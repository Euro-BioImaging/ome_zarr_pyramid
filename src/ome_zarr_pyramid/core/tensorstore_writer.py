"""Async TensorStore producer-consumer writer for OME-Zarr pyramids.

Ported from eubi_bridge's ``core/writers.py``. Provides a queue-based
reader/writer pipeline (`write_with_queue_async`) plus higher-level
orchestration for building and downscaling a multiscale pyramid
(`downscale_with_tensorstore_async`, `store_multiscale_async`).
"""

import asyncio
import gc
import itertools
import os
import shutil
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import tensorstore as ts
import zarr
from zarr.storage import LocalStore

from ome_zarr_pyramid.core.pyramid import NGFFMetadataHandler, Pyramid
from ome_zarr_pyramid.utils.array_utils import (
    autocompute_chunk_shape,
    get_array_chunks,
    get_chunk_shape,
    parse_memory,
)
from ome_zarr_pyramid.utils.compressor_config import CompressorConfig
from ome_zarr_pyramid.utils.logging_config import get_logger

logger = get_logger(__name__)

ZARR_V2 = 2
ZARR_V3 = 3
DEFAULT_DIMENSION_SEPARATOR = "/"


def _zarr_group(store, overwrite: bool, zarr_format: int) -> zarr.Group:
    """Create or open a zarr group, handling keyword differences across zarr versions."""
    try:
        return zarr.group(store, overwrite=overwrite, zarr_format=zarr_format)  # type: ignore[arg-type]
    except TypeError:
        try:
            return zarr.group(store, overwrite=overwrite, zarr_version=zarr_format)  # type: ignore[call-arg,arg-type]
        except TypeError:
            return zarr.group(store, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Zarr array creation
# ---------------------------------------------------------------------------

def _create_zarr_v2_array(
        store_path: Union[Path, str],
        shape: Tuple[int, ...],
        chunks: Tuple[int, ...],
        dtype: Any,
        compressor_config: CompressorConfig,
        dimension_separator: str,
        overwrite: bool,
) -> zarr.Array:
    compressor = compressor_config.build(zarr_format=ZARR_V2)
    return zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        store=store_path,
        compressor=compressor,
        dimension_separator=dimension_separator,
        overwrite=overwrite,
        zarr_format=ZARR_V2,
    )


def _create_zarr_v3_array(
        store: Any,
        shape: Tuple[int, ...],
        chunks: Tuple[int, ...],
        dtype: Any,
        compressor_config: CompressorConfig,
        shards: Optional[Tuple[int, ...]],
        dimension_names: Optional[List[str]] = None,
        overwrite: bool = False,
) -> zarr.Array:
    compressor = compressor_config.build(zarr_format=ZARR_V3)
    # For Zarr v3, only include compressors if not None (no compression)
    compressors = [compressor] if compressor is not None else []
    return zarr.create_array(
        store=store,
        shape=shape,
        chunks=chunks,
        shards=shards,
        dimension_names=dimension_names,
        dtype=dtype,
        compressors=compressors,
        overwrite=overwrite,
        zarr_format=ZARR_V3,
    )


def _create_zarr_array(
        store_path: Union[Path, str],
        shape: Tuple[int, ...],
        chunks: Tuple[int, ...],
        dtype: Any,
        compressor_config: Optional[CompressorConfig] = None,
        zarr_format: int = ZARR_V2,
        overwrite: bool = False,
        shards: Optional[Tuple[int, ...]] = None,
        dimension_separator: str = DEFAULT_DIMENSION_SEPARATOR,
        dimension_names: Optional[List[str]] = None,
) -> zarr.Array:
    """Create a Zarr array with specified format and compression settings."""
    compressor_config = compressor_config or CompressorConfig()
    dtype = _normalize_dtype(dtype, None)  # coerce ts/str dtypes -> np.dtype for zarr.create
    chunks = tuple(np.minimum(shape, chunks).tolist())

    # For sharding: ensure shards are compatible with chunks
    if shards is not None:
        shards = tuple(np.array(shards).flatten().tolist())
        adjusted_shards = []
        for shard_size, chunk_size, dim_size in zip(shards, chunks, shape):
            if shard_size % chunk_size != 0 and chunk_size > 0:
                adjusted = (dim_size // max(1, dim_size // shard_size)) if shard_size > 0 else chunk_size
                adjusted_shards.append(min(adjusted, dim_size))
            else:
                adjusted_shards.append(shard_size)
        shards = tuple(adjusted_shards)

    store = LocalStore(store_path)

    if zarr_format not in (ZARR_V2, ZARR_V3):
        raise ValueError(f"Unsupported Zarr format: {zarr_format}")

    if zarr_format == ZARR_V2:
        return _create_zarr_v2_array(
            store_path=store_path,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor_config=compressor_config,
            dimension_separator=dimension_separator,
            overwrite=overwrite,
        )

    return _create_zarr_v3_array(
        store=store,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor_config=compressor_config,
        shards=shards,
        dimension_names=dimension_names,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Shared write helpers
# ---------------------------------------------------------------------------

def _normalize_dtype(dtype, arr) -> np.dtype:
    """Coerce dtype to np.dtype, falling back to arr.dtype when None."""
    if dtype is None:
        return np.dtype(arr.dtype)
    if isinstance(dtype, str):
        return np.dtype(dtype)
    try:
        return np.dtype(dtype.name)
    except Exception:
        return np.dtype(dtype)


def _align_shards(shards, chunks) -> tuple:
    """Return a shard tuple whose every element is a multiple of the matching chunk size."""
    if shards is None:
        return tuple(chunks)
    shards = np.asarray(shards)
    chunks_arr = np.asarray(chunks)
    if not np.allclose(np.mod(shards, chunks_arr), 0):
        shards = np.multiply(np.floor_divide(shards, chunks_arr), chunks_arr)
    return tuple(int(s) for s in np.ravel(shards))


def _read_region(arr, region_slice):
    """Unified region reader for dask, zarr, and numpy arrays."""
    if hasattr(arr, 'compute'):
        return arr[region_slice].compute()
    return np.asarray(arr[region_slice])


def _compute_region_shape(input_shape, final_chunks, region_size_mb, dtype=None, input_chunks=None):
    """Compute optimal region shape with deterministic algorithm.

    Uses LCM (Least Common Multiple) to maintain alignment with both
    input chunks and output chunks.

    Algorithm:
    1. Start with a single output chunk
    2. Expand dimensions in reverse order (last -> first) until region_size_mb reached
    3. Use LCM of input_chunks and output_chunks for expansion increments
    4. Stop when budget exhausted or dimension complete

    Parameters
    ----------
    input_shape : tuple of int
        Shape of input array.
    final_chunks : tuple of int
        Output chunk shape (zarr chunks).
    region_size_mb : float or str
        Target size of read regions. Can be a number in MB or a string like '1GB', '512MB'.
    dtype : numpy.dtype, optional
        Data type for computing element size.
    input_chunks : tuple of int, optional
        Input chunk shape (for alignment).

    Returns
    -------
    tuple of int
        Optimal region shape.

    Example
    -------
    >>> _compute_region_shape((50, 179, 2, 339, 415), (1, 64, 1, 64, 64), 8.0)
    (1, 128, 2, 339, 415)
    """
    region_size_mb = parse_memory(region_size_mb)

    if dtype is None:
        element_size = 2
    else:
        try:
            element_size = int(np.dtype(dtype).itemsize)
        except Exception:
            element_size = 2

    target_bytes = region_size_mb * 1024 * 1024

    input_arr = np.array(input_shape, dtype=np.int64)
    output_chunk_arr = np.array(final_chunks, dtype=np.int64)

    if input_chunks is None:
        input_chunk_arr = output_chunk_arr.copy()
    else:
        input_chunk_arr = np.array(input_chunks, dtype=np.int64)

    # STEP 1: Start with one output chunk
    region_arr = output_chunk_arr.copy()
    current_bytes = np.prod(region_arr) * element_size

    # If single output chunk exceeds target, use it anyway (can't split chunks)
    if current_bytes >= target_bytes:
        return tuple(region_arr.tolist())

    # STEP 2: Compute expansion increments using LCM (maintains both alignments).
    # When lcm(input_chunk, output_chunk) >= dim_size the increment is larger
    # than the entire dimension: the first expansion step jumps straight to full
    # extent, making fine-grained region sizing impossible. This happens when
    # gcd(input_chunk, output_chunk) is very small relative to the chunk sizes.
    # Fall back to output_chunk in that case so the budget can be honoured.
    expansion_increments = np.zeros(len(region_arr), dtype=np.int64)
    for i in range(len(region_arr)):
        gcd = np.gcd(input_chunk_arr[i], output_chunk_arr[i])
        lcm = (input_chunk_arr[i] * output_chunk_arr[i]) // gcd
        expansion_increments[i] = output_chunk_arr[i] if lcm >= input_arr[i] else lcm

    # STEP 3: Expand dimensions in reverse order (last -> first)
    for dim in reversed(range(len(region_arr))):
        while region_arr[dim] < input_arr[dim]:
            increment = expansion_increments[dim]
            remaining = input_arr[dim] - region_arr[dim]

            if remaining <= increment:
                new_size = input_arr[dim]
            else:
                new_size = region_arr[dim] + increment
                future_remaining = input_arr[dim] - new_size
                if 0 < future_remaining < increment:
                    new_size = input_arr[dim]

            test_region = region_arr.copy()
            test_region[dim] = new_size
            new_bytes = np.prod(test_region) * element_size

            if new_bytes <= target_bytes:
                region_arr[dim] = new_size
                current_bytes = new_bytes
            else:
                break

        if current_bytes >= target_bytes:
            break

    # STEP 4: Verify output chunk alignment for PARTIAL dimensions only.
    # Full dimensions don't need alignment (they include all chunks anyway).
    for i in range(len(region_arr)):
        if region_arr[i] >= input_arr[i]:
            continue

        if output_chunk_arr[i] > 0 and region_arr[i] % output_chunk_arr[i] != 0:
            aligned_size = (region_arr[i] // output_chunk_arr[i]) * output_chunk_arr[i]
            region_arr[i] = max(output_chunk_arr[i], aligned_size)

    return tuple(region_arr.tolist())


def wrap_output_path(output_path: str):
    """Resolve an output path to a local filesystem path or an s3fs mapping."""
    if output_path.startswith('https://'):
        try:
            import s3fs
        except ImportError as e:
            raise ImportError(
                "writing to an https:// (S3) path needs s3fs: "
                "pip install 'ome_zarr_pyramid[s3]'"
            ) from e
        endpoint_url = 'https://' + output_path.replace('https://', '').split('/')[0]
        relpath = output_path.replace(endpoint_url, '')
        fs = s3fs.S3FileSystem(
            client_kwargs={
                'endpoint_url': endpoint_url,
            },
            endpoint_url=endpoint_url
        )
        fs.makedirs(relpath, exist_ok=True)
        mapped = fs.get_mapper(relpath)
    else:
        os.makedirs(output_path, exist_ok=True)
        mapped = os.path.abspath(output_path)
    return mapped


def _get_or_create_multimeta(gr: zarr.Group,
                              axis_order: Union[str, Sequence[str]],
                              unit_list: List[str],
                              version: str) -> NGFFMetadataHandler:
    """Read existing or create new metadata handler for a zarr group.

    Parameters
    ----------
    gr : zarr.Group
        Zarr group to read metadata from or write metadata to.
    axis_order : Union[str, Sequence[str]]
        Axis names in order (e.g. ``'zyx'`` or ``['z', 'y', 'x']``).
    unit_list : List[str]
        List of strings indicating the units of each axis.
    version : str
        Version of NGFF to create if no metadata exists.

    Returns
    -------
    NGFFMetadataHandler
        Metadata handler for the zarr group.
    """
    handler = NGFFMetadataHandler()
    handler.connect_to_group(gr)
    try:
        handler.read_metadata()
    except (FileNotFoundError, KeyError, ValueError):
        handler.create_new(version=version)
        handler.parse_axes(axis_order=axis_order, units=unit_list)
    return handler


# ---------------------------------------------------------------------------
# Queue-based producer-consumer writer
# ---------------------------------------------------------------------------

async def write_with_queue_async(
    arr: Union[da.Array, zarr.Array],
    output_path: Union[Path, str],
    output_chunks: Optional[Tuple[int, ...]] = None,
    output_shards: Optional[Tuple[int, ...]] = None,
    zarr_format: int = 2,
    dtype: Optional[np.dtype] = None,
    dimension_names: Optional[List[str]] = None,
    compressor: Optional[str] = 'blosc',
    compressor_params: Optional[dict] = None,
    num_readers: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    region_size_mb: float = 8.0,
    queue_size: Optional[int] = None,
    gc_interval: float = 15.0,
    overwrite: bool = False,
    verbose: bool = False,
    **kwargs
) -> 'ts.TensorStore':
    """Queue-based writer with producer-consumer threading pattern.

    Architecture:
    - Reader threads: call `_read_region()` to read from the input array and
      enqueue `(slice, data)` tuples.
    - Writer threads: pop from the queue and submit async TensorStore writes.
    - Monitor thread: progress logging every 2 seconds.
    - Queue buffering decouples read/write for pipeline throughput.

    Parameters
    ----------
    arr : Union[da.Array, zarr.Array]
        Input array to write.
    output_path : Union[Path, str]
        Path to the output zarr array.
    output_chunks : Tuple[int, ...], optional
        Output chunk shape. Defaults to the input array's chunk shape.
    zarr_format : int, optional
        Zarr format version (2 or 3). Default is 2.
    dtype : Optional[np.dtype], optional
        Output data type. If None, uses the input array's dtype.
    dimension_names : Optional[List[str]], optional
        Names for each dimension (e.g. ['t', 'c', 'z', 'y', 'x']).
    compressor : str, optional
        Compression algorithm ('blosc', 'gzip', 'zstd', etc.). Default is 'blosc'.
    compressor_params : Optional[dict], optional
        Parameters for the compressor (e.g. {'cname': 'zstd', 'clevel': 1}).
    num_readers : Optional[int], optional
        Number of reader threads. Default is 2 * max_concurrency.
    max_concurrency : Optional[int], optional
        Number of writer threads. Default is 4.
    region_size_mb : float, optional
        Target size of read regions in MB. Default is 8.0.
    queue_size : Optional[int], optional
        Maximum queue size. Default is min(128, max(8, num_readers)).
    gc_interval : float, optional
        Seconds between garbage collections. Default is 15.0.
    overwrite : bool, optional
        If True, delete existing data before writing. Default is False.
    verbose : bool, optional
        Enable verbose logging. Default is False.

    Returns
    -------
    ts.TensorStore
        TensorStore handle to the written array.
    """
    # === DEFAULTS ===
    if max_concurrency is None:
        max_concurrency = 4
    if num_readers is None:
        num_readers = 2 * max_concurrency
    if queue_size is None:
        queue_size = min(128, max(8, num_readers))

    dtype = _normalize_dtype(dtype, arr)
    if output_chunks is None:
        output_chunks = get_chunk_shape(arr)
    output_shards = _align_shards(output_shards, output_chunks)

    # === CREATE ARRAY WITH ZARR LIBRARY FIRST ===
    # This ensures all compressor parameters are applied correctly
    if compressor_params is None:
        compressor_params = {}

    output_path_str = str(output_path)
    if overwrite and os.path.exists(output_path_str):
        shutil.rmtree(output_path_str)
    os.makedirs(output_path_str, exist_ok=True)

    compressor_config = CompressorConfig(
        name=compressor,
        params=compressor_params
    )
    _create_zarr_array(
        store_path=output_path_str,
        shape=arr.shape,
        chunks=output_chunks,
        shards=output_shards,
        dtype=dtype,
        compressor_config=compressor_config,
        zarr_format=zarr_format,
        dimension_names=dimension_names,
        overwrite=overwrite,
    )

    # === COMPUTE REGION SHAPE ===
    input_chunks = get_array_chunks(arr)

    region_shape = _compute_region_shape(
        input_shape=arr.shape,
        final_chunks=output_chunks,
        region_size_mb=region_size_mb,
        dtype=dtype,
        input_chunks=input_chunks
    )
    # If arr is a dask array, rechunk it to match the read region shape
    if isinstance(arr, da.Array):
        arr = arr.rechunk(region_shape)

    # === OPEN WITH TENSORSTORE FOR WRITING ===
    # TensorStore will use the metadata already written by the zarr library
    spec_dict = {
        'driver': 'zarr' if zarr_format == 2 else 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': output_path_str
        },
        'open': True
    }

    ts_store = await ts.open(spec_dict)

    if verbose:
        logger.info(f"Queue-based writer: {num_readers} readers, {max_concurrency} writers, region_shape={region_shape}")

    # === THREADED WRITE FUNCTION ===
    def _run_threaded_write():
        """Synchronous function that runs the threaded write pipeline."""
        state = {
            'completed': 0,
            'failed': 0,
            'total': 0,
            'lock': threading.Lock(),
            'error': None,
            'done_reading': False
        }

        # Compute total regions
        total_regions = 1
        for dim_size, region_size in zip(arr.shape, region_shape):
            total_regions *= int(np.ceil(dim_size / region_size))
        state['total'] = total_regions

        q = Queue(maxsize=queue_size)

        # Generate region indices
        region_indices = []
        ranges = [range(0, dim_size, region_size) for dim_size, region_size in zip(arr.shape, region_shape)]
        for idx_tuple in itertools.product(*ranges):
            region_slice = tuple(
                slice(start, min(start + region_size, dim_size))
                for start, region_size, dim_size in zip(idx_tuple, region_shape, arr.shape)
            )
            region_indices.append(region_slice)

        # Atomic index counter
        index_lock = threading.Lock()
        index_counter = [0]

        # === READER THREAD ===
        def reader_thread():
            """Read regions and enqueue them."""
            last_gc = time.time()
            while True:
                with index_lock:
                    if index_counter[0] >= len(region_indices):
                        break
                    idx = index_counter[0]
                    index_counter[0] += 1

                region_slice = region_indices[idx]
                if verbose:
                    logger.info(f"Reader thread reading region {idx+1}/{len(region_indices)}: {region_slice}")
                    logger.info(f"arr shape: {arr.shape}, region shape: {[s.stop - s.start for s in region_slice]}")

                try:
                    data = _read_region(arr, region_slice)
                    q.put((region_slice, data))

                    if time.time() - last_gc > gc_interval:
                        gc.collect()
                        last_gc = time.time()

                except Exception as e:
                    with state['lock']:
                        if state['error'] is None:
                            state['error'] = e
                    logger.error(f"Reader thread error at {region_slice}: {e}")
                    break

        # === WRITER THREAD ===
        def writer_thread():
            """Write regions from the queue."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _async_writer():
                while True:
                    try:
                        region_slice, data = q.get(timeout=1.0)
                    except Exception:
                        if state['done_reading'] and q.empty():
                            break
                        continue

                    try:
                        if verbose:
                            logger.info(f"Writer thread writing region: {region_slice}")
                            logger.info(f"Data shape: {data.shape}, expected shape: {[s.stop - s.start for s in region_slice]}")
                        await ts_store[region_slice].write(data)

                        with state['lock']:
                            state['completed'] += 1

                        q.task_done()

                    except Exception as e:
                        with state['lock']:
                            state['failed'] += 1
                            if state['error'] is None:
                                state['error'] = e
                        logger.error(f"Writer thread error at {region_slice}: {e}")
                        q.task_done()

            loop.run_until_complete(_async_writer())
            loop.close()

        # === MONITOR THREAD ===
        def monitor_progress():
            """Log progress every 2 seconds."""
            while True:
                time.sleep(2.0)
                with state['lock']:
                    completed = state['completed']
                    total = state['total']
                    if total > 0:
                        pct = 100.0 * completed / total
                        if verbose:
                            logger.info(f"Write progress: {completed}/{total} regions ({pct:.1f}%)")
                    if completed + state['failed'] >= total:
                        break

        # === START THREADS ===
        readers = [threading.Thread(target=reader_thread, daemon=True) for _ in range(num_readers)]
        writers = [threading.Thread(target=writer_thread, daemon=True) for _ in range(max_concurrency)]
        monitor = threading.Thread(target=monitor_progress, daemon=True)

        for t in readers:
            t.start()
        for t in writers:
            t.start()
        monitor.start()

        for t in readers:
            t.join()

        state['done_reading'] = True

        q.join()
        for t in writers:
            t.join()

        monitor.join(timeout=5.0)

        if state['error'] is not None:
            raise state['error']

        if verbose:
            logger.info(f"Write complete: {state['completed']}/{state['total']} regions")

    # === RUN THREADED WRITE IN EXECUTOR ===
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run_threaded_write)

    return ts_store


# ---------------------------------------------------------------------------
# Pyramid-level orchestration
# ---------------------------------------------------------------------------

async def downscale_with_tensorstore_async(
        base_store: Union[str, Path, 'ts.TensorStore'],
        scale_factor,
        n_layers,
        downscale_method: str = 'simple',
        min_dimension_size: Optional[int] = None,
        smart_scale_factor=None,
        max_concurrent_downscale_layers: int = 3,
        **kwargs
) -> Pyramid:
    """Downscale a base layer into the remaining pyramid levels via TensorStore.

    Opens the parent group as a `Pyramid`, runs `update_downscaler` (which
    automatically uses TensorStore for zarr-backed arrays), then writes each
    downscaled level concurrently with `write_with_queue_async`.
    """
    if isinstance(base_store, ts.TensorStore):
        base_array_path = base_store.kvstore.path
    else:
        base_array_path = str(base_store)

    gr_path = os.path.dirname(base_array_path)
    pyr = Pyramid(gr_path)

    region_size_mb = kwargs.get('region_size_mb', 8.0)
    max_concurrency = kwargs.pop('max_concurrency', None)

    logger.info(
        f"Updating downscaler with scale_factor={scale_factor}, "
        f"n_layers={n_layers}, smart_scale_factor={smart_scale_factor}"
    )
    update_kwargs: dict = dict(
        scale_factor=scale_factor,
        n_layers=n_layers,
        downscale_method=downscale_method,
        smart_scale_factor=smart_scale_factor,
    )
    if min_dimension_size is not None:
        update_kwargs['min_dimension_size'] = min_dimension_size
    await pyr.update_downscaler(**update_kwargs)

    downscaled_arrays = pyr.downscaler._downscaled_arrays
    logger.info(f"Downscaler created {len(downscaled_arrays)} layers for writing")

    grpath = str(pyr.gr.store.root)
    basepath = pyr.meta.resolution_paths[0]
    base_layer = pyr.layers[basepath]
    zarr_format = pyr.meta.zarr_format

    # Derive compressor name/params from the base layer's actual codec
    compressor_config = CompressorConfig.from_array(base_layer)
    if compressor_config is None:
        compressor_name = None
        compressor_params: dict = {}
    else:
        compressor_name = compressor_config.name
        compressor_params = compressor_config.params

    # Fixed region size for downscale writes regardless of the caller's
    # base-layer region_size_mb -- keeps downscale I/O fast even for
    # large output chunk sizes.
    downscale_region_size_mb = 16.0
    logger.info(f"Downscaling with fixed region_size_mb={downscale_region_size_mb} MB")

    excluded_kwargs = {
        'max_concurrency', 'dtype', 'compressor', 'compressor_params',
        'zarr_format', 'region_size_mb', 'output_chunks', 'output_shards',
        'dimension_names', 'arr', 'output_path',
    }

    coros = []
    total_layers = len(downscaled_arrays) - 1
    for idx, arr in enumerate(downscaled_arrays):
        if idx == 0:
            continue
        key = str(idx)
        logger.info(f"Preparing layer {key} ({idx}/{total_layers}) for writing...")
        logger.info(f"Layer {key} shape: {arr.shape}, dtype: {arr.dtype}")
        shards = tuple(base_layer.shards) if base_layer.shards is not None else base_layer.chunks

        params = dict(
            arr=arr,
            output_path=os.path.join(grpath, key),
            output_chunks=tuple(base_layer.chunks),
            output_shards=shards,
            compressor=compressor_name,
            compressor_params=compressor_params,
            zarr_format=zarr_format,
            dimension_names=list(pyr.axes),
            dtype=np.dtype(arr.dtype.name),
            region_size_mb=downscale_region_size_mb,
            max_concurrency=max_concurrency,
            **{k: v for k, v in kwargs.items() if k not in excluded_kwargs}
        )
        coros.append((key, write_with_queue_async(**params)))

    n_concurrent = max(1, min(max_concurrent_downscale_layers, len(coros)))
    logger.info(
        f"Starting concurrent writes for {len(coros)} downscaled layers "
        f"(max {n_concurrent} at a time)..."
    )

    semaphore = asyncio.Semaphore(n_concurrent)

    async def _bounded(coro):
        async with semaphore:
            return await coro

    results = await asyncio.gather(
        *[_bounded(coro) for _, coro in coros],
        return_exceptions=True,
    )
    for (key, _), result in zip(coros, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to write layer {key}: {result}", exc_info=result)
            raise result

    logger.info("All downscaled layers written successfully")

    # Register the downscaled levels in the parent group's NGFF metadata
    for idx in range(1, len(downscaled_arrays)):
        pyr.meta.add_dataset(path=str(idx), scale=pyr.downscaler.dm.scales[idx].tolist(), overwrite=True)
    pyr.meta.save_changes()

    return Pyramid(gr_path)


async def store_multiscale_async(
    arr: Union[da.Array, zarr.Array],
    output_path: Union[Path, str],
    axes: Sequence[str],
    scales: Sequence[float],
    units: Sequence[str],
    zarr_format: int = 2,
    auto_chunk: bool = True,
    output_chunks: Optional[Tuple[int, ...]] = None,
    output_shard_coefficients: Optional[Tuple[int, ...]] = None,
    overwrite: bool = False,
    channel_meta: Optional[Any] = None,
    *,
    scale_factors: Optional[Tuple[Union[int, float], ...]] = None,
    n_layers=None,
    min_dimension_size: Optional[int] = None,
    downscale_method: str = 'simple',
    smart_scale_factor: Optional[Tuple[Union[int, float], ...]] = None,
    num_readers: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    region_size_mb: float = 8.0,
    queue_size: Optional[int] = None,
    gc_interval: float = 15.0,
    max_concurrent_downscale_layers: int = 3,
    **kwargs
) -> Pyramid:
    """Write a base array to a new NGFF zarr group and (optionally) downscale it.

    Creates the output group, writes the base layer with `write_with_queue_async`,
    then -- if `scale_factors` is given -- builds the remaining pyramid levels
    via `downscale_with_tensorstore_async`.
    """
    verbose = kwargs.get('verbose', False)
    output_shards = kwargs.get('output_shards', None)
    target_chunk_mb = kwargs.get('target_chunk_mb', 1)
    dtype = kwargs.get('dtype', arr.dtype)
    if dtype is None:
        dtype = arr.dtype
    elif isinstance(dtype, str):
        dtype = np.dtype(dtype)
    compressor = kwargs.get('compressor', 'blosc')
    compressor_params = kwargs.get('compressor_params', {})
    logger.info(f"Compressor selected for output: {compressor} with params: {compressor_params}")

    # Parse chunks
    if auto_chunk or output_chunks is None:
        if verbose:
            logger.info(f"Auto-computing chunks for {output_path} with target chunk size {target_chunk_mb} MB")
        chunks = autocompute_chunk_shape(
            arr.shape,
            axes=axes,
            target_chunk_mb=target_chunk_mb,
            dtype=dtype
        )
    else:
        chunks = output_chunks

    chunks = np.minimum(chunks, arr.shape).tolist()
    chunks = tuple(int(item) for item in chunks)

    # Parse shards
    if output_shards is not None:
        shards = output_shards
    elif output_shard_coefficients is not None:
        shards = tuple(int(c * s) for c, s in zip(chunks, output_shard_coefficients))
    else:
        shards = chunks
    shards = tuple(int(item) for item in shards)

    # Make (or overwrite) the top-level group
    outpath = wrap_output_path(str(output_path))
    gr = _zarr_group(outpath, overwrite=overwrite, zarr_format=zarr_format)

    base_store_path = os.path.join(str(outpath), '0')
    version = '0.5' if zarr_format == 3 else '0.4'
    meta = _get_or_create_multimeta(gr, axis_order=axes, unit_list=list(units), version=version)

    if channel_meta == 'auto':
        if 'c' in axes:
            idx = list(axes).index('c')
            size = arr.shape[idx]
        else:
            size = 1
        meta.autocompute_omerometa(size, arr.dtype)
    elif channel_meta is not None:
        if verbose:
            logger.info(f"Adding channel metadata: {channel_meta}")
        meta.metadata['omero']['channels'] = channel_meta

    meta.save_changes()

    if verbose:
        logger.info(f"Starting to write base layer to {base_store_path}")
        logger.info(f"The region_size_mb is set to {region_size_mb} MB for base layer writing.")

    base_start_time = time.time()
    await write_with_queue_async(
        arr=arr,
        output_path=base_store_path,
        output_chunks=chunks,
        zarr_format=zarr_format,
        dtype=dtype,
        dimension_names=list(axes),
        compressor=compressor,
        compressor_params=compressor_params,
        num_readers=num_readers,
        max_concurrency=max_concurrency,
        region_size_mb=region_size_mb,
        queue_size=queue_size,
        gc_interval=gc_interval,
        overwrite=overwrite,
        verbose=verbose,
        output_shards=shards,
    )
    base_elapsed = (time.time() - base_start_time) / 60
    logger.info(f"Base layer written in {base_elapsed:.2f} minutes")

    # Add base layer to metadata
    meta.add_dataset(path='0', scale=list(scales), overwrite=True)
    meta.save_changes()

    if scale_factors is not None:
        logger.info("Starting downscaling...")
        downscale_start = time.time()
        try:
            pyr = await downscale_with_tensorstore_async(
                base_store=base_store_path,
                scale_factor=scale_factors,
                n_layers=n_layers,
                min_dimension_size=min_dimension_size,
                downscale_method=downscale_method,
                smart_scale_factor=smart_scale_factor,
                max_concurrency=max_concurrency,
                queue_size=queue_size,
                region_size_mb=region_size_mb,
                num_readers=num_readers,
                gc_interval=gc_interval,
                verbose=verbose,
                max_concurrent_downscale_layers=max_concurrent_downscale_layers,
            )
            downscale_elapsed = (time.time() - downscale_start) / 60
            logger.info(f"Downscaling completed in {downscale_elapsed:.2f} minutes")
        except Exception as e:
            downscale_elapsed = (time.time() - downscale_start) / 60
            logger.error(f"Downscaling failed after {downscale_elapsed:.2f} minutes: {e}", exc_info=True)
            raise
    else:
        pyr = Pyramid(gr)

    return pyr
