"""High-performance I/O module for OME Zarr Pyramids using region-wise processing.

Implements async vectorized writes with TensorStore, queue-based producer-consumer
pipelines, and threading for optimal performance on multi-resolution datasets.
Based on the dyna_zarr architecture for efficient large-scale I/O.
"""

import asyncio
import gc
import itertools
import logging
import os
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import tensorstore as ts
import zarr

from ome_zarr_pyramid.core import tensorstore_writer
from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.utils.array_utils import get_chunk_shape
from ome_zarr_pyramid.utils.logging_config import get_logger
from ome_zarr_pyramid.utils.storage_utils import make_kvstore

logger = get_logger(__name__)

# Suppress pending task warnings from zarr's async operations
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Task was destroyed but it is pending.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*cannot schedule new futures after shutdown.*")


def _attach_image_label(group: zarr.Group, image_label: dict, version: str) -> None:
    """Attach OME ``image-label`` metadata to a label-image group (version-aware:
    0.5 nests it under the ``ome`` attr; 0.4 uses a top-level ``image-label``)."""
    if version == "0.5":
        ome = dict(group.attrs.get("ome", {}))
        ome["image-label"] = image_label
        group.attrs["ome"] = ome
    else:  # 0.4
        group.attrs["image-label"] = image_label


def _register_label_name(labels_group: zarr.Group, name: str, version: str) -> None:
    """Add ``name`` to a ``labels`` group's label list (the NGFF labels/ convention)."""
    if version == "0.5":
        ome = dict(labels_group.attrs.get("ome", {}))
        ome.setdefault("version", version)
        names = list(ome.get("labels", []))
        if name not in names:
            names.append(name)
        ome["labels"] = names
        labels_group.attrs["ome"] = ome
    else:  # 0.4
        names = list(labels_group.attrs.get("labels", []))
        if name not in names:
            names.append(name)
        labels_group.attrs["labels"] = names


def _read_label_names(labels_group: zarr.Group) -> list:
    """The registered label names in a ``labels/`` group (0.5 nests them under the
    ``ome`` attr, 0.4 uses a top-level ``labels`` list)."""
    if "ome" in labels_group.attrs:
        return list(dict(labels_group.attrs["ome"]).get("labels", []))
    return list(labels_group.attrs.get("labels", []))


def _is_remote_store(path) -> bool:
    """True for an object-store URL (https:// / s3://) that must NOT be treated as a
    local filesystem `Path` (which would mangle the URL)."""
    s = str(path)
    return s.startswith("https://") or s.startswith("s3://")


def _is_dyna_pyramid(pyramid) -> bool:
    """True if the pyramid's in-memory layers are dyna_zarr DynamicArrays (the memory-bounded
    pull backend), so writing should stream via dyna_zarr.io.write. False for zarr/dask/numpy
    layers, or when dyna_zarr isn't installed."""
    try:
        from dyna_zarr import DynamicArray
    except ImportError:
        return False
    if pyramid.meta is None:
        return False
    try:
        first = pyramid.layers[pyramid.meta.resolution_paths[0]]
    except Exception:
        return False
    return isinstance(first, DynamicArray)


def _resolve_dyna_chunks(chunk_shape, level_path, arr):
    """Storage chunks for a dyna level write: an explicit ``chunk_shape`` (tuple, or a
    ``{level: tuple}`` dict), else the array's own chunks (preserving the read chunking)."""
    if chunk_shape is None:
        return arr.chunks
    if isinstance(chunk_shape, dict):
        return chunk_shape.get(int(level_path), chunk_shape.get(str(level_path), arr.chunks))
    return tuple(chunk_shape)


def _store_join(path, *parts):
    """Join sub-paths onto a store: URL-safe (forward slashes) for remote stores, a
    `pathlib.Path` for local ones."""
    if _is_remote_store(path):
        return str(path).rstrip("/") + "/" + "/".join(str(p).strip("/") for p in parts)
    p = Path(path)
    for part in parts:
        p = p / str(part)
    return p


def _open_store_group(path, zarr_format: int, mode: str = "a") -> zarr.Group:
    """Open/create a zarr group for attr read/write, S3-aware: an https:// path is
    resolved to an s3fs mapping; a local path is used directly."""
    if _is_remote_store(path):
        store = tensorstore_writer.wrap_output_path(str(path))
    else:
        store = str(path)
    return zarr.open_group(store, mode=mode, zarr_format=zarr_format)


def _iter_region_slices(shape, region_shape):
    """Yield the tuple-of-slices tiling `shape` into `region_shape` blocks."""
    ranges = [range(0, int(s), int(r)) for s, r in zip(shape, region_shape)]
    for start in itertools.product(*ranges):
        yield tuple(slice(st, min(st + int(r), int(s))) for st, r, s in zip(start, region_shape, shape))


def _compute_region_shape_for_layer(
    layer_shape: Tuple[int, ...],
    layer_chunks: Tuple[int, ...],
    region_size_mb: float = 8.0,
    dtype: Union[type, np.dtype] = np.float32
) -> Tuple[int, ...]:
    """Compute region shape given target region size in MB.
    
    Parameters
    ----------
    layer_shape : tuple
        Shape of the layer
    layer_chunks : tuple
        Chunk shape for the layer
    region_size_mb : float
        Target region size in MB
    dtype : type or np.dtype
        Data type
        
    Returns
    -------
    tuple
        Region shape that fits approximately in region_size_mb
    """
    # Ensure all inputs are integers
    layer_shape = tuple(int(s) for s in layer_shape)
    layer_chunks = tuple(int(c) for c in layer_chunks)
    
    itemsize = np.dtype(dtype).itemsize
    bytes_per_element = itemsize
    target_bytes = int(region_size_mb * 1024 * 1024)
    
    # Start with layer chunks, expand to fill region_size_mb
    region_shape = list(layer_chunks)
    for dim in range(len(region_shape)):
        max_size = layer_shape[dim]
        # Find how many chunks fit in target bytes
        current_bytes = int(np.prod(region_shape) * bytes_per_element)
        if current_bytes < target_bytes:
            # Try to expand this dimension
            expand_factor = max(1, target_bytes // (current_bytes or 1))
            region_shape[dim] = min(max_size, region_shape[dim] * expand_factor)
    
    return tuple(region_shape)


def _write_layer_process_worker(task: dict) -> Tuple[str, int, int]:
    """Module-level worker that writes one pyramid layer inside a dedicated process.

    Defined at module level so it is importable by spawned worker processes on
    Windows (which uses 'spawn' instead of 'fork').  Each worker independently
    re-opens its own zarr store handles, so no file descriptors or async event
    loops are shared across process boundaries.

    Within the process a single reader thread produces regions into a queue and
    *threads_per_layer* writer threads drain it concurrently.

    Parameters
    ----------
    task : dict
        source_zarr_path   – str, filesystem path to the source zarr group root
        source_layer_path  – str, layer key inside the source group
        dest_zarr_path     – str, filesystem path to the destination zarr group root
        dest_layer_path    – str, layer key inside the destination group
        region_slices_raw  – list[list[list[int,int]]], per-region list of
                             [start, stop] pairs for each dimension
        threads_per_layer  – int, writer-thread count (and queue depth) per process
        verbose            – bool

    Returns
    -------
    tuple of (dest_layer_path, regions_written, total_regions)
    """
    import zarr
    import numpy as np
    import threading
    import gc
    from queue import Queue, Empty  # noqa: F401 (needed in worker process)

    source_zarr_path  = task['source_zarr_path']
    source_layer_path = task['source_layer_path']
    dest_zarr_path    = task['dest_zarr_path']
    dest_layer_path   = task['dest_layer_path']
    region_slices_raw = task['region_slices_raw']
    threads_per_layer = max(1, task['threads_per_layer'])
    verbose           = task['verbose']

    source_array = zarr.open_group(source_zarr_path, mode='r')[source_layer_path]
    dest_array   = zarr.open_group(dest_zarr_path,   mode='r+')[dest_layer_path]

    region_slices = [
        tuple(slice(s[0], s[1]) for s in reg)
        for reg in region_slices_raw
    ]
    total   = len(region_slices)
    queue   = Queue(maxsize=threads_per_layer * 4)
    written: list = [0]
    error:   list = [None]

    def _reader():
        try:
            for slices in region_slices:
                if error[0]:
                    break
                queue.put((slices, np.asarray(source_array[slices])))
        except Exception as exc:
            error[0] = exc
        finally:
            for _ in range(threads_per_layer):
                queue.put(None)   # sentinel per writer

    def _writer():
        while True:
            try:
                item = queue.get(timeout=2.0)
            except Exception:
                continue
            if item is None:
                break
            slices, data = item
            dest_array[slices] = data
            written[0] += 1
            queue.task_done()

    reader_t  = threading.Thread(target=_reader, daemon=True)
    writer_ts = [threading.Thread(target=_writer, daemon=True) for _ in range(threads_per_layer)]

    reader_t.start()
    for wt in writer_ts:
        wt.start()
    reader_t.join()
    for wt in writer_ts:
        wt.join()

    if error[0]:
        raise error[0]

    gc.collect()
    return dest_layer_path, written[0], total


def _create_layer_array(dest_path: Path,
                         layer_path: str,
                         shape: Tuple[int, ...],
                         chunks: Tuple[int, ...],
                         dtype,
                         pyramid: Pyramid) -> None:
    """Create a destination layer array, honoring the pyramid's zarr format and compressor."""
    zarr_format = pyramid.meta.zarr_format if pyramid.meta else 2
    tensorstore_writer._create_zarr_array(
        store_path=str(dest_path / layer_path),
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor_config=pyramid.compressor,
        zarr_format=zarr_format,
        overwrite=False,
        dimension_names=list(pyramid.axes) if zarr_format == 3 else None,
    )


class PyramidIO:
    """High-performance I/O handler for NGFF-compliant OME Zarr Pyramids.
    
    Features:
    - Region-wise processing for memory efficiency
    - Async TensorStore writes for parallelism
    - Queue-based producer-consumer pipeline
    - Multi-threaded readers and writers
    - Automatic garbage collection
    """

    def __init__(self, verbose: bool = True, enable_gc: bool = True):
        """Initialize PyramidIO handler.
        
        Parameters
        ----------
        verbose : bool, optional
            Enable verbose logging. Default is True.
        enable_gc : bool, optional
            Enable garbage collection during writes. Default is True.
        """
        self.verbose = verbose
        self.enable_gc = enable_gc

    def read_pyramid(self,
                     path: Union[str, Path],
                     include_labels: bool = True) -> Pyramid:
        """Read a pyramid from an NGFF-compliant zarr store.

        Parameters
        ----------
        path : str or Path
            Path to the zarr store containing the NGFF pyramid
        include_labels : bool
            Also discover the image's NGFF ``labels/`` collection and populate the
            returned pyramid's `Pyramid.labels` (lazy - only metadata + lazy arrays
            are loaded). Default True; set False to skip the scan.

        Returns
        -------
        Pyramid
            Loaded pyramid with metadata and arrays
        """
        path = Path(path) if isinstance(path, str) else path

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if self.verbose:
            logger.info(f"[PyramidIO] Reading pyramid from: {path}")

        try:
            pyramid = Pyramid()
            pyramid.from_ngff(str(path))

            # remember the on-disk chunking so ops that rechunk for processing can be
            # written back with the ORIGINAL storage chunks by default (see rechunk).
            try:
                base = pyramid.dask_arrays[pyramid.meta.resolution_paths[0]]
                pyramid._storage_chunks = tuple(base.chunksize)
            except Exception:
                pass

            if include_labels:
                self._discover_labels(pyramid, path)

            if self.verbose:
                logger.info(f"[PyramidIO] Loaded {pyramid.nlayers} layers, axes: {pyramid.axes}")

            return pyramid

        except Exception as e:
            logger.error(f"[PyramidIO] Failed to read pyramid: {str(e)}")
            raise

        finally:
            gc.collect()

    def read_labels(self,
                    path: Union[str, Path],
                    name: Optional[str] = None) -> Pyramid:
        """Read a single OME-Zarr label image as a `Pyramid`, with its
        ``image-label`` metadata parsed (surfaced via `pyr.meta.image_label`).

        `path` may point directly at a label-image group, or at a source image with
        `name` given - then ``<path>/labels/<name>`` is read. The label is a normal
        multiscale pyramid, so every `Pyramid` operation works on it.
        """
        path = Path(path) if isinstance(path, str) else path
        if name is not None:
            path = path / "labels" / name
        # a label image has no nested labels/; skip discovery
        return self.read_pyramid(path, include_labels=False)

    def _discover_labels(self, pyramid: Pyramid, path: Path) -> None:
        """Populate ``pyramid._labels`` from the image's NGFF ``labels/`` group, if
        present. Best-effort and lazy: failures leave the collection empty."""
        labels_dir = Path(path) / "labels"
        if not labels_dir.exists():
            return
        try:
            grp = zarr.open_group(str(labels_dir), mode="r")
            names = _read_label_names(grp)
        except Exception:
            return
        for nm in names:
            try:
                pyramid._labels[nm] = self.read_labels(labels_dir / nm)
            except Exception:
                continue

    def write_pyramid(self,
                      pyramid: Pyramid,
                      path: Union[str, Path],
                      layers: Optional[List[str]] = None,
                      overwrite: bool = False,
                      max_workers: int = 4,
                      region_size_mb: float = 8.0,
                      gc_interval: float = 15.0,
                      use_multiprocessing: bool = False,
                      backend: str = 'sync',
                      compressor: Optional[str] = None,
                      compressor_params: Optional[dict] = None,
                      max_concurrency: int = 4,
                      num_readers: Optional[int] = None,
                      queue_size: Optional[int] = None,
                      max_concurrent_layers: int = 3,
                      chunk_shape=None,
                      chunk_size_mb=None,
                      include_labels: bool = True,
                      labels_group_path: Optional[Union[str, Path]] = None,
                      label_name: Optional[str] = None,
                      storage_options: Optional[dict] = None) -> None:
        """Write a pyramid using region-wise processing.

        Handles BOTH image and label pyramids: a label image is detected via
        ``pyramid.meta.is_label`` (it carries an ``image-label`` block) and its
        metadata is attached automatically; pass ``labels_group_path`` + ``label_name``
        to also register it under a parent ``labels/`` group (the NGFF convention).
        An IMAGE pyramid's attached `labels` (from ``add_image_label``) are written
        recursively into ``<path>/labels/<name>``.

        Storage chunking (decoupled from any PROCESSING tile size an op imposed):
        pass ``chunk_shape`` (per-axis tuple, or a ``{level: tuple}`` dict) or
        ``chunk_size_mb`` (scalar / per-level sequence / ``{level: mb}``) to set the
        on-disk chunk shape - see :meth:`Pyramid.rechunk`. When neither is given, the
        pyramid's recorded ``_storage_chunks`` (from read) is restored, so a
        read -> process -> write round-trip keeps the original chunking even if an op
        rechunked the dask arrays to its tile size.

        Two parallelism strategies are available:

        * ``use_multiprocessing=False`` (default): one reader thread per layer
          feeds a shared pool of *max_workers* writer threads.  Works with any
          input array type (zarr, dask, numpy).

        * ``use_multiprocessing=True``: one OS process per layer, each with its
          own reader + writer threads.  Provides complete event-loop isolation
          and true CPU-level parallelism across layers.  Requires the source
          pyramid to be backed by a zarr store on disk.

        Parameters
        ----------
        pyramid : Pyramid
            Pyramid instance to write
        path : str or Path
            Output path for the zarr store
        layers : list of str, optional
            Specific layer indices to write. If None, writes all layers.
        overwrite : bool, optional
            If True, overwrite existing store. Default is False.
        max_workers : int, optional
            Total worker budget (threads or processes+threads). Default is 4.
        region_size_mb : float, optional
            Target region size in MB. Default is 8.0.
        gc_interval : float, optional
            Seconds between GC runs (threading path only). Default is 15.0.
        use_multiprocessing : bool, optional
            Use process-per-layer parallelism. Default is False.
        backend : str, optional
            ``'sync'`` (default) uses the threaded zarr-array writer above.
            ``'tensorstore'`` uses the async TensorStore producer-consumer
            writer (see :func:`write_pyramid_async`).
        compressor, compressor_params, max_concurrency, num_readers,
        queue_size, max_concurrent_layers : optional
            Only used when ``backend='tensorstore'`` - see
            :func:`write_pyramid_async`. ``compressor=None`` (default) uses
            ``pyramid.compressor`` (the input's own codec, or blosc for
            non-zarr-backed pyramids).

        Notes
        -----
        Regardless of ``backend``, the output zarr format (v2/v3) and array
        compressor are taken from ``pyramid.meta.zarr_format`` /
        ``pyramid.compressor`` - i.e. the metadata carried by the in-memory
        ``Pyramid`` object, which by default mirrors the source it was read
        from.
        """
        if pyramid.meta is None:
            raise ValueError("Pyramid not initialized")

        # REMOTE object store (https:// / s3://): keep the URL as a STRING (Path would
        # mangle 'https://' -> 'https:\\') and use the SAME sync writer as local - its
        # arrays are created on the group and its region writes go through zarr, so an
        # s3fs/fsspec-backed group works identically. (The TensorStore backend uses a
        # LOCAL 'file' kvstore and cannot write to S3; multiprocessing can't share the
        # remote store.) One writer, one source of truth.
        remote = _is_remote_store(path)
        if remote:
            if storage_options is not None:   # e.g. {'anon': True} for a public bucket
                tensorstore_writer.set_s3_storage_options(**storage_options)
            if getattr(pyramid, '_downscale_plan', None) is not None and layers is None:
                raise NotImplementedError(
                    "writing a DEFERRED downscale() pyramid to a remote store is not "
                    "supported (it re-reads the base from the store; S3 read is not "
                    "implemented). Use downscale(defer=False), or write a single level."
                )
            backend = 'sync'
            use_multiprocessing = False
        else:
            path = Path(path) if isinstance(path, str) else path

        # capture any attached labels NOW (top-level image write only): the storage
        # rechunk below rebinds `pyramid` to a clone that does not carry `_labels`.
        # They are written into `<path>/labels/<name>` after the image (see
        # `_write_attached_labels`). `layers is not None` = an internal sub-write.
        write_lbls = layers is None and include_labels and bool(getattr(pyramid, '_labels', None))
        attached_labels = dict(getattr(pyramid, '_labels', {})) if write_lbls else {}

        # LABEL image? detect via meta.is_label and capture its image-label block now
        # (so it survives the rechunk rebind below); attached after the arrays. Register
        # under a parent labels/ group FIRST, so the nested group is created inside it.
        label_block = pyramid.meta.image_label if layers is None else None
        label_version, label_zarr_format = pyramid.meta.version, pyramid.meta.zarr_format
        if layers is None and labels_group_path is not None and label_name is not None:
            lg = _open_store_group(labels_group_path, label_zarr_format, mode="a")
            _register_label_name(lg, label_name, label_version)

        # A DEFERRED-downscale pyramid (from `Pyramid.downscale(defer=True)`) carries
        # only level 0 plus a plan; expand it PROGRESSIVELY FROM DISK so an expensive
        # lazy base is computed once, not re-run per output level. (Skip when a
        # specific `layers` subset is requested - that path is used internally here.)
        plan = getattr(pyramid, '_downscale_plan', None)
        if plan is not None and layers is None:
            self._write_with_plan(
                pyramid, path, plan, overwrite=overwrite,
                chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb,
                max_workers=max_workers, region_size_mb=region_size_mb,
                gc_interval=gc_interval, use_multiprocessing=use_multiprocessing,
                backend=backend, compressor=compressor,
                compressor_params=compressor_params, max_concurrency=max_concurrency,
                num_readers=num_readers, queue_size=queue_size,
                max_concurrent_layers=max_concurrent_layers,
            )
            self._write_side_metadata(
                path, label_block=label_block, label_version=label_version,
                label_zarr_format=label_zarr_format, attached_labels=attached_labels,
                overwrite=overwrite, backend=backend, compressor=compressor,
                compressor_params=compressor_params)
            return

        # dyna_zarr backend: the pyramid's layers are DynamicArrays (a memory-bounded,
        # pull-based op chain). Write each level via dyna_zarr.io.write, which streams the
        # whole read -> op-chain -> write pipeline region by region. Reuses the same group +
        # NGFF-metadata machinery as the sync path (connect_to_group / save_changes).
        if layers is None and _is_dyna_pyramid(pyramid):
            if remote:
                raise NotImplementedError(
                    "writing a dyna_zarr-backed pyramid to a remote store is not supported "
                    "yet; write locally (the dyna backend's remote write is still local-only)")
            self._write_pyramid_dyna(
                pyramid, path, overwrite=overwrite, chunk_shape=chunk_shape,
                region_size_mb=region_size_mb, max_workers=max_workers)
            self._write_side_metadata(
                path, label_block=label_block, label_version=label_version,
                label_zarr_format=label_zarr_format, attached_labels=attached_labels,
                overwrite=overwrite, backend=backend, compressor=compressor,
                compressor_params=compressor_params)
            return

        # apply the storage chunking before writing (explicit spec, else restore the
        # recorded on-disk chunks). Skipped for internal layer-subset writes, which
        # are already rechunked by _write_with_plan.
        if layers is None:
            if chunk_shape is not None or chunk_size_mb is not None:
                pyramid = pyramid.rechunk(chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb)
            elif getattr(pyramid, '_storage_chunks', None) is not None:
                pyramid = pyramid.rechunk()

        if backend == 'tensorstore':
            self.write_pyramid_tensorstore(
                pyramid=pyramid,
                path=path,
                layers=layers,
                overwrite=overwrite,
                compressor=compressor,
                compressor_params=compressor_params,
                region_size_mb=region_size_mb,
                max_concurrency=max_concurrency,
                num_readers=num_readers,
                queue_size=queue_size,
                gc_interval=gc_interval,
                max_concurrent_layers=max_concurrent_layers,
            )
            self._write_side_metadata(
                path, label_block=label_block, label_version=label_version,
                label_zarr_format=label_zarr_format, attached_labels=attached_labels,
                overwrite=overwrite, backend=backend, compressor=compressor,
                compressor_params=compressor_params)
            return

        if self.verbose:
            logger.info(f"[PyramidIO] Writing pyramid to: {path}")
            logger.info(f"[PyramidIO] Pyramid has {pyramid.nlayers} layers")
        
        try:
            # Determine which layers to write
            if layers is None:
                layers_to_write = pyramid.meta.resolution_paths
            else:
                layers_to_write = [str(layer) for layer in layers]
            
            # Create output zarr group, honoring the pyramid's zarr format (v2/v3).
            # Remote paths resolve to an s3fs/fsspec mapping so the group (and every
            # array + region write on it) targets the object store.
            group_store = tensorstore_writer.wrap_output_path(str(path)) if remote else str(path)
            zarr_group = tensorstore_writer._zarr_group(
                group_store, overwrite=overwrite, zarr_format=pyramid.meta.zarr_format
            )

            if use_multiprocessing:
                self._write_all_layers_multiprocessing(
                    pyramid=pyramid,
                    zarr_group=zarr_group,
                    layers_to_write=layers_to_write,
                    dest_path=path,
                    max_workers=max_workers,
                    region_size_mb=region_size_mb,
                )
            else:
                self._write_all_layers_unified(
                    pyramid=pyramid,
                    zarr_group=zarr_group,
                    layers_to_write=layers_to_write,
                    dest_path=path,
                    max_workers=max_workers,
                    region_size_mb=region_size_mb,
                    gc_interval=gc_interval,
                )
            
            # Write NGFF metadata - always write to the destination group, even
            # if the in-memory metadata hasn't changed since it was read.
            pyramid.meta.connect_to_group(zarr_group)
            pyramid.meta._pending_changes = True
            pyramid.meta.save_changes()

            # Allow any pending async tasks to complete
            time.sleep(0.1)
            gc.collect()
            
            if self.verbose:
                logger.info(f"[PyramidIO] Successfully wrote pyramid to: {path}")

        except Exception as e:
            logger.error(f"[PyramidIO] Failed to write pyramid: {str(e)}")
            raise

        finally:
            gc.collect()

        # attach the label image-label block + write any attached labels (normal exit)
        self._write_side_metadata(
            path, label_block=label_block, label_version=label_version,
            label_zarr_format=label_zarr_format, attached_labels=attached_labels,
            overwrite=overwrite, backend=backend, compressor=compressor,
            compressor_params=compressor_params)

    def _write_pyramid_dyna(self, pyramid, path, *, overwrite, chunk_shape,
                            region_size_mb, max_workers):
        """Write a dyna_zarr-backed pyramid: each level is streamed to disk via
        dyna_zarr.io.write (memory-bounded), then the shared NGFF group metadata is written.
        Local stores only for now."""
        from dyna_zarr import io as dyna_io
        path = Path(path) if isinstance(path, str) else path
        zf = pyramid.meta.zarr_format
        zarr_group = tensorstore_writer._zarr_group(str(path), overwrite=overwrite, zarr_format=zf)
        das = pyramid.dynamic_arrays
        for p in pyramid.meta.resolution_paths:
            arr = das[str(p)]
            chunks = _resolve_dyna_chunks(chunk_shape, p, arr)
            dyna_io.write(arr, str(path / str(p)),
                          chunks=tuple(chunks) if chunks is not None else None,
                          zarr_format=int(zf), region_size_mb=region_size_mb,
                          max_workers=max_workers)
        # write the multiscales / omero metadata onto the group (same as the sync path)
        pyramid.meta.connect_to_group(zarr_group)
        pyramid.meta._pending_changes = True
        pyramid.meta.save_changes()

    def _write_side_metadata(self, path, *, label_block, label_version, label_zarr_format,
                             attached_labels: dict, overwrite: bool, backend: str = 'sync',
                             compressor=None, compressor_params=None) -> None:
        """Post-array-write side metadata, shared by every write_pyramid exit:
        (1) if this is a LABEL image, attach its ``image-label`` block to the written
        group (belt-and-suspenders across backends; `save_changes` already writes it on
        the sync path); (2) write any attached image `labels` into ``<path>/labels/<name>``."""
        if label_block is not None:
            group = _open_store_group(path, label_zarr_format, mode="a")
            _attach_image_label(group, label_block, label_version)
        if attached_labels:
            self._write_attached_labels(attached_labels, path, overwrite=overwrite,
                                        backend=backend, compressor=compressor,
                                        compressor_params=compressor_params)

    def _write_attached_labels(self, labels: dict, path, *, overwrite: bool,
                               backend: str = 'sync', compressor=None,
                               compressor_params=None) -> None:
        """Write each entry of a pyramid's `labels` collection as an OME-Zarr label
        image under ``<path>/labels/<name>`` (registering it in the ``labels/`` group),
        carrying each label's own ``image-label`` metadata. The counterpart of
        `read_pyramid`'s label discovery, so an add_image_label -> write -> read round
        trips."""
        if not labels:
            return
        labels_group = _store_join(path, "labels")   # S3-aware (URL join for remote)
        for name, label_pyr in labels.items():
            # unified: write_pyramid detects the label via meta.is_label, attaches its
            # image-label block and registers it in the labels/ group. (label_pyr has no
            # nested `labels`, so this does not recurse further.)
            self.write_pyramid(
                pyramid=label_pyr,
                path=str(_store_join(labels_group, name)),
                labels_group_path=str(labels_group),
                label_name=name,
                overwrite=overwrite,
                backend=backend, compressor=compressor, compressor_params=compressor_params,
            )


    def _write_with_plan(self, pyramid: Pyramid, path: Path, plan: dict, *,
                         overwrite: bool, chunk_shape=None, chunk_size_mb=None,
                         **write_kwargs) -> None:
        """Write a DEFERRED-downscale pyramid progressively from disk.

        1. stream the base (level 0) to the store ONCE (the expensive lazy graph
           runs a single time here);
        2. re-read that on-disk base;
        3. expand the downscale plan on the DISK-backed base (each coarser level is
           a cheap read+downsample of the stored parent, not a recompute of the base);
        4. write the remaining levels (>=1) and the full multiscale metadata.

        Never materialises the whole volume in memory. Reused for both `write_pyramid`
        and `write_labels` (labels downsample with 'simple' = nearest, which the plan
        records)."""
        base_path0 = pyramid.meta.resolution_paths[0]

        def _rechunked(p):
            """Apply the requested storage chunking (explicit, else restore recorded)."""
            if chunk_shape is not None or chunk_size_mb is not None:
                return p.rechunk(chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb)
            if getattr(p, '_storage_chunks', None) is not None:
                return p.rechunk()
            return p

        # 1) write base only (rechunked to storage chunks), plan removed to avoid recursion
        saved = pyramid.__dict__.pop('_downscale_plan', None)
        try:
            base = _rechunked(pyramid)
            self.write_pyramid(base, path, layers=[base_path0], overwrite=overwrite, **write_kwargs)
        finally:
            if saved is not None:
                pyramid._downscale_plan = saved

        # 2) re-read the on-disk base (zarr-backed -> downscaling reads from DISK)
        disk = self.read_pyramid(str(path))

        # 3) expand the plan for the level count / shapes / scales (metadata only)
        dk = {k: plan[k] for k in ('n_layers', 'min_dimension_size', 'scale_factor',
                                   'downscale_method', 'backend', 'smart_scale_factor')}
        full = disk.downscale(defer=False, **dk)
        fpaths = full.meta.resolution_paths
        extra = list(fpaths[1:])
        if not extra:
            return

        # 4) build coarser levels as DASK arrays by downsampling the ON-DISK base
        # (reads the stored base, never the expensive source graph; keeps everything
        # dask so chunking is controllable and there is no tensorstore dtype snag).
        import dask.array as da  # local import
        from ome_zarr_pyramid.utils.scale import mean_downscale, median_downscale, simple_downscale
        method = {'simple': simple_downscale, 'mean': mean_downscale,
                  'median': median_downscale}.get(plan.get('downscale_method', 'simple'), simple_downscale)
        base_da = disk.dask_arrays[disk.meta.resolution_paths[0]]
        ndim = base_da.ndim
        level_arrays = [base_da]
        for i in range(1, len(fpaths)):
            tgt = full.dask_arrays[fpaths[i]].shape
            factor = tuple(max(1, int(round(base_da.shape[a] / tgt[a]))) for a in range(ndim))
            level_arrays.append(method(base_da, scale_factor=factor))

        unit_clean = [u for u in full.meta.unit_list if u is not None]
        full_dask = Pyramid().from_arrays(
            arrays=level_arrays, axis_order=full.meta.axis_order,
            unit_list=unit_clean if unit_clean else None,
            scales=[full.meta.get_scale(p) for p in fpaths],
            version=full.meta.version,
            name=full.meta.multiscales.get('name', 'unnamed') if full.meta.metadata else 'unnamed',
        )
        full_dask._storage_chunks = getattr(pyramid, '_storage_chunks', None)
        full_dask = _rechunked(full_dask)
        # `full_dask` was rebuilt via from_arrays (axes/scales/units/name only), so
        # overlay the SOURCE pyramid's non-dataset metadata (omero channels/colours/
        # windows and any custom top-level attrs). Without this, writing the extra
        # levels re-saves the group attrs and clobbers omero set via set_channels /
        # set_display_range. Keep full_dask's own `multiscales` (all levels, correct
        # scales/translations).
        import copy as _copy
        src_md = pyramid.meta.metadata if pyramid.meta is not None else None
        if src_md and full_dask.meta is not None and full_dask.meta.metadata is not None:
            # mirror the source's non-`multiscales` metadata EXACTLY: add its
            # omero/image-label/custom attrs, and DROP any keys full_dask auto-added
            # that the source lacks (e.g. from_arrays' empty omero on a label image).
            # Keep full_dask's own `multiscales` (all levels + correct scales).
            ms = full_dask.meta.metadata.get('multiscales')
            new_md = {k: _copy.deepcopy(v) for k, v in src_md.items() if k != 'multiscales'}
            if ms is not None:
                new_md['multiscales'] = ms
            full_dask.meta.metadata = new_md
            full_dask.meta._pending_changes = True
        self.write_pyramid(full_dask, path, layers=extra, overwrite=False, **write_kwargs)

    def write_labels(self,
                     label_pyramid: Pyramid,
                     path: Union[str, Path],
                     image_label: Optional[dict] = None,
                     labels_group_path: Optional[Union[str, Path]] = None,
                     label_name: Optional[str] = None,
                     overwrite: bool = False,
                     **write_kwargs) -> None:
        """Write `label_pyramid` as an OME-Zarr label image.

        Thin wrapper over `write_pyramid`, which now handles label images directly
        (detected via `meta.is_label`): it attaches the `image-label` block and, given
        `labels_group_path` + `label_name`, registers the label under the parent
        `labels` group. Prefer ``write_pyramid(label_pyr, path, labels_group_path=...,
        label_name=...)``; this wrapper is kept for convenience and the explicit
        `image_label` override.

        Parameters
        ----------
        label_pyramid : Pyramid
            The (integer) label pyramid to write.
        path : str or Path
            Destination for the label-image group (e.g. `<image>/labels/<name>`).
        image_label : dict, optional
            An explicit `image-label` metadata object (`version`/`colors`/
            `properties`/`source`) to attach. Default None: use the block already on
            the pyramid (`label_pyramid.meta.image_label`, set via
            `Pyramid.set_image_label` / `from_label_arrays`), so
            ``write_labels(label_pyr, path)`` just works. An explicit dict overrides it.
        labels_group_path : str or Path, optional
            Path to the parent `labels` group to register the label in. If given
            with `label_name`, that group is created/updated with the name.
        label_name : str, optional
            Name to register in the parent `labels` group.
        overwrite : bool
            Overwrite an existing label image at `path`.
        **write_kwargs
            Forwarded to `write_pyramid` (backend, compressor, workers, ...).
        """
        if label_pyramid.meta is None:
            raise ValueError("label_pyramid not initialized")
        # An explicit `image_label` dict overrides the block on the pyramid: stamp it
        # onto a metadata-only clone (do not mutate the caller's pyramid).
        if image_label is not None:
            label_pyramid = label_pyramid._clone_metadata_only()
            if label_pyramid.meta is not None and label_pyramid.meta.metadata is not None:
                label_pyramid.meta.metadata['image-label'] = image_label
                label_pyramid.meta._pending_changes = True
        # write_pyramid detects the label via meta.is_label, attaches its image-label
        # block and (with labels_group_path + label_name) registers it in the labels/ group.
        self.write_pyramid(label_pyramid, path=path, overwrite=overwrite,
                           labels_group_path=labels_group_path, label_name=label_name,
                           **write_kwargs)

    async def write_pyramid_async(self,
                                   pyramid: Pyramid,
                                   path: Union[str, Path],
                                   layers: Optional[List[str]] = None,
                                   overwrite: bool = False,
                                   compressor: Optional[str] = None,
                                   compressor_params: Optional[dict] = None,
                                   region_size_mb: float = 8.0,
                                   max_concurrency: int = 4,
                                   num_readers: Optional[int] = None,
                                   queue_size: Optional[int] = None,
                                   gc_interval: float = 15.0,
                                   max_concurrent_layers: int = 3) -> None:
        """Write a pyramid using the TensorStore async producer-consumer writer.

        Each layer is written via
        :func:`ome_zarr_pyramid.core.tensorstore_writer.write_with_queue_async`,
        with up to *max_concurrent_layers* layers written concurrently.

        ``compressor=None`` (default) uses ``pyramid.compressor`` - the
        codec carried over from the input (or blosc, for non-zarr-backed
        pyramids). Pass an explicit ``compressor``/``compressor_params`` to
        override it for this write only.
        """
        if pyramid.meta is None:
            raise ValueError("Pyramid not initialized")

        # keep object-store URLs as strings (Path would mangle 'https://' -> 'https:\\')
        if not _is_remote_store(path):
            path = Path(path) if isinstance(path, str) else path

        if compressor is None:
            if pyramid.compressor is not None:
                compressor = pyramid.compressor.name
                if compressor_params is None:
                    compressor_params = pyramid.compressor.params
            else:
                compressor = 'blosc'
        if compressor_params is None:
            compressor_params = {}

        if layers is None:
            layers_to_write = pyramid.meta.resolution_paths
        else:
            layers_to_write = [str(layer) for layer in layers]

        zarr_format = pyramid.meta.zarr_format

        if self.verbose:
            logger.info(f"[PyramidIO] Writing pyramid to: {path} (backend=tensorstore)")
            logger.info(f"[PyramidIO] Writing layers: {layers_to_write}")

        group_store = tensorstore_writer.wrap_output_path(str(path)) if _is_remote_store(path) else str(path)
        tensorstore_writer._zarr_group(group_store, overwrite=overwrite, zarr_format=zarr_format)

        semaphore = asyncio.Semaphore(max(1, min(max_concurrent_layers, len(layers_to_write))))

        async def _write_layer(layer_path: str) -> None:
            array = pyramid.layers[layer_path]
            async with semaphore:
                await tensorstore_writer.write_with_queue_async(
                    arr=array,
                    output_path=str(_store_join(path, layer_path)),
                    output_chunks=get_chunk_shape(array),
                    zarr_format=zarr_format,
                    dtype=array.dtype,
                    dimension_names=list(pyramid.axes),
                    compressor=compressor,
                    compressor_params=compressor_params,
                    region_size_mb=region_size_mb,
                    max_concurrency=max_concurrency,
                    num_readers=num_readers,
                    queue_size=queue_size,
                    gc_interval=gc_interval,
                    overwrite=overwrite,
                    verbose=self.verbose,
                )

        results = await asyncio.gather(
            *(_write_layer(layer_path) for layer_path in layers_to_write),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                raise result

        zarr_group = _open_store_group(path, zarr_format, mode='r+')
        pyramid.meta.connect_to_group(zarr_group)
        pyramid.meta._pending_changes = True
        pyramid.meta.save_changes()

        gc.collect()

        if self.verbose:
            logger.info(f"[PyramidIO] Successfully wrote pyramid to: {path}")

    def write_pyramid_tensorstore(self,
                                   pyramid: Pyramid,
                                   path: Union[str, Path],
                                   **kwargs) -> None:
        """Synchronous wrapper around :func:`write_pyramid_async`."""
        asyncio.run(self.write_pyramid_async(pyramid, path, **kwargs))

    def _write_all_layers_unified(self,
                                   pyramid: Pyramid,
                                   zarr_group: zarr.Group,
                                   layers_to_write: List[str],
                                   dest_path: Union[str, Path],
                                   max_workers: int = 4,
                                   region_size_mb: float = 8.0,
                                   gc_interval: float = 15.0) -> None:
        """Write all layers concurrently using a unified queue-based pipeline.
        
        All regions from all layers are placed in a single queue and processed
        together by reader/writer threads, avoiding async task accumulation.
        """
        # Setup zarr arrays and collect region info for all layers
        layer_info = {}
        total_regions = 0
        
        for layer_path in layers_to_write:
            array = pyramid.layers[layer_path]
            shape = tuple(int(s) for s in array.shape)
            # normalize to a numpy dtype: TensorStore-backed layers (e.g. from
            # `Pyramid.downscale()` on a read pyramid) expose a ts dtype, which
            # zarr.create / np.dtype can't consume directly.
            dtype = tensorstore_writer._normalize_dtype(array.dtype, array)
            if hasattr(array, 'chunksize'):
                # dask array: `.chunks` is a tuple of per-axis chunk-size tuples;
                # `.chunksize` is the (uniform) tuple of ints we want.
                chunks = tuple(int(c) for c in array.chunksize)
            elif hasattr(array, 'chunks'):
                # zarr array: `.chunks` is already a tuple of ints.
                chunks = tuple(int(c) for c in array.chunks)
            else:
                chunks = tuple([256] * len(shape))
            
            # Create the layer array ON THE GROUP, so it lands in the group's store -
            # LOCAL or an s3fs/fsspec mapping alike (single store-agnostic writer).
            if layer_path not in zarr_group:
                zf = pyramid.meta.zarr_format if pyramid.meta else 2
                tensorstore_writer.create_group_array(
                    zarr_group, layer_path, shape, chunks, dtype, zf,
                    pyramid.compressor, list(pyramid.axes) if zf == 3 else None)

            zarr_array = zarr_group[layer_path]
            
            # Compute region shape
            region_shape = _compute_region_shape_for_layer(
                shape, chunks, region_size_mb, dtype
            )
            
            # Generate region slices for this layer
            region_slices = []
            for start_indices in itertools.product(*[range(0, int(s), int(r))
                                                       for s, r in zip(shape, region_shape)]):
                slices = tuple(
                    slice(start, min(start + int(region_shape[i]), int(shape[i])))
                    for i, start in enumerate(start_indices)
                )
                region_slices.append(slices)
            
            layer_info[layer_path] = {
                'array': array,
                'zarr_array': zarr_array,
                'shape': shape,
                'dtype': dtype,
                'region_slices': region_slices,
                'regions_count': len(region_slices)
            }
            total_regions += len(region_slices)
            
            if self.verbose:
                logger.info(f"[PyramidIO] Layer {layer_path}: {len(region_slices)} regions")
        
        if self.verbose:
            logger.info(f"[PyramidIO] Writing {total_regions} total regions across {len(layers_to_write)} layers")
        
        # Single unified queue for all regions from all layers
        queue: Queue = Queue(maxsize=max_workers * 4)
        state: Dict = {
            'error': None,
            'regions_written': 0,
            'start_time': time.time(),
        }
        shutdown_flag = threading.Event()
        
        # Counter tracking how many per-layer readers have finished.
        # When the last reader finishes it sends sentinels to stop all writers.
        readers_finished  = [0]
        readers_lock      = threading.Lock()
        num_layer_readers = len(layers_to_write)

        def make_layer_reader(lp: str, info: dict):
            """Factory: returns a reader function scoped to one layer."""
            def _reader():
                try:
                    array = info['array']
                    for region_slice in info['region_slices']:
                        if state.get('error'):
                            break
                        if isinstance(array, da.Array):
                            data = array[region_slice].compute()
                        else:
                            data = np.asarray(array[region_slice])
                        queue.put((lp, region_slice, data))
                except Exception as e:
                    logger.error(f"[LayerReader-{lp}] Error: {str(e)}")
                    state['error'] = e
                finally:
                    with readers_lock:
                        readers_finished[0] += 1
                        if readers_finished[0] == num_layer_readers:
                            # Last reader done — unblock all writers
                            for _ in range(max_workers):
                                queue.put(None)
            return _reader

        def unified_writer_thread():
            """Consumer: write regions to their respective zarr arrays."""
            try:
                while not shutdown_flag.is_set():
                    if state.get('error'):
                        break
                    
                    try:
                        item = queue.get(timeout=1.0)
                    except Empty:
                        continue
                    
                    if item is None:
                        break
                    
                    layer_path, region_slice, data = item
                    zarr_array = layer_info[layer_path]['zarr_array']
                    
                    # Write to zarr array (async)
                    zarr_array[region_slice] = data
                    state['regions_written'] += 1
                    
                    if self.verbose and state['regions_written'] % max(1, total_regions // 20) == 0:
                        logger.info(f"[UnifiedWriter] Wrote region {state['regions_written']}/{total_regions}")
                    
                    queue.task_done()
            
            except Exception as e:
                logger.error(f"[UnifiedWriter] Error: {str(e)}")
                state['error'] = e
        
        # One reader per layer (concurrent reads across all layers) + shared writers
        readers = [
            threading.Thread(
                target=make_layer_reader(lp, info),
                daemon=True,
                name=f"LayerReader-{lp}"
            )
            for lp, info in layer_info.items()
        ]
        writers = [
            threading.Thread(target=unified_writer_thread, daemon=True, name=f"UnifiedWriter-{i}")
            for i in range(max_workers)
        ]

        for r in readers:
            r.start()
        for w in writers:
            w.start()
        
        # Wait for completion
        last_gc = time.time()
        while state['regions_written'] < total_regions:
            if state.get('error'):
                raise state['error']
            
            # Periodic GC
            now = time.time()
            if self.enable_gc and now - last_gc > gc_interval:
                gc.collect()
                last_gc = now
            
            time.sleep(0.1)
        
        shutdown_flag.set()
        
        # Wait for all threads to finish
        for r in readers:
            r.join(timeout=10)
        for w in writers:
            w.join(timeout=10)
        
        if state.get('error'):
            raise state['error']
        
        # Flush all zarr arrays to ensure async compression completes
        for layer_path, info in layer_info.items():
            zarr_array = info['zarr_array']
            try:
                if hasattr(zarr_array.store, 'map') and hasattr(zarr_array.store.map, 'flush'):
                    zarr_array.store.map.flush()
            except:
                pass
        
        gc.collect()
        
        elapsed = time.time() - state['start_time']
        throughput = total_regions / elapsed if elapsed > 0 else 0
        
        if self.verbose:
            logger.info(f"[PyramidIO] Wrote all {total_regions} regions in {elapsed:.1f}s ({throughput:.1f} regions/s)")

    # ------------------------------------------------------------------
    # Multiprocessing path (one OS process per layer)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_store_path(zarr_group: zarr.Group) -> Optional[str]:
        """Return the filesystem root of a zarr group's store, or None."""
        store = zarr_group.store
        # zarr v3 LocalStore
        if hasattr(store, 'root'):
            return str(store.root)          # type: ignore[attr-defined]
        # zarr v2 DirectoryStore
        if hasattr(store, 'path'):
            return str(store.path)          # type: ignore[attr-defined]
        # zarr v2 FSStore
        if hasattr(store, 'dir_path'):
            try:
                return str(store.dir_path())  # type: ignore[attr-defined]
            except Exception:
                pass
        return None

    def _write_all_layers_multiprocessing(self,
                                          pyramid: Pyramid,
                                          zarr_group: zarr.Group,
                                          layers_to_write: List[str],
                                          dest_path: Union[str, Path],
                                          max_workers: int = 4,
                                          region_size_mb: float = 8.0) -> None:
        """Write all layers in parallel using one OS process per layer.

        Each process re-opens its own zarr handles, giving complete event-loop
        isolation and true CPU parallelism across layers.  Within each process,
        *threads_per_layer* writer threads drain a per-layer read queue.

        Requires the source pyramid to be backed by an accessible zarr store on
        disk (i.e. ``pyramid.gr`` must not be None).  For in-memory pyramids
        use ``use_multiprocessing=False``.
        """
        if pyramid.gr is None:
            raise ValueError(
                "Multiprocessing mode requires a disk-backed pyramid (pyramid.gr is None). "
                "Call write_pyramid(..., use_multiprocessing=False) for in-memory pyramids."
            )

        source_zarr_path = self._get_store_path(pyramid.gr)
        if source_zarr_path is None:
            raise ValueError(
                "Cannot determine source zarr path for multiprocessing. "
                "Use write_pyramid(..., use_multiprocessing=False) instead."
            )

        dest_zarr_path = str(dest_path)
        num_layers      = len(layers_to_write)
        # Distribute worker budget evenly; every layer gets at least 1 writer thread
        threads_per_layer = max(1, max_workers // num_layers)

        start_time   = time.time()
        total_regions = 0
        tasks: List[dict] = []

        for layer_path in layers_to_write:
            array  = pyramid.layers[layer_path]
            shape  = tuple(int(s) for s in array.shape)
            dtype  = tensorstore_writer._normalize_dtype(array.dtype, array)  # ts dtype -> np.dtype
            chunks = tuple(int(c) for c in getattr(array, 'chunks', tuple([256] * len(shape))))

            # Pre-create destination array in main process (metadata write; safe)
            if layer_path not in zarr_group:
                _create_layer_array(Path(dest_path), layer_path, shape, chunks, dtype, pyramid)

            region_shape = _compute_region_shape_for_layer(shape, chunks, region_size_mb, dtype)

            # Serialise slices as plain [[start,stop],...] lists (slices aren't picklable)
            region_slices_raw = [
                [[int(start_indices[i]),
                  int(min(start_indices[i] + int(region_shape[i]), int(shape[i])))]
                 for i in range(len(shape))]
                for start_indices in itertools.product(
                    *[range(0, int(s), int(r)) for s, r in zip(shape, region_shape)]
                )
            ]
            total_regions += len(region_slices_raw)

            if self.verbose:
                logger.info(f"[PyramidIO] Layer {layer_path}: {len(region_slices_raw)} regions")

            tasks.append({
                'source_zarr_path':  source_zarr_path,
                'source_layer_path': layer_path,
                'dest_zarr_path':    dest_zarr_path,
                'dest_layer_path':   layer_path,
                'region_slices_raw': region_slices_raw,
                'threads_per_layer': threads_per_layer,
                'verbose':           self.verbose,
            })

        if self.verbose:
            logger.info(
                f"[PyramidIO] Writing {total_regions} total regions across "
                f"{num_layers} layers using {num_layers} processes × "
                f"{threads_per_layer} threads each"
            )

        with ProcessPoolExecutor(max_workers=num_layers) as executor:
            futures = {executor.submit(_write_layer_process_worker, t): t['dest_layer_path']
                       for t in tasks}
            for fut in as_completed(futures):
                layer_path, written, total = fut.result()   # propagates exceptions
                if self.verbose:
                    logger.info(f"[PyramidIO] Layer {layer_path}: wrote {written}/{total} regions")

        elapsed    = time.time() - start_time
        throughput = total_regions / elapsed if elapsed > 0 else 0
        if self.verbose:
            logger.info(
                f"[PyramidIO] Wrote all {total_regions} regions in "
                f"{elapsed:.1f}s ({throughput:.1f} regions/s)"
            )


class IO:
    """Convenience wrapper for high-performance pyramid I/O operations."""

    def read_pyramid(self, path: Union[str, Path], include_labels: bool = True) -> Pyramid:
        """Read a pyramid from a path. `include_labels` also loads the image's NGFF
        ``labels/`` collection into `Pyramid.labels` (lazy). See
        :meth:`PyramidIO.read_pyramid`."""
        return PyramidIO().read_pyramid(path, include_labels=include_labels)

    def read_labels(self, path: Union[str, Path], name: Optional[str] = None) -> Pyramid:
        """Read a single OME-Zarr label image as a `Pyramid` (with ``image-label``
        metadata). See :meth:`PyramidIO.read_labels`."""
        return PyramidIO().read_labels(path, name=name)

    def write_pyramid(self,
                      pyramid: Pyramid,
                      path: Union[str, Path],
                      layers: Optional[List[str]] = None,
                      overwrite: bool = False,
                      max_workers: int = 4,
                      region_size_mb: float = 8.0,
                      use_multiprocessing: bool = False,
                      backend: str = 'sync',
                      **kwargs) -> None:
        """Write a pyramid to a path using high-performance async processing.

        ``backend='tensorstore'`` dispatches to the async TensorStore writer;
        any extra ``**kwargs`` (``compressor``, ``compressor_params``,
        ``max_concurrency``, ``num_readers``, ``queue_size``,
        ``max_concurrent_layers``, ``gc_interval``) are forwarded to it.
        """
        return PyramidIO().write_pyramid(
            pyramid=pyramid,
            path=path,
            layers=layers,
            overwrite=overwrite,
            max_workers=max_workers,
            region_size_mb=region_size_mb,
            use_multiprocessing=use_multiprocessing,
            backend=backend,
            **kwargs,
        )

    def write_labels(self,
                     label_pyramid: Pyramid,
                     path: Union[str, Path],
                     image_label: Optional[dict] = None,
                     labels_group_path: Optional[Union[str, Path]] = None,
                     label_name: Optional[str] = None,
                     overwrite: bool = False,
                     **kwargs) -> None:
        """Write a label pyramid as an OME-Zarr label image (image-label metadata
        + optional registration in a parent `labels` group). See
        :meth:`PyramidIO.write_labels`."""
        return PyramidIO().write_labels(
            label_pyramid=label_pyramid,
            path=path,
            image_label=image_label,
            labels_group_path=labels_group_path,
            label_name=label_name,
            overwrite=overwrite,
            **kwargs,
        )
