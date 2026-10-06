"""High-performance I/O module for OME Zarr Pyramids using region-wise processing.

Implements async vectorized writes with TensorStore, queue-based producer-consumer
pipelines, and threading for optimal performance on multi-resolution datasets.
Based on the dyna_zarr architecture for efficient large-scale I/O.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import gc
import itertools
import logging
import os
import threading
import time
import warnings
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorstore as ts
import zarr

from ome_zarr_pyramid.core import tensorstore_writer
from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.utils.array_utils import get_array_chunks, get_chunk_shape
from ome_zarr_pyramid.utils.logging_config import get_logger
from ome_zarr_pyramid.utils.optional_deps import is_dask_array, is_installed, require
from ome_zarr_pyramid.utils.storage_utils import make_kvstore

if TYPE_CHECKING:
    import dask.array as da

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
    """True for a URL (s3://, gs://, az://, https://, ...) that must NOT be treated as a
    local filesystem `Path` (which would mangle it). ``file://`` is local."""
    s = str(path)
    return "://" in s and not s.startswith("file://")


def _is_legacy_https(path) -> bool:
    """ozp's original remote convention: ``https://<s3-endpoint>/<bucket>/<key>`` with
    s3fs-style storage_options, written by the 'sync' engine. Kept for existing callers;
    the dyna engine takes ``s3://bucket/key`` with obstore-style options instead."""
    return str(path).startswith("https://")


def _object_store_group(url: str, zarr_format: int, mode: str,
                        storage_options: Optional[dict] = None) -> zarr.Group:
    """A zarr group on an object store, through obstore - with the SAME storage_options
    names dyna uses for the arrays (``endpoint``, ``region``, ``skip_signature``,
    ``client_options``, ...; credentials from the standard environment variables)."""
    try:
        import obstore
    except ImportError as exc:
        raise ImportError("remote OME-Zarr (s3://, gs://, az://) needs obstore: "
                          "pip install 'ome_zarr_pyramid[remote]'") from exc
    from zarr.storage import ObjectStore
    store = ObjectStore(obstore.store.from_url(url, **(storage_options or {})),
                        read_only=(mode == "r"))
    return zarr.open_group(store, mode=mode, zarr_format=zarr_format)


_ENGINES = ('auto', 'dyna', 'sync', 'tensorstore')


def _resolve_engine(pyramid, backend, *, remote, use_multiprocessing, path=None) -> str:
    """The write engine for ``write_pyramid(backend=...)``.

    ``'auto'`` picks dyna wherever it can stream the pyramid. An EXPLICIT engine is
    honoured or refused, never swapped for another (fail loudly)."""
    if backend not in _ENGINES:
        raise ValueError(f"backend must be one of {_ENGINES}, got {backend!r}")
    has_dask = pyramid._has_dask_layer()
    if use_multiprocessing and has_dask and backend in ('auto', 'dyna'):
        raise ValueError(
            "use_multiprocessing cannot write dask-backed levels (their graphs are not "
            "shipped to worker processes; dask parallelises them itself). Drop "
            "use_multiprocessing.")
    legacy_https = remote and _is_legacy_https(path)
    if backend == 'auto':
        # the legacy https:// convention (s3fs options) keeps the sync writer; every
        # other target - local, s3://, gs://, az:// - goes through dyna
        engine = 'sync' if legacy_https else 'dyna'
        if use_multiprocessing and engine != 'dyna':
            raise ValueError(
                "use_multiprocessing runs on the dyna engine, which takes s3:// URLs with "
                "storage_options={'endpoint': ...}, not the legacy https:// form. Drop "
                "use_multiprocessing, or use the s3:// form.")
        return engine
    if backend == 'dyna':
        if legacy_https:
            raise ValueError(
                "the dyna engine writes object stores as s3://bucket/key (or gs://, az://) "
                "with obstore-style storage_options, e.g. {'endpoint': 'https://host'}; the "
                "https://host/bucket/key form is the legacy 'sync' convention. Use the "
                "s3:// form, or backend='sync'.")
        return 'dyna'
    if use_multiprocessing:
        raise ValueError(
            f"use_multiprocessing runs on the dyna engine (one process per level); "
            f"backend={backend!r} has no multiprocessing mode. Use backend='auto' or "
            f"'dyna', or drop use_multiprocessing.")
    if backend == 'tensorstore' and remote:
        raise ValueError(
            "backend='tensorstore' writes local stores only. Use backend='auto' or "
            "'dyna' for a remote target.")
    if backend == 'sync' and remote and not legacy_https:
        raise ValueError(
            f"backend='sync' writes remote stores only in the legacy https://host/bucket/key "
            f"form; {str(path)!r} needs the dyna engine (backend='auto' or 'dyna').")
    return backend


def _dyna_write_level(arr, output_path: str, kwargs: dict) -> str:
    """Write one level with ``dyna_zarr.io.write``; returns what it reported on stdout.

    Module-level so a spawned worker process can import it (Windows uses spawn)."""
    import contextlib
    import io as _stdio
    from dyna_zarr import io as dyna_io
    buf = _stdio.StringIO()
    with contextlib.redirect_stdout(buf):    # dyna reports grid decisions on stdout
        dyna_io.write(arr, output_path, **kwargs)
    return buf.getvalue()


def _dask_write_level(x, output_path: str, kwargs: dict) -> str:
    """The dask pump: push a dask level into a dyna sink (``io.create_sink``).

    dask schedules its own graph (threaded, ``max_workers`` threads); dyna creates and
    owns the output with the same settings io.write would use. The graph is rechunked to
    the sink's write unit (chunk, or shard when sharded), so every concurrent write is a
    whole unit. Measured on 1.2 GB: 1.24 s, vs 3.5 s into a zarr-python array and 5.2 s
    through the 'sync' writer (reports/write_audit.md)."""
    import dask
    import dask.array as da
    from dyna_zarr import io as dyna_io
    sink_kw = {k: v for k, v in kwargs.items() if k not in ('region_size_mb', 'max_workers')}
    sink = dyna_io.create_sink(output_path, x.shape, x.dtype, **sink_kw)
    with dask.config.set(scheduler='threads', num_workers=int(kwargs.get('max_workers', 4))):
        da.store(x.rechunk(sink.write_unit), sink, lock=False)
    return ""


def _per_level(spec, level: int, name: str, tuple_form: bool = False):
    """One level's value from a per-level option: a ``{level: value}`` dict, a
    per-level sequence (scalar options), or one value for every level."""
    if isinstance(spec, dict):
        if level in spec:
            v = spec[level]
        elif str(level) in spec:
            v = spec[str(level)]
        else:
            raise ValueError(f"{name} dict has no entry for resolution level {level}")
    elif not tuple_form and isinstance(spec, (tuple, list)):
        if level >= len(spec):
            raise ValueError(f"{name} has {len(spec)} entries but resolution level "
                             f"{level} was requested")
        v = spec[level]
    else:
        v = spec
    return tuple(int(x) for x in v) if tuple_form else v


def _resolve_compressor(pyramid, compressor, compressor_params):
    """The codec to write: an explicit name (+ params) or ``CompressorConfig``, else
    the pyramid's own (None: let the writer inherit the input's / use its default)."""
    from ome_zarr_pyramid.utils.compressor_config import CompressorConfig
    if compressor is None:
        return pyramid.compressor
    if isinstance(compressor, CompressorConfig):
        return compressor
    return CompressorConfig(name=compressor, params=dict(compressor_params or {}))


_BLOSC_SHUFFLE = {'noshuffle': 0, 'shuffle': 1, 'bitshuffle': 2}


def _to_dyna_codecs(cfg):
    """ozp ``CompressorConfig`` -> dyna ``Codecs`` (None passes through)."""
    if cfg is None:
        return None
    from dyna_zarr import Codecs
    name = (cfg.name or 'none').lower()
    p = dict(cfg.params or {})
    if name in ('', 'none'):
        return Codecs(None)
    if name == 'blosc':
        shuffle = p.get('shuffle', 1)
        if isinstance(shuffle, str):
            shuffle = _BLOSC_SHUFFLE.get(shuffle.lower(), 1)
        return Codecs('blosc', clevel=int(p.get('clevel', 5)), cname=p.get('cname', 'lz4'),
                      shuffle=int(shuffle))
    if name in ('zstd', 'gzip', 'bz2'):
        default = {'zstd': 1, 'gzip': 5, 'bz2': 1}[name]
        return Codecs(name, clevel=int(p.get('level', p.get('clevel', default))))
    raise ValueError(f"compressor {cfg.name!r} cannot be written by the dyna engine "
                     f"(blosc, zstd, gzip, bz2 or none)")


def _inherited_shard_coefficients(base):
    """The base level's sharding as per-axis coefficients (shard = coefficient x chunk),
    so every level of the pyramid is sharded alike; None when the base is unsharded."""
    shards, chunks = getattr(base, 'shards', None), getattr(base, 'chunks', None)
    if not shards or not chunks or len(shards) != len(chunks):
        return None
    if any(int(s) % int(c) for s, c in zip(shards, chunks)):
        return None
    coefs = tuple(int(s) // int(c) for s, c in zip(shards, chunks))
    return None if all(k == 1 for k in coefs) else coefs


def _store_join(path, *parts):
    """Join sub-paths onto a store: URL-safe (forward slashes) for remote stores, a
    `pathlib.Path` for local ones."""
    if _is_remote_store(path):
        return str(path).rstrip("/") + "/" + "/".join(str(p).strip("/") for p in parts)
    p = Path(path)
    for part in parts:
        p = p / str(part)
    return p


def _open_store_group(path, zarr_format: int, mode: str = "a",
                      storage_options: Optional[dict] = None) -> zarr.Group:
    """Open/create a zarr group for attr read/write: local directly; an object-store URL
    (s3://, gs://, az://) through obstore with dyna-style storage_options; the legacy
    https:// form through an s3fs mapping (options set by `set_s3_storage_options`)."""
    if _is_remote_store(path):
        if _is_legacy_https(path):
            store = tensorstore_writer.wrap_output_path(str(path))
        else:
            return _object_store_group(str(path), zarr_format, mode, storage_options)
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
                     include_labels: bool = True,
                     storage_options: Optional[dict] = None) -> Pyramid:
        """Read a pyramid from an NGFF-compliant zarr store.

        Parameters
        ----------
        path : str or Path
            Path to the zarr store containing the NGFF pyramid, or an object-store URL
            (s3://, gs://, az://).
        include_labels : bool
            Also discover the image's NGFF ``labels/`` collection and populate the
            returned pyramid's `Pyramid.labels` (lazy - only metadata + lazy arrays
            are loaded). Default True; set False to skip the scan. Local stores only.
        storage_options : dict, optional
            For a URL: obstore-style options (``endpoint``, ``region``,
            ``skip_signature``, ``client_options``, ...), the same names the writer and
            dyna use. Credentials come from the standard environment variables.

        Returns
        -------
        Pyramid
            Loaded pyramid with metadata and arrays
        """
        remote = _is_remote_store(path) and not _is_legacy_https(path)
        if not remote:
            path = Path(path) if isinstance(path, str) else path
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: {path}")

        if self.verbose:
            logger.info(f"[PyramidIO] Reading pyramid from: {path}")

        try:
            pyramid = Pyramid()
            if remote:
                # the format is read from the store; open_group detects v2 / v3
                pyramid.from_ngff(_object_store_group(str(path), None, "r", storage_options))
                include_labels = False   # label discovery walks a local directory tree
            else:
                pyramid.from_ngff(str(path))

            # remember the on-disk chunking so ops that rechunk for processing can be
            # written back with the ORIGINAL storage chunks by default (see rechunk).
            try:
                base = pyramid.layers[pyramid.meta.resolution_paths[0]]
                pyramid._storage_chunks = tuple(int(c) for c in base.chunks)
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
                      backend: str = 'auto',
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
                      storage_options: Optional[dict] = None,
                      shard_coefficients=None,
                      io_backend: str = 'tensorstore') -> None:
        """Write a pyramid using region-wise processing.

        Write engines (``backend``):

        * ``'auto'`` (default): ``'dyna'``, except a remote (https://) target, which
          still goes to ``'sync'`` until the remote dyna path lands.
        * ``'dyna'``: dyna owns every write. numpy, zarr, TensorStore and DynamicArray
          levels are PULLED by ``dyna_zarr.io.write`` (memory-bounded); dask levels are
          PUSHED into a dyna sink by ``dask.array.store`` (the dask pump; dask schedules
          its own graph on threads). The fastest engine measured
          (reports/write_audit.md), and the only one that writes shards. Refuses remote
          targets (for now).
        * ``'sync'``: the threaded zarr-python writer (legacy).
        * ``'tensorstore'``: the async TensorStore writer (legacy, local only).

        ``compressor`` / ``compressor_params`` (a name + params, or a
        ``CompressorConfig``) override the pyramid's codec on every engine.
        ``shard_coefficients`` (a per-axis tuple, or ``{level: tuple}``; shard =
        coefficients x chunks, zarr v3 only) and ``io_backend`` (dyna's byte backend:
        ``'tensorstore'`` or ``'zarrista'``) need the dyna engine and are refused by the
        others. Without ``shard_coefficients`` a sharded base level's sharding is kept
        and applied to every level.

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

        Parallelism:

        * ``use_multiprocessing=False`` (default): threads - each level is written
          with ``max_workers`` concurrent regions, one level after another.
        * ``use_multiprocessing=True`` (dyna engine only): one spawned process per
          level, all levels at once, each with its own ``max_workers`` regions - so
          peak memory is up to (number of levels) x one write. Works for every pyramid
          the dyna engine writes (zarr, numpy, TensorStore, op chains: they pickle).

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
            Process-per-level parallelism on the dyna engine. Default is False.
        backend : str, optional
            The write engine: ``'auto'`` (default), ``'dyna'``, ``'sync'`` or
            ``'tensorstore'`` - see above.
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
        engine = _resolve_engine(pyramid, backend, remote=remote,
                                 use_multiprocessing=use_multiprocessing, path=path)
        if engine != 'dyna':
            # options only the dyna engine implements: refuse rather than silently drop
            dropped = [name for name, given in (
                ("shard_coefficients", shard_coefficients is not None),
                ("io_backend", io_backend != 'tensorstore')) if given]
            if dropped:
                raise ValueError(
                    f"{', '.join(dropped)} need(s) the dyna write engine, but this write "
                    f"uses {engine!r}. Pass backend='dyna' (or 'auto' for a pyramid dyna "
                    f"can stream), or drop {', '.join(dropped)}.")
        if remote and engine != 'dyna':
            # the legacy https:// path: s3fs-style options, set module-wide
            if storage_options is not None:   # e.g. {'anon': True} for a public bucket
                tensorstore_writer.set_s3_storage_options(**storage_options)
            if (getattr(pyramid, '_downscale_plan', None) is not None
                    and getattr(pyramid, '_downscale_plan_active', False)
                    and layers is None):
                raise NotImplementedError(
                    "writing a DEFERRED downscale() pyramid to a legacy https:// target is "
                    "not supported. Use the s3:// form (dyna engine), downscale(defer=False), "
                    "or write a single level."
                )
        elif not remote:
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
            lg = _open_store_group(labels_group_path, label_zarr_format, mode="a",
                                   storage_options=storage_options)
            _register_label_name(lg, label_name, label_version)

        # A DEFERRED-downscale pyramid (from `Pyramid.downscale(defer=True)`) carries
        # only level 0 plus a plan; expand it PROGRESSIVELY FROM DISK so an expensive
        # lazy base is computed once, not re-run per output level. (Skip when a
        # specific `layers` subset is requested - that path is used internally here.)
        # THE PLAN WINS. A pyramid can carry both real levels and a plan, because ops
        # propagate the plan across every level they transform. When both are present the
        # plan is the more recent intent - it says what the pyramid should become - so it
        # is applied and the pre-existing coarser levels are rebuilt from the base. (The
        # plan path writes level 0 and re-reads it to expand the rest, so it narrows to
        # the base itself; feeding it several levels used to make it look for a level it
        # had not written yet - `KeyError: '1'`.)
        # Only an ACTIVE plan is written. A plan inherited through an op is DORMANT: it
        # records what the source's levels were (factors, shapes, count) but commits to
        # nothing, so an op chain does not silently produce levels the caller never asked
        # for. `Pyramid.downscale()` activates it - that is what asking looks like.
        plan = getattr(pyramid, '_downscale_plan', None)
        if plan is not None and not getattr(pyramid, '_downscale_plan_active', False):
            plan = None
        if plan is not None and layers is None:
            extra = ({'shard_coefficients': shard_coefficients, 'io_backend': io_backend,
                      'storage_options': storage_options}
                     if engine == 'dyna' else {})
            self._write_with_plan(
                pyramid, path, plan, overwrite=overwrite,
                chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb,
                max_workers=max_workers, region_size_mb=region_size_mb,
                gc_interval=gc_interval, use_multiprocessing=use_multiprocessing,
                backend=engine, compressor=compressor,
                compressor_params=compressor_params, max_concurrency=max_concurrency,
                num_readers=num_readers, queue_size=queue_size,
                max_concurrent_layers=max_concurrent_layers, **extra,
            )
            self._write_side_metadata(
                path, label_block=label_block, label_version=label_version,
                label_zarr_format=label_zarr_format, attached_labels=attached_labels,
                overwrite=overwrite, backend=backend, compressor=compressor,
                compressor_params=compressor_params, storage_options=storage_options)
            return

        # dyna engine: every level through dyna_zarr.io.write, which streams the whole
        # read -> op-chain -> write pipeline region by region (numpy / zarr / TensorStore
        # levels are wrapped as DynamicArrays). Also serves the internal layer-subset
        # writes of the plan path.
        if engine == 'dyna':
            self._write_pyramid_dyna(
                pyramid, path, layers=layers, overwrite=overwrite,
                chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb,
                shard_coefficients=shard_coefficients, compressor=compressor,
                compressor_params=compressor_params, region_size_mb=region_size_mb,
                max_workers=max_workers, io_backend=io_backend,
                use_multiprocessing=use_multiprocessing, storage_options=storage_options)
            if layers is not None:
                return    # internal sub-write: the caller writes the side metadata
            self._write_side_metadata(
                path, label_block=label_block, label_version=label_version,
                label_zarr_format=label_zarr_format, attached_labels=attached_labels,
                overwrite=overwrite, backend=backend, compressor=compressor,
                compressor_params=compressor_params, storage_options=storage_options)
            return

        # apply the storage chunking before writing (explicit spec, else restore the
        # recorded on-disk chunks). Skipped for internal layer-subset writes, which
        # are already rechunked by _write_with_plan.
        if layers is None:
            if chunk_shape is not None or chunk_size_mb is not None:
                pyramid = pyramid.rechunk(chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb)
            elif getattr(pyramid, '_storage_chunks', None) is not None:
                pyramid = pyramid.rechunk()

        if engine == 'tensorstore':
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
                compressor_params=compressor_params, storage_options=storage_options)
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

            self._write_all_layers_unified(
                pyramid=pyramid,
                zarr_group=zarr_group,
                layers_to_write=layers_to_write,
                dest_path=path,
                max_workers=max_workers,
                region_size_mb=region_size_mb,
                gc_interval=gc_interval,
                compressor_config=_resolve_compressor(pyramid, compressor,
                                                      compressor_params),
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
            compressor_params=compressor_params, storage_options=storage_options)

    def _write_pyramid_dyna(self, pyramid, path, *, layers, overwrite, chunk_shape,
                            chunk_size_mb, shard_coefficients, compressor, compressor_params,
                            region_size_mb, max_workers, io_backend,
                            use_multiprocessing=False, storage_options=None):
        """The dyna engine: every level streamed through ``dyna_zarr.io.write``
        (memory-bounded), then the NGFF group metadata. Local or object store (s3://,
        gs://, az://): the arrays through dyna, the group through obstore, both with the
        same ``storage_options``.

        dyna owns the bytes: chunk solving, codecs, shards, overwrite rule. ozp only
        resolves what dyna cannot know - its per-level forms (``{level: ...}`` dicts and
        per-level sequences), the chunking recorded on the pyramid, the pyramid's codec,
        the base level's sharding (applied to every level) and the v3 dimension names.
        """
        remote = _is_remote_store(path)
        path = str(path) if remote else Path(path)
        zf = int(pyramid.meta.zarr_format)
        paths = [str(p) for p in pyramid.meta.resolution_paths]
        todo = paths if layers is None else [str(p) for p in layers]
        # dask levels stay dask - they are PUSHED into a dyna sink (the dask pump);
        # everything else is wrapped as a DynamicArray and PULLED by io.write
        from dyna_zarr import DynamicArray
        das = {}
        for p in paths:
            a = pyramid.layers[p]
            das[p] = a if (is_dask_array(a) or isinstance(a, DynamicArray)) else DynamicArray(a)

        codecs = _to_dyna_codecs(_resolve_compressor(pyramid, compressor, compressor_params))
        if shard_coefficients is None and zf == 3:
            shard_coefficients = _inherited_shard_coefficients(das[paths[0]])
        if shard_coefficients is not None and zf != 3:
            raise ValueError(
                f"shard_coefficients need zarr v3, but this pyramid is OME-Zarr "
                f"{pyramid.meta.version} (zarr v{zf}).")
        recorded = dict(getattr(pyramid, '_level_chunks', None) or {})
        storage = getattr(pyramid, '_storage_chunks', None)

        if remote:
            zarr_group = _object_store_group(path, zf, "w" if overwrite else "a",
                                             storage_options)
        else:
            zarr_group = tensorstore_writer._zarr_group(str(path), overwrite=overwrite,
                                                        zarr_format=zf)
        jobs = []                                    # (level path, array, output, kwargs)
        for lvl, p in enumerate(paths):
            if p not in todo:
                continue
            arr = das[p]
            kw = dict(zarr_format=zf, region_size_mb=region_size_mb, max_workers=max_workers,
                      backend=io_backend, overwrite=False)
            # chunks: an exact shape -> chunks=, a size -> chunk_size_mb= (dyna solves it);
            # neither -> what the pyramid recorded, else dyna's default (the input's own)
            if chunk_shape is not None:
                kw['chunks'] = _per_level(chunk_shape, lvl, 'chunk_shape', tuple_form=True)
            elif chunk_size_mb is not None:
                kw['chunk_size_mb'] = float(_per_level(chunk_size_mb, lvl, 'chunk_size_mb'))
            elif p in recorded:
                kw['chunks'] = tuple(recorded[p])
            elif storage is not None and len(storage) == arr.ndim:
                kw['chunks'] = tuple(min(int(c), int(s)) for c, s in zip(storage, arr.shape))
            elif is_dask_array(arr):
                kw['chunks'] = tuple(int(c) for c in get_array_chunks(arr))   # dask's own grid
            if codecs is not None:
                kw['compressor'] = codecs
            if shard_coefficients is not None:
                kw['shard_coefficients'] = _per_level(shard_coefficients, lvl,
                                                      'shard_coefficients', tuple_form=True)
            if zf == 3:
                kw['dimension_names'] = tuple(pyramid.axes)
            if remote and storage_options:
                kw['storage_options'] = dict(storage_options)
            jobs.append((p, arr, str(_store_join(path, p)), kw))

        if use_multiprocessing and len(jobs) > 1:
            # one spawned process per level, all levels at once (the arrays pickle:
            # zarr / TensorStore by reference, numpy staging by value, op chains as-is)
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=len(jobs),
                                     mp_context=multiprocessing.get_context("spawn")) as ex:
                futures = [(p, ex.submit(_dyna_write_level, arr, out, kw))
                           for p, arr, out, kw in jobs]
                logs = [(p, f.result()) for p, f in futures]   # re-raises a worker error
        else:
            logs = [(p, (_dask_write_level if is_dask_array(arr) else _dyna_write_level)(
                        arr, out, kw)) for p, arr, out, kw in jobs]
        if self.verbose:
            for p, log in logs:
                if log.strip():
                    logger.info(f"[PyramidIO] level {p}: " + " | ".join(
                        ln.strip() for ln in log.splitlines() if ln.strip()))
        # write the multiscales / omero metadata onto the group (same as the sync path)
        pyramid.meta.connect_to_group(zarr_group)
        pyramid.meta._pending_changes = True
        pyramid.meta.save_changes()

    def _write_side_metadata(self, path, *, label_block, label_version, label_zarr_format,
                             attached_labels: dict, overwrite: bool, backend: str = 'sync',
                             compressor=None, compressor_params=None,
                             storage_options=None) -> None:
        """Post-array-write side metadata, shared by every write_pyramid exit:
        (1) if this is a LABEL image, attach its ``image-label`` block to the written
        group (belt-and-suspenders across backends; `save_changes` already writes it on
        the sync path); (2) write any attached image `labels` into ``<path>/labels/<name>``."""
        if label_block is not None:
            group = _open_store_group(path, label_zarr_format, mode="a",
                                      storage_options=storage_options)
            _attach_image_label(group, label_block, label_version)
        if attached_labels:
            self._write_attached_labels(attached_labels, path, overwrite=overwrite,
                                        backend=backend, compressor=compressor,
                                        compressor_params=compressor_params,
                                        storage_options=storage_options)

    def _write_attached_labels(self, labels: dict, path, *, overwrite: bool,
                               backend: str = 'sync', compressor=None,
                               compressor_params=None, storage_options=None) -> None:
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
                storage_options=storage_options,
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

        # 1) write base only (rechunked to storage chunks), plan removed to avoid recursion.
        # NARROW to level 0 first: the pyramid may still carry coarser levels (ops
        # propagate the plan across all of them), and writing with `layers=[base]` alone
        # would leave the group metadata advertising levels that were never written, so
        # the re-read in step 2 fails with `KeyError`. The plan rebuilds those levels.
        saved = pyramid.__dict__.pop('_downscale_plan', None)
        try:
            source = (pyramid.select_levels(base_path0)
                      if len(pyramid.meta.resolution_paths) > 1 else pyramid)
            source.__dict__.pop('_downscale_plan', None)
            base = _rechunked(source)
            self.write_pyramid(base, path, layers=[base_path0], overwrite=overwrite, **write_kwargs)
        finally:
            if saved is not None:
                pyramid._downscale_plan = saved

        # 2) re-read the on-disk base (zarr-backed -> downscaling reads from DISK)
        storage_options = write_kwargs.get('storage_options')
        disk = self.read_pyramid(str(path), include_labels=False,
                                 storage_options=storage_options)

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
        from ome_zarr_pyramid.utils.scale import (as_tensorstore, crop_to_level, mean_downscale,
                                                  median_downscale, simple_downscale)
        # dask levels only for the legacy engines; the dyna engine streams DynamicArray
        # levels (it cannot consume dask graphs)
        if is_installed("dask") and write_kwargs.get('backend') in ('sync', 'tensorstore'):
            method = {'simple': simple_downscale, 'mean': mean_downscale,
                      'median': median_downscale}.get(plan.get('downscale_method', 'simple'),
                                                      simple_downscale)
            base_da = disk.dask_arrays[disk.meta.resolution_paths[0]]
        else:
            # no dask: the stored base as a DynamicArray; stride levels by lazy slicing,
            # mean / median by TensorStore's downsample of the stored base, cut to the
            # level sizes dask's coarsen gives
            base_da = disk._arrays()[disk.meta.resolution_paths[0]]
            base_ts = as_tensorstore(base_da)

            def method(base, scale_factor, _m=plan.get('downscale_method', 'simple')):
                if _m not in ('mean', 'median'):
                    return simple_downscale(base, scale_factor=scale_factor)
                from dyna_zarr import DynamicArray
                level = ts.downsample(base_ts, [int(f) for f in scale_factor], method=_m)
                return DynamicArray(crop_to_level(level, base.shape, scale_factor, _m))
        ndim = base_da.ndim
        level_arrays = [base_da]
        level_factors = [tuple([1] * ndim)]
        # Per-level factors, SOLVED against each target shape rather than rounded from
        # the ratio. `round(base/tgt)` silently produced an off-by-one level whenever the
        # rounded ratio did not actually reproduce `tgt` (e.g. base=101 tgt=50), and it
        # cannot express an irregular progression at all. `_solve_axis_factor` verifies.
        from ome_zarr_pyramid.utils.scale import _solve_axis_factor, _level_size
        dmethod = plan.get('downscale_method', 'simple')
        planned = plan.get('level_scale_factors')
        pshapes = plan.get('level_shapes')
        for i in range(1, len(fpaths)):
            # When the plan carries DERIVED per-level factors, they - not the shapes
            # `downscale(defer=False)` regenerates - are authoritative. That expansion
            # applies one `scale_factor` repeatedly, so it cannot reproduce an irregular
            # or anisotropic progression (base 64 with levels 64/32 -> level 2 factor
            # (2,4,4), which `scale_factor**2` renders as (1,1,64,16,16)).
            if planned is not None and i < len(planned):
                factor = tuple(int(f) for f in planned[i])
                tgt = (tuple(pshapes[i]) if pshapes is not None and i < len(pshapes)
                       else tuple(_level_size(base_da.shape[a], factor[a], dmethod)
                                  for a in range(ndim)))
            else:
                tgt = full.layers[fpaths[i]].shape
                # The factor the level's SCALE metadata states comes first: it is what the
                # written store will claim. The shape alone is ambiguous whenever an extent
                # is not divisible (41 -> 6 fits factors 7 AND 8), and the solver returns
                # the smallest fit - 7 - while the metadata said 8: pixels and physical
                # coordinates silently disagreed.
                s0, si = full.meta.get_scale(fpaths[0]), full.meta.get_scale(fpaths[i])
                stated = [si[a] / s0[a] if s0[a] else None for a in range(ndim)]
                factor = []
                for a in range(ndim):
                    st = stated[a]
                    if (st is not None and abs(st - round(st)) < 1e-6 and round(st) >= 1
                            and _level_size(base_da.shape[a], int(round(st)), dmethod) == tgt[a]):
                        factor.append(int(round(st)))
                        continue
                    f = _solve_axis_factor(base_da.shape[a], tgt[a], dmethod)
                    factor.append(int(f) if f is not None
                                  else max(1, int(round(base_da.shape[a] / tgt[a]))))
                factor = tuple(factor)
            got = tuple(_level_size(base_da.shape[a], factor[a], dmethod) for a in range(ndim))
            if got != tuple(tgt):
                raise ValueError(
                    f"cannot reproduce level {i} of the downscale plan exactly: "
                    f"base {tuple(base_da.shape)} with factor {factor} gives {got}, "
                    f"but the plan requires {tuple(tgt)}. No integer downscale factor "
                    f"maps the base to that shape under method '{dmethod}' (no backend "
                    f"accepts a target shape). Re-plan with .downscale(...) or use "
                    f"drop_downscale_plan()."
                )
            level_arrays.append(method(base_da, scale_factor=factor))
            level_factors.append(factor)

        def _build(arrays):
            """The full pyramid over `arrays`, carrying the source's metadata."""
            unit_clean = [u for u in full.meta.unit_list if u is not None]
            full_dask = Pyramid().from_arrays(
                arrays=arrays, axis_order=full.meta.axis_order,
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
            return full_dask

        level_kwargs = dict(write_kwargs)
        if (level_kwargs.get('backend') == 'dyna' and level_kwargs.get('compressor') is None
                and pyramid.compressor is not None):
            # full_dask was rebuilt from arrays, so its compressor fell back to the
            # default: write the coarser levels with the SOURCE's codec, like level 0
            level_kwargs['compressor'] = pyramid.compressor

        # CASCADE ('simple' on the dyna engine): level i from the STORED level i-1, not
        # from L0. Stride composes exactly (L0[::a][::b] == L0[::a*b]), so the pixels are
        # identical, while L0 is read once instead of once per level (1.2 GB / 4 levels:
        # 3.9 -> 2.1 GB read, every level still written in whole chunks). Not for
        # mean/median, which do not compose exactly (rounding twice; median of medians),
        # nor with use_multiprocessing (a cascade is sequential by nature).
        cascade = (level_kwargs.get('backend') == 'dyna' and dmethod == 'simple'
                   and not level_kwargs.get('use_multiprocessing'))
        if not cascade:
            self.write_pyramid(_build(level_arrays), path, layers=extra, overwrite=False,
                               **level_kwargs)
            return
        from dyna_zarr import DynamicArray
        for i in range(1, len(fpaths)):
            prev, cur = level_factors[i - 1], level_factors[i]
            if all(c % p == 0 for c, p in zip(cur, prev)):
                if _is_remote_store(path):
                    from dyna_zarr import io as dyna_io
                    parent = dyna_io.read(str(_store_join(path, fpaths[i - 1])),
                                          storage_options=storage_options)
                else:
                    parent = DynamicArray(zarr.open_array(str(Path(path) / fpaths[i - 1]),
                                                          mode='r'))
                rel = tuple(c // p for c, p in zip(cur, prev))
                stepped = simple_downscale(parent, scale_factor=rel)
                if tuple(stepped.shape) == tuple(level_arrays[i].shape):
                    level_arrays[i] = stepped
                # else: keep the from-L0 level (an irregular progression)
            self.write_pyramid(_build(level_arrays), path, layers=[fpaths[i]],
                               overwrite=False, **level_kwargs)

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
                                   gc_interval: float = 15.0,
                                   compressor_config=None) -> None:
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
            # the storage chunks `rechunk` recorded, else the array's own (dask's
            # chunksize, zarr's / a DynamicArray's chunks, a TensorStore's layout)
            recorded = (getattr(pyramid, '_level_chunks', None) or {}).get(str(layer_path))
            own = get_array_chunks(array)
            if recorded is not None:
                chunks = tuple(int(c) for c in recorded)
            elif own is not None:
                chunks = tuple(int(c) for c in own)
            else:
                chunks = tuple([256] * len(shape))
            
            # Create the layer array ON THE GROUP, so it lands in the group's store -
            # LOCAL or an s3fs/fsspec mapping alike (single store-agnostic writer).
            if layer_path not in zarr_group:
                zf = pyramid.meta.zarr_format if pyramid.meta else 2
                tensorstore_writer.create_group_array(
                    zarr_group, layer_path, shape, chunks, dtype, zf,
                    compressor_config if compressor_config is not None else pyramid.compressor,
                    list(pyramid.axes) if zf == 3 else None)

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
                        if is_dask_array(array):
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


class IO:
    """Convenience wrapper for high-performance pyramid I/O operations."""

    def read_pyramid(self, path: Union[str, Path], include_labels: bool = True,
                     storage_options: Optional[dict] = None) -> Pyramid:
        """Read a pyramid from a path or object-store URL. `include_labels` also loads
        the image's NGFF ``labels/`` collection into `Pyramid.labels` (lazy). See
        :meth:`PyramidIO.read_pyramid`."""
        return PyramidIO().read_pyramid(path, include_labels=include_labels,
                                        storage_options=storage_options)

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
                      backend: str = 'auto',
                      **kwargs) -> None:
        """Write a pyramid to a path. ``backend`` picks the write engine
        (``'auto'``, ``'dyna'``, ``'sync'``, ``'tensorstore'``); extra ``**kwargs``
        (``compressor``, ``chunk_shape``, ``shard_coefficients``, ...) are forwarded.
        See :meth:`PyramidIO.write_pyramid`.
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
