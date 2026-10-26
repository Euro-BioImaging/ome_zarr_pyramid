"""Reading and writing OME-Zarr pyramids.

The NGFF side (multiscales, omero, labels, the downscale plan) lives here; the bytes are
written by ``dyna_zarr.io``, region by region within a memory budget, local or remote.
The engines write_pyramid used before 0.1.0 are kept at the end of `PyramidIO` for
reference only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import asyncio
import gc
import itertools
import logging
from types import SimpleNamespace
import os
import threading
import time
import warnings
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorstore as ts
import zarr

from ome_zarr_pyramid.core import tensorstore_writer
from ome_zarr_pyramid.core._compat import downscale_processes, renamed
from ome_zarr_pyramid.core.nodes import nests_in_ome
from ome_zarr_pyramid.core.pyramid import Pyramid, level_translation
from ome_zarr_pyramid.core.validation import validating
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
    0.5 and 0.6 nest it under the ``ome`` attr; 0.4 uses a top-level ``image-label``)."""
    if nests_in_ome(version):
        ome = dict(group.attrs.get("ome", {}))
        ome["image-label"] = image_label
        group.attrs["ome"] = ome
    else:  # 0.4
        group.attrs["image-label"] = image_label


def _register_label_name(labels_group: zarr.Group, name: str, version: str, *,
                         image_path=None, label_system: Optional[str] = None,
                         storage_options: Optional[dict] = None) -> None:
    """Add ``name`` to a ``labels`` group's label list (the NGFF labels/ convention). In
    0.6 the image at `image_path` (the group holding the labels group) also gets the
    label's link to `label_system`, the label's intrinsic system (3.28)."""
    if nests_in_ome(version):
        ome = dict(labels_group.attrs.get("ome", {}))
        ome.setdefault("version", version)
        names = list(ome.get("labels", []))
        if name not in names:
            names.append(name)
        ome["labels"] = names
        labels_group.attrs["ome"] = ome
        if str(version) == "0.6" and image_path is not None:
            _link_label_06(image_path, name, label_system, storage_options)
    else:  # 0.4
        names = list(labels_group.attrs.get("labels", []))
        if name not in names:
            names.append(name)
        labels_group.attrs["labels"] = names


def _link_label_06(image_path, name: str, label_system: Optional[str],
                   storage_options: Optional[dict] = None) -> None:
    """Give the 0.6 image at `image_path` the identity link from its intrinsic system to
    label `name`'s `label_system` (3.28; the spec's "labels" section), unless it has a
    link for that label already. Nothing when no 0.6 image is there (yet)."""
    from ome_zarr_pyramid.core.ngff06 import INTRINSIC
    try:
        image = _open_store_group(image_path, 3, mode="a", storage_options=storage_options)
    except (FileNotFoundError, ValueError, TypeError):
        return
    ome = dict(image.attrs.get("ome") or {})
    mss = list(ome.get("multiscales") or [])
    if str(ome.get("version")) != "0.6" or not mss:
        return
    target = f"labels/{name}"
    ms = dict(mss[0])
    cts = list(ms.get("coordinateTransformations") or [])
    for t in cts:
        for end in (t.get("input"), t.get("output")):
            if isinstance(end, dict) and str(end.get("path", "")).strip("/") == target:
                return
    own = ((ms.get("datasets") or [{}])[0].get("coordinateTransformations") or [{}])[0]
    intrinsic = (own.get("output") or {}).get("name") or INTRINSIC
    cts.append({"type": "identity", "input": {"name": intrinsic},
                "output": {"name": label_system or INTRINSIC, "path": target}})
    ms["coordinateTransformations"] = cts
    ome["multiscales"] = [ms] + mss[1:]
    image.attrs["ome"] = ome


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
    s3fs-style storage_options. `write_pyramid` translates it with
    `_legacy_https_target`; the native form is ``s3://bucket/key`` with obstore-style
    options."""
    return str(path).startswith("https://")


# s3fs option -> obstore option, for the legacy https:// form
_S3FS_NAMES = {"key": "access_key_id", "secret": "secret_access_key",
               "token": "session_token"}
_S3FS_CLIENT_KWARGS = {"endpoint_url": "endpoint", "region_name": "region"}


def _legacy_https_target(path, storage_options: Optional[dict] = None):
    """``(s3://bucket/key, obstore options)`` for a legacy ``https://host/bucket/key``.

    The endpoint is ``https://host`` and the bucket stays in the path
    (``virtual_hosted_style_request=False``), as in the URL. The s3fs option names the
    old writer took are renamed: ``anon=True`` -> ``skip_signature``, ``key`` /
    ``secret`` / ``token`` -> ``access_key_id`` / ``secret_access_key`` /
    ``session_token``, ``client_kwargs`` ``endpoint_url`` / ``region_name`` ->
    ``endpoint`` / ``region``. Other names pass through as obstore options. (Explicit
    credentials need ``io_backend='zarrista'``; the TensorStore backend takes them from
    the environment and refuses them as options.)"""
    host, _, key = str(path)[len("https://"):].partition("/")
    if not key.strip("/"):
        raise ValueError(f"{str(path)!r}: the https:// form is https://host/bucket/key.")
    opts = {"endpoint": f"https://{host}", "virtual_hosted_style_request": False}
    for name, value in (storage_options or {}).items():
        if name == "anon":
            if value:
                opts["skip_signature"] = True
        elif name in _S3FS_NAMES:
            opts[_S3FS_NAMES[name]] = value
        elif name == "client_kwargs":
            unknown = sorted(set(value) - set(_S3FS_CLIENT_KWARGS))
            if unknown:
                raise ValueError(
                    f"storage_options client_kwargs {unknown} have no equivalent in the "
                    f"obstore options ozp writes with; pass obstore options instead "
                    f"(endpoint, region, client_options, ...).")
            opts.update({_S3FS_CLIENT_KWARGS[k]: v for k, v in value.items()})
        else:
            opts[name] = value
    return f"s3://{key}", opts


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


#: ``write_pyramid(backend=...)``: the storage backend, as dyna-zarr names it (decision
#: C-14, 0.5.0; ``io_backend=`` is its old name)
_STORAGE_BACKENDS = ('tensorstore', 'zarrista')
#: ``write_pyramid(engine=...)``: dyna-zarr's execution loop
_DYNA_ENGINES = ('fused', 'pipeline')
#: what ``backend=`` named before 0.5.0: ozp's write engine, which is always dyna
_ENGINES = ('auto', 'dyna')
# Removed in 0.1.0. Their code stays in this module for reference
# (`PyramidIO.write_pyramid_tensorstore`, `PyramidIO._write_all_layers_unified`) but
# write_pyramid no longer calls it: the dyna engine was faster, used less memory, and
# was the only one with shards and a memory budget. ('tensorstore' was one too: since
# 0.5.0 backend='tensorstore' is the storage backend, as in dyna-zarr.)
_REMOVED_ENGINES = ('sync',)
_REMOVED_SETTINGS = ('max_concurrency', 'queue_size', 'max_concurrent_layers')


def _storage_backend(backend, io_backend) -> str:
    """The storage backend of ``write_pyramid(backend=..., io_backend=...)``: ``backend=``
    names it as dyna-zarr does (``'tensorstore'``, the default, or ``'zarrista'``);
    ``io_backend=`` is its old name and still works. The engine words ``backend=`` took
    before 0.5.0 (``'auto'``, ``'dyna'``: one engine, dyna) warn and are ignored; a removed
    engine raises."""
    if backend in _REMOVED_ENGINES:
        raise ValueError(
            f"backend={backend!r} was a write engine removed in ome_zarr_pyramid 0.1.0. The "
            f"default engine writes the same output, faster and with less memory: drop it. "
            f"(backend= names the storage backend now: 'tensorstore' or 'zarrista'.)")
    if backend in _ENGINES:
        warnings.warn(
            f"write_pyramid(backend={backend!r}): backend= names the storage backend since "
            f"ome_zarr_pyramid 0.5.0 ('tensorstore' or 'zarrista', as in dyna-zarr); the write "
            f"engine is always dyna, so drop backend={backend!r} (dyna's own loop is "
            f"engine='fused' or 'pipeline')", FutureWarning, stacklevel=4)
        backend = None
    for name, value in (("backend", backend), ("io_backend", io_backend)):
        if value is not None and value not in _STORAGE_BACKENDS:
            raise ValueError(f"{name}={value!r}: the storage backend is one of "
                             f"{_STORAGE_BACKENDS}, as in dyna-zarr")
    if backend is not None and io_backend is not None and backend != io_backend:
        raise ValueError(f"backend={backend!r} and io_backend={io_backend!r} disagree: "
                         f"io_backend is the old name of backend; give one")
    return backend or io_backend or 'tensorstore'


def _resolve_engine(pyramid, backend, *, max_downscale_processes) -> str:
    """ozp's write engine: always dyna. (`backend` is a word ``backend=`` took before 0.5.0,
    'auto' or 'dyna'.)"""
    if backend not in _ENGINES:
        raise ValueError(f"backend must be one of {_ENGINES}, got {backend!r}")
    if max_downscale_processes and pyramid._has_dask_layer():
        raise ValueError(
            "max_downscale_processes > 1 cannot write dask-backed levels (their graphs are not "
            "shipped to worker processes; dask parallelises them itself). Drop "
            "max_downscale_processes.")
    return 'dyna'


def _refuse_removed_settings(given: dict) -> None:
    """write_pyramid's catch-all keywords: the removed engines' settings get an
    explanation, anything else the usual TypeError."""
    removed = [k for k in given if k in _REMOVED_SETTINGS]
    if removed:
        raise ValueError(
            f"{', '.join(removed)} belonged to the write engines removed in "
            f"ome_zarr_pyramid 0.1.0 and no longer do anything. Drop them: parallelism is "
            f"max_threads, memory is memory_budget_mb.")
    if given:
        raise TypeError(f"write_pyramid() got unexpected keyword argument(s) "
                        f"{', '.join(map(repr, given))}")


def _dyna_write_level(arr, output_path: str, kwargs: dict) -> str:
    """Write one level with ``dyna_zarr.io.write``; returns the messages it reported.

    Module-level so a spawned worker process can import it (Windows uses spawn)."""
    from dyna_zarr import io as dyna_io
    # dyna's capture, not redirect_stdout: that swaps sys.stdout for the whole process,
    # so concurrent writes (threads) swallowed each other's and the host's output
    with dyna_io.capture_messages() as messages:
        dyna_io.write(arr, output_path, **kwargs)
    return "\n".join(messages)


def _plan_level_factors(plan: dict, full, base_shape, dmethod: str):
    """Per-level cumulative factors and target shapes of a downscale plan, level 0 first.

    Factors are SOLVED against each target shape rather than rounded from the ratio:
    `round(base/tgt)` silently produced an off-by-one level whenever the rounded ratio
    did not reproduce `tgt` (e.g. base=101 tgt=50), and it cannot express an irregular
    progression at all. Raises when a target is unreachable by any integer factor."""
    from ome_zarr_pyramid.utils.scale import _fits, _level_size, _solve_axis_factor
    fpaths = full.meta.resolution_paths
    ndim = len(base_shape)
    factors, targets = [tuple([1] * ndim)], [tuple(int(s) for s in base_shape)]
    planned = plan.get('level_scale_factors')
    pshapes = plan.get('level_shapes')
    for i in range(1, len(fpaths)):
        # When the plan carries DERIVED per-level factors, they - not the shapes
        # `downscale(defer=False)` regenerates - are authoritative. That expansion applies
        # one `scale_factor` repeatedly, so it cannot reproduce an irregular or anisotropic
        # progression (base 64 with levels 64/32 -> level 2 factor (2,4,4), which
        # `scale_factor**2` renders as (1,1,64,16,16)).
        if planned is not None and i < len(planned):
            factor = tuple(int(f) for f in planned[i])
            tgt = (tuple(pshapes[i]) if pshapes is not None and i < len(pshapes)
                   else tuple(_level_size(base_shape[a], factor[a], dmethod)
                              for a in range(ndim)))
        else:
            tgt = full.layers[fpaths[i]].shape
            # The factor the level's SCALE metadata states comes first: it is what the
            # written store will claim. The shape alone is ambiguous whenever an extent is
            # not divisible (41 -> 6 fits factors 7 AND 8), and the solver returns the
            # smallest fit - 7 - while the metadata said 8: pixels and physical
            # coordinates silently disagreed.
            s0, si = full.meta.get_scale(fpaths[0]), full.meta.get_scale(fpaths[i])
            stated = [si[a] / s0[a] if s0[a] else None for a in range(ndim)]
            factor = []
            for a in range(ndim):
                st = stated[a]
                if (st is not None and abs(st - round(st)) < 1e-6 and round(st) >= 1
                        and _fits(base_shape[a], int(round(st)), tgt[a])):
                    factor.append(int(round(st)))
                    continue
                f = _solve_axis_factor(base_shape[a], tgt[a], dmethod)
                factor.append(int(f) if f is not None
                              else max(1, int(round(base_shape[a] / tgt[a]))))
            factor = tuple(factor)
        got = tuple(_level_size(base_shape[a], factor[a], dmethod) for a in range(ndim))
        if not all(_fits(base_shape[a], factor[a], tgt[a]) for a in range(ndim)):
            raise ValueError(
                f"cannot reproduce level {i} of the downscale plan exactly: "
                f"base {tuple(base_shape)} with factor {factor} gives {got}, "
                f"but the plan requires {tuple(tgt)}. No integer downscale factor "
                f"maps the base to that shape under method '{dmethod}' (no backend "
                f"accepts a target shape). Re-plan with .downscale(...) or use "
                f"drop_downscale_plan()."
            )
        factors.append(factor)
        targets.append(tuple(int(t) for t in tgt))
    return factors, targets


def _full_pyramid(pyramid, full, arrays, rechunked, method):
    """The full pyramid over `arrays` (one per level of `full`, built from level 0 with
    downscale `method`), carrying the source `pyramid`'s metadata: what a plan-expanded
    write saves on the group."""
    import copy as _copy
    fpaths = full.meta.resolution_paths
    unit_clean = [u for u in full.meta.unit_list if u is not None]
    full_dask = Pyramid().from_arrays(
        arrays=arrays, axis_order=full.meta.axis_order,
        unit_list=unit_clean if unit_clean else None,
        scales=[full.meta.get_scale(p) for p in fpaths],
        version=full.meta.version,
        name=full.meta.multiscales.get('name', 'unnamed') if full.meta.metadata else 'unnamed',
    )
    full_dask._storage_chunks = getattr(pyramid, '_storage_chunks', None)
    full_dask = rechunked(full_dask)
    # from_arrays carries scales only: put the source's physical origin (its base
    # translation) plus each level's alignment under `method` on every level, or a
    # written pyramid loses its position
    base_translation = (pyramid.meta.get_base_translation()
                        if pyramid.meta is not None else None)
    if base_translation is not None and full_dask.meta is not None:
        base_scale = full.meta.get_scale(fpaths[0])
        for p in fpaths:
            full_dask.meta.update_translation(p, level_translation(
                base_translation, base_scale, full.meta.get_scale(p), method))
    # `full_dask` was rebuilt via from_arrays (axes/scales/units/name only), so overlay the
    # SOURCE pyramid's non-dataset metadata (omero channels/colours/windows and any custom
    # top-level attrs). Without this, writing the extra levels re-saves the group attrs and
    # clobbers omero set via set_channels / set_display_range. Mirror the source's
    # non-`multiscales` metadata EXACTLY (drop keys from_arrays auto-added that the source
    # lacks, e.g. an empty omero on a label image); keep full_dask's own `multiscales`.
    src_md = pyramid.meta.metadata if pyramid.meta is not None else None
    if src_md and full_dask.meta is not None and full_dask.meta.metadata is not None:
        ms = full_dask.meta.metadata.get('multiscales')
        new_md = {k: _copy.deepcopy(v) for k, v in src_md.items() if k != 'multiscales'}
        if ms is not None:
            new_md['multiscales'] = ms
        full_dask.meta.metadata = new_md
        full_dask.meta._pending_changes = True
    if full_dask.meta is not None and full_dask.meta.metadata and len(fpaths) > 1:
        # the downscale method that made the levels (NGFF: multiscales `type`; the
        # strict schemas require it)
        full_dask.meta.multiscales['type'] = method
    return full_dask


def _process_rss_mb():
    """This process's resident memory in MiB, or None without psutil."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:                                       # noqa: BLE001
        return None


def _ts_level_reducer(block, factors, method):
    """A level computed by TensorStore's downsample (downscale_backend='tensorstore')."""
    return ts.downsample(ts.array(block), [int(f) for f in factors],
                         method=method).read().result()


def _dyna_level_kwargs(pyramid, lvl, p, arr, *, zf, max_threads, io_backend, region_size_mb,
                       chunk_shape, chunk_size_mb, recorded, storage, codecs,
                       shard_coefficients, remote, storage_options, loop=None) -> dict:
    """``dyna_zarr.io.write`` options for one level: the dyna engine's rules, shared by
    the per-level writes and the one-pass level writer. `loop`: dyna's ``engine``,
    ``num_readers`` and ``gc_interval``, those given (per-level writes only)."""
    kw = dict(zarr_format=zf, max_threads=max_threads, backend=io_backend, overwrite=False)
    kw.update(loop or {})
    if not is_dask_array(arr):
        # dyna writes a dask level with dask's own scheduler (the pump), which has no
        # read regions and refuses region options
        kw['region_size_mb'] = region_size_mb
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
    if codecs is not None:
        kw['compressor'] = codecs
    if shard_coefficients is not None:
        kw['shard_coefficients'] = _per_level(shard_coefficients, lvl,
                                              'shard_coefficients', tuple_form=True)
    if zf == 3:
        kw['dimension_names'] = tuple(pyramid.axes)
    if remote and storage_options:
        kw['storage_options'] = dict(storage_options)
    return kw


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
    if type(compressor).__name__ == "Codecs" and \
            type(compressor).__module__.split(".")[0] == "dyna_zarr":
        # dyna-zarr's own form (decision C-14): its codec, level and blosc settings
        if compressor_params:
            raise ValueError("compressor_params go with a codec name; a dyna Codecs holds "
                             "its own settings")
        name = compressor.compressor or "none"
        if name == "blosc":
            params = {"cname": compressor.cname, "clevel": int(compressor.clevel),
                      "shuffle": int(compressor.shuffle)}
        elif name in ("zstd", "gzip", "bz2"):
            params = {"level": int(compressor.clevel)}
        else:
            params = {}
        return CompressorConfig(name=str(name), params=params)
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


class CodecChangedWarning(UserWarning):
    """A write uses another codec than its input's: the input's codec has no form in the
    output's zarr format (bz2 and plain lz4 are zarr v2 only), or the writer does not
    know it (zlib, lzma, zarr-python's numcodecs.* v3 codecs). The pixels are the same."""


#: what a write uses when the input's codec cannot be written
_FALLBACK_CODEC = ("blosc", 5, "lz4")


def _codec_label(codecs) -> str:
    name = getattr(codecs, "compressor", None)
    return f"blosc-{codecs.cname}" if name == "blosc" else str(name)


def _writable_codecs(codecs, source, zarr_format: int, explicit: bool):
    """The codecs a write uses: `codecs` (from ``compressor=`` or the pyramid's own) or,
    when None, the ones dyna would inherit from `source` (the base level). When those
    cannot be written in `zarr_format`, or the input's codec is one the writer does not
    know, the write uses blosc-lz4 and says so (`CodecChangedWarning`). A codec the
    caller asked for (`explicit`) is never replaced: its write fails, naming it."""
    import warnings

    from dyna_zarr import Codecs

    fallback = Codecs(_FALLBACK_CODEC[0], clevel=_FALLBACK_CODEC[1], cname=_FALLBACK_CODEC[2])
    hint = "pass compressor= to choose another"
    if codecs is None:
        codecs = getattr(source, "codecs", None)
        raw = getattr(source, "_compressors", None)
        if codecs is None:
            if raw:                                 # compressed, with a codec dyna lacks
                first = raw[0]
                try:
                    label = first.to_dict().get("name")
                except Exception:                   # noqa: BLE001
                    label = getattr(first, "codec_id", type(first).__name__)
                warnings.warn(f"the input's codec {label!r} cannot be written (the writer "
                              f"knows blosc, zstd and gzip, plus bz2 and lz4 in zarr v2); "
                              f"writing blosc-lz4 instead ({hint})", CodecChangedWarning,
                              stacklevel=4)
                return fallback
            return None                             # nothing to inherit: dyna's default
    try:
        if zarr_format == 3:
            codecs.to_v3_config("uint8")
        else:
            codecs.to_v2_config()
    except ValueError:
        if explicit:
            raise
        warnings.warn(f"the input's codec {_codec_label(codecs)} has no zarr v{zarr_format} "
                      f"form; writing blosc-lz4 instead ({hint})", CodecChangedWarning,
                      stacklevel=4)
        return fallback
    return codecs


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


def _place_by_scene(pyramid, path, apply: bool, storage_options=None) -> None:
    """For an image the scene form placed (``eubi:scene`` on its group, plan 5.5): with
    `apply`, read the scene on the group it names and add the image's position to every
    level (and its labels'), then drop the back-pointer (the image now holds its position,
    the single-image form); without, warn once naming that group."""
    import posixpath

    from ome_zarr_pyramid.core import ngff06
    from ome_zarr_pyramid.core.eubi_keys import SCENE_KEY
    from ome_zarr_pyramid.core.spatial import SYSTEMS_KEY, system_from_json

    pointer = (pyramid.meta.custom_blocks or {}).get(SCENE_KEY) if pyramid.meta else None
    if not isinstance(pointer, str):
        return
    here = str(path).replace("\\", "/").rstrip("/")
    owner = posixpath.normpath(posixpath.join(here, pointer))
    if not apply:
        warnings.warn(f"{path}: this image is placed by the NGFF 0.6 scene on {owner}; its "
                      f"levels hold only their alignment. Read it with apply_scene=True, or "
                      f"read {owner} with PyramidCollection.from_zarr", UserWarning,
                      stacklevel=4)
        return
    group = _open_store_group(owner, 3, mode="r", storage_options=storage_options)
    attrs = group.attrs.asdict()
    rel = posixpath.relpath(here, owner)
    key = (("image", 0),)
    systems = {}
    for obj in attrs.get(SYSTEMS_KEY) or []:
        cs = system_from_json(obj)
        systems[cs.id] = cs
    _, _, positions, _, names = ngff06.graph_from_scenes({"": attrs}, {rel: key}, systems, {},
                                                         str(owner))
    if key not in positions:
        warnings.warn(f"{path}: the scene on {owner} has no position for this image; its "
                      f"levels are left as they are", UserWarning, stacklevel=4)
        return
    if not ngff06.check_intrinsic(pyramid, names.get(key), str(path)):
        return
    p = positions[key]
    for pyr in [pyramid, *(getattr(pyramid, "_labels", None) or {}).values()]:
        for level in pyr.meta.resolution_paths:
            t = pyr.meta.get_translation(level) or [0.0] * len(p)
            pyr.meta.update_translation(level, [float(a) + float(b) for a, b in zip(t, p)])
    pyramid.meta.set_custom(SCENE_KEY, None)


def _store_parent(path):
    """The group above `path` in its store (the counterpart of `_store_join`)."""
    if _is_remote_store(path):
        return str(path).rstrip("/").rpartition("/")[0]
    return Path(path).parent


#: write_pyramid(windows=...)
_WINDOW_MODES = ("keep", "dtype", "data", "none", "auto")
#: windows="auto": the percentiles of the coarsest level that set start / end ...
AUTO_WINDOW_PERCENTILES = (0.1, 99.9)
#: ... when that level is at most this size; a larger one keeps the dtype's range
AUTO_WINDOW_MAX_MB = 64.0


def _apply_windows(pyramid, mode: str):
    """*pyramid* with every channel's omero window set from its dtype (``"dtype"``:
    the full range as ``min``/``max`` and ``start``/``end``) or removed (``"none"``)."""
    if not ((pyramid.meta.metadata or {}).get("omero") or {}).get("channels"):
        pyramid = pyramid.set_channels({})                  # defaults (from_arrays has none)
    n = len(pyramid.meta.metadata["omero"]["channels"])
    if mode == "none":
        return pyramid.set_channels({i: {"window": None} for i in range(n)})
    dtype = pyramid.layers[pyramid.meta.resolution_paths[0]].dtype
    dtype = np.dtype(getattr(dtype, "numpy_dtype", dtype))    # a TensorStore level too
    if dtype == np.bool_:
        lo, hi = 0, 1
    elif np.issubdtype(dtype, np.integer):
        lo, hi = int(np.iinfo(dtype).min), int(np.iinfo(dtype).max)
    else:
        lo, hi = float(np.finfo(dtype).min), float(np.finfo(dtype).max)
    window = {"min": lo, "max": hi, "start": lo, "end": hi}
    return pyramid.set_channels({i: {"window": dict(window)} for i in range(n)})


def _auto_window_limits(level, axes, where) -> Optional[tuple]:
    """``(starts, ends)`` per channel: the AUTO_WINDOW_PERCENTILES of `level` (the
    coarsest level of a pyramid, axes `axes`), read one channel at a time; None when the
    level is over AUTO_WINDOW_MAX_MB (the dtype's range stays, with a log line). A
    channel without finite values gets None."""
    dtype = np.dtype(getattr(level.dtype, "numpy_dtype", level.dtype))
    size_mb = int(np.prod(level.shape)) * dtype.itemsize / 2**20
    if size_mb > AUTO_WINDOW_MAX_MB:
        logger.info(f"windows='auto': the coarsest level of {where} is {size_mb:.0f} MB "
                    f"(over {AUTO_WINDOW_MAX_MB:.0f} MB): windows from the dtype")
        return None
    axes = list(axes)
    c = axes.index('c') if 'c' in axes else None
    n = level.shape[c] if c is not None else 1
    integer = np.issubdtype(dtype, np.integer)
    lows, highs = [], []
    for k in range(n):                               # one channel in memory at a time
        key = tuple(k if i == c else slice(None) for i in range(len(axes)))
        data = np.asarray(level[key])
        if data.dtype == np.bool_:
            data = data.astype(np.uint8)
        if not np.isfinite(data).any():
            lows.append(None)
            highs.append(None)
            continue
        lo, hi = np.nanpercentile(data, AUTO_WINDOW_PERCENTILES)
        lows.append(int(np.floor(lo)) if integer else float(lo))
        highs.append(int(np.ceil(hi)) if integer else float(hi))
    return lows, highs


def _data_window_limits(pyramid) -> tuple:
    """``(starts, ends)`` per channel: the minimum and maximum of level 0, streamed in
    bounded memory (``Pyramid.set_display_range``)."""
    measured = pyramid.set_display_range("minmax")
    base = pyramid.layers[pyramid.meta.resolution_paths[0]]
    dtype = np.dtype(getattr(base.dtype, "numpy_dtype", base.dtype))
    cast = int if np.issubdtype(dtype, np.integer) else float
    windows = [ch.get("window") or {} for ch in measured.meta.metadata["omero"]["channels"]]
    return ([cast(w["start"]) if w.get("start") is not None else None for w in windows],
            [cast(w["end"]) if w.get("end") is not None else None for w in windows])


#: update_metadata(windows=...)
_UPDATE_WINDOW_MODES = ("keep", "dtype", "data", "auto", "none")


def _write_ome_xml(path, xml: str, series, version: str,
                   storage_options: Optional[dict] = None, *, layout: bool = False) -> None:
    """``<path>/OME/METADATA.ome.xml``, in the output's NGFF `version`, written through
    the zarr store of the output (local and object-store outputs alike).

    - `layout` True (a bioformats2raw layout, ``write_series``): ``OME`` is a zarr group
      whose attributes list the image groups, *series* (0.4: ``series``, later versions:
      ``ome.series`` beside ``ome.version``), as bioformats2raw lays it out.
    - `layout` False (a single image, ``write_pyramid`` / ``update_metadata``): the XML
      is a plain file and ``OME`` no group. NGFF has an ``OME`` group only in a
      bioformats2raw layout, whose root carries ``bioformats2raw.layout``; a group with
      ``series`` under an image root (what ozp 0.3.0 wrote, its ``series`` naming the
      level-0 array) is rejected by validators that walk the hierarchy. Such a group is
      removed (:func:`_drop_stray_ome_group`), the XML kept."""
    from ome_zarr_pyramid.core.ngff_plate import zarr_format_of
    from ome_zarr_pyramid.core.pyramid import _run_coroutine
    from zarr.core.buffer import default_buffer_prototype
    if not isinstance(xml, str) or not xml.strip():
        raise ValueError("ome_xml: an OME-XML document (a non-empty string) is needed")
    version = str(version)
    zarr_format = zarr_format_of(version)          # raises for a version ozp does not write
    root = _open_store_group(path, zarr_format, mode="a", storage_options=storage_options)
    if layout:
        series = [str(s) for s in (series if series is not None else [0])]
        ome = root.require_group("OME")
        if version == "0.4":
            ome.attrs["series"] = series
        else:
            ome.attrs["ome"] = {"version": version, "series": series}
        store_path = ome.store_path
    else:
        _drop_stray_ome_group(root, path)
        store_path = root.store_path / "OME"
    key = f"{store_path.path}/METADATA.ome.xml" if store_path.path else "METADATA.ome.xml"
    root = getattr(store_path.store, "root", None)
    if root is not None:
        # local disk: straight to the final name. zarr's own write goes through a
        # temporary "METADATA.ome.<32 hex>.partial", 37 characters longer, which crossed
        # Windows' 260-character path limit where the file itself fits.
        target = Path(root) / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(xml.encode("utf-8"))      # as given (text mode adds CRs)
        return
    data = default_buffer_prototype().buffer.from_bytes(xml.encode("utf-8"))
    _run_coroutine(store_path.store.set(key, data))


def _drop_stray_ome_group(root: zarr.Group, path) -> bool:
    """Remove the ``OME`` group's metadata (``zarr.json``; zarr v2: ``.zgroup`` and
    ``.zattrs``) under an image root that is no bioformats2raw layout, keeping
    ``METADATA.ome.xml``: the stray group ozp 0.3.0 wrote with an image's OME-XML. True
    when one was removed (with a warning naming the store)."""
    from ome_zarr_pyramid.core.pyramid import _run_coroutine
    attrs = root.attrs.asdict()
    ome = attrs.get("ome") if isinstance(attrs.get("ome"), dict) else attrs
    if "bioformats2raw.layout" in ome or "plate" in ome:
        return False
    store, prefix = root.store_path.store, root.store_path.path
    names = ("zarr.json",) if root.metadata.zarr_format == 3 else (".zgroup", ".zattrs")
    keys = ["/".join(p for p in (prefix, "OME", n) if p) for n in names]

    async def drop():
        found = [k for k in keys if await store.exists(k)]
        for k in found:
            await store.delete(k)
        return found
    removed = _run_coroutine(drop())
    if removed:
        warnings.warn(f"{path}: removed the OME group's metadata ({', '.join(removed)}): an "
                      f"image root that is no bioformats2raw layout holds its OME-XML as a "
                      f"plain file, OME/METADATA.ome.xml (ozp 0.3.0 also wrote a group, "
                      f"which validators reject)", UserWarning, stacklevel=3)
    return bool(removed)


def _open_store_group(path, zarr_format: int, mode: str = "a",
                      storage_options: Optional[dict] = None) -> zarr.Group:
    """Open/create a zarr group for attr read/write: local directly; an object-store URL
    (s3://, gs://, az://) through obstore with dyna-style storage_options. (The https://
    s3fs branch only serves the removed engines; write_pyramid translates https:// to
    s3:// before it gets here.)"""
    if _is_remote_store(path):
        if _is_legacy_https(path):
            store = tensorstore_writer.wrap_output_path(str(path))
        else:
            return _object_store_group(str(path), zarr_format, mode, storage_options)
    else:
        store = str(path)
    return zarr.open_group(store, mode=mode, zarr_format=zarr_format)


def _existing_group_version(path, storage_options: Optional[dict] = None) -> Optional[str]:
    """The NGFF version a label written into the existing group at `path` must take, or
    None when no group is there. A zarr v2 group is 0.4 (ozp writes v2 only as 0.4); a
    v3 group is its ``ome.version``, 0.5 when it records none. A v3 group declaring
    another version (0.7, say) raises :class:`UnsupportedNGFFVersion`: ozp writes 0.4,
    0.5 and 0.6."""
    if _is_remote_store(path):
        try:
            group = _object_store_group(str(path), None, "r", storage_options)
        except FileNotFoundError:
            return None
        zarr_format, attrs = group.metadata.zarr_format, dict(group.attrs)
    else:
        p = Path(path)
        if (p / "zarr.json").is_file():            # zarr-python reads v3 first too
            zarr_format = 3
            attrs = zarr.open_group(str(p), mode="r", zarr_format=3).attrs
        elif (p / ".zgroup").is_file():
            zarr_format, attrs = 2, {}
        else:
            return None
    if zarr_format == 2:
        return "0.4"
    version = str(dict(attrs.get("ome") or {}).get("version", "0.5"))
    if version not in ("0.5", "0.6"):
        from ome_zarr_pyramid.core.nodes import UnsupportedNGFFVersion
        raise UnsupportedNGFFVersion(
            f"{path}: an NGFF {version} group; ozp writes labels into NGFF 0.4 / 0.5 / "
            f"0.6 stores only (write the label to a store of its own, or convert the image "
            f"to version='0.5' first)")
    return version


def _labels_target_version(labels_group_path, storage_options: Optional[dict] = None
                           ) -> Optional[str]:
    """The NGFF version of the store a label goes into through ``labels_group_path``: the
    labels group's own, or, before that group exists, its image's (the parent group).
    None when neither exists (a new store: the label keeps its own version)."""
    version = _existing_group_version(labels_group_path, storage_options)
    if version is None:
        if _is_remote_store(labels_group_path):
            parent = str(labels_group_path).rstrip("/").rpartition("/")[0]
            if parent.endswith(":/"):                # the labels group is a bucket root
                return None
        else:
            parent = Path(labels_group_path).parent
        version = _existing_group_version(parent, storage_options)
    return version


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
    """Reads and writes NGFF OME-Zarr pyramids (and their labels); every level is
    written by ``dyna_zarr.io``, region by region and memory-bounded."""

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
                     storage_options: Optional[dict] = None,
                     apply_scene: Optional[bool] = False) -> Pyramid:
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
        apply_scene : bool or None
            For an image an NGFF 0.6 scene places (the scene form, plan 5.5: its group
            carries ``eubi:scene``, the path of the group holding the scene): its own
            levels hold only their alignment, its position is in the scene. True reads
            the scene (one more request) and adds the position to the levels; False
            (the default) leaves the levels as they are, with a warning naming the
            scene's group; None does neither (``PyramidCollection.from_zarr`` places
            its images itself).

        Returns
        -------
        Pyramid
            Loaded pyramid with metadata and arrays

        Raises
        ------
        UnsupportedNGFFVersion
            The store declares an NGFF version ozp does not read (0.7, say); raised
            from its attributes, before any array or label is opened.
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
            if apply_scene is not None:
                _place_by_scene(pyramid, path, apply_scene, storage_options)

            if self.verbose:
                logger.info(f"[PyramidIO] Loaded {pyramid.nlayers} layers, axes: {pyramid.axes}")

            return pyramid

        except Exception as e:
            logger.error(f"[PyramidIO] Failed to read pyramid: {str(e)}")
            raise

    def read_labels(self,
                    path: Union[str, Path],
                    name: Optional[str] = None,
                    storage_options: Optional[dict] = None) -> Pyramid:
        """Read a single OME-Zarr label image as a `Pyramid`, with its
        ``image-label`` metadata parsed (surfaced via `pyr.meta.image_label`).

        `path` may point directly at a label-image group, or at a source image with
        `name` given - then ``<path>/labels/<name>`` is read. The label is a normal
        multiscale pyramid, so every `Pyramid` operation works on it. An object-store
        URL works too, with ``storage_options`` as for `read_pyramid`.
        """
        if name is not None:
            path = _store_join(path, "labels", name)    # URL-safe for remote stores
        # a label image has no nested labels/; skip discovery
        return self.read_pyramid(path, include_labels=False, storage_options=storage_options,
                                 apply_scene=None)

    def _discover_labels(self, pyramid: Pyramid, path: Path) -> None:
        """Populate ``pyramid._labels`` from the image's NGFF ``labels/`` group, if
        present. Lazy and best-effort, never silent: a ``labels/`` group that cannot be
        read, or a label that cannot be opened (one declaring an NGFF version ozp does
        not read, 0.6, among them), is left out with a ``UserWarning`` naming it and
        why; the image itself still opens."""
        labels_dir = Path(path) / "labels"
        if not labels_dir.exists():
            return
        try:
            grp = zarr.open_group(str(labels_dir), mode="r")
            names = _read_label_names(grp)
        except Exception as exc:
            warnings.warn(f"the labels of {path} are left out: {type(exc).__name__} "
                          f"reading {labels_dir}: {exc}", UserWarning, stacklevel=3)
            return
        for nm in names:
            try:
                pyramid._labels[nm] = self.read_labels(labels_dir / nm)
            except Exception as exc:
                warnings.warn(f"label {nm!r} of {path} is left out: {type(exc).__name__} "
                              f"reading {labels_dir / nm}: {exc}", UserWarning,
                              stacklevel=3)

    @validating
    def write_pyramid(self,
                      pyramid: Pyramid,
                      path: Union[str, Path],
                      layers: Optional[List[str]] = None,
                      overwrite: bool = False,
                      max_threads: Optional[int] = None,
                      region_size_mb: float = 8.0,
                      max_downscale_processes: Optional[int] = None,
                      backend: Optional[str] = None,
                      compressor: Optional[str] = None,
                      compressor_params: Optional[dict] = None,
                      chunk_shape=None,
                      chunk_size_mb=None,
                      include_labels: bool = True,
                      labels_group_path: Optional[Union[str, Path]] = None,
                      label_name: Optional[str] = None,
                      storage_options: Optional[dict] = None,
                      shard_coefficients=None,
                      io_backend: Optional[str] = None,
                      dtype=None,
                      window_from_data: bool = False,
                      memory_budget_mb: Optional[float] = None,
                      windows: str = "keep",
                      ome_xml: Optional[str] = None,
                      ome_series: Optional[Sequence] = None,
                      max_workers: Optional[int] = None,
                      use_multiprocessing: Optional[bool] = None,
                      chunks=None,
                      engine: Optional[str] = None,
                      num_readers: Optional[int] = None,
                      gc_interval: Optional[float] = None,
                      fill_derived: Optional[bool] = None,
                      levels: Optional[List[str]] = None,
                      **removed) -> None:
        """Write a pyramid using region-wise processing.

        ``chunks`` is ``chunk_shape`` under dyna-zarr's name (``io.write(chunks=...)``);
        give one of the two.

        ``dyna_zarr.io.write`` writes every level, local or remote (``s3://`` and the
        other object-store URLs, with ``storage_options``). numpy, zarr, TensorStore and
        DynamicArray levels are pulled region by region (memory-bounded); dask levels are
        run by dask's own scheduler on threads and pushed into the same kind of output.
        The options are dyna-zarr's words (decision C-14, 0.5.0): ``backend`` is the
        storage backend, ``'tensorstore'`` (default) or ``'zarrista'`` (``io_backend`` is
        its old name and still works); ``engine`` is dyna's execution loop, ``'fused'``
        (dyna's default) or ``'pipeline'``, with ``num_readers`` (pipeline only) and
        ``gc_interval``; they apply to levels written one by one (a deferred downscale's
        one-pass write has dyna's own loop and refuses them). ``backend='auto'`` /
        ``'dyna'``, what ``backend`` named before 0.5.0 (ozp's engine, always dyna), warn
        and are ignored; the ``'sync'`` engine, removed in 0.1.0, is refused, as are its
        settings (``max_concurrency``, ``queue_size``, ``max_concurrent_layers``).

        A legacy ``https://host/bucket/key`` target is written as ``s3://bucket/key``
        with ``endpoint='https://host'``; its s3fs option names (``anon``, ``key``,
        ``secret``, ``token``, ``client_kwargs``) are translated.

        ``compressor`` / ``compressor_params`` (a name + params, a ``CompressorConfig``,
        or dyna-zarr's ``Codecs``) override the pyramid's codec.

        ``fill_derived`` (default True; decision 9.12-S): fill the NGFF SHOULD fields ozp
        derives exactly where the source lacks them - a multiscale's empty ``metadata`` and
        its ``type`` ``"unknown"`` when nothing names the method (and, in ``write_plate``,
        each acquisition's ``maximumfieldcount``). False writes them only as the source
        has them; the strict schemas then report them (``validate=True`` warns).
        ``shard_coefficients`` (a per-axis tuple, or ``{level: tuple}``; shard =
        coefficients x chunks, zarr v3 only) shards the levels, and ``backend``
        picks dyna's byte backend (``'tensorstore'`` or ``'zarrista'``). Without
        ``shard_coefficients`` a sharded base level's sharding is kept and applied to
        every level.

        Handles BOTH image and label pyramids: a label image is detected via
        ``pyramid.meta.is_label`` (it carries an ``image-label`` block) and its
        metadata is attached automatically; pass ``labels_group_path`` + ``label_name``
        to also register it under a parent ``labels/`` group (the NGFF convention).
        A label written into an existing store that way takes that store's NGFF
        version (the labels group's, or its image's). An IMAGE pyramid's attached
        `labels` (from ``add_image_label``) are written recursively into
        ``<path>/labels/<name>``, in the image's version.

        Storage chunking (decoupled from any PROCESSING tile size an op imposed):
        ``chunk_shape`` (a per-axis tuple, or a ``{level: tuple}`` dict) is used as
        given, each side capped at the level's extent. ``chunk_size_mb`` (scalar /
        per-level sequence / ``{level: mb}``) is a size that dyna turns into a shape
        aligned with each level's own grid. When neither is given, the
        pyramid's recorded ``_storage_chunks`` (from read) is restored, so a
        read -> process -> write round-trip keeps the original chunking even if an op
        rechunked the dask arrays to its tile size.

        Parallelism:

        * ``max_threads`` (default 4): regions processed at once, one per thread -
          each region is read, run through the op chain and written by one thread.
          An upper limit: ``memory_budget_mb`` can lower it.
        * ``max_downscale_processes`` (default None: off): write up to that
          many levels at once, one level per spawned process, each with its own
          ``max_threads``. Rarely useful and to be deprecated: it turns off the one-pass
          write (every level then reads level 0 again), level 0 holds most of the data,
          and peak memory is up to ``max_downscale_processes`` x one write.
          Works for zarr, numpy, TensorStore and op-chain levels (they pickle), not for
          dask levels.

        ``max_threads`` and ``max_downscale_processes`` were called ``max_workers`` and
        ``use_multiprocessing`` before 0.1.0; the old names still work, with a
        ``FutureWarning``.

        Parameters
        ----------
        pyramid : Pyramid
            Pyramid instance to write
        path : str or Path
            Output path for the zarr store
        layers : list of str, optional
            Specific level paths to write. If None, writes all levels. ``levels`` is the
            same, by the name the docs and dyna-zarr use.
        overwrite : bool, optional
            If True, overwrite existing store. Default is False.
        max_threads : int, optional
            Regions processed at once, one per thread (per process). Default is 4.
        region_size_mb : float, optional
            Target region size in MB. Default is 8.0.
        max_downscale_processes : int, optional
            Levels written at once in separate processes. Default None (off). Rarely
            useful; to be deprecated (see above).
        backend : str, optional
            The storage backend: ``'tensorstore'`` (default) or ``'zarrista'``, as in
            dyna-zarr (``io_backend`` is its old name) - see above.
        engine, num_readers, gc_interval : optional
            dyna-zarr's execution loop and its settings, passed to ``dyna_zarr.io.write``
            for each level - see above.
        dtype : optional
            Write every level as this dtype, saturating instead of wrapping
            (``Pyramid.saturate_cast``: uint16 4000 -> uint8 255, floats rounded,
            NaN -> 0); the omero windows are mapped the same way.
        memory_budget_mb : float, optional
            The memory this write may add to the process (what the process already
            holds is not counted). dyna caps its threads by it,
            and a local zarr v3 pyramid with shards larger than ``region_size_mb`` is
            processed in y/x tiles that keep the open output chunks within it, whatever
            the plane size (either ``io_backend``; zarrista needs zarrista-bounded).
            None: no cap.
        windows : str, optional
            The channels' omero display windows: ``"keep"`` (default) as the pyramid
            has them; ``"dtype"`` the full range of the dtype as WRITTEN (after
            ``dtype``: a uint16 -> uint8 write gets 0-255), as ``min``/``max`` and
            ``start``/``end``; ``"data"`` ``start``/``end`` = the min/max of the level 0
            data as written, measured while level 0 is written (no second read; not with
            ``max_downscale_processes``); ``"auto"`` ``min``/``max`` from the dtype and
            ``start``/``end`` = the 0.1 and 99.9 percentiles of the coarsest level as
            written, read back after the write (a quick overview) - or, when that level is
            over 64 MB (no downscaling, a deep stack kept in z), the dtype's range, as
            ``"dtype"``; ``"none"`` no window (viewers choose). For other percentiles or
            another level, use :meth:`Pyramid.set_display_range` before writing.
        window_from_data : bool, optional
            The older spelling of ``windows="data"``.
        compressor, compressor_params : optional
            Override the pyramid's codec (see above). ``compressor=None`` (default)
            uses ``pyramid.compressor`` (the input's own codec, or blosc for
            non-zarr-backed pyramids).
        ome_xml : str, optional
            An OME-XML document written with the pyramid, as the plain file
            ``OME/METADATA.ome.xml`` (no ``OME`` group: NGFF has one only in a
            bioformats2raw layout, see ``PyramidCollection.write_series``): in the same
            write, local or remote, so the output is not opened again afterwards. ozp
            writes it as given; building it is the caller's.
        ome_series : sequence, optional
            Deprecated, no effect (a single image gets no ``OME`` group since ozp
            0.3.1); a warning says so.

        Notes
        -----
        The output zarr format (v2/v3) and array
        compressor are taken from ``pyramid.meta.zarr_format`` /
        ``pyramid.compressor`` - i.e. the metadata carried by the in-memory
        ``Pyramid`` object, which by default mirrors the source it was read
        from.
        """
        if pyramid.meta is None:
            raise ValueError("Pyramid not initialized")
        _refuse_removed_settings(removed)
        if levels is not None:            # the level paths to write, by dyna's name (C-16)
            if layers is not None:
                raise ValueError("write_pyramid: give levels= or layers=, not both "
                                 "(levels is the name of layers since 0.5.0)")
            layers = levels
        if fill_derived is not None:
            from ome_zarr_pyramid.core.validation import filling
            with filling(fill_derived):
                return self.write_pyramid(
                    pyramid, path, layers=layers, overwrite=overwrite, max_threads=max_threads,
                    region_size_mb=region_size_mb,
                    max_downscale_processes=max_downscale_processes, backend=backend,
                    compressor=compressor, compressor_params=compressor_params,
                    chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb,
                    include_labels=include_labels, labels_group_path=labels_group_path,
                    label_name=label_name, storage_options=storage_options,
                    shard_coefficients=shard_coefficients, io_backend=io_backend, dtype=dtype,
                    window_from_data=window_from_data, memory_budget_mb=memory_budget_mb,
                    windows=windows, ome_xml=ome_xml, ome_series=ome_series,
                    max_workers=max_workers, use_multiprocessing=use_multiprocessing,
                    chunks=chunks, engine=engine, num_readers=num_readers,
                    gc_interval=gc_interval)
        if ome_xml is not None and (not isinstance(ome_xml, str) or not ome_xml.strip()):
            raise ValueError("ome_xml: an OME-XML document (a non-empty string) is needed")
        if ome_series is not None:
            warnings.warn("write_pyramid(ome_series=...) has no effect since ozp 0.3.1: a "
                          "single image's OME-XML is a plain file, OME/METADATA.ome.xml, with "
                          "no OME group (NGFF has one only in a bioformats2raw layout; "
                          "write_series writes that)", DeprecationWarning, stacklevel=2)
            ome_series = None
        if windows not in _WINDOW_MODES:
            raise ValueError(f"windows={windows!r}: one of {_WINDOW_MODES}")
        if window_from_data:
            if windows not in ("keep", "data"):
                raise ValueError(f"window_from_data=True is windows='data'; it cannot be "
                                 f"combined with windows={windows!r}")
            windows = "data"
        if windows != "keep" and layers is None and pyramid.meta.is_label:
            raise ValueError(f"windows={windows!r} sets omero channel windows; a label "
                             f"image has none.")
        if chunks is not None:
            if chunk_shape is not None:
                raise ValueError("write_pyramid: give chunks= or chunk_shape=, not both "
                                 "(chunks is dyna-zarr's name for chunk_shape)")
            chunk_shape = chunks
        max_threads = renamed(max_threads, max_workers, new="max_threads",
                              old="max_workers", where="write_pyramid", default=4)
        max_downscale_processes = downscale_processes(max_downscale_processes, use_multiprocessing,
                                                      where="write_pyramid")
        io_backend = _storage_backend(backend, io_backend)
        backend = None                   # the storage backend travels as io_backend below
        if engine is not None and engine not in _DYNA_ENGINES:
            raise ValueError(f"engine={engine!r}: dyna-zarr's loop is one of {_DYNA_ENGINES}")
        loop = {k: v for k, v in (("engine", engine), ("num_readers", num_readers),
                                  ("gc_interval", gc_interval)) if v is not None}
        _resolve_engine(pyramid, 'auto', max_downscale_processes=max_downscale_processes)
        # the legacy https://host/bucket/key form becomes s3://bucket/key + endpoint
        if _is_legacy_https(path):
            if labels_group_path is not None and _is_legacy_https(labels_group_path):
                labels_group_path, _ = _legacy_https_target(labels_group_path,
                                                            storage_options)
            path, storage_options = _legacy_https_target(path, storage_options)
        if windows == "auto" and layers is None:
            return self._write_auto_windows(
                pyramid, path, overwrite=overwrite, max_threads=max_threads,
                region_size_mb=region_size_mb,
                max_downscale_processes=max_downscale_processes, backend=backend,
                compressor=compressor, compressor_params=compressor_params,
                chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb,
                include_labels=include_labels, labels_group_path=labels_group_path,
                label_name=label_name, storage_options=storage_options,
                shard_coefficients=shard_coefficients, io_backend=io_backend, dtype=dtype,
                memory_budget_mb=memory_budget_mb, ome_xml=ome_xml, ome_series=ome_series,
                **loop)
        if windows == "data" and layers is None:
            return self._write_measuring_windows(
                pyramid, path, overwrite=overwrite, max_threads=max_threads,
                region_size_mb=region_size_mb,
                max_downscale_processes=max_downscale_processes, backend=backend,
                compressor=compressor, compressor_params=compressor_params,
                chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb,
                include_labels=include_labels, labels_group_path=labels_group_path,
                label_name=label_name, storage_options=storage_options,
                shard_coefficients=shard_coefficients, io_backend=io_backend, dtype=dtype,
                memory_budget_mb=memory_budget_mb, ome_xml=ome_xml, ome_series=ome_series,
                **loop)
        if dtype is not None:
            pyramid = pyramid.saturate_cast(dtype)
        if windows in ("dtype", "none") and layers is None:
            pyramid = _apply_windows(pyramid, windows)

        # an object-store URL stays a STRING (Path would mangle 's3://' -> 's3:\')
        if not _is_remote_store(path):
            path = Path(path) if isinstance(path, str) else path

        # a label written into an existing store (labels_group_path) takes that store's
        # NGFF version, as add_image_label does: v2 and v3 groups do not nest, so a v3
        # label under a v2 labels/ group (or the reverse) is invisible to readers, and a
        # second labels/ group of the other format hides the labels already there.
        if layers is None and labels_group_path is not None:
            target = _labels_target_version(labels_group_path, storage_options)
            if target is not None and target != str(pyramid.meta.version):
                pyramid = pyramid.to_version(target)

        # capture any attached labels NOW (top-level image write only): the storage
        # rechunk below rebinds `pyramid` to a clone that does not carry `_labels`.
        # They are written into `<path>/labels/<name>` after the image (see
        # `_write_attached_labels`). `layers is not None` = an internal sub-write.
        write_lbls = layers is None and include_labels and bool(getattr(pyramid, '_labels', None))
        attached_labels = dict(getattr(pyramid, '_labels', {})) if write_lbls else {}
        # a label is written in its image's NGFF version: zarr v2 and v3 groups do not
        # nest, so a v3 label under a v2 image (or the reverse) is invisible to readers.
        attached_labels = {nm: (lbl if str(lbl.meta.version) == str(pyramid.meta.version)
                                else lbl.to_version(str(pyramid.meta.version)))
                           for nm, lbl in attached_labels.items()}

        # LABEL image? detect via meta.is_label and capture its image-label block now
        # (so it survives the rechunk rebind below); attached after the arrays. Register
        # under a parent labels/ group FIRST, so the nested group is created inside it.
        label_block = pyramid.meta.image_label if layers is None else None
        label_version, label_zarr_format = pyramid.meta.version, pyramid.meta.zarr_format
        if layers is None and labels_group_path is not None and label_name is not None:
            lg = _open_store_group(labels_group_path, label_zarr_format, mode="a",
                                   storage_options=storage_options)
            from ome_zarr_pyramid.core.ngff06 import intrinsic_of
            _register_label_name(lg, label_name, label_version,
                                 image_path=_store_parent(labels_group_path),
                                 label_system=intrinsic_of(pyramid.meta.multiscales),
                                 storage_options=storage_options)
        side = dict(label_block=label_block, label_version=label_version,
                    label_zarr_format=label_zarr_format, attached_labels=attached_labels,
                    overwrite=overwrite, backend=io_backend, compressor=compressor,
                    compressor_params=compressor_params, storage_options=storage_options,
                    ome_xml=ome_xml, ome_series=ome_series)

        # A DEFERRED-downscale pyramid (from `Pyramid.downscale(defer=True)`) carries
        # only level 0 plus a plan; the levels are written from ONE read of level 0
        # (or, where that cannot run, from the stored level 0), so an expensive lazy
        # base is computed once, not re-run per output level. (Skip when a
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
            if engine not in (None, 'fused') or loop.get("num_readers") is not None:
                raise ValueError(
                    f"engine={engine!r} / num_readers apply to levels written one by one; "
                    f"this pyramid's levels are computed in one pass from level 0 (a "
                    f"deferred downscale), which has dyna's own loop: drop them")
            self._write_with_plan(
                pyramid, path, plan, overwrite=overwrite,
                chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb,
                max_threads=max_threads, region_size_mb=region_size_mb,
                max_downscale_processes=max_downscale_processes,
                compressor=compressor, compressor_params=compressor_params,
                shard_coefficients=shard_coefficients, io_backend=io_backend,
                storage_options=storage_options, memory_budget_mb=memory_budget_mb,
            )
            self._write_side_metadata(path, **side)
            return

        # every level through dyna_zarr.io.write, which streams the whole
        # read -> op-chain -> write pipeline region by region (numpy / zarr / TensorStore
        # levels are wrapped as DynamicArrays). Also serves the internal layer-subset
        # writes of the plan path.
        self._write_pyramid_dyna(
            pyramid, path, layers=layers, overwrite=overwrite,
            chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb,
            shard_coefficients=shard_coefficients, compressor=compressor,
            compressor_params=compressor_params, region_size_mb=region_size_mb,
            max_threads=max_threads, io_backend=io_backend,
            max_downscale_processes=max_downscale_processes, storage_options=storage_options,
            memory_budget_mb=memory_budget_mb, loop=loop)
        if layers is not None:
            return    # internal sub-write: the caller writes the side metadata
        self._write_side_metadata(path, **side)

    def _write_auto_windows(self, pyramid, path, **kwargs) -> None:
        """write_pyramid(windows="auto"): written with the dtype's windows, then each
        channel's start / end set to the AUTO_WINDOW_PERCENTILES of the COARSEST written
        level - read back from the output, so it is the data as written (after
        ``dtype``) and the source is not read again. A coarsest level over
        AUTO_WINDOW_MAX_MB (no downscaling, a deep stack kept in z) keeps the dtype's
        windows, with a log line."""
        self.write_pyramid(pyramid, path, windows="dtype", **kwargs)
        storage_options = kwargs.get('storage_options')
        zarr_format = int(pyramid.meta.zarr_format)
        if _is_remote_store(path):
            group = _object_store_group(str(path), zarr_format, "r", storage_options)
        else:
            group = zarr.open_group(str(path), mode="r", zarr_format=zarr_format)
        attrs = group.attrs.asdict()
        from ome_zarr_pyramid.core.ngff06 import ozp_multiscales
        multiscale = ozp_multiscales(attrs, str(path))[1][0]
        axes = [a['name'] if isinstance(a, dict) else a for a in multiscale['axes']]
        coarsest = group[multiscale['datasets'][-1]['path']]
        found = _auto_window_limits(coarsest, axes, path)
        if found is None:
            return
        lows, highs = found
        self._set_written_windows(path, zarr_format, SimpleNamespace(min=lows, max=highs),
                                  storage_options)

    def update_metadata(self, path, *, scales: Optional[dict] = None,
                        units: Optional[dict] = None,
                        channels: Optional[dict] = None, windows: str = "keep",
                        ome_xml=None, ome_series=None,
                        storage_options: Optional[dict] = None,
                        fill_derived: Optional[bool] = None) -> Pyramid:
        """Edit the metadata of the OME-Zarr image at `path` in place; its pixels are
        never touched (read, not rewritten). Returns the updated `Pyramid`.

        - `scales` / `units`: ``{axis: value}`` for level 0 (``Pyramid.update_scales``:
          every coarser level keeps its factor) and the axes' units.
        - `channels`: ``Pyramid.set_channels`` overrides (``{index or label: {field:
          value}}``).
        - `windows`: the omero windows. ``"keep"`` leaves them; ``"dtype"`` sets the
          dtype's full range; ``"data"`` sets start / end to each channel's minimum /
          maximum of level 0 (read through, streamed; min / max the dtype's range);
          ``"auto"`` sets start / end to the AUTO_WINDOW_PERCENTILES of the coarsest
          level (the dtype's range when that is over AUTO_WINDOW_MAX_MB), as
          ``write_pyramid(windows="auto")``; ``"none"`` removes them.
        - `ome_xml`: an OME-XML document (a string, or a function of the updated
          pyramid returning one) written to ``OME/METADATA.ome.xml``, as by
          ``write_pyramid``; `ome_series` as there (no effect).

        A store ozp 0.3.0 wrote with an OME-XML holds a stray ``OME`` group under the
        image root, which validators reject; this repairs it, removing the group's
        metadata and keeping ``OME/METADATA.ome.xml`` (with a warning).

        Local stores and object-store URLs (with `storage_options`)."""
        if windows not in _UPDATE_WINDOW_MODES:
            raise ValueError(f"update_metadata: windows={windows!r}: one of "
                             f"{_UPDATE_WINDOW_MODES}")
        if fill_derived is not None:
            from ome_zarr_pyramid.core.validation import filling
            with filling(fill_derived):
                return self.update_metadata(path, scales=scales, units=units,
                                            channels=channels, windows=windows,
                                            ome_xml=ome_xml, ome_series=ome_series,
                                            storage_options=storage_options)
        pyramid = self.read_pyramid(path, include_labels=False,
                                    storage_options=storage_options, apply_scene=None)
        order = pyramid.meta.axis_order
        if "c" in order and pyramid.meta.unit_list[order.index("c")]:
            # NGFF gives a channel axis no unit, and some viewers (Neuroglancer) reject
            # the whole image over one: an update repairs it
            pyramid.meta.update_unitlist([None if ax == "c" else u
                                          for ax, u in zip(order, pyramid.meta.unit_list)])
        if scales:
            pyramid.update_scales(**{ax: float(v) for ax, v in scales.items()})
        if units:
            pyramid.update_units(**units)
        if channels:
            pyramid = pyramid.set_channels(channels)
        if windows in ("dtype", "none"):
            pyramid = _apply_windows(pyramid, windows)
        elif windows in ("data", "auto"):
            pyramid = _apply_windows(pyramid, "dtype")
            meta = pyramid.meta
            found = (_data_window_limits(pyramid) if windows == "data" else
                     _auto_window_limits(pyramid.layers[meta.resolution_paths[-1]],
                                         meta.axis_order, path))
            if found is not None:
                pyramid = pyramid.set_channels({
                    k: {"window": {"start": lo, "end": hi}}
                    for k, (lo, hi) in enumerate(zip(*found)) if lo is not None})
        zarr_format = int(pyramid.meta.zarr_format)
        if _is_remote_store(path):
            group = _object_store_group(str(path), zarr_format, "a", storage_options)
        else:
            group = zarr.open_group(str(path), mode="r+", zarr_format=zarr_format)
        pyramid.meta.zarr_group = group
        pyramid.meta._pending_changes = True
        pyramid.meta.save_changes()
        _drop_stray_ome_group(group, path)
        if ome_xml is not None:
            xml = ome_xml(pyramid) if callable(ome_xml) else ome_xml
            _write_ome_xml(path, xml, ome_series, pyramid.meta.version, storage_options)
        return pyramid

    def _write_measuring_windows(self, pyramid, path, **kwargs) -> None:
        """write_pyramid(window_from_data=True): the write, with level 0 observed by
        ``dyna_zarr.io.Extrema`` on the channel axis, then the measured per-channel
        min/max set as the omero window start/end on the written group."""
        from dyna_zarr import io as dyna_io
        if kwargs['max_downscale_processes']:
            raise ValueError(
                "window_from_data measures level 0 while it is written, but with "
                "max_downscale_processes > 1 level 0 is written in another process. Drop "
                "max_downscale_processes, or set the windows before writing with "
                "Pyramid.set_display_range().")
        if pyramid.meta.is_label:
            raise ValueError("window_from_data sets omero channel windows; a label image "
                             "has none.")
        if not ((pyramid.meta.metadata or {}).get('omero') or {}).get('channels'):
            pyramid = pyramid.set_channels({})     # defaults (from_arrays has none)
        axes = pyramid.meta.axis_order
        observer = dyna_io.Extrema(axis=axes.index('c') if 'c' in axes else None)
        self._level0_observer = observer
        try:
            self.write_pyramid(pyramid, path, **kwargs)
        finally:
            self._level0_observer = None
        self._set_written_windows(path, int(pyramid.meta.zarr_format), observer,
                                  kwargs.get('storage_options'))

    @staticmethod
    def _set_written_windows(path, zarr_format, observer, storage_options) -> None:
        """Set omero window start/end from ``observer`` on the group written at ``path``
        (OME-Zarr 0.5 keeps omero under ``ome``, 0.4 at the top level)."""
        if _is_remote_store(path):
            group = _object_store_group(str(path), zarr_format, "a", storage_options)
        else:
            group = zarr.open_group(str(path), mode="r+", zarr_format=zarr_format)
        attrs = group.attrs.asdict()
        holder = attrs.get('ome') if isinstance(attrs.get('ome'), dict) else attrs
        channels = (holder.get('omero') or {}).get('channels') or []
        if not channels:
            return
        for k, (lo, hi) in enumerate(zip(observer.min, observer.max)):
            if k < len(channels) and lo is not None:
                window = channels[k].setdefault('window', {})
                window['start'], window['end'] = lo, hi
        if holder is attrs:
            group.attrs.update({'omero': holder['omero']})
        else:
            group.attrs.update({'ome': holder})

    def _write_pyramid_dyna(self, pyramid, path, *, layers, overwrite, chunk_shape,
                            chunk_size_mb, shard_coefficients, compressor, compressor_params,
                            region_size_mb, max_threads, io_backend,
                            max_downscale_processes=None, storage_options=None,
                            memory_budget_mb=None, loop=None):
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
        # dask levels stay dask (io.write runs them with dask's scheduler); everything
        # else is wrapped as a DynamicArray and pulled region by region by io.write
        from dyna_zarr import DynamicArray
        das = {}
        for p in paths:
            a = pyramid.layers[p]
            das[p] = a if (is_dask_array(a) or isinstance(a, DynamicArray)) else DynamicArray(a)

        codecs = _writable_codecs(
            _to_dyna_codecs(_resolve_compressor(pyramid, compressor, compressor_params)),
            das[paths[0]], zf, explicit=compressor is not None)
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
            kw = _dyna_level_kwargs(
                pyramid, lvl, p, arr, zf=zf, max_threads=max_threads, io_backend=io_backend,
                region_size_mb=region_size_mb, chunk_shape=chunk_shape,
                chunk_size_mb=chunk_size_mb, recorded=recorded, storage=storage,
                codecs=codecs, shard_coefficients=shard_coefficients, remote=remote,
                storage_options=storage_options, loop=loop)
            if lvl == 0 and getattr(self, '_level0_observer', None) is not None:
                kw['observe'] = self._level0_observer
            if memory_budget_mb is not None:
                kw['memory_budget_mb'] = float(memory_budget_mb)
            jobs.append((p, arr, str(_store_join(path, p)), kw))

        if max_downscale_processes and len(jobs) > 1:
            # one spawned process per level, max_downscale_processes at once (the arrays pickle:
            # zarr / TensorStore by reference, numpy staging by value, op chains as-is)
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(len(jobs), max_downscale_processes),
                                     mp_context=multiprocessing.get_context("spawn")) as ex:
                futures = [(p, ex.submit(_dyna_write_level, arr, out, kw))
                           for p, arr, out, kw in jobs]
                logs = [(p, f.result()) for p, f in futures]   # re-raises a worker error
        else:
            logs = [(p, _dyna_write_level(arr, out, kw)) for p, arr, out, kw in jobs]
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
                             attached_labels: dict, overwrite: bool, backend: Optional[str] = None,
                             compressor=None, compressor_params=None,
                             storage_options=None, ome_xml=None, ome_series=None) -> None:
        """Post-array-write side metadata, shared by every write_pyramid exit:
        (1) if this is a LABEL image, attach its ``image-label`` block to the written
        group; (2) write any attached image `labels` into ``<path>/labels/<name>``;
        (3) the OME-XML (``ome_xml``) into ``<path>/OME/METADATA.ome.xml``."""
        if label_block is not None:
            group = _open_store_group(path, label_zarr_format, mode="a",
                                      storage_options=storage_options)
            _attach_image_label(group, label_block, label_version)
        if attached_labels:
            self._write_attached_labels(attached_labels, path, overwrite=overwrite,
                                        backend=backend, compressor=compressor,
                                        compressor_params=compressor_params,
                                        storage_options=storage_options)
        if ome_xml is not None:
            _write_ome_xml(path, ome_xml, ome_series, label_version, storage_options)

    def _write_attached_labels(self, labels: dict, path, *, overwrite: bool,
                               backend: Optional[str] = None, compressor=None,
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


    def _write_plan_one_pass(self, pyramid, base, path, plan, dk, *, overwrite, rechunked,
                             write_kwargs) -> bool:
        """Write a deferred-downscale pyramid from ONE read of the source, if it can.

        ``dyna_zarr.io.write_levels`` writes level 0 and as many coarser levels as its
        region budget allows from the same regions of the source (every chunk written
        whole, once); the remaining small levels follow from the stored last level
        (stride / min / max compose) or from the stored level 0 (mean / median / mode do
        not). The old path wrote level 0, re-read it, and wrote one level at a time.

        Returns False, having written nothing, where it does not apply:
        max_downscale_processes, a dask base, a single level, or a plan reproducing
        floor-sized levels (those are cropped on the old path).
        """
        if write_kwargs.get('max_downscale_processes'):
            return False
        from ome_zarr_pyramid.utils.scale import _level_size, downscale_level
        base_path0 = base.meta.resolution_paths[0]
        base_arr = base._arrays()[base_path0]
        if is_dask_array(base_arr):
            return False
        dmethod = plan.get('downscale_method', 'simple')
        dbackend = plan.get('downscale_backend', 'tensorstore')
        full = base.downscale(defer=False, **dk)          # lazy: shapes and scales only
        fpaths = full.meta.resolution_paths
        if len(fpaths) < 2:
            return False
        factors, targets = _plan_level_factors(plan, full, base_arr.shape, dmethod)
        if any(tuple(_level_size(b, f, dmethod) for b, f in zip(base_arr.shape, fac)) != t
               for fac, t in zip(factors, targets)):
            return False

        from dyna_zarr import DynamicArray, operations as dyna_ops
        from dyna_zarr import io as dyna_io

        def lazy_level(src, factor):
            lvl = downscale_level(src, factor, dmethod, dbackend)
            return lvl if hasattr(lvl, '_with_transform') else DynamicArray(lvl)

        arrays = [base_arr] + [lazy_level(base_arr, f) for f in factors[1:]]
        built = _full_pyramid(pyramid, full, arrays, rechunked, dmethod)

        remote = _is_remote_store(path)
        path = str(path) if remote else Path(path)
        zf = int(built.meta.zarr_format)
        storage_options = write_kwargs.get('storage_options')
        io_backend = write_kwargs.get('io_backend', 'tensorstore')
        codecs = _to_dyna_codecs(_resolve_compressor(base, write_kwargs.get('compressor'),
                                                     write_kwargs.get('compressor_params')))
        shard_coefficients = write_kwargs.get('shard_coefficients')
        if shard_coefficients is None and zf == 3:
            shard_coefficients = _inherited_shard_coefficients(base_arr)
        if shard_coefficients is not None and zf != 3:
            raise ValueError(
                f"shard_coefficients need zarr v3, but this pyramid is OME-Zarr "
                f"{built.meta.version} (zarr v{zf}).")
        common = dict(zf=zf, max_threads=write_kwargs.get('max_threads', 4),
                      io_backend=io_backend, region_size_mb=write_kwargs.get('region_size_mb'),
                      chunk_shape=write_kwargs.get('chunk_shape'),
                      chunk_size_mb=write_kwargs.get('chunk_size_mb'), codecs=codecs,
                      shard_coefficients=shard_coefficients, remote=remote,
                      storage_options=storage_options)
        composable = dmethod in ('simple', 'min', 'max')

        # per-level output options: level 0 as the base pyramid would be written, the
        # rest as the built pyramid's levels; a level without explicit chunks keeps what
        # the old path gave it - the grid of the array it was derived from
        outputs, out_chunks = [], []
        for i, (p, factor) in enumerate(zip(fpaths, factors)):
            owner = base if i == 0 else built
            kw = _dyna_level_kwargs(owner, i, p, arrays[i],
                                    recorded=dict(getattr(owner, '_level_chunks', None) or {}),
                                    storage=getattr(owner, '_storage_chunks', None), **common)
            chunks = kw.get('chunks')
            if chunks is None and kw.get('chunk_size_mb') is None:
                if i == 0:
                    grid = base_arr.chunks
                else:
                    parent = i - 1 if composable and all(
                        c % q == 0 for c, q in zip(factor, factors[i - 1])) else 0
                    stand_in = dyna_ops.empty(targets[parent], dtype=base_arr.dtype,
                                              chunks=out_chunks[parent])
                    rel = tuple(c // q for c, q in zip(factor, factors[parent]))
                    grid = lazy_level(stand_in, rel).chunks
                if grid is not None:
                    chunks = tuple(min(int(c), int(n)) for c, n in zip(grid, targets[i]))
            out_chunks.append(chunks if chunks is not None else tuple(targets[i]))
            outputs.append(dict(path=str(_store_join(path, p)), factors=factor,
                                chunks=chunks, chunk_size_mb=kw.get('chunk_size_mb'),
                                shard_coefficients=kw.get('shard_coefficients')))

        if remote:
            zarr_group = _object_store_group(path, zf, "w" if overwrite else "a",
                                             storage_options)
        else:
            zarr_group = tensorstore_writer._zarr_group(str(path), overwrite=overwrite,
                                                        zarr_format=zf)
        level_kw = dict(method=dmethod, max_threads=common['max_threads'],
                        region_size_mb=common['region_size_mb'], compressor=codecs,
                        zarr_format=zf, backend=io_backend,
                        storage_options=dict(storage_options) if remote and storage_options
                        else None, overwrite=False,
                        dimension_names=tuple(built.axes) if zf == 3 else None,
                        reducer=(_ts_level_reducer if dbackend == 'tensorstore'
                                 and dmethod != 'simple' else None),
                        memory_budget_mb=write_kwargs.get('memory_budget_mb'))
        src, src_factor, i, log = base_arr, factors[0], 0, []
        observer = getattr(self, '_level0_observer', None)
        budget = write_kwargs.get('memory_budget_mb')
        start_rss = _process_rss_mb() if budget is not None else None
        while i < len(outputs):
            rel = [dict(o, factors=tuple(f // q for f, q in zip(o['factors'], src_factor)))
                   for o in outputs[i:]]
            if budget is not None and i > 0 and start_rss is not None:
                # the whole write may add `budget`: a later pass gets what the earlier
                # ones have not kept (freed memory the allocator did not hand back,
                # reader caches; ~90 MB after a nematostella first pass)
                grown = max(0.0, (_process_rss_mb() or start_rss) - start_rss)
                level_kw['memory_budget_mb'] = max(budget / 4, budget - grown)
            with dyna_io.capture_messages() as messages:     # thread-safe, unlike
                n = dyna_io.write_levels(src, rel, **level_kw,  # redirect_stdout
                                         observe=observer if i == 0 else None)
            log.append("\n".join(messages))
            i += n
            if i < len(outputs):
                # the rest from a stored level: the last one written where every remaining
                # factor is a multiple of its own (stride / min / max compose), else level 0
                parent = i - 1 if composable and all(
                    all(f % q == 0 for f, q in zip(o['factors'], factors[i - 1]))
                    for o in outputs[i:]) else 0
                # read with the write's own byte backend: dyna's default reader
                # (TensorStore) holds ~2x a block while reading it, zarrista ~1.15x
                src = dyna_io.read(outputs[parent]['path'], backend=io_backend,
                                   storage_options=level_kw['storage_options'])
                src_factor = factors[parent]
        if self.verbose:
            for text in log:
                logger.info("[PyramidIO] one pass: " + " | ".join(
                    ln.strip() for ln in text.splitlines() if ln.strip()))
        built.meta.connect_to_group(zarr_group)
        built.meta._pending_changes = True
        built.meta.save_changes()
        return True

    def _write_with_plan(self, pyramid: Pyramid, path: Path, plan: dict, *,
                         overwrite: bool, chunk_shape=None, chunk_size_mb=None,
                         **write_kwargs) -> None:
        """Write a DEFERRED-downscale pyramid; the base is computed ONCE either way.

        Level 0 and the coarser levels are written from one read of the source
        (`_write_plan_one_pass`). Where that cannot run (``max_downscale_processes``
        above 1, a dask level 0, a plan reproducing floor-sized legacy levels):

        1. stream the base (level 0) to the store ONCE;
        2. re-read that on-disk base;
        3. expand the downscale plan on the DISK-backed base (each coarser level is
           a cheap read+downsample of the stored base, not a recompute);
        4. write the remaining levels (>=1) and the full multiscale metadata.

        Never materialises the whole volume in memory. Reused for both `write_pyramid`
        and `write_labels` (labels downsample with 'simple' = nearest, which the plan
        records)."""
        base_path0 = pyramid.meta.resolution_paths[0]
        # a chunk SIZE is dyna's to turn into a shape, as on the plain route
        write_kwargs['chunk_size_mb'] = chunk_size_mb

        def _rechunked(p):
            """Apply the requested storage chunking (explicit, else restore recorded)."""
            if chunk_shape is not None:
                return p.rechunk(chunk_shape=chunk_shape)
            if getattr(p, '_storage_chunks', None) is not None:
                return p.rechunk()
            return p

        # 1) the base: level 0 only, rechunked to storage chunks, plan removed to avoid
        # recursion (written below unless the one-pass write does everything).
        # NARROW to level 0 first: the pyramid may still carry coarser levels (ops
        # propagate the plan across all of them), and writing with `layers=[base]` alone
        # would leave the group metadata advertising levels that were never written, so
        # the re-read in step 2 fails with `KeyError`. The plan rebuilds those levels.
        saved = pyramid.__dict__.pop('_downscale_plan', None)
        try:
            source = (pyramid.select_levels(0)
                      if len(pyramid.meta.resolution_paths) > 1 else pyramid)
            source.__dict__.pop('_downscale_plan', None)
            base = _rechunked(source)
        finally:
            if saved is not None:
                pyramid._downscale_plan = saved
        dk = {k: plan[k] for k in ('n_layers', 'min_dimension_size', 'scale_factor',
                                   'downscale_method', 'backend', 'smart_scale_factor')}
        dk['downscale_backend'] = plan.get('downscale_backend', 'tensorstore')

        # ONE PASS: level 0 and the coarser levels from ONE read of the
        # source, without re-reading the stored level 0 (see _write_plan_one_pass).
        if self._write_plan_one_pass(pyramid, base, path, plan, dk, overwrite=overwrite,
                                     rechunked=_rechunked, write_kwargs=write_kwargs):
            return
        self.write_pyramid(base, path, layers=[base_path0], overwrite=overwrite, **write_kwargs)

        # 2) re-read the on-disk base (zarr-backed -> downscaling reads from DISK)
        storage_options = write_kwargs.get('storage_options')
        disk = self.read_pyramid(str(path), include_labels=False,
                                 storage_options=storage_options, apply_scene=None)

        # 3) expand the plan for the level count / shapes / scales (metadata only)
        full = disk.downscale(defer=False, **dk)
        fpaths = full.meta.resolution_paths
        extra = list(fpaths[1:])
        if not extra:
            return

        # 4) build the coarser levels by downsampling the ON-DISK base (reads the stored
        # base, never the expensive source graph)
        from ome_zarr_pyramid.utils.scale import crop_to_level, downscale_level
        dmethod = plan.get('downscale_method', 'simple')
        dbackend = plan.get('downscale_backend', 'tensorstore')
        base_da = disk._arrays()[disk.meta.resolution_paths[0]]

        def method(base, scale_factor):
            # TensorStore's downsample or dyna's coarsen (downscale_backend): the same
            # pixels; dask stays dask. A TensorStore view is wrapped as a DynamicArray.
            level = downscale_level(base, scale_factor, dmethod, dbackend)
            if not is_dask_array(level) and not hasattr(level, '_with_transform'):
                from dyna_zarr import DynamicArray
                level = DynamicArray(level)
            return level
        level_factors, targets = _plan_level_factors(plan, full, base_da.shape, dmethod)
        level_arrays = [base_da]
        for factor, tgt in zip(level_factors[1:], targets[1:]):
            level = method(base_da, scale_factor=factor)
            if tuple(level.shape) != tuple(tgt):
                level = crop_to_level(level, tgt)     # a floor-sized level being reproduced
            level_arrays.append(level)

        def _build(arrays):
            """The full pyramid over `arrays`, carrying the source's metadata."""
            return _full_pyramid(pyramid, full, arrays, _rechunked, dmethod)

        level_kwargs = dict(write_kwargs)
        if level_kwargs.get('compressor') is None and pyramid.compressor is not None:
            # full_dask was rebuilt from arrays, so its compressor fell back to the
            # default: write the coarser levels with the SOURCE's codec, like level 0
            level_kwargs['compressor'] = pyramid.compressor

        # CASCADE: level i from the STORED level i-1, not from L0, for the
        # methods that compose exactly: stride (L0[::a][::b] == L0[::a*b]) and min / max
        # (the min of window minima is the min of the union window, partial edge windows
        # included). The pixels are identical, while L0 is read once instead of once per
        # level (1.2 GB / 4 levels: 3.9 -> 2.1 GB read). Not for mean / median / mode,
        # which do not compose (rounding twice, edge windows reweighted, median of
        # medians), nor with max_downscale_processes (a cascade is sequential by nature).
        cascade = (dmethod in ('simple', 'min', 'max')
                   and not level_kwargs.get('max_downscale_processes'))
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
                stepped = method(parent, scale_factor=rel)
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
            with `label_name`, that group is created/updated with the name. When that
            group, or the image holding it, already exists, the label is written in
            that store's NGFF version (`Pyramid.to_version`), whatever its own: a 0.5
            label from ``from_label_arrays()`` goes into a 0.4 image as 0.4 (zarr v2).
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

    # ------------------------------------------------------------------------------
    # REMOVED ENGINES, kept for reference. write_pyramid has not called the code below
    # since 0.1.0 (backend='tensorstore': write_pyramid_async / write_pyramid_tensorstore;
    # backend='sync': _write_all_layers_unified). On a 1 GB copy the default engine took
    # 1.14 s and +0.4 GB; 'tensorstore' 2.08 s and +1.1 GB; 'sync' 3.4 s and +0.34 GB.
    # Neither wrote shards or honoured a memory budget. Not tested any more.
    # ------------------------------------------------------------------------------

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
        Removed from `write_pyramid` in 0.1.0; kept for reference (see above).

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
        """Synchronous wrapper around :func:`write_pyramid_async` (removed engine,
        kept for reference)."""
        asyncio.run(self.write_pyramid_async(pyramid, path, **kwargs))

    def _write_all_layers_unified(self,
                                   pyramid: Pyramid,
                                   zarr_group: zarr.Group,
                                   layers_to_write: List[str],
                                   dest_path: Union[str, Path],
                                   max_threads: int = 4,
                                   region_size_mb: float = 8.0,
                                   gc_interval: float = 15.0,
                                   compressor_config=None) -> None:
        """Write all layers concurrently using a unified queue-based pipeline (the
        removed 'sync' engine, kept for reference; it needs the pyramid already
        rechunked to the storage chunks and an open zarr group).

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
        queue: Queue = Queue(maxsize=max_threads * 4)
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
                            for _ in range(max_threads):
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
            for i in range(max_threads)
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
    """Convenience wrapper for high-performance pyramid I/O operations.

    ``ome_zarr_pyramid.io`` is one shared instance. Keep this class STATELESS: options
    are call arguments. Anything stored on the shared ``io`` would apply to every thread
    and library in the process, and spawned workers would not see it (they import a
    fresh one). A future per-user default belongs in the constructor of the user's own
    ``IO(...)``, on the instance, never on the class."""

    def read_pyramid(self, path: Union[str, Path], include_labels: bool = True,
                     storage_options: Optional[dict] = None,
                     apply_scene: Optional[bool] = False) -> Pyramid:
        """Read a pyramid from a path or object-store URL. `include_labels` also loads
        the image's NGFF ``labels/`` collection into `Pyramid.labels` (lazy);
        `apply_scene=True` adds the position an NGFF 0.6 scene gives the image to its
        levels. See :meth:`PyramidIO.read_pyramid`.

        `path` may also be an open ``zarr.Group`` (what ``Pyramid().from_ngff`` takes);
        its labels are not looked up and the other options do not apply."""
        if isinstance(path, zarr.Group):
            return Pyramid().from_ngff(path)
        return PyramidIO().read_pyramid(path, include_labels=include_labels,
                                        storage_options=storage_options,
                                        apply_scene=apply_scene)

    def update_metadata(self, path: Union[str, Path], **kwargs):
        """Edit the metadata of what is stored at `path` in place (scales, units, channels,
        windows, OME-XML); the pixels are never rewritten. See
        :meth:`PyramidIO.update_metadata` for the options.

        An image: returns the updated `Pyramid`. A plate, a well or a bioformats2raw
        layout: every image gets the same edit, and the collection read again is returned
        (``ome_xml`` is refused there: it describes one image, or a whole series layout,
        which ``write_series(ome_xml=...)`` writes)."""
        from ome_zarr_pyramid.core.collection import PyramidCollection

        storage_options = kwargs.get("storage_options")
        coll = PyramidCollection._from_zarr(path, storage_options=storage_options)
        if list(coll) == [()]:
            return PyramidIO().update_metadata(path, **kwargs)
        if kwargs.get("ome_xml") is not None:
            raise ValueError(f"update_metadata: {path} holds {len(coll)} images; ome_xml "
                             f"describes one image (or a series layout: write_series(ome_xml=...))")
        locations = []
        for key in coll:
            node = coll._pyramids[key]
            loader = getattr(node, "loader", None)
            path_of = getattr(loader, "path_of", None)
            if path_of is None or getattr(node, "path", None) is None:
                raise ValueError(f"update_metadata: cannot tell where image {dict(key)} of "
                                 f"{path} is stored (an RFC 8 collection?); edit its images "
                                 f"one by one with io.update_metadata(image_path, ...)")
            locations.append(path_of(node))
        for location in locations:
            PyramidIO().update_metadata(location, **kwargs)
        return PyramidCollection._from_zarr(path, storage_options=storage_options)

    def read_labels(self, path: Union[str, Path], name: Optional[str] = None,
                    storage_options: Optional[dict] = None) -> Pyramid:
        """Read a single OME-Zarr label image as a `Pyramid` (with ``image-label``
        metadata). See :meth:`PyramidIO.read_labels`."""
        return PyramidIO().read_labels(path, name=name, storage_options=storage_options)

    def write_pyramid(self,
                      pyramid: Pyramid,
                      path: Union[str, Path],
                      layers: Optional[List[str]] = None,
                      overwrite: bool = False,
                      max_threads: Optional[int] = None,
                      region_size_mb: float = 8.0,
                      max_downscale_processes: Optional[int] = None,
                      backend: Optional[str] = None,
                      **kwargs) -> str:
        """Write a pyramid to a path or object-store URL and return `path`. The options are
        dyna-zarr's words: ``backend`` the storage backend (``'tensorstore'`` or
        ``'zarrista'``; ``io_backend`` is its old name), ``engine`` / ``num_readers`` /
        ``gc_interval`` dyna's loop, ``chunks`` the chunk shape, ``compressor`` a name,
        ``CompressorConfig`` or dyna ``Codecs``. Extra ``**kwargs`` (``chunk_shape``,
        ``shard_coefficients``, ...) are forwarded. See :meth:`PyramidIO.write_pyramid`.
        """
        PyramidIO().write_pyramid(
            pyramid=pyramid,
            path=path,
            layers=layers,
            overwrite=overwrite,
            max_threads=max_threads,
            region_size_mb=region_size_mb,
            max_downscale_processes=max_downscale_processes,
            backend=backend,
            **kwargs,
        )
        return str(path)

    def write_labels(self,
                     label_pyramid: Pyramid,
                     path: Union[str, Path],
                     image_label: Optional[dict] = None,
                     labels_group_path: Optional[Union[str, Path]] = None,
                     label_name: Optional[str] = None,
                     overwrite: bool = False,
                     **kwargs) -> str:
        """Write a label pyramid as an OME-Zarr label image (image-label metadata
        + optional registration in a parent `labels` group) and return `path`. The usual
        route is ``pyr = pyr.add_image_label(label_pyr, name)`` then ``pyr.write(path)``, which writes
        the label under its image. See
        :meth:`PyramidIO.write_labels`."""
        PyramidIO().write_labels(
            label_pyramid=label_pyramid,
            path=path,
            image_label=image_label,
            labels_group_path=labels_group_path,
            label_name=label_name,
            overwrite=overwrite,
            **kwargs,
        )
        return str(path)

    def write(self, obj, path, **options):
        """Write a `Pyramid` or a `PyramidCollection` to `path`, as dyna-zarr's ``io.write``
        writes an array: a pyramid with :meth:`write_pyramid` (returns `path`), a
        collection with ``PyramidCollection.write`` (the structure follows the collection,
        ``version`` picks the format: 0.4 / 0.5 / 0.6 or ``"rfc8"``). Returns where to read
        it back, so ``io.read(io.write(obj, path))`` is what was written.
        ``pyr.write`` and ``coll.write`` are the shortcuts."""
        from ome_zarr_pyramid.core.collection import PyramidCollection
        if isinstance(obj, PyramidCollection):
            return obj._write(path, **options)
        if isinstance(obj, Pyramid):
            return self.write_pyramid(obj, path, **options)
        raise TypeError(f"io.write: a Pyramid or a PyramidCollection, not {type(obj).__name__}")

    def read(self, path, *, storage_options: Optional[dict] = None, include_labels: bool = True,
             apply_scene: Optional[bool] = False, add_missing_translation: bool = False):
        """What is stored at `path`, as :meth:`write` takes it back: a `Pyramid` for an
        image, a `PyramidCollection` for a plate, a well, a bioformats2raw layout or an RFC 8
        collection (also when it holds one image). `include_labels` and `apply_scene` are
        :meth:`read_pyramid`'s, for an image; `add_missing_translation` is
        :meth:`read_collection`'s, for a collection. :meth:`read_pyramid` and
        :meth:`read_collection` give one type whatever the store holds."""
        if isinstance(path, zarr.Group):
            return Pyramid().from_ngff(path)
        if isinstance(path, dict):
            raise TypeError("io.read reads one store; this looks like what "
                            "coll.write(directory, stem) returns, {key: path} of one store "
                            "per image: read each path")
        try:
            coll = self.read_collection(path, storage_options=storage_options,
                                        add_missing_translation=add_missing_translation)
        except zarr.errors.GroupNotFoundError as exc:
            raise FileNotFoundError(f"io.read: {path} is no OME-Zarr store (no zarr group at "
                                    f"its root); a directory of stores is read one store at "
                                    f"a time") from exc
        if list(coll) != [()]:
            return coll
        return self.read_pyramid(path, include_labels=include_labels,
                                 storage_options=storage_options, apply_scene=apply_scene)

    # -- collections (the class methods of PyramidCollection are shortcuts to these) ----
    def read_collection(self, path, *, storage_options: Optional[dict] = None,
                        add_missing_translation: bool = False):
        """The `PyramidCollection` stored at `path`: an NGFF 0.4 / 0.5 / 0.6 plate (keyed
        ``plate``, ``acquisition``, ``well``, ``field``, as micro-reader keys plate
        images), a well on its own, a bioformats2raw layout (``series``), an RFC 8
        collection (also a standalone ``.json`` document), or a single image (the one key
        ``()``). The plate and well metadata are read now; each image when it is first
        used. `storage_options` as for :meth:`read_pyramid`. A store declaring an NGFF
        version ozp does not read raises ``UnsupportedNGFFVersion``; an image placed in a
        coordinate system without a level-0 translation raises ``InvariantError`` when it
        opens, unless ``add_missing_translation=True`` places it at the system's origin.
        ``PyramidCollection.from_zarr`` is the shortcut."""
        from ome_zarr_pyramid.core.collection import PyramidCollection
        return PyramidCollection._from_zarr(path, storage_options=storage_options,
                                            add_missing_translation=add_missing_translation)

    def write_collection(self, collection, path, **options) -> str:
        """Write `collection` as NGFF RFC 8: each image as an RFC 8 multiscale under
        `path`, then the collection document. Options: ``document`` (a standalone
        ``.json``), ``overwrite``, ``storage_options``, ``writer`` and the options of
        :meth:`write_pyramid`. Returns `path` (the standalone `document`, when
        given): ``io.read`` reads it back.
        ``PyramidCollection.write_collection`` is the shortcut."""
        return collection._write_collection(path, **options)

    def write_plate(self, collection, path, **options) -> str:
        """Write a plate of `collection` as an NGFF plate (``version`` 0.4 / 0.5 / 0.6, or
        ``"rfc8"``): each image, then the plate and well metadata. Options: ``plate`` (its
        number, default 0), ``version``, ``overwrite``, ``storage_options``, ``writer``,
        ``document``, ``validate`` and the options of :meth:`write_pyramid`. Returns
        `path` (`document`, when given). ``PyramidCollection.write_plate`` is the shortcut."""
        return collection._write_plate(path, **options)

    def write_series(self, collection, path, **options) -> str:
        """Write every image of `collection` as one series of a bioformats2raw layout,
        then the root metadata and the OME-XML. Options: ``version``, ``overwrite``,
        ``ome_xml``, ``storage_options``, ``writer``, ``document``, ``validate`` and the
        options of :meth:`write_pyramid`. Returns `path` (the standalone `document`, when
        given): ``io.read`` reads it back.
        ``PyramidCollection.write_series`` is the shortcut."""
        return collection._write_series(path, **options)
