"""Pyramids joined along one axis (``concatenate``): lazy, every level, metadata kept.

The pyramids must describe one image: everything but their size along the joined axis
has to agree, and anything that does not agree is an error naming it, never one input's
value taken silently. Joining along ``c`` concatenates the channels as well (labels,
colours, windows, each as it was); joining along another axis needs the same channels.
"""
from __future__ import annotations

import copy
import math
from typing import Any, List, Sequence

from ome_zarr_pyramid.core.spatial import SYSTEM_KEY
from ome_zarr_pyramid.utils.optional_deps import is_dask_array

#: how far two scales or positions may differ, relative to their size, and still agree
REL_TOL = 1e-6


def _close(a: Sequence[float], b: Sequence[float], skip: int = -1) -> bool:
    return all(math.isclose(float(x), float(y), rel_tol=REL_TOL, abs_tol=1e-12)
               for i, (x, y) in enumerate(zip(a, b)) if i != skip)


def _channels(pyr, n: int) -> List[dict]:
    """*pyr*'s omero channels, or its defaults when it has none for its *n* channels."""
    from ome_zarr_pyramid.core.pyramid import generate_channel_metadata
    channels = ((pyr.meta.metadata or {}).get("omero") or {}).get("channels") or []
    if len(channels) == n:
        return copy.deepcopy(channels)
    dtype = pyr.layers[pyr.meta.resolution_paths[0]].dtype
    return generate_channel_metadata(n, dtype)["omero"]["channels"]


def _same_channels(a: List[dict], b: List[dict]) -> bool:
    return [(c.get("label"), c.get("color")) for c in a] == \
           [(c.get("label"), c.get("color")) for c in b]


def concatenate(pyramids: Sequence[Any], axis: str = "c"):
    """One ``Pyramid`` joining *pyramids* along *axis* (``"c"`` by default), level by
    level, lazily (dyna-zarr, or dask when an input is dask-backed).

    The pyramids must agree in everything but their size along *axis*: axes, number of
    levels, each level's shape on the other axes, data type, every level's scale and
    units, and the position on the other axes. Otherwise a ``ValueError`` names what
    differs (joining images of different pixel sizes or places into one would misplace
    them), as do inputs placed in different coordinate systems (or some placed, some not;
    see ``ome_zarr_pyramid.core.spatial.pyramid_system``). Label images and pyramids with a
    pending ``downscale(defer=True)`` plan are refused: concatenate first, then downscale.

    - Along ``c``: the channels are the inputs' channels in order, each with its label,
      colour and window (an input without channel metadata gets ozp's defaults).
    - Along another axis: the inputs must have the same channels (labels, colours); the
      position along *axis* is the first input's (where the joined image starts).
    - Everything else (name, NGFF version, custom metadata blocks, codec, storage
      chunks) comes from the first pyramid.
    """
    from ome_zarr_pyramid.core.pyramid import Pyramid

    pyrs = list(pyramids)
    if not pyrs:
        raise ValueError("concatenate: no pyramids given")
    for n, p in enumerate(pyrs):
        if not isinstance(p, Pyramid) or p.meta is None:
            raise TypeError(f"concatenate: input {n} is not an initialized Pyramid")
    first = pyrs[0]
    axes = first.meta.axis_order
    axis = str(axis).lower()
    if axis not in axes:
        raise ValueError(f"concatenate: axis {axis!r} is not in the pyramids' axes {axes!r}")
    k = axes.index(axis)
    paths0 = first.meta.resolution_paths

    for n, p in enumerate(pyrs):
        where = f"concatenate: input {n}"
        if p.meta.axis_order != axes:
            raise ValueError(f"{where} has axes {p.meta.axis_order!r}, input 0 {axes!r}")
        if p.meta.is_label:
            raise ValueError(f"{where} is a label image; label images are not concatenated")
        if getattr(p, "_downscale_plan", None) is not None:
            raise ValueError(f"{where} has a pending downscale plan: concatenate the "
                             f"pyramids first, then downscale")
        paths = p.meta.resolution_paths
        if len(paths) != len(paths0):
            raise ValueError(f"{where} has {len(paths)} levels, input 0 {len(paths0)}")
        if list(p.meta.unit_list) != list(first.meta.unit_list):
            raise ValueError(f"{where} has units {p.meta.unit_list}, input 0 "
                             f"{first.meta.unit_list}")
        system, system0 = (q.meta.custom_blocks.get(SYSTEM_KEY) for q in (p, first))
        if system != system0:
            raise ValueError(f"{where} is placed in coordinate system {system!r}, input 0 in "
                             f"{system0!r} (None: unplaced): a joined image has one position, "
                             f"in one system")
        for level, (q, q0) in enumerate(zip(paths, paths0)):
            a, a0 = p.layers[q], first.layers[q0]
            if a.dtype != a0.dtype:
                raise ValueError(f"{where} level {level} is {a.dtype}, input 0 {a0.dtype}")
            if any(s != s0 for i, (s, s0) in enumerate(zip(a.shape, a0.shape)) if i != k):
                raise ValueError(f"{where} level {level} has shape {tuple(a.shape)}, input 0 "
                                 f"{tuple(a0.shape)}: only {axis!r} may differ")
            scale, scale0 = p.meta.get_scale(q), first.meta.get_scale(q0)
            if not _close(scale, scale0):
                raise ValueError(f"{where} level {level} has scale {scale}, input 0 {scale0}")
            tr, tr0 = p.meta.get_translation(q), first.meta.get_translation(q0)
            tr = tr or [0.0] * len(axes)
            tr0 = tr0 or [0.0] * len(axes)
            if not _close(tr, tr0, skip=k):
                raise ValueError(f"{where} level {level} is at {tr}, input 0 at {tr0}: "
                                 f"only the position along {axis!r} may differ")

    sizes = [p.layers[p.meta.resolution_paths[0]].shape[k] for p in pyrs]
    if axis == "c":
        channels = [ch for p, n in zip(pyrs, sizes) for ch in _channels(p, n)]
    else:
        channels = _channels(first, first.layers[paths0[0]].shape[axes.index("c")]
                             if "c" in axes else 1)
        for n, (p, size) in enumerate(zip(pyrs, sizes)):
            mine = _channels(p, p.layers[p.meta.resolution_paths[0]].shape[axes.index("c")]
                             if "c" in axes else 1)
            if not _same_channels(mine, channels):
                raise ValueError(f"concatenate: input {n} has other channels than input 0; "
                                 f"along {axis!r} the channels must be the same")

    # the levels: dask when any input is dask-backed, else dyna-zarr (as ozp's operators)
    use_dask = any(p._has_dask_layer() for p in pyrs)
    levels = [p.dask_arrays if use_dask else p._arrays() for p in pyrs]
    joined = []
    for level, q0 in enumerate(paths0):
        parts = [lv[p.meta.resolution_paths[level]] for lv, p in zip(levels, pyrs)]
        if use_dask:
            import dask.array as da
            joined.append(da.concatenate(parts, axis=k))
        else:
            from dyna_zarr import operations
            joined.append(operations.concatenate(parts, axis=k))

    new = Pyramid().from_arrays(
        joined, axis_order=axes, unit_list=first.meta.unit_list,
        scales=[first.meta.get_scale(q) for q in paths0], version=first.meta.version,
        name=first.meta.tag)
    # the first input's metadata, with the joined datasets and channels
    md = copy.deepcopy(first.meta.metadata)
    datasets = new.meta.metadata["multiscales"][0]["datasets"]
    for ds, ds0 in zip(datasets, md["multiscales"][0]["datasets"]):
        ds["coordinateTransformations"] = copy.deepcopy(ds0["coordinateTransformations"])
    md["multiscales"][0]["datasets"] = datasets
    md.setdefault("omero", {})["channels"] = channels
    new.meta.metadata = md
    new.meta.mark_changed()
    new._compressor = getattr(first, "_compressor", None)
    for attr in ("_storage_chunks", "_level_chunks"):
        if getattr(first, attr, None) is not None:
            setattr(new, attr, getattr(first, attr))
    return new
