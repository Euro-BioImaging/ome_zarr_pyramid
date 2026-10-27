"""Pyramids from microscopy images, and collections of them.

Two levels, mirroring a microscopy reader (micro-reader's ``File`` / ``Image``):

* :meth:`Pyramid.from_image` (``pyramid_from_image`` here): ONE image -> a ``Pyramid``. Its
  axes, pixel sizes and units, stage position, channels and stored resolution levels
  become the NGFF multiscale; every level is a lazy dyna-zarr array over the image.
* :meth:`PyramidCollection.from_file`: every image of a file -> a ``PyramidCollection``,
  keyed by the image's index in the file (series, tile, view, illumination, ...).

Nothing here imports micro-reader. An "image" is anything shaped like one:

========================  =============================================================
``axes``                  str, one letter per axis, a subset of ``tczyx`` (``s``, RGB
                          samples, is refused: open with ``samples="channels"``)
``shape``, ``dtype``      as an array
``__getitem__``           a region read returning a NumPy array (what dyna-zarr's
                          ``from_array`` wraps), plus optional ``read_unit``
``array``                 optional: the image's own lazy array (a dyna-zarr
                          ``DynamicArray``, a dask array, a zarr array), used as it
                          is instead of region reads through ``__getitem__``: a dask
                          graph is not recomputed once per region, and dask stays
                          dask (written by dask's own scheduler)
``resolutions``           optional: the stored levels, finest first (the image itself
                          first); each level is image-like
``metadata``              optional, with any of: ``pixel_sizes`` {axis: quantity},
                          ``position`` {axis: quantity}, ``channels`` (objects with
                          ``name``, ``color``, ``emission_wavelength``,
                          ``excitation_wavelength``), ``index_values`` {name: quantity}
``index``                 optional {name: int}: where the image sits in its file
``name``                  optional str
========================  =============================================================

A *quantity* has ``value`` and ``unit`` (an NGFF / UDUNITS unit name, e.g.
``"micrometer"``) and ``to(unit)``. A Bio-Formats source presenting these attributes takes
the same path.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from ome_zarr_pyramid.core.validation import validating

#: the axes an NGFF multiscale may have here
_AXES = "tczyx"
#: an image index entry that is a resolution LEVEL, not a separate image
_RESOLUTION = "resolution"
#: the index that numbers a file's series (bioformats2raw's series; micro-reader 0.1.0
#: names it so). New code names it through this constant only.
SERIES_INDEX = "series"
#: the indices that say where an image sits (its container); any other index (tile,
#: view, illumination, phase, part, ...) tells apart images in one container
CONTAINERS = ("plate", "acquisition", "well", "field", SERIES_INDEX)
#: micro-reader's image index names (``micro_reader._base.INDEX_NAMES``; a test keeps the
#: two the same). ``select`` knows these and any other name the collection's keys carry.
INDEX_NAMES = ("plate", "acquisition", "well", "field", SERIES_INDEX, "tile", "rotation",
               "view", "illumination", "phase", "block", "emission", "excitation",
               "lifetime", "sequence", "part", "resolution")


#: ``select``'s other address of a plate's wells (0-based numbers or names); not index
#: names, so not in the keys
WELL_ADDRESS = ("row", "column")
#: a value of one of these types is "any of" its items, in ``select``
_SEQUENCES = (list, tuple, range, set, frozenset)
#: the navigation classes (ozp's and micro-reader's): not index values
_NAVIGATION = ("Plate", "Well", "Field", "Acquisition")


def check_index_names(index, keys_have, where: str, extra=()) -> None:
    """``select``'s rule for index names, as micro-reader's ``select``: a name is known when
    it is one of micro-reader's :data:`INDEX_NAMES`, one the keys carry (`keys_have`; an
    index name ozp keeps as read, eubi-bridge 0.2.x's say) or one of `extra`; an unknown
    one raises ``KeyError``. A known name no image has selects nothing."""
    unknown = [name for name in index
               if name not in INDEX_NAMES and name not in keys_have and name not in extra]
    if not unknown:
        return
    if "scene" in unknown:
        # micro-reader 0.1.0 renamed the index; this goes with micro-reader's guard (3.24)
        raise KeyError(f"{where}: the image index 'scene' is now called {SERIES_INDEX!r} "
                       f"(micro-reader 0.1.0); use {SERIES_INDEX!r}")
    raise KeyError(f"{where}: unknown indices {sorted(unknown)}; the indices are "
                   f"{sorted(set(INDEX_NAMES) | set(keys_have))}")


def index_values(name: str, value, where: str = "select") -> tuple:
    """``select``'s values for `name`, as micro-reader's: one value, or a list, tuple,
    range or set of them (any of). A plate, well, field or acquisition object is refused:
    it is no index value."""
    values = tuple(value) if isinstance(value, _SEQUENCES) else (value,)
    for v in values:
        kind = type(v)
        if kind.__name__ in _NAVIGATION and \
                kind.__module__.split(".")[0] in ("ome_zarr_pyramid", "micro_reader"):
            raise TypeError(
                f"{where}({name}=...): a {kind.__name__} object is no index value; give its "
                f"index (well=plate.well_index('C3')) or its name (well='C3'), or take the "
                f"part itself with .as_collection()")
    return values


def _on_axis(values, number: int, labels) -> bool:
    """Whether a well's row (or column) `number` is among `values`: 0-based numbers, or
    names (`labels`: the plate's row or column names); None takes every one."""
    if values is None:
        return True
    name = labels[number] if 0 <= number < len(labels) else None
    return any((v == name) if isinstance(v, str) else (v == number) for v in values)


def _io():
    """The IO a shortcut delegates to (imported here: io.py imports this module's
    neighbours at load time). IO holds no state, so a fresh one is the shared ``io``."""
    from ome_zarr_pyramid.core.io import IO
    return IO()


def _written(path, document=None) -> str:
    """What a collection writer returns: where to read the collection back with
    ``io.read`` (the standalone RFC 8 document when there is one, else the store)."""
    return str(document) if document is not None else str(path)


def _pruned(tree, key_of: Dict[int, Any], *, keep_wells: bool = True, copies=None):
    """A copy of `tree` holding only the images `key_of` names (``id`` of a node, or of
    the pyramid it holds -> its key; the copies are added to it, for the writers). Kept:
    every collection that still holds an image, an alias whose image is kept, and a label
    or derived image whose sources are all kept. With `keep_wells` a kept plate keeps
    every well, those left without images empty, so "well": w still reads back as
    wells[w] (the plate's own numbers win on reading, plan 5.11); ``drop_empty`` drops
    them. A plate keeps its rows and columns. `copies`, when given, is filled with ``id``
    of each node -> its copy. Nothing is opened."""
    from ome_zarr_pyramid.core.nodes import (
        CollectionNode, LabelsAttribute, MultiscaleNode, Ref, is_plate, is_well)

    out = tree.copy()
    for original, copied in zip(tree.walk(), out.walk()):
        if copies is not None:
            copies[id(original)] = copied
        key = key_of.get(id(original), key_of.get(id(getattr(original, "_pyramid", None))))
        if key is not None:
            key_of[id(copied)] = key

    def held(node) -> bool:
        first = getattr(node, "_alias_of", None)
        return id(node) in key_of or (first is not None and (
            id(first) in key_of or id(getattr(first, "_pyramid", None)) in key_of))

    def derived(node) -> bool:
        return "labels" in node.attributes or "source" in node.attributes

    def prune(coll) -> None:
        kept = []
        for child in coll.children:
            if isinstance(child, CollectionNode):
                prune(child)
                if child.children or (keep_wells and is_plate(coll) and is_well(child)):
                    kept.append(child)
            elif not isinstance(child, MultiscaleNode) or held(child) or derived(child):
                kept.append(child)
        coll._nodes = kept

    def refs(value) -> List[str]:
        if isinstance(value, Ref):
            return [value.id]
        if isinstance(value, LabelsAttribute):
            return [r.id for r in value.source]
        if isinstance(value, (list, tuple)):
            return [i for v in value for i in refs(v)]
        return []

    def drop_orphans(coll, ids) -> None:
        kept = []
        for child in coll.children:
            if isinstance(child, CollectionNode):
                drop_orphans(child, ids)
                if child.children or (keep_wells and is_plate(coll) and is_well(child)):
                    kept.append(child)
            elif isinstance(child, MultiscaleNode) and derived(child) and not held(child):
                sources = refs(child.attributes.get("labels")) + refs(child.attributes.get("source"))
                if all(i in ids for i in sources):
                    kept.append(child)
            else:
                kept.append(child)
        coll._nodes = kept

    prune(out)
    drop_orphans(out, {n.id for n in out.walk() if isinstance(n, MultiscaleNode) and held(n)})
    return out


def _renamed_graph(graph: dict, rename: Dict[Any, Any]) -> dict:
    """`graph` (``_graph_for``'s) with the image keys renamed by `rename` (old -> new):
    placements and every edge end that is an image key."""
    from dataclasses import replace

    from ome_zarr_pyramid.core.spatial import Transformation, is_image_key

    def end(value):
        return rename.get(value, value) if is_image_key(value) else value

    def edge(e):
        params = e.params
        if e.type == "sequence":
            params = dict(params)
            params["transformations"] = [edge(t) if isinstance(t, Transformation) else t
                                         for t in params.get("transformations") or []]
        return replace(e, input=end(e.input), output=end(e.output), params=params)

    return {"coordinate_systems": graph["coordinate_systems"],
            "system_of": {rename.get(k, k): sid for k, sid in graph["system_of"].items()},
            "coordinate_transformations": [edge(e) for e in graph["coordinate_transformations"]]}


class OperationMismatchError(ValueError):
    """A collection operation that does not fit every image: nothing is done (decision C-5,
    all or nothing). The error names the images and how to narrow the collection first."""


class _MappedLoader:
    """The loader of an image of a mapped collection (C2): opens the source image (its own
    node, so an image opened once is not read twice) and applies the operation."""

    def __init__(self, source, op):
        self.source, self.op = source, op

    def __call__(self, node):
        return self.op(self.source.pyramid)


def _channel_count(row) -> int:
    labels = row.get("channels") or ()
    return len(labels) if labels else int(row.get("size_c") or 1)


def _fits_set_channels(row, args, kwargs):
    """(reason, hint) when ``set_channels(overrides)`` does not fit the image of `row`."""
    overrides = args[0] if args else kwargs.get("overrides") or {}
    labels = tuple(row.get("channels") or ())
    for key in overrides:
        if isinstance(key, str):
            if key not in labels:
                return (f"no channel labelled {key!r} (labels: {list(labels)})",
                        f"coll.filter(channels=lambda labels: {key!r} in labels)")
        elif not 0 <= int(key) < _channel_count(row):
            return (f"{_channel_count(row)} channel(s), no channel {int(key)}",
                    f"coll.filter(size_c=lambda n: (n or 1) > {int(key)})")
    return None


def _fits_set_display_range(row, args, kwargs):
    level = kwargs.get("stats_level")
    if isinstance(level, int) and not isinstance(level, bool) and \
            not 0 <= level < int(row.get("nlevels") or 0):
        return (f"{row.get('nlevels')} level(s), no level {level}",
                f"coll.filter(nlevels=lambda n: n > {level})")
    return None


def _row_set_channels(row, args, kwargs):
    """The table row after ``set_channels(overrides)``: the new labels (an image without
    channels gets ``set_channels``' defaults, ``Channel {i}``, first)."""
    overrides = args[0] if args else kwargs.get("overrides") or {}
    labels = list(row.get("channels") or ()) or [f"Channel {i}" for i in range(_channel_count(row))]
    index = {label: i for i, label in enumerate(labels) if label is not None}
    for key, fields in overrides.items():
        if isinstance(fields, dict) and "label" in fields:
            labels[index[key] if isinstance(key, str) else int(key)] = fields["label"]
    row["channels"] = tuple(labels)
    return row


def _row_dtype(row, dtype):
    """`row` with its pixels in `dtype` (dtype and byte count)."""
    dtype = np.dtype(dtype)
    row["dtype"] = dtype.name
    if row.get("shape"):
        row["nbytes"] = int(np.prod(row["shape"])) * dtype.itemsize
    return row


def _row_rechunk(row, args, kwargs):
    """The row after ``rechunk``: a chunk shape for every level gives level 0's chunks
    (capped at its extent); a size, or one shape per level, is solved by the writer:
    None, the row is read from the image."""
    shape = args[0] if args else kwargs.get("chunk_shape")
    if isinstance(shape, (tuple, list)) and row.get("shape") and len(shape) == len(row["shape"]):
        row["chunks"] = tuple(min(int(c), int(n)) for c, n in zip(shape, row["shape"]))
        return row
    return None


def _row_to_version(row, args, kwargs):
    row["version"] = str(args[0] if args else kwargs.get("version"))
    return row


_COMPARISONS = ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__")
_UNARY = ("__neg__", "__pos__", "__abs__", "__invert__")

#: how an operation changes an image's table row, so a mapped collection's table comes
#: from its source's (metadata only) instead of opening each image; an operation not here
#: (or one giving None) has its rows read from the images it gives: arithmetic with a
#: number, whose dtype follows numpy's promotion
_ROWS = {"set_channels": _row_set_channels,
         "set_display_range": lambda row, args, kwargs: row,      # windows: not in the table
         "set_custom_metadata": lambda row, args, kwargs: row,    # custom blocks: neither
         "rechunk": _row_rechunk, "to_version": _row_to_version,
         "saturate_cast": lambda row, args, kwargs: _row_dtype(row, args[0] if args else kwargs["dtype"]),
         **{name: (lambda row, args, kwargs: _row_dtype(row, bool)) for name in _COMPARISONS},
         **{name: (lambda row, args, kwargs: row) for name in _UNARY}}


def _fits_rechunk(row, args, kwargs):
    shape = args[0] if args else kwargs.get("chunk_shape")
    ndim = len(row.get("axes") or "")
    if isinstance(shape, (tuple, list)) and len(shape) != ndim:
        return (f"{ndim} axes ({row.get('axes')!r}), not {len(shape)}",
                f"coll.filter(axes={row.get('axes')!r})")
    return None

_AXIS_ALIASES = {"channels": "c", "channel": "c", "time": "t"}
#: from this many images to open for their table rows on, they are opened in threads
_PARALLEL_ROWS = 8


def _sizes(row) -> Dict[str, int]:
    return dict(zip(row.get("axes") or "", row.get("shape") or ()))


def _fits_isel(row, args, kwargs):
    """``isel(**indexers)``: every axis there, every integer and list index in range."""
    sizes = _sizes(row)
    for name, index in kwargs.items():
        axis = _AXIS_ALIASES.get(name, name)
        if axis not in sizes:
            return (f"no axis {axis!r} (axes {row.get('axes')!r})",
                    f"coll.filter(axes=lambda axes: {axis!r} in axes)")
        wanted = list(index) if isinstance(index, (list, tuple, range)) else \
            [index] if isinstance(index, (int, np.integer)) else []
        for i in wanted:
            if not -sizes[axis] <= int(i) < sizes[axis]:
                return (f"{sizes[axis]} along {axis}, no index {int(i)}",
                        f"coll.filter(size_{axis}=lambda n: (n or 0) > {abs(int(i))})")
    return None


def _fits_squeeze(row, args, kwargs):
    axes = args[0] if args else kwargs.get("axes")
    sizes = _sizes(row)
    for axis in (axes or ""):
        if axis in sizes and sizes[axis] != 1:
            return (f"{sizes[axis]} along {axis}: squeeze drops size-1 axes only",
                    f"coll.filter(size_{axis}=1)")
    return None


def _levels_asked(args) -> List[int]:
    out = []
    for level in args:
        if isinstance(level, (list, tuple, range)):
            out += [int(v) for v in level]
        elif isinstance(level, (int, np.integer)):
            out.append(int(level))
    return out


def _fits_select_levels(row, args, kwargs):
    """``select_levels(*levels)``: the image has every level asked for (a negative level
    counts from the coarsest)."""
    n = int(row.get("nlevels") or 0)
    asked = _levels_asked(args)
    needed = max([k + 1 for k in asked if k >= 0] + [-k for k in asked if k < 0] + [0])
    if n < needed:
        return (f"{n} level(s); select_levels{tuple(asked)} needs {needed}",
                f"coll.filter(nlevels=lambda n: n >= {needed})")
    return None


def _dropped_isel(row, args, kwargs) -> set:
    """The axes ``isel`` drops: those given an integer."""
    return {_AXIS_ALIASES.get(name, name) for name, index in kwargs.items()
            if isinstance(index, (int, np.integer))}


def _dropped_squeeze(row, args, kwargs) -> set:
    axes = args[0] if args else kwargs.get("axes")
    sizes = _sizes(row)
    names = axes if axes is not None else "".join(a for a in sizes if a not in "yx")
    return {a for a in names if sizes.get(a) == 1 and a not in "yx"}


#: the axes an operation drops from an image (from its table row): the graph rule of C4
_DROPS = {"isel": _dropped_isel, "squeeze": _dropped_squeeze}
#: operations that change an image's geometry (shape, levels, axes): RFC 8 label nodes of
#: the tree do not follow them yet (decision C-2), so they warn
_GEOMETRY = ("isel", "squeeze", "select_levels", "downscale")

#: an operation's fit check, run on the table before anything is done (C3): a function of
#: an image's row and the call's arguments -> None, or (why it does not fit, a filter that
#: narrows the collection to the images it fits)
_FITS = {"set_channels": _fits_set_channels, "set_display_range": _fits_set_display_range,
         "isel": _fits_isel, "squeeze": _fits_squeeze, "select_levels": _fits_select_levels,
         "rechunk": _fits_rechunk}


class VaryingPropertyError(ValueError):
    """A collection's read property (``axes``, ``dtype``, ...) whose value differs between
    its images: there is no one value to return (decision C-8). The error names the
    images that differ; ``coll.table`` gives each image's value."""


def _crowded(keys) -> List[Tuple[List, List[str]]]:
    """The groups of `keys` that share one container (equal containment indices) but are
    several images: ``[(keys, the other index names whose values vary)]``, in key order."""
    groups: Dict[Tuple, List] = {}
    for key in keys:
        place = tuple((n, v) for n, v in key if n in CONTAINERS)
        groups.setdefault(place, []).append(key)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        names = {n for k in members for n, _ in k if n not in CONTAINERS}   # once per group
        values: Dict[str, set] = {name: set() for name in names}
        for key in members:
            index = dict(key)
            for name in names:
                values[name].add(index.get(name))
        out.append((members, sorted(n for n, vals in values.items() if len(vals) > 1)))
    return out


def _quantity(q, unit: Optional[str] = None) -> Optional[Tuple[float, str]]:
    """(value, unit) of a quantity, converted to `unit` when given and convertible."""
    if q is None:
        return None
    if unit is not None and getattr(q, "unit", unit) != unit and hasattr(q, "to"):
        try:
            q = q.to(unit)
        except Exception:                                   # noqa: BLE001
            return None                                     # no conversion: say nothing
    return float(q.value), str(q.unit)


class PositionUnitError(ValueError):
    """An image's stage position along an axis is in a unit ozp cannot convert to that
    axis's unit, so it cannot become the axis's translation."""


def _position(q, unit: Optional[str], *, image: str, axis: str) -> float:
    """A position quantity's value in `unit`, the axis's own unit. Unlike `_quantity`,
    a failed conversion raises: a position that silently became 0 would move the image."""
    if unit is None or getattr(q, "unit", unit) == unit:
        return float(q.value)
    exc = None
    if hasattr(q, "to"):
        try:
            return float(q.to(unit).value)
        except Exception as err:                             # noqa: BLE001
            exc = err
    raise PositionUnitError(
        f"image {image!r}: the position along axis {axis!r} is in {str(q.unit)!r}, which "
        f"cannot be converted to the axis unit {unit!r}. Give the position in a length "
        f"unit convertible to {unit!r}, or clear image.metadata.position.") from exc


def _hex(color) -> Optional[str]:
    if not color:
        return None
    text = str(color).lstrip("#")
    return text.upper() if len(text) in (3, 6) else None


#: which of an image's stored resolution levels ``from_image`` takes
RESOLUTION_POLICIES = ("all", "base")


def pyramid_from_image(image, *, version: str = "0.5", name: Optional[str] = None,
                       acquisition_block: Optional[str] = None, resolutions: str = "all"):
    """The ``Pyramid`` for one image (see the module docstring for what an image is).

    - Levels: the image's stored resolutions (finest first), each a lazy dyna-zarr array;
      ``resolutions="base"`` takes only the full-resolution image (to downscale it
      afresh). A coarser level's scale is the base scale times the size ratio per axis.
    - Scale and units: from ``metadata.pixel_sizes``. A missing z or y size takes x's
      value and unit; any other axis without one keeps ome_zarr_pyramid's default
      (1.0, micrometer / second; none for ``c``).
    - Translation: ``metadata.position`` in each axis's own scale unit, on every level
      (the stored levels are the image's own, so they share level 0's origin), written
      whenever the image has a position, (0, 0) included. A position along an axis in a
      unit that cannot be converted to the axis's unit raises ``PositionUnitError``.
    - Channels: name and colour into omero. A channel the image describes without a
      name is ``Channel {i}`` (counted from 0), without a colour white (OME's default
      colour); colours are generated only when the image says nothing about its
      channels. These are eubi-bridge's defaults, so its outputs stay the same.
    - What NGFF has no field for (channel wavelengths, values along other image indices)
      is ``pyramid.acquisition``. With `acquisition_block` it is also stored, as
      ``{"acquisition": ...}``, in that custom metadata block (see
      ``Pyramid.set_custom_metadata``), so it is written with the pyramid.
    - The image's index in its file is ``pyramid.source_index``.
    """
    from dyna_zarr import from_array

    from ome_zarr_pyramid.core.pyramid import Pyramid
    from ome_zarr_pyramid.utils import defaults

    axes = str(image.axes).lower()
    if "s" in axes:
        raise ValueError(
            f"image axes {axes!r} have RGB samples ('s'); an OME-Zarr pyramid has t/c/z/y/x "
            f"only. Open the file with samples='channels' (micro-reader) so samples become "
            f"channels.")
    if set(axes) - set(_AXES) or len(set(axes)) != len(axes):
        raise ValueError(f"image axes {axes!r}: expected distinct letters from {_AXES!r}")

    if resolutions not in RESOLUTION_POLICIES:
        raise ValueError(f"resolutions={resolutions!r}: one of {RESOLUTION_POLICIES}")
    levels = list(getattr(image, "resolutions", None) or [image])
    if tuple(levels[0].shape) != tuple(image.shape):
        levels = [image] + levels
    if resolutions == "base":
        levels = levels[:1]
    base_shape = tuple(int(n) for n in image.shape)

    meta = getattr(image, "metadata", None)
    pixel_sizes = dict(getattr(meta, "pixel_sizes", None) or {})
    base_scale, units, given = [], [], set()
    for ax in axes:
        # a channel axis has no unit in NGFF (ozp's default table says "Channel")
        default_unit = None if ax == "c" else defaults.unit_map.get(ax)
        q = _quantity(pixel_sizes.get(ax)) if ax != "c" else None
        if q is None:
            base_scale.append(float(defaults.scale_map.get(ax, 1.0)))
            units.append(default_unit)
        else:
            base_scale.append(q[0])
            units.append(q[1])
            given.add(ax)
    if "x" in given:
        # a missing z or y pixel size takes x's (eubi-bridge's rule)
        x = axes.index("x")
        for ax in "zy":
            if ax in axes and ax not in given:
                base_scale[axes.index(ax)], units[axes.index(ax)] = base_scale[x], units[x]
    scales = []
    for level in levels:
        ratio = [b / max(1, int(n)) for b, n in zip(base_shape, level.shape)]
        scales.append([s * (r if ax in "zyx" else 1.0) for s, r, ax in zip(base_scale, ratio, axes)])

    # a level's own lazy array when it offers one (an eubi-bridge LegacyImage over a
    # dask or Bio-Formats reader), else its region reads wrapped by dyna-zarr
    arrays = [getattr(level, "array", None) if getattr(level, "array", None) is not None
              else from_array(level) for level in levels]
    label = name or getattr(image, "name", None) or "unnamed"
    pyr = Pyramid().from_arrays(arrays, axis_order=axes, unit_list=units, scales=scales,
                                version=version, name=label)

    position = dict(getattr(meta, "position", None) or {})
    if position:
        # written whenever the image has a position, (0, 0) included; an axis the
        # position does not name, and t / c, stay 0. Stored levels are the image's own
        # (a vendor pyramid): they keep level 0's translation (alignment 0).
        unit_of = dict(zip(axes, units))
        translation = [_position(position[ax], unit_of.get(ax), image=label, axis=ax)
                       if ax in "zyx" and position.get(ax) is not None else 0.0
                       for ax in axes]
        for path in pyr.meta.resolution_paths:
            pyr.meta.update_translation(path, translation)

    channels = list(getattr(meta, "channels", None) or [])
    n_channels = base_shape[axes.index("c")] if "c" in axes else 1
    if channels and len(channels) == n_channels:
        from ome_zarr_pyramid.core.pyramid import auto_channel_color
        given = [_hex(getattr(ch, "color", None)) for ch in channels]
        taken, k = {c for c in given if c}, 0
        overrides = {}
        for i, ch in enumerate(channels):
            # unnamed: the default label "Channel {i}" stays. No colour: one channel
            # white (shown as grey); of several, the next of ozp's distinct colours
            # (red, green, blue, ...) no other channel has, so they stay apart (C13)
            color = given[i]
            if color is None and n_channels == 1:
                color = "FFFFFF"
            elif color is None:
                while auto_channel_color(k) in taken:
                    k += 1
                color = auto_channel_color(k)
                taken.add(color)
                k += 1
            fields = {"color": color}
            if getattr(ch, "name", None):
                fields["label"] = str(ch.name)
            overrides[i] = fields
        pyr = pyr.set_channels(overrides)

    acquisition: Dict[str, Any] = {}
    waves = []
    for ch in channels:
        entry = {}
        for key in ("emission_wavelength", "excitation_wavelength"):
            q = _quantity(getattr(ch, key, None), "nanometer")
            if q is not None:
                entry[key] = {"value": q[0], "unit": q[1]}
        waves.append(entry)
    if any(waves):
        acquisition["channel_wavelengths"] = waves
    index_values = dict(getattr(meta, "index_values", None) or {})
    if index_values:
        acquisition["index_values"] = {k: {"value": float(q.value), "unit": str(q.unit)}
                                       for k, q in index_values.items()}
    if acquisition_block and acquisition:
        block = dict(pyr.custom_metadata.get(acquisition_block) or {})
        block["acquisition"] = acquisition
        pyr.meta.set_custom(acquisition_block, block)
    pyr.acquisition = acquisition
    pyr.source_index = dict(getattr(image, "index", None) or {})
    return pyr


@dataclass
class Component:
    """One connected component of a collection's graph (plan 3.6, see
    :meth:`PyramidCollection.components`): its shared system ids, its entries (in the
    collection's order), its edges other than positions, and its owner (the path of
    names from the root to the collection that writes it; None for a single entry)."""
    systems: set
    entries: list
    edges: list
    owner: Optional[Tuple[str, ...]]


def _no_translation(where, key, sid) -> str:
    return (f"{where}I4: entry {dict(key)} is placed in {sid!r} but has no level-0 "
            f"translation, so its position is unknown. Pass add_missing_translation=True "
            f"(to PyramidCollection(...) or PyramidCollection.from_zarr) to place it at the "
            f"origin of {sid!r}, or remove its coordinate system")


def _with_zero_translation(pyr):
    """A metadata copy of `pyr` with translation 0 on every level (decision O-2)."""
    out = pyr._clone_metadata_only()
    for attr in ("acquisition", "source_index", "joined_index", "source_series"):
        if hasattr(pyr, attr):
            setattr(out, attr, getattr(pyr, attr))
    n = len(out.meta.axis_order)
    for path in out.meta.resolution_paths:
        if out.meta.get_translation(path) is None:
            out.meta.update_translation(path, [0.0] * n)
    return out


def _translation_on_load(node, key, sid, add: bool, where: str) -> None:
    """Check, when a placed image of a store opens, that it has a level-0 translation (I4):
    add zeros with `add`, else raise naming the argument (decision O-2)."""
    from ome_zarr_pyramid.core.spatial import InvariantError
    loader = node.loader

    def load(n, loader=loader):
        pyr = loader(n)
        if pyr.meta.get_translation(pyr.meta.resolution_paths[0]) is None:
            if not add:
                raise InvariantError(_no_translation(f"{where}: ", key, sid))
            n_axes = len(pyr.meta.axis_order)
            for path in pyr.meta.resolution_paths:
                if pyr.meta.get_translation(path) is None:
                    pyr.meta.update_translation(path, [0.0] * n_axes)
        return pyr
    for attr in ("location", "base", "path_of"):
        if hasattr(loader, attr):
            setattr(load, attr, getattr(loader, attr))
    node.loader = load


def _read_scenes(groups, key_of_path, out, systems, system_of, where):
    """The 0.6 scenes on `groups` taken into the graph read so far (``ngff06.
    graph_from_scenes``): ``(systems, system_of, edges)``. Each placed image's position is
    folded into its levels when it opens; opens no image now."""
    from ome_zarr_pyramid.core import ngff06
    if not any(ngff06.scene_of(attrs) for attrs in groups.values()):
        return systems, system_of, []
    held = {cs.id: cs for cs in systems}
    systems, system_of, positions, edges, names = ngff06.graph_from_scenes(
        groups, key_of_path, held, system_of, where)
    for key, p in positions.items():
        node = out.get(key)
        if node is not None and getattr(node, "loader", None) is not None:
            ngff06.fold_on_load(node, p, names.get(key), where)
    return systems, system_of, edges


def _tree_of_keys(tree, entries):
    """A series layout read from a store as the tree of its images' keys (plan 3.18): the
    tree :func:`~ome_zarr_pyramid.core.plates.tree_from_file` builds, whose leaves open
    the store's image groups as the layout's nodes did (nothing is opened here). Returns
    ``(tree, {key: leaf})``; the root keeps the layout's attributes."""
    from ome_zarr_pyramid.core.nodes import MultiscaleNode
    from ome_zarr_pyramid.core.plates import tree_from_file

    shim = type("_Keys", (), {"plates": []})()
    new = tree_from_file(shim, entries, name=tree.name)
    leaf_of = {}
    for leaf in new.walk():
        inner = getattr(leaf, "_pyramid", None)
        if isinstance(leaf, MultiscaleNode) and isinstance(inner, MultiscaleNode):
            leaf._pyramid, leaf.loader, leaf.path = None, inner.loader, inner.path
            leaf_of[id(inner)] = leaf
    new.attributes.update(tree.attributes)
    new.ngff_fields = dict(tree.ngff_fields)
    return new, {key: leaf_of[id(node)] for key, node in entries.items()}


def _end_text(end) -> str:
    return str(dict(end)) if isinstance(end, tuple) else repr(getattr(end, "name", end))


def _path_text(path) -> str:
    return "the root" if not path else "/".join(path)


def _key(index: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    """A hashable collection key: the image index without its resolution level."""
    return tuple((str(k), int(v)) for k, v in index.items() if k != _RESOLUTION)


def _tree_entries(tree) -> Iterator[Tuple[Tuple[Tuple[str, int], ...], Any]]:
    """(key, image node) for every image of a plate or well tree, keyed as micro-reader
    keys plate images: ``(("plate", 0), ("acquisition", a), ("well", w), ("field", f))``.
    ``w`` counts the plate's wells, ``a`` is the acquisition's position in the plate's
    list (0 when an image names none), ``f`` counts the well's images of that
    acquisition. An acquisition attribute on a sub-collection of the well (RFC 8's
    "tall" form) applies to its images. Label and other derived images (a ``labels`` or
    ``source`` attribute) are not fields and are left out. A well read on its own has no
    plate: its keys are ``(("acquisition", id), ("field", f))``."""
    from ome_zarr_pyramid.core.nodes import CollectionNode, MultiscaleNode, is_plate

    def images(well):
        for child in well.nodes:
            if isinstance(child, CollectionNode):
                acq = child.attributes.get("acquisition")
                for image in child.nodes:
                    if isinstance(image, MultiscaleNode):
                        yield image, image.attributes.get("acquisition", acq)
            elif isinstance(child, MultiscaleNode):
                yield child, child.attributes.get("acquisition")

    def fields(well, acquisition_index, prefix):
        counts: Dict[int, int] = {}
        for image, acq in images(well):
            if "labels" in image.attributes or "source" in image.attributes:
                continue
            a = acquisition_index(acq) if acq is not None else 0
            f = counts.get(a, 0)
            counts[a] = f + 1
            yield prefix(a) + (("field", f),), image

    from ome_zarr_pyramid.core.plates import plate_nodes

    plates = plate_nodes(tree)
    if plates:                                 # a plate, or a collection of plates
        for p, node in enumerate(plates):
            plate = node.attributes["plate"]
            for w, well in enumerate(node.nodes):
                yield from fields(well, lambda ref, plate=plate: plate.index_of("acquisitions", ref),
                                  lambda a, p=p, w=w: (("plate", p), ("acquisition", a), ("well", w)))
    else:
        yield from fields(tree, lambda ref: int(ref.id) if ref.id.isdigit() else 0,
                          lambda a: (("acquisition", a),))


def _same(a, b) -> bool:
    """Whether two property values are equal (False when they cannot say: arrays)."""
    try:
        return bool(a == b)
    except Exception:                                       # noqa: BLE001
        return a is b


def _order_key(value):
    """Sorts values of one column: None last, then by type, then by value."""
    return (value is None, type(value).__name__, value if value is not None else 0)


def _sorted_values(values) -> list:
    try:
        return sorted(values, key=_order_key)
    except TypeError:                          # values of one type that do not order
        return sorted(values, key=lambda v: (v is None, repr(v)))


class PyramidCollection(Mapping):
    """Pyramids keyed by where each image sits in its source: ``(("series", 0),
    ("view", 1))``. A read-only mapping (``len``, iteration over keys, ``items()``);
    ``coll[{"series": 0, "view": 1}]`` and ``coll.select(view=1)`` look entries up by
    index; :meth:`write` writes every entry.

    A collection read from OME-Zarr (:meth:`from_zarr`) keeps its node tree as
    :attr:`tree` (see :mod:`ome_zarr_pyramid.core.nodes`), and opens each pyramid on
    first access: listing, selecting and indexing keys open nothing.

    Where the images are (plan 1.3-1.5, 3.3, 3.4): :attr:`coordinate_systems` are the
    shared coordinate systems, by id; :attr:`system_of` names, per entry, the system its
    position is in (the position itself is the entry's level-0 translation, in the image's
    own units); an entry not in it is unplaced. :attr:`coordinate_transformations` holds
    the edges that are not positions (affine edges, registrations, other types as read).

    Built from pyramids in memory, without a graph, the collection lifts the system each
    pyramid keeps (plan 3.26; see :func:`ome_zarr_pyramid.core.spatial.pyramid_system`).
    One id is one system only among pyramids of one source (read from one store, or taken
    from one collection with :meth:`pyramid`): a pyramid of another source whose id is
    taken gets ``<set>--<id>`` (8.16), unless ``join_systems=True`` says they are one
    space. Take an entry out with :meth:`pyramid` to keep where it is.

    A pyramid that names a system but has no level-0 translation raises
    :class:`~ome_zarr_pyramid.core.spatial.InvariantError` (I4): its position is
    unknown. ``add_missing_translation=True`` places it at the system's origin instead
    (translation 0 on every level, on a copy: the pyramid given is not changed;
    decision O-2). :meth:`from_zarr` takes the same argument.
    """

    def __init__(self, pyramids: Optional[Dict[Tuple[Tuple[str, int], ...], Any]] = None,
                 name: Optional[str] = None, tree=None, *,
                 coordinate_systems: Optional[Iterable] = None,
                 system_of: Optional[Dict[Tuple[Tuple[str, int], ...], str]] = None,
                 coordinate_transformations: Optional[Iterable] = None,
                 join_systems: bool = False, add_missing_translation: bool = False):
        from ome_zarr_pyramid.core.spatial import system_from, transformation_from

        # a value is a Pyramid, or a MultiscaleNode opened on first access
        self._pyramids = dict(pyramids or {})
        self.name = name
        #: the node tree the collection was read from (a plate, a well, an image), or None
        self.tree = tree
        # the table's rows built so far, by key (metadata only; see `table`)
        self._rows: Dict[Tuple, Dict[str, Any]] = {}
        # the index names of the collection this one was selected from (see `select`)
        self._names: frozenset = frozenset()
        systems: Dict[str, Any] = {}
        for cs in coordinate_systems or ():
            cs = system_from(cs)
            if cs.id in systems:
                from ome_zarr_pyramid.core.spatial import InvariantError
                raise InvariantError(f"I1: two coordinate systems are called {cs.id!r}")
            systems[cs.id] = cs
        self._systems = systems
        self._system_of = {tuple(k): str(v) for k, v in (system_of or {}).items()}
        self._edges = [transformation_from(t) for t in coordinate_transformations or ()]
        # where the entries came from (3.26): a collection taken from this one shares it
        self._source = ("set", id(self))
        if coordinate_systems is None and system_of is None and coordinate_transformations is None:
            if self._lift(join_systems):
                if add_missing_translation:
                    self._add_missing_translations()
                self.check_invariants()

    def _add_missing_translations(self) -> None:
        """Give each placed, loaded pyramid without a level-0 translation zeros on every
        level, on a metadata copy (the pyramid given is not changed; decision O-2)."""
        from ome_zarr_pyramid.core.pyramid import Pyramid
        for key in self._system_of:
            pyr = self._pyramids.get(key)
            if isinstance(pyr, Pyramid) and pyr.meta is not None and                     pyr.meta.get_translation(pyr.meta.resolution_paths[0]) is None:
                self._pyramids[key] = _with_zero_translation(pyr)

    def _lift(self, join: bool) -> bool:
        """Take the systems the loaded pyramids keep (3.26, 8.16); True when any did. Lazy
        entries (an image node of a store) are not opened."""
        import dataclasses

        from ome_zarr_pyramid.core.pyramid import Pyramid
        from ome_zarr_pyramid.core.spatial import (
            clean_id_part, merge_systems, pyramid_system, source_name, source_token)

        ids: Dict[Tuple, str] = {}            # (source, id in the pyramid) -> id here
        prefixes: Dict[Any, str] = {}         # source -> its "<set>" part
        for key, value in self._pyramids.items():
            if not isinstance(value, Pyramid) or value.meta is None:
                continue
            sid, cs = pyramid_system(value)
            if sid is None:
                continue
            source = None if join else source_token(value)
            here = ids.get((source, sid))
            if here is None:
                here = sid
                if sid in self._systems:      # another source's system has the id
                    if source not in prefixes:
                        named = source_name(value)
                        prefixes[source] = (clean_id_part(named) if named
                                            else f"set{len(prefixes) + 1}")
                    here = f"{prefixes[source]}--{sid}"
                    n = 2
                    while here in self._systems:
                        here = f"{prefixes[source]}{n}--{sid}"
                        n += 1
                ids[(source, sid)] = here
            mine = cs if here == cs.id else dataclasses.replace(cs, id=here)
            self._systems[here] = (merge_systems(self._systems[here], mine)
                                   if here in self._systems else mine)
            self._system_of[key] = here
        return bool(ids)

    # -- where the images are (3.3, 3.4) ------------------------------------------
    @property
    def coordinate_systems(self) -> Dict[str, Any]:
        """The shared coordinate systems, by id (a copy: build a new collection to change
        them)."""
        return dict(self._systems)

    @property
    def system_of(self) -> Dict[Tuple[Tuple[str, int], ...], str]:
        """For each placed entry, the id of the system its position (its level-0
        translation) is in. An entry not listed is unplaced."""
        return dict(self._system_of)

    @property
    def coordinate_transformations(self) -> List[Any]:
        """The graph's edges other than positions (a copy)."""
        return list(self._edges)

    def check_invariants(self, *, images: bool = True) -> None:
        """Raise :class:`~ome_zarr_pyramid.core.spatial.InvariantError` when the graph breaks
        the model's invariants (plan 3.22): I1 (ids pattern-safe and unique), I2 (a
        position's system is a listed, nominal one), I4 (a placed entry has a level-0
        translation), I5 (every edge end is a listed system, an entry of the collection or
        an outside reference) and I8 (a placed image's axes are among its system's, when
        the system's axes are known). Opens every placed pyramid's metadata, unless
        ``images=False`` (I4 and I8 are then left out: a store read lazily)."""
        from ome_zarr_pyramid.core.spatial import (
            ExternalRef, InvariantError, PATTERN_SAFE, is_image_key, steps)
        where = f"{self.name}: " if self.name else ""
        for sid in self._systems:
            if not PATTERN_SAFE.fullmatch(sid):
                raise InvariantError(f"{where}I1: system id {sid!r} is not pattern-safe")
        for key, sid in self._system_of.items():
            if key not in self._pyramids:
                raise InvariantError(f"{where}system_of names {dict(key)}, which is not an "
                                     f"entry of the collection")
            cs = self._systems.get(sid)
            if cs is None:
                raise InvariantError(f"{where}I2: entry {dict(key)} is placed in {sid!r}, "
                                     f"which is not one of the systems {list(self._systems)}")
            if cs.estimated:
                raise InvariantError(f"{where}I2: entry {dict(key)} is placed in {sid!r}, an "
                                     f"estimated system; positions go into nominal ones only")
            if not images:
                continue
            pyr = self._open(self._pyramids[key])
            if pyr.meta.get_translation(pyr.meta.resolution_paths[0]) is None:
                raise InvariantError(_no_translation(where, key, sid))
            if cs.axes is not None:
                names = {a.name for a in cs.axes}
                missing = [a for a in pyr.meta.axis_order if a not in names]
                if missing:
                    raise InvariantError(f"{where}I8: entry {dict(key)} has axes "
                                         f"{missing} that its system {sid!r} does not have")

        def check_end(end, edge):
            if end is None or isinstance(end, ExternalRef):
                return
            if is_image_key(end):
                if end not in self._pyramids:
                    raise InvariantError(f"{where}I5: an edge ({edge.type}) starts or ends at "
                                         f"{dict(end)}, which is not an entry")
            elif end not in self._systems:
                raise InvariantError(f"{where}I5: an edge ({edge.type}) starts or ends at "
                                     f"{end!r}, which is not one of the systems "
                                     f"{list(self._systems)}")

        for edge in self._edges:
            for step in (edge,) + (steps(edge) if edge.type == "sequence" else ()):
                check_end(step.input, edge)
                check_end(step.output, edge)

    def validate(self, *, images: bool = True) -> "PyramidCollection":
        """Check the collection as ``Pyramid.validate`` checks one image, and return it (so
        it chains): the graph's invariants (:meth:`check_invariants`), then every image's
        ``validate()``. ``images=False`` checks the graph only and opens no image (a store
        read lazily). Raises on the first problem found."""
        self.check_invariants(images=images)
        if images:
            for key in self._pyramids:
                self[key].validate()
        return self

    def components(self, *, warn: bool = True) -> List["Component"]:
        """The graph's connected components (plan 3.6), the unit a writer gives one owner.
        Nodes: the shared systems and every entry's own system; edges: the positions
        (:attr:`system_of`) and every edge of :attr:`coordinate_transformations`, in either
        direction (an end outside the set joins nothing). An entry without any edge is in
        no component (unplaced). A component with two or more entries has an owner: the
        smallest collection of :attr:`tree` holding all of them, as the path of names
        from the root (``()``: the root; no tree: ``()``); one with a single entry has
        none (it is written in the single-image form).

        When the edges that are not positions join components that would otherwise have
        different owners, `warn` gives a ``UserWarning`` naming the edges and the new
        owner (a plate-wide registration makes the plate the owner of every well's
        system, so no well is a scene of its own)."""
        from ome_zarr_pyramid.core.spatial import ExternalRef, is_image_key, steps

        parent: Dict[Any, Any] = {}

        def find(node):
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def join(a, b):
            parent[find(a)] = find(b)

        def node_of(end):
            if end is None or isinstance(end, ExternalRef):
                return None
            return ("entry", end) if is_image_key(end) else ("system", end)

        for key, sid in self._system_of.items():
            join(("entry", key), ("system", sid))
        positions = {node: find(node) for node in list(parent)}
        groups = {}
        for node, root in positions.items():
            if node[0] == "entry":
                groups.setdefault(root, []).append(node[1])
        before = {root: self._owner_of(keys) for root, keys in groups.items() if len(keys) > 1}
        joining = []
        for edge in self._edges:
            ends = [node_of(e) for step in (edge,) + (steps(edge) if edge.type == "sequence" else ())
                    for e in (step.input, step.output)]
            ends = [e for e in ends if e is not None]
            for a, b in zip(ends, ends[1:]):
                if find(a) != find(b):
                    joining.append((edge, a, b))
                join(a, b)
        out: Dict[Any, Component] = {}
        for node in list(parent):
            c = out.setdefault(find(node), Component(set(), [], [], None))
            if node[0] == "entry":
                c.entries.append(node[1])
            else:
                c.systems.add(node[1])
        for edge in self._edges:
            end = next((node_of(e) for e in (edge.input, edge.output) if node_of(e) is not None), None)
            if end is not None:
                out[find(end)].edges.append(edge)
        components = []
        for root, c in out.items():
            if not c.entries:
                continue                        # systems linked only to each other
            order = {k: i for i, k in enumerate(self._pyramids)}
            entries = sorted(c.entries, key=lambda k: order.get(k, len(order)))
            owner = self._owner_of(entries) if len(entries) > 1 else None
            components.append(Component(c.systems, entries, c.edges, owner))
            owners = {before[r] for r, keys in groups.items()
                      if r in before and find(("entry", keys[0])) == root}
            if warn and len(owners) > 1:
                import warnings
                named = sorted({f"{e.type} {_end_text(e.input)} -> {_end_text(e.output)}"
                                for e, a, b in joining if find(a) == root})
                warnings.warn(f"{len(named)} edge(s) ({'; '.join(named[:3])}"
                              f"{' ...' if len(named) > 3 else ''}) join systems that "
                              f"{sorted(map(_path_text, owners))} would own; they are "
                              f"written together under {_path_text(owner)}",
                              UserWarning, stacklevel=2)
        return components

    def _owner_of(self, keys) -> Tuple[str, ...]:
        """The path of names from the root to the smallest collection of the tree
        holding every entry of `keys` (``()``: the root, also without a tree)."""
        from ome_zarr_pyramid.core.nodes import CollectionNode
        if self.tree is None or not isinstance(self.tree, CollectionNode):
            return ()
        ids = {}
        for key, value in self._pyramids.items():
            ids[id(value)] = key
        where: Dict[Any, Tuple[str, ...]] = {}

        def walk(node, path):
            for child in node.nodes:
                if isinstance(child, CollectionNode):
                    walk(child, path + (child.name,))
                    continue
                key = ids.get(id(child), ids.get(id(getattr(child, "_pyramid", None))))
                if key is not None:
                    where[key] = path
        walk(self.tree, ())
        paths = [where.get(k, ()) for k in keys]
        common = paths[0]
        for p in paths[1:]:
            n = 0
            while n < min(len(common), len(p)) and common[n] == p[n]:
                n += 1
            common = common[:n]
        return common

    @classmethod
    def from_zarr(cls, path, *, storage_options: Optional[dict] = None,
                  add_missing_translation: bool = False) -> "PyramidCollection":
        """The collection stored at `path`; a shortcut for ``io.read_collection``, where the layouts it reads are described."""
        return _io().read_collection(path, storage_options=storage_options,
                                     add_missing_translation=add_missing_translation)

    @classmethod
    def _from_zarr(cls, path, *, storage_options: Optional[dict] = None,
                   add_missing_translation: bool = False) -> "PyramidCollection":
        """The collection stored at `path`: an NGFF 0.4 / 0.5 / 0.6 plate (keyed ``plate``,
        ``acquisition``, ``well``, ``field``, as micro-reader keys plate images), a well
        on its own (keyed as in its plate when it carries ``eubi:images``, else
        ``acquisition``, ``field``), a bioformats2raw layout (``series``), an RFC 8
        collection, or a single image (the one key ``()``). The plate and well metadata
        are read now; each image when it is first used. `storage_options` as for
        ``io.read_pyramid`` (an object-store URL). A store declaring an NGFF version ozp
        does not read (0.7, say) raises :class:`~ome_zarr_pyramid.UnsupportedNGFFVersion`
        here, from its attributes.

        An image the store places in a coordinate system but that has no level-0
        translation raises :class:`~ome_zarr_pyramid.core.spatial.InvariantError` (I4)
        when it opens; ``add_missing_translation=True`` places it at the system's origin
        instead (decision O-2)."""
        add = add_missing_translation
        from ome_zarr_pyramid.core.io import _open_store_group
        from ome_zarr_pyramid.core.ngff_plate import (
            ImageOpener, _split, _stem, read_plate, read_well)
        from ome_zarr_pyramid.core.nodes import MultiscaleNode, NodePath, check_read_version
        from ome_zarr_pyramid.core.pyramid import NGFFMetadataHandler

        from ome_zarr_pyramid.core import rfc8

        if str(path).lower().endswith(".json"):                # a standalone RFC 8 document
            return cls._from_tree(rfc8.read_rfc8(path, storage_options=storage_options), add)
        from zarr.errors import ContainsArrayError
        try:
            root = _open_store_group(str(path), None, mode="r", storage_options=storage_options)
        except ContainsArrayError:            # an array: an RFC 8 singlescale image
            return cls._from_tree(rfc8.read_rfc8(path, storage_options=storage_options), add)
        attrs = root.attrs.asdict()
        if rfc8.is_rfc8(attrs.get("ome")):
            return cls._from_tree(rfc8.read_rfc8(path, storage_options=storage_options), add)
        version, ngff, _ = _split(attrs)
        if "multiscales" in ngff and not ("plate" in ngff or "well" in ngff):
            version = NGFFMetadataHandler.declared_version(attrs)
        # refused from the root's attributes, before any child group or array is opened
        # (a lazy image would otherwise fail only on first use); a group with neither a
        # version nor NGFF metadata gets the "not an OME-Zarr image" error below
        if version is not None or ngff:
            check_read_version(version, str(path))
        if "plate" in ngff:
            tree = read_plate(path, storage_options=storage_options, _group=root)
        elif "well" in ngff:
            tree = read_well(path, storage_options=storage_options, _group=root)
        elif "multiscales" in ngff:
            tree = MultiscaleNode(_stem(path) or "image", path=NodePath("zarr", "."),
                                  loader=ImageOpener(path, storage_options))
            # a lone image keeps its system (3.26), from the attributes read already
            from ome_zarr_pyramid.core.spatial import SYSTEM_KEY, SYSTEMS_KEY, system_from_json
            custom = _split(attrs)[2]
            sid = custom.get(SYSTEM_KEY)
            systems = [system_from_json(o) for o in custom.get(SYSTEMS_KEY) or []]
            coll = cls({(): tree}, name=tree.name, tree=tree, coordinate_systems=systems,
                       system_of={(): str(sid)} if sid is not None else {},
                       coordinate_transformations=())
            return coll._finish_store_read(path, add)
        elif "bioformats2raw.layout" in ngff:
            from ome_zarr_pyramid.core.ngff_series import read_series
            tree = read_series(path, storage_options=storage_options, _group=root)
            # keyed as micro-reader keys a file's series (bioformats2raw's series), or by
            # their eubi:index (3.18)
            entries = [(((SERIES_INDEX, i),), node) for i, node in enumerate(tree.nodes)]
            where = {id(n): (n.path.path, tree) for _, n in entries}
            return cls._keyed_from_store(entries, where, tree, root, path, series=True, add=add)
        else:
            raise ValueError(f"{path}: not an OME-Zarr image, well or plate (its metadata "
                             f"has no 'multiscales', 'well' or 'plate')")
        entries = list(_tree_entries(tree))
        if "plate" not in ngff:
            # a well on its own: keyed and placed as in its plate, from its eubi:images
            # and the systems on it (decision O-5); its spec keys when it has none
            where = {id(n): (n.path.path, tree) for _, n in entries if n.path is not None}
            return cls._keyed_from_store(entries, where, tree, root, path, series=False,
                                         listing_groups={"": tree.attributes}, add=add)
        where = {}
        for well in tree.nodes:
            for image in well.walk():
                if isinstance(image, MultiscaleNode) and image.path is not None:
                    where[id(image)] = (f"{well.path.path}/{image.path.path}", well)
        return cls._keyed_from_store(entries, where, tree, root, path, series=False, add=add)

    @classmethod
    def _keyed_from_store(cls, entries, where, tree, root, path, *, series: bool,
                          listing_groups: Optional[Mapping[str, Any]] = None, add: bool = False):
        """The collection of a plate or series layout read from a store, with the keys
        and the graph its ``eubi:`` keys hold (plan 3.18, 5.7, 5.11): keys from
        ``eubi:images`` (``eubi:index`` wins for numbers the spec does not state: field,
        series, tile, ...; the spec's plate, acquisition and well win, a disagreement
        warns), ``system_of`` from it, the systems and edges from the groups read. A
        store written with ``eubi:version`` whose listing group lacks ``eubi:images``
        has its image groups read in one batch. A series layout whose every image has an
        index gets the tree of its keys (series flattened on writing get their series
        level back). `listing_groups` names the groups whose ``eubi:`` keys hold the graph
        (default: the root and, for a plate, its wells). Opens no image."""
        import warnings

        from ome_zarr_pyramid.core import eubi_keys as ek
        from ome_zarr_pyramid.core.ngff_plate import _group_attrs, _split

        root_attrs = tree.attributes
        marked = ek.VERSION_KEY in root_attrs
        ek.check_version(root_attrs, str(path))
        found: Dict[int, Optional[dict]] = {}
        missing = []
        for key, node in entries:
            if id(node) not in where:          # an image without a path: its spec key
                continue
            image_path, listing = where[id(node)]
            listed = listing.attributes.get(ek.IMAGES_KEY)
            if isinstance(listed, Mapping):
                found[id(node)] = listed.get(node.path.path)
            elif marked:
                missing.append((node, image_path))
        if missing:                            # one batch of the image groups (3.18)
            for (node, _), attrs in zip(missing, _group_attrs(root, [p for _, p in missing])):
                custom = _split(attrs)[2] if attrs else {}
                if ek.INDEX_KEY in custom:
                    found[id(node)] = {"index": custom[ek.INDEX_KEY],
                                       "coordinate_system": custom.get(ek.SYSTEM_KEY)}
        out, system_of, key_of_path = {}, {}, {}
        disagree = []
        for key, node in entries:
            entry = found.get(id(node))
            if isinstance(entry, Mapping) and isinstance(entry.get("index"), Mapping):
                written = ek.key_from_json(entry["index"])
                if not series:                 # the spec's numbers win (5.11)
                    spec = dict(key)
                    names = [n for n, _ in written]
                    for n in ("well", "acquisition", "plate"):
                        if n in spec and n not in names:
                            names.insert(0, n)
                    values = dict(written)
                    for n in ("plate", "acquisition", "well"):
                        if n in spec:
                            if n in values and values[n] != spec[n]:
                                disagree.append((dict(key), dict(written)))
                            values[n] = spec[n]
                    written = tuple((n, values[n]) for n in names)
                key = written
                if entry.get("coordinate_system") is not None:
                    system_of[key] = str(entry["coordinate_system"])
            if key in out:
                raise ValueError(f"{path}: two images have the index {dict(key)} "
                                 f"({ek.INDEX_KEY} / {ek.IMAGES_KEY})")
            out[key] = node
            if id(node) in where:
                key_of_path[where[id(node)][0]] = key
        if disagree:
            warnings.warn(f"{path}: {len(disagree)} image(s) whose {ek.INDEX_KEY} disagrees "
                          f"with the plate's own numbering (e.g. {disagree[0][1]} for "
                          f"{disagree[0][0]}); the plate's is kept (plan 5.11)",
                          UserWarning, stacklevel=3)
        if listing_groups is not None:
            groups = dict(listing_groups)
        else:
            groups = {"": root_attrs}
            if not series:
                groups.update({well.path.path: well.attributes for well in tree.nodes})
        systems, edges = ek.graph_from_groups(groups, key_of_path, str(path))
        systems, system_of, scene_edges = _read_scenes(groups, key_of_path, out, systems,
                                                       system_of, str(path))
        edges = list(edges) + scene_edges
        if series and marked and len(found) == len(entries) and out:
            tree, out = _tree_of_keys(tree, out)   # flattened series get their level back
        coll = cls(out, name=tree.name, tree=tree, coordinate_systems=systems,
                   system_of=system_of, coordinate_transformations=edges)
        return coll._finish_store_read(path, add)

    def _finish_store_read(self, path, add_missing_translation: bool = False) -> "PyramidCollection":
        """Scope the ids to the store (5.24) and check the graph without opening an image;
        a system named without its definition gets one with unknown axes, with a warning
        (3.26). Each placed image is checked for its level-0 translation when it opens
        (I4; zeros with `add_missing_translation`, decision O-2)."""
        import warnings

        from ome_zarr_pyramid.core.spatial import CoordinateSystem
        bare = sorted(set(self._system_of.values()) - set(self._systems))
        if bare:
            warnings.warn(f"{path}: coordinate system(s) {bare} are named without their "
                          f"definition; they get unknown axes, filled when written",
                          UserWarning, stacklevel=3)
            for sid in bare:
                self._systems[sid] = CoordinateSystem(sid)
        self._source = ("store", str(path))
        self.check_invariants(images=False)
        from ome_zarr_pyramid.core.nodes import MultiscaleNode
        for key, sid in self._system_of.items():
            node = self._pyramids.get(key)
            if isinstance(node, MultiscaleNode) and getattr(node, "loader", None) is not None \
                    and getattr(node, "_pyramid", None) is None:
                _translation_on_load(node, key, sid, add_missing_translation, str(path))
        return self

    @classmethod
    def _from_tree(cls, tree, add: bool = False) -> "PyramidCollection":
        """A collection over an RFC 8 tree: plates keyed as micro-reader keys plate
        images, any other collection ``(("series", i),)``, a single image ``()``."""
        from ome_zarr_pyramid.core.plates import plate_nodes
        from ome_zarr_pyramid.core.rfc8 import tree_entries

        from ome_zarr_pyramid.core import rfc8_keys

        entries = list(_tree_entries(tree) if plate_nodes(tree) else tree_entries(tree))
        entries, systems, system_of, edges = rfc8_keys.graph_from_tree(tree, entries, tree.name)
        out = {}
        for key, node in entries:
            if key in out:
                raise ValueError(f"{tree.name}: two images have the index {dict(key)}")
            out[key] = node
        coll = cls(out, name=tree.name, tree=tree, coordinate_systems=systems,
                   system_of=system_of, coordinate_transformations=edges)
        return coll._finish_store_read(tree.name, add)

    @staticmethod
    def _open(value):
        """A pyramid, opening a lazy image node."""
        from ome_zarr_pyramid.core.nodes import MultiscaleNode
        return value.pyramid if isinstance(value, MultiscaleNode) else value

    @classmethod
    def from_file(cls, file, *, version: str = "0.5", **kwargs) -> "PyramidCollection":
        """A collection from every image of an opened file (micro-reader ``File``, or
        anything with ``images``). Images that are only another resolution level of an
        image (an index with ``resolution`` > 0, from ``resolutions="all"``) are folded
        into that image's pyramid, so one entry is one whole image. `kwargs` go to
        :meth:`Pyramid.from_image`.

        Keys are the images' indices as the file gives them (micro-reader 0.1.0:
        ``(("series", i), ...)``; plate images ``plate``, ``acquisition``, ``well``,
        ``field``); an index name ozp does not know (the old series name eubi-bridge
        0.2.x uses) is kept as it is. :attr:`tree` is built from the indices (see
        :func:`ome_zarr_pyramid.core.plates.tree_from_file`): the file's plates, and its
        series, each a collection when it holds several images.

        Where the images are (3.3, 3.4): :attr:`coordinate_systems` and
        :attr:`coordinate_transformations` are the file's (``file.coordinate_systems``,
        ``file.coordinate_transformations``), and an image whose
        ``metadata.coordinate_system`` names a system is placed in it, its position being
        its level-0 translation. A file without ``coordinate_systems`` (eubi-bridge's own
        objects) gives each system its images name a minimal definition: the id, and the
        axes of the images in it (3.26). An image that names no system is unplaced, its
        position still written as its translation (5.12)."""
        if isinstance(file, (str, os.PathLike)):
            raise TypeError(
                f"from_file takes an opened file, not a path: "
                f"PyramidCollection.from_file(micro_reader.open({str(file)!r})). An OME-Zarr "
                f"store is read with PyramidCollection.from_zarr(path) or io.read(path)")
        out = {}
        system_of = {}
        for image in file.images:
            index = dict(getattr(image, "index", None) or {})
            if int(index.get(_RESOLUTION, 0)) != 0:
                continue                        # a level of an image already in the set
            key = _key(index)
            if key in out:
                raise ValueError(f"two images share the index {dict(key)}")
            out[key] = pyramid_from_image(image, version=version, **kwargs)
            sid = getattr(getattr(image, "metadata", None), "coordinate_system", None)
            if sid is not None:
                system_of[key] = str(sid)
        from ome_zarr_pyramid.core.plates import file_stem, tree_from_file
        from ome_zarr_pyramid.core.spatial import merge_systems, minimal_system
        path = getattr(file, "path", None)
        tree = tree_from_file(file, out, name=file_stem(path) if path else None)
        systems = getattr(file, "coordinate_systems", None)
        if systems is None:
            # the file does not describe its systems (eubi-bridge's own objects): each
            # one an image names gets a minimal definition, its axes the images' own (3.26)
            minimal: Dict[str, Any] = {}
            for key, sid in system_of.items():
                mine = minimal_system(sid, out[key])
                minimal[sid] = merge_systems(minimal[sid], mine) if sid in minimal else mine
            systems = list(minimal.values())
        coll = cls(out, name=path, tree=tree, coordinate_systems=systems,
                   system_of=system_of,
                   coordinate_transformations=getattr(file, "coordinate_transformations",
                                                      None) or ())
        if path:
            coll._source = ("file", str(path))
        coll.check_invariants()
        return coll

    def write_plate(self, path, *, plate: int = 0, version: Optional[str] = None,
                    overwrite: bool = False, storage_options: Optional[dict] = None,
                    writer: Optional[Callable] = None, document: Optional[str] = None,
                    **write_kwargs) -> str:
        """Write plate `plate` of the collection as an NGFF plate; a shortcut for
        ``io.write_plate``. See :meth:`_write_plate` for the options."""
        return _io().write_plate(self, path, plate=plate, version=version,
                                 overwrite=overwrite, storage_options=storage_options,
                                 writer=writer, document=document, **write_kwargs)

    @validating
    def _write_plate(self, path, *, plate: int = 0, version: Optional[str] = None,
                     overwrite: bool = False, storage_options: Optional[dict] = None,
                     writer: Optional[Callable] = None, document: Optional[str] = None,
                     **write_kwargs) -> str:
        """Write plate `plate` of the collection as an NGFF plate: each image in turn
        (``io.write_pyramid``, or `writer`, with `write_kwargs`), then the plate and
        well metadata. Returns `path` (`document`, when given), for ``io.read``. See
        :func:`ome_zarr_pyramid.core.ngff_plate.write_plate` for `version` and
        `overwrite`. ``version="rfc8"`` writes it as RFC 8 (see `write_collection`;
        `document` puts the plate's RFC 8 document in a standalone ``.json``).

        A field holding several images (its images' keys agree in plate, acquisition,
        well and field but differ in tile, view, ...) raises
        :class:`~ome_zarr_pyramid.core.ngff_plate.PlateFieldError` in NGFF 0.4 / 0.5 (and
        0.6): open the file with ``tiles="mosaic"``, or write ``version="rfc8"``. Images
        outside the plate (a slide's label or preview, keyed without a plate) are not
        part of it."""
        self._refuse_empty("write_plate")
        from ome_zarr_pyramid.core.ngff_plate import check_fields, write_plate
        from ome_zarr_pyramid.core.nodes import MultiscaleNode
        from ome_zarr_pyramid.core.plates import plate_nodes

        nodes = plate_nodes(self.tree)
        if not nodes:
            raise ValueError("the collection holds no plate (see .plates)")
        if not 0 <= plate < len(nodes):
            raise IndexError(f"plate {plate}: the collection holds {len(nodes)} plate(s)")
        node = nodes[plate]
        key_of = {id(v): k for k, v in self._pyramids.items()}
        fields = []                            # (well name, [the index of each image])
        for well in node.nodes:
            indices = []
            for image in well.walk():
                if not isinstance(image, MultiscaleNode) or "labels" in image.attributes \
                        or "source" in image.attributes:
                    continue                   # label and derived images are not entries
                key = key_of.get(id(image), key_of.get(id(getattr(image, "_pyramid", None))))
                if key is None:
                    raise ValueError(
                        "this collection is a selection of its plate; write the plate from "
                        "the whole collection (a plate is written with all its images)")
                indices.append(dict(key))
            fields.append((well.name, indices))
        if version == "rfc8":
            return self._write_rfc8(node, key_of, path, document=document, overwrite=overwrite,
                                    storage_options=storage_options, writer=writer,
                                    **write_kwargs)
        check_fields(node.name, fields)        # one image per field, in every version
        node = self._plate_with_keys(node, key_of, version)
        write_plate(node, path, version=version, overwrite=overwrite,
                    storage_options=storage_options, writer=writer, **write_kwargs)
        return _written(path)

    def _plate_with_keys(self, node, key_of, version):
        """A copy of plate `node` carrying the ``eubi:`` keys of 0.4 / 0.5 (plan 5.7): each
        image a stamped clone with its ``eubi:index`` (``pyramid(key)``), each well its
        ``eubi:images`` and ``eubi:kind``, the plate ``eubi:version`` and ``eubi:kind``,
        and each component's systems and edges on its owner (3.6). The collection and its
        tree are not changed."""
        from ome_zarr_pyramid.core import eubi_keys as ek
        from ome_zarr_pyramid.core.ngff_plate import DEFAULT_VERSION, image_paths, well_path
        from ome_zarr_pyramid.core.nodes import MultiscaleNode

        out = node.copy()
        for n in out.walk():
            n.attributes = ek.strip(n.attributes)
        attr = out.attributes["plate"]
        paths, listing, leaves = {}, {}, []
        for well, copied in zip(node.nodes, out.nodes):
            where = well_path(attr, copied)
            for image, image_copy, image_path in zip(well.nodes, copied.nodes,
                                                     image_paths(copied)):
                if not isinstance(image, MultiscaleNode) or "labels" in image.attributes \
                        or "source" in image.attributes:
                    continue
                key = key_of.get(id(image), key_of.get(id(getattr(image, "_pyramid", None))))
                paths[key], listing[key] = f"{where}/{image_path}", where
                leaves.append((image_copy, key))
        written = str(version or node.ngff_fields.get("version") or DEFAULT_VERSION)
        groups, pyramids = ek.graph_attrs(self._derive(list(paths)), paths, listing,
                                          set(listing.values()), written)
        out.attributes.update(groups.get("", {}))
        out.attributes[ek.KIND_KEY] = "plate"
        for copied in out.nodes:
            copied.attributes.update(groups.get(well_path(attr, copied), {}))
            copied.attributes[ek.KIND_KEY] = "well"
        for image_copy, key in leaves:
            image_copy.pyramid = pyramids[key]
            image_copy.attributes[ek.INDEX_KEY] = ek.index_json(key)
        return out

    def write_series(self, path, *, version: Optional[str] = None, overwrite: bool = False,
                     ome_xml: Optional[str] = None, storage_options: Optional[dict] = None,
                     writer: Optional[Callable] = None, document: Optional[str] = None,
                     **write_kwargs) -> str:
        """Write every entry as one series of a bioformats2raw layout; a shortcut for
        ``io.write_series``. See :meth:`_write_series` for the options."""
        return _io().write_series(self, path, version=version, overwrite=overwrite,
                                  ome_xml=ome_xml, storage_options=storage_options,
                                  writer=writer, document=document, **write_kwargs)

    @validating
    def _write_series(self, path, *, version: Optional[str] = None, overwrite: bool = False,
                      ome_xml: Optional[str] = None, storage_options: Optional[dict] = None,
                      writer: Optional[Callable] = None, document: Optional[str] = None,
                      **write_kwargs) -> str:
        """Write every entry as one series of a bioformats2raw layout (``<path>/0``,
        ``<path>/1``, ..., in key order; a collection read from that layout keeps its
        series paths), then the root metadata and the ``OME`` group with its OME-XML.
        Returns `path` (`document`, when given), for ``io.read``. `ome_xml` is stored as ``OME/METADATA.ome.xml``
        (default: the source layout's when it still fits, else a minimal one). See
        :func:`ome_zarr_pyramid.core.ngff_series.write_series`.

        A series holding several images (keys with one series value that differ in a
        tile, view, ... index) is flattened: each image is written as a series of its
        own, in key order, with one ``UserWarning`` per call.

        ``version="rfc8"``: RFC 8 replaces the layout with a collection of the series
        images, so that is written (see `write_collection`); `ome_xml` is not used."""
        import warnings

        from ome_zarr_pyramid.core.ngff_series import LAYOUT_KEY, write_series

        self._refuse_empty("write_series")
        from ome_zarr_pyramid.core.nodes import CollectionNode, MultiscaleNode

        tree = self.tree
        key_of = self._key_of_nodes()          # each image node's key (for the writers)
        if isinstance(tree, CollectionNode) and LAYOUT_KEY in tree.attributes \
                and "plate" not in tree.attributes \
                and {id(n) for n in tree.nodes} == {id(v) for v in self._pyramids.values()}:
            node = tree                    # read from a series layout, all of it: as read
            keys = [key_of[id(n)] for n in tree.nodes]
        else:                              # any other collection: series 0, 1, ... in key order
            keys = sorted(self._pyramids)
            crowded = _crowded(keys) if version != "rfc8" else []
            if crowded:
                names = sorted({n for _, varied in crowded for n in varied})
                many = len(crowded)
                warnings.warn(
                    f"write_series: {many} of the collection's series hold"
                    f"{'' if many > 1 else 's'} several images (they differ in {names}); "
                    f"each image is written as a series of its own, in key order "
                    f"({len(keys)} series): the bioformats2raw layout cannot say which "
                    f"{' / '.join(names)} an image is", UserWarning, stacklevel=2)
            node = CollectionNode("series")
            for i, key in enumerate(keys):
                child = node.add(MultiscaleNode(str(i), self[key]))
                key_of[id(child)] = key
        if version == "rfc8":
            return self._write_rfc8(node, key_of, path, document=document, overwrite=overwrite,
                                    storage_options=storage_options, writer=writer,
                                    **write_kwargs)
        node = self._series_with_keys(node, keys, version)
        write_series(node, path, version=version, overwrite=overwrite, ome_xml=ome_xml,
                     storage_options=storage_options, writer=writer, **write_kwargs)
        return _written(path)

    def _series_with_keys(self, node, keys, version):
        """A copy of series layout `node` (its images are the entries `keys`, in order)
        carrying the ``eubi:`` keys of 0.4 / 0.5 (plan 5.7): each series image a stamped clone with its ``eubi:index``, the root
        ``eubi:version``, ``eubi:images`` and the graph (the root is the only collection
        group of the layout, so it owns every component)."""
        from ome_zarr_pyramid.core import eubi_keys as ek
        from ome_zarr_pyramid.core.ngff_plate import DEFAULT_VERSION, image_paths
        from ome_zarr_pyramid.core.nodes import MultiscaleNode

        out = node.copy()
        out.attributes = ek.strip(out.attributes)
        paths, leaves = {}, []
        images = [(c, p) for n, c, p in zip(node.nodes, out.nodes, image_paths(out))
                  if isinstance(n, MultiscaleNode)]
        for (image_copy, series_path), key in zip(images, keys):
            paths[key] = series_path
            leaves.append((image_copy, key))
        written = str(version or node.ngff_fields.get("version") or DEFAULT_VERSION)
        groups, pyramids = ek.graph_attrs(self._derive(list(paths)), paths,
                                          {k: "" for k in paths}, [""], written)
        out.attributes.update(groups.get("", {}))
        for image_copy, key in leaves:
            image_copy.pyramid = pyramids[key]
        return out

    def write_collection(self, path, *, version: str = "rfc8", document: Optional[str] = None,
                         overwrite: bool = False, storage_options: Optional[dict] = None,
                         writer: Optional[Callable] = None,
                         **write_kwargs) -> str:
        """Write the collection as NGFF RFC 8; a shortcut for ``io.write_collection``. See :meth:`_write_collection` for the options."""
        return _io().write_collection(self, path, version=version, document=document,
                                      overwrite=overwrite, storage_options=storage_options,
                                      writer=writer, **write_kwargs)

    @validating
    def _write_collection(self, path, *, version: str = "rfc8", document: Optional[str] = None,
                          overwrite: bool = False, storage_options: Optional[dict] = None,
                          writer: Optional[Callable] = None, **write_kwargs) -> str:
        """Write the collection as NGFF RFC 8 (``version="rfc8"``, the only one: a 0.4 /
        0.5 plate or series is `write_plate` / `write_series`): each image as an RFC 8
        multiscale under `path`, then the collection document in `path`'s ``zarr.json``,
        or in the standalone `document` (``.json``). The tree as read or built
        (plates, wells, nested collections, attributes) is written as it is; a
        collection without one is written as a flat collection of its images, in key
        order.         Returns `path` (`document`, when given), for ``io.read``. See
        :func:`ome_zarr_pyramid.core.rfc8.write_rfc8`.

        A selection (``coll.select(plate=0, well=3)``) is written with the part of the
        tree that holds its images: collections left empty are dropped, as are label and
        derived images whose source is not written; names, attributes and the plate's
        rows and columns stay as read."""
        from ome_zarr_pyramid.core.nodes import CollectionNode, MultiscaleNode
        from ome_zarr_pyramid.core.rfc8 import tree_entries

        self._refuse_empty("write_collection")

        if version != "rfc8":
            raise ValueError(f"write_collection writes version='rfc8'; NGFF {version} plates "
                             f"and series are write_plate / write_series")

        tree = self.tree
        key_of = self._key_of_nodes()
        if isinstance(tree, MultiscaleNode):
            node = CollectionNode(tree.name, [tree])
        elif isinstance(tree, CollectionNode):
            held = {id(v) for v in self._pyramids.values()}
            images = [n for _, n in (_tree_entries(tree) if self.plates else tree_entries(tree))]
            if any(id(n) not in held and id(getattr(n, "_pyramid", None)) not in held
                   and getattr(n, "_alias_of", None) is None for n in images):
                # a selection (coll.select(well=3)): RFC 8 holds any tree, so the tree cut
                # to the images the collection holds is written
                node = _pruned(tree, key_of)
            else:
                node = tree
        else:
            node = CollectionNode(str(self.name or "collection"))
            for i, key in enumerate(sorted(self._pyramids)):
                child = node.add(MultiscaleNode(str(i), self[key]))
                key_of[id(child)] = key
        return self._write_rfc8(node, key_of, path, document=document, overwrite=overwrite,
                                storage_options=storage_options, writer=writer, **write_kwargs)

    def _write_rfc8(self, node, key_of, path, **options) -> str:
        """`node` (the part of the tree being written) as RFC 8; returns where to read it
        back (the standalone document when there is one, else `path`)."""
        from ome_zarr_pyramid.core import rfc8_keys
        from ome_zarr_pyramid.core.rfc8 import write_rfc8

        write_rfc8(rfc8_keys.annotate(self, node, key_of), path, **options)
        return _written(path, options.get("document"))

    def _refuse_empty(self, where: str) -> None:
        """A collection without images is not written: an empty store reads back as
        nothing, and an empty selection is mostly a value that matched no image."""
        if not self._pyramids:
            raise ValueError(
                f"{where}: the collection holds no image, so nothing is written. Was it "
                f"selected empty? select() compares index values: well=3 or "
                f"well=coll.plates[0].well_index('C3'), not a Well object "
                f"(coll.plates[0].wells[3].as_collection() gives a well's images)")

    def _key_of_nodes(self) -> Dict[int, Any]:
        """``id`` of each entry's value (its tree node, or its pyramid) -> its key (for the
        writers; a node a writer builds is added by its own ``id``)."""
        return {id(v): k for k, v in self._pyramids.items()}

    @property
    def plates(self) -> list:
        """The collection's plates (``[]`` when it holds none), navigated as in
        micro-reader: ``coll.plates[0].wells[3].fields[0].image`` (see
        :mod:`ome_zarr_pyramid.core.plates`). From a micro-reader file they match
        ``file.plates``; from an NGFF plate they follow its plate and well metadata."""
        from ome_zarr_pyramid.core.plates import plates_of
        return plates_of(self)

    # -- mapping ---------------------------------------------------------------
    def _lookup(self, key):
        if isinstance(key, Mapping):
            key = _key(key)
        return self._pyramids[tuple(key)]

    def __getitem__(self, key):
        return self._open(self._lookup(key))

    def pyramid(self, key):
        """The entry at `key` as a pyramid of its own (plan 3.26, 8.25): a metadata clone
        (sharing the arrays) that keeps the system the entry is placed in, as the custom
        metadata ``eubi:coordinate_system`` and ``eubi:coordinate_systems`` (its id and
        definition), or none when the entry is unplaced. Written alone, it keeps where it
        is; put back into a collection with others from this one, it shares their systems.
        ``coll[key]`` is the shared object, never changed."""
        from ome_zarr_pyramid.core.plates import file_stem
        from ome_zarr_pyramid.core.spatial import stamp_system

        if isinstance(key, Mapping):
            key = _key(key)
        key = tuple(key)
        shared = self[key]
        clone = shared._clone_metadata_only()
        for attr in ("acquisition", "source_index", "joined_index", "source_series"):
            if hasattr(shared, attr):
                setattr(clone, attr, copy.deepcopy(getattr(shared, attr)))
        sid = self._system_of.get(key)
        stamp_system(clone, self._systems[sid] if sid is not None else None)
        clone._spatial_source = self._source
        clone._spatial_set_name = file_stem(self.name) if self.name else None
        return clone

    def only(self):
        """The one pyramid of a one-image collection (a file holding one image, say), as
        :meth:`pyramid` gives it: it keeps the system it is placed in. A collection of
        any other size raises ``ValueError`` listing its keys: pick one with
        ``coll.pyramid(key)`` or narrow it with ``select`` / ``filter`` first."""
        if len(self._pyramids) != 1:
            keys = [dict(k) for k in self._pyramids]
            shown = ", ".join(map(str, keys[:5])) + (" ..." if len(keys) > 5 else "")
            raise ValueError(f"only(): the collection holds {len(keys)} images, not one"
                             f"{': ' + shown if keys else ''}; take one with "
                             f"coll.pyramid(key), or narrow it with select / filter first")
        return self.pyramid(next(iter(self._pyramids)))

    def __iter__(self) -> Iterator[Tuple[Tuple[str, int], ...]]:
        return iter(self._pyramids)

    def __len__(self) -> int:
        return len(self._pyramids)

    def __repr__(self) -> str:
        names = sorted({k for key in self._pyramids for k, _ in key})
        return f"PyramidCollection({len(self)} pyramids, indexed by {names})"

    def _index_names(self) -> List[str]:
        """The index names of the collection's keys, sorted (those of the collection it
        was selected from included, so an empty selection still knows them)."""
        return sorted(self._names.union(n for key in self._pyramids for n, _ in key))

    def select(self, **index) -> "PyramidCollection":
        """The entries whose index matches every given value, as micro-reader's
        ``File.select`` (which gives a list of images): ``select(view=1)``,
        ``select(plate=0, well=12)``. An image without an index given does not match it.
        ``select`` slices along the indices (containment and orthogonal); for queries over
        the table's columns (dtype, sizes, units, functions of them) use :meth:`filter`
        (decision C-6).

        - A value may be a list, tuple, range or set: any of them (``illumination=[2, 3]``).
        - ``well`` is a well's place in ``plates[p].wells`` (as in micro-reader) or its name
          (``well="C3"``, in each plate).
        - ``row`` and ``column`` select a plate's wells by row and column: 0-based numbers
          (as ``Well.row`` / ``Well.column``: ``column=2`` is the column named "3") or
          names (``row="C"``, ``column="3"``), one or several (``row=["B", "C"]``,
          ``column=range(2, 5)``). They are another address of the well, so they do not
          go with ``well``.
        - A ``Plate``, ``Well``, ``Field`` or ``Acquisition`` object is no value (TypeError):
          ``coll.plates[0].wells[3].as_collection()`` is a well's images.

        A name that is neither one of micro-reader's index names nor one the collection's
        keys carry raises ``KeyError``; a known name no image has gives an empty
        collection. The tree is kept as it is (a plate keeps every well); ``drop_empty()``
        drops what holds no image."""
        check_index_names(index, self._index_names(), "select", extra=WELL_ADDRESS)
        wanted = {name: index_values(name, value) for name, value in index.items()}
        in_wells = self._keys_in_wells(wanted)
        keys = []
        for key in self._pyramids:
            if in_wells is not None and key not in in_wells:
                continue
            values = dict(key)
            if all(values.get(name) in allowed for name, allowed in wanted.items()):
                keys.append(key)
        return self._derive(keys)

    def _keys_in_wells(self, wanted: Dict[str, tuple]) -> Optional[set]:
        """The keys of the wells `wanted` names by name, row or column (those entries are
        taken out of `wanted`), or None when it names wells by place only (matched as
        any other index)."""
        rows, columns = wanted.pop("row", None), wanted.pop("column", None)
        wells = wanted.get("well")
        if wells is not None and (rows is not None or columns is not None):
            raise TypeError("select: give well= or row= / column=, not both (row and column "
                            "are another address of the well)")
        names = {v for v in wells if isinstance(v, str)} if wells is not None else set()
        if rows is None and columns is None and not names:
            return None
        plates = self.plates
        if not plates:
            raise ValueError("select: row=, column= and well names select a plate's wells; "
                             "the collection holds no plate")
        if wells is not None:
            del wanted["well"]
        places = {v for v in wells if not isinstance(v, str)} if wells is not None else set()
        out = set()
        for plate in plates:
            for w, well in enumerate(plate.wells):
                if wells is not None:
                    hit = w in places or well.name in names
                else:
                    hit = _on_axis(rows, well.row, plate.rows) and \
                        _on_axis(columns, well.column, plate.columns)
                if hit:
                    out.update(key for field in well.fields for key in field.keys)
        return out

    def drop_empty(self) -> "PyramidCollection":
        """This collection with the containers that hold none of its images dropped from
        its tree: a field without images, then a well without fields, a plate without
        wells and any other collection left empty (as pandas' ``dropna``:
        ``coll.select(plate=0, well=12).drop_empty()``). The positional indices are
        numbered anew to match: ``"plate": p`` is ``plates[p]`` and ``"well": w`` is
        ``plates[p].wells[w]``, as in micro-reader, so a well's key can change. Every other
        index (field, series, tile, view, illumination, ...) keeps its value: it is the
        source's own number. The coordinate-system graph follows the new keys; a plate keeps
        its rows and columns. This collection is not changed."""
        from ome_zarr_pyramid.core.nodes import CollectionNode, MultiscaleNode
        from ome_zarr_pyramid.core.plates import plate_nodes

        keys = list(self._pyramids)
        tree = self.tree
        if not isinstance(tree, CollectionNode):
            return self._derive(keys)
        key_of = self._key_of_nodes()
        copies: Dict[int, Any] = {}
        out = _pruned(tree, key_of, keep_wells=False, copies=copies)
        rename: Dict[Any, Any] = {}
        new_plates = plate_nodes(out)
        for plate in self.plates:
            copied = copies.get(id(plate.node))
            p_new = next((i for i, n in enumerate(new_plates) if n is copied), None)
            if p_new is None:
                continue
            for well in plate.wells:
                copied_well = copies.get(id(well.node))
                w_new = next((i for i, n in enumerate(copied.children) if n is copied_well),
                             None)
                if w_new is None:
                    continue
                for field in well.fields:
                    for key in field.keys:
                        rename[key] = tuple((n, p_new if n == "plate" else
                                             w_new if n == "well" else v) for n, v in key)
        values = {}
        for key in keys:
            value = self._pyramids[key]
            if isinstance(value, MultiscaleNode):
                value = copies.get(id(value), value)   # the tree's own node, for the writers
            values[rename.get(key, key)] = value
        new = type(self)(values, name=self.name, tree=out,
                         **_renamed_graph(self._graph_for(keys), rename))
        new._names = frozenset(self._index_names())
        new._source = self._source
        return new

    # -- operations: every Pyramid operation through one _map (decision C-9) -------------
    def _map(self, name: str, *args, **kwargs) -> "PyramidCollection":
        """`Pyramid.<name>(*args, **kwargs)` on every image (C2), as a new collection with
        the same keys, a copy of the tree and the same coordinate-system graph. Lazy: an
        image not opened yet is opened, and the operation applied, when it is first used
        (read, or written); an image already open, or held in memory, gets it now (its
        arrays stay lazy). An operation that does not fit every image is refused before
        anything is done, naming the images (C3, decision C-5); the check runs on the
        table, so it opens no image. Only ozp's own operations go through here: a user's
        function on every image is pyrops'."""
        from ome_zarr_pyramid.core.nodes import MultiscaleNode, Node

        fits, drops = _FITS.get(name), _DROPS.get(name)
        dropped: Dict[Any, set] = {}
        if fits is not None or drops is not None:
            table = self.table
            rows = list(zip(table.keys, table.rows))
            if fits is not None:
                self._refuse(name, [(key, found) for key, row in rows
                                    for found in [fits(row, args, kwargs)] if found is not None])
            if drops is not None:
                dropped = {key: drops(row, args, kwargs) for key, row in rows}
                # the graph rule (C4): an image an edge runs through keeps its axes
                ends = self._edge_ends()
                self._refuse(name, [
                    (key, (f"an edge of the coordinate-system graph (affine, registration) "
                           f"runs through it, and it would lose axis {', '.join(sorted(gone))}",
                           f"a slice keeps the axis: {name}("
                           f"{sorted(gone)[0]}=slice(i, i + 1))" if name == "isel" else
                           "leave those images out first: coll.select(...)"))
                    for key, gone in dropped.items() if gone and key in ends])
        # RFC 8 label nodes follow their image's geometry (decision C-2), by the Pyramid
        # rule itself: each label attached to its image, the operation applied, the label
        # taken back. Done now (images with label nodes are opened, metadata only), so a
        # label that cannot follow refuses the operation before anything is done.
        followed = self._labels_followed(name, args, kwargs) if name in _GEOMETRY else {}

        def op(pyr):
            return getattr(pyr, name)(*args, **kwargs)

        tree, out, copies = self.tree, None, {}
        if isinstance(tree, Node):
            out = tree.copy()
            pairs = list(zip(tree.walk(), out.walk()))
            copies = {id(o): c for o, c in pairs}
            for o, c in pairs:                    # an alias repeats the image's copy
                first = getattr(o, "_alias_of", None)
                if first is not None and id(first) in copies:
                    c._alias_of = copies[id(first)]
                    c.loader = lambda _n, first=c._alias_of: first.pyramid
        values, results = {}, {}
        for key, value in self._pyramids.items():
            if isinstance(value, MultiscaleNode):
                node = copies.get(id(value)) or value.copy()
                if value.loaded:
                    node.pyramid = op(value.pyramid)
                else:
                    node.pyramid = None
                    node.loader = _MappedLoader(value, op)
                values[key] = node
            else:                                 # a pyramid in memory
                results[id(value)] = values[key] = op(value)
        for o, c in (zip(tree.walk(), out.walk()) if out is not None else ()):
            held = getattr(o, "_pyramid", None)
            if isinstance(o, MultiscaleNode) and id(held) in results:
                c.pyramid = results[id(held)]
            if id(o) in followed:                     # an RFC 8 label node: its label after
                c.pyramid = followed[id(o)]
        graph = self._graph_for(list(self._pyramids))
        if any(dropped.values()):
            graph["coordinate_systems"] = self._systems_without(graph, dropped)
        new = type(self)(values, name=self.name, tree=out, **graph)
        new._names = frozenset(self._index_names())
        new._source = self._source
        if name in _ROWS:                         # its table from this one's, opening nothing
            new._rows_from = (self, _ROWS[name], args, kwargs)
        return new

    def _refuse(self, name: str, misfits) -> None:
        """Decision C-5: `misfits` (``[(key, (why, a narrowing hint))]``) refuse the whole
        operation, naming up to five images."""
        if not misfits:
            return
        shown = "; ".join(f"{dict(k)}: {why}" for k, (why, _) in misfits[:5])
        more = f" and {len(misfits) - 5} more" if len(misfits) > 5 else ""
        raise OperationMismatchError(
            f"{name}: does not fit {len(misfits)} of the collection's {len(self)} images, so "
            f"nothing is done: {shown}{more}. Narrow the collection first, e.g. "
            f"{misfits[0][1][1]}")

    def _edge_ends(self) -> set:
        """The image keys an edge of the graph runs through (not placements)."""
        from ome_zarr_pyramid.core.spatial import is_image_key, steps
        out = set()
        for edge in self._edges:
            for step in (edge,) + (steps(edge) if edge.type == "sequence" else ()):
                out.update(end for end in (step.input, step.output) if is_image_key(end))
        return out

    @staticmethod
    def _systems_without(graph: dict, dropped: Dict[Any, set]) -> list:
        """The graph's systems, a system losing an axis every image placed in it dropped
        (an image may have fewer axes than its system, so one that keeps it keeps it)."""
        from ome_zarr_pyramid.core.spatial import CoordinateSystem
        out = []
        for cs in graph["coordinate_systems"]:
            members = [k for k, sid in graph["system_of"].items() if sid == cs.id]
            if cs.axes is not None and members:
                gone = set.intersection(*(set(dropped.get(k, ())) for k in members))
                if gone:
                    cs = CoordinateSystem(cs.id, cs.name,
                                          tuple(a for a in cs.axes if a.name not in gone),
                                          cs.estimated, dict(cs.extra))
            out.append(cs)
        return out

    def _labels_followed(self, name: str, args, kwargs) -> Dict[int, Any]:
        """``id`` of each RFC 8 label node of the tree whose image is in the collection ->
        its label after `name` (decision C-2): the label attached to its image, the
        Pyramid operation applied (its label rule), the label taken back. A label that
        cannot follow raises :class:`LabelMismatchError` naming the image."""
        from ome_zarr_pyramid.core.nodes import LabelsAttribute, MultiscaleNode, Node
        from ome_zarr_pyramid.core.pyramid import LabelMismatchError

        tree = self.tree
        if not isinstance(tree, Node):
            return {}
        by_id = {n.id: n for n in tree.walk() if isinstance(n, MultiscaleNode) and n.id}
        key_of = self._key_of_nodes()
        out: Dict[int, Any] = {}
        for node in tree.walk():
            if not isinstance(node, MultiscaleNode):
                continue
            labels = node.attributes.get("labels")
            if not isinstance(labels, LabelsAttribute) or len(labels.source) != 1:
                continue
            image = by_id.get(labels.source[0].id)
            key = key_of.get(id(image), key_of.get(id(getattr(image, "_pyramid", None)))) \
                if image is not None else None
            if key is None:
                continue                          # its image is not in this collection
            try:
                with_label = self[key].add_image_label(node.pyramid, "label")
                out[id(node)] = getattr(with_label, name)(*args, **kwargs).labels["label"]
            except LabelMismatchError as exc:
                raise LabelMismatchError(f"image {dict(key)}: {exc}") from None
        return out

    # -- axes and levels (C4): the Pyramid methods on every image ------------------------
    def isel(self, **indexers) -> "PyramidCollection":
        """`Pyramid.isel` on every image (``coll.isel(c=0)``, ``coll.isel(z=slice(10, 20))``):
        an int drops the axis, a slice or list keeps it; scales and translations follow, so
        each image keeps its place. An axis or index an image does not have refuses the
        call, naming the images. Dropping an axis of an image an edge of the
        coordinate-system graph runs through is refused (a slice keeps the axis); a system
        whose images all drop an axis loses it. Lazy."""
        return self._map("isel", **indexers)

    def squeeze(self, axes: Optional[str] = None) -> "PyramidCollection":
        """`Pyramid.squeeze` on every image: size-1 axes dropped (each image its own, or
        those of `axes`, which must be size 1 in every image). The graph as for
        :meth:`isel`. Lazy."""
        return self._map("squeeze", axes)

    def select_levels(self, *levels) -> "PyramidCollection":
        """`Pyramid.select_levels` on every image (``coll.select_levels(0, 1, 2)``). A level
        some image does not have refuses the call, naming the images:
        ``coll.filter(nlevels=lambda n: n >= 3).select_levels(0, 1, 2)``. Lazy."""
        return self._map("select_levels", *levels)

    def downscale(self, **options) -> "PyramidCollection":
        """`Pyramid.downscale` on every image (``coll.downscale(n_layers=4)``): each image's
        own plan from its level 0, built when it is written. Lazy."""
        return self._map("downscale", **options)

    # -- storage and values (C5) ---------------------------------------------------------
    def rechunk(self, chunk_shape=None, chunk_size_mb=None) -> "PyramidCollection":
        """`Pyramid.rechunk` on every image: the storage chunks written (a shape, or a size
        dyna solves per image). A chunk shape of another length than an image's axes refuses
        the call, naming the images. Lazy."""
        return self._map("rechunk", chunk_shape=chunk_shape, chunk_size_mb=chunk_size_mb)

    def to_version(self, version: str) -> "PyramidCollection":
        """`Pyramid.to_version` on every image (``"0.4"``, ``"0.5"``, ``"0.6"``). A plate or
        series is written in the version ``write`` is given; this sets each image's own.
        Lazy."""
        return self._map("to_version", version)

    def saturate_cast(self, dtype) -> "PyramidCollection":
        """`Pyramid.saturate_cast` on every image: ``dtype``, clipped instead of wrapped
        (uint16 4000 -> uint8 255). Lazy."""
        return self._map("saturate_cast", dtype)

    def _operate(self, name: str, other) -> "PyramidCollection":
        """An operator with a number, image by image (``coll > 128``, ``coll * 2``,
        ``65535 - coll``). Two collections do not combine yet (decision C-4)."""
        import numbers
        if isinstance(other, (numbers.Number, np.generic)):
            return self._map(name, other)
        if isinstance(other, PyramidCollection):
            raise TypeError(
                f"{name}: two collections do not combine image by image yet (decision C-4); "
                f"combine their images one by one, e.g. "
                f"PyramidCollection({{k: a[k] - b[k] for k in a}})")
        raise TypeError(f"{name}: a collection combines with a number, not a "
                        f"{type(other).__name__}; apply it to each image (coll[key])")

    # arithmetic, comparison and bitwise operators with a number, as on a Pyramid; == and
    # != are elementwise too (decision C-1): with anything but a number they say whether
    # it is the same collection, so `coll in some_list` works
    def __add__(self, o): return self._operate("__add__", o)
    def __radd__(self, o): return self._operate("__radd__", o)
    def __sub__(self, o): return self._operate("__sub__", o)
    def __rsub__(self, o): return self._operate("__rsub__", o)
    def __mul__(self, o): return self._operate("__mul__", o)
    def __rmul__(self, o): return self._operate("__rmul__", o)
    def __truediv__(self, o): return self._operate("__truediv__", o)
    def __rtruediv__(self, o): return self._operate("__rtruediv__", o)
    def __floordiv__(self, o): return self._operate("__floordiv__", o)
    def __rfloordiv__(self, o): return self._operate("__rfloordiv__", o)
    def __mod__(self, o): return self._operate("__mod__", o)
    def __rmod__(self, o): return self._operate("__rmod__", o)
    def __pow__(self, o): return self._operate("__pow__", o)
    def __rpow__(self, o): return self._operate("__rpow__", o)
    def __lt__(self, o): return self._operate("__lt__", o)
    def __le__(self, o): return self._operate("__le__", o)
    def __gt__(self, o): return self._operate("__gt__", o)
    def __ge__(self, o): return self._operate("__ge__", o)
    def __and__(self, o): return self._operate("__and__", o)
    def __rand__(self, o): return self._operate("__rand__", o)
    def __or__(self, o): return self._operate("__or__", o)
    def __ror__(self, o): return self._operate("__ror__", o)
    def __xor__(self, o): return self._operate("__xor__", o)
    def __rxor__(self, o): return self._operate("__rxor__", o)

    def __eq__(self, o):
        import numbers
        if isinstance(o, (numbers.Number, np.generic)):
            return self._map("__eq__", o)
        return NotImplemented                     # anything else: the same collection or not

    def __ne__(self, o):
        import numbers
        if isinstance(o, (numbers.Number, np.generic)):
            return self._map("__ne__", o)
        return NotImplemented

    __hash__ = None                               # a Mapping: not hashable, as before

    def __neg__(self): return self._map("__neg__")
    def __pos__(self): return self._map("__pos__")
    def __abs__(self): return self._map("__abs__")
    def __invert__(self): return self._map("__invert__")

    def set_channels(self, overrides) -> "PyramidCollection":
        """`Pyramid.set_channels` on every image: the same channel changes everywhere
        (``coll.set_channels({0: {"label": "DAPI", "color": "blue"}})``). A channel index
        or label an image does not have refuses the call, naming the images. Lazy."""
        return self._map("set_channels", overrides)

    def set_display_range(self, method: str = "minmax", **options) -> "PyramidCollection":
        """`Pyramid.set_display_range` on every image: each image's windows from its own
        data (``method``, ``p_low``, ``p_high``, ``stats_level``, ...). Computed when an
        image is first used, so a write reads each image's data once more, image by image.
        A ``stats_level`` some image does not have refuses the call."""
        return self._map("set_display_range", method, **options)

    def set_custom_metadata(self, name: str, block) -> "PyramidCollection":
        """`Pyramid.set_custom_metadata` on every image: the block `name` set to `block`
        (``None`` removes it). Lazy."""
        return self._map("set_custom_metadata", name, block)

    def _graph_for(self, keys) -> dict:
        """The part of the graph the entries `keys` keep (3.25): their placements, the
        edges whose image ends are all among them, and the systems those name (a
        system nothing names is left out)."""
        from ome_zarr_pyramid.core.spatial import is_image_key, steps

        kept = set(keys)
        system_of = {k: s for k, s in self._system_of.items() if k in kept}

        def ends(edge):
            for step in (edge,) + (steps(edge) if edge.type == "sequence" else ()):
                yield step.input
                yield step.output

        edges = [e for e in self._edges
                 if all(end in kept for end in ends(e) if is_image_key(end))]
        named = set(system_of.values()) | {end for e in edges for end in ends(e)
                                           if isinstance(end, str)}
        systems = [cs for sid, cs in self._systems.items() if sid in named]
        return {"coordinate_systems": systems, "system_of": system_of,
                "coordinate_transformations": edges}

    def _derive(self, keys) -> "PyramidCollection":
        """A collection of these keys, in this order: the same tree (plates keep every
        well, so "well": w stays wells[w]; their fields hold the kept images only), the
        part of the graph they keep (3.25) and the table rows already built."""
        keys = list(keys)
        new = type(self)({k: self._pyramids[k] for k in keys}, name=self.name, tree=self.tree,
                         **self._graph_for(keys))
        new._rows = {k: self._rows[k] for k in keys if k in self._rows}
        new._names = frozenset(self._index_names())
        new._source = self._source
        return new

    def _branch(self, keys, tree, name) -> "PyramidCollection":
        """A collection of `keys` with `tree` (a node of this collection's tree: a plate, a
        well, a field) as its tree: what a plate, well or field's ``as_collection`` gives."""
        keys = list(keys)
        new = type(self)({k: self._pyramids[k] for k in keys}, name=name, tree=tree,
                         **self._graph_for(keys))
        new._rows = {k: self._rows[k] for k in keys if k in self._rows}
        new._names = frozenset(self._index_names())
        new._source = self._source
        if getattr(self, "_rows_from", None) is not None:
            new._rows_from = self._rows_from
        return new

    # -- the table and its queries ----------------------------------------------
    @property
    def table(self):
        """The collection as a table, one row per image: its key columns, its metadata
        (axes, shape, dtype, levels, chunks, codec, scale, units, channels, ...; see
        ``ome_zarr_pyramid.core.query``) and, for plates, ``plate_name``, ``well_name``,
        ``row_name``, ``column_name`` and ``acquisition_name``. Built from metadata only, once: images not opened yet are read
        from their metadata files in concurrent batches."""
        from ome_zarr_pyramid.core.query import CollectionTable
        self._build_rows()
        return CollectionTable(list(self._pyramids), [self._rows[k] for k in self._pyramids])

    def _build_rows(self) -> None:
        from ome_zarr_pyramid.core.ngff_plate import ImageOpener
        from ome_zarr_pyramid.core.nodes import MultiscaleNode
        from ome_zarr_pyramid.core.query import row_from_pyramid, rows_from_files

        missing = [k for k in self._pyramids if k not in self._rows]
        if not missing:
            return
        derived = getattr(self, "_rows_from", None)
        if derived is not None:                  # a mapped collection: its source's rows,
            source, update, args, kwargs = derived    # changed as the operation changes them
            source._build_rows()
            for key in missing:
                if key in source._rows:
                    row = update(dict(source._rows[key]), args, kwargs)
                    if row is not None:               # None: read it from the image
                        self._rows[key] = row
            missing = [k for k in missing if k not in self._rows]
            if not missing:
                return
        places = self._plate_places()
        on_disk: Dict[int, List] = {}                  # storage_options id -> [(key, location)]
        options: Dict[int, Any] = {}
        to_open = []                                   # [(key, value, location)]
        for key in missing:
            value = self._pyramids[key]
            opener = getattr(value, "loader", None) if isinstance(value, MultiscaleNode) else None
            location = opener.path_of(value) if isinstance(opener, ImageOpener) and value.path else None
            if location is not None and not value.loaded:
                so = opener.storage_options
                on_disk.setdefault(id(so), []).append((key, location))
                options[id(so)] = so
            else:
                to_open.append((key, value, location))
        # images that have no metadata file to read (in memory, or the result of an
        # operation): opened and read, each in its own thread when there are many (each
        # key opens its own node; the rows are the ones a serial loop gives)
        if len(to_open) >= _PARALLEL_ROWS:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(16, len(to_open))) as pool:
                rows = list(pool.map(
                    lambda job: row_from_pyramid(self._open(job[1]), path=job[2]), to_open))
        else:
            rows = [row_from_pyramid(self._open(value), path=location)
                    for _, value, location in to_open]
        for (key, _, _), row in zip(to_open, rows):
            self._rows[key] = row
        for group, jobs in on_disk.items():
            rows = rows_from_files([loc for _, loc in jobs], options[group])
            for (key, location), row in zip(jobs, rows):
                # not an NGFF 0.4 / 0.5 image after all: opened, metadata only
                self._rows[key] = row if row is not None else row_from_pyramid(
                    self._open(self._pyramids[key]), path=location)
        for key in missing:
            row = self._rows[key]
            row.update(dict(key))
            row.update(places.get(id(self._pyramids[key]), {}))

    def _plate_places(self) -> Dict[int, Dict[str, Any]]:
        """Plate name, well name, row and column names and acquisition name of every image
        of the collection's plates, by the id of the value that holds it."""
        from ome_zarr_pyramid.core.nodes import MultiscaleNode
        from ome_zarr_pyramid.core.plates import plate_nodes

        out: Dict[int, Dict[str, Any]] = {}
        for plate in plate_nodes(self.tree):
            attr = plate.attributes["plate"]
            name = plate.ngff_fields.get("plate", {}).get("name", plate.name)
            for well in plate.nodes:
                ref = well.attributes.get("well")
                row = attr.rows[attr.index_of("rows", ref.row)] if ref else None
                col = attr.columns[attr.index_of("columns", ref.column)] if ref else None
                for image in well.walk():
                    if not isinstance(image, MultiscaleNode):
                        continue
                    acq = image.attributes.get("acquisition")
                    acq_name = None
                    if acq is not None:
                        try:
                            item = attr.acquisitions[attr.index_of("acquisitions", acq)]
                            acq_name = item.name
                        except KeyError:
                            pass
                    place = {"plate_name": name, "well_name": well.name,
                             "row_name": row.name or row.id if row else None,
                             "column_name": col.name or col.id if col else None,
                             "acquisition_name": acq_name}
                    out[id(image)] = place
                    if image._pyramid is not None:
                        out[id(image._pyramid)] = place
        return out

    def _check_columns(self, names) -> None:
        from ome_zarr_pyramid.core.query import COLUMN_ALIASES
        columns = self.table.columns
        unknown = [n for n in names if n not in columns and n not in COLUMN_ALIASES]
        if unknown:
            raise KeyError(f"no column {unknown[0]!r}; the columns are {list(columns)}")

    def filter(self, predicate: Optional[Callable] = None, /, **conditions) -> "PyramidCollection":
        """The images that pass every condition, a query over the table's columns (to slice
        along the indices, :meth:`select`; decision C-6): a column name with a value
        (equal) or a function of the value (true), e.g. ``filter(dtype="uint16", unit_x="nanometer",
        nlevels=lambda n: n >= 3)``. `predicate`, if given, is a function of each image's
        record (a dict of its columns): ``filter(lambda image: image["size_z"] ==
        image["size_c"])``. Keys are columns too: ``filter(well=3)``. Nothing is read beyond
        the table's metadata."""
        from ome_zarr_pyramid.core.query import matches
        self._check_columns(conditions)
        rows = self.table.rows
        return self._derive([k for k, row in zip(self._pyramids, rows)
                             if matches(row, predicate, conditions)])

    def unique(self, column: str) -> list:
        """The distinct values of a column, sorted (None last): ``unique("dtype")``."""
        self._check_columns([column])
        return _sorted_values(set(self.table[column]))

    @property
    def columns(self) -> Tuple[str, ...]:
        """The names of the table's columns: the keys, then the image columns, then (for
        plates) the plate columns. Reads the metadata the table needs."""
        return self.table.columns

    def count(self, column: str) -> Dict[Any, int]:
        """The number of images per value of a column: ``count("codec")`` gives
        ``{'blosc-lz4': 2}``. The number of images is ``len(coll)``."""
        self._check_columns([column])
        values = self.table[column]
        return {v: values.count(v) for v in _sorted_values(set(values))}

    def group_by(self, *columns: str) -> Dict[Any, "PyramidCollection"]:
        """One collection per distinct value of `columns` (a tuple of values for several):
        ``group_by("acquisition")``, ``group_by("dtype", "nlevels")``."""
        if not columns:
            raise ValueError("group_by: name at least one column")
        self._check_columns(columns)
        groups: Dict[Any, List] = {}
        for key, row in zip(self._pyramids, self.table.rows):
            value = row.get(columns[0]) if len(columns) == 1 else tuple(row.get(c) for c in columns)
            groups.setdefault(value, []).append(key)
        return {v: self._derive(groups[v]) for v in _sorted_values(set(groups))}

    def sort_by(self, *columns: str, reverse: bool = False) -> "PyramidCollection":
        """The collection with its images in the order of `columns` (None last):
        ``sort_by("well", "field")``, ``sort_by("nbytes", reverse=True)``."""
        if not columns:
            raise ValueError("sort_by: name at least one column")
        self._check_columns(columns)
        rows = dict(zip(self._pyramids, self.table.rows))
        order = sorted(self._pyramids, key=lambda k: tuple(_order_key(rows[k].get(c)) for c in columns),
                       reverse=reverse)
        return self._derive(order)

    def concatenate(self, index, axis: str = "c") -> "PyramidCollection":
        """Fold one or more image indices into an axis: ``concatenate("view")`` joins
        every entry that differs only in its view along ``c`` (one image per illumination,
        say, with all views as channels); ``concatenate(("view", "illumination"))`` folds
        both. Returns a new collection keyed by the remaining indices.

        Each group is joined with :func:`ome_zarr_pyramid.concatenate` (lazily, and
        refused when the entries do not agree in pixel size, position, shape or type), in
        the order of the folded indices' values: the first index outermost. Each joined
        pyramid has ``joined_index``: for every position along *axis* (every channel, for
        ``c``), the folded indices' values it came from, e.g. ``[{"view": 0}, {"view": 0},
        {"view": 1}, ...]``, to name its channels.

        The joined image is placed in its parts' coordinate system (3.25); parts in
        different systems, or some placed and some not, raise ``ValueError``. Edges that
        start or end at a folded part are dropped, with a warning (they belonged to one
        part, not to the joined image). The new collection gets a tree built from its keys
        (with the plates of this one)."""
        from ome_zarr_pyramid.core.concatenate import concatenate

        names = (index,) if isinstance(index, str) else tuple(index)
        if not names:
            raise ValueError("concatenate: no index to fold")
        having = [all(name in dict(key) for name in names) for key in self._pyramids]
        if not any(having):
            raise ValueError(f"concatenate: no entry is indexed by {list(names)} "
                             f"(the indices are {sorted({k for key in self._pyramids for k, _ in key})})")
        if not all(having):
            raise ValueError(f"concatenate: some entries are not indexed by {list(names)}")
        groups: Dict[Tuple, list] = {}
        for key in self._pyramids:
            rest = tuple((k, v) for k, v in key if k not in names)
            groups.setdefault(rest, []).append(key)
        out = {}
        system_of = {}
        for rest, keys in groups.items():
            keys.sort(key=lambda key: tuple(dict(key)[name] for name in names))
            placed = {self._system_of.get(key) for key in keys}
            if len(placed) > 1:
                raise ValueError(
                    f"concatenate: the entries joined into {dict(rest)} are in different "
                    f"coordinate systems or some are unplaced: "
                    f"{ {str(dict(k)): self._system_of.get(k) for k in keys} }; a joined image "
                    f"has one position, in one system")
            if None not in placed:
                system_of[rest] = placed.pop()
            parts = [self._open(self._pyramids[key]) for key in keys]
            joined = concatenate(parts, axis=axis)
            positions = []
            for key, part in zip(keys, parts):
                values = {name: dict(key)[name] for name in names}
                along = part.meta.axis_order.index(str(axis).lower())
                size = part.layers[part.meta.resolution_paths[0]].shape[along]
                positions += [dict(values) for _ in range(size)]
            joined.joined_index = positions
            joined.source_index = dict(rest)
            out[rest] = joined
        # every entry is a folded part: an edge at an image belonged to that part only;
        # edges between systems stay
        graph = self._graph_for([])
        dropped = len(self._edges) - len(graph["coordinate_transformations"])
        if dropped:
            import warnings
            warnings.warn(f"concatenate: {dropped} edge(s) that start or end at a joined "
                          f"entry are dropped: each belonged to one part, not to the joined "
                          f"image", UserWarning, stacklevel=2)
        named = set(system_of.values()) | {cs.id for cs in graph["coordinate_systems"]}
        systems = [cs for sid, cs in self._systems.items() if sid in named]
        from ome_zarr_pyramid.core.plates import tree_from_file
        shim = type("_Keys", (), {"plates": self.plates})()
        tree = tree_from_file(shim, out, name=getattr(self.tree, "name", None) or self.name)
        new = type(self)(out, name=self.name, tree=tree, coordinate_systems=systems,
                         system_of=system_of,
                         coordinate_transformations=graph["coordinate_transformations"])
        new._source = self._source
        return new

    @property
    def varying(self) -> Tuple[str, ...]:
        """The index names whose value differs between entries (what tells them apart)."""
        values: Dict[str, set] = {}
        for key in self._pyramids:
            for name, value in key:
                values.setdefault(name, set()).add(value)
        return tuple(name for name, vals in values.items() if len(vals) > 1)

    # -- read properties shared with Pyramid (decision C-8) -----------------------------
    def _shared(self, name: str, column: Optional[str] = None, exact: bool = False):
        """The value of Pyramid property `name` that every image of the collection has.
        With a table `column`, the images are compared through the table (metadata only)
        and the value is the table's, or with `exact` the first image's own (the table
        keeps less: a dtype without its byte order); without a column, every image is
        opened. Images that differ raise :class:`VaryingPropertyError`."""
        if not self._pyramids:
            raise ValueError(f"{name}: the collection holds no image")
        keys = list(self._pyramids)
        values = self.table[column] if column is not None else [getattr(self[k], name) for k in keys]
        distinct: List[Tuple[Any, Any]] = []
        for key, value in zip(keys, values):
            if not any(_same(value, seen) for seen, _ in distinct):
                distinct.append((value, key))
        if len(distinct) > 1:
            shown = "; ".join(f"{dict(k)}: {v!r}" for v, k in distinct[:3])
            where = (f"coll.table[{column!r}] gives each image's value" if column is not None
                     else f"read it per image (coll[key].{name}); coll.table describes each image")
            raise VaryingPropertyError(
                f"{name} differs between the collection's images ({len(distinct)} different "
                f"values, e.g. {shown}{' ...' if len(distinct) > 3 else ''}): there is no one "
                f"{name} to return; {where}")
        return getattr(self[keys[0]], name) if exact else values[0]

    @property
    def axes(self) -> str:
        """The axes every image has (``"czyx"``, ...); images that differ raise
        :class:`VaryingPropertyError`. Opens no image."""
        return self._shared("axes", "axes")

    @property
    def dtype(self):
        """The pixel type every image has; images that differ raise
        :class:`VaryingPropertyError`. Opens one image."""
        return self._shared("dtype", "dtype", exact=True)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The level-0 shape every image has; images that differ raise
        :class:`VaryingPropertyError`. Opens no image."""
        return self._shared("shape", "shape")

    @property
    def nlayers(self) -> int:
        """The number of resolution levels every image has; images that differ raise
        :class:`VaryingPropertyError`. Opens no image."""
        return self._shared("nlayers", "nlevels")

    @property
    def nlevels(self) -> int:
        """The number of resolution levels all the images have (``nlayers``, by the name
        the table, the docs and dyna-zarr use; decision C-16)."""
        return self.nlayers

    @property
    def compressor(self):
        """The compressor every image has; images that differ raise
        :class:`VaryingPropertyError`. Opens every image."""
        return self._shared("compressor")

    @property
    def scale_factor_dict(self) -> dict:
        """The downscale factors every image has; images that differ raise
        :class:`VaryingPropertyError`. Opens every image."""
        return self._shared("scale_factor_dict")

    @property
    def custom_metadata(self) -> dict:
        """The custom metadata blocks every image has; images that differ raise
        :class:`VaryingPropertyError`. Opens every image."""
        return self._shared("custom_metadata")

    @property
    def supports_dyna(self) -> bool:
        """Whether every image can be processed with dyna-zarr (``Pyramid.supports_dyna``
        for all of them; True for an empty collection). Opens every image."""
        return all(self[k].supports_dyna for k in self._pyramids)

    def suffix(self, key, names: Optional[Dict[str, str]] = None) -> str:
        """The file-name suffix for an entry: one ``_<name><value>`` per varying index,
        e.g. ``_view1_illumination0``. `names` renames indices (``{"illumination":
        "illu"}``)."""
        names = names or {}
        key = _key(key) if isinstance(key, Mapping) else tuple(key)
        parts = [f"_{names.get(name, name)}{value}" for name, value in key
                 if name in self.varying]
        return "".join(parts)

    def update_metadata(self, *, path=None, **options):
        """Rewrite the metadata of each image of the store the collection was read from
        (or of `path`), as ``io.update_metadata(path, ...)``; returns the store read
        again. A collection not read from an OME-Zarr store needs `path`."""
        if path is None:
            kind, where = self._source if isinstance(self._source, tuple) else (None, None)
            if kind != "store":
                raise ValueError("update_metadata: this collection was not read from an "
                                 "OME-Zarr store; pass path=, or call "
                                 "IO().update_metadata(path, ...)")
            path = where
        return _io().update_metadata(path, **options)

    def write(self, path=None, stem: Optional[str] = None, *, version: Optional[str] = None,
              names: Optional[Dict[str, str]] = None, writer: Optional[Callable] = None,
              directory=None, **options):
        """Write the collection to `path`; a shortcut for ``io.write(coll, path, ...)``.
        Returns where to read it back: ``io.read(coll.write(path))`` is the written
        collection (`path`, or the standalone RFC 8 `document` when one is given).

        What is written follows the collection; `version` picks the format:

        - ``"0.4"``, ``"0.5"``, ``"0.6"`` (default: the collection's own, as read, else
          0.5): a plate when the collection is one plate (as ``write_plate``), one image
          when it holds one image keyed ``()`` (a store read, or ``pyr.as_collection()``),
          else a bioformats2raw series layout (as ``write_series``). Several plates raise:
          NGFF 0.4 to 0.6 hold one plate per store.
        - ``"rfc8"``: the collection's tree as an NGFF RFC 8 collection (as
          ``write_collection``), several plates too.

        `options` are those of the writer used (``overwrite``, ``storage_options``,
        ``validate``, ``document``, ...) and of ``io.write_pyramid`` for each image.

        ``coll.write(directory, stem)`` writes each image as a store of its own,
        ``<directory>/<stem><suffix>.ome.zarr`` (`names` renames indices in the suffix),
        and returns ``{key: path}`` of each store."""
        if directory is not None:
            if path is not None:
                raise TypeError("write: give path or directory, not both")
            path = directory
        if path is None:
            raise TypeError("write: missing the path to write to")
        return _io().write(self, path, stem=stem, version=version, names=names,
                           writer=writer, **options)

    def _write(self, path, stem: Optional[str] = None, *, version: Optional[str] = None,
               names: Optional[Dict[str, str]] = None, writer: Optional[Callable] = None,
               **options):
        """The implementation behind ``io.write(coll, ...)`` (see :meth:`write`)."""
        self._refuse_empty("write")
        if stem is not None:
            if version is not None:
                options["version"] = version
            return self._write_images(path, stem, names=names, writer=writer, **options)
        if names is not None:
            raise ValueError("write: names= renames the indices in the store names of "
                             "coll.write(directory, stem); give a stem")
        if version == "rfc8":
            return self._write_collection(path, writer=writer, **options)
        plates = self.plates
        if len(plates) > 1:
            raise ValueError(f"write: the collection holds {len(plates)} plates and NGFF "
                             f"{version or 'up to 0.6'} holds one plate per store; write each "
                             f"with write_plate(path, plate=i), or all of them with "
                             f"version='rfc8'")
        if plates and all(any(n in ("plate", "well") for n, _ in key) for key in self._pyramids):
            return self._write_plate(path, version=version, writer=writer, **options)
        if list(self._pyramids) != [()]:
            return self._write_series(path, version=version, writer=writer, **options)
        pyr = self.pyramid(())
        if version is not None and str(pyr.meta.version) != str(version):
            pyr = pyr.to_version(version)
        (writer or _io().write_pyramid)(pyr, path, **options)
        return str(path)

    @validating
    def _write_images(self, directory, stem: str, *, names: Optional[Dict[str, str]] = None,
                      writer: Optional[Callable] = None, **write_kwargs) -> Dict[Any, str]:
        """Write each entry as ``<directory>/<stem><suffix>.ome.zarr``; returns
        {key: path}. `write_kwargs` go to ``io.write_pyramid`` (or `writer`).

        Each image keeps where it is (plan 3.26, 5.7): written as :meth:`pyramid` gives
        it (``eubi:coordinate_system`` and its definition), with its key as
        ``eubi:index`` and ``eubi:version`` (each output is a root of its own; decision
        O-4). The graph's other edges (affine edges, registrations) have no place in a
        store of one image: they are left out with a warning (plan 3.10); `write_series`,
        `write_plate` and `write_collection` keep them."""
        import os
        import warnings

        from ome_zarr_pyramid.core import eubi_keys as ek
        from ome_zarr_pyramid.core.io import IO
        self._refuse_empty("write")
        if self._edges:
            kinds = sorted({e.type for e in self._edges})
            warnings.warn(f"write: {len(self._edges)} edge(s) of the coordinate-system graph "
                          f"({', '.join(kinds)}) are not written, since each store holds one "
                          f"image; write_series, write_plate or write_collection keep them "
                          f"(plan 3.10)", UserWarning, stacklevel=2)
        write = writer or IO().write_pyramid
        paths = {}
        for key in self._pyramids:
            pyr = self.pyramid(key)
            pyr.meta.set_custom(ek.INDEX_KEY, ek.index_json(key))
            pyr.meta.set_custom(ek.VERSION_KEY, ek.EUBI_VERSION)
            path = os.path.join(str(directory), f"{stem}{self.suffix(key, names)}.ome.zarr")
            write(pyr, path, **write_kwargs)
            paths[key] = path
        return paths
