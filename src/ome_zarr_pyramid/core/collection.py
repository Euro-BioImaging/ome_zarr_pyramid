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
        overrides = {}
        for i, ch in enumerate(channels):
            # unnamed: the default label "Channel {i}" stays; no colour: white
            fields = {"color": _hex(getattr(ch, "color", None)) or "FFFFFF"}
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

    @validating
    def write_plate(self, path, *, plate: int = 0, version: Optional[str] = None,
                    overwrite: bool = False, storage_options: Optional[dict] = None,
                    writer: Optional[Callable] = None, document: Optional[str] = None,
                    **write_kwargs) -> Dict[str, str]:
        """Write plate `plate` of the collection as an NGFF plate: each image in turn
        (``io.write_pyramid``, or `writer`, with `write_kwargs`), then the plate and
        well metadata. Returns ``{"<well>/<image>": path}``. See
        :func:`ome_zarr_pyramid.core.ngff_plate.write_plate` for `version` and
        `overwrite`. ``version="rfc8"`` writes it as RFC 8 (see `write_collection`;
        `document` puts the plate's RFC 8 document in a standalone ``.json``).

        A field holding several images (its images' keys agree in plate, acquisition,
        well and field but differ in tile, view, ...) raises
        :class:`~ome_zarr_pyramid.core.ngff_plate.PlateFieldError` in NGFF 0.4 / 0.5 (and
        0.6): open the file with ``tiles="mosaic"``, or write ``version="rfc8"``. Images
        outside the plate (a slide's label or preview, keyed without a plate) are not
        part of it."""
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
            from ome_zarr_pyramid.core import rfc8_keys
            from ome_zarr_pyramid.core.rfc8 import write_rfc8
            return write_rfc8(rfc8_keys.annotate(self, node, key_of), path, document=document,
                              overwrite=overwrite, storage_options=storage_options,
                              writer=writer, **write_kwargs)
        check_fields(node.name, fields)        # one image per field, in every version
        node = self._plate_with_keys(node, key_of, version)
        return write_plate(node, path, version=version, overwrite=overwrite,
                           storage_options=storage_options, writer=writer, **write_kwargs)

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

    @validating
    def write_series(self, path, *, version: Optional[str] = None, overwrite: bool = False,
                     ome_xml: Optional[str] = None, storage_options: Optional[dict] = None,
                     writer: Optional[Callable] = None, document: Optional[str] = None,
                     **write_kwargs) -> Dict[str, str]:
        """Write every entry as one series of a bioformats2raw layout (``<path>/0``,
        ``<path>/1``, ..., in key order; a collection read from that layout keeps its
        series paths), then the root metadata and the ``OME`` group with its OME-XML.
        Returns ``{series path: path}``. `ome_xml` is stored as ``OME/METADATA.ome.xml``
        (default: the source layout's when it still fits, else a minimal one). See
        :func:`ome_zarr_pyramid.core.ngff_series.write_series`.

        A series holding several images (keys with one series value that differ in a
        tile, view, ... index) is flattened: each image is written as a series of its
        own, in key order, with one ``UserWarning`` per call.

        ``version="rfc8"``: RFC 8 replaces the layout with a collection of the series
        images, so that is written (see `write_collection`); `ome_xml` is not used."""
        import warnings

        from ome_zarr_pyramid.core.ngff_series import LAYOUT_KEY, write_series
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
            from ome_zarr_pyramid.core import rfc8_keys
            from ome_zarr_pyramid.core.rfc8 import write_rfc8
            return write_rfc8(rfc8_keys.annotate(self, node, key_of), path,
                              document=document, overwrite=overwrite,
                              storage_options=storage_options, writer=writer, **write_kwargs)
        node = self._series_with_keys(node, keys, version)
        return write_series(node, path, version=version, overwrite=overwrite, ome_xml=ome_xml,
                            storage_options=storage_options, writer=writer, **write_kwargs)

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
                         writer: Optional[Callable] = None, **write_kwargs) -> Dict[str, str]:
        """Write the collection as NGFF RFC 8 (``version="rfc8"``, the only one: a 0.4 /
        0.5 plate or series is `write_plate` / `write_series`): each image as an RFC 8
        multiscale under `path`, then the collection document in `path`'s ``zarr.json``,
        or in the standalone `document` (``.json``). The tree as read or built
        (plates, wells, nested collections, attributes) is written as it is; a
        collection without one is written as a flat collection of its images, in key
        order. Returns ``{image path: written path}``. See
        :func:`ome_zarr_pyramid.core.rfc8.write_rfc8`."""
        from ome_zarr_pyramid.core.nodes import CollectionNode, MultiscaleNode
        from ome_zarr_pyramid.core.rfc8 import write_rfc8, tree_entries

        if version != "rfc8":
            raise ValueError(f"write_collection writes version='rfc8'; NGFF {version} plates "
                             f"and series are write_plate / write_series")
        from ome_zarr_pyramid.core import rfc8_keys

        tree = self.tree
        key_of = self._key_of_nodes()
        if isinstance(tree, MultiscaleNode):
            node = CollectionNode(tree.name, [tree])
        elif isinstance(tree, CollectionNode):
            held = {id(v) for v in self._pyramids.values()}
            images = [n for _, n in (_tree_entries(tree) if self.plates else tree_entries(tree))]
            if any(id(n) not in held and id(getattr(n, "_pyramid", None)) not in held
                   and getattr(n, "_alias_of", None) is None for n in images):
                raise ValueError("this collection is a selection of its tree; write it from "
                                 "the whole collection")
            node = tree
        else:
            node = CollectionNode(str(self.name or "collection"))
            for i, key in enumerate(sorted(self._pyramids)):
                child = node.add(MultiscaleNode(str(i), self[key]))
                key_of[id(child)] = key
        return write_rfc8(rfc8_keys.annotate(self, node, key_of),
                          path, document=document, overwrite=overwrite,
                          storage_options=storage_options, writer=writer, **write_kwargs)

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
        """The entries whose index matches every given value: ``select(view=1)``. A name
        the collection's keys do not have raises ``KeyError`` naming the ones they have;
        a known name whose value matches nothing gives an empty collection. (A plate
        field's ``Field.select`` only filters.)"""
        known = self._index_names()
        unknown = [name for name in index if name not in known]
        if unknown:
            hint = ""
            # micro-reader 0.1.0 renamed the index; this hint goes together with
            # micro-reader's index guard (plan 3.24)
            if "scene" in unknown and SERIES_INDEX in known:
                hint = f"; 'scene' is now called {SERIES_INDEX!r} (micro-reader 0.1.0)"
            raise KeyError(f"select: the collection has no index {unknown[0]!r}; its "
                           f"indices are {known}{hint}")
        return self._derive([k for k in self._pyramids
                             if all(dict(k).get(name) == value for name, value in index.items())])

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
        places = self._plate_places()
        on_disk: Dict[int, List] = {}                  # storage_options id -> [(key, location)]
        options: Dict[int, Any] = {}
        for key in missing:
            value = self._pyramids[key]
            opener = getattr(value, "loader", None) if isinstance(value, MultiscaleNode) else None
            location = opener.path_of(value) if isinstance(opener, ImageOpener) and value.path else None
            if location is not None and not value.loaded:
                so = opener.storage_options
                on_disk.setdefault(id(so), []).append((key, location))
                options[id(so)] = so
            else:
                self._rows[key] = row_from_pyramid(self._open(value), path=location)
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
        columns = self.table.columns
        unknown = [n for n in names if n not in columns]
        if unknown:
            raise KeyError(f"no column {unknown[0]!r}; the columns are {list(columns)}")

    def filter(self, predicate: Optional[Callable] = None, /, **conditions) -> "PyramidCollection":
        """The images that pass every condition: a column name with a value (equal) or a
        function of the value (true), e.g. ``filter(dtype="uint16", unit_x="nanometer",
        levels=lambda n: n >= 3)``. `predicate`, if given, is a function of each image's
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
        ``group_by("acquisition")``, ``group_by("dtype", "levels")``."""
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

    def suffix(self, key, names: Optional[Dict[str, str]] = None) -> str:
        """The file-name suffix for an entry: one ``_<name><value>`` per varying index,
        e.g. ``_view1_illumination0``. `names` renames indices (``{"illumination":
        "illu"}``)."""
        names = names or {}
        key = _key(key) if isinstance(key, Mapping) else tuple(key)
        parts = [f"_{names.get(name, name)}{value}" for name, value in key
                 if name in self.varying]
        return "".join(parts)

    @validating
    def write(self, directory, stem: str, *, names: Optional[Dict[str, str]] = None,
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
