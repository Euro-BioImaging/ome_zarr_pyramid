"""The ``eubi:`` keys of NGFF 0.4 / 0.5 output (plan section 5, 5.7, 5.11, 5.28, 8.2, 8.4,
8.5): what an image's index, a collection's kind, the image list of a listing group and
the coordinate-system graph become in a version that has no field for them, and how a
reader takes them back.

Our keys sit beside ``ome`` (0.5) or at the top level (0.4), never inside ``ome``:

- ``eubi:version`` (the root): the version of this key schema, :data:`EUBI_VERSION`.
- ``eubi:index`` (every image group): the image's key, as a JSON object.
- ``eubi:images`` (a well, a bioformats2raw root): ``{path: {"index": ...,
  "coordinate_system": id or None}}``, one entry per image child, so a reader keys and
  places the images without opening them (3.18).
- ``eubi:kind`` (every collection group but the root): ``plate``, ``well``, ...
- ``eubi:coordinate_system`` / ``eubi:coordinate_systems`` (every placed image group: its
  system and its definition, 3.26) and, on the group owning a component (3.6), the
  component's systems.
- ``eubi:coordinate_transformations`` (the owner group): the edges 0.4 / 0.5 cannot hold,
  with a warning (8.5).

A position is the image's level-0 translation (the single-image form); 0.4 / 0.5 never
get a scene. Nothing here opens an image: the writer is handed the pyramids, the reader
reads the groups the plate, well and series readers read anyway.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ome_zarr_pyramid.core.spatial import (
    Axis, CoordinateSystem, ExternalRef, SYSTEM_KEY, SYSTEMS_KEY, is_image_key, merge_systems,
    stamp_system, steps, system_from_json, system_to_json, transformation_from_json,
    transformation_to_json)

#: the version of the ``eubi:`` key schema written (8.2); a reader meeting a higher one
#: warns once and passes the keys it does not know through
EUBI_VERSION = 1

VERSION_KEY = "eubi:version"
INDEX_KEY = "eubi:index"
IMAGES_KEY = "eubi:images"
KIND_KEY = "eubi:kind"
EDGES_KEY = "eubi:coordinate_transformations"
SCENE_KEY = "eubi:scene"

#: every key this module writes: a write replaces them all, so keys read from a store
#: never outlive the collection's state (5.11)
GRAPH_KEYS = (VERSION_KEY, INDEX_KEY, IMAGES_KEY, KIND_KEY, SYSTEM_KEY, SYSTEMS_KEY,
              EDGES_KEY, SCENE_KEY)

#: the order of axis types in a system the writer fills from several images (3.12)
_TYPE_ORDER = {"time": 0, "space": 2}


def index_json(key) -> Dict[str, int]:
    """An image key as the JSON object of ``eubi:index``."""
    return {str(name): int(value) for name, value in key}


def key_from_json(obj) -> Tuple[Tuple[str, int], ...]:
    """The image key an ``eubi:index`` object holds (``resolution`` left out)."""
    if not isinstance(obj, Mapping):
        raise ValueError(f"{INDEX_KEY} {obj!r} is not an object")
    return tuple((str(k), int(v)) for k, v in obj.items() if k != "resolution")


# -- the axes of a shared system (3.12) ----------------------------------------------

def fill_axes(cs: CoordinateSystem, pyramids: Sequence) -> CoordinateSystem:
    """`cs` with its axes filled from the images written into it (3.12), when it has none:
    the images' axes when every image has the same names in the same order, else their
    union ordered by type (time, channel and other types, space; ties by first
    occurrence). Units: the images' when all agree, else micrometer (space) or second
    (time). A system with axes, or with no image, is returned as it is."""
    if cs.axes is not None or not pyramids:
        return cs
    orders = [tuple(p.meta.axis_order) for p in pyramids]
    units: Dict[str, set] = {}
    seen: List[str] = []
    for p in pyramids:
        unit_list = list(p.meta.unit_list or []) + [None] * len(p.meta.axis_order)
        for name, unit in zip(p.meta.axis_order, unit_list):
            units.setdefault(name, set()).add(unit)
            if name not in seen:
                seen.append(name)
    if len(set(orders)) == 1:
        names = list(orders[0])
    else:
        names = sorted(seen, key=lambda n: (_TYPE_ORDER.get(Axis(n).type, 1), seen.index(n)))
    axes = []
    for name in names:
        kind = Axis(name).type
        found = units[name] - {None}
        if kind == "channel":
            unit = None
        elif len(found) == 1:
            unit = found.pop()
        else:
            unit = "second" if kind == "time" else "micrometer" if found else None
        axes.append(Axis(name, unit=unit))
    return CoordinateSystem(cs.id, cs.name, tuple(axes), cs.estimated, dict(cs.extra))


def filled_systems(coll, pyramids: Mapping) -> Dict[str, CoordinateSystem]:
    """`coll`'s systems, each with its axes filled (:func:`fill_axes`) from the images
    written into it: those placed in it and those an edge links to it (`pyramids`: the
    pyramids written, by key)."""
    members: Dict[str, List] = {}
    for key, sid in coll.system_of.items():
        members.setdefault(sid, []).append(key)
    for edge in coll.coordinate_transformations:
        ends = [e for s in (edge,) + (steps(edge) if edge.type == "sequence" else ())
                for e in (s.input, s.output)]
        for sid in (e for e in ends if isinstance(e, str)):
            members.setdefault(sid, []).extend(k for k in ends if is_image_key(k))
    return {sid: fill_axes(cs, [pyramids[k] for k in dict.fromkeys(members.get(sid, ()))
                                if k in pyramids])
            for sid, cs in coll.coordinate_systems.items()}


# -- writing --------------------------------------------------------------------------

def _owner(paths: Sequence[str], groups: Iterable[str]) -> str:
    """The deepest of `groups` (paths of collection groups, "" the root) holding every
    image path of `paths`."""
    best = ""
    for g in groups:
        if g and all(p.startswith(g + "/") for p in paths) and len(g) > len(best):
            best = g
    return best


def _relative(path: str, group: str) -> str:
    return path[len(group) + 1:] if group else path


def graph_attrs(coll, image_paths: Mapping, listing: Mapping, groups: Iterable[str],
                version: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[Any, Any]]:
    """The ``eubi:`` keys of a 0.4 / 0.5 layout of `coll` (plan 5.7, 5.5, 5.6, 5.21):
    ``({group path: {key: value}}, {image key: pyramid to write})``.

    `image_paths` maps each key to its image group's path from the root (``"A/1/0"``),
    `listing` each key to the path of the group that lists it (its well, or ``""``),
    `groups` names the collection groups (a component's owner is the deepest holding all
    its images; a component of one image is defined on its listing group, so a reader
    finds every system without opening an image). Each pyramid is ``coll.pyramid(key)``
    (stamped with its system, 3.26) with its ``eubi:index``. Edges 0.4 / 0.5 cannot hold
    go to their owner's ``eubi:coordinate_transformations``, with one warning."""
    groups = set(groups) | {""}
    attrs: Dict[str, Dict[str, Any]] = {"": {VERSION_KEY: EUBI_VERSION}}
    system_of = coll.system_of
    pyramids = {}
    for key in image_paths:
        pyr = coll.pyramid(key)
        pyr.meta.set_custom(INDEX_KEY, index_json(key))
        pyr.meta.set_custom(SCENE_KEY, None)
        pyramids[key] = pyr
    images: Dict[str, Dict[str, Any]] = {}
    for key, path in image_paths.items():
        where = listing[key]
        images.setdefault(where, {})[_relative(path, where)] = {
            "index": index_json(key), "coordinate_system": system_of.get(key)}
    for where, entries in images.items():
        attrs.setdefault(where, {})[IMAGES_KEY] = entries

    systems = filled_systems(coll, pyramids)
    for key, pyr in pyramids.items():          # the image's own copy, axes filled too
        if system_of.get(key) in systems:
            stamp_system(pyr, systems[system_of[key]])

    placed_systems, placed_edges = set(), set()
    held: List[Tuple[str, int]] = []
    components = []
    for c in coll.components(warn=False):
        paths = [image_paths[k] for k in c.entries if k in image_paths]
        if paths:
            components.append((c, paths, _owner(paths, groups) if len(paths) > 1
                               else listing[c.entries[0]]))
    owned: Dict[str, int] = {}
    for c, paths, owner in components:
        owned[owner] = owned.get(owner, 0) + 1
    failed: List[str] = []
    for c, paths, owner in components:
        block = attrs.setdefault(owner, {})
        if version == "0.6" and len(paths) > 1:
            problems = _scene_06(coll, c, owner, owned[owner], image_paths, systems, pyramids,
                                 block)
            if not problems:
                placed_systems |= c.systems
                placed_edges |= {id(e) for e in c.edges}
                continue
            failed.append(f"{owner or 'the root'}: {'; '.join(problems)}")
        block.setdefault(SYSTEMS_KEY, []).extend(
            system_to_json(systems[s]) for s in sorted(c.systems) if s in systems)
        placed_systems |= c.systems
        if c.edges:
            block.setdefault(EDGES_KEY, []).extend(
                _edge_json(e, owner, image_paths) for e in c.edges)
            placed_edges |= {id(e) for e in c.edges}
            held.append((owner, len(c.edges)))
    # systems no entry reaches (linked only to each other) and their edges: the root
    rest = [s for s in systems if s not in placed_systems]
    if rest:
        attrs[""].setdefault(SYSTEMS_KEY, []).extend(system_to_json(systems[s]) for s in rest)
    loose = [e for e in coll.coordinate_transformations if id(e) not in placed_edges]
    if loose:
        attrs[""].setdefault(EDGES_KEY, []).extend(_edge_json(e, "", image_paths) for e in loose)
        held.append(("", len(loose)))
    if failed:
        warnings.warn(f"NGFF 0.6: no scene on {len(failed)} group(s), whose images keep their "
                      f"positions in their own levels (the single-image form, plan 5.18): "
                      f"{' | '.join(failed)}", UserWarning, stacklevel=3)
    if held:
        n = sum(k for _, k in held)
        kinds = sorted({e.type for e in coll.coordinate_transformations})
        where = sorted({g or "the root" for g, _ in held})
        hint = ("a scene could hold them where plan 5.18 allows one" if version == "0.6"
                else "NGFF 0.6 or RFC 8 holds them in a scene")
        warnings.warn(f"NGFF {version} has no place for {n} edge(s) of the coordinate-system "
                      f"graph ({', '.join(kinds)}); they are kept under {EDGES_KEY} on "
                      f"{', '.join(where)} (plan 8.5); {hint}",
                      UserWarning, stacklevel=3)
    return attrs, pyramids


def _scene_06(coll, c, owner: str, n_owned: int, image_paths: Mapping, systems: Mapping,
              pyramids: Dict[Any, Any], block: Dict[str, Any]) -> List[str]:
    """Write component `c` as a 0.6 scene on `owner` (plan 5.5, 5.18, 3.20, 3.21, 9.17's
    rules), when 5.18 lets it: ``block`` gets the ``scene`` (the shared systems first,
    nominal before estimated, with no ``path``; one position edge per image; the other
    edges composed with the positions that left the levels) and the systems' ``eubi:``
    entries without axes (5.28); each image's pyramid in `pyramids` keeps only its
    alignment and gets the ``eubi:scene`` back-pointer. Returns the problems that stop
    it (nothing written then), [] when written."""
    import posixpath

    from ome_zarr_pyramid.core.ngff06 import intrinsic_of
    from ome_zarr_pyramid.core.rfc8_keys import _composed, _position_edge, _scene_problems, \
        _shifted

    under = [k for k, p in image_paths.items() if not owner or p.startswith(owner + "/")]
    problems = _scene_problems(c, owner, under, pyramids, systems, coll.system_of)
    if n_owned > 1:
        problems.insert(0, f"it owns {n_owned} unconnected parts of the graph (5.18(a))")
    if problems:
        return problems
    rel = {k: _relative(image_paths[k], owner) for k in c.entries}
    names = {k: intrinsic_of(pyramids[k].meta.multiscales) for k in c.entries}

    def end(value):
        if value is None:
            return None
        if is_image_key(value):
            return {"name": names[value], "path": rel[value]}
        if isinstance(value, ExternalRef):
            return {"name": value.name, "path": value.path}
        return {"name": value}

    shared = sorted((s for s in c.systems if s in systems),
                    key=lambda s: (systems[s].estimated, s))
    edges, positions = [], {}
    for key in c.entries:
        sid = coll.system_of.get(key)
        if sid is None:
            continue
        pyr = pyramids[key]
        positions[key] = list(pyr.meta.get_translation(pyr.meta.resolution_paths[0])
                              or [0.0] * len(pyr.meta.axis_order))
        edges.append(transformation_to_json(_position_edge(pyr, key, systems[sid]), endpoint=end))
    for edge in c.edges:
        edges.append(transformation_to_json(_composed(edge, positions), endpoint=end))
    scene = {
        "coordinateSystems": [{"name": s, "axes": system_to_json(systems[s])["axes"]}
                              for s in shared if systems[s].axes is not None],
        "coordinateTransformations": edges}
    # the semantic check (plan 5.23): every image below the scene attached, by its name
    from ome_zarr_pyramid.core.validation import check_scene, raise_for_metadata
    below = {rel[k]: (names[k], list(pyramids[k].meta.axis_order)) for k in c.entries}
    problems = check_scene(scene, below)
    if problems:
        raise_for_metadata(f"{owner or 'the root'} (scene)", problems, "0.6")
    block["scene"] = scene
    block.setdefault(SYSTEMS_KEY, []).extend(system_to_json(systems[s], axes=False)
                                             for s in shared)
    for key, p in positions.items():
        moved = _shifted(pyramids[key], p)
        moved.meta.set_custom(SCENE_KEY, posixpath.relpath(owner or ".", image_paths[key]))
        pyramids[key] = moved
    return []


def _edge_json(edge, owner: str, image_paths: Mapping) -> dict:
    """An edge as 5.28 JSON on the group at `owner`: an image end is its own system,
    ``{"name": "physical", "path": <image path relative to the owner>}``."""
    def end(value):
        if value is None:
            return None
        if is_image_key(value):
            return {"name": "physical", "path": _relative(image_paths[value], owner)}
        if isinstance(value, ExternalRef):
            return {"name": value.name, "path": value.path}
        return {"name": value}
    return transformation_to_json(edge, endpoint=end)


def strip(attributes: Mapping[str, Any]) -> Dict[str, Any]:
    """`attributes` without the keys of :data:`GRAPH_KEYS` and a 0.6 ``scene`` (read from
    a store; its positions are folded into the images, its other edges are in the graph,
    so a write puts the collection's state in their place)."""
    return {k: v for k, v in attributes.items() if k not in GRAPH_KEYS and k != "scene"}


# -- reading --------------------------------------------------------------------------

def check_version(attrs: Mapping[str, Any], where: str) -> None:
    """Warn once when a store's ``eubi:version`` is newer than this ozp knows (8.2)."""
    found = attrs.get(VERSION_KEY)
    if isinstance(found, int) and not isinstance(found, bool) and found > EUBI_VERSION:
        warnings.warn(f"{where}: eubi:version {found} is newer than this ozp reads "
                      f"({EUBI_VERSION}); keys it does not know are passed through",
                      UserWarning, stacklevel=3)


def graph_from_groups(groups: Mapping[str, Mapping[str, Any]], key_of_path: Mapping[str, Any],
                      where: str) -> Tuple[List[CoordinateSystem], list]:
    """The systems and edges written on the groups read (``{group path: attributes}``;
    ``""`` the root): ``eubi:coordinate_systems`` merged by id (one store, one space per
    id, 5.24), ``eubi:coordinate_transformations`` with their image ends resolved to keys
    through `key_of_path` (image group path -> key); an end resolving to no image of the
    set is an :class:`ExternalRef`."""
    systems: Dict[str, CoordinateSystem] = {}
    edges = []
    for group, attrs in groups.items():
        for obj in attrs.get(SYSTEMS_KEY) or []:
            cs = system_from_json(obj)
            systems[cs.id] = merge_systems(systems[cs.id], cs) if cs.id in systems else cs

        def end(obj, group=group):
            if obj is None:
                return None
            if isinstance(obj, str):
                return obj
            name = obj.get("name", obj.get("id"))
            if "path" in obj:
                path = "/".join(p for p in (group, str(obj["path"]).strip("/")) if p)
                key = key_of_path.get(path)
                return key if key is not None else ExternalRef(str(obj["path"]), str(name))
            return str(name)
        for obj in attrs.get(EDGES_KEY) or []:
            edges.append(transformation_from_json(obj, endpoint=end))
    return list(systems.values()), edges
