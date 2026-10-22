"""The coordinate-system graph and ``eubi:`` keys in NGFF RFC 8 output, and how a reader
takes them back (plan 9.10: 3.11, 3.28, 5.2, 5.4, 5.5-5.10, 5.15, 5.18, 5.21, 8.3, 8.6,
8.19, 8.35).

Writing (:func:`annotate`) works on a copy of the collection's tree, never the tree
itself:

- Every image node carries ``eubi:index`` (its key) and ``eubi:coordinate_system``; every
  collection ``eubi:kind``; the root ``eubi:version``. A plate whose fields hold several
  images has every field as a sub-collection (8.3); an unnamed plate keeps its original
  name as ``eubi:name`` (8.6).
- A component of the graph (3.6) with two or more images gets a ``scene`` attribute on
  its owner collection when 5.18 holds there: the owner holds that component only, every
  image under it is in it, and each image's axes equal its system's. The scene lists the
  shared system first, then the others (estimated last), and one edge per placed image
  into its system: a translation, or ``sequence[scale, translation]`` when the units
  differ (3.20). The images' levels then hold only their alignment (their position
  leaves them), and each image node carries the back-pointer ``eubi:scene``. The
  component's other edges go into the scene too, composed with the positions that left
  the levels; each attached label gets its image's edge (3.28).
- Otherwise (5.18 fails, or a component of one image) the single-image form: the
  position stays in the levels, the systems go under ``eubi:coordinate_systems`` and the
  other edges under ``eubi:coordinate_transformations``, with a warning.

Reading (:func:`graph_from_tree`) reverses it on a tree read from RFC 8: keys from
``eubi:index``, systems from ``eubi:coordinate_systems`` and the scenes, positions from the
scenes' edges (3.20: p = t / s), folded into the levels when an image opens; an image met
twice in the document is read once (3.11).
"""
from __future__ import annotations

import posixpath
import re
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ome_zarr_pyramid.core import eubi_keys as ek
from ome_zarr_pyramid.core.nodes import CollectionNode, MultiscaleNode, is_plate, is_well
from ome_zarr_pyramid.core.spatial import (
    CoordinateSystem, ExternalRef, PATTERN_SAFE, SYSTEM_KEY, SYSTEMS_KEY, Transformation,
    TransformError, compose, is_image_key, merge_systems, scale_translation, stamp_system,
    steps, system_from_json, system_to_json, transformation_from_json,
    transformation_to_json, unit_factor)

#: the image-local system every RFC 8 image ozp writes declares (its levels output to it)
PHYSICAL = "physical"
#: an image node's back-pointer to the collection whose scene places it (5.5)
SCENE_KEY = ek.SCENE_KEY
#: the original name of a node RFC 8 needed a generated one for (8.6)
NAME_KEY = "eubi:name"
#: marks, while reading, an edge end that is an attached label (3.28), not a graph node
_LABEL = "eubi:label"


def _is_derived(node) -> bool:
    return "labels" in node.attributes or "source" in node.attributes


# -- ids (8.35) -------------------------------------------------------------------------

def rfc8_id(sid: str, taken: set) -> str:
    """A system id as an RFC 8 id: characters outside ``[a-zA-Z0-9-_.]`` as ``_xHH_``,
    a numeric suffix where the document already uses it. Ids ozp makes pass unchanged."""
    escaped = re.sub(r"[^A-Za-z0-9_.-]", lambda m: "".join(f"_x{b:02X}_" for b in
                                                          m.group(0).encode("utf-8")), sid)
    candidate, n = escaped, 1
    while candidate in taken:
        n += 1
        candidate = f"{escaped}_{n}"
    taken.add(candidate)
    return candidate


# -- placeholders write_rfc8 resolves once the paths are known --------------------------

class _ImageRef:
    """An image's (or an attached label's) own system as an RFC 8 reference."""

    def __init__(self, node, label: Optional[str] = None):
        self.node, self.label = node, label

    def to_rfc8(self, where) -> dict:
        rel = (where.label[(id(self.node), self.label)] if self.label is not None
               else where.image[id(self.node)])
        return {"id": PHYSICAL, "path": {"type": "zarr", "path": f"./{rel}"}}


def _end(value, refs: Mapping, ids: Mapping, where) -> Optional[dict]:
    if value is None:
        return None
    if is_image_key(value):
        return refs[value].to_rfc8(where)
    if isinstance(value, ExternalRef):
        return {"id": value.name, "path": {"type": "zarr", "path": value.path}}
    return {"id": ids.get(value, value)}


class _Edges:
    """Edges (with the references of their image ends) as RFC 8 transformation objects."""

    def __init__(self, edges: List[Tuple[Transformation, Mapping]], ids: Mapping):
        self.edges, self.ids = edges, ids

    def to_rfc8(self, where) -> list:
        return [transformation_to_json(e, endpoint=lambda v, refs=refs: _end(v, refs, self.ids, where))
                for e, refs in self.edges]


class _Scene(_Edges):
    """A collection's ``scene`` attribute (RFC 8): its systems, then its edges."""

    def __init__(self, systems: List[dict], edges, ids):
        super().__init__(edges, ids)
        self.systems = systems

    def to_rfc8(self, where) -> dict:
        return {"coordinateSystems": self.systems, "coordinateTransformations": super().to_rfc8(where)}


class _BackPointer:
    """``eubi:scene`` on an image node: the owner collection's folder, relative to the
    image's own (5.5)."""

    def __init__(self, image, owner):
        self.image, self.owner = image, owner

    def to_rfc8(self, where) -> str:
        return posixpath.relpath(where.prefix.get(id(self.owner)) or ".", where.image[id(self.image)])


# -- writing ------------------------------------------------------------------------------

def _translation(values, key, sid) -> Transformation:
    return Transformation("translation", key, sid, {"translation": [float(v) for v in values]})


def _shifted(pyr, p):
    """`pyr` (a clone of its own) with `p` taken off every level's translation, and off
    its attached labels' (3.28): the levels keep only their alignment."""
    out = pyr._clone_metadata_only()
    for attr in ("acquisition", "source_index", "joined_index", "source_series"):
        if hasattr(pyr, attr):
            setattr(out, attr, getattr(pyr, attr))
    for path in out.meta.resolution_paths:
        t = out.meta.get_translation(path) or [0.0] * len(p)
        out.meta.update_translation(path, [float(a) - float(b) for a, b in zip(t, p)])
    labels = {}
    for name, label in (getattr(pyr, "_labels", {}) or {}).items():
        moved = label._clone_metadata_only()
        for path in moved.meta.resolution_paths:
            t = moved.meta.get_translation(path) or [0.0] * len(p)
            moved.meta.update_translation(path, [float(a) - float(b) for a, b in zip(t, p)])
        labels[name] = moved
    out._labels = labels
    return out


def _position_edge(pyr, key, cs) -> Optional[Transformation]:
    """The edge that places `key` in system `cs` (3.20, 5.5): its level-0 translation as a
    translation in the system's units, or ``sequence[scale, translation]`` when a unit
    differs; None when a unit cannot be converted."""
    p = pyr.meta.get_translation(pyr.meta.resolution_paths[0]) or [0.0] * len(pyr.meta.axis_order)
    units = list(pyr.meta.unit_list or []) + [None] * len(p)
    factors = []
    for ax, unit, axis in zip(pyr.meta.axis_order, units, cs.axes):
        f = unit_factor(unit, axis.unit) if axis.type in ("space", "time") else 1.0
        if f is None:
            return None
        factors.append(f)
    t = [f * v for f, v in zip(factors, p)]
    if all(f == 1.0 for f in factors):
        return _translation(t, key, cs.id)
    return Transformation("sequence", key, cs.id, {"transformations": [
        Transformation("scale", params={"scale": factors}),
        Transformation("translation", params={"translation": t})]})


def _scene_problems(c, owner, under, pyramids, systems, system_of) -> List[str]:
    """Why the component `c` cannot be a scene on `owner` (5.18); [] when it can."""
    problems = []
    entries = set(c.entries)
    stray = [k for k in under if k not in entries]
    if stray:
        problems.append(f"images under it outside the component ({[dict(k) for k in stray[:3]]})")
    for key in c.entries:
        sid = system_of.get(key)
        if sid is None:
            continue
        cs, pyr = systems[sid], pyramids[key]
        names = [a.name for a in cs.axes or ()]
        if names != list(pyr.meta.axis_order):
            problems.append(f"{dict(key)}: axes {list(pyr.meta.axis_order)}, its system "
                            f"{sid!r} {names} (5.18(c))")
        elif _position_edge(pyr, key, cs) is None:
            problems.append(f"{dict(key)}: units {pyr.meta.unit_list} do not convert to "
                            f"{sid!r}'s")
    for edge in c.edges:
        touches = [e for s in (edge,) + (steps(edge) if edge.type == "sequence" else ())
                   for e in (s.input, s.output) if is_image_key(e)]
        if touches and not edge.modelled:
            problems.append(f"a {edge.type} edge at an image (ozp composes only the modelled "
                            f"types with the position)")
    return problems


def annotate(coll, node, key_of: Mapping[int, Any], *, written: str = "rfc8"):
    """A copy of tree `node` (the part of ``coll.tree`` being written) carrying the
    ``eubi:`` keys and the scenes of RFC 8 (see the module docstring); `key_of` maps
    ``id`` of an image node, or of the pyramid it holds, to its key. Returns the copy;
    the collection and its tree are not changed."""
    out = node.copy()
    pairs = list(zip(node.walk(), out.walk()))
    copy_of = {id(o): c for o, c in pairs}
    parent: Dict[int, Any] = {}
    for _, c in pairs:
        for child in c.children if isinstance(c, CollectionNode) else ():
            parent[id(child)] = c
    for _, c in pairs:
        c.attributes = ek.strip({k: v for k, v in c.attributes.items()
                                 if k not in (NAME_KEY, "scene")})

    images: Dict[Any, Any] = {}               # key -> image node (copy)
    aliases = []                              # (copy, key of the image it repeats)
    for o, c in pairs:
        if not isinstance(o, MultiscaleNode) or _is_derived(o):
            continue
        first = getattr(o, "_alias_of", None)
        key = key_of.get(id(first if first is not None else o),
                         key_of.get(id(getattr(first if first is not None else o, "_pyramid", None))))
        if key is None:
            continue
        if first is not None or key in images:
            aliases.append((c, key))
        else:
            images[key] = c
    keys = list(images)
    sub = coll._derive(keys)
    pyramids = {k: coll.pyramid(k) for k in keys}
    systems = ek.filled_systems(sub, pyramids)
    system_of = sub.system_of
    for key, pyr in pyramids.items():
        pyr.meta.set_custom(ek.INDEX_KEY, ek.index_json(key))
        pyr.meta.set_custom(SCENE_KEY, None)
        if system_of.get(key) in systems:
            stamp_system(pyr, systems[system_of[key]])

    # the tree: kinds, the root's version, an unnamed plate's name, fields as collections
    out.attributes[ek.VERSION_KEY] = ek.EUBI_VERSION
    for c in [c for _, c in pairs]:
        if not isinstance(c, CollectionNode):
            continue
        kind = "plate" if is_plate(c) else "well" if is_well(c) else c.kind
        if kind is not None:
            c.attributes[ek.KIND_KEY] = kind
        if is_plate(c):
            original = c.ngff_fields.get("plate", {}).get("name")
            if isinstance(original, str) and original != c.name:
                c.attributes[NAME_KEY] = original
            _fields_as_collections(c, images)
    parent = {}
    for c in out.walk():
        for child in c.children if isinstance(c, CollectionNode) else ():
            parent[id(child)] = c

    taken = {n.id for n in out.walk() if n.id is not None}
    ids = {sid: rfc8_id(sid, taken) for sid in systems}
    refs = {k: _ImageRef(images[k]) for k in keys}
    for key in keys:
        images[key].attributes[ek.INDEX_KEY] = ek.index_json(key)
        images[key].attributes[ek.SYSTEM_KEY] = system_of.get(key)

    # the graph: one scene per owner that 5.18 allows, the single-image form elsewhere
    by_names = _paths_of(coll.tree)
    components = [c for c in sub.components(warn=False)]
    owners = [c.owner for c in components if c.owner is not None]
    fallback: List[Tuple[Any, list]] = []    # (group, problems) of components left eubi
    placed_systems, placed_edges = set(), set()
    for c in components:
        if c.owner is None:                  # one image: the single-image form
            continue
        owner = copy_of.get(id(by_names.get(c.owner)), out)
        under = [k for k in keys if _under(images[k], owner, parent)]
        problems = _scene_problems(c, owner, under, pyramids, systems, system_of)
        if owners.count(c.owner) > 1:
            problems.insert(0, "it holds several components (5.18(a))")
        if problems:
            fallback.append((owner, problems))
            continue
        edges = []
        positions = {}
        for key in c.entries:
            sid = system_of.get(key)
            if sid is not None:
                edge = _position_edge(pyramids[key], key, systems[sid])
                p = pyramids[key].meta.get_translation(pyramids[key].meta.resolution_paths[0])
                positions[key] = p or [0.0] * len(pyramids[key].meta.axis_order)
                edges.append((edge, {key: refs[key]}))
                for label in (getattr(pyramids[key], "_labels", {}) or {}):
                    edges.append((edge, {key: _ImageRef(images[key], label)}))
        for edge in c.edges:
            edges.append((_composed(edge, positions), refs))
        for key, p in positions.items():
            pyramids[key] = _shifted(pyramids[key], p)
            images[key].attributes[SCENE_KEY] = _BackPointer(images[key], owner)
        order = sorted(c.systems, key=lambda s: (systems[s].estimated,
                                                 s not in set(system_of.values()), s))
        owner.attributes["scene"] = _Scene([_system_rfc8(systems[s], ids[s]) for s in order],
                                           edges, ids)
        owner.attributes[SYSTEMS_KEY] = [_system_eubi(systems[s], ids[s], axes=False) for s in order]
        placed_systems |= c.systems
        placed_edges |= {id(e) for e in c.edges}

    # everything left: systems on the group listing their images (or owning them), edges
    # under eubi: with one warning
    held = []
    for c in components:
        if c.systems <= placed_systems and all(id(e) in placed_edges for e in c.edges):
            continue
        group = (copy_of.get(id(by_names.get(c.owner)), out) if c.owner is not None
                 else parent.get(id(images[c.entries[0]]), out))
        block = group.attributes.setdefault(SYSTEMS_KEY, [])
        block.extend(_system_eubi(systems[s], ids[s]) for s in sorted(c.systems))
        if c.edges:
            group.attributes.setdefault(ek.EDGES_KEY, _Edges([], ids)).edges.extend(
                (e, refs) for e in c.edges)
            held.append((group, len(c.edges)))
        placed_systems |= c.systems
    rest = [s for s in systems if s not in placed_systems]
    if rest:
        out.attributes.setdefault(SYSTEMS_KEY, []).extend(_system_eubi(systems[s], ids[s]) for s in rest)
    loose = [e for e in sub.coordinate_transformations
             if not any(id(e) in {id(x) for x in c.edges} for c in components)]
    if loose:
        out.attributes.setdefault(ek.EDGES_KEY, _Edges([], ids)).edges.extend((e, refs) for e in loose)
        held.append((out, len(loose)))
    for group, problems in fallback:
        warnings.warn(f"RFC 8: collection {group.name!r} is not written as a scene "
                      f"({'; '.join(problems)}); its images keep their positions in their "
                      f"own translations and its systems are under {SYSTEMS_KEY} (plan 5.18)",
                      UserWarning, stacklevel=3)
    if held:
        warnings.warn(f"RFC 8: {sum(n for _, n in held)} edge(s) have no scene to go into; "
                      f"they are kept under {ek.EDGES_KEY} on "
                      f"{', '.join(sorted({g.name for g, _ in held}))}", UserWarning, stacklevel=3)

    for key in keys:
        images[key].pyramid = pyramids[key]
    for c, key in aliases:                    # written as the image's own path (3.11)
        c.pyramid = pyramids[key]
        c._alias = True
    return out


def _paths_of(tree) -> Dict[Tuple[str, ...], Any]:
    """Every collection of `tree` by its path of names from the root (as
    :meth:`PyramidCollection.components` names owners)."""
    out: Dict[Tuple[str, ...], Any] = {}

    def walk(node, path):
        out[path] = node
        for child in node.nodes if isinstance(node, CollectionNode) else ():
            if isinstance(child, CollectionNode):
                walk(child, path + (child.name,))
    if isinstance(tree, CollectionNode):
        walk(tree, ())
    return out


def _under(node, owner, parent) -> bool:
    while node is not None:
        if node is owner:
            return True
        node = parent.get(id(node))
    return False


def _fields_as_collections(plate, images: Mapping) -> None:
    """8.3: a plate any of whose fields holds several images has every field as a
    sub-collection (``field<n>``), so a one-image field reads back with its index too."""
    from ome_zarr_pyramid.core.plates import _image_name
    key_of = {id(n): k for k, n in images.items()}
    wells = [w for w in plate.nodes if isinstance(w, CollectionNode)]
    if not any(isinstance(c, CollectionNode) for w in wells for c in w.nodes):
        return
    for well in wells:
        kept = []
        for child in list(well.nodes):
            key = key_of.get(id(child))
            if isinstance(child, MultiscaleNode) and key is not None and "field" in dict(key):
                f = dict(key)["field"]
                child.name = _image_name(key, ("plate", "well", "field"))
                field = CollectionNode(f"field{f}", [child], kind="field", index=f)
                field.attributes[ek.KIND_KEY] = "field"
                kept.append(field)
            else:
                kept.append(child)
        well._nodes = []
        for child in kept:
            well.add(child)


def _system_rfc8(cs: CoordinateSystem, rid: str) -> dict:
    out = {"id": rid}
    if cs.name and cs.name != cs.id:
        out["name"] = cs.name
    if cs.axes is not None:
        out["axes"] = system_to_json(cs)["axes"]
    return out


def _system_eubi(cs: CoordinateSystem, rid: str, *, axes: bool = True) -> dict:
    """5.28's entry, plus the RFC 8 id when 8.35 had to escape the verbatim one."""
    out = system_to_json(cs, axes=axes and cs.axes is not None)
    if rid != cs.id:
        out["eubi:id"] = rid
    return out


def _composed(edge: Transformation, positions: Mapping) -> Transformation:
    """`edge` for images whose position has left their levels (the scene form): starting
    at such an image, its position is applied first; ending at one, taken off last."""
    start, end = edge.input, edge.output
    if not edge.modelled or (start not in positions and end not in positions):
        return edge
    parts = []
    if start in positions:
        parts.append(_translation(positions[start], None, None))
    parts.append(Transformation(edge.type, None, None, edge.params, edge.name))
    if end in positions:
        parts.append(_translation([-v for v in positions[end]], None, None))
    ndim = len(positions.get(start) or positions.get(end))
    return compose(*parts, ndim=ndim, input=start, output=end)


# -- reading ------------------------------------------------------------------------------

def _location(node) -> Optional[str]:
    loader = getattr(node, "loader", None)
    if node.path is None or not hasattr(loader, "location"):
        return None
    return posixpath.normpath(str(loader.location(node)).replace("\\", "/"))


def _loaded_with(node, p) -> None:
    """Fold position `p` into the levels of `node`'s pyramid when it opens (3.20)."""
    loader = node.loader

    def load(n, loader=loader, p=tuple(p)):
        pyr = loader(n)
        for path in pyr.meta.resolution_paths:
            t = pyr.meta.get_translation(path) or [0.0] * len(p)
            pyr.meta.update_translation(path, [float(a) + float(b) for a, b in zip(t, p)])
        return pyr
    load.location = getattr(loader, "location", None)
    node.loader = load


def graph_from_tree(tree, entries: List[Tuple[Any, Any]], where: str):
    """(entries, coordinate_systems, system_of, edges) of an RFC 8 tree read (keys,
    positions, systems and edges, module docstring). `entries` are the default
    ``[(key, image node)]`` (the spec's numbering); an image met twice in the document
    keeps its first entry (3.11). Opens no image."""
    # one image, several parents: the first occurrence holds it
    seen: Dict[str, Any] = {}
    kept = []
    for key, node in entries:
        loc = _location(node)
        if loc is not None and loc in seen:
            first = seen[loc]
            node._alias_of = first
            node.loader = lambda _n, first=first: first.pyramid
            continue
        if loc is not None:
            seen[loc] = node
        kept.append((key, node))
    label_at: Dict[str, Any] = {}
    for node in tree.walk() if isinstance(tree, CollectionNode) else ():
        if isinstance(node, CollectionNode) and node.kind is None and \
                isinstance(node.attributes.get(ek.KIND_KEY), str):
            node.kind = node.attributes[ek.KIND_KEY]
        if is_plate(node) and isinstance(node.attributes.get(NAME_KEY), str):
            node.ngff_fields.setdefault("plate", {})["name"] = node.attributes[NAME_KEY]
        if isinstance(node, MultiscaleNode) and _is_derived(node) and _location(node):
            label_at[_location(node)] = node

    plate = bool(isinstance(tree, CollectionNode) and any(is_plate(n) for n in tree.walk()))
    out, system_of, key_at = [], {}, {}
    disagree = []
    for key, node in kept:
        index = node.attributes.get(ek.INDEX_KEY)
        if isinstance(index, Mapping):
            written = ek.key_from_json(index)
            if plate:                           # the spec's numbers win (5.11)
                spec, values = dict(key), dict(written)
                names = [n for n, _ in written]
                for n in ("well", "acquisition", "plate"):
                    if n in spec and n not in names:
                        names.insert(0, n)
                for n in ("plate", "acquisition", "well"):
                    if n in spec:
                        if n in values and values[n] != spec[n]:
                            disagree.append((dict(key), dict(written)))
                        values[n] = spec[n]
                written = tuple((n, values[n]) for n in names)
            key = written
        sid = node.attributes.get(ek.SYSTEM_KEY)
        if sid is not None:
            system_of[key] = str(sid)
        out.append((key, node))
        if _location(node) is not None:
            key_at[_location(node)] = key
    if disagree:
        warnings.warn(f"{where}: {len(disagree)} image(s) whose {ek.INDEX_KEY} disagrees with "
                      f"the plate's own numbering (e.g. {disagree[0][1]} for {disagree[0][0]}); "
                      f"the plate's is kept (plan 5.11)", UserWarning, stacklevel=3)

    # systems: eubi: definitions (verbatim ids, estimated) and the scenes' own
    systems: Dict[str, CoordinateSystem] = {}
    rid_to_sid: Dict[str, str] = {}
    scenes = []
    for node in tree.walk() if isinstance(tree, CollectionNode) else ():
        for obj in node.attributes.get(SYSTEMS_KEY) or []:
            cs = system_from_json(obj)
            rid_to_sid[str(obj.get("eubi:id", cs.id))] = cs.id
            systems[cs.id] = merge_systems(systems[cs.id], cs) if cs.id in systems else cs
        scene = node.attributes.get("scene")
        if isinstance(scene, Mapping):
            scenes.append((node, scene))
    for node, scene in scenes:
        for obj in scene.get("coordinateSystems") or []:
            rid = str(obj.get("id"))
            sid = rid_to_sid.get(rid, rid)
            cs = system_from_json({"name": sid, "axes": obj.get("axes"),
                                   **({"eubi:name": obj["name"]} if obj.get("name") else {})})
            if sid in systems:
                held = systems[sid]
                systems[sid] = (held if held.axes is not None else
                                CoordinateSystem(sid, held.name, cs.axes, held.estimated, held.extra))
            else:
                systems[sid] = cs

    def end(obj, base):
        if obj is None:
            return None
        if isinstance(obj, str):
            return rid_to_sid.get(obj, obj)
        rid = str(obj.get("id", obj.get("name")))
        path = obj.get("path")
        if path is None:
            return rid_to_sid.get(rid, rid)
        text = path.get("path") if isinstance(path, Mapping) else str(path)
        loc = posixpath.normpath(posixpath.join(base, str(text))) if base else str(text)
        if loc in key_at:
            return key_at[loc]
        if loc in label_at:
            return ExternalRef(loc, _LABEL)     # an attached label's edge (3.28)
        return ExternalRef(str(text), rid)

    # edges: positions out of the scene edges, the rest kept (composed back to memory)
    raw = []
    for node, scene in scenes:
        base = _base_of(node)
        for obj in scene.get("coordinateTransformations") or []:
            raw.append(_edge(obj, lambda o, b=base: end(o, b)))
    for node in tree.walk() if isinstance(tree, CollectionNode) else ():
        base = _base_of(node)
        for obj in node.attributes.get(ek.EDGES_KEY) or []:
            raw.append(_edge(obj, lambda o, b=base: end(o, b)))
    positions: Dict[Any, List[float]] = {}
    label_positions: Dict[str, List[float]] = {}
    edges = []
    for edge in raw:
        if edge is None:
            continue
        start = edge.input
        if isinstance(start, ExternalRef) and start.name == _LABEL:
            p = _position_of(edge)
            if p is not None:
                label_positions[start.path] = p
            continue                            # a label link is never an I5 edge (3.28)
        if is_image_key(start) and isinstance(edge.output, str) \
                and system_of.get(start) == edge.output and edge.modelled:
            p = _position_of(edge, systems.get(edge.output))
            if p is not None:
                positions[start] = p
                continue
        edges.append(edge)
    edges = [_uncomposed(e, positions) for e in edges]
    for key, node in out:
        if key in positions:
            _loaded_with(node, positions[key])
    for loc, p in label_positions.items():
        _loaded_with(label_at[loc], p)
    return out, list(systems.values()), system_of, edges


def _base_of(node) -> str:
    """The folder the references of `node`'s attributes are relative to: its document's."""
    for child in node.walk():
        loader = getattr(child, "loader", None)
        if hasattr(loader, "base"):
            return posixpath.normpath(str(loader.base).replace("\\", "/"))
        if hasattr(loader, "location") and child.path is not None and callable(loader.location):
            loc = str(loader.location(child)).replace("\\", "/")
            rel = posixpath.normpath(str(child.path.path))
            return posixpath.normpath(loc[: len(loc) - len(rel)].rstrip("/")) if loc.endswith(rel) else ""
    return ""


def _edge(obj, end):
    try:
        return transformation_from_json(obj, endpoint=end)
    except (TypeError, ValueError):
        return None


def _position_of(edge: Transformation, cs: Optional[CoordinateSystem] = None) -> Optional[List[float]]:
    """p = t / s of a position edge (3.20), in the image's own units; None when the edge
    is not a scale-and-translation."""
    ndim = None
    for step in (edge,) + (steps(edge) if edge.type == "sequence" else ()):
        for key in ("translation", "scale"):
            if key in step.params:
                ndim = len(step.params[key])
                break
        if ndim:
            break
    if ndim is None:
        return None
    try:
        st = scale_translation(edge, ndim)
    except TransformError:
        return None
    if st is None:
        return None
    s, t = st
    if np.any(s == 0):
        return None
    return [float(v) for v in t / s]


def _uncomposed(edge: Transformation, positions: Mapping) -> Transformation:
    """An edge read from a scene, for images whose position is folded back into their
    levels: the reverse of :func:`_composed`."""
    start, end = edge.input, edge.output
    if not edge.modelled or (start not in positions and end not in positions):
        return edge
    parts = []
    if start in positions:
        parts.append(_translation([-v for v in positions[start]], None, None))
    parts.append(Transformation(edge.type, None, None, edge.params, edge.name))
    if end in positions:
        parts.append(_translation(positions[end], None, None))
    ndim = len(positions.get(start) or positions.get(end))
    return compose(*parts, ndim=ndim, input=start, output=end)
