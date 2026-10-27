"""NGFF 0.6 multiscales (plan "NGFF 0.6 and RFC 8: one spatial model", 5.4, 5.6, 5.17,
3.23, 3.28; spec "OME-Zarr 0.6", sections "multiscales", "coordinateTransformations").

ozp holds every image in one form in memory, whatever version it was read from: the
multiscale's ``axes`` and, per level, a ``scale`` and an optional ``translation``. 0.6
writes the same facts differently:

- the axes live in a coordinate system of the multiscale (``coordinateSystems``), the
  image's *intrinsic* one: the system every level's transformation outputs to;
- a level holds ONE transformation: ``scale``, ``identity`` or
  ``sequence[scale, translation]``, from ``{"path": <level>}`` to ``{"name": <intrinsic>}``;
- the multiscale may hold further systems and transformations among them, and links
  to the intrinsic systems of its label images (``labels/<name>``).

:func:`from_06` turns a 0.6 multiscale into ozp's form and :func:`to_06` back. What
ozp's form has no place for is kept as read under :data:`STASH` inside the multiscale
dict, so a read-write round trip writes it back unchanged: the intrinsic system's name,
its other coordinate systems, its other transformations (3.23: a foreign image's inline
systems and the edges among them are no graph nodes; ozp writes them back inline) and
the label links that are not identities. Identity label links are dropped on reading
and written again, one per label (3.28). :func:`strip_stash` removes the stash; writing
another version than 0.6 calls it and keeps what the stash held under
:data:`FOREIGN_KEY`, with a warning (3.10: nothing is dropped without a word).

A level transformation 0.6 does not allow raises :class:`Unreadable06`: ozp refuses an
image it would misread (5.17)."""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: the name ozp gives the intrinsic system of an image it writes (5.4)
INTRINSIC = "physical"
#: where the parts of a 0.6 multiscale ozp's form has no place for wait to be written back
STASH = "eubi:ngff06"
#: the custom block (beside ``ome``) holding a stash's foreign parts in 0.4 / 0.5 output
FOREIGN_KEY = "eubi:ngff06"

#: the multiscale fields that :func:`from_06` turns into ozp's form
_CONVERTED = ("coordinateSystems", "coordinateTransformations", "datasets")


class Unreadable06(ValueError):
    """A 0.6 multiscale ozp cannot read without misreading it: a level transformation
    other than scale, identity or sequence[scale, translation], levels that output to
    different systems, or an intrinsic system that is not listed."""


def _is_label_end(end) -> bool:
    return isinstance(end, Mapping) and isinstance(end.get("path"), str) and \
        end["path"].strip("/").startswith("labels/")


def _level(ds: Mapping[str, Any], n: int, where: str) -> Tuple[str, List[dict]]:
    """(output system name, ozp's transformations) of one 0.6 dataset."""
    cts = ds.get("coordinateTransformations")
    if not isinstance(cts, list) or len(cts) != 1 or not isinstance(cts[0], Mapping):
        raise Unreadable06(f"{where}: level {ds.get('path')!r} must hold exactly one "
                           f"transformation (scale, identity or sequence[scale, "
                           f"translation]); it holds {cts!r}")
    t = cts[0]
    out = t.get("output")
    name = out.get("name") if isinstance(out, Mapping) else out
    if not isinstance(name, str) or not name:
        raise Unreadable06(f"{where}: level {ds.get('path')!r}: its transformation names "
                           f"no output coordinate system")
    kind = t.get("type")
    if kind == "scale":
        return name, [{"type": "scale", "scale": [float(v) for v in t["scale"]]}]
    if kind == "identity":
        return name, [{"type": "scale", "scale": [1.0] * n}]
    if kind == "sequence":
        parts = t.get("transformations") or []
        if len(parts) == 2 and parts[0].get("type") == "scale" \
                and parts[1].get("type") == "translation":
            return name, [{"type": "scale", "scale": [float(v) for v in parts[0]["scale"]]},
                          {"type": "translation",
                           "translation": [float(v) for v in parts[1]["translation"]]}]
    raise Unreadable06(f"{where}: level {ds.get('path')!r}: a {kind!r} transformation "
                       f"{t!r}; NGFF 0.6 allows only scale, identity or sequence[scale, "
                       f"translation] here")


def from_06(ms: Mapping[str, Any], where: str = "multiscale") -> Dict[str, Any]:
    """ozp's form of the 0.6 multiscale `ms` (a new dict; `ms` is not changed)."""
    datasets = ms.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise Unreadable06(f"{where}: a 0.6 multiscale needs at least one dataset")
    systems = [s for s in ms.get("coordinateSystems") or [] if isinstance(s, Mapping)]
    by_name = {s.get("name"): s for s in systems}
    first = datasets[0].get("coordinateTransformations") or [{}]
    out0 = first[0].get("output") if isinstance(first[0], Mapping) else None
    intrinsic = out0.get("name") if isinstance(out0, Mapping) else out0
    if intrinsic not in by_name:
        raise Unreadable06(f"{where}: the levels output to {intrinsic!r}, which is not one of "
                           f"the multiscale's coordinateSystems {list(by_name)}")
    axes = copy.deepcopy(by_name[intrinsic].get("axes") or [])
    new_ds = []
    for ds in datasets:
        name, cts = _level(ds, len(axes), where)
        if name != intrinsic:
            raise Unreadable06(f"{where}: level {ds.get('path')!r} outputs to {name!r}, level "
                               f"{datasets[0].get('path')!r} to {intrinsic!r}; every level must "
                               f"output to the same system")
        new_ds.append({**{k: copy.deepcopy(v) for k, v in ds.items()
                          if k != "coordinateTransformations"},
                       "coordinateTransformations": cts})
    stash: Dict[str, Any] = {"intrinsic": intrinsic}
    others = [copy.deepcopy(s) for s in systems if s.get("name") != intrinsic]
    if others:
        stash["coordinateSystems"] = others
    edges, links = [], {}
    for t in ms.get("coordinateTransformations") or []:
        if not isinstance(t, Mapping):
            continue
        label_end = next((e for e in (t.get("input"), t.get("output")) if _is_label_end(e)), None)
        if label_end is not None:
            if t.get("type") != "identity":     # identities are written again (3.28)
                links[label_end["path"].strip("/")] = copy.deepcopy(t)
            continue
        edges.append(copy.deepcopy(t))
    if edges:
        stash["coordinateTransformations"] = edges
    if links:
        stash["label_links"] = links
    out = {k: copy.deepcopy(v) for k, v in ms.items() if k not in _CONVERTED}
    out["axes"] = axes
    out["datasets"] = new_ds
    out[STASH] = stash
    return out


def _dataset_06(ds: Mapping[str, Any], intrinsic: str) -> Dict[str, Any]:
    cts = ds.get("coordinateTransformations") or []
    scale = next((t["scale"] for t in cts if t.get("type") == "scale"), None)
    translation = next((t["translation"] for t in cts if t.get("type") == "translation"), None)
    path = str(ds.get("path"))
    ends = {"input": {"path": path}, "output": {"name": intrinsic}}
    if scale is None:
        raise ValueError(f"level {path!r} has no scale")
    if translation is None:
        t = {"type": "scale", "scale": list(scale), **ends}
    else:                                       # one sequence per level (5.4)
        t = {"type": "sequence", **ends,
             "transformations": [{"type": "scale", "scale": list(scale)},
                                 {"type": "translation", "translation": list(translation)}]}
    return {**{k: copy.deepcopy(v) for k, v in ds.items() if k != "coordinateTransformations"},
            "coordinateTransformations": [t]}


def to_06(ms: Mapping[str, Any], labels=None) -> Dict[str, Any]:
    """The 0.6 multiscale of `ms` in ozp's form (with its stash, if read from 0.6).
    `labels`: the image's label images, ``{name: the label's intrinsic system name}`` (or
    a list of names, each with ozp's :data:`INTRINSIC`); each gets an identity link from
    the intrinsic system to the label's (3.28), unless the stash holds another link for
    it."""
    if labels is not None and not isinstance(labels, Mapping):
        labels = {name: INTRINSIC for name in labels}
    stash = ms.get(STASH) or {}
    intrinsic = stash.get("intrinsic") or INTRINSIC
    out = {k: copy.deepcopy(v) for k, v in ms.items()
           if k not in ("axes", "datasets", "version", STASH)}
    out["coordinateSystems"] = [{"name": intrinsic, "axes": copy.deepcopy(ms.get("axes") or [])},
                                *copy.deepcopy(stash.get("coordinateSystems") or [])]
    out["datasets"] = [_dataset_06(ds, intrinsic) for ds in ms.get("datasets") or []]
    cts = copy.deepcopy(stash.get("coordinateTransformations") or [])
    links = stash.get("label_links") or {}
    for name, label_intrinsic in (labels or {}).items():
        path = f"labels/{name}"
        if path in links:
            cts.append(copy.deepcopy(links[path]))
        else:
            cts.append({"type": "identity", "input": {"name": intrinsic},
                        "output": {"name": label_intrinsic or INTRINSIC, "path": path}})
    if cts:
        out["coordinateTransformations"] = cts
    # the downscale method (5.23; strict_image requires it): the one ozp made the levels
    # with, else the one read, else unknown (vendor levels, a single level)
    if stash.get("type"):
        out["type"] = stash["type"]
    from ome_zarr_pyramid.core.validation import FILL_DERIVED
    if FILL_DERIVED.get():                   # write_pyramid(fill_derived=False): as given
        out.setdefault("type", "unknown")
        out.setdefault("metadata", {})
    return out


def intrinsic_of(ms: Mapping[str, Any]) -> str:
    """The name the intrinsic system of `ms` (ozp's form) is written with in 0.6."""
    return (ms.get(STASH) or {}).get("intrinsic") or INTRINSIC


def strip_stash(ms: Dict[str, Any], version: str, where: str = "") -> Optional[Dict[str, Any]]:
    """Remove the stash from `ms` (in place) before writing `version` (not 0.6). Returns
    the foreign parts it held (systems, transformations, label links), for the caller to
    keep under :data:`FOREIGN_KEY`, with a warning; None when it held none."""
    stash = ms.pop(STASH, None) or {}
    foreign = {k: v for k, v in stash.items() if k not in ("intrinsic", "type")}
    if not foreign:
        return None
    warnings.warn(f"{where or 'image'}: NGFF {version} has no place for the 0.6 coordinate "
                  f"systems and transformations read with this image "
                  f"({', '.join(sorted(foreign))}); they are kept under {FOREIGN_KEY!r} beside "
                  f"'ome' (plan 3.10). Write version='0.6' to keep them in place",
                  UserWarning, stacklevel=4)
    return copy.deepcopy(foreign)


def ozp_multiscales(attrs: Mapping[str, Any], where: str = "") -> Tuple[Optional[str], List[dict]]:
    """(declared version, multiscales in ozp's form) of an image group's raw attributes,
    for readers that look at the JSON without opening the image (``query``)."""
    ome = attrs.get("ome") if isinstance(attrs.get("ome"), Mapping) else attrs
    mss = ome.get("multiscales") or []
    if isinstance(mss, Mapping):
        mss = [mss]
    version = ome.get("version") or (mss[0].get("version") if mss and isinstance(mss[0], Mapping)
                                     else None)
    if str(version) == "0.6":
        return "0.6", [from_06(ms, where) for ms in mss]
    return (str(version) if version is not None else None), [dict(ms) for ms in mss]


# -- scenes (reading) ------------------------------------------------------------------

def scene_of(attrs: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The 0.6 ``scene`` object of a group: from its raw attributes (``ome.scene``) or
    from a node's attributes, where the readers keep NGFF keys (``scene``)."""
    ome = attrs.get("ome")
    if isinstance(ome, Mapping) and isinstance(ome.get("scene"), Mapping):
        return ome["scene"]
    scene = attrs.get("scene")
    return scene if isinstance(scene, Mapping) else None


def graph_from_scenes(groups: Mapping[str, Mapping[str, Any]], key_of_path: Mapping[str, Any],
                      systems: Mapping[str, Any], system_of: Mapping[Any, str], where: str):
    """What the 0.6 scenes on `groups` (``{group path: attributes}``, ``""`` the root) say
    (plan 3.20, 3.21, 5.5, 5.6, 5.13): ``(systems, system_of, positions, edges, names)``.

    - systems: the scenes' coordinate systems, a 0.6 name being the id (2.31), merged
      with the ``eubi:`` definitions in `systems` (those give the readable name and the
      estimated marker);
    - positions: ``{key: p}``, p in the image's own units, from each image's position
      edge: the edge between its own system and a non-estimated shared one that is a
      scale-and-translation with the unit factor as scale (3.20); an edge running system
      -> image is inverted first. The innermost scene comes first: `groups` are taken
      deepest path first;
    - system_of: `system_of` with each placed image's shared system (the edge wins over a
      differing ``eubi:coordinate_system``, with a warning, 3.20);
    - edges: every other edge, with the moved positions composed back out (3.21).

    An end ``{name, path}`` is the system *name* of the image at *path* (relative to the
    scene's group): an image of the set gives its key (its own system is taken as the
    one named; the fifth item returned, ``{key: name}``, lets the opener check it with
    :func:`fold_on_load`), anything else an :class:`ExternalRef` kept as read (3.29).
    Opens no image."""
    import posixpath

    from ome_zarr_pyramid.core.rfc8_keys import _position_of, _uncomposed
    from ome_zarr_pyramid.core.spatial import (
        CoordinateSystem, ExternalRef, TransformError, invert, is_image_key, merge_systems,
        system_from_json, transformation_from_json)

    out_systems = dict(systems)
    placed = dict(system_of)
    names: Dict[Any, str] = {}
    raw = []
    for group in sorted(groups, key=lambda g: -len([p for p in g.split("/") if p])):
        scene = scene_of(groups[group])
        if scene is None:
            continue
        for obj in scene.get("coordinateSystems") or []:
            cs = system_from_json(obj)
            held = out_systems.get(cs.id)
            if held is None:
                out_systems[cs.id] = cs
            elif held.axes is None:
                out_systems[cs.id] = CoordinateSystem(cs.id, held.name, cs.axes, held.estimated,
                                                      held.extra)

        def end(obj, group=group):
            if not isinstance(obj, Mapping):
                return None
            name = str(obj.get("name"))
            path = obj.get("path")
            if path in (None, ""):
                return name
            loc = posixpath.normpath(posixpath.join(group, str(path)) if group else str(path))
            if loc in key_of_path:
                names[key_of_path[loc]] = name
                return key_of_path[loc]
            return ExternalRef(str(path), name)
        for obj in scene.get("coordinateTransformations") or []:
            try:
                raw.append(transformation_from_json(obj, endpoint=end))
            except (TypeError, ValueError) as err:
                warnings.warn(f"{where}: a scene transformation ozp cannot read is skipped: "
                              f"{obj!r} ({err})", UserWarning, stacklevel=3)
    positions: Dict[Any, List[float]] = {}
    moved: Dict[Any, str] = {}
    edges = []
    for edge in raw:
        a, b = edge.input, edge.output
        image, sid, forward = (a, b, True) if is_image_key(a) else (b, a, False)
        cs = out_systems.get(sid) if isinstance(sid, str) else None
        if is_image_key(image) and cs is not None and not cs.estimated and edge.modelled \
                and image not in positions:
            e = edge
            if not forward:
                try:
                    n = len(cs.axes) if cs.axes else None
                    e = invert(edge, n) if n else None
                except TransformError:
                    e = None
            p = _position_of(e) if e is not None else None
            if p is not None:
                positions[image] = p
                moved[image] = sid
                continue
        edges.append(edge)
    differ = [(k, placed[k], s) for k, s in moved.items() if k in placed and placed[k] != s]
    if differ:
        k, was, now = differ[0]
        warnings.warn(f"{where}: {len(differ)} image(s) carry an eubi:coordinate_system other "
                      f"than their scene edge's (e.g. {dict(k)}: {was!r}, edge {now!r}); the "
                      f"edge's is kept and the key is rewritten on the next write (plan 3.20)",
                      UserWarning, stacklevel=3)
    placed.update(moved)
    edges = [_uncomposed(e, positions) for e in edges]
    return list(out_systems.values()), placed, positions, edges, names


def check_intrinsic(pyr, expected: Optional[str], where: str) -> bool:
    """True when the image a scene edge named by system `expected` has that as its
    intrinsic system (ozp folds positions only into the intrinsic system); False, with
    a warning, when the name is one of its other systems."""
    if expected is None:
        return True
    ms = pyr.meta.multiscales if pyr.meta is not None else {}
    got = intrinsic_of(ms)
    if got == expected or str(pyr.meta.version) != "0.6":
        return True
    warnings.warn(f"{where}: the scene places system {expected!r} of this image, which is "
                  f"not its intrinsic system {got!r}; the position is not folded into its "
                  f"levels", UserWarning, stacklevel=3)
    return False


def fold_on_load(node, p, expected: Optional[str], where: str) -> None:
    """Fold position `p` into the levels of image `node` when it opens (3.20), after
    checking that the scene named its intrinsic system (:func:`check_intrinsic`)."""
    loader = node.loader

    def load(n, loader=loader, p=tuple(p)):
        pyr = loader(n)
        if check_intrinsic(pyr, expected, f"{where}: {n.name}"):
            for path in pyr.meta.resolution_paths:
                t = pyr.meta.get_translation(path) or [0.0] * len(p)
                pyr.meta.update_translation(path, [float(a) + float(b) for a, b in zip(t, p)])
        return pyr
    for attr in ("location", "base", "path_of"):
        if hasattr(loader, attr):
            setattr(load, attr, getattr(loader, attr))
    node.loader = load
