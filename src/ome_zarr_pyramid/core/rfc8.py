"""NGFF RFC 8 JSON ("Collections and Extensibility") read into the collection model of
:mod:`ome_zarr_pyramid.core.nodes`, and written from it (``version="rfc8"``).

RFC 8 is a draft (no NGFF version assigned yet), so ozp reads generously and writes
plainly:

- **Reading.** A document is the ``ome`` key of a zarr group's ``zarr.json``, or of a
  standalone ``.json`` file. Every node behind a path opens on first use, relative to
  the document that names it: a collection's children from another JSON file or zarr
  group; an image from an RFC 8 multiscale node (singlescale levels with scale /
  translation transforms and a coordinate system: inline, in the levels' own
  ``zarr.json``, or in the multiscale's attributes) or from an ordinary NGFF 0.4 / 0.5
  image (as the RFC's own example files do). References are read in each spelling
  the RFC and its example files use (``{"id": ...}``, ``{"ref": ...}``, a plain string).
- **Writing.** One document holds the collection tree (in the root group's
  ``zarr.json``, or in a standalone ``.json``); each image is a zarr v3 group whose
  ``ome`` is an RFC 8 multiscale node with its levels inline, a ``physical`` coordinate
  system, and a scale (+ translation) per level. The version written is ``"0.x"``, the
  RFC's own placeholder. What RFC 8 has no place for (an image's ``omero``, custom
  metadata blocks, a root's keys that are not RFC 8) is written beside ``ome`` under
  ozp's custom-metadata rule, and read back from there. An image's attached label
  images are written as sibling multiscale nodes with a ``labels`` attribute naming
  their source, the RFC's model of labels.
"""
from __future__ import annotations

import copy
import json
import os
import posixpath
import re
from typing import Any, Dict, List, Optional, Tuple

from ome_zarr_pyramid.core.collection import SERIES_INDEX
from ome_zarr_pyramid.core.nodes import (
    CollectionNode, ID_PATTERN, InvalidCollection, LabelsAttribute, MultiscaleNode, Node,
    NodePath, Ref, attributes_to_json, is_plate, is_well, ref_id)

#: the ``version`` written in RFC 8 metadata: the RFC's placeholder (Bugra, 2026-10-17)
RFC8_VERSION = "0.x"
#: the node types RFC 8 defines
NODE_TYPES = ("collection", "multiscale", "singlescale")
#: the ``ome`` keys of a document's root that are not node fields
_ROOT_ONLY = ("version",)
#: attributes of a root node that stay in the RFC 8 document, not beside ``ome``
_RFC8_ATTRIBUTES = ("plate", "well", "acquisition", "labels", "scene", "coordinateSystems",
                    "coordinateTransformations", "source")


def is_rfc8(ome: Any) -> bool:
    """Whether an ``ome`` block is an RFC 8 node (it has a node ``type``)."""
    return isinstance(ome, dict) and isinstance(ome.get("type"), str)


# -- locations ------------------------------------------------------------------------

def _is_url(location: str) -> bool:
    return "://" in location and not location.startswith("file://")


def _local(location: str) -> str:
    if location.startswith("file://"):
        rest = location[len("file://"):]
        if re.match(r"^/[A-Za-z]:", rest):           # file:///C:/...
            rest = rest[1:]
        return rest
    return location


def resolve(base: str, path: str) -> str:
    """`path` (relative, ``file://`` or a URL) against `base`, the location of the
    document that names it: a zarr group, or the folder of a JSON file."""
    if "://" in path:
        return _local(path) if path.startswith("file://") else path
    if _is_url(base):
        scheme, rest = base.split("://", 1)
        return f"{scheme}://" + posixpath.normpath(posixpath.join(rest, path))
    return os.path.normpath(os.path.join(_local(base), path))


def _read_json(location: str, storage_options: Optional[dict]) -> Dict[str, Any]:
    if _is_url(location):
        import obstore
        folder, name = location.rsplit("/", 1)
        store = obstore.store.from_url(folder, **(storage_options or {}))
        return json.loads(bytes(obstore.get(store, name).bytes()))
    with open(_local(location), "rb") as f:
        return json.loads(f.read())


def _group(location: str, storage_options: Optional[dict], mode: str = "r",
           zarr_format: Optional[int] = None):
    from ome_zarr_pyramid.core.io import _open_store_group
    return _open_store_group(location, zarr_format, mode=mode, storage_options=storage_options)


def _document(location: str, kind: str, storage_options: Optional[dict]
              ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """(the ``ome`` node, the other attributes, the base for its paths) of the document
    at `location`: a JSON file (`kind` ``"json"``) or a zarr group or array (``"zarr"``)."""
    if kind == "json":
        doc = _read_json(location, storage_options)
        base = location.rsplit("/", 1)[0] if _is_url(location) else os.path.dirname(_local(location))
        attrs = doc
        if "ome" not in doc and isinstance(doc.get("attributes"), dict):
            attrs = doc["attributes"]          # a zarr.json-shaped file (webknossos example)
    else:
        import zarr
        try:
            attrs = _group(location, storage_options).attrs.asdict()
        except (zarr.errors.ContainsArrayError, ValueError, FileNotFoundError, KeyError):
            attrs = _array(location, storage_options).attrs.asdict()
        base = location
    ome = attrs.get("ome")
    if not isinstance(ome, dict):
        raise InvalidCollection(f"{location}: no 'ome' metadata")
    return ome, {k: v for k, v in attrs.items() if k != "ome"}, base


def _array(location: str, storage_options: Optional[dict]):
    import zarr
    if _is_url(location):
        parent, name = location.rstrip("/").rsplit("/", 1)
        return _group(parent, storage_options)[name]
    return zarr.open_array(_local(location), mode="r")


# -- reading --------------------------------------------------------------------------

def _systems_in(doc: Any) -> Dict[str, Any]:
    """Every coordinate system a document defines, by id (ids are unique in it)."""
    found: Dict[str, Any] = {}
    if isinstance(doc, dict):
        for system in doc.get("coordinateSystems") or []:
            if isinstance(system, dict) and isinstance(system.get("id"), str):
                found[system["id"]] = system
        for value in doc.values():
            if isinstance(value, (dict, list)):
                found.update(_systems_in(value))
    elif isinstance(doc, list):
        for value in doc:
            found.update(_systems_in(value))
    return found


class Loader:
    """Opens what a node of an RFC 8 document points to, relative to that document.
    `systems` are the document's coordinate systems, for inline images whose levels
    output to one defined elsewhere in it (on their collection)."""

    def __init__(self, base: str, storage_options: Optional[dict] = None,
                 systems: Optional[Dict[str, Any]] = None):
        self.base = base
        self.storage_options = storage_options
        self.systems = systems or {}

    def location(self, node: Node) -> str:
        return resolve(self.base, node.path.path)

    def __call__(self, node: Node):
        if isinstance(node, MultiscaleNode):
            if node.path is None:                    # its levels are inline, here
                return pyramid_from_rfc8(node.to_json(), self.base, self.storage_options,
                                         systems=self.systems)
            return open_image(self.location(node), self.storage_options)
        if isinstance(node, CollectionNode):
            location = self.location(node)
            ome, _, base = _document(location, node.path.type, self.storage_options)
            child = Node.from_json(_strip(ome), loader=Loader(base, self.storage_options,
                                                              _systems_in(ome)))
            if not isinstance(child, CollectionNode):
                raise InvalidCollection(f"{location}: a {child.type} node, where the "
                                        f"collection {node.name!r} points")
            # the attributes the parent document gives the node complement or override
            # the node's own (RFC 8, Collection node)
            for key, value in child.attributes.items():
                node.attributes.setdefault(key, value)
            return child.nodes
        raise InvalidCollection(f"{node!r}: ozp opens collections and multiscales")


def _strip(ome: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in ome.items() if k not in _ROOT_ONLY}


def open_image(location: str, storage_options: Optional[dict] = None):
    """The ``Pyramid`` at `location`: an RFC 8 multiscale node, or an NGFF 0.4 / 0.5
    image."""
    from ome_zarr_pyramid.core.io import IO
    try:
        ome, others, base = _document(location, "zarr", storage_options)
    except InvalidCollection:
        ome = None
    if is_rfc8(ome):
        if ome["type"] != "multiscale":
            raise InvalidCollection(f"{location}: a {ome['type']} node, where an image is "
                                    f"expected")
        pyr = pyramid_from_rfc8(ome, base, storage_options)
        _restore_side_blocks(pyr, others)
        return pyr
    return IO().read_pyramid(location, storage_options=storage_options)


def read_rfc8(path, *, storage_options: Optional[dict] = None) -> Node:
    """The node tree of the RFC 8 document at `path`: a zarr group, or a standalone
    ``.json`` file. Nodes behind paths open on first use. The root's attributes include
    the keys found beside ``ome`` in a zarr group (custom metadata blocks)."""
    location = str(path)
    kind = "json" if location.lower().endswith(".json") else "zarr"
    ome, others, base = _document(location, kind, storage_options)
    if not is_rfc8(ome):
        raise InvalidCollection(f"{location}: its 'ome' metadata is not an RFC 8 node")
    stored_type = ome["type"]
    if stored_type == "singlescale":
        # one level on its own (an array with its node): an image of that one level
        level = {k: v for k, v in _strip(ome).items() if k != "path"}
        level["path"] = {"type": "zarr", "path": "."}
        ome = {"type": "multiscale", "name": ome.get("name") or "image", "nodes": [level],
               "version": ome.get("version")}
    node = Node.from_json(_strip(ome), loader=Loader(base, storage_options, _systems_in(ome)))
    if stored_type == "multiscale" and kind == "zarr" and node.path is None:
        # an image on its own: opened with the blocks beside its `ome`
        node.loader = lambda _node: open_image(location, storage_options)
    elif kind == "zarr":
        node.attributes.update(copy.deepcopy(others))
    node.ngff_fields["version"] = ome.get("version")
    node.ngff_fields["rfc8"] = True
    return node


def _axes_of(system: Dict[str, Any]) -> List[Dict[str, Any]]:
    axes = system.get("axes") or []
    return [a if isinstance(a, dict) else {"name": str(a)} for a in axes]


def pyramid_from_rfc8(ome: Dict[str, Any], base: str, storage_options: Optional[dict] = None,
                      systems: Optional[Dict[str, Any]] = None):
    """A ``Pyramid`` from an RFC 8 multiscale node whose paths are relative to `base`.

    Each singlescale level's transforms are taken from its own attributes, from the
    multiscale's ``coordinateTransformations`` naming it as input, or from the level
    array's own ``zarr.json``; its axes from the coordinate system they output to."""
    from ome_zarr_pyramid.core.pyramid import Pyramid

    attrs = ome.get("attributes") or {}
    systems = {**(systems or {}),
               **{s.get("id"): s for s in attrs.get("coordinateSystems") or [] if isinstance(s, dict)}}
    shared = attrs.get("coordinateTransformations") or []
    levels = [n for n in ome.get("nodes") or [] if isinstance(n, dict)
              and n.get("type") == "singlescale"]
    if not levels:
        raise InvalidCollection(f"multiscale {ome.get('name')!r} lists no singlescale levels")
    arrays, scales, translations, axes = [], [], [], None
    for level in levels:
        if "path" not in level:
            raise InvalidCollection(f"multiscale {ome.get('name')!r}: level "
                                    f"{level.get('name')!r} has no path")
        location = resolve(base, level["path"]["path"])
        arr = _array(location, storage_options)
        own = dict(level.get("attributes") or {})
        transforms = own.get("coordinateTransformations")
        level_systems = {**systems, **{s.get("id"): s for s in own.get("coordinateSystems") or []
                                       if isinstance(s, dict)}}
        if not transforms:
            transforms = [t for t in shared if ref_id(t.get("input")) == level.get("id")]
        if not transforms:                       # distributed: in the array's zarr.json
            stored = (arr.attrs.asdict().get("ome") or {}).get("attributes") or {}
            transforms = stored.get("coordinateTransformations") or []
            level_systems.update({s.get("id"): s for s in stored.get("coordinateSystems") or []
                                  if isinstance(s, dict)})
        scale = translation = None
        output = None
        for t in transforms:
            output = output or ref_id(t.get("output"))
            if t.get("type") == "scale":
                scale = [float(v) for v in t["scale"]]
            elif t.get("type") == "translation":
                translation = [float(v) for v in t["translation"]]
        if scale is None:
            raise InvalidCollection(f"multiscale {ome.get('name')!r}: level "
                                    f"{level.get('name')!r} has no scale transformation")
        system = level_systems.get(output) or (next(iter(level_systems.values()))
                                                if len(level_systems) == 1 else None)
        if system is None:
            raise InvalidCollection(f"multiscale {ome.get('name')!r}: no coordinate system "
                                    f"{output!r} for level {level.get('name')!r}")
        level_axes = _axes_of(system)
        if axes is None:
            axes = level_axes
        elif [a.get("name") for a in axes] != [a.get("name") for a in level_axes]:
            raise InvalidCollection(f"multiscale {ome.get('name')!r}: its levels have "
                                    f"different axes")
        arrays.append(arr)
        scales.append(scale)
        translations.append(translation)
    names = [str(a.get("name")) for a in axes]
    if any(len(n) != 1 or n not in "tczyx" for n in names) or len(set(names)) != len(names):
        raise InvalidCollection(f"multiscale {ome.get('name')!r}: axes {names}; ozp images "
                                f"have distinct axes from t, c, z, y, x")
    pyr = Pyramid().from_arrays(arrays, axis_order="".join(names),
                                unit_list=[a.get("unit") for a in axes], scales=scales,
                                version="0.5", name=ome.get("name"))
    for path, translation in zip(pyr.meta.resolution_paths, translations):
        if translation is not None:
            pyr.meta.update_translation(path, translation)
    return pyr


def _restore_side_blocks(pyr, others: Dict[str, Any]) -> None:
    """Put an RFC 8 image's ``omero`` and custom blocks (stored beside ``ome``) back on
    its pyramid."""
    if not others:
        return
    meta = pyr.meta.metadata
    if isinstance(others.get("omero"), dict):
        meta["omero"] = copy.deepcopy(others["omero"])
    for key, value in others.items():
        if key != "omero":
            pyr.meta.set_custom(key, value)
    pyr.meta.mark_changed()


# -- writing --------------------------------------------------------------------------

def multiscale_json(pyr, *, name: str, id: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(the RFC 8 multiscale node of `pyr`, the blocks written beside it): its levels
    inline as singlescale nodes at ``./<level path>``, each with a scale (and a
    translation, when the level has one) into the ``physical`` coordinate system."""
    meta = pyr.meta
    ms = (meta.metadata.get("multiscales") or [{}])[0]
    axes = copy.deepcopy(ms.get("axes") or [])
    datasets = {str(d.get("path")): d for d in ms.get("datasets") or []}
    levels = []
    for path in meta.resolution_paths:
        transforms = [{"type": "scale", "scale": [float(v) for v in meta.get_scale(path)],
                       "input": {"id": str(path)}, "output": {"id": "physical"}}]
        stored = datasets.get(str(path), {}).get("coordinateTransformations") or []
        if any(t.get("type") == "translation" for t in stored):
            transforms.append({"type": "translation",
                               "translation": [float(v) for v in meta.get_translation(path)],
                               "input": {"id": str(path)}, "output": {"id": "physical"}})
        levels.append({"type": "singlescale", "id": str(path), "name": str(path),
                       "path": {"type": "zarr", "path": f"./{path}"},
                       "attributes": {"coordinateTransformations": transforms}})
    node: Dict[str, Any] = {"type": "multiscale"}
    if id is not None:
        node["id"] = id
    node.update({"name": name, "nodes": levels,
                 "attributes": {"coordinateSystems": [{"id": "physical", "axes": axes}]}})
    beside: Dict[str, Any] = {}
    if isinstance(meta.metadata.get("omero"), dict):
        beside["omero"] = copy.deepcopy(meta.metadata["omero"])
    beside.update(copy.deepcopy(meta.custom_blocks))
    return node, beside


def _label_attributes(label) -> List[Dict[str, Any]]:
    """RFC 8 ``labelAttributes`` from a label image's ``image-label`` colours /
    properties (``label-value`` -> ``labelValue``, ``rgba`` -> ``color``)."""
    il = getattr(label.meta, "image_label", None) or {}
    by_value: Dict[Any, Dict[str, Any]] = {}
    for entry in il.get("colors") or []:
        value = entry.get("label-value")
        out = by_value.setdefault(value, {"labelValue": value})
        if "rgba" in entry:
            out["color"] = list(entry["rgba"])
    for entry in il.get("properties") or []:
        value = entry.get("label-value")
        out = by_value.setdefault(value, {"labelValue": value})
        out.update({k: v for k, v in entry.items() if k != "label-value"})
    return list(by_value.values())


def _safe_id(text: str, taken: set) -> str:
    stem = re.sub(r"[^A-Za-z0-9\-_.]", "_", text) or "node"
    candidate, n = stem, 1
    while candidate in taken:
        n += 1
        candidate = f"{stem}_{n}"
    taken.add(candidate)
    return candidate


def write_rfc8(node: CollectionNode, path, *, document: Optional[str] = None,
               overwrite: bool = False, storage_options: Optional[dict] = None,
               writer=None, **write_kwargs) -> Dict[str, str]:
    """Write a collection tree as RFC 8 (version ``"0.x"``): every image in turn as an
    RFC 8 multiscale group under `path` (``io.write_pyramid`` or `writer`, with
    `write_kwargs`), then the collection document: in `path`'s ``zarr.json``, or, with
    `document`, in that standalone ``.json`` file (its image paths relative to it).
    Returns ``{image path in the collection: written path}``.

    Images keep the path they have in the tree (joined through their collections),
    else a well's ``<row>/<column>/<n>`` in a plate or ``<n>`` in any collection, where
    collections and images share one counter (``0`` a series collection, ``1`` the
    image next to it). Two images at one path, or an image group inside another, are
    refused before anything is written. An existing `path` is refused unless
    `overwrite` (which clears it first); the collection document is written last."""
    from zarr.errors import ContainsArrayError, ContainsGroupError

    from ome_zarr_pyramid.core.io import IO, _store_join
    from ome_zarr_pyramid.core.ngff_plate import well_path

    if not isinstance(node, CollectionNode):
        raise ValueError(f"{node!r}: write_rfc8 writes a collection")
    taken = {n.id for n in node.walk() if n.id is not None}
    jobs: List[Tuple[str, Any, str, Optional[str]]] = []   # (rel path, pyramid, name, id)
    label_of: Dict[str, str] = {}                          # label path -> its image's path

    def own_part(parent: CollectionNode, child: Node) -> Optional[str]:
        if isinstance(child, CollectionNode):
            if is_plate(parent) and is_well(child):
                return well_path(parent.attributes["plate"], child)
            if child.path is not None and child.path.type == "zarr":
                return child.path.path
            return None
        if isinstance(child, MultiscaleNode) and child.path is not None:
            return child.path.path
        return None

    def parts_of(coll: CollectionNode) -> Dict[int, str]:
        """Where each collection and image child of `coll` goes in it: its own path,
        else the next number no sibling takes (one counter for collections and images,
        so a collection's folder never holds an image group of the same name)."""
        children = [c for c in coll.nodes if isinstance(c, (CollectionNode, MultiscaleNode))]
        own = {id(c): own_part(coll, c) for c in children}
        used = {posixpath.normpath(p).split("/")[0] for p in own.values() if p is not None}
        out, n = {}, 0
        for child in children:
            if own[id(child)] is not None:
                out[id(child)] = own[id(child)]
                continue
            while str(n) in used:
                n += 1
            out[id(child)] = str(n)
            used.add(str(n))
        return out

    def join(prefix: str, part: str) -> str:
        return posixpath.normpath(posixpath.join(prefix, part)) if prefix else posixpath.normpath(part)

    def build(coll: CollectionNode, prefix: str, root: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": "collection"}
        if coll.id is not None:
            out["id"] = coll.id
        out["name"] = coll.name
        children = []
        parts = parts_of(coll)
        for child in coll.nodes:
            if isinstance(child, CollectionNode):
                children.append(build(child, join(prefix, parts[id(child)]), False))
            elif isinstance(child, MultiscaleNode):
                rel = join(prefix, parts[id(child)])
                pyr = child.pyramid
                labels = dict(getattr(pyr, "_labels", {}) or {})
                ident = child.id if child.id is not None or not labels else _safe_id(child.name, taken)
                jobs.append((rel, pyr, child.name, ident))
                entry: Dict[str, Any] = {"type": "multiscale"}
                if ident is not None:
                    entry["id"] = ident
                entry.update({"name": child.name, "path": {"type": "zarr", "path": f"./{rel}"}})
                if child.attributes:
                    entry["attributes"] = attributes_to_json(child.attributes)
                entry.update(copy.deepcopy(child.extra))
                children.append(entry)
                for label_name, label in labels.items():
                    label_rel = f"{rel}/labels/{label_name}"
                    jobs.append((label_rel, label, label_name, None))
                    label_of[label_rel] = rel
                    attributes: Dict[str, Any] = {"labels": LabelsAttribute(
                        [Ref(ident)], _label_attributes(label)).to_json()}
                    if "acquisition" in child.attributes:
                        attributes["acquisition"] = child.attributes["acquisition"].to_json()
                    children.append({"type": "multiscale",
                                     "name": f"{child.name} {label_name}",
                                     "path": {"type": "zarr", "path": f"./{label_rel}"},
                                     "attributes": attributes})
            else:
                children.append(child.to_json())
        out["nodes"] = children
        attributes = {k: v for k, v in coll.attributes.items()
                      if not (root and k not in _RFC8_ATTRIBUTES and ":" not in k)}
        attributes.pop("bioformats2raw.layout", None)     # replaced by the collection itself
        if attributes:
            out["attributes"] = attributes_to_json(attributes)
        out.update(copy.deepcopy(coll.extra))
        return out

    doc = build(node, "", True)
    beside = {k: copy.deepcopy(v.to_json() if hasattr(v, "to_json") else v)
              for k, v in node.attributes.items()
              if k not in _RFC8_ATTRIBUTES and ":" not in k and k != "bioformats2raw.layout"}
    _check_paths(node.name, jobs, label_of)

    try:
        root = _group(str(path), storage_options, mode="w" if overwrite else "w-", zarr_format=3)
    except (ContainsGroupError, ContainsArrayError, FileExistsError) as exc:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it") from exc
    write = writer or IO().write_pyramid
    written = {}
    for rel, pyr, name, ident in jobs:
        target = str(_store_join(str(path), rel))
        version_05 = pyr if str(pyr.meta.version) == "0.5" else pyr.to_version("0.5")
        bare = version_05._clone_metadata_only()
        bare._labels = {}                        # labels are nodes of their own here
        kw = dict(write_kwargs)
        if storage_options is not None:
            kw["storage_options"] = storage_options
        write(bare, target, **kw)
        ome, side = multiscale_json(version_05, name=name, id=ident)
        group = _group(target, storage_options, mode="a", zarr_format=3)
        group.attrs.put({"ome": {"version": RFC8_VERSION, **ome}, **side})
        written[rel] = target
    if document is None:
        root.attrs.put({"ome": {"version": RFC8_VERSION, **_relative(doc, "")}, **beside})
    else:
        doc_dir = os.path.dirname(os.path.abspath(_local(str(document))))
        offset = os.path.relpath(os.path.abspath(_local(str(path))), doc_dir).replace(os.sep, "/")
        with open(_local(str(document)), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"ome": {"version": RFC8_VERSION, **_relative(doc, offset)}, **beside},
                      f, indent=2)
    return written


def _check_paths(name: str, jobs, label_of: Dict[str, str]) -> None:
    """Refuse, before anything is written, two images at one path and an image group
    inside another image group. A label sits in its own image's ``labels/`` group
    (`label_of`: a label's path -> its image's path)."""
    names = [rel for rel, _, _, _ in jobs]
    if len(set(names)) != len(names):
        raise ValueError(f"collection {name!r}: two images would be written at the same "
                         f"path ({sorted({n for n in names if names.count(n) > 1})})")
    images = set(names)
    for rel in names:
        parts = rel.split("/")
        for i in range(1, len(parts)):
            outer = "/".join(parts[:i])
            if outer in images and label_of.get(rel) != outer:
                raise ValueError(f"collection {name!r}: image {rel!r} would be written "
                                 f"inside the image group {outer!r}")


def _relative(doc: Dict[str, Any], offset: str) -> Dict[str, Any]:
    """`doc` with its image paths made relative to a document `offset` away from the
    images' root (``""``: the document is in the root group)."""
    if not offset:
        return doc
    out = copy.deepcopy(doc)

    def walk(node):
        for child in node.get("nodes") or []:
            path = child.get("path")
            if isinstance(path, dict) and path.get("type") == "zarr" and path["path"].startswith("./"):
                path["path"] = "./" + posixpath.normpath(posixpath.join(offset, path["path"][2:]))
            walk(child)
    walk(out)
    return out


def tree_entries(tree) -> List[Tuple[Tuple, MultiscaleNode]]:
    """The flat keys of a collection that is not a plate: ``(("series", i),)`` for each
    image in tree order (as micro-reader keys a file's series), label and other derived
    images (a ``labels`` or ``source`` attribute) left out. The tree is kept as read: a
    nested collection's images are numbered on, in tree order."""
    out = []

    def walk(coll):
        for child in coll.nodes:
            if isinstance(child, CollectionNode):
                walk(child)
            elif isinstance(child, MultiscaleNode) and "labels" not in child.attributes \
                    and "source" not in child.attributes:
                out.append((((SERIES_INDEX, len(out)),), child))
    if isinstance(tree, MultiscaleNode):
        return [((), tree)]
    walk(tree)
    return out
