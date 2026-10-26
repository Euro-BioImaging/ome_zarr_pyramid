"""NGFF 0.4 / 0.5 plates (the ``plate`` and ``well`` metadata) into the collection model
of :mod:`ome_zarr_pyramid.core.nodes`, and back.

A plate becomes a collection with a ``plate`` attribute, each well a collection with a
``well`` attribute (behind the well's path, e.g. ``"B/3"``), each image of a well a
multiscale node behind its path (``"0"``), with an ``acquisition`` attribute when the
well names one. Rows and columns get their name as id; an acquisition's integer id
becomes a string id. Wells are named row name + column name (``"B3"``), as in
micro-reader.

Nothing is lost on the way: what the model has no field for is kept and written back.

- The group's other attributes (``barcode``, ``_creator``, ``bioformats2raw.layout``,
  ...) become node attributes, written back under ozp's custom-metadata rule: beside
  ``ome`` in 0.5 (NGFF's own keys inside it), at the top level in 0.4.
- Keys of the plate, well, row, column, acquisition or image JSON ozp does not know are
  kept with the item they belong to.
- The plate's ``name`` and ``field_count`` are kept as read (``ngff_fields``).

Reading checks the plate and refuses a broken one, naming every problem: an index out
of range, a well path or image path used twice, an acquisition id used twice, an image
whose acquisition the plate does not list, a listed well without a group.

The images are not opened when the plate is read: each image node gets a loader
(:class:`ImageOpener`) that reads it with ``io.read_pyramid`` on first use.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ome_zarr_pyramid.core.nodes import (
    READ_VERSIONS, CollectionNode, InvalidCollection, MultiscaleNode, Node, NodePath,
    PlateAttribute, PlateItem, Ref, TYPED_ATTRIBUTES, WellAttribute, check_read_version,
    nests_in_ome)

#: the NGFF versions whose plate layout this module reads and writes (the readers'
#: one list, ``nodes.READ_VERSIONS``)
VERSIONS = READ_VERSIONS
#: the version a plate or a series layout is written in when nothing else decides
DEFAULT_VERSION = "0.5"
#: the zarr format each NGFF version this module writes is stored in
_ZARR_FORMATS = {"0.4": 2, "0.5": 3, "0.6": 3}


def zarr_format_of(version) -> int:
    """The zarr format an NGFF `version` is stored in: 0.4 -> 2, 0.5 and 0.6 -> 3. Any
    other version raises (ozp writes 0.4, 0.5 and 0.6 only)."""
    try:
        return _ZARR_FORMATS[str(version)]
    except KeyError:
        raise ValueError(f"NGFF version {version!r}: ozp writes the versions "
                         f"{tuple(_ZARR_FORMATS)}") from None


class PlateFieldError(ValueError):
    """A field of a plate holds several images (tiles, views, parts, ...). An NGFF plate
    (0.4, 0.5 and 0.6) holds one image per field, so the plate cannot be written as it
    is; the message names the ways out."""


def check_fields(plate: str, wells) -> None:
    """Raise :class:`PlateFieldError` when a field of the plate holds two or more images.
    `wells`: ``(well name, [image index, ...])`` per well, an index being the image's
    ``{name: value}`` in its source (``None`` or one without ``field`` when unknown: a
    well built by hand without field numbers, whose images are each their own field).
    Images share a field when their containment indices (plate, acquisition, well,
    field, series) agree; the other indices (tile, view, part, ...) tell them apart."""
    from ome_zarr_pyramid.core.collection import CONTAINERS, _RESOLUTION

    for well, indices in wells:
        places: Dict[Tuple, List[Dict[str, Any]]] = {}
        for index in indices:
            if not index or "field" not in index:
                continue
            index = {k: v for k, v in index.items() if k != _RESOLUTION}
            place = tuple(sorted((k, v) for k, v in index.items() if k in CONTAINERS))
            places.setdefault(place, []).append(index)
        for place, members in places.items():
            if len({tuple(sorted(m.items())) for m in members}) < 2:
                continue                   # one image (or the same index twice: not a field's)
            names = sorted({k for m in members for k in m if k not in CONTAINERS
                            and len({m2.get(k) for m2 in members}) > 1})
            where = dict(place)
            acq = f" (acquisition {where['acquisition']})" if "acquisition" in where else ""
            raise PlateFieldError(
                f"plate {plate!r}: well {well!r}, field {where['field']}{acq} holds "
                f"{len(members)} images (they differ in {names}); an NGFF plate (0.4, 0.5 "
                f"and 0.6) holds one image per field. Ways out: (1) open the file with "
                f'tiles="mosaic" (micro-reader >= 0.1), so each field is one image; (2) write '
                f'with version="rfc8" (experimental), which keeps every image of the '
                f"field; (3) for an NGFF source, join the field's images with pyrops "
                f"first.")


def _index_of(image: MultiscaleNode) -> Optional[Dict[str, Any]]:
    """An image node's index in its source, when the tree says it: an ``eubi:index``
    attribute, else the ``source_index`` of a pyramid already held (``from_image``).
    An image behind a path that is not open yet has none (nothing is opened here)."""
    index = image.attributes.get("eubi:index")
    if isinstance(index, dict):
        return index
    pyr = image._pyramid
    return dict(getattr(pyr, "source_index", None) or {}) if pyr is not None else None


_PLATE_FIELDS = ("rows", "columns", "acquisitions", "wells", "name", "field_count", "version")
_ENTRY_FIELDS = ("path", "rowIndex", "columnIndex")
_WELL_FIELDS = ("images", "version")
_IMAGE_FIELDS = ("path", "acquisition")


def _ngff_keys() -> frozenset:
    from ome_zarr_pyramid.core.pyramid import NGFFMetadataHandler
    return NGFFMetadataHandler.NGFF_KEYS


def _split(attrs: Mapping[str, Any]) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
    """(version, NGFF keys, custom keys) of a group's attributes. 0.5 keeps NGFF keys
    inside ``ome``; a custom key found there is read as custom (and written beside
    ``ome``), the rule ozp applies to images."""
    attrs = copy.deepcopy(dict(attrs))
    keys = _ngff_keys()
    if "ome" in attrs and isinstance(attrs["ome"], dict):
        ome = attrs.pop("ome")
        version = ome.pop("version", None)
        ngff = {k: v for k, v in ome.items() if k in keys}
        custom = {k: v for k, v in ome.items() if k not in keys}
        custom.update(attrs)
        return (str(version) if version is not None else None), ngff, custom
    ngff = {k: v for k, v in attrs.items() if k in keys}
    custom = {k: v for k, v in attrs.items() if k not in keys}
    # 0.4 records its version inside the plate / well JSON; a group without one is 0.4
    inner = next((ngff[k] for k in ("plate", "well") if isinstance(ngff.get(k), dict)), {})
    return str(inner.get("version", "0.4")), ngff, custom


def _join(version: str, ngff: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    if nests_in_ome(version):
        return {"ome": {"version": version, **ngff}, **custom}
    return {**ngff, **custom}


def _attributes_out(node: Node, own: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """A node's attributes other than `own`, as (NGFF keys, custom keys) JSON."""
    keys = _ngff_keys()
    ngff, custom = {}, {}
    for key, value in node.attributes.items():
        if key == own:
            continue
        if key in TYPED_ATTRIBUTES:
            raise ValueError(f"{node!r}: the {key!r} attribute has no place in an NGFF "
                             f"0.4/0.5 {own}")
        value = value.to_json() if hasattr(value, "to_json") else copy.deepcopy(value)
        (ngff if key in keys else custom)[key] = value
    return ngff, custom


def _rest(data: Mapping[str, Any], known) -> Dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in data.items() if k not in known}


def _is_index(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# -- reading -----------------------------------------------------------------------

def plate_from_attrs(attrs: Mapping[str, Any],
                     wells: Mapping[str, Optional[Mapping[str, Any]]], *,
                     name: Optional[str] = None) -> CollectionNode:
    """The plate collection from a plate group's attributes and its wells' (``{well
    path: attributes}``; ``None`` for a well whose group is missing). `name` names a
    plate whose JSON has no (or an empty) name. Raises :class:`InvalidCollection`
    naming every problem found."""
    version, ngff, custom = _split(attrs)
    plate = ngff.pop("plate", None)
    if not isinstance(plate, dict):
        raise InvalidCollection("not an NGFF plate: the group has no 'plate' metadata")
    check_read_version(version, f"plate {name!r}" if name else "plate")
    problems: List[str] = []

    def items(key: str) -> List[PlateItem]:
        out = []
        for i, entry in enumerate(plate.get(key) or []):
            if not isinstance(entry, dict):
                problems.append(f"{key}[{i}] is not an object")
                continue
            if key == "acquisitions":
                ident = entry.get("id")
                if not _is_index(ident):
                    problems.append(f"acquisitions[{i}]: id {ident!r} is not an integer")
                    continue
                ident, label = str(ident), entry.get("name")
            else:
                ident = label = entry.get("name")
                if not isinstance(ident, str) or not ident:
                    problems.append(f"{key}[{i}] has no name")
                    continue
            try:
                out.append(PlateItem(ident, label, _rest(entry, ("id", "name"))))
            except ValueError as exc:
                problems.append(f"{key}[{i}]: {exc}")
        return out

    if "rows" not in plate or "columns" not in plate:
        problems.append("the plate lists no 'rows' or no 'columns'")
    attr = PlateAttribute(items("rows"), items("columns"), items("acquisitions"),
                          _rest(plate, _PLATE_FIELDS))
    raw_name = plate.get("name")
    node = CollectionNode(raw_name if isinstance(raw_name, str) and raw_name
                          else (name or "plate"),
                          attributes={"plate": attr, **ngff, **custom})
    node.ngff_fields = {"version": version, "node_name": node.name,
                        "plate": {k: copy.deepcopy(plate[k]) for k in ("name", "field_count")
                                  if k in plate}}

    seen_paths = set()
    for i, entry in enumerate(plate.get("wells") or []):
        where = f"wells[{i}]"
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            problems.append(f"{where} has no path")
            continue
        path = entry["path"]
        where = f"well {path!r}"
        if path in seen_paths:
            problems.append(f"{where} is listed twice")
            continue
        seen_paths.add(path)
        r, c = entry.get("rowIndex"), entry.get("columnIndex")
        bad = [f"{key} {value!r} is not an index into the plate's {len(axis)} {kind}"
               for key, value, axis, kind in (("rowIndex", r, attr.rows, "rows"),
                                              ("columnIndex", c, attr.columns, "columns"))
               if not _is_index(value) or not 0 <= value < len(axis)]
        if bad:
            problems.append(f"{where}: " + ", ".join(bad))
            continue
        row, col = attr.rows[r], attr.columns[c]
        well_attrs = wells.get(path)
        if well_attrs is None:
            problems.append(f"{where} is listed in the plate but has no group (or no "
                            f"metadata)")
            continue
        well = _well_from_attrs(well_attrs, path, f"{row.name}{col.name}",
                                WellAttribute(Ref(row.id), Ref(col.id)), problems)
        if well is None:
            continue
        well.ngff_fields["entry"] = _rest(entry, _ENTRY_FIELDS)
        try:
            node.add(well)
        except ValueError:
            problems.append(f"{where}: another well already sits at row {row.name!r}, "
                            f"column {col.name!r}")
    problems += node.problems()
    if problems:
        raise InvalidCollection(f"plate {node.name!r}: " + "; ".join(problems))
    return node


def _well_from_attrs(attrs, path: Optional[str], name: str, ref: Optional[WellAttribute],
                     problems: List[str]) -> Optional[CollectionNode]:
    where = f"well {path or name!r}"
    _, ngff, custom = _split(attrs)
    well = ngff.pop("well", None)
    if not isinstance(well, dict):
        problems.append(f"{where}: the group has no 'well' metadata")
        return None
    own = {"well": ref} if ref is not None else {}
    node = CollectionNode(name, [], path=NodePath("zarr", path) if path else None,
                          attributes={**own, **ngff, **custom})
    node.ngff_fields["well"] = _rest(well, _WELL_FIELDS)
    for j, image in enumerate(well.get("images") or []):
        if not isinstance(image, dict) or not isinstance(image.get("path"), str) \
                or not image["path"]:
            problems.append(f"{where}: images[{j}] has no path")
            continue
        attributes = {}
        if "acquisition" in image:
            if not _is_index(image["acquisition"]):
                problems.append(f"{where}: image {image['path']!r} has acquisition "
                                f"{image['acquisition']!r}, not an integer")
                continue
            attributes["acquisition"] = Ref(str(image["acquisition"]))
        field = MultiscaleNode(image["path"], path=NodePath("zarr", image["path"]),
                               attributes=attributes)
        field.ngff_fields["entry"] = _rest(image, _IMAGE_FIELDS)
        try:
            node.add(field)
        except ValueError:
            problems.append(f"{where}: image {image['path']!r} is listed twice")
    return node


def _stem(path: str) -> str:
    base = os.path.basename(str(path).rstrip("/\\"))
    for ext in (".ome.zarr", ".zarr"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return base


class ImageOpener:
    """The loader of an image node read from a plate or a well: reads the image at
    ``<base>/<node path>`` with ``io.read_pyramid`` (with `storage_options` for a URL)."""

    def __init__(self, base, storage_options: Optional[dict] = None):
        self.base = str(base)
        self.storage_options = storage_options

    def path_of(self, node: Node) -> str:
        from ome_zarr_pyramid.core.io import _store_join
        if node.path.path in ("", ".", "./"):           # the image at the base itself
            return self.base
        return str(_store_join(self.base, node.path.path))

    def __call__(self, node: Node):
        from ome_zarr_pyramid.core.io import IO
        # positions a 0.6 scene gives are folded by the collection reader
        return IO().read_pyramid(self.path_of(node), storage_options=self.storage_options,
                                 apply_scene=None)

    def __repr__(self) -> str:
        return f"ImageOpener({self.base!r})"


def _attach_openers(well: CollectionNode, base, storage_options) -> None:
    opener = ImageOpener(base, storage_options)
    for image in well.children:
        if isinstance(image, MultiscaleNode) and image.path is not None:
            image.loader = opener


def _group_attrs(root, paths: List[str]) -> List[Optional[Dict[str, Any]]]:
    """The attributes of the groups at `paths` under `root`, fetched together (one
    concurrent batch from the store, not one round trip per group: a 384-well plate
    reads in about the time of its JSON locally, and in parallel requests remotely).
    ``None`` for a path with no metadata."""
    import asyncio
    import json

    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    store, prefix = root.store_path.store, root.store_path.path
    name = "zarr.json" if root.metadata.zarr_format == 3 else ".zattrs"
    keys = ["/".join(p for p in (prefix, path.strip("/"), name) if p) for path in paths]

    async def fetch():
        proto = default_buffer_prototype()
        return await asyncio.gather(*(store.get(key, prototype=proto) for key in keys))

    out: List[Optional[Dict[str, Any]]] = []
    for buf in sync(fetch()) if keys else []:
        if buf is None:
            out.append(None)
            continue
        doc = json.loads(buf.to_bytes())
        out.append(doc.get("attributes", {}) if name == "zarr.json" else doc)
    return out


def _root(path, storage_options, root):
    if root is not None:                    # already open (PyramidCollection.from_zarr)
        return root
    from ome_zarr_pyramid.core.io import _open_store_group
    return _open_store_group(str(path), None, mode="r", storage_options=storage_options)


def read_plate(path, *, storage_options: Optional[dict] = None, _group=None
               ) -> CollectionNode:
    """Read an NGFF 0.4 or 0.5 plate's metadata (the plate and every well) into a plate
    collection. The images are opened on first use (``node.pyramid``). Raises
    :class:`InvalidCollection` naming every problem of a broken plate."""
    root = _root(path, storage_options, _group)
    attrs = root.attrs.asdict()
    version, ngff, _ = _split(attrs)
    if isinstance(ngff.get("plate"), dict):
        # from the plate's own attributes, before any well group is opened
        check_read_version(version, f"{path}: plate")
    plate = ngff.get("plate")
    well_paths = list(dict.fromkeys(
        entry["path"] for entry in ((plate.get("wells") or []) if isinstance(plate, dict) else [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)))
    wells = dict(zip(well_paths, _group_attrs(root, well_paths)))
    for well_path, well_attrs in wells.items():
        # a well declaring a version ozp does not read (0.6) is refused now, not when
        # one of its images is first used
        declared = _split(well_attrs)[0] if isinstance(well_attrs, Mapping) else None
        if declared is not None:
            check_read_version(declared, f"{path}: well {well_path!r}")
    plate = plate_from_attrs(attrs, wells, name=_stem(path))
    from ome_zarr_pyramid.core.io import _store_join
    for well in plate.children:
        _attach_openers(well, _store_join(str(path), well.path.path), storage_options)
    return plate


def well_from_attrs(attrs: Mapping[str, Any], *, name: str = "well") -> CollectionNode:
    """A well read on its own (not through its plate): a collection of its images. It
    has no ``well`` attribute, since only the plate says which row and column it is."""
    problems: List[str] = []
    version, _, _ = _split(attrs)
    check_read_version(version, f"well {name!r}")
    node = _well_from_attrs(attrs, None, name, None, problems)
    if problems:
        raise InvalidCollection("; ".join(problems))
    node.ngff_fields["version"] = version
    return node


def read_well(path, *, storage_options: Optional[dict] = None, _group=None
              ) -> CollectionNode:
    """Read an NGFF 0.4 or 0.5 well group on its own; its images are opened on first
    use."""
    root = _root(path, storage_options, _group)
    attrs = root.attrs.asdict()
    check_read_version(_split(attrs)[0], f"{path}: well")
    well = well_from_attrs(attrs, name=_stem(path) or "well")
    _attach_openers(well, str(path), storage_options)
    return well


# -- writing -----------------------------------------------------------------------

def plate_to_attrs(node: CollectionNode, version: Optional[str] = None
                   ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """(plate group attributes, {well path: well group attributes}) in NGFF `version`
    (``"0.4"`` or ``"0.5"``; default: the version the plate was read as, else 0.5).
    Refuses a tree that breaks a rule, or that 0.4/0.5 cannot hold (a well holding
    collections: the "tall" acquisition form)."""
    version = str(version or node.ngff_fields.get("version") or DEFAULT_VERSION)
    if version not in VERSIONS:
        raise ValueError(f"version {version!r}: NGFF plates are written as {VERSIONS}")
    attr = node.attributes.get("plate")
    if not isinstance(attr, PlateAttribute):
        raise ValueError(f"{node!r} is not a plate: it has no 'plate' attribute")
    node.validate()

    plate: Dict[str, Any] = {}
    if attr.acquisitions:
        acquisitions = []
        for a in attr.acquisitions:
            if not a.id.isdigit():
                raise ValueError(f"acquisition id {a.id!r}: NGFF {version} plates number "
                                 f"acquisitions with integers")
            out: Dict[str, Any] = {"id": int(a.id)}
            if a.name is not None:
                out["name"] = a.name
            acquisitions.append({**out, **copy.deepcopy(a.extra)})
        plate["acquisitions"] = acquisitions
    plate["columns"] = [{"name": c.name or c.id, **copy.deepcopy(c.extra)} for c in attr.columns]
    plate["rows"] = [{"name": r.name or r.id, **copy.deepcopy(r.extra)} for r in attr.rows]

    entries, wells, field_count = [], {}, 0
    for well in node.nodes:
        ref = well.attributes.get("well")
        if not isinstance(ref, WellAttribute):
            raise ValueError(f"{well!r} in plate {node.name!r} has no 'well' attribute")
        r, c = attr.index_of("rows", ref.row), attr.index_of("columns", ref.column)
        path = well_path(attr, well)
        if path in wells:
            raise ValueError(f"plate {node.name!r}: two wells have the path {path!r}")
        entries.append({"path": path, "rowIndex": r, "columnIndex": c,
                        **copy.deepcopy(well.ngff_fields.get("entry", {}))})
        wells[path] = _well_to_attrs(well, version)
        field_count = max(field_count, len(wells[path]["ome"]["well"]["images"]
                                           if nests_in_ome(version) else
                                           wells[path]["well"]["images"]))
    plate["wells"] = entries
    from ome_zarr_pyramid.core.validation import FILL_DERIVED
    if FILL_DERIVED.get():                   # write_plate(fill_derived=False): as read
        _fill_maximumfieldcount(plate, wells, version)

    read = node.ngff_fields.get("plate")
    renamed = node.name != node.ngff_fields.get("node_name")
    if read is None or renamed:                  # built here, or renamed since read
        plate["name"] = node.name
        plate["field_count"] = field_count
    else:                                        # as read
        if "name" in read:
            plate["name"] = copy.deepcopy(read["name"])
        if "field_count" in read:
            plate["field_count"] = field_count
    plate.update(copy.deepcopy(attr.extra))
    if version == "0.4":
        plate["version"] = version
    ngff, custom = _attributes_out(node, "plate")
    return _join(version, {**ngff, "plate": plate}, custom), wells


def _fill_maximumfieldcount(plate: Dict[str, Any], wells: Dict[str, Dict[str, Any]],
                            version: str) -> None:
    """Give each acquisition without one its ``maximumfieldcount`` (NGFF SHOULD; the strict
    plate schema requires it, decision 9.12-S): the most fields of that acquisition in one
    well, counted from the wells written. A value the source gives is kept. With several
    acquisitions only fields that name theirs count; with one, every field does."""
    acquisitions = plate.get("acquisitions") or []
    if not acquisitions:
        return
    most: Dict[Any, int] = {}
    for attrs in wells.values():
        images = (attrs["ome"] if nests_in_ome(version) else attrs)["well"]["images"]
        if len(acquisitions) == 1:
            per = {acquisitions[0]["id"]: len(images)}
        else:
            per = {}
            for image in images:
                if "acquisition" in image:
                    per[image["acquisition"]] = per.get(image["acquisition"], 0) + 1
        for ident, n in per.items():
            most[ident] = max(most.get(ident, 0), n)
    for a in acquisitions:
        if "maximumfieldcount" not in a and most.get(a["id"]):
            a["maximumfieldcount"] = most[a["id"]]


def well_path(attr: PlateAttribute, well: Node) -> str:
    """Where a well is written in its plate: its own path, else ``"<row>/<column>"``
    (NGFF's rule: the row and column names)."""
    if well.path is not None:
        return well.path.path
    ref = well.attributes["well"]
    row = attr.rows[attr.index_of("rows", ref.row)]
    col = attr.columns[attr.index_of("columns", ref.column)]
    return f"{row.name or row.id}/{col.name or col.id}"


def image_paths(well: CollectionNode) -> List[str]:
    """Where each image of a well is written in it: its own path, else the next free
    number (``"0"``, ``"1"``, ...; NGFF image paths are alphanumeric)."""
    used = {image.path.path for image in well.nodes if image.path is not None}
    out, n = [], 0
    for image in well.nodes:
        if image.path is not None:
            out.append(image.path.path)
            continue
        while str(n) in used:
            n += 1
        out.append(str(n))
        used.add(str(n))
    return out


def _well_to_attrs(well: Node, version: str) -> Dict[str, Any]:
    if not isinstance(well, CollectionNode):
        raise ValueError(f"{well!r}: a well must be a collection")
    images = []
    for image, path in zip(well.nodes, image_paths(well)):
        if not isinstance(image, MultiscaleNode):
            raise ValueError(f"well {well.name!r} holds {image!r}: an NGFF {version} well "
                             f"lists images only (the 'wide' form, one acquisition "
                             f"attribute per image)")
        if "labels" in image.attributes:
            raise ValueError(f"well {well.name!r} holds the label image {image!r}: in "
                             f"NGFF {version} labels sit in their image's 'labels' group")
        entry: Dict[str, Any] = {"path": path}
        acq = image.attributes.get("acquisition")
        if acq is not None:
            if not acq.id.isdigit():
                raise ValueError(f"{image!r}: acquisition id {acq.id!r} is not an integer")
            entry["acquisition"] = int(acq.id)
        entry.update(copy.deepcopy(image.ngff_fields.get("entry", {})))
        images.append(entry)
    body: Dict[str, Any] = {"images": images,
                            **copy.deepcopy(well.ngff_fields.get("well", {}))}
    if version == "0.4":
        body["version"] = version
    ngff, custom = _attributes_out(well, "well")
    return _join(version, {**ngff, "well": body}, custom)


def write_plate_metadata(node: CollectionNode, path, *, version: Optional[str] = None,
                         storage_options: Optional[dict] = None) -> Dict[str, Any]:
    """Write a plate's metadata: the plate group's attributes and every well group's
    (groups are created as needed; their existing attributes are replaced). The images
    are not written. Returns the plate group's attributes."""
    from ome_zarr_pyramid.core.io import _open_store_group

    version = str(version or node.ngff_fields.get("version") or DEFAULT_VERSION)
    plate_attrs, wells = plate_to_attrs(node, version)
    root = _open_store_group(str(path), zarr_format_of(version), mode="a",
                             storage_options=storage_options)
    root.attrs.put(plate_attrs)
    for path_in_plate, attrs in wells.items():
        root.require_group(path_in_plate).attrs.put(attrs)
    return plate_attrs


def write_plate(node: CollectionNode, path, *, version: Optional[str] = None,
                overwrite: bool = False, storage_options: Optional[dict] = None,
                writer=None, **write_kwargs) -> Dict[str, str]:
    """Write a plate: every image, one at a time, with ``io.write_pyramid`` (or
    `writer`) at ``<path>/<well path>/<image path>``, then the plate and well metadata.
    Returns ``{"<well path>/<image path>": written path}``.

    - `version`: ``"0.4"`` or ``"0.5"``; default: the version the plate was read as,
      else its images' version (when they share one), else 0.5. Images in another
      version are converted (``Pyramid.to_version``).
    - The whole plate is checked before any pixel is written, and its metadata is
      written last, so a write that stops half way leaves no plate metadata.
    - A field holding several images (images whose index, from an ``eubi:index``
      attribute or a held pyramid's ``source_index``, agrees in plate, acquisition,
      well, field and series but not in tile, view, ...) raises
      :class:`PlateFieldError`, in every version. Images without field numbers (a
      well built by hand) are each their own field.
    - An existing `path` is refused unless `overwrite` (which clears it first).
    - `write_kwargs` go to every image's write (``max_threads``, ``compressor``,
      ``chunk_shape``, ``memory_budget_mb``, ...)."""
    from ome_zarr_pyramid.core.io import IO, _open_store_group, _store_join

    attr = node.attributes.get("plate")
    if not isinstance(attr, PlateAttribute):
        raise ValueError(f"{node!r} is not a plate: it has no 'plate' attribute")
    # one image per field, in every version: before any version-specific rule
    # (walked: a field of several images is a collection of its own in the well)
    check_fields(node.name, [(well.name, [_index_of(image) for child in well.nodes
                                          for image in child.walk()
                                          if isinstance(image, MultiscaleNode)])
                             for well in node.nodes])
    # every rule of the layout first: before any image is opened (or pixel written)
    plate_to_attrs(node, str(version or node.ngff_fields.get("version") or DEFAULT_VERSION))
    jobs = []                                  # (well path, image path, pyramid)
    for well in node.nodes:
        for image, image_path in zip(well.nodes, image_paths(well)):
            if not isinstance(image, MultiscaleNode):
                continue                       # plate_to_attrs below names the problem
            jobs.append((well_path(attr, well), image_path, image.pyramid))
    versions = sorted({str(pyr.meta.version) for _, _, pyr in jobs})
    version = str(version or node.ngff_fields.get("version")
                  or (versions[0] if len(versions) == 1 else DEFAULT_VERSION))
    plate_to_attrs(node, version)              # every rule, before any pixel
    # an image in another version is written in the plate's (Pyramid.to_version)
    jobs = [(w, i, pyr if str(pyr.meta.version) == version else pyr.to_version(version))
            for w, i, pyr in jobs]

    from zarr.errors import ContainsArrayError, ContainsGroupError
    try:
        _open_store_group(str(path), zarr_format_of(version),
                          mode="w" if overwrite else "w-", storage_options=storage_options)
    except (ContainsGroupError, ContainsArrayError, FileExistsError) as exc:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it") from exc
    write = writer or IO().write_pyramid
    written = {}
    for in_plate, image_path, pyr in jobs:
        target = str(_store_join(str(path), in_plate, image_path))
        if storage_options is not None:
            write(pyr, target, storage_options=storage_options, **write_kwargs)
        else:
            write(pyr, target, **write_kwargs)
        written[f"{in_plate}/{image_path}"] = target
    from ome_zarr_pyramid.core.validation import filling
    with filling(write_kwargs.get("fill_derived")):        # the plate's own SHOULD fields too
        write_plate_metadata(node, path, version=version, storage_options=storage_options)
    return written
