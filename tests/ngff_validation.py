"""Check a written OME-Zarr store as a whole: every group against the official NGFF
schemas of every role it has (the strict schemas included), and the hierarchy, which no
schema checks.

Why both: ozp 0.3.0 wrote an ``OME`` group with ``series: ["0"]`` under every single
image that had an OME-XML. Each group on its own validated (the OME group against
``ome.schema``, the image against ``image.schema``), and ome-zarr-models and ome-zarr-py
read the stores. But NGFF has an OME group only in a bioformats2raw layout, and its
``series`` named the level-0 array, not an image group. Only a check of where each group
sits catches that, so this one walks the whole store:

- each group's roles, from its NGFF keys: image (``multiscales``; with ``image-label`` also
  label), plate, well, bioformats2raw root, the OME group of a layout (``series``), labels
  group; each role against its schema and its strict schema (vendored in
  ``tests/ngff_schemas/<version>``, the files of ngff.openmicroscopy.org);
- an OME group only under a bioformats2raw root, its ``series`` naming image groups;
- an image's dataset paths are arrays; a labels group sits in an image and lists label
  images; a plate's wells and a well's fields are groups with their own metadata;
- a group holding ``ome`` metadata with no role is a problem.

``problems(path)`` returns the problems found, ``[]`` for a valid store. Local stores."""
from __future__ import annotations

import json
import posixpath
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

SCHEMAS = Path(__file__).parent / "ngff_schemas"


@lru_cache(maxsize=None)
def _schemas(version: str):
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012
    registry, by_name = Registry(), {}
    for f in sorted((SCHEMAS / version).glob("*.schema")):
        schema = json.loads(f.read_text(encoding="utf-8"))
        # the strict schemas name no $schema; the others are draft 2020-12
        resource = Resource.from_contents(schema, default_specification=DRAFT202012)
        registry = registry.with_resource(schema["$id"], resource)
        by_name[f.stem] = schema
    return registry, by_name


def _schema_errors(attrs: dict, version: str, role: str) -> List[str]:
    import jsonschema
    registry, by_name = _schemas(version)
    if role not in by_name:
        return []
    schema = by_name[role]
    cls = jsonschema.validators.validator_for(schema, default=jsonschema.Draft202012Validator)
    return [f"{role}.schema: {e.message[:160]} (at {'/'.join(map(str, e.absolute_path)) or '.'})"
            for e in cls(schema, registry=registry).iter_errors(attrs)]


def _walk(root: Path) -> Tuple[Dict[str, Tuple[int, dict]], set]:
    """{group path: (zarr format, attributes)} and the set of array paths, "" the root."""
    groups, arrays = {}, set()
    for meta in root.rglob("zarr.json"):
        rel = meta.parent.relative_to(root).as_posix()
        rel = "" if rel == "." else rel
        doc = json.loads(meta.read_text(encoding="utf-8"))
        if doc.get("node_type") == "group":
            groups[rel] = (3, doc.get("attributes") or {})
        elif doc.get("node_type") == "array":
            arrays.add(rel)
    for meta in root.rglob(".zgroup"):
        rel = meta.parent.relative_to(root).as_posix()
        rel = "" if rel == "." else rel
        attrs_file = meta.parent / ".zattrs"
        attrs = json.loads(attrs_file.read_text(encoding="utf-8")) if attrs_file.exists() else {}
        groups[rel] = (2, attrs)
    for meta in root.rglob(".zarray"):
        rel = meta.parent.relative_to(root).as_posix()
        arrays.add("" if rel == "." else rel)
    return groups, arrays


def _ngff(fmt: int, attrs: dict) -> dict:
    if fmt == 3:
        return attrs.get("ome") if isinstance(attrs.get("ome"), dict) else {}
    return attrs


def _version(ngff: dict) -> str:
    if ngff.get("version"):
        return str(ngff["version"])
    ms = ngff.get("multiscales")
    if isinstance(ms, list) and ms and isinstance(ms[0], dict) and ms[0].get("version"):
        return str(ms[0]["version"])
    for key in ("plate", "well"):
        if isinstance(ngff.get(key), dict) and ngff[key].get("version"):
            return str(ngff[key]["version"])
    return "0.4"


def _join(*parts: str) -> str:
    return posixpath.normpath(posixpath.join(*[p for p in parts if p])) if any(parts) else ""


def problems(path) -> List[str]:
    root = Path(path)
    groups, arrays = _walk(root)
    out: List[str] = []
    if "" not in groups:
        return [f"{path}: no group at the root"]
    for rel, (fmt, attrs) in sorted(groups.items()):
        where = rel or "the root"
        ngff = _ngff(fmt, attrs)
        version = _version(ngff)
        parent = posixpath.dirname(rel) if rel else None
        parent_ngff = _ngff(*groups[parent]) if parent is not None and parent in groups else {}
        roles = []
        if "multiscales" in ngff:
            roles += ["image", "strict_image"]
            if "image-label" in ngff:
                roles += ["label", "strict_label"]
            for ms in ngff["multiscales"] or []:
                for ds in ms.get("datasets") or []:
                    if _join(rel, str(ds.get("path"))) not in arrays:
                        out.append(f"{where}: dataset {ds.get('path')!r} is not an array")
        if "plate" in ngff:
            roles += ["plate", "strict_plate"]
            for well in (ngff["plate"].get("wells") or []):
                w = _join(rel, well.get("path", ""))
                if "well" not in _ngff(*groups.get(w, (3, {}))):
                    out.append(f"{where}: well {well.get('path')!r} is no well group")
        if "well" in ngff:
            roles += ["well", "strict_well"]
            for image in (ngff["well"].get("images") or []):
                i = _join(rel, image.get("path", ""))
                if "multiscales" not in _ngff(*groups.get(i, (3, {}))):
                    out.append(f"{where}: field {image.get('path')!r} is no image group")
        if "bioformats2raw.layout" in ngff:
            roles.append("bf2raw")
        if "series" in ngff:
            roles.append("ome")
            if posixpath.basename(rel) != "OME" or "bioformats2raw.layout" not in parent_ngff:
                out.append(f"{where}: an OME group with 'series' belongs only under a "
                           f"bioformats2raw root (its parent has no 'bioformats2raw.layout')")
            for s in ngff.get("series") or []:
                target = _join(parent or "", str(s))
                if target in arrays:
                    out.append(f"{where}: series {s!r} is an array, not an image group")
                elif "multiscales" not in _ngff(*groups.get(target, (3, {}))):
                    out.append(f"{where}: series {s!r} is no image group")
        if "labels" in ngff and isinstance(ngff["labels"], list):
            roles.append("labels")
            if "multiscales" not in parent_ngff or posixpath.basename(rel) != "labels":
                out.append(f"{where}: a labels group belongs inside an image, named 'labels'")
            for name in ngff["labels"]:
                lbl = _ngff(*groups.get(_join(rel, name), (3, {})))
                if "multiscales" not in lbl or "image-label" not in lbl:
                    out.append(f"{where}: label {name!r} is no label image")
        if not roles and ngff and set(ngff) - {"version"}:
            out.append(f"{where}: NGFF keys {sorted(ngff)} that make no NGFF node")
        if not roles and fmt == 3 and isinstance(attrs.get("ome"), dict) and set(attrs["ome"]) == {"version"}:
            out.append(f"{where}: an 'ome' block with only a version, no NGFF node")
        for role in roles:
            out += [f"{where}: {e}" for e in _schema_errors(attrs, version, role)]
    return out
