"""NGFF validation (plan 5.23): a semantic check on every write, and an opt-in check of
a whole store against the official schemas.

The semantic check needs no extra dependency and runs whenever ozp writes metadata:

- :func:`check_multiscales` on each image's multiscales as they are saved: 2 to 5 axes
  with unique names, 2 or 3 of them space axes; each level a scale, then at most one
  translation, each as long as the axes, scales > 0; in 0.6 one transformation per level,
  from the level's path, every level into the same system, one of the multiscale's own;
- :func:`check_scene` on each 0.6 scene ozp writes: every edge end with a ``path`` names an
  image below the scene by its intrinsic system, every image below the scene is attached
  by an edge, and an image's position goes into a scene system with the image's axes
  (5.18(c)).

A problem there is an ozp bug, or metadata set by hand that NGFF does not allow: the
write stops with :class:`NGFFValidationError` before the metadata is written. (The OME-XML
of a bioformats2raw layout is checked where it is made: one Image per series group,
``OMEXMLError``.)

The store check, :func:`validate`, walks a written store and checks every group against
the official schemas of every role it has, plus what no schema checks: where each group
sits. It needs the ``validate`` extra (``pip install 'ome_zarr_pyramid[validate]'``:
jsonschema, referencing). The writers take ``validate=True`` to run it after writing;
``ozp.validate(path)`` runs it on any store. The schemas are vendored in
``ome_zarr_pyramid/schemas/<version>`` for 0.4, 0.5 and 0.6.

The strict schemas (``strict_image``, ``strict_plate``, ...) turn the spec's SHOULDs into
requirements: a plate's name, an image's name and metadata, a label's colors, each
acquisition's ``maximumfieldcount``. Their findings are not problems (decision 9.12-S):
``validate(path, strict=True)`` adds them, and ``validate=True`` warns with
:class:`NGFFValidationWarning`. A store copied from another tool often lacks some of these;
ozp writes what it can derive exactly (``maximumfieldcount``, an empty ``metadata``) and
keeps the rest as read, so a store ozp makes on its own meets them.

Why the walk: ozp 0.3.0 wrote an ``OME`` group with ``series: ["0"]`` under every single
image that had an OME-XML. Each group on its own validated, and readers read the stores.
But NGFF has an OME group only in a bioformats2raw layout, and its ``series`` named the
level-0 array, not an image group. Only a check of where each group sits catches that.
What the walk checks:

- each group's roles, from its NGFF keys: image (``multiscales``; with ``image-label``
  also label), plate, well, bioformats2raw root, the OME group of a layout (``series``),
  labels group, and in 0.6 a scene (``scene``, beside any other role); each role against
  its schema and its strict schema; each image's multiscales by :func:`check_multiscales`;
- a scene's edge ends that name a ``path`` are image groups below the scene's group;
- an OME group only under a bioformats2raw root, its ``series`` naming image groups;
- an image's dataset paths are arrays; a labels group sits in an image and lists label
  images; a plate's wells and a well's fields are groups with their own metadata;
- a group holding ``ome`` metadata with no role is a problem.

RFC 8 groups (version ``"0.x"``) have no published schema yet: the walk skips them.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import json
import os
import posixpath
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"
VERSIONS = ("0.4", "0.5", "0.6")
EXTRA = "pip install 'ome_zarr_pyramid[validate]'"
RFC8_VERSION = "0.x"


#: whether a write fills the SHOULD fields ozp can derive exactly (decision 9.12-S): an
#: acquisition's ``maximumfieldcount``, a multiscale's empty ``metadata`` and its ``type``
#: "unknown" when nothing says the method. ``write_pyramid(fill_derived=False)`` (and every
#: writer) writes them only when the source has them; a write sets it for its own duration.
FILL_DERIVED = contextvars.ContextVar("ozp_fill_derived", default=True)


@contextlib.contextmanager
def filling(value: Optional[bool]):
    """Writes inside fill the derivable SHOULD fields when `value` is True, not when False;
    None keeps the setting around (a write inside another write)."""
    if value is None:
        yield
        return
    token = FILL_DERIVED.set(bool(value))
    try:
        yield
    finally:
        FILL_DERIVED.reset(token)


class NGFFValidationWarning(UserWarning):
    """A written store meets every NGFF requirement but not every recommendation the strict
    schemas check (a plate's name, an image's name, a label's colors, ...)."""


class NGFFValidationError(ValueError):
    """Metadata or a store that NGFF does not allow. ``problems`` lists each one."""

    def __init__(self, message: str, problems: Sequence[str]):
        super().__init__(message)
        self.problems = list(problems)


def _raise(what: str, problems: Sequence[str], hint: str) -> None:
    if problems:
        shown = "\n  - ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise NGFFValidationError(f"{what}:\n  - {shown}{more}\n{hint}", problems)


# -- the semantic check, on every write ----------------------------------------------

def _numbers(value, n: int) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == n
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value))


def check_multiscales(ngff: Mapping[str, Any], version: str) -> List[str]:
    """The problems of the ``multiscales`` in `ngff` (the NGFF keys of an image group, in
    `version`'s form), [] when there are none. See the module docstring."""
    out: List[str] = []
    for i, ms in enumerate(ngff.get("multiscales") or []):
        at = f"multiscales[{i}]"
        if not isinstance(ms, Mapping):
            out.append(f"{at} is not an object")
            continue
        systems = ms.get("coordinateSystems") or [] if version == "0.6" else []
        if version == "0.6":
            if not systems or not isinstance(systems[0], Mapping):
                out.append(f"{at}: no coordinateSystems (0.6 names the axes there)")
                continue
            axes = systems[0].get("axes") or []
        else:
            axes = ms.get("axes") or []
        names = [a.get("name") if isinstance(a, Mapping) else None for a in axes]
        n = len(axes)
        if not 2 <= n <= 5:
            out.append(f"{at}: {n} axes {names}; NGFF allows 2 to 5")
        if len(set(names)) != n or None in names:
            out.append(f"{at}: axis names {names} are not unique names")
        space = [a.get("name") for a in axes if isinstance(a, Mapping) and a.get("type") == "space"]
        if not 2 <= len(space) <= 3:
            out.append(f"{at}: {len(space)} space axes {space}; NGFF needs 2 or 3")
        datasets = ms.get("datasets") or []
        if not datasets:
            out.append(f"{at}: no datasets (levels)")
        outputs = []
        for ds in datasets:
            path = ds.get("path") if isinstance(ds, Mapping) else None
            level = f"{at} level {path!r}"
            cts = (ds.get("coordinateTransformations") if isinstance(ds, Mapping) else None) or []
            if version == "0.6":
                if len(cts) != 1:
                    out.append(f"{level}: {len(cts)} transformations; 0.6 has one per level")
                    continue
                t = cts[0]
                outputs.append((t.get("output") or {}).get("name"))
                if (t.get("input") or {}).get("path") != path:
                    out.append(f"{level}: its transformation starts at "
                               f"{(t.get('input') or {}).get('path')!r}, not at the level")
                steps = t.get("transformations") or [] if t.get("type") == "sequence" else [t]
            else:
                steps = cts
            kinds = [s.get("type") for s in steps]
            if kinds not in (["scale"], ["scale", "translation"]):
                out.append(f"{level}: transformations {kinds}; a level has a scale, then at "
                           f"most one translation")
            for s in steps:
                if s.get("type") == "scale":
                    if not _numbers(s.get("scale"), n):
                        out.append(f"{level}: scale {s.get('scale')} is not {n} numbers")
                    elif any(v <= 0 for v in s["scale"]):
                        out.append(f"{level}: scale {s['scale']} is not > 0 on every axis")
                elif s.get("type") == "translation" and not _numbers(s.get("translation"), n):
                    out.append(f"{level}: translation {s.get('translation')} is not {n} numbers")
        if version == "0.6" and outputs:
            known = {s.get("name") for s in systems if isinstance(s, Mapping)}
            if len(set(outputs)) != 1:
                out.append(f"{at}: the levels map into {sorted(set(map(str, outputs)))}; "
                           f"every level maps into the same system")
            elif outputs[0] not in known:
                out.append(f"{at}: the levels map into {outputs[0]!r}, which is none of the "
                           f"multiscale's systems {sorted(map(str, known))}")
    return out


def check_scene(scene: Mapping[str, Any], images: Mapping[str, Tuple[str, Sequence[str]]]
                ) -> List[str]:
    """The problems of a 0.6 `scene` ozp is about to write, [] when there are none.
    `images` maps each image group below the scene (its path relative to the scene's
    group) to (its intrinsic system's name, its axis names)."""
    out: List[str] = []
    systems = {s.get("name"): [a.get("name") for a in s.get("axes") or []]
               for s in scene.get("coordinateSystems") or []}
    attached = set()
    for i, t in enumerate(scene.get("coordinateTransformations") or []):
        ends = [t.get("input"), t.get("output")]
        for end in ends:
            if not isinstance(end, Mapping) or end.get("path") is None:
                continue
            path = str(end["path"])
            if path not in images:
                out.append(f"scene edge {i}: path {path!r} is no image below the scene")
                continue
            attached.add(path)
            if end.get("name") != images[path][0]:
                out.append(f"scene edge {i}: names {end.get('name')!r} at {path!r}, whose "
                           f"intrinsic system is {images[path][0]!r}")
        source, target = ends
        if (isinstance(source, Mapping) and source.get("path") in images
                and isinstance(target, Mapping) and target.get("path") is None):
            name = target.get("name")
            if name not in systems:
                out.append(f"scene edge {i}: places {source['path']!r} in {name!r}, which is "
                           f"not one of the scene's systems {sorted(map(str, systems))}")
            elif list(systems[name]) != list(images[source["path"]][1]):
                out.append(f"scene edge {i}: {source['path']!r} has axes "
                           f"{list(images[source['path']][1])}, {name!r} has {systems[name]} "
                           f"(5.18(c))")
    for path in sorted(set(images) - attached):
        out.append(f"image {path!r} is below the scene but no edge attaches it")
    return out


def raise_for_metadata(where: str, problems: Sequence[str], version: str) -> None:
    """Stop a write whose metadata has `problems` (from the checks above)."""
    _raise(f"{where or 'image'}: ozp would write NGFF {version} metadata that NGFF does not "
           f"allow", problems,
           "This group's metadata was not written (its arrays may be). If the values came "
           "from you (scales, axes, transformations), correct them; otherwise this is an "
           "ozp bug, please report it.")


# -- the store check, opt-in -----------------------------------------------------------

def require_schemas() -> None:
    """Fail now, before anything is written, when the ``validate`` extra is missing."""
    try:
        import jsonschema  # noqa: F401
        import referencing  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"validating against the NGFF schemas needs jsonschema and "
                          f"referencing, which are optional: {EXTRA}") from exc


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


_META = {"zarr.json", ".zgroup", ".zarray", ".zattrs"}


def _node(rel: str, read: Mapping[str, Any], groups: Dict[str, Tuple[int, dict]],
          arrays: set) -> None:
    """Record the node in folder `rel`; `read` maps each metadata file there to a function
    returning its text."""
    if "zarr.json" in read:
        doc = json.loads(read["zarr.json"]())
        if doc.get("node_type") == "group":
            groups[rel] = (3, doc.get("attributes") or {})
        elif doc.get("node_type") == "array":
            arrays.add(rel)
    elif ".zarray" in read:
        arrays.add(rel)
    elif ".zgroup" in read:
        groups[rel] = (2, json.loads(read[".zattrs"]()) if ".zattrs" in read else {})


def _walk_local(root: Path):
    """{group path: (zarr format, attributes)} and the set of array paths ("" the root).
    An array's folder is not entered: its chunks can be millions of files."""
    groups: Dict[str, Tuple[int, dict]] = {}
    arrays: set = set()
    for here, dirs, files in os.walk(root):
        rel = Path(here).relative_to(root).as_posix()
        rel = "" if rel == "." else rel
        read = {name: functools.partial((Path(here) / name).read_text, encoding="utf-8")
                for name in _META.intersection(files)}
        _node(rel, read, groups, arrays)
        if rel in arrays:
            dirs[:] = []
    return groups, arrays


def _walk_remote(url: str, storage_options: Optional[dict]):
    """As :func:`_walk_local`, for an object-store URL (through obstore, with the
    storage_options names the writers take). Lists every key under the URL."""
    try:
        import obstore
    except ImportError as exc:
        raise ImportError("validating a remote store (s3://, gs://, az://) needs obstore: "
                          "pip install 'ome_zarr_pyramid[remote]'") from exc
    store = obstore.store.from_url(url, **(storage_options or {}))
    folders: Dict[str, Dict[str, Any]] = {}
    for batch in obstore.list(store):
        for item in batch:
            key = str(item["path"])
            rel, name = posixpath.split(key)
            if name in _META:
                folders.setdefault(rel, {})[name] = functools.partial(
                    lambda k: bytes(obstore.get(store, k).bytes()).decode("utf-8"), key)
    groups: Dict[str, Tuple[int, dict]] = {}
    arrays: set = set()
    for rel, read in folders.items():
        _node(rel, read, groups, arrays)
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


def validate(path, *, schemas: bool = True, strict: bool = False,
             storage_options: Optional[dict] = None) -> List[str]:
    """The problems of the OME-Zarr store at `path` (local, or an object-store URL with
    `storage_options`), [] for a valid one. Every group is checked against the official
    schemas of its roles for its version (0.4, 0.5, 0.6), and where each group sits (see
    the module docstring). `strict=True` adds the findings of the strict schemas, which
    turn the spec's SHOULDs into requirements (a plate's name, an image's name and
    metadata, a label's colors, ...). `schemas=False` skips the schemas: then no extra is
    needed. Needs ``pip install 'ome_zarr_pyramid[validate]'``."""
    errors, should = _findings(path, schemas, storage_options)
    return errors + should if strict else errors


def _findings(path, schemas: bool, storage_options: Optional[dict]) -> Tuple[List[str], List[str]]:
    """(problems, strict-schema findings) of the store at `path`."""
    if schemas:
        require_schemas()
    remote = "://" in str(path) and not str(path).startswith("file://")
    if remote:
        groups, arrays = _walk_remote(str(path), storage_options)
    else:
        root = Path(str(path)[len("file://"):] if str(path).startswith("file://") else path)
        if not root.is_dir():
            return [f"{path}: no such store"], []
        groups, arrays = _walk_local(root)
    out: List[str] = []
    should: List[str] = []
    if "" not in groups:
        return [f"{path}: no group at the root"], []
    rfc8 = False
    for rel, (fmt, attrs) in sorted(groups.items()):
        where = rel or "the root"
        ngff = _ngff(fmt, attrs)
        version = _version(ngff)
        if version == RFC8_VERSION:     # RFC 8: no published schema yet, its structure only
            rfc8 = True
            out += rfc8_problems(ngff, lambda p, rel=rel: _join(rel, p), groups, arrays, where)
            continue
        parent = posixpath.dirname(rel) if rel else None
        parent_ngff = _ngff(*groups[parent]) if parent is not None and parent in groups else {}
        roles = []
        if "multiscales" in ngff:
            roles += ["image", "strict_image"]
            if "image-label" in ngff:
                roles += ["label", "strict_label"]
            if version in VERSIONS:
                out += [f"{where}: {p}" for p in check_multiscales(ngff, version)]
            for ms in ngff["multiscales"] or []:
                for ds in (ms.get("datasets") or []) if isinstance(ms, dict) else []:
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
        if "scene" in ngff:
            roles.append("scene")
            for t in (ngff["scene"].get("coordinateTransformations") or []):
                for end in (t.get("input"), t.get("output")):
                    if isinstance(end, dict) and end.get("path"):
                        target = _join(rel, str(end["path"]))
                        if "multiscales" not in _ngff(*groups.get(target, (3, {}))):
                            out.append(f"{where}: scene edge end {end['path']!r} is no image "
                                       f"group below the scene")
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
        if schemas and version in VERSIONS:
            for role in roles:
                found = [f"{where}: {e}" for e in _schema_errors(attrs, version, role)]
                (should if role.startswith("strict_") else out).extend(found)
    if rfc8:
        _warn_rfc8(path)
    return out, should


def _warn_rfc8(path) -> None:
    from ome_zarr_pyramid.core.rfc8 import RFC8Warning
    warnings.warn(f"{path}: RFC 8 has no published schema yet; only its structure is checked "
                  f"(node types, each level an array, each referenced image a group, each "
                  f"transformation into a declared coordinate system)", RFC8Warning,
                  stacklevel=3)


def rfc8_problems(node: Any, resolve, groups: Mapping, arrays: set, where: str) -> List[str]:
    """The structural problems of RFC 8 node `node` (a document's ``ome``, or a node in
    it): every node of an RFC 8 type, every singlescale level's path an array, every
    multiscale or collection a path names a group of that kind, every level's
    transformation output a coordinate system its multiscale declares. `resolve` turns a
    path of the document into a store path (the keys of `groups` and `arrays`)."""
    out: List[str] = []
    if not isinstance(node, dict):
        return [f"{where}: an RFC 8 node is no JSON object"]
    kind = node.get("type")
    name = node.get("name") or node.get("id") or kind
    path = node.get("path")
    target = resolve(str(path["path"])) if isinstance(path, dict) and \
        path.get("type") == "zarr" and path.get("path") is not None else None
    if kind not in ("collection", "multiscale", "singlescale"):
        return [f"{where}: RFC 8 node {name!r} has type {kind!r}, not collection, "
                f"multiscale or singlescale"]
    if kind == "singlescale":
        if target is None:
            out.append(f"{where}: level {name!r} names no zarr path")
        elif target not in arrays:
            out.append(f"{where}: level {name!r} at {path['path']!r} is not an array")
        return out
    if target is not None and "nodes" not in node:      # a reference: the group says more
        if target not in groups:
            return [f"{where}: {kind} {name!r} at {path['path']!r} is no group"]
        found = _ngff(*groups[target])
        if kind == "multiscale" and not (found.get("type") == "multiscale"
                                         or "multiscales" in found):
            out.append(f"{where}: multiscale {name!r} at {path['path']!r} is no image group")
        if kind == "collection" and found.get("type") != "collection":
            out.append(f"{where}: collection {name!r} at {path['path']!r} is no collection")
        return out
    children = node.get("nodes")
    if not isinstance(children, list) or (kind == "multiscale" and not children):
        return [f"{where}: {kind} {name!r} lists no nodes"]
    systems = {cs.get("id") for cs in ((node.get("attributes") or {}).get("coordinateSystems")
                                       or []) if isinstance(cs, dict)}
    for child in children:
        out += rfc8_problems(child, resolve, groups, arrays, where)
        if kind != "multiscale" or not isinstance(child, dict):
            continue
        for t in ((child.get("attributes") or {}).get("coordinateTransformations") or []):
            output = (t.get("output") or {}).get("id") if isinstance(t, dict) else None
            if output not in systems:
                out.append(f"{where}: level {child.get('name')!r} of {name!r} transforms into "
                           f"{output!r}, which its multiscale does not declare")
    return out


def rfc8_document_problems(document, root) -> List[str]:
    """The structural problems of a standalone RFC 8 document (a ``.json`` file whose paths
    are relative to it) describing the local store at `root`."""
    with open(document, encoding="utf-8") as f:
        ome = json.load(f).get("ome", {})
    groups, arrays = _walk_local(Path(root))
    base = os.path.dirname(os.path.abspath(str(document)))

    def resolve(p: str) -> str:
        rel = os.path.relpath(os.path.normpath(os.path.join(base, p)), os.path.abspath(str(root)))
        return "" if rel == "." else rel.replace(os.sep, "/")
    return rfc8_problems(ome, resolve, groups, arrays, str(document))


def validating(write):
    """Give a writer ``validate=False``: with True, the ``validate`` extra is checked
    before writing, and the written store (`path`, else each path the writer returns) is
    checked after: :class:`NGFFValidationError` lists its problems, and
    :class:`NGFFValidationWarning` the strict schemas' findings (SHOULDs the store does not
    meet, often metadata copied as read from another tool's store; decision 9.12-S)."""
    signature = inspect.signature(write)

    @functools.wraps(write)
    def wrapper(*args, **kwargs):
        if not kwargs.pop("validate", False):
            return write(*args, **kwargs)
        require_schemas()
        result = write(*args, **kwargs)
        bound = signature.bind_partial(*args, **kwargs).arguments
        options = bound.get("storage_options")
        if "path" in bound:
            path = bound["path"]
            if str(path).startswith("https://"):     # the legacy form, written as s3://
                from ome_zarr_pyramid.core.io import _legacy_https_target
                path, options = _legacy_https_target(path, options)
            targets = [path]
        else:
            targets = list(result.values()) if isinstance(result, Mapping) else []
        document = bound.get("document")
        for target in targets:
            errors, should = _findings(target, True, options)
            if document is not None:         # a standalone RFC 8 document beside the store
                errors += rfc8_document_problems(document, target)
            _raise(f"{target}: the written store is not valid NGFF", errors,
                   "The store is written; fix what is listed, or report it as an ozp bug "
                   "if ozp made it.")
            if should:
                kinds = sorted({f.split(": ", 1)[-1].split(" (at ")[0] for f in should})
                warnings.warn(
                    f"{target}: valid NGFF, but {len(should)} recommendation(s) of the strict "
                    f"schemas are not met ({'; '.join(kinds)}). They are SHOULDs of the spec; "
                    f"ozp writes what it can derive and keeps the rest as read. "
                    f"ozp.validate(path, strict=True) lists each one.",
                    NGFFValidationWarning, stacklevel=2)
        return result

    params = list(signature.parameters.values())
    flag = inspect.Parameter("validate", inspect.Parameter.KEYWORD_ONLY, default=False)
    tail = [p for p in params if p.kind == inspect.Parameter.VAR_KEYWORD]
    wrapper.__signature__ = signature.replace(
        parameters=[p for p in params if p.kind != inspect.Parameter.VAR_KEYWORD] + [flag] + tail)
    return wrapper
