"""The bioformats2raw series layout (NGFF 0.4 / 0.5 "bf2raw") into the collection model
of :mod:`ome_zarr_pyramid.core.nodes`, and back.

The layout: a root group whose metadata has ``"bioformats2raw.layout": 3``, one
OME-Zarr image per series at paths ``0``, ``1``, ..., and optionally an ``OME`` group
holding the file's OME-XML (``OME/METADATA.ome.xml``) and the list of series paths
(``series``; without it the series sit at consecutive integer paths from ``0``).

It reads as a collection whose children are the series images, keyed ``(("scene", i),)``
as micro-reader keys a file's scenes. The images open on first use; the OME-XML is read
only when the collection is written again.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from ome_zarr_pyramid.core.ngff_plate import (
    VERSIONS, ImageOpener, _attributes_out, _group_attrs, _join, _root, _split, _stem,
    image_paths)
from ome_zarr_pyramid.core.nodes import (
    CollectionNode, InvalidCollection, MultiscaleNode, NodePath)

#: the root key that marks the layout, and the layout version this module writes
LAYOUT_KEY = "bioformats2raw.layout"
LAYOUT_VERSION = 3
_XML = "METADATA.ome.xml"


def _ome_series(attrs: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """The ``series`` list of an OME group's attributes (0.4: top level, 0.5: in ``ome``)."""
    if not isinstance(attrs, dict):
        return None
    series = attrs.get("series", (attrs.get("ome") or {}).get("series"))
    if isinstance(series, list) and all(isinstance(s, (str, int)) for s in series):
        return [str(s) for s in series]
    return None


def read_series(path, *, storage_options: Optional[dict] = None, _group=None
                ) -> CollectionNode:
    """Read a bioformats2raw series layout into a collection of its series images (each
    opened on first use). Raises :class:`InvalidCollection` naming every problem: a
    listed series without an image, a series group that is not an image."""
    root = _root(path, storage_options, _group)
    version, ngff, custom = _split(root.attrs.asdict())
    if version not in VERSIONS:
        raise InvalidCollection(f"{path}: version {version!r}; ozp reads NGFF {VERSIONS}")
    layout = ngff.pop(LAYOUT_KEY, None)
    if layout is None:
        raise InvalidCollection(f"{path}: no {LAYOUT_KEY!r} in its metadata")
    (ome_attrs,) = _group_attrs(root, ["OME"])
    listed = _ome_series(ome_attrs)
    if listed is not None:
        paths = listed
    else:                                  # consecutive integer paths from 0
        names = {str(k) for k in root.group_keys()}
        paths, i = [], 0
        while str(i) in names:
            paths.append(str(i))
            i += 1
    problems = []
    node = CollectionNode(_stem(path) or "series",
                          attributes={LAYOUT_KEY: layout, **ngff, **custom})
    opener = ImageOpener(str(path), storage_options)
    for series_path, attrs in zip(paths, _group_attrs(root, paths)):
        if attrs is None:
            problems.append(f"series {series_path!r} is listed but has no image")
            continue
        _, image_ngff, _ = _split(attrs)
        if "multiscales" not in image_ngff:
            problems.append(f"series {series_path!r} is not an image (no 'multiscales')")
            continue
        node.add(MultiscaleNode(series_path, path=NodePath("zarr", series_path),
                                loader=opener))
    if problems:
        raise InvalidCollection(f"{path}: " + "; ".join(problems))
    node.ngff_fields = {"version": version, "series": listed,
                        "ome_xml": {"path": str(path), "storage_options": storage_options}
                        if ome_attrs is not None else None}
    return node


def _source_xml(node: CollectionNode) -> Optional[str]:
    """The OME-XML of the layout `node` was read from, or None."""
    source = node.ngff_fields.get("ome_xml")
    if not source:
        return None
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    root = _root(source["path"], source["storage_options"], None)
    store, prefix = root.store_path.store, root.store_path.path
    key = "/".join(p for p in (prefix, "OME", _XML) if p)
    buf = sync(store.get(key, prototype=default_buffer_prototype()))
    return buf.to_bytes().decode("utf-8") if buf is not None else None


def write_series(node: CollectionNode, path, *, version: Optional[str] = None,
                 overwrite: bool = False, ome_xml: Optional[str] = None,
                 storage_options: Optional[dict] = None, writer=None,
                 **write_kwargs) -> Dict[str, str]:
    """Write a collection of images in the bioformats2raw layout: each image in turn at
    its path (``0``, ``1``, ... unless it has its own) with ``io.write_pyramid`` (or
    `writer`, with `write_kwargs`), then the root metadata and the ``OME`` group.
    Returns ``{series path: written path}``.

    - `version`: ``"0.4"`` or ``"0.5"``; default: the version it was read as, else the
      images' (when they share one), else 0.5. Images in another version are converted.
    - `ome_xml`: the OME-XML to store as ``OME/METADATA.ome.xml``; default: the one of
      the layout the collection was read from. Without one, the ``OME`` group is
      written only when the source had one (its ``series`` list).
    - An existing `path` is refused unless `overwrite` (which clears it first); the
      metadata is written last, so a write that stops half way is not a layout."""
    from zarr.errors import ContainsArrayError, ContainsGroupError

    from ome_zarr_pyramid.core.io import (
        IO, _open_store_group, _store_join, _write_ome_xml)

    images = [(p, image.pyramid) for image, p in zip(node.nodes, image_paths(node))
              if isinstance(image, MultiscaleNode)]
    if not images:
        raise ValueError(f"{node!r} holds no images to write as series")
    versions = sorted({str(pyr.meta.version) for _, pyr in images})
    version = str(version or node.ngff_fields.get("version")
                  or (versions[0] if len(versions) == 1 else "0.5"))
    if version not in VERSIONS:
        raise ValueError(f"version {version!r}: the series layout is written as {VERSIONS}")
    images = [(p, pyr if str(pyr.meta.version) == version else pyr.to_version(version))
              for p, pyr in images]
    ngff, custom = _attributes_out(node, LAYOUT_KEY)
    xml = ome_xml if ome_xml is not None else _source_xml(node)

    zarr_format = 3 if version == "0.5" else 2
    try:
        root = _open_store_group(str(path), zarr_format, mode="w" if overwrite else "w-",
                                 storage_options=storage_options)
    except (ContainsGroupError, ContainsArrayError, FileExistsError) as exc:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it") from exc
    write = writer or IO().write_pyramid
    written = {}
    for series_path, pyr in images:
        target = str(_store_join(str(path), series_path))
        if storage_options is not None:
            write(pyr, target, storage_options=storage_options, **write_kwargs)
        else:
            write(pyr, target, **write_kwargs)
        written[series_path] = target
    layout = node.attributes.get(LAYOUT_KEY, LAYOUT_VERSION)
    root.attrs.put(_join(version, {LAYOUT_KEY: copy.deepcopy(layout), **ngff}, custom))
    series = [p for p, _ in images]
    if xml is not None:
        _write_ome_xml(str(path), xml, series, zarr_format, storage_options)
    elif node.ngff_fields.get("series") is not None:
        ome = root.require_group("OME")
        ome.attrs.put({"series": series} if zarr_format == 2
                      else {"ome": {"version": version, "series": series}})
    return written
