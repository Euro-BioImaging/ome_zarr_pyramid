"""The bioformats2raw series layout (NGFF 0.4 / 0.5 "bf2raw") into the collection model
of :mod:`ome_zarr_pyramid.core.nodes`, and back.

The layout: a root group whose metadata has ``"bioformats2raw.layout": 3``, one
OME-Zarr image per series at paths ``0``, ``1``, ..., and optionally an ``OME`` group
holding the file's OME-XML (``OME/METADATA.ome.xml``) and the list of series paths
(``series``; without it the series sit at consecutive integer paths from ``0``).

It reads as a collection whose children are the series images, keyed ``(("series", i),)``
as micro-reader keys a file's series. The images open on first use; the OME-XML is read
only when the collection is written again.

Written, the ``OME`` group always holds an OME-XML with one ``Image`` per series group,
in group order: the source layout's when it still fits, else a minimal one
(:func:`minimal_ome_xml`).
"""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

from ome_zarr_pyramid.core.ngff_plate import (
    DEFAULT_VERSION, VERSIONS, ImageOpener, _attributes_out, _group_attrs, _join, _root,
    _split, _stem, image_paths, zarr_format_of)
from ome_zarr_pyramid.core.nodes import (
    CollectionNode, InvalidCollection, MultiscaleNode, NodePath, check_read_version)

#: the root key that marks the layout, and the layout version this module writes
LAYOUT_KEY = "bioformats2raw.layout"
LAYOUT_VERSION = 3
_XML = "METADATA.ome.xml"
#: the OME-XML schema the minimal document is written in
OME_NAMESPACE = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
#: numpy dtype name -> OME PixelType
_PIXEL_TYPES = {"uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "int8": "int8",
                "int16": "int16", "int32": "int32", "float32": "float", "float64": "double",
                "bool": "bit", "complex64": "complex", "complex128": "double-complex"}


class OMEXMLError(ValueError):
    """An OME-XML that cannot be written with a series layout: one passed explicitly whose
    ``Image`` count differs from the series groups written (or that is not XML), or a
    pixel type OME-XML has no name for."""


class SeriesOpener(ImageOpener):
    """The loader of a series layout's images: the pyramid it opens also records which
    series of which layout it is (``source_series``), so a write of it in another
    collection knows the OME-XML it came with (and whether that still fits)."""

    def __init__(self, base, storage_options: Optional[dict] = None,
                 series: Optional[List[str]] = None):
        super().__init__(base, storage_options)
        self.series = list(series or [])

    def __call__(self, node):
        pyr = super().__call__(node)
        pyr.source_series = {"layout": self.base, "storage_options": self.storage_options,
                             "series": node.path.path,
                             "position": self.series.index(node.path.path)}
        return pyr


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
    check_read_version(version, str(path))
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
    opener = SeriesOpener(str(path), storage_options, paths)
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


def _layout_xml(path, storage_options) -> Optional[str]:
    """The OME-XML of the layout at `path` (``OME/METADATA.ome.xml``), or None."""
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    root = _root(path, storage_options, None)
    store, prefix = root.store_path.store, root.store_path.path
    key = "/".join(p for p in (prefix, "OME", _XML) if p)
    buf = sync(store.get(key, prototype=default_buffer_prototype()))
    return buf.to_bytes().decode("utf-8") if buf is not None else None


def _source_xml(node: CollectionNode) -> Optional[str]:
    """The OME-XML of the layout `node` was read from, or None."""
    source = node.ngff_fields.get("ome_xml")
    if not source:
        return None
    return _layout_xml(source["path"], source["storage_options"])


def count_images(xml: str) -> int:
    """The number of ``Image`` elements under an OME-XML document's root, in any
    namespace. Raises ``xml.etree.ElementTree.ParseError`` for text that is not XML."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    return sum(1 for child in root if child.tag == "Image" or str(child.tag).endswith("}Image"))


def _dimension_order(axes: str) -> str:
    """OME's DimensionOrder for ozp axes (a subset of ``tczyx``): X and Y first, then the
    image's other axes innermost first, then the absent ones (``yx`` -> XYZCT, ``cyx`` ->
    XYCZT, ``tczyx`` -> XYZCT)."""
    rest = [a.upper() for a in reversed(axes) if a not in "xy"]
    rest += [a for a in "ZCT" if a not in rest]
    return "XY" + "".join(rest)


def minimal_ome_xml(images) -> str:
    """A minimal OME-XML for a series layout: one ``Image`` per ``(series path, pyramid)``
    of `images`, in that order, with ``ID="Image:<path>"``, the pyramid's name, and a
    ``Pixels`` element (``ID="Pixels:<path>"``, sizes from level 0, 1 for an absent axis,
    the pixel type and the dimension order) holding ``MetadataOnly``: the pixels are the
    series' arrays. No planes and no channel metadata. Raises :class:`OMEXMLError` for a
    pixel type OME-XML has no name for."""
    import xml.etree.ElementTree as ET

    import numpy as np

    # unqualified tags under a default xmlns: ElementTree's default_namespace refuses
    # unqualified attributes, and every OME attribute is one
    root = ET.Element("OME", {"xmlns": OME_NAMESPACE, "Creator": "ome_zarr_pyramid"})
    for path, pyr in images:
        axes = str(pyr.meta.axis_order).lower()
        dtype = np.dtype(pyr.dtype).name
        pixel_type = _PIXEL_TYPES.get(dtype)
        if pixel_type is None:
            raise OMEXMLError(
                f"series {path!r}: OME-XML has no pixel type for {dtype} (it names "
                f"{sorted(set(_PIXEL_TYPES))}); cast the image to one of them "
                f"(Pyramid.saturate_cast), or pass ome_xml= with one Image per series")
        image = ET.SubElement(root, "Image", {"ID": f"Image:{path}"})
        if pyr.meta.tag:
            image.set("Name", str(pyr.meta.tag))
        size = dict(zip(axes, (int(n) for n in pyr.shape)))
        pixels = ET.SubElement(image, "Pixels", {
            "ID": f"Pixels:{path}", "DimensionOrder": _dimension_order(axes),
            "Type": pixel_type, **{f"Size{a.upper()}": str(size.get(a, 1)) for a in "xyzct"}})
        ET.SubElement(pixels, "MetadataOnly")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def _carried(node: CollectionNode, images) -> Tuple[Optional[str], Optional[str], bool]:
    """(the source OME-XML the images come with, where it is, whether the written groups
    are still the source's series in its order). A layout read and written whole carries
    its own; images opened from a layout and put in another collection carry theirs
    (``source_series``, see :class:`SeriesOpener`)."""
    source = node.ngff_fields.get("ome_xml")
    if source:                                 # the layout as read, written whole
        return _source_xml(node), str(source["path"]), True
    sources = [getattr(pyr, "source_series", None) for _, pyr in images]
    first = next((s for s in sources if s), None)
    if first is None:
        return None, None, False
    xml = _layout_xml(first["layout"], first["storage_options"])
    if xml is None:                            # the source layout has no OME-XML
        return None, None, False
    fits = (all(s and s["layout"] == first["layout"] for s in sources)
            and [s["position"] for s in sources] == list(range(len(sources))))
    return xml, first["layout"], fits


def _xml_for(node: CollectionNode, images, ome_xml: Optional[str]) -> str:
    """The OME-XML written with `images` (``[(series path, pyramid)]``): `ome_xml` when it
    holds one Image per series (else :class:`OMEXMLError`); else the carried source one
    when it still fits; else a minimal one (replacing a carried one with a warning)."""
    from xml.etree.ElementTree import ParseError

    n = len(images)
    if ome_xml is not None:
        if not isinstance(ome_xml, str) or not ome_xml.strip():
            raise ValueError("ome_xml: an OME-XML document (a non-empty string) is needed")
        try:
            count = count_images(ome_xml)
        except ParseError as exc:
            raise OMEXMLError(f"ome_xml is not an XML document ({exc}); pass an OME-XML "
                              f"with one Image per series, or omit ome_xml (ozp then "
                              f"writes a minimal one)") from exc
        if count != n:
            raise OMEXMLError(
                f"ome_xml holds {count} Image element{'' if count == 1 else 's'}, but {n} "
                f"series are written: a bioformats2raw layout's OME-XML has one Image per "
                f"series group, in group order. Pass one Image per series, or omit "
                f"ome_xml (ozp then writes a minimal one)")
        return ome_xml
    carried, where, fits = _carried(node, images)
    if carried is not None:
        try:
            found: Optional[int] = count_images(carried)
        except ParseError:
            found = None
        if fits and found == n:
            return carried                     # as read, unchanged
        held = "is not XML" if found is None else \
            f"holds {found} Image element{'' if found == 1 else 's'}"
        warnings.warn(
            f"write_series: the source OME-XML of {where} {held} and does not fit the {n} "
            f"series written; it is replaced by a minimal OME-XML with one Image per "
            f"series", UserWarning, stacklevel=4)
    return minimal_ome_xml(images)


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
    - The ``OME`` group (its ``series`` list) and ``OME/METADATA.ome.xml`` are always
      written, with one ``Image`` per series group in group order. `ome_xml`: the
      OME-XML to store; one with another number of Images raises
      :class:`OMEXMLError`, before anything is written. Without it: the OME-XML of the
      layout the images were read from, when the written groups are still its series
      in its order; else a minimal one (:func:`minimal_ome_xml`), with a ``UserWarning``
      when it replaces a source OME-XML that no longer fits.
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
                  or (versions[0] if len(versions) == 1 else DEFAULT_VERSION))
    if version not in VERSIONS:
        raise ValueError(f"version {version!r}: the series layout is written as {VERSIONS}")
    xml = _xml_for(node, images, ome_xml)      # before any pixel is written
    images = [(p, pyr if str(pyr.meta.version) == version else pyr.to_version(version))
              for p, pyr in images]
    ngff, custom = _attributes_out(node, LAYOUT_KEY)

    try:
        root = _open_store_group(str(path), zarr_format_of(version),
                                 mode="w" if overwrite else "w-",
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
    _write_ome_xml(str(path), xml, [p for p, _ in images], version, storage_options,
                   layout=True)
    return written
