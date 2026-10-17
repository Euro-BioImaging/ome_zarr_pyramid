"""Pyramids from microscopy images, and collections of them.

Two levels, mirroring a microscopy reader (micro-reader's ``File`` / ``Image``):

* :meth:`Pyramid.from_image` (``pyramid_from_image`` here): ONE image -> a ``Pyramid``. Its
  axes, pixel sizes and units, stage position, channels and stored resolution levels
  become the NGFF multiscale; every level is a lazy dyna-zarr array over the image.
* :meth:`PyramidCollection.from_file`: every image of a file -> a ``PyramidCollection``,
  keyed by the image's index in the file (scene, tile, view, illumination, ...).

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

from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np

#: the axes an NGFF multiscale may have here
_AXES = "tczyx"
#: an image index entry that is a resolution LEVEL, not a separate image
_RESOLUTION = "resolution"


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
    - Translation: ``metadata.position`` in each axis's own scale unit, on every level.
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
        unit_of = dict(zip(axes, units))
        translation = []
        for ax in axes:
            q = _quantity(position.get(ax), unit_of.get(ax)) if ax in "zyx" else None
            translation.append(q[0] if q is not None else 0.0)
        if any(translation):
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


class PyramidCollection(Mapping):
    """Pyramids keyed by where each image sits in its source: ``(("scene", 0),
    ("view", 1))``. A read-only mapping (``len``, iteration over keys, ``items()``);
    ``coll[{"scene": 0, "view": 1}]`` and ``coll.select(view=1)`` look entries up by
    index; :meth:`write` writes every entry.

    A collection read from OME-Zarr (:meth:`from_zarr`) keeps its node tree as
    :attr:`tree` (see :mod:`ome_zarr_pyramid.core.nodes`), and opens each pyramid on
    first access: listing, selecting and indexing keys open nothing.
    """

    def __init__(self, pyramids: Optional[Dict[Tuple[Tuple[str, int], ...], Any]] = None,
                 name: Optional[str] = None, tree=None):
        # a value is a Pyramid, or a MultiscaleNode opened on first access
        self._pyramids = dict(pyramids or {})
        self.name = name
        #: the node tree the collection was read from (a plate, a well, an image), or None
        self.tree = tree

    @classmethod
    def from_zarr(cls, path, *, storage_options: Optional[dict] = None) -> "PyramidCollection":
        """The collection stored at `path`: an NGFF 0.4/0.5 plate (keyed ``plate``,
        ``acquisition``, ``well``, ``field``, as micro-reader keys plate images), a well
        on its own (``acquisition``, ``field``), or a single image (the one key ``()``).
        The plate and well metadata are read now; each image when it is first used.
        `storage_options` as for ``io.read_pyramid`` (an object-store URL)."""
        from ome_zarr_pyramid.core.io import _open_store_group
        from ome_zarr_pyramid.core.ngff_plate import (
            ImageOpener, _split, _stem, read_plate, read_well)
        from ome_zarr_pyramid.core.nodes import MultiscaleNode, NodePath

        from ome_zarr_pyramid.core import rfc8

        if str(path).lower().endswith(".json"):                # a standalone RFC 8 document
            return cls._from_tree(rfc8.read_rfc8(path, storage_options=storage_options))
        from zarr.errors import ContainsArrayError
        try:
            root = _open_store_group(str(path), None, mode="r", storage_options=storage_options)
        except ContainsArrayError:            # an array: an RFC 8 singlescale image
            return cls._from_tree(rfc8.read_rfc8(path, storage_options=storage_options))
        attrs = root.attrs.asdict()
        if rfc8.is_rfc8(attrs.get("ome")):
            return cls._from_tree(rfc8.read_rfc8(path, storage_options=storage_options))
        _, ngff, _ = _split(attrs)
        if "plate" in ngff:
            tree = read_plate(path, storage_options=storage_options, _group=root)
        elif "well" in ngff:
            tree = read_well(path, storage_options=storage_options, _group=root)
        elif "multiscales" in ngff:
            tree = MultiscaleNode(_stem(path) or "image", path=NodePath("zarr", "."),
                                  loader=ImageOpener(path, storage_options))
            return cls({(): tree}, name=tree.name, tree=tree)
        elif "bioformats2raw.layout" in ngff:
            from ome_zarr_pyramid.core.ngff_series import read_series
            tree = read_series(path, storage_options=storage_options, _group=root)
            # keyed as micro-reader keys a file's scenes (bioformats2raw's series)
            return cls({(("scene", i),): node for i, node in enumerate(tree.nodes)},
                       name=tree.name, tree=tree)
        else:
            raise ValueError(f"{path}: not an OME-Zarr image, well or plate (its metadata "
                             f"has no 'multiscales', 'well' or 'plate')")
        return cls(dict(_tree_entries(tree)), name=tree.name, tree=tree)

    @classmethod
    def _from_tree(cls, tree) -> "PyramidCollection":
        """A collection over an RFC 8 tree: plates keyed as micro-reader keys plate
        images, any other collection ``(("scene", i),)``, a single image ``()``."""
        from ome_zarr_pyramid.core.plates import plate_nodes
        from ome_zarr_pyramid.core.rfc8 import tree_entries

        entries = (_tree_entries(tree) if plate_nodes(tree) else tree_entries(tree))
        return cls(dict(entries), name=tree.name, tree=tree)

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
        :meth:`Pyramid.from_image`."""
        out = {}
        for image in file.images:
            index = dict(getattr(image, "index", None) or {})
            if int(index.get(_RESOLUTION, 0)) != 0:
                continue                        # a level of an image already in the set
            key = _key(index)
            if key in out:
                raise ValueError(f"two images share the index {dict(key)}")
            out[key] = pyramid_from_image(image, version=version, **kwargs)
        from ome_zarr_pyramid.core.ngff_plate import _stem
        from ome_zarr_pyramid.core.plates import tree_from_file
        path = getattr(file, "path", None)
        tree = tree_from_file(file, out, name=_stem(str(path)) if path else None)
        return cls(out, name=path, tree=tree)

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

        Images whose index has more than plate / acquisition / well / field (a tile, a
        view, ...) are written as separate images of their well, with a warning: NGFF
        0.4/0.5 wells cannot say which tile or view an image is."""
        import warnings

        from ome_zarr_pyramid.core.ngff_plate import write_plate
        from ome_zarr_pyramid.core.nodes import MultiscaleNode
        from ome_zarr_pyramid.core.plates import plate_nodes

        nodes = plate_nodes(self.tree)
        if not nodes:
            raise ValueError("the collection holds no plate (see .plates)")
        if not 0 <= plate < len(nodes):
            raise IndexError(f"plate {plate}: the collection holds {len(nodes)} plate(s)")
        node = nodes[plate]
        held = {id(v) for v in self._pyramids.values()}
        mine = [k for k in self._pyramids if dict(k).get("plate", 0) == plate]
        for well in node.nodes:
            for image in well.walk():
                if not isinstance(image, MultiscaleNode) or "labels" in image.attributes \
                        or "source" in image.attributes:
                    continue                   # label and derived images are not entries
                if id(image) not in held and id(getattr(image, "_pyramid", None)) not in held:
                    raise ValueError(
                        "this collection is a selection of its plate; write the plate from "
                        "the whole collection (a plate is written with all its images)")
        if version == "rfc8":
            from ome_zarr_pyramid.core.rfc8 import write_rfc8
            return write_rfc8(node, path, document=document, overwrite=overwrite,
                              storage_options=storage_options, writer=writer, **write_kwargs)
        extra = sorted({name for k in mine for name, _ in k}
                       - {"plate", "acquisition", "well", "field"})
        if extra:
            warnings.warn(f"plate {plate}: its images are also indexed by {extra}; they are "
                          f"written as separate images of their well, which NGFF 0.4/0.5 "
                          f"cannot label with {extra}", stacklevel=2)
        return write_plate(node, path, version=version, overwrite=overwrite,
                           storage_options=storage_options, writer=writer, **write_kwargs)

    def write_series(self, path, *, version: Optional[str] = None, overwrite: bool = False,
                     ome_xml: Optional[str] = None, storage_options: Optional[dict] = None,
                     writer: Optional[Callable] = None, document: Optional[str] = None,
                     **write_kwargs) -> Dict[str, str]:
        """Write every entry as one series of a bioformats2raw layout (``<path>/0``,
        ``<path>/1``, ..., in key order; a collection read from that layout keeps its
        series paths), then the root metadata and the ``OME`` group. Returns
        ``{series path: path}``. `ome_xml` is stored as ``OME/METADATA.ome.xml``
        (default: the source layout's). See
        :func:`ome_zarr_pyramid.core.ngff_series.write_series`.

        ``version="rfc8"``: RFC 8 replaces the layout with a collection of the series
        images, so that is written (see `write_collection`); `ome_xml` is not used."""
        from ome_zarr_pyramid.core.ngff_series import LAYOUT_KEY, write_series
        from ome_zarr_pyramid.core.nodes import CollectionNode, MultiscaleNode

        tree = self.tree
        if isinstance(tree, CollectionNode) and LAYOUT_KEY in tree.attributes \
                and "plate" not in tree.attributes \
                and {id(n) for n in tree.nodes} == {id(v) for v in self._pyramids.values()}:
            node = tree                    # read from a series layout, all of it: as read
        else:                              # any other collection: series 0, 1, ... in key order
            node = CollectionNode("series")
            for i, key in enumerate(sorted(self._pyramids)):
                node.add(MultiscaleNode(str(i), self[key]))
        if version == "rfc8":
            from ome_zarr_pyramid.core.rfc8 import write_rfc8
            return write_rfc8(node, path, document=document, overwrite=overwrite,
                              storage_options=storage_options, writer=writer, **write_kwargs)
        return write_series(node, path, version=version, overwrite=overwrite, ome_xml=ome_xml,
                            storage_options=storage_options, writer=writer, **write_kwargs)

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
        tree = self.tree
        if isinstance(tree, MultiscaleNode):
            node = CollectionNode(tree.name, [tree])
        elif isinstance(tree, CollectionNode):
            held = {id(v) for v in self._pyramids.values()}
            images = [n for _, n in (_tree_entries(tree) if self.plates else tree_entries(tree))]
            if any(id(n) not in held and id(getattr(n, "_pyramid", None)) not in held
                   for n in images):
                raise ValueError("this collection is a selection of its tree; write it from "
                                 "the whole collection")
            node = tree
        else:
            node = CollectionNode(str(self.name or "collection"))
            for i, key in enumerate(sorted(self._pyramids)):
                node.add(MultiscaleNode(str(i), self[key]))
        return write_rfc8(node, path, document=document, overwrite=overwrite,
                          storage_options=storage_options, writer=writer, **write_kwargs)

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

    def __iter__(self) -> Iterator[Tuple[Tuple[str, int], ...]]:
        return iter(self._pyramids)

    def __len__(self) -> int:
        return len(self._pyramids)

    def __repr__(self) -> str:
        names = sorted({k for key in self._pyramids for k, _ in key})
        return f"PyramidCollection({len(self)} pyramids, indexed by {names})"

    def select(self, **index) -> "PyramidCollection":
        """The entries whose index matches every given value: ``select(view=1)``."""
        keep = {k: p for k, p in self._pyramids.items()
                if all(dict(k).get(name) == value for name, value in index.items())}
        # the same tree: plates keep every well (so "well": w stays wells[w]), and their
        # fields hold the selected images only
        return type(self)(keep, name=self.name, tree=self.tree)

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
        {"view": 1}, ...]``, to name its channels."""
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
        for rest, keys in groups.items():
            keys.sort(key=lambda key: tuple(dict(key)[name] for name in names))
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
        return type(self)(out, name=self.name)

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

    def write(self, directory, stem: str, *, names: Optional[Dict[str, str]] = None,
              writer: Optional[Callable] = None, **write_kwargs) -> Dict[Any, str]:
        """Write each entry as ``<directory>/<stem><suffix>.ome.zarr``; returns
        {key: path}. `write_kwargs` go to ``io.write_pyramid`` (or `writer`)."""
        import os

        from ome_zarr_pyramid.core.io import IO
        write = writer or IO().write_pyramid
        paths = {}
        for key, value in self._pyramids.items():
            pyr = self._open(value)
            path = os.path.join(str(directory), f"{stem}{self.suffix(key, names)}.ome.zarr")
            write(pyr, path, **write_kwargs)
            paths[key] = path
        return paths
