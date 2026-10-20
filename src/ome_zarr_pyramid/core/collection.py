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


class PositionUnitError(ValueError):
    """An image's stage position along an axis is in a unit ozp cannot convert to that
    axis's unit, so it cannot become the axis's translation."""


def _position(q, unit: Optional[str], *, image: str, axis: str) -> float:
    """A position quantity's value in `unit`, the axis's own unit. Unlike `_quantity`,
    a failed conversion raises: a position that silently became 0 would move the image."""
    if unit is None or getattr(q, "unit", unit) == unit:
        return float(q.value)
    exc = None
    if hasattr(q, "to"):
        try:
            return float(q.to(unit).value)
        except Exception as err:                             # noqa: BLE001
            exc = err
    raise PositionUnitError(
        f"image {image!r}: the position along axis {axis!r} is in {str(q.unit)!r}, which "
        f"cannot be converted to the axis unit {unit!r}. Give the position in a length "
        f"unit convertible to {unit!r}, or clear image.metadata.position.") from exc


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
    - Translation: ``metadata.position`` in each axis's own scale unit, on every level
      (the stored levels are the image's own, so they share level 0's origin), written
      whenever the image has a position, (0, 0) included. A position along an axis in a
      unit that cannot be converted to the axis's unit raises ``PositionUnitError``.
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
        # written whenever the image has a position, (0, 0) included; an axis the
        # position does not name, and t / c, stay 0. Stored levels are the image's own
        # (a vendor pyramid): they keep level 0's translation (alignment 0).
        unit_of = dict(zip(axes, units))
        translation = [_position(position[ax], unit_of.get(ax), image=label, axis=ax)
                       if ax in "zyx" and position.get(ax) is not None else 0.0
                       for ax in axes]
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


class PyramidCollection(Mapping):
    """Pyramids keyed by where each image sits in its source: ``(("scene", 0),
    ("view", 1))``. A read-only mapping (``len``, iteration over keys, ``items()``);
    ``coll[{"scene": 0, "view": 1}]`` and ``coll.select(view=1)`` look entries up by
    index; :meth:`write` writes every entry.

    The home for the NGFF collections proposal once it is adopted: today each entry is
    written as its own OME-Zarr image.
    """

    def __init__(self, pyramids: Optional[Dict[Tuple[Tuple[str, int], ...], Any]] = None,
                 name: Optional[str] = None):
        self._pyramids = dict(pyramids or {})
        self.name = name

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
        return cls(out, name=getattr(file, "path", None))

    # -- mapping ---------------------------------------------------------------
    def _lookup(self, key):
        if isinstance(key, Mapping):
            key = _key(key)
        return self._pyramids[tuple(key)]

    def __getitem__(self, key):
        return self._lookup(key)

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
        return type(self)(keep, name=self.name)

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
            parts = [self._pyramids[key] for key in keys]
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
        for key, pyr in self._pyramids.items():
            path = os.path.join(str(directory), f"{stem}{self.suffix(key, names)}.ome.zarr")
            write(pyr, path, **write_kwargs)
            paths[key] = path
        return paths
