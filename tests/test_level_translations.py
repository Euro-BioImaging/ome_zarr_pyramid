"""Per-level translations (plan 3.19, 5.14, 9.4a).

A level's translation is the image's position (level 0's translation) plus the level's
alignment: 0 for levels made with the stride method 'simple', (F - 1) / 2 level-0 pixels
= (s_k - s_0) / 2 for levels made with a window method (mean, median, min, max, mode).
Levels taken from a file keep what they store. An image at (0, 0) still writes its
translation, and a position ozp cannot convert raises instead of becoming 0.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ome_zarr_pyramid import IO, PositionUnitError, Pyramid, PyramidCollection
from ome_zarr_pyramid.core.pyramid import STRIDE_METHODS, level_translation
from ome_zarr_pyramid.utils.scale import DOWNSCALE_METHODS

WINDOW_METHODS = ("mean", "median", "min", "max", "mode")


class Q:
    """A quantity: value + unit, convertible between length units (not 'pixel')."""
    _to_m = {"micrometer": 1e-6, "nanometer": 1e-9, "millimeter": 1e-3}

    def __init__(self, value, unit):
        self.value, self.unit = value, unit

    def to(self, unit):
        return Q(self.value * self._to_m[self.unit] / self._to_m[unit], unit)


class FakeImage:
    """Image-like: a NumPy array with micro-reader's attribute names."""

    def __init__(self, data, axes, metadata=None, index=None, levels=(), name="img"):
        self._data, self.axes, self.metadata, self.name = data, axes, metadata, name
        self.shape, self.dtype = data.shape, data.dtype
        self.index = index or {"position": 0}
        self.resolutions = [self] + [FakeImage(lv, axes, metadata, index) for lv in levels]

    def __getitem__(self, key):
        return self._data[key]


def _datasets(path):
    """The written multiscale datasets, from zarr.json (0.5) or .zattrs (0.4)."""
    if os.path.exists(os.path.join(path, "zarr.json")):
        attrs = json.load(open(os.path.join(path, "zarr.json")))["attributes"]["ome"]
    else:
        attrs = json.load(open(os.path.join(path, ".zattrs")))
    return attrs["multiscales"][0]["datasets"]


def _written_translations(path):
    """Each written level's translation, or None where a level has none."""
    out = []
    for ds in _datasets(path):
        tr = [t["translation"] for t in ds["coordinateTransformations"]
              if t["type"] == "translation"]
        out.append(tr[0] if tr else None)
    return out


def _yx(translation=(5.0, 3.0)):
    """64 x 64 uint16, 0.5 um pixels, level 0 at (y 5, x 3) um."""
    pyr = Pyramid().from_array(np.arange(4096, dtype="uint16").reshape(64, 64),
                               axis_order="yx", unit_list=["micrometer"] * 2,
                               scale=[0.5, 0.5])
    if translation is not None:
        pyr.meta.update_translation("0", list(translation))
    return pyr


def _expected(method):
    if method in STRIDE_METHODS:
        return [[5.0, 3.0]] * 3
    return [[5.0, 3.0], [5.25, 3.25], [5.75, 3.75]]


# the shared rule ----------------------------------------------------------------------

def test_the_helper_covers_every_downscale_method():
    assert set(STRIDE_METHODS) | set(WINDOW_METHODS) == set(DOWNSCALE_METHODS)
    for m in DOWNSCALE_METHODS:
        got = [level_translation([5, 3], [0.5, 0.5], [s, s], m) for s in (0.5, 1.0, 2.0)]
        assert got == _expected(m)


def test_the_helper_gives_no_negative_zero_and_refuses_an_unknown_method():
    got = level_translation([-0.25, 0.0], [0.5, 0.5], [1.0, 0.5], "mean")
    assert got == [0.0, 0.0] and all(str(v) == "0.0" for v in got)
    assert str(level_translation([-0.0], [1.0], [2.0], "simple")[0]) == "0.0"
    with pytest.raises(ValueError, match="downscale_method"):
        level_translation([0.0], [1.0], [2.0], "bilinear")


# the two downscale paths --------------------------------------------------------------

@pytest.mark.parametrize("defer", [True, False])
@pytest.mark.parametrize("method", DOWNSCALE_METHODS)
def test_written_translations_follow_the_method(tmp_path, method, defer):
    pyr = _yx().downscale(n_layers=3, scale_factor=2, downscale_method=method, defer=defer)
    expected = _expected(method)
    # in memory (get_downscaled_pyramid builds the visible levels on both paths) ...
    assert [pyr.meta.get_translation(p) for p in pyr.meta.resolution_paths] == expected
    # ... and written: the deferred write saves io._full_pyramid's group attributes
    path = str(tmp_path / f"{method}.ome.zarr")
    IO().write_pyramid(pyr, path, overwrite=True)
    assert _written_translations(path) == expected
    assert [ds["coordinateTransformations"][0]["scale"] for ds in _datasets(path)] == \
        [[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]
    back = IO().read_pyramid(path)
    assert [back.meta.get_translation(p) for p in back.meta.resolution_paths] == expected


@pytest.mark.parametrize("method", ["simple", "mean"])
def test_the_dask_deferred_write_follows_the_method(tmp_path, method):
    # a dask base takes the older deferred route (level 0 written, levels from disk)
    da = pytest.importorskip("dask.array")
    pyr = Pyramid().from_array(da.from_array(np.arange(4096, dtype="uint16").reshape(64, 64),
                                             chunks=32),
                               axis_order="yx", unit_list=["micrometer"] * 2, scale=[0.5, 0.5])
    pyr.meta.update_translation("0", [5, 3])
    path = str(tmp_path / f"dask_{method}.ome.zarr")
    IO().write_pyramid(pyr.downscale(n_layers=3, scale_factor=2, downscale_method=method),
                       path, overwrite=True)
    assert _written_translations(path) == _expected(method)


@pytest.mark.parametrize("defer", [True, False])
def test_t_and_c_stay_zero(tmp_path, defer):
    data = np.zeros((2, 3, 8, 32, 32), "uint16")
    pyr = Pyramid().from_array(data, axis_order="tczyx",
                               unit_list=["second", "Channel", "micrometer", "micrometer",
                                          "micrometer"],
                               scale=[1.0, 1.0, 2.0, 0.5, 0.5])
    pyr.meta.update_translation("0", [0, 0, 10, 5, 3])
    pyr = pyr.downscale(n_layers=3, scale_factor=[1, 1, 2, 2, 2], downscale_method="mean",
                        defer=defer)
    path = str(tmp_path / "tczyx.ome.zarr")
    IO().write_pyramid(pyr, path, overwrite=True)
    assert _written_translations(path) == [[0.0, 0.0, 10.0, 5.0, 3.0],
                                           [0.0, 0.0, 11.0, 5.25, 3.25],
                                           [0.0, 0.0, 13.0, 5.75, 3.75]]


def test_no_base_translation_writes_none(tmp_path):
    pyr = _yx(translation=None).downscale(n_layers=3, scale_factor=2, downscale_method="mean",
                                          defer=False)
    path = str(tmp_path / "none.ome.zarr")
    IO().write_pyramid(pyr, path, overwrite=True)
    assert _written_translations(path) == [None, None, None]


# levels taken from a file keep what they store -----------------------------------------

def test_zero_translations_are_written_without_downscaling(tmp_path):
    base = np.arange(4096, dtype="uint16").reshape(64, 64)
    pyr = Pyramid().from_arrays([base, base[::2, ::2]], axis_order="yx",
                                unit_list=["micrometer"] * 2, scales=[[0.5, 0.5], [1.0, 1.0]])
    for p in pyr.meta.resolution_paths:
        pyr.meta.update_translation(p, [0, 0])
    path = str(tmp_path / "zero.ome.zarr")
    IO().write_pyramid(pyr, path, overwrite=True)
    assert _written_translations(path) == [[0.0, 0.0], [0.0, 0.0]]


def test_stored_translations_survive_a_read_and_write(tmp_path):
    base = np.arange(4096, dtype="uint16").reshape(64, 64)
    pyr = Pyramid().from_arrays([base, base[::2, ::2]], axis_order="yx",
                                unit_list=["micrometer"] * 2, scales=[[0.5, 0.5], [1.0, 1.0]])
    pyr.meta.update_translation("0", [1, 2])
    pyr.meta.update_translation("1", [7, 7])
    first, second = str(tmp_path / "a.ome.zarr"), str(tmp_path / "b.ome.zarr")
    IO().write_pyramid(pyr, first, overwrite=True)
    IO().write_pyramid(IO().read_pyramid(first), second, overwrite=True)
    assert _written_translations(first) == [[1.0, 2.0], [7.0, 7.0]]
    assert _datasets(second) == _datasets(first)                     # byte for byte


# from_image / from_file positions ------------------------------------------------------

def _file(position, levels=1, axes="yx", pixel=0.5):
    data = np.arange(4096, dtype="uint16").reshape(64, 64)
    meta = SimpleNamespace(pixel_sizes={"y": Q(pixel, "micrometer"), "x": Q(pixel, "micrometer")},
                           position=position, channels=[], index_values={})
    lv = [data[::2 ** k, ::2 ** k] for k in range(1, levels)]
    return SimpleNamespace(images=[FakeImage(data, axes, meta, levels=lv)], path="f.czi")


def test_an_image_at_zero_writes_zero_on_every_level(tmp_path):
    coll = PyramidCollection.from_file(_file({"y": Q(0, "micrometer"), "x": Q(0, "micrometer")},
                                             levels=3))
    (pyr,) = coll.values()
    assert [pyr.meta.get_translation(p) for p in pyr.meta.resolution_paths] == [[0.0, 0.0]] * 3
    path = str(tmp_path / "zero.ome.zarr")
    IO().write_pyramid(pyr, path, overwrite=True)
    assert _written_translations(path) == [[0.0, 0.0]] * 3


def test_vendor_levels_keep_level_zeros_translation():
    # a stored (vendor) pyramid: alignment 0 on every level, whatever its scale
    coll = PyramidCollection.from_file(_file({"y": Q(5, "micrometer"), "x": Q(3000, "nanometer")},
                                             levels=3))
    (pyr,) = coll.values()
    assert [pyr.meta.get_scale(p) for p in pyr.meta.resolution_paths] == \
        [[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]
    assert [pyr.meta.get_translation(p) for p in pyr.meta.resolution_paths] == [[5.0, 3.0]] * 3


def test_a_position_along_one_axis_leaves_the_others_at_zero():
    (pyr,) = PyramidCollection.from_file(_file({"x": Q(3, "micrometer")})).values()
    assert pyr.meta.get_translation("0") == [0.0, 3.0]


def test_an_image_downscaled_afresh_gets_the_window_alignment():
    pyr = Pyramid.from_image(_file({"y": Q(5, "micrometer"), "x": Q(3, "micrometer")}).images[0])
    full = pyr.downscale(n_layers=3, scale_factor=2, downscale_method="mean", defer=False)
    assert [full.meta.get_translation(p) for p in full.meta.resolution_paths] == _expected("mean")


def test_an_unconvertible_position_unit_raises():
    with pytest.raises(PositionUnitError) as err:
        PyramidCollection.from_file(_file({"x": Q(3, "pixel")}))
    assert isinstance(err.value, ValueError)
    text = str(err.value)
    assert "'x'" in text and "'pixel'" in text and "'micrometer'" in text and "'img'" in text
    assert "image.metadata.position" in text


def test_a_quantity_without_conversion_in_another_unit_raises():
    class Bare:                                    # no .to(): cannot be converted
        value, unit = 3.0, "millimeter"
    with pytest.raises(PositionUnitError, match="'millimeter'"):
        Pyramid.from_image(_file({"x": Bare()}).images[0])
