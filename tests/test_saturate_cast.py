"""Pyramid.saturate_cast and write_pyramid(dtype=): narrowing never wraps.

uint16 4000 must land as uint8 255, not 160; the omero windows follow the same
mapping; a deferred downscale plan and attached label images survive the cast.
"""
import numpy as np
import pytest
import zarr

from ome_zarr_pyramid.core.io import IO
from ome_zarr_pyramid.core.pyramid import Pyramid

SHAPE, AXES = (2, 8, 32, 32), "czyx"


@pytest.fixture
def data():
    d = np.random.default_rng(0).integers(0, 1000, size=SHAPE, dtype="uint16")
    d[0, 0, 0, :4] = [4000, 65535, 255, 256]
    return d


def _sat(a):
    return np.minimum(a, 255).astype("uint8")


def _windows(pyr):
    return [ch["window"] for ch in pyr.meta.metadata["omero"]["channels"]]


@pytest.mark.parametrize("backing", ["numpy", "dask"])
def test_saturate_cast_levels_and_windows(data, backing):
    arr = data
    if backing == "dask":
        da = pytest.importorskip("dask.array")
        arr = da.from_array(data, chunks=(1, 4, 16, 16))
    pyr = Pyramid().from_arrays([arr], axis_order=AXES, version="0.5")
    before = _windows(pyr)
    out = pyr.saturate_cast("uint8")
    level = out.layers["0"]
    assert np.dtype(level.dtype) == np.uint8
    np.testing.assert_array_equal(np.asarray(level[...]), _sat(data))
    assert pyr.layers["0"].dtype == np.uint16                # input untouched
    for b, a in zip(before, _windows(out)):
        for key in ("min", "max", "start", "end"):
            assert a[key] == min(max(round(b[key]), 0), 255)


def test_write_pyramid_dtype_saturates_every_level(data, zarr_path):
    pyr = Pyramid().from_arrays([data], axis_order=AXES, version="0.5")
    IO().write_pyramid(pyr.downscale(n_layers=2, scale_factor=(1, 1, 2, 2), defer=True),
                       zarr_path, overwrite=True, dtype="uint8")
    g = zarr.open_group(zarr_path, mode="r")
    np.testing.assert_array_equal(g["0"][...], _sat(data))
    np.testing.assert_array_equal(g["1"][...], _sat(data)[:, :, ::2, ::2])
    assert g["1"].dtype == np.uint8
    back = IO().read_pyramid(zarr_path)
    assert all(w["max"] <= 255 for w in _windows(back))


def test_write_pyramid_dtype_keeps_attached_labels(data, zarr_path):
    lab = np.zeros(SHAPE[1:], dtype="uint32")
    lab[2:6, 8:20, 8:20] = 70000                               # beyond uint8: must NOT be cast
    labels = Pyramid().from_label_arrays([lab], axis_order=AXES[1:], version="0.5")
    pyr = Pyramid().from_arrays([data], axis_order=AXES, version="0.5")
    IO().write_pyramid(pyr.add_image_label(labels, name="seg"), zarr_path, overwrite=True,
                       dtype="uint8")
    seg = IO().read_pyramid(f"{zarr_path}/labels/seg")
    np.testing.assert_array_equal(np.asarray(seg.layers["0"][...]), lab)
    np.testing.assert_array_equal(zarr.open_group(zarr_path, mode="r")["0"][...], _sat(data))


def test_float_pyramid_to_integer_rounds(zarr_path):
    f = np.array([[254.7, -3.2], [np.nan, 1e9]], dtype="float32").reshape(1, 1, 2, 2)
    pyr = Pyramid().from_arrays([f], axis_order=AXES, version="0.5")
    IO().write_pyramid(pyr, zarr_path, overwrite=True, dtype="uint8")
    np.testing.assert_array_equal(zarr.open_group(zarr_path, mode="r")["0"][...].ravel(),
                                  [255, 0, 0, 255])
