"""IO: write a Pyramid to an OME-Zarr store and read it back, checking that
pixel data and NGFF metadata survive the round trip (both 0.4 and 0.5)."""

import numpy as np
import pytest

from ome_zarr_pyramid.core.io import IO


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_roundtrip_data_and_axes(make_pyramid, zarr_path, compute, version):
    pyr, data = make_pyramid(shape=(2, 8, 16, 16), axis_order="czyx",
                             scales=[[1.0, 2.0, 0.5, 0.5]])
    pyr.meta.version = version
    IO().write_pyramid(pyr, zarr_path, overwrite=True)
    back = IO().read_pyramid(zarr_path)

    assert back.axes == "czyx"
    assert back.meta.get_base_scale() == [1.0, 2.0, 0.5, 0.5]
    np.testing.assert_array_equal(compute(back), data)


def test_roundtrip_preserves_channels(make_pyramid, zarr_path):
    pyr, _ = make_pyramid(shape=(3, 16, 16), axis_order="cyx", channels=True)
    IO().write_pyramid(pyr, zarr_path, overwrite=True)
    back = IO().read_pyramid(zarr_path)
    assert [c["label"] for c in back.meta.get_channels()] == ["ch0", "ch1", "ch2"]


def test_overwrite(make_pyramid, zarr_path):
    pyr, _ = make_pyramid(shape=(2, 16, 16), axis_order="cyx")
    IO().write_pyramid(pyr, zarr_path, overwrite=True)
    # a second write to the same path must succeed with overwrite=True
    IO().write_pyramid(pyr, zarr_path, overwrite=True)
    assert IO().read_pyramid(zarr_path).nlayers == 1


def test_deferred_downscale_expands_on_write(make_pyramid, zarr_path):
    """downscale(defer=True) records a plan (nlayers stays 1); the extra levels
    are materialized only when the pyramid is written."""
    pyr, _ = make_pyramid(shape=(2, 64, 64), axis_order="cyx", scales=[[1.0, 1.0, 1.0]])
    planned = pyr.downscale(n_layers=3, defer=True)
    assert planned.nlayers == 1                         # nothing built yet

    IO().write_pyramid(planned, zarr_path, overwrite=True)
    back = IO().read_pyramid(zarr_path)
    assert back.nlayers == 3
    shapes = [back.dask_arrays[p].shape for p in back.meta.resolution_paths]
    assert shapes[0] == (2, 64, 64)
    assert shapes[1][-1] < shapes[0][-1]                # coarser levels shrink
