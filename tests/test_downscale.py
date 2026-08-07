"""Downscaling, level selection, and storage rechunking."""

import numpy as np
import pytest


def test_downscale_sync_builds_levels(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 64, 64), axis_order="cyx", scales=[[1.0, 1.0, 1.0]])
    ds = pyr.downscale(n_layers=3, defer=False)
    assert ds.nlayers == 3
    shapes = [ds.dask_arrays[p].shape for p in ds.meta.resolution_paths]
    assert shapes[0] == (2, 64, 64)
    # each level halves the spatial axes (2x factor)
    assert shapes[1] == (2, 32, 32)
    assert shapes[2] == (2, 16, 16)


def test_downscale_scale_metadata_doubles(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 64, 64), axis_order="cyx", scales=[[1.0, 0.5, 0.5]])
    ds = pyr.downscale(n_layers=2, defer=False)
    assert ds.meta.get_scale("1") == pytest.approx([1.0, 1.0, 1.0])   # 0.5 * 2


def test_downscale_defer_is_lazy(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 64, 64), axis_order="cyx")
    planned = pyr.downscale(n_layers=3, defer=True)
    assert planned.nlayers == 1                                   # no levels materialized


def test_min_dimension_size_caps_levels(make_pyramid):
    pyr, _ = make_pyramid(shape=(1, 128, 128), axis_order="cyx")
    ds = pyr.downscale(min_dimension_size=32, defer=False)
    smallest = ds.dask_arrays[ds.meta.resolution_paths[-1]].shape
    assert min(smallest[-2:]) >= 32


def test_get_downscaled_pyramid_preserves_level0(make_pyramid, compute):
    """get_downscaled_pyramid() must reproduce the source at level 0 (data + scale)."""
    pyr, data = make_pyramid(shape=(2, 64, 64), axis_order="cyx", scales=[[1.0, 0.5, 0.5]])
    ds = pyr.get_downscaled_pyramid()
    assert ds.meta.resolution_paths[0] == "0"
    assert ds.meta.get_base_scale() == [1.0, 0.5, 0.5]
    np.testing.assert_array_equal(compute(ds), data)


def test_get_downscaled_pyramid_auto_depth(make_pyramid):
    """With no downscaler configured, get_downscaled_pyramid() chooses the depth
    automatically: keep halving until the largest spatial dim would drop below 64
    (the `min_dimension_size` default) - not a single no-op level."""
    pyr, _ = make_pyramid(shape=(3, 512, 512), axis_order="cyx")
    ds = pyr.get_downscaled_pyramid()
    shapes = [ds.dask_arrays[p].shape for p in ds.meta.resolution_paths]
    assert ds.nlayers == 4                              # 512 -> 256 -> 128 -> 64
    assert min(shapes[-1][-2:]) >= 64
    assert min(shapes[-1][-2:]) < 128
    assert all(s[0] == 3 for s in shapes)               # the c axis is never downscaled


def test_get_downscaled_pyramid_small_stays_single(make_pyramid):
    """A base already at/under the min-dimension threshold yields a single level."""
    pyr, _ = make_pyramid(shape=(2, 48, 48), axis_order="cyx")
    assert pyr.get_downscaled_pyramid().nlayers == 1


def test_select_levels_contiguous(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 64, 64), axis_order="cyx")
    ds = pyr.downscale(n_layers=4, defer=False)
    sub = ds.select_levels(1, 2)
    assert sub.nlayers == 2
    # the selected finest level becomes level 0 of the output
    assert sub.dask_arrays["0"].shape == ds.dask_arrays["1"].shape


def test_select_levels_non_contiguous(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 128, 128), axis_order="cyx")
    ds = pyr.downscale(n_layers=4, defer=False)      # 128,64,32,16
    sub = ds.select_levels(0, 2, 3)                  # skip level 1
    assert sub.nlayers == 3
    assert sub.dask_arrays["0"].shape == ds.dask_arrays["0"].shape
    assert sub.dask_arrays["1"].shape == ds.dask_arrays["2"].shape
    assert sub.dask_arrays["2"].shape == ds.dask_arrays["3"].shape


def test_select_levels_accepts_list_and_slice(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 128, 128), axis_order="cyx")
    ds = pyr.downscale(n_layers=4, defer=False)
    assert ds.select_levels([0, 2]).nlayers == 2
    assert ds.select_levels(slice(1, 3)).nlayers == 2
    assert ds.select_levels(2).nlayers == 1          # single level
    assert ds.select_levels(-1).nlayers == 1         # negative index (coarsest)


def test_rechunk_chunk_shape(pyr3d):
    pyr, _ = pyr3d
    rc = pyr.rechunk(chunk_shape=(3, 8, 8))
    assert rc.dask_arrays["0"].chunksize == (3, 8, 8)


def test_rechunk_chunk_size_mb(pyr3d):
    pyr, _ = pyr3d
    rc = pyr.rechunk(chunk_size_mb=0.001)
    # a tiny target produces chunks no larger than the array
    assert all(c <= s for c, s in zip(rc.dask_arrays["0"].chunksize, pyr.shape))
