"""Elementwise operator dunders on Pyramid (native dask, all levels, metadata
preserved). See the pyramid-operators feature."""

import numpy as np


def test_add_scalar(pyr3d, compute):
    pyr, data = pyr3d
    out = pyr + 1
    assert out.axes == "cyx"
    np.testing.assert_allclose(compute(out), data + 1)


def test_subtract_and_divide(pyr3d, compute):
    pyr, data = pyr3d
    np.testing.assert_allclose(compute(pyr - 2.0), data - 2.0)
    np.testing.assert_allclose(compute(pyr / 2.0), data / 2.0)


def test_multiply_two_pyramids(make_pyramid, compute):
    a, da = make_pyramid(shape=(2, 16, 16), axis_order="cyx", seed=1)
    b, db = make_pyramid(shape=(2, 16, 16), axis_order="cyx", seed=2)
    np.testing.assert_allclose(compute(a * b), da * db)


def test_comparison_yields_bool(pyr3d, compute):
    pyr, data = pyr3d
    out = pyr > 10
    assert compute(out).dtype == np.bool_
    np.testing.assert_array_equal(compute(out), data > 10)


def test_invert_mask(make_pyramid, compute):
    mask, data = make_pyramid(shape=(2, 16, 16), axis_order="cyx")
    m = mask > 50
    np.testing.assert_array_equal(compute(~m), ~(data > 50))


def test_operator_preserves_metadata(pyr4d):
    pyr, _ = pyr4d
    out = pyr * 2
    assert out.axes == "czyx"
    assert out.meta.get_base_scale() == pyr.meta.get_base_scale()


def test_operator_applies_to_all_levels(make_pyramid, compute):
    pyr, _ = make_pyramid(shape=(2, 64, 64), axis_order="cyx")
    ds = pyr.downscale(n_layers=3, defer=False)
    out = ds + 5
    assert out.nlayers == 3
    for p in out.meta.resolution_paths:
        np.testing.assert_allclose(compute(out, p), compute(ds, p) + 5)
