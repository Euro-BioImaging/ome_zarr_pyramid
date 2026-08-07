"""`Pyramid.isel` / `__getitem__`: int, slice, and list (fancy) selection,
coordinate-metadata updates, omero channel subsetting, and error handling."""

import numpy as np
import pytest


# --- int / slice (axis-dropping and strided) ---

def test_int_drops_axis(pyr3d, compute):
    pyr, data = pyr3d
    out = pyr.isel(c=1)
    assert out.axes == "yx"
    np.testing.assert_array_equal(compute(out), data[1])


def test_slice_keeps_axis(pyr3d, compute):
    pyr, data = pyr3d
    out = pyr.isel(c=slice(1, 3))
    assert out.axes == "cyx"
    np.testing.assert_array_equal(compute(out), data[1:3])


def test_strided_slice_rescales(pyr4d):
    pyr, _ = pyr4d
    out = pyr.isel(z=slice(0, 8, 2))                    # step 2 on z (scale 2.0)
    z = out.axes.index("z")
    assert out.meta.get_base_scale()[z] == pytest.approx(4.0)


# --- list (fancy) selection: the axis is KEPT ---

def test_list_keeps_axis(pyr3d, compute):
    pyr, data = pyr3d
    out = pyr.isel(c=[2, 0])                            # note: order preserved
    assert out.axes == "cyx"
    assert out.shape[0] == 2
    np.testing.assert_array_equal(compute(out), data[[2, 0]])


def test_single_element_list_keeps_axis(pyr3d):
    pyr, _ = pyr3d
    kept = pyr.isel(c=[1])
    dropped = pyr.isel(c=1)
    assert kept.axes == "cyx" and kept.shape[0] == 1
    assert dropped.axes == "yx"


def test_negative_indices(pyr3d, compute):
    pyr, data = pyr3d
    np.testing.assert_array_equal(compute(pyr.isel(c=[-1, -2])), data[[2, 1]])


def test_non_uniform_list_gathers(pyr4d, compute):
    pyr, data = pyr4d
    out = pyr.isel(z=[0, 1, 5])
    np.testing.assert_array_equal(compute(out), data[:, [0, 1, 5]])


def test_multi_list_orthogonal(pyr3d, compute):
    pyr, data = pyr3d
    out = pyr.isel(c=[0, 2], y=[0, 5, 10])
    expected = data[[0, 2]][:, [0, 5, 10], :]           # orthogonal, not cross-product
    assert out.shape == expected.shape
    np.testing.assert_array_equal(compute(out), expected)


def test_list_and_int_mix(pyr3d, compute):
    pyr, data = pyr3d
    out = pyr.isel(c=[0, 2], x=3)
    assert out.axes == "cy"
    np.testing.assert_array_equal(compute(out), data[[0, 2]][:, :, 3])


# --- coordinate metadata for lists ---

def test_uniform_list_scale_exact(pyr4d):
    pyr, _ = pyr4d
    out = pyr.isel(z=[0, 2, 4, 6])                      # step 2 -> exact rescale
    z = out.axes.index("z")
    assert out.meta.get_base_scale()[z] == pytest.approx(4.0)


def test_non_uniform_list_keeps_scale(pyr4d):
    pyr, _ = pyr4d
    out = pyr.isel(z=[0, 1, 5])                         # irregular -> scale unchanged
    z = out.axes.index("z")
    assert out.meta.get_base_scale()[z] == pytest.approx(2.0)


# --- omero channel subsetting ---

def test_omero_subset_on_channel_list(make_pyramid):
    pyr, _ = make_pyramid(shape=(4, 16, 16), axis_order="cyx", channels=True)
    out = pyr.isel(c=[1, 3])
    labels = [c["label"] for c in out.meta.get_channels()]
    assert labels == ["ch1", "ch3"]


# --- __getitem__ routes through isel ---

def test_getitem_list(pyr3d, compute):
    pyr, data = pyr3d
    np.testing.assert_array_equal(compute(pyr[[2, 0]]), data[[2, 0]])


def test_getitem_slice(pyr3d, compute):
    pyr, data = pyr3d
    np.testing.assert_array_equal(compute(pyr[1:3]), data[1:3])


# --- long-name axis aliases ---

def test_channels_alias(pyr3d, compute):
    pyr, data = pyr3d
    np.testing.assert_array_equal(compute(pyr.isel(channels=[0, 2])), data[[0, 2]])


# --- errors ---

def test_error_float_index(pyr3d):
    pyr, _ = pyr3d
    with pytest.raises(TypeError):
        pyr.isel(c=1.5)


def test_error_unknown_axis(pyr3d):
    pyr, _ = pyr3d
    with pytest.raises(ValueError):
        pyr.isel(q=0)


def test_error_non_integer_list(pyr3d):
    pyr, _ = pyr3d
    with pytest.raises(TypeError):
        pyr.isel(c=[1.5, 2.0])


def test_error_out_of_range_list(pyr3d):
    pyr, _ = pyr3d
    with pytest.raises(IndexError):
        pyr.isel(c=[99])


def test_error_empty_list(pyr3d):
    pyr, _ = pyr3d
    with pytest.raises(ValueError):
        pyr.isel(c=[])
