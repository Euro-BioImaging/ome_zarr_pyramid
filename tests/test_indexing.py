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


# --- crops as eubi-bridge makes them (roadmap C2) ---

def test_a_channel_crop_keeps_the_channels_in_step_and_moves_nothing():
    from ome_zarr_pyramid import Pyramid
    a = np.arange(4 * 6 * 16 * 16, dtype="u2").reshape(4, 6, 16, 16)
    pyr = Pyramid().from_arrays([a, a[:, ::2, ::2, ::2]], axis_order="czyx",
                                scales=[[1, 2.0, 0.5, 0.5], [1, 4.0, 1.0, 1.0]], version="0.5")
    for p in pyr.meta.resolution_paths:
        pyr.meta.update_translation(p, [0.0, 10.0, 100.0, 200.0])
    pyr = pyr.set_channels({i: {"label": f"ch{i}"} for i in range(4)})
    for sel, labels in ((slice(1, 3), ["ch1", "ch2"]), (slice(0, 4, 2), ["ch0", "ch2"]),
                        ([2, 0], ["ch2", "ch0"]), (1, ["ch1"])):
        out = pyr.isel(c=sel)
        assert [c["label"] for c in out.meta.metadata["omero"]["channels"]] == labels
        for p in out.meta.resolution_paths:                 # every level
            assert out.meta.get_translation(p)[-3:] == [10.0, 100.0, 200.0]
            if "c" in out.axes:
                assert out.meta.get_translation(p)[0] == 0.0       # a channel has no place
                assert out.meta.get_scale(p)[0] == 1               # ... nor a step
    # a spatial crop moves the position by start * pixel size, on every level
    out = pyr.isel(z=slice(2, 6), y=slice(4, 12))
    assert out.meta.get_translation("0") == [0.0, 14.0, 102.0, 200.0]
    assert out.meta.get_translation("1") == [0.0, 14.0, 102.0, 200.0]


def test_isel_adds_no_zero_translation():
    """A level without a translation gets none from a selection starting at 0 (squeeze,
    a crop from the origin); a crop elsewhere gets the shifted one."""
    import numpy as np
    from ome_zarr_pyramid import Pyramid
    pyr = Pyramid().from_arrays([np.zeros((1, 1, 4, 32, 32), "u1")], axis_order="tczyx",
                                scales=[[1, 1, 2, 0.5, 0.5]])
    ct = lambda p: p.meta.multiscales["datasets"][0]["coordinateTransformations"]
    assert [c["type"] for c in ct(pyr.squeeze())] == ["scale"]
    assert [c["type"] for c in ct(pyr.isel(y=slice(0, 8)))] == ["scale"]
    moved = pyr.isel(y=slice(4, 8))
    assert ct(moved)[1] == {"type": "translation", "translation": [0.0, 0.0, 0.0, 2.0, 0.0]}
