"""Pyramid construction via `from_arrays`: axes, shape, dtype, layers, scales."""

import numpy as np


def test_from_arrays_basic(pyr3d):
    pyr, data = pyr3d
    assert pyr.axes == "cyx"
    assert pyr.meta.ndim == 3
    assert pyr.shape == data.shape
    assert str(pyr.dtype) == "float32"
    assert pyr.nlayers == 1
    assert pyr.meta.resolution_paths == ["0"]


def test_dask_arrays_match_source(pyr3d, compute):
    pyr, data = pyr3d
    assert set(pyr.dask_arrays) == {"0"}
    np.testing.assert_array_equal(compute(pyr), data)


def test_dynamic_arrays_match_source(pyr3d):
    pyr, data = pyr3d
    dyn = pyr.dynamic_arrays["0"]
    assert type(dyn).__name__ == "DynamicArray"
    np.testing.assert_array_equal(np.asarray(dyn.compute()), data)


def test_scales_recorded(pyr4d):
    pyr, _ = pyr4d
    assert pyr.meta.get_base_scale() == [1.0, 2.0, 0.5, 0.5]


def test_axis_order_variants(make_pyramid):
    pyr, data = make_pyramid(shape=(4, 6, 8, 8, 8), axis_order="tczyx")
    assert pyr.axes == "tczyx"
    assert pyr.meta.ndim == 5
    assert pyr.shape == (4, 6, 8, 8, 8)


def test_layers_property(pyr3d):
    pyr, _ = pyr3d
    assert set(pyr.layers) == {"0"}
