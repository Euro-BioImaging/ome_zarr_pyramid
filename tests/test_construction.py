"""Pyramid construction via `from_arrays`: axes, shape, dtype, layers, scales."""

import numpy as np
import pytest

from ome_zarr_pyramid.core.pyramid import Pyramid


def test_from_arrays_basic(pyr3d):
    pyr, data = pyr3d
    assert pyr.axes == "cyx"
    assert pyr.meta.ndim == 3
    assert pyr.shape == data.shape
    assert str(pyr.dtype) == "float32"
    assert pyr.nlayers == 1
    assert pyr.meta.resolution_paths == ["0"]


def test_dask_arrays_match_source(pyr3d, compute):
    import pytest
    pytest.importorskip("dask")
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


def test_dask_arrays_refuses_dyna_layers(tmp_path):
    """`dask_arrays` must REFUSE a dyna layer rather than materialize the level.

    There is no lazy dask view of a pull-model chain, so converting one means reading the
    whole level - a silent, unbounded cost triggered by touching a property. Callers should
    use `dynamic_arrays`, or write the pyramid first (zarr layers convert lazily).
    """
    import numpy as np
    import pytest
    from ome_zarr_pyramid import IO, Pyramid

    pytest.importorskip("dyna_zarr")
    pytest.importorskip("dask")
    data = (np.arange(8 * 16 * 16) % 7).reshape(1, 1, 8, 16, 16).astype("int32")
    path = tmp_path / "src.zarr"
    IO().write_pyramid(Pyramid().from_array(data, scale=[1, 1, 1, 1, 1]),
                       str(path), overwrite=True)
    disk = IO().read_pyramid(str(path))

    # zarr layers: lazy, no error
    assert disk.dask_arrays["0"].shape == data.shape

    # a pyramid actually BUILT from DynamicArray layers -> refused
    # (`.layers` returns a fresh dict, so assigning into it would not stick)
    dyna_pyr = Pyramid().from_arrays(
        [disk.dynamic_arrays[p] for p in disk.meta.resolution_paths],
        axis_order=disk.meta.axis_order,
        unit_list=disk.meta.unit_list,
        scales=[disk.meta.get_scale(p) for p in disk.meta.resolution_paths],
    )
    assert type(dyna_pyr.layers["0"]).__name__ == "DynamicArray"
    with pytest.raises(TypeError, match="DynamicArray"):
        dyna_pyr.dask_arrays


class _ReaderSource:
    """A reader's region source: shape, dtype, __getitem__ - no zarr, no dask."""

    def __init__(self, data, read_unit=None):
        self._data, self.shape, self.dtype = data, data.shape, data.dtype
        self.read_unit = read_unit

    def __getitem__(self, key):
        return self._data[key]


def test_from_arrays_wraps_any_array_like_source(zarr_path):
    from dyna_zarr import DynamicArray
    from ome_zarr_pyramid.core.io import IO
    data = np.random.default_rng(0).integers(0, 999, (2, 16, 64, 48), dtype="uint16")
    pyr = Pyramid().from_arrays([_ReaderSource(data, read_unit=(1, 8, 32, 48))],
                                axis_order="czyx", version="0.5")
    assert isinstance(pyr.layers["0"], DynamicArray)
    assert pyr.layers["0"].chunks == (1, 8, 32, 48)          # the reader's own unit
    IO().write_pyramid(pyr.downscale(n_layers=2, defer=True), zarr_path, overwrite=True)
    back = IO().read_pyramid(zarr_path)
    np.testing.assert_array_equal(np.asarray(back.layers["0"][...]), data)
    single = Pyramid().from_arrays(_ReaderSource(data), axis_order="czyx")   # no list
    assert single.nlayers == 1 and single.shape == data.shape


def test_from_arrays_refuses_an_axis_order_of_the_wrong_rank():
    with pytest.raises(ValueError, match="one axis letter per dimension"):
        Pyramid().from_arrays([np.zeros((2, 3, 4, 5))], axis_order="zyx")
