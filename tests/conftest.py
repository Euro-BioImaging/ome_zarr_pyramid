"""Shared fixtures for the ome_zarr_pyramid test suite.

The public surface under test is the working core: `Pyramid` (construction,
indexing, downscaling, metadata, operators) and `IO` (NGFF read/write). Legacy
process/creation/CLI modules are intentionally out of scope.
"""

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid.core.pyramid import Pyramid


def _default_chunks(shape):
    """A sensible chunking: split each axis in half once it is larger than 8."""
    return tuple(s if s <= 8 else (s + 1) // 2 for s in shape)


@pytest.fixture
def make_pyramid():
    """Factory: build a single-level `Pyramid` from a fresh random array.

    Returns `(pyramid, numpy_data)` so a test can compare results against the
    exact source array. Layers are zarr arrays (so both `dask_arrays` and
    `dynamic_arrays` are exercisable).
    """
    def _make(shape=(3, 32, 32), axis_order="cyx", scales=None, chunks=None,
              dtype="float32", seed=0, channels=False):
        data = (np.random.default_rng(seed).random(shape) * 100 + 1).astype(dtype)
        if chunks is None:
            chunks = _default_chunks(shape)
        layer = zarr.array(np.ascontiguousarray(data), chunks=chunks)
        pyr = Pyramid().from_arrays([layer], axis_order=axis_order, scales=scales)
        if channels and "c" in axis_order:
            n = shape[axis_order.index("c")]
            pyr = pyr.set_channels({i: {"label": f"ch{i}"} for i in range(n)})
        return pyr, data
    return _make


@pytest.fixture
def pyr3d(make_pyramid):
    """A 3-D `cyx` pyramid (3 channels, 32x32) with anisotropic-free scale."""
    return make_pyramid(shape=(3, 32, 32), axis_order="cyx", scales=[[1.0, 0.5, 0.5]])


@pytest.fixture
def pyr4d(make_pyramid):
    """A 4-D `czyx` pyramid with a non-trivial z scale (for level/coord tests)."""
    return make_pyramid(shape=(2, 8, 16, 16), axis_order="czyx",
                        scales=[[1.0, 2.0, 0.5, 0.5]])


@pytest.fixture
def compute():
    """`compute(pyr, level="0")` -> the materialized NumPy array of a level."""
    def _c(pyr, level="0"):
        return np.asarray(pyr.dask_arrays[level].compute())
    return _c


@pytest.fixture
def zarr_path(tmp_path):
    """A fresh, not-yet-created .zarr path under the test's tmp dir."""
    return str(tmp_path / "image.zarr")
