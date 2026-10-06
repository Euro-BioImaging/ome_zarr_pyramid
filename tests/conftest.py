"""Shared fixtures for the ome_zarr_pyramid test suite.

The public surface under test is the working core: `Pyramid` (construction,
indexing, downscaling, metadata, operators) and `IO` (NGFF read/write). Legacy
process/creation/CLI modules are intentionally out of scope.
"""

import os
import sys


class _NoDask:
    """``OZP_BLOCK_DASK=1``: dask (an optional extra) is not installed, as far
    as this test run can tell -- the whole suite runs without it, and the tests
    of dask-only features skip (``pytest.importorskip("dask")``)."""

    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in ("dask", "distributed"):
            raise ModuleNotFoundError(f"No module named {name!r} (blocked: OZP_BLOCK_DASK)",
                                      name=name)
        return None


if os.environ.get("OZP_BLOCK_DASK") == "1":
    sys.meta_path.insert(0, _NoDask())

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import zarr  # noqa: E402

from ome_zarr_pyramid.core.pyramid import Pyramid  # noqa: E402


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
    """`compute(pyr, level="0")` -> the materialized NumPy array of a level
    (zarr, dask, DynamicArray or TensorStore: with or without dask)."""
    def _c(pyr, level="0"):
        return np.asarray(pyr.layers[level])
    return _c


@pytest.fixture
def zarr_path(tmp_path):
    """A fresh, not-yet-created .zarr path under the test's tmp dir."""
    return str(tmp_path / "image.zarr")
