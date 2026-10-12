"""Downscale methods x downscale backends: every level equals ``ts.downsample`` of level 0.

Two backends compute the window methods: 'tensorstore' (default) and 'dyna' (dyna's
``coarsen``, which reproduces TensorStore bit for bit). The reference for each level is
TensorStore's downsample of level 0 by the CUMULATIVE factor, so this also pins the
cascade (simple/min/max levels are written from the stored level above), the ceil edge
rule on odd extents, dask pyramids and eager (defer=False) levels.
"""
import numpy as np
import pytest
import tensorstore as ts

from ome_zarr_pyramid import IO, Pyramid
from ome_zarr_pyramid.utils.scale import DOWNSCALE_METHODS

SHAPE = (1, 2, 21, 43, 45)          # odd extents: partial windows at every far edge
AXES = "tczyx"
LEVELS = 3
FACTOR = (1, 1, 2, 2, 2)


def _data(dtype="uint16"):
    rng = np.random.default_rng(0)
    if dtype.startswith("float"):
        return rng.normal(100, 30, SHAPE).astype(dtype)
    return rng.integers(0, 6, SHAPE).astype(dtype)      # few values: mode/median ties


def _reference(base, method, level):
    f = [fa ** level for fa in FACTOR]
    m = "stride" if method == "simple" else method
    return ts.downsample(ts.array(base), f, method=m).read().result()


def _check_levels(pyr_or_path, base, method):
    back = IO().read_pyramid(str(pyr_or_path)) if not isinstance(pyr_or_path, Pyramid) \
        else pyr_or_path
    paths = back.meta.resolution_paths
    assert len(paths) == LEVELS
    for i, p in enumerate(paths):
        got = np.asarray(back.layers[p][...] if not hasattr(back.layers[p], "compute")
                         else back.layers[p].compute())
        want = _reference(base, method, i)
        assert got.shape == want.shape, (method, i, got.shape, want.shape)
        np.testing.assert_array_equal(got, want, err_msg=f"{method} level {i}")


@pytest.mark.parametrize("downscale_backend", ["tensorstore", "dyna"])
@pytest.mark.parametrize("method", DOWNSCALE_METHODS)
def test_written_levels_match_tensorstore(tmp_path, method, downscale_backend):
    base = _data()
    pyr = Pyramid().from_arrays([base], axis_order=AXES, version="0.5")
    out = tmp_path / "p.zarr"
    IO().write_pyramid(pyr.downscale(n_layers=LEVELS, scale_factor=FACTOR,
                                     downscale_method=method,
                                     downscale_backend=downscale_backend),
                       str(out), overwrite=True)
    _check_levels(out, base, method)


@pytest.mark.parametrize("downscale_backend", ["tensorstore", "dyna"])
@pytest.mark.parametrize("method", ["mean", "min", "max"])
def test_float_levels_match_tensorstore(tmp_path, method, downscale_backend):
    base = _data("float32")
    pyr = Pyramid().from_arrays([base], axis_order=AXES, version="0.5")
    out = tmp_path / "f.zarr"
    IO().write_pyramid(pyr.downscale(n_layers=LEVELS, scale_factor=FACTOR,
                                     downscale_method=method,
                                     downscale_backend=downscale_backend),
                       str(out), overwrite=True)
    _check_levels(out, base, method)


@pytest.mark.parametrize("downscale_backend", ["tensorstore", "dyna"])
@pytest.mark.parametrize("method", DOWNSCALE_METHODS)
def test_eager_levels_match_tensorstore(method, downscale_backend):
    base = _data()
    pyr = Pyramid().from_arrays([base], axis_order=AXES, version="0.5")
    full = pyr.downscale(n_layers=LEVELS, scale_factor=FACTOR, downscale_method=method,
                         downscale_backend=downscale_backend, defer=False)
    _check_levels(full, base, method)


@pytest.mark.parametrize("method", DOWNSCALE_METHODS)
def test_dask_pyramid_levels_match_tensorstore(tmp_path, method):
    """Dask levels stay dask and give the same pixels (they used to truncate integer
    means and drop partial edge windows)."""
    da = pytest.importorskip("dask.array")
    base = _data()
    pyr = Pyramid().from_arrays([da.from_array(base, chunks=(1, 1, 8, 16, 16))],
                                axis_order=AXES, version="0.5")
    full = pyr.downscale(n_layers=LEVELS, scale_factor=FACTOR, downscale_method=method,
                         defer=False)
    assert all(hasattr(full.layers[p], "dask") for p in full.meta.resolution_paths)
    _check_levels(full, base, method)
    out = tmp_path / "d.zarr"
    IO().write_pyramid(pyr.downscale(n_layers=LEVELS, scale_factor=FACTOR,
                                     downscale_method=method), str(out), overwrite=True)
    _check_levels(out, base, method)


def test_partial_edge_window_is_kept():
    pyr = Pyramid().from_arrays([np.ones((1, 1, 1, 101, 101), "uint8")], axis_order=AXES,
                                version="0.5")
    full = pyr.downscale(n_layers=2, scale_factor=(1, 1, 1, 2, 2), downscale_method="mean")
    assert full.layers["1"].shape == (1, 1, 1, 51, 51)


def test_unknown_method_or_backend_is_refused():
    pyr = Pyramid().from_arrays([_data()], axis_order=AXES, version="0.5")
    with pytest.raises(ValueError, match="downscale_method"):
        pyr.downscale(downscale_method="gauss")
    with pytest.raises(ValueError, match="downscale_backend"):
        pyr.downscale(downscale_method="mean", downscale_backend="zarrs")


@pytest.mark.parametrize("downscale_backend,expect_dyna", [("tensorstore", False), ("dyna", True)])
def test_backend_is_really_used(tmp_path, monkeypatch, downscale_backend, expect_dyna):
    """The backend choice reaches the writer: dyna's coarsen runs only for 'dyna'."""
    import dyna_zarr.operations as dyna_ops
    calls = []
    orig = dyna_ops.coarsen
    monkeypatch.setattr(dyna_ops, "coarsen", lambda *a, **k: calls.append(1) or orig(*a, **k))
    pyr = Pyramid().from_arrays([_data()], axis_order=AXES, version="0.5")
    IO().write_pyramid(pyr.downscale(n_layers=LEVELS, scale_factor=FACTOR,
                                     downscale_method="median",
                                     downscale_backend=downscale_backend),
                       str(tmp_path / "r.zarr"), overwrite=True)
    assert bool(calls) is expect_dyna
