"""write_pyramid(window_from_data=True): omero window start/end measured during the write.

The per-channel min/max of level 0 AS WRITTEN (after a dtype cast) become each channel's
window start/end, from the write's own read - no second pass over the output. The
window's min/max (the slider range) are kept.
"""
import json

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid

AXES = "czyx"
SHAPE = (3, 9, 41, 37)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    d = rng.integers(100, 50000, size=SHAPE, dtype="uint16")
    d[1] //= 9
    d[2] = np.minimum(d[2], 3000)
    return d


def _pyr(data, version="0.5", chunks=(1, 4, 16, 16)):
    zf = 3 if version == "0.5" else 2
    a = zarr.create_array(store=zarr.storage.MemoryStore(), shape=data.shape, chunks=chunks,
                          dtype=data.dtype, zarr_format=zf)
    a[...] = data
    return Pyramid().from_arrays([a], axis_order=AXES, version=version)


def _windows(path):
    attrs = zarr.open_group(str(path), mode="r").attrs.asdict()
    holder = attrs.get("ome", attrs)
    return [c["window"] for c in holder["omero"]["channels"]]


def _expect(d):
    return d.min(axis=(1, 2, 3)).tolist(), d.max(axis=(1, 2, 3)).tolist()


@pytest.mark.parametrize("version", ["0.5", "0.4"])
@pytest.mark.parametrize("plan", [True, False])
def test_windows_are_level0_minmax(data, tmp_path, version, plan):
    pyr = _pyr(data, version)
    if plan:
        pyr = pyr.downscale(n_layers=3)               # the one-pass writer
    IO().write_pyramid(pyr, str(tmp_path / "o.zarr"), overwrite=True, window_from_data=True)
    lows, highs = _expect(data)
    windows = _windows(tmp_path / "o.zarr")
    assert [w["start"] for w in windows] == lows and [w["end"] for w in windows] == highs
    # no channels given: defaults are made, their min/max (the dtype range) kept
    assert all((w["min"], w["max"]) == (0, 65535) for w in windows)
    assert all(type(w["start"]) is int for w in windows)


def test_the_plan_fallback_path(data, tmp_path):
    # a dask base keeps the old plan path (level 0 written first, levels from it)
    da = pytest.importorskip("dask.array")
    pyr = Pyramid().from_arrays([da.from_array(data, chunks=(1, 4, 16, 16))],
                                axis_order=AXES, version="0.5").downscale(n_layers=3)
    IO().write_pyramid(pyr, str(tmp_path / "o.zarr"), overwrite=True, window_from_data=True)
    windows = _windows(tmp_path / "o.zarr")
    assert ([w["start"] for w in windows], [w["end"] for w in windows]) == _expect(data)


def test_measured_after_the_cast(data, tmp_path):
    IO().write_pyramid(_pyr(data).downscale(n_layers=2), str(tmp_path / "o.zarr"),
                       overwrite=True, dtype="uint8", window_from_data=True)
    written = zarr.open_array(str(tmp_path / "o.zarr" / "0"))[...]
    windows = _windows(tmp_path / "o.zarr")
    assert ([w["start"] for w in windows], [w["end"] for w in windows]) == _expect(written)


def test_without_it_windows_are_untouched(data, tmp_path):
    pyr = _pyr(data).downscale(n_layers=2)
    expect = json.dumps(pyr.meta.metadata["omero"]["channels"], sort_keys=True)
    IO().write_pyramid(pyr, str(tmp_path / "o.zarr"), overwrite=True)
    assert json.dumps([{**c} for c in zarr.open_group(str(tmp_path / "o.zarr"), mode="r")
                       .attrs.asdict()["ome"]["omero"]["channels"]], sort_keys=True) == expect


def test_refused_where_level0_is_not_observed(data, tmp_path):
    with pytest.raises(ValueError, match="window_from_data"):
        IO().write_pyramid(_pyr(data).downscale(n_layers=2), str(tmp_path / "o.zarr"),
                           overwrite=True, window_from_data=True, max_downscale_processes=2)
