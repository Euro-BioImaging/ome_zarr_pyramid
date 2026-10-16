"""write_pyramid(windows=...): the channels' omero display windows, in ozp so every
caller gets the same three modes (eubi-bridge's channel_intensity_limits maps onto them).
A "dtype" window describes the dtype AS WRITTEN: after a uint16 -> uint8 write, 0-255
(advertising the source range rendered the image black)."""
import os

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid


def _pyr(dtype="uint16", nc=2):
    a = (np.arange(nc * 4 * 16 * 16) % 3000 + 100).astype(dtype).reshape(nc, 4, 16, 16)
    pyr = Pyramid().from_arrays([a], axis_order="czyx", version="0.5")
    return pyr.set_channels({0: {"label": "DAPI",
                                 "window": {"min": 7, "max": 9, "start": 7, "end": 9}}})


def _written(tmp_path, pyr, **kw):
    path = str(tmp_path / "w.ome.zarr")
    IO().write_pyramid(pyr, path, overwrite=True, **kw)
    return dict(zarr.open_group(path, mode="r").attrs)["ome"]["omero"]["channels"]


def test_keep_is_the_default(tmp_path):
    chans = _written(tmp_path, _pyr())
    assert chans[0]["window"] == {"min": 7, "max": 9, "start": 7, "end": 9}


def test_dtype_describes_the_written_dtype(tmp_path):
    full16 = {"min": 0, "max": 65535, "start": 0, "end": 65535}
    assert [c["window"] for c in _written(tmp_path, _pyr(), windows="dtype")] == [full16] * 2
    full8 = {"min": 0, "max": 255, "start": 0, "end": 255}
    chans = _written(tmp_path, _pyr(), windows="dtype", dtype="uint8")
    assert [c["window"] for c in chans] == [full8] * 2
    f32 = np.finfo(np.float32)
    w = _written(tmp_path, _pyr("float32"), windows="dtype")[0]["window"]
    assert w["start"] == pytest.approx(float(f32.min)) and w["end"] == pytest.approx(float(f32.max))


def test_data_measures_level_0_and_keeps_min_max(tmp_path):
    pyr = _pyr()
    a = np.asarray(pyr.layers["0"][...])
    for kw in ({"windows": "data"}, {"window_from_data": True}):       # old spelling too
        chans = _written(tmp_path, pyr, **kw)
        for c, ch in enumerate(chans):
            assert (ch["window"]["start"], ch["window"]["end"]) == (a[c].min(), a[c].max())
        assert (chans[0]["window"]["min"], chans[0]["window"]["max"]) == (7, 9)


def test_none_writes_no_window(tmp_path):
    chans = _written(tmp_path, _pyr(), windows="none")
    assert all("window" not in c for c in chans)
    assert chans[0]["label"] == "DAPI"                                    # the rest is kept


def test_a_pyramid_without_channels_gets_them_first(tmp_path):
    a = np.zeros((3, 4, 16, 16), "u1")
    chans = _written(tmp_path, Pyramid().from_arrays([a], axis_order="czyx", version="0.5"),
                     windows="dtype")
    assert len(chans) == 3 and chans[2]["window"]["end"] == 255


def test_refused(tmp_path):
    path = str(tmp_path / "x.ome.zarr")
    with pytest.raises(ValueError, match="windows="):
        IO().write_pyramid(_pyr(), path, windows="bright")
    with pytest.raises(ValueError, match="cannot be combined"):
        IO().write_pyramid(_pyr(), path, windows="none", window_from_data=True)
    label = Pyramid().from_arrays([np.zeros((4, 16, 16), "u2")], axis_order="zyx",
                                  version="0.5").set_image_label()
    with pytest.raises(ValueError, match="label"):
        IO().write_pyramid(label, path, windows="dtype")
    assert not os.path.exists(path)


# -- windows="auto": percentiles of the coarsest written level, else the dtype's range --

def _bright(shape=(2, 8, 256, 256)):
    """Channel 0 in 1000-1999, channel 1 in 3000-3999, plus one hot pixel each."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, 1000, shape).astype("u2")
    a[0] += 1000
    a[1] += 3000
    a[:, 0, 0, 0] = 65000
    return a


@pytest.mark.parametrize("deferred", [False, True])
def test_auto_takes_percentiles_of_the_coarsest_level(tmp_path, deferred):
    a = _bright()
    pyr = Pyramid().from_arrays([a], axis_order="czyx", version="0.5")
    pyr = pyr.downscale(n_layers=3, defer=deferred)
    chans = _written(tmp_path, pyr, windows="auto")
    path = tmp_path / "w.ome.zarr"
    coarsest = np.asarray(zarr.open_group(str(path), mode="r")["2"])
    for k, ch in enumerate(chans):
        lo, hi = np.percentile(coarsest[k], (0.1, 99.9))
        assert ch["window"] == {"min": 0, "max": 65535, "start": int(np.floor(lo)),
                                "end": int(np.ceil(hi))}
        assert ch["window"]["end"] < 65000                  # the hot pixel does not count
    assert 1000 <= chans[0]["window"]["start"] and chans[1]["window"]["start"] >= 3000


def test_auto_measures_what_is_written(tmp_path):
    chans = _written(tmp_path, Pyramid().from_arrays([_bright()], axis_order="czyx", version="0.5"),
                     windows="auto", dtype="uint8")
    assert all(ch["window"] == {"min": 0, "max": 255, "start": 255, "end": 255}
               for ch in chans)                              # saturated by the cast


def test_auto_falls_back_to_the_dtype_when_the_coarsest_level_is_big(tmp_path, monkeypatch,
                                                                     caplog):
    import logging
    from ome_zarr_pyramid.core import io as ozp_io
    monkeypatch.setattr(ozp_io, "AUTO_WINDOW_MAX_MB", 0.5)  # one level of 2 MB
    with caplog.at_level(logging.INFO, logger="ome_zarr_pyramid.core.io"):
        chans = _written(tmp_path, Pyramid().from_arrays([_bright()], axis_order="czyx", version="0.5"),
                         windows="auto")
    assert all(ch["window"] == {"min": 0, "max": 65535, "start": 0, "end": 65535}
               for ch in chans)
    assert "windows from the dtype" in caplog.text


def test_auto_on_floats_ignores_nan(tmp_path):
    a = np.linspace(0, 1, 64 * 64, dtype="f4").reshape(1, 64, 64)
    a[0, 0, :8] = np.nan
    chans = _written(tmp_path, Pyramid().from_arrays([a], axis_order="cyx", version="0.5"), windows="auto")
    lo, hi = np.nanpercentile(a, (0.1, 99.9))
    assert chans[0]["window"]["start"] == pytest.approx(lo, rel=1e-3)
    assert chans[0]["window"]["end"] == pytest.approx(hi, rel=1e-3)
