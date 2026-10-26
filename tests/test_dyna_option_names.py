"""write_pyramid's options use dyna-zarr's words (decision C-14, ozp 0.5.0): backend= is the
storage backend (io_backend= its old name), engine= / num_readers= / gc_interval= are dyna's
loop, compressor= takes dyna's Codecs too."""
import warnings

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid


@pytest.fixture
def pyr():
    data = np.arange(2 * 32 * 48, dtype="uint16").reshape(2, 32, 48)
    return Pyramid().from_arrays([data, data[:, ::2, ::2]], axis_order="cyx")


def _level0(path):
    return zarr.open_group(path, mode="r")["0"]


def _seen_backends(monkeypatch):
    """The storage backend dyna_zarr.io.write is given, per level written."""
    import dyna_zarr
    seen, write = [], dyna_zarr.io.write

    def spy(*args, **kw):
        seen.append(kw.get("backend"))
        return write(*args, **kw)
    monkeypatch.setattr(dyna_zarr.io, "write", spy)
    return seen


@pytest.mark.parametrize("call, expected", [({}, "tensorstore"),
                                            (dict(backend="tensorstore"), "tensorstore"),
                                            (dict(backend="zarrista"), "zarrista"),
                                            (dict(io_backend="zarrista"), "zarrista"),
                                            (dict(backend="zarrista", io_backend="zarrista"), "zarrista")])
def test_backend_is_the_storage_backend(pyr, tmp_path, monkeypatch, call, expected):
    if expected == "zarrista":
        pytest.importorskip("zarrista")
    seen = _seen_backends(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)              # no warning for these
        IO().write_pyramid(pyr, str(tmp_path / "a.zarr"), **call)
    assert seen and set(seen) == {expected}
    np.testing.assert_array_equal(_level0(str(tmp_path / "a.zarr"))[...], np.asarray(pyr.layers["0"]))


def test_backend_and_io_backend_must_agree(pyr, tmp_path):
    with pytest.raises(ValueError, match="disagree: io_backend is the old name of backend"):
        IO().write_pyramid(pyr, str(tmp_path / "a.zarr"), backend="tensorstore",
                           io_backend="zarrista")
    with pytest.raises(ValueError, match="the storage backend is one of"):
        IO().write_pyramid(pyr, str(tmp_path / "b.zarr"), backend="zarr-python")


def test_engine_num_readers_and_gc_interval_reach_dyna(pyr, tmp_path, monkeypatch):
    import dyna_zarr
    seen, write = [], dyna_zarr.io.write

    def spy(*args, **kw):
        seen.append({k: kw.get(k) for k in ("engine", "num_readers", "gc_interval")})
        return write(*args, **kw)
    monkeypatch.setattr(dyna_zarr.io, "write", spy)
    IO().write_pyramid(pyr, str(tmp_path / "a.zarr"), engine="pipeline", num_readers=2,
                       gc_interval=5.0)
    assert seen and all(s == {"engine": "pipeline", "num_readers": 2, "gc_interval": 5.0}
                        for s in seen)
    np.testing.assert_array_equal(_level0(str(tmp_path / "a.zarr"))[...], np.asarray(pyr.layers["0"]))
    with pytest.raises(ValueError, match="engine='turbo'"):
        IO().write_pyramid(pyr, str(tmp_path / "b.zarr"), engine="turbo")


def test_a_one_pass_write_refuses_a_loop_it_does_not_have(pyr, tmp_path):
    deferred = Pyramid().from_arrays([np.asarray(pyr.layers["0"])], axis_order="cyx") \
        .downscale(n_layers=2, defer=True)
    with pytest.raises(ValueError, match="computed in one pass"):
        IO().write_pyramid(deferred, str(tmp_path / "a.zarr"), engine="pipeline")
    IO().write_pyramid(deferred, str(tmp_path / "b.zarr"), engine="fused")   # dyna's own


def test_compressor_takes_a_dyna_codecs(pyr, tmp_path):
    from dyna_zarr import Codecs
    for codecs, name in ((Codecs("zstd", clevel=3), "zstd"), (Codecs("blosc", cname="zstd", clevel=4), "blosc")):
        out = str(tmp_path / f"{name}.zarr")
        IO().write_pyramid(pyr, out, compressor=codecs)
        back = Pyramid.from_zarr(out)
        assert back.compressor.name == name
        np.testing.assert_array_equal(np.asarray(back.layers["0"]), np.asarray(pyr.layers["0"]))
    with pytest.raises(ValueError, match="holds its own settings"):
        IO().write_pyramid(pyr, str(tmp_path / "x.zarr"), compressor=Codecs("zstd"),
                           compressor_params={"level": 3})
