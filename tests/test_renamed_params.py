"""max_workers -> max_threads, use_multiprocessing -> max_downscale_processes (0.1.0): the old
names still work, with a FutureWarning; old and new together are an error."""
import numpy as np
import pytest
import zarr

from ome_zarr_pyramid.core.io import IO
from ome_zarr_pyramid.core.pyramid import Pyramid


@pytest.fixture
def pyr():
    data = np.arange(2 * 16 * 32 * 32, dtype="uint16").reshape(2, 16, 32, 32)
    arr = zarr.create_array(store=zarr.storage.MemoryStore(), shape=data.shape,
                            chunks=(1, 8, 16, 16), dtype=data.dtype)
    arr[...] = data
    return Pyramid().from_arrays([arr], axis_order="czyx", version="0.5")


def test_max_threads_sets_the_dyna_threads(pyr, tmp_path, caplog):
    with caplog.at_level("INFO"):
        IO().write_pyramid(pyr, str(tmp_path / "a.zarr"), max_threads=3)
    assert "fused, 3 threads" in caplog.text


def test_max_workers_still_works(pyr, tmp_path, caplog):
    with caplog.at_level("INFO"),             pytest.warns(FutureWarning, match="max_workers= is deprecated.*max_threads="):
        IO().write_pyramid(pyr, str(tmp_path / "a.zarr"), max_workers=3)
    assert "fused, 3 threads" in caplog.text


def test_use_multiprocessing_still_works(pyr, tmp_path):
    ref = tmp_path / "ref.zarr"
    IO().write_pyramid(pyr.downscale(n_layers=2, defer=True), str(ref))
    out = tmp_path / "mp.zarr"
    with pytest.warns(FutureWarning, match="use_multiprocessing= is deprecated.*max_downscale_processes"):
        IO().write_pyramid(pyr.downscale(n_layers=2, defer=True), str(out),
                           use_multiprocessing=True)
    for lvl in ("0", "1"):
        np.testing.assert_array_equal(zarr.open_array(str(out / lvl))[...],
                                      zarr.open_array(str(ref / lvl))[...])


@pytest.mark.parametrize("kw", [dict(max_threads=2, max_workers=2),
                                dict(max_downscale_processes=2, use_multiprocessing=True)])
def test_old_and_new_together_is_an_error(pyr, tmp_path, kw):
    with pytest.raises(TypeError, match="pass max_(threads|downscale_processes)= only"):
        IO().write_pyramid(pyr, str(tmp_path / "a.zarr"), **kw)


def test_one_process_means_this_process(pyr, tmp_path):
    # max_downscale_processes=1 is no multiprocessing, so the one-pass write (and windows) work
    IO().write_pyramid(pyr, str(tmp_path / "a.zarr"), max_downscale_processes=1,
                       window_from_data=True)


@pytest.mark.parametrize("kw", [dict(queue_size=8), dict(max_concurrency=2),
                                dict(max_concurrent_layers=1)])
def test_removed_engine_settings_are_refused(pyr, tmp_path, kw):
    # they belonged to the engines removed in 0.1.0; ignoring them would hide a mistake
    with pytest.raises(ValueError, match="removed in ome_zarr_pyramid 0.1.0"):
        IO().write_pyramid(pyr, str(tmp_path / "a.zarr"), **kw)
    assert not (tmp_path / "a.zarr").exists()


def test_an_unknown_keyword_is_still_a_type_error(pyr, tmp_path):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        IO().write_pyramid(pyr, str(tmp_path / "a.zarr"), chunks_mb=4)


def test_the_package_io(tmp_path):
    # `from ome_zarr_pyramid import io`: one shared, stateless IO; IO() still works
    import pickle
    from ome_zarr_pyramid import IO, io
    assert isinstance(io, IO) and vars(io) == {}
    assert type(pickle.loads(pickle.dumps(io.write_pyramid)).__self__) is IO
    pyr = Pyramid().from_arrays([np.ones((4, 16, 16), "u2")], axis_order="zyx", version="0.4")
    io.write_pyramid(pyr, str(tmp_path / "a.zarr"))
    assert io.read_pyramid(str(tmp_path / "a.zarr")).nlayers == 1


def test_downscale_processes_logs_that_it_is_rarely_useful(pyr, tmp_path, caplog, monkeypatch):
    from ome_zarr_pyramid.core import _compat
    monkeypatch.setattr(_compat, "_warned_downscale_processes", False)
    with caplog.at_level("WARNING", logger="ome_zarr_pyramid"):
        IO().write_pyramid(pyr.downscale(n_layers=2, defer=True), str(tmp_path / "a.zarr"),
                           max_downscale_processes=1)                       # off: silent
        assert "rarely faster" not in caplog.text
        IO().write_pyramid(pyr.downscale(n_layers=2, defer=True), str(tmp_path / "b.zarr"),
                           max_downscale_processes=2)
        IO().write_pyramid(pyr.downscale(n_layers=2, defer=True), str(tmp_path / "c.zarr"),
                           max_downscale_processes=2)
    assert caplog.text.count("rarely faster") == 1                         # once per process
