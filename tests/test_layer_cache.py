"""Pyramid.layers opens each level's zarr array once per group and set of level paths
(it re-opened every level on every access: an isel of a 3-level image opened 24 arrays)."""
import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid


def _store(tmp_path, name="a.zarr", value=1, shape=(2, 32, 32)):
    out = str(tmp_path / name)
    IO().write_pyramid(Pyramid().from_arrays([np.full(shape, value, "u2")], axis_order="cyx")
                       .downscale(n_layers=3), out)
    return out


def test_each_level_is_opened_once(tmp_path, monkeypatch):
    path = _store(tmp_path)
    opened = []
    real = zarr.Group.__getitem__

    def counting(self, key):
        if key in ("0", "1", "2"):
            opened.append(key)
        return real(self, key)
    monkeypatch.setattr(zarr.Group, "__getitem__", counting)
    pyr = Pyramid.from_zarr(path)                         # the read opens them once
    for _ in range(5):
        assert list(pyr.layers) == ["0", "1", "2"]
    pyr.isel(c=0), pyr.select_levels(1), pyr.set_channels({0: {"label": "a"}})
    assert sorted(opened) == ["0", "1", "2"]


def test_the_dict_is_a_copy_and_the_arrays_are_the_store_s(tmp_path):
    pyr = Pyramid.from_zarr(_store(tmp_path, value=7))
    levels = pyr.layers
    levels.pop("0")
    assert list(pyr.layers) == ["0", "1", "2"]
    np.testing.assert_array_equal(np.asarray(pyr.layers["0"]), 7)


def test_another_group_or_other_level_paths_open_again(tmp_path):
    pyr = Pyramid.from_zarr(_store(tmp_path, "a.zarr", value=1))
    assert np.asarray(pyr.layers["0"]).max() == 1
    pyr.from_ngff(zarr.open_group(_store(tmp_path, "b.zarr", value=2), mode="r"))
    assert np.asarray(pyr.layers["0"]).max() == 2        # the new group's arrays
    pyr.meta.multiscales["datasets"] = [d for d in pyr.meta.multiscales["datasets"]
                                        if d["path"] != "2"]
    assert list(pyr.meta.resolution_paths) == ["0", "1"]   # other level paths: opened again
    assert list(pyr.layers) == ["0", "1"]


def test_a_store_overwritten_after_it_was_read_is_read_again(tmp_path):
    out = _store(tmp_path, value=1)
    old = Pyramid.from_zarr(out)
    old.layers                                            # cached
    IO().write_pyramid(Pyramid().from_arrays([np.full((2, 16, 16), 3, "u2")], axis_order="cyx"),
                       out, overwrite=True)
    fresh = Pyramid.from_zarr(out)                        # a new read sees the new store
    assert fresh.layers["0"].shape == (2, 16, 16)
    np.testing.assert_array_equal(np.asarray(fresh.layers["0"]), 3)
