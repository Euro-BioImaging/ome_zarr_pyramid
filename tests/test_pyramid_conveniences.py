"""Conveniences eubi-bridge's own NGFF code had and ozp lacked (found migrating eubi to
ozp, 2026-10-12): squeeze, update_scales / update_units by axis name, a public
mark_changed, and no unit on a channel axis."""
import json
import os

import numpy as np
import pytest

from ome_zarr_pyramid import IO, Pyramid

DATA = np.arange(1 * 3 * 1 * 32 * 32, dtype="uint16").reshape(1, 3, 1, 32, 32)


def _pyr():
    pyr = Pyramid().from_arrays([DATA], axis_order="tczyx", scales=[[1.0, 1.0, 4.0, 0.5, 0.5]],
                                version="0.5").downscale(n_layers=2, scale_factor=(1, 1, 1, 2, 2),
                                                         defer=False)
    pyr = pyr.set_channels({0: {"label": "a"}, 1: {"label": "b"}, 2: {"label": "c"}})
    for p in pyr.meta.resolution_paths:
        pyr.meta.update_translation(p, [0.0, 0.0, 7.0, 10.0, 20.0])
    return pyr


def test_squeeze_drops_size_1_axes_and_keeps_the_rest_of_the_metadata():
    sq = _pyr().squeeze()
    assert sq.axes == "cyx" and tuple(sq.base_array.shape) == (3, 32, 32)
    assert sq.meta.get_scale("0") == [1.0, 0.5, 0.5]
    assert sq.meta.get_scale("1") == [1.0, 1.0, 1.0]
    assert sq.meta.get_translation("0") == [0.0, 10.0, 20.0]
    assert [c["label"] for c in sq.meta.metadata["omero"]["channels"]] == ["a", "b", "c"]
    np.testing.assert_array_equal(np.asarray(sq.base_array[...]), DATA[0, :, 0])


def test_squeeze_named_axes_only_and_refusals():
    assert _pyr().squeeze("t").axes == "czyx"
    with pytest.raises(ValueError, match="not size 1"):
        _pyr().squeeze("c")
    flat = Pyramid().from_arrays([np.zeros((1, 32), "uint8")], axis_order="yx")
    assert flat.squeeze().axes == "yx"                      # y and x always stay
    no_op = Pyramid().from_arrays([np.zeros((2, 8, 8), "uint8")], axis_order="zyx")
    assert no_op.squeeze() is no_op


def test_update_scales_by_axis_rescales_every_level():
    pyr = _pyr().update_scales(z=2.0, x=0.25)
    assert pyr.meta.get_scale("0") == [1.0, 1.0, 2.0, 0.5, 0.25]
    assert pyr.meta.get_scale("1") == [1.0, 1.0, 2.0, 1.0, 0.5]           # x keeps factor 2
    with pytest.raises(ValueError, match="no axis"):
        pyr.update_scales(q=1.0)


def test_update_units_by_axis():
    pyr = _pyr().update_units(z="nanometer", t="millisecond")
    assert pyr.meta.unit_list == ["millisecond", None, "nanometer", "micrometer", "micrometer"]


def test_a_channel_axis_has_no_unit(tmp_path):
    pyr = _pyr()
    pyr.meta.update_unitlist(["second", "micrometer", "micrometer", "micrometer", "micrometer"])
    axes = pyr.meta.metadata["multiscales"][0]["axes"]
    assert "unit" not in next(a for a in axes if a["name"] == "c")
    assert next(a for a in axes if a["name"] == "z")["unit"] == "micrometer"


def test_mark_changed_makes_save_write_a_direct_edit(tmp_path):
    path = str(tmp_path / "a.ome.zarr")
    IO().write_pyramid(_pyr(), path, overwrite=True)
    meta = IO().read_pyramid(path).meta
    meta.metadata["omero"]["channels"][0]["label"] = "edited"
    meta.save_changes()                                    # not marked: nothing written
    assert IO().read_pyramid(path).meta.metadata["omero"]["channels"][0]["label"] == "a"
    meta.mark_changed()
    meta.save_changes()
    assert IO().read_pyramid(path).meta.metadata["omero"]["channels"][0]["label"] == "edited"
