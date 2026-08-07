"""Metadata handling: channels/omero, scales, translations, units, rename."""

import pytest


def test_set_channels_labels_and_colors(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 16, 16), axis_order="cyx")
    pyr = pyr.set_channels({0: {"label": "DAPI", "color": "red"},
                            1: {"label": "GFP", "color": "green"}})
    chans = pyr.meta.get_channels()
    assert [c["label"] for c in chans] == ["DAPI", "GFP"]
    assert chans[0]["color"].upper() == "FF0000"


def test_set_channels_is_sparse(make_pyramid):
    pyr, _ = make_pyramid(shape=(3, 16, 16), axis_order="cyx", channels=True)
    updated = pyr.set_channels({1: {"label": "middle"}})
    labels = [c["label"] for c in updated.meta.get_channels()]
    assert labels == ["ch0", "middle", "ch2"]           # only channel 1 touched


def test_set_channels_returns_new_pyramid(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 16, 16), axis_order="cyx", channels=True)
    updated = pyr.set_channels({0: {"label": "x"}})
    assert updated is not pyr
    assert pyr.meta.get_channels()[0]["label"] == "ch0"  # original unchanged


def test_set_display_range(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 16, 16), axis_order="cyx")
    out = pyr.set_display_range("minmax")
    windows = [c.get("window") for c in out.meta.get_channels()]
    assert all(w is not None and w["start"] <= w["end"] for w in windows)


def test_scales_and_translation(pyr4d):
    pyr, _ = pyr4d
    assert pyr.meta.get_base_scale() == [1.0, 2.0, 0.5, 0.5]
    # translation defaults to None/zero until an op that shifts it (e.g. crop)
    sub = pyr.isel(z=slice(2, 6))
    z = sub.axes.index("z")
    assert sub.meta.get_base_translation()[z] == pytest.approx(2 * 2.0)  # start * scale


def test_rename(pyr3d):
    pyr, _ = pyr3d
    renamed = pyr.rename("mydata")
    assert renamed.meta.tag == "mydata"
