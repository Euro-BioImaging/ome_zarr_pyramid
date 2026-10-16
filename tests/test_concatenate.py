"""Pyramids joined along one axis (``ome_zarr_pyramid.concatenate``, roadmap B6).

Joining puts several images into one coordinate system, so everything but their size along
the joined axis must agree, and every disagreement is an error naming it: eubi-bridge used
to take the last view's metadata silently. Values are checked against ``np.concatenate``
on every level.
"""
import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid, concatenate

RNG = np.random.default_rng(3)
SCALES = [[1.0, 2.0, 0.5, 0.5], [1.0, 4.0, 1.0, 1.0]]
AT = [0.0, 10.0, -3.0, 7.5]


def _pyr(nc, *, z=4, dtype="uint16", scales=SCALES, at=AT, labels=None, units=None, nt=None):
    """A two-level czyx (or tczyx with *nt*) pyramid with *nc* channels."""
    shape0 = (nc, z, 16, 16) if nt is None else (nt, nc, z, 16, 16)
    a0 = RNG.integers(0, 60000, shape0).astype(dtype)
    a1 = np.ascontiguousarray(a0[..., ::2, ::2, ::2])
    axes = "czyx" if nt is None else "tczyx"
    sc = scales if nt is None else [[1.0] + s for s in scales]
    pyr = Pyramid().from_arrays([a0, a1], axis_order=axes, scales=sc, unit_list=units,
                                version="0.5", name=f"img{nc}")
    tr = at if nt is None else [0.0] + at
    for path in pyr.meta.resolution_paths:
        pyr.meta.update_translation(path, tr)
    if labels is not None:
        pyr = pyr.set_channels({i: {"label": lab, "color": "FF00FF" if i else "00FF00",
                                    "window": {"start": 5 + i, "end": 500 + i}}
                                for i, lab in enumerate(labels)})
    return pyr, (a0, a1)


def _values(pyr):
    return [np.asarray(pyr.layers[p][...]) for p in pyr.meta.resolution_paths]


def test_along_c_values_channels_and_metadata():
    p1, (a0, a1) = _pyr(2, labels=["DAPI", "GFP"])
    p2, (b0, b1) = _pyr(3, labels=["A", "B", "C"])
    p1 = p1.set_custom_metadata("my_app", {"from": "first"})
    out = concatenate([p1, p2], axis="c")
    assert hasattr(out.layers[out.meta.resolution_paths[0]], "_with_transform")   # lazy dyna
    got = _values(out)
    np.testing.assert_array_equal(got[0], np.concatenate([a0, b0], axis=0))
    np.testing.assert_array_equal(got[1], np.concatenate([a1, b1], axis=0))
    chans = out.meta.metadata["omero"]["channels"]
    assert [c["label"] for c in chans] == ["DAPI", "GFP", "A", "B", "C"]
    assert [c["color"] for c in chans] == ["00FF00", "FF00FF", "00FF00", "FF00FF", "FF00FF"]
    assert [c["window"]["start"] for c in chans] == [5, 6, 5, 6, 7]       # each kept
    assert out.meta.get_scale("0") == SCALES[0] and out.meta.get_scale("1") == SCALES[1]
    assert out.meta.get_translation("0") == AT == out.meta.get_translation("1")
    assert out.meta.tag == "img2" and out.custom_metadata == {"my_app": {"from": "first"}}


def test_an_input_without_channel_metadata_gets_the_defaults():
    p1, _ = _pyr(1, labels=["DAPI"])
    p2, _ = _pyr(2)                                         # from_arrays: no omero channels
    labels = [c["label"] for c in concatenate([p1, p2]).meta.metadata["omero"]["channels"]]
    assert labels == ["DAPI", "Channel 0", "Channel 1"]


def test_along_t_the_channels_must_be_the_same_and_the_first_position_is_kept():
    p1, (a0, _a1) = _pyr(2, nt=2, labels=["DAPI", "GFP"])
    p2, (b0, _b1) = _pyr(2, nt=3, labels=["DAPI", "GFP"], at=AT)
    p2.meta.update_translation("0", [60.0] + AT)            # later in time: allowed
    p2.meta.update_translation("1", [60.0] + AT)
    out = concatenate([p1, p2], axis="t")
    np.testing.assert_array_equal(_values(out)[0], np.concatenate([a0, b0], axis=0))
    assert out.meta.get_translation("0") == [0.0] + AT
    assert [c["label"] for c in out.meta.metadata["omero"]["channels"]] == ["DAPI", "GFP"]
    p3, _ = _pyr(2, nt=1, labels=["DAPI", "RFP"])
    with pytest.raises(ValueError, match="other channels"):
        concatenate([p1, p3], axis="t")


def test_float_noise_in_scales_and_positions_still_agrees():
    p1, _ = _pyr(1)
    p2, _ = _pyr(1, scales=[[1.0, 2.0 + 1e-12, 0.5, 0.5], [1.0, 4.0, 1.0 * (1 + 1e-9), 1.0]],
                 at=[0.0, 10.0, -3.0, 7.5 + 1e-12])
    assert concatenate([p1, p2]).layers["0"].shape[0] == 2


@pytest.mark.parametrize("change, match", [
    (dict(z=5), "shape"),
    (dict(dtype="uint8"), "uint8"),
    (dict(scales=[[1.0, 2.0, 0.5, 0.5], [1.0, 4.0, 1.0, 1.1]]), "level 1 has scale"),
    (dict(scales=[[1.0, 2.5, 0.5, 0.5], [1.0, 5.0, 1.0, 1.0]]), "level 0 has scale"),
    (dict(at=[0.0, 10.0, -3.0, 8.0]), "only the position along 'c'"),
    (dict(units=["micrometer", "nanometer", "micrometer"]), "units"),
])
def test_inputs_that_disagree_are_refused_naming_what_differs(change, match):
    p1, _ = _pyr(1)
    p2, _ = _pyr(1, **change)
    with pytest.raises(ValueError, match=match):
        concatenate([p1, p2])


def test_other_refusals():
    p1, _ = _pyr(1)
    two_levels = p1
    one_level = Pyramid().from_arrays([np.zeros((1, 4, 16, 16), "u2")], axis_order="czyx",
                                      scales=[SCALES[0]], version="0.5")
    with pytest.raises(ValueError, match="levels"):
        concatenate([two_levels, one_level])
    with pytest.raises(ValueError, match="axes"):
        concatenate([p1, Pyramid().from_arrays([np.zeros((4, 16, 16), "u2")], axis_order="zyx",
                                               version="0.5")])
    with pytest.raises(ValueError, match="not in the pyramids' axes"):
        concatenate([p1, p1], axis="t")
    with pytest.raises(ValueError, match="label image"):
        concatenate([p1, p1.set_image_label()])
    planned = Pyramid().from_arrays([np.zeros((1, 8, 32, 32), "u2")], axis_order="czyx",
                                    version="0.5").downscale(n_layers=2, scale_factor=(1, 2, 2, 2))
    with pytest.raises(ValueError, match="pending downscale"):
        concatenate([planned, planned])
    with pytest.raises(ValueError, match="no pyramids"):
        concatenate([])
    with pytest.raises(TypeError, match="not an initialized Pyramid"):
        concatenate([p1, np.zeros(3)])


def test_dask_inputs_join_with_dask():
    da = pytest.importorskip("dask.array")
    a0 = RNG.integers(0, 100, (2, 4, 16, 16)).astype("u2")
    b0 = RNG.integers(0, 100, (1, 4, 16, 16)).astype("u2")
    p1 = Pyramid().from_arrays([da.from_array(a0)], axis_order="czyx", version="0.5")
    p2 = Pyramid().from_arrays([da.from_array(b0)], axis_order="czyx", version="0.5")
    out = concatenate([p1, p2])
    level = out.layers[out.meta.resolution_paths[0]]
    assert hasattr(level, "dask")
    np.testing.assert_array_equal(np.asarray(level), np.concatenate([a0, b0]))


def test_a_joined_pyramid_writes_and_reads_back(tmp_path):
    p1, (a0, a1) = _pyr(2, labels=["DAPI", "GFP"])
    p2, (b0, b1) = _pyr(1, labels=["A"])
    path = str(tmp_path / "joined.ome.zarr")
    IO().write_pyramid(concatenate([p1, p2]), path, overwrite=True)
    back = IO().read_pyramid(path)
    np.testing.assert_array_equal(_values(back)[0], np.concatenate([a0, b0]))
    np.testing.assert_array_equal(_values(back)[1], np.concatenate([a1, b1]))
    ome = dict(zarr.open_group(path, mode="r").attrs)["ome"]
    assert [c["label"] for c in ome["omero"]["channels"]] == ["DAPI", "GFP", "A"]
    assert ome["multiscales"][0]["datasets"][0]["coordinateTransformations"][1][
        "translation"] == AT


# -- B5 and C7 ------------------------------------------------------------------------------

def test_from_image_can_take_only_the_full_resolution_level():
    from test_collection import _image
    img = _image()                                           # two stored levels
    assert Pyramid.from_image(img).meta.resolution_paths == ["0", "1"]
    assert Pyramid.from_image(img, resolutions="base").meta.resolution_paths == ["0"]
    with pytest.raises(ValueError, match="resolutions"):
        Pyramid.from_image(img, resolutions="first")


def test_a_channel_can_have_no_window(tmp_path):
    p1, _ = _pyr(2, labels=["DAPI", "GFP"])
    pyr = p1.set_channels({0: {"window": None}})             # None removes the field
    chans = pyr.meta.metadata["omero"]["channels"]
    assert "window" not in chans[0] and chans[1]["window"]["start"] == 6
    path = str(tmp_path / "nowin.ome.zarr")
    IO().write_pyramid(pyr, path, overwrite=True)
    written = dict(zarr.open_group(path, mode="r").attrs)["ome"]["omero"]["channels"]
    assert "window" not in written[0] and "window" in written[1]
