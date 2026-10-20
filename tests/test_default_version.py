"""5.27 (E3): ozp writes NGFF 0.5 (zarr v3) by default, from every constructor and from
every collection whose source is not an NGFF store; a store read as 0.4 or 0.5 is written
back in its own version when no `version` is given, and in the one asked for otherwise."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import Pyramid, PyramidCollection, io

from test_collection import FakeImage
from test_collection_zarr import _plate
from test_series import _layout

DATA = np.zeros((8, 8), "u1")


def _constructed(how, version=None):
    kw = {} if version is None else {"version": version}
    if how == "from_array":
        return Pyramid().from_array(DATA, axis_order="yx", **kw)
    if how == "from_arrays":
        return Pyramid().from_arrays([DATA, DATA[::2, ::2]], axis_order="yx", **kw)
    return Pyramid().from_label_arrays([DATA], axis_order="yx", **kw)


def _format_on_disk(path):
    """2 or 3, from the files the group or array holds (and never both)."""
    v3 = os.path.exists(os.path.join(path, "zarr.json"))
    v2 = any(os.path.exists(os.path.join(path, f)) for f in (".zattrs", ".zgroup", ".zarray"))
    assert v3 != v2, os.listdir(path)
    return 3 if v3 else 2


def _ngff_version(path):
    """The NGFF version a written group declares: `ome.version` (0.5), or the 0.4 spots."""
    if _format_on_disk(path) == 3:
        return json.load(open(os.path.join(path, "zarr.json")))["attributes"]["ome"]["version"]
    attrs = json.load(open(os.path.join(path, ".zattrs")))
    assert "ome" not in attrs
    for key in ("multiscales", "plate", "well"):
        if key in attrs:
            block = attrs[key][0] if key == "multiscales" else attrs[key]
            return block["version"]
    return "0.4"                     # a 0.4 bioformats2raw root says its version nowhere


@pytest.mark.parametrize("how", ["from_array", "from_arrays", "from_label_arrays"])
def test_constructors_default_to_05(tmp_path, how):
    pyr = _constructed(how)
    assert pyr.meta.version == "0.5" and pyr.meta.zarr_format == 3
    if how == "from_label_arrays":
        assert pyr.meta.image_label["version"] == "0.5"
    out = str(tmp_path / "img.ome.zarr")
    io.write_pyramid(pyr, out)
    assert _format_on_disk(out) == 3 and _ngff_version(out) == "0.5"
    assert zarr.open_group(out, mode="r").metadata.zarr_format == 3


@pytest.mark.parametrize("how", ["from_array", "from_arrays", "from_label_arrays"])
def test_constructors_still_write_04_when_asked(tmp_path, how):
    pyr = _constructed(how, version="0.4")
    assert pyr.meta.version == "0.4" and pyr.meta.zarr_format == 2
    out = str(tmp_path / "img.ome.zarr")
    io.write_pyramid(pyr, out)
    assert _format_on_disk(out) == 2 and _ngff_version(out) == "0.4"


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_a_store_is_written_back_in_its_own_version(tmp_path, version):
    src = str(tmp_path / "src.ome.zarr")
    io.write_pyramid(_constructed("from_arrays", version), src)
    out = str(tmp_path / "out.ome.zarr")
    io.write_pyramid(io.read_pyramid(src), out)
    assert _ngff_version(out) == version and _format_on_disk(out) == (3 if version == "0.5" else 2)
    # and a deferred downscale of it too
    again = str(tmp_path / "again.ome.zarr")
    io.write_pyramid(io.read_pyramid(src).downscale(n_layers=2), again)
    assert _ngff_version(again) == version


def _wells(path):
    return [os.path.join(path, r, c) for r in ("A", "B") for c in ("1", "2")
            if os.path.isdir(os.path.join(path, r, c))]


@pytest.mark.parametrize("asked, expected", [(None, "0.4"), ("0.5", "0.5"), ("0.4", "0.4")])
def test_a_04_plate_keeps_its_version_unless_told(tmp_path, asked, expected):
    src, _ = _plate(tmp_path, "0.4")
    out = str(tmp_path / "out.zarr")
    kw = {} if asked is None else {"version": asked}
    PyramidCollection.from_zarr(src).write_plate(out, **kw)
    assert _ngff_version(out) == expected
    wells = _wells(out)
    assert wells and all(_ngff_version(w) == expected for w in wells)
    assert all(_ngff_version(os.path.join(w, "0")) == expected for w in wells)


@pytest.mark.parametrize("asked, expected", [(None, "0.4"), ("0.5", "0.5")])
def test_a_04_series_layout_keeps_its_version_unless_told(tmp_path, asked, expected):
    src, _ = _layout(tmp_path, "0.4")
    out = str(tmp_path / "out.ome.zarr")
    kw = {} if asked is None else {"version": asked}
    PyramidCollection.from_zarr(src).write_series(out, **kw)
    assert _format_on_disk(out) == (3 if expected == "0.5" else 2)
    assert _ngff_version(os.path.join(out, "0")) == expected
    assert _ngff_version(os.path.join(out, "1")) == expected


def _fake_file():
    images = [FakeImage(np.full((6, 8), i, "u2"), "yx", index={"series": i}) for i in range(2)]
    return SimpleNamespace(path="/data/two.czi", images=images)


def test_a_microscopy_file_is_written_as_05(tmp_path):
    coll = PyramidCollection.from_file(_fake_file())
    out = str(tmp_path / "out.ome.zarr")
    coll.write_series(out)
    assert _format_on_disk(out) == 3
    assert json.load(open(os.path.join(out, "OME", "zarr.json")))["attributes"]["ome"]["version"] == "0.5"
    assert _ngff_version(os.path.join(out, "0")) == "0.5"
    written = coll.write(str(tmp_path), "x")
    assert written and all(_ngff_version(p) == "0.5" for p in written.values())


def test_a_collection_of_constructed_pyramids_is_written_as_05(tmp_path):
    # the A3 tester's case: built with from_array and no version, written as series
    coll = PyramidCollection({(("series", i),): Pyramid().from_array(DATA + i, axis_order="yx")
                              for i in range(2)})
    out = str(tmp_path / "out.ome.zarr")
    coll.write_series(out)
    assert _format_on_disk(out) == 3
    assert os.path.exists(os.path.join(out, "OME", "zarr.json"))
    assert _ngff_version(os.path.join(out, "1")) == "0.5"


# ---------------------------------------------------------------- labels follow their image
# zarr v2 and v3 groups do not nest: a v3 label under a v2 image is invisible to zarr-python
# and every 0.4 reader. With the 0.5 default, "read a 0.4 store, add a label, write" would
# give exactly that, so a label takes its image's version (A5 tester r1, failures 1 and 2).

LABEL = np.zeros((8, 8), "u2")


def _assert_tree(out, version):
    fmt = 3 if version == "0.5" else 2
    for node in ("", "labels", os.path.join("labels", "nuc"), os.path.join("labels", "nuc", "0")):
        assert _format_on_disk(os.path.join(out, node)) == fmt, node
    assert _ngff_version(os.path.join(out, "labels", "nuc")) == version
    grp = zarr.open_group(out, mode="r")
    assert "nuc" in grp["labels"] and grp["labels/nuc"].metadata.zarr_format == fmt


@pytest.mark.parametrize("image_v, label_v", [("0.4", None), ("0.4", "0.5"), ("0.5", "0.4")])
def test_add_image_label_takes_the_images_version(tmp_path, image_v, label_v):
    src = str(tmp_path / "src.ome.zarr")
    io.write_pyramid(Pyramid().from_array(DATA, axis_order="yx", version=image_v), src)
    img = io.read_pyramid(src)
    kw = {} if label_v is None else {"version": label_v}
    lbl = Pyramid().from_label_arrays([LABEL], axis_order="yx", **kw)
    before = lbl.meta.version
    with_label = img.add_image_label(lbl, "nuc")
    assert lbl.meta.version == before                       # the caller's label is untouched
    attached = with_label.labels["nuc"]
    assert attached.meta.version == image_v
    assert attached.meta.zarr_format == (3 if image_v == "0.5" else 2)
    assert attached.meta.image_label["version"] == image_v
    assert attached.meta.image_label["source"] == {"image": "../../"}
    out = str(tmp_path / "out.ome.zarr")
    io.write_pyramid(with_label, out)
    _assert_tree(out, image_v)
    back = io.read_pyramid(out)
    assert back.labels.names == ["nuc"] and back.labels["nuc"].meta.version == image_v
    np.testing.assert_array_equal(np.asarray(back.labels["nuc"].base_array), LABEL)


def test_write_pyramid_writes_a_label_in_its_images_version(tmp_path):
    # a label attached some other way than add_image_label (here: set directly) is
    # still written in the image's version
    img = Pyramid().from_array(DATA, axis_order="yx", version="0.4")
    img._labels = {"nuc": Pyramid().from_label_arrays([LABEL], axis_order="yx")}
    out = str(tmp_path / "out.ome.zarr")
    io.write_pyramid(img, out)
    _assert_tree(out, "0.4")
    assert img._labels["nuc"].meta.version == "0.5"         # the pyramid itself is untouched


@pytest.mark.parametrize("asked", [None, "0.4"])
def test_write_series_of_a_04_image_with_a_default_label_is_04(tmp_path, asked):
    img = Pyramid().from_array(DATA, axis_order="yx", version="0.4")
    img._labels = {"nuc": Pyramid().from_label_arrays([LABEL], axis_order="yx")}
    coll = PyramidCollection({(("series", 0),): img})
    out = str(tmp_path / "out.ome.zarr")
    coll.write_series(out, **({} if asked is None else {"version": asked}))
    assert _format_on_disk(out) == 2
    _assert_tree(os.path.join(out, "0"), "0.4")


# a label written straight into an existing store (io.write_labels / write_pyramid with
# labels_group_path, the route pyrametric's write_label_pyramid takes) takes that store's
# version too (A5 tester r2, failure 1)

@pytest.mark.parametrize("image_v, label_v", [("0.4", None), ("0.5", "0.4"), ("0.4", "0.4")])
def test_write_labels_into_an_existing_store_takes_its_version(tmp_path, image_v, label_v):
    src = str(tmp_path / "img.ome.zarr")
    io.write_pyramid(Pyramid().from_array(DATA, axis_order="yx", version=image_v), src)
    kw = {} if label_v is None else {"version": label_v}
    lbl = Pyramid().from_label_arrays([LABEL], axis_order="yx", **kw)
    before = lbl.meta.version
    io.write_labels(lbl, os.path.join(src, "labels", "nuc"),
                    labels_group_path=os.path.join(src, "labels"), label_name="nuc")
    assert lbl.meta.version == before                       # the caller's label is untouched
    _assert_tree(src, image_v)
    back = io.read_pyramid(src)
    assert back.labels.names == ["nuc"] and back.labels["nuc"].meta.version == image_v
    assert back.labels["nuc"].meta.image_label["version"] == image_v
    np.testing.assert_array_equal(np.asarray(back.labels["nuc"].base_array), LABEL)


def test_write_labels_next_to_an_existing_label_keeps_both(tmp_path):
    src = str(tmp_path / "img.ome.zarr")
    labels = os.path.join(src, "labels")
    io.write_pyramid(Pyramid().from_array(DATA, axis_order="yx", version="0.4"), src)
    io.write_labels(Pyramid().from_label_arrays([LABEL], axis_order="yx", version="0.4"),
                    os.path.join(labels, "a"), labels_group_path=labels, label_name="a")
    io.write_labels(Pyramid().from_label_arrays([LABEL], axis_order="yx"),
                    os.path.join(labels, "b"), labels_group_path=labels, label_name="b")
    assert sorted(os.listdir(labels)) == [".zattrs", ".zgroup", "a", "b"]   # no zarr.json
    assert sorted(io.read_pyramid(src).labels.names) == ["a", "b"]
    assert sorted(zarr.open_group(src, mode="r")["labels"].attrs["labels"]) == ["a", "b"]
    for name in ("a", "b"):
        assert _format_on_disk(os.path.join(labels, name)) == 2


def test_write_labels_into_a_new_store_keeps_the_labels_version(tmp_path):
    labels = str(tmp_path / "new.ome.zarr" / "labels")
    io.write_labels(Pyramid().from_label_arrays([LABEL], axis_order="yx"),
                    os.path.join(labels, "nuc"), labels_group_path=labels, label_name="nuc")
    assert _format_on_disk(labels) == 3
    assert _ngff_version(os.path.join(labels, "nuc")) == "0.5"


def test_write_labels_into_an_06_store_is_refused(tmp_path):
    from ome_zarr_pyramid import UnsupportedNGFFVersion
    src = tmp_path / "img06.ome.zarr"
    zarr.open_group(str(src), mode="w", zarr_format=3).attrs["ome"] = {"version": "0.6"}
    labels = str(src / "labels")
    with pytest.raises(UnsupportedNGFFVersion, match="0.6"):
        io.write_labels(Pyramid().from_label_arrays([LABEL], axis_order="yx"),
                        os.path.join(labels, "nuc"), labels_group_path=labels,
                        label_name="nuc")
    assert not os.path.exists(labels)                       # refused before anything is written
