"""ozp refuses an NGFF version it cannot read with UnsupportedNGFFVersion, from the
group's declared version, at open, before any array is read (plan 9.3, 5.17). Built for
0.6 (9.3); since 0.6 is read (9.11), the fixtures declare 0.7, a version ozp does not
know, and the names ending in "06" stand for "a version ozp does not read".

The fixtures are attributes only (no arrays at all), so a refusal proves nothing past
the root's attributes was read: reading further would fail differently. 0.4 / 0.5 / 0.6
stores and RFC 8 ("0.x") collections still open."""
import json
import os

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection, UnsupportedNGFFVersion, io
from ome_zarr_pyramid.core import ngff_plate
from ome_zarr_pyramid.core.nodes import READ_VERSIONS, InvalidCollection

MULTISCALE = {"name": "img",
              "axes": [{"name": "y", "type": "space", "unit": "micrometer"},
                       {"name": "x", "type": "space", "unit": "micrometer"}],
              "datasets": [{"path": "0", "coordinateTransformations": [
                  {"type": "scale", "scale": [1.0, 1.0]}]}]}


def _group(path, ome):
    g = zarr.open_group(str(path), mode="w", zarr_format=3)
    g.attrs.put({"ome": {"version": "0.7", **ome}})
    return str(path)


def _image06(tmp_path):
    return _group(tmp_path / "img.zarr", {"multiscales": [MULTISCALE]})


def _series06(tmp_path):
    root = _group(tmp_path / "layout.zarr", {"bioformats2raw.layout": 3, "series": ["0"]})
    _group(tmp_path / "layout.zarr" / "0", {"multiscales": [MULTISCALE]})
    return root


def _well06(tmp_path):
    return _group(tmp_path / "well.zarr", {"well": {"images": [{"path": "0"}]}})


def _plate06(tmp_path):
    # the listed well has no group: a reader that opened wells before checking the
    # version would fail on that instead
    return _group(tmp_path / "plate.zarr", {"plate": {
        "rows": [{"name": "A"}], "columns": [{"name": "1"}],
        "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}]}})


def _write(tmp_path, version, name):
    data = np.arange(2 * 16 * 24, dtype="u2").reshape(2, 16, 24)
    pyr = Pyramid().from_arrays([data], axis_order="cyx", scales=[[1, 0.5, 0.5]],
                                version=version, name=name)
    path = str(tmp_path / f"{name}.zarr")
    IO().write_pyramid(pyr, path)
    return path, data


def _relabelled(tmp_path):
    """A real 0.5 image written by ozp whose zarr.json then says 0.7."""
    path, _ = _write(tmp_path, "0.5", "relabelled")
    meta = os.path.join(path, "zarr.json")
    with open(meta, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["attributes"]["ome"]["version"] == "0.5"
    doc["attributes"]["ome"]["version"] = "0.7"
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path


FIXTURES = {"image": _image06, "series": _series06, "well": _well06, "plate": _plate06,
            "relabelled 0.5": _relabelled}
OPENERS = {"io.read_pyramid": io.read_pyramid,
           "Pyramid().from_ngff": lambda p: Pyramid().from_ngff(p),
           "PyramidCollection.from_zarr": PyramidCollection.from_zarr}


def _check(exc):
    text = str(exc.value)
    for part in ("'0.7'", "0.4", "0.5"):
        assert part in text, text
    assert isinstance(exc.value, ValueError)


@pytest.mark.parametrize("opener", OPENERS)
@pytest.mark.parametrize("fixture", FIXTURES)
def test_a_06_store_is_refused_at_open(tmp_path, fixture, opener):
    path = FIXTURES[fixture](tmp_path)
    with pytest.raises(UnsupportedNGFFVersion) as exc:
        OPENERS[opener](path)
    _check(exc)


def test_a_06_series_image_is_refused_too(tmp_path):
    root = _series06(tmp_path)
    with pytest.raises(UnsupportedNGFFVersion) as exc:
        io.read_pyramid(os.path.join(root, "0"))
    _check(exc)


@pytest.mark.parametrize("fixture, reader", [(_well06, ngff_plate.read_well),
                                             (_plate06, ngff_plate.read_plate)])
def test_the_plate_and_well_readers_refuse_06(tmp_path, fixture, reader):
    with pytest.raises(UnsupportedNGFFVersion) as exc:
        reader(fixture(tmp_path))
    _check(exc)


def test_the_error_is_exported_typed_and_names_the_way_out(tmp_path):
    with pytest.raises(UnsupportedNGFFVersion) as exc:
        io.read_pyramid(_image06(tmp_path))
    # the plate / well readers raised InvalidCollection for a version they did not
    # read: callers catching that still catch this
    assert issubclass(UnsupportedNGFFVersion, InvalidCollection)
    assert issubclass(UnsupportedNGFFVersion, ValueError)
    assert READ_VERSIONS == ("0.4", "0.5", "0.6")
    text = str(exc.value)
    assert "img.zarr" in text and "write it as 0.6" in text


def test_a_04_layout_declaring_an_older_version_is_refused(tmp_path):
    """The check is on the declared version: a 0.4-style group whose multiscale says
    0.3 is refused as well, rather than read with 0.4 rules."""
    path, _ = _write(tmp_path, "0.4", "old")
    g = zarr.open_group(path, mode="r+")
    ms = g.attrs["multiscales"]
    ms[0]["version"] = "0.3"
    g.attrs["multiscales"] = ms
    for opener in OPENERS.values():
        with pytest.raises(UnsupportedNGFFVersion, match="NGFF version '0.3'"):
            opener(path)


# -- positive controls ----------------------------------------------------------------

@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_04_and_05_images_still_open(tmp_path, version):
    path, data = _write(tmp_path, version, f"v{version}")
    pyr = io.read_pyramid(path)
    assert pyr.meta.version == version
    assert pyr.meta.zarr_format == (3 if version == "0.5" else 2)
    np.testing.assert_array_equal(np.asarray(pyr.layers["0"][:]), data)
    np.testing.assert_array_equal(np.asarray(Pyramid().from_ngff(path).layers["0"][:]), data)
    coll = PyramidCollection.from_zarr(path)
    np.testing.assert_array_equal(np.asarray(coll[()].layers["0"][:]), data)


def test_an_rfc8_collection_still_opens(tmp_path):
    first, data = _write(tmp_path, "0.5", "a")
    pyr = io.read_pyramid(first)
    # keyed by an index name no rename touches; read back keys are taken as they come
    coll = PyramidCollection({(("position", 0),): pyr, (("position", 1),): pyr})
    out = str(tmp_path / "rfc8.zarr")
    coll.write_collection(out, version="rfc8")
    for group in (out, os.path.join(out, "0")):
        assert zarr.open_group(group, mode="r").attrs["ome"]["version"] == "0.x"
    back = PyramidCollection.from_zarr(out)
    assert len(back) == 2
    np.testing.assert_array_equal(np.asarray(back[sorted(back.keys())[1]].layers["0"][:]),
                                  data)


def test_an_rfc8_node_reaching_the_ngff_reader_is_not_refused_on_its_version(tmp_path):
    """An RFC 8 multiscale group is read by node type (rfc8.open_image); read with
    io.read_pyramid it fails as before (it is not an NGFF 0.4/0.5 image), but not with
    a version refusal: "0.x" is let through explicitly, the check is not widened."""
    from ome_zarr_pyramid.core import rfc8
    first, data = _write(tmp_path, "0.5", "a")
    out = str(tmp_path / "rfc8.zarr")
    PyramidCollection({(("position", 0),): io.read_pyramid(first)}).write_collection(out)
    image = os.path.join(out, "0")
    np.testing.assert_array_equal(np.asarray(rfc8.open_image(image).layers["0"][:]), data)
    with pytest.raises(Exception) as exc:
        io.read_pyramid(image)
    assert not isinstance(exc.value, UnsupportedNGFFVersion)


def test_a_group_without_ngff_metadata_is_still_not_an_image(tmp_path):
    path = str(tmp_path / "plain.zarr")
    zarr.open_group(path, mode="w", zarr_format=3).attrs.put({"ome": {"lab": 1}})
    with pytest.raises(ValueError, match="not an OME-Zarr image") as exc:
        PyramidCollection.from_zarr(path)
    assert not isinstance(exc.value, UnsupportedNGFFVersion)


def _v3(path, attrs):
    zarr.open_group(str(path), mode="w", zarr_format=3).attrs.put(attrs)
    return str(path)


def test_a_06_well_in_a_05_plate_is_refused_at_open(tmp_path):
    plate = _v3(tmp_path / "pl.zarr", {"ome": {"version": "0.5", "plate": {
        "rows": [{"name": "A"}], "columns": [{"name": "1"}],
        "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}]}}})
    _v3(tmp_path / "pl.zarr" / "A", {})
    _v3(tmp_path / "pl.zarr" / "A" / "1", {"ome": {"version": "0.7", "well": {
        "images": [{"path": "0"}]}}})
    with pytest.raises(UnsupportedNGFFVersion, match=r"well 'A/1'.*0\.7"):
        PyramidCollection.from_zarr(plate)


def test_a_06_label_in_a_05_image_is_left_out_with_a_warning(tmp_path):
    img = str(tmp_path / "img.zarr")
    io.write_pyramid(Pyramid().from_array(np.zeros((8, 8), "uint8"), axis_order="yx",
                                          version="0.5", name="i"), img)
    label = os.path.join(img, "labels", "cells")
    io.write_pyramid(Pyramid().from_array(np.ones((8, 8), "uint8"), axis_order="yx",
                                          version="0.5", name="cells"), label)
    zarr.open_group(os.path.join(img, "labels"), mode="a").attrs.put(
        {"ome": {"version": "0.5", "labels": ["cells"]}})
    assert list(io.read_pyramid(img).labels) == ["cells"]
    with open(os.path.join(label, "zarr.json")) as f:
        doc = json.load(f)
    doc["attributes"]["ome"]["version"] = "0.7"
    with open(os.path.join(label, "zarr.json"), "w") as f:
        json.dump(doc, f)
    with pytest.warns(UserWarning, match=r"label 'cells' of .* is left out: "
                                         r"UnsupportedNGFFVersion reading .*cells.*0\.7"):
        assert list(io.read_pyramid(img).labels) == []
    with pytest.raises(UnsupportedNGFFVersion):
        io.read_labels(img, "cells")


def _image_with_labels(tmp_path, names):
    img = str(tmp_path / "img.zarr")
    io.write_pyramid(Pyramid().from_array(np.zeros((8, 8), "uint8"), axis_order="yx",
                                          version="0.5", name="i"), img)
    io.write_pyramid(Pyramid().from_array(np.ones((8, 8), "uint8"), axis_order="yx",
                                          version="0.5", name="cells"),
                     os.path.join(img, "labels", "cells"))
    zarr.open_group(os.path.join(img, "labels"), mode="a").attrs.put(
        {"ome": {"version": "0.5", "labels": names}})
    return img


def test_a_label_that_cannot_be_opened_is_left_out_with_a_warning(tmp_path):
    # any failure, not only a version ozp does not read: never dropped without a word
    img = _image_with_labels(tmp_path, ["cells", "ghost"])
    with pytest.warns(UserWarning, match=r"label 'ghost' of .* is left out: \w+ reading "
                                         r".*ghost") as caught:
        pyr = io.read_pyramid(img)
    assert list(pyr.labels) == ["cells"]
    assert len([w for w in caught if "left out" in str(w.message)]) == 1


def test_a_labels_group_that_cannot_be_read_is_left_out_with_a_warning(tmp_path):
    img = _image_with_labels(tmp_path, ["cells"])
    with open(os.path.join(img, "labels", "zarr.json"), "w") as f:
        f.write("{not json")
    with pytest.warns(UserWarning, match=r"the labels of .* are left out: \w+ reading "
                                         r".*labels"):
        pyr = io.read_pyramid(img)
    assert list(pyr.labels) == []
    assert pyr.nlayers == 1                     # the image itself still opens


def test_labels_that_open_give_no_warning(tmp_path):
    import warnings
    img = _image_with_labels(tmp_path, ["cells"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert list(io.read_pyramid(img).labels) == ["cells"]


def test_05_wells_in_a_05_plate_still_read(tmp_path):
    plate = _v3(tmp_path / "pl.zarr", {"ome": {"version": "0.5", "plate": {
        "rows": [{"name": "A"}], "columns": [{"name": "1"}],
        "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}]}}})
    _v3(tmp_path / "pl.zarr" / "A", {})
    _v3(tmp_path / "pl.zarr" / "A" / "1", {"ome": {"version": "0.5", "well": {
        "images": [{"path": "0"}]}}})
    node = ngff_plate.read_plate(plate)
    assert [w.path.path for w in node.children] == ["A/1"]
