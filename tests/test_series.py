"""The bioformats2raw series layout: read as a collection of its series (keyed like
micro-reader's series, each opened on first use) and written again with the same root
metadata, OME group and OME-XML; any collection can be written in the layout."""
import os

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection
from ome_zarr_pyramid.core.nodes import InvalidCollection, MultiscaleNode

XML = ('<?xml version="1.0" encoding="UTF-8"?>\n<OME xmlns="http://www.openmicroscopy.org/'
       'Schemas/OME/2016-06"><Image ID="Image:0" Name="first"/><Image ID="Image:1" '
       'Name="second"/></OME>\n')


def _image(value, version):
    data = np.full((2, 16, 20), value, dtype="u2")
    data[:, 0, 0] = 100 + value
    return Pyramid().from_arrays([data], axis_order="cyx", scales=[[1, 0.5, 0.5]],
                                 version=version, name=f"series {value}"), data


def _layout(tmp_path, version, n=2, series=True, xml=True, extra=None):
    """A layout as bioformats2raw writes it: series 0..n-1, the root marker, and an OME
    group listing the series with the OME-XML."""
    path = str(tmp_path / "file.ome.zarr")
    data = []
    for i in range(n):
        pyr, d = _image(i, version)
        IO().write_pyramid(pyr, os.path.join(path, str(i)))
        data.append(d)
    zf = 3 if version == "0.5" else 2
    root = zarr.open_group(path, mode="a", zarr_format=zf)
    marker = {"bioformats2raw.layout": 3}
    root.attrs.put({"ome": {"version": "0.5", **marker}, **(extra or {})} if zf == 3
                   else {**marker, **(extra or {})})
    if series:
        ome = root.require_group("OME")
        names = [str(i) for i in range(n)]
        ome.attrs.put({"series": names} if zf == 2 else {"ome": {"version": "0.5", "series": names}})
        if xml:
            with open(os.path.join(path, "OME", "METADATA.ome.xml"), "w", newline="") as f:
                f.write(XML)
    return path, data


def _opened(coll):
    return [n.name for n in coll.tree.walk() if isinstance(n, MultiscaleNode) and n.loaded]


def _validate(path):
    """ome-zarr-models' bioformats2raw model of the output's version accepts it (asked
    directly: its open_ome_zarr does not try that model for 0.5)."""
    pytest.importorskip("ome_zarr_models")
    group = zarr.open_group(path, mode="r")
    if group.metadata.zarr_format == 3:
        from ome_zarr_models.v05.bioformats2raw import BioFormats2Raw
    else:
        from ome_zarr_models.v04.bioformats2raw import BioFormats2Raw
    BioFormats2Raw.from_zarr(group)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_a_layout_round_trips(tmp_path, version):
    src, data = _layout(tmp_path, version, extra={"lab:run": 7})
    coll = PyramidCollection.from_zarr(src)
    assert list(coll) == [(("series", 0),), (("series", 1),)] and _opened(coll) == []
    np.testing.assert_array_equal(np.asarray(coll[{"series": 1}].layers["0"]), data[1])
    assert _opened(coll) == ["1"]
    out = str(tmp_path / "copy.ome.zarr")
    assert sorted(coll.write_series(out)) == ["0", "1"]
    for group in ("", "OME"):
        a = zarr.open_group(os.path.join(src, group), mode="r").attrs.asdict()
        b = zarr.open_group(os.path.join(out, group), mode="r").attrs.asdict()
        # as read, plus the eubi: keys of 0.4 / 0.5 on the root (plan 5.7)
        assert {k: v for k, v in b.items() if not k.startswith("eubi:")} == a, group
    root = zarr.open_group(out, mode="r").attrs.asdict()
    assert root["eubi:version"] == 1
    assert root["eubi:images"]["1"] == {"index": {"series": 1}, "coordinate_system": None}
    with open(os.path.join(out, "OME", "METADATA.ome.xml"), newline="") as f:
        assert f.read() == XML
    back = PyramidCollection.from_zarr(out)
    for key in coll:
        np.testing.assert_array_equal(np.asarray(back[key].layers["0"]), np.asarray(coll[key].layers["0"]))
        assert back[key].meta.tag == coll[key].meta.tag
    _validate(out)


def test_without_an_ome_group_the_series_are_the_consecutive_integers(tmp_path):
    src, _ = _layout(tmp_path, "0.5", n=3, series=False)
    os.rename(os.path.join(src, "2"), os.path.join(src, "4"))       # a gap ends the series
    coll = PyramidCollection.from_zarr(src)
    assert [n.name for n in coll.tree] == ["0", "1"]
    out = str(tmp_path / "out.zarr")
    coll.write_series(out)
    # none in: ozp writes the OME group with a minimal OME-XML, one Image per series
    assert zarr.open_group(os.path.join(out, "OME"), mode="r").attrs.asdict() == {
        "ome": {"version": "0.5", "series": ["0", "1"]}}
    with open(os.path.join(out, "OME", "METADATA.ome.xml"), encoding="utf-8") as f:
        assert f.read().count("<Image ") == 2


def test_problems_are_named(tmp_path):
    src, _ = _layout(tmp_path, "0.4")
    zarr.open_group(os.path.join(src, "OME"), mode="a").attrs["series"] = ["0", "1", "2"]
    zarr.open_group(os.path.join(src, "1"), mode="a").attrs.put({"x": 1})   # not an image
    with pytest.raises(InvalidCollection) as err:
        PyramidCollection.from_zarr(src)
    assert "series '2' is listed but has no image" in str(err.value)
    assert "series '1' is not an image" in str(err.value)


def test_any_collection_is_written_as_series_in_either_version(tmp_path):
    from test_collection import FakeImage
    images = [FakeImage(np.full((2, 8, 8), v, "u2"), "cyx", index={"series": v}) for v in (0, 1, 2)]
    coll = PyramidCollection.from_file(type("F", (), {"images": images, "path": None, "plates": []})(),
                                       version="0.4")
    out = str(tmp_path / "out.zarr")
    xml3 = XML.replace('</OME>', '<Image ID="Image:2" Name="third"/></OME>')
    coll.write_series(out, version="0.5", ome_xml=xml3)          # one Image per series
    root = zarr.open_group(out, mode="r")
    assert root.metadata.zarr_format == 3
    attrs = root.attrs.asdict()
    assert attrs["ome"] == {"version": "0.5", "bioformats2raw.layout": 3}
    assert attrs["eubi:version"] == 1 and sorted(attrs["eubi:images"]) == ["0", "1", "2"]
    assert zarr.open_group(os.path.join(out, "OME"), mode="r").attrs.asdict() == {
        "ome": {"version": "0.5", "series": ["0", "1", "2"]}}
    back = PyramidCollection.from_zarr(out)
    assert len(back) == 3 and back[{"series": 2}].meta.version == "0.5"
    np.testing.assert_array_equal(np.asarray(back[{"series": 2}].layers["0"]), np.full((2, 8, 8), 2))
    _validate(out)


def test_an_existing_target_needs_overwrite(tmp_path):
    src, _ = _layout(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(src)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        coll.write_series(src)
    out = str(tmp_path / "out.zarr")
    coll.write_series(out)
    coll.write_series(out, overwrite=True)
    assert len(PyramidCollection.from_zarr(out)) == 2


def test_a_plate_in_the_layout_stays_a_plate():
    path = os.path.join(os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy")),
                        "plates", "ome-zarr", "idr0010", "idr0010_76-45.ome.zarr")
    if not os.path.isdir(path):
        pytest.skip("idr0010 is not here")
    coll = PyramidCollection.from_zarr(path)
    assert len(coll.plates) == 1 and coll.tree.attributes["bioformats2raw.layout"] == 3
