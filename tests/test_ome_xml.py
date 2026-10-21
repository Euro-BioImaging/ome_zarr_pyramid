"""write_pyramid(ome_xml=...): the OME-XML written with the pyramid (roadmap D5), as
bioformats2raw lays it out and eubi-bridge wrote it, in the same write: the output is not
opened again afterwards. Remote: tests/test_write_remote.py."""
import json
import os

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid

XML = ('<?xml version="1.0" encoding="UTF-8"?>\n'
       '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
       '<Image ID="Image:0" Name="µ-test"/></OME>')


def _pyr(version="0.5"):
    a = np.arange(2 * 8 * 32 * 32, dtype="u2").reshape(2, 8, 32, 32)
    return Pyramid().from_arrays([a], axis_order="czyx", version=version)


def _read(path):
    with open(os.path.join(path, "OME", "METADATA.ome.xml"), encoding="utf-8") as fh:
        return fh.read()


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_written_beside_the_levels_with_its_series(tmp_path, version):
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(_pyr(version), path, overwrite=True, ome_xml=XML, ome_series=[2])
    assert _read(path) == XML                                 # as given, UTF-8
    attrs = dict(zarr.open_group(os.path.join(path, "OME"), mode="r").attrs)
    if version == "0.4":
        assert attrs == {"series": ["2"]}
    else:
        assert attrs == {"ome": {"version": "0.5", "series": ["2"]}}
    back = IO().read_pyramid(path)                            # the image reads as before
    assert back.meta.resolution_paths == ["0"]


def test_the_default_series_and_every_write_route(tmp_path):
    for name, pyr, kw in (
            ("plain", _pyr(), {}),
            ("windows", _pyr(), {"window_from_data": True}),
            ("deferred", _pyr().downscale(n_layers=2, scale_factor=(1, 2, 2, 2)), {})):
        path = str(tmp_path / f"{name}.ome.zarr")
        IO().write_pyramid(pyr, path, overwrite=True, ome_xml=XML, **kw)
        assert _read(path) == XML, name
        meta = json.load(open(os.path.join(path, "OME", "zarr.json"), encoding="utf-8"))
        assert meta["attributes"]["ome"]["series"] == ["0"], name


def test_without_ome_xml_nothing_is_written(tmp_path):
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(_pyr(), path, overwrite=True)
    assert not os.path.exists(os.path.join(path, "OME"))


def test_an_empty_document_is_refused_before_anything_is_written(tmp_path):
    path = str(tmp_path / "img.ome.zarr")
    with pytest.raises(ValueError, match="ome_xml"):
        IO().write_pyramid(_pyr(), path, overwrite=True, ome_xml="  ")
    assert not os.path.exists(path)


def test_local_xml_is_written_under_its_own_name(tmp_path, monkeypatch):
    """On local disk the XML goes straight to METADATA.ome.xml: zarr's own write uses a
    temporary name 37 characters longer, which crossed Windows' 260-character limit in a
    deep output folder (eubi-bridge reference case lif_fret)."""
    from zarr.storage import LocalStore
    real_set = LocalStore.set

    async def no_xml_through_the_store(self, key, value, *args, **kwargs):
        assert not key.endswith("METADATA.ome.xml"), "went through zarr's temporary file"
        return await real_set(self, key, value, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "set", no_xml_through_the_store)
    path = str(tmp_path / "x.ome.zarr")
    IO().write_pyramid(_pyr(), path, ome_xml=XML)
    with open(os.path.join(path, "OME", "METADATA.ome.xml"), "rb") as fh:
        assert fh.read() == XML.encode("utf-8")              # byte for byte


# -- 5.26 / 8.31: the OME-XML of a series layout (write_series) ----------------------

def _series_pyr(value, axes="cyx", dtype="u2", shape=(2, 8, 12), name=None):
    return Pyramid().from_arrays([np.full(shape, value, dtype)], axis_order=axes,
                                 name=name or f"image {value}", version="0.5")


def _images(xml):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    assert root.tag == "{http://www.openmicroscopy.org/Schemas/OME/2016-06}OME"
    return [child for child in root if child.tag.endswith("}Image")]


def _pixels(image):
    (pixels,) = [c for c in image if c.tag.endswith("}Pixels")]
    return pixels


def test_minimal_ome_xml_on_its_own():
    from ome_zarr_pyramid.core.ngff_series import minimal_ome_xml
    xml = minimal_ome_xml([
        ("0", _series_pyr(1, "yx", "u1", (8, 12), name="plain")),
        ("1", _series_pyr(1, "cyx", "u2", (3, 8, 12))),
        ("2", _series_pyr(1, "tczyx", "f4", (2, 3, 4, 8, 12))),
        ("3", _series_pyr(1, "zyx", "f8", (4, 8, 12))),
        ("4", _series_pyr(1, "yx", "bool", (8, 12)))])
    images = _images(xml)
    assert [i.get("ID") for i in images] == [f"Image:{n}" for n in range(5)]
    assert images[0].get("Name") == "plain"
    got = [(_pixels(i).get("ID"), _pixels(i).get("DimensionOrder"), _pixels(i).get("Type"),
            tuple(int(_pixels(i).get(f"Size{a}")) for a in "XYZCT")) for i in images]
    assert got == [("Pixels:0", "XYZCT", "uint8", (12, 8, 1, 1, 1)),
                   ("Pixels:1", "XYCZT", "uint16", (12, 8, 1, 3, 1)),
                   ("Pixels:2", "XYZCT", "float", (12, 8, 4, 3, 2)),
                   ("Pixels:3", "XYZCT", "double", (12, 8, 4, 1, 1)),
                   ("Pixels:4", "XYZCT", "bit", (12, 8, 1, 1, 1))]
    for image in images:
        assert [c.tag.split("}")[1] for c in _pixels(image)] == ["MetadataOnly"]


def test_a_pixel_type_ome_xml_cannot_name_raises():
    from ome_zarr_pyramid import OMEXMLError
    from ome_zarr_pyramid.core.ngff_series import minimal_ome_xml
    with pytest.raises(OMEXMLError, match="int64"):
        minimal_ome_xml([("0", _series_pyr(1, dtype="i8"))])


def _coll(n=4):
    from ome_zarr_pyramid import PyramidCollection
    from ome_zarr_pyramid.core.collection import SERIES_INDEX
    return PyramidCollection({((SERIES_INDEX, i),): _series_pyr(i) for i in range(n)})


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_write_series_always_writes_a_fitting_ome_xml(tmp_path, version):
    out = str(tmp_path / "s.zarr")
    _coll().write_series(out, version=version)
    images = _images(_read(out))
    assert [i.get("ID") for i in images] == ["Image:0", "Image:1", "Image:2", "Image:3"]
    assert [i.get("Name") for i in images] == [f"image {i}" for i in range(4)]
    assert all(_pixels(i).get("Type") == "uint16" and _pixels(i).get("SizeC") == "2" for i in images)
    series = ["0", "1", "2", "3"]
    if version == "0.5":
        meta = json.load(open(os.path.join(out, "OME", "zarr.json"), encoding="utf-8"))
        assert meta["attributes"] == {"ome": {"version": "0.5", "series": series}}
    else:
        attrs = json.load(open(os.path.join(out, "OME", ".zattrs"), encoding="utf-8"))
        assert attrs == {"series": series}
        # a 0.4 layout says its version nowhere as ome.version
        for folder, _, files in os.walk(out):
            if ".zattrs" in files:
                doc = json.load(open(os.path.join(folder, ".zattrs"), encoding="utf-8"))
                assert "ome" not in doc, folder
            assert "zarr.json" not in files, folder


def test_an_explicit_ome_xml_with_another_image_count_raises(tmp_path):
    from ome_zarr_pyramid import OMEXMLError
    out = tmp_path / "s.zarr"
    with pytest.raises(OMEXMLError) as err:
        _coll().write_series(str(out), ome_xml=XML)
    assert "holds 1 Image element," in str(err.value) and "but 4 series" in str(err.value)
    assert "omit ome_xml" in str(err.value) and not out.exists()
    with pytest.raises(OMEXMLError, match="not an XML document"):
        _coll().write_series(str(out), ome_xml="<OME><Image ID='Image:0'></OME>")
    four = XML.replace("</OME>", "".join(f'<Image ID="Image:{i}"/>' for i in (1, 2, 3)) + "</OME>")
    _coll().write_series(str(out), ome_xml=four)
    assert _read(str(out)) == four                     # as given


def _two_series(tmp_path):
    from ome_zarr_pyramid import PyramidCollection
    from test_series import XML as TWO, _layout
    src, _ = _layout(tmp_path, "0.5")
    return PyramidCollection.from_zarr(src), TWO


def test_a_carried_ome_xml_that_no_longer_fits_is_replaced_with_a_warning(tmp_path):
    from ome_zarr_pyramid import PyramidCollection
    coll, _ = _two_series(tmp_path)
    key = list(coll)[1]
    for i, part in enumerate((coll.select(**dict(key)), PyramidCollection({key: coll[key]}))):
        out = str(tmp_path / f"one{i}.zarr")
        with pytest.warns(UserWarning, match=r"holds 2 Image elements and does not fit the 1 "
                                             r"series written; it is replaced"):
            part.write_series(out)
        (image,) = _images(_read(out))
        assert image.get("ID") == "Image:0"


def test_a_carried_ome_xml_that_fits_is_written_unchanged(tmp_path):
    import warnings
    from ome_zarr_pyramid import PyramidCollection
    coll, two = _two_series(tmp_path)
    for i, whole in enumerate((coll, PyramidCollection({k: coll[k] for k in coll}))):
        out = str(tmp_path / f"whole{i}.zarr")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            whole.write_series(out)
        assert not [w for w in caught if issubclass(w.category, UserWarning)]
        with open(os.path.join(out, "OME", "METADATA.ome.xml"), newline="") as f:
            assert f.read() == two


def test_a_file_collection_written_as_series_gets_a_minimal_ome_xml(tmp_path):
    from types import SimpleNamespace

    from ome_zarr_pyramid import PyramidCollection
    from test_collection import FakeImage
    images = [FakeImage(np.full((3, 8, 10), v, "u1"), "cyx", index={"series": v}, name=f"s{v}")
              for v in range(3)]
    coll = PyramidCollection.from_file(SimpleNamespace(images=images, path=None, plates=[]))
    out = str(tmp_path / "f.zarr")
    coll.write_series(out, version="0.5")
    found = _images(_read(out))
    assert [(i.get("ID"), i.get("Name"), _pixels(i).get("Type"), _pixels(i).get("SizeX"),
             _pixels(i).get("SizeY"), _pixels(i).get("SizeC")) for i in found] == [
        (f"Image:{v}", f"s{v}", "uint8", "10", "8", "3") for v in range(3)]


def test_the_ome_group_follows_the_output_version():
    from ome_zarr_pyramid.core.io import _write_ome_xml
    from ome_zarr_pyramid.core.ngff_plate import zarr_format_of
    assert (zarr_format_of("0.4"), zarr_format_of("0.5")) == (2, 3)
    for version in ("0.6", "rfc8", None):
        with pytest.raises(ValueError, match="ozp writes the versions"):
            zarr_format_of(version)
    with pytest.raises(ValueError, match="ozp writes the versions"):
        _write_ome_xml("never-written.zarr", XML, ["0"], "0.6")
    assert not os.path.exists("never-written.zarr")
