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
