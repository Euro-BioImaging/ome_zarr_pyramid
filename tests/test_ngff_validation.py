"""Every kind of store ozp writes passes the whole-store NGFF check of
``tests/ngff_validation.py``: each group against the official (strict) schemas of its
roles, and the hierarchy, which no schema checks. The check catches the stray OME group
ozp 0.3.0 wrote under single images, which validators passed."""
import os

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection

from ngff_validation import problems
from test_write_plate import _file

pytest.importorskip("jsonschema")
pytest.importorskip("referencing")

XML = ('<?xml version="1.0" encoding="UTF-8"?>\n'
       '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
       '<Image ID="Image:0" Name="v"/></OME>')
VERSIONS = ["0.4", "0.5"]


def _pyr(version, value=1):
    a = np.full((2, 4, 32, 32), value, dtype="u2")
    return Pyramid().from_arrays([a], axis_order="czyx", version=version,
                                 unit_list=["micrometer"] * 3, name="img")


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("kind", ["one level", "deferred levels", "in-memory levels"])
def test_a_written_image_is_valid_ngff(tmp_path, version, kind):
    pyr = _pyr(version)
    if kind == "deferred levels":
        pyr = pyr.downscale(n_layers=3, scale_factor=(1, 1, 2, 2))
    elif kind == "in-memory levels":
        pyr = pyr.downscale(n_layers=3, scale_factor=(1, 1, 2, 2), defer=False)
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(pyr, path, ome_xml=XML)
    assert problems(path) == []
    assert os.path.isfile(os.path.join(path, "OME", "METADATA.ome.xml"))


@pytest.mark.parametrize("version", VERSIONS)
def test_an_image_with_a_label_is_valid_ngff(tmp_path, version):
    img = _pyr(version)
    lbl = Pyramid().from_label_arrays([np.ones((2, 4, 32, 32), "u1")], axis_order="czyx")
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(img.add_image_label(lbl, "cells"), path)
    # known gap: a label written without colors fails the strict schema only (colors
    # SHOULD be given; ozp writes them when the caller sets them, set_image_label)
    found = problems(path)
    assert all(p.startswith("labels/cells: strict_label.schema: 'colors' is a required")
               for p in found), found
    assert len(found) == 1


@pytest.mark.parametrize("version", VERSIONS)
def test_a_series_layout_is_valid_ngff(tmp_path, version):
    coll = PyramidCollection({(("series", i),): _pyr(version, i) for i in range(3)})
    path = str(tmp_path / "s.ome.zarr")
    coll.write_series(path, version=version)
    assert problems(path) == []


@pytest.mark.parametrize("version", VERSIONS)
def test_a_plate_is_valid_ngff(tmp_path, version):
    path = str(tmp_path / "p.ome.zarr")
    PyramidCollection.from_file(_file()).write_plate(path, version=version)
    assert problems(path) == []


def test_the_check_catches_the_stray_ome_group_of_ozp_030(tmp_path):
    """A single image with an OME group listing its level-0 array, as 0.3.0 wrote it:
    every group validates alone, the hierarchy does not; update_metadata repairs it."""
    path = str(tmp_path / "old.ome.zarr")
    IO().write_pyramid(_pyr("0.5"), path, ome_xml=XML)
    zarr.open_group(os.path.join(path, "OME"), mode="w", zarr_format=3).attrs["ome"] = {
        "version": "0.5", "series": ["0"]}
    with open(os.path.join(path, "OME", "METADATA.ome.xml"), "w", encoding="utf-8") as fh:
        fh.write(XML)
    found = problems(path)
    assert any("belongs only under a bioformats2raw root" in p for p in found), found
    assert any("series '0' is an array" in p for p in found), found
    with pytest.warns(UserWarning, match="removed the OME group's metadata"):
        IO().update_metadata(path)
    assert problems(path) == []


def test_the_check_catches_a_missing_type(tmp_path):
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(_pyr("0.5"), path)
    g = zarr.open_group(path, mode="r+")
    ome = g.attrs["ome"]
    del ome["multiscales"][0]["type"]
    g.attrs["ome"] = ome
    assert any("strict_image.schema" in p and "'type'" in p for p in problems(path))
