"""NGFF validation in ozp itself (plan 5.23).

The semantic check runs on every write without any extra: each image's multiscales as
they are saved (``check_multiscales``) and each 0.6 scene ozp writes (``check_scene``).
The store check is opt-in: ``ozp.validate(path)`` and the writers' ``validate=True``,
against the official schemas vendored in the package (the ``validate`` extra)."""
import os
import sys

import numpy as np
import pytest
import zarr

import ome_zarr_pyramid as ozp
from ome_zarr_pyramid import IO, NGFFValidationError, Pyramid, PyramidCollection
from ome_zarr_pyramid.core.validation import SCHEMAS, check_multiscales, check_scene

XML = ('<?xml version="1.0" encoding="UTF-8"?>\n'
       '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
       '<Image ID="Image:0" Name="v"/></OME>')
SPACE = {"type": "space", "unit": "micrometer"}


def _pyr(version="0.5", value=1):
    a = np.full((2, 4, 32, 32), value, dtype="u2")
    return Pyramid().from_arrays([a], axis_order="czyx", version=version,
                                 unit_list=["micrometer"] * 3, name="img")


def _ms(axes="zyx", scale=(1.0, 0.5, 0.5), translation=None):
    cts = [{"type": "scale", "scale": list(scale)}]
    if translation is not None:
        cts.append({"type": "translation", "translation": list(translation)})
    return {"multiscales": [{
        "axes": [{"name": a, **SPACE} for a in axes],
        "datasets": [{"path": "0", "coordinateTransformations": cts}]}]}


def _ms06(outputs=("physical",), systems=("physical",), sequence=False):
    def level(i, out):
        t = {"type": "scale", "scale": [1.0, 2.0 ** i, 2.0 ** i]}
        if sequence:
            t = {"type": "sequence", "transformations": [
                t, {"type": "translation", "translation": [0.0, 0.0, 0.0]}]}
        return {"path": str(i), "coordinateTransformations": [
            {**t, "input": {"path": str(i)}, "output": {"name": out}}]}
    return {"multiscales": [{
        "coordinateSystems": [{"name": s, "axes": [{"name": a, **SPACE} for a in "zyx"]}
                              for s in systems],
        "datasets": [level(i, o) for i, o in enumerate(outputs)]}]}


# -- check_multiscales -----------------------------------------------------------------

def test_valid_multiscales_have_no_problems():
    assert check_multiscales(_ms(), "0.5") == []
    assert check_multiscales(_ms(translation=(0, 1, 2)), "0.4") == []
    assert check_multiscales(_ms06(outputs=("physical", "physical")), "0.6") == []
    assert check_multiscales(_ms06(sequence=True), "0.6") == []


@pytest.mark.parametrize("ms, found", [
    (_ms(scale=(1.0, 0.0, 0.5)), "is not > 0"),
    (_ms(scale=(1.0, 0.5)), "is not 3 numbers"),
    (_ms(translation=(0, 1)), "translation [0, 1] is not 3 numbers"),
    (_ms(axes="x", scale=(1.0,)), "1 axes"),
    (_ms(axes="tczyxw", scale=(1.0,) * 6), "6 axes"),
    (_ms(axes="zyy"), "not unique names"),
])
def test_a_bad_multiscale_is_named(ms, found):
    assert any(found in p for p in check_multiscales(ms, "0.5")), check_multiscales(ms, "0.5")


def test_two_or_three_space_axes():
    ms = _ms()
    ms["multiscales"][0]["axes"][0]["type"] = "channel"
    ms["multiscales"][0]["axes"][1]["type"] = "channel"
    assert any("1 space axes" in p for p in check_multiscales(ms, "0.5"))


def test_a_translation_before_the_scale_is_named():
    ms = _ms()
    ms["multiscales"][0]["datasets"][0]["coordinateTransformations"] = [
        {"type": "translation", "translation": [0, 0, 0]}, {"type": "scale", "scale": [1, 1, 1]}]
    assert any("a level has a scale, then at most one translation" in p
               for p in check_multiscales(ms, "0.5"))


def test_06_levels_map_into_one_system_of_the_multiscale():
    found = check_multiscales(_ms06(outputs=("physical", "other"),
                                    systems=("physical", "other")), "0.6")
    assert any("every level maps into the same system" in p for p in found), found
    found = check_multiscales(_ms06(outputs=("stage",)), "0.6")
    assert any("which is none of the multiscale's systems" in p for p in found), found
    ms = _ms06()
    ms["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["input"]["path"] = "1"
    assert any("starts at '1', not at the level" in p for p in check_multiscales(ms, "0.6"))
    ms = _ms06()
    ms["multiscales"][0]["datasets"][0]["coordinateTransformations"] *= 2
    assert any("2 transformations; 0.6 has one per level" in p
               for p in check_multiscales(ms, "0.6"))


# -- check_scene -----------------------------------------------------------------------

def _scene(paths=("0", "1"), name="physical", system_axes="zyx"):
    return {"coordinateSystems": [{"name": "stage", "axes": [{"name": a} for a in system_axes]}],
            "coordinateTransformations": [
                {"type": "translation", "translation": [0, 0, 0],
                 "input": {"name": name, "path": p}, "output": {"name": "stage"}}
                for p in paths]}


IMAGES = {"0": ("physical", ["z", "y", "x"]), "1": ("physical", ["z", "y", "x"])}


def test_a_valid_scene_has_no_problems():
    assert check_scene(_scene(), IMAGES) == []


def test_a_scene_names_each_problem():
    assert any("no edge attaches it" in p for p in check_scene(_scene(paths=("0",)), IMAGES))
    assert any("'7' is no image below the scene" in p
               for p in check_scene(_scene(paths=("0", "1", "7")), IMAGES))
    assert any("whose intrinsic system is 'physical'" in p
               for p in check_scene(_scene(name="other"), IMAGES))
    assert any("(5.18(c))" in p for p in check_scene(_scene(system_axes="yx"), IMAGES))
    scene = _scene()
    scene["coordinateTransformations"][0]["output"]["name"] = "elsewhere"
    assert any("not one of the scene's systems" in p for p in check_scene(scene, IMAGES))


# -- the semantic check on every write -------------------------------------------------

def test_saving_metadata_ngff_does_not_allow_stops_before_writing_it(tmp_path):
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(_pyr(), path)
    before = dict(zarr.open_group(path, mode="r").attrs)
    back = IO().read_pyramid(path)
    back.meta.update_all_datasets(scale=[1.0, 0.0, 0.5, 0.5])
    with pytest.raises(NGFFValidationError, match="is not > 0") as err:
        back.meta.save_changes()
    assert "metadata was not written" in str(err.value)
    assert err.value.problems
    assert dict(zarr.open_group(path, mode="r").attrs) == before


def test_every_06_scene_is_checked_as_it_is_written(tmp_path, monkeypatch):
    from ome_zarr_pyramid.core import validation
    from test_spatial_collection import _cs, _file, _placed
    images = [_placed({"series": 0}, "stage", value=1),
              _placed({"series": 1}, "stage", y=50, value=2)]
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")]))
    monkeypatch.setattr(validation, "check_scene", lambda scene, below: ["planted problem"])
    with pytest.raises(NGFFValidationError, match="planted problem"):
        coll.write_series(str(tmp_path / "s.ome.zarr"), version="0.6")


# -- the store check, opt-in -----------------------------------------------------------

def test_the_schemas_ship_with_the_package():
    for version in ("0.4", "0.5", "0.6"):
        assert (SCHEMAS / version / "strict_image.schema").is_file()
    assert (SCHEMAS / "0.6" / "scene.schema").is_file()


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_write_pyramid_validate_true(tmp_path, version):
    pytest.importorskip("jsonschema")
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(_pyr(version), path, ome_xml=XML, validate=True)
    assert ozp.validate(path) == []


def test_validate_true_raises_for_a_store_ngff_does_not_allow(tmp_path, monkeypatch):
    """A writer that leaves a stray OME group behind (as ozp 0.3.0 did) is caught."""
    pytest.importorskip("jsonschema")
    from ome_zarr_pyramid.core import io as ozp_io
    real = ozp_io._write_ome_xml

    def stray(path, xml, series, version, storage_options=None, **kw):
        real(path, xml, series, version, storage_options, **kw)
        zarr.open_group(os.path.join(str(path), "OME"), mode="w", zarr_format=3).attrs[
            "ome"] = {"version": "0.5", "series": ["0"]}

    monkeypatch.setattr(ozp_io, "_write_ome_xml", stray)
    with pytest.raises(NGFFValidationError, match="belongs only under a bioformats2raw root"):
        IO().write_pyramid(_pyr(), str(tmp_path / "img.ome.zarr"), ome_xml=XML, validate=True)


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_collection_writers_validate_the_whole_store(tmp_path, version):
    pytest.importorskip("jsonschema")
    from test_write_plate import _file
    PyramidCollection.from_file(_file()).write_plate(str(tmp_path / "p.ome.zarr"),
                                                     version=version, validate=True)
    coll = PyramidCollection({(("series", i),): _pyr(version, i) for i in range(2)})
    coll.write_series(str(tmp_path / "s.ome.zarr"), version=version, validate=True)
    written = coll.write(str(tmp_path), "one", validate=True)
    assert len(written) == 2 and all(ozp.validate(p) == [] for p in written.values())


def test_without_the_extra_validate_true_fails_before_writing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    path = tmp_path / "img.ome.zarr"
    with pytest.raises(ImportError, match=r"ome_zarr_pyramid\[validate\]"):
        IO().write_pyramid(_pyr(), str(path), validate=True)
    assert not path.exists()
    with pytest.raises(ImportError, match=r"ome_zarr_pyramid\[validate\]"):
        ozp.validate(str(tmp_path))
    IO().write_pyramid(_pyr(), str(path))                 # without validate: unaffected
    assert ozp.validate(str(path), schemas=False) == []   # the hierarchy needs no extra


def test_rfc8_groups_are_skipped(tmp_path):
    pytest.importorskip("jsonschema")
    coll = PyramidCollection({(("series", i),): _pyr("0.5", i) for i in range(2)})
    path = str(tmp_path / "c.zarr")
    with pytest.warns(Warning):
        coll.write_collection(path)
    assert ozp.validate(path) == []


def test_a_missing_store_is_a_problem(tmp_path):
    assert ozp.validate(str(tmp_path / "nothing.zarr"), schemas=False) == [
        f"{tmp_path / 'nothing.zarr'}: no such store"]


# -- strict-schema findings warn (decision 9.12-S) -----------------------------------------

def _label_without_colors(tmp_path, version="0.5"):
    img = _pyr(version)
    lbl = Pyramid().from_label_arrays([np.ones((2, 4, 32, 32), "u1")], axis_order="czyx")
    path = str(tmp_path / "img.ome.zarr")
    return img.add_image_label(lbl, "cells"), path


def test_strict_findings_are_returned_only_on_request(tmp_path):
    pytest.importorskip("jsonschema")
    pyr, path = _label_without_colors(tmp_path)
    IO().write_pyramid(pyr, path)
    assert ozp.validate(path) == []
    strict = ozp.validate(path, strict=True)
    assert len(strict) == 1 and "strict_label.schema: 'colors' is a required" in strict[0]


def test_validate_true_warns_for_strict_findings_and_does_not_raise(tmp_path):
    pytest.importorskip("jsonschema")
    pyr, path = _label_without_colors(tmp_path)
    with pytest.warns(ozp.NGFFValidationWarning, match="'colors' is a required property"):
        IO().write_pyramid(pyr, path, validate=True)


def test_a_store_ozp_makes_alone_raises_no_warning(tmp_path):
    import warnings
    pytest.importorskip("jsonschema")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ozp.NGFFValidationWarning)
        IO().write_pyramid(_pyr("0.5"), str(tmp_path / "img.ome.zarr"), validate=True)


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_ozp_writes_an_empty_multiscale_metadata(tmp_path, version):
    """strict_image requires `metadata`; ozp writes {} where the image has none."""
    pytest.importorskip("jsonschema")
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(_pyr(version), path)
    attrs = dict(zarr.open_group(path, mode="r").attrs)
    assert (attrs.get("ome") or attrs)["multiscales"][0]["metadata"] == {}
    assert ozp.validate(path, strict=True) == []


def test_maximumfieldcount_is_derived_per_acquisition():
    """The most fields of each acquisition in one well, counted from what is written; a
    value the source gives is kept."""
    from ome_zarr_pyramid.core.ngff_plate import plate_from_attrs, plate_to_attrs

    def well(*acquisitions):
        return {"well": {"version": "0.4", "images": [
            {"path": str(i), "acquisition": a} for i, a in enumerate(acquisitions)]}}

    plate = {"plate": {"columns": [{"name": "1"}, {"name": "2"}], "rows": [{"name": "A"}],
                       "acquisitions": [{"id": 0}, {"id": 1, "maximumfieldcount": 9}],
                       "version": "0.4",
                       "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0},
                                 {"path": "A/2", "rowIndex": 0, "columnIndex": 1}]}}
    node = plate_from_attrs(plate, {"A/1": well(0, 0, 1), "A/2": well(0, 1)})
    assert plate_to_attrs(node)[0]["plate"]["acquisitions"] == [
        {"id": 0, "maximumfieldcount": 2}, {"id": 1, "maximumfieldcount": 9}]
