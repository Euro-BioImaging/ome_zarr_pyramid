"""NGFF 0.4 / 0.5 plates into the collection model and back: the JSON of real plates
(idr0010 0.5, MoBIE and Fractal 0.4) is written back unchanged, a plate built in memory
is written in either version, and a broken plate is refused with every problem named."""
import copy
import glob
import os

import pytest
import zarr

from ome_zarr_pyramid.core.ngff_plate import (
    plate_from_attrs, plate_to_attrs, read_plate, write_plate_metadata)
from ome_zarr_pyramid.core.nodes import (
    CollectionNode, InvalidCollection, LabelsAttribute, MultiscaleNode, PlateAttribute,
    PlateItem, Ref, WellAttribute, is_plate, is_well)

PLATES = os.path.join(os.environ.get("MICROSCOPY_DATA",
                                     os.path.expanduser("~/Desktop/data/microscopy")),
                      "plates", "ome-zarr")
IDR0010 = os.path.join(PLATES, "idr0010", "idr0010_76-45.ome.zarr")
MOBIE = os.path.join(PLATES, "mobie-minimal", "minimal-mobie", "zarr-files",
                     "20230616-Plate2-umsfgfpmodc-drug.zarr")
REAL = [p for p in [IDR0010, MOBIE] + sorted(glob.glob(os.path.join(PLATES, "zenodo-*", "*.zarr")))
        if os.path.isdir(p)]


def _attrs(path):
    root = zarr.open_group(path, mode="r")
    return root, root.attrs.asdict()


def _expected(attrs):
    """What ozp writes for a read plate: the same attributes, with a custom key found
    inside `ome` moved beside it (ozp's custom-metadata rule; idr0010's `_creator`)."""
    attrs = copy.deepcopy(attrs)
    ome = attrs.get("ome")
    if isinstance(ome, dict):
        from ome_zarr_pyramid.core.pyramid import NGFFMetadataHandler
        for key in [k for k in ome if k != "version" and k not in NGFFMetadataHandler.NGFF_KEYS]:
            attrs[key] = ome.pop(key)
    return attrs


@pytest.mark.skipif(not REAL, reason="the local NGFF plates are not here")
@pytest.mark.parametrize("path", REAL, ids=os.path.basename)
def test_real_plates_write_back_unchanged(path, tmp_path):
    root, attrs = _attrs(path)
    node = read_plate(path)
    plate_attrs, wells = plate_to_attrs(node)
    assert plate_attrs == _expected(attrs)
    for well_path, well_attrs in wells.items():
        assert well_attrs == _expected(root[well_path].attrs.asdict())
    # and on disk, in the version it was read as (under the same folder name: a plate
    # whose JSON has no name is named after its folder)
    out = str(tmp_path / os.path.basename(path))
    write_plate_metadata(node, out)
    out_root, out_attrs = _attrs(out)
    assert out_attrs == _expected(attrs)
    assert out_root.metadata.zarr_format == root.metadata.zarr_format
    assert read_plate(out) == node


@pytest.mark.skipif(not os.path.isdir(IDR0010), reason="idr0010 is not here")
def test_idr0010_maps_into_the_model():
    plate = read_plate(IDR0010)
    assert is_plate(plate) and plate.name == "idr0010_76-45"     # its JSON name is ""
    attr = plate.attributes["plate"]
    assert (len(attr.rows), len(attr.columns), len(plate)) == (12, 32, 384)
    assert attr.rows[1] == PlateItem("B", "B")
    assert plate.attributes["bioformats2raw.layout"] == 3
    assert plate.attributes["_creator"]["name"] == "ome2024-ngff-challenge"
    well = plate["B1"]
    assert is_well(well) and well.path.path == "B/1"
    assert well.attributes["well"] == WellAttribute(Ref("B"), Ref("1"))
    assert [(f.name, f.path.path) for f in well] == [("0", "0")]


@pytest.mark.skipif(not os.path.isdir(MOBIE), reason="the MoBIE plate is not here")
def test_a_04_plate_written_as_05(tmp_path):
    plate = read_plate(MOBIE)
    out = str(tmp_path / "plate.zarr")
    write_plate_metadata(plate, out, version="0.5")
    root, attrs = _attrs(out)
    assert root.metadata.zarr_format == 3
    assert attrs["barcode"] == "N/A" and attrs["order_name"] == "None"   # beside `ome`
    assert attrs["ome"]["version"] == "0.5" and "version" not in attrs["ome"]["plate"]
    assert root["G/15"].attrs.asdict() == {"ome": {"version": "0.5",
                                                    "well": {"images": [{"path": "0"}]}}}
    again = read_plate(out)
    assert again.ngff_fields["version"] == "0.5"
    assert plate_to_attrs(again, "0.4")[0] == _attrs(MOBIE)[1]           # and back to 0.4


def _built():
    """A two-acquisition plate built in memory (the 'wide' form)."""
    plate = CollectionNode("screen 1", attributes={"plate": PlateAttribute(
        rows=[PlateItem("A", "A"), PlateItem("B", "B")],
        columns=[PlateItem("1", "1"), PlateItem("2", "2")],
        acquisitions=[PlateItem("0", "first"), PlateItem("1", "second",
                                                           {"starttime": 1700000000})]),
        "lab:barcode": "P-001"})
    for row, col in (("A", "2"), ("B", "1")):
        well = plate.add(CollectionNode(f"{row}{col}", attributes={
            "well": WellAttribute(Ref(row), Ref(col))}))
        for acq in ("0", "1"):
            well.add(MultiscaleNode(acq, attributes={"acquisition": Ref(acq)}))
    return plate


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_a_plate_built_in_memory_is_written_in_either_version(tmp_path, version):
    out = str(tmp_path / "plate.zarr")
    write_plate_metadata(_built(), out, version=version)
    root, attrs = _attrs(out)
    plate_json = attrs["ome"]["plate"] if version == "0.5" else attrs["plate"]
    assert plate_json["wells"] == [{"path": "A/2", "rowIndex": 0, "columnIndex": 1},
                                   {"path": "B/1", "rowIndex": 1, "columnIndex": 0}]
    assert plate_json["acquisitions"] == [{"id": 0, "name": "first"},
                                          {"id": 1, "name": "second", "starttime": 1700000000}]
    assert plate_json["name"] == "screen 1" and plate_json["field_count"] == 2
    assert ("version" in plate_json) == (version == "0.4")
    assert attrs["lab:barcode"] == "P-001"
    well = root["B/1"].attrs.asdict()
    images = (well["ome"] if version == "0.5" else well)["well"]["images"]
    assert images == [{"path": "0", "acquisition": 0}, {"path": "1", "acquisition": 1}]
    back = read_plate(out)
    assert [w.name for w in back] == ["A2", "B1"]
    assert back["A2"]["1"].attributes["acquisition"] == Ref("1")
    assert back.attributes["plate"] == _built().attributes["plate"]


def test_only_the_wide_form_and_no_labels_are_written_in_04_05():
    tall = _built()
    tall["A2"].add(CollectionNode("acquisition 2"))
    with pytest.raises(ValueError, match="lists images only"):
        plate_to_attrs(tall, "0.5")
    labelled = _built()
    labelled["A2"].add(MultiscaleNode("nuclei", id="lab", attributes={
        "labels": LabelsAttribute([Ref("A2_0")])}))
    labelled["A2"]["0"].id = "A2_0"
    with pytest.raises(ValueError, match="'labels' group"):
        plate_to_attrs(labelled, "0.4")
    with pytest.raises(ValueError, match="written as"):
        plate_to_attrs(_built(), "rfc8")


def _plate_json(**changes):
    plate = {"columns": [{"name": "1"}, {"name": "2"}], "rows": [{"name": "A"}],
             "acquisitions": [{"id": 0}, {"id": 1}], "version": "0.4",
             "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0},
                       {"path": "A/2", "rowIndex": 0, "columnIndex": 1}]}
    plate.update(changes)
    return {"plate": plate}


def _well(*acquisitions):
    return {"well": {"version": "0.4", "images": [
        {"path": str(i), "acquisition": a} for i, a in enumerate(acquisitions)]}}


def test_a_valid_synthetic_plate_reads():
    plate = plate_from_attrs(_plate_json(), {"A/1": _well(0, 1), "A/2": _well(1)})
    assert [w.name for w in plate] == ["A1", "A2"]
    assert plate_to_attrs(plate)[0] == _plate_json()


def test_a_broken_plate_is_refused_naming_every_problem():
    attrs = _plate_json(
        acquisitions=[{"id": 0}, {"id": 0}],                               # id twice
        wells=[{"path": "A/1", "rowIndex": 0, "columnIndex": 0},
               {"path": "A/1", "rowIndex": 0, "columnIndex": 0},           # path twice
               {"path": "A/3", "rowIndex": 0, "columnIndex": 2},           # out of range
               {"path": "B/1", "rowIndex": 1, "columnIndex": 0},           # out of range
               {"path": "A/2", "rowIndex": 0, "columnIndex": 1},           # no group
               {"path": "X/1", "rowIndex": 0, "columnIndex": 0}])          # same row/col
    wells = {"A/1": _well(0, 7), "X/1": _well(0)}                          # 7: no such acq.
    with pytest.raises(InvalidCollection) as err:
        plate_from_attrs(attrs, wells)
    text = str(err.value)
    for part in ("acquisition id '0' is used twice",
                 "well 'A/1' is listed twice",
                 "well 'A/3': columnIndex 2 is not an index into the plate's 2 columns",
                 "well 'B/1': rowIndex 1 is not an index into the plate's 1 rows",
                 "well 'A/2' is listed in the plate but has no group",
                 "well 'X/1': another well already sits at row 'A', column '1'",
                 "acquisition '7' is not one of the plate's acquisitions"):
        assert part in text, part


@pytest.mark.parametrize("attrs, message", [
    ({"multiscales": []}, "no 'plate' metadata"),
    ({"ome": {"version": "0.6", "plate": {"rows": [], "columns": []}}}, "version '0.6'"),
    (_plate_json(rows=[{"label": "A"}]), r"rows\[0\] has no name"),
    (_plate_json(acquisitions=[{"id": "first"}]), "id 'first' is not an integer"),
])
def test_what_is_not_a_readable_plate(attrs, message):
    with pytest.raises(InvalidCollection, match=message):
        plate_from_attrs(attrs, {"A/1": _well(0), "A/2": _well(0)})


def test_a_well_image_listed_twice_is_refused():
    well = {"well": {"images": [{"path": "0"}, {"path": "0"}]}}
    with pytest.raises(InvalidCollection, match="image '0' is listed twice"):
        plate_from_attrs(_plate_json(acquisitions=[]), {"A/1": well, "A/2": _well()})
