"""ozp's select and micro-reader's select give the same images for the same arguments
(they ship together: ozp 0.5.0 with micro-reader's next release). One plate CZI written
with micro-reader's own test writer, read by micro-reader, then turned into a collection."""
import os
import sys

import pytest

micro_reader = pytest.importorskip("micro_reader")
TESTS = os.path.join(os.path.dirname(micro_reader.__file__), "..", "..", "..", "tests")
if not os.path.isfile(os.path.join(TESTS, "test_hcs.py")):
    pytest.skip("micro-reader's tests (its CZI writer) are not next to it", allow_module_level=True)
sys.path.insert(0, os.path.abspath(TESTS))
from test_hcs import _plate_xml, _write                       # noqa: E402

from ome_zarr_pyramid import PyramidCollection                 # noqa: E402

WELLS = {0: (3, 5, "C5"), 1: (2, 2, "B2"), 2: (2, 2, "B2"), 3: (4, 7, "D7")}
CALLS = [dict(well=1), dict(well="B2"), dict(well=["B2", "D7"]), dict(row="C"), dict(row=2),
         dict(column="5"), dict(column=4), dict(row=["B", "C"]), dict(row=range(1, 3), column="2"),
         dict(row="H"), dict(field=[0, 1]), dict(well="B2", field=range(1, 5)),
         dict(plate=0, acquisition=0, well=0), dict(view=0),
         # each place by position and by name, as the table's columns
         dict(well_name="B2"), dict(well_name=["B2", "D7"]), dict(row_name="C"),
         dict(column_name="5", row_name=["B", "C"]), dict(row_name="C", column=4),
         dict(plate_name=""), dict(plate=""), dict(plate_name="other"), dict(well=[1, "B2"]),
         dict(acquisition_name="run 2"),
         # slices of positions, as ranges; open-ended too
         dict(well=slice(0, 2)), dict(well=slice(1, None)), dict(row=slice(2, None)),
         dict(column=slice(0, 5, 2)), dict(row=slice(1, 3), column=slice(1, None)),
         dict(field=slice(None, 1)), dict(well=[slice(2, None), "B2"]), dict(plate=slice(0, 1))]


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("plate") / "plate.czi")
    _write(path, WELLS, _plate_xml(WELLS))
    f = micro_reader.open(path, plates=True)
    yield f, PyramidCollection.from_file(f)
    f.close()


@pytest.mark.parametrize("call", CALLS, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()))
def test_the_same_images(plate, call):
    f, coll = plate
    micro = sorted(tuple(sorted(dict(image.index).items())) for image in f.select(**call))
    ozp = sorted(tuple(sorted(dict(key).items())) for key in coll.select(**call))
    assert ozp == micro


@pytest.mark.parametrize("call, error", [(dict(well=0, row=0), TypeError),
                                         (dict(well_name="B2", row_name="B"), TypeError),
                                         (dict(well_name=3), TypeError),
                                         (dict(well_nam="B2"), KeyError),
                                         (dict(well=slice(-1, None)), ValueError),
                                         (dict(row=slice(0, 4, 0)), ValueError),
                                         (dict(well_name=slice(0, 2)), TypeError),
                                         (dict(colour=1), KeyError),
                                         (dict(scene=0), KeyError)])
def test_the_same_refusals(plate, call, error):
    f, coll = plate
    with pytest.raises(error):
        f.select(**call)
    with pytest.raises(error):
        coll.select(**call)
    well = f.plates[0].wells[0]
    with pytest.raises(TypeError):
        f.select(well=well)
    with pytest.raises(TypeError):
        coll.select(well=well)                         # micro-reader's Well too


def test_every_place_keyword_is_a_table_column_with_the_same_values(plate):
    # select's keywords and the table's columns are one vocabulary: an image's own value in
    # a column selects it
    _, coll = plate
    for key, row in zip(coll, coll.table):
        for column in ("plate", "plate_name", "well", "well_name", "row", "row_name",
                       "column", "column_name", "acquisition"):
            assert key in coll.select(**{column: row[column]}), (column, row[column])
    assert {"row", "column"} <= set(coll.table.columns)


@pytest.mark.parametrize("cut", [lambda p: p.rows[1:3], lambda p: p.rows[1:4].columns[1:5],
                                 lambda p: p.columns[1:5].rows[1:4], lambda p: p.rows["D"],
                                 lambda p: p.rows[["B", 3]], lambda p: p.columns[-8:]],
                         ids=["rows", "rows-columns", "columns-rows", "name", "list", "negative"])
def test_the_same_parts_of_a_plate(plate, cut):
    # navigation is the same in both (decision C-19): a part of a plate, its wells, kind
    f, coll = plate
    micro, ozp = cut(f.plates[0]), cut(coll.plates[0])
    assert [w.name for w in micro.wells] == [w.name for w in ozp.wells]
    assert micro.kind == ozp.kind == "plate" and micro.is_part == ozp.is_part
    assert micro.row_names == ozp.row_names and micro.column_names == ozp.column_names


def test_the_same_levels(plate):
    f, coll = plate
    for micro, ozp in ((f.plates[0], coll.plates[0]),
                       (f.plates[0].wells[0], coll.plates[0].wells[0]),
                       (f.plates[0].wells[0].fields[0], coll.plates[0].wells[0].fields[0])):
        assert micro.kind == ozp.kind
        assert micro.parent.kind == ozp.parent.kind or (micro.parent.kind, ozp.parent.kind) == \
            ("file", "collection")                       # a file is micro-reader's collection
        for name in ("plates", "wells", "fields"):
            assert hasattr(micro, name) == hasattr(ozp, name), (micro.kind, name)
