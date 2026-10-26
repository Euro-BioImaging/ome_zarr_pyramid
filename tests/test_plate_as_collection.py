"""select as in micro-reader, and a plate / well / field of a collection as a collection
of its own (``coll.plates[p].as_collection()``): the navigation stays micro-reader's."""
import copy
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ome_zarr_pyramid import IO, PyramidCollection
from ome_zarr_pyramid.core.collection import INDEX_NAMES

from test_write_plate import PLATE, _file


def _two_plates():
    images = list(_file().images)
    for image in _file().images:
        other = copy.copy(image)
        other.index = {**image.index, "plate": 1}
        images.append(other)
    second = copy.deepcopy(PLATE)
    return PyramidCollection.from_file(SimpleNamespace(path="/data/screen.idx.xml",
                                                       images=images, plates=[PLATE, second]))


def _pixels(pyr):
    return np.asarray(pyr.layers[pyr.meta.resolution_paths[0]])


# -- select, as micro-reader's ------------------------------------------------------

def test_index_names_are_micro_readers():
    micro = pytest.importorskip("micro_reader._base")
    assert INDEX_NAMES == micro.INDEX_NAMES


def test_select_knows_every_micro_reader_index_and_the_keys_own():
    coll = _two_plates()
    assert len(coll.select(view=0)) == 0              # known, no image has it: empty
    assert len(coll.select(plate=1)) == 3
    with pytest.raises(KeyError, match=r"unknown indices \['colour'\]"):
        coll.select(colour=1)
    with pytest.raises(KeyError, match="'scene' is now called 'series'"):
        coll.select(scene=0)
    own = PyramidCollection({(("lab_run", 0),): coll[next(iter(coll))]})
    assert len(own.select(lab_run=0)) == 1            # a name the keys carry is known


def test_field_select_checks_names_as_micro_reader_does():
    field = _two_plates().plates[0].wells[0].fields[0]
    assert field.select(view=1).keys == ()
    with pytest.raises(KeyError, match="unknown indices"):
        field.select(colour=1)


# -- as_collection ------------------------------------------------------------------

def test_a_plate_as_a_collection_holds_that_plate_only(tmp_path):
    coll = _two_plates()
    second = coll.plates[1].as_collection()
    assert sorted(second) == sorted(coll.select(plate=1))
    assert len(second.plates) == 1 and second.tree is coll.plates[1].node
    nav = second.plates[0]                             # navigation as before, keys kept
    assert [w.name for w in nav.wells] == [w.name for w in coll.plates[1].wells]
    assert nav.wells[0].fields[1].keys == coll.plates[1].wells[0].fields[1].keys
    with pytest.warns(UserWarning, match="disagrees with the plate's own numbering"):
        back = IO().read(second.write(str(tmp_path / "plate1.zarr")))
        list(back)
    # a plate store holds one plate, read back as plate 0 (as write_plate(plate=1) does)
    assert len(back) == 3 and len(back.plates) == 1
    for key in second:
        np.testing.assert_array_equal(_pixels(back[tuple((n, 0 if n == "plate" else v)
                                                          for n, v in key)]),
                                      _pixels(coll[key]))


def test_a_well_and_a_field_as_collections():
    coll = _two_plates()
    well = coll.plates[0].wells[0]
    wc = well.as_collection()
    assert sorted(wc) == sorted(k for f in well.fields for k in f.keys)
    assert wc.tree is well.node and wc.name == well.name
    field = well.fields[1]
    fc = field.as_collection()
    assert list(fc) == list(field.keys) and fc.only() is not None


def test_the_graph_is_cut_to_the_branch():
    coll = _two_plates()
    second = coll.plates[1].as_collection()
    assert set(second.system_of) <= set(second)


# -- loud where a mistake would be silent ------------------------------------------

def test_an_empty_collection_is_not_written(tmp_path):
    coll = _two_plates()
    empty = coll.select(plate=0, well=7)              # no such well: nothing selected
    assert len(empty) == 0
    for write in (lambda: empty.write(str(tmp_path / "a.zarr"), version="rfc8"),
                  lambda: empty.write(str(tmp_path / "b.zarr")),
                  lambda: empty.write_plate(str(tmp_path / "c.zarr")),
                  lambda: empty.write_series(str(tmp_path / "d.zarr")),
                  lambda: empty.write_collection(str(tmp_path / "e.zarr")),
                  lambda: empty.write(str(tmp_path / "f"), "img")):
        with pytest.raises(ValueError, match="holds no image.*well_index"):
            write()
    assert not any(tmp_path.iterdir())


def test_from_file_with_a_path_names_the_readers(tmp_path):
    with pytest.raises(TypeError, match=r"micro_reader.open.*from_zarr\(path\) or io.read\(path\)"):
        PyramidCollection.from_file(tmp_path / "x.ome.zarr")
    with pytest.raises(TypeError, match="not a path"):
        PyramidCollection.from_file(str(tmp_path / "x.czi"))


# -- select: names, rows, columns, several values ------------------------------------

def test_select_by_well_name_row_and_column():
    coll = _two_plates()                              # wells B3 (fields 0, 2) and A1 (field 0)
    plate = coll.plates[0]
    b3 = plate.well_index("B3")
    by_place = coll.select(plate=0, well=b3)
    assert sorted(coll.select(plate=0, well="B3")) == sorted(by_place)
    assert len(coll.select(well="B3")) == 4           # B3 of both plates
    row_b = plate.rows.index("B")
    assert sorted(coll.select(plate=0, row="B")) == sorted(by_place)
    assert sorted(coll.select(plate=0, row=row_b)) == sorted(by_place)   # 0-based
    column_3 = plate.columns.index("3")
    assert sorted(coll.select(plate=0, row="B", column=column_3)) == sorted(by_place)
    assert sorted(coll.select(plate=0, column=["1", "3"])) == sorted(coll.select(plate=0))
    assert len(coll.select(plate=0, row=range(0, 1))) == 1                 # row A: A1
    assert len(coll.select(row="H")) == 0             # a row with no image: nothing


def test_select_takes_several_values_for_any_index():
    coll = _two_plates()
    assert len(coll.select(field=[0, 2])) == len(coll)
    assert len(coll.select(plate=[1], field=range(1, 3))) == 1
    assert len(coll.select(well=["B3", "A1"], plate=0)) == 3


def test_select_refuses_what_is_no_index_value():
    coll = _two_plates()
    plate = coll.plates[0]
    with pytest.raises(TypeError, match="a Well object is no index value"):
        coll.select(well=plate.wells[0])
    with pytest.raises(TypeError, match="not both"):
        coll.select(well=0, row="B")
    with pytest.raises(TypeError, match="Plate object"):
        coll.select(plate=[plate])
    flat = PyramidCollection({(("series", 0),): coll[next(iter(coll))]})
    with pytest.raises(ValueError, match="holds no plate"):
        flat.select(row=0)
    with pytest.raises(KeyError, match="unknown indices"):
        plate.wells[0].fields[0].select(row=0)        # a field sits in one well


# -- drop_empty -----------------------------------------------------------------------

def test_drop_empty_drops_what_holds_no_image_and_numbers_plates_and_wells_anew(tmp_path):
    coll = _two_plates()
    part = coll.select(plate=1, well="B3", field=2)
    assert len(part.plates) == 2 and len(part.plates[1].wells) == 2     # the tree as it was
    small = part.drop_empty()
    assert len(small.plates) == 1 and [w.name for w in small.plates[0].wells] == ["B3"]
    (key,) = small
    assert dict(key) == {"plate": 0, "acquisition": 0, "well": 0, "field": 2}   # field kept
    np.testing.assert_array_equal(_pixels(small[key]), _pixels(coll[next(iter(part))]))
    assert sorted(part) != sorted(small) and len(coll.plates) == 2       # nothing else changed
    # the keys read back as they are, in RFC 8 and as an NGFF 0.5 plate (no longer a selection)
    for version in ("rfc8", "0.5"):
        back = PyramidCollection.from_zarr(small.write(str(tmp_path / f"{version}.zarr"),
                                                       version=version))
        assert sorted(back) == sorted(small), version
        np.testing.assert_array_equal(_pixels(back[key]), _pixels(small[key]))


def test_drop_empty_keeps_a_full_collection_as_it_is():
    coll = _two_plates()
    same = coll.drop_empty()
    assert sorted(same) == sorted(coll) and len(same.plates) == 2
    assert [len(p.wells) for p in same.plates] == [len(p.wells) for p in coll.plates]


def test_the_graph_follows_the_new_keys():
    from ome_zarr_pyramid.core.collection import _renamed_graph
    from ome_zarr_pyramid.core.spatial import Transformation
    a = (("plate", 1), ("well", 3), ("field", 0))
    b = (("plate", 1), ("well", 5), ("field", 0))
    rename = {a: (("plate", 0), ("well", 0), ("field", 0)),
              b: (("plate", 0), ("well", 1), ("field", 0))}
    edge = Transformation("translation", a, b, {"translation": [0.0, 1.0]})
    seq = Transformation("sequence", a, "stage", {"transformations": [
        Transformation("scale", None, None, {"scale": [1.0, 2.0]})]})
    out = _renamed_graph({"coordinate_systems": ["stage"], "system_of": {a: "stage"},
                          "coordinate_transformations": [edge, seq]}, rename)
    assert out["system_of"] == {rename[a]: "stage"}
    first, second = out["coordinate_transformations"]
    assert (first.input, first.output) == (rename[a], rename[b])
    assert first.params == {"translation": [0.0, 1.0]}
    assert (second.input, second.output) == (rename[a], "stage")
