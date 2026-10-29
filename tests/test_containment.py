"""Containment (decision C-19): every level knows its kind and parent, accessors go down
only, a selection shows what it holds, and a plate's rows / columns slice it into a part
of the plate - as micro-reader's navigation (test_select_like_micro checks the two)."""
import pytest

from test_plate_as_collection import _two_plates


def _names(wells):
    return [w.name for w in wells]


def test_each_level_knows_its_kind_and_parent():
    coll = _two_plates()                         # two plates; wells B3 (fields 0, 2), A1
    plate = coll.plates[0]
    well = plate.wells[0]
    field = well.fields[0]
    assert (coll.kind, plate.kind, well.kind, field.kind) == ("collection", "plate", "well", "field")
    assert field.parent is well and well.parent is plate and plate.parent is coll
    assert coll.parent is None
    pcoll = plate.as_collection()
    wcoll = well.as_collection()
    fcoll = field.as_collection()
    assert (pcoll.kind, wcoll.kind, fcoll.kind) == ("plate", "well", "field")
    assert pcoll.parent is coll
    assert wcoll.parent.kind == "plate" and sorted(wcoll.parent) == sorted(pcoll)
    assert fcoll.parent.kind == "well" and sorted(fcoll.parent) == sorted(wcoll)
    assert fcoll.images == tuple(coll[k] for k in field.keys)


def test_accessors_go_down_only():
    coll = _two_plates()
    pcoll = coll.plates[0].as_collection()
    wcoll = coll.plates[0].wells[0].as_collection()
    assert _names(pcoll.wells) == _names(coll.plates[0].wells)
    assert [f.index for f in wcoll.fields] == [f.index for f in coll.plates[0].wells[0].fields]
    assert pcoll.row_names == coll.plates[0].row_names
    for obj, name, message in ((pcoll, "plates", "a plate holds no plates.*holds .wells"),
                               (wcoll, "wells", "a well holds no wells.*holds .fields"),
                               (coll, "wells", "a collection holds no wells.*holds .plates"),
                               (coll.plates[0], "plates", "a plate holds no plates"),
                               (coll.plates[0].wells[0], "wells", r"a well holds no wells.*\.parent")):
        with pytest.raises(AttributeError, match=message):
            getattr(obj, name)


def test_a_selection_shows_what_it_holds():
    coll = _two_plates()
    pcoll = coll.plates[0].as_collection()
    rowb = pcoll.select(row="B")
    assert rowb.kind == "plate" and rowb.parent is coll        # still the plate, a part of it
    assert _names(rowb.wells) == ["B3"]                         # not A1, which it does not hold
    assert rowb.well_index("B3") == pcoll.well_index("B3")      # the index its images carry
    with pytest.raises(AttributeError, match="a plate holds no plates"):
        rowb.plates
    part = coll.select(plate=1, row="B")                        # a collection's selection
    assert len(part.plates) == 1 and _names(part.plates[0].wells) == ["B3"]


def test_rows_and_columns_slice_a_plate_as_select_does():
    coll = _two_plates()
    plate = coll.plates[0]
    part = plate.rows[1:]                                       # row B
    assert part.kind == "plate" and part.is_part and not plate.is_part
    assert _names(part.wells) == ["B3"]
    assert sorted(part.as_collection()) == sorted(coll.select(plate=0, row=slice(1, None)))
    cut = plate.columns[2:].rows["B"]                           # either order
    assert _names(cut.wells) == _names(plate.rows["B"].columns[2:].wells) == ["B3"]
    assert _names(plate.rows[["A", 1]].wells) == _names(plate.wells)
    assert _names(plate.columns[-1].wells) == ["B3"]            # from the layout's end
    assert part.row_names == plate.row_names and len(plate.rows) == len(plate.row_names)
    assert part.parent is coll and part.well_index("B3") == plate.well_index("B3")
    assert part.as_collection().kind == "plate"
    with pytest.raises(TypeError, match="row_names"):
        list(plate.rows)
    with pytest.raises(KeyError, match="no row 'Z'"):
        plate.rows["Z"]
