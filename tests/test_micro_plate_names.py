"""Plates from micro-reader 0.2 and 0.3 (ozp 0.4.1).

micro-reader 0.3.0 made ``Plate.rows`` / ``Plate.columns`` slicers of the plate (iterating
one raises TypeError) and moved the names to ``Plate.row_names`` / ``column_names``. ozp
0.4.0 iterated ``plate.rows``, so every plate read through it failed once micro-reader
0.3.0 was installed. Both styles give the same tree now.
"""
from types import SimpleNamespace

import pytest

from ome_zarr_pyramid.core.plates import plate_nodes, tree_from_file


class Slicer:
    """micro-reader 0.3's ``plate.rows``: a length, no iteration."""

    def __init__(self, names):
        self._n = len(names)

    def __len__(self):
        return self._n

    def __iter__(self):
        raise TypeError("plate.rows is a slicer, not a sequence: its names are plate.row_names")


def _file(style):
    rows, columns = ("A", "B"), ("1", "2", "3")
    wells = [SimpleNamespace(name="A2", row=0, column=1), SimpleNamespace(name="B3", row=1, column=2)]
    plate = SimpleNamespace(name="p", wells=wells, acquisitions=[])
    if style == "0.3":
        plate.rows, plate.columns = Slicer(rows), Slicer(columns)
        plate.row_names, plate.column_names = rows, columns
    else:
        plate.rows, plate.columns = rows, columns
    return SimpleNamespace(plates=[plate], path="plate.xml")


@pytest.mark.parametrize("style", ["0.2", "0.3"])
def test_a_plate_from_either_micro_reader(style):
    entries = {(("plate", 0), ("well", w), ("field", 0)): object() for w in (0, 1)}
    tree = tree_from_file(_file(style), entries)
    (plate,) = plate_nodes(tree)
    attr = plate.attributes["plate"]
    assert [r.name for r in attr.rows] == ["A", "B"]
    assert [c.name for c in attr.columns] == ["1", "2", "3"]
    assert [w.name for w in plate.children] == ["A2", "B3"]


def test_a_well_outside_the_layout_names_it_in_either_style():
    f = _file("0.3")
    f.plates[0].wells.append(SimpleNamespace(name="C1", row=2, column=0))
    with pytest.raises(Exception, match=r"C1.*2 x 3 layout \(rows \['A', 'B'\]"):
        tree_from_file(f, {})
