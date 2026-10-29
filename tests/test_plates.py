"""coll.plates, navigated as in micro-reader: a plate file opened with micro-reader has
the same plates, wells, fields and indices in ozp (real vendor plates, and a hand-made
file that needs no micro-reader), and an NGFF plate reads into the same structure."""
import glob
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ome_zarr_pyramid import PyramidCollection
from ome_zarr_pyramid.core.nodes import InvalidCollection, is_plate
from ome_zarr_pyramid.core.plates import Acquisition, Plate, Well

from test_collection_zarr import _opened, _plate

DATA = os.path.join(os.environ.get("MICROSCOPY_DATA",
                                   os.path.expanduser("~/Desktop/data/microscopy")), "plates")


class FakeImage:
    """A plate image with micro-reader's attribute names."""

    def __init__(self, index, value):
        self._data = np.full((2, 8, 12), value, dtype="u2")
        self.axes, self.shape, self.dtype = "cyx", self._data.shape, self._data.dtype
        self.index, self.name, self.metadata = index, f"img{value}", None
        self.resolutions = [self]

    def __getitem__(self, key):
        return self._data[key]


def _file(plates):
    """Wells B3 and A1 of a 2 x 3 plate; B3 has fields 0 and 2, each in two views."""
    images, value = [], 0
    for w, fields in ((0, (0, 2)), (1, (0,))):
        for f in fields:
            for view in (0, 1):
                images.append(FakeImage({"plate": 0, "acquisition": 0, "well": w, "field": f,
                                         "view": view}, value))
                value += 1
    return SimpleNamespace(path="/data/screen.idx.xml", images=images, plates=plates)


PLATE = SimpleNamespace(name="", rows=("A", "B"), columns=("1", "2", "3"), acquisitions=(),
                        wells=(SimpleNamespace(row=1, column=2, name="B3"),
                               SimpleNamespace(row=0, column=0, name="A1")))


def test_a_plate_file_navigates_as_in_micro_reader():
    coll = PyramidCollection.from_file(_file([PLATE]))
    (plate,) = coll.plates
    assert plate == Plate("", ("A", "B"), ("1", "2", "3"),
                          (Well(1, 2, "B3"), Well(0, 0, "A1")), ())
    assert plate.well_index("A1") == 1 and is_plate(plate.node)
    b3 = plate.wells[0]
    assert [f.index for f in b3.fields] == [0, 2]
    field = b3.fields[1]
    assert [dict(k)["view"] for k in field.keys] == [0, 1]
    with pytest.raises(ValueError, match="several view: pick one with select"):
        field.tiles
    one = field.select(view=1).image
    assert np.asarray(one.layers["0"]).max() == 3          # well 0, field 2, view 1
    with pytest.raises(KeyError, match="no well 'C1'"):
        plate.well_index("C1")
    # the tree holds the same pyramids as the flat collection; a field of two views is
    # a collection of its own in the well (plan 3.17)
    well_node = plate.node["B3"]
    assert (plate.node.kind, plate.node.index, well_node.kind, well_node.index) == (
        "plate", 0, "well", 0)
    assert [n.name for n in well_node] == ["field0", "field2"]
    field2 = well_node["field2"]
    assert (field2.kind, field2.index, [n.name for n in field2]) == (
        "field", 2, ["acquisition0_view0", "acquisition0_view1"])
    assert field2["acquisition0_view1"].pyramid is one
    # a selection keeps every well (index "well": w stays wells[w]), fields narrowed
    picked = coll.select(view=0).plates[0]
    assert [w.name for w in picked.wells] == ["B3", "A1"]
    assert [len(f.keys) for f in picked.wells[0].fields] == [1, 1]


def test_a_file_that_is_not_a_plate_has_no_plates():
    coll = PyramidCollection.from_file(_file([]))
    assert coll.plates == []
    # the images still have a tree: each under the root, named by its index values
    assert coll.tree.kind is None and coll.tree.name == "screen.idx"
    assert [n.name for n in coll.tree][:2] == ["plate0_acquisition0_well0_field0_view0",
                                              "plate0_acquisition0_well0_field0_view1"]
    assert len(coll.tree) == 6


def test_a_well_outside_the_layout_is_refused():
    bad = SimpleNamespace(**{**vars(PLATE), "rows": ("A",)})
    with pytest.raises(InvalidCollection, match=r"wells \['B3'\] outside its 1 x 3 layout"):
        PyramidCollection.from_file(_file([bad]))


def test_an_ngff_plate_navigates_the_same_way(tmp_path):
    path, data = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(path)
    (plate,) = coll.plates
    assert plate.name == "screen" and plate.row_names == ("A", "B") and plate.column_names == ("1", "2")
    assert plate.wells == (Well(0, 1, "A2"), Well(1, 0, "B1"))
    assert plate.acquisitions == (Acquisition(0, "first"), Acquisition(1, "second"))
    field = plate.wells[1].fields[0]
    assert len(field.keys) == 2 and _opened(coll) == []
    image = field.select(acquisition=1).image
    np.testing.assert_array_equal(np.asarray(image.layers["0"]), data[(1, 1)])
    assert _opened(coll) == ["1"]


def _plate_files():
    patterns = ("**/Index.idx.xml", "**/*.HTD", "**/*.xdce", "**/MeasurementData.mlf")
    return sorted({p for pat in patterns for p in glob.glob(os.path.join(DATA, pat), recursive=True)})


def _summary(plates, keys_of):
    return [(p.name, tuple(p.row_names), tuple(p.column_names), tuple((a.id, a.name) for a in p.acquisitions),
             [(w.row, w.column, w.name, [(f.index, sorted(keys_of(f))) for f in w.fields])
              for w in p.wells]) for p in plates]


@pytest.mark.skipif(not _plate_files(), reason="the local vendor plates are not here")
@pytest.mark.parametrize("path", _plate_files(), ids=lambda p: os.path.relpath(p, DATA)[:60])
def test_vendor_plates_match_micro_reader(path):
    micro_reader = pytest.importorskip("micro_reader")
    try:
        f = micro_reader.open(path)
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"micro-reader cannot open it: {exc}")
    if not f.plates:
        pytest.skip("micro-reader reads no plate from it")
    try:
        coll = PyramidCollection.from_file(f)
    except InvalidCollection as exc:
        if "outside its" in str(exc):
            pytest.skip(f"micro-reader puts a well outside the plate layout: {exc}")
        raise
    mr = _summary(f.plates, lambda fl: [tuple(sorted((k, v) for k, v in i.index.items()
                                                     if k != "resolution")) for i in fl.images])
    ozp = _summary(coll.plates, lambda fl: [tuple(sorted(k)) for k in fl.keys])
    assert ozp == mr
    for well, mr_well in zip(coll.plates[0].wells, f.plates[0].wells):
        for field, mr_field in zip(well.fields, mr_well.fields):
            pyr = field.images[0]
            assert tuple(pyr.layers[pyr.meta.resolution_paths[0]].shape) == tuple(mr_field.images[0].shape)
