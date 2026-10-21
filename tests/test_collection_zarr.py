"""PyramidCollection.from_zarr: a plate, a well and a single image open as collections
keyed as micro-reader keys plate images; nothing but metadata is read until an image is
used, and an image then reads the same pixels as io.read_pyramid on its path."""
import glob
import os

import numpy as np
import pytest

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection
from ome_zarr_pyramid.core.collection import _tree_entries
from ome_zarr_pyramid.core.ngff_plate import write_plate_metadata
from ome_zarr_pyramid.core.nodes import (
    CollectionNode, LabelsAttribute, MultiscaleNode, PlateAttribute, PlateItem, Ref,
    WellAttribute)

PLATES = os.path.join(os.environ.get("MICROSCOPY_DATA",
                                     os.path.expanduser("~/Desktop/data/microscopy")),
                      "plates", "ome-zarr")
REAL = [p for p in [os.path.join(PLATES, "idr0010", "idr0010_76-45.ome.zarr"),
                    os.path.join(PLATES, "mobie-minimal", "minimal-mobie", "zarr-files",
                                 "20230616-Plate2-umsfgfpmodc-drug.zarr")]
        + sorted(glob.glob(os.path.join(PLATES, "zenodo-*", "*.zarr"))) if os.path.isdir(p)]


def _image(value, version):
    data = np.full((2, 16, 24), value, dtype="u2")
    data[:, 0, 0] = np.arange(2) + 100 * value
    pyr = Pyramid().from_arrays([data], axis_order="cyx", scales=[[1, 0.5, 0.5]],
                                version=version, name=f"img{value}")
    return pyr, data


def _plate(tmp_path, version):
    """A 2 x 2 plate with wells A2 and B1, each with one field in two acquisitions."""
    plate = CollectionNode("screen", attributes={"plate": PlateAttribute(
        rows=[PlateItem("A", "A"), PlateItem("B", "B")],
        columns=[PlateItem("1", "1"), PlateItem("2", "2")],
        acquisitions=[PlateItem("0", "first"), PlateItem("1", "second")])})
    path = tmp_path / "screen.zarr"
    data = {}
    for w, (row, col) in enumerate((("A", "2"), ("B", "1"))):
        well = plate.add(CollectionNode(f"{row}{col}", attributes={
            "well": WellAttribute(Ref(row), Ref(col))}))
        for acq in (0, 1):
            well.add(MultiscaleNode(str(acq), attributes={"acquisition": Ref(str(acq))}))
            pyr, data[(w, acq)] = _image(10 * w + acq, version)
            IO().write_pyramid(pyr, str(path / row / col / str(acq)))
    write_plate_metadata(plate, str(path), version=version)
    return str(path), data


def _opened(coll):
    return [n.name for n in coll.tree.walk() if isinstance(n, MultiscaleNode) and n.loaded]


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_a_plate_opens_lazily_keyed_like_micro_reader(tmp_path, version):
    path, data = _plate(tmp_path, version)
    coll = PyramidCollection.from_zarr(path)
    assert coll.name == "screen" and coll.tree.name == "screen"
    assert {tuple(name for name, _ in k) for k in coll} == {
        ("plate", "acquisition", "well", "field")}
    assert sorted(tuple(dict(k)[n] for n in ("well", "acquisition", "field")) for k in coll) \
        == [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    assert coll.varying == ("acquisition", "well")
    second = coll.select(acquisition=1)
    assert len(second) == 2 and _opened(coll) == []           # nothing read yet
    key = {"plate": 0, "acquisition": 1, "well": 1, "field": 0}
    pyr = coll[key]
    assert _opened(coll) == ["1"]                              # only that image
    np.testing.assert_array_equal(np.asarray(pyr.layers[pyr.meta.resolution_paths[0]]),
                                  data[(1, 1)])
    assert coll[key] is pyr                                    # opened once


def test_a_well_on_its_own_and_a_single_image(tmp_path):
    path, data = _plate(tmp_path, "0.5")
    well = PyramidCollection.from_zarr(os.path.join(path, "B", "1"))
    assert sorted(well) == [(("acquisition", 0), ("field", 0)), (("acquisition", 1), ("field", 0))]
    np.testing.assert_array_equal(np.asarray(well[{"acquisition": 0, "field": 0}].layers["0"]),
                                  data[(1, 0)])
    image = PyramidCollection.from_zarr(os.path.join(path, "A", "2", "1"))
    assert list(image) == [()] and image.tree.loaded is False
    np.testing.assert_array_equal(np.asarray(image[()].layers["0"]), data[(0, 1)])


def test_what_from_zarr_does_not_open(tmp_path):
    import zarr
    zarr.open_group(str(tmp_path / "plain.zarr"), mode="w").attrs["x"] = 1
    with pytest.raises(ValueError, match="not an OME-Zarr image, well or plate"):
        PyramidCollection.from_zarr(str(tmp_path / "plain.zarr"))
    g = zarr.open_group(str(tmp_path / "series.zarr"), mode="w", zarr_format=3)
    g.attrs["ome"] = {"version": "0.5", "bioformats2raw.layout": 3}
    # a series layout with no series yet opens, empty (tests/test_series.py for the rest)
    assert len(PyramidCollection.from_zarr(str(tmp_path / "series.zarr"))) == 0


def test_tree_entries_follow_tall_acquisitions_and_skip_derived_images():
    plate = CollectionNode("p", attributes={"plate": PlateAttribute(
        rows=[PlateItem("A")], columns=[PlateItem("1")],
        acquisitions=[PlateItem("acq_0"), PlateItem("acq_1")])})
    well = plate.add(CollectionNode("A1", attributes={"well": WellAttribute(Ref("A"), Ref("1"))}))
    for a in ("acq_0", "acq_1"):
        sub = well.add(CollectionNode(a, attributes={"acquisition": Ref(a)}))
        sub.add(MultiscaleNode("f0", id=f"{a}_f0"))
        sub.add(MultiscaleNode("f1"))
        sub.add(MultiscaleNode("nuclei", attributes={"labels": LabelsAttribute([Ref(f"{a}_f0")])}))
        sub.add(MultiscaleNode("corrected", attributes={"source": [{"id": f"{a}_f0"}]}))
    keys = [(dict(k)["acquisition"], dict(k)["field"], node.name) for k, node in _tree_entries(plate)]
    assert keys == [(0, 0, "f0"), (0, 1, "f1"), (1, 0, "f0"), (1, 1, "f1")]


@pytest.mark.skipif(not REAL, reason="the local NGFF plates are not here")
@pytest.mark.parametrize("path", REAL, ids=os.path.basename)
def test_real_plates_open_without_reading_images(path):
    coll = PyramidCollection.from_zarr(path)
    n_images = sum(len(well) for well in coll.tree)
    assert len(coll) == n_images and _opened(coll) == []
    key = next(iter(coll))
    node = coll._pyramids[key]
    field, direct = coll[key], IO().read_pyramid(node.loader.path_of(node))
    base = field.meta.resolution_paths[0]
    assert field.meta.resolution_paths == direct.meta.resolution_paths
    np.testing.assert_array_equal(np.asarray(field.layers[base]), np.asarray(direct.layers[base]))
