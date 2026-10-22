"""The tree of a collection opened from a file (plan 3.17), the series keys (7.4) and
`select` on an index name the collection does not have (9.8).

`PyramidCollection.from_file` takes any object with `images`: the tree is built from
the images' indices alone (and the file's `plates`, when it has them). Hand-made images
prove it needs no micro-reader; the last tests run micro-reader 0.1.0 on real files.
"""
import json
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from ome_zarr_pyramid import PyramidCollection
from ome_zarr_pyramid.core import rfc8
from ome_zarr_pyramid.core.nodes import CollectionNode, MultiscaleNode

from test_collection import FakeImage

DATA = os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy"))


def _img(index, value=0):
    return FakeImage(np.full((2, 8, 8), value, "u2"), "cyx", index=index, name=f"img{value}")


def _file(*indices, plates=None, path="/data/sample.czi"):
    images = [_img(index, v) for v, index in enumerate(indices)]
    return SimpleNamespace(path=path, images=images, **({"plates": plates} if plates else {}))


def _shape(node):
    """The tree as nested (name, kind, index, children) / image names."""
    if isinstance(node, MultiscaleNode):
        return node.name
    return (node.name, node.kind, node.index, [_shape(n) for n in node])


# -- trees -------------------------------------------------------------------------

def test_a_plain_file_has_a_root_with_its_one_image():
    coll = PyramidCollection.from_file(_file({"series": 0}))
    assert list(coll) == [(("series", 0),)]
    assert isinstance(coll.tree, CollectionNode)
    assert _shape(coll.tree) == ("sample", None, None, ["series0"])
    assert coll.tree["series0"].pyramid is coll[{"series": 0}]


def test_a_series_of_several_images_is_a_collection_one_image_collapses():
    coll = PyramidCollection.from_file(_file({"series": 0, "tile": 0}, {"series": 0, "tile": 1},
                                             {"series": 0, "tile": 2}, {"series": 1}))
    assert _shape(coll.tree) == ("sample", None, None, [
        ("series0", "series", 0, ["tile0", "tile1", "tile2"]), "series1"])
    # keys are unchanged by the tree
    assert list(coll) == [(("series", 0), ("tile", 0)), (("series", 0), ("tile", 1)),
                          (("series", 0), ("tile", 2)), (("series", 1),)]
    assert coll.tree["series0"]["tile2"].pyramid is coll[{"series": 0, "tile": 2}]
    assert coll.tree["series1"].pyramid is coll[{"series": 1}]


def test_images_are_named_by_the_indices_no_collection_holds():
    coll = PyramidCollection.from_file(_file(
        {"series": 0, "view": 0, "illumination": 0}, {"series": 0, "view": 1, "illumination": 0},
        {"series": 1, "tile": 4}))
    assert _shape(coll.tree) == ("sample", None, None, [
        ("series0", "series", 0, ["view0_illumination0", "view1_illumination0"]),
        "series1_tile4"])


def test_images_without_a_series_index_sit_under_the_root():
    plain = _file({"series": 0}, path="/data/one.tif")
    plain.images[0].index = {}                 # no index at all
    coll = PyramidCollection.from_file(plain)
    assert list(coll) == [()] and _shape(coll.tree) == ("one", None, None, ["image"])
    coll = PyramidCollection.from_file(_file({"tile": 0}, {"tile": 1}))
    assert _shape(coll.tree) == ("sample", None, None, ["tile0", "tile1"])


PLATE = SimpleNamespace(name="P", rows=("A", "B"), columns=("1", "2"), acquisitions=(),
                        wells=(SimpleNamespace(row=0, column=1, name="A2"),))


def test_a_plate_and_an_image_outside_it():
    coll = PyramidCollection.from_file(_file(
        {"plate": 0, "acquisition": 0, "well": 0, "field": 0},
        {"plate": 0, "acquisition": 0, "well": 0, "field": 1, "tile": 0},
        {"plate": 0, "acquisition": 0, "well": 0, "field": 1, "tile": 1},
        {"series": 0}, plates=[PLATE]))
    assert _shape(coll.tree) == ("sample", None, None, [
        ("P", "plate", 0, [("A2", "well", 0, [
            "acquisition0_field0", ("field1", "field", 1, ["acquisition0_tile0", "acquisition0_tile1"])])]),
        "series0"])                              # the label: under the root, not in the plate
    assert len(coll) == 4 and (("series", 0),) in coll
    assert [len(f.keys) for f in coll.plates[0].wells[0].fields] == [1, 2]


def test_the_old_series_name_is_kept_as_it_is(tmp_path):
    """eubi-bridge 0.2.x passes its own image objects indexed {"scene": i}: the keys, the
    output names and select keep that name, and nothing warns (plan 9.8)."""
    images = [FakeImage(np.full((2, 8, 8), k, "u2"), "cyx", index={"scene": k}) for k in (0, 1)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coll = PyramidCollection.from_file(SimpleNamespace(images=images, path="/d/x.czi"))
        assert list(coll) == [(("scene", 0),), (("scene", 1),)]
        assert coll.suffix((("scene", 1),)) == "_scene1"
        assert len(coll.select(scene=1)) == 1
        assert coll[(("scene", 1),)] is coll.select(scene=1)[{"scene": 1}]
        assert caught == []
        paths = coll.write(tmp_path, "x", overwrite=True)
    # (zarr's own DeprecationWarnings aside) the write says nothing either
    assert [w for w in caught if not issubclass(w.category, DeprecationWarning)] == []
    assert sorted(os.path.basename(p) for p in paths.values()) == [
        "x_scene0.ome.zarr", "x_scene1.ome.zarr"]
    # an unknown name is orthogonal: the images sit under the root, named by it
    assert _shape(coll.tree) == ("x", None, None, ["scene0", "scene1"])


def test_the_output_names_say_series(tmp_path):
    coll = PyramidCollection.from_file(_file({"series": 0}, {"series": 1}))
    assert coll.suffix((("series", 1),)) == "_series1"
    paths = coll.write(tmp_path, "f", overwrite=True)
    assert sorted(os.path.basename(p) for p in paths.values()) == [
        "f_series0.ome.zarr", "f_series1.ome.zarr"]


def test_a_tree_from_file_is_written_and_read_back(tmp_path):
    coll = PyramidCollection.from_file(_file({"series": 0}, {"series": 1}))
    coll.write_collection(str(tmp_path / "c.zarr"))
    back = PyramidCollection.from_zarr(str(tmp_path / "c.zarr"))
    assert list(back) == [(("series", 0),), (("series", 1),)]
    assert [n.name for n in back.tree] == ["series0", "series1"]
    # the RFC 8 attribute keeps its own name
    assert "scene" in rfc8._RFC8_ATTRIBUTES
    # a selection of the tree is refused, as for a collection read from a store
    with pytest.raises(ValueError, match="selection of its tree"):
        coll.select(series=0).write_collection(str(tmp_path / "part.zarr"))


def test_a_concatenated_collection_gets_a_tree_of_its_keys_and_writes(tmp_path):
    # plan 3.25: concatenate carries the tree (built from the joined keys)
    coll = PyramidCollection.from_file(_file({"series": 0, "view": 0}, {"series": 0, "view": 1}))
    joined = coll.concatenate("view")
    assert _shape(joined.tree) == ("sample", None, None, ["series0"])
    assert joined.tree["series0"].pyramid is joined[{"series": 0}]
    assert list(joined.write_series(str(tmp_path / "s.zarr"))) == ["0"]
    assert len(joined.write(tmp_path, "j", overwrite=True)) == 1


def _image_groups(root):
    """The paths (relative to `root`) of every multiscale image group written."""
    out = []
    for folder, _, files in os.walk(root):
        if "zarr.json" in files:
            with open(os.path.join(folder, "zarr.json"), encoding="utf-8") as f:
                attrs = json.load(f).get("attributes", {})
            if attrs.get("ome", {}).get("type") == "multiscale":
                out.append(os.path.relpath(folder, root).replace(os.sep, "/"))
    return sorted(out)


def _no_nesting(groups):
    return [(a, b) for a in groups for b in groups if b.startswith(a + "/")
            and not b.startswith(a + "/labels/")]


@pytest.mark.parametrize("indices", [
    [{"series": 0, "tile": 0}, {"series": 0, "tile": 1}, {"series": 1}],
    [{"series": 0}, {"series": 1, "tile": 0}, {"series": 1, "tile": 1}],
    [{"series": 0, "tile": 0}, {"series": 0, "tile": 1}, {"series": 1}, {"series": 2, "tile": 0},
     {"series": 2, "tile": 1}]])
def test_a_root_of_series_collections_and_images_writes_every_image(tmp_path, indices):
    """A series collection and a one-image series under one root: collections and images
    are numbered on one counter, so no image group lands in another (TEST-r1 #1)."""
    coll = PyramidCollection.from_file(_file(*indices))
    coll.write_collection(str(tmp_path / "c.zarr"))
    groups = _image_groups(str(tmp_path / "c.zarr"))
    assert len(groups) == len(indices) and _no_nesting(groups) == []
    back = PyramidCollection.from_zarr(str(tmp_path / "c.zarr"))
    values = sorted(int(np.asarray(back[k].layers["0"][...]).max()) for k in back)
    assert values == list(range(len(indices)))


def test_a_plate_and_an_image_outside_it_are_written_apart(tmp_path):
    """w96_A1+A2.czi's label next to its plate: the label is not written inside the
    plate's folder, nor the plate inside the label (TEST-r1 #1)."""
    coll = PyramidCollection.from_file(_file(
        {"plate": 0, "acquisition": 0, "well": 0, "field": 0},
        {"plate": 0, "acquisition": 0, "well": 0, "field": 1, "tile": 0},
        {"plate": 0, "acquisition": 0, "well": 0, "field": 1, "tile": 1},
        {"series": 0}, plates=[PLATE]))
    coll.write_collection(str(tmp_path / "c.zarr"))
    groups = _image_groups(str(tmp_path / "c.zarr"))
    assert len(groups) == 4 and _no_nesting(groups) == []
    # a plate with a field of several images has every field as a sub-collection (8.3)
    assert groups == ["0/A/2/0/0", "0/A/2/1/0", "0/A/2/1/1", "1"]


def test_write_rfc8_refuses_an_image_inside_another_before_writing(tmp_path):
    from ome_zarr_pyramid.core.nodes import NodePath
    pyr = PyramidCollection.from_file(_file({"series": 0}))[(("series", 0),)]
    tree = CollectionNode("c", [MultiscaleNode("a", pyr, path=NodePath("zarr", "x")),
                                MultiscaleNode("b", pyr, path=NodePath("zarr", "x/y"))])
    with pytest.raises(ValueError, match="inside the image group 'x'"):
        rfc8.write_rfc8(tree, str(tmp_path / "c.zarr"))
    assert not (tmp_path / "c.zarr").exists()


@pytest.mark.parametrize("path, stem", [
    ("/d/xyz__multiple_images.czi", "xyz__multiple_images"), ("/d/cells.ome.tif", "cells"),
    ("/d/cells.OME.TIFF", "cells"), ("/d/p.ome.zarr/", "p"), ("/d/p.zarr", "p"),
    ("/d/screen.idx.xml", "screen.idx"), ("noext", "noext")])
def test_the_root_is_named_by_the_file_stem(path, stem):
    from ome_zarr_pyramid.core.plates import file_stem
    assert file_stem(path) == stem
    assert PyramidCollection.from_file(_file({"series": 0}, path=path)).tree.name == stem


def test_a_folder_keeps_its_whole_name(tmp_path):
    from ome_zarr_pyramid.core.plates import file_stem
    folder = tmp_path / "plate.v2"
    folder.mkdir()
    assert file_stem(str(folder)) == "plate.v2"


# -- select -------------------------------------------------------------------------

def test_select_raises_on_an_index_the_collection_does_not_have():
    coll = PyramidCollection.from_file(_file({"series": 0, "view": 0}, {"series": 1, "view": 1}))
    with pytest.raises(KeyError, match=r"no index 'nothing'; its indices are \['series', 'view'\]"):
        coll.select(nothing=1)
    with pytest.raises(KeyError, match=r"'scene' is now called 'series' \(micro-reader 0.1.0\)"):
        coll.select(scene=0)
    # a known name whose value matches nothing: an empty collection
    empty = coll.select(view=9)
    assert len(empty) == 0
    # ... which still knows the names it was selected by
    assert len(empty.select(series=0)) == 0
    with pytest.raises(KeyError, match="no index 'tile'"):
        empty.select(tile=0)
    assert len(coll.select(series=0)) == 1


def test_select_names_no_series_hint_without_series():
    coll = PyramidCollection.from_file(_file({"tile": 0}, {"tile": 1}))
    with pytest.raises(KeyError) as err:
        coll.select(scene=0)
    assert "now called" not in str(err.value) and "['tile']" in str(err.value)


def test_a_field_select_only_filters():
    coll = PyramidCollection.from_file(_file(
        {"plate": 0, "acquisition": 0, "well": 0, "field": 0}, plates=[PLATE]))
    field = coll.plates[0].wells[0].fields[0]
    assert len(field.select(view=3).keys) == 0


# -- micro-reader 0.1.0 on real files -------------------------------------------------

def _real(rel, **kwargs):
    mr = pytest.importorskip("micro_reader")
    if not str(getattr(mr, "__version__", "0")).startswith("0.1"):
        pytest.skip("needs micro-reader 0.1 (the series index)")
    path = os.path.join(DATA, rel)
    if not os.path.exists(path):
        pytest.skip(f"sample not found: {path}")
    return mr.open(path, **kwargs)


@pytest.mark.parametrize("rel", ["formats/czi/assorted/xyz__multiple_images.czi",
                                 "formats/lif/ome-samples/20191025 Test FRET 585. 423, 426.lif"])
def test_a_real_multi_series_file(rel):
    file = _real(rel)
    coll = PyramidCollection.from_file(file)
    n = len(file.images)
    assert n > 1 and list(coll) == [(("series", i),) for i in range(n)]
    assert [node.name for node in coll.tree] == [f"series{i}" for i in range(n)]
    assert all(isinstance(node, MultiscaleNode) for node in coll.tree)
    assert coll.suffix((("series", 1),)) == "_series1"
    assert len(coll.select(series=0)) == 1
    with pytest.raises(KeyError, match="series"):
        coll.select(scene=0)


def test_a_real_file_mixing_mosaics_and_a_single_position_writes(tmp_path):
    """S=3_1Pos_2Mosaic: series 0 and 2 are mosaics, series 1 one position (TEST-r1 #1)."""
    file = _real("formats/czi/zenodo-7015307/S=3_1Pos_2Mosaic_T=2=Z=3_CH=2.czi",
                 tiles="separate")
    coll = PyramidCollection.from_file(file)
    assert [type(n).__name__ for n in coll.tree] == ["CollectionNode", "MultiscaleNode",
                                                    "CollectionNode"]
    coll.write_collection(str(tmp_path / "c.zarr"))
    groups = _image_groups(str(tmp_path / "c.zarr"))
    assert len(groups) == len(coll) and _no_nesting(groups) == []
    assert len(PyramidCollection.from_zarr(str(tmp_path / "c.zarr"))) == len(coll)
