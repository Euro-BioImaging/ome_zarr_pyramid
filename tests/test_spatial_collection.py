"""Where a collection's images are (plan 3.3, 3.4, 3.22, 3.25): PyramidCollection's
coordinate_systems, system_of and coordinate_transformations, from a file and through
select / filter / concatenate. Files are hand-made objects with micro-reader's field names,
so no micro-reader is needed."""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from ome_zarr_pyramid import PyramidCollection
from ome_zarr_pyramid.core.spatial import Axis, CoordinateSystem, InvariantError, Transformation

from test_collection import FakeImage, Q


def _cs(id, estimated=False, axes=None):
    return SimpleNamespace(id=id, name=None, axes=axes, estimated=estimated, extra={})


def _img(index, system=None, position=None, value=0, axes="cyx"):
    meta = SimpleNamespace(
        pixel_sizes={"y": Q(0.5, "micrometer"), "x": Q(0.5, "micrometer")},
        position=position or {}, channels=[], index_values={}, coordinate_system=system)
    return FakeImage(np.full((2, 8, 8), value, "u2"), axes, meta, index=index, name=f"img{value}")


def _placed(index, system="stage", y=10.0, x=-3.0, **kw):
    return _img(index, system, {"y": Q(y, "micrometer"), "x": Q(x, "micrometer")}, **kw)


def _file(images, systems=(), edges=(), path="/data/sample.czi"):
    return SimpleNamespace(path=path, images=images, coordinate_systems=tuple(systems),
                           coordinate_transformations=tuple(edges))


def _tiles(n=3, system="stage"):
    return [_placed({"series": 0, "tile": t}, system, y=10.0 * t, value=t) for t in range(n)]


# -- from a file --------------------------------------------------------------------

def test_from_file_takes_the_files_systems_and_each_images_system():
    images = _tiles(2) + [_img({"series": 1}, value=9)]           # series 1 unplaced
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")]))
    assert list(coll.coordinate_systems) == ["stage"]
    assert coll.coordinate_systems["stage"] == CoordinateSystem("stage")
    assert coll.system_of == {(("series", 0), ("tile", 0)): "stage",
                              (("series", 0), ("tile", 1)): "stage"}
    assert coll.coordinate_transformations == []
    # the position is the level-0 translation, in the image's own units
    pyr = coll[{"series": 0, "tile": 1}]
    assert pyr.meta.get_translation("0")[-2:] == [10.0, -3.0]


def test_a_file_that_does_not_describe_its_systems_gives_none_and_keeps_translations():
    # eubi-bridge's own objects, micro-reader before 0.2: no coordinate_systems
    images = [_placed({"series": 0}, system=None)]
    images[0].metadata.coordinate_system = None
    coll = PyramidCollection.from_file(SimpleNamespace(path="/x.czi", images=images))
    assert coll.coordinate_systems == {} and coll.system_of == {}
    assert coll[{"series": 0}].meta.get_translation("0")[-2:] == [10.0, -3.0]


def test_an_undeclared_or_estimated_system_breaks_i2():
    with pytest.raises(InvariantError, match="I2.*not one of the systems"):
        PyramidCollection.from_file(_file(_tiles(1, "slide"), [_cs("stage")]))
    with pytest.raises(InvariantError, match="I2.*estimated"):
        PyramidCollection.from_file(_file(_tiles(1, "global-registered"),
                                          [_cs("global-registered", estimated=True)]))


def test_a_system_whose_axes_lack_an_images_axis_breaks_i8():
    yx = (Axis("y", unit="micrometer"), Axis("x", unit="micrometer"))
    with pytest.raises(InvariantError, match="I8.*'c'"):
        PyramidCollection.from_file(_file(_tiles(1), [_cs("stage", axes=yx)]))


def test_edges_are_kept_and_their_ends_checked():
    key = (("series", 0), ("tile", 0))
    reg = Transformation("translation", key, "global-registered", {"translation": [0, 0, 1]})
    coll = PyramidCollection.from_file(_file(
        _tiles(2), [_cs("stage"), _cs("global-registered", estimated=True)], [reg]))
    assert coll.coordinate_transformations == [reg]
    stray = Transformation("translation", (("series", 7),), "global-registered",
                           {"translation": [0, 0, 1]})
    with pytest.raises(InvariantError, match="I5.*not an entry"):
        PyramidCollection.from_file(_file(_tiles(2), [_cs("stage"),
                                                      _cs("global-registered", True)], [stray]))
    with pytest.raises(InvariantError, match="I5.*not one of the systems"):
        PyramidCollection.from_file(_file(_tiles(2), [_cs("stage")], [reg]))


def test_two_systems_with_one_id_break_i1():
    with pytest.raises(InvariantError, match="I1"):
        PyramidCollection.from_file(_file(_tiles(1), [_cs("stage"), _cs("stage")]))


# -- derived collections (3.25) -------------------------------------------------------

def _wells():
    images = [_placed({"series": 0, "tile": 0}, "well-B2", value=0),
              _placed({"series": 0, "tile": 1}, "well-B2", value=1),
              _placed({"series": 1, "tile": 0}, "well-B3", value=2)]
    key = (("series", 0), ("tile", 1))
    reg = Transformation("translation", key, "global-registered", {"translation": [0, 0, 1]})
    return PyramidCollection.from_file(_file(
        images, [_cs("well-B2"), _cs("well-B3"), _cs("global-registered", True)], [reg]))


def test_select_keeps_the_graph_of_the_kept_entries():
    coll = _wells()
    b3 = coll.select(series=1)
    assert list(b3.coordinate_systems) == ["well-B3"]
    assert b3.system_of == {(("series", 1), ("tile", 0)): "well-B3"}
    assert b3.coordinate_transformations == []
    tile1 = coll.select(series=0, tile=1)
    assert list(tile1.coordinate_systems) == ["well-B2", "global-registered"]
    assert len(tile1.coordinate_transformations) == 1
    tile1.check_invariants()


def test_filter_and_sort_by_derive_the_same_way():
    coll = _wells()
    assert coll.filter(series=0).system_of == {k: "well-B2" for k in coll if dict(k)["series"] == 0}
    assert coll.sort_by("series", reverse=True).system_of == coll.system_of


# -- concatenate (3.25) ---------------------------------------------------------------

def test_a_joined_image_is_placed_in_its_parts_system():
    images = [_placed({"series": 0, "view": v}, "stage", value=v) for v in range(2)]
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")]))
    joined = coll.concatenate("view")
    assert joined.system_of == {(("series", 0),): "stage"}
    assert list(joined.coordinate_systems) == ["stage"]
    assert joined.tree is not None
    joined.check_invariants()


def test_parts_in_different_systems_are_not_joined():
    images = [_placed({"series": 0, "view": 0}, "stage", value=0),
              _placed({"series": 0, "view": 1}, "slide", value=1)]
    coll = PyramidCollection.from_file(_file(images, [_cs("stage"), _cs("slide")]))
    with pytest.raises(ValueError, match="different coordinate systems"):
        coll.concatenate("view")
    images[1] = _img({"series": 0, "view": 1}, value=1)                 # unplaced
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")]))
    with pytest.raises(ValueError, match="some are unplaced"):
        coll.concatenate("view")


def test_edges_at_joined_parts_are_dropped_with_a_warning():
    images = [_placed({"series": 0, "view": v}, "stage", value=v) for v in range(2)]
    key = (("series", 0), ("view", 1))
    reg = Transformation("translation", key, "global-registered", {"translation": [0, 0, 1]})
    coll = PyramidCollection.from_file(_file(
        images, [_cs("stage"), _cs("global-registered", True)], [reg]))
    with pytest.warns(UserWarning, match="1 edge"):
        joined = coll.concatenate("view")
    assert joined.coordinate_transformations == []
    assert list(joined.coordinate_systems) == ["stage"]
