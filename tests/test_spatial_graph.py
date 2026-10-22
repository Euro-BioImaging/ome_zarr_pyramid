"""The graph of section 6's shapes (plan 3.6, 6.3, 6.6, 9.9's done-when): components and
owners, affine views, per-well systems through group_by, the raw form of types ozp does
not model, and a lone pyramid's system through a write."""
import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from ome_zarr_pyramid import IO, PyramidCollection
from ome_zarr_pyramid.core.spatial import (
    Transformation, matrix, pyramid_system, transformation_from_json, transformation_to_json)

from test_spatial_collection import _cs, _file, _img, _placed

UM = "micrometer"


# -- 6.6: a plate, one system per well ------------------------------------------------

PLATE = SimpleNamespace(name="P", rows=("A", "B"), columns=("1", "2", "3"), acquisitions=(),
                        wells=(SimpleNamespace(row=0, column=0, name="A1"),
                               SimpleNamespace(row=1, column=2, name="B3")))


def _plate(registration=None):
    images, edges = [], []
    for w, well in enumerate(("A1", "B3")):
        for f in range(2):
            key = {"plate": 0, "acquisition": 0, "well": w, "field": f}
            images.append(_placed(key, f"well-{well}", y=100.0 * f, value=2 * w + f))
            if registration == "per-well":
                edges.append(Transformation("translation", tuple(key.items()),
                                            f"well-{well}-registered", {"translation": [0, 0, 1]}))
            elif registration == "plate-wide":
                edges.append(Transformation("translation", tuple(key.items()),
                                            "plate-registered", {"translation": [0, 0, 1]}))
    systems = [_cs("well-A1"), _cs("well-B3")]
    if registration == "per-well":
        systems += [_cs("well-A1-registered", True), _cs("well-B3-registered", True)]
    elif registration == "plate-wide":
        systems.append(_cs("plate-registered", True))
    file = _file(images, systems, edges, path="/data/screen.idx.xml")
    file.plates = [PLATE]
    return PyramidCollection.from_file(file)


def test_each_well_is_a_component_owned_by_its_well():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        components = _plate().components()
    assert [(sorted(c.systems), len(c.entries), c.owner) for c in components] == [
        (["well-A1"], 2, ("P", "A1")), (["well-B3"], 2, ("P", "B3"))]


def test_group_by_well_gives_each_well_its_system_and_its_edges():
    coll = _plate("per-well")
    groups = coll.group_by("well")
    assert list(groups) == [0, 1]
    for w, well in enumerate(("A1", "B3")):
        g = groups[w]
        assert list(g.coordinate_systems) == [f"well-{well}", f"well-{well}-registered"]
        assert set(g.system_of.values()) == {f"well-{well}"}
        assert len(g.coordinate_transformations) == 2
        assert {e.output for e in g.coordinate_transformations} == {f"well-{well}-registered"}
        g.check_invariants()
    # unique and count read the table, not the graph; the derived collections keep it
    assert coll.unique("well") == [0, 1]
    assert coll.sort_by("well", reverse=True).system_of == coll.system_of


def test_a_plate_wide_registration_makes_one_plate_component_and_warns():
    coll = _plate("plate-wide")
    with pytest.warns(UserWarning, match=r"join systems that \['P/A1', 'P/B3'\] would own; "
                                         r"they are written together under P"):
        (component,) = coll.components()
    assert component.systems == {"well-A1", "well-B3", "plate-registered"}
    assert component.owner == ("P",) and len(component.entries) == 4
    assert len(component.edges) == 4


def test_a_lone_entry_has_no_owner_and_an_unplaced_one_no_component():
    images = [_placed({"series": 0}, "stage"), _img({"series": 1}, value=1)]
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")]))
    (c,) = coll.components()
    assert c.entries == [(("series", 0),)] and c.owner is None


# -- 6.3: BigDataViewer views with affine edges ---------------------------------------

def test_views_with_affine_edges_are_one_component_under_their_series():
    turned = [[1, 0, 0, 0], [0, 0, -1, 5], [0, 1, 0, 0]]        # c, y, x: y' = 5 - x, x' = y
    images = [_img({"series": 0, "view": v}, value=v, axes="cyx") for v in range(2)]
    edges = [Transformation("affine", (("series", 0), ("view", v)), "global",
                            {"affine": turned if v else [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]})
             for v in range(2)]
    coll = PyramidCollection.from_file(_file(images, [_cs("global")], edges))
    assert coll.system_of == {}                         # no translation-only position
    (c,) = coll.components()
    assert c.systems == {"global"} and c.owner == ("series0",) and len(c.edges) == 2
    m = matrix(c.edges[1], 3)
    assert np.allclose(m @ [0, 1, 0, 1], [0, 5, 1, 1])  # a quarter turn in y / x
    # a view taken out keeps its edge only while its key is in the set
    assert coll.select(view=1).coordinate_transformations == [edges[1]]


# -- what ozp does not model, kept as read --------------------------------------------

@pytest.mark.parametrize("raw", [
    {"type": "rotation", "rotation": [[0, -1], [1, 0]], "input": "a", "output": "b"},
    {"type": "bijection", "input": "a", "output": "b", "forward": {"type": "scale", "scale": [2, 2]},
     "inverse": {"type": "scale", "scale": [0.5, 0.5]}},
    {"type": "coordinates", "path": "coordinateTransformations/c", "input": "a", "output": "b",
     "name": "warp"},
])
def test_an_unmodelled_type_round_trips_byte_equal(raw):
    t = transformation_from_json(raw)
    assert not t.modelled
    assert json.dumps(transformation_to_json(t), sort_keys=True) == json.dumps(raw, sort_keys=True)


# -- a lone pyramid through a write (3.26) ----------------------------------------------

def test_an_entry_written_alone_reads_back_with_its_position_system_and_axes(tmp_path):
    file = _file([_placed({"series": 0}, "stage", y=7.0, x=2.0)],
                 [SimpleNamespace(id="stage", name="the stage", axes=None, estimated=False, extra={})])
    coll = PyramidCollection.from_file(file)
    path = str(tmp_path / "one.ome.zarr")
    IO().write_pyramid(coll.pyramid({"series": 0}), path, overwrite=True)
    back = IO().read_pyramid(path)
    assert back.meta.get_translation("0")[-2:] == [7.0, 2.0]
    assert pyramid_system(back) == ("stage", coll.coordinate_systems["stage"])
    again = PyramidCollection({(("series", 0),): back})
    assert again.coordinate_systems == coll.coordinate_systems


def test_a_file_with_only_images_reads_back_without_an_i2_error(tmp_path):
    images = [_placed({"series": s}, "stage", value=s) for s in range(2)]
    coll = PyramidCollection.from_file(SimpleNamespace(path="/x.czi", images=images))
    for s in range(2):
        IO().write_pyramid(coll.pyramid({"series": s}), str(tmp_path / f"{s}.ome.zarr"),
                           overwrite=True)
    back = PyramidCollection({(("series", s),): IO().read_pyramid(str(tmp_path / f"{s}.ome.zarr"))
                              for s in range(2)}, join_systems=True)
    assert back.coordinate_systems == coll.coordinate_systems
    assert set(back.system_of.values()) == {"stage"}


def test_a_prefix_in_one_collection_leaves_another_collections_view_alone():
    left = PyramidCollection.from_file(_file([_placed({"series": 0}, "stage")], [_cs("stage")],
                                             path="/data/left.czi"))
    right = PyramidCollection.from_file(_file([_placed({"series": 0}, "stage")], [_cs("stage")],
                                              path="/data/right.czi"))
    mine = left.pyramid({"series": 0})
    alone = PyramidCollection({(("a", 0),): mine})
    merged = PyramidCollection({(("b", 0),): right.pyramid({"series": 0}), (("a", 0),): mine})
    assert merged.system_of[(("a", 0),)] == "left--stage"
    assert alone.system_of == {(("a", 0),): "stage"}
    assert pyramid_system(mine)[0] == "stage"             # the pyramid is not changed
    assert left.system_of == {(("series", 0),): "stage"}
