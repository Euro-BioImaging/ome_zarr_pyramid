"""The graph and eubi: keys in RFC 8 output (plan 9.10: 3.11, 3.28, 5.2, 5.4, 5.5-5.10,
5.18, 8.3, 8.6, 8.19, 8.21): scenes on the collection owning each component, positions
read back from their edges, fields as sub-collections, labels following their image,
aliases, the experimental warning."""
import json
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.core.rfc8 import RFC8Warning
from ome_zarr_pyramid.core.spatial import Transformation, matrix

from test_collection import FakeImage, Q
from test_eubi_keys import _plate_file, _positions, _same_graph
from test_spatial_collection import _cs, _file, _img, _placed

UM = "micrometer"


def _doc(path):
    return zarr.open_group(path, mode="r").attrs.asdict()["ome"]


def _write(coll, path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RFC8Warning)
        return coll.write_collection(path, **kw)


def _node(doc, *names):
    node = doc
    for name in names:
        node = next(n for n in node["nodes"] if n["name"] == name)
    return node


def test_writing_rfc8_warns_that_it_is_experimental(tmp_path):
    coll = PyramidCollection.from_file(_file([_placed({"series": 0}, "stage")], [_cs("stage")]))
    with pytest.warns(RFC8Warning, match="experimental"):
        coll.write_collection(str(tmp_path / "c.zarr"))


# -- scenes (5.5, 5.18) -----------------------------------------------------------------

def test_each_well_is_the_scene_of_its_system(tmp_path):                         # 6.6
    coll = _plate_file(per_well=True, fields=(0, 2))
    out = str(tmp_path / "p.zarr")
    _write(coll, out)
    doc = _doc(out)                                # the file's collection, holding plate P
    assert doc["attributes"]["eubi:version"] == 1
    assert _node(doc, "P")["attributes"]["eubi:kind"] == "plate"
    well = _node(doc, "P", "B2")
    scene = well["attributes"]["scene"]
    assert [s["id"] for s in scene["coordinateSystems"]] == ["well-B2"]
    assert [a["name"] for a in scene["coordinateSystems"][0]["axes"]] == ["c", "y", "x"]
    edge = scene["coordinateTransformations"][1]
    assert edge["type"] == "translation" and edge["translation"][-2:] == [200.0, 7.0]
    assert edge["input"] == {"id": "physical", "path": {"type": "zarr", "path": "./0/B/2/1"}}
    assert edge["output"] == {"id": "well-B2"}
    image = _node(doc, "P", "B2", "acquisition0_field2")
    assert image["attributes"]["eubi:index"]["field"] == 2
    assert image["attributes"]["eubi:coordinate_system"] == "well-B2"
    assert image["attributes"]["eubi:scene"] == ".."               # its well, one up
    # the image's own levels hold only their alignment: the position left them
    level = zarr.open_group(os.path.join(out, "0", "B", "2", "1"), mode="r").attrs["ome"]["nodes"][0]
    (t,) = level["attributes"]["coordinateTransformations"]
    assert t["type"] in ("scale", "sequence")                      # one transform (5.4)
    if t["type"] == "sequence":
        assert t["transformations"][1]["translation"][-2:] == [0.0, 0.0]
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)
    assert [f.index for f in back.plates[0].wells[0].fields] == [0, 2]   # gapped (8.4)


def test_a_tiling_set_is_a_scene_on_its_series(tmp_path):                         # 6.4
    images = [_placed({"series": 0, "tile": t}, "stage", y=10.0 * t, x=-2.0 * t, value=t)
              for t in range(4)]
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")], path="/data/matl.omp2info"))
    out = str(tmp_path / "s.zarr")
    _write(coll, out)
    scene = _node(_doc(out), "series0")["attributes"]["scene"]
    assert len(scene["coordinateTransformations"]) == 4
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)


def test_a_unit_difference_gives_a_scale_and_translation(tmp_path):
    a = _placed({"series": 0, "tile": 0}, "stage", y=1.0, x=2.0, value=1)
    b = FakeImage(np.zeros((2, 8, 8), "u2"), "cyx", SimpleNamespace(
        pixel_sizes={"y": Q(500.0, "nanometer"), "x": Q(500.0, "nanometer")},
        position={"y": Q(3000.0, "nanometer"), "x": Q(4000.0, "nanometer")}, channels=[],
        index_values={}, coordinate_system="stage"), index={"series": 0, "tile": 1})
    coll = PyramidCollection.from_file(_file([a, b], [_cs("stage")]))
    out = str(tmp_path / "s.zarr")
    _write(coll, out)
    edges = _node(_doc(out), "series0")["attributes"]["scene"]["coordinateTransformations"]
    assert edges[0]["type"] == "translation"
    assert edges[1]["type"] == "sequence"                          # nanometer into micrometer
    assert edges[1]["transformations"][0]["scale"][-2:] == [0.001, 0.001]
    assert edges[1]["transformations"][1]["translation"][-2:] == [3.0, 4.0]
    back = PyramidCollection.from_zarr(out)
    assert _positions(back) == _positions(coll)                    # in each image's own units


def test_images_whose_axes_differ_from_their_system_stay_single_images(tmp_path):  # 6.7
    zyx = FakeImage(np.zeros((3, 8, 8), "u2"), "zyx", SimpleNamespace(
        pixel_sizes={"z": Q(2.0, UM), "y": Q(0.5, UM), "x": Q(0.5, UM)},
        position={"z": Q(1.0, UM), "y": Q(4.0, UM), "x": Q(5.0, UM)}, channels=[],
        index_values={}, coordinate_system="stage"), index={"series": 0, "tile": 0})
    yx = FakeImage(np.zeros((8, 8), "u2"), "yx", SimpleNamespace(
        pixel_sizes={"y": Q(0.5, UM), "x": Q(0.5, UM)}, position={"y": Q(6.0, UM), "x": Q(7.0, UM)},
        channels=[], index_values={}, coordinate_system="stage"), index={"series": 0, "tile": 1})
    coll = PyramidCollection.from_file(_file([zyx, yx], [_cs("stage")]))
    out = str(tmp_path / "s.zarr")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coll.write_collection(out)
    assert any("is not written as a scene" in str(w.message) and "5.18(c)" in str(w.message)
               for w in caught)
    series = _node(_doc(out), "series0")
    assert "scene" not in series["attributes"]
    assert [s["name"] for s in series["attributes"]["eubi:coordinate_systems"]] == ["stage"]
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)


def test_affine_views_go_into_the_scene_unchanged(tmp_path):                      # 6.3
    turned = [[1, 0, 0, 0], [0, 0, -1, 5], [0, 1, 0, 0]]
    images = [_img({"series": 0, "view": v}, value=v) for v in range(2)]
    edges = [Transformation("affine", (("series", 0), ("view", v)), "global",
                            {"affine": turned if v else [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]})
             for v in range(2)]
    coll = PyramidCollection.from_file(_file(images, [_cs("global")], edges))
    out = str(tmp_path / "s.zarr")
    _write(coll, out)
    scene = _node(_doc(out), "series0")["attributes"]["scene"]
    assert [t["type"] for t in scene["coordinateTransformations"]] == ["affine", "affine"]
    back = PyramidCollection.from_zarr(out)
    for mine, theirs in zip(coll.coordinate_transformations, back.coordinate_transformations):
        assert (theirs.input, theirs.output) == (mine.input, mine.output)
        assert np.allclose(matrix(theirs, 3), matrix(mine, 3))


def test_a_registration_from_a_placed_image_is_composed_with_its_position(tmp_path):
    images = [_placed({"series": 0, "tile": t}, "stage", y=10.0 * t, x=1.0, value=t)
              for t in range(2)]
    reg = Transformation("translation", (("series", 0), ("tile", 1)), "stage-registered",
                         {"translation": [0.0, 0.5, -0.25]})
    coll = PyramidCollection.from_file(_file(
        images, [_cs("stage"), _cs("stage-registered", estimated=True)], [reg]))
    out = str(tmp_path / "s.zarr")
    _write(coll, out)
    scene = _node(_doc(out), "series0")["attributes"]["scene"]
    assert [s["id"] for s in scene["coordinateSystems"]] == ["stage", "stage-registered"]
    written = scene["coordinateTransformations"][-1]
    # the image's levels no longer hold its position (10, 1): the edge now adds it
    assert written["translation"][-2:] == [10.5, 0.75]
    back = PyramidCollection.from_zarr(out)
    assert back.coordinate_transformations == [reg]
    assert back.coordinate_systems["stage-registered"].estimated is True


# -- fields, names, acquisitions (8.3, 8.6, 5.15) ------------------------------------------

PLATE = SimpleNamespace(name="", rows=("A",), columns=("1", "2"), acquisitions=(),
                        wells=(SimpleNamespace(row=0, column=0, name="A1"),
                               SimpleNamespace(row=0, column=1, name="A2")))


def _tiled_plate():
    """6.1 tiled: well A1 field 0 of two tiles and a one-image field 2; well A2 one image."""
    keys = [{"field": 0, "tile": 0}, {"field": 0, "tile": 1}, {"field": 2}]
    images = [_placed({"plate": 0, "acquisition": 0, "well": 0, **k}, "stage", y=10.0 * i,
                      value=i) for i, k in enumerate(keys)]
    images.append(_placed({"plate": 0, "acquisition": 0, "well": 1, "field": 0}, "stage",
                          y=99.0, value=9))
    file = _file(images, [_cs("stage")], path="/data/w96.czi")
    file.plates = [PLATE]
    return PyramidCollection.from_file(file)


def test_a_tiled_plate_keeps_its_tiles_fields_and_empty_name(tmp_path):            # 6.1
    coll = _tiled_plate()
    out = str(tmp_path / "p.zarr")
    _write(coll, out)
    doc = _doc(out)
    plate = _node(doc, "plate0")
    assert plate["attributes"]["eubi:name"] == ""                             # 8.6
    a1 = _node(doc, "plate0", "A1")
    assert [n["name"] for n in a1["nodes"]] == ["field0", "field2"]              # 8.3
    assert all(n["attributes"]["eubi:kind"] == "field" for n in a1["nodes"])
    a2 = _node(doc, "plate0", "A2")
    assert [n["name"] for n in a2["nodes"]] == ["field0"]
    back = PyramidCollection.from_zarr(out)
    assert list(back) == list(coll)                                # tiles, fields 0 / 2
    assert back.plates[0].name == ""                               # back as ""
    assert back.system_of == coll.system_of and _positions(back) == _positions(coll)


def test_labels_in_a_field_follow_their_image(tmp_path):                          # 3.28
    coll = _tiled_plate()
    key = (("plate", 0), ("acquisition", 0), ("well", 0), ("field", 0), ("tile", 1))
    tile = coll[key]
    labels = Pyramid().from_label_arrays([np.ones((8, 8), "u1")], axis_order="yx",
                                         scales=[[0.5, 0.5]], version="0.5", name="nuclei")
    labels.meta.update_translation("0", [10.0, -3.0])              # its image's position
    node = next(n for n in coll.tree.walk() if getattr(n, "_pyramid", None) is tile)
    node.pyramid = coll._pyramids[key] = tile.add_image_label(labels, "nuclei")
    out = str(tmp_path / "p.zarr")
    _write(coll, out)
    field = _node(_doc(out), "plate0", "A1", "field0")
    names = [n["name"] for n in field["nodes"]]
    assert names == ["acquisition0_tile0", "acquisition0_tile1", "acquisition0_tile1 nuclei"]
    # one stage across both wells: the plate owns the component, so its scene holds them
    edges = _node(_doc(out), "plate0")["attributes"]["scene"]["coordinateTransformations"]
    label_edges = [e for e in edges if e["input"]["path"]["path"].endswith("labels/nuclei")]
    assert len(label_edges) == 1 and label_edges[0]["translation"][-2:] == [10.0, -3.0]
    back = PyramidCollection.from_zarr(out)
    assert list(back) == list(coll)                                # the label is no tile
    label_node = _back_label(back)
    assert label_node.pyramid.meta.get_translation("0") == [10.0, -3.0]   # folded back


def _back_label(coll):
    for node in coll.tree.walk():
        if "labels" in node.attributes:
            return node
    raise AssertionError("no label node")


@pytest.mark.parametrize("form", ["tall", "wide"])
def test_the_tall_and_wide_acquisition_forms_round_trip(tmp_path, form):            # 5.15
    """The RFC's two HCS examples (test_nodes.TALL / WIDE), with real images: written
    back, each keeps its form (acquisition sub-collections, or an attribute per image)."""
    import copy

    from ome_zarr_pyramid import IO
    from test_nodes import TALL, WIDE
    doc = copy.deepcopy(TALL if form == "tall" else WIDE)
    for node in doc["nodes"][0]["nodes"]:
        for image in ([node] if node["type"] == "multiscale" else node["nodes"]):
            target = str(tmp_path / "src" / image["path"]["path"][2:])
            IO().write_pyramid(Pyramid().from_arrays([np.zeros((4, 4), "u1")], axis_order="yx",
                                                     version="0.5", name=image["name"]), target)
    src = str(tmp_path / "src" / "plate.json")
    with open(src, "w") as f:
        json.dump({"ome": {"version": "0.x", **doc}}, f)
    coll = PyramidCollection.from_zarr(src)
    out = str(tmp_path / "out.zarr")
    _write(coll, out)
    back = PyramidCollection.from_zarr(out)
    assert list(back) == list(coll) and back.plates == coll.plates
    well = _node(_doc(out), "well A01")
    if form == "tall":
        assert [n["type"] for n in well["nodes"]] == ["collection", "collection"]
        assert [n["attributes"]["acquisition"] for n in well["nodes"]] == [
            {"id": "acq_0"}, {"id": "acq_1"}]
    else:
        assert all(n["type"] == "multiscale" and "acquisition" in n["attributes"]
                   for n in well["nodes"])


# -- one image, two parents (3.11, 8.19) -------------------------------------------------

def test_an_image_with_two_parents_reads_once_and_is_written_back_as_an_alias(tmp_path):
    from ome_zarr_pyramid import IO
    img = str(tmp_path / "img.zarr")
    IO().write_pyramid(Pyramid().from_arrays([np.zeros((4, 4), "u1")], axis_order="yx",
                                             version="0.5", name="img"), img)
    src = str(tmp_path / "doc.json")
    node = {"type": "multiscale", "name": "img", "path": {"type": "zarr", "path": "./img.zarr"}}
    with open(src, "w") as f:
        json.dump({"ome": {"version": "0.x", "type": "collection", "name": "root", "nodes": [
            {"type": "collection", "name": "a", "nodes": [node]},
            {"type": "collection", "name": "b", "nodes": [dict(node, name="same image")]}]}}, f)
    coll = PyramidCollection.from_zarr(src)
    assert len(coll) == 1                                          # one entry, one key
    b = coll.tree["b"]["same image"]
    assert b.pyramid is coll[list(coll)[0]]                        # sharing its pyramid
    out = str(tmp_path / "out.zarr")
    _write(coll, out)
    doc = _doc(out)
    paths = [_node(doc, "a", "img")["path"]["path"], _node(doc, "b", "same image")["path"]["path"]]
    assert paths[0] == paths[1]                                    # written as one path
    assert len(PyramidCollection.from_zarr(out)) == 1


# -- a standalone document rebases the scene's paths too -----------------------------------

def test_a_standalone_document_keeps_the_scene(tmp_path):
    coll = _plate_file(per_well=True)
    out = str(tmp_path / "p.zarr")
    document = str(tmp_path / "docs" / "plate.json")
    os.makedirs(os.path.dirname(document))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RFC8Warning)
        coll.write_plate(out, version="rfc8", document=document)
    back = PyramidCollection.from_zarr(document)
    _same_graph(back, coll)
