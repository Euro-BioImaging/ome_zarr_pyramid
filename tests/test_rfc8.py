"""NGFF RFC 8 JSON: the RFC's example shapes read (collections inline, behind JSON files
and zarr groups; multiscales consolidated, distributed and as single levels; references
spelled {"id"}, {"ref"} and {"$ref"}), and what ozp writes with version="rfc8" reads back
as the same tree, with the same pixels and image metadata."""
import glob
import json
import os
import warnings

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.core.nodes import CollectionNode, LabelsAttribute, MultiscaleNode, Ref
from ome_zarr_pyramid.core.rfc8 import RFC8_VERSION, read_rfc8

from test_collection_zarr import _opened, _plate
from conftest import written_images

AXES = [{"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}]
DATA = (np.arange(16 * 16 * 16) % 200).astype("u1").reshape(16, 16, 16)


def _write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f)


def _levels(group_path, distributed=False):
    """Levels 0..2 of DATA as zarr v3 arrays in `group_path`; the singlescale nodes for
    them (inline transforms) or, `distributed`, their own zarr.json holding the node."""
    g = zarr.open_group(group_path, mode="a", zarr_format=3)
    nodes = []
    for i in range(3):
        s = 2 ** i
        arr = g.create_array(str(i), data=DATA[::s, ::s, ::s], chunks=(8, 8, 8))
        transforms = [{"type": "scale", "scale": [float(s)] * 3, "input": {"ref": f"s{i}"},
                       "output": {"ref": "world"}}]
        if distributed:
            arr.attrs["ome"] = {"version": "0.7dev0", "type": "singlescale", "id": f"s{i}",
                                "name": f"scale {i}", "attributes": {
                                    "coordinateSystems": [{"id": "world", "axes": AXES}],
                                    "coordinateTransformations": transforms}}
            nodes.append({"id": f"s{i}", "type": "singlescale", "name": f"scale {i}",
                          "path": {"type": "zarr", "path": f"./{i}"}})
        else:
            nodes.append({"id": f"s{i}", "type": "singlescale", "name": f"scale {i}",
                          "path": {"type": "zarr", "path": f"./{i}"},
                          "attributes": {"coordinateTransformations": transforms}})
    return g, nodes


def _multiscale(path, distributed=False):
    g, nodes = _levels(path, distributed)
    ome = {"version": "0.7dev0", "id": "img", "type": "multiscale", "name": "test multiscale",
           "nodes": nodes}
    if not distributed:
        ome["attributes"] = {"coordinateSystems": [{"id": "world", "name": "world", "axes": AXES}]}
    g.attrs["ome"] = ome
    return path


def _assert_data(pyr):
    assert pyr.axes == "zyx" and pyr.meta.resolution_paths == ["0", "1", "2"]
    assert [pyr.meta.get_scale(p) for p in pyr.meta.resolution_paths] == [[1.0] * 3, [2.0] * 3, [4.0] * 3]
    assert pyr.meta.unit_list == ["micrometer"] * 3
    for i, p in enumerate(pyr.meta.resolution_paths):
        s = 2 ** i
        np.testing.assert_array_equal(np.asarray(pyr.layers[p]), DATA[::s, ::s, ::s])


@pytest.mark.parametrize("distributed", [False, True], ids=["consolidated", "distributed"])
def test_rfc8_multiscales_read(tmp_path, distributed):
    path = _multiscale(str(tmp_path / "ms.zarr"), distributed)
    coll = PyramidCollection.from_zarr(path)
    assert list(coll) == [()] and coll.tree.type == "multiscale"
    _assert_data(coll[()])


def test_a_single_level_on_its_own_reads(tmp_path):
    path = str(tmp_path / "single.zarr")
    arr = zarr.create_array(path, data=DATA, chunks=(8, 8, 8), zarr_format=3)
    arr.attrs["ome"] = {"version": "0.7dev0", "type": "singlescale", "id": "s0", "name": "s0",
                        "attributes": {"coordinateSystems": [{"id": "world", "axes": AXES}],
                                       "coordinateTransformations": [{
                                           "type": "scale", "scale": [0.5, 0.5, 0.5],
                                           "input": {"id": "s0"}, "output": {"id": "world"}}]}}
    pyr = PyramidCollection.from_zarr(path)[()]
    assert pyr.meta.get_scale(pyr.meta.resolution_paths[0]) == [0.5] * 3
    np.testing.assert_array_equal(np.asarray(pyr.layers[pyr.meta.resolution_paths[0]]), DATA)


def test_collections_inline_behind_json_and_zarr_read_lazily(tmp_path):
    _multiscale(str(tmp_path / "images" / "a.zarr"))
    _multiscale(str(tmp_path / "images" / "b.zarr"), distributed=True)
    # a child collection in its own JSON file, naming an image relative to itself
    _write_json(str(tmp_path / "sub" / "child.json"), {"ome": {
        "version": "0.8", "id": "child", "type": "collection", "name": "sub-collection",
        "attributes": {"ex:a": 0, "ex:c": 3},
        "nodes": [{"type": "multiscale", "name": "b", "id": "b",
                   "path": {"type": "zarr", "path": "../images/b.zarr"},
                   "attributes": {"ngio:source": {"ref": {"path": {"type": "json",
                                                                   "path": "../parent.json"},
                                                          "id": "a"}}}}]}})
    parent = str(tmp_path / "parent.json")
    _write_json(parent, {"ome": {
        "version": "0.8", "id": "parent", "type": "collection", "name": "base-collection",
        "nodes": [{"type": "multiscale", "name": "a", "id": "a",
                   "path": {"type": "zarr", "path": "./images/a.zarr"}, "attributes": {}},
                  {"type": "collection", "name": "sub-collection", "id": "child-ref",
                   "path": {"type": "json", "path": "./sub/child.json"},
                   "attributes": {"ex:a": 1, "ex:b": 2}}]}})
    coll = PyramidCollection.from_zarr(parent)
    assert [dict(k)["series"] for k in coll] == [0, 1] and _opened(coll) == []
    sub = coll.tree["sub-collection"]
    assert sub.attributes == {"ex:a": 1, "ex:b": 2, "ex:c": 3}      # the parent's win
    assert sub["b"].attributes["ngio:source"]["ref"]["id"] == "a"   # kept as written
    _assert_data(coll[{"series": 0}])
    _assert_data(coll[{"series": 1}])
    # the same collection in a zarr group's zarr.json
    g = zarr.open_group(str(tmp_path / "group.zarr"), mode="w", zarr_format=3)
    g.attrs["ome"] = {"version": "0.x", "type": "collection", "name": "in a group", "nodes": [
        {"type": "multiscale", "name": "a", "path": {"type": "zarr", "path": "../images/a.zarr"}}]}
    _assert_data(PyramidCollection.from_zarr(str(tmp_path / "group.zarr"))[{"series": 0}])


def test_the_rfc_hcs_examples_read_as_plates(tmp_path):
    from test_nodes import TALL, WIDE
    for name, doc in (("wide", WIDE), ("tall", TALL)):
        path = str(tmp_path / f"{name}.json")
        _write_json(path, {"ome": {"version": "0.x", **doc}})
        coll = PyramidCollection.from_zarr(path)
        (plate,) = coll.plates
        assert plate.row_names == ("A",) and plate.column_names == ("1",)
        assert [w.name for w in plate.wells] == ["well A01"]
        assert sorted(dict(k)["acquisition"] for k in coll) == ([0] if name == "wide" else [0, 1])


def _labelled_plate(tmp_path):
    """The 2-well, 2-acquisition plate of test_collection_zarr, read from 0.5, with
    channel names, a translation, a custom block and a label image on one field."""
    src, data = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(src)
    plate = coll.plates[0].node
    first = plate["A2"]["0"]
    pyr = first.pyramid.set_channels({0: {"label": "DAPI", "color": "0000FF"}, 1: {"label": "GFP"}})
    pyr = pyr.set_custom_metadata("my_app", {"run": 7})
    pyr.meta.update_translation("0", [0.0, 3.0, 4.0])
    labels = Pyramid().from_label_arrays([np.ones((16, 24), "u1")], axis_order="yx",
                                         scales=[[0.5, 0.5]], version="0.5", name="nuclei")
    labels = labels.set_image_label(colors=[{"label-value": 1, "rgba": [255, 0, 0, 255]}])
    first.pyramid = pyr.add_image_label(labels, "nuclei")
    first.id = "A2_0"
    return coll, data


def _shape(node):
    """A tree as (type, name, attributes, children), without paths and generated ids,
    and without the eubi: keys RFC 8 output adds (tested in test_rfc8_graph.py)."""
    attrs = {k: v.to_json() if hasattr(v, "to_json") else v for k, v in node.attributes.items()
             if not k.startswith("eubi:")}
    kids = node.children if isinstance(node, CollectionNode) else []
    return (node.type, node.name, json.dumps(attrs, sort_keys=True), [_shape(c) for c in kids])


def _meta(pyr):
    m = pyr.meta
    return (pyr.axes, [m.get_scale(p) for p in m.resolution_paths],
            [m.get_translation(p) for p in m.resolution_paths], m.unit_list,
            [(c.get("label"), c.get("color")) for c in m.metadata["omero"]["channels"]],
            {k: v for k, v in pyr.custom_metadata.items() if not k.startswith("eubi:")})


@pytest.mark.parametrize("standalone", [False, True], ids=["in-the-group", "standalone-json"])
def test_a_plate_written_as_rfc8_reads_back_the_same(tmp_path, standalone):
    coll, data = _labelled_plate(tmp_path)
    out = str(tmp_path / "rfc8.zarr")
    document = str(tmp_path / "docs" / "plate.json") if standalone else None
    if document:
        os.makedirs(os.path.dirname(document))
    written = coll.write_plate(out, version="rfc8", document=document)
    assert written_images(written if not document else out, out) == [
        "A/2/0", "A/2/1", "B/1/0", "B/1/1"]
    assert written == (document or out)                     # where io.read reads it back
    assert os.path.isdir(os.path.join(out, "A", "2", "0", "labels", "nuclei"))
    doc = json.load(open(document)) if document else zarr.open_group(out, mode="r").attrs.asdict()
    assert doc["ome"]["version"] == RFC8_VERSION and doc["ome"]["type"] == "collection"
    image = zarr.open_group(os.path.join(out, "A", "2", "0"), mode="r").attrs.asdict()
    assert image["ome"]["type"] == "multiscale" and image["my_app"] == {"run": 7}
    assert "multiscales" not in image["ome"]
    back = PyramidCollection.from_zarr(document or out)
    assert _opened(back) == []
    assert back.plates == coll.plates
    original = coll.plates[0].node
    label_node = back.plates[0].node["A2"]["0 nuclei"]
    assert label_node.attributes["labels"] == LabelsAttribute(
        [Ref("A2_0")], [{"labelValue": 1, "color": [255, 0, 0, 255]}])
    # the same tree, plus the label image as a node of its own (RFC 8's labels)
    expected = _shape(original)
    got = _shape(back.tree)
    a2 = got[3][0]
    got_a2 = (a2[0], a2[1], a2[2], [c for c in a2[3] if c[1] != "0 nuclei"])
    assert (got[0], got[1], got[2], [got_a2] + got[3][1:]) == expected
    for key in coll:
        assert _meta(back[key]) == _meta(coll[key])
        np.testing.assert_array_equal(np.asarray(back[key].layers["0"]), np.asarray(coll[key].layers["0"]))


def test_series_and_any_collection_are_written_as_rfc8(tmp_path):
    from test_series import _layout
    src, data = _layout(tmp_path, "0.4")
    series = PyramidCollection.from_zarr(src)
    out = str(tmp_path / "series_rfc8.zarr")
    series.write_series(out, version="rfc8")
    root = zarr.open_group(out, mode="r").attrs.asdict()
    assert "bioformats2raw.layout" not in json.dumps(root)          # replaced by the collection
    back = PyramidCollection.from_zarr(out)
    assert list(back) == list(series)
    for i, expected in enumerate(data):
        np.testing.assert_array_equal(np.asarray(back[{"series": i}].layers["0"]), expected)
        assert back[{"series": i}].meta.version == "0.5"
    # a collection with no tree: a flat collection of its images
    from test_collection import FakeImage
    images = [FakeImage(np.full((2, 8, 8), v, "u2"), "cyx", index={"series": v}) for v in (0, 1)]
    plain = PyramidCollection.from_file(type("F", (), {"images": images, "path": None,
                                                       "plates": []})(), version="0.4")
    out2 = str(tmp_path / "plain.zarr")
    plain.write_collection(out2)
    again = PyramidCollection.from_zarr(out2)
    np.testing.assert_array_equal(np.asarray(again[{"series": 1}].layers["0"]), np.full((2, 8, 8), 1))


def test_write_collection_refuses(tmp_path):
    coll, _ = _labelled_plate(tmp_path)
    with pytest.raises(ValueError, match="writes version='rfc8'"):
        coll.write_collection(str(tmp_path / "a.zarr"), version="0.5")
    coll.write_collection(str(tmp_path / "c.zarr"))
    with pytest.raises(FileExistsError, match="overwrite=True"):
        coll.write_collection(str(tmp_path / "c.zarr"))


def test_a_selection_is_written_with_its_part_of_the_tree(tmp_path):
    coll, data = _labelled_plate(tmp_path)
    well = coll.plates[0].well_index("A2")
    part = coll.select(plate=0, well=well)
    out = part.write(str(tmp_path / "part.zarr"), version="rfc8")
    back = PyramidCollection.from_zarr(out)
    assert sorted(back) == sorted(part)
    # the plate keeps its wells (B1 empty), so every key reads back as it was; its views
    # show the wells holding images (decision C-19), B1 stays in the tree underneath
    assert [w.name for w in back.plates[0].wells] == ["A2"]
    assert back.plates[0].well_index("B1") == coll.plates[0].well_index("B1")
    assert [w.name for w in back._plate_views()[0].wells] == ["A2", "B1"]
    assert back.plates[0].row_names == coll.plates[0].row_names      # rows / columns as read
    for key in part:
        np.testing.assert_array_equal(np.asarray(back[key].layers["0"]),
                                      np.asarray(coll[key].layers["0"]))
    assert os.path.isdir(os.path.join(out, "A", "2", "0", "labels", "nuclei"))
    assert len(coll.plates[0].wells) == 2                             # the collection's tree is untouched
    # by acquisition: each well keeps the images of that acquisition
    first = coll.select(acquisition=0)
    again = PyramidCollection.from_zarr(first.write(str(tmp_path / "acq.zarr"), version="rfc8"))
    assert sorted(again) == sorted(first)


def test_a_label_whose_image_is_left_out_is_left_out_too(tmp_path):
    coll, _ = _labelled_plate(tmp_path)
    other = coll.select(well=coll.plates[0].well_index("B1"))
    out = other.write(str(tmp_path / "b1.zarr"), version="rfc8")
    assert not os.path.exists(os.path.join(out, "A", "2"))
    assert sorted(PyramidCollection.from_zarr(out)) == sorted(other)


EXAMPLES = os.environ.get("RFC8_EXAMPLES")


@pytest.mark.skipif(not EXAMPLES or not os.path.isdir(EXAMPLES),
                    reason="set RFC8_EXAMPLES to a checkout of normanrz/ngff-rfc8-collection-examples")
def test_the_rfc_example_repository_reads():
    found = 0
    for p in sorted(glob.glob(os.path.join(EXAMPLES, "**", "*.json"), recursive=True)):
        try:
            doc = json.load(open(p))
        except ValueError:
            continue
        if not isinstance(doc.get("ome") or (doc.get("attributes") or {}).get("ome"), dict):
            continue
        target = os.path.dirname(p) if p.endswith("zarr.json") else p
        coll = PyramidCollection.from_zarr(target)
        found += 1
        if "zarr.json" in p and "https" not in json.dumps(doc):
            pyr = coll[next(iter(coll))]                # local images: they open
            assert pyr.nlayers >= 1
    assert found >= 10
