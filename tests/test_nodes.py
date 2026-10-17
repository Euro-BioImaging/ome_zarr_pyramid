"""The in-memory collection model (NGFF RFC 8 nodes): RFC 8 JSON in and out unchanged,
typed plate/well/acquisition/labels attributes, unknown keys kept, children behind a
path opened only on first use, copies, and the rules a tree must follow."""
import copy

import numpy as np
import pytest

from ome_zarr_pyramid import Pyramid
from ome_zarr_pyramid.core.nodes import (
    CollectionNode, InvalidCollection, LabelsAttribute, MultiscaleNode, Node, NodePath,
    PlateAttribute, PlateItem, Ref, SinglescaleNode, WellAttribute, is_plate, is_well)

#: RFC 8's "wide" HCS example (acquisition attribute on each image of the well),
#: plus extension keys of other tools
WIDE = {
    "type": "collection", "name": "hcs-plate-001",
    "attributes": {
        "plate": {"acquisitions": [{"id": "acq_0", "name": "Acquisition Round 1",
                                    "starttime": 1343731272000}],
                  "columns": [{"id": "1", "name": "1"}],
                  "rows": [{"id": "A", "name": "A"}]},
        "mobie:grid": {"cols": 1}},
    "nodes": [{
        "type": "collection", "name": "well A01",
        "attributes": {"well": {"column": {"id": "1"}, "row": {"id": "A"}}},
        "nodes": [
            {"id": "A01_0", "type": "multiscale", "name": "A01_0",
             "path": {"type": "zarr", "path": "./A/01/001.img"},
             "attributes": {"acquisition": {"id": "acq_0"}}},
            {"type": "multiscale", "name": "A01_0_ill_corrected",
             "path": {"type": "zarr", "path": "./A/01/001_ill_corrected.img"},
             "attributes": {"acquisition": {"id": "acq_0"}, "source": [{"id": "A01_0"}]}},
            {"type": "multiscale", "name": "A01_0_nuclei",
             "path": {"type": "zarr", "path": "./A/01/001_nuclei.img"},
             "attributes": {"acquisition": {"id": "acq_0"},
                            "labels": {"source": [{"id": "A01_0"}]}}}]}]}

#: RFC 8's "tall" layout: one sub-collection per acquisition inside the well
TALL = {
    "type": "collection", "name": "hcs-plate-001",
    "attributes": {"plate": {
        "acquisitions": [{"id": "acq_0", "name": "Acquisition Round 1"},
                         {"id": "acq_1", "name": "Acquisition Round 2"}],
        "columns": [{"id": "1", "name": "1"}], "rows": [{"id": "A", "name": "A"}]}},
    "nodes": [{
        "type": "collection", "name": "well A01",
        "attributes": {"well": {"column": {"id": "1"}, "row": {"id": "A"}}},
        "nodes": [
            {"type": "collection", "name": f"acquisition {i}",
             "attributes": {"acquisition": {"id": f"acq_{i}"}},
             "nodes": [{"type": "multiscale", "name": f"A01_{i}",
                        "path": {"type": "zarr", "path": f"./A/01/{i}.img"}}]}
            for i in (0, 1)]}]}

#: a multiscale listing its levels inline (RFC 8's first example)
MULTISCALE = {
    "type": "multiscale", "name": "our_first_example",
    "attributes": {"coordinateSystems": [{"name": "physical", "axes": [
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}]}]},
    "nodes": [{"id": "s0", "name": "s0", "type": "singlescale",
               "path": {"type": "zarr", "path": "./s0"},
               "attributes": {"coordinateTransformations": [
                   {"type": "scale", "scale": [0.5, 0.5], "input": "s0",
                    "output": "physical"}]}}]}


@pytest.mark.parametrize("doc", [WIDE, TALL, MULTISCALE], ids=["wide", "tall", "levels"])
def test_rfc8_json_reads_and_writes_back_unchanged(doc):
    node = Node.from_json(copy.deepcopy(doc))
    assert node.to_json() == doc
    assert node.validate() if isinstance(node, CollectionNode) else True


def test_known_attributes_are_typed():
    plate = Node.from_json(WIDE)
    assert is_plate(plate) and not is_well(plate)
    attr = plate.attributes["plate"]
    assert isinstance(attr, PlateAttribute)
    assert attr.rows == [PlateItem("A", "A")]
    assert attr.acquisitions[0].extra == {"starttime": 1343731272000}
    well = plate["well A01"]
    assert is_well(well)
    assert well.attributes["well"] == WellAttribute(Ref("A"), Ref("1"))
    assert attr.index_of("rows", well.attributes["well"].row) == 0
    nuclei = well["A01_0_nuclei"]
    assert isinstance(nuclei, MultiscaleNode)
    assert nuclei.attributes["acquisition"] == Ref("acq_0")
    assert nuclei.attributes["labels"] == LabelsAttribute([Ref("A01_0")])
    assert plate.find("A01_0") is well["A01_0"]
    with pytest.raises(KeyError, match="no row with id 'B'"):
        attr.index_of("rows", "B")


def test_unknown_keys_types_and_fields_are_kept():
    doc = {"type": "collection", "name": "root", "version": "0.x",
           "fractal:run": {"id": 7},
           "attributes": {"_creator": {"name": "omero-zarr", "version": "0.5.5"},
                          "mobie:grid": [1, 2]},
           "nodes": [{"type": "mobie:view", "name": "a view", "id": "v1",
                      "nodes": [{"anything": True}], "attributes": {"x": 1}},
                     {"type": "singlescale", "name": "s", "nodes": ["kept"],
                      "path": {"type": "zarr", "path": "./s"}}]}
    node = Node.from_json(copy.deepcopy(doc))
    assert node.extra == {"version": "0.x", "fractal:run": {"id": 7}}
    view = node["a view"]
    assert type(view) is Node and view.type == "mobie:view"
    assert view.extra == {"nodes": [{"anything": True}]}
    assert isinstance(node["s"], SinglescaleNode) and node["s"].extra == {"nodes": ["kept"]}
    assert node.to_json() == doc


def test_references_written_as_plain_strings_stay_strings():
    doc = {"type": "multiscale", "name": "nuclei", "path": {"type": "zarr", "path": "./n"},
           "attributes": {"labels": {"source": ["raw"], "labelAttributes": [
               {"labelValue": 1, "color": [255, 0, 0, 255]}]}}}
    node = Node.from_json(copy.deepcopy(doc))
    assert node.attributes["labels"].source[0].id == "raw"
    assert node.to_json() == doc


def _pyramid():
    return Pyramid().from_arrays([np.zeros((2, 16, 16), "u1")], axis_order="cyx",
                                 scales=[[1, 1, 1]], version="0.5", name="img")


def _tree(loader):
    """A plate built in memory: one well holding a pyramid, one well behind a path,
    and an image referenced from another document."""
    pyr = _pyramid()
    plate = CollectionNode("plate", attributes={"plate": PlateAttribute(
        rows=[PlateItem("A"), PlateItem("B")], columns=[PlateItem("1")],
        acquisitions=[PlateItem("0", "first")]), "lab:barcode": "P-001"}, id="p")
    a1 = plate.add(CollectionNode("A1", attributes={"well": WellAttribute(Ref("A"), Ref("1"))}))
    a1.add(MultiscaleNode("field 0", pyr, id="A1_0",
                          attributes={"acquisition": Ref("0")}))
    a1.add(MultiscaleNode("A1_0_labels", path=NodePath("zarr", "./A/1/0/labels/nuclei"),
                          attributes={"labels": LabelsAttribute([Ref("A1_0")])},
                          loader=loader))
    plate.add(CollectionNode("B1", path=NodePath("json", "./B1.json"), loader=loader,
                             attributes={"well": WellAttribute(Ref("B"), Ref("1"))}))
    plate.add(MultiscaleNode("reference", path=NodePath("zarr", "https://example.org/x.zarr"),
                             attributes={"acquisition": Ref("0", NodePath("json", "../o.json"))},
                             loader=loader))
    return plate, pyr


def test_an_in_memory_tree_survives_a_copy_unchanged():
    calls = []
    plate, pyr = _tree(lambda node: calls.append(node.name))
    twin = plate.copy()
    assert twin == plate and twin is not plate
    assert twin.validate() is twin
    held = twin["A1"]["field 0"]
    assert held.pyramid is pyr                    # pyramids are shared, not copied
    assert not twin["B1"].loaded and not twin["reference"].loaded
    assert calls == []                            # copying, comparing, validating open nothing
    # the copy is independent of the original
    twin.attributes["plate"].rows.append(PlateItem("C"))
    twin["A1"].attributes["lab:note"] = "edited"
    twin["A1"].add(MultiscaleNode("field 1", _pyramid()))
    assert [r.id for r in plate.attributes["plate"].rows] == ["A", "B"]
    assert "lab:note" not in plate["A1"].attributes
    assert len(plate["A1"]) == 2 and len(twin["A1"]) == 3
    assert twin != plate


def test_children_behind_a_path_open_on_first_use_only():
    calls = []

    def loader(node):
        calls.append(node.name)
        if node.type == "collection":
            return [MultiscaleNode("field 0", _pyramid())]
        return _pyramid()

    plate, _ = _tree(loader)
    well = plate["B1"]
    assert not well.loaded and calls == []
    assert [n.name for n in well] == ["field 0"]
    assert well.loaded and calls == ["B1"]
    len(well), well["field 0"]
    assert calls == ["B1"]                        # opened once
    labels = plate["A1"]["A1_0_labels"]
    assert isinstance(labels.pyramid, Pyramid) and calls == ["B1", "A1_0_labels"]
    # a collection behind a path is written as its path, opened or not
    assert well.to_json() == {
        "type": "collection", "name": "B1", "path": {"type": "json", "path": "./B1.json"},
        "attributes": {"well": {"column": {"id": "1"}, "row": {"id": "B"}}}}


def test_without_a_loader_a_path_says_so():
    node = MultiscaleNode("img", path=NodePath("zarr", "./img.zarr"))
    with pytest.raises(LookupError, match="'./img.zarr' .* no loader"):
        node.pyramid
    with pytest.raises(LookupError, match="holds no pyramid"):
        MultiscaleNode("empty").pyramid


def test_a_pyramid_in_memory_has_no_rfc8_json_yet():
    with pytest.raises(NotImplementedError, match="written together with the pyramid"):
        MultiscaleNode("img", _pyramid()).to_json()


@pytest.mark.parametrize("make, message", [
    (lambda: CollectionNode(""), "non-empty string"),
    (lambda: CollectionNode("c", id="has space"), "letters, digits"),
    (lambda: Ref("a/b"), "letters, digits"),
    (lambda: Node.from_json({"type": "collection", "name": "c", "nodes": [],
                             "path": {"type": "json", "path": "c.json"}}), "not both"),
    (lambda: CollectionNode("c", [MultiscaleNode("x"), MultiscaleNode("x")]),
     "already has a node named 'x'"),
    (lambda: Node.from_json({"type": "collection", "name": "c", "nodes": {}}), "must be a list"),
    (lambda: Node.from_json({"type": "collection", "name": "c",
                             "attributes": {"plate": {"rows": []}}}), "'rows' and 'columns'"),
    (lambda: Node.from_json({"name": "c"}), "needs a type"),
])
def test_malformed_nodes_are_refused(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_validation_names_every_problem():
    doc = copy.deepcopy(WIDE)
    plate_attr = doc["attributes"]["plate"]
    plate_attr["rows"].append({"id": "A"})                         # duplicate row id
    well = doc["nodes"][0]
    well["attributes"]["well"]["column"] = {"id": "7"}             # not a plate column
    images = well["nodes"]
    images[1]["id"] = "A01_0"                                      # duplicate node id
    images[2]["attributes"]["acquisition"] = {"id": "acq_9"}       # not a plate acquisition
    images[2]["attributes"]["labels"]["source"] = [{"id": "nope"}]  # names no node
    found = Node.from_json(doc).problems()
    assert len(found) == 5
    with pytest.raises(InvalidCollection) as err:
        Node.from_json(doc).validate()
    text = str(err.value)
    for part in ("row id 'A' is used twice", "column '7' is not one of the plate's columns",
                 "id 'A01_0' is used twice", "acquisition 'acq_9' is not one of",
                 "label source 'nope' names no node"):
        assert part in text


def test_wells_and_acquisitions_need_a_plate_above_them():
    tree = CollectionNode("root", [
        CollectionNode("w", [MultiscaleNode("img", path=NodePath("zarr", "./img"),
                                            attributes={"acquisition": Ref("0")})],
                       attributes={"well": WellAttribute(Ref("A"), Ref("1"))})])
    found = tree.problems()
    assert found == ["well 'w' is not inside a plate",
                     "MultiscaleNode('img', path='./img') has an acquisition but is not "
                     "inside a plate"]
