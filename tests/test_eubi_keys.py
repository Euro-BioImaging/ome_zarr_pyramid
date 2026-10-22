"""The eubi: keys of NGFF 0.4 / 0.5 (plan 9.15: 5.5-5.7, 5.11, 5.21, 5.28, 8.2, 8.4, 8.5):
section 6's shapes written and read back with the same keys, positions, systems and
edges; where each key sits; reading opens no image and needs no extra batch."""
import json
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import PyramidCollection
from ome_zarr_pyramid.core import ngff_plate
from ome_zarr_pyramid.core.spatial import Axis, Transformation, matrix, pyramid_system

from test_collection import FakeImage, Q
from test_spatial_collection import _cs, _file, _img, _placed

UM = "micrometer"


def _attrs(path, group=""):
    return zarr.open_group(os.path.join(path, group), mode="r").attrs.asdict()


def _quiet(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _positions(coll):
    return {k: coll[k].meta.get_translation("0") for k in coll.system_of}


def _same_graph(back, coll):
    assert list(back) == list(coll)
    assert back.system_of == coll.system_of
    assert set(back.coordinate_systems) == set(coll.coordinate_systems)
    for sid, cs in coll.coordinate_systems.items():
        got = back.coordinate_systems[sid]
        assert (got.name, got.estimated) == (cs.name, cs.estimated)
        assert got.axes is not None                    # filled when written (3.12)
    assert _positions(back) == _positions(coll)


# -- plates: 6.1 (mosaic) and 6.6 ---------------------------------------------------------

def _plate_file(per_well, fields=(0, 1)):
    plate = SimpleNamespace(name="P", rows=("A", "B"), columns=("1", "2", "3", "4"),
                            acquisitions=(), wells=(SimpleNamespace(row=1, column=1, name="B2"),
                                                    SimpleNamespace(row=1, column=3, name="B4")))
    images = []
    for w, well in enumerate(("B2", "B4")):
        for f in fields:
            sid = f"well-{well}" if per_well else "stage"
            images.append(_placed({"plate": 0, "acquisition": 0, "well": w, "field": f}, sid,
                                  y=100.0 * f + w, x=7.0, value=10 * w + f))
    systems = [_cs("well-B2"), _cs("well-B4")] if per_well else [_cs("stage")]
    file = _file(images, systems, path="/data/screen.czi")
    file.plates = [plate]
    return PyramidCollection.from_file(file)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_a_mosaic_plate_in_one_stage_system_round_trips(tmp_path, version):     # 6.1
    coll = _plate_file(per_well=False)
    out = str(tmp_path / "p.zarr")
    coll.write_plate(out, version=version)
    root = _attrs(out)
    assert root["eubi:version"] == 1 and root["eubi:kind"] == "plate"
    assert "eubi:version" not in root.get("ome", {})              # beside ome, never in it
    # one component across both wells: the plate group owns it (3.6)
    assert [s["name"] for s in root["eubi:coordinate_systems"]] == ["stage"]
    well = _attrs(out, "B/2")
    assert well["eubi:kind"] == "well"
    assert well["eubi:images"]["1"] == {"index": {"plate": 0, "acquisition": 0, "well": 0,
                                                  "field": 1}, "coordinate_system": "stage"}
    image = _attrs(out, "B/2/1")
    assert image["eubi:index"] == {"plate": 0, "acquisition": 0, "well": 0, "field": 1}
    assert image["eubi:coordinate_system"] == "stage"            # the single-image form
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)


def test_a_plate_with_one_system_per_well_puts_each_on_its_well(tmp_path):      # 6.6
    coll = _plate_file(per_well=True, fields=(0, 2))
    out = str(tmp_path / "p.zarr")
    coll.write_plate(out)
    assert "eubi:coordinate_systems" not in _attrs(out)
    for well in ("B2", "B4"):
        systems = _attrs(out, f"B/{well[1]}")["eubi:coordinate_systems"]
        assert [s["name"] for s in systems] == [f"well-{well}"]
        assert [a["name"] for a in systems[0]["axes"]] == ["c", "y", "x"]
        assert systems[0]["eubi:estimated"] is False
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)
    assert [f.index for f in back.plates[0].wells[0].fields] == [0, 2]    # 8.4
    assert [(sorted(c.systems), c.owner) for c in back.components()] == [
        (["well-B2"], ("B2",)), (["well-B4"], ("B4",))]      # the read plate is the root


# -- series: 6.2, 6.4, 6.7 --------------------------------------------------------------

def test_series_with_an_unplaced_label_and_preview_round_trip(tmp_path):         # 6.2
    images = [_placed({"series": 0}, "stage", value=1), _placed({"series": 1}, "stage", y=50,
                                                                value=2),
              _img({"series": 2}, value=3), _img({"series": 3}, value=4)]   # label, preview
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")]))
    out = str(tmp_path / "s.zarr")
    coll.write_series(out)
    root = _attrs(out)
    assert root["eubi:images"]["2"] == {"index": {"series": 2}, "coordinate_system": None}
    assert "eubi:coordinate_system" not in _attrs(out, "2")
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)


def test_a_tiling_set_flattened_to_series_gets_its_tiles_back(tmp_path):         # 6.4
    images = [_placed({"series": 0, "tile": t}, "stage", y=10.0 * t, x=-2.0 * t, value=t)
              for t in range(25)]
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")], path="/data/matl.omp2info"))
    out = str(tmp_path / "s.zarr")
    with pytest.warns(UserWarning, match="each image is written as a series of its own"):
        coll.write_series(out)
    assert _attrs(out, "24")["eubi:index"] == {"series": 0, "tile": 24}
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)
    assert [n.name for n in back.tree] == ["series0"] and len(back.tree["series0"]) == 25


def test_mixed_rank_images_share_one_system_with_the_union_of_their_axes(tmp_path):  # 6.7
    zyx = FakeImage(np.zeros((3, 8, 8), "u2"), "zyx", SimpleNamespace(
        pixel_sizes={"z": Q(2.0, UM), "y": Q(0.5, UM), "x": Q(0.5, UM)},
        position={"z": Q(1.0, UM), "y": Q(4.0, UM), "x": Q(5.0, UM)}, channels=[],
        index_values={}, coordinate_system="stage"), index={"series": 0})
    yx = FakeImage(np.zeros((8, 8), "u2"), "yx", SimpleNamespace(
        pixel_sizes={"y": Q(0.5, UM), "x": Q(0.5, UM)}, position={"y": Q(6.0, UM), "x": Q(7.0, UM)},
        channels=[], index_values={}, coordinate_system="stage"), index={"series": 1})
    coll = PyramidCollection.from_file(_file([zyx, yx], [_cs("stage")]))
    out = str(tmp_path / "s.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)                 # 0.4 / 0.5: works, no warning
        coll.write_series(out)
    (system,) = _attrs(out)["eubi:coordinate_systems"]
    assert [a["name"] for a in system["axes"]] == ["z", "y", "x"]
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)


# -- 6.3: affine views, flattened to series -----------------------------------------------

def test_affine_views_keep_their_edges_under_eubi_with_a_warning(tmp_path):
    turned = [[1, 0, 0, 0], [0, 0, -1, 5], [0, 1, 0, 0]]
    images = [_img({"series": 0, "view": v}, value=v) for v in range(2)]
    edges = [Transformation("affine", (("series", 0), ("view", v)), "global",
                            {"affine": turned if v else [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]})
             for v in range(2)]
    coll = PyramidCollection.from_file(_file(images, [_cs("global")], edges))
    out = str(tmp_path / "s.zarr")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coll.write_series(out)
    said = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("has no place for 2 edge(s)" in m and "eubi:coordinate_transformations" in m
               for m in said)
    written = _attrs(out)["eubi:coordinate_transformations"]
    assert written[1]["input"] == {"name": "physical", "path": "1"}
    assert written[1]["output"] == {"name": "global"}
    back = PyramidCollection.from_zarr(out)
    assert list(back) == list(coll)
    for mine, theirs in zip(coll.coordinate_transformations, back.coordinate_transformations):
        assert (theirs.input, theirs.output) == (mine.input, mine.output)
        assert np.allclose(matrix(theirs, 3), matrix(mine, 3))
    (c,) = back.components()
    assert c.systems == {"global"} and len(c.entries) == 2


# -- the JSON shapes (5.28) ----------------------------------------------------------------

_END = {"type": "object", "required": ["name"], "properties": {
    "name": {"type": "string"}, "path": {"type": "string"}}, "additionalProperties": False}
_SCHEMA = {
    "type": "object",
    "properties": {
        "eubi:version": {"const": 1},
        "eubi:images": {"type": "object", "additionalProperties": {
            "type": "object", "required": ["index", "coordinate_system"],
            "additionalProperties": False, "properties": {
                "index": {"type": "object", "additionalProperties": {"type": "integer"}},
                "coordinate_system": {"type": ["string", "null"]}}}},
        "eubi:coordinate_systems": {"type": "array", "items": {
            "type": "object", "required": ["name", "eubi:name", "eubi:estimated"],
            "properties": {
                "name": {"type": "string", "pattern": "^[a-zA-Z0-9_.-]+$"},
                "eubi:name": {"type": "string"}, "eubi:estimated": {"type": "boolean"},
                "axes": {"type": "array", "items": {
                    "type": "object", "required": ["name", "type"],
                    "properties": {"name": {"type": "string"}, "type": {"type": "string"},
                                   "unit": {"type": "string"}}}}}}},
        "eubi:coordinate_transformations": {"type": "array", "items": {
            "type": "object", "required": ["type", "input", "output"],
            "properties": {"type": {"type": "string"}, "input": _END, "output": _END}}},
    },
}


def test_the_written_keys_follow_the_json_shapes_of_5_28(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    reg = Transformation("translation", (("series", 0),), "global-registered",
                         {"translation": [0, 1, 2]})
    images = [_placed({"series": s}, "stage", value=s) for s in range(2)]
    coll = PyramidCollection.from_file(_file(
        images, [_cs("stage"), _cs("global-registered", estimated=True)], [reg]))
    out = str(tmp_path / "s.zarr")
    _quiet(lambda: coll.write_series(out))
    root = _attrs(out)
    jsonschema.validate(root, _SCHEMA)
    assert {s["name"]: s["eubi:estimated"] for s in root["eubi:coordinate_systems"]} == {
        "stage": False, "global-registered": True}                       # 5.21
    jsonschema.validate(_attrs(out, "0"), _SCHEMA)
    back = PyramidCollection.from_zarr(out)
    assert back.coordinate_transformations == [reg]
    assert back.coordinate_systems["global-registered"].estimated is True


# -- lone images and the reader's rules ----------------------------------------------------

def test_an_entry_written_alone_keeps_its_index_and_system(tmp_path):
    images = [_placed({"series": s}, "stage", value=s) for s in range(2)]
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")]))
    paths = coll.write(str(tmp_path), "f")
    attrs = _attrs(paths[(("series", 1),)])
    assert attrs["eubi:index"] == {"series": 1} and attrs["eubi:version"] == 1
    lone = PyramidCollection.from_zarr(paths[(("series", 1),)])
    assert lone.system_of == {(): "stage"} and list(lone.coordinate_systems) == ["stage"]


def test_a_newer_key_schema_warns_and_a_disagreeing_index_keeps_the_plates(tmp_path):
    coll = _plate_file(per_well=False)
    out = str(tmp_path / "p.zarr")
    coll.write_plate(out)
    group = zarr.open_group(out, mode="a")
    group.attrs["eubi:version"] = 2
    well = zarr.open_group(os.path.join(out, "B/2"), mode="a")
    images = well.attrs["eubi:images"]
    images["0"]["index"]["well"] = 5                    # the plate says well 0
    well.attrs["eubi:images"] = images
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = PyramidCollection.from_zarr(out)
    said = " ".join(str(w.message) for w in caught)
    assert "eubi:version 2 is newer" in said and "the plate's is kept" in said
    assert list(back) == list(coll)


def test_a_collection_without_eubi_keys_reads_as_before(tmp_path):
    coll = _plate_file(per_well=False)
    out = str(tmp_path / "p.zarr")
    coll.write_plate(out)
    for group in ["", "B/2", "B/4"]:
        g = zarr.open_group(os.path.join(out, group), mode="a")
        for key in [k for k in g.attrs if k.startswith("eubi:")]:
            del g.attrs[key]
    back = PyramidCollection.from_zarr(out)
    assert back.system_of == {} and back.coordinate_systems == {}
    assert [dict(k)["field"] for k in back] == [0, 1, 0, 1]


# -- reading opens nothing (3.18; done-when T4) ------------------------------------------

def _counting(monkeypatch):
    calls = {"open": 0, "batch": 0}
    real_open, real_batch = ngff_plate.ImageOpener.__call__, ngff_plate._group_attrs

    def opened(self, node):
        calls["open"] += 1
        return real_open(self, node)

    def batch(root, paths):
        calls["batch"] += 1
        return real_batch(root, paths)
    monkeypatch.setattr(ngff_plate.ImageOpener, "__call__", opened)
    monkeypatch.setattr(ngff_plate, "_group_attrs", batch)
    return calls


def test_from_zarr_opens_no_image_and_needs_one_batch_more_only_without_the_lists(
        tmp_path, monkeypatch):
    coll = _plate_file(per_well=False, fields=(0, 1, 2))
    out = str(tmp_path / "p.zarr")
    coll.write_plate(out)
    calls = _counting(monkeypatch)
    listed = PyramidCollection.from_zarr(out)
    assert calls == {"open": 0, "batch": 1}            # the wells, as before
    for well in ("B/2", "B/4"):                        # a hand-edited copy: no lists
        g = zarr.open_group(os.path.join(out, well), mode="a")
        del g.attrs["eubi:images"]
    calls.update(open=0, batch=0)
    stripped = PyramidCollection.from_zarr(out)
    assert calls == {"open": 0, "batch": 2}            # plus the image groups, at once
    assert list(stripped) == list(listed) and stripped.system_of == listed.system_of
