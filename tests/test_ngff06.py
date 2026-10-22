"""NGFF 0.6 (plan 9.11: 3.20, 3.23, 3.28, 5.4-5.6, 5.17, 5.27): images read into ozp's one
form and written back in 0.6's; what that form has no place for kept as read; labels'
links; plates, wells and series layouts in 0.6 (single-image form); scenes in files
ozp reads place their images. The default output stays 0.5 (5.27)."""
import json
import os
import warnings

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection, io
from ome_zarr_pyramid.core import ngff06
from ome_zarr_pyramid.core.query import row_from_json
from ome_zarr_pyramid.core.spatial import InvariantError

from test_eubi_keys import _plate_file, _same_graph
from test_spatial_collection import _cs, _file, _img, _placed

UM = "micrometer"
AXES = [{"name": "y", "type": "space", "unit": UM}, {"name": "x", "type": "space", "unit": UM}]


def _attrs(path, group=""):
    return zarr.open_group(os.path.join(path, group), mode="r").attrs.asdict()


def _quiet(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _image06(path, data, *, levels=None, intrinsic="physical", extra=None):
    """A hand-written 0.6 image: zarr v3 arrays and the spec's JSON (not ozp's writer).
    `levels`: one 0.6 transformation per level (without input / output)."""
    g = zarr.open_group(str(path), mode="w", zarr_format=3)
    levels = levels or [{"type": "scale", "scale": [0.5, 0.5]}]
    datasets = []
    for i, t in enumerate(levels):
        arr = g.create_array(str(i), shape=data[:, ::2 ** i][:, :, ::2 ** i].shape if data.ndim == 3
                             else data[::2 ** i, ::2 ** i].shape, dtype=data.dtype)
        arr[...] = data[:, ::2 ** i, ::2 ** i] if data.ndim == 3 else data[::2 ** i, ::2 ** i]
        datasets.append({"path": str(i), "coordinateTransformations": [
            {**t, "input": {"path": str(i)}, "output": {"name": intrinsic}}]})
    ms = {"name": "img", "coordinateSystems": [{"name": intrinsic, "axes": AXES}],
          "datasets": datasets, **(extra or {})}
    g.attrs.put({"ome": {"version": "0.6", "multiscales": [ms]}})
    return str(path)


DATA = np.arange(16 * 16, dtype="u2").reshape(16, 16)


# -- one image ------------------------------------------------------------------------------

def test_an_image_written_as_06_has_the_specs_form_and_reads_back(tmp_path):
    pyr = Pyramid().from_arrays([DATA, DATA[::2, ::2]], axis_order="yx",
                                scales=[[0.5, 0.5], [1.0, 1.0]], unit_list=[UM, UM],
                                version="0.6")
    pyr.meta.update_translation("1", [0.25, 0.25])
    out = str(tmp_path / "i.zarr")
    IO().write_pyramid(pyr, out)
    assert zarr.open_group(out, mode="r").metadata.zarr_format == 3
    ome = _attrs(out)["ome"]
    assert ome["version"] == "0.6"
    (ms,) = ome["multiscales"]
    assert "axes" not in ms and "version" not in ms
    assert ms["coordinateSystems"] == [{"name": "physical", "axes": AXES}]
    l0, l1 = (d["coordinateTransformations"] for d in ms["datasets"])
    assert l0 == [{"type": "scale", "scale": [0.5, 0.5], "input": {"path": "0"},
                   "output": {"name": "physical"}}]
    assert l1 == [{"type": "sequence", "input": {"path": "1"}, "output": {"name": "physical"},
                   "transformations": [{"type": "scale", "scale": [1.0, 1.0]},
                                       {"type": "translation", "translation": [0.25, 0.25]}]}]
    back = io.read_pyramid(out)
    assert back.meta.version == "0.6" and back.meta.zarr_format == 3
    assert back.meta.axis_order == "yx"
    assert back.meta.get_scale("1") == [1.0, 1.0]
    assert back.meta.get_translation("1") == [0.25, 0.25]
    np.testing.assert_array_equal(np.asarray(back.layers["0"]), DATA)


def test_identity_and_sequence_levels_are_read(tmp_path):
    path = _image06(tmp_path / "i.zarr", DATA, levels=[
        {"type": "identity"},
        {"type": "sequence", "transformations": [{"type": "scale", "scale": [2.0, 2.0]},
                                                 {"type": "translation", "translation": [0.5, 0.5]}]}])
    pyr = io.read_pyramid(path)
    assert pyr.meta.get_scale("0") == [1.0, 1.0]
    assert pyr.meta.get_scale("1") == [2.0, 2.0]
    assert pyr.meta.get_translation("1") == [0.5, 0.5]


@pytest.mark.parametrize("level, match", [
    ({"type": "translation", "translation": [1.0, 1.0]}, "a 'translation' transformation"),
    ({"type": "sequence", "transformations": [{"type": "translation", "translation": [1, 1]},
                                              {"type": "scale", "scale": [1, 1]}]},
     "a 'sequence' transformation"),
    ({"type": "affine", "affine": [[1, 0, 0], [0, 1, 0]]}, "a 'affine' transformation"),
])
def test_a_level_06_does_not_allow_is_refused_not_misread(tmp_path, level, match):
    path = _image06(tmp_path / "i.zarr", DATA, levels=[level])
    with pytest.raises(ngff06.Unreadable06, match=match):
        io.read_pyramid(path)


def test_levels_into_a_system_that_is_not_listed_are_refused(tmp_path):
    path = _image06(tmp_path / "i.zarr", DATA)
    g = zarr.open_group(path, mode="r+")
    ome = g.attrs["ome"]
    ome["multiscales"][0]["coordinateSystems"][0]["name"] = "elsewhere"
    g.attrs["ome"] = ome
    with pytest.raises(ngff06.Unreadable06, match="not one of the multiscale's coordinateSystems"):
        io.read_pyramid(path)


FOREIGN = {
    "coordinateSystems": [{"name": "intrinsic", "axes": AXES},
                          {"name": "array", "axes": [{"name": "y", "type": "space"},
                                                     {"name": "x", "type": "space"}]}],
    "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0],
                                   "input": {"name": "intrinsic"}, "output": {"name": "array"}}],
}


def _foreign(tmp_path):
    path = _image06(tmp_path / "f.zarr", DATA, intrinsic="intrinsic")
    g = zarr.open_group(path, mode="r+")
    ome = g.attrs["ome"]
    ome["multiscales"][0].update(FOREIGN)
    g.attrs["ome"] = ome
    return path


def test_a_foreign_images_extra_systems_and_edges_are_written_back_as_read(tmp_path):
    pyr = io.read_pyramid(_foreign(tmp_path))
    assert pyr.meta.axis_order == "yx"              # the intrinsic system's axes
    out = str(tmp_path / "again.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        IO().write_pyramid(pyr, out)                # 0.6 in, 0.6 out: no warning
    (ms,) = _attrs(out)["ome"]["multiscales"]
    assert ms["coordinateSystems"] == FOREIGN["coordinateSystems"]       # names kept (3.23)
    assert ms["coordinateTransformations"] == FOREIGN["coordinateTransformations"]
    assert ms["datasets"][0]["coordinateTransformations"][0]["output"] == {"name": "intrinsic"}


def test_written_as_05_the_foreign_parts_are_kept_under_eubi_with_a_warning(tmp_path):
    pyr = io.read_pyramid(_foreign(tmp_path))
    out = str(tmp_path / "v05.zarr")
    with pytest.warns(UserWarning, match="NGFF 0.5 has no place for the 0.6 coordinate"):
        IO().write_pyramid(pyr.to_version("0.5"), out)
    attrs = _attrs(out)
    (ms,) = attrs["ome"]["multiscales"]
    assert ngff06.STASH not in ms and "coordinateSystems" not in ms
    kept = attrs[ngff06.FOREIGN_KEY]
    assert kept["coordinateSystems"] == FOREIGN["coordinateSystems"][1:]
    assert kept["coordinateTransformations"] == FOREIGN["coordinateTransformations"]
    # and back in place when that image is written as 0.6 again
    again = str(tmp_path / "v06.zarr")
    IO().write_pyramid(io.read_pyramid(out).to_version("0.6"), again)
    (ms,) = _attrs(again)["ome"]["multiscales"]
    assert ms["coordinateSystems"][1:] == FOREIGN["coordinateSystems"][1:]
    assert ngff06.FOREIGN_KEY not in _attrs(again)


def test_a_read_05_image_written_as_06_gives_no_warning(tmp_path):
    src = str(tmp_path / "v05.zarr")
    IO().write_pyramid(Pyramid().from_arrays([DATA], axis_order="yx", scales=[[0.5, 0.5]],
                                             version="0.5"), src)
    out = str(tmp_path / "v06.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        IO().write_pyramid(io.read_pyramid(src).to_version("0.6"), out)
    assert _attrs(out)["ome"]["version"] == "0.6"


def test_the_table_row_of_a_06_image(tmp_path):
    path = _image06(tmp_path / "i.zarr", DATA, levels=[
        {"type": "sequence", "transformations": [{"type": "scale", "scale": [0.5, 0.5]},
                                                 {"type": "translation", "translation": [3.0, 4.0]}]}])
    row = row_from_json(_attrs(path), [{"shape": (16, 16), "dtype": "uint16"}], path=path)
    assert row["version"] == "0.6" and row["axes"] == "yx"
    assert row["scale"] == (0.5, 0.5) and row["translation"] == (3.0, 4.0)
    assert row["units"] == (UM, UM)


def test_the_default_output_stays_05():
    assert Pyramid().from_arrays([DATA], axis_order="yx").meta.version == "0.5"


# -- labels (3.28) --------------------------------------------------------------------------

def test_a_label_gets_one_identity_link_and_reading_drops_it(tmp_path):
    img = Pyramid().from_array(DATA, axis_order="yx", version="0.6")
    lbl = Pyramid().from_label_arrays([(DATA > 100).astype("u1")], axis_order="yx")
    out = str(tmp_path / "i.zarr")
    io.write_pyramid(img.add_image_label(lbl, "cells"), out)
    (ms,) = _attrs(out)["ome"]["multiscales"]
    assert ms["coordinateTransformations"] == [
        {"type": "identity", "input": {"name": "physical"},
         "output": {"name": "physical", "path": "labels/cells"}}]
    label_ome = _attrs(out, "labels/cells")["ome"]
    assert label_ome["version"] == "0.6" and "image-label" in label_ome
    back = io.read_pyramid(out)
    assert back.labels.names == ["cells"]
    assert "coordinateTransformations" not in back.meta.multiscales     # dropped on reading
    again = str(tmp_path / "again.zarr")
    io.write_pyramid(back, again)                   # ... and written again, once
    (ms,) = _attrs(again)["ome"]["multiscales"]
    assert len(ms["coordinateTransformations"]) == 1


def test_a_label_link_that_is_not_an_identity_is_kept_as_read(tmp_path):
    link = {"type": "scale", "scale": [2.0, 2.0], "input": {"name": "physical"},
            "output": {"name": "physical", "path": "labels/cells"}}
    ms = ngff06.from_06({"coordinateSystems": [{"name": "physical", "axes": AXES}],
                         "datasets": [{"path": "0", "coordinateTransformations": [
                             {"type": "scale", "scale": [1, 1], "input": {"path": "0"},
                              "output": {"name": "physical"}}]}],
                         "coordinateTransformations": [link]})
    assert ms[ngff06.STASH]["label_links"] == {"labels/cells": link}
    assert ngff06.to_06(ms, ["cells"])["coordinateTransformations"] == [link]


# -- collections in 0.6: the 0.5 layout, positions in a scene where 5.18 allows ---------------

def _levels0(path, group):
    ct = _attrs(path, group)["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
    return ct["transformations"][1]["translation"] if ct["type"] == "sequence" else None


def test_a_mosaic_plate_gets_one_scene_on_the_plate(tmp_path):                        # 6.1
    coll = _plate_file(per_well=False)
    out = str(tmp_path / "p.zarr")
    coll.write_plate(out, version="0.6")
    for group in ("", "B/2", "B/2/1"):
        assert _attrs(out, group)["ome"]["version"] == "0.6", group
    scene = _attrs(out)["ome"]["scene"]
    assert [s["name"] for s in scene["coordinateSystems"]] == ["stage"]
    assert "path" not in scene["coordinateSystems"][0]              # never null (9.17)
    edge = scene["coordinateTransformations"][1]
    assert edge["input"] == {"name": "physical", "path": "B/2/1"}
    assert edge["output"] == {"name": "stage"} and edge["type"] == "translation"
    image = _attrs(out, "B/2/1")
    assert image["eubi:coordinate_system"] == "stage" and image["eubi:scene"] == "../../.."
    assert _levels0(out, "B/2/1") in (None, [0.0, 0.0, 0.0])        # alignment only
    assert "axes" not in _attrs(out)["eubi:coordinate_systems"][0]   # the scene holds them
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)
    assert back[next(iter(back))].meta.version == "0.6"


def test_one_system_per_well_gives_each_well_its_scene(tmp_path):                     # 6.6
    coll = _plate_file(per_well=True, fields=(0, 2))
    out = str(tmp_path / "p.zarr")
    coll.write_plate(out, version="0.6")
    assert "scene" not in _attrs(out)["ome"]
    for well in ("B/2", "B/4"):
        scene = _attrs(out, well)["ome"]["scene"]
        assert [s["name"] for s in scene["coordinateSystems"]] == [f"well-B{well[-1]}"]
        assert [e["input"]["path"] for e in scene["coordinateTransformations"]] == ["0", "1"]
    assert _attrs(out, "B/2/1")["eubi:scene"] == ".."
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)
    assert [f.index for f in back.plates[0].wells[0].fields] == [0, 2]


def test_a_series_layout_gets_a_scene_on_its_root(tmp_path):
    images = [_placed({"series": 0}, "stage", value=1), _placed({"series": 1}, "stage", y=50,
                                                                value=2)]
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")]))
    out = str(tmp_path / "s.zarr")
    coll.write_series(out, version="0.6")
    ome = _attrs(out)["ome"]
    assert ome["bioformats2raw.layout"] == 3 and "scene" in ome
    assert _attrs(out, "OME")["ome"]["version"] == "0.6"
    assert os.path.isfile(os.path.join(out, "OME", "METADATA.ome.xml"))   # 9.17 rule 4
    assert _attrs(out, "1")["eubi:scene"] == ".."
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)


def test_tiles_of_a_series_are_placed_by_the_roots_scene(tmp_path):                   # 6.4
    images = [_placed({"series": 0, "tile": t}, "stage", y=10.0 * t, x=-2.0 * t, value=t)
              for t in range(4)]
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")], path="/data/matl.omp2info"))
    out = str(tmp_path / "s.zarr")
    _quiet(lambda: coll.write_series(out, version="0.6"))
    assert len(_attrs(out)["ome"]["scene"]["coordinateTransformations"]) == 4
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)
    assert [n.name for n in back.tree] == ["series0"]


def test_an_unplaced_label_and_preview_keep_the_root_from_being_a_scene(tmp_path):   # 6.2
    images = [_placed({"series": 0}, "stage", value=1), _placed({"series": 1}, "stage", y=50,
                                                                value=2),
              _img({"series": 2}, value=3), _img({"series": 3}, value=4)]
    coll = PyramidCollection.from_file(_file(images, [_cs("stage")]))
    out = str(tmp_path / "s.zarr")
    with pytest.warns(UserWarning, match=r"no scene on 1 group.*outside the component"):
        coll.write_series(out, version="0.6")
    assert "scene" not in _attrs(out)["ome"]
    assert "eubi:scene" not in _attrs(out, "0")
    assert _levels0(out, "1")[-2:] == [50.0, -3.0]                     # the single-image form
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)


def test_mixed_rank_images_stay_in_the_single_image_form(tmp_path):                   # 6.7
    from types import SimpleNamespace
    from test_collection import FakeImage, Q
    zyx = FakeImage(np.zeros((3, 8, 8), "u2"), "zyx", SimpleNamespace(
        pixel_sizes={"z": Q(2.0, UM), "y": Q(0.5, UM), "x": Q(0.5, UM)},
        position={"z": Q(1.0, UM), "y": Q(4.0, UM), "x": Q(5.0, UM)}, channels=[],
        index_values={}, coordinate_system="stage"), index={"series": 0})
    yx = FakeImage(np.zeros((8, 8), "u2"), "yx", SimpleNamespace(
        pixel_sizes={"y": Q(0.5, UM), "x": Q(0.5, UM)}, position={"y": Q(6.0, UM), "x": Q(7.0, UM)},
        channels=[], index_values={}, coordinate_system="stage"), index={"series": 1})
    coll = PyramidCollection.from_file(_file([zyx, yx], [_cs("stage")]))
    out = str(tmp_path / "s.zarr")
    with pytest.warns(UserWarning, match=r"5\.18\(c\)"):
        coll.write_series(out, version="0.6")
    assert "scene" not in _attrs(out)["ome"]
    back = PyramidCollection.from_zarr(out)
    _same_graph(back, coll)


def test_affine_views_are_edges_of_the_roots_scene(tmp_path):                        # 6.3
    from ome_zarr_pyramid.core.spatial import Transformation, matrix
    turned = [[1, 0, 0, 0], [0, 0, -1, 5], [0, 1, 0, 0]]
    images = [_img({"series": 0, "view": v}, value=v) for v in range(2)]
    edges = [Transformation("affine", (("series", 0), ("view", v)), "global",
                            {"affine": turned if v else [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]})
             for v in range(2)]
    coll = PyramidCollection.from_file(_file(images, [_cs("global")], edges))
    out = str(tmp_path / "s.zarr")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coll.write_series(out, version="0.6")
    assert not any("has no place for" in str(w.message) for w in caught)
    scene = _attrs(out)["ome"]["scene"]
    assert [s["name"] for s in scene["coordinateSystems"]] == ["global"]
    assert [e["type"] for e in scene["coordinateTransformations"]] == ["affine", "affine"]
    assert "eubi:coordinate_transformations" not in _attrs(out)
    back = PyramidCollection.from_zarr(out)
    assert list(back) == list(coll)
    for mine, theirs in zip(coll.coordinate_transformations, back.coordinate_transformations):
        assert (theirs.input, theirs.output) == (mine.input, mine.output)
        assert np.allclose(matrix(theirs, 3), matrix(mine, 3))


def test_a_lone_read_of_a_scene_form_image_warns_or_applies_the_scene(tmp_path):
    images = [_placed({"series": 0}, "stage", value=1), _placed({"series": 1}, "stage", y=50,
                                                                value=2)]
    out = str(tmp_path / "s.zarr")
    PyramidCollection.from_file(_file(images, [_cs("stage")])).write_series(out, version="0.6")
    one = os.path.join(out, "1")
    with pytest.warns(UserWarning, match="placed by the NGFF 0.6 scene on .*apply_scene=True"):
        pyr = io.read_pyramid(one)
    assert pyr.meta.get_translation("0")[-2:] == [0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        placed = io.read_pyramid(one, apply_scene=True)
    assert placed.meta.get_translation("0")[-2:] == [50.0, -3.0]
    assert "eubi:scene" not in placed.meta.custom_blocks     # it holds its position now


# -- scenes written by other tools -----------------------------------------------------------

def _scene(group_path, systems, edges):
    g = zarr.open_group(str(group_path), mode="r+")
    ome = dict(g.attrs["ome"])
    ome["scene"] = {"coordinateSystems": systems, "coordinateTransformations": edges}
    g.attrs["ome"] = ome


STAGE = {"name": "stage", "axes": AXES}


def _edge(path, t, *, reverse=False, name="physical"):
    ends = {"input": {"name": name, "path": path}, "output": {"name": "stage"}}
    if reverse:
        ends = {"input": ends["output"], "output": ends["input"]}
    return {**t, **ends}


def _series06(tmp_path, n=2):
    root = tmp_path / "s.zarr"
    g = zarr.open_group(str(root), mode="w", zarr_format=3)
    g.attrs.put({"ome": {"version": "0.6", "bioformats2raw.layout": 3}})
    for i in range(n):
        _image06(root / str(i), DATA + i)
    return root


def test_a_scene_on_a_bioformats2raw_root_places_its_images(tmp_path):
    root = _series06(tmp_path)
    _scene(root, [STAGE], [
        _edge("0", {"type": "translation", "translation": [10.0, 20.0]}),
        _edge("1", {"type": "translation", "translation": [-5.0, 2.0]}, reverse=True)])
    coll = PyramidCollection.from_zarr(str(root))
    assert set(coll.system_of.values()) == {"stage"}
    assert coll.coordinate_systems["stage"].axes is not None
    assert coll[{"series": 0}].meta.get_translation("0") == [10.0, 20.0]
    assert coll[{"series": 1}].meta.get_translation("0") == [5.0, -2.0]   # inverted (3.20)
    assert coll.coordinate_transformations == []


def test_a_scene_edge_across_units_is_a_position_in_the_images_units(tmp_path):
    root = _series06(tmp_path, n=1)
    _scene(root, [{"name": "stage", "axes": [{"name": "y", "type": "space", "unit": "nanometer"},
                                             {"name": "x", "type": "space", "unit": "nanometer"}]}],
           [_edge("0", {"type": "sequence", "transformations": [
               {"type": "scale", "scale": [1000.0, 1000.0]},
               {"type": "translation", "translation": [3000.0, 4000.0]}]})])
    coll = PyramidCollection.from_zarr(str(root))
    assert coll[{"series": 0}].meta.get_translation("0") == [3.0, 4.0]


def test_a_scene_on_a_well_places_its_fields(tmp_path):
    well = tmp_path / "w.zarr"
    g = zarr.open_group(str(well), mode="w", zarr_format=3)
    g.attrs.put({"ome": {"version": "0.6", "well": {"images": [{"path": "0"}, {"path": "1"}]}}})
    for i in range(2):
        _image06(well / str(i), DATA)
    _scene(well, [STAGE], [_edge(str(i), {"type": "translation", "translation": [0.0, 8.0 * i]})
                           for i in range(2)])
    coll = PyramidCollection.from_zarr(str(well))
    assert len(coll.system_of) == 2
    assert [coll[k].meta.get_translation("0") for k in coll] == [[0.0, 0.0], [0.0, 8.0]]


def test_a_scene_on_a_plate_places_its_images_by_their_paths(tmp_path):
    plate = tmp_path / "p.zarr"
    g = zarr.open_group(str(plate), mode="w", zarr_format=3)
    g.attrs.put({"ome": {"version": "0.6", "plate": {
        "name": "P", "rows": [{"name": "A"}], "columns": [{"name": "1"}],
        "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}]}}})
    zarr.open_group(str(plate / "A"), mode="w", zarr_format=3)
    w = zarr.open_group(str(plate / "A" / "1"), mode="w", zarr_format=3)
    w.attrs.put({"ome": {"version": "0.6", "well": {"images": [{"path": "0"}, {"path": "1"}]}}})
    for i in range(2):
        _image06(plate / "A" / "1" / str(i), DATA)
    _scene(plate, [STAGE], [_edge(f"A/1/{i}", {"type": "translation",
                                                "translation": [2.0 * i, 1.0]}) for i in range(2)])
    coll = PyramidCollection.from_zarr(str(plate))
    assert [coll[k].meta.get_translation("0") for k in coll] == [[0.0, 1.0], [2.0, 1.0]]
    assert set(coll.system_of.values()) == {"stage"}


def test_other_scene_edges_join_the_graph(tmp_path):
    root = _series06(tmp_path)
    turn = {"type": "affine", "affine": [[0.0, -1.0, 5.0], [1.0, 0.0, 0.0]]}
    _scene(root, [STAGE], [
        _edge("0", {"type": "translation", "translation": [1.0, 1.0]}),
        {**turn, "input": {"name": "physical", "path": "1"},
         "output": {"name": "physical", "path": "0"}}])
    coll = PyramidCollection.from_zarr(str(root))
    (edge,) = coll.coordinate_transformations
    assert (edge.input, edge.output) == ((("series", 1),), (("series", 0),))
    (c,) = coll.components()
    assert len(c.entries) == 2


def test_a_scene_naming_a_system_that_is_not_the_images_own_is_not_folded(tmp_path):
    root = _series06(tmp_path, n=1)
    _scene(root, [STAGE], [_edge("0", {"type": "translation", "translation": [9.0, 9.0]},
                                 name="array")])
    # its position is then unknown: placed without a translation, it raises I4 (O-2) ...
    coll = PyramidCollection.from_zarr(str(root))
    with pytest.warns(UserWarning, match="not its intrinsic system 'physical'"):
        with pytest.raises(InvariantError, match="add_missing_translation=True"):
            coll[{"series": 0}]
    # ... unless the reader asks to place it at the origin
    coll = PyramidCollection.from_zarr(str(root), add_missing_translation=True)
    with pytest.warns(UserWarning, match="not its intrinsic system 'physical'"):
        pyr = coll[{"series": 0}]
    assert pyr.meta.get_translation("0") == [0.0, 0.0]


def test_a_read_scene_is_written_again_and_applied_once(tmp_path):
    root = _series06(tmp_path)
    _scene(root, [STAGE], [_edge(str(i), {"type": "translation", "translation": [0.0, 8.0 * i]})
                           for i in range(2)])
    coll = PyramidCollection.from_zarr(str(root))
    out = str(tmp_path / "again.zarr")
    _quiet(lambda: coll.write_series(out, version="0.6"))
    scene = _attrs(out)["ome"]["scene"]               # one scene, made from the model
    assert len(scene["coordinateTransformations"]) == 2
    assert _levels0(out, "1") in (None, [0.0, 0.0])
    back = PyramidCollection.from_zarr(out)
    assert [back[k].meta.get_translation("0") for k in back] == [[0.0, 0.0], [0.0, 8.0]]


def test_the_multiscale_type_is_the_downscale_method_or_unknown(tmp_path):     # 5.23
    one = Pyramid().from_arrays([DATA], axis_order="yx", scales=[[0.5, 0.5]], version="0.6")
    made = one.downscale(n_layers=2, downscale_method="mean", defer=False)
    for name, pyr, want in (("one", one, "unknown"), ("made", made, "mean")):
        out = str(tmp_path / f"{name}.zarr")
        IO().write_pyramid(pyr, out)
        (ms,) = _attrs(out)["ome"]["multiscales"]
        assert ms["type"] == want and "metadata" in ms, name
    v05 = str(tmp_path / "v05.zarr")
    IO().write_pyramid(made.to_version("0.5"), v05)              # 0.5 names it too (0.3.1)
    assert _attrs(v05)["ome"]["multiscales"][0]["type"] == "mean"
