"""A lone Pyramid keeps its coordinate system (plan 3.26, 8.16, 8.25): the custom
metadata eubi:coordinate_system / eubi:coordinate_systems, written and read back;
coll.pyramid(key) stamps a clone; a collection built from pyramids lifts their systems,
merging ids only within one source; concatenate keeps one system."""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection, concatenate
from ome_zarr_pyramid.core.spatial import (
    SYSTEM_KEY, SYSTEMS_KEY, Axis, CoordinateSystem, InvariantError, pyramid_system,
    system_to_json)

from test_collection import Q
from test_spatial_collection import _cs, _file, _img, _placed

UM = "micrometer"
CYX = (Axis("c"), Axis("y", unit=UM), Axis("x", unit=UM))


def test_from_image_keeps_a_minimal_definition_of_the_images_system():
    pyr = Pyramid.from_image(_placed({"series": 0}, "stage"))
    assert pyramid_system(pyr) == ("stage", CoordinateSystem("stage", axes=CYX))
    assert pyr.custom_metadata[SYSTEM_KEY] == "stage"
    # an image that names no system is unplaced: no blocks, its translation still written
    alone = Pyramid.from_image(_placed({"series": 0}, None))
    assert pyramid_system(alone) == (None, None) and SYSTEM_KEY not in alone.custom_metadata
    assert alone.meta.get_translation("0")[-2:] == [10.0, -3.0]


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_the_system_is_written_and_read_back(tmp_path, version):
    pyr = Pyramid.from_image(_placed({"series": 0}, "stage"), version=version)
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(pyr, path, overwrite=True)
    attrs = zarr.open_group(path, mode="r").attrs.asdict()
    # beside "ome" (0.5) or beside "multiscales" (0.4): a custom block, not NGFF's
    assert attrs[SYSTEM_KEY] == "stage"
    assert attrs[SYSTEMS_KEY] == [system_to_json(CoordinateSystem("stage", axes=CYX))]
    assert SYSTEM_KEY not in attrs.get("ome", {})
    back = IO().read_pyramid(path)
    assert pyramid_system(back) == pyramid_system(pyr)


def test_an_id_without_its_definition_gets_unknown_axes_and_a_warning():
    pyr = Pyramid.from_image(_placed({"series": 0}, None))
    pyr.meta.set_custom(SYSTEM_KEY, "stage")
    with pytest.warns(UserWarning, match="without its definition"):
        assert pyramid_system(pyr) == ("stage", CoordinateSystem("stage"))


# -- taking an entry out, putting it back (8.25) ----------------------------------------

def _stage_file(path="/data/sample.czi"):
    yx = (Axis("y", unit=UM), Axis("x", unit=UM))
    cs = SimpleNamespace(id="stage", name="the stage", axes=CYX, estimated=False, extra={})
    images = [_placed({"series": s}, "stage", y=10.0 * s, value=s) for s in range(2)]
    images.append(_img({"series": 2}, value=9))                          # unplaced
    return _file(images, [cs], path=path), yx


def test_coll_pyramid_returns_a_clone_stamped_with_the_files_definition():
    file, _ = _stage_file()
    coll = PyramidCollection.from_file(file)
    out = coll.pyramid({"series": 1})
    assert pyramid_system(out) == ("stage", coll.coordinate_systems["stage"])
    assert coll.coordinate_systems["stage"].name == "the stage"
    assert out.meta.get_translation("0")[-2:] == [10.0, -3.0]
    assert out.source_index == {"series": 1}
    # the shared entry is never changed
    assert SYSTEM_KEY not in coll[{"series": 1}].custom_metadata
    assert pyramid_system(coll.pyramid({"series": 2})) == (None, None)


def test_entries_taken_out_of_one_set_go_back_into_one_system():
    coll = PyramidCollection.from_file(_stage_file()[0])
    back = PyramidCollection({k: coll.pyramid(k) for k in coll})
    assert back.coordinate_systems == coll.coordinate_systems
    assert back.system_of == coll.system_of
    # a selection is the same source
    part = coll.select(series=0)
    mixed = PyramidCollection({(("a", 0),): part.pyramid({"series": 0}),
                               (("a", 1),): coll.pyramid({"series": 1})})
    assert set(mixed.system_of.values()) == {"stage"}


def test_one_id_from_two_sources_is_two_systems_unless_joined():
    a = PyramidCollection.from_file(_stage_file("/data/left.czi")[0])
    b = PyramidCollection.from_file(_stage_file("/data/right.czi")[0])
    both = {(("side", 0),): a.pyramid({"series": 0}), (("side", 1),): b.pyramid({"series": 0})}
    apart = PyramidCollection(both)
    assert apart.system_of == {(("side", 0),): "stage", (("side", 1),): "right--stage"}
    assert apart.coordinate_systems["right--stage"].name == "the stage"
    joined = PyramidCollection(both, join_systems=True)
    assert set(joined.system_of.values()) == {"stage"} and list(joined.coordinate_systems) == ["stage"]


def test_joining_merges_minimal_definitions_and_refuses_disagreeing_axes():
    two_d = Pyramid.from_image(_placed({"series": 0}, "stage"))
    three_d = _placed({"series": 1}, "stage")
    three_d = Pyramid.from_image(type(three_d)(np.zeros((2, 3, 8, 8), "u2"), "czyx",
                                               three_d.metadata, index={"series": 1}))
    coll = PyramidCollection({(("i", 0),): two_d, (("i", 1),): three_d}, join_systems=True)
    assert [a.name for a in coll.coordinate_systems["stage"].axes] == ["c", "z", "y", "x"]
    other = Pyramid.from_image(_placed({"series": 2}, "stage"))
    other.meta.set_custom(SYSTEMS_KEY, [system_to_json(CoordinateSystem(
        "stage", axes=(Axis("c"), Axis("y", unit="nanometer"), Axis("x", unit=UM))))])
    with pytest.raises(InvariantError, match="axis 'y'"):
        PyramidCollection({(("i", 0),): two_d, (("i", 2),): other}, join_systems=True)


def test_two_reads_of_one_store_are_one_source(tmp_path):
    path = str(tmp_path / "plate1.ome.zarr")
    IO().write_pyramid(Pyramid.from_image(_placed({"series": 0}, "stage")), path, overwrite=True)
    other = str(tmp_path / "plate2.ome.zarr")
    IO().write_pyramid(Pyramid.from_image(_placed({"series": 0}, "stage")), other, overwrite=True)
    same = PyramidCollection({(("r", 0),): IO().read_pyramid(path),
                              (("r", 1),): IO().read_pyramid(path)})
    assert set(same.system_of.values()) == {"stage"}
    apart = PyramidCollection({(("r", 0),): IO().read_pyramid(path),
                               (("r", 1),): IO().read_pyramid(other)})
    assert apart.system_of[(("r", 1),)] == "plate2--stage"


def test_a_file_without_definitions_gives_minimal_ones():
    images = [_placed({"series": 0}, "stage"),
              _placed({"series": 1}, "stage", value=1)]
    coll = PyramidCollection.from_file(SimpleNamespace(path="/x.czi", images=images))
    assert coll.coordinate_systems == {"stage": CoordinateSystem("stage", axes=CYX)}
    assert set(coll.system_of.values()) == {"stage"}


# -- concatenate keeps one system ---------------------------------------------------------

def test_concatenate_keeps_the_system_and_refuses_two():
    a = Pyramid.from_image(_placed({"series": 0}, "stage", value=1))
    b = Pyramid.from_image(_placed({"series": 0}, "stage", value=2))
    joined = concatenate([a, b], axis="c")
    assert pyramid_system(joined) == pyramid_system(a)
    slide = Pyramid.from_image(_placed({"series": 0}, "slide", value=3))
    with pytest.raises(ValueError, match="coordinate system 'slide'"):
        concatenate([a, slide], axis="c")
    unplaced = Pyramid.from_image(_placed({"series": 0}, None, value=4))
    with pytest.raises(ValueError, match="None: unplaced"):
        concatenate([a, unplaced], axis="c")
