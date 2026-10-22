"""Decisions O-2 and O-5 (Spatial Model Workplan, 2026-10-21).

O-2: a pyramid placed in a system without a level-0 translation raises I4, naming
``add_missing_translation=True``, which places it at the system's origin instead (on a
copy: the pyramid given is not changed). O-5: a well opened on its own is keyed and placed
as in its plate, from its ``eubi:images`` and the systems on it."""
import os
import warnings

import numpy as np
import pytest

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection
from ome_zarr_pyramid.core.spatial import CoordinateSystem, InvariantError, stamp_system

from test_eubi_keys import _plate_file, _positions

KEY = (("series", 0),)


def _unplaced_level0():
    pyr = Pyramid().from_arrays([np.zeros((8, 8), "u2")], axis_order="yx", scales=[[0.5, 0.5]])
    stamp_system(pyr, CoordinateSystem("stage"))
    assert pyr.meta.get_translation("0") is None
    return pyr


def test_a_placed_pyramid_without_a_translation_raises_naming_the_way_out():
    with pytest.raises(InvariantError, match=r"I4.*add_missing_translation=True"):
        PyramidCollection({KEY: _unplaced_level0()})


def test_add_missing_translation_places_it_at_the_origin_on_a_copy():
    pyr = _unplaced_level0()
    coll = PyramidCollection({KEY: pyr}, add_missing_translation=True)
    assert coll.system_of == {KEY: "stage"}
    assert coll[KEY].meta.get_translation("0") == [0.0, 0.0]
    assert pyr.meta.get_translation("0") is None              # the given pyramid is unchanged


def test_a_store_read_raises_when_the_image_opens_unless_asked(tmp_path):
    path = str(tmp_path / "img.zarr")
    IO().write_pyramid(_unplaced_level0(), path)
    coll = PyramidCollection.from_zarr(path)                  # nothing is opened yet
    assert coll.system_of == {(): "stage"}
    with pytest.raises(InvariantError, match=r"add_missing_translation=True"):
        coll[()]
    placed = PyramidCollection.from_zarr(path, add_missing_translation=True)
    assert placed[()].meta.get_translation("0") == [0.0, 0.0]


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_a_well_read_alone_keeps_its_plate_keys_and_systems(tmp_path, version):
    coll = _plate_file(per_well=True, fields=(0, 2))           # one system per well
    out = str(tmp_path / "p.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coll.write_plate(out, version=version)
    well = PyramidCollection.from_zarr(os.path.join(out, "B", "2"))
    mine = [k for k in coll if dict(k)["well"] == 0]
    assert list(well) == mine                                  # field numbers 0, 2 kept
    assert well.system_of == {k: "well-B2" for k in mine}
    assert well.coordinate_systems["well-B2"].axes is not None
    assert _positions(well) == {k: v for k, v in _positions(coll).items() if k in mine}


def test_a_well_of_a_plate_wide_system_names_it_and_warns_for_its_definition(tmp_path):
    coll = _plate_file(per_well=False)                         # "stage" is on the plate
    out = str(tmp_path / "p.zarr")
    coll.write_plate(out)
    with pytest.warns(UserWarning, match=r"\['stage'\] are named without their definition"):
        well = PyramidCollection.from_zarr(os.path.join(out, "B", "4"))
    mine = [k for k in coll if dict(k)["well"] == 1]
    assert list(well) == mine and set(well.system_of.values()) == {"stage"}
    assert _positions(well) == {k: v for k, v in _positions(coll).items() if k in mine}


def test_write_warns_that_it_leaves_the_other_edges_out(tmp_path):              # O-4
    from test_spatial_collection import _cs, _file, _img
    from ome_zarr_pyramid.core.spatial import Transformation
    images = [_img({"series": 0, "view": v}, value=v) for v in range(2)]
    edges = [Transformation("affine", (("series", 0), ("view", v)), "global",
                            {"affine": [[1, 0, 0, v], [0, 1, 0, 0], [0, 0, 1, 0]]}) for v in range(2)]
    coll = PyramidCollection.from_file(_file(images, [_cs("global")], edges))
    with pytest.warns(UserWarning, match=r"2 edge\(s\) .*\(affine\) are not written"):
        paths = coll.write(str(tmp_path), "views")
    assert len(paths) == 2
