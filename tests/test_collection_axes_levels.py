"""Axis and level operations on collections (Workplan C4): isel, squeeze, select_levels and
downscale through _map, lazily, refused as a whole when they do not fit (C-5), and the
coordinate-system graph rule: dropping an axis is refused for an image an edge runs
through; a system whose images all drop an axis loses it."""
import numpy as np
import pytest

from ome_zarr_pyramid import OperationMismatchError, PyramidCollection
from ome_zarr_pyramid.core.spatial import Axis, CoordinateSystem, Transformation

from test_api_symmetry import _same_files
from test_collection_zarr import _opened, _plate
from test_query import _mixed


def _level0(pyr):
    return np.asarray(pyr.layers[pyr.meta.resolution_paths[0]])


def test_isel_on_a_store_plate_is_lazy_and_writes_what_each_image_gives(tmp_path):
    path, _ = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(path)
    first = coll.isel(c=0)
    assert _opened(first) == [] and _opened(coll) == []
    back = PyramidCollection.from_zarr(first.write(str(tmp_path / "c0.zarr")))
    assert sorted(back) == sorted(coll) and len(back.plates) == 1
    for key in coll:
        expected = coll[key].isel(c=0)
        assert back[key].axes == expected.axes == "yx"
        np.testing.assert_array_equal(_level0(back[key]), _level0(expected))


def test_a_slice_keeps_the_axis_and_the_place():
    coll = _same_files(2)
    part = coll.isel(x=slice(4, 8))
    for key in coll:
        assert part[key].axes == coll[key].axes
        np.testing.assert_array_equal(_level0(part[key]), _level0(coll[key])[..., 4:8])
        x = coll[key].axes.index("x")
        before = coll[key].meta.get_translation("0") or [0.0] * len(coll[key].axes)
        assert part[key].meta.get_translation("0")[x] == pytest.approx(
            before[x] + 4 * coll[key].meta.get_scale("0")[x])


def test_what_does_not_fit_every_image_is_refused_naming_them():
    coll = _mixed()            # series 2: czyx with 3 channels; series 3: 3 levels; others 2 / 1
    with pytest.raises(OperationMismatchError, match=r"does not fit 3 .*no axis 'z'"):
        coll.isel(z=0)
    with pytest.raises(OperationMismatchError, match=r"2 along c, no index 2"):
        coll.isel(c=2)
    with pytest.raises(OperationMismatchError,
                       match=r"1 level\(s\); select_levels\(0, 1, 2\) needs 3.*"
                             r"coll\.filter\(nlevels=lambda n: n >= 3\)"):
        coll.select_levels(0, 1, 2)
    with pytest.raises(OperationMismatchError, match="squeeze drops size-1 axes only"):
        coll.squeeze("c")
    deep = coll.filter(nlevels=lambda n: n >= 3).select_levels(0, 1, 2)
    assert [deep[k].nlevels for k in deep] == [3]
    assert [coll.filter(axes="czyx").isel(z=0)[k].axes for k in coll.filter(axes="czyx")] == ["cyx"]


def test_select_levels_and_downscale_on_every_image(tmp_path):
    coll = _same_files(2)
    planned = coll.downscale(n_layers=3)
    back = PyramidCollection.from_zarr(planned.write(str(tmp_path / "d.zarr")))
    assert [back[k].nlevels for k in back] == [3, 3]
    coarse = back.select_levels(1, 2)
    assert [coarse[k].nlevels for k in coarse] == [2, 2]
    np.testing.assert_array_equal(_level0(coarse[next(iter(coarse))]),
                                  np.asarray(back[next(iter(back))].layers["1"]))


def _placed(edge=False):
    coll = _same_files(2)                                     # cyx images
    keys = list(coll)
    stage = CoordinateSystem("stage", axes=(Axis("c"), Axis("y"), Axis("x")))
    edges = [Transformation("translation", keys[0], keys[1], {"translation": [0.0, 1.0, 2.0]})] \
        if edge else []
    return PyramidCollection({k: coll[k] for k in keys}, coordinate_systems=[stage],
                             system_of={k: "stage" for k in keys},
                             coordinate_transformations=edges), keys


def test_a_system_loses_an_axis_all_its_images_drop():
    coll, keys = _placed()
    flat = coll.isel(c=0)
    assert [a.name for a in flat.coordinate_systems["stage"].axes] == ["y", "x"]
    assert flat.system_of == coll.system_of
    one = PyramidCollection({keys[0]: coll[keys[0]], keys[1]: coll[keys[1]].isel(c=[0])},
                            coordinate_systems=list(coll.coordinate_systems.values()),
                            system_of=dict(coll.system_of))
    squeezed = one.squeeze()                                  # only the second image drops c
    assert [a.name for a in squeezed.coordinate_systems["stage"].axes] == ["c", "y", "x"]


def test_dropping_an_axis_an_edge_runs_through_is_refused():
    coll, keys = _placed(edge=True)
    with pytest.raises(OperationMismatchError,
                       match=r"an edge of the coordinate-system graph .* would lose axis c.*"
                             r"isel\(c=slice\(i, i \+ 1\)\)"):
        coll.isel(c=0)
    kept = coll.isel(c=slice(0, 1))                           # a slice keeps the axis
    assert all(kept[k].axes == "cyx" for k in kept) and len(kept.coordinate_transformations) == 1


def test_isel_of_an_image_with_unitless_axes():
    # idr0010's images: t and z without a unit; every isel raised ("Number of units ...")
    from ome_zarr_pyramid import Pyramid
    pyr = Pyramid().from_arrays([np.zeros((1, 2, 1, 8, 8), "u2")], axis_order="tczyx",
                                unit_list=[None, None, "micrometer", "micrometer"])
    pyr.meta.update_unitlist([None, None, None, "micrometer", "micrometer"])
    assert pyr.meta.unit_list == [None, None, None, "micrometer", "micrometer"]
    for kw, axes in ((dict(c=0), "tzyx"), (dict(t=0), "czyx"), (dict(c=slice(0, 1)), "tczyx")):
        out = pyr.isel(**kw)
        assert out.axes == axes
        assert out.meta.unit_list == [None if a in "tcz" else "micrometer" for a in axes]
