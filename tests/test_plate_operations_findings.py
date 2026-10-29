"""What the operations on real plates (Workplan C8, spatial-model/tools/plate_operations.py)
found, each on a small stand-in for the plate it showed up on."""
import os

import numpy as np
import pytest

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection
from ome_zarr_pyramid.core.nodes import CollectionNode, MultiscaleNode

from test_collection_zarr import _plate


def _shapes(pyr):
    return [tuple(pyr.layers[p].shape) for p in pyr.meta.resolution_paths]


def test_an_image_with_unitless_axes_rechunks_downscales_and_writes(tmp_path):
    # idr0010: t and z without a unit; rechunk and the plan's writer dropped the Nones
    # ("Number of units must match number of axes except channel")
    pyr = Pyramid().from_arrays([np.arange(2 * 3 * 64 * 64, dtype="u2").reshape(1, 2, 3, 64, 64)],
                                axis_order="tczyx", unit_list=[None, None, "micrometer", "micrometer"])
    pyr.meta.update_unitlist([None, None, None, "micrometer", "micrometer"])
    assert pyr.rechunk().meta.unit_list == pyr.meta.unit_list
    out = str(tmp_path / "img.ome.zarr")
    pyr.downscale(n_layers=2, downscale_method="mean").write(out)
    back = Pyramid.from_zarr(out)
    assert back.nlevels == 2 and back.meta.unit_list == [None, None, None, "micrometer", "micrometer"]


def _floor_sized(odd=35):
    """An image (czyx) and a label (zyx) whose level 1 is floor-sized (35 -> 17), as
    Fractal writes them."""
    data = np.arange(2 * odd * odd, dtype="u2").reshape(2, 1, odd, odd)
    half = data[..., : odd - 1, : odd - 1].reshape(2, 1, odd // 2, 2, odd // 2, 2).mean(axis=(3, 5))
    image = Pyramid().from_arrays([data, half.astype("u2")], axis_order="czyx",
                                  scales=[[1, 1, 1, 1], [1, 1, 2, 2]])
    cells = (np.indices((1, odd, odd)).sum(axis=0) % 3 + 1).astype("u4")
    label = Pyramid().from_label_arrays([cells, cells[:, : odd - 1 : 2, : odd - 1 : 2]],
                                        axis_order="zyx", scales=[[1, 1, 1], [1, 2, 2]])
    return image.add_image_label(label, "cells")


def test_a_deferred_downscale_shows_the_levels_it_writes(tmp_path):
    # Fractal: a plan derived from floor-sized levels (535 -> 267) reproduces them on write,
    # but the levels it showed were ceil-sized (268): the object and its store disagreed
    planned = _floor_sized().downscale(n_layers=3, downscale_method="mean")
    shown = _shapes(planned)
    assert shown[1] == (2, 1, 17, 17)
    out = str(tmp_path / "img.ome.zarr")
    planned.write(out)
    back = Pyramid.from_zarr(out)
    assert _shapes(back) == shown
    for p, q in zip(planned.meta.resolution_paths, back.meta.resolution_paths):
        np.testing.assert_array_equal(np.asarray(planned.layers[p]), np.asarray(back.layers[q]))


def test_a_label_follows_the_plan_on_its_own_axes(tmp_path):
    # the label (zyx) was given the image's per-axis factor (czyx) when the writer expanded
    # the plan ("operands could not be broadcast together with shapes (3,) (4,)"), and its
    # levels were not cut to the image's floor-sized ones
    planned = _floor_sized().downscale(n_layers=3, downscale_method="mean")
    label = planned.labels["cells"]
    assert [s[1:] for s in _shapes(planned)] == _shapes(label)
    out = str(tmp_path / "img.ome.zarr")
    planned.write(out)
    back = Pyramid.from_zarr(out)
    written = back.labels["cells"]
    assert [s[1:] for s in _shapes(back)] == _shapes(written)
    for p, q in zip(label.meta.resolution_paths, written.meta.resolution_paths):
        values = np.asarray(written.layers[q])
        np.testing.assert_array_equal(np.asarray(label.layers[p]), values)
        assert set(np.unique(values)) <= {1, 2, 3}            # picked, never averaged


def _plate_and_overview(tmp_path):
    """The test plate with one image beside it, as a CZI's slide overview is."""
    src, _ = _plate(tmp_path, "0.5")
    plate_node = PyramidCollection.from_zarr(src).plates[0].node
    overview = Pyramid().from_arrays([np.ones((1, 32, 48), "u2")], axis_order="cyx")
    tree = CollectionNode("w96", [plate_node, MultiscaleNode("overview", overview)])
    return PyramidCollection._from_tree(tree)


def test_a_plate_with_an_image_beside_it(tmp_path):
    coll = _plate_and_overview(tmp_path)
    assert (("series", 0),) in coll and len(coll.plates) == 1
    # NGFF holds the plate alone: refused, not written as a series (the plate was lost)
    with pytest.raises(ValueError, match=r"a plate and 1 image\(s\) outside it.*plates\[0\]"
                                         r"\.as_collection\(\)"):
        coll.write(str(tmp_path / "x.zarr"), version="0.5")
    assert not os.path.exists(tmp_path / "x.zarr")
    plate = IO().read(coll.plates[0].as_collection().write(str(tmp_path / "p.zarr"), version="0.5"))
    assert len(plate.plates) == 1 and len(plate) == len(coll) - 1
    overview = IO().read(coll.select(series=0).write(str(tmp_path / "o.zarr"), version="0.5"))
    assert len(overview) == 1
    # RFC 8 keeps both, and reads the image beside the plate back (it was written, not read)
    back = IO().read(coll.write(str(tmp_path / "r.zarr"), version="rfc8"))
    assert sorted(back) == sorted(coll)
    np.testing.assert_array_equal(np.asarray(back[(("series", 0),)].layers["0"]),
                                  np.ones((1, 32, 48), "u2"))
