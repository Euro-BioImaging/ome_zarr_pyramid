"""Label images follow their image's geometry (decision C-2, Workplan C7): attached labels
and a collection's RFC 8 label nodes go through isel, squeeze, select_levels and downscale
(on the axes they have, never the channel axis; downscaled with "simple", never averaged),
stay as they are under shape-preserving operations, and a label that cannot follow refuses
the operation (they were dropped silently)."""
import os

import numpy as np
import pytest

from ome_zarr_pyramid import IO, LabelMismatchError, Pyramid, PyramidCollection


def _image(levels=2):
    data = np.arange(2 * 4 * 32 * 32, dtype="u2").reshape(2, 4, 32, 32)
    return Pyramid().from_arrays([data], axis_order="czyx").downscale(n_layers=levels, defer=False)


def _label(shape=(4, 32, 32), levels=2):
    values = (np.indices(shape).sum(axis=0) % 3 + 1).astype("u4")     # 1, 2, 3 stripes
    return Pyramid().from_label_arrays([values], axis_order="zyx").downscale(
        n_layers=levels, defer=False)


def _px(pyr, level=None):
    return np.asarray(pyr.layers[level or pyr.meta.resolution_paths[0]])


@pytest.mark.parametrize("op, on_label", [
    (lambda p: p.isel(z=slice(0, 2)), lambda l: l.isel(z=slice(0, 2))),
    (lambda p: p.isel(z=1, x=slice(4, 20, 2)), lambda l: l.isel(z=1, x=slice(4, 20, 2))),
    (lambda p: p.isel(c=0), lambda l: l),                      # never the channel axis
    (lambda p: p[:, 1:3], lambda l: l.isel(z=slice(1, 3))),
    (lambda p: p.select_levels(1), lambda l: l.select_levels(1)),
    (lambda p: p.rechunk((1, 2, 16, 16)), lambda l: l),        # the same grid: as it is
    (lambda p: p * 2, lambda l: l),
])
def test_an_attached_label_follows_its_image(op, on_label):
    pyr = _image().add_image_label(_label(), "cells")
    out = op(pyr)
    assert list(out.labels) == ["cells"]
    expected = on_label(_label())
    got = out.labels["cells"]
    assert got.axes == expected.axes and got.nlayers == expected.nlayers
    for path in got.meta.resolution_paths:
        np.testing.assert_array_equal(_px(got, path), _px(expected, path))


def test_a_label_is_downscaled_by_picking_not_averaging():
    pyr = _image(levels=1).add_image_label(_label(levels=1), "cells")
    out = pyr.downscale(n_layers=3, downscale_method="mean", defer=False)
    label = out.labels["cells"]
    assert label.nlayers == out.nlayers == 3
    for path in label.meta.resolution_paths:
        assert set(np.unique(_px(label, path))) <= {1, 2, 3}      # no mean of 1 and 2
    deferred = pyr.downscale(n_layers=3)                          # a plan: the label too
    assert getattr(deferred.labels["cells"], "_downscale_plan")["downscale_method"] == "simple"


def test_squeeze_drops_the_same_axes():
    image = Pyramid().from_arrays([np.zeros((2, 4, 8, 8), "u2")], axis_order="czyx")
    label = Pyramid().from_label_arrays([np.ones((1, 8, 8), "u4")], axis_order="zyx")
    out = image.isel(z=[0]).add_image_label(label, "cells").squeeze()
    assert out.axes == "cyx" and out.labels["cells"].axes == "yx"


def test_a_label_that_cannot_follow_refuses_the_operation():
    half = _image().add_image_label(_label(shape=(4, 16, 16)), "half")
    with pytest.raises(LabelMismatchError, match=r"'half' cannot follow its image \(16 along x, the image 32\)"):
        half.isel(x=slice(0, 8))
    half.isel(z=slice(0, 2))                                      # z sizes agree: fine
    short = _image(levels=2).add_image_label(_label(levels=1), "short")
    with pytest.raises(LabelMismatchError, match=r"1 level\(s\), no level 1"):
        short.select_levels(1)
    with pytest.raises(LabelMismatchError, match="16 along y"):
        _image(levels=1).add_image_label(_label(shape=(4, 16, 32), levels=1), "y").downscale(n_layers=2)
    assert issubclass(LabelMismatchError, ValueError)


def test_written_labels_match_their_image(tmp_path):
    out = str(tmp_path / "img.ome.zarr")
    _image().add_image_label(_label(), "cells").isel(z=slice(1, 3)).write(out)
    back = Pyramid.from_zarr(out)
    assert _px(back).shape[1:] == _px(back.labels["cells"]).shape
    np.testing.assert_array_equal(_px(back.labels["cells"]), _px(_label().isel(z=slice(1, 3))))


# -- a collection's RFC 8 label nodes: the same rule -----------------------------------

def _rfc8_plate(tmp_path, label_shape=None):
    """test_rfc8's labelled plate written as RFC 8 and read back: its label image is a
    node of the tree. `label_shape` replaces the label (a misfit)."""
    from test_rfc8 import _labelled_plate
    coll, _ = _labelled_plate(tmp_path)
    if label_shape is not None:
        field = coll.plates[0].node["A2"]["0"]
        misfit = Pyramid().from_label_arrays([np.ones(label_shape, "u1")], axis_order="yx",
                                             version="0.5", name="nuclei")
        field.pyramid._labels = {"nuclei": misfit}
    return PyramidCollection.from_zarr(coll.write_collection(str(tmp_path / "c.zarr")))


def _label_nodes(coll):
    from ome_zarr_pyramid.core.nodes import MultiscaleNode
    return [n for n in coll.tree.walk() if isinstance(n, MultiscaleNode) and "labels" in n.attributes]


def test_rfc8_label_nodes_follow_and_still_match_when_written(tmp_path):
    coll = _rfc8_plate(tmp_path)
    (node,) = _label_nodes(coll)
    before = _px(node.pyramid)
    part = coll.isel(x=slice(0, 8))
    (followed,) = _label_nodes(part)
    np.testing.assert_array_equal(_px(followed.pyramid), before[..., 0:8])
    back = PyramidCollection.from_zarr(part.write(str(tmp_path / "part.zarr"), version="rfc8"))
    (written,) = _label_nodes(back)
    image = next(n for n in back.tree.walk() if getattr(n, "id", None) ==
                 written.attributes["labels"].source[0].id)
    assert _px(written.pyramid).shape == _px(image.pyramid).shape[1:]
    assert _px(written.pyramid).shape[-1] == 8
    assert _px(node.pyramid).shape == before.shape                  # the source is unchanged


def test_an_rfc8_label_that_cannot_follow_refuses_naming_the_image(tmp_path):
    # the image is 24 pixels wide at 0.5: a label 10 wide at 1 covers another area (12 at 1
    # would cover the same, at half the resolution, and follow: decision C-18)
    coll = _rfc8_plate(tmp_path, label_shape=(16, 10))
    with pytest.raises(LabelMismatchError, match=r"image \{.*'well': 0.*\}: isel: .*10 along x"):
        coll.isel(x=slice(0, 4))
    coll.isel(y=slice(0, 4))                                       # y sizes agree: fine


def test_an_rfc8_plate_read_back_writes_again_as_it_was(tmp_path):
    # a read RFC 8 document's image paths are relative to it ("./A/2/0" in an inline
    # well), not to the well: they were joined twice ("A/2/A/2/0") and refused
    coll = _rfc8_plate(tmp_path)
    again = PyramidCollection.from_zarr(coll.write(str(tmp_path / "again.zarr"), version="rfc8"))
    assert sorted(again) == sorted(coll)
    assert (tmp_path / "again.zarr" / "A" / "2" / "0" / "labels" / "nuclei").is_dir()
    assert not (tmp_path / "again.zarr" / "A" / "2" / "A").exists()
    (node,) = _label_nodes(again)
    np.testing.assert_array_equal(_px(node.pyramid), _px(_label_nodes(coll)[0].pyramid))
