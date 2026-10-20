"""9.4b / 5.2: a plate field holding several images (tiles, views, parts) raises in NGFF
0.4 / 0.5 (and 0.6: the check sits before any version-specific code), naming the ways
out in order: tiles="mosaic", version="rfc8", pyrops. Images outside the plate (a slide
label or preview, keyed without a plate) are not part of it and raise no warning."""
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from ome_zarr_pyramid import PlateFieldError, Pyramid, PyramidCollection
from ome_zarr_pyramid.core.collection import SERIES_INDEX
from ome_zarr_pyramid.core.ngff_plate import write_plate
from ome_zarr_pyramid.core.nodes import (
    CollectionNode, MultiscaleNode, PlateAttribute, PlateItem, Ref, WellAttribute)

from test_plates import PLATE, FakeImage

CZI = os.path.join(os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy")),
                   "formats", "czi")
TILED = os.path.join(CZI, "zenodo-7015307", "W96_B2+B4_S=2_T=1=Z=1_C=1_Tile=5x9.czi")
A1A2 = os.path.join(CZI, "challenge", "w96_A1+A2.czi")


def _file(tiles=2, extra=()):
    """PLATE's wells B3 (field 0 with `tiles` tiles) and A1 (field 0, one image), plus
    `extra` images outside the plate."""
    images, value = [], 0
    for w, n in ((0, tiles), (1, 1)):
        for t in range(n):
            index = {"plate": 0, "acquisition": 0, "well": w, "field": 0}
            if n > 1:
                index["tile"] = t
            images.append(FakeImage(index, value))
            value += 1
    images += [FakeImage(index, 50 + i) for i, index in enumerate(extra)]
    return SimpleNamespace(path="/data/tiled.czi", images=images, plates=[PLATE])


def _ways_out_in_order(message):
    return message.index('tiles="mosaic"') < message.index('version="rfc8"') < message.index("pyrops")


@pytest.mark.parametrize("version", ["0.4", "0.5", None])
def test_a_field_with_two_tiles_raises_before_anything_is_written(tmp_path, version):
    coll = PyramidCollection.from_file(_file())
    out = tmp_path / "p.zarr"
    with pytest.raises(PlateFieldError) as err:
        coll.write_plate(str(out), version=version)
    message = str(err.value)
    assert _ways_out_in_order(message)
    assert "well 'B3', field 0" in message and "holds 2 images" in message
    assert "['tile']" in message and "experimental" in message
    # tiles="mosaic" does not exist in micro-reader 0.0.4: the way out says so
    assert 'tiles="mosaic" (micro-reader >= 0.1)' in message
    assert isinstance(err.value, ValueError) and not out.exists()


def test_rfc8_keeps_every_tile(tmp_path):
    coll = PyramidCollection.from_file(_file(tiles=3))
    written = coll.write_plate(str(tmp_path / "p.zarr"), version="rfc8")
    assert len(written) == 4                       # three tiles in B3, one image in A1


def test_one_image_per_field_writes_without_a_warning(tmp_path):
    coll = PyramidCollection.from_file(_file(tiles=1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        written = coll.write_plate(str(tmp_path / "p.zarr"))
    assert sorted(written) == ["A/1/0", "B/3/0"]
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_images_outside_the_plate_are_left_out_without_a_warning(tmp_path):
    # a CZI slide label / preview is keyed by series only: it is no plate image, so it
    # neither warns (the old "also indexed by" warning counted it as plate 0) nor raises
    coll = PyramidCollection.from_file(_file(tiles=1, extra=[{SERIES_INDEX: 0}]))
    assert len(coll) == 3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        written = coll.write_plate(str(tmp_path / "p.zarr"))
    assert sorted(written) == ["A/1/0", "B/3/0"]
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def _pyr(value, index=None):
    pyr = Pyramid().from_arrays([np.full((2, 8, 12), value, "u2")], axis_order="cyx")
    if index is not None:
        pyr.source_index = index
    return pyr


def _hand_built(*wells):
    attr = PlateAttribute(rows=[PlateItem("A", "A")], columns=[PlateItem("c1", "1"), PlateItem("c2", "2")])
    plate = CollectionNode("hand", attributes={"plate": attr})
    for c, images in enumerate(wells):
        well = plate.add(CollectionNode(f"A{c + 1}", attributes={"well": WellAttribute(
            Ref("A"), Ref(f"c{c + 1}"))}))
        for i, image in enumerate(images):
            well.add(image if isinstance(image, MultiscaleNode) else MultiscaleNode(f"i{i}", image))
    return plate


def test_a_hand_built_well_with_two_images_of_one_field_raises(tmp_path):
    field = {"plate": 0, "well": 0, "field": 0}
    plate = _hand_built([_pyr(1, {**field, "tile": 0}), _pyr(2, {**field, "tile": 1})], [_pyr(3)])
    for version in ("0.4", "0.5"):
        with pytest.raises(PlateFieldError) as err:
            write_plate(plate, str(tmp_path / f"p{version}.zarr"), version=version)
        assert _ways_out_in_order(str(err.value)) and "well 'A1', field 0" in str(err.value)
    # the same through an eubi:index attribute on the image nodes
    nodes = [MultiscaleNode(f"t{t}", _pyr(t), attributes={"eubi:index": {**field, "tile": t}})
             for t in (0, 1)]
    with pytest.raises(PlateFieldError):
        write_plate(_hand_built(nodes, [_pyr(3)]), str(tmp_path / "attr.zarr"))


def test_a_hand_built_well_without_field_numbers_writes_each_image_as_a_field(tmp_path):
    plate = _hand_built([_pyr(1), _pyr(2)], [_pyr(3)])
    written = write_plate(plate, str(tmp_path / "p.zarr"))
    assert sorted(written) == ["A/1/0", "A/1/1", "A/2/0"]
    # images of different fields, or the same index twice (two sources), are no crowd
    plate = _hand_built([_pyr(1, {"well": 0, "field": 0}), _pyr(2, {"well": 0, "field": 1}),
                         _pyr(4, {"well": 0, "field": 1})], [_pyr(3)])
    assert len(write_plate(plate, str(tmp_path / "q.zarr"))) == 4


# -- real files (skipped when absent) ------------------------------------------------

def _micro_reader():
    micro_reader = pytest.importorskip("micro_reader")
    from importlib.metadata import version
    # micro-reader 0.0.x calls the mosaic policy "stitched"
    return micro_reader, ("stitched" if version("micro-reader").startswith("0.0.") else "mosaic")


@pytest.mark.skipif(not os.path.isfile(TILED), reason="the tiled CZI plate is not here")
def test_a_tiled_czi_plate_raises_and_writes_as_mosaic(tmp_path):
    micro_reader, mosaic = _micro_reader()
    coll = PyramidCollection.from_file(micro_reader.open(TILED, tiles="separate", plates=True))
    for version in ("0.4", "0.5"):
        with pytest.raises(PlateFieldError) as err:
            coll.write_plate(str(tmp_path / f"sep{version}.zarr"), version=version)
        assert _ways_out_in_order(str(err.value)) and "holds 45 images" in str(err.value)
    coll = PyramidCollection.from_file(micro_reader.open(TILED, tiles=mosaic, plates=True))
    out = str(tmp_path / "mosaic.zarr")
    assert sorted(coll.write_plate(out, version="0.5")) == ["B/2/0", "B/4/0"]
    back = PyramidCollection.from_zarr(out)
    assert sum(len(f.keys) for w in back.plates[0].wells for f in w.fields) == 2


@pytest.mark.skipif(not os.path.isfile(A1A2), reason="the w96_A1+A2 CZI plate is not here")
def test_a_czi_plate_with_one_image_per_field_writes_without_a_warning(tmp_path):
    micro_reader, _ = _micro_reader()
    coll = PyramidCollection.from_file(micro_reader.open(A1A2, plates=True))
    out = str(tmp_path / "a1a2.zarr")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        written = coll.write_plate(out)
    assert sorted(written) == ["A/1/0", "A/2/0"]
    assert os.path.isdir(os.path.join(out, "A", "1", "0")) and os.path.isdir(os.path.join(out, "A", "2", "0"))
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
