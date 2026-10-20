"""9.4b / 5.3 / 8.9: a series holding several images (tiles, views, ...) is written in the
bioformats2raw layout (0.4 / 0.5) as further series, one per image, in key order, with
ONE UserWarning per write_series call; one image per series writes without one."""
import os
import warnings

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.core.collection import SERIES_INDEX


def _pyr(value):
    return Pyramid().from_arrays([np.full((2, 8, 12), value, "u2")], axis_order="cyx",
                                 name=f"image {value}")


def _user_warnings(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    return result, [w for w in caught if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_tiles_of_a_series_become_further_series_with_one_warning(tmp_path, version):
    entries = {((SERIES_INDEX, 0), ("tile", t)): _pyr(t) for t in range(3)}
    entries[((SERIES_INDEX, 1),)] = _pyr(9)
    out = str(tmp_path / "s.zarr")
    written, caught = _user_warnings(
        lambda: PyramidCollection(entries).write_series(out, version=version))
    assert sorted(written) == ["0", "1", "2", "3"]
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "1 of the collection's series holds several images" in message
    assert "['tile']" in message and "key order (4 series)" in message
    back = PyramidCollection.from_zarr(out)
    values = [int(np.asarray(back[{SERIES_INDEX: i}].layers["0"])[0, 0, 0]) for i in range(4)]
    assert values == [0, 1, 2, 9]                       # key order: series 0's tiles, then 1


def test_one_image_per_series_writes_without_a_warning(tmp_path):
    entries = {((SERIES_INDEX, s),): _pyr(s) for s in range(3)}
    written, caught = _user_warnings(
        lambda: PyramidCollection(entries).write_series(str(tmp_path / "s.zarr")))
    assert sorted(written) == ["0", "1", "2"] and caught == []


def test_every_varying_sub_index_is_named(tmp_path):
    entries = {((SERIES_INDEX, s), ("view", v), ("illumination", i)): _pyr(10 * s + 2 * v + i)
               for s in (0, 1) for v in (0, 1) for i in (0, 1)}
    _, caught = _user_warnings(
        lambda: PyramidCollection(entries).write_series(str(tmp_path / "s.zarr")))
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "2 of the collection's series hold several images" in message
    assert "['illumination', 'view']" in message and "(8 series)" in message


def test_a_layout_read_and_written_whole_does_not_warn(tmp_path):
    entries = {((SERIES_INDEX, s),): _pyr(s) for s in range(2)}
    src = str(tmp_path / "src.zarr")
    PyramidCollection(entries).write_series(src, version="0.5")
    coll = PyramidCollection.from_zarr(src)
    _, caught = _user_warnings(lambda: coll.write_series(str(tmp_path / "copy.zarr")))
    assert caught == []
    assert os.path.isfile(os.path.join(tmp_path, "copy.zarr", "OME", "METADATA.ome.xml"))
    assert zarr.open_group(str(tmp_path / "copy.zarr"), mode="r").attrs["ome"]["bioformats2raw.layout"] == 3
