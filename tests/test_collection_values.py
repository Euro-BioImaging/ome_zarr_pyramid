"""Storage, values and operators on collections (Workplan C5): rechunk, to_version,
saturate_cast and operators with a number through _map, lazily; == and != elementwise
(decision C-1); two collections do not combine yet (decision C-4)."""
import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import OperationMismatchError, PyramidCollection

from test_api_symmetry import _same_files
from test_collection_zarr import _opened, _plate
from test_query import _mixed


def _level0(pyr):
    return np.asarray(pyr.layers[pyr.meta.resolution_paths[0]])


def test_a_mask_collection_is_lazy_and_writes_as_a_plate(tmp_path):
    path, _ = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(path)
    threshold = 1000                                  # opening an image first would apply it now
    mask = coll > threshold
    assert _opened(mask) == [] and _opened(coll) == []
    assert set(mask.table["dtype"]) == {"bool"} and _opened(mask) == []   # the table: no image
    back = PyramidCollection.from_zarr(mask.write(str(tmp_path / "mask.zarr")))
    assert sorted(back) == sorted(coll) and len(back.plates) == 1
    for key in coll:
        np.testing.assert_array_equal(_level0(back[key]), _level0(coll[key]) > threshold)


def test_operators_with_a_number_give_what_each_image_gives():
    coll = _same_files(2)
    key = next(iter(coll))
    a = _level0(coll[key])
    cases = {"==": (coll == 3, a == 3), "!=": (coll != 3, a != 3), "*": (coll * 2, a * 2),
             "r-": (65535 - coll, 65535 - a), "<": (3 < coll, 3 < a), "-x": (-coll, -a),
             "&": ((coll > 1) & True, (a > 1) & True), "np": (coll + np.uint16(1), a + np.uint16(1))}
    for name, (result, expected) in cases.items():
        assert isinstance(result, PyramidCollection) and sorted(result) == sorted(coll), name
        np.testing.assert_array_equal(_level0(result[key]), expected, err_msg=name)


def test_two_collections_do_not_combine_yet():
    coll = _same_files(2)
    with pytest.raises(TypeError, match=r"do not combine image by image yet \(decision C-4\)"):
        coll - coll
    with pytest.raises(TypeError, match="combines with a number, not a Pyramid"):
        coll + coll[next(iter(coll))]
    # == with anything but a number: the same collection or not (so `in` works)
    assert (coll == coll) is True and (coll == _same_files(2)) is False
    assert coll in [_same_files(2), coll] and (coll != _same_files(2)) is True


def test_storage_and_casts_with_their_rows_from_metadata(tmp_path):
    path, _ = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(path)
    small = coll.saturate_cast("uint8").to_version("0.4").rechunk((1, 4, 4))
    table = small.table
    assert set(table["dtype"]) == {"uint8"} and set(table["version"]) == {"0.4"}
    assert set(table["chunks"]) == {(1, 4, 4)} and _opened(small) == []
    out = small.write(str(tmp_path / "small.zarr"), version="0.4")
    back = PyramidCollection.from_zarr(out)
    key = sorted(back)[0]
    assert back[key].dtype == np.uint8 and back[key].meta.version == "0.4"
    np.testing.assert_array_equal(_level0(back[key]), np.clip(_level0(coll[key]), 0, 255))


def test_a_chunk_shape_of_another_length_is_refused():
    coll = _mixed()                       # series 2 is czyx, the others cyx
    with pytest.raises(OperationMismatchError, match=r"does not fit 1 .*4 axes \('czyx'\), not 3"):
        coll.rechunk((1, 4, 4))
    assert len(coll.filter(axes="cyx").rechunk((1, 4, 4))) == 3


def test_rows_read_in_threads_equal_the_rows_read_one_by_one(tmp_path, monkeypatch):
    import ome_zarr_pyramid.core.collection as C
    from types import SimpleNamespace
    from test_query import Img
    images = [Img({"series": s}) for s in range(12)]       # 12 >= the threshold: threads

    def table(parallel):
        monkeypatch.setattr(C, "_PARALLEL_ROWS", 8 if parallel else 10 ** 9)
        coll = PyramidCollection.from_file(SimpleNamespace(path=None, images=images, plates=[]))
        return coll.isel(x=slice(1, 9, 2), c=[1]).table.rows
    assert table(True) == table(False)


def test_the_table_has_no_stale_shards_after_an_axis_drop(tmp_path):
    from ome_zarr_pyramid import IO, Pyramid
    data = np.zeros((2, 64, 64), "u1")
    out = str(tmp_path / "sharded.zarr")
    IO().write_pyramid(Pyramid().from_arrays([data], axis_order="cyx"), out,
                       chunks=(1, 16, 16), shard_coefficients=(2, 2, 2))
    coll = PyramidCollection.from_zarr(out)
    assert coll.table["shards"] == [(2, 32, 32)]
    assert coll.isel(c=0).table["shards"] == [None]       # was (2, 32, 32) on a 2-D image
    assert coll.isel(x=slice(0, 32)).table["shards"] == [(2, 32, 32)]
