"""A collection as a table: rows from metadata only (from opened pyramids, or straight from
the metadata files of images not opened yet, which must give the same row), and the
SQL-style queries over it: filter, unique, count, group_by, sort_by."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection
from ome_zarr_pyramid.core.query import IMAGE_COLUMNS, row_from_pyramid

from test_collection_zarr import _opened, _plate
from test_plates import PLATE, FakeImage

PLATES = os.path.join(os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy")),
                      "plates", "ome-zarr")
IDR0010 = os.path.join(PLATES, "idr0010", "idr0010_76-45.ome.zarr")


class Img:
    """An image with micro-reader's names, of any dtype, shape and level count."""

    def __init__(self, index, shape=(2, 8, 12), dtype="u2", axes="cyx", levels=1):
        self._data = np.zeros(shape, dtype=dtype)
        self.axes, self.shape, self.dtype, self.index = axes, self._data.shape, self._data.dtype, index
        self.name, self.metadata = f"img{index}", None
        self.resolutions = [self] + [Img(index, tuple(s if a == "c" else s // 2 ** k
                                                      for a, s in zip(axes, shape)), dtype, axes)
                                     for k in range(1, levels)]

    def __getitem__(self, key):
        return self._data[key]


def _mixed():
    """Four series that differ: dtype, axes, level count."""
    images = [Img({"series": 0}), Img({"series": 1}, dtype="u1"),
              Img({"series": 2}, shape=(3, 4, 16, 16), axes="czyx"), Img({"series": 3}, levels=3)]
    return PyramidCollection.from_file(SimpleNamespace(path=None, images=images, plates=[]))


def test_the_table_of_a_file_collection():
    coll = _mixed()
    table = coll.table
    assert len(table) == 4 and table.key_columns == ("series",)
    assert set(IMAGE_COLUMNS) <= set(table.columns)
    first = table.rows[0]
    assert (first["series"], first["axes"], first["shape"], first["dtype"]) == (0, "cyx", (2, 8, 12), "uint16")
    assert (first["size_c"], first["size_z"], first["nbytes"]) == (2, None, 2 * 8 * 12 * 2)
    assert table["dtype"] == ["uint16", "uint8", "uint16", "uint16"]
    assert "CollectionTable: 4 images" in repr(table)


def test_queries():
    coll = _mixed()
    u16 = coll.filter(dtype="uint16")
    assert [dict(k)["series"] for k in u16] == [0, 2, 3]
    assert [dict(k)["series"] for k in coll.filter(size_z=lambda n: n is not None)] == [2]
    assert [dict(k)["series"] for k in coll.filter(lambda row: row["nbytes"] > 1000)] == [2]
    assert [dict(k)["series"] for k in coll.filter(levels=lambda n: n >= 3)] == [3]
    assert coll.table.count("levels") == {1: 3, 3: 1}
    assert coll.table.unique("dtype") == ["uint16", "uint8"]                # sorted as text
    assert coll.table.unique("size_z") == [4, None]                         # None last
    assert len(coll) == 4 and coll.table.count("axes") == {"cyx": 3, "czyx": 1}
    assert coll.table.columns[:1] == ("series",)
    groups = coll.group_by("axes")
    assert list(groups) == ["cyx", "czyx"] and len(groups["cyx"]) == 3
    by_two = coll.group_by("dtype", "axes")
    assert set(by_two) == {("uint16", "cyx"), ("uint8", "cyx"), ("uint16", "czyx")}
    assert [dict(k)["series"] for k in coll.sort_by("nbytes", reverse=True)][:1] == [2]
    assert [dict(k)["series"] for k in coll.sort_by("dtype", "series")] == [0, 2, 3, 1]
    # results are collections: chain, index, and their rows are not built again
    sub = coll.filter(dtype="uint16").filter(axes="cyx")
    assert len(sub) == 2 and set(sub._rows) == set(sub._pyramids)
    assert isinstance(sub[{"series": 0}], Pyramid)


def test_unknown_columns_are_refused():
    coll = _mixed()
    for call in (lambda: coll.filter(colour="red"), lambda: coll.table.unique("colour"),
                 lambda: coll.table.count("colour"), lambda: coll.group_by("colour"), lambda: coll.sort_by("colour")):
        with pytest.raises(KeyError, match="no column 'colour'; the columns are"):
            call()


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_rows_from_files_equal_rows_from_pyramids(tmp_path, version):
    path, _ = _plate(tmp_path, version)
    coll = PyramidCollection.from_zarr(path)
    table = coll.table
    assert _opened(coll) == []                                     # read from the files
    for key, row in zip(coll, table.rows):
        node = coll._pyramids[key]
        expected = row_from_pyramid(coll[key], path=node.loader.path_of(node))
        assert {c: row[c] for c in expected} == expected
    assert table.rows[0]["version"] == version
    # per-axis columns, from the c/y/x pyramids of _plate (scale 1, 0.5, 0.5)
    first = table.rows[0]
    assert (first["scale_y"], first["scale_x"], first["scale_c"], first["scale_z"]) == (0.5, 0.5, 1.0, None)
    assert (first["unit_x"], first["unit_c"], first["translation_x"]) == ("micrometer", None, 0.0)
    assert len(coll.filter(unit_x="micrometer", scale_x=lambda s: s < 1)) == 4
    assert len(coll.filter(lambda image: image["size_y"] < image["size_x"])) == 4
    assert table.rows[0]["codec"] is not None and table.rows[0]["levels"] == 1
    # plate columns
    assert coll.table.unique("well_name") == ["A2", "B1"]
    assert coll.table.unique("acquisition_name") == ["first", "second"]
    assert coll.table.count("row_name") == {"A": 2, "B": 2}
    assert coll.table.count("column_name") == {"1": 2, "2": 2}
    assert [dict(k)["acquisition"] for k in coll.filter(acquisition_name="second")] == [1, 1]
    # a filtered plate keeps its plate views and writes
    second = coll.filter(acquisition_name="second")
    assert [len(w.fields[0].keys) for w in second.plates[0].wells] == [1, 1]


def test_sharded_and_multi_level_images_read_the_same_both_ways(tmp_path):
    data = np.arange(2 * 64 * 64, dtype="u2").reshape(2, 64, 64)
    pyr = Pyramid().from_arrays([data], axis_order="cyx", scales=[[1, 0.5, 0.5]], version="0.5",
                                name="sharded").downscale(n_layers=3)
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(pyr, path, chunk_shape=(1, 16, 16), shard_coefficients=(1, 2, 2))
    coll = PyramidCollection.from_zarr(path)                       # a single image: key ()
    files = coll.table.rows[0]
    opened = row_from_pyramid(coll[()], path=coll._pyramids[()].loader.path_of(coll._pyramids[()]))
    assert {c: files[c] for c in opened} == opened
    assert files["levels"] == 3 and files["chunks"] == (1, 16, 16) and files["shards"] == (1, 32, 32)
    assert files["level_shapes"] == ((2, 64, 64), (2, 32, 32), (2, 16, 16))


def test_a_vendor_style_plate_has_plate_columns():
    images = [FakeImage({"plate": 0, "acquisition": 0, "well": w, "field": f}, w * 10 + f)
              for w, fields in ((0, (0, 2)), (1, (0,))) for f in fields]
    coll = PyramidCollection.from_file(SimpleNamespace(path="/data/s.idx.xml", images=images, plates=[PLATE]))
    assert coll.table.count("well_name") == {"A1": 1, "B3": 2}
    assert coll.table.unique("row_name") == ["A", "B"] and coll.table.unique("plate_name") == [""]


@pytest.mark.skipif(not os.path.isdir(IDR0010), reason="idr0010 is not here")
def test_idr0010_table_opens_no_image():
    import time
    coll = PyramidCollection.from_zarr(IDR0010)
    t = time.perf_counter()
    table = coll.table
    assert time.perf_counter() - t < 5.0                          # well under 1 s locally
    assert len(table) == 384 and _opened(coll) == []
    assert coll.table.unique("dtype") == ["uint16"] and coll.table.unique("levels") == [3]
    key = list(coll)[17]
    node = coll._pyramids[key]
    expected = row_from_pyramid(coll[key], path=node.loader.path_of(node))
    assert {c: table.rows[17][c] for c in expected} == expected


def test_what_the_table_holds_is_read_from_the_table():
    # coll.columns / .unique / .count moved to coll.table (decision C-19: coll.columns is a
    # plate's column slicer); the old names still answer, with a FutureWarning
    from test_plate_as_collection import _two_plates
    coll = _two_plates()
    with pytest.warns(FutureWarning, match=r"coll\.columns is coll\.table\.columns"):
        assert coll.columns == coll.table.columns
    with pytest.warns(FutureWarning, match=r"coll\.table\.unique"):
        assert coll.unique("well_name") == coll.table.unique("well_name") == ["A1", "B3"]
    with pytest.warns(FutureWarning, match=r"coll\.table\.count"):
        assert coll.count("well_name") == coll.table.count("well_name")
    # a plate's collection: rows / columns slice it, as select does
    pcoll = coll.plates[0].as_collection()
    part = pcoll.rows["B"].columns[2:]
    assert part.kind == "plate" and [w.name for w in part.wells] == ["B3"]
    assert sorted(part) == sorted(pcoll.select(row="B", column=slice(2, None)))
    assert sorted(pcoll.columns[0]) == sorted(pcoll.select(column=0))
    with pytest.raises(AttributeError, match="a collection holds no rows"):
        coll.rows
