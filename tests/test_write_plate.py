"""write_plate: a plate (from a micro-reader file, or read from NGFF) written one image
at a time, then its metadata; read back with the same layout, pixels and metadata, and
readable by ome-zarr-py (napari-ome-zarr's reader) and valid for ome-zarr-models."""
import glob
import json
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import PyramidCollection
from ome_zarr_pyramid.core.nodes import InvalidCollection, LabelsAttribute, MultiscaleNode, Ref

from test_collection_zarr import _plate
from test_plates import DATA, PLATE, FakeImage
from conftest import written_images


def _file(views=False):
    """Wells B3 (fields 0 and 2) and A1 (field 0) of PLATE; with `views`, two per field."""
    images, value = [], 0
    for w, fields in ((0, (0, 2)), (1, (0,))):
        for f in fields:
            for view in ((0, 1) if views else (None,)):
                index = {"plate": 0, "acquisition": 0, "well": w, "field": f}
                if view is not None:
                    index["view"] = view
                images.append(FakeImage(index, value))
                value += 1
    return SimpleNamespace(path="/data/screen.idx.xml", images=images, plates=[PLATE])


def _pixels(pyr):
    return np.asarray(pyr.layers[pyr.meta.resolution_paths[0]])


def _meta(pyr):
    m = pyr.meta
    return (pyr.axes, [m.get_scale(p) for p in m.resolution_paths], m.unit_list,
            [(c.get("label"), c.get("color")) for c in m.metadata["omero"]["channels"]])


def _viewers(path):
    """What a plate viewer needs: ome-zarr-py (the reader of napari-ome-zarr) finds a
    plate and builds its stitched grid; ome-zarr-models validates it as HCS. Returns the
    grid's shape, or None where ome-zarr-py is not installed (it needs dask): the rest
    of a test runs either way."""
    try:
        import ome_zarr_models
        from ome_zarr import reader
        from ome_zarr.io import parse_url
    except ImportError:
        return None
    node = next(iter(reader.Reader(parse_url(path))()))
    assert any(isinstance(s, reader.Plate) for s in node.specs)
    assert type(ome_zarr_models.open_ome_zarr(zarr.open_group(path, mode="r"))).__name__ == "HCS"
    return node.data[0].shape


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_a_file_plate_round_trips(tmp_path, version):
    coll = PyramidCollection.from_file(_file(), version=version)
    out = str(tmp_path / "screen.zarr")
    written = coll.write_plate(out)
    assert written_images(written, out) == ["A/1/0", "B/3/0", "B/3/1"]
    back = PyramidCollection.from_zarr(out)
    assert back.plates == coll.plates                     # name, rows, columns, wells
    for p_mine, p_back in zip(coll.plates[0].wells, back.plates[0].wells):
        assert [f.index for f in p_back.fields] == [f.index for f in p_mine.fields]
        for mine, theirs in zip(p_mine.fields, p_back.fields):
            np.testing.assert_array_equal(_pixels(mine.image), _pixels(theirs.image))
            assert _meta(mine.image) == _meta(theirs.image)
    attrs = zarr.open_group(out, mode="r").attrs.asdict()
    plate = attrs["ome"]["plate"] if version == "0.5" else attrs["plate"]
    assert plate["name"] == "" and plate["field_count"] == 2
    if version == "0.4":                                  # zarr v2: every group is marked
        assert os.path.exists(os.path.join(out, "B", ".zgroup"))
    grid = _viewers(out)
    assert grid is None or grid[-2:] == (2 * 8, 3 * 12)    # 2 rows x 3 columns of 8 x 12


def test_gapped_field_numbers_survive(tmp_path):
    # micro-reader numbers the fields present (0, 2); NGFF lists a well's images in
    # order, and the well's eubi:images keeps their numbers (plan 8.4)
    coll = PyramidCollection.from_file(_file())
    coll.write_plate(str(tmp_path / "p.zarr"))
    back = PyramidCollection.from_zarr(str(tmp_path / "p.zarr"))
    assert [f.index for f in coll.plates[0].wells[0].fields] == [0, 2]
    assert [f.index for f in back.plates[0].wells[0].fields] == [0, 2]
    assert list(back) == list(coll)


def test_a_field_with_several_images_raises_but_rfc8_keeps_them(tmp_path):
    # 9.4b / 5.2: an NGFF plate field is one image (more in test_write_plate_raises.py)
    from ome_zarr_pyramid import PlateFieldError
    coll = PyramidCollection.from_file(_file(views=True))
    with pytest.raises(PlateFieldError, match=r"they differ in \['view'\]"):
        coll.write_plate(str(tmp_path / "p.zarr"))
    assert not os.path.exists(tmp_path / "p.zarr")
    written = coll.write_plate(str(tmp_path / "r.zarr"), version="rfc8")
    assert len(written_images(written, tmp_path / "r.zarr")) == 6


def test_an_ngff_plate_is_copied(tmp_path):
    src, data = _plate(tmp_path, "0.4")
    out = str(tmp_path / "copy.zarr")
    PyramidCollection.from_zarr(src).write_plate(out)
    copied = zarr.open_group(out, mode="r").attrs.asdict()
    assert {k: v for k, v in copied.items() if not k.startswith("eubi:")} ==         zarr.open_group(src, mode="r").attrs.asdict()
    assert copied["eubi:version"] == 1 and copied["eubi:kind"] == "plate"
    back = PyramidCollection.from_zarr(out)
    for (w, a), expected in data.items():
        key = {"plate": 0, "acquisition": a, "well": w, "field": 0}
        np.testing.assert_array_equal(_pixels(back[key]), expected)


def test_an_existing_target_needs_overwrite(tmp_path):
    out = str(tmp_path / "p.zarr")
    coll = PyramidCollection.from_file(_file())
    coll.write_plate(out)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        coll.write_plate(out)
    zarr.open_group(out, mode="a").create_group("stale")
    coll.write_plate(out, overwrite=True)
    assert not os.path.exists(os.path.join(out, "stale"))     # cleared first
    assert len(PyramidCollection.from_zarr(out)) == 3


@pytest.mark.parametrize("source, target", [("0.4", "0.5"), ("0.5", "0.4")])
def test_images_are_converted_to_the_plate_version(tmp_path, source, target):
    coll = PyramidCollection.from_file(_file(), version=source)
    out = str(tmp_path / "p.zarr")
    coll.write_plate(out, version=target)
    back = PyramidCollection.from_zarr(out)
    for mine, theirs in zip(coll.plates[0].wells, back.plates[0].wells):
        for f_mine, f_back in zip(mine.fields, theirs.fields):
            assert f_back.image.meta.version == target
            np.testing.assert_array_equal(_pixels(f_back.image), _pixels(f_mine.image))
            assert _meta(f_back.image) == _meta(f_mine.image)
    assert zarr.open_group(out, mode="r").metadata.zarr_format == (3 if target == "0.5" else 2)


def test_problems_are_found_before_any_pixel_is_written(tmp_path):
    out = tmp_path / "p.zarr"
    labelled = PyramidCollection.from_file(_file())
    labelled.plates[0].node["B3"].add(MultiscaleNode("nuclei", attributes={
        "labels": LabelsAttribute([Ref("x")])}))
    with pytest.raises((ValueError, InvalidCollection)):
        labelled.write_plate(str(out))
    assert not out.exists()


def test_what_write_plate_refuses(tmp_path):
    coll = PyramidCollection.from_file(_file())
    with pytest.raises(ValueError, match="is a selection of its plate"):
        coll.select(field=0).write_plate(str(tmp_path / "a.zarr"))
    with pytest.raises(IndexError, match="holds 1 plate"):
        coll.write_plate(str(tmp_path / "b.zarr"), plate=1)
    plain = PyramidCollection.from_file(SimpleNamespace(path=None, plates=[],
                                                        images=_file().images[:1]))
    with pytest.raises(ValueError, match="holds no plate"):
        plain.write_plate(str(tmp_path / "c.zarr"))


HARMONY = sorted(glob.glob(os.path.join(DATA, "harmony", "*", "**", "Index.idx.xml"), recursive=True))


@pytest.mark.skipif(not HARMONY, reason="the local Harmony plates are not here")
@pytest.mark.parametrize("version", ["0.4", "0.5"])
@pytest.mark.parametrize("index", [p for p in HARMONY if "idr0034-subset" in p or "zenodo-7841360" in p],
                         ids=lambda p: os.path.relpath(p, DATA)[:40])
def test_harmony_to_ngff_and_back(tmp_path, index, version):
    micro_reader = pytest.importorskip("micro_reader")
    f = micro_reader.open(index)
    coll = PyramidCollection.from_file(f, version=version)
    out = str(tmp_path / "plate.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coll.write_plate(out)
    back = PyramidCollection.from_zarr(out)
    layout = [(p.name, p.rows, p.columns, [(w.row, w.column, w.name, [
        (fl.index, sorted(tuple(sorted((k, v) for k, v in i.index.items() if k != "resolution"))
                          for i in fl.images)) for fl in w.fields]) for w in p.wells]) for p in f.plates]
    assert [(p.name, p.rows, p.columns, [(w.row, w.column, w.name, [
        (fl.index, sorted(tuple(sorted(k)) for k in fl.keys)) for fl in w.fields]) for w in p.wells])
        for p in back.plates] == [(n, tuple(r), tuple(c), w) for n, r, c, w in layout]
    for key in coll:
        assert _meta(coll[key]) == _meta(back[key])
        np.testing.assert_array_equal(_pixels(coll[key]), _pixels(back[key]))
    _viewers(out)
