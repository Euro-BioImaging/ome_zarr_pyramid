"""One API for Pyramid and PyramidCollection, with IO as the one read/write hub (Workplan
C9; plan 'Operations on collections', section 'Symmetry with Pyramid'; decisions C-8 to
C-11): the class methods are shortcuts to IO (which keeps no settings: options are per
call, as in dyna-zarr's `io`), a collection's read properties give the one value its images share or raise
VaryingPropertyError, `only()` and `as_collection()` turn one into the other, and every
old name keeps working."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection, VaryingPropertyError
from ome_zarr_pyramid.core.spatial import pyramid_system

from test_collection_zarr import _opened, _plate
from test_query import Img, _mixed
from test_spatial_pyramid import _stage_file
from conftest import written_images


@pytest.fixture
def calls(monkeypatch):
    """The IO methods called, in order (each still does its work)."""
    seen = []
    for name in ("read_pyramid", "write_pyramid", "read_collection", "write_collection",
                 "write_plate", "write_series", "write", "update_metadata"):
        method = getattr(IO, name)

        def recording(self, *args, _name=name, _method=method, **kwargs):
            seen.append(_name)
            return _method(self, *args, **kwargs)
        monkeypatch.setattr(IO, name, recording)
    return seen


def _same_files(coll_size=2):
    images = [Img({"series": s}) for s in range(coll_size)]
    return PyramidCollection.from_file(SimpleNamespace(path=None, images=images, plates=[]))


def _level0(pyr):
    return np.asarray(pyr.layers[pyr.meta.resolution_paths[0]])


# -- Pyramid: the shortcuts to IO -------------------------------------------------------

def test_pyramid_write_and_from_zarr_are_io_write_pyramid_and_read_pyramid(tmp_path):
    pyr = _same_files(1).only()
    path = str(tmp_path / "img.ome.zarr")
    assert pyr.write(path) == path
    back = Pyramid.from_zarr(path)
    by_io = IO().read_pyramid(path)
    np.testing.assert_array_equal(_level0(back), _level0(by_io))
    assert back.axes == by_io.axes == "cyx"
    # the old ways still work (decision C-11)
    old = Pyramid().from_ngff(path)
    np.testing.assert_array_equal(_level0(old), _level0(back))


def test_the_pyramid_shortcuts_go_through_io(tmp_path, calls):
    pyr = _same_files(1).only()
    path = str(tmp_path / "img.ome.zarr")
    pyr.write(path)
    Pyramid.from_zarr(path, include_labels=False)
    assert calls == ["write_pyramid", "read_pyramid"]


def test_io_takes_no_settings_object():
    with pytest.raises(TypeError):
        Pyramid.from_zarr("x.ome.zarr", io=IO())


def test_read_pyramid_takes_an_open_zarr_group(tmp_path):
    path = str(tmp_path / "img.ome.zarr")
    _same_files(1).only().write(path)
    pyr = IO().read_pyramid(zarr.open_group(path, mode="r"))
    np.testing.assert_array_equal(_level0(pyr), _level0(Pyramid.from_zarr(path)))


def test_chunks_is_the_dyna_name_of_chunk_shape(tmp_path):
    pyr = _same_files(1).only()
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(pyr, path, chunks=(1, 4, 4))
    assert tuple(Pyramid.from_zarr(path).layers["0"].chunks) == (1, 4, 4)
    with pytest.raises(ValueError, match="chunks"):
        IO().write_pyramid(pyr, str(tmp_path / "b.ome.zarr"), chunks=(1, 4, 4),
                           chunk_shape=(1, 4, 4))


def test_pyramid_update_metadata_edits_the_store_it_was_read_from(tmp_path):
    path = str(tmp_path / "img.ome.zarr")
    _same_files(1).only().write(path)
    pyr = Pyramid.from_zarr(path)
    updated = pyr.update_metadata(scales={"x": 0.25}, units={"x": "nanometer"})
    again = Pyramid.from_zarr(path)
    x = again.axes.index("x")
    assert again.meta.get_scale("0")[x] == pytest.approx(0.25)
    assert again.meta.unit_list[x] == "nanometer"
    assert updated.meta.get_scale("0")[x] == pytest.approx(0.25)


def test_update_metadata_of_a_pyramid_built_in_memory_says_what_to_do():
    with pytest.raises(ValueError, match=r"pass path=, or call IO\(\).update_metadata"):
        _same_files(1).only().update_metadata(scales={"x": 1.0})


def test_a_pyramid_as_a_collection_and_back():
    pyr = _same_files(1).only()
    coll = pyr.as_collection()
    assert list(coll) == [()] and coll[()] is pyr and coll.name == pyr.meta.tag
    back = coll.only()
    np.testing.assert_array_equal(_level0(back), _level0(pyr))


def test_a_placed_pyramid_keeps_its_system_through_as_collection_and_only():
    file, _ = _stage_file()
    coll = PyramidCollection.from_file(file)
    placed = coll.pyramid({"series": 1})
    assert list(placed.coordinate_systems) == ["stage"]
    assert placed.coordinate_systems["stage"] == coll.coordinate_systems["stage"]
    assert placed.coordinate_transformations == []
    alone = placed.as_collection()
    assert alone.system_of == {(): "stage"} and list(alone.coordinate_systems) == ["stage"]
    assert pyramid_system(alone.only()) == pyramid_system(placed)
    assert coll.pyramid({"series": 2}).coordinate_systems == {}      # unplaced


def test_a_pyramid_table_is_the_collection_row_without_keys():
    coll = _same_files(1)
    row = coll.table.rows[0]
    alone = coll.only().table
    assert len(alone) == 1 and alone.key_columns == ()
    assert {c: alone.rows[0][c] for c in ("axes", "shape", "dtype", "levels")} == \
        {c: row[c] for c in ("axes", "shape", "dtype", "levels")}


# -- PyramidCollection: the shortcuts to IO ----------------------------------------------

@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_from_zarr_is_io_read_collection_and_opens_nothing(tmp_path, version, calls):
    path, _ = _plate(tmp_path, version)
    del calls[:]
    coll = PyramidCollection.from_zarr(path)
    assert calls == ["read_collection"]
    assert sorted(coll) == sorted(IO().read_collection(path))
    assert _opened(coll) == []


def test_the_collection_writers_go_through_io(tmp_path, calls):
    path, _ = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(path)
    del calls[:]
    coll.write_plate(str(tmp_path / "p.zarr"))
    coll.write_collection(str(tmp_path / "c.zarr"))
    assert calls[0] == "write_plate" and "write_collection" in calls
    series = _same_files(2)
    written = IO().write_series(series, str(tmp_path / "s.ome.zarr"))
    assert written_images(written, tmp_path / "s.ome.zarr") == ["0", "1"]
    again = PyramidCollection.from_zarr(str(tmp_path / "p.zarr"))
    assert sorted(again) == sorted(coll)


# -- one write, the layout chosen (or found) -----------------------------------------------

def test_write_auto_writes_a_plate_as_a_plate(tmp_path, calls):
    path, _ = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(path)
    del calls[:]
    out = str(tmp_path / "p.zarr")
    written = coll.write(out)
    assert calls[0] == "write" and "write_plate" not in calls and "plate" in zarr.open_group(out).attrs["ome"]
    assert sorted(PyramidCollection.from_zarr(out)) == sorted(coll)
    assert written


def test_write_auto_writes_images_as_series_and_one_image_as_an_image(tmp_path):
    written = _same_files(2).write(str(tmp_path / "s.ome.zarr"))
    assert written_images(written, tmp_path / "s.ome.zarr") == ["0", "1"]
    pyr = _same_files(1).only()
    one = str(tmp_path / "one.ome.zarr")
    assert pyr.as_collection().write(one) == one
    np.testing.assert_array_equal(_level0(Pyramid.from_zarr(one)), _level0(pyr))


def test_write_version_picks_the_format_and_the_old_directory_stem_form(tmp_path):
    coll = _same_files(2)
    by_stem = coll.write(str(tmp_path), "img")                    # as before: one store each
    assert sorted(os.path.basename(p) for p in by_stem.values()) ==         sorted(f"img{coll.suffix(k)}.ome.zarr" for k in coll)
    named = coll.write(directory=str(tmp_path / "d"), stem="img")
    assert sorted(named) == sorted(by_stem)
    rfc8 = coll.write(str(tmp_path / "c.zarr"), version="rfc8")
    assert len(written_images(rfc8, tmp_path / "c.zarr")) == 2
    assert zarr.open_group(rfc8, mode="r").attrs["ome"]["version"] == "0.x"
    series = IO().write(coll, str(tmp_path / "s.ome.zarr"), version="0.4")
    assert "bioformats2raw.layout" in zarr.open_group(series, mode="r").attrs
    one = _same_files(1).only()
    v4 = one.as_collection().write(str(tmp_path / "one.ome.zarr"), version="0.4")
    assert Pyramid.from_zarr(v4).meta.version == "0.4"


def test_write_refuses_what_it_cannot_write(tmp_path):
    coll = _same_files(2)
    with pytest.raises(TypeError):
        coll.write(str(tmp_path / "x"), layout="series")         # removed before release
    with pytest.raises(ValueError, match="give a stem"):
        coll.write(str(tmp_path / "x"), names={"series": "s"})
    with pytest.raises(TypeError, match="not both"):
        coll.write(str(tmp_path / "x"), directory=str(tmp_path / "y"))
    with pytest.raises(TypeError, match="Pyramid or a PyramidCollection"):
        IO().write(object(), str(tmp_path / "x"))


def test_collection_update_metadata_edits_every_image_of_its_store(tmp_path):
    path = str(tmp_path / "s.ome.zarr")
    _same_files(2).write_series(path)
    coll = PyramidCollection.from_zarr(path)
    again = coll.update_metadata(scales={"x": 0.5})
    for key in again:
        x = again[key].axes.index("x")
        assert again[key].meta.get_scale("0")[x] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="pass path="):
        _same_files(2).update_metadata(scales={"x": 1.0})


def test_validate_still_reaches_the_writer_through_the_shortcut(tmp_path):
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    written = _same_files(2).write_series(str(tmp_path / "s.ome.zarr"), validate=True)
    assert written_images(written, tmp_path / "s.ome.zarr") == ["0", "1"]


# -- PyramidCollection: read properties shared with Pyramid (decision C-8) -----------------

def test_shared_properties_give_the_one_value():
    coll = _same_files(3)
    first = coll[next(iter(coll))]
    assert coll.axes == first.axes == "cyx"
    assert coll.dtype == first.dtype and coll.shape == first.shape == (2, 8, 12)
    assert coll.nlayers == first.nlayers == 1
    assert coll.compressor == first.compressor
    assert coll.scale_factor_dict == first.scale_factor_dict
    assert coll.custom_metadata == first.custom_metadata
    assert coll.supports_dyna is first.supports_dyna is True


def test_a_property_that_differs_raises_and_points_to_the_table():
    coll = _mixed()                     # dtype, axes and level count differ
    for name in ("dtype", "axes", "shape", "nlayers"):
        with pytest.raises(VaryingPropertyError, match=r"differs between the collection's images.*coll\.table"):
            getattr(coll, name)
    assert issubclass(VaryingPropertyError, ValueError)
    # narrowed to images that agree, the value comes back
    assert coll.filter(dtype="uint16", axes="cyx").dtype == np.dtype("uint16")


def test_shared_axes_shape_and_levels_open_no_image_of_a_store(tmp_path):
    path, _ = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(path)
    first = IO().read_pyramid(path + "/A/2/0")
    assert (coll.axes, coll.shape, coll.nlayers) == (first.axes, tuple(first.shape), first.nlayers)
    assert _opened(coll) == []
    assert coll.dtype == first.dtype and len(_opened(coll)) == 1     # the exact dtype: one image


def test_an_empty_collection_has_no_value_to_share():
    with pytest.raises(ValueError, match="holds no image"):
        _same_files(2).select(series=7).axes


# -- only() and validate() ----------------------------------------------------------------

def test_only_refuses_a_collection_of_another_size_listing_its_keys():
    with pytest.raises(ValueError, match=r"holds 2 images, not one: \{'series': 0\}, \{'series': 1\}"):
        _same_files(2).only()
    with pytest.raises(ValueError, match="holds 0 images"):
        _same_files(2).select(series=7).only()
    assert _same_files(2).select(series=1).only().source_index == {"series": 1}


def test_validate_checks_the_graph_and_every_image_and_chains(tmp_path):
    coll = _same_files(2)
    assert coll.validate() is coll
    path, _ = _plate(tmp_path, "0.5")
    lazy = PyramidCollection.from_zarr(path)
    assert lazy.validate(images=False) is lazy and _opened(lazy) == []
    lazy.validate()
    assert len(_opened(lazy)) == len(lazy)
    lazy.check_invariants()            # the old name stays (decision C-11)


# -- io.read: what the store holds (the other half of io.write) ---------------------------

def test_io_read_gives_a_pyramid_for_an_image_and_a_collection_for_a_layout(tmp_path):
    one = str(tmp_path / "img.ome.zarr")
    _same_files(1).only().write(one)
    pyr = IO().read(one)
    assert isinstance(pyr, Pyramid) and pyr.axes == "cyx"
    series = str(tmp_path / "s.ome.zarr")
    _same_files(2).write_series(series)
    coll = IO().read(series)
    assert isinstance(coll, PyramidCollection) and sorted(coll) == sorted(_same_files(2))
    single_series = str(tmp_path / "s1.ome.zarr")
    _same_files(1).write_series(single_series)       # a layout of one image stays a layout
    assert isinstance(IO().read(single_series), PyramidCollection)
    path, _ = _plate(tmp_path, "0.5")
    assert isinstance(IO().read(path), PyramidCollection)
    assert isinstance(IO().read(zarr.open_group(one, mode="r")), Pyramid)


def test_io_read_then_io_write_keeps_the_kind(tmp_path):
    one = str(tmp_path / "img.ome.zarr")
    _same_files(1).only().write(one)
    assert IO().write(IO().read(one), str(tmp_path / "copy.ome.zarr")) == str(tmp_path / "copy.ome.zarr")
    path, _ = _plate(tmp_path, "0.5")
    out = str(tmp_path / "plate_copy.zarr")
    copied = IO().read(IO().write(IO().read(path), out))
    assert isinstance(copied, PyramidCollection) and sorted(copied) == sorted(IO().read(path))
    assert "plate" in zarr.open_group(out).attrs["ome"]


# -- write returns where io.read reads the collection back ---------------------------

def test_every_format_returns_what_io_read_reads_back(tmp_path):
    path, _ = _plate(tmp_path, "0.5")
    plate = PyramidCollection.from_zarr(path)
    series = _same_files(2)
    for coll, version in ((plate, None), (plate, "rfc8"), (series, "0.5"), (series, "rfc8")):
        out = str(tmp_path / f"{version}_{len(coll)}.zarr")
        written = coll.write(out, version=version)
        assert written == out, version
        back = IO().read(written)
        assert sorted(back) == sorted(coll), version
        for key in coll:
            np.testing.assert_array_equal(_level0(back[key]), _level0(coll[key]))
    doc = str(tmp_path / "c.json")
    assert series.write(str(tmp_path / "d.zarr"), version="rfc8", document=doc) == doc
    assert sorted(IO().read(doc)) == sorted(series)
    # one store per image: no top-level store, so {key: path} of each, as before
    by_stem = series.write(str(tmp_path / "d"), "img")
    assert sorted(by_stem) == sorted(series)
    assert all(isinstance(IO().read(p), Pyramid) for p in by_stem.values())


# -- validate=True checks an RFC 8 store's structure ------------------------------------

def test_write_collection_validate_checks_the_rfc8_structure(tmp_path):
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    from ome_zarr_pyramid import RFC8Warning, validate
    out = str(tmp_path / "c.zarr")
    with pytest.warns(RFC8Warning, match="only its structure is checked"):
        _same_files(2).write_collection(out, validate=True)
    with pytest.warns(RFC8Warning):
        assert validate(out) == []
    # a level removed: the check names it
    import shutil
    shutil.rmtree(os.path.join(out, "0", "0"))
    with pytest.warns(RFC8Warning):
        problems = validate(out)
    assert any("level '0'" in p and "is not an array" in p for p in problems), problems


def test_a_standalone_rfc8_document_is_checked_too(tmp_path):
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    from ome_zarr_pyramid import RFC8Warning
    from ome_zarr_pyramid.core.validation import rfc8_document_problems
    out, doc = str(tmp_path / "c.zarr"), str(tmp_path / "c.json")
    with pytest.warns(RFC8Warning):
        _same_files(2).write_collection(out, document=doc, validate=True)
    assert rfc8_document_problems(doc, out) == []
    import shutil
    shutil.rmtree(os.path.join(out, "1"))
    assert any("is no group" in p for p in rfc8_document_problems(doc, out))


def test_io_read_says_what_to_do_with_one_store_per_image(tmp_path):
    by_stem = _same_files(2).write(str(tmp_path), "img")
    with pytest.raises(TypeError, match="read each path"):
        IO().read(by_stem)
    with pytest.raises(FileNotFoundError, match="is no OME-Zarr store"):
        IO().read(str(tmp_path))


def test_a_single_image_collection_holds_no_plates():
    """.plates is [] for a collection of one image, as before 0.5.0 (decision C-19 made
    it raise for plates, wells and fields only)."""
    coll = _same_files(1).only().as_collection()
    assert coll.kind == "image" and coll.plates == []