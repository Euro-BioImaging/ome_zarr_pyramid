"""Pyramid operations on collections through one private _map (C2), lazily; an operation
that does not fit every image is refused as a whole (C3, decision C-5); the metadata
operations set_channels, set_display_range and set_custom_metadata (C6)."""
import numpy as np
import pytest

from ome_zarr_pyramid import OperationMismatchError, PyramidCollection

from test_api_symmetry import _same_files
from test_collection_zarr import _opened, _plate
from test_query import _mixed


def _labels(pyr):
    return [c.get("label") for c in pyr.meta.metadata["omero"]["channels"]]


def test_a_store_collection_maps_without_opening_an_image(tmp_path):
    path, _ = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(path)
    named = coll.set_channels({0: {"label": "DAPI", "color": "blue"}})
    assert _opened(named) == [] and _opened(coll) == []         # nothing read yet
    assert sorted(named) == sorted(coll) and named.tree is not coll.tree
    key = sorted(named)[0]
    assert _labels(named[key])[0] == "DAPI"                       # applied on first use
    assert named[key].meta.metadata["omero"]["channels"][0]["color"] == "0000FF"
    assert "DAPI" not in _labels(coll[key]) if "omero" in coll[key].meta.metadata else True
    # written: every image of the plate carries the change, the layout is the source's
    back = PyramidCollection.from_zarr(named.write(str(tmp_path / "named.zarr")))
    assert sorted(back) == sorted(coll) and len(back.plates) == 1
    assert all(_labels(back[k])[0] == "DAPI" for k in back)


def test_an_image_already_open_and_one_in_memory_get_it_now():
    coll = _same_files(2)                                         # pyramids in memory
    tagged = coll.set_custom_metadata("lab", {"run": 7})
    assert all(tagged[k].custom_metadata["lab"] == {"run": 7} for k in tagged)
    assert all("lab" not in coll[k].custom_metadata for k in coll)
    np.testing.assert_array_equal(np.asarray(tagged[next(iter(tagged))].layers["0"]),
                                  np.asarray(coll[next(iter(coll))].layers["0"]))


def test_set_display_range_is_computed_when_an_image_is_used(tmp_path):
    path, data = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(path)
    ranged = coll.set_channels({0: {"label": "a"}}).set_display_range("minmax")   # chained
    assert _opened(ranged) == [] and _opened(coll) == []
    key = sorted(ranged)[0]
    window = ranged[key].meta.metadata["omero"]["channels"][0]["window"]
    level0 = np.asarray(ranged[key].layers["0"])[0]
    assert (window["start"], window["end"]) == (level0.min(), level0.max())


def test_an_operation_that_does_not_fit_every_image_is_refused_naming_them():
    coll = _mixed()                                   # series 2 has 3 channels, others 2
    with pytest.raises(OperationMismatchError, match=r"does not fit 3 of the collection's 4 "
                       r"images, so nothing is done: \{'series': 0\}: 2 channel\(s\), no "
                       r"channel 2.*coll\.filter\(size_c="):
        coll.set_channels({2: {"label": "third"}})
    with pytest.raises(OperationMismatchError, match="no channel labelled 'DAPI'"):
        coll.set_channels({"DAPI": {"color": "blue"}})
    with pytest.raises(OperationMismatchError, match=r"no level 2.*nlevels=lambda n: n > 2"):
        coll.set_display_range(stats_level=2)         # only series 3 has 3 levels
    narrowed = coll.filter(size_c=lambda n: (n or 1) > 2).set_channels({2: {"label": "third"}})
    assert [_labels(narrowed[k])[2] for k in narrowed] == ["third"]
    assert issubclass(OperationMismatchError, ValueError)


def test_a_refused_store_collection_opens_nothing(tmp_path):
    path, _ = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(path)
    with pytest.raises(OperationMismatchError):
        coll.set_channels({9: {"label": "x"}})
    assert _opened(coll) == []


def test_a_mapped_collection_keeps_its_graph_and_selections_still_work(tmp_path):
    from test_spatial_pyramid import _stage_file
    file, _ = _stage_file()
    coll = PyramidCollection.from_file(file)
    tagged = coll.set_custom_metadata("lab", {"run": 1})
    assert tagged.coordinate_systems == coll.coordinate_systems
    assert tagged.system_of == coll.system_of
    part = tagged.select(series=1)
    assert len(part) == 1 and part[next(iter(part))].custom_metadata["lab"] == {"run": 1}


def test_the_table_of_a_mapped_collection_opens_nothing(tmp_path):
    path, _ = _plate(tmp_path, "0.5")
    coll = PyramidCollection.from_zarr(path)
    named = coll.set_channels({0: {"label": "DAPI"}}).set_custom_metadata("lab", 1)
    assert set(named.table["channels"]) == {("DAPI", "Channel 1")}
    assert len(named.filter(channels=lambda labels: "DAPI" in labels)) == len(coll)
    part = named.select(well=0)                           # a selection stays lazy too
    assert part.table["channels"][0] == ("DAPI", "Channel 1")
    assert _opened(named) == [] and _opened(part) == [] and _opened(coll) == []
