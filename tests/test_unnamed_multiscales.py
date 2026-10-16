"""A multiscales without ``name`` (optional in NGFF, 0.4 and 0.5): read, edited and
written without inventing one; ``meta.tag`` is None. (It raised KeyError on read and on
every step that copies the metadata.)"""
import json

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO

AXES = [{"name": "c", "type": "channel"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}]


def _unnamed_store(path, zarr_format):
    g = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    data = np.arange(2 * 32 * 40, dtype="u2").reshape(2, 32, 40)
    g.create_array("0", data=data, chunks=(1, 16, 16))
    g.create_array("1", data=data[:, ::2, ::2], chunks=(1, 16, 16))
    multiscales = {"axes": AXES, "datasets": [
        {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 0.5, 0.5]}]},
        {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [1, 1.0, 1.0]}]}]}
    if zarr_format == 2:
        g.attrs["multiscales"] = [{"version": "0.4", **multiscales}]
    else:
        g.attrs["ome"] = {"version": "0.5", "multiscales": [multiscales]}
    return data


def _multiscales(path):
    attrs = zarr.open_group(str(path), mode="r").attrs.asdict()
    return attrs.get("ome", attrs)["multiscales"][0]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_unnamed_multiscales_round_trip(tmp_path, zarr_format):
    data = _unnamed_store(tmp_path / "in.zarr", zarr_format)
    pyr = IO().read_pyramid(str(tmp_path / "in.zarr"))
    assert pyr.meta.tag is None
    edited = pyr.set_channels({0: {"label": "DAPI"}})        # copies the metadata
    assert edited.meta.tag is None
    IO().write_pyramid(edited, str(tmp_path / "out.zarr"))
    assert "name" not in _multiscales(tmp_path / "out.zarr")
    back = IO().read_pyramid(str(tmp_path / "out.zarr"))
    np.testing.assert_array_equal(np.asarray(back.layers["0"]), data)
    assert back.rename("nuclei").meta.tag == "nuclei"


def test_a_pyramid_built_without_a_name_writes_none(tmp_path):
    from ome_zarr_pyramid import Pyramid
    pyr = Pyramid().from_arrays([np.zeros((2, 8, 8), "u1")], axis_order="cyx",
                                scales=[[1, 1, 1]], version="0.5", name=None)
    assert pyr.meta.tag is None
    IO().write_pyramid(pyr, str(tmp_path / "out.zarr"))
    assert "name" not in _multiscales(tmp_path / "out.zarr")
    meta = json.loads((tmp_path / "out.zarr" / "zarr.json").read_text())
    assert "name" not in meta["attributes"]["ome"]["multiscales"][0]


def test_levels_named_other_than_numbers_survive_metadata_edits(tmp_path):
    """Level paths are any NGFF path ('s0', 's1'): a metadata-only copy (rename,
    set_channels) keeps them (it keyed the levels 0..n-1 under the source's paths)."""
    g = zarr.open_group(str(tmp_path / "in.zarr"), mode="w", zarr_format=2)
    data = np.arange(2 * 32 * 40, dtype="u2").reshape(2, 32, 40)
    g.create_array("s0", data=data, chunks=(1, 16, 16))
    g.create_array("s1", data=data[:, ::2, ::2], chunks=(1, 16, 16))
    g.attrs["multiscales"] = [{"version": "0.4", "name": "img", "axes": AXES, "datasets": [
        {"path": p, "coordinateTransformations": [{"type": "scale", "scale": [1, s, s]}]}
        for p, s in (("s0", 0.5), ("s1", 1.0))]}]
    pyr = IO().read_pyramid(str(tmp_path / "in.zarr")).rename("other")
    pyr = pyr.set_channels({0: {"label": "DAPI"}})
    assert pyr.meta.resolution_paths == ["s0", "s1"]
    np.testing.assert_array_equal(np.asarray(pyr.base_array), data)
    IO().write_pyramid(pyr, str(tmp_path / "out.zarr"))
    out = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert sorted(out.array_keys()) == ["s0", "s1"]
    np.testing.assert_array_equal(out["s1"][...], data[:, ::2, ::2])
