"""Pyramid.to_version: the same pixels and metadata written as OME-Zarr 0.4 (zarr v2) or
0.5 (zarr v3), labels included; going there and back gives the metadata it started with."""
import json
import os

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid


def _pyramid(version):
    data = np.arange(2 * 6 * 32 * 40, dtype="u2").reshape(2, 6, 32, 40)
    pyr = Pyramid().from_arrays([data, data[:, :, ::2, ::2]], axis_order="czyx",
                                unit_list=[None, "micrometer", "micrometer", "micrometer"],
                                scales=[[1, 2, 0.5, 0.5], [1, 2, 1, 1]], version=version,
                                name="cells")
    pyr = pyr.set_channels({0: {"label": "DAPI", "color": "0000FF"}, 1: {"label": "GFP"}})
    pyr = pyr.set_custom_metadata("my_app", {"run": 7})
    labels = Pyramid().from_label_arrays([np.zeros((6, 32, 40), "u1"), np.zeros((6, 16, 20), "u1")],
                                         axis_order="zyx", scales=[[2, 0.5, 0.5], [2, 1, 1]],
                                         version=version, name="nuclei")
    return pyr.add_image_label(labels, "nuclei"), data


def _summary(pyr):
    m = pyr.meta
    return (pyr.axes, m.tag, m.resolution_paths, [m.get_scale(p) for p in m.resolution_paths],
            m.unit_list, m.metadata["omero"]["channels"], pyr.custom_metadata)


@pytest.mark.parametrize("source, target", [("0.5", "0.4"), ("0.4", "0.5")])
def test_converted_pyramids_write_and_read_back_the_same(tmp_path, source, target):
    pyr, data = _pyramid(source)
    converted = pyr.to_version(target)
    assert converted.meta.version == target and pyr.meta.version == source   # a new pyramid
    out = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(converted, out)
    group = zarr.open_group(out, mode="r")
    assert group.metadata.zarr_format == (3 if target == "0.5" else 2)
    attrs = group.attrs.asdict()
    if target == "0.5":
        assert attrs["ome"]["version"] == "0.5"
        assert "version" not in attrs["ome"]["multiscales"][0]
        assert attrs["my_app"] == {"run": 7}                                    # beside `ome`
        assert group["0"].metadata.dimension_names == tuple("czyx")
    else:
        assert attrs["multiscales"][0]["version"] == "0.4" and "ome" not in attrs
        assert attrs["my_app"] == {"run": 7}
    back = IO().read_pyramid(out)
    assert back.meta.version == target and _summary(back) == _summary(pyr)
    np.testing.assert_array_equal(np.asarray(back.layers["0"]), data)
    label = back.labels["nuclei"]
    assert label.meta.version == target and label.meta.image_label["version"] == target


def test_there_and_back_restores_the_metadata():
    pyr, _ = _pyramid("0.5")
    again = pyr.to_version("0.4").to_version("0.5")
    assert again.meta.metadata == pyr.meta.metadata
    assert again.labels["nuclei"].meta.metadata == pyr.labels["nuclei"].meta.metadata
    assert pyr.to_version("0.5").meta.metadata == pyr.meta.metadata          # same version


def test_a_sharded_pyramid_goes_to_04_without_shards(tmp_path):
    pyr, data = _pyramid("0.5")
    sharded = str(tmp_path / "sharded.ome.zarr")
    IO().write_pyramid(pyr, sharded, chunk_shape=(1, 2, 16, 16), shard_coefficients=(1, 1, 2, 2))
    read = IO().read_pyramid(sharded)
    assert read.layers["0"].shards is not None
    out = str(tmp_path / "v2.ome.zarr")
    IO().write_pyramid(read.to_version("0.4"), out)
    arr = zarr.open_group(out, mode="r")["0"]
    assert arr.metadata.zarr_format == 2 and tuple(arr.chunks) == (1, 2, 16, 16)
    np.testing.assert_array_equal(arr[...], data)


def test_only_known_versions():
    pyr, _ = _pyramid("0.5")
    with pytest.raises(ValueError, match="'0.7' is not one of"):
        pyr.to_version("0.7")
