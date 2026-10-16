"""IO().update_metadata: edit an OME-Zarr image's metadata in place (scales, units,
channels, windows, OME-XML). The pixels are never rewritten, everything not asked for is
kept (custom blocks, translations, other channels' fields), and both OME-Zarr 0.4 and 0.5
work. eubi-bridge's update_pixel_meta / update_channel_meta are this one call."""
import json
import os

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid

AXES = "tczyx"


def _store(tmp_path, version="0.5", name="img.zarr"):
    a = (np.random.default_rng(1).integers(100, 3000, (1, 2, 8, 64, 80))).astype("u2")
    pyr = Pyramid().from_arrays([a], axis_order=AXES, scales=[[1.0, 1.0, 2.0, 0.5, 0.5]],
                                unit_list=["second", None, "micrometer", "micrometer",
                                           "micrometer"], version=version)
    pyr = pyr.set_channels({0: {"label": "DAPI", "color": "0000FF"},
                            1: {"label": "GFP", "color": "00FF00"}})
    for path in pyr.meta.resolution_paths:
        pyr.meta.update_translation(path, [0.0, 0.0, 10.0, 20.0, 30.0])
    pyr.meta.set_custom("eubi_bridge", {"acquisition": {"source_path": "x.tif"}})
    path = tmp_path / name
    IO().write_pyramid(pyr.downscale(n_layers=3, defer=True), str(path), overwrite=True,
                       windows="dtype")
    return path, a


def _attrs(path):
    attrs = dict(zarr.open_group(str(path), mode="r").attrs)
    return attrs, attrs.get("ome", attrs)


def _array_files(path):
    """Every chunk / array-metadata file and its modification time."""
    found = {}
    for root, _, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, path)
            if rel.split(os.sep)[0] in ("0", "1", "2"):
                found[rel] = os.stat(full).st_mtime_ns
    return found


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_scales_and_units_on_every_level_nothing_else_changes(tmp_path, version):
    path, _ = _store(tmp_path, version)
    before_attrs, before = _attrs(path)
    files = _array_files(path)
    IO().update_metadata(str(path), scales={"z": 1.5, "x": 0.3}, units={"t": "millisecond"})
    after_attrs, after = _attrs(path)
    def transforms(ds):
        found = {t["type"]: t[t["type"]] for t in ds["coordinateTransformations"]}
        return found["scale"], found.get("translation")

    datasets = after["multiscales"][0]["datasets"]
    old = before["multiscales"][0]["datasets"]
    assert len(datasets) == len(old) == 3
    base = transforms(old[0])[0]
    for new_ds, old_ds in zip(datasets, old):
        (new_s, new_t), (old_s, old_t) = transforms(new_ds), transforms(old_ds)
        factor = [o / b for o, b in zip(old_s, base)]           # each level's own factor
        assert new_s == pytest.approx([1.0, 1.0, 1.5 * factor[2], 0.5 * factor[3],
                                       0.3 * factor[4]])
        assert new_t == old_t                                          # translations kept
    units = {a["name"]: a.get("unit") for a in after["multiscales"][0]["axes"]}
    assert units == {"t": "millisecond", "c": None, "z": "micrometer", "y": "micrometer",
                     "x": "micrometer"}
    assert after["omero"] == before["omero"]
    assert after_attrs["eubi_bridge"] == before_attrs["eubi_bridge"]   # custom block kept
    assert _array_files(path) == files                                 # no array rewritten


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_channels_merge_and_keep_the_rest(tmp_path, version):
    path, _ = _store(tmp_path, version)
    IO().update_metadata(str(path), channels={1: {"label": "mCherry", "color": "magenta"}})
    channels = _attrs(path)[1]["omero"]["channels"]
    assert [(c["label"], c["color"]) for c in channels] == [("DAPI", "0000FF"),
                                                            ("mCherry", "FF00FF")]
    assert channels[1]["window"] == {"min": 0, "max": 65535, "start": 0, "end": 65535}


def test_windows_keep_dtype_data_auto_none(tmp_path):
    path, a = _store(tmp_path)
    io = IO()
    io.update_metadata(str(path), channels={0: {"window": {"start": 5, "end": 6}}})
    assert _attrs(path)[1]["omero"]["channels"][0]["window"]["start"] == 5      # keep
    io.update_metadata(str(path), windows="dtype")
    full = {"min": 0, "max": 65535, "start": 0, "end": 65535}
    assert [c["window"] for c in _attrs(path)[1]["omero"]["channels"]] == [full, full]
    io.update_metadata(str(path), windows="data")
    for k, c in enumerate(_attrs(path)[1]["omero"]["channels"]):
        assert c["window"] == {"min": 0, "max": 65535, "start": int(a[:, k].min()),
                               "end": int(a[:, k].max())}
    io.update_metadata(str(path), windows="auto")
    group = zarr.open_group(str(path), mode="r")
    coarsest = np.asarray(group["2"])
    for k, c in enumerate(_attrs(path)[1]["omero"]["channels"]):
        lo, hi = np.percentile(coarsest[:, k], (0.1, 99.9))
        assert c["window"] == {"min": 0, "max": 65535, "start": int(np.floor(lo)),
                               "end": int(np.ceil(hi))}
    io.update_metadata(str(path), windows="none")
    assert all("window" not in c for c in _attrs(path)[1]["omero"]["channels"])
    with pytest.raises(ValueError, match="windows='bogus'"):
        io.update_metadata(str(path), windows="bogus")


def test_the_ome_xml_from_the_updated_pyramid(tmp_path):
    path, _ = _store(tmp_path)
    seen = []

    def xml(pyr):
        seen.append(pyr.meta.metadata["omero"]["channels"][0]["label"])
        return '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"/>'

    IO().update_metadata(str(path), channels={0: {"label": "Hoechst"}}, ome_xml=xml)
    assert seen == ["Hoechst"]
    assert (path / "OME" / "METADATA.ome.xml").read_bytes().startswith(b"<OME")
    assert json.loads((path / "OME" / "zarr.json").read_text())["attributes"]["ome"][
        "series"] == ["0"]


def test_a_channel_axis_unit_is_removed(tmp_path):
    """NGFF gives a channel axis no unit; a store written with one ("Channel") is
    repaired by any update (Neuroglancer rejects the whole store over it)."""
    path, _ = _store(tmp_path, "0.4")
    attrs = json.loads((path / ".zattrs").read_text())
    for axis in attrs["multiscales"][0]["axes"]:
        if axis["name"] == "c":
            axis["unit"] = "Channel"
    (path / ".zattrs").write_text(json.dumps(attrs))
    IO().update_metadata(str(path))
    axes = _attrs(path)[1]["multiscales"][0]["axes"]
    assert "unit" not in next(a for a in axes if a["name"] == "c")
