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
    assert not (path / "OME" / "zarr.json").exists()          # a plain file (ozp 0.3.1)


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


OME_IMAGE = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
             '<Image ID="Image:0" Name="x"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
             'Type="uint16" SizeX="80" SizeY="64" SizeZ="8" SizeC="2" SizeT="1" '
             'PhysicalSizeX="0.5" PhysicalSizeXUnit="µm" PhysicalSizeY="0.5" '
             'PhysicalSizeYUnit="µm" PhysicalSizeZ="2.0" PhysicalSizeZUnit="µm">'
             '<Channel ID="Channel:0:0"/></Pixels></Image></OME>')


def _with_xml(tmp_path, xml=OME_IMAGE):
    path, _ = _store(tmp_path)
    IO().update_metadata(str(path), ome_xml=xml)
    return path


def _xml(path):
    return (path / "OME" / "METADATA.ome.xml").read_text(encoding="utf-8")


def test_new_scales_reach_the_ome_xml(tmp_path):
    """review cycle 1, N-25: the multiscales took the new sizes, the OME-XML beside the
    image kept the old ones."""
    path = _with_xml(tmp_path)
    IO().update_metadata(str(path), scales={"x": 0.25, "y": 0.25})
    xml = _xml(path)
    assert 'PhysicalSizeX="0.25" PhysicalSizeXUnit="µm"' in xml
    assert 'PhysicalSizeY="0.25"' in xml
    assert 'PhysicalSizeZ="2.0"' in xml                     # an axis not edited stays
    assert '<Channel ID="Channel:0:0"/>' in xml and 'Name="x"' in xml   # the rest as written
    IO().update_metadata(str(path), units={"z": "nanometer"})
    assert 'PhysicalSizeZ="2.0" PhysicalSizeZUnit="nm"' in _xml(path)


def test_a_unitless_axis_loses_its_ome_xml_size(tmp_path):
    """OME's default unit is the micrometre: a size left without a unit would claim it."""
    path = _with_xml(tmp_path)
    IO().update_metadata(str(path), units={"x": None})
    xml = _xml(path)
    assert "PhysicalSizeX=" not in xml and "PhysicalSizeXUnit" not in xml
    assert 'PhysicalSizeY="0.5"' in xml


def test_an_ome_xml_of_several_images_is_left_with_a_warning(tmp_path):
    two = OME_IMAGE.replace("</OME>", '<Image ID="Image:1"><Pixels ID="Pixels:1" '
                            'PhysicalSizeX="9"/></Image></OME>')
    path = _with_xml(tmp_path, two)
    with pytest.warns(UserWarning, match="describes 2 images"):
        IO().update_metadata(str(path), scales={"x": 0.25})
    assert _xml(path) == two


def test_no_ome_xml_none_written(tmp_path):
    path, _ = _store(tmp_path)
    IO().update_metadata(str(path), scales={"x": 0.25})
    assert not (path / "OME" / "METADATA.ome.xml").exists()


def test_new_scales_keep_the_coarser_levels_on_level_0(tmp_path):
    """review cycle 2, R2-06: a mean level's half-pixel shift stayed as it was when the
    scales changed, so the level sat off level 0 again. Its offset from level 0 scales
    with the new size; level 0's own translation (a stage position) stays."""
    a = np.zeros((1, 1, 4, 32, 32), "u2")
    pyr = Pyramid().from_arrays([a], axis_order=AXES, scales=[[1.0, 1.0, 1.0, 1.0, 1.0]],
                                unit_list=["second", None, "micrometer", "micrometer",
                                           "micrometer"], version="0.5")
    path = tmp_path / "m.zarr"
    IO().write_pyramid(pyr.downscale(n_layers=2, scale_factor=2, downscale_method="mean", defer=True), str(path),
                       overwrite=True)
    before = IO().read_pyramid(str(path)).meta
    t1 = before.get_translation(before.resolution_paths[1])
    assert t1[-1] == pytest.approx(0.5)                               # (2 - 1) / 2
    after = IO().update_metadata(str(path), scales={"x": 0.5, "y": 0.5}).meta
    p0, p1 = after.resolution_paths[:2]
    assert after.get_scale(p1)[-1] == pytest.approx(1.0)
    t1 = after.get_translation(p1)
    assert t1[-1] == pytest.approx(0.25) and t1[-2] == pytest.approx(0.25)   # (1 - 0.5) / 2
    assert t1[-3] == pytest.approx(before.get_translation(before.resolution_paths[1])[-3])
    stored = IO().read_pyramid(str(path)).meta
    assert stored.get_translation(stored.resolution_paths[1])[-1] == pytest.approx(0.25)


def test_a_stage_position_stays_when_the_scales_change(tmp_path):
    path, _ = _store(tmp_path)                       # translation [0, 0, 10, 20, 30] on every level
    IO().update_metadata(str(path), scales={"x": 1.0})
    meta = IO().read_pyramid(str(path)).meta
    for p in meta.resolution_paths:
        assert meta.get_translation(p) == pytest.approx([0.0, 0.0, 10.0, 20.0, 30.0])


def test_min_dimension_size_counts_the_levels_of_a_stored_pyramid(tmp_path):
    """review cycle 2, R2-05: on a pyramid read from a store, the source's level count won
    over min_dimension_size in silence (the README's own example); left out, the source's
    count is kept."""
    a = np.zeros((1, 1, 1, 256, 256), "u2")
    pyr = Pyramid().from_arrays([a], axis_order=AXES, scales=[[1.0] * 5],
                                unit_list=["second", None, "micrometer", "micrometer",
                                           "micrometer"], version="0.5")
    path = tmp_path / "s.zarr"
    IO().write_pyramid(pyr.downscale(n_layers=3, defer=True), str(path), overwrite=True)
    stored = IO().read_pyramid(str(path))
    assert [stored.downscale(min_dimension_size=m).nlayers for m in (64, 128, 200)] == [3, 2, 1]
    assert stored.downscale().nlayers == 3
    with pytest.raises(TypeError, match="method"):
        stored.downscale(method="mean")
