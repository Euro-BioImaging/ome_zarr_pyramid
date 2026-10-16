
"""Pyramid.from_image and PyramidCollection.from_file.

Most tests use a hand-made image-like object, so they prove ozp needs no micro-reader:
anything with axes, shape, dtype, region reads and (optionally) resolutions / metadata /
index works. The last test runs the real micro-reader on an OME-TIFF it can open.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection


class Q:
    """A quantity: value + unit, convertible between micrometer and nanometer."""
    _to_m = {"micrometer": 1e-6, "nanometer": 1e-9, "millimeter": 1e-3, "second": 1.0}

    def __init__(self, value, unit):
        self.value, self.unit = value, unit

    def to(self, unit):
        return Q(self.value * self._to_m[self.unit] / self._to_m[unit], unit)


class FakeImage:
    """Image-like: a NumPy array with micro-reader's attribute names."""

    def __init__(self, data, axes, metadata=None, index=None, levels=(), name="img"):
        self._data, self.axes, self.metadata, self.name = data, axes, metadata, name
        self.shape, self.dtype = data.shape, data.dtype
        self.index = index or {"scene": 0}
        self.resolutions = [self] + [FakeImage(lv, axes, metadata, index) for lv in levels]

    def __getitem__(self, key):
        return self._data[key]


def _image(index=None, axes="czyx"):
    data = np.arange(2 * 8 * 32 * 32, dtype="uint16").reshape(2, 8, 32, 32)
    meta = SimpleNamespace(
        pixel_sizes={"z": Q(2.0, "micrometer"), "y": Q(0.5, "micrometer"), "x": Q(500.0, "nanometer")},
        position={"y": Q(10.0, "micrometer"), "x": Q(-3.0, "millimeter")},
        channels=[SimpleNamespace(name="DAPI", color="#0000FF", emission_wavelength=Q(461, "nanometer"),
                                  excitation_wavelength=None),
                  SimpleNamespace(name="GFP", color="#00FF00", emission_wavelength=None,
                                  excitation_wavelength=Q(0.488, "micrometer"))],
        index_values={})
    return FakeImage(data, axes, meta, index, levels=[data[:, :, ::2, ::2]])


def test_from_image_maps_levels_scales_units_position_and_channels():
    pyr = Pyramid.from_image(_image())
    m = pyr.meta
    assert pyr.axes == "czyx" and m.resolution_paths == ["0", "1"]
    assert m.unit_list == [None, "micrometer", "micrometer", "nanometer"]   # x keeps its own unit
    assert m.get_scale("0") == [1.0, 2.0, 0.5, 500.0]
    assert m.get_scale("1") == [1.0, 2.0, 1.0, 1000.0]                     # y/x halved
    # position in each axis's own scale unit: x is in nanometers, given in millimeters
    assert m.get_translation("0") == m.get_translation("1") == [0.0, 0.0, 10.0, -3.0e6]
    chans = m.metadata["omero"]["channels"]
    assert [(c["label"], c["color"]) for c in chans] == [("DAPI", "0000FF"), ("GFP", "00FF00")]
    waves = pyr.acquisition["channel_wavelengths"]
    assert waves[0] == {"emission_wavelength": {"value": 461.0, "unit": "nanometer"}}
    assert waves[1]["excitation_wavelength"]["unit"] == "nanometer"
    assert waves[1]["excitation_wavelength"]["value"] == pytest.approx(488.0)
    assert pyr.source_index == {"scene": 0}


def test_acquisition_is_written_in_the_named_custom_block(tmp_path):
    assert Pyramid.from_image(_image()).custom_metadata == {}             # only when asked
    source = Pyramid.from_image(_image(), acquisition_block="my_app")
    pyr = source.set_channels({0: {"active": False}})                     # survives edits
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(pyr, path, overwrite=True)
    attrs = json.load(open(os.path.join(path, "zarr.json")))["attributes"]
    assert attrs["my_app"]["acquisition"] == source.acquisition
    assert "my_app" not in attrs["ome"]


def test_from_image_levels_are_lazy_reads_of_the_image(tmp_path):
    img = _image()
    pyr = Pyramid.from_image(img)
    assert hasattr(pyr.layers["0"], "_with_transform")                     # a DynamicArray
    path = str(tmp_path / "img.ome.zarr")
    IO().write_pyramid(pyr, path, overwrite=True)
    back = IO().read_pyramid(path)
    np.testing.assert_array_equal(np.asarray(back.layers["0"][...]), img._data)
    np.testing.assert_array_equal(np.asarray(back.layers["1"][...]), img._data[:, :, ::2, ::2])
    assert back.meta.get_translation("0") == [0.0, 0.0, 10.0, -3.0e6]


def test_an_image_without_metadata_gets_ozp_defaults():
    img = FakeImage(np.zeros((4, 16, 16), "uint8"), "zyx")
    pyr = Pyramid.from_image(img, name="plain")
    assert pyr.meta.get_scale("0") == [1.0, 1.0, 1.0]
    assert pyr.meta.get_translation("0") is None
    assert pyr.acquisition == {}


# eubi-bridge's defaults (roadmap B2-B4), so its outputs stay the same ------------------

def test_a_missing_z_or_y_pixel_size_takes_x():
    meta = SimpleNamespace(pixel_sizes={"x": Q(250.0, "nanometer")}, channels=[], position={})
    pyr = Pyramid.from_image(FakeImage(np.zeros((1, 4, 8, 8), "u2"), "tzyx", meta))
    assert pyr.meta.get_scale("0") == [1.0, 250.0, 250.0, 250.0]
    assert pyr.meta.unit_list == ["second", "nanometer", "nanometer", "nanometer"]
    meta.pixel_sizes["z"] = Q(2.0, "micrometer")                 # a z of its own is kept
    pyr = Pyramid.from_image(FakeImage(np.zeros((1, 4, 8, 8), "u2"), "tzyx", meta))
    assert pyr.meta.get_scale("0")[1:] == [2.0, 250.0, 250.0]
    assert pyr.meta.unit_list[1:] == ["micrometer", "nanometer", "nanometer"]


def test_described_channels_without_a_name_or_colour():
    unknown = SimpleNamespace(name=None, color=None, emission_wavelength=None,
                              excitation_wavelength=None)
    named = SimpleNamespace(name="DAPI", color=None, emission_wavelength=None,
                            excitation_wavelength=None)
    meta = SimpleNamespace(pixel_sizes={}, position={}, channels=[named, unknown, unknown])
    pyr = Pyramid.from_image(FakeImage(np.zeros((3, 8, 8), "u2"), "cyx", meta))
    chans = pyr.meta.metadata["omero"]["channels"]
    # by position, counted from 0; no colour of their own: white (OME's default)
    assert [(c["label"], c["color"]) for c in chans] == [
        ("DAPI", "FFFFFF"), ("Channel 1", "FFFFFF"), ("Channel 2", "FFFFFF")]
    # one channel without a c axis, as a single-channel file
    meta.channels = [unknown]
    pyr = Pyramid.from_image(FakeImage(np.zeros((8, 8), "u2"), "yx", meta))
    assert [(c["label"], c["color"]) for c in pyr.meta.metadata["omero"]["channels"]] == [
        ("Channel 0", "FFFFFF")]


def test_channels_the_image_says_nothing_about_get_distinct_colours():
    from ome_zarr_pyramid.core.pyramid import auto_channel_color
    pyr = Pyramid.from_image(FakeImage(np.zeros((20, 4, 4), "u1"), "cyx"))
    pyr = pyr.set_channels({})                       # the defaults, as a write fills them in
    colours = [c["color"] for c in pyr.meta.metadata["omero"]["channels"]]
    assert colours[:7] == ["FF0000", "00FF00", "0000FF", "FF00FF", "00FFFF", "FFFF00", "FFFFFF"]
    assert len(set(colours)) == 20                               # generated, not cycled
    assert colours == [auto_channel_color(i) for i in range(20)]
    # eubi-bridge's 8th to 12th colours (eubi_bridge.utils.metadata_utils.auto_channel_color)
    assert colours[7:12] == ["265CFF", "B8FF66", "BF1CA4", "66FFEB", "FFA426"]


def test_rgb_samples_are_refused_with_the_way_out():
    img = FakeImage(np.zeros((8, 8, 3), "uint8"), "yxs")
    with pytest.raises(ValueError, match="samples='channels'"):
        Pyramid.from_image(img)


def test_collection_from_file_keys_selects_and_folds_resolutions(tmp_path):
    images = [_image({"scene": s, "view": v}) for s in (0, 1) for v in (0, 1)]
    # resolutions="all" lists a level as its own image: folded, not a fifth entry
    level = _image({"scene": 0, "view": 0, "resolution": 1})
    coll = PyramidCollection.from_file(SimpleNamespace(images=images + [level], path="f.czi"))
    assert len(coll) == 4 and coll.varying == ("scene", "view")
    assert coll[{"scene": 1, "view": 0}].source_index == {"scene": 1, "view": 0}
    assert len(coll.select(view=1)) == 2
    assert coll.suffix({"scene": 1, "view": 0}) == "_scene1_view0"
    paths = coll.select(scene=0).write(tmp_path, "f", overwrite=True)
    assert sorted(os.path.basename(p) for p in paths.values()) == [
        "f_view0.ome.zarr", "f_view1.ome.zarr"]          # only what varies in THIS set


def test_two_images_with_one_index_are_refused():
    with pytest.raises(ValueError, match="share the index"):
        PyramidCollection.from_file(SimpleNamespace(images=[_image(), _image()]))


def test_with_micro_reader_on_an_ome_tiff(tmp_path):
    mr = pytest.importorskip("micro_reader")
    tifffile = pytest.importorskip("tifffile")
    path = str(tmp_path / "two_scenes.ome.tif")
    data = np.arange(2 * 3 * 16 * 16, dtype="uint16").reshape(2, 3, 16, 16)
    with tifffile.TiffWriter(path, ome=True) as tw:
        for s in range(2):
            tw.write(data[s] + s, metadata={"axes": "CYX", "PhysicalSizeX": 0.25,
                                             "PhysicalSizeY": 0.25,
                                             "Channel": {"Name": ["a", "b", "c"]}})
    coll = PyramidCollection.from_file(mr.open(path, samples="channels"))
    assert len(coll) == 2
    pyr = coll[{"scene": 1}]
    assert pyr.meta.get_scale("0")[-2:] == [0.25, 0.25]
    assert [c["label"] for c in pyr.meta.metadata["omero"]["channels"]] == ["a", "b", "c"]
    out = str(tmp_path / "scene1.ome.zarr")
    IO().write_pyramid(pyr, out, overwrite=True)
    np.testing.assert_array_equal(np.asarray(IO().read_pyramid(out).layers["0"][...]),
                                  data[1] + 1)


# -- an image's own lazy array (eubi-bridge's LegacyImage over its older readers) -------

class _NoRegionReads(FakeImage):
    """An image offering its own lazy array; reading it region by region is an error."""

    def __init__(self, data, axes, array, metadata=None):
        super().__init__(data, axes, metadata)
        self.array = array
        self.resolutions = [self]

    def __getitem__(self, key):
        raise AssertionError("region read: the image's own array should have been used")


def test_from_image_uses_the_images_own_dask_array(tmp_path):
    da = pytest.importorskip("dask.array")
    data = np.arange(2 * 8 * 32 * 32, dtype="uint16").reshape(2, 8, 32, 32)
    lazy = da.from_array(data, chunks=(1, 8, 16, 16))
    pyr = Pyramid.from_image(_NoRegionReads(data, "czyx", lazy))
    level = pyr.layers[pyr.meta.resolution_paths[0]]
    assert level is lazy                                     # dask stays dask
    out = str(tmp_path / "own.ome.zarr")
    IO().write_pyramid(pyr.downscale(n_layers=2), out)
    back = IO().read_pyramid(out)
    np.testing.assert_array_equal(np.asarray(back.layers["0"][...]), data)


def test_from_image_uses_the_images_own_dyna_array():
    from dyna_zarr import from_array
    data = np.arange(2 * 8 * 32 * 32, dtype="uint16").reshape(2, 8, 32, 32)
    own = from_array(data)
    pyr = Pyramid.from_image(_NoRegionReads(data, "czyx", own))
    assert pyr.layers[pyr.meta.resolution_paths[0]] is own
