"""Custom metadata blocks: an application's own block next to the NGFF metadata.

OME-Zarr 0.5 reserves the ``ome`` attribute for NGFF, so a custom block (eubi-bridge's
``eubi_bridge``, ...) is stored beside it; in 0.4 everything is at the top level. A block
written inside ``ome`` by older tools (eubi-bridge <= 0.1.5) is still read, and the next
save moves it out.
"""
import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid
from ome_zarr_pyramid.core.pyramid import NGFFMetadataHandler

A = np.arange(2 * 8 * 16 * 16, dtype="uint16").reshape(2, 8, 16, 16)
BLOCK = {"version": "1.2", "acquisition": {"index_values": {"angle": {"value": 90.0, "unit": "degree"}}}}


def _pyr(version):
    return Pyramid().from_arrays([A], axis_order="czyx", version=version)


def _attrs(path):
    return dict(zarr.open_group(str(path), mode="r").attrs)


def _handler(path):
    h = NGFFMetadataHandler()
    h.connect_to_group(str(path), mode="r+")
    return h.read_metadata()


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_written_where_the_spec_puts_it_and_read_back(tmp_path, version):
    pyr = _pyr(version).set_custom_metadata("my_app", BLOCK)
    assert pyr.custom_metadata == {"my_app": BLOCK}
    out = tmp_path / "o.zarr"
    IO().write_pyramid(pyr, str(out), overwrite=True)
    attrs = _attrs(out)
    assert attrs["my_app"] == BLOCK                     # beside `ome` / top level
    if version == "0.5":
        assert set(attrs) == {"ome", "my_app"}
        assert "my_app" not in attrs["ome"]
    assert IO().read_pyramid(str(out)).custom_metadata == {"my_app": BLOCK}


def test_set_returns_a_new_pyramid_and_none_removes():
    pyr = _pyr("0.5")
    with_block = pyr.set_custom_metadata("my_app", BLOCK)
    assert pyr.custom_metadata == {}                     # the original is unchanged
    assert with_block.set_custom_metadata("my_app", None).custom_metadata == {}
    view = with_block.custom_metadata
    view["my_app"]["version"] = "edited"                 # a copy, not the live block
    assert with_block.custom_metadata["my_app"]["version"] == "1.2"


@pytest.mark.parametrize("name", ["ome", "omero", "multiscales", "image-label", "", 3])
def test_reserved_names_are_refused(name):
    with pytest.raises(ValueError, match="reserved"):
        _pyr("0.5").set_custom_metadata(name, BLOCK)


def test_a_legacy_block_inside_ome_is_read_and_moved_out_on_save(tmp_path):
    out = tmp_path / "legacy.zarr"
    IO().write_pyramid(_pyr("0.5"), str(out), overwrite=True)
    group = zarr.open_group(str(out), mode="r+")
    ome = dict(group.attrs["ome"])
    ome["eubi_bridge"] = BLOCK                            # where eubi-bridge <= 0.1.5 wrote it
    group.attrs["ome"] = ome

    assert IO().read_pyramid(str(out)).custom_metadata == {"eubi_bridge": BLOCK}
    h = _handler(out)
    h.set_custom("eubi_bridge", dict(BLOCK, version="2.0"))
    h.save_changes()
    attrs = _attrs(out)
    assert "eubi_bridge" not in attrs["ome"]
    assert attrs["eubi_bridge"]["version"] == "2.0"


def test_the_block_beside_ome_wins_over_a_legacy_one(tmp_path):
    out = tmp_path / "both.zarr"
    IO().write_pyramid(_pyr("0.5").set_custom_metadata("my_app", BLOCK), str(out), overwrite=True)
    group = zarr.open_group(str(out), mode="r+")
    group.attrs["ome"] = dict(group.attrs["ome"], my_app={"version": "old"})
    assert IO().read_pyramid(str(out)).custom_metadata["my_app"] == BLOCK


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_removing_a_block_in_place_deletes_the_attribute(tmp_path, version):
    out = tmp_path / "o.zarr"
    IO().write_pyramid(_pyr(version).set_custom_metadata("my_app", BLOCK), str(out), overwrite=True)
    h = _handler(out)
    h.set_custom("my_app", None)
    h.save_changes()
    assert "my_app" not in _attrs(out)
    assert IO().read_pyramid(str(out)).custom_metadata == {}


def test_edits_and_ops_carry_the_block():
    pyr = _pyr("0.5").set_custom_metadata("my_app", BLOCK)
    for derived in (pyr + 1, pyr.rename("x"), pyr.set_channels({0: {"label": "a"}}),
                    pyr.isel(c=[0])):
        assert derived.custom_metadata == {"my_app": BLOCK}
