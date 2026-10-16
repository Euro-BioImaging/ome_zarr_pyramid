"""Metadata that must survive a write (found comparing with eubi-bridge's writer).

- An application's own top-level block (eubi-bridge's 'eubi_bridge') was dropped on every
  OME-Zarr 0.4 write and read: only multiscales / omero / image-label were kept.
- The base translation was lost on levels a downscale plan builds (scales only), and the
  eager path kept it on level 0 only. A translation is a physical origin: every level.
- A non-NGFF '_creator' block was written into every new pyramid.
"""
import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid

A = np.arange(2 * 16 * 32 * 32, dtype="uint16").reshape(2, 16, 32, 32)
T = [0.0, 0.0, -5.0, 7.0]


def _attrs(path):
    at = dict(zarr.open_group(str(path), mode="r").attrs)
    return at.get("ome", at)


def _custom(path, name):
    """A custom block where the spec puts it: beside `ome` (0.5), top level (0.4)."""
    at = dict(zarr.open_group(str(path), mode="r").attrs)
    assert name not in at.get("ome", {})
    return at[name]


def _pyr(version, mode):
    pyr = Pyramid().from_arrays([A], axis_order="czyx", scales=[[1, 0.5, 0.1, 0.1]],
                                version=version)
    pyr.meta.update_translation("0", T)
    if mode == "deferred":
        pyr = pyr.downscale(n_layers=3, scale_factor=(1, 2, 2, 2))
    elif mode == "eager":
        pyr = pyr.downscale(n_layers=3, scale_factor=(1, 2, 2, 2), defer=False)
    pyr.meta.metadata["my_app"] = {"version": "x", "nested": [1, 2]}
    pyr.meta._pending_changes = True
    return pyr


@pytest.mark.parametrize("version", ["0.4", "0.5"])
@pytest.mark.parametrize("mode", ["single", "deferred", "eager"])
def test_custom_block_and_translation_survive(tmp_path, version, mode):
    out = tmp_path / "o.zarr"
    IO().write_pyramid(_pyr(version, mode), str(out), overwrite=True)
    at = _attrs(out)
    assert _custom(out, "my_app") == {"version": "x", "nested": [1, 2]}
    assert "_creator" not in at
    datasets = at["multiscales"][0]["datasets"]
    assert len(datasets) == (1 if mode == "single" else 3)
    for ds in datasets:
        tr = [ct["translation"] for ct in ds["coordinateTransformations"]
              if ct["type"] == "translation"]
        assert tr == [T]
    back = IO().read_pyramid(str(out))
    assert back.meta.metadata["my_app"] == {"version": "x", "nested": [1, 2]}


def test_read_modify_write_keeps_the_block(tmp_path):
    src, out = tmp_path / "s.zarr", tmp_path / "o.zarr"
    IO().write_pyramid(_pyr("0.4", "single"), str(src), overwrite=True)
    IO().write_pyramid(IO().read_pyramid(str(src)) + 1, str(out), overwrite=True)
    assert _custom(out, "my_app")["version"] == "x"
