"""A multiscale-level ``coordinateTransformations`` (NGFF 0.4 / 0.5) applies to every
level, after the level's own. On read it is folded into each level, so a level's scale
and translation are the true ones, and a write keeps them (per level)."""
import json
import os

import numpy as np
import pytest

from ome_zarr_pyramid import IO, Pyramid


def _written(tmp_path, version):
    pyr = Pyramid().from_arrays([np.zeros((8, 32, 32), "u2")], axis_order="zyx",
                                scales=[[2.0, 0.5, 0.5]], version=version)
    path = str(tmp_path / f"a_{version}.ome.zarr")
    IO().write_pyramid(pyr.downscale(n_layers=2), path, overwrite=True)
    return path


def _edit_multiscale(path, version, edit):
    """Rewrite the stored multiscale metadata with ``edit(ms)``."""
    name = ".zattrs" if version == "0.4" else "zarr.json"
    with open(os.path.join(path, name)) as fh:
        doc = json.load(fh)
    ms = (doc["multiscales"][0] if version == "0.4"
          else doc["attributes"]["ome"]["multiscales"][0])
    edit(ms)
    with open(os.path.join(path, name), "w") as fh:
        json.dump(doc, fh)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_a_translation_for_all_levels_is_detected(tmp_path, version):
    path = _written(tmp_path, version)

    def move_up(ms):
        for ds in ms["datasets"]:
            ds["coordinateTransformations"] = [t for t in ds["coordinateTransformations"]
                                               if t["type"] == "scale"]
        ms["coordinateTransformations"] = [{"type": "translation",
                                            "translation": [10.0, 20.0, 30.0]}]
    _edit_multiscale(path, version, move_up)
    meta = IO().read_pyramid(path).meta
    assert meta.get_translation("0") == [10.0, 20.0, 30.0]
    assert meta.get_translation("1") == [10.0, 20.0, 30.0]
    assert meta.get_scale("1") == [4.0, 1.0, 1.0]


def test_scales_and_translations_compose_as_the_spec_says(tmp_path):
    path = _written(tmp_path, "0.4")

    def both(ms):
        ms["datasets"][1]["coordinateTransformations"].append(
            {"type": "translation", "translation": [1.0, 0.25, 0.25]})
        ms["coordinateTransformations"] = [
            {"type": "identity"},
            {"type": "scale", "scale": [1.0, 2.0, 2.0]},
            {"type": "translation", "translation": [5.0, 6.0, 7.0]}]
    _edit_multiscale(path, "0.4", both)
    meta = IO().read_pyramid(path).meta
    # level 1: scale (4, 1, 1) x (1, 2, 2); translation (1, .25, .25) x (1, 2, 2) + (5, 6, 7)
    assert meta.get_scale("1") == [4.0, 2.0, 2.0]
    assert meta.get_translation("1") == [6.0, 6.5, 7.5]
    # level 0 has no translation of its own: only the multiscale one
    assert meta.get_scale("0") == [2.0, 1.0, 1.0]
    assert meta.get_translation("0") == [5.0, 6.0, 7.0]


def test_a_write_keeps_them(tmp_path):
    path = _written(tmp_path, "0.4")
    _edit_multiscale(path, "0.4", lambda ms: ms.__setitem__(
        "coordinateTransformations", [{"type": "translation", "translation": [3.0, 2.0, 1.0]}]))
    out = str(tmp_path / "out.ome.zarr")
    IO().write_pyramid(IO().read_pyramid(path), out, overwrite=True)
    with open(os.path.join(out, ".zattrs")) as fh:
        ms = json.load(fh)["multiscales"][0]
    assert "coordinateTransformations" not in ms           # stored per level now
    back = IO().read_pyramid(out).meta
    assert back.get_translation("0") == [3.0, 2.0, 1.0]
    assert back.get_translation("1") == [3.0, 2.0, 1.0]
