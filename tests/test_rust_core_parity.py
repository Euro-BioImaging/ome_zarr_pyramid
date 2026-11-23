"""ome_zarr_pyramid in Rust (ome-zarr-pyramid-core, Workplan DR9) reads what ozp reads.

Pyramids written by ozp in NGFF 0.4, 0.5 and 0.6 (and edited as ozp's own tests edit
them): every level's index-to-world matrix from the Rust crate must give ozp's scale and
translation, with the same axes, units, paths and array shapes. Beyond ozp, the crate
places 0.6 levels with an affine or a rotation, which ozp refuses.
"""
import json
import os
import zipfile

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid

core = pytest.importorskip("ome_zarr_pyramid_core")

UM = "micrometer"


def _rust(path):
    return json.loads(core.read_image(str(path)))


def _written(tmp_path, version, translations=True):
    data = np.arange(8 * 32 * 40, dtype="u2").reshape(8, 32, 40)
    pyr = Pyramid().from_arrays([data], axis_order="zyx", scales=[[2.0, 0.5, 0.25]],
                                unit_list=[UM, UM, UM], version=version)
    pyr = pyr.downscale(n_layers=3)
    if translations:
        for i, p in enumerate(pyr.meta.resolution_paths):
            pyr.meta.update_translation(p, [1.0 * i, -2.5, 0.125 * i])
    path = str(tmp_path / f"a_{version}.ome.zarr")
    IO().write_pyramid(pyr, path, overwrite=True)
    return path


def _edit_multiscale(path, version, edit):
    """Rewrite the stored multiscale metadata with ``edit(ms)`` (as ozp's tests do)."""
    name = ".zattrs" if version == "0.4" else "zarr.json"
    with open(os.path.join(path, name)) as fh:
        doc = json.load(fh)
    ms = (doc["multiscales"][0] if version == "0.4"
          else doc["attributes"]["ome"]["multiscales"][0])
    edit(ms)
    with open(os.path.join(path, name), "w") as fh:
        json.dump(doc, fh)


def _assert_parity(path, rust=None):
    """Rust's levels against ozp's reading of the same image."""
    pyr = IO().read_pyramid(path)
    meta = pyr.meta
    r = rust if rust is not None else _rust(path)
    ms = r["multiscales"][0]
    assert r["version"] == str(meta.version)
    assert [a["name"] for a in ms["axes"]] == list(meta.axis_order)
    units = meta.unit_dict
    assert [a.get("unit") for a in ms["axes"]] == [units.get(a["name"]) for a in ms["axes"]]
    assert [lv["path"] for lv in ms["levels"]] == list(meta.resolution_paths)
    n = len(ms["axes"])
    for lv in ms["levels"]:
        m = np.asarray(lv["matrix"], dtype=float)      # NGFF's affine form: n x (n + 1)
        assert m.shape == (n, n + 1)
        np.testing.assert_array_equal(m[:, :n], np.diag(np.diag(m[:, :n])))
        scale = meta.get_scale(lv["path"])
        trans = meta.get_translation(lv["path"]) or [0.0] * n
        np.testing.assert_allclose(np.diag(m[:, :n]), scale, rtol=0, atol=1e-12)
        np.testing.assert_allclose(m[:n, n], trans, rtol=0, atol=1e-12)
        shape = zarr.open_array(os.path.join(path, lv["path"]), mode="r").shape
        assert tuple(lv["shape"]) == tuple(shape)


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
@pytest.mark.parametrize("translations", [False, True])
def test_levels_place_as_ozp_reads_them(tmp_path, version, translations):
    _assert_parity(_written(tmp_path, version, translations))


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_a_multiscale_transformation_applies_after_each_level(tmp_path, version):
    """ozp's composition: world = s_ms * (s_ds * i + t_ds) + t_ms."""
    path = _written(tmp_path, version)

    def outer(ms):
        ms["coordinateTransformations"] = [
            {"type": "identity"}, {"type": "scale", "scale": [1.0, 2.0, 2.0]},
            {"type": "translation", "translation": [5.0, 6.0, 7.0]}]
    _edit_multiscale(path, version, outer)
    _assert_parity(path)


def test_a_zipped_pyramid_reads_as_the_folder(tmp_path):
    folder = _written(tmp_path, "0.5")
    zpath = tmp_path / "a.ozx"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(folder):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, folder).replace(os.sep, "/"))
    from_zip = json.loads(core.read_image_zip(str(zpath)))
    assert from_zip["multiscales"] == _rust(folder)["multiscales"]
    _assert_parity(folder, from_zip)


def test_both_refuse_an_unknown_version(tmp_path):
    path = _written(tmp_path, "0.5")
    with open(os.path.join(path, "zarr.json")) as fh:
        doc = json.load(fh)
    doc["attributes"]["ome"]["version"] = "0.7"
    with open(os.path.join(path, "zarr.json"), "w") as fh:
        json.dump(doc, fh)
    with pytest.raises(core.OmeError, match="0.7"):
        _rust(path)
    with pytest.raises(ValueError):
        IO().read_pyramid(path)


def test_rust_places_a_rotated_06_level_that_ozp_refuses(tmp_path):
    """A 0.6 level with an affine (a quarter turn and a shift): ozp refuses it, the
    crate gives its matrix."""
    g = zarr.open_group(str(tmp_path / "r.zarr"), mode="w", zarr_format=3)
    g.create_array("0", shape=(4, 6), dtype="u1")
    affine = [[0.0, -0.5, 3.0], [0.5, 0.0, 4.0]]
    g.attrs.put({"ome": {"version": "0.6", "multiscales": [{
        "name": "r",
        "coordinateSystems": [{"name": "physical", "axes": [
            {"name": "y", "type": "space", "unit": UM},
            {"name": "x", "type": "space", "unit": UM}]}],
        "datasets": [{"path": "0", "coordinateTransformations": [{
            "type": "affine", "affine": affine,
            "input": {"path": "0"}, "output": {"name": "physical"}}]}]}]}})
    r = _rust(tmp_path / "r.zarr")
    m = np.asarray(r["multiscales"][0]["levels"][0]["matrix"])
    np.testing.assert_array_equal(m, np.asarray(affine))
    np.testing.assert_array_equal(m @ [2.0, 3.0, 1.0], [1.5, 5.0])
    assert r["multiscales"][0]["system"] == "physical"
    with pytest.raises(ValueError):
        IO().read_pyramid(str(tmp_path / "r.zarr"))
