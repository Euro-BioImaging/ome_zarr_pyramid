"""write_pyramid straight into a zip (Workplan C16): ``X.ome.zarr.zip`` (the BioImage
Archive's layout, the image's folder inside) or ``X.ozx`` (NGFF RFC-9), byte for byte what a
folder write holds, read back with read_pyramid; and the refusals."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ome_zarr_pyramid import IO, Pyramid

dyna_io = pytest.importorskip("dyna_zarr").io
if not hasattr(dyna_io, "open_zip"):
    pytest.skip("dyna-zarr without zip writing", allow_module_level=True)

DATA = (np.random.default_rng(3).integers(0, 900, (2, 8, 64, 96))
        + np.arange(96)).astype("uint16")


def _image(version="0.5", defer=True):
    return Pyramid().from_arrays([DATA], axis_order="czyx", version=version).downscale(
        n_layers=3, defer=defer)


def _folder(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in Path(root).rglob("*") if p.is_file()}


def _zipped(zip_path, prefix=""):
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.testzip() is None
        return {n[len(prefix):]: zf.read(n) for n in zf.namelist() if n.startswith(prefix)}


@pytest.mark.parametrize("version, defer", [("0.5", True), ("0.5", False), ("0.4", True)])
def test_the_zip_holds_what_a_folder_write_holds(tmp_path, version, defer):
    IO().write_pyramid(_image(version, defer), tmp_path / "img.ome.zarr")
    IO().write_pyramid(_image(version, defer), tmp_path / "img.ome.zarr.zip")
    assert _zipped(tmp_path / "img.ome.zarr.zip", "img.ome.zarr/") == \
        _folder(tmp_path / "img.ome.zarr")
    assert not (tmp_path / "img.ome.zarr.zip.part").exists()
    back = IO().read_pyramid(tmp_path / "img.ome.zarr.zip")
    assert back.meta.resolution_paths == ["0", "1", "2"]
    np.testing.assert_array_equal(np.asarray(back.layers["0"][...]), DATA)


def test_metadata_comes_first_and_is_listed_once(tmp_path):
    IO().write_pyramid(_image(), tmp_path / "img.ome.zarr.zip", windows="auto")
    with zipfile.ZipFile(tmp_path / "img.ome.zarr.zip") as zf:
        names = zf.namelist()
    assert len(names) == len(set(names))
    assert names[0] == "img.ome.zarr/zarr.json"
    first_data = next(i for i, n in enumerate(names) if not n.endswith("zarr.json"))
    assert all(not n.endswith("zarr.json") for n in names[first_data:])
    # windows="auto" read the coarsest level back from the zip being written
    omero = IO().read_pyramid(tmp_path / "img.ome.zarr.zip").meta.metadata["omero"]
    assert omero["channels"][0]["window"]["end"] < 65535


def test_rfc9_ozx(tmp_path):
    IO().write_pyramid(_image(), tmp_path / "img.ozx")
    with zipfile.ZipFile(tmp_path / "img.ozx") as zf:
        assert json.loads(zf.comment)["ome"]["version"] == "0.5"
        assert zf.namelist()[0] == "zarr.json"
    back = IO().read_pyramid(tmp_path / "img.ozx")
    np.testing.assert_array_equal(np.asarray(back.layers["2"][...]),
                                  np.asarray(IO().read_pyramid(tmp_path / "img.ozx").layers["2"][...]))
    with pytest.raises(ValueError, match="0.5 or newer"):
        IO().write_pyramid(_image("0.4"), tmp_path / "old.ozx")
    assert not (tmp_path / "old.ozx").exists()


def test_attached_labels_go_into_the_zip(tmp_path):
    values = (np.indices((8, 64, 96)).sum(axis=0) % 3 + 1).astype("u4")
    label = Pyramid().from_label_arrays([values], axis_order="zyx").downscale(
        n_layers=2, defer=False)
    img = _image(defer=False).add_image_label(label, "cells")
    IO().write_pyramid(img, tmp_path / "img.ome.zarr")
    IO().write_pyramid(img, tmp_path / "img.ome.zarr.zip")
    zipped = _zipped(tmp_path / "img.ome.zarr.zip", "img.ome.zarr/")
    assert "labels/cells/0/zarr.json" in zipped
    assert zipped == _folder(tmp_path / "img.ome.zarr")


def test_refusals_leave_nothing(tmp_path):
    with pytest.raises(ValueError, match="one writer"):
        IO().write_pyramid(_image(), tmp_path / "a.ome.zarr.zip", max_downscale_processes=2)
    with pytest.raises(ValueError, match="not open for writing"):
        IO().write_pyramid(_image(), tmp_path / "b.ome.zarr.zip" / "inner")
    assert list(tmp_path.iterdir()) == []


def test_an_existing_zip_needs_overwrite(tmp_path):
    IO().write_pyramid(_image(), tmp_path / "img.ome.zarr.zip")
    before = (tmp_path / "img.ome.zarr.zip").read_bytes()
    with pytest.raises(FileExistsError):
        IO().write_pyramid(_image(), tmp_path / "img.ome.zarr.zip")
    assert (tmp_path / "img.ome.zarr.zip").read_bytes() == before
    IO().write_pyramid(_image(defer=False), tmp_path / "img.ome.zarr.zip", overwrite=True)
    assert IO().read_pyramid(tmp_path / "img.ome.zarr.zip").meta.resolution_paths == ["0", "1", "2"]
