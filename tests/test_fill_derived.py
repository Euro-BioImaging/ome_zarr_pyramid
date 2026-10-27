"""fill_derived (decision 9.12-S): ozp fills the NGFF SHOULD fields it derives exactly - a
multiscale's empty metadata and type "unknown", a plate acquisition's maximumfieldcount -
by default; fill_derived=False writes them only as the source has them."""
import json
import os

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid, PyramidCollection
from ome_zarr_pyramid.core.validation import FILL_DERIVED

from test_collection_zarr import _plate


def _ms(path):
    with open(os.path.join(path, "zarr.json"), encoding="utf-8") as f:
        ome = json.load(f)["attributes"]["ome"]
    return ome["multiscales"][0]


def _bare():
    """A pyramid without the derivable fields (as one read from another tool's store: an
    ozp-built one carries an empty metadata from the start)."""
    pyr = Pyramid().from_arrays([np.zeros((2, 16, 16), "u2")], axis_order="cyx")
    del pyr.meta.metadata["multiscales"][0]["metadata"]
    return pyr


@pytest.mark.parametrize("version", ["0.5", "0.6"])
def test_an_image_without_the_derived_fields(tmp_path, version):
    pyr = _bare().to_version(version) if version != "0.5" else _bare()
    filled, plain = str(tmp_path / "filled.zarr"), str(tmp_path / "plain.zarr")
    IO().write_pyramid(pyr, filled)
    IO().write_pyramid(pyr, plain, fill_derived=False)          # the same pyramid, after
    assert _ms(filled)["metadata"] == {} and _ms(filled)["type"] == "unknown"
    assert "metadata" not in _ms(plain) and "type" not in _ms(plain)
    np.testing.assert_array_equal(np.asarray(Pyramid.from_zarr(plain).layers["0"]), 0)
    assert FILL_DERIVED.get() is True                     # the setting ends with the write


def test_a_known_downscale_type_is_written_either_way(tmp_path):
    pyr = _bare().downscale(n_layers=2, downscale_method="mean")
    out = str(tmp_path / "a.zarr")
    IO().write_pyramid(pyr, out, fill_derived=False)
    assert _ms(out)["type"] == "mean"                      # the method ozp used: not a fill


def test_a_plate_without_maximumfieldcount(tmp_path):
    src, _ = _plate(tmp_path, "0.5")                        # 2 wells, 2 acquisitions
    group = zarr.open_group(src, mode="r+")
    ome = group.attrs.asdict()["ome"]
    for a in ome["plate"]["acquisitions"]:
        a.pop("maximumfieldcount", None)                    # as another tool writes it
    group.attrs["ome"] = ome
    coll = PyramidCollection.from_zarr(src)
    filled = coll.write_plate(str(tmp_path / "filled.zarr"))
    plain = coll.write_plate(str(tmp_path / "plain.zarr"), fill_derived=False)
    acq = lambda p: zarr.open_group(p, mode="r").attrs["ome"]["plate"]["acquisitions"]
    assert all("maximumfieldcount" in a for a in acq(filled))
    assert not any("maximumfieldcount" in a for a in acq(plain))
    assert sorted(PyramidCollection.from_zarr(plain)) == sorted(coll)


def test_validate_warns_about_what_is_not_filled(tmp_path):
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    from ome_zarr_pyramid import NGFFValidationWarning
    with pytest.warns(NGFFValidationWarning, match="recommendation"):
        IO().write_pyramid(_bare(), str(tmp_path / "a.zarr"), fill_derived=False, validate=True)
    IO().write_pyramid(_bare(), str(tmp_path / "b.zarr"), validate=True)    # filled: no warning


def test_update_metadata_takes_it_too(tmp_path):
    out = str(tmp_path / "a.zarr")
    IO().write_pyramid(_bare(), out, fill_derived=False)
    IO().update_metadata(out, scales={"x": 0.5}, fill_derived=False)
    assert "metadata" not in _ms(out)
    IO().update_metadata(out, scales={"x": 0.5})
    assert _ms(out)["metadata"] == {}
