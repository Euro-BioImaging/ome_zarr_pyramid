"""Level names beside the layer names (decision C-16): pyr.levels / nlevels, coll.nlevels,
write_pyramid(levels=), and the table's nlevels column (levels still reads)."""
import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid

from test_api_symmetry import _same_files


def _pyr():
    data = np.arange(2 * 32 * 32, dtype="uint16").reshape(2, 32, 32)
    return Pyramid().from_arrays([data, data[:, ::2, ::2]], axis_order="cyx")


def test_pyramid_levels_are_its_layers():
    pyr = _pyr()
    assert pyr.nlevels == pyr.nlayers == 2
    assert list(pyr.levels) == list(pyr.layers) == ["0", "1"]
    np.testing.assert_array_equal(np.asarray(pyr.levels["1"]), np.asarray(pyr.layers["1"]))


def test_write_pyramid_levels_is_layers(tmp_path):
    pyr = _pyr()
    import os
    out = str(tmp_path / "a.zarr")
    IO().write_pyramid(pyr, out, levels=["1"])                 # level 1 only
    assert os.path.isdir(os.path.join(out, "1")) and not os.path.isdir(os.path.join(out, "0"))
    with pytest.raises(ValueError, match="levels= or layers=, not both"):
        IO().write_pyramid(pyr, str(tmp_path / "b.zarr"), levels=["0"], layers=["0"])


def test_the_table_says_nlevels_and_still_reads_levels():
    coll = _same_files(2)
    table = coll.table
    assert "nlevels" in table.columns and "levels" not in table.columns
    assert table["nlevels"] == table["levels"] == [1, 1]
    assert coll.nlevels == coll.nlayers == 1
    assert len(coll.filter(levels=1)) == len(coll.filter(nlevels=1)) == 2
    assert list(coll.group_by("levels")) == list(coll.group_by("nlevels")) == [1]
    assert len(coll.filter(lambda image: image["levels"] == image["nlevels"])) == 2
    assert "levels" not in coll.table.to_pandas().columns if _has_pandas() else True


def _has_pandas():
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False
