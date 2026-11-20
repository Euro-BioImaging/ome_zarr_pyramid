"""memory_budget_mb when it is too small for dyna's smallest write (dyna-zarr 0.2.0+,
decision DR-5): the caller's budget is a limit, so the first pass raises
MemoryBudgetError; a later pass gets a budget ozp lowered itself (what the earlier
passes kept), so it warns and writes with dyna's smallest pass instead of stopping a
pyramid whose first levels are already written."""
import importlib

import numpy as np
import pytest

from ome_zarr_pyramid import IO, Pyramid

dyna_zarr = pytest.importorskip("dyna_zarr")
from dyna_zarr import io as dyna_io  # noqa: E402  (the io namespace class)

pytestmark = pytest.mark.skipif(not hasattr(dyna_io, "soft_memory_budget"),
                                reason="dyna-zarr before 0.2.0 only warns")


def _pyramid(levels=3):
    data = np.arange(2 * 32 * 64 * 64, dtype="uint16").reshape(1, 2, 32, 64, 64)
    pyr = Pyramid().from_arrays([data], axis_order="tczyx", version="0.5")
    return pyr.downscale(n_layers=levels, scale_factor=(1, 1, 2, 2, 2),
                         downscale_method="mean"), data


def _one_level_per_pass(monkeypatch):
    """Make every write_levels call write one level; return the soft flag per pass."""
    soft = importlib.import_module("dyna_zarr.io")._SOFT_BUDGET
    real = dyna_io.write_levels
    seen = []

    def write_levels(src, outputs, **kw):
        seen.append(soft.get())
        return real(src, outputs[:1], **kw)

    monkeypatch.setattr(dyna_io, "write_levels", write_levels)
    return seen


def test_a_budget_below_the_first_pass_raises(tmp_path):
    pyr, _ = _pyramid()
    with pytest.raises(dyna_zarr.MemoryBudgetError) as err:
        IO().write_pyramid(pyr, str(tmp_path / "p.zarr"), overwrite=True,
                           memory_budget_mb=0.001)
    assert err.value.needed_mb > 0.001


def test_later_passes_write_in_a_soft_budget(tmp_path, monkeypatch):
    seen = _one_level_per_pass(monkeypatch)
    pyr, data = _pyramid()
    out = tmp_path / "p.zarr"
    IO().write_pyramid(pyr, str(out), overwrite=True, memory_budget_mb=1024)
    assert seen == [False, True, True]
    back = IO().read_pyramid(str(out))
    np.testing.assert_array_equal(
        np.asarray(back.layers[back.meta.resolution_paths[0]][...]), data)


def test_without_a_budget_no_pass_is_soft(tmp_path, monkeypatch):
    seen = _one_level_per_pass(monkeypatch)
    pyr, _ = _pyramid()
    IO().write_pyramid(pyr, str(tmp_path / "p.zarr"), overwrite=True)
    assert seen == [False, False, False]
