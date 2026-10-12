"""Two things found by the ecosystem audit (2026-10-12):

- Level 0 was looked up by the NAME '0' (`base_array`, `update_downscaler`), so a store
  whose levels are named otherwise (`s0`, `s1`, ...: NGFF allows any path) raised KeyError.
  Level 0 is `resolution_paths[0]`.
- `downscale(defer=False)` and `get_downscaled_pyramid` called `asyncio.run`, which fails
  inside a running event loop: every Jupyter cell, and eubi-bridge's async conversion.
"""
import asyncio
import json
import os

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid import IO, Pyramid

DATA = np.arange(8 * 64 * 64, dtype="uint16").reshape(8, 64, 64)


def _store_with_level_names(tmp_path, names=("s0", "s1")):
    """A written 2-level OME-Zarr 0.4 store, its levels renamed to `names`."""
    path = str(tmp_path / "named.ome.zarr")
    pyr = Pyramid().from_arrays([DATA], axis_order="zyx", version="0.4")
    IO().write_pyramid(pyr.downscale(n_layers=2), path, overwrite=True)
    attrs_path = os.path.join(path, ".zattrs")
    with open(attrs_path) as fh:
        attrs = json.load(fh)
    for ds, name in zip(attrs["multiscales"][0]["datasets"], names):
        os.rename(os.path.join(path, ds["path"]), os.path.join(path, name))
        ds["path"] = name
    with open(attrs_path, "w") as fh:
        json.dump(attrs, fh)
    return path


def test_level_0_is_the_first_resolution_path_not_the_name_0(tmp_path):
    pyr = IO().read_pyramid(_store_with_level_names(tmp_path))
    assert pyr.meta.resolution_paths[0] == "s0"
    np.testing.assert_array_equal(np.asarray(pyr.base_array[...]), DATA)


@pytest.mark.parametrize("defer", [False, True])
def test_a_store_with_other_level_names_downscales_and_writes(tmp_path, defer):
    pyr = IO().read_pyramid(_store_with_level_names(tmp_path))
    out = str(tmp_path / "out.ome.zarr")
    IO().write_pyramid(pyr.downscale(n_layers=3, defer=defer), out, overwrite=True)
    back = IO().read_pyramid(out)
    assert back.nlayers == 3
    np.testing.assert_array_equal(np.asarray(back.base_array[...]), DATA)


def test_downscale_works_inside_a_running_event_loop():
    pyr = Pyramid().from_arrays([DATA], axis_order="zyx")

    async def notebook_cell():          # what a Jupyter cell is: code in a running loop
        eager = pyr.downscale(n_layers=3, defer=False)
        auto = Pyramid().from_arrays([DATA], axis_order="zyx").get_downscaled_pyramid()
        return eager.nlayers, auto.nlayers

    eager, auto = asyncio.run(notebook_cell())
    assert eager == 3
    assert auto >= 1


def test_downscale_leaves_the_callers_plan_in_place():
    # an op hands a downscaled pyramid's plan on, dormant; downscale() on the result
    # activates it, and the pyramid it was called on keeps its own plan
    pyr = Pyramid().from_arrays([DATA], axis_order="zyx").downscale(n_layers=3) + 1
    before = getattr(pyr, "_downscale_plan", None)
    assert before is not None
    assert pyr.downscale().nlayers == 3
    assert getattr(pyr, "_downscale_plan", None) == before
