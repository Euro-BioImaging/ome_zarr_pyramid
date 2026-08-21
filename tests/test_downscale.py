"""Downscaling, level selection, and storage rechunking."""

import numpy as np
import pytest


def test_downscale_sync_builds_levels(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 64, 64), axis_order="cyx", scales=[[1.0, 1.0, 1.0]])
    ds = pyr.downscale(n_layers=3, defer=False)
    assert ds.nlayers == 3
    shapes = [ds.dask_arrays[p].shape for p in ds.meta.resolution_paths]
    assert shapes[0] == (2, 64, 64)
    # each level halves the spatial axes (2x factor)
    assert shapes[1] == (2, 32, 32)
    assert shapes[2] == (2, 16, 16)


def test_downscale_scale_metadata_doubles(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 64, 64), axis_order="cyx", scales=[[1.0, 0.5, 0.5]])
    ds = pyr.downscale(n_layers=2, defer=False)
    assert ds.meta.get_scale("1") == pytest.approx([1.0, 1.0, 1.0])   # 0.5 * 2


def test_downscale_defer_is_lazy(make_pyramid):
    """Deferred means NO COMPUTE - not invisible levels.

    The plan used to be write-only state: the pyramid reported one level while a write
    emitted the full stack. The levels are now resolved lazily (~3 ms) so the object and
    the store agree; what stays deferred is the PIXEL derivation, which the writer does
    from the on-disk base (see test_deferred_downscale_expands_on_write).
    """
    pyr, _ = make_pyramid(shape=(2, 64, 64), axis_order="cyx")
    planned = pyr.downscale(n_layers=3, defer=True)
    assert planned.nlayers == 3
    assert planned.meta.resolution_paths == ["0", "1", "2"]
    assert getattr(planned, "_downscale_plan", None) is not None   # writer still uses it
    # lazy: the levels are unmaterialized arrays, nothing was computed
    assert all(hasattr(planned.layers[q], "shape") for q in planned.meta.resolution_paths)


def test_min_dimension_size_caps_levels(make_pyramid):
    pyr, _ = make_pyramid(shape=(1, 128, 128), axis_order="cyx")
    ds = pyr.downscale(min_dimension_size=32, defer=False)
    smallest = ds.dask_arrays[ds.meta.resolution_paths[-1]].shape
    assert min(smallest[-2:]) >= 32


def test_get_downscaled_pyramid_preserves_level0(make_pyramid, compute):
    """get_downscaled_pyramid() must reproduce the source at level 0 (data + scale)."""
    pyr, data = make_pyramid(shape=(2, 64, 64), axis_order="cyx", scales=[[1.0, 0.5, 0.5]])
    ds = pyr.get_downscaled_pyramid()
    assert ds.meta.resolution_paths[0] == "0"
    assert ds.meta.get_base_scale() == [1.0, 0.5, 0.5]
    np.testing.assert_array_equal(compute(ds), data)


def test_get_downscaled_pyramid_auto_depth(make_pyramid):
    """With no downscaler configured, get_downscaled_pyramid() chooses the depth
    automatically: keep halving until the largest spatial dim would drop below 64
    (the `min_dimension_size` default) - not a single no-op level."""
    pyr, _ = make_pyramid(shape=(3, 512, 512), axis_order="cyx")
    ds = pyr.get_downscaled_pyramid()
    shapes = [ds.dask_arrays[p].shape for p in ds.meta.resolution_paths]
    assert ds.nlayers == 4                              # 512 -> 256 -> 128 -> 64
    assert min(shapes[-1][-2:]) >= 64
    assert min(shapes[-1][-2:]) < 128
    assert all(s[0] == 3 for s in shapes)               # the c axis is never downscaled


def test_get_downscaled_pyramid_small_stays_single(make_pyramid):
    """A base already at/under the min-dimension threshold yields a single level."""
    pyr, _ = make_pyramid(shape=(2, 48, 48), axis_order="cyx")
    assert pyr.get_downscaled_pyramid().nlayers == 1


def test_select_levels_contiguous(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 64, 64), axis_order="cyx")
    ds = pyr.downscale(n_layers=4, defer=False)
    sub = ds.select_levels(1, 2)
    assert sub.nlayers == 2
    # the selected finest level becomes level 0 of the output
    assert sub.dask_arrays["0"].shape == ds.dask_arrays["1"].shape


def test_select_levels_non_contiguous(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 128, 128), axis_order="cyx")
    ds = pyr.downscale(n_layers=4, defer=False)      # 128,64,32,16
    sub = ds.select_levels(0, 2, 3)                  # skip level 1
    assert sub.nlayers == 3
    assert sub.dask_arrays["0"].shape == ds.dask_arrays["0"].shape
    assert sub.dask_arrays["1"].shape == ds.dask_arrays["2"].shape
    assert sub.dask_arrays["2"].shape == ds.dask_arrays["3"].shape


def test_select_levels_accepts_list_and_slice(make_pyramid):
    pyr, _ = make_pyramid(shape=(2, 128, 128), axis_order="cyx")
    ds = pyr.downscale(n_layers=4, defer=False)
    assert ds.select_levels([0, 2]).nlayers == 2
    assert ds.select_levels(slice(1, 3)).nlayers == 2
    assert ds.select_levels(2).nlayers == 1          # single level
    assert ds.select_levels(-1).nlayers == 1         # negative index (coarsest)


def test_rechunk_chunk_shape(pyr3d):
    pyr, _ = pyr3d
    rc = pyr.rechunk(chunk_shape=(3, 8, 8))
    assert rc.dask_arrays["0"].chunksize == (3, 8, 8)


def test_rechunk_chunk_size_mb(pyr3d):
    pyr, _ = pyr3d
    rc = pyr.rechunk(chunk_size_mb=0.001)
    # a tiny target produces chunks no larger than the array
    assert all(c <= s for c, s in zip(rc.dask_arrays["0"].chunksize, pyr.shape))


# --- a pyramid with BOTH real levels and a deferred plan ---------------------------------
# Shape-preserving ops carry a plan forward across every level they transform, so a
# multi-level pyramid can end up carrying one. The rules: THE PLAN WINS on write (it is the
# more recent intent, and says what the pyramid should become); `.downscale()` re-plans
# from level 0; ops keep carrying the plan.

def _multi_level(tmp_path):
    import numpy as np
    from ome_zarr_pyramid import IO, Pyramid
    data = (np.arange(1 * 1 * 32 * 64 * 64) % 251).reshape(1, 1, 32, 64, 64).astype("uint8")
    base = tmp_path / "base.zarr"
    IO().write_pyramid(Pyramid().from_array(data, scale=[1, 1, 1, 1, 1]), str(base),
                       overwrite=True)
    planned = IO().read_pyramid(str(base)).downscale(n_layers=3)
    multi = tmp_path / "multi.zarr"
    IO().write_pyramid(planned, str(multi), overwrite=True)
    return IO().read_pyramid(str(multi))


def _stale_plan(pyr, n_layers=5, active=True):
    """Attach a plan directly, as an op does when it carries one forward.

    `active` picks the state: a plan INHERITED through an op is dormant (it records what
    the source's levels were but commits to nothing), while `Pyramid.downscale()` marks
    one active. Only an active plan is expanded and written.
    """
    pyr._downscale_plan = {
        "n_layers": n_layers, "min_dimension_size": 64, "scale_factor": None,
        "downscale_method": "simple", "backend": "numpy", "smart_scale_factor": None,
    }
    if active:
        pyr._downscale_plan_active = True
    return pyr


def test_op_carries_plan_across_all_levels(tmp_path):
    """An op transforms every existing level AND keeps the plan."""
    pyr = _stale_plan(_multi_level(tmp_path))
    out = pyr + 1
    assert len(out.meta.resolution_paths) == 3
    assert getattr(out, "_downscale_plan", None) is not None


def test_plan_wins_over_existing_levels(tmp_path):
    """A carried plan is applied on write, rebuilding the coarser levels from the base.

    The plan path writes level 0 and re-reads it to expand the rest, so it narrows to the
    base; feeding it several levels used to leave the group advertising levels that were
    never written (`KeyError: '1'`).
    """
    from ome_zarr_pyramid import IO
    pyr = _stale_plan(_multi_level(tmp_path), n_layers=5)      # 3 real levels, plan says 5
    out = tmp_path / "written.zarr"
    IO().write_pyramid(pyr, str(out), overwrite=True)
    assert len(IO().read_pyramid(str(out)).meta.resolution_paths) == 5


def test_downscale_replans_from_level_zero(tmp_path):
    """An explicit `.downscale(n)` overrides whatever levels/plan were there."""
    from ome_zarr_pyramid import IO
    pyr = _stale_plan(_multi_level(tmp_path))
    replanned = pyr.downscale(n_layers=4)
    # the plan (4 levels) wins over whatever levels were there, and is now visible
    assert len(replanned.meta.resolution_paths) == 4
    out = tmp_path / "replanned.zarr"
    IO().write_pyramid(replanned, str(out), overwrite=True)
    assert len(IO().read_pyramid(str(out)).meta.resolution_paths) == 4


# --- downscale() MERGES with a carried plan ----------------------------------------------
# Ops propagate a plan across a pipeline so the intent survives; `downscale` must not be the
# method that silently discards it. Explicit arguments win, omitted ones inherit.

def _planned(pyr, **over):
    plan = {"n_layers": 5, "min_dimension_size": 64, "scale_factor": [1, 1, 1, 2, 2],
            "downscale_method": "simple", "backend": "numpy", "smart_scale_factor": None}
    plan.update(over)
    pyr._downscale_plan = plan
    return pyr


def test_downscale_without_plan_uses_defaults(tmp_path):
    pyr = _multi_level(tmp_path).select_levels("0")
    plan = pyr.downscale()._downscale_plan
    assert plan["n_layers"] is None and plan["scale_factor"] is None


def test_downscale_on_planned_pyramid_is_a_noop(tmp_path):
    """No arguments = nothing asked for = nothing changed."""
    pyr = _planned(_multi_level(tmp_path).select_levels("0"))
    plan = pyr.downscale()._downscale_plan
    assert plan["n_layers"] == 5
    assert plan["scale_factor"] == [1, 1, 1, 2, 2]


def test_explicit_argument_overrides_carried_plan(tmp_path):
    pyr = _planned(_multi_level(tmp_path).select_levels("0"))
    plan = pyr.downscale(n_layers=4)._downscale_plan
    assert plan["n_layers"] == 4
    assert plan["scale_factor"] == [1, 1, 1, 2, 2]      # inherited, not reset


def test_omitted_argument_inherits_scale_factor(tmp_path):
    """The case that motivated merging: a bare downscale used to reset scale_factor to
    None, silently substituting 2x-on-yx and moving the output onto a different grid."""
    pyr = _planned(_multi_level(tmp_path).select_levels("0"))
    plan = pyr.downscale(scale_factor=[1, 1, 2, 2, 2])._downscale_plan
    assert plan["scale_factor"] == [1, 1, 2, 2, 2]
    assert plan["n_layers"] == 5                        # inherited


def test_drop_downscale_plan_keeps_real_levels(tmp_path):
    """The explicit way out: forget the intent, keep the data."""
    from ome_zarr_pyramid import IO
    pyr = _planned(_multi_level(tmp_path))              # 3 real levels + plan says 5
    dropped = pyr.drop_downscale_plan()
    assert getattr(dropped, "_downscale_plan", None) is None
    assert len(dropped.meta.resolution_paths) == 3
    out = tmp_path / "dropped.zarr"
    IO().write_pyramid(dropped, str(out), overwrite=True)
    assert len(IO().read_pyramid(str(out)).meta.resolution_paths) == 3
