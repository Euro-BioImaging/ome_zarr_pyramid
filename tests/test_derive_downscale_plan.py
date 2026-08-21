"""Deriving a downscale plan from a pyramid's EXISTING levels.

`scale_factor=None` used to mean "assume 2 on z/y/x", which silently rewrote the shapes
of any source not built that way. These tests pin the replacement: solve for the factors
actually used, per level and per axis, verify them, and refuse rather than guess.

The governing constraint is that NO backend accepts a target shape - `ts.downsample`,
stride slicing and `da.coarsen` all take integer FACTORS - so a level is reproducible
only if some factor maps the base onto it under that method's rounding.
"""

import numpy as np
import pytest

from ome_zarr_pyramid import IO, Pyramid
from ome_zarr_pyramid.utils.scale import (
    _level_size, _solve_axis_factor, derive_downscale_plan, derive_scale_factors,
)


def _pyr(shapes):
    return Pyramid().from_arrays(
        arrays=[np.random.randint(0, 255, s, dtype='uint8') for s in shapes],
        axis_order='tczyx',
        scales=[[1, 1, 2 ** i, 2 ** i, 2 ** i] for i in range(len(shapes))],
    )


# --- rounding rules ---------------------------------------------------------------------

@pytest.mark.parametrize("base,factor,expected", [
    (101, 2, 51), (255, 2, 128), (7, 2, 4), (100, 2, 50), (99, 3, 33), (10, 1, 10),
])
def test_stride_rounds_up(base, factor, expected):
    """stride / ts.downsample round CEIL. Pinned because reachability depends on it."""
    assert _level_size(base, factor, 'simple') == expected


@pytest.mark.parametrize("base,factor,expected", [
    (101, 2, 50), (255, 2, 127), (7, 2, 3), (100, 2, 50),
])
def test_coarsen_rounds_down(base, factor, expected):
    """da.coarsen(trim_excess=True) rounds FLOOR - the opposite of stride."""
    assert _level_size(base, factor, 'mean') == expected


# --- solving one axis -------------------------------------------------------------------

def test_solves_exact_factor():
    assert _solve_axis_factor(100, 50, 'simple') == 2
    assert _solve_axis_factor(100, 25, 'simple') == 4
    assert _solve_axis_factor(64, 64, 'simple') == 1


def test_refuses_unreachable_axis():
    """No integer factor maps 10 -> 7: 2 gives 5, 3 gives 4. Must be None, not 1."""
    assert _solve_axis_factor(10, 7, 'simple') is None


def test_rounding_decides_reachability():
    """101 -> 50 is floor-only; 101 -> 51 is ceil-only. Each method sees just one."""
    assert _solve_axis_factor(101, 51, 'simple') == 2
    assert _solve_axis_factor(101, 50, 'simple') is None
    assert _solve_axis_factor(101, 50, 'mean') == 2
    assert _solve_axis_factor(101, 51, 'mean') is None


# --- deriving whole progressions --------------------------------------------------------

def test_regular_progression():
    got = derive_scale_factors([(100, 100), (50, 50), (25, 25)], 'simple')
    assert got == [(1, 1), (2, 2), (4, 4)]


def test_irregular_progression():
    """Level 1 at 2x, level 2 at 5x. A single repeated factor CANNOT express this."""
    got = derive_scale_factors([(100, 100), (50, 50), (20, 20)], 'simple')
    assert got == [(1, 1), (2, 2), (5, 5)]


def test_anisotropic_progression():
    got = derive_scale_factors([(64, 64, 64), (64, 32, 32), (32, 16, 16)], 'simple')
    assert got == [(1, 1, 1), (1, 2, 2), (2, 4, 4)]


def test_refuses_unreachable_level():
    assert derive_scale_factors([(10, 10), (7, 7)], 'simple') is None


def test_refuses_floor_source_under_ceil_method():
    """A floor-rounded odd level is unreachable by a ceil method - refuse, not shift."""
    assert derive_scale_factors([(101, 101), (50, 50)], 'simple') is None
    assert derive_scale_factors([(101, 101), (50, 50)], 'mean') is not None


def test_derived_factors_round_trip():
    """Whatever is returned must actually regenerate the input shapes."""
    shapes = [(64, 64, 64), (64, 32, 32), (32, 16, 16)]
    for factor, want in zip(derive_scale_factors(shapes, 'simple'), shapes):
        assert tuple(_level_size(b, f, 'simple')
                     for b, f in zip(shapes[0], factor)) == want


def test_single_level_has_no_plan():
    assert derive_downscale_plan([(64, 64)], 'simple') is None


# --- the Pyramid entry point ------------------------------------------------------------

def test_pyramid_derives_its_own_levels():
    plan = _pyr([(1, 1, 64, 64, 64), (1, 1, 64, 32, 32), (1, 1, 32, 16, 16)]
                ).derive_downscale_plan()
    assert plan['n_layers'] == 3
    assert plan['level_scale_factors'][2] == (1, 1, 2, 4, 4)


def test_pyramid_warns_and_returns_none_when_underivable():
    with pytest.warns(RuntimeWarning, match="could not derive a downscale plan"):
        assert _pyr([(1, 1, 10, 10, 10), (1, 1, 7, 7, 7)]).derive_downscale_plan() is None


def test_derivation_is_silent_when_suppressed():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # any warning would fail the test
        assert _pyr([(1, 1, 10, 10, 10), (1, 1, 7, 7, 7)]
                    ).derive_downscale_plan(warn=False) is None


# --- end to end -------------------------------------------------------------------------

@pytest.mark.parametrize("shapes", [
    [(1, 1, 64, 64, 64), (1, 1, 64, 32, 32), (1, 1, 32, 16, 16)],   # anisotropic
    [(1, 1, 100, 100, 100), (1, 1, 50, 50, 50), (1, 1, 20, 20, 20)],  # irregular
    [(1, 1, 64, 64, 64), (1, 1, 32, 32, 32)],                        # regular
])
def test_write_reproduces_source_shapes_exactly(tmp_path, shapes):
    """The whole point: re-writing a pyramid must not resize its levels."""
    src = _pyr(shapes)
    out = tmp_path / "rt.zarr"
    IO().write_pyramid(src.downscale(), str(out), overwrite=True)
    back = IO().read_pyramid(str(out))
    got = [tuple(back.layers[p].shape) for p in back.meta.resolution_paths]
    assert got == [tuple(s) for s in shapes]


def test_explicit_scale_factor_is_not_overridden(tmp_path):
    """Derivation fills in only when the caller named no factor."""
    src = _pyr([(1, 1, 64, 64, 64), (1, 1, 64, 32, 32)])
    planned = src.downscale(scale_factor=(1, 1, 2, 2, 2), n_layers=3)
    assert planned._downscale_plan['scale_factor'] == (1, 1, 2, 2, 2)
    assert 'level_scale_factors' not in planned._downscale_plan


def test_plan_survives_an_elementwise_op(tmp_path):
    """Ops carry the plan; the derived per-level factors must ride along intact."""
    shapes = [(1, 1, 64, 64, 64), (1, 1, 64, 32, 32), (1, 1, 32, 16, 16)]
    out = _pyr(shapes).downscale() + 1
    assert out._downscale_plan['level_scale_factors'][2] == (1, 1, 2, 4, 4)
    out._downscale_plan_active = True
    dest = tmp_path / "op.zarr"
    IO().write_pyramid(out, str(dest), overwrite=True)
    back = IO().read_pyramid(str(dest))
    assert [tuple(back.layers[p].shape)
            for p in back.meta.resolution_paths] == [tuple(s) for s in shapes]


# --- defaults ---------------------------------------------------------------------------

def test_default_scale_factor_is_isotropic_in_space():
    """z is halved with y/x. It was 1 (plane-wise), which made a raw pyramid and a
    label pyramid of the same mask come out at DIFFERENT shapes."""
    from ome_zarr_pyramid.utils import defaults
    assert defaults.scale_factor_map == {'t': 1, 'c': 1, 'z': 2, 'y': 2, 'x': 2}


def test_config_scale_factor_map_agrees_with_defaults():
    """`core/config.py` carries a duplicate of the map; they must not drift apart."""
    from ome_zarr_pyramid.core import config
    from ome_zarr_pyramid.utils import defaults
    assert config.scale_factor_map == defaults.scale_factor_map


def test_default_downscale_is_stride(tmp_path):
    """'simple' (== stride / nearest) is the default for RAW data too, not just labels:
    averaging is opt-in. Verified on the written levels, not just the plan field."""
    base = np.arange(8 * 8 * 8, dtype='uint16').reshape(1, 1, 8, 8, 8)
    pyr = Pyramid().from_array(base, scale=[1, 1, 1, 1, 1])
    assert pyr.downscale(n_layers=2)._downscale_plan['downscale_method'] == 'simple'
    out = tmp_path / "stride.zarr"
    IO().write_pyramid(pyr.downscale(n_layers=2), str(out), overwrite=True)
    back = IO().read_pyramid(str(out))
    np.testing.assert_array_equal(np.asarray(back.layers['1']),
                                  base[:, :, ::2, ::2, ::2])


def test_default_levels_halve_every_spatial_axis(tmp_path):
    pyr = Pyramid().from_array(np.zeros((1, 1, 32, 32, 32), dtype='uint8'),
                               scale=[1, 1, 1, 1, 1])
    out = tmp_path / "iso.zarr"
    IO().write_pyramid(pyr.downscale(n_layers=3), str(out), overwrite=True)
    back = IO().read_pyramid(str(out))
    assert [tuple(back.layers[p].shape) for p in back.meta.resolution_paths] == [
        (1, 1, 32, 32, 32), (1, 1, 16, 16, 16), (1, 1, 8, 8, 8)]


# --- plan survives level-narrowing ops --------------------------------------------------

def test_merge_keeps_derived_per_level_keys():
    """`.downscale()` on a pyramid carrying a derived plan must not drop the per-level
    factors. The merge rebuilds the recipe from its own six fields, so
    `level_scale_factors`/`level_shapes` fell out and the plan silently degraded to a
    single repeated factor."""
    src = _pyr([(1, 1, 64, 64, 64), (1, 1, 64, 32, 32), (1, 1, 32, 16, 16)])
    merged = src.downscale().downscale()
    assert merged._downscale_plan['level_scale_factors'][2] == (1, 1, 2, 4, 4)


def test_explicit_scale_factor_supersedes_derived_keys():
    """Naming a factor must discard the derived per-level ones, not fight them."""
    src = _pyr([(1, 1, 64, 64, 64), (1, 1, 64, 32, 32)])
    merged = src.downscale().downscale(scale_factor=(1, 1, 2, 2, 2))
    assert merged._downscale_plan['scale_factor'] == (1, 1, 2, 2, 2)
    assert 'level_scale_factors' not in merged._downscale_plan


def test_narrowing_op_preserves_level_count(tmp_path):
    """A Model-B op keeps only the finest level; without a derived plan a 3-level
    input was written back out with ONE level."""
    shapes = [(1, 1, 64, 64, 64), (1, 1, 32, 32, 32), (1, 1, 16, 16, 16)]
    disk = tmp_path / "src.zarr"
    IO().write_pyramid(_pyr(shapes), str(disk), overwrite=True)
    src = IO().read_pyramid(str(disk))
    assert len(src.meta.resolution_paths) == 3

    narrowed = src.select_levels(src.meta.resolution_paths[0])
    plan = src.derive_downscale_plan()
    assert plan is not None
    narrowed._downscale_plan = {
        'n_layers': plan['n_layers'], 'min_dimension_size': 64,
        'scale_factor': plan['scale_factor'],
        'downscale_method': plan['downscale_method'], 'backend': 'numpy',
        'smart_scale_factor': None,
        'level_scale_factors': plan['level_scale_factors'],
        'level_shapes': plan['level_shapes'],
    }
    # an inherited plan is DORMANT; `.downscale()` is what activates it. Simulate that
    # activation directly, since the point here is the level count, not the API surface.
    narrowed._downscale_plan_active = True
    out = tmp_path / "out.zarr"
    IO().write_pyramid(narrowed, str(out), overwrite=True)
    back = IO().read_pyramid(str(out))
    assert [tuple(back.layers[p].shape)
            for p in back.meta.resolution_paths] == [tuple(s) for s in shapes]


# --- a plan is lazy, not invisible --------------------------------------------------------

def test_plan_levels_are_visible():
    """The plan must not be write-only state.

    It used to be: `nlayers`, `resolution_paths`, `layers` and the scales all reported a
    single level while a write emitted the full stack, so the object silently disagreed
    with the store it produced.
    """
    pyr = _pyr([(1, 1, 64, 64, 64)])
    planned = pyr.downscale(n_layers=3)
    assert planned.nlayers == 3
    assert planned.meta.resolution_paths == ["0", "1", "2"]
    assert [tuple(planned.layers[p].shape) for p in planned.meta.resolution_paths] == [
        (1, 1, 64, 64, 64), (1, 1, 32, 32, 32), (1, 1, 16, 16, 16)]
    assert len(planned.meta.get_scale("2")) == 5


def test_plan_survives_materialization():
    """The writer still needs the plan after the levels are resolved."""
    planned = _pyr([(1, 1, 64, 64, 64)]).downscale(n_layers=3)
    assert getattr(planned, "_downscale_plan", None) is not None


def test_visible_levels_are_not_computed():
    """Resolving must stay lazy: no block of the expensive base may run."""
    import dask.array as da
    calls = {"n": 0}

    def expensive(block):
        calls["n"] += 1
        return block * 2

    base = da.from_array(np.zeros((1, 1, 64, 64, 64), dtype='uint8'),
                         chunks=(1, 1, 32, 32, 32)).map_blocks(expensive, dtype='uint8')
    planned = Pyramid().from_array(base, scale=[1, 1, 1, 1, 1]).downscale(n_layers=3)
    assert planned.nlayers == 3
    _ = [planned.layers[p].shape for p in planned.meta.resolution_paths]
    # dask may call the function once with a 0-sized probe block to infer meta; what must
    # not happen is a real pass over the data.
    assert calls["n"] <= 1, f"resolving the plan computed the base ({calls['n']} blocks)"


def test_write_still_computes_the_base_once(tmp_path):
    """Visible levels must not cost the optimization the plan exists for: writing
    derives the coarser levels from the ON-DISK base, so an expensive lazy base is
    computed once, not once per level."""
    import dask.array as da
    calls = {"n": 0}

    def expensive(block):
        calls["n"] += 1
        return block * 2

    raw = da.from_array(np.random.randint(0, 50, (1, 1, 64, 64, 64), dtype='uint8'),
                        chunks=(1, 1, 32, 32, 32))
    planned = Pyramid().from_array(raw.map_blocks(expensive, dtype='uint8'),
                                   scale=[1, 1, 1, 1, 1]).downscale(n_layers=3)
    n_blocks = len(raw.to_delayed().ravel())
    calls["n"] = 0
    IO().write_pyramid(planned, str(tmp_path / "once.zarr"), overwrite=True)
    assert calls["n"] == n_blocks, (
        f"base ran {calls['n'] / n_blocks:.1f} passes; the plan exists to keep it at 1")


# --- dormant vs active --------------------------------------------------------------------
# A plan has two states. Inherited through an op it is DORMANT: it records what the source's
# levels were (count, per-axis factors, shapes) but commits to nothing, so a pipeline never
# produces levels the caller did not ask for. `downscale()` ACTIVATES it - that is what
# asking looks like - and only an active plan is expanded into levels and written.

def _dormant(shapes, tmp_path, name="src"):
    """A single-level pyramid carrying a dormant plan, as an op would leave it."""
    disk = tmp_path / f"{name}.zarr"
    IO().write_pyramid(_pyr(shapes), str(disk), overwrite=True)
    src = IO().read_pyramid(str(disk))
    plan = src.derive_downscale_plan()
    narrowed = src.select_levels(src.meta.resolution_paths[0])
    narrowed._downscale_plan = {
        'n_layers': plan['n_layers'], 'min_dimension_size': 64,
        'scale_factor': plan['scale_factor'],
        'downscale_method': plan['downscale_method'], 'backend': 'numpy',
        'smart_scale_factor': None,
        'level_scale_factors': plan['level_scale_factors'],
        'level_shapes': plan['level_shapes'],
    }
    return narrowed


SHAPES = [(1, 1, 64, 64, 64), (1, 1, 32, 32, 32), (1, 1, 16, 16, 16)]


def test_dormant_plan_reports_one_level(tmp_path):
    d = _dormant(SHAPES, tmp_path)
    assert d.nlayers == 1
    assert getattr(d, '_downscale_plan', None) is not None      # recorded
    assert not getattr(d, '_downscale_plan_active', False)      # but not committed


def test_dormant_plan_is_not_written(tmp_path):
    """A pipeline must not emit levels nobody asked for."""
    d = _dormant(SHAPES, tmp_path)
    out = tmp_path / "dormant.zarr"
    IO().write_pyramid(d, str(out), overwrite=True)
    assert len(IO().read_pyramid(str(out)).meta.resolution_paths) == 1


def test_activation_uses_the_derived_levels(tmp_path):
    """A bare `downscale()` means 'give me the levels my source had'."""
    activated = _dormant(SHAPES, tmp_path).downscale()
    assert activated.nlayers == 3
    assert [tuple(activated.layers[p].shape)
            for p in activated.meta.resolution_paths] == [tuple(s) for s in SHAPES]
    out = tmp_path / "active.zarr"
    IO().write_pyramid(activated, str(out), overwrite=True)
    back = IO().read_pyramid(str(out))
    assert [tuple(back.layers[p].shape)
            for p in back.meta.resolution_paths] == [tuple(s) for s in SHAPES]


def test_activation_arguments_override_the_derived_count(tmp_path):
    assert _dormant(SHAPES, tmp_path).downscale(n_layers=2).nlayers == 2


def test_drop_undoes_an_activated_plan(tmp_path):
    """The escape hatch must really drop levels the plan created - clearing the recipe
    alone left them behind, because an active plan has already expanded into metadata."""
    dropped = _dormant(SHAPES, tmp_path).downscale().drop_downscale_plan()
    assert dropped.nlayers == 1
    out = tmp_path / "dropped.zarr"
    IO().write_pyramid(dropped, str(out), overwrite=True)
    assert len(IO().read_pyramid(str(out)).meta.resolution_paths) == 1


def test_drop_keeps_real_source_levels(tmp_path):
    """Dropping must not narrow levels that were never a plan's doing."""
    disk = tmp_path / "real.zarr"
    IO().write_pyramid(_pyr(SHAPES), str(disk), overwrite=True)
    real = IO().read_pyramid(str(disk))
    assert real.drop_downscale_plan().nlayers == 3


def test_dormant_plan_keeps_its_derivation(tmp_path):
    """Dormancy changes only whether the plan is binding, never what it derived."""
    d = _dormant(SHAPES, tmp_path)
    assert d._downscale_plan['level_scale_factors'] == [
        (1, 1, 1, 1, 1), (1, 1, 2, 2, 2), (1, 1, 4, 4, 4)]
