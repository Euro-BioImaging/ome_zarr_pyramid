"""One source of truth for a downscale's levels (``resolve_levels`` + ``Pyramid._build_levels``):
the eager ``downscale(defer=False)``, a deferred plan's visible levels and the store it writes
have the same levels - shapes, scales, translations, pixels - and, unless a factor is named,
a pyramid that already has levels keeps its own factors and shapes.

They used to be worked out in several places: the eager path forced 2 on z / y / x on any
source, and a deferred downscale of floor-sized levels (Fractal: 535 -> 267) showed 268
where it wrote 267."""
import numpy as np
import pytest

from ome_zarr_pyramid import Pyramid
from ome_zarr_pyramid.utils.scale import resolve_levels


def _source(shapes, factors):
    """A czyx pyramid with `shapes` (level 0 first), each level's scale level 0's times its
    factor, and a translation on level 0."""
    rng = np.random.default_rng(0)
    arrays = [rng.integers(0, 250, s, dtype="u2") for s in shapes]
    scales = [[1.0, 0.5 * f[1], 0.25 * f[2], 0.25 * f[3]] for f in factors]
    pyr = Pyramid().from_arrays(arrays, axis_order="czyx", scales=scales)
    pyr.meta.update_translation(pyr.meta.resolution_paths[0], [0.0, 3.0, 4.0, 5.0])
    return pyr


SOURCES = {
    # name: (level shapes, level factors)
    "regular": ([(2, 16, 64, 64), (2, 8, 32, 32)], [(1, 1, 1, 1), (1, 2, 2, 2)]),
    "anisotropic": ([(2, 8, 64, 64), (2, 8, 32, 32)], [(1, 1, 1, 1), (1, 1, 2, 2)]),
    "floor-sized": ([(2, 1, 35, 35), (2, 1, 17, 17)], [(1, 1, 1, 1), (1, 1, 2, 2)]),
    "irregular": ([(2, 4, 64, 64), (2, 4, 64, 32), (2, 4, 32, 16)],
                  [(1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 4)]),
}
CASES = [
    ("regular", {}), ("anisotropic", {}), ("floor-sized", {"n_layers": 3}), ("irregular", {}),
    ("anisotropic", {"n_layers": 4}),                       # more levels than the source
    ("regular", {"scale_factor": (1, 1, 2, 2), "n_layers": 3}),   # a named factor wins
]


def _levels(pyr):
    m = pyr.meta
    return [(tuple(pyr.layers[p].shape), [float(v) for v in m.get_scale(p)],
             [float(v) for v in (m.get_translation(p) or [])]) for p in m.resolution_paths]


def _same_pixels(a, b):
    for p, q in zip(a.meta.resolution_paths, b.meta.resolution_paths):
        np.testing.assert_array_equal(np.asarray(a.layers[p]), np.asarray(b.layers[q]))


@pytest.mark.parametrize("method", ["mean", "simple"])
@pytest.mark.parametrize("name, options", CASES)
def test_eager_deferred_and_written_levels_are_the_same(tmp_path, name, options, method):
    src = _source(*SOURCES[name])
    eager = src.downscale(defer=False, downscale_method=method, **options)
    deferred = src.downscale(downscale_method=method, **options)
    assert _levels(eager) == _levels(deferred)
    _same_pixels(eager, deferred)
    out = str(tmp_path / "out.ome.zarr")
    deferred.write(out)
    back = Pyramid.from_zarr(out)
    assert _levels(back) == _levels(deferred)
    _same_pixels(back, deferred)


@pytest.mark.parametrize("defer", [True, False])
def test_a_source_keeps_its_own_factors_unless_one_is_named(defer):
    shapes, factors = SOURCES["anisotropic"]
    src = _source(shapes, factors)
    kept = src.downscale(defer=defer, n_layers=3)
    # z (8) is never downscaled, as in the source; y / x halve on, past its last level
    assert [s for s, _, _ in _levels(kept)] == [(2, 8, 64, 64), (2, 8, 32, 32), (2, 8, 16, 16)]
    named = src.downscale(defer=defer, n_layers=3, scale_factor=(1, 2, 2, 2))
    assert [s for s, _, _ in _levels(named)] == [(2, 8, 64, 64), (2, 4, 32, 32), (2, 2, 16, 16)]


def test_floor_sized_levels_are_kept_eagerly_too():
    eager = _source(*SOURCES["floor-sized"]).downscale(defer=False, downscale_method="mean")
    assert [s for s, _, _ in _levels(eager)] == [(2, 1, 35, 35), (2, 1, 17, 17)]


def test_select_levels_of_a_plan_shows_what_it_writes(tmp_path):
    planned = _source(*SOURCES["floor-sized"]).downscale(n_layers=3, downscale_method="mean")
    kept = planned.select_levels(1, 2)
    out = str(tmp_path / "kept.ome.zarr")
    kept.write(out)
    back = Pyramid.from_zarr(out)
    assert [s for s, _, _ in _levels(back)] == [s for s, _, _ in _levels(kept)]
    assert _levels(back)[0][0] == (2, 1, 17, 17)            # level 1 as the plan built it


def test_resolve_levels_rules():
    # the source's factors and shapes, its last step on past them
    factors, shapes = resolve_levels((1, 64, 256, 256), "czyx", n_layers=5,
                                     level_scale_factors=[(1, 1, 1, 1), (1, 1, 2, 2), (1, 2, 4, 4)])
    assert factors[3:] == [(1, 4, 8, 8), (1, 8, 16, 16)]
    # a recorded shape is kept only where its factor gives it (floor or ceil)
    _, shapes = resolve_levels((1, 1, 35, 35), "czyx", level_scale_factors=[(1, 1, 1, 1), (1, 1, 2, 2)],
                               level_shapes=[(1, 1, 35, 35), (1, 1, 17, 17)])
    assert shapes[1] == (1, 1, 17, 17)
    _, shapes = resolve_levels((1, 1, 35, 35), "czyx", level_scale_factors=[(1, 1, 1, 1), (1, 1, 2, 2)],
                               level_shapes=[(1, 1, 35, 35), (1, 1, 15, 15)])
    assert shapes[1] == (1, 1, 18, 18)
    # no source levels: 2 on z / y / x, as many levels as keep 64 px or more
    factors, _ = resolve_levels((1, 1, 512, 512), "czyx")
    assert factors == [(1, 1, 1, 1), (1, 1, 2, 2), (1, 1, 4, 4), (1, 1, 8, 8)]
