"""write_pyramid's engines: 'auto' streams through dyna, explicit engines are honoured
or refused (never swapped), and every storage option reaches the store.

Distilled from the write-path audit (benchmarks/audit_write_matrix.py,
reports/write_audit.md). Each test pins one finding: before, the codec was silently
ignored, shards were never written, the tensorstore engine raised NameError, a
DynamicArray pyramid ignored the requested engine, and a deferred downscale of an
in-memory zarr source lost its chunk grid.
"""
import numpy as np
import pytest
import zarr

from ome_zarr_pyramid.core.io import IO, PyramidIO
from ome_zarr_pyramid.core.pyramid import Pyramid

SHAPE, CHUNKS, AXES = (2, 24, 40, 40), (1, 8, 16, 16), "czyx"


@pytest.fixture
def data():
    return np.random.default_rng(0).integers(0, 60000, size=SHAPE, dtype="uint16")


@pytest.fixture
def route(monkeypatch):
    """Records which writer each write_pyramid call ends up in."""
    calls = []
    for meth, tag in (("_write_pyramid_dyna", "dyna"), ("_write_all_layers_unified", "sync"),
                      ("write_pyramid_tensorstore", "ts")):
        orig = getattr(PyramidIO, meth)

        def spy(self, *a, __o=orig, __t=tag, **k):
            calls.append(__t)
            return __o(self, *a, **k)
        monkeypatch.setattr(PyramidIO, meth, spy)
    return calls


def _zarr_pyr(data, version="0.5", **kw):
    zf = 3 if version == "0.5" else 2
    arr = zarr.create_array(store=zarr.storage.MemoryStore(), shape=SHAPE, chunks=CHUNKS,
                            dtype=data.dtype, zarr_format=zf, **kw)
    arr[...] = data
    return Pyramid().from_arrays([arr], axis_order=AXES, version=version)


def _level(path, level="0"):
    return zarr.open_group(str(path), mode="r")[level]


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_auto_streams_a_non_dask_pyramid_through_dyna(data, zarr_path, route, version):
    IO().write_pyramid(_zarr_pyr(data, version), zarr_path, overwrite=True)
    assert route == ["dyna"]
    a = _level(zarr_path)
    np.testing.assert_array_equal(a[...], data)
    assert tuple(a.chunks) == CHUNKS


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_a_dask_pyramid_is_pumped_into_a_dyna_sink(data, zarr_path, route, version):
    # the dask pump: dask schedules, dyna owns the output (codec, chunks, names)
    da = pytest.importorskip("dask.array")
    pyr = Pyramid().from_arrays([da.from_array(data, chunks=CHUNKS) + 1], axis_order=AXES,
                                version=version)
    IO().write_pyramid(pyr, zarr_path, overwrite=True, compressor="zstd")
    assert route == ["dyna"]
    a = _level(zarr_path)
    np.testing.assert_array_equal(a[...], data + 1)
    assert tuple(a.chunks) == CHUNKS
    assert "zstd" in type(a.compressors[0]).__name__.lower()
    if version == "0.5":
        assert a.metadata.dimension_names == tuple(AXES)


def test_dask_pump_shards_and_cascades_a_deferred_downscale(data, zarr_path):
    da = pytest.importorskip("dask.array")
    pyr = Pyramid().from_arrays([da.from_array(data, chunks=CHUNKS)], axis_order=AXES,
                                version="0.5").downscale(n_layers=3, defer=True)
    IO().write_pyramid(pyr, zarr_path, overwrite=True, shard_coefficients=(1, 3, 2, 2))
    assert _level(zarr_path).shards == (1, 24, 32, 32)
    np.testing.assert_array_equal(_level(zarr_path, "2")[...], data[:, ::4, ::4, ::4])


def test_multiprocessing_refuses_dask_levels(data, zarr_path):
    da = pytest.importorskip("dask.array")
    pyr = Pyramid().from_arrays([da.from_array(data, chunks=CHUNKS)], axis_order=AXES)
    with pytest.raises(ValueError, match="dask-backed"):
        IO().write_pyramid(pyr, zarr_path, overwrite=True, use_multiprocessing=True)


@pytest.mark.parametrize("engine,tag", [("sync", "sync"), ("tensorstore", "ts")])
def test_an_explicit_engine_is_honoured_for_a_dyna_pyramid(data, zarr_path, route, engine, tag):
    pyr = _zarr_pyr(data) * 1                      # DynamicArray levels
    IO().write_pyramid(pyr, zarr_path, overwrite=True, backend=engine)
    assert route == [tag]
    np.testing.assert_array_equal(_level(zarr_path)[...], data)


@pytest.mark.parametrize("engine", ["auto", "sync"])
def test_compressor_reaches_the_store(data, zarr_path, engine):
    IO().write_pyramid(_zarr_pyr(data), zarr_path, overwrite=True, backend=engine,
                       compressor="zstd")
    assert type(_level(zarr_path).compressors[0]).__name__ == "ZstdCodec"


def test_v3_levels_carry_dimension_names(data, zarr_path):
    IO().write_pyramid(_zarr_pyr(data), zarr_path, overwrite=True)
    assert _level(zarr_path).metadata.dimension_names == tuple(AXES)


def test_shard_coefficients_shard_every_level(data, zarr_path):
    pyr = _zarr_pyr(data).downscale(n_layers=3, defer=True)
    IO().write_pyramid(pyr, zarr_path, overwrite=True, shard_coefficients=(1, 3, 2, 2))
    a0 = _level(zarr_path)
    assert a0.shards == (1, 24, 32, 32)
    assert all(_level(zarr_path, k).shards is not None for k in ("1", "2"))


def test_a_sharded_source_stays_sharded_through_a_deferred_downscale(data, tmp_path):
    src = tmp_path / "src.zarr"
    IO().write_pyramid(_zarr_pyr(data), str(src), overwrite=True,
                       shard_coefficients=(2, 3, 2, 2))
    out = tmp_path / "out.zarr"
    IO().write_pyramid(IO().read_pyramid(str(src)).downscale(n_layers=3, defer=True),
                       str(out), overwrite=True)
    assert _level(out).shards == (2, 24, 32, 32)
    assert all(_level(out, k).shards is not None for k in ("1", "2"))


def test_options_the_engine_cannot_honour_are_refused(data, zarr_path):
    with pytest.raises(ValueError, match="dyna write engine"):
        IO().write_pyramid(_zarr_pyr(data), zarr_path, overwrite=True, backend="sync",
                           shard_coefficients=(1, 1, 1, 1))
    with pytest.raises(ValueError, match="zarr v3"):
        IO().write_pyramid(_zarr_pyr(data, "0.4"), zarr_path, overwrite=True,
                           shard_coefficients=(1, 1, 1, 1))


def test_deferred_downscale_of_an_in_memory_zarr_keeps_its_chunks(data, zarr_path, route):
    IO().write_pyramid(_zarr_pyr(data).downscale(n_layers=3, defer=True), zarr_path,
                       overwrite=True)
    assert set(route) == {"dyna"}
    assert tuple(_level(zarr_path).chunks) == CHUNKS
    assert len(list(zarr.open_group(zarr_path, mode="r").array_keys())) == 3


def test_tensorstore_engine_writes(data, zarr_path):
    # regression: raised NameError (is_dask_array) on every non-dyna source
    IO().write_pyramid(_zarr_pyr(data), zarr_path, overwrite=True, backend="tensorstore")
    np.testing.assert_array_equal(_level(zarr_path)[...], data)


@pytest.mark.parametrize("multiprocessing", [False, True])
def test_simple_levels_cascade_with_pixels_identical_to_striding_l0(tmp_path, multiprocessing):
    # 'simple' levels are derived from the stored PARENT level (one L0 read for all
    # levels); stride composes exactly, so each level must equal L0[::2**i] on z/y/x.
    # Odd extents exercise the ceil-sized edge. multiprocessing derives from L0 instead.
    odd = np.random.default_rng(1).integers(0, 999, size=(2, 23, 41, 37), dtype="uint16")
    arr = zarr.create_array(store=zarr.storage.MemoryStore(), shape=odd.shape,
                            chunks=(1, 8, 16, 16), dtype=odd.dtype)
    arr[...] = odd
    pyr = Pyramid().from_arrays([arr], axis_order=AXES, version="0.5")
    out = tmp_path / "out.zarr"
    IO().write_pyramid(pyr.downscale(n_layers=4, scale_factor=(1, 2, 2, 2), defer=True),
                       str(out), overwrite=True, use_multiprocessing=multiprocessing)
    for i in range(4):
        s = 2 ** i
        np.testing.assert_array_equal(_level(out, str(i))[...], odd[:, ::s, ::s, ::s])


def test_mean_levels_still_derive_from_l0(data, tmp_path, route):
    out = tmp_path / "out.zarr"
    IO().write_pyramid(_zarr_pyr(data).downscale(n_layers=3, downscale_method="mean",
                                                 defer=True), str(out), overwrite=True)
    # base + ONE write of both coarser levels (from L0), not one write per level
    assert route == ["dyna", "dyna"]
    expect = data[:, :12, :20, :20].reshape(2, 6, 2, 10, 2, 10, 2).mean(axis=(2, 4, 6))
    np.testing.assert_allclose(_level(out, "1")[:, :6, :10, :10], expect, atol=1)


def test_multiprocessing_writes_every_level_in_its_own_process(data, tmp_path):
    # repaired: it was unreachable (a read pyramid lost `gr` in the storage-chunk
    # restore, so it always raised "requires a disk-backed pyramid")
    src = tmp_path / "src.zarr"
    IO().write_pyramid(_zarr_pyr(data).downscale(n_layers=3, defer=True), str(src),
                       overwrite=True)
    pyr = IO().read_pyramid(str(src)) * 1            # an op chain: pickled to the workers
    out = tmp_path / "out.zarr"
    IO().write_pyramid(pyr, str(out), overwrite=True, use_multiprocessing=True)
    for k in ("0", "1", "2"):
        np.testing.assert_array_equal(_level(out, k)[...], _level(src, k)[...])
    assert tuple(_level(out).chunks) == CHUNKS


def test_multiprocessing_is_refused_where_it_cannot_run(data, zarr_path):
    with pytest.raises(ValueError, match="dyna engine"):
        IO().write_pyramid(_zarr_pyr(data), zarr_path, overwrite=True, backend="sync",
                           use_multiprocessing=True)
