# ome_zarr_pyramid

Read, write and downscale **OME-Zarr (NGFF)** image pyramids through a single, lazy
`Pyramid` object, memory-bound by design, so it scales from a small tile to
large-scale volumes with the same code.

- **One object, all levels.** A `Pyramid` wraps every resolution level as a lazy
  array ([dyna-zarr](https://pypi.org/project/dyna-zarr/) `DynamicArray`s; `dask`
  arrays if you build it from dask) plus the full NGFF metadata (axes, scales, units,
  omero, translations).
- **Lazy & memory-bound.** Nothing is computed until you write or `.compute()`. The
  writer streams region by region; the whole volume is never held in RAM.
- **NGFF-native.** Reads/writes multiscales v0.4 and v0.5 (with `dimension_names`),
  sharded or not, local or on object storage (`s3://`, `gs://`, `az://`).
- **Deferred, progressive downscaling.** `downscale()` records a plan and builds
  nothing; the writer streams the base to disk **once** and derives each coarser level
  from the stored level above it. An expensive base (e.g. a segmentation) is computed
  a single time, not once per level.
- **Elementwise algebra.** `Pyramid` objects support arithmetic, comparison and
  bitwise operators across every level at once (`img > 128`, `a * b`, `~mask`).

> Image **processing** (filters, segmentation, features, morphology, clustering, …)
> lives in the sibling package **`ome_zarr_pro`**, which builds on this one.

## Install

```bash
pip install ome_zarr_pyramid              # core: zarr, tensorstore, dyna-zarr (no dask)
pip install "ome_zarr_pyramid[dask]"      # + dask: build pyramids from dask arrays
pip install "ome_zarr_pyramid[remote]"    # + obstore: s3://, gs://, az:// stores
```

## Quickstart

```python
from ome_zarr_pyramid import Pyramid, IO

pyr = IO().read_pyramid("image.ome.zarr")           # -> Pyramid (lazy, nothing loaded)
print(pyr.axes, pyr.nlayers)                        # 'tczyx', 5
print(pyr.base_array.shape, pyr.base_array.dtype)   # full-resolution level 0

IO().write_pyramid(pyr, "copy.ome.zarr", overwrite=True)
```

### Everything is a `Pyramid`

**Every operation returns a new `Pyramid`.** Selecting (`isel`, `select_levels`), the
elementwise operators (`+ - * / > == & ~ …`), `downscale` and `rechunk` all produce a
fresh, lazy `Pyramid`, with **all resolution levels and metadata preserved**, so they
compose and chain naturally. Nothing is materialised until you `.compute()` an array or write
the pyramid. Throughout this README, `pyr` and any variable ending in `_pyr` are
`Pyramid` objects.

```python
mask_pyr = (IO().read_pyramid("image.ome.zarr").isel(c=0) > 128)   # Pyramid -> Pyramid -> Pyramid
IO().write_pyramid(mask_pyr.downscale(n_layers=4), "mask.ome.zarr", overwrite=True)
```

## Inspect the pyramid & its metadata

```python
pyr = IO().read_pyramid("image.ome.zarr")

pyr.axes                       # 'tczyx' (axis order)
pyr.meta.resolution_paths      # ['0', '1', '2', '3']
pyr.meta.unit_list             # ['second', None, 'micrometer', 'micrometer', 'micrometer']

for p in pyr.meta.resolution_paths:
    arr = pyr.dynamic_arrays[p]                     # lazy array for this level
    scale = pyr.meta.get_scale(p)                   # physical pixel size per axis
    print(p, arr.shape, arr.chunks, scale)

import numpy as np
level0 = np.asarray(pyr.base_array)                 # materialise level 0 to numpy
```

## Select regions, channels and levels

Each of these returns a **new `Pyramid`** (a sub-pyramid), not a bare array:

```python
channel0_pyr = pyr.isel(c=0)                 # -> Pyramid: channel 0 (drops the 'c' axis)
zrange_pyr   = pyr.isel(z=slice(10, 40))     # -> Pyramid: a z-range (scale/translation updated)
frame_pyr    = pyr.isel(t=0, c=1)            # -> Pyramid: one timepoint, one channel

top3_pyr     = pyr.select_levels(0, 1, 2)    # -> Pyramid: the finest three resolution levels
skip_pyr     = pyr.select_levels(0, 2, 4)    # -> Pyramid: an arbitrary (non-contiguous) subset
```

`isel` is xarray-style: an **int** selects one position and drops that axis, a
**slice** keeps a strided sub-range, applied across every resolution level, with the
coordinate metadata (scale, translation, dropped axes, omero channels) updated to match.

## Elementwise algebra (all levels, lazy)

`Pyramid` behaves like an array under operators, but each operation returns **another
`Pyramid`** (every level transformed lazily, metadata preserved), so results chain:

```python
mask_pyr   = pyr > 128                   # -> Pyramid: boolean mask (thresholding)
scaled_pyr = pyr * 2 + 10                # -> Pyramid: arithmetic with scalars
diff_pyr   = a_pyr - b_pyr               # -> Pyramid: combine two aligned pyramids
band_pyr   = (pyr > 50) & (pyr < 200)    # -> Pyramid: bitwise combine of two mask pyramids
inv_pyr    = ~mask_pyr                    # -> Pyramid: unary ops (~, -, abs())

IO().write_pyramid(mask_pyr, "mask.ome.zarr", overwrite=True)
```

Supported: `+ - * / // % **`, `< <= > >= == !=`, `& | ^`, and unary `- + abs() ~`.
Operands may be scalars or other `Pyramid` objects; the result is always a `Pyramid`.

## Downscale and write (memory-bound, base computed once)

```python
pyr = IO().read_pyramid("image.ome.zarr")

# downscale() is DEFERRED and returns a Pyramid: it records a plan, builds nothing.
full_pyr = pyr.downscale(n_layers=4)                 # -> Pyramid (exactly 4 levels)
full_pyr = pyr.downscale(min_dimension_size=128)     # -> Pyramid (until largest axis < 128)

# The write streams level 0 once, then builds the coarser levels from what it stored.
IO().write_pyramid(full_pyr, "out.ome.zarr", overwrite=True)
```

Downsampling defaults to **stride** (nearest-neighbour, `downscale_method="simple"`)
at an **isotropic 2x on the spatial axes** (`z`/`y`/`x`; `t`/`c` stay 1). On a pyramid
that already has levels, the per-axis factors are instead **derived from those levels**
so an anisotropic or irregular source round-trips unchanged — pass `scale_factor=`
explicitly to override, and see `Pyramid.derive_downscale_plan()`.

With stride, each level is built from the stored level above it, so level 0 is read
back only once. `downscale_method="mean"` and `"median"` build every level from level 0
instead, because averaging twice is not the same as averaging once.

This matters when level 0 is an expensive lazy graph: the base is computed a **single**
time (during its own write) instead of being recomputed for every pyramid level.

## Control storage chunking (independent of processing)

Storage chunk shape is decoupled from whatever chunking an upstream operation imposed.
Pass a chunk **shape** (used as is) or a target chunk **size in MB** (approximate:
dyna-zarr picks a dtype-aware chunk that fits the size and lines up with the source's
chunks):

```python
full_pyr = pyr.downscale(n_layers=3)                 # 'tczyx', 3 levels

# target MB per chunk: one value for all levels, or one per level
IO().write_pyramid(full_pyr, "out.ome.zarr", overwrite=True, chunk_size_mb=1.0)
IO().write_pyramid(full_pyr, "out.ome.zarr", overwrite=True, chunk_size_mb=(4, 1, 0.5))

# exact chunk shape, one entry per axis: one shape for all levels, or one per level
IO().write_pyramid(full_pyr, "out.ome.zarr", overwrite=True, chunk_shape=(1, 1, 32, 256, 256))
IO().write_pyramid(full_pyr, "out.ome.zarr", overwrite=True,
                   chunk_shape={0: (1, 1, 32, 256, 256),
                                1: (1, 1, 32, 128, 128),
                                2: (1, 1, 16, 64, 64)})

# or rechunk the Pyramid itself -> returns a new Pyramid
rechunked_pyr = pyr.rechunk(chunk_size_mb=2.0)       # -> Pyramid
```

A per-level dict or sequence needs an entry for every level.

When neither is given, a pyramid read from disk is written back with its **original**
on-disk chunking.

## Build a pyramid from your own arrays

```python
import dask.array as da
from ome_zarr_pyramid import Pyramid, IO

lvl0 = da.zeros((2, 512, 512), chunks=(1, 256, 256), dtype="uint16")  # (c, y, x)

image_pyr = Pyramid().from_arrays(       # -> Pyramid
    [lvl0],
    axis_order="cyx",
    unit_list=[None, "micrometer", "micrometer"],
    scales=[[1, 0.325, 0.325]],      # physical pixel size at level 0
    version="0.4",                    # NGFF version ('0.4' or '0.5')
    name="my_image",
)

IO().write_pyramid(image_pyr.downscale(n_layers=3), "my_image.ome.zarr", overwrite=True)
```

A level can be any array-like object: NumPy, zarr, dask, TensorStore, a
`micro_reader.Image`, or your own reader with `shape`, `dtype` and `__getitem__`. It is
read region by region when you write, so a large file never has to fit in memory:

```python
import micro_reader

image = micro_reader.open("image.czi").images[0].as_axes("canonical", samples="channels")
image_pyr = Pyramid().from_arrays([image], axis_order="tczyx", version="0.5")
```

## Write options

Every write goes through [dyna-zarr](https://pypi.org/project/dyna-zarr/) by default
(`backend="auto"`). Lazy levels are streamed region by region. Levels built from dask
arrays are pushed by dask into the same writer.

```python
IO().write_pyramid(pyr, "out.ome.zarr", overwrite=True,     # pyr: 'tczyx'
                   compressor="zstd",                  # codec (default: the input's own)
                   shard_coefficients=(1, 1, 1, 4, 4), # shards of 4x4 chunks in y/x (OME-Zarr 0.5)
                   max_workers=8,                      # regions in flight
                   region_size_mb=16)                  # size of one region
```

- A sharded input stays sharded, on every level.
- `io_backend="zarrista"` writes the bytes with the optional zarrista backend of
  dyna-zarr instead of TensorStore.
- `use_multiprocessing=True` writes each level in its own process. It is rarely
  faster: level 0 holds most of the data.

### Object storage

Use an `s3://`, `gs://` or `az://` URL and pass the store settings as
`storage_options`. Credentials come from the usual environment variables
(`AWS_ACCESS_KEY_ID`, ...).

```python
opts = {"endpoint": "https://s3.example.com", "region": "eu-west-1"}
IO().write_pyramid(pyr, "s3://bucket/out.ome.zarr", overwrite=True, storage_options=opts)
back = IO().read_pyramid("s3://bucket/out.ome.zarr", storage_options=opts)
```

The older `https://host/bucket/key` form still works for writing, with s3fs-style
options (`pip install "ome_zarr_pyramid[s3]"`).

### Other writers

`backend="sync"` (a threaded zarr-python writer) and `backend="tensorstore"` are kept
for compatibility. They are slower, local or `https://` only, and do not write
shards. Options they cannot honour raise an error rather than being ignored.

## License

MIT. See [LICENSE](LICENSE).