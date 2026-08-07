# ome_zarr_pyramid

Read, write and downscale **OME-Zarr (NGFF)** image pyramids through a single, lazy,
dask-backed `Pyramid` object, memory-bound by design, so it scales from a small tile
to large-scale volumes with the same code.

- **One object, all levels.** A `Pyramid` wraps every resolution level as a lazy
  `dask` array plus the full NGFF metadata (axes, scales, units, omero, translations).
- **Lazy & memory-bound.** Nothing is computed until you write or `.compute()`. The
  region writer streams tile-by-tile; the whole volume is never held in RAM.
- **NGFF-native.** Reads/writes multiscales v0.4 and v0.5, TensorStore or threaded
  backends, local or S3.
- **Deferred, progressive downscaling.** `downscale()` records a plan and builds
  nothing; the writer streams the base to disk **once** and derives coarser levels
  from the stored base. An expensive base (e.g. a segmentation) is computed a single
  time, not once per level.
- **Elementwise algebra.** `Pyramid` objects support arithmetic, comparison and
  bitwise operators across every level at once (`img > 128`, `a * b`, `~mask`).

> Image **processing** (filters, segmentation, features, morphology, clustering, …)
> lives in the sibling package **`ome_zarr_pro`**, which builds on this one.

## Install

```bash
pip install ome_zarr_pyramid          # core: zarr, dask, tensorstore
pip install "ome_zarr_pyramid[s3]"    # + s3fs, for writing to https:// (S3) stores
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
    arr = pyr.dask_arrays[p]                        # lazy dask array for this level
    scale = pyr.meta.get_scale(p)                   # physical pixel size per axis
    print(p, arr.shape, arr.chunksize, scale)

import numpy as np
level0 = np.asarray(pyr.base_array.compute())       # materialise level 0 to numpy
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

# The write streams level 0 once, then derives coarser levels from the on-disk base.
IO().write_pyramid(full_pyr, "out.ome.zarr", overwrite=True)
```

This matters when level 0 is an expensive lazy graph: the base is computed a **single**
time (during its own write) instead of being recomputed for every pyramid level.

## Control storage chunking (independent of processing)

Storage chunk shape is decoupled from whatever chunking an upstream operation imposed.
Pass a chunk **shape** or a target chunk **size in MB** (isotropic: square in 2-D,
cube in 3-D, dtype-aware):

```python
# target MB per chunk (scalar, or per-level sequence / dict)
IO().write_pyramid(full_pyr, "out.ome.zarr", overwrite=True, chunk_size_mb=1.0)
IO().write_pyramid(full_pyr, "out.ome.zarr", overwrite=True, chunk_size_mb=(4, 1, 0.5))

# explicit chunk shape (one tuple for all levels, or per-level dict)
IO().write_pyramid(full_pyr, "out.ome.zarr", overwrite=True, chunk_shape=(1, 1, 256, 256))
IO().write_pyramid(full_pyr, "out.ome.zarr", overwrite=True,
                   chunk_shape={0: (1, 1, 256, 256), 1: (1, 1, 128, 128)})

# or rechunk the Pyramid itself -> returns a new Pyramid
rechunked_pyr = pyr.rechunk(chunk_size_mb=2.0)       # -> Pyramid
```

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

## Write backends & options

```python
IO().write_pyramid(pyr, "out.ome.zarr", overwrite=True,
                   backend="sync",        # 'sync' (threaded) or 'tensorstore'
                   max_workers=4)
IO().write_pyramid(pyr, "https://s3.example.com/bucket/out.ome.zarr")  # needs [s3]
```

## License

MIT. See [LICENSE](LICENSE).


```python
pyr = Pyramid().from_arrays(
  [lvl0],
  axis_order = 'zyx',
  ...
)
pyr_full = pyr.downscale(n_layers = 8)
pyr_subset = pyr_full.select_levels(range(3, 8))   # levels 3..7 (contiguous run)
pyr_subset_computed = (pyr_subset ** 2) > 0.5
IO().write_pyramid(pyr_subset_computed, "out.ome.zarr")

```