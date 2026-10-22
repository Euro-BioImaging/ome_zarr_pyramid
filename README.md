# ome_zarr_pyramid

Read, write and downscale **OME-Zarr (NGFF)** image pyramids through a single, lazy
`Pyramid` object, memory-bound by design, so it scales from a small tile to
large-scale volumes with the same code.

- **One object, all levels.** A `Pyramid` wraps every resolution level as a lazy
  array ([dyna-zarr](https://pypi.org/project/dyna-zarr/) `DynamicArray`s; `dask`
  arrays if you build it from dask) plus the full NGFF metadata (axes, scales, units,
  omero, translations).
- **Lazy & memory-bound.** Nothing is computed until you write the pyramid or read a
  level into memory (`np.asarray(pyr.base_array)`). The writer streams region by region;
  the whole volume is never held in RAM.
- **NGFF-native.** Reads/writes multiscales v0.4 and v0.5 (with `dimension_names`),
  sharded or not, local or on object storage (`s3://`, `gs://`, `az://`).
- **Deferred downscaling.** `downscale()` records a plan and builds nothing. The
  writer reads the base **once** and writes every level from that read, so an expensive
  base (e.g. a segmentation) is computed a single time, not once per level.
- **Elementwise algebra.** `Pyramid` objects support arithmetic, comparison and
  bitwise operators across every level at once (`img > 128`, `a * b`, `~mask`).
- **Collections.** Plates, wells, a file's series and other groups of images are one
  `PyramidCollection`: read from OME-Zarr or from a microscopy file, written as an NGFF
  plate, a bioformats2raw layout or an NGFF RFC 8 collection.

## Install

```bash
pip install ome_zarr_pyramid              # core: zarr, tensorstore, dyna-zarr (no dask)
pip install "ome_zarr_pyramid[dask]"      # + dask: build pyramids from dask arrays
pip install "ome_zarr_pyramid[remote]"    # + obstore: s3://, gs://, az:// stores
```

## Quickstart

```python
from ome_zarr_pyramid import Pyramid, io

pyr = io.read_pyramid("image.ome.zarr")             # -> Pyramid (lazy, nothing loaded)
print(pyr.axes, pyr.nlayers)                        # 'tczyx', 5
print(pyr.base_array.shape, pyr.base_array.dtype)   # full-resolution level 0

io.write_pyramid(pyr, "copy.ome.zarr", overwrite=True)
```

### Everything is a `Pyramid`

**Every operation returns a new `Pyramid`.** Selecting (`isel`, `select_levels`), the
elementwise operators (`+ - * / > == & ~ …`), `downscale` and `rechunk` all produce a
fresh, lazy `Pyramid`, with **all resolution levels and metadata preserved**, so they
compose and chain naturally. The lazy `Pyramid` object can be materialised in memory or 
streamed to disc with simple methods. 

Throughout this README, `pyr` and any variable ending in `_pyr` are
`Pyramid` objects, and `io` is `from ome_zarr_pyramid import io`. 

```python
mask_pyr = (io.read_pyramid("image.ome.zarr").isel(c=0) > 128)   # Pyramid -> Pyramid -> Pyramid
io.write_pyramid(mask_pyr.downscale(n_layers=4), "mask.ome.zarr", overwrite=True)
```

## Inspect the pyramid & its metadata

```python
pyr = io.read_pyramid("image.ome.zarr")

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

## Slicing and level selection

Each of these returns a **new `Pyramid`** (a sub-pyramid), not a bare array:

```python
channel0_pyr = pyr.isel(c=0)                 # -> Pyramid: channel 0 (drops the 'c' axis)
zrange_pyr   = pyr.isel(z=slice(10, 40))     # -> Pyramid: a z-range (scale/translation updated)
frame_pyr    = pyr.isel(t=0, c=1)            # -> Pyramid: one timepoint, one channel

top3_pyr     = pyr.select_levels(0, 1, 2)    # -> Pyramid: the finest three resolution levels
skip_pyr     = pyr.select_levels(0, 2, 4)    # -> Pyramid: an arbitrary (non-contiguous) subset
```

`isel` is xarray-style: an **int** selects one position and drops that axis, a
**slice** keeps a strided sub-range, **applied across every resolution level**, with the
coordinate metadata (scale, translation, dropped axes, omero channels) updated to match.

`Pyramid` also supports NumPy-style indexing. It does the same as `isel`, but via positional
indexing:

```python
zrange_pyr = pyr[:, :, 10:40]                # 'tczyx': same as pyr.isel(z=slice(10, 40))
frame_pyr  = pyr[0, 1]                       # same as pyr.isel(t=0, c=1)
tail_pyr   = pyr[..., 100:200]               # ... fills the axes in between
```

The output of isel or numpy-style indexing is also a `Pyramid`. The indices passed are applied
to the level-0 directly and `adapted to each lower resolution level based on the pyramid's scale factors`.

## Elementwise algebra (all levels, lazy)

`Pyramid` behaves like an array under operators, but each operation returns **another
`Pyramid`** (every level transformed lazily, metadata preserved), so results chain:

```python
mask_pyr   = pyr > 128                   # -> Pyramid: boolean mask (thresholding)
scaled_pyr = pyr * 2 + 10                # -> Pyramid: arithmetic with scalars
diff_pyr   = a_pyr - b_pyr               # -> Pyramid: combine two aligned pyramids
band_pyr   = (pyr > 50) & (pyr < 200)    # -> Pyramid: bitwise combine of two mask pyramids
inv_pyr    = ~mask_pyr                    # -> Pyramid: unary ops (~, -, abs())

io.write_pyramid(mask_pyr, "mask.ome.zarr", overwrite=True)
```

Supported: `+ - * / // % **`, `< <= > >= == !=`, `& | ^`, and unary `- + abs() ~`.
Operands may be scalars or other `Pyramid` objects; the result is always a `Pyramid`.

## Downscale and write (memory-bound, base computed once)

```python
pyr = io.read_pyramid("image.ome.zarr")

# downscale() is DEFERRED and returns a Pyramid: it records a plan, builds nothing.
full_pyr = pyr.downscale(n_layers=4)                 # -> Pyramid (exactly 4 levels)
full_pyr = pyr.downscale(min_dimension_size=128)     # -> Pyramid (largest axis stays >= 128)

# The write reads level 0 once and writes the coarser levels from the same reads.
io.write_pyramid(full_pyr, "out.ome.zarr", overwrite=True)
```

The write works in **one pass**. It reads level 0 block by block, and from each block it
computes and writes level 0 and every coarser level at once. So each block is read only
once, and an expensive lazy level 0 (e.g. a segmentation) is computed a single time, not
once per level. The threads (`max_threads`) work on different blocks in parallel. If the
smallest levels do not fit in the memory budget in the same pass, they are built afterwards
from what was already written to disk, with the same pixels.

`downscale()` **rebuilds the pyramid from level 0**. If `downscale()` is run on
a source pyramid that already has levels, the existing coarse levels are replaced by 
the newer ones produced by `downscale()`. In this scenario, the scale factors `downscale` uses by default 
are derived from **the shape ratios between the source levels** and used for recreating the new
coarse levels from the level-0. This ensures that the input and output pyramids have consistent scale 
factors. To override this behaviour and enforce a specific scale factor, pass `scale_factor=` explicitly. 
See `Pyramid.derive_downscale_plan()`.

In the scenario that the source has no coarse levels, `downscale()` will default to an **isotropic 2x on the 
spatial axes** (`t` and `c` remain 1). 

Downscaling uses window method defaulting to **stride** (nearest-neighbour, `downscale_method="simple"`).

```python
mask_pyr = pyr.isel(c=0) > 128
io.write_pyramid(mask_pyr, "a.ome.zarr")               # each level: its own level, thresholded
io.write_pyramid(mask_pyr.downscale(), "b.ome.zarr")   # each level: level 0's mask, downscaled
```

Operating on existing resolution levels is fine for arithmetics such as `pyr * 2`. For masks, labels and 
most other operations, it is recommended to use `downscale()`: otherwise the coarser levels may not match level 0. 

Methods:

* `"simple"`: stride. Works with both raw data and labels.
* `"mean"`: works (and recommended) with raw data. Does **not** work with labels.
* `"mode"`: also works with labels.
* `"median"`, `"min"`, `"max"`: supported but generally not recommended.

They give the same pixels as TensorStore's `downsample`.

Each coarser level is placed where its pixels come from. With `"simple"`, a level's
first pixel is level 0's first pixel, so every level has level 0's translation. With
the other methods, a level's first pixel is the average (or median, ...) of a window
of level-0 pixels, so it sits at the window's centre: the level's translation is level
0's plus half the difference of the two pixel sizes. For 0.5 µm pixels at (5, 3) µm,
downscaled twice by 2 with `"mean"`, the levels are at (5, 3), (5.25, 3.25) and
(5.75, 3.75) µm. Levels read from a store keep the translations stored there.

`downscale_backend` chooses where the window methods run. Both produce the same pixels:

```python
pyr.downscale(downscale_method="mean")                            # TensorStore (default)
pyr.downscale(downscale_method="mean", downscale_backend="dyna")  # dyna's coarsen
```

## Control storage chunking (independent of processing)

If no chunk parameter is supplied, operation chains will carry the source pyramid's **original** 
on-disk chunking, and `write_pyramid(input_pyr, path.zarr)` will simply create the output store
with `input_pyr`'s chunk shape. Supplying explicit output chunk parameters override this behaviour. 

Different options are:

1. Pass a chunk **shape**. This is a tuple representing chunk shape or a dict of tuples representing chunk
shape per level. These are used as **exact** values the output chunks will have, and therefore, the user
is responsible for any misalignment between input and output chunks.

2. Alternatively, pass a target chunk **size in MB** or a sequence of chunk sizes per level. dyna-zarr picks 
a dtype-aware chunk that *approximately* fits the size and will handle the alignment between this chunk and 
the source's chunks.

```python
full_pyr = pyr.downscale(n_layers=3)                 # 'tczyx', 3 levels

# target MB per chunk: one value for all levels, or one per level
io.write_pyramid(full_pyr, "out.ome.zarr", overwrite=True, chunk_size_mb=1.0)
io.write_pyramid(full_pyr, "out.ome.zarr", overwrite=True, chunk_size_mb=(4, 1, 0.5))

# exact chunk shape, one entry per axis: one shape for all levels, or one per level
io.write_pyramid(full_pyr, "out.ome.zarr", overwrite=True, chunk_shape=(1, 1, 32, 256, 256))
io.write_pyramid(full_pyr, "out.ome.zarr", overwrite=True,
                   chunk_shape={0: (1, 1, 32, 256, 256),
                                1: (1, 1, 32, 128, 128),
                                2: (1, 1, 16, 64, 64)})

# or rechunk the Pyramid itself -> returns a new Pyramid
rechunked_pyr = pyr.rechunk(chunk_size_mb=2.0)       # -> Pyramid
```

A per-level dict or sequence needs an entry for every level.

## Build a pyramid from your own arrays

```python
import dask.array as da
from ome_zarr_pyramid import Pyramid, io

lvl0 = da.zeros((2, 512, 512), chunks=(1, 256, 256), dtype="uint16")  # (c, y, x)

image_pyr = Pyramid().from_arrays(       # -> Pyramid
    [lvl0],
    axis_order="cyx",
    unit_list=[None, "micrometer", "micrometer"],
    scales=[[1, 0.325, 0.325]],      # physical pixel size at level 0
    version="0.5",                    # NGFF version: '0.5' (zarr v3, the default) or '0.4'
    name="my_image",
)

io.write_pyramid(image_pyr.downscale(n_layers=3), "my_image.ome.zarr", overwrite=True)
```

A level can be any array-like object: NumPy, zarr, dask, TensorStore, a
`micro_reader.Image` (see micro-reader), or your own reader with `shape`, `dtype` and `__getitem__`. 
Executing a write command (such as `io.write_pyramid()`) on a lazy level object will read the level 
region by region from its source and stream it to the output store on disc, enabling memory-bounded
IO operations. 

```python
import micro_reader

image = micro_reader.open("image.czi", samples="channels").images[0]
# Here, image is a lazy micro_reader.Image object. It can be read region by region.
image_pyr = Pyramid.from_image(image)
# Its stored resolution levels, pixel sizes, stage position and channels become the
# pyramid's levels and metadata; the levels stay lazy and can be streamed to disc.
```

For every image of a file at once, `PyramidCollection.from_file(micro_reader.open(...))`
gives one lazy pyramid per image (see [Collections](#collections-plates-series-and-other-groups-of-images)).
Each pyramid is keyed by the image's index in the file, as micro-reader gives it:

```python
coll = PyramidCollection.from_file(micro_reader.open("sample.czi"))
list(coll)                    # [(('series', 0),), (('series', 1),)]
pyr = coll[{"series": 1}]
coll.select(series=0)         # -> PyramidCollection
coll.write("out", "sample")   # out/sample_series0.ome.zarr, out/sample_series1.ome.zarr
coll.tree                     # the file as nodes: its plates, and its series
```

A file's series are numbered by the index `series` (micro-reader 0.1.0 and later).
`select` with an index name the collection does not have raises a `KeyError` that lists
the names it has; a known name with a value no image has gives an empty collection.
Any other object with `images` works too, with whatever index names its images carry.

In `coll.tree`, named after the file (`sample`), a series that holds several images
(the tiles of a scan, several views) is a collection of its own; a series of one image
is just that image. A plate's wells hold their fields the same way.
`coll.write_collection("sample.zarr")` writes this tree as an RFC 8 collection. Read
back, its images are keyed `("series", 0)`, `("series", 1)`, ... in tree order, tiles
included: which tile an image was is not kept yet.

## Collections: plates, series and other groups of images

A `PyramidCollection` holds many pyramids: the wells and fields of a plate, the series
of a file, or any other group of images. Opening one reads its metadata only; each
image is opened when you use it.

```python
from ome_zarr_pyramid import PyramidCollection

plate_coll = PyramidCollection.from_zarr("plate.ome.zarr")   # an NGFF plate; no image read yet
plate = plate_coll.plates[0]
plate.rows, plate.columns                     # ('A', 'B', ...), ('1', '2', ...)
well = plate.wells[plate.well_index("B3")]    # Well(row=1, column=2, name='B3')
pyr = well.fields[0].image                    # -> Pyramid, opened now
```

Plates are navigated as in micro-reader: rows and columns count from 0,
`plate.wells[w]` is the well with index `"well": w`, and `well.fields[f]` its f-th field.
When a field has one image per acquisition, pick one first:
`well.fields[0].select(acquisition=1).image`.

Every image also has a flat key, the same one micro-reader gives a plate image:

```python
list(plate_coll)[0]          # (('plate', 0), ('acquisition', 0), ('well', 0), ('field', 0))
pyr = plate_coll[{"plate": 0, "acquisition": 0, "well": 3, "field": 0}]
second_run = plate_coll.select(acquisition=1)                # -> PyramidCollection
```

### Write a plate

A plate opened with micro-reader (Harmony, MetaXpress, IN Cell, CellVoyager, ...) or
read from OME-Zarr is written as an NGFF plate, one image after the other:

```python
import micro_reader

plate_coll = PyramidCollection.from_file(micro_reader.open("Images/Index.idx.xml"))
plate_coll.write_plate("plate.ome.zarr")                       # NGFF 0.5
plate_coll.write_plate("plate_v04.ome.zarr", version="0.4")    # NGFF 0.4
```

The write options of `io.write_pyramid` (`max_threads`, `compressor`,
`memory_budget_mb`, ...) apply to every image. The plate and well metadata are written
last, so a plate that stops half way is not mistaken for a complete one.

An NGFF plate holds one image per field. A field with several images (the tiles of a
tiled scan, several views) raises `PlateFieldError`, which names the ways out: open the
file with `tiles="mosaic"` (micro-reader 0.1 or later) so each field is one image; write
`version="rfc8"` (experimental), which keeps every tile; or, for a plate read from
NGFF, join a field's images into one with pyrops. Images that are not in the plate (a
slide's label or preview) are left out.

### Series (the bioformats2raw layout)

```python
series_coll = PyramidCollection.from_zarr("converted.ome.zarr")  # series 0, 1, ...
pyr = series_coll[{"series": 1}]
series_coll.write_series("copy.ome.zarr")      # the same layout, with its OME-XML
```

Any collection can be written this way, for example all series of a file opened with
micro-reader.
A series that holds several images (tiles, views) is written as one series per image,
in key order, with a warning.

The `OME` group always gets an OME-XML with one `Image` per series, in series order:
the one of the layout the images were read from while it still fits, else a short one
ozp writes (size, pixel type and dimension order of each series). When it replaces the
source's, it warns. An `ome_xml=` you pass must have one `Image` per series, or the
write stops with `OMEXMLError` before anything is written.

### NGFF RFC 8 collections

[RFC 8](https://ngff.openmicroscopy.org/rfc/8/index.html) describes plates, series and
any other group of images as one tree of nodes. It is still a proposal; ozp reads it
and writes it when you ask for `version="rfc8"`:

```python
plate_coll.write_plate("plate_rfc8.zarr", version="rfc8")            # a plate as RFC 8
plate_coll.write_collection("out.zarr", document="collection.json")   # the tree in its own .json
coll = PyramidCollection.from_zarr("collection.json")                 # a .json or a zarr group
coll.tree                                                             # the nodes and their attributes
```

## Write options

Every write goes through [dyna-zarr](https://pypi.org/project/dyna-zarr/). Lazy levels
are streamed region by region.

```python
io.write_pyramid(pyr, "out.ome.zarr", overwrite=True,     # pyr: 'tczyx'
                   compressor="zstd",                  # codec (default: the input's own)
                   shard_coefficients=(1, 1, 1, 4, 4), # shards of 4x4 chunks in y/x (OME-Zarr 0.5)
                   max_threads=8,                      # regions at once, one per thread
                   region_size_mb=16)                  # size of one region
```

- A sharded input stays sharded, on every level.
- `memory_budget_mb=1024` keeps the write within that much memory, on top of what your
  process already holds. Shards can be much larger than that: **shards are also written chunk
  by chunk**. For example, a 17.6 GB stack of 2347 planes, written as a 6-level pyramid
  with 960³ shards, stayed under 1 GB with a 1.5 GB budget. It does not apply to levels
  built from dask arrays (dask's own scheduler writes those), and raises there.
- `io_backend="zarrista"` writes the bytes with the optional zarrista backend of
  dyna-zarr instead of TensorStore.
- Renamed in 0.1.0: `max_workers` is now `max_threads`, and `use_multiprocessing` is now
  `max_downscale_processes`. The old names still work, with a `FutureWarning`. The old
  `backend="sync"` and `backend="tensorstore"` write engines are gone: asking for them raises.
- `window_from_data=True` sets each channel's display window (`start`/`end` in the omero metadata) 
  to the min/max of level 0 as written, measured during the write itself, so the output is
  not read again. The window's `min`/`max` stay as they are. For percentiles or
  another level, use `Pyramid.set_display_range()` before writing.

### OME-Zarr 0.4, 0.5 or 0.6

A pyramid is written in its own version (the one it was read or built as).
Built in ozp (`from_array`, `from_arrays`, `from_label_arrays`, `from_image`,
`PyramidCollection.from_file`), it is 0.5 unless you pass `version="0.4"` or
`version="0.6"`. `to_version` gives the same pyramid in another one:

```python
io.write_pyramid(pyr.to_version("0.4"), "out_v04.ome.zarr")    # 0.4, zarr v2
io.write_pyramid(pyr.to_version("0.6"), "out_v06.ome.zarr")    # 0.6, zarr v3
```

Going to 0.4, the levels are written without shards (zarr v2 has none). When the
input's codec does not exist in the format written (bz2 or plain lz4 in 0.5), or is
one the writer does not know (zlib, lzma), blosc-lz4 is written instead, with a
warning; pass `compressor=` to choose another.

ozp reads and writes 0.4, 0.5 and 0.6. In 0.6 a plate, well or series can hold a
`scene` that places its images; ozp writes one where the images share a coordinate
system, and places the images of a scene written by another tool. A version ozp does
not know raises `UnsupportedNGFFVersion` (from `ome_zarr_pyramid`), which names the
version found and the versions ozp reads. NGFF RFC 8 collections (version `"0.x"`) open
as before.

### Check a store

Every write checks the metadata it saves (axes, scales, translations, 0.6 scenes) and
stops with `NGFFValidationError` when NGFF does not allow it. To check a whole store
against the official NGFF schemas, install the `validate` extra
(`pip install 'ome_zarr_pyramid[validate]'`):

```python
import ome_zarr_pyramid as ozp

problems = ozp.validate("out.ome.zarr")         # [] when the store is valid
io.write_pyramid(pyr, "out.ome.zarr", validate=True)    # raises if the result is not
```

`validate` checks each group against the schemas of the version it declares and where
each group sits, which no schema checks: for example an `OME` group outside a
bioformats2raw layout. The strict schemas check the spec's recommendations too (a plate's
name, an image's name, a label's colors): `ozp.validate(path, strict=True)` lists those,
and `validate=True` warns about them (`NGFFValidationWarning`) instead of raising. A plate
copied from another tool often misses some; ozp fills the ones it can derive. It works on `s3://` and other object-store URLs
too (`storage_options=`). `write_plate`, `write_series` and `PyramidCollection.write`
also take `validate=True`.

### Object storage

Use an `s3://`, `gs://` or `az://` URL and pass the store settings as
`storage_options`. Credentials come from the usual environment variables
(`AWS_ACCESS_KEY_ID`, ...).

```python
opts = {"endpoint": "https://s3.example.com", "region": "eu-west-1"}
io.write_pyramid(pyr, "s3://bucket/out.ome.zarr", overwrite=True, storage_options=opts)
back = io.read_pyramid("s3://bucket/out.ome.zarr", storage_options=opts)
```

The older `https://host/bucket/key` form still works for writing: it is written as
`s3://bucket/key` with `https://host` as the endpoint.


## License

MIT. See [LICENSE](LICENSE).