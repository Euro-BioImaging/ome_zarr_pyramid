# Changelog

## 0.3.1 (2026-10-21)

### Fixes

- A single image written with an OME-XML (`write_pyramid(ome_xml=...)`,
  `update_metadata(ome_xml=...)`, and so eubi-bridge 0.3.0's outputs) no longer gets an
  `OME` zarr group. 0.3.0 wrote one with `series: ["0"]` under the image root, copied from
  the bioformats2raw layout: NGFF has an `OME` group only under a bioformats2raw root,
  and its `series` named the level-0 array instead of an image group. Each group still
  validated on its own, so ome-zarr-models, ome-zarr-py and the schemas passed the stores,
  but a validator that walks the whole hierarchy rejects them. The XML is now the plain
  file `OME/METADATA.ome.xml`. Pixels and the image's own metadata were never affected.
  `write_series` (a bioformats2raw layout) keeps its `OME` group, as the spec has it.
- `update_metadata(path)` repairs a store 0.3.0 wrote that way: it removes the stray
  group's `zarr.json` (zarr v2: `.zgroup`, `.zattrs`), keeps `OME/METADATA.ome.xml`, and
  warns. No pixel is read or rewritten.
- Every written multiscale names its downscaling `type` (NGFF SHOULD; the strict schemas
  require it): the downscale method when ozp made the levels (`"simple"`, `"mean"`, ...),
  the one read with the image, else `"unknown"` (levels given as they are, a single level).

### Deprecated

- `write_pyramid(ome_series=...)` / `update_metadata(ome_series=...)` have no effect (a
  single image gets no `OME` group) and give a `DeprecationWarning`.

### Tests

- `tests/ngff_validation.py` checks every kind of written store as a whole: each group
  against the official NGFF 0.4 / 0.5 schemas of every role it has, strict ones included
  (vendored in `tests/ngff_schemas`), and the hierarchy no schema checks (an `OME` group
  only under a bioformats2raw root with `series` naming image groups, datasets are arrays,
  labels, wells and fields where their parents list them). Known gap it reports: a label
  written without `colors` fails `strict_label` only.

## 0.3.0 (2026-10-21)

### Breaking changes

- `Pyramid().from_array`, `from_arrays` and `from_label_arrays` now build NGFF 0.5
  pyramids (zarr v3) when no `version` is given; before, they built 0.4 (zarr v2). Pass
  `version="0.4"` to keep writing 0.4. With this, 0.5 is the default everywhere ozp
  makes an image itself: these three, `Pyramid.from_image`,
  `PyramidCollection.from_file`, and `write_plate` / `write_series` of a collection
  that was not read from an NGFF store and whose images do not all share one version
  (when they do, that version is written). A store read as 0.4 or 0.5 is still written back
  in its own version when `version` is not given (`io.write_pyramid`, `write_plate`,
  `write_series`), and in the one asked for otherwise. ozp does not write NGFF 0.6.
- Opening a store that declares an NGFF version ozp does not read now raises
  `UnsupportedNGFFVersion` (exported from `ome_zarr_pyramid`). Before, a 0.6 store was
  read as if it were 0.4 on zarr v2. This covers images, plates, wells and series
  layouts, through `io.read_pyramid`, `Pyramid().from_ngff`,
  `PyramidCollection.from_zarr` and the plate and well readers. The error names the
  version found and the versions ozp reads (0.4 and 0.5), and comes from the store's
  metadata, before any image data is read. Only the version the store declares counts:
  a 0.5 store relabelled "0.6" is refused too, and so is a store declaring an older
  version such as 0.3. A plate is refused when one of its wells declares such a
  version, when the plate is read (not when an image of that well is first used). A
  label declaring such a version is left out of its image's `labels` with a
  `UserWarning` (see Fixes); `io.read_labels` raises for it. NGFF RFC 8 collections
  (version "0.x") open as before.
- `UnsupportedNGFFVersion` is a subclass of `InvalidCollection` (and so of
  `ValueError`): the plate, well and series readers used to raise `InvalidCollection`
  for a version they did not read, so code catching that keeps working.
- `write_plate` raises `PlateFieldError` (a `ValueError`, exported from
  `ome_zarr_pyramid`) when a field of the plate holds several images (tiles, views,
  parts), in NGFF 0.4 and 0.5. Before, the images were written as separate fields of
  their well with a warning, so viewers showed one tile as "the" field. The message
  names the ways out, in this order: open the file with `tiles="mosaic"`
  (micro-reader 0.1 or later; micro-reader 0.0.4 does not have it) so each field is one
  image; write `version="rfc8"` (experimental),
  which keeps every tile; for an NGFF source, join the images with pyrops. Wells built
  by hand without field numbers are written as before: each image is its own field.
- A series holding several images (tiles, views, ...) is written by `write_series`
  as further series, one per image, in key order (as before), and now with one
  `UserWarning` per call naming how many series held several images and which index
  names varied: the bioformats2raw layout cannot say which tile or view an image is.
- `write_series` always writes the `OME` group and `OME/METADATA.ome.xml`, with one
  `Image` per series group in group order. Without `ome_xml=`, it is the source
  layout's when the written series are still the source's, in its order; otherwise
  ozp writes a minimal one (`ID="Image:<path>"`, the image's name, a `Pixels` element
  with sizes, pixel type and dimension order, `MetadataOnly`). A source OME-XML that
  no longer fits (a selection of a layout's series) is replaced with a `UserWarning`.
  Before, the source's was copied whatever was written, and nothing was written for a
  collection read from a microscopy file (a bioformats2raw root without OME-XML makes
  napari-ome-zarr 0.11 fail).
- `write_series(ome_xml=...)` with another number of `Image` elements than series
  written, or text that is not XML, raises `OMEXMLError` (a `ValueError`, exported),
  naming both counts, before anything is written. A pixel type OME-XML has no name
  for (int64, uint64, float16) raises `OMEXMLError` when ozp writes the minimal one.
- The index that numbers a file's series is called `series`, as in micro-reader 0.1.0
  (it was `scene`). ozp's own keys follow: a bioformats2raw layout and an RFC 8
  collection that is not a plate read as `(("series", i),)`, and
  `PyramidCollection.write` names its outputs `<stem>_series1.ome.zarr` (it was
  `_scene1`). `from_file` keeps the index names it is given, so objects whose images
  still carry `{"scene": i}` (eubi-bridge 0.2.x) keep the keys `("scene", i)`, the
  `_scene1` names and `select(scene=...)`. RFC 8's own `scene` attribute is unchanged.
- `PyramidCollection.select` raises `KeyError` on an index name the collection does not
  have, naming the names it has. Before, it returned an empty collection without a
  word. A name it has with a value no image has still gives an empty collection. While
  micro-reader keeps its guard on the old name, `select(scene=...)` on a collection
  keyed by `series` adds "'scene' is now called 'series' (micro-reader 0.1.0)"; the
  hint goes when that guard goes. `Field.select` (plates) still only filters.
- `PyramidCollection.from_file` always builds `coll.tree`, from the images' indices
  (before, only for a plate file; otherwise it was `None`). The root is named by the
  file's stem (`xyz.czi` -> `xyz`, `cells.ome.tif` -> `cells`; a folder keeps its
  name); a plate holds its wells; a series or a plate field of several images (tiles,
  views) is a collection of its own (`CollectionNode.kind` `"series"` / `"field"`,
  `.index` its number), and a series or field of one image is that image. Images
  outside any plate (a slide's label or preview) sit under the root. So
  `write_collection` of a `from_file` collection writes this tree, and refuses a
  selection of it, as for a collection read from a store. Read back with `from_zarr`,
  such a tree keys every image as a series, in tree order (`S=2_2x2_CH=1.czi`'s 2
  series of 4 tiles come back as `("series", 0)` to `("series", 7)`): the tile index
  comes back when ozp writes the images' indices (plan 9.10).

### Fixes

- `io.read_pyramid` no longer leaves an image's label out without a word. A label that
  cannot be opened (a missing group, broken metadata, or an NGFF version ozp does not
  read, such as 0.6) and a `labels` group that cannot be read are left out of
  `pyr.labels` with a `UserWarning` naming the label, its path and the error; the image
  still opens. Before, they were skipped silently.
- `write_collection` (RFC 8) numbers the collections and the images of one collection
  on one counter, so a series collection and a one-image series next to it no longer
  get the same folder, and a plate and an image beside it (a CZI label) are written
  apart. Before, a collection took its position among all children and an image its
  position among the images only: the write stopped half way with "already holds a
  zarr group", or wrote the plate's images inside the label's image group. Two images
  at one path, or an image group inside another, are now refused before anything is
  written.
- Levels made by `downscale()` with a window method (`mean`, `median`, `min`, `max`,
  `mode`) now carry their own translation: level 0's plus half the difference between
  the level's pixel size and level 0's (the centre of the window the level's first
  pixel is computed from). Before, every level carried level 0's translation, which
  shifted coarse levels by up to half a coarse pixel. `"simple"` (the default) is
  unchanged: every level keeps level 0's translation. Both `defer=True` and
  `defer=False` write the same. Levels read from a store, and the stored levels of a
  microscopy image (`Pyramid.from_image`, `PyramidCollection.from_file`), keep their
  translations.
- An image whose stage position is (0, 0) now gets a translation of 0 on every level;
  before, an all-zero position wrote no translation at all.
- A stage position in a unit ozp cannot convert to the axis's unit (for example
  `pixel` on a micrometer axis) now raises `PositionUnitError` (a `ValueError`,
  exported from `ome_zarr_pyramid`), naming the image, the axis and both units. Before,
  that axis silently got position 0. An axis the position does not name still gets 0.
- A label image is written in its image's NGFF version, so a 0.4 image keeps its labels
  in zarr v2 and a 0.5 image in zarr v3. `add_image_label` converts a label in the other
  version to the image's (as `to_version` does), and `io.write_pyramid`, `write_series`
  and `write_plate` write any attached label in its image's version. A label written
  straight into an existing store (`io.write_labels`, or `io.write_pyramid` with
  `labels_group_path`) takes that store's version: the `labels` group's, or its
  image's before that group exists; a store declaring NGFF 0.6 is refused with
  `UnsupportedNGFFVersion`. Before, a label kept
  its own version: a 0.4 store with a label built by `from_label_arrays()` (0.5 by
  default now) got zarr v3 labels inside a zarr v2 image, which zarr-python and 0.4
  readers do not see, and a 0.4 label on a 0.5 image the reverse.
- `write_plate` no longer warns that a plate's images are "also indexed by" an index
  when the collection holds images outside the plate (a CZI slide's label or preview,
  keyed without a plate): they were counted as plate 0.
- The `OME` group's attributes follow the output's NGFF version (0.4: `series`; 0.5:
  `ome.version` and `ome.series`); before, `update_metadata(ome_xml=...)` and the
  series writer derived them from the zarr format, with 0.5 written in.

### Supported consumers

- eubi-bridge 0.3.0 and eubi-bridge-lite 0.3.0, which require
  `ome_zarr_pyramid[remote]>=0.3.0,<0.4` (and micro-reader 0.1.0).
- eubi-bridge 0.1.x and 0.2.x are not supported by this release: their pin releases
  (eubi-bridge and eubi-bridge-lite 0.1.7 and 0.2.1) require ozp `<0.3` and stay on
  ozp 0.2.x.
- ozp 0.3.0 itself requires `dyna-zarr>=0.1.0,<0.2` (0.2.0 had no upper bound).

### Guards

- ozp has no guard. The old eubi-bridge releases are kept off this release by their
  own bounds (the pin releases above) and by micro-reader 0.1.0's guard against them.
- Kept for those older objects while micro-reader keeps that guard: `from_file` leaves
  an image's `scene` index as it is (keys `("scene", i)`, output names `_scene<N>`,
  `select(scene=...)`), and `select(scene=...)` on a collection keyed by `series` says
  that `scene` is now called `series`. Both go in the same release that removes
  micro-reader's guard (plan 3.24).
