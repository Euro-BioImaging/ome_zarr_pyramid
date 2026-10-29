# Changelog

## 0.4.1 (2026-10-28): plates from micro-reader 0.3

A patch for users who stay on ome_zarr_pyramid 0.4 (0.5.0 has the same fix).

### Fixed

- **Plates read with micro-reader 0.3.0 failed**: micro-reader 0.3.0 made `Plate.rows` /
  `Plate.columns` slicers of the plate, which cannot be iterated, and moved the names to
  `row_names` / `column_names`. Building a collection from a plate file (`from_file`)
  iterated `plate.rows` and raised `TypeError`. The names are read from `row_names` /
  `column_names` when the plate has them, else from `rows` / `columns` (micro-reader
  0.2), so both give the same plate.

## 0.4.0 (2026-10-22): coordinate systems, NGFF 0.6, RFC 8 scenes, validation

A minor release under the release policy: field collections are renamed and fields are
no longer renumbered on read (see Breaking changes), so it is not a patch. It ships the
spatial model's trains T2 and T3 together (decision T2-2): the coordinate-system graph,
`eubi:` keys in 0.4 / 0.5, RFC 8 scenes, NGFF 0.6 reading and writing (opt-in with
`version="0.6"`; the default stays 0.5) and whole-store validation. eubi-bridge 0.3.0 and
0.3.1 bound ome_zarr_pyramid below 0.4 and keep 0.3.x; eubi-bridge 0.3.2 is the first that
reads 0.4.0. It reads micro-reader 0.2.0's coordinate systems; with micro-reader 0.1.x,
whose files declare none, positions stay unplaced as before.

### New

- `ome_zarr_pyramid.core.spatial`: the spatial model's values (`Axis`,
  `CoordinateSystem`, `Transformation`, `ExternalRef`), ids (`system_id`,
  `clean_id_part`), the transform model (identity, translation, scale, affine and
  sequence as matrices; `compose`, `invert`; other types are kept as read and raise
  `TransformError` when used) and their JSON (`eubi:` entries).
- `PyramidCollection.coordinate_systems`, `system_of` and `coordinate_transformations`:
  where the images are. `from_file` takes them from micro-reader's file
  (`file.coordinate_systems`, each image's `metadata.coordinate_system`) and checks the
  model's invariants (`check_invariants`, `InvariantError`). A file without
  `coordinate_systems` (eubi-bridge's own objects) gets minimal definitions. `select`,
  `filter` and `sort_by` keep the part of the graph their entries keep; `concatenate`
  places a joined image in its parts' system and drops edges at its parts, with a
  warning.
- A lone `Pyramid` keeps its system: `Pyramid.from_image` and `coll.pyramid(key)`
  write it as the custom metadata `eubi:coordinate_system` and
  `eubi:coordinate_systems`, read back with the pyramid. A collection built from such
  pyramids takes their systems; one id from two sources (two files, two stores) is two
  systems (`<set>--<id>`) unless `join_systems=True`.

- NGFF 0.4 / 0.5 output keeps the model in `eubi:` keys beside `ome` (plan 9.15):
  `write_plate`, `write_series` and `write` write each image's key (`eubi:index`) and
  system (`eubi:coordinate_system`, with its definition), each well's and series root's
  image list (`eubi:images`), each plate's and well's `eubi:kind`, the systems and edges
  of each connected part on the group that holds it (`eubi:coordinate_systems`,
  `eubi:coordinate_transformations`; edges NGFF 0.4 / 0.5 cannot hold give a warning),
  and `eubi:version` 1 on the root. A system without axes gets them from its images.
  `PyramidCollection.from_zarr` reads them back without opening an image: the same keys
  (gapped field numbers 0, 2 stay 0, 2), positions, systems and edges; a series layout
  whose series were flattened (tiles written as series of their own) gets its series
  level back. A store without these keys reads as before.
- `PyramidCollection.components()`: the graph's connected parts and the collection that
  owns each.
- NGFF RFC 8 output keeps the model too (plan 9.10): each connected part of the graph
  becomes a `scene` attribute on the collection that owns it, when every image under
  that collection belongs to it and each image's axes equal its system's (else the
  positions stay in the images, with a warning). A scene lists the shared system first
  and one edge per image into it (a translation, or a scale and translation when the
  units differ); the images' own levels then hold only their alignment. Other edges go
  into the scene, adjusted to that; an attached label image gets its image's edge.
  Images, collections and the root carry `eubi:index`, `eubi:coordinate_system`,
  `eubi:kind` and `eubi:version`; a plate whose fields hold several images has every
  field as a sub-collection; an unnamed plate keeps `""` as `eubi:name` and reads back
  with it. `PyramidCollection.from_zarr` reads all of it back without opening an image
  (positions are added to the levels when an image opens), and reads an image that two
  collections list once (one key), writing it back as one path. Writing RFC 8 gives an
  `RFC8Warning`: RFC 8 is a draft.
- Exported: `RFC8Warning`, `InvariantError`, `TransformError`.
- `add_missing_translation=True` on `PyramidCollection(...)` and
  `PyramidCollection.from_zarr`: an image placed in a coordinate system but without a
  level-0 translation is placed at that system's origin (translation 0; a copy, the
  pyramid given is not changed). Without it, such an image raises `InvariantError`
  (rule I4), now with a message naming the argument; from a store, when the image opens.
- `PyramidCollection.from_zarr` on a well alone keys and places its images as in its
  plate, from the well's `eubi:images` and the systems on it: the plate's keys (plate,
  well and field numbers, gaps kept), each image's coordinate system and position. A
  well without these keys reads as before (acquisition and field, the spec's numbering).
- `PyramidCollection.write` (one store per entry) warns when the collection has edges
  other than positions (affine edges, registrations): a store of one image has no place
  for them. They were left out without a word before.
- NGFF 0.6 (plan 9.11, first part): ozp reads 0.6 images, labels, plates, wells and
  series layouts, and writes them with `version="0.6"` (zarr v3; the default stays 0.5).
  A 0.6 image is held like any other (axes, and a scale and translation per level) and
  written back in 0.6's form: the axes in a coordinate system named `physical`, one
  transformation per level (`scale`, or `sequence[scale, translation]`). A foreign
  image's other coordinate systems and transformations, and the name of its own system,
  are written back as read; written as 0.4 / 0.5 they are kept under `eubi:ngff06`, with
  a warning. A label gets an identity link from its image. A level transformation 0.6
  does not allow (a translation alone, an affine) raises `Unreadable06` instead of
  being misread. A `scene` written by another tool places its images: their positions
  are added to their levels when they open (the image's own system must be the one the
  scene names, else a warning), and its other edges join the graph.
- `write_plate` and `write_series` with `version="0.6"` put each connected part of the
  graph into a `scene` on the plate, well or series root that holds it (plan 5.5, 5.18),
  when every image under that group belongs to it and each image's axes equal its
  system's: the shared system first, one edge per image into it (a translation, or a
  scale and translation when the units differ), the other edges adjusted to it; the
  images' own levels then hold only their alignment, and each image group points back
  at the scene's group (`eubi:scene`). Otherwise the positions stay in the images, with
  a warning naming the group and why. Checked against the official 0.6 schemas of each
  role and with ngff-zarr 0.49 and ome-zarr-py 0.21 (plan 9.17).
  `io.read_pyramid(path, apply_scene=True)` reads such an image alone with its position;
  without it the read warns, naming the scene's group. 0.6 output writes the multiscale
  `type`: the downscale method that made the levels, else `"unknown"`.
- NGFF validation (plan 5.23). Every write checks the metadata it is about to save,
  with no extra dependency: an image's multiscales (2 to 5 axes with unique names, 2 or
  3 of them space axes; per level a scale, then at most one translation, each as long
  as the axes, scales above 0; in 0.6 one transformation per level, every level into the
  same system of the image's own) and each 0.6 scene (every image below it attached, by
  its own system's name, into a scene system with the image's axes). A problem stops the
  write of that group's metadata with `NGFFValidationError` (from `ome_zarr_pyramid`),
  which lists each problem in `.problems`.
- `ome_zarr_pyramid.validate(path)` checks a whole store, local or remote
  (`storage_options`), and returns its problems (`[]` for a valid one): every group
  against the official 0.4, 0.5 or 0.6 schemas of each role it has (the schemas ship with
  the package), plus where each group sits, which no schema checks (an `OME` group only
  under a bioformats2raw root, series and wells and fields that are groups, dataset paths
  that are arrays, scene edges that end at images). `validate(path, strict=True)` also
  returns what the strict schemas find: the spec's SHOULDs (a plate's name, an image's
  name and metadata, a label's colors, ...). `io.write_pyramid`, `write_plate`,
  `write_series` and `PyramidCollection.write` take `validate=True` to run it on what they
  wrote: they raise `NGFFValidationError` for a problem and warn with
  `NGFFValidationWarning` for strict findings, which a store copied from another tool
  often has (Fractal and MoBIE plates, for example). Both need the new `validate` extra
  (`pip install 'ome_zarr_pyramid[validate]'`: jsonschema, referencing); without it
  they raise naming the extra, before anything is written. `validate(path,
  schemas=False)` checks only where the groups sit and needs no extra. RFC 8 groups
  (version `"0.x"`) have no published schema yet and are skipped.
- ozp writes two SHOULD fields where they are missing and it can derive them exactly:
  each plate acquisition's `maximumfieldcount` (the most fields of that acquisition in
  one well) and an empty multiscale `metadata` in 0.4 / 0.5 (0.6 wrote one already). A
  value read with the store is kept. So a store ozp makes on its own meets the strict
  schemas.

### Breaking changes

- NGFF 0.6 stores are read instead of refused with `UnsupportedNGFFVersion` (which stays
  for versions ozp does not know, 0.7 say); `zarr_format_of("0.6")` is 3.
- Metadata NGFF does not allow (a scale of 0, a translation shorter than the axes, one
  space axis, ...) is no longer written as given: the write stops with
  `NGFFValidationError` (a `ValueError`) before that group's metadata is saved. This
  includes `io.update_metadata` on a store whose metadata has such a value.

- RFC 8 output: a level's scale and translation are one `sequence` (was two edges, plan
  5.4; both read); a plate of fields with several images writes every field as a
  sub-collection, so a one-image field's path moves one level down.
- Plate and series output carries the `eubi:` keys above (new attributes beside `ome`);
  a well's fields read back with their own numbers, no longer renumbered 0, 1, ...; a
  flattened series layout reads back keyed by series and tile.
- `concatenate` refuses pyramids placed in different coordinate systems (or some placed,
  some not); `PyramidCollection.concatenate` refused them already.
- `PyramidCollection.from_file`'s tree follows micro-reader's `File.collection` rule
  (plan 3.17, checked by a test against micro-reader): acquisition makes no collection,
  so a field holding images of several acquisitions is one field collection; a field
  collection is named `field<n>` (was `acquisition0_field<n>`) and its images keep
  their acquisition in their names (`acquisition0_tile3`, was `tile3`). The root's
  children are in the order of the images (the plates where the first plate image is).

## 0.3.1 (2026-10-21)

### Fixes

- A single image written with an OME-XML (`write_pyramid(ome_xml=...)`,
  `update_metadata(ome_xml=...)`, and so eubi-bridge 0.3.0's outputs) still gets its
  OME-XML, at the same place, `OME/METADATA.ome.xml`. What changes: the `OME` folder
  holding it is no longer a zarr group (no `OME/zarr.json`; zarr v2: no `.zgroup`,
  `.zattrs`), just a folder with the XML file. 0.3.0 made it a group with
  `series: ["0"]`, copied from the bioformats2raw layout: NGFF has an `OME` group only
  under a bioformats2raw root, and its `series` named the level-0 array instead of an
  image group. Each group still validated on its own, so ome-zarr-models, ome-zarr-py and
  the schemas passed the stores, but a validator that walks the whole hierarchy rejects
  them. Pixels and the image's own metadata were never affected. `write_series` (a
  bioformats2raw layout) keeps its `OME` group, as the spec has it.
- `update_metadata(path)` repairs a store 0.3.0 wrote that way: it removes only the
  group metadata, `OME/zarr.json` (zarr v2: `.zgroup`, `.zattrs`), keeps
  `OME/METADATA.ome.xml`, and warns. No pixel is read or rewritten.
- Every written multiscale names its downscaling `type` (NGFF SHOULD; the strict schemas
  require it): the downscale method when ozp made the levels (`"simple"`, `"mean"`, ...),
  the one read with the image, else `"unknown"` (levels given as they are, a single level).

### Deprecated

- `write_pyramid(ome_series=...)` / `update_metadata(ome_series=...)` have no effect
  (they set the `series` of the `OME` group, which a single image no longer has; its
  OME-XML is written as before) and give a `DeprecationWarning`.

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
