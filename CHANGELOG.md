# Changelog

## 0.5.0 (2026-10-28): one API for Pyramid and PyramidCollection, IO as the hub

A minor release (decision C-11): names are added and every existing name keeps working;
one return value changes (see Changed). eubi-bridge needs no change (it does not read
those return values).

### New

- `IO` is the one read/write hub: `read_collection`, `write_plate`, `write_series` and
  `write_collection` join `read_pyramid`, `write_pyramid`, `update_metadata` and the label
  methods. The class methods are shortcuts to them: `PyramidCollection.from_zarr`
  and its writers now go through `IO`. `IO` keeps no settings: options are per call, as in
  dyna-zarr's `io`.
- `Pyramid.from_zarr(path)` and `pyr.write(path)` (returns the path), shortcuts for
  `io.read_pyramid` / `io.write_pyramid`; `pyr.update_metadata(...)` edits the store the
  pyramid was read from (`path=` for one it cannot name).
- A collection answers the read properties of a `Pyramid`: `axes`, `dtype`, `shape`,
  `nlayers`, `compressor`, `scale_factor_dict`, `custom_metadata` give the one value all
  its images have, and raise the new `VaryingPropertyError` (a `ValueError`, pointing to
  `coll.table`) when they differ (decision C-8). `axes`, `shape` and `nlayers` are read from
  the table and open no image. `supports_dyna` is true when every image supports dyna.
- `PyramidCollection.only()`: the one pyramid of a one-image collection (keeping its
  coordinate system), else a `ValueError` listing the keys. `Pyramid.as_collection()`:
  the pyramid as a collection of one, under the key `()`.
- `PyramidCollection.validate(images=True)`: the graph's invariants and every image's
  `validate()`, returning the collection; `check_invariants` stays.
- `Pyramid.table` (a one-row `CollectionTable`), `Pyramid.coordinate_systems` (`{id:
  system}` or `{}`) and `Pyramid.coordinate_transformations` (`[]`), as on a collection.
- One write for a collection: `coll.write(path)` / `io.write(coll, path)` writes what the
  collection is: a plate as a plate, one image (key `()`) as an image, anything else as a
  series layout; `version=` picks the format (`"0.4"`, `"0.5"`, `"0.6"`, or `"rfc8"` for
  the whole tree, several plates too). The old `coll.write(directory, stem)` (one store
  per image) is unchanged. `io.write(pyr, path)` writes a pyramid.
- `coll.update_metadata(...)` and `io.update_metadata(path, ...)` on a plate or series:
  each image of the store is edited; the store is read again and returned.
- The writers return where they wrote: `io.write_pyramid` and `io.write_labels` return the
  path (they returned `None`).
- `io.read(path)`, the other half of `io.write` (`io.read(io.write(obj, path))` is what
  was written): a `Pyramid` for an image store, a
  `PyramidCollection` for a plate, a well, a bioformats2raw layout or an RFC 8 collection
  (also when it holds one image). `io.read_pyramid` / `io.read_collection` give one type
  whatever the store holds.
- `io.read_pyramid` takes an open `zarr.Group`.
- `validate=True` on `write_collection` (and on `write_plate` / `write_series` with
  `version="rfc8"`) checks the written RFC 8 store, and a standalone RFC 8 document:
  RFC 8 has no published schema, so its structure is checked (node types, every level an
  array, every referenced image a group, every transformation into a declared coordinate
  system), with an `RFC8Warning` saying so. `ozp.validate(path)` checks an RFC 8 store
  the same way; it passed every RFC 8 store unchecked before.
- `as_collection()` on a plate, a well and a field of `coll.plates`: that part as a
  collection of its own (its keys, its tree, the part of the coordinate-system graph it
  keeps), so `coll.plates[1].as_collection().write(path)` writes one plate. The navigation
  objects stay micro-reader's.
- A selection writes as RFC 8: `coll.select(plate=0, well=3).write(path, version="rfc8")`
  (and `write_collection`) writes the part of the tree that holds the selected images. A
  kept plate keeps every well (empty where nothing was selected) and its rows and columns,
  so the keys read back as they were; labels of images left out are left out. It was
  refused ("a selection of its tree").
- `select` takes micro-reader's new arguments: a well by name (`well="C3"`), wells by
  `row` / `column` (0-based numbers or names; not together with `well`), and a list,
  tuple, range or set as any value (any of them). A test checks that ozp and
  micro-reader select the same images for the same arguments.
- `select`'s keywords are the table's columns: each place of a plate has a position and a
  name, `plate` / `plate_name`, `acquisition` / `acquisition_name`, `well` / `well_name`,
  `row` / `row_name`, `column` / `column_name` (`select(well_name="B2")` raised
  `KeyError`). The table gains `row` and `column`, the well's 0-based row and column, beside
  `row_name` and `column_name`. A name given to a position keyword is still its name
  (`well="B2"`); a number given to a name keyword raises `TypeError` naming the position
  keyword. The `KeyError` for an unknown keyword lists `row`, `column` and the names too.
- `select` (and `Field.select`) take slices of positions, as ranges: `well=slice(0, 12)`,
  `row=slice(2, None)` (open-ended: row C onward), `column=slice(0, None, 2)`. A slice
  selected nothing, silently (it was compared as one value); a negative bound or a step
  below 1 raises `ValueError` (which position is last depends on what is selected), a
  slice on a name keyword `TypeError`. As micro-reader's.
- Containment (decision C-19), as micro-reader's navigation (a test keeps the two the
  same): every level has `.kind` and `.parent`. A plate, well or field view knows its
  kind (`"plate"`, `"well"`, `"field"`) and the level above (`well.parent` is its plate,
  `plate.parent` the collection). A collection is `"collection"` as read or built, and
  `"plate"`, `"well"` or `"field"` when taken from navigation (`as_collection()`; a
  selection or operation of it keeps its kind and parent). Accessors go down only: a
  collection holds `.plates`, a plate's collection `.wells` (and `.row_names`,
  `.column_names`, `.well_index`), a well's `.fields`, a field's `.images`; another
  level's accessor raises `AttributeError` naming the way. `plate.rows[...]` /
  `plate.columns[...]` give a part of the plate, still a plate (`plate.rows[2:5].columns[3:7]`,
  as `select(row=..., column=...)`). A selection's plates and wells are the ones holding
  its images: `pcoll.select(row="B").wells` is row B's wells (they listed every well of
  the tree, empty ones too); the tree underneath stays whole for writing.
- `coll.drop_empty()`: the collection with the fields, wells, plates and other
  collections that hold no image dropped from its tree; plates and wells are numbered
  anew (`"well": w` is `plates[p].wells[w]`), every other index keeps its value, the
  coordinate-system graph follows. `coll.select(well="C3").drop_empty()` writes as an
  NGFF plate of one well, or as RFC 8.
- The write options use dyna-zarr's words (decision C-14): `backend=` is the storage
  backend (`"tensorstore"`, default, or `"zarrista"`); `io_backend=`, its old name, still
  works without a warning; giving both with different values raises. `engine=`
  (`"fused"` / `"pipeline"`), `num_readers=` and `gc_interval=` reach dyna's write of
  each level (a deferred downscale's one-pass write refuses `engine="pipeline"`).
  `compressor=` takes dyna's `Codecs`. Attached labels are written with the same storage
  backend as their image (they always used TensorStore).
- Level names beside the layer names (decision C-16): `pyr.levels` and `pyr.nlevels`
  (`layers`, `nlayers`), `coll.nlevels`, `write_pyramid(levels=)` (`layers=`). The layer
  names stay, without a warning. The table's column `levels` (the number of levels) is
  `nlevels`; `levels` still reads (`table["levels"]`, `filter(levels=...)`,
  `group_by("levels")`, `image["levels"]` in a predicate), listings show `nlevels`.
- `write_pyramid(fill_derived=False)` (decision 9.12-S), also on `update_metadata` and
  every collection writer: the NGFF SHOULD fields ozp derives exactly (a multiscale's
  empty `metadata` and `type` "unknown", an acquisition's `maximumfieldcount`) are
  written only as the source has them. Default True, as before. A write no longer adds
  them to the pyramid it is given, only to the store.
- `Pyramid.layers` opens each level's zarr array once (per group and set of level paths)
  instead of on every access: operations on a stored pyramid read its arrays' metadata
  once (an `isel` of a 3-level image opened 24 arrays). The table of `isel(c=0)` on
  idr0010's 384 images: 4.0 s -> 1.2 s. A store overwritten after it was read is read
  again with `io.read_pyramid`.
- The table rows of images with no metadata file to read (the lazy results of collection
  operations, images in memory) are read in threads from eight images on: the same rows,
  about twice as fast (idr0010's 384 images after `isel`: 8 s -> 4 s).
- Storage, values and operators on a whole collection (Workplan C5): `coll.rechunk(...)`,
  `coll.to_version(...)`, `coll.saturate_cast(...)` and operators with a number
  (arithmetic, comparisons, bitwise, unary: `coll > 128` is a mask collection), lazy.
  `coll == 3` / `coll != 3` are elementwise (decision C-1); with anything but a number they
  say whether it is the same collection. Two collections do not combine yet: `coll - other`
  raises `TypeError` (decision C-4).
- Axis and level operations on a whole collection (Workplan C4): `coll.isel(...)`,
  `coll.squeeze(...)`, `coll.select_levels(...)`, `coll.downscale(...)`, lazy, refused as a
  whole when an axis, index or level is missing in some image. The coordinate-system graph:
  a system whose images all drop an axis loses it; dropping an axis of an image an edge
  (affine, registration) runs through is refused. RFC 8 label nodes of the tree follow
  their images (Workplan C7, below).
- Operations on a whole collection (Workplan C2, C3, C6): `coll.set_channels(...)`,
  `coll.set_display_range(...)` and `coll.set_custom_metadata(...)` apply the `Pyramid`
  method to every image and return a new collection (same keys, tree and coordinate
  systems). Lazy: an image not opened yet gets the operation when it is first used or
  written, and the table of the result comes from the source's metadata. An operation
  that does not fit every image raises `OperationMismatchError` (a `ValueError`, new)
  before anything is done, naming the images and a `filter` that narrows the collection
  (decision C-5).
- `chunks=` on `io.write_pyramid` (and every writer passing write options), the dyna-zarr
  name of `chunk_shape=`; giving both raises.
- Label images follow their image (Workplan C7, decision C-2). `isel`, `squeeze`,
  `select_levels` and `downscale` of a pyramid apply to its attached labels too: on the
  axes a label has, never the channel axis, and a label is downscaled with `"simple"`
  (picking, never averaging), whatever method the image uses. Shape-preserving
  operations (`rechunk`, operators, `saturate_cast`, metadata) keep them as they are. On a
  collection, its RFC 8 label nodes follow their images the same way. A label that cannot
  follow (another size along an indexed axis, fewer levels) raises the new
  `LabelMismatchError` (a `ValueError`) naming the label, and on a collection the image,
  before anything is done.
- A label at another resolution than its image follows it in physical space (decision
  C-18): Fractal's plates store labels at a quarter of the image's resolution, covering the
  same area. Along each axis the pixel sizes must differ by a whole-number ratio and the
  label's size be the image's at that ratio. `downscale` gives the label the image's
  factors (its levels stay aligned: 540 -> 270 -> 135 beside 2160 -> 1080 -> 540); `isel`
  maps an index through the ratio (a slice when it lands on whole label pixels, an integer
  to the label pixel holding it); `select_levels` takes the same levels.

### Fixed

- **An RFC 8 plate written as NGFF 0.4, 0.5 or 0.6** (`PyramidCollection.from_zarr(rfc8)
  .write_plate(out, version="0.6")`) put each field at its path from the collection's root
  inside its well (`B/2/B/2/0`, not a valid NGFF image path); in 0.6 the well's scene then
  named images it could not find, and every field read back at the origin. A well writes
  each image at the last part of its path (found by eubi-bridge's OME-Zarr plate input,
  Workplan E2).

Found by the operations on real plates (Workplan C8):

- `Pyramid.rechunk`, and writing a downscale plan, of an image with a unitless axis besides
  the channel axis (idr0010: `t` and `z` without a unit) raised "Number of units must match
  number of axes except channel", as `isel` did: the unitless axes' units were dropped.
- The levels a pyramid shows are the levels it writes. A downscale's levels were worked out
  in several places (the eager `downscale(defer=False)`, the levels a deferred plan shows,
  `select_levels` on a plan, the writer's two routes, the label rule), each its own way, and
  they disagreed: a deferred downscale of floor-sized levels (Fractal's plates: 535 -> 267)
  showed 268 and wrote 267, one of an irregular source (64 -> 64 / 32 -> 32 / 16) wrote
  other levels than it showed, and the eager and deferred results differed for most
  sources. Now `resolve_levels` works out every level's factor and shape once, the plan
  holds them (`level_scale_factors`, `level_shapes`), and one builder makes the levels
  from them everywhere.
- An attached label with fewer axes than its image (a `zyx` label on a `czyx` image) could
  not follow a downscale plan when written ("operands could not be broadcast together"),
  and did not get the image's floor-sized levels.
- `coll.write(path)` of a collection holding a plate and images beside it (a CZI's slide
  overview) wrote everything as a series, losing the plate, without a word. It is refused
  now, naming the ways out: the plate alone (`coll.plates[0].as_collection().write(path)`),
  the other images (`coll.select(...)`), or all of it as RFC 8.
- An RFC 8 store holding a plate and images beside it read back the plate only: the images
  beside it are keyed `(("series", i),)` now, as micro-reader keys them.

- Writing an RFC 8 collection read from a store as RFC 8 again was refused ("image
  'A/2/A/2/0/labels/nuclei' would be written inside the image group 'A/2/A/2/0'"): the
  document's image paths, relative to the document, were joined to their well's folder a
  second time, and a label node read back was not known as its image's label.
- A deferred `downscale()` of an in-memory pyramid whose levels were built eagerly raised
  `TypeError` (`int(None)`): a TensorStore view without a storage grid was taken for one.
- The table's `shards` of an image that lost an axis (`isel(c=0)` of a sharded image) was
  its source's tuple, one entry too long; it is None now (dyna-zarr copied the shards down
  its transforms; fixed there too, for dyna-zarr 0.1.1).
- `Pyramid.isel` (and `squeeze`, `[...]`) of an image with a unitless axis besides the
  channel axis (idr0010: `t` and `z` without a unit) raised "Number of units must match
  number of axes except channel": the unitless axes' units were dropped.

### Changed

- **Breaking:** a plate view's `rows` and `columns` slice the plate (decision C-19); the
  names are `row_names` and `column_names`, as micro-reader's. `len(plate.rows)` still
  works. ozp reads micro-reader 0.2's plates (`rows` held the names) as before.
- `.plates` of a plate's (or well's) collection raises `AttributeError` naming `.wells`
  (it gave the plate itself); a collection as read keeps `.plates`.
- What the table holds is read from the table: `coll.columns`, `coll.unique(column)` and
  `coll.count(column)` are `coll.table.columns`, `coll.table.unique(column)` and
  `coll.table.count(column)`. The old names still answer, with a `FutureWarning`, and will
  be removed; on a plate's collection `coll.columns` is the column slicer (decision C-19),
  beside `coll.rows`: `pcoll.rows[2:5].columns[3:7]` is the plate's collection with those
  images. `filter`, `group_by` and `sort_by` stay on the collection: they give collections.
- `downscale(defer=False)` of a pyramid that already has levels uses the pyramid's own
  factors and level shapes, as `downscale()` (deferred) always did, unless a `scale_factor`
  is named: it forced 2 on z / y / x, so an anisotropic source (z kept) came out halved
  along z. Levels past the source's last one continue with its last step (they repeated
  level 1's factor). A plan now always holds every level's factor and shape.
- `isel`, `squeeze`, `select_levels`, `downscale`, `rechunk` and the operators no longer
  drop a pyramid's attached labels: they follow it (see New), or the operation raises
  `LabelMismatchError` when one cannot.
- `PyramidCollection.__eq__` / `__ne__` with a number are elementwise (decision C-1), as on
  a `Pyramid`; the Mapping comparison (same keys and images) is gone, nothing used it.
- `Pyramid.from_image` / `PyramidCollection.from_file` (C13): channels the file describes
  without a colour get ozp's distinct colours (red, green, blue, ...; skipping colours
  other channels have) instead of white, so a multichannel image no longer shows all
  grey. A single channel stays white. Colours the file gives are kept. eubi-bridge's own
  colour filling (emission wavelengths, transmitted light) runs after and is unchanged.
- `backend="auto"` / `"dyna"` (ozp's one engine, the meaning `backend=` had) give a
  `FutureWarning` and are ignored; `backend="tensorstore"` writes with TensorStore
  storage instead of raising "removed engine"; `num_readers=` and `gc_interval=` are
  dyna's settings again instead of being refused as the removed engines'.
- `select` refuses a `Plate`, `Well`, `Field` or `Acquisition` object as a value
  (`TypeError`; it selected nothing), as micro-reader's does.
- `select` names indices as micro-reader's `select` does: any of micro-reader's index
  names is known, with the names the keys carry; a known index no image has selects
  nothing (it raised `KeyError` when the keys lacked it), and `'scene'` always gets the
  rename hint. `Field.select` checks names the same way (it took any).
- `write_plate`, `write_series` and `write_collection` return the path written (the
  standalone RFC 8 `document`, when one is given), as `pyr.write` does, so
  `io.read(coll.write(path))` reads the written collection back. They returned
  `{path in the layout: written path}`. `coll.write(directory, stem)` (one store per
  image, no top-level store) still returns `{key: path}`.

### Not in this release

- Length and iteration: `len(pyr)` / iterating levels is not added, because `pyr[0]`
  already slices pixels (positional indexing), so `len` and `[]` would disagree.
- The Pyramid operations on collections (`isel`, `rechunk`, `set_channels`, ... through
  one private `_map`, decision C-9) wait on decisions C-1 to C-6 (Workplan C2-C6).
  `concatenate` stays collection-only (decision C-10).

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
