# Changelog

## 0.2.1 (unreleased)

### Fixes

- Levels made by `downscale()` now carry a translation that follows the downscale
  method: with a window method (`mean`, `median`, `min`, `max`, `mode`) a level sits
  (F - 1) / 2 level-0 pixels further in than level 0 (the centre of the window its
  first pixel is computed from); before, every level carried level 0's translation.
  `"simple"` (the default) is unchanged. `defer=True` and `defer=False` write the same.
- An image whose stage position is all zero, such as (0, 0), now gets a translation of
  0 on every level; before, no translation was written at all.
- A stage position in a unit ozp cannot convert to the axis's unit (for example `pixel`
  on a micrometer axis) now raises `PositionUnitError` (a `ValueError`, exported from
  `ome_zarr_pyramid`), naming the image, the axis and both units; before, that axis
  silently got position 0.
