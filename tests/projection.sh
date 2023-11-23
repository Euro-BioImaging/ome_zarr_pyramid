apply_projection

apply_projection -h

apply_projection data/filament.zarr data/res

apply_projection -r 0,1 data/filament.zarr data/res

apply_projection -r 0,1 -a y data/filament.zarr data/res

apply_projection -r 0,1 -a y --projection_type mean data/filament.zarr data/res

