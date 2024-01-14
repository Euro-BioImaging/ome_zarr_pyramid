# ome_zarr_pyramid

A generalised object for OME-Zarr layout.

It reads the OME-Zarr data and provides update methods such as
rechunking, re-compressing, changing data type, etc. The updates can
be in-place or can be written to another store.

Currently, multi-series and multi-labeled OME-Zarr datasets are supported.
High-content-screening layout is currently not supported.



