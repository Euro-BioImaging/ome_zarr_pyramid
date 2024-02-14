# ome_zarr_pyramid

A package to work with OME-Zarr datasets in a Python environment.

It provides a "Pyramid" object which can read, update and write 
OME-Zarr datasets. Various update methods such as rechunking, re-compressing, 
changing data type, etc are available.

Individual OME-Zarr pyramids including image-label objects are supported.
Collection layouts, including high-content-screening layout are currently 
not supported.



