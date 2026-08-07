"""ome_zarr_pyramid: read, write and downscale OME-Zarr (NGFF) pyramids.

The lazy, dask-backed :class:`~ome_zarr_pyramid.core.pyramid.Pyramid` plus
:class:`~ome_zarr_pyramid.core.io.IO` (NGFF read/write, memory-bound region writer,
deferred progressive downscaling) are the public surface. Image PROCESSING operations
(filters, segmentation, features, ...) live in the sibling ``pyrops`` package.
"""

from ome_zarr_pyramid.core.io import IO
from ome_zarr_pyramid.core.pyramid import Pyramid

__all__ = ["Pyramid", "IO"]
