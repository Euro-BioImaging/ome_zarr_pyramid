"""ome_zarr_pyramid: read, write and downscale OME-Zarr (NGFF) pyramids.

The lazy :class:`~ome_zarr_pyramid.core.pyramid.Pyramid` (dyna-zarr arrays; dask when
built from dask) plus ``io`` (NGFF read/write, memory-bound writer, deferred
downscaling) are the public surface::

    from ome_zarr_pyramid import io
    pyr = io.read_pyramid("image.ome.zarr")
    io.write_pyramid(pyr.downscale(n_layers=4), "out.ome.zarr")

``io`` is one shared :class:`~ome_zarr_pyramid.core.io.IO`; ``IO()`` still works.
"""

__version__ = "0.4.1"
__author__ = "Bugra Oezdemir"

from ome_zarr_pyramid.core.io import IO
from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.core.collection import PositionUnitError, PyramidCollection
from ome_zarr_pyramid.core.concatenate import concatenate
from ome_zarr_pyramid.core.nodes import UnsupportedNGFFVersion
from ome_zarr_pyramid.core.ngff_plate import PlateFieldError
from ome_zarr_pyramid.core.ngff_series import OMEXMLError
from ome_zarr_pyramid.core.rfc8 import RFC8Warning
from ome_zarr_pyramid.core.spatial import InvariantError, TransformError
from ome_zarr_pyramid.core.validation import NGFFValidationError, NGFFValidationWarning, validate

#: The package's reader/writer, shared by everyone in this process. It holds no state:
#: settings go in the call (or in your own ``IO(...)``), never on this object, since
#: anything stored here would apply to every thread and library in the process.
io = IO()

__all__ = ["Pyramid", "PyramidCollection", "concatenate", "IO", "io", "UnsupportedNGFFVersion",
           "PositionUnitError", "PlateFieldError", "OMEXMLError", "RFC8Warning", "InvariantError",
           "TransformError", "validate", "NGFFValidationError", "NGFFValidationWarning"]
