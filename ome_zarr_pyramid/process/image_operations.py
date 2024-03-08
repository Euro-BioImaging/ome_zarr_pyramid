# from ome_zarr_pyramid.core.Hierarchy import OMEZarr, MultiScales
import warnings
from multiprocessing import Pool, Manager

from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.core import config, convenience as cnv
from ome_zarr_pyramid.process.parameter_control import OperationParams, BaseParams
from ome_zarr_pyramid.process import process_utilities as putils, custom_operations
from ome_zarr_pyramid.process.process_core import BaseProtocol

import itertools
import zarr
import pandas as pd
from pathlib import Path
import numpy as np
import dask.array as da
import dask.bag as db
import os, copy
import numcodecs
import dask_image.ndfilters as ndfilters
import dask_image.ndmorph
import dask_image.dispatch
import dask_image.ndmeasure as ndmeasure
import dask
from typing import (Union, Tuple, Dict, Any, Iterable, List, Optional)

def filter_wrapper(filter_name):
    def decorator(func):
        @wraps(func)
        def wrapper(pyr_or_path, **kwargs):
            if isinstance(pyr_or_path, str):
                oz = Pyramid()
                oz.from_zarr(pyr_or_path)
            else:
                assert isinstance(pyr_or_path, Pyramid)
                oz = pyr_or_path
            imfilt = ImageFilters(oz, filter_name=filter_name, **kwargs)
            imfilt.run_cycle()
            output = imfilt.output
            if "write_to" in kwargs:
                output.to_zarr(kwargs["write_to"])
            return output
        return wrapper
    return decorator

@filter_wrapper('gaussian_filter')
def gaussian_filter(pyr_or_path, **kwargs):
    # This function body can remain static
    pass

@filter_wrapper('gaussian_laplace')
def gaussian_laplace(pyr_or_path, **kwargs):
    # This function body can remain static
    pass

@filter_wrapper('median_filter')
def median_filter(pyr_or_path, **kwargs):
    # This function body can remain static
    pass
