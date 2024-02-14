# from ome_zarr_pyramid.core.Hierarchy import OMEZarr, MultiScales
import warnings

from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.core import config, convenience as cnv
from ome_zarr_pyramid.process import process_utilities as utils
from ome_zarr_pyramid.process.parameter_control import FilterParams, BaseParams

import itertools
import inspect
import importlib
import zarr
import pandas as pd
from pathlib import Path
import numpy as np
import dask.array as da
import dask.bag as db
import os, copy
import numcodecs
import dask_image.ndfilters
import dask_image.ndmorph
import dask_image.dispatch
import dask_image.ndmeasure as ndmeasure
import dask
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from functools import wraps

################################### UTILITIES #########################################

def apply_to_pyramid(pyr,
                     resolutions,
                     filter,
                     params
                     ):
    oz = Pyramid()
    for rsl in resolutions:
        scl = pyr.get_scale(rsl)
        arr = pyr[rsl]
        # params["image"] = arr
        res = filter(image = arr, **params)
        oz.add_layer(res,
                     rsl,
                     scale=scl,
                     axis_order=pyr.axis_order,
                     unitlist=pyr.unit_list,
                     zarr_meta={'chunks': res.chunksize,
                                'dtype': pyr.dtype,
                                'compressor': pyr.compressor,
                                'dimension_separator': pyr.dimension_separator}
                     )
    res = oz.copy()
    return res

def get_functions_with_params(module_name):
    module = importlib.import_module(module_name)
    functions = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        try:
            signature = inspect.signature(obj)
            parameters = {}
            for param_name, param in signature.parameters.items():
                if param.default != inspect.Parameter.empty:
                    parameters[param_name] = param.default
                else:
                    parameters[param_name] = None
            functions[name] = parameters
        except ValueError:  # Unable to get signature
            pass
    return functions

################################### CLASSES ############################################

class _ImageFilters:
    def __init__(self,
                 filter_name: str = None,
                 **kwargs
                 ):
        self._filter_params = FilterParams(filter_name, **kwargs)
    def print_params(self):
        print(self.filter_params)
    @property
    def _filter_collection(self):
        return self._filter_params._meta
    @property
    def filter_name(self):
        return self._filter_params._filter_name
    @property
    def filter_params(self):
        p = copy.deepcopy(self._filter_params._params)
        p.pop('image')
        return p
    def filter(self,
               *args,
               **kwargs
               ):
        filter = getattr(dask_image.ndfilters, self.filter_name)
        return filter(*args, **kwargs)

class ImageFilters(utils.ArrayManipulation, _ImageFilters):
    def __init__(self,
                 *args,
                 filter_name: str = "gaussian",
                 **kwargs
                 ):
            _ImageFilters.__init__(self, filter_name=filter_name, **kwargs)
            utils.ArrayManipulation.__init__(self, *args, **kwargs)
    @property
    def base_params(self):
        return self._base_params._params
    def update_param(self,
                     key,
                     value
                     ): ### TODO!!!
        pass
    def run(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.filter_params.keys():
                print(f"The given param {key} is not defined for the function {self.filter_name}.")
                print(f"The defined parameters are: {self.filter_params}")
            elif value != self.filter_params[key]:
                self._filter_params._update_param(key, value)
        # for key, value in self.filter_params.items():
        #     if key == "image":
        #         pass
        #     else:
        #         if value is None:
        #             raise ValueError(f"The value for the parameter {key} cannot be None.")
        output = apply_to_pyramid(self.input, self._base_params.resolutions, self.filter, self.filter_params)
        super().__setattr__("output", output)

################################## FUNCTIONS ############################################
# hh = imfilt._filter_collection

# def gaussian_filter(pyr_or_path, **kwargs):
#     if isinstance(pyr_or_path, str):
#         oz = Pyramid()
#         oz.from_zarr(pyr_or_path)
#     else:
#         assert isinstance(pyr_or_path, Pyramid)
#         oz = pyr_or_path
#     imfilt = ImageFilters(oz, filter_name = 'gaussian', **kwargs)
#     imfilt.run_cycle()
#     output = imfilt.output
#     if "write_to" in kwargs.keys():
#         output.to_zarr(kwargs["write_to"])
#     return output

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

#
cpath = f"/home/oezdemir/PycharmProjects/test_pyenv/data/astronaut_channel_0.zarr"
oz = Pyramid()
oz.from_zarr(cpath)
res = gaussian_laplace(oz, sigma = (0.2, 1, 1))


# h = BaseParams(oz, ['0'], True, 'nomen')

# imfilt = ImageFilters(oz, resolutions = ['0'], filter_name="gaussian", output_name = 'nomen2', sigma = (0.2, 2, 2))
# hh = imfilt._filter_collection
# imfilt.run_cycle(output_name = 'whatever', sigma = (0.2, 1, 1))
# dir(imfilt._base_params)
# # imfilt.filter_name = "gaussian"
# imfilt.sigma = 2



# bpath = f"/home/oezdemir/PycharmProjects/test_pyenv/data/filament.zarr"

# boz = Pyramid()
# boz.from_zarr(cpath)
#
# h = utils.ArrayManipulation(oz, resolutions = ['0', '1'])
# h.filter_name = None
# h = oz.copy()
# utils.aspyramid(oz)

# imfilt = utils.ArrayManipulation(oz, resolutions = ['0'], filter_name="gaussian", sigma = (0.2, 2, 2))

# imfilt = ImageFilters(oz, resolutions = ['0'], filter_name="gaussian", sigma = (0.2, 2, 2))
# imfilt.sigma = 2
#
# imfilt.run_cycle(resolutions = ['0', '1'])
# imfilt.run_cycle(sigma = 2)
#
# imfilt.run_cycle(input = oz, sigma = (0.2, 2, 2))
#
# imfilt.filter_name
# imfilt.filter_params