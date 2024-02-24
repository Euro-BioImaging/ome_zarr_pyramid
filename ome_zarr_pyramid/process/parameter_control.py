# from ome_zarr_pyramid.core.Hierarchy import OMEZarr, MultiScales
import warnings

from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.core import config, convenience as cnv
from ome_zarr_pyramid.process import process_utilities as putils

import itertools
import inspect
import importlib
import zarr
import numpy as np
import os, copy
import numcodecs
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )

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

class BaseParams:
    def __init__(self,
                 input: Pyramid = None,
                 resolutions: list[str] = None,
                 drop_singlet_axes: bool = True,
                 output_name: str = 'nomen'
                 ):
        input = putils.aspyramid(input)
        if output_name is None:
            output_name = input.tag
        if resolutions is None:
            resolutions = input.resolution_paths
        base_params = {'input': input,
                       'resolutions': [cnv.asstr(rsl) for rsl in resolutions],
                       'drop_singlet_axes': drop_singlet_axes,
                       'output_name': output_name
                       }
        super().__setattr__("_params", base_params)
        for key, value in base_params.items():
            super().__setattr__(key, value)
    def _update_param(self,
                      name,
                      value
                      ):
        if name not in self._params.keys():
            raise ValueError(f"The name must be one of the base parameters: {self._params.keys()}.")
        self._params[name] = value
        super().__setattr__(name, value)
    def __setattr__(self,
                    key,
                    value
                    ):
        raise ValueError("Parameters cannot be updated by assignment.")

class FilterParams:
    def __init__(self,
                 filter_name,
                 **kwargs
                 ):
        meta = get_functions_with_params(f"dask_image.ndfilters")
        super().__setattr__("_meta", meta)
        self._set_filter(filter_name)
        for key, value in kwargs.items():
            if key in self._params.keys():
                self._update_param(key, value)
    def _set_filter(self,
                    filter_name
                    ):
        filter_params = copy.deepcopy(self._meta[filter_name])
        super().__setattr__("_filter_name", filter_name)
        super().__setattr__("_params", filter_params)
        for key, value in filter_params.items():
            super().__setattr__(key, value)
    def _update_param(self,
                      name,
                      value
                      ):
        if name not in self._params.keys():
            raise ValueError(f"The parameter {name} is not defined for the filter {self._filter_name}")
        self._params[name] = value
        super().__setattr__(name, value)
    def __setattr__(self,
                    key,
                    value
                    ):
        raise ValueError("Parameters cannot be updated by assignment.")


# bp = BaseParams()
# bp._params
# fp = FilterParams("gaussian")
# fp._params

# pc = FilterParams("gaussian")
# pc._update_param("sigma", 2)
# pc.params

# class SuperClass:
#     def __init__(self,
#                  paramset: dict
#                  ):
#         copied = copy.deepcopy(paramset)
#         pc = ParamControl(copied)
#         self.pc = copy.deepcopy(pc)
#         setattr(self.pc, "run", self.shout)
#     def shout(self):
#         print("HALLOOOOOO!!!!!")
#
# animals = {"animal1": "sheep", "animal2": "cow"}
# pc = ParamControl(animals)
# pc.animal1 = "panda"
# pc._paramset
#
# sc = SuperClass(animals)
# sc.pc.animal1 = "panda"
#
# sc.pc._paramset