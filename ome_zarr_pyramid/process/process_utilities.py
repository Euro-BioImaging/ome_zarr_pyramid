# from ome_zarr_pyramid.core.Hierarchy import OMEZarr, MultiScales
import warnings

from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.core import config, convenience as cnv

import itertools
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
from functools import wraps
import importlib, inspect
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )

def aspyramid(obj):
    print(obj)
    assert hasattr(obj, 'refpath') & hasattr(obj, 'physical_size')
    if isinstance(obj, Pyramid):
        return obj.copy()
    else:
        raise TypeError(f"Object is must be an instance of the {Pyramid} class.")

def validate_pyramid_uniformity(pyramids,
                                resolutions = 'all',
                                axes = 'all'
                                ):
    """
    Compares pyramids by the following criteria: Axis order, unit order and array shape.
    Only the layers specified in the resolutions variable are compared.
    """
    full_meta = []
    for pyramid in pyramids:
        pyr = aspyramid(pyramid)
        if resolutions == 'all':
            resolutions = pyr.resolution_paths
        else:
            pyr.shrink(resolutions)
        full_meta.append(pyr.array_meta)
    indices = np.arange(len(full_meta)).tolist()
    combinations = list(itertools.combinations(indices, 2))
    for id1, id2 in combinations:
        assert pyramids[id1].axis_order == pyramids[id1].axis_order, f'Axis order is not uniform in the pyramid list.'
        assert pyramids[id1].unit_list == pyramids[id1].unit_list, f'Unit list is not uniform in the pyramid list.'
        for pth in resolutions:
            if axes == 'all':
                assert full_meta[id1][pth]['shape'] == full_meta[id2][pth]['shape'], \
                    f'If the parameter "axes" is "all", then the array shape must be uniform in all pyramids.'
            else:
                dims1 = [pyramids[id1].axis_order.index(it) for it in axes]
                dims2 = [pyramids[id2].axis_order.index(it) for it in axes]
                shape1 = [full_meta[id1][pth]['shape'][it] for it in dims1]
                shape2 = [full_meta[id2][pth]['shape'][it] for it in dims2]
                assert shape1 == shape2, f'The shape of the two arrays must match except for the concatenation axis.'
    return pyramids


def pyramids_are_elementwise_compatible(pyr1, pyr2): pass

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
    return module, functions

def on_pyramid(module_path,
               filter_name
               ):
    def decorator(func):
        @wraps(func)
        def wrapper(input,
                    **kwargs
                    ):
            # input validation - make sure all parameters match
            module, funclist = get_functions_with_params(module_path)
            arguments = funclist[filter_name]
            ### get the Pyramid object.
            if isinstance(input, Pyramid): # maybe also validate the pyramid here
                pyr = input
            else:
                assert isinstance(input, str), "If input is not a Pyramid, a path to an OME-Zarr dataset must be given."
                pyr = Pyramid()
                pyr.from_zarr(input, include_imglabels=False)
            func_kwargs = {key: value for key, value in kwargs.items() if
                           key not in ['output_path']}  # add other parameters that are not function-specific.
            for key in func_kwargs.keys():
                if key not in arguments.keys():
                    raise ValueError(f"The parameter '{key}' is not defined for the function '{filter_name}'.\n"
                                     f"The valid parameters are '{arguments.keys()}'")
            if 'axis' in func_kwargs.keys():
                func_kwargs['axis'] = pyr.index(func_kwargs['axis'])
            # run operation
            # print(f"kwargs: {kwargs.keys()}")
            filter = getattr(module, filter_name)
            output = apply_to_pyramid(pyr, filter, **func_kwargs)
            # postprocessing
            if 'output_path' in kwargs.keys():
                output.to_zarr(kwargs['output_path'])
            return output
        return wrapper
    return decorator














