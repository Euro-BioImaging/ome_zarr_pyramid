# from ome_zarr_pyramid.core.Hierarchy import OMEZarr, MultiScales
import warnings

from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.core import config, convenience as cnv
from ome_zarr_pyramid.process import process_utilities as putils

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
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )

############################# Image manipulation functions ################################

import importlib, inspect
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

# oz = Pyramid()
# oz.from_zarr(f"data/filament.zarr")
#
# arr0 = oz[0]

#
# h = reductive_apply(oz, resolutions=[0, 1], operation = max, params = {'axis': 'z'}, axes_key='axis')
# hh = apply(oz, resolutions=[0, 1], operation = power, params = {'power': [3]})
#
# arr1 = da.zeros_like(arr0)
# conc = da.concatenate([arr0, arr1], axis = 0)
#

# static array-to-array operations:

def add(image: da.array,
        scalar: [int, float]
        ):
    return da.add(image, scalar)

def subtract(image: da.array,
            scalar: [int, float]
            ):
    return da.add(image, scalar)

def power(image: da.Array,
          power: [int, float]
          ):
    return da.power(image, power)

def subset(image: da.Array,
           axis: (str, tuple, list),
           interval: Iterable[slice],
           n_resolutions: int = 1,
           scale_factor: (int, float) = 2,
           planewise: bool = True
           ):
    return image[interval]

def split(image: da.Array,
          axis: (int, tuple, list),
          n_sections: (int, tuple, list) = 1
          # offset: (int, tuple, list) = 1, # TODO
          # increment: (int, tuple, list) = 1 # TODO
          ):
    # arr = oz1[1]
    # image = arr
    # axis = [0, 1]
    # n_sections = [3, 5]
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(n_sections, int):
        n_sections = [n_sections] * len(axis)
    assert len(n_sections) == len(axis)
    shape = image.shape
    axlens = [shape[i] for i in axis]
    increment = [axlen // n for axlen, n in zip(axlens, n_sections)]
    slices = []
    increments = [None] * len(shape)
    for idx, i in enumerate(axis):
        if idx in axis:
            increments[i] = increment[idx]
    for idx, i in enumerate(increments):
        if i in [None, 0]:
            increments[idx] = 1
    for ax, size in enumerate(shape):
        step = increments[ax]
        if ax in axis:
            splitters = [(i, i + step) for i in range(0, size, step)]
            slcs = tuple([slice(*splitter) for splitter in splitters])
        else:
            slcs = tuple([slice(None, None)])
        slices.append(slcs)
    combins = list(itertools.product(*slices))
    subsets = []
    for combin in combins:
        subsets.append(image[combin])
    return subsets

# static collection-to-array operations:
def concatenate(imlist: da.Array,
                axis: int
                ):
    return da.concatenate(imlist, axis)

def block(imlist: da.Array, # TODO: https://docs.dask.org/en/stable/generated/dask.array.block.html#dask.array.block
          axis: int
          ):
    return da.block(imlist, axis)

def maxmul(imlist: da.Array # TODO extend to the other statistical operators.
           ):
    merged = da.stack(imlist, axis = 0)
    return da.max(merged, axis = 0, keepdims = True)

def minmul (imlist: da.Array,
            axis: int
            ):
    return da.min(imlist, axis)

def meanmul (imlist: da.Array,
            axis: int
            ):
    return da.concatenate(imlist, axis)

# reductive array-to-array operations:
def max(image: da.Array,
        axis: int,
        keepdims: bool = True
        ):
    return da.max(image, axis = axis, keepdims = keepdims)

def mean(image: da.Array,
        axis: int,
        keepdims: bool = True
        ):
    return da.mean(image, axis = axis, keepdims = keepdims)

def min(image: da.Array,
        axis: int,
        keepdims: bool = True
        ):
    return da.min(image, axis = axis, keepdims = keepdims)

def median(image: da.Array,
        axis: int,
        keepdims: bool = True
        ):
    return da.median(image, axis = axis, keepdims = keepdims)

def sum(image: da.Array,
        axis: int,
        keepdims: bool = True
        ):
    return da.sum(image, axis = axis, keepdims = keepdims)

# expansive array-to-array operations
def expand_dims(image: da.Array,
                newaxis: (str, tuple, list),
                newunit: (str, tuple, list),
                newscale: (str, tuple, list),
                position: (int, tuple, list)
                ):
    return da.expand_dims(image, axis = position)

# expansive collection-to-array operations
def stack(imlist: list,
          newaxis: (str, tuple, list),
          newunit: (str, tuple, list),
          newscale: (str, tuple, list),
          position: int
          ):
    if hasattr(position, '__len__'):
        if len(position) > 1:
            raise ValueError(f"The 'position' variable must be a single integer but a list of multiple items is given.")
        elif len(position) == 0:
            raise ValueError(f"The 'position' variable must be a single integer but an empty list is given.")
        else:
            position = position[0]
    return da.stack(imlist, axis = position)


# reductive array-to-collection operations

# def split(pyr: Pyramid,
#            axes: str = 'zc',
#            steps: tuple = None
#            ):
#     if steps is None:
#         steps = tuple([1] * len(axes))
#     else:
#         assert len(axes) == len(steps), f'Axes and steps must be of equal length.'
#     axlens = [pyr.axislen(ax) for ax in axes]
#     slices = []
#     for ax, size, step in zip(axes, axlens, steps):
#         splitters = [(i, i+step) for i in range(0, size, step)]
#         slices.append(splitters)
#     combins = list(itertools.product(*slices))
#     subsets = []
#     for combin in combins:
#         slicer = {ax: slc for ax, slc in zip(axes, combin)}
#         subset = pyr.copy().subset(slicer)
#         subsets.append((slicer, subset))
#     # for _, sub in subsets:
#     #     print(sub.shape)
#     return subsets

# def split_layers():
#     pass

def update_axis_order():
    pass



