import copy
import inspect, itertools, os
# from dataclasses import dataclass
from attrs import define, field, setters
import numcodecs; numcodecs.blosc.use_threads = False
from joblib import Parallel, delayed, parallel_backend

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.core import config
from ome_zarr_pyramid.process.core.multiscale_apply_general import ApplyAndRescale, ApplyToPyramid

import zarr, warnings
import numpy as np, dask.array as da


def _get_scale_factor_from_dict(input: dict):
    assert isinstance(input, dict)
    if input.__len__() == 1:
        raise Exception(f"No scale factor can be obtained from monoresolution Pyramids.")
    assert '1' in input.keys(), f"To be able to detect the scale factor, the path 1 must exist."
    scale_factor = np.around(input['1']).tolist()
    return scale_factor

def _get_scale_factor_from_pyramid(input: Pyramid):
    if input.nlayers == 1:
        raise Exception(f"No scale factor can be obtained from monoresolution Pyramids.")
    return _get_scale_factor_from_dict(input.scale_factors)

def _generate_scale_factor_per_path(scale_factor, pth):
    exp = int(pth)
    return np.power(scale_factor, exp)

def _extend_shapes(current_shapes: dict, scale_factor: (list, tuple), paths: (tuple, list)):
    assert isinstance(current_shapes, dict)
    mainshape = current_shapes['0']
    shapes = copy.deepcopy(current_shapes)
    for pth in paths:
        if str(pth) not in current_shapes.keys():
            factor = _generate_scale_factor_per_path(scale_factor, pth)
            newshape = np.divide(mainshape, factor)
            newshape[newshape <= 1] = 1.
            shapes[str(pth)] = tuple(np.around(newshape).astype(int).tolist())
    return shapes

def _get_scale_factors_from_shapes(shapes: dict):
    return {pth: np.divide(shapes['0'], shapes[pth]).tolist() for pth in shapes.keys()}




