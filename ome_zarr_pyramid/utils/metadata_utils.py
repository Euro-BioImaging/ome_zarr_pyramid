import warnings, time, shutil, zarr, itertools, multiprocessing, re, numcodecs, dask, os, copy, inspect
from pathlib import Path
import numpy as np
import dask.array as da

from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from skimage import transform
from joblib import Parallel, delayed, parallel_backend
import numcodecs; numcodecs.blosc.use_threads = False
from ome_zarr_pyramid.core.pyramid import Pyramid


def update_scales(input: Union[Pyramid, str, Path],
                  new_scales: Union[dict, tuple, list, None] = None,
                  **kwargs
                  ):
    if not isinstance(input, Pyramid):
        assert isinstance(input, (str, Path))
        input = Pyramid().from_zarr(input)
    if isinstance(new_scales, (tuple, list)):
        assert len(new_scales) == input.ndim
        input.update_scales(new_scales)
        return input
    else:
        scales = input.scales[input.refpath]
        if 't_scale' in kwargs.keys():
            idx = input.index('t')
            scales[idx] = kwargs['t_scale']
        if 'c_scale' in kwargs.keys():
            idx = input.index('c')
            scales[idx] = kwargs['c_scale']
        if 'z_scale' in kwargs.keys():
            idx = input.index('z')
            scales[idx] = kwargs['z_scale']
        if 'y_scale' in kwargs.keys():
            idx = input.index('y')
            scales[idx] = kwargs['y_scale']
        if 'x_scale' in kwargs.keys():
            idx = input.index('x')
            scales[idx] = kwargs['x_scale']
        input.update_scales(scales, hard = True)
        return input


def update_units(input: Union[Pyramid, str, Path],
                 new_units: Union[dict, tuple, list, None] = None,
                 **kwargs
                 ):
    if not isinstance(input, Pyramid):
        assert isinstance(input, (str, Path))
        input = Pyramid().from_zarr(input)
    if isinstance(new_units, (tuple, list)):
        assert len(new_units) == input.ndim
        input.update_units(new_units)
        return input
    else:
        units = list(input.unit_list)
        if 't_unit' in kwargs.keys():
            idx = input.index('t')
            units[idx] = kwargs['t_unit']
        if 'c_unit' in kwargs.keys():
            idx = input.index('c')
            units[idx] = kwargs['c_unit']
        if 'z_unit' in kwargs.keys():
            idx = input.index('z')
            units[idx] = kwargs['z_unit']
        if 'y_unit' in kwargs.keys():
            idx = input.index('y')
            units[idx] = kwargs['y_unit']
        if 'x_unit' in kwargs.keys():
            idx = input.index('x')
            units[idx] = kwargs['x_unit']
        input.update_unitlist(units, hard = True)
        return input


class MetadataUpdater:
    def update_scales(self,
                      input: Union[Pyramid, str, Path],
                      new_scales: Union[dict, tuple, list, None] = None,
                      **kwargs
                      ):
        _ = update_scales(input, new_scales, **kwargs)
        return
    def update_units(self,
                    input: Union[Pyramid, str, Path],
                    new_units: Union[dict, tuple, list, None] = None,
                    ** kwargs
                     ):
        _ = update_units(input, new_units, **kwargs)
        return


class MetadataReporter:
    def __init__(self):
        # TODO: set metadata reporting methods here.
        raise NotImplementedError