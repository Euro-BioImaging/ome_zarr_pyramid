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
                  new_scale: Union[dict, tuple, list, None] = None,
                  **kwargs
                  ):
    if not isinstance(input, Pyramid):
        assert isinstance(input, (str, Path))
        input = Pyramid().from_zarr(input)
    if isinstance(new_scale, (tuple, list)):
        assert len(new_scale) == input.ndim
        input.update_scales(new_scale)
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
        input.update_scales(scales)
        return input




