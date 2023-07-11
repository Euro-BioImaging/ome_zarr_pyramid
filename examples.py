from src.OMEZarr import OMEZarr
from src.core import utils
import numpy as np
import dask.array as da
import os
import numcodecs
import zarr

basepath = 'data/filament.zarr'

oz = OMEZarr(basepath)

oz.baseobj.typeobj
oz.baseobj.path_order
oz.baseobj.dtype('0')
oz.baseobj.chunks('0')

oz.labelobj.image_labels['original'].typeobj
oz.labelobj.image_labels['original'].dtype('0')

newdir = 'data/filament_ex.zarr'
oz.baseobj.extract(newdir = newdir, paths = ['0', '1'])

noz = OMEZarr(newdir)
noz.baseobj.typeobj
noz.baseobj.path_order
noz.baseobj.chunks('0')
noz.baseobj.rechunk((10, 70, 70))
noz.baseobj.chunks('0')

noz.baseobj.get_scale('0')
noz.baseobj.get_scale('1')
