from OME_Zarr.src.OMEZarr import OMEZarr
from OME_Zarr.src.core import utils
import numcodecs, numpy as np
import napari

basepath = 'OME_Zarr/data/filament.zarr'

oz = OMEZarr(basepath)

oz.baseobj.typeobj
oz.baseobj.path_order
oz.baseobj.dtype('0')
oz.baseobj.chunks('0')

oz.labelobj['original'].typeobj
oz.labelobj['original'].dtype('0')

newdir = 'OME_Zarr/data/filament_ex.zarr'
oz.baseobj.extract(newdir = newdir, paths = ['0', '1'])

noz = OMEZarr(newdir)
noz.baseobj.typeobj
noz.baseobj.path_order
noz.baseobj.get_scale('0')
noz.baseobj.get_scale('1')

noz.baseobj.chunks('0')
noz.baseobj.rechunk((10, 70, 70))
noz.baseobj.chunks('0')

