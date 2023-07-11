from OME_Zarr.src.OMEZarr import OMEZarr, OMEZarrObject as ozobj
from OME_Zarr.src.core import utils
from backups.oldversions.OMEZarr_v2 import OMEZarr as omezarr
import numcodecs, numpy as np
import napari

basepath = 'OME_Zarr/data/filament.zarr'
newpath = '/home/oezdemir/PycharmProjects/ZarrSeg/OME_Zarr/data/filament_ex.zarr'
newpath1 = '/home/oezdemir/PycharmProjects/ZarrSeg/OME_Zarr/data/filament_ex1.zarr'

obj = ozobj(newpath)

obj.dtype('0')

obj.rebase(newpath1, paths = ['0', '1'], zarr_meta = {'dtype': np.float32})

obj.dtype('0')

obj.rebase(newpath1, paths = ['0', '1'], zarr_meta = {'dimension_separator': '.'})
obj.asflat()

obj1 = ozobj(newpath1)
obj1.rebase(newpath1, paths = ['0', '1'], zarr_meta = {'dimension_separator': '.'})


obj1.dtype('0')

napari.view_path(newpath)

napari.view_path(newpath1)