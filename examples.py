# from OME_Zarr.src.OMEZarr import OMEZarr
import pandas as pd

from src.core import utils
# from OME_Zarr.src.OMEZarr import OMEZarr
from src.core.Hierarchy import OMEZarr
import numpy as np
import dask.array as da
import dask.bag as db
import os, copy
import numcodecs
import zarr, napari
import dask_image.ndfilters
import dask_image.ndmorph
import dask_image.dispatch
import dask_image.ndmeasure as ndmeasure
import dask as da
import pandas as pd

# basepath = 'OME_Zarr/data/filament_plus.zarr'
basepath = '/home/oezdemir/tests/concat-stack-crop1.zarr'
basepath = 'data/filament.zarr'
# basepath = '/home/oezdemir/tests/concat-stack-crop1.zarr'

# newdir = 'data/filament_plus2_labels_extracted.zarr'
data1 = OMEZarr(basepath)

lbl0 = da.array.from_zarr(data1.labels.original.layers['0'])



slcs = utils.locate_labels(lbl0)

index = da.array.unique(lbl0).compute()
areas = ndmeasure.labeled_comprehension(lbl0, lbl0, index, np.count_nonzero, object, False)
hareas = areas.compute()






data1.labels.original.labelmeta

lbls1 = data1.labels
lbls1.has_multiscales_meta()

dir(lbls1)
dir(data1)

lbls1.rebase(newdir = newdir,
             include_subdirs = True,
             paths = ['0', '1'],
             subset = {'z': (10, 21), 'y': (50, 100)}) ### Note that if you do this with hybrid, only the top-level multiscales will be rebased now.



lblsread = OMEZarr(newdir)
lbls2 = lblsread.original
lbl0 = da.array.from_zarr(data1.labels.original.layers['0'])
# lbl0 = da.array.from_zarr(lbls2.layers['0'])
mask = lbl0 == 10
# h = np.unique(lbl0)
dir(ndmeasure)




hhh = np.where(lbl0)
for item in hhh:
    c = item.max()
    c.compute()
    print(item.max())


lblints = ndmeasure.find_objects(lbl0)
clblints = lblints.compute()
dir(lblints)
len(lblints)





roi = lblints.loc[10].compute()
roii = tuple(roi.to_numpy().tolist()[0])
sliced = lbl0[roii]
# napari.view_image(res0)
napari.view_image(sliced)
# napari.view_image(roi)



# h1 = data1.grs['filament_plus1.zarr/labels/original']
# h2 = data1.labels.original

# data1.extract(newdir = 'OME_Zarr/data/filament_plus1_extracted_full.zarr') ### Note that if you do this with hybrid, only the top-level multiscales will be rebased now.




# data1.rebase_multiscales('OME_Zarr/data/filament_plus1_rebased_full.zarr')
# data.rebase(newdir = 'OME_Zarr/data/filament_plus1_rebased.zarr') ### Note that if you do this with hybrid, only the top-level multiscales will be rebased now.

# h = data.grs['other/original']

# h1 = data[0]
#
#
# h.rebase(newdir = 'OME_Zarr/data/filament_plus1_others_original0.zarr')
#
#

# arr = np.array(h.layers['0'])
# hh = utils.index_nth_dimension(arr, dimensions = [0, 1], intervals = [(12, 27), (50, 100)])

h.rebase(newdir = 'OME_Zarr/data/filament_plus1_others_original_subset.zarr', subset = {'z': (10, 21), 'y': (50, 100)})

h.resolution_paths
h.grpath
# h.reduce_pyramid(['0', '1'])

h.split_indices('z')

h.rebase(newdir = 'OME_Zarr/data/filament_plus1_others_original.zarr')
h.rebase(newdir = 'OME_Zarr/data/filament_plus1_others_original1.zarr', paths = ['0', '1'])

h.chunks()

h.rechunk([7, 70, 70])

h.chunks()
h['0'].chunks

h.asflat()

data.other.original.layers['0'].chunks

dir(h)

h.chunks('0')
h.array_meta







h = dict(vox.groups())

vox.has_axes
lvox = LabelVoxel(basepath)

for name, gr in vox.groups():
    print(gr)

hh = Voxel(gr)
zarr.Group.__init__(gr)


vox.basename
dir(vox)

oz = OMEZarr(basepath)




# oz.baseobj.typeobj
# oz.baseobj.path_order
# oz.baseobj.dtype('0')
# oz.baseobj.chunks('0')

# oz.labelobj.image_labels['original'].typeobj
# oz.labelobj.image_labels['original'].dtype('0')

newdir = 'OME_Zarr/data/filament_ex.zarr'
# oz.baseobj.array_meta['0']['dimension_separator']
# oz.baseobj.copy(newdir, overwrite = True)

oz = OMEZarr(newdir)
oz.baseobj.rebase(newdir = newdir, paths = ['0', '1', '2'])

newnewdir = 'OME_Zarr/data/filament_ex_ex.zarr'


# dir(zarr)

# gr = zarr.group(newnewdir, overwrite = True)




subset = {'z': 12,
          'y': (20, 140),
          'x': (20, 150)
          }
# h = oz.baseobj.get_sliced_layers(subset)
# oz.baseobj.extract_dataset(None, 'yx')

oz.baseobj.rebase(newdir = newnewdir, paths = ['0', '1'])

oz.baseobj.reslice(subset)

oz.baseobj.rebase(newdir = newnewdir, paths = ['0', '1'], subset = subset)
oz.baseobj.basepath == newnewdir

h = oz.baseobj.base['0']
hd = da.from_zarr(h)
hd.chunks
hd.chunksize

oz.baseobj.extract(newdir = newdir, paths = ['0', '1'])

noz = OMEZarr(newdir)
noz = OMEZarr(newnewdir)

obj = noz.baseobj

met = obj.extract_dataset(['0'], axes = 'yx')











class Footballer:
    def __init__(self):
        print('I am footballer')

class Basketballer:
    def __init__(self):
        print('I am basketballer')
    def shoot(self):
        print('I can shoot as basketballer')

class Player:
    def __new__(cls, name):
        if name == 'footballer':
            obj = object.__new__(Footballer)
        elif name == 'basketballer':
            obj = object.__new__(Basketballer)
        else:
            obj = object.__new__(Player)
        return obj
    def __init__(self, name):
        print("Init is called")

h = Player('footballers')


class Team(Player, Basketballer):
    def __init__(self, name):
        item = Player.__new__()
        Player.__init__(self, name)


h = Team('footbsaller')










noz.baseobj.typeobj
noz.baseobj.path_order

noz.baseobj.chunks()
noz.baseobj.rechunk((10, 70, 70))
noz.baseobj.chunks()

noz.baseobj.get_scale('0')
noz.baseobj.get_scale('1')

noz.baseobj.asflat()
noz.baseobj.asnested()

noz.baseobj.dtype('0')
noz.baseobj.astype(np.float32)
noz.baseobj.dtype('0')

napari.view_path(oz.baseobj.basepath)


















