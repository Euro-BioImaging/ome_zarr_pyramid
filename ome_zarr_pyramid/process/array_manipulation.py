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

class ArrayManipulation:
    def __init__(self,
                 pyr: Pyramid = None,
                 resolutions: list[str] = None,
                 drop_singlet_axes: bool = True,
                 output_name: str = 'nomen',
                 **kwargs
                 ):
        self.params = kwargs
        if pyr is None:
            warnings.warn(f"The input object is not of type Pyramid.")
        else:
            self.set_input(pyr, resolutions, drop_singlet_axes, output_name)
        self.run(**kwargs)
        self.get_output()
        print("One cycle run.")
    def __setattr__(self,
                    key,
                    value
                    ):
        setattr(self, key, value)
        self.__init__(self.pyr,
                      self.resolutions,
                      self.drop_singlet_axes,
                      self.output_name,
                      **self.params
                      )

    def set_input(self,
                  pyr: Pyramid = None,
                  resolutions: list[str] = None,
                  drop_singlet_axes: bool = True,
                  output_name: str = 'nomen'
                  ):
        self.pyr = aspyramid(pyr)
        if output_name is None:
            output_name = self.pyr.tag
        self.output_name = output_name
        if resolutions is None:
            resolutions = self.pyr.resolution_paths
        self.resolutions = resolutions
        self.drop_singlet_axes = drop_singlet_axes
    def run(self, **kwargs):
        print(f"No function selected.")
    def get_output(self):
        if self.drop_singlet_axes:
            self.pyr.drop_singlet_axes()
        self.pyr.retag(self.output_name)
        return self.pyr




################################### CLASSES ############################################


class ApplyProjection(ArrayManipulation):
    def __init__(self,
                 *args,
                 projection_type: str = 'max',
                 axis: str = 'z',
                 **kwargs
                 ):
        ArrayManipulation.__init__(self,
                                   *args,
                                   **kwargs,
                                   projection_type = projection_type,
                                   axis = axis
                                   )
    def run(self,
            projection_type: str = 'max',
            axis: str = 'z'
            ):
        projections = {
            'max': da.max,
            'min': da.min,
            'mean': da.mean,
            'median': da.median,
            'sum': da.sum
        }
        assert projection_type in projections.keys(), f'projection_type must be one of {projections.keys()}'
        if axis not in self.pyr.axis_order:
            raise ValueError(
                f"The projection dimension {axis} is not included in the image dimensions, which are {self.pyr.axis_order}")
        oz = Pyramid()
        for rsl in self.resolutions:
            idx = self.pyr.axis_order.index(axis)
            scl = self.pyr.get_scale(rsl)
            arr = self.pyr[rsl]
            res = projections[projection_type](arr, axis=idx)
            res = da.expand_dims(res, idx)
            oz.add_layer(res,
                         rsl,
                         scale=scl,
                         axis_order=self.pyr.axis_order,
                         unitlist=self.pyr.unit_list,
                         zarr_meta={'chunks': res.chunksize,
                                    'dtype': self.pyr.dtype,
                                    'compressor': self.pyr.compressor,
                                    'dimension_separator': self.pyr.dimension_separator}
                         )
        self.pyr = oz


######################################################################################



######################################################################################

def apply_projection(pyramid: (str, Pyramid) = None,
                     output_name: str = None,
                     resolutions: list[str] = None,
                     axis: str = 'z',
                     projection_type: str = 'max',
                     drop_singlet_axes=True,
                     ):
    """"""
    projections = {
        'max': da.max,
        'min': da.min,
        'mean': da.mean,
        'median': da.median,
        'sum': da.sum
    }
    pyr = aspyramid(pyramid)
    if resolutions is None:
        resolutions = pyr.resolution_paths
    assert projection_type in projections.keys(), f'projection_type must be one of {projections.keys()}'

    if output_name is None:
        output_name = pyr.tag
    if axis not in pyr.axis_order:
        raise ValueError(
            f"The projection dimension {axis} is not included in the image dimensions, which are {pyr.axis_order}")
    idx = pyr.axis_order.index(axis)
    newaxes = pyr.axis_order
    newunits = pyr.unit_list
    if drop_singlet_axes:
        newaxes = newaxes.replace(axis, '')
        newunits.pop(idx)
    oz = Pyramid()
    for rsl in resolutions:
        idx = pyr.axis_order.index(axis)
        scl = pyr.get_scale(rsl)
        arr = pyr[rsl]
        res = projections[projection_type](arr, axis=idx)
        if drop_singlet_axes:
            scl.pop(idx)
        else:
            res = da.expand_dims(res, idx)

        oz.add_layer(res, rsl, scale=scl, axis_order=newaxes, unitlist=newunits,
                     zarr_meta={'chunks': res.chunksize,
                                'dtype': pyr.dtype,
                                'compressor': pyr.compressor,
                                'dimension_separator': pyr.dimension_separator}
                     )

    return oz.retag(output_name)

def apply_projection_cmd(input_path: str,
                          output_path: str,
                          output_name: str = None,
                          resolutions: list[str] = None,
                          axis: str = 'z',
                          projection_type: str = 'max',
                          drop_singlet_axes: bool = True,
                          overwrite: bool = False
                          ):
    """"""
    pyr = Pyramid()
    pyr.from_zarr(input_path)
    res = apply_projection(pyr,
                           output_name,
                           resolutions,
                           axis,
                           projection_type,
                           drop_singlet_axes
                           )
    res.to_zarr(output_path, overwrite = overwrite)
    return

######################################################################################

def concatenate(pyramids: Iterable[Pyramid],
                 output_name: str = None,
                 resolutions: Union[str, list] = 'all',
                 axis: str = 'z',
                 ):
    """"""
    assert len(pyramids) > 0, f'Input list must be of length greater than 0.'
    if axis == 'all':
        axes = 'all'
    else:
        axes = ''.join([ax for ax in pyramids[0].axis_order if ax != axis])
    pyramids = validate_pyramid_uniformity(pyramids, resolutions, axes = axes)
    concat_layers = {}
    for pyramid in pyramids:
        for pth in pyramid.resolution_paths:
            if pth not in concat_layers.keys():
                concat_layers[pth] = []
            concat_layers[pth].append(pyramid.layers[pth])
    pyr = pyramids[0]
    if resolutions == 'all':
        resolutions = pyr.resolution_paths
    if output_name is None:
        output_name = pyr.tag
    axes = pyr.axis_order
    units = pyr.unit_list
    idx = axes.index(axis)
    zarr_meta = {}
    for rsl in resolutions:
        scl = pyr.get_scale(rsl)
        layers = concat_layers[rsl]
        res = da.concatenate(layers, idx, allow_unknown_chunksizes=False)
        zmeta = {
                 'array': res,
                 'axis_order': axes,
                 'unit_list': units,
                 'scale': scl,
                 'chunks': res.chunksize,
                 'shape': res.shape,
                 'compressor': pyr.compressor,
                 'dtype': pyr.dtype,
                 'dimension_separator': pyr.dimension_separator
        }
        zarr_meta[rsl] = zmeta
    pyr = Pyramid()
    pyr.from_dict(zarr_meta)
    return pyr.retag(output_name)

# def _collect_omezarrs(in_path,
#                       filter_by: Union[str, None] = None
#                       ):
#     names = os.listdir(in_path)
#     if filter_by is not None:
#         paths = [os.path.join(in_path, name) for name in names if filter_by in name]
#     else:
#         paths = [os.path.join(in_path, name) for name in names]
#     input_list = []
#     for pth in paths:
#         try:
#             input_list.append(OMEZarr(pth))
#         except:
#             pass
#     return input_list
#
def split_axes(pyr: Pyramid,
               axes: str = 'zc',
               steps: tuple = None
               ):
    if steps is None:
        steps = tuple([1] * len(axes))
    else:
        assert len(axes) == len(steps), f'Axes and steps must be of equal length.'
    axlens = [pyr.axislen(ax) for ax in axes]
    slices = []
    for ax, size, step in zip(axes, axlens, steps):
        splitters = [(i, i+step) for i in range(0, size, step)]
        slices.append(splitters)
    combins = list(itertools.product(*slices))
    subsets = []
    for combin in combins:
        slicer = {ax: slc for ax, slc in zip(axes, combin)}
        subset = pyr.copy().subset(slicer)
        subsets.append((slicer, subset))
    # for _, sub in subsets:
    #     print(sub.shape)
    return subsets

def split_layers():
    pass

def join_layers(pyramids: list,
                refpath = '0'
                ):
    pass

def subset():
    pass

def update_axis_order():
    pass

def merge_datasets():
    pass




# pyr = Pyramid()
# # pyr.from_dict({'0': {'array': np.zeros((50, 100, 100))}})
# pyr.from_zarr('/home/oezdemir/PycharmProjects/test_pyenv/OME_Zarr/data/filament4.zarr')
# rescaled = pyr.copy().rescale(scale_factor = 2, resolutions = 7)
# subset = rescaled.copy().subset({'z': (1,)})
# subset1 = rescaled.copy().subset([(0, 1), (10, 200)]).rescale(2, resolutions = 3)
# h = subset1 + subset1
# subset2 = rescaled.copy().subset(0:3,10:20).rescale(2, resolutions = 3)


# rescaled.to_zarr('/home/oezdemir/PycharmProjects/test_pyenv/OME_Zarr/data/rescaled1.zarr')
# rescaled1 = pyr.rescale(scale_factor = 1.5, resolutions = 3)
# rescaled2 =  pyr.rescale(scale_factor = 3, resolutions = 3)


# pyr._array_meta_['0']['compressor'] = pyr.layers['0']._meta['compressor']

# basepath1 = 'OME_Zarr/data/filament3.zarr'
# respath1 =  'OME_Zarr/data/res3.zarr'
# #
# pyr = Pyramid()
# pyr.from_zarr(basepath1)
# pyr.rechunk((12, 36, 72))
# pyr.rechunk((30, 100, 100))
# pyr.to_zarr(respath1, True)
#
# pyr2 = pyr + pyr
# pyr3 = pyr - pyr
#
# np.sum(pyr[0]).compute()
# np.sum(pyr2[0]).compute()
# np.sum(pyr3[0]).compute()
#
# pyr.mean('1')
# pyr2.mean('1')