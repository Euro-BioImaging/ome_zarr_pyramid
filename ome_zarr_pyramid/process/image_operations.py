# from ome_zarr_pyramid.core.Hierarchy import OMEZarr, MultiScales
import warnings

from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.core import config, convenience as cnv
from ome_zarr_pyramid.process.parameter_control import FilterParams, BaseParams
from ome_zarr_pyramid.process import process_utilities as putils
from ome_zarr_pyramid.process.process_core import BaseProtocol

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

################################### CLASSES ############################################

class ApplyProjection(BaseProtocol):
    def __init__(self,
                 *args,
                 projection_type: str = 'max',
                 axis: str = 'z',
                 **kwargs
                 ):
        BaseProtocol.__init__(self,
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
        if axis not in self.input.axis_order:
            raise ValueError(
                f"The projection dimension {axis} is not included in the image dimensions, which are {self.input.axis_order}")
        oz = Pyramid()
        for rsl in self._base_params.resolutions:
            idx = self.input.axis_order.index(axis)
            scl = self.input.get_scale(rsl)
            arr = self.input[rsl]
            res = projections[projection_type](arr, axis=idx)
            res = da.expand_dims(res, idx)
            oz.add_layer(res,
                         rsl,
                         scale=scl,
                         axis_order=self.input.axis_order,
                         unitlist=self.input.unit_list,
                         zarr_meta={'chunks': res.chunksize,
                                    'dtype': self.input.dtype,
                                    'compressor': self.input.compressor,
                                    'dimension_separator': self.input.dimension_separator}
                         )
        super().__setattr__("output", oz)


######################################################################################

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
    projector = ApplyProjection(input = pyr,
                                 resolutions = resolutions,
                                 drop_singlet_axes = drop_singlet_axes,
                                 output_name = output_name
                                 )
    projector.run_cycle(axis = axis,
                        projection_type = projection_type
                        )
    projector.output.to_zarr(output_path, overwrite = overwrite)
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
    pyramids = putils.validate_pyramid_uniformity(pyramids, resolutions, axes = axes)
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


