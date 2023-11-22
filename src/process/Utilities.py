from src.core.Hierarchy import OMEZarr
from src.core import utils

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

from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    Iterable,
    List,
    Optional
)

def apply_projection(path_or_omezarr,
                     output_directory,
                     output_name = None,
                     resolutions: Union = 'all',
                     along = 'z',
                     projection_type = 'max',
                     label_name: Union[str, None] = None
                     ):
    projections = {
        'max': da.max,
        'min': da.min,
        'mean': da.mean,
        'median': da.median
    }
    if hasattr(path_or_omezarr, 'grname'):
        inputs = path_or_omezarr
    elif isinstance(path_or_omezarr, (str, Path)):
        inputs = OMEZarr(path_or_omezarr)
    if not inputs.is_multiscales:
        raise TypeError("Currently, projections can only be applied to multiscales groups.")
    if label_name is None:
        data = inputs
    else:
        assert inputs.is_label_image, f'Input is not  a label image.'
        label_path = inputs.labels.base_attrs['labels']
        assert label_name in label_path, f'The given label_name is not in the label path: {label_path}'
        labels = inputs.labels
        data = getattr(labels, label_name)
    if resolutions == 'all':
        resolutions = data.resolution_paths
    assert projection_type in projections.keys(), f'projection_type must be one of {projections.keys()}'
    resolutions = [str(rsl) for rsl in resolutions]
    if output_name is None: output_name = data.identifier
    # print(output_name)
    idx = data.axis_order.index(along)
    newshape = list(data.shape())
    newshape.pop(idx)
    newchunks = list(data.chunks())
    newchunks.pop(idx)
    axes = data.axis_order.replace(along, '')
    units = data.unit_list
    # print(units)
    oz = OMEZarr(os.path.join(output_directory, f'{output_name}_res-{resolutions}_{projection_type}-project.ome.zarr'),
                 shape_of_new = newshape,
                 chunks_of_new = newchunks,
                 axis_order = axes,
                 unit_order= units
                 )
    for rsl in resolutions:
        z_ax = data.axis_order.index(along)
        scl = data.get_scale(rsl)
        arr_z = data[rsl]
        arr = da.from_zarr(arr_z)
        proj = projections[projection_type](arr, axis = z_ax)
        proj_scale = tuple([scl[i] for i in range(len(scl)) if i != z_ax])
        zmeta = {'chunks': proj.chunksize,
                 'shape': proj.shape,
                 'compressor': data.compressor(),
                 'dtype': data.dtype(),
                 'dimension_separator': data.dimension_separator()
                 }
        if rsl in oz.resolution_paths:
            oz.del_dataset(rsl)
        if along in oz.axis_order:
            oz.del_axis(along)
        oz.set_array(rsl, proj, scale = proj_scale, zarr_meta = zmeta)
    return oz
