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


