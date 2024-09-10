import warnings, time, shutil, zarr, itertools, multiprocessing, re, numcodecs, dask, os, copy, inspect, napari
import numpy as np
import dask.array as da

from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from skimage import transform
from joblib import Parallel, delayed, parallel_backend
import numcodecs; numcodecs.blosc.use_threads = False

# from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.utils.general_utils import *
from ome_zarr_pyramid.utils.assignment_utils import assign_array
from skimage.util import view_as_blocks



### Downscaling methods
def blockwise_median(a, factors):
    """https://stackoverflow.com/questions/50485458/median-downsampling-in-python"""
    assert a.ndim == len(factors), \
        "blocks must have same dimensionality as the input image"
    # assert not (np.array(a.shape) % factors).any(), \
    #     "factors must divide cleanly into the input image shape"
    block_view = view_as_blocks(a, factors)
    assert block_view.shape[a.ndim:] == factors
    block_axes = [*range(a.ndim, 2*a.ndim)]
    return np.median(block_view, axis=block_axes)

def blockwise_min(a, factors):
    """https://stackoverflow.com/questions/50485458/median-downsampling-in-python"""
    assert a.ndim == len(factors), \
        "blocks must have same dimensionality as the input image"
    assert not (np.array(a.shape) % factors).any(), \
        "factors must divide cleanly into the input image shape"
    block_view = view_as_blocks(a, factors)
    assert block_view.shape[a.ndim:] == factors
    block_axes = [*range(a.ndim, 2*a.ndim)]
    return np.min(block_view, axis=tuple(block_axes))

def blockwise_max(a, factors):
    """https://stackoverflow.com/questions/50485458/median-downsampling-in-python"""
    assert a.ndim == len(factors), \
        "blocks must have same dimensionality as the input image"
    assert not (np.array(a.shape) % factors).any(), \
        "factors must divide cleanly into the input image shape"
    block_view = view_as_blocks(a, factors)
    assert block_view.shape[a.ndim:] == factors
    block_axes = [*range(a.ndim, 2*a.ndim)]
    return np.max(block_view, axis=tuple(block_axes))

def downscale_local_softmax(block, factors):
    padder = np.sum((factors, block.shape), axis = 0) % factors
    padder = tuple([(0, item) for item in padder])
    padded = np.pad(block, padder)
    min = blockwise_min(padded, factors)
    max = blockwise_max(padded, factors)
    resized = np.where(min > 0, min, max)
    return resized

##################################


def _calculate_input_slice_bases(input_array, input_block_size):
    input_shape = np.array(input_array.shape)
    bases = calculate_slice_bases(input_shape, input_block_size)
    return bases

def _calculate_input_slices(input_array, input_block_size):
    bases = _calculate_input_slice_bases(input_array, input_block_size)
    slcs = get_slices_from_slice_bases(bases)
    return slcs

def _map_input_base_to_output_base(input_slice_base: Tuple[np.ndarray], scale_factor):
    stacked = np.vstack(input_slice_base).T
    factor = np.array(scale_factor).reshape(1, -1)
    rescaled = np.divide(stacked, factor)
    rescaled[0] = np.floor(rescaled[0])
    rescaled[1] = np.ceil(rescaled[1])
    output_slice_base = tuple(rescaled.T.astype(int))
    return output_slice_base

def _calculate_output_slice_bases(input_array, input_block_size, scale_factor):

    assert np.all(np.remainder(input_block_size, scale_factor) == 0)
    input_slice_bases = _calculate_input_slice_bases(input_array, input_block_size)

    bases = [_map_input_base_to_output_base(input_base, scale_factor) for input_base in input_slice_bases]
    return bases

def _calculate_output_slices(input_array, input_block_size, scale_factor):
    bases = _calculate_output_slice_bases(input_array, input_block_size, scale_factor)
    slcs = get_slices_from_slice_bases(bases)
    return slcs

def _downscale_step(input_array,
                    rootpath,
                    basename,
                    scale_factor,
                    min_input_block_size = None,
                    overwrite = False,
                    n_jobs = 8,
                    downscale_func = transform.downscale_local_mean
                    #**kwargs
                    ):

    assert isinstance(input_array, zarr.Array)
    if isinstance(input_array.store, zarr.DirectoryStore):
        nextpath = str(int(basename) + 1)
        output_store = os.path.join(rootpath, nextpath)
    else:
        output_store = zarr.MemoryStore()
    if min_input_block_size is None:
        input_block_size = input_array.chunks
    else:
        input_block_size = min_input_block_size
    input_block_size = make_divisible(input_block_size, scale_factor)
    input_slices = _calculate_input_slices(input_array, input_block_size)
    output_shape = calculate_output_shape(input_array.shape, scale_factor)
    output_slices = _calculate_output_slices(input_array, input_block_size, scale_factor)
    output_array = create_like(input_array, shape = output_shape, store = output_store, overwrite = overwrite, #**kwargs
                                )
    output_array = assign_array(dest = output_array,
                                source = input_array,
                                dest_slices = output_slices,
                                source_slices = input_slices,
                                func = downscale_func,
                                factors = scale_factor,
                                n_jobs = n_jobs
                                )
    return output_array

def downscale_multiscales(input_array: zarr.Array, # top level array
                          rootpath: str, # group path of all arrays
                          n_layers: Union[Tuple, List],
                          scale_factor: Union[Tuple, List],
                          downscale_func = transform.downscale_local_mean,
                          min_input_block_size: tuple = None,
                          overwrite_layers: tuple = False,
                          n_jobs = 8,
                          # **kwargs
                          ) -> dict:
    """
    :param input_array:
    :param n_layers:
    :param scale_factor:
    :param min_input_block_size:
    :param overwrite_layers:
    :return:

    This function is used to rescale the arrays of a Pyramid object.
    """

    paths = np.arange(n_layers - 1)
    multiscales = {str(paths[0]): input_array}
    processed = input_array
    scale_factor = np.floor(np.array(scale_factor) + 0.5).astype(int)

    for i in paths:
        factor = tuple(scale_factor)
        basename = str(i)
        processed = _downscale_step(processed,
                                    rootpath,
                                    basename,
                                    scale_factor = factor,
                                    min_input_block_size = min_input_block_size,
                                    overwrite = overwrite_layers,
                                    n_jobs = n_jobs,
                                    downscale_func = downscale_func,
                                    # **kwargs
                                    )
        nextpath = str(i + 1)
        multiscales[nextpath] = processed
    return multiscales

def get_scale_factors_from_rescaled(rescaled: dict, refpath = '0'):
    assert refpath in rescaled.keys()
    mainshape = rescaled['0'].shape
    return {pth: np.divide(mainshape, item.shape) for pth, item in rescaled.items()}

def get_scales_from_rescaled(rescaled: dict,
                             refscale: (tuple, list),
                             refpath = '0'
                             ):
    factors = get_scale_factors_from_rescaled(rescaled, refpath)
    return {pth: tuple(np.multiply(refscale, factor).tolist()) for pth, factor in factors.items()}


