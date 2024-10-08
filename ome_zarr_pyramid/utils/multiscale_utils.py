import warnings, time, shutil, zarr, itertools, multiprocessing, re, numcodecs, dask, os, copy, inspect
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

# def _parse_syncdir(input, syncdir = None):
#     if syncdir is not None and syncdir != 'same' and syncdir != 'default':
#         pass
#     elif syncdir == 'default':
#         syncdir = os.path.expanduser('~') + '/.syncdir'
#     elif syncdir == 'same':
#         if input.synchronizer is not None:
#             syncdir = input.synchronizer.path
#         else:
#             warnings.warn(f"The input array has no synchronizer. Default synchronizer path is being used.")
#             syncdir = os.path.join(os.path.expanduser('~'), '.syncdir', self.basename)
#             syncdir = os.path.expanduser('~') + '/.syncdir'
#     else:
#         syncdir = None
#     return syncdir

def _downscale_step(input_array,
                    root: zarr.Group,
                    basename,
                    scale_factor,
                    min_input_block_size = None,
                    overwrite = False,
                    n_jobs = 8,
                    downscale_func = transform.downscale_local_mean,
                    use_synchronizer = 'multiprocessing',
                    syncdir = None,
                    backend = 'dask',
                    verbose = True
                    #**kwargs
                    ):

    assert isinstance(input_array, zarr.Array)
    if min_input_block_size is None:
        input_block_size = input_array.chunks
    else:
        input_block_size = min_input_block_size
    input_block_size = make_divisible(input_block_size, scale_factor)
    input_slices = _calculate_input_slices(input_array, input_block_size)
    output_shape = calculate_output_shape(input_array.shape, scale_factor)
    output_slices = _calculate_output_slices(input_array, input_block_size, scale_factor)

    # print(f"syncdir: {syncdir}")
    output_array = create_in_group_like(input_array,
                                     gr = root,
                                     shape = output_shape,
                                     # store = output_store,
                                     overwrite = overwrite, #**kwargs
                                     use_synchronizer = use_synchronizer,
                                     syncdir = syncdir,
                                     path_in_group = basename
                                     )

    # print(f'output_sync: {output_array.synchronizer.path}')
    output_array = assign_array(dest = output_array,
                                source = input_array,
                                dest_slices = output_slices,
                                source_slices = input_slices,
                                func = downscale_func,
                                factors = scale_factor,
                                n_jobs = n_jobs,
                                backend = backend,
                                verbose = verbose
                                )
    # print(f'output: {output_array.synchronizer}')
    return output_array

def downscale_multiscales(
                          #input_array: zarr.Array, # top level array
                          root: (str, zarr.Group), # group path of all arrays
                          n_layers: Union[Tuple, List],
                          scale_factor: Union[Tuple, List],
                          downscale_func = transform.downscale_local_mean,
                          min_input_block_size: tuple = None,
                          overwrite_layers: tuple = False,
                          n_jobs = 8,
                          use_synchronizer: str = None,
                          syncdir: str = None,
                          refpath = None,
                          backend='dask',
                          verbose=True
                          # **kwargs
                          ) -> zarr.Group:
    """
    :param input_array:
    :param n_layers:
    :param scale_factor:
    :param min_input_block_size:
    :param overwrite_layers:
    :return:

    This function is used to rescale the arrays of a Pyramid object.
    """

    # handle the synchronizer
    if not isinstance(root, zarr.Group):
        root = zarr.group(root)
    rootpath = root.store.path
    pyrname = os.path.basename(rootpath)
    if syncdir is not None and syncdir != 'same' and syncdir != 'default':
        pass
    elif syncdir == 'default':
        syncdir = os.path.join(os.path.expanduser('~'), '.syncdir', pyrname)
    elif syncdir == 'same':
        if root.synchronizer is not None:
            syncdir = root.synchronizer.path
        else:
            warnings.warn(f"The input array has no synchronizer. Default synchronizer path is being used.")
            syncdir = os.path.join(os.path.expanduser('~'), '.syncdir', pyrname)
    else:
        syncdir = None

    try:
        shutil.rmtree(syncdir)
    except:
        pass
    ###

    paths = np.arange(n_layers - 1)
    synchronizer = zarr.ProcessSynchronizer(syncdir)
    multiscales = root

    keys = sorted([int(i) for i in multiscales.array_keys()])
    if refpath is None:
        refpath = keys[0]

    processed = multiscales[refpath]
    scale_factor = np.floor(np.array(scale_factor) + 0.5).astype(int)

    # Delete non-reference arrays
    for key in keys:
        if key != refpath:
            multiscales.pop(key)

    for i in paths:
        factor = tuple(scale_factor)
        basename = str(i + 1)
        try:
            syncpath = synchronizer[basename].path
        except:
            syncpath = synchronizer.path
        processed = _downscale_step(processed,
                                    root = multiscales,
                                    basename = basename,
                                    scale_factor = factor,
                                    min_input_block_size = min_input_block_size,
                                    overwrite = overwrite_layers,
                                    # overwrite=True,
                                    n_jobs = n_jobs,
                                    downscale_func = downscale_func,
                                    use_synchronizer = use_synchronizer,
                                    syncdir = syncpath,
                                    backend = backend,
                                    verbose = verbose
                                    # **kwargs
                                    )
    return multiscales

def get_scale_factors_from_rescaled(rescaled: zarr.Group, refpath = '0'):
    assert refpath in rescaled.keys()
    mainshape = rescaled['0'].shape
    return {pth: np.divide(mainshape, item.shape) for pth, item in rescaled.items()}

def get_scales_from_rescaled(rescaled: zarr.Group,
                             refscale: (tuple, list),
                             refpath = '0'
                             ):
    factors = get_scale_factors_from_rescaled(rescaled, refpath)
    return {pth: tuple(np.multiply(refscale, factor).tolist()) for pth, factor in factors.items()}


