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


def _calculate_input_slice_bases(input_array, input_block_size):
    input_shape = np.array(input_array.shape)
    bases = calculate_slice_bases(input_shape, input_block_size)
    return bases

def _calculate_input_slices(input_array, input_block_size):
    bases = _calculate_input_slice_bases(input_array, input_block_size)
    slcs = get_slices_from_slice_bases(bases)
    return slcs

def _map_input_base_to_output_base(input_slice_base: Tuple[np.ndarray], scale_factor):
    # input_slice_base = input_slice_bases[0]
    stacked = np.vstack(input_slice_base).T
    factor = np.array(scale_factor).reshape(1, -1)
    rescaled = np.divide(stacked, factor)
    rescaled[0] = np.floor(rescaled[0])
    rescaled[1] = np.ceil(rescaled[1])
    output_slice_base = tuple(rescaled.T.astype(int))
    # print(input_slice_base)
    # print(output_slice_base)
    return output_slice_base

def _calculate_output_slice_bases(input_array, input_block_size, scale_factor):
    ### TODO: output slice shapeleri input slice shapelerinin scale_factor ile bölünmesinden hesaplansin
    # scale_factor = (1, 2, 2)
    # input_block_size = (22, 22, 22)
    # input_array = zarr.zeros((100, 25, 25), chunks=(20, 20, 20))
    # input_block_size = make_divisible(input_block_size, scale_factor)

    assert np.all(np.remainder(input_block_size, scale_factor) == 0)
    input_slice_bases = _calculate_input_slice_bases(input_array, input_block_size)

    # output_block_size = calculate_output_block_size(input_block_size, scale_factor)
    # output_shape = calculate_output_shape(input_array.shape, scale_factor)
    # bases = calculate_slice_bases(output_shape, output_block_size)
    bases = [_map_input_base_to_output_base(input_base, scale_factor) for input_base in input_slice_bases]
    return bases

def _calculate_output_slices(input_array, input_block_size, scale_factor):
    bases = _calculate_output_slice_bases(input_array, input_block_size, scale_factor)
    slcs = get_slices_from_slice_bases(bases)
    return slcs

# def _get_root_path(zarr_array):
#     root = zarr_array.store.path
#     return root

# def _get_path_name(zarr_array):
#     pth = zarr_array.basename
#     return pth


def _downscale_step(input_array, # top level array from which to downscale
                    rootpath, # group path of all arrays
                    basename, # the basename of this input_array
                    scale_factor,
                    min_input_block_size = None,
                    overwrite = False,
                    n_jobs = 8
                    #**kwargs
                    ):
    # fpath = f"/home/oezdemir/PycharmProjects/dask_distributed/ome_zarr_pyramid/data3/filament.zarr/0"
    # basename = '0'
    # rootpath = f"/home/oezdemir/PycharmProjects/dask_distributed/ome_zarr_pyramid/data3/filament.zarr"
    # input_array = zarr.open_array(fpath, chunks=(11, 11, 11))
    # scale_factor = (1, 3, 3)
    # min_input_block_size = None
    # overwrite = True
    # n_jobs = 8

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
    # print(input_block_size)
    input_block_size = make_divisible(input_block_size, scale_factor)
    # print(input_block_size)
    # print(scale_factor)
    input_slices = _calculate_input_slices(input_array, input_block_size)
    output_shape = calculate_output_shape(input_array.shape, scale_factor)
    output_slices = _calculate_output_slices(input_array, input_block_size, scale_factor)
    output_array = create_like(input_array, shape = output_shape, store = output_store, overwrite = overwrite, #**kwargs
                                )
    output_array = assign_array(dest = output_array,
                                source = input_array,
                                dest_slices = output_slices,
                                source_slices = input_slices,
                                func = transform.downscale_local_mean,
                                factors = scale_factor,
                                n_jobs = n_jobs
                                )
    # for in_slice, out_slice in zip(input_slices, output_slices): # TODO: parallelize
    #     output_array[out_slice] = transform.downscale_local_mean(input_array[in_slice], scale_factor)
    #     ### TODO: Branch out for grayscale and label images.
    return output_array

def downscale_multiscales(input_array: zarr.Array, # top level array
                          rootpath: str, # group path of all arrays
                          n_layers: Union[Tuple, List],
                          scale_factor: Union[Tuple, List],
                          min_input_block_size: tuple = None,
                          overwrite_layers: tuple = False,
                          n_jobs = 8
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
    # n_layers = 5
    # scale_factor = (1, 2, 2)
    # min_input_block_size = (21, 21, 21)
    # arr = np.random.rand(300, 300, 300)
    # refarray = zarr.array(arr, chunks=(11, 11, 11),
    #                       store=f"/home/oezdemir/PycharmProjects/dask_distributed/ome_zarr_pyramid/data3/test1/0")
    # overwrite_layers = True
    # input_array = refarray
    # n_jobs = 8

    paths = np.arange(n_layers - 1)
    multiscales = {str(paths[0]): input_array}
    processed = input_array
    scale_factor = np.floor(np.array(scale_factor) + 0.5).astype(int)

    for i in paths:
        factor = tuple(scale_factor)
        basename = str(i)
        # if i < len(scale_factor):
        #     if hasattr(scale_factor[i], '__len__'):
        #         factor = tuple(scale_factor[i])
        processed = _downscale_step(processed,
                                    rootpath,
                                    basename,
                                    scale_factor = factor,
                                    min_input_block_size = min_input_block_size,
                                    overwrite = overwrite_layers,
                                    n_jobs = n_jobs
                                    # **kwargs
                                    )
        # if not isinstance(input_array.store, zarr.DirectoryStore):
        nextpath = str(i + 1)
        # else:
        #     nextpath = _get_path_name(processed)
        multiscales[nextpath] = processed
        # print(multiscales.keys())
    return multiscales

# fpath = f"/home/oezdemir/PycharmProjects/dask_distributed/ome_zarr_pyramid/data3/filament.zarr/0"
# gpath = f"/home/oezdemir/PycharmProjects/dask_distributed/ome_zarr_pyramid/data3/filament.zarr"
# # gr = zarr.group(gpath)
# # list(gr.keys())
# # del gr[3]
# # gr = zarr.group(gpath, overwrite = True)
# arrz = zarr.open_array(fpath, chunks = (11, 11, 11))
# scale_factor = (1, 3, 3)
# arrz_dsc = downscale_multiscales(arrz, rootpath = gpath, n_layers = 5, scale_factor = scale_factor, overwrite_layers = True)

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

# scales = get_scales_from_rescaled(arrz_dsc, refscale = (0.5, 0.25, 0.5))
# v = napari.Viewer()
# v.add_image(arrz)
# v.add_image(arrz_dsc['0'])
# v.add_image(arrz_dsc['1'])
