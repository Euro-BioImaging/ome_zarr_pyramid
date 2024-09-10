import copy, inspect, itertools, os, zarr
from attrs import define, field, setters
import numcodecs; numcodecs.blosc.use_threads = False
from joblib import Parallel, delayed, parallel_backend
import numpy as np

from ome_zarr_pyramid.utils.general_utils import *


def _assign_block(dest: zarr.Array,
                  source: zarr.Array,
                  slc_dest: slice,
                  slc_source: slice,
                  func = None,
                  **func_args
                  ):
    if func is None:
        dest[slc_dest] = source[slc_source]
    else:
        dest[slc_dest] = func(source[slc_source], **func_args)
    return dest

def assign_array(dest: zarr.Array, # TODO: think about adding synchronizer to this.
                 source: zarr.Array,
                 crop_indices: tuple = None,
                 insert_at: (list, tuple, np.ndarray) = None,
                 block_size = None,
                 dest_slices = None,
                 source_slices = None,
                 func = None,
                 n_jobs = 8,
                 require_sharedmem = None,
                 **func_args
                 ):
    if crop_indices is None:
        crop_indices = tuple([(0, item) for item in source.shape])
    if insert_at is None:
        insert_at = (0, 0, 0)
    t = np.array(crop_indices)
    maxima = t[:, 1]
    minima = t[:, 0]
    shape = maxima - minima
    if block_size is None:
        block_size = source.chunks
    if source_slices is None:
        source_indices = calculate_slice_bases(
                                                crop_indices = crop_indices,
                                                block_size = block_size,
                                                max_shape = source.shape
                                                )
        source_slices = get_slices_from_slice_bases(source_indices)
    if dest_slices is None:
        insert_minima = np.array(insert_at)
        insert_maxima = np.add(shape, insert_at)
        dest_location = tuple(np.vstack((insert_minima, insert_maxima)).T.tolist())
        dest_indices = calculate_slice_bases(
                                              crop_indices = dest_location,
                                              block_size = block_size,
                                              max_shape = dest.shape
                                              )
        dest_slices = get_slices_from_slice_bases(dest_indices)


    has_synchronizer = False
    if hasattr(dest, 'synchronizer'):
        has_synchronizer = dest.synchronizer is not None

    if not isinstance(dest.store, zarr.DirectoryStore) or not has_synchronizer:
        for slc_source, slc_dest in zip(source_slices, dest_slices):
            dest = _assign_block(dest = dest, source = source, slc_dest = slc_dest, slc_source = slc_source, func = func, **func_args)
    else:
        with parallel_backend('multiprocessing'):
            with Parallel(n_jobs = n_jobs,
                          require = require_sharedmem
                          ) as parallel:
                _ = parallel(
                    delayed(_assign_block)(dest = dest, source = source, slc_dest = slc_dest, slc_source = slc_source, func = func, **func_args)
                    for slc_source, slc_dest in zip(source_slices, dest_slices)
                )
    return dest

def basic_assign(dest: zarr.Array, # TODO: think about adding synchronizer to this.
                 source: zarr.Array,
                 dest_slice: tuple = None, # TODO: support dict
                 source_slice: (list, tuple, np.ndarray) = None, # TODO: support dict
                 block_size = None,
                 n_jobs = 8,
                 require_sharedmem = None,
                 ):
    if dest_slice is None:
        dest_slice = tuple([0, size] for size in dest.shape)
    if source_slice is None:
        source_slice = tuple([0, size] for size in source.shape)

    assert isinstance(dest_slice, (tuple, list, np.ndarray))
    dslc = np.array(dest_slice)
    assert dslc.ndim == 2
    assert dslc[:, 0].size == dest.ndim

    if np.any(np.greater(dslc[:, 1], dest.shape)):
        raise ValueError(f"Destination slice outside destination array's shape.")
    if np.any(np.less(dslc[:, 0], 0)):
        raise ValueError(f"Destination slice outside destination array's shape.")

    if not np.isscalar(source):
        assert isinstance(source_slice, (tuple, list, np.ndarray))
        sslc = np.array(source_slice)
        assert sslc.ndim == 2
        assert sslc[:, 0].size == source.ndim

        if np.any(np.greater(sslc[:, 1], source.shape)):
            raise ValueError(f"Source slice outside source array's shape.")
        if np.any(np.less(sslc[:, 0], 0)):
            raise ValueError(f"Source slice outside source array's shape.")

    sshape = sslc[:, 1] - sslc[:, 0]
    dshape = dslc[:, 1] - dslc[:, 0]
    assert np.allclose(sshape, dshape), f"The source and destination slices must be equal in shape."

    #######
    if block_size is None:
        try:
            block_size = source.chunks
        except: # if source is a numpy array, it does not have chunks
            block_size = dest.chunks
    source_indices = calculate_slice_bases(
                                            crop_indices = sslc,
                                            block_size = block_size,
                                            max_shape = source.shape
                                            )
    source_slices = get_slices_from_slice_bases(source_indices)

    insert_minima = dslc[:, 0]
    insert_maxima = dslc[:, 1]
    dest_location = tuple(np.vstack((insert_minima, insert_maxima)).T.tolist())
    dest_indices = calculate_slice_bases(
                                          crop_indices = dest_location,
                                          block_size = block_size,
                                          max_shape = dest.shape
                                          )
    dest_slices = get_slices_from_slice_bases(dest_indices)

    has_synchronizer = False
    if hasattr(dest, 'synchronizer'):
        has_synchronizer = dest.synchronizer is not None

    if not isinstance(dest.store, zarr.DirectoryStore) or not has_synchronizer:
        for slc_source, slc_dest in zip(source_slices, dest_slices):
            dest = _assign_block(dest = dest, source = source, slc_dest = slc_dest, slc_source = slc_source)
    else:
        with parallel_backend('multiprocessing'):
            with Parallel(n_jobs = n_jobs,
                          require = require_sharedmem
                          ) as parallel:
                _ = parallel(
                    delayed(_assign_block)(dest = dest, source = source, slc_dest = slc_dest, slc_source = slc_source)
                    for slc_source, slc_dest in zip(source_slices, dest_slices)
                )
    return dest
