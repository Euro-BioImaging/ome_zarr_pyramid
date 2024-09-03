import warnings, time, shutil, zarr, itertools, multiprocessing, re, numcodecs, dask, os, copy, inspect
import numpy as np
import dask.array as da

from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from skimage import transform
from joblib import Parallel, delayed, parallel_backend
import numcodecs; numcodecs.blosc.use_threads = False

# from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection


def make_divisible(base, divisor):
    remainder = np.remainder(base, divisor)
    return np.add(base, divisor).astype(int) - remainder

def calculate_output_block_size(input_block_size, scale_factor):
    assert np.all(np.remainder(input_block_size, scale_factor)) == 0
    return np.ceil(np.divide(input_block_size, scale_factor)).astype(int)

def calculate_output_shape(input_shape, scale_factor):
    return np.ceil(np.divide(input_shape, scale_factor)).astype(int)

def get_slices_from_slice_bases(combin):
    slices = []
    for triple in combin:
        slcs = tuple([slice(*item) for item in triple])
        slices.append(slcs)
    return slices

# def _calculate_slice_bases(shape, block_size):
#     assert len(shape) == len(block_size)
#     indices = [np.arange(0, item, increment) for item, increment in zip(shape, block_size)]
#     for idx, (item, shape) in enumerate(zip(indices, shape)):
#         indices[idx] = np.append(item, shape)
#     bases = []
#     for item in indices:
#         joined = np.vstack((item[:-1], item[1:])).T.astype(int)
#         bases.append(joined)
#     combin = list(itertools.product(*bases))
#     return combin

def calculate_slice_bases(
                           crop_indices,
                           block_size,
                           max_shape = None
                           ):
    t = np.array(crop_indices)
    if t.ndim > 1:
        maxima = t[:, 1]
        minima = t[:, 0]
    else:
        minima = np.zeros_like(t)
        maxima = t.copy()
    shape = maxima - minima
    if not max_shape is None:
        assert not np.any(np.greater(maxima, max_shape))
    assert len(shape) == len(block_size)
    indices = [np.arange(mini, maxi, increment) for mini, maxi, increment in zip(minima, maxima, block_size)]
    for idx, (item, size) in enumerate(zip(indices, maxima)):
        indices[idx] = np.append(item, size)
    bases = []
    for item in indices:
        joined = np.vstack((item[:-1], item[1:])).T.astype(int)
        bases.append(joined)
    combin = list(itertools.product(*bases))
    return combin


def create_like(zarray,
                store = None, # leave as None for KVStore
                use_synchronizer  = False,
                syncdir = None,
                **kwargs
                ):
    # handle different synchronizer scenarios.
    synchronizer = None
    if store is None:
        store = zarr.MemoryStore()
    elif isinstance(store, str) and not store.startswith('http'):
        if 'dimension_separator' in kwargs.keys():
            dimension_separator = kwargs.get('dimension_separator')
        else:
            # print("even here")
            if hasattr(zarray, '_dimension_separator'):
                if zarray._dimension_separator is None:
                    dimension_separator = '.'
                else:
                    dimension_separator = zarray._dimension_separator
            elif hasattr(zarray.store, '_dimension_separator'):
                if zarray.store._dimension_separator is None:
                    dimension_separator = '.'
                else:
                    dimension_separator = zarray.store._dimension_separator
            else:
                dimension_separator = '/'
        store = zarr.DirectoryStore(store, dimension_separator = dimension_separator)

    if hasattr(store, 'path'):
        if isinstance(store.path, str):
            if use_synchronizer is None:
                synchronizer = None
            elif use_synchronizer == 'multiprocessing':
                if syncdir is None:
                    raise TypeError()
                synchronizer = zarr.ProcessSynchronizer(syncdir)
            elif use_synchronizer == 'multithreading':
                if syncdir is None:
                    raise TypeError()
                synchronizer = zarr.ThreadSynchronizer()
            else:
                synchronizer = None

    if 'overwrite' in kwargs.keys():
        overwrite = kwargs.get("overwrite")
    else:
        overwrite = False

    params = {
        'shape': zarray.shape if 'shape' not in kwargs.keys() else kwargs['shape'],
        'chunks': zarray.chunks if 'chunks' not in kwargs.keys() else kwargs['chunks'],
        'dtype': zarray.dtype if 'dtype' not in kwargs.keys() else kwargs['dtype'],
        'compressor': zarray.compressor if 'compressor' not in kwargs.keys() else kwargs['compressor'],
        'overwrite': overwrite
    }
    if hasattr(zarray, 'dimension_separator'):
        params['dimension_separator'] = zarray.dimension_separator

    res = zarr.create(**params,
                      store = store,
                      # overwrite = overwrite,
                      synchronizer = synchronizer
                      )
    return res



# synchdir = f"/home/oezdemir/PycharmProjects/dask_distributed/ome_zarr_pyramid/data3/sync"
# fpath = f"/home/oezdemir/PycharmProjects/dask_distributed/ome_zarr_pyramid/data3/filament.zarr"
# synchronizer = zarr.ProcessSynchronizer(synchdir)
# gr = zarr.open_group(fpath, synchronizer = synchronizer)
# gr.synchronizer.path



