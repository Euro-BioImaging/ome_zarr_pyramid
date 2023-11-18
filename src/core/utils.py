import os, json
import numpy as np, pandas as pd
import dask
import zarr, json, shutil, os, copy
from dask import array as da
import dask_image.ndmeasure as ndmeasure
import dask as da
import s3fs
from skimage.transform import resize as skresize
import numcodecs
from rechunker import rechunk
import shutil, tempfile

from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    Iterable,
    List,
    Optional
)

def index_nth_dimension(array,
                        dimensions = 2, # a scalar or iterable
                        intervals = None # a scalar, an iterable of scalars, a list of tuple or None
                        ):
    if isinstance(array, zarr.Array):
        array = da.from_zarr(array)
    allinds = np.arange(array.ndim).astype(int)
    if np.isscalar(dimensions):
        dimensions = [dimensions]
    if intervals is None or np.isscalar(intervals):
        intervals = np.repeat(intervals, len(dimensions))
    assert len(intervals) == len(dimensions) ### KALDIM
    interval_dict = {item: interval for item, interval in zip(dimensions, intervals)}
    shape = array.shape
    slcs = []
    for idx, dimlen in zip(allinds, shape):
        if idx not in dimensions:
            slc = slice(dimlen)
        else:
            try:
                slc = slice(interval_dict[idx][0], interval_dict[idx][1])
            except:
                slc = interval_dict[idx]
        slcs.append(slc)
    slcs = tuple(slcs)
    indexed = array[slcs]
    return indexed

def transpose_dict(dictionary):
    keys, values = [], []
    for key, value in dictionary.items():
        keys.append(key)
        values.append(value)
    return keys, values

def argsorter(s):
    return sorted(range(len(s)), key = lambda k: s[k])

def is_generic_collection(group):
    res = False
    basepath = group.store.path
    basename = os.path.basename(basepath)
    paths = list(group.keys())
    attrs = dict(group.attrs)
    attrkeys, attrvalues = transpose_dict(attrs)
    if basename in attrkeys and (len(paths) > 0):
        if len(attrs[basename]) == len(paths):
            res = True
            for item0, item1 in zip(attrs[basename], paths):
                if item0 != item1:
                    res = False
    return res

def get_collection_paths(directory,
                         return_all = False
                         ):
    gr = zarr.group(directory)
    groupkeys = list(gr.group_keys())
    arraykeys = list(gr.array_keys())
    grouppaths = [os.path.join(directory, item) for item in groupkeys]
    arraypaths = [os.path.join(directory, item) for item in arraykeys]
    collection_paths = []
    multiscales_paths = []
    while len(grouppaths) > 0:
        if is_generic_collection(gr) or 'bioformats2raw.layout' in gr.attrs:
            collection_paths.append(directory)
        if 'multiscales' in list(gr.attrs.keys()):
            multiscales_paths.append(directory)
        directory = grouppaths[0]
        grouppaths.pop(0)
        gr = zarr.group(directory)
        groupkeys = list(gr.group_keys())
        arraykeys = list(gr.array_keys())
        grouppaths += [os.path.join(directory, item) for item in groupkeys]
        arraypaths += [os.path.join(directory, item) for item in arraykeys]
    if is_generic_collection(gr) or 'bioformats2raw.layout' in gr.attrs:
        collection_paths.append(directory)
    if 'multiscales' in list(gr.attrs.keys()):
        multiscales_paths.append(directory)
    out = [item for item in collection_paths]
    for mpath in multiscales_paths:
        s = os.path.dirname(mpath)
        if s in collection_paths:
            pass
        else:
            if mpath not in out:
                out.append(mpath)
    if return_all:
        return out, multiscales_paths, arraypaths
    return out

@dask.delayed
def delayed_resize(img, output_shape):
    return skresize(img, output_shape)

def rescale(zarray,   ### Input is a single zarr array. Maybe divide this into two functions or methods, 1) get scales, 2) rescale
            resolutions = 7,
            planewise = True,
            compute = False
            ):
    ### get_scales function will calculate scales from integer or tuple
    divisors = (2 ** np.arange(resolutions)).astype(int).reshape(1, -1)
    shape = np.array(zarray.shape)
    if planewise:
        ones = np.repeat(np.ones(resolutions).reshape(1, -1), zarray.ndim - 2, axis = 0)
        plane = np.repeat(divisors, 2, axis = 0)
        div = np.vstack((ones, plane)).T
    else:
        div = np.vstack([divisors] * zarray.ndim).T
    scales = shape // div
    scales[scales < 1] = 1
    scales = da.array(scales)
    #### until here make one function
    rescaled = [delayed_resize(zarray, scale) for scale in scales]
    if compute:
        res, *(_) = dask.compute(rescaled)
    else:
        darrays = []
        for level in range(len(rescaled)):
            nres = rescaled[level]
            nresdask = da.from_delayed(nres, shape = scales[level], dtype = zarray.dtype)
            nresdask = nresdask.rechunk(zarray.chunks)
            darrays.append(nresdask)
        res = darrays
    return res


def locate_labels(label_img,
                  ) -> dict:
    df = ndmeasure.find_objects(label_img).compute()
    dlbl = df.T.to_dict()
    return dlbl