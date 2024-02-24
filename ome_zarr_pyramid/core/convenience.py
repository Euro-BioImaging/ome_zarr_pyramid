import os, json
import warnings

import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import dask
import zarr, json, shutil, os, copy
from dask import array as da
from dask_image import ndmorph, ndinterp, ndfilters
import dask
import s3fs
from skimage.transform import resize as skresize
from skimage import measure
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

def asstr(s):
    if isinstance(s, str):
        return s
    elif isinstance(s, int):
        return str(s)
    else:
        raise TypeError(f"Input must be either of types {str, int}")

def asdask(data, chunks = 'auto'):
    assert isinstance(data, (da.Array, zarr.Array, np.ndarray)), f'data must be of type: {da.Array, zarr.Array, np.ndarray}'
    if isinstance(data, zarr.Array):
        return da.from_zarr(data)
    elif isinstance(data, np.ndarray):
        return da.from_array(data, chunks = chunks)
    return data

def parse_as_list(path_or_paths: Union[Iterable, str, int, float]
                   ):
    if isinstance(path_or_paths, (str, int, float)):
        inputs = [path_or_paths]
    else:
        inputs = path_or_paths
    return inputs

def includes(group1: Union[Iterable, str, int, float],
             group2: Union[Iterable, str, int, float]
             ):
    """Convenience function that checks if group1 includes group2 completely."""
    gr1 = parse_as_list(group1)
    gr2 = parse_as_list(group2)
    return all([item in gr1 for item in gr2])

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

def rescale(zarray,   ### Input is a single zarr array. Maybe divide this into two functions or methods, 1) get scales, 2) rescale
            zscale,
            axis_order,
            resolutions = 7,
            planewise = True,
            scale_factor = 2
            ) -> Iterable[da.Array]:
    """ Rescale an array with given resolutions and scale factor """
    print(f'input array type: {type(zarray)}')
    arr = asdask(zarray)
    zscale = np.array(zscale).astype(float)
    scale_fact = np.ones_like(zscale)
    for ax in axis_order:
        idx = axis_order.index(ax)
        if planewise:
            if ax in 'yx':
                scale_fact[idx] = scale_factor
        else:
            scale_fact[idx] = scale_factor
    div = np.round(np.vstack([scale_fact ** item for item in np.arange(resolutions)]), 3)
    shape = np.array(zarray.shape)
    new_shapes = shape // div
    new_shapes[new_shapes < 1] = 1
    #### until here make one function
    ret = {}
    for i, new_shape in enumerate(new_shapes):
        matrix = np.zeros((arr.ndim + 1, arr.ndim + 1))
        matrix[arr.ndim, arr.ndim] = 1
        scl = [None] * len(axis_order)
        for j in axis_order:
            idx = axis_order.index(j)
            matrix[idx, idx] = shape[idx] / new_shape[idx]
            scl[idx] = (shape[idx] / new_shape[idx]) * zscale[idx]
        output_shape = new_shape
        image_transformed = ndinterp.affine_transform(
                            arr,
                            matrix = matrix,
                            output_shape = output_shape,
                            output_chunks = arr.chunksize
                            )
        ret[str(i)] = (image_transformed, scl)
    return ret

def locate_labels(label_img,
                  ) -> dict:
    df = ndmeasure.find_objects(label_img).compute()
    dlbl = df.T.to_dict()
    return dlbl

def convert_np_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def is_valid_json(my_dict):
    try:
        json.dumps(my_dict)
        return True
    except:
        warnings.warn(f"Object is not a valid json!")
        return False

# import json
def turn2json(my_dict):
    stringified = json.dumps(my_dict, default = convert_np_types)
    return json.loads(stringified)

def get_display(label_image,
                colormap = 'viridis',
                lim = 255,
                chunks = 'auto'
                ):
    if isinstance(label_image, np.ndarray):
        label_image = da.from_array(label_image, chunks)
    lbld = label_image
    uqs, sizes = da.compute(da.unique(lbld, return_counts = True))[0]
    num = len(uqs)
    allcmaps = plt.colormaps()
    assert colormap in allcmaps, "colormap must be one of the items in 'cmaps'. Run 'plt.colormaps()' for the full list."
    cmap = mpl.colormaps.get_cmap(colormap)
    idx = np.linspace(0, lim, num).astype(int)
    if hasattr(cmap, 'colors'):
        colors = (np.array(cmap.colors) * lim * 1.)[idx]
    else:
        colors = np.array([cmap(i) for i in (idx / lim)]) * lim * 1.

    uqs = uqs[np.argsort(sizes)]
    uqs = np.array(([uqs[-1].tolist()] + uqs[:-1].tolist()))
    display_metadata = {
                            "colors": [
                                {"label-value": i0, "rgba": item.tolist() }
                                for i0, item in zip(uqs, colors)
                            ]
                        }
    return display_metadata

# h = get_display(lbld)

# lpyr = LabelPyramid()
# lpyr.add_imglabels(h['colors'])

def get_properties(label_image,
                   image = None,
                   properties = ['label', 'area', 'area_convex', 'intensity_max', 'intensity_mean', 'solidity']
                   ):
    if not isinstance(label_image, np.ndarray):
        larr = np.array(label_image)
    else:
        larr = label_image
    if image is not None:
        if not isinstance(image, np.ndarray):
            arr = np.array(image)
        else:
            arr = image

    props = measure.regionprops_table(larr, arr,
                                      properties = properties)

    return props
