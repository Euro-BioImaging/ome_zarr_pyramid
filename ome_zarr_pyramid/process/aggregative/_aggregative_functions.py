import numpy as np, zarr, copy
from scipy import ndimage as ndi
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from ome_zarr_pyramid.process.core._blockwise_general import create_like


def _get_shapes_from_arraylist(arraylist):
    return np.array([arr.shape for arr in arraylist])

def _get_uniform_axes(shapes: (tuple, list)):
    collection = np.array(shapes)
    uqs = np.array([np.unique(collection[:, i]).size for i in range(collection.shape[1])])
    axes = uqs == 1
    return axes

def _get_final_shape_for_concatenation(shapes: (tuple, list),
                    axis: int = 0
                    ):
    u_axes = _get_uniform_axes(shapes)
    other = np.delete(u_axes, axis)
    if not np.all(other):
        raise ValueError(f"Axes other than the concatenation axis '{axis}' must match.")
    final_shape = shapes[0].copy()
    final_shape[axis] = np.sum(shapes[:, axis])
    return final_shape

def _get_final_shape_for_stacking(shapes: (tuple, list),
                    axis: int = 0
                    ):
    u_axes = _get_uniform_axes(shapes)
    other = np.delete(u_axes, axis)
    if not np.all(other):
        raise ValueError(f"Axes other than the concatenation axis '{axis}' must match.")
    final_shape = shapes[0].copy()
    final_shape[axis] = np.sum(shapes[:, axis])
    return final_shape

def concatenate_zarrs(arraylist: List[zarr.Array],
                      axis = 0,
                      ):
    shapes = _get_shapes_from_arraylist(arraylist)
    shape = _get_final_shape_for_concatenation(shapes, axis = axis)
    concatenated = create_like(arraylist[0],
                               shape = shape,
                               )
    slcset = []
    start = 0
    for shape, arr in zip(shapes, arraylist):
        stop = start + shape[axis]
        slcs = [slice(None, None)] * len(shape)
        slcs[axis] = slice(start, stop)
        slcset.append(tuple(slcs))
        start = stop
    for idx, slcs in enumerate(slcset):
        concatenated[slcs][:] = arraylist[idx][:]
    return concatenated

def stack_zarrs(arraylist: List[zarr.Array],
                  axis = 0,
                  ): pass

def sum_zarrs(arraylist: List[zarr.Array],
              axis = 0,
              ): pass

# TODO: similarly, projective max, min, sum, etc. for collections of zarr arrays.
