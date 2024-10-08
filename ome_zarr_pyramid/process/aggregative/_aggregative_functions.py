import numpy as np, zarr, copy, inspect
from scipy import ndimage as ndi
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
# from ome_zarr_pyramid.process.core._blockwise_general import create_like
from ome_zarr_pyramid.creation.pyramid_creation import PyramidCreator
from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.utils import assignment_utils as asutils
from pathlib import Path


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
                  ): raise NotImplementedError()

def sum_zarrs(arraylist: List[zarr.Array],
              axis = 0,
              ): raise NotImplementedError()

# TODO: projective max, min, sum, etc. for collections of zarr arrays.


### BLOCK FUNCTIONS

def _compute_final_shape_from_dictofzarrs(data, axis_order):
    def recursive_shape(data):
        if isinstance(data, dict):
            combined_shape = None
            for axis, subdata in data.items():
                axis_index = axis_order.index(axis)
                sub_shape = None
                for item in subdata:
                    item_shape = recursive_shape(item)
                    if sub_shape is None:
                        sub_shape = item_shape
                    else:
                        for i in range(len(sub_shape)):
                            if i == axis_index:
                                sub_shape[i] += item_shape[i]
                            else:
                                if sub_shape[i] != item_shape[i]:
                                    raise ValueError(f"Non-concatenation dimensions must match. "
                                                     f"Dimension {i} mismatch: {sub_shape[i]} vs {item_shape[i]}.")
                if combined_shape is None:
                    combined_shape = sub_shape
                else:
                    for i in range(len(combined_shape)):
                        if i == axis_index:
                            combined_shape[i] += sub_shape[i]
                        else:
                            if combined_shape[i] != sub_shape[i]:
                                raise ValueError(f"Non-concatenation dimensions must match. "
                                                 f"Dimension {i} mismatch: {combined_shape[i]} vs {sub_shape[i]}.")
            return combined_shape
        elif isinstance(data, list):
            list_shape = None
            for item in data:
                item_shape = recursive_shape(item)
                if list_shape is None:
                    list_shape = item_shape
                else:
                    for i in range(len(list_shape)):
                        if list_shape[i] != item_shape[i]:
                            raise ValueError(f"All list items must have the same shape. "
                                             f"Dimension {i} mismatch: {list_shape[i]} vs {item_shape[i]}.")
            return list_shape
        else:
            return list(data.shape)
    return recursive_shape(data)

def __find_sub_shape(data):
    if isinstance(data, dict):
        for subdata in data.values():
            for item in subdata:
                return __find_sub_shape(item)
    elif isinstance(data, list):
        return __find_sub_shape(data[0])
    else:
        return data.shape

def _find_block_slices(data, current_shape, axis_order, current_slices=None):
    if current_slices is None:
        current_slices = {axis: slice(0, current_shape[i]) for i, axis in enumerate(axis_order)}
    flat_slices = []
    if isinstance(data, dict):
        for axis, subdata in data.items():
            axis_index = axis_order.index(axis)
            start = current_slices[axis].start
            for item in subdata:
                sub_shape = __find_sub_shape(item)
                end = start + sub_shape[axis_index]
                new_slices = current_slices.copy()
                new_slices[axis] = slice(start, end)
                flat_slices.extend(_find_block_slices(item, sub_shape, axis_order, new_slices))
                start = end
    elif isinstance(data, list):
        for item in data:
            flat_slices.extend(_find_block_slices(item, current_shape, axis_order, current_slices))
    else:
        flat_slices.append((data, current_slices))
    return flat_slices

def _calculate_block_slices(data: Union[Pyramid], axis_order = 'tczyx'):
    assert isinstance(data, dict), f"Input data must be a dict."
    # Calculate the final shape
    final_shape = _compute_final_shape_from_dictofzarrs(arrs, axis_order=axis_order)
    # Find the slices for each Pyramid's reference array
    block_slices = _find_block_slices(arrs, final_shape, axis_order)
    refpyr, _ = block_slices[0]
    if refpyr.axis_order != axis_order:
        return _calculate_block_slices(data, refpyr.axis_order)
    else:
        return final_shape, block_slices

#### BLOCK FUNCTIONS ####
#########################