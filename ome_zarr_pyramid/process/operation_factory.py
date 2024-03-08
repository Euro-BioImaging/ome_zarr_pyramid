# from ome_zarr_pyramid.core.Hierarchy import OMEZarr, MultiScales
import warnings
from multiprocessing import Pool, Manager

from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.core import config, convenience as cnv
from ome_zarr_pyramid.process.parameter_control import OperationParams, BaseParams
from ome_zarr_pyramid.process import process_utilities as putils, custom_operations
# from ome_zarr_pyramid.process.process_core import BaseProtocol

import itertools
import zarr
import pandas as pd
from pathlib import Path
import numpy as np
import dask.array as da
import dask.bag as db
import os, copy
import numcodecs
import dask_image.ndfilters as ndfilters
import dask_image.ndmorph
import dask_image.dispatch
import dask_image.ndmeasure as ndmeasure
import dask
from typing import (Union, Tuple, Dict, Any, Iterable, List, Optional)
from functools import wraps

################################### MODULE UTILITIES ######################################
def get_common_set(lol):
    flat = []
    res = []
    for l in lol: flat += l
    for item in flat:
        if all(item in l for l in lol):
            if item not in res:
                res.append(item)
    return res

def get_common_indices(axes):
    common_axes = get_common_set(axes)
    common_axes_indices = []
    for ax in axes:
        ids = []
        for c in common_axes:
            ids.append(ax.index(c))
        common_axes_indices.append(ids)
    return common_axes_indices

class AxisHandler:
    def __init__(self,
                 axes: list # This is a list of list
                 ):
        self.axes = axes
    def get_common_axes(self):
        return get_common_set(self.axes)
    def get_incomplete_axes(self):
        common = self.get_common_axes()
        missing = []
        indices_of_missing = []
        for ax in self.axes:
            miss = [i for i in ax if i not in common]
            missing.append(miss)
            ids = [ax.index(i) for i in ax if i not in common]
            indices_of_missing.append(ids)
        return missing, indices_of_missing
    @property
    def common_axes(self):
        return self.get_common_axes()
        # axes = oper5.input_dataset.axes
        # scales = oper5.input_dataset.scales
        # units = oper5.input_dataset.units

# axes = oper4.input_dataset.axes
# axh = AxisHandler(axes)
# axh.get_incomplete_axes()

# def method_iterator(method):
#     def wrapper(self, *args, **kwargs):
#         for pyramid in self.pyramids:
#             getattr(pyramid, method)(*args, **kwargs)
#     return wrapper

def insert_indices(iterable1, iterable2, indices):
    endlen = (len(iterable1) + len(iterable2))
    end_indices = [None] * endlen
    other_indices = [i for i in range(endlen) if i not in indices]
    for i, j in zip(other_indices, iterable1):
        end_indices[i] = j
    for i, j in zip(indices, iterable2):
        end_indices[i] = j
    return end_indices


def apply_method(args):
    pyramid, method, method_args, method_kwargs = args
    getattr(pyramid, method)(*method_args, **method_kwargs)
    return pyramid

def concurrent_pyramid_methods(method):
    def decorator(cls):
        def wrapper(self, *args, **kwargs):
            with Pool() as pool:
                results = []
                for pyramid in self.pyramids:
                    results.append(pool.apply_async(apply_method, args=((pyramid, method, args, kwargs),)))
                pool.close()
                pool.join()
                self.pyramids = [result.get() for result in results]
        setattr(cls, method, wrapper)
        return cls
    return decorator

def pyramid_methods(method):
    def decorator(cls):
        def wrapper(self, *args, **kwargs):
            for pyramid in self.pyramids:
                getattr(pyramid, method)(*args, **kwargs)
        setattr(cls, method, wrapper)
        return cls
    return decorator

@pyramid_methods('rechunk')
@pyramid_methods('recompress')
@pyramid_methods('rescale')
@pyramid_methods('shrink')
@pyramid_methods('del_layer')
@pyramid_methods('astype')
@pyramid_methods('asflat')
@pyramid_methods('asnested')
@pyramid_methods('expand_dims')
@pyramid_methods('drop_singlet_axes')
@pyramid_methods('retag')
class PyramidCollection: # Change to PyramidIO
    # TODO: provide a sorting method, based on the Pyramid tag, which should be automatically assigned a unique value.
    def __init__(self,
                 input=None,
                 stringent=False
                 ):
        self.pyramids = []
        if input in [None, []]:
            pass
        else:
            input = [pyr.copy() for pyr in input]
            for pyr in input:
                self.add_pyramid(pyr,
                                 stringent = stringent
                                 )

    @property
    def size(self):
        return len(self.pyramids)

    @property
    def refpath(self):
        try:
            return self.resolution_paths[0]
        except:
            return ValueError(f"No reference path can be specified. Perhaps, paths are not unique?")

    def drop_pyramid(self,
                     idx,
                     ):
        self.pyramids.pop(idx)

    def add_pyramid(self,
                    pyr,
                    stringent=False
                    ):
        # pyr.retag(unique_name_or_number)
        self.pyramids.append(pyr)
        assert self._validate_pyramidal_collection(self.pyramids) # Change to _validate_and_read or have separate methods for reading and validation
        if stringent:
            assert self._paths_are_unique(), f"The resolution paths are not consistent for all Pyramids."
        if not self._paths_are_unique():
            self.equalise_resolutions()
        if len(self.resolution_paths) == 0:
            raise ValueError(f"The pyramids cannot be processed together. No matching resolution paths")

    def _paths_are_unique(self):  # make sure each pyramid has the same number of resolution layers
        return all(self.pyramids[0].resolution_paths == item.resolution_paths for item in self.pyramids)

    @property
    def _nlayers(self):
        return [pyr.nlayers for pyr in self.pyramids]

    @property
    def _resolution_paths(self):
        return [pyr.resolution_paths for pyr in self.pyramids]

    @property
    def resolution_paths(self):
        if self._paths_are_unique():
            return self.pyramids[0].resolution_paths
        else:
            return get_common_set(self._resolution_paths)

    def equalise_resolutions(self):
        # print(f"Resolutions are being equalised.")
        paths = self.resolution_paths
        self.shrink(paths)

    # def rescale(self,
    #             scale_factor: Union[list, tuple, int, float],
    #             resolutions: int = None,
    #             planewise: bool = True
    #             ):
    #     for pyr in self.pyramids:
    #         pyr.rescale(scale_factor = scale_factor,
    #                     resolutions = resolutions,
    #                     planewise = planewise
    #                     )

    def _validate_pyramidal_collection(self, input): # Change to _validate_and_read
        if input is None:
            input = []
        if isinstance(input, Pyramid):
            input = [input]
        else:
            assert isinstance(input, (tuple, list)), f"Input must be tuple or list."
        for pyr in input:
            assert isinstance(pyr, Pyramid), f"Each input item must be an instance of Pyramid class."
        self.pyramids = input
        return True

    @property
    def arrays(self):
        l = {key: [] for key in self.resolution_paths}
        for key in self.resolution_paths:
            arrays = [pyr[key] for pyr in self.pyramids]
            l[key] = arrays
        return l

    @property
    def axes(self):
        axes = []
        for pyr in self.pyramids:
            axes.append(pyr.axis_order)
        return axes

    @property
    def shapes(self):
        shapes = []
        for pyr in self.pyramids:
            shapes.append(pyr.shape)
        return shapes

    @property
    def has_uniform_axes(
            self):  ### THESE WILL BE METHODS INSTEAD OF PROPERTIES AND WILL HAVE AN AXIS PARAMETER TO CONTROL WHICH AXES TO COMPARE.
        return all([self.axes[0] == item for item in self.axes])

    def get_uniform_dimensions(self): # TODO
        pass

    def has_uniform_axes_except(
                                self,
                                axes
                                ):
        if self.has_uniform_axes:
            return True
        if not hasattr(axes, '__len__'):
            axes = [axes]
        template = [c for c in list(self.axes[0]) if c not in axes]
        for item in self.axes:
            item_temp = [c for c in list(item) if c not in axes]
            if template != item_temp:
                return False
        return True

    @property
    def units(self):
        units = []
        for pyr in self.pyramids:
            units.append(pyr.unit_list)
        return units

    @property
    def has_uniform_units(self):
        return all([self.units[0] == item for item in self.units])

    @property
    def scales(self):
        l = {key: [] for key in self.resolution_paths}
        for key in self.resolution_paths:
            scales = [pyr.scales[key] for pyr in self.pyramids]
            l[key] = scales
        return l

    @property
    def has_uniform_scales(self):
        checklist = []
        for key, scale in self.scales.items():
            checklist.append(all([scale[0] == item for item in scale]))
        return all(checklist)

    @property
    def has_uniform_all(self):
        return self.has_uniform_axes & self.has_uniform_scales & self.has_uniform_units

    @property
    def chunks(self):
        chunks = []
        for pyr in self.pyramids:
            chunks.append(pyr.chunks)
        return chunks

class _ImageOperations:
    def __init__(self,
                 operation_module_name: str = 'dask',
                 operation_name: str = None,
                 **kwargs
                 ):
        self._custom_operation_categories = {
            "add": "static_array_to_array",
            "around": "static_array_to_array",
            "power": "static_array_to_array",
            "subset": "static_array_to_array",
            "concatenate": "static_collection_to_array",
            "block": "static_collection_to_array",
            "max": "reductive_array_to_array",
            "min": "reductive_array_to_array",
            "mean": "reductive_array_to_array",
            "median": "reductive_array_to_array",
            "sum": "reductive_array_to_array",
            "expand_dims": "expansive_array_to_array",
            "stack": "expansive_collection_to_array",
            "split": "reductive_array_to_collection",
            "argwhere": "reductive_array_to_coords" # TODO: these should go to a separate module.
                                                    # TODO: Keep this module for "pyramid(s) in, pyramid(s) out" operations
        }
        self._operation_source_map = {
            'dask': custom_operations,
            'dask_ndfilters': ndfilters
        }
        assert operation_module_name in self._operation_source_map.keys(), f"Module name is not recognised. Currently it must be one of '{self._operation_source_map.keys()}'"
        self._operation_params = OperationParams(operation_module_name, operation_name, **kwargs)


    def print_params(self):
        print(self.operation_params)

    @property
    def _operation_collection(self):
        return self._operation_params._meta

    @property
    def operation_module_name(self):
        return self._operation_params._module_name

    @property
    def operation_name(self):
        return self._operation_params._filter_name

    @property
    def operation_category(self):
        if self.operation_module_name == 'dask':
            return self._custom_operation_categories[self.operation_name]
        elif self.operation_module_name == 'dask_ndfilters':
            return 'static_array_to_array'

    @property
    def operation_params(self):
        p = copy.deepcopy(self._operation_params._params)
        for item in ['image', 'arr', 'array', 'layer', 'images', 'imlist', 'arrays', 'layers']:
            if item in self._operation_params._params:
                p.pop(item)
        return p

    @property
    def _operation_source(self):
        return self._operation_source_map[self.operation_module_name]

    def operation(self,
                  *args,
                  **kwargs
                  ):
        if 'reductive' in self.operation_category and 'axis' not in self.operation_params.keys():
            raise ValueError(f"All reductive operations including '{self.operation_name}' must have the parameter 'axis'.")
        if 'expansive' in self.operation_category and 'newaxis' not in self.operation_params.keys():
            raise ValueError(f"All expansive operations including '{self.operation_name}' must have the parameter 'newaxis'.")
        operation = getattr(self._operation_source, self.operation_name)  # operation can be a class or function
        return operation(*args, **kwargs)


class Operation(_ImageOperations):
    def __init__(self,
                 operation_name,
                 operation_module_name = 'dask'
                 ):
        # self.operation_params = OperationParams(operation_name)
        _ImageOperations.__init__(self, operation_module_name = operation_module_name, operation_name=operation_name)
        self._base_params = BaseParams(resolutions = None, drop_singlet_axes = None, output_name = None)
        self.input_dataset = None
        self.notices = []

    @property
    def base_params(self):
        return self._base_params._params

    @property
    def _data_and_operation_are_compatible(self):
        if self.input_dataset is None:
            return ValueError(f"PyramidCollection does not exist.")
        if "array_to" in self.operation_category:
            self.message = f"'{self.operation_name}' is a '{self.operation_category}' operation and thus requires a single input image."
            return self.input_dataset.size == 1  # The array_to_array and array_to_collection operations require single input images.
        elif "collection_to" in self.operation_category:
            self.message = f"'{self.operation_name}' is a '{self.operation_category}' operation and thus requires multiple input images."
            return self.input_dataset.size > 1  # The collection_to_array operations require multiple input images.
        else:
            warnings.warn(f"Not yet implemented.")
        if "to_collection" in self.operation_category:
            if self.base_params['output_name'] is None:
                return True
            else:
                self.message = f"Currently the output name has to be automatically derived for multi-output operations. Ignoring the given output name '{self.base_params['output_name']}"
                return False

    def set_input(self,
                  input: Iterable[Pyramid]
                  ):
        if isinstance(input, Pyramid):
            input = [input]
        self.input_dataset = PyramidCollection(input)
        assert self._data_and_operation_are_compatible, self.message
        if 'axis' in self.operation_params.keys():  ### This means there will be an operation involving array axes
            if "collection_to_array" in self.operation_category:
                assert self.input_dataset.has_uniform_all, f"All pyramids in the collection must have the same axes, units and scales."

    def reconstruct_pyramid_collection(self,
                                       result_arrays,
                                       layer_meta,
                                       # output_dataset_size
                                       ):
        key0 = list(result_arrays.keys())[0]
        output_dataset_size = len(result_arrays[key0])
        result_pyrs = [Pyramid() for _ in range(output_dataset_size)]
        for pth in result_arrays.keys():
            res_array_list = result_arrays[pth]
            meta = layer_meta[pth]
            for idx, array in enumerate(res_array_list):
                # print(pth, idx)
                pyr = result_pyrs[idx]
                pyr.add_layer(array,
                              pth,
                              scale=meta['scale'],
                              zarr_meta={'dtype': array.dtype,
                                         'chunks': meta['chunks'],
                                         'shape': array.shape,
                                         'compressor': meta['compressor'],
                                         'dimension_separator': meta['dimension_separator']
                                         },
                              axis_order=meta['axis_order'],
                              unitlist=meta['unit_list']
                              )
                # result_pyrs[idx] = pyr
                # print(pyr.resolution_paths)
        output_dataset = PyramidCollection(result_pyrs)
        return output_dataset

    def get_axis_as_int(self):
        if 'axis' not in self.operation_params.keys():
            return ValueError(f"The 'axis' is not a valid parameter for the operation '{self.operation_name}'")
        if self.input_dataset is None:
            return ValueError(f"There is no input dataset.")
        if not self.input_dataset.has_uniform_all:
            return ValueError(
                f"The '{self.operation_name}' operation requires matching axes, units and scales for all pyramids.")
        if isinstance(self.operation_params['axis'], str):
            opparams = copy.deepcopy(self.operation_params)
            axis = tuple([self.input_dataset.axes[0].index(idx) for idx in opparams['axis']])
            if len(axis) == 1:
                axis = axis[0]
            return axis
        elif isinstance(self.operation_params['axis'], (int, tuple, list)):
            return self.operation_params['axis']
        else:
            return ValueError(f"The 'axis' parameter cannot be of types other than int, str, tuple or list.")
    @property
    def axis_as_int(self):
        return self.get_axis_as_int()

    def get_interval_as_slice(self): ### downscale interval for each resolution.
        if isinstance(self.axis_as_int, ValueError):
            return f"Axis parameter is not defined."
        if 'interval' not in self.operation_params.keys():
            return ValueError(f"The 'interval' is not a valid parameter for the operation '{self.operation_name}'.")
        if self.input_dataset.size != 1:
            return ValueError(f"The input pyramid number must be exactly 1 for the opertaion '{self.operation_name}'.")
        indices = self.axis_as_int
        intervals = copy.deepcopy(self.operation_params['interval'])
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(intervals[0], int): # this means interval is provided as a flat Iterable, thus must wrap it.
            intervals = [self.operation_params['interval']]
        assert len(indices) == len(intervals), f"The index length must match the provided number of intervals."
        axes = self.input_dataset.axes[0]
        shape = self.input_dataset.shapes[0]
        intervals_ = [None] * len(axes)
        for idx, i in enumerate(indices):
            if intervals[idx][1] - intervals[idx][0] > shape[i]:
                intervals_[i] = slice(None, None)
            else:
                intervals_[i] = slice(*intervals[idx])
        for i, _ in enumerate(axes):
            if i not in indices:
                intervals_[i] = slice(None, None)
        return tuple(intervals_)
    @property
    def interval_as_slice(self):
        return self.get_interval_as_slice()

    def update_param(self,
                     param_name: str,
                     param_value: any
                     ):
        if param_name in ['resolutions', 'drop_singlet_axes', 'output_name']:
            self._base_params._update_param(param_name, param_value)
        elif param_name in self.operation_params.keys():
            self._operation_params._update_param(param_name, param_value)

    def _parse_params(self, **kwargs):  # This is bound to the apply_operation function
        # self.extra_params = {}
        if self.input_dataset is None:
            raise ValueError(f"The input PyramidCollection object does not exist.")
        if len(kwargs.keys()) > 0:
            for key, value in kwargs.items():
                self.update_param(key, value)

    def _static_array_to_array(self,
                               params: dict
                               ):
        if "interval" in self.operation_params.keys():
            assert "n_resolutions" in self.operation_params, f"The parameter 'n_resolutions' is missing"
            assert "scale_factor" in self.operation_params, f"The parameter 'scale_factor' is missing"
            assert "planewise" in self.operation_params, f"The parameter 'planewise' is missing"
            self.input_dataset.shrink([self.input_dataset.refpath])
            self.notices.append(f"For the operation '{self.operation_name}', The input dataset has been restricted to its reference resolution layer.")
        for key, array_list in self.input_dataset.arrays.items():
            # print(params)
            # print(array_list)
            res_array_list = [self.operation(array, **params) for array in array_list]
            assert len(res_array_list) == 1, f"'{self.operation_category}' cannot have multiple outputs"
            assert len(self.input_dataset.pyramids) == 1, f"'{self.operation_category}' cannot have multiple inputs"
            self.result_arrays[key] = res_array_list
            self.layer_meta[key] = self.input_dataset.pyramids[0].layer_meta[key]
        # output_dataset_size = len(self.input_dataset.pyramids)  # output_dataset_size is equal to input dataset size
        self.output_dataset = self.reconstruct_pyramid_collection(self.result_arrays, self.layer_meta)
        if self.output_dataset.chunks[0] != self.input_dataset.chunks[0]: # TODO: this needs to be added to the notices.
            self.output_dataset.rechunk(self.input_dataset.chunks[0])
            # warnings.warn(f"Chunking changed. Updated back to '{self.output_dataset.chunks[0]}'")
        # print(f"Output chunk size: '{self.output_dataset.chunks[0]}'")
        if "interval" in self.operation_params.keys(): # A rescaling postprocess is needed if the parameters contain interval.
            self.output_dataset.rescale(resolutions = params['n_resolutions'],
                                        scale_factor = params['scale_factor'],
                                        planewise = params['planewise']
                                        )
            self.layer_meta = self.output_dataset.pyramids[0].layer_meta # Layer meta must be updated according to the postprocess

    def _static_collection_to_array(self,
                                    params: dict
                                    ):
        if not self.input_dataset.has_uniform_all:
            raise ValueError(f"Input pyramids must exactly match in axis, units and scales.")
        # self._parse_params(**kwargs)
        for key, array_list in self.input_dataset.arrays.items():
            res_array_list = [self.operation(array_list, **params)]
            assert len(res_array_list) == 1, f"'{self.operation_category}' cannot have multiple outputs"
            assert len(self.input_dataset.pyramids) > 1, f"'{self.operation_category}' must have multiple inputs"
            self.result_arrays[key] = res_array_list
            self.layer_meta[key] = self.input_dataset.pyramids[0].layer_meta[key]
        # output_dataset_size = 1  # output_dataset_size is equal to 1 (map reduction)
        self.output_dataset = self.reconstruct_pyramid_collection(self.result_arrays, self.layer_meta,
                                                                  # output_dataset_size
                                                                  )

    def _reductive_array_to_array(self,
                                  params: dict
                                  ):
        for key, array_list in self.input_dataset.arrays.items():
            res_array_list = [self.operation(array, **params) for array in array_list]
            assert len(res_array_list) == 1, f"'{self.operation_category}' cannot have multiple outputs"
            assert len(self.input_dataset.pyramids) == 1, f"'{self.operation_category}' cannot have multiple inputs"
            self.result_arrays[key] = res_array_list
            array = self.result_arrays[key][0]
            layer_meta = self.input_dataset.pyramids[0].layer_meta[key]
            layer_meta['dtype'] = array.dtype
            if hasattr((self.operation_params['axis']), '__len__'):
                axes = self.operation_params['axis']
            else:
                axes = [self.operation_params['axis']]
            if self.operation_params['keepdims']:
                indices = [i for i, j in enumerate(layer_meta['axis_order'])]
            else:
                indices = [i for i, j in enumerate(layer_meta['axis_order']) if j not in axes]
            layer_meta['axis_order'] = ''.join([layer_meta['axis_order'][i] for i in indices])
            layer_meta['unit_list'] = [layer_meta['unit_list'][i] for i in indices]
            layer_meta['scale'] = [layer_meta['scale'][i] for i in indices]
            layer_meta['chunks'] = array.chunksize
            self.layer_meta[key] = layer_meta
            # output_dataset_size = len(self.input_dataset.pyramids)
        self.output_dataset = self.reconstruct_pyramid_collection(self.result_arrays, self.layer_meta,
                                                                  # output_dataset_size
                                                                  )

    def _reductive_array_to_collection(self,
                                       params: dict
                                       ):
        for key, array_list in self.input_dataset.arrays.items():
            res_array_list = [self.operation(array, **params) for array in array_list][0]
            # assert len(res_array_list) == 1, f"'{self.operation_category}' cannot have multiple outputs"
            assert len(self.input_dataset.pyramids) == 1, f"'{self.operation_category}' cannot have multiple inputs"
            self.result_arrays[key] = res_array_list
            array = self.result_arrays[key][0]
            layer_meta = self.input_dataset.pyramids[0].layer_meta[key]
            layer_meta['dtype'] = array.dtype
            layer_meta['chunks'] = array.chunksize
            self.layer_meta[key] = layer_meta
            # output_dataset_size = len(self.input_dataset.pyramids)
        self.output_dataset = self.reconstruct_pyramid_collection(self.result_arrays, self.layer_meta)

    def _expansive_array_to_array(self,
                                  params: dict
                                  ):
        for key, array_list in self.input_dataset.arrays.items():
            res_array_list = [self.operation(array, **params) for array in array_list]
            assert len(res_array_list) == 1, f"'{self.operation_category}' cannot have multiple outputs"
            assert len(self.input_dataset.pyramids) == 1, f"'{self.operation_category}' cannot have multiple inputs"
            self.result_arrays[key] = res_array_list
            array = self.result_arrays[key][0]
            layer_meta = self.input_dataset.pyramids[0].layer_meta[key]
            layer_meta['dtype'] = array.dtype
            newaxis = self.operation_params['newaxis']
            assert isinstance(newaxis, str), "The newaxis parameter must be of type str."
            newunit = self.operation_params['newunit']
            newscale = self.operation_params['newscale']
            position = self.operation_params['position']
            layer_meta['axis_order'] = ''.join(insert_indices(layer_meta['axis_order'], newaxis, position))
            layer_meta['unit_list'] = insert_indices(layer_meta['unit_list'], newunit, position)
            layer_meta['scale'] = insert_indices(layer_meta['scale'], newscale, position)
            layer_meta['chunks'] = array.chunksize
            self.layer_meta[key] = layer_meta
            self.output_dataset = self.reconstruct_pyramid_collection(self.result_arrays, self.layer_meta)

    def _expansive_collection_to_array(self,
                                  params: dict
                                  ):
        for key, array_list in self.input_dataset.arrays.items():
            res_array_list = [self.operation(array_list, **params)]
            assert len(res_array_list) == 1, f"'{self.operation_category}' cannot have multiple outputs"
            assert len(self.input_dataset.pyramids) > 1, f"'{self.operation_category}' must have multiple inputs"
            self.result_arrays[key] = res_array_list
            array = self.result_arrays[key][0]
            layer_meta = self.input_dataset.pyramids[0].layer_meta[key]
            layer_meta['dtype'] = array.dtype
            newaxis = self.operation_params['newaxis']
            assert isinstance(newaxis, str), "The newaxis parameter must be of type str."
            newunit = self.operation_params['newunit']
            newscale = self.operation_params['newscale']
            position = self.operation_params['position']
            layer_meta['axis_order'] = ''.join(insert_indices(layer_meta['axis_order'], newaxis, position))
            layer_meta['unit_list'] = insert_indices(layer_meta['unit_list'], newunit, position)
            layer_meta['scale'] = insert_indices(layer_meta['scale'], newscale, position)
            layer_meta['chunks'] = array.chunksize
            self.layer_meta[key] = layer_meta
            self.output_dataset = self.reconstruct_pyramid_collection(self.result_arrays, self.layer_meta)

    def apply_operation(self, **kwargs):
        self.result_arrays = {}
        self.layer_meta = {}
        self._parse_params(**kwargs)
        if 'axis' in self.operation_params.keys(): # "axis" parameter must be updated if it exists and it exists only with reductive operations
            params = {**self.operation_params, 'axis': self.axis_as_int}
        else:
            params = copy.deepcopy(self.operation_params)
        if 'interval' in self.operation_params.keys(): # "axis" parameter must be updated if it exists and it exists only with reductive operations
            params = {**params, 'interval': self.interval_as_slice}
        else:
            pass
        # print(params)
        if self.operation_category == 'static_array_to_array':
            self._static_array_to_array(params)
        elif self.operation_category == 'static_collection_to_array':
            self._static_collection_to_array(params)
        elif self.operation_category == 'reductive_array_to_array': # Assumes that the axis parameter exists.
            self._reductive_array_to_array(params)
        elif self.operation_category == 'reductive_array_to_collection':
            self._reductive_array_to_collection(params)
        elif self.operation_category == 'expansive_array_to_array': # Assumes that the axis and index parameters exist.
            self._expansive_array_to_array(params)
        elif self.operation_category == 'expansive_collection_to_array': # Assumes that the axis and index parameters exist.
            self._expansive_collection_to_array(params)

        # postprocessing
        if self.base_params['resolutions'] is not None:
            self.output_dataset.rescale(scale_factor = 2,
                                        resolutions = self.base_params['resolutions'],
                                        planewise = True
                                        )
        if self.base_params['drop_singlet_axes'] is not None:
            self.output_dataset.drop_singlet_axes()
        if self.base_params['output_name'] is not None:
            if 'to_collection' in self.operation_category:
                warnings.warn(f"Currently the output name has to be automatically derived for multi-output operations. Ignoring the given output name '{self.base_params['output_name']}")
            self.output_dataset.retag(self.base_params['output_name'])


def ndfilter_wrapper(filter_name):
    def decorator(func):
        @wraps(func)
        def wrapper(pyr_or_path, **kwargs):
            # preprocessing
            if isinstance(pyr_or_path, str):
                oz = Pyramid()
                oz.from_zarr(pyr_or_path)
            else:
                assert isinstance(pyr_or_path, Pyramid)
                oz = pyr_or_path
            # run operation
            imfilt = Operation(filter_name, operation_module_name='dask_ndfilters')
            imfilt.set_input([oz])
            for key, value in kwargs.items():
                imfilt.update_param(key, value)
                if key not in imfilt.operation_params.keys():
                    if key not in imfilt.base_params.keys():
                        if key != "write_to":
                            raise ValueError(f"The parameter '{key}' is not defined for the '{filter_name}'.")
            imfilt.apply_operation()
            # postprocessing
            output = imfilt.output_dataset.pyramids[0]
            if "write_to" in kwargs:
                # print(output.array_meta)
                output.to_zarr(kwargs["write_to"])
            return output
        return wrapper
    return decorator

# cpath = f"/home/oezdemir/PycharmProjects/test_pyenv/data/filament.zarr"
# oz = Pyramid()
# oz.from_zarr(cpath)
#
# oz1 = oz.rescale(2, 5)
# oz2 = oz.rescale(2, 5)
# oz3 = oz.rescale(2, 5)
# oz1.del_layer('0')
# oz3.del_layer('4')
#
# pyrs = PyramidCollection([oz1, oz2, oz3])
# # pyrs.shrink(['0', '1', '4'])
# pyrs.rescale(scale_factor = 2, resolutions = 5)


# oz1.del_layer('0')
# oz3.del_layer('4')
#
# # oper = Operation('sum')
# # oper.axis_as_int
# # oper.set_input([oz1])
# # oper.apply_operation(axis = 'z', keepdims = True)
# #
# # # outputs = oper.output_dataset
# # # outputs.resolution_paths
# #
# # oper1 = Operation('concatenate')
# # oper1.set_input([oz1, oz2, oz3])
# # oper1.apply_operation(axis = 'z')
# #
# #
# # oper2 = Operation('add')
# # oper2.set_input([oz2])
# # oper2.apply_operation(scalar = 2)
# #
# # oper3 = Operation('expand_dims')
# # oper3.set_input([oz3])
# # oper3.apply_operation(newaxis = 'tc', newunit = ["sec", "Channel"], newscale = [20, 1], position = [0, 1])
# #
# # oper4 = Operation('stack')
# # oper4.set_input([oz1, oz2, oz3])
# # oper4.apply_operation(newaxis = 't', newunit = ["sec"], newscale = [20], position = [0])
# #
# # pyr1 = oper.output_dataset.pyramids[0]
# # pyr2 = oper.input_dataset.pyramids[0]
# #
# # oz.expand_dims('c', 0, new_scale = 1, new_unit = 'Frame')
# #
# # oper5 = Operation('concatenate')
# # oper5.set_input([pyr1, pyr2])
# # oper5.apply_operation(axis = 'z')
# #
# oper6 = Operation('subset')
# oper6.set_input([oz1])
#
#
# oper6._operation_params._update_param('axis', 'xz')
# oper6._operation_params._update_param('axis', (1, 2))
# oper6._operation_params._update_param('interval', ((0, 10), (0, 30)))
# # oper6._operation_params._update_param('n_resolutions', 3)
#
#
# oper6.interval_as_slice
# oper6.apply_operation()
#
# oper7 = Operation('split')
# oper7.set_input([oz1])
# oper7._operation_params._update_param('axis', 'zy')
# oper7._operation_params._update_param('n_sections', (10, 100))
# oper7.apply_operation()


# pyr = oper4.output_da
# taset.pyramids[0]
# pyr.parse_axes('tzyx', unit_list = ['sec', 'slice', 'space', 'space'], overwrite=True)

# outputs = oper.output_dataset
# outputs.resolution_paths


################################### MODULE CLASSES ########################################

#
# class ImageOperations(BaseProtocol, _ImageOperations):
#     def __init__(self,
#                  *args,
#                  operation_name: str = "projection",
#                  **kwargs
#                  ):
#         _ImageOperations.__init__(self, operation_name=operation_name, **kwargs)
#         BaseProtocol.__init__(self, *args, **kwargs)
#
#     @property
#     def base_params(self):
#         return self._base_params._params
#     def update_param(self,
#                      key,
#                      value
#                      ): ### TODO!!!
#         pass
#     def update_operation(self,
#                       operation_name,
#                       **kwargs
#                       ): ### TODO!!!
#         pass
#     def run(self, **kwargs): # as this runs, the base_params and operation_params are already known.
#         for key, value in kwargs.items():
#             if key not in self.operation_params.keys():
#                 print(f"The given param {key} is not defined for the function {self.operation_name}.")
#                 print(f"The defined parameters are: {self.operation_params}")
#             elif value != self.operation_params[key]:
#                 self._operation_params._update_param(key, value)
#
#         output = self.operation(self.input, operation_params = self.operation_params)
#         super().__setattr__("output", output)
