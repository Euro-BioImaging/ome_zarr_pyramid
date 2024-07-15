import warnings, zarr, os, copy, inspect
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.core._blockwise_general import LazyFunction
from ome_zarr_pyramid.process.morphology._blockwise_label import BlockwiseLabelRunner, LabelProfiler
from ome_zarr_pyramid.process.core.multiscale_apply_general import ApplyToPyramid, ApplyAndRescale, \
    _parse_subset_indices, _parse_args, _parse_axes_for_directional


def _parse_params_for_label(input, profiler, *args, **kwargs):
    overlap = tuple([3] * input.ndim)
    return args, kwargs, overlap


class ApplyLabelToPyramid(ApplyToPyramid):
    def __init__(self,
                 input: Pyramid,
                 *args,
                 min_block_size=None,
                 block_overlap_sizes=None,
                 subset_indices: dict = None,
                 ### zarr parameters
                 store: str = None,
                 compressor='auto',
                 dimension_separator=None,
                 dtype=None,
                 overwrite=False,
                 ###
                 func=None,
                 n_jobs=None,
                 monoresolution=False,
                 **kwargs
                 ):
        ApplyToPyramid.__init__(self,
                                input = input,
                                min_block_size = min_block_size,
                                block_overlap_sizes = block_overlap_sizes,
                                subset_indices = subset_indices,
                                store = store,
                                compressor = compressor,
                                dimension_separator = dimension_separator,
                                dtype = dtype,
                                overwrite = overwrite,
                                func = func,
                                n_jobs = n_jobs,
                                monoresolution = monoresolution,
                                *args, **kwargs
                                )

    def set_function(self, func):
        self.profiler = LabelProfiler(func)
        self.lazyfunc = LazyFunction(func, profiler=LabelProfiler)

    def set_runner(self, runner = None):
        if runner is None:
            runner = BlockwiseLabelRunner
        self.runner = runner

    def parse_params(self, *args, **kwargs):
        if self.profiler.is_label_filter:
            self.args, self.kwargs, overlap = _parse_params_for_label(self.input, self.profiler, *args, **kwargs)
            if self.block_overlap_sizes is None:
                self.block_overlap_sizes = overlap
            else:
                self.args = args
                self.kwargs = kwargs
        else:
            raise Exception(f"This is not a label function.")

    def _write_single_layer(self,
                            out,
                            blockwise,
                            pth,
                            meta,
                            **kwargs
                            ):
        out.add_layer(blockwise,
                      pth,
                      scale=self.input.get_scale(pth),
                      translation=self.input.get_translation(pth),
                      zarr_meta={'dtype': meta['dtype'],
                                 'chunks': meta['chunks'],
                                 'shape': blockwise.shape,
                                 'compressor': meta['compressor'],
                                 'dimension_separator': meta['dimension_separator']
                                 },
                      axis_order=self.input.axis_order,
                      unitlist=self.input.unit_list
                      )

        blockwise.write_meta()
        blockwise.clean_sync_folder()
        relocated = blockwise.write_binary(sequential=True, relocate=True)
        blockwise._set_input_array(relocated)
        _ = blockwise.write_binary(sequential=True, relocate=False, repetitions=100, # TODO: parse these parameters
                                   switch_slicing_direction = True
                                   )
        return out


class ApplyLabelAndRescale(ApplyAndRescale):
    def __init__(self,
                 input: Pyramid,
                 *args,
                 min_block_size=None,
                 block_overlap_sizes=None,
                 subset_indices: dict = None,
                 ### zarr parameters
                 store: str = None,
                 compressor='auto',
                 dimension_separator = None,
                 dtype=None,
                 overwrite=False,
                 ###
                 func=None,
                 n_jobs = None,
                 monoresolution = False,
                 **kwargs
                ):
        ApplyAndRescale.__init__(self,
                                input = input,
                                min_block_size = min_block_size,
                                block_overlap_sizes = block_overlap_sizes,
                                subset_indices = subset_indices,
                                store = store,
                                compressor = compressor,
                                dimension_separator = dimension_separator,
                                dtype = dtype,
                                overwrite = overwrite,
                                func = func,
                                n_jobs = n_jobs,
                                monoresolution = monoresolution,
                                *args, **kwargs
                                )


    def set_function(self, func):
        self.profiler = LabelProfiler(func)
        self.lazyfunc = LazyFunction(func, profiler=LabelProfiler)

    def set_runner(self, runner = None):
        if runner is None:
            runner = BlockwiseLabelRunner
        self.runner = runner

    def parse_params(self, *args, **kwargs):
        if self.profiler.is_label_filter:
            self.args, self.kwargs, overlap = _parse_params_for_label(self.input, self.profiler, *args, **kwargs)
            if self.block_overlap_sizes is None:
                self.block_overlap_sizes = overlap
            else:
                self.args = args
                self.kwargs = kwargs
        else:
            raise Exception(f"This is not a label function.")

    def _write_single_layer(self,
                            # out,
                            blockwise,
                            pth,
                            meta,
                            scales = None,
                            sequential = False,
                            downscaling_per_layer=False,
                            **kwargs
                            ):
        self.output.add_layer(blockwise,
                              pth,
                              scale=scales[pth],
                              # translation=self.input.get_translation(pth),
                              zarr_meta={'dtype': meta['dtype'],
                                         'chunks': meta['chunks'],
                                         'shape': blockwise.shape,
                                         'compressor': meta['compressor'],
                                         'dimension_separator': meta['dimension_separator']
                                         },
                              axis_order=self.input.axis_order,
                              unitlist=self.input.unit_list
                              )

        blockwise.write_meta()
        blockwise.clean_sync_folder()
        relocated = blockwise.write_binary(sequential=True, relocate=True, only_downscale=downscaling_per_layer)
        blockwise._set_input_array(relocated)
        if downscaling_per_layer:
            layer = relocated
        else:
            layer = blockwise.write_binary(sequential = True,
                                           relocate = False, repetitions = 100, # TODO: parse these parameters
                                           switch_slicing_direction = True, only_downscale = False
                                           )
        return layer

