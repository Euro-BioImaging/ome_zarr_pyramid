import zarr
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.core._blockwise_general import LazyFunction
from ome_zarr_pyramid.process.filtering._blockwise_filter import FilterProfiler, BlockwiseFilterRunner
from ome_zarr_pyramid.process.core.multiscale_apply_general import ApplyToPyramid, ApplyAndRescale, \
    _parse_subset_indices, _parse_args, _parse_axes_for_directional


def _parse_params_for_scipy_ndimage(input, profiler, *args, **kwargs):
    def _get_overlap_sizes(fp):
        if isinstance(fp, (tuple, list)):
            shape = np.array(fp)
        else:
            shape = np.array(fp.shape)
        overlap = shape // 2 + 3
        return tuple(overlap.tolist())

    def _fp_from_gaussian_params(sigma, truncate = 4.0, radius = None, **kwargs):
        if radius is None:
            radius = (truncate * np.array(sigma) + 0.5).astype(int)
        if np.isscalar(radius):
            radius = [radius] * input.ndim
        fp = tuple(2 * np.array(radius) + 1)
        return fp

    if 'axis' in kwargs.keys():
        axletters = kwargs.get('axis')
        if axletters is None:
            axis = -1
        else:
            axis = input.index(axletters, scalar_sensitive=False)
        kwargs['axis'] = tuple(axis)

    if 'footprint' in kwargs.keys():
        fp = kwargs.get('footprint')
        if np.isscalar(fp):
            fp = tuple([fp] * input.ndim)
        if not isinstance(fp, np.ndarray):
            fp = np.ones(fp)
        assert input.ndim == fp.ndim
        kwargs['footprint'] = fp
    elif 'size' in kwargs.keys():
        size = kwargs.get('size')
        if np.isscalar(size):
            size = tuple([size] * input.ndim)
        fp = np.ones(size)
        kwargs['footprint'] = fp
    elif 'weights' in kwargs.keys():
        fp = kwargs.get('weights')
        if not isinstance(fp, np.ndarray):
            raise TypeError(f"The parameter weights must be of type numpyp.ndarray.")
        if not input.ndim == fp.ndim:
            raise TypeError(f"The parameter weights must have the same number of dimensions as input.")
    elif 'sigma' in kwargs.keys():
        fp = _fp_from_gaussian_params(**kwargs)
    else:
        args = list(args)
        keys = list(profiler.params.keys())
        idx_fp = keys.index('footprint')
        if len(args) >= idx_fp:
            pass
        else:
            idx_s = keys.index('size')
            if len(args) >= idx_s:
                size = args[idx_s]
                if np.isscalar(size):
                    size = tuple([size] * input.ndim)
                fp = np.ones(size)
                args[idx_fp] = fp
            else:
                raise Exception(f"The footprint parameter is missing for the function '{func}'.")
    overlap = _get_overlap_sizes(fp)
    return args, kwargs, overlap


class ApplyFilterToPyramid(ApplyToPyramid):
    def __init__(self,
                 input: Pyramid,
                 *args,
                 min_block_size=None,
                 block_overlap_sizes=None,
                 subset_indices: dict = None,
                 ### zarr parameters
                 store: str = None,
                 compressor='auto',
                 chunks = None,
                 dimension_separator = None,
                 dtype=None,
                 overwrite=False,
                 ###
                 func=None,
                 runner = None,
                 n_jobs = None,
                 ### parameters to manage multiscaling
                 rescale_output = False, # overrides select_layers
                 select_layers: (int, str, list, tuple) = 'all',
                 scale_factor = None,
                 n_scales = None,
                 **kwargs
                 ):
        super().__init__(
                        input = input,
                        *args,
                        min_block_size = min_block_size,
                        block_overlap_sizes = block_overlap_sizes,
                        subset_indices = subset_indices,
                        store = store,
                        compressor = compressor,
                        chunks = chunks,
                        dimension_separator = dimension_separator,
                        dtype = dtype,
                        overwrite = overwrite,
                        func = func,
                        runner = runner,
                        n_jobs = n_jobs,
                        rescale_output = rescale_output,
                        select_layers = select_layers,
                        scale_factor = scale_factor,
                        n_scales = n_scales,
                        **kwargs
                        )

    def set_function(self, func):
        self.profiler = FilterProfiler(func)
        self.lazyfunc = LazyFunction(func, profiler=FilterProfiler)

    def parse_params(self, *args, **kwargs):
        if self.profiler.is_scipy_ndimage_filter:
            self.args, self.kwargs, overlap = _parse_params_for_scipy_ndimage(self.input, self.profiler, *args, **kwargs)
            if self.block_overlap_sizes is None:
                self.block_overlap_sizes = overlap
        else:
            raise Exception(f"Currently only scipy.ndimage filters are supported.")

#
# class ApplyFilterAndRescale(ApplyAndRescale):
#     def __init__(self,
#                  input: Pyramid,
#                  min_block_size=None,
#                  block_overlap_sizes=None,
#                  subset_indices: dict = None,
#                  ### zarr parameters
#                  store: str = None,
#                  compressor='auto',
#                  dimension_separator = None,
#                  dtype=None,
#                  overwrite=False,
#                  ###
#                  func=None,
#                  n_jobs = None,
#                  monoresolution = False,
#                  *args,
#                  **kwargs
#                 ):
#         super().__init__(
#                         input = input,
#                         min_block_size = min_block_size,
#                         block_overlap_sizes = block_overlap_sizes,
#                         subset_indices = subset_indices,
#                         store = store,
#                         compressor = compressor,
#                         dimension_separator = dimension_separator,
#                         dtype = dtype,
#                         overwrite = overwrite,
#                         func = func,
#                         n_jobs = n_jobs,
#                         monoresolution = monoresolution,
#                          *args,
#                          **kwargs
#                         )
#     def set_function(self, func):
#         self.profiler = FilterProfiler(func)
#         self.lazyfunc = LazyFunction(func, profiler=FilterProfiler)
#
#     def parse_params(self, *args, **kwargs):
#         if self.profiler.is_scipy_ndimage_filter:
#             self.args, self.kwargs, overlap = _parse_params_for_scipy_ndimage(self.input, self.profiler, *args, **kwargs)
#             if self.block_overlap_sizes is None:
#                 self.block_overlap_sizes = overlap
#         else:
#             raise Exception(f"Currently only scipy.ndimage filters are supported.")

