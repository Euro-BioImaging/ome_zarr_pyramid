from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.core._blockwise_general import LazyFunction
from ome_zarr_pyramid.process.thresholding import _local_threshold as lt, _global_threshold as gt
from ome_zarr_pyramid.process.thresholding._blockwise_threshold import BlockwiseThresholdRunner, ThresholderProfiler

from ome_zarr_pyramid.process.core.multiscale_apply_general import ApplyToPyramid, ApplyAndRescale, \
    _parse_subset_indices, _parse_args, _parse_axes_for_directional


def _parse_params_for_local_threshold(input, profiler, *args, **kwargs):
    def _get_overlap_sizes(window_shape):
        if isinstance(window_shape, (tuple, list)):
            shape = np.array(window_shape)
        elif isinstance(window_shape, np.ndimage):
            shape = window_shape
        overlap = shape // 2 + 3
        return tuple(overlap.tolist())
    if 'window_shape' in kwargs.keys():
        window_shape = kwargs['window_shape']
        assert hasattr(window_shape, '__len__')
        assert len(window_shape) == input.ndim
        overlap = _get_overlap_sizes(window_shape)
    elif 'window_size' in kwargs.keys():
        window_size = kwargs.get('window_size')
        if np.isscalar(window_size):
            window_size = input.ndim * [window_size]
            kwargs['window_size'] = tuple(window_size)
        else:
            assert len(window_size) == input.ndim
        overlap = _get_overlap_sizes(window_size)
    else:
        raise ValueError(f"At least one of window_shape or window_size must be provided.")
    return args, kwargs, overlap


class ApplyThresholdToPyramid(ApplyToPyramid):
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
                 global_thresholder = None,
                 global_threshold_params = {},
                 global_threshold_level = None,
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
                        chunks=chunks,
                        dimension_separator = dimension_separator,
                        dtype = dtype,
                        overwrite = overwrite,
                        func = func,
                        n_jobs = n_jobs,
                        rescale_output = rescale_output,
                        select_layers = select_layers,
                        scale_factor = scale_factor,
                        n_scales = n_scales,
                        **kwargs
                        )
        self._global_thresholder = global_thresholder
        if self.profiler.requires_global_threshold and self.profiler.requires_local_threshold:
            assert self._global_thresholder is not None, f"The function {self.profiler.name} requires a global thresholder. Please assign one."
        self._global_threshold_params = global_threshold_params
        self._global_threshold_level = global_threshold_level

    def set_function(self, func):
        self.profiler = ThresholderProfiler(func)
        self.lazyfunc = LazyFunction(func, profiler=ThresholderProfiler)

    def set_runner(self, runner = None):
        if runner is None:
            runner = BlockwiseThresholdRunner
        self.runner = runner

    def parse_params(self, *args, **kwargs):
        if self.profiler.is_skimage_thresholder:
            if self.profiler.requires_local_threshold:
                self.args, self.kwargs, overlap = _parse_params_for_local_threshold(self.input, self.profiler, *args, **kwargs)
                if self.block_overlap_sizes is None:
                    self.block_overlap_sizes = overlap
            else:
                self.args = args
                self.kwargs = kwargs
        else:
            raise Exception(f"Currently only skimage thresholders are supported.")

    def _write_single_layer(self,
                            # out,
                            blockwise,
                            pth,
                            meta,
                            scales = None, # only for compatibility
                            sequential = False,
                            downscaling_per_layer = False,
                            **kwargs
                            ):
        if hasattr(self, 'slurm_params'):
            blockwise.set_slurm_params(self.slurm_params)
        self.output.add_layer(blockwise,
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

        if blockwise.func.func.__class__ == ThresholderProfiler:
            if self.profiler.requires_global_threshold:
                if self._global_thresholder is not None:
                    blockwise.set_global_thresholder(self._global_thresholder, **self._global_threshold_params)
                if self._global_threshold_level is not None:
                    blockwise.set_threshold_level(self._global_threshold_level)
                blockwise.compute_global_threshold()
        else:
            # print(f"not here")
            pass

        blockwise.write_meta()
        _ = blockwise.write_binary(sequential=False,
                                   only_downscale=False
                                   )
        return self.output


class ApplyThresholdAndRescale(ApplyAndRescale):
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
                 global_thresholder=None,
                 global_threshold_params={},
                 global_threshold_level=None,
                 **kwargs
                ):
        ApplyAndRescale.__init__(self,
                                input = input,
                                *args,
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
                                **kwargs
                                )
        self._global_thresholder = global_thresholder
        if self.profiler.requires_global_threshold and self.profiler.requires_local_threshold:
            assert self._global_thresholder is not None, f"The function {self.profiler.name} requires a global thresholder. Please assign one."
        self._global_threshold_params = global_threshold_params
        self._global_threshold_level = global_threshold_level

    def set_function(self, func):
        self.profiler = ThresholderProfiler(func)
        self.lazyfunc = LazyFunction(func, profiler=ThresholderProfiler)

    def set_runner(self, runner = None):
        if runner is None:
            runner = BlockwiseThresholdRunner
        self.runner = runner

    def parse_params(self, *args, **kwargs):
        if self.profiler.is_skimage_thresholder:
            if self.profiler.requires_local_threshold:
                self.args, self.kwargs, overlap = _parse_params_for_local_threshold(self.input, self.profiler, *args, **kwargs)
                if self.block_overlap_sizes is None:
                    self.block_overlap_sizes = overlap
            else:
                self.args = args
                self.kwargs = kwargs
        else:
            raise Exception(f"Currently only skimage thresholders are supported.")

    def _write_single_layer(self,
                            # out,
                            blockwise,
                            pth,
                            meta,
                            scales = None,
                            sequential = False,
                            downscaling_per_layer = False,
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

        if blockwise.func.func.__class__ == ThresholderProfiler:
            if self.profiler.requires_global_threshold:
                if self._global_thresholder is not None:
                    blockwise.set_global_thresholder(self._global_thresholder, **self._global_threshold_params)
                if self._global_threshold_level is not None:
                    blockwise.set_threshold_level(self._global_threshold_level)
                blockwise.compute_global_threshold()
        else:
            pass

        blockwise.write_meta()
        layer = blockwise.write_binary(sequential=False,
                                       only_downscale=downscaling_per_layer
                                       )
        return layer
