import tempfile, warnings, multiprocessing, itertools, shutil, glob, zarr, os, copy, inspect
from skimage import transform
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional, final )

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.core._blockwise_general import FunctionProfiler, BlockwiseRunner, LazyFunction
from ome_zarr_pyramid.process import process_utilities as putils


def _parse_args(*args):
    parsed_args = list(args)
    if len(args) > 0:
        if isinstance(args[0], str):
            if os.path.exists(args[0]):
                parsed_args[0] = Pyramid().from_zarr(args[0])
                assert putils.validate_pyramid_uniformity(
                    (input, parsed_args[0])), f"The input x1 is not elementwise compatible with the input x2."
            else:
                pass
        else:
            pass
    else:
        pass
    args = tuple(parsed_args)
    return args

def _parse_axes_for_directional(input, funpf, *args, **kwargs):
    """For reductive, expansive and directional"""
    args = _parse_args(*args)
    if 'axis' in kwargs.keys():
        axletters = kwargs.get('axis')
        axis = input.index(axletters, scalar_sensitive=False)
        if isinstance(axis, int):
            axis = (axis,)
        kwargs['axis'] = tuple(axis)
    else:
        args = list(args)
        keys = list(funpf.params.keys())
        idx_ax = keys.index('axis')
        if not len(args) >= idx_ax:
            raise Exception(f"The axis parameter is required for the function '{func}'.")
        axletters = args[idx_ax]
        axis = input.index(axletters, scalar_sensitive=False)
        if isinstance(axis, int):
            axis = (axis,)
        args[idx_ax] = tuple(axis)
        args = tuple(args)
    return args, kwargs

def _parse_axes_for_mutative(): pass
"""For disruptive and generative functions"""

def _parse_subset_indices(input, subset_indices):
    if subset_indices is not None:
        sub_ids = [(0, size) for size in input.shape]
        for i, ax in enumerate(input.axis_order):
            if ax in subset_indices.keys():
                sub_ids[i] = subset_indices[ax]
        sub_ids_start, sub_ids_stop = np.array(sub_ids).T
        array_shapes = np.array([input.array_meta[pth]['shape'] for pth in input.resolution_paths])
        array_shape_ratios = array_shapes / array_shapes.max(axis = 0)
        sub_ids_starts = np.multiply(sub_ids_start, array_shape_ratios).astype(int)
        sub_ids_stops = np.multiply(sub_ids_stop, array_shape_ratios).astype(int)
        sub_ids = {input.resolution_paths[i]: tuple([(first, second) for first, second in zip(start, stop)])
                   for i, (start, stop) in enumerate(zip(sub_ids_starts, sub_ids_stops))}
    else:
        sub_ids = {i: None
                   for i in input.resolution_paths}
    return sub_ids

def _parse_scale_factor(): pass


class ApplyToPyramid:
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
        assert isinstance(input, (Pyramid, PyramidCollection))
        self.input = input
        self.block_overlap_sizes = block_overlap_sizes
        self.set_function(func)
        self.set_runner(runner)
        self.parse_params(*args, **kwargs)
        self.min_block_size = min_block_size
        self.subset_indices = _parse_subset_indices(self.input, subset_indices)
        ### zarr parameters
        self.rescale_output = rescale_output
        self.select_layers = select_layers
        self.n_scales = n_scales
        self.store = store
        self.scale_factor = scale_factor
        self.downscale_func = None
        if compressor is None:
            self.compressor = self.input.compressor
        else:
            self.compressor = compressor
        if chunks is None:
            self.chunks = self.input.chunks
        else:
            self.chunks = chunks
        if dimension_separator is None:
            self.dimension_separator = self.input.dimension_separator
        else:
            self.dimension_separator = dimension_separator
        if dtype is None:
            self.dtype = self.input.dtype
        else:
            self.dtype = dtype
        self.overwrite = overwrite
        ###
        if n_jobs is None:
            self.n_jobs = multiprocessing.cpu_count() // 2
        else:
            self.n_jobs = n_jobs
        self.blockwises = {}

    def set_downscaler(self, downscale_func):
        self.downscale_func = downscale_func

    def set_function(self, func):
        self.profiler = FunctionProfiler(func)
        self.lazyfunc = LazyFunction(func, profiler = FunctionProfiler)

    def set_runner(self, runner = None):
        if runner is None:
            runner = BlockwiseRunner
        self.runner = runner

    def parse_params(self, *args, **kwargs):
        if self.profiler.is_reductive or self.profiler.is_directional or self.profiler.is_expansive:
            self.args, self.kwargs = _parse_axes_for_directional(self.input, self.profiler, *args, **kwargs)
        else:
            self.args = _parse_args(*args)
            self.kwargs = kwargs

    def _parse_default_scale_factor(self):
        scale_factor = [1] * self.input.ndim
        indices_yx = self.input.index('yx')
        if isinstance(indices_yx[0], (list, tuple)):
            indices_yx = indices_yx[0]
        for idx in indices_yx:
            scale_factor[idx] = 2
        return scale_factor

    def parse_select_layers(self):
        if self.select_layers == 'all':
            paths = self.input.resolution_paths
        elif isinstance(self.select_layers, int):
            paths = [str(pth) for pth in np.arange(self.select_layers)]
        elif isinstance(self.select_layers, (tuple, list)):
            paths = [str(pth) for pth in self.select_layers]
        else:
            raise ValueError(f"Resolution layers could not be detected.")
        return paths

    def _write_single_layer(self,
                            # out,
                            blockwise,
                            pth,
                            meta,
                            scales = None,
                            parallel_backend = 'dask',
                            # sequential = False,
                            downscaling_per_layer=False,
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

        blockwise.write_meta()
        _ = blockwise.write_binary(parallel_backend = parallel_backend,
                                   # only_downscale=False
                                   )
        return self.output

    def add_layers(self):
        select_paths = self.parse_select_layers()

        if self.rescale_output and self.select_layers != 'all':
            warnings.warn(f"Note that if rescale_output is True, the 'select_layers' parameter is ignored.")
            warnings.warn(f"To control the number of layers to be created, use n_layers parameter instead.")

        if self.rescale_output:
            paths = [self.input.refpath] ### parse only the refpath if the scale_factor is specified so that the output can be rescaled.
            if self.n_scales is None:
                self.n_scales = self.input.nlayers
            if self.downscale_func is None:
                self.set_downscaler(transform.downscale_local_mean)
            if self.scale_factor is None:
                self.scale_factor = self._parse_default_scale_factor()
        else:
            paths = select_paths
            assert self.n_scales is None, f"If rescale_output is False, the parameter n_scales must be None (default)."
            assert self.downscale_func is None, f"If rescale_output is False, the parameter downscale_func must be None (default)."
            assert self.scale_factor is None, f"If rescale_output is False, the parameter scale_factor must be None (default)."

        self.output = Pyramid()
        if self.overwrite:
            _ = zarr.group(self.store, overwrite = True)

        # syncdir = tempfile.mkdtemp()

        if not hasattr(self, 'syncdir'):
            self.syncdir = None

        for i, pth in enumerate(paths):
            layer = self.input.layers[pth]
            if self.store is not None:
                arraypath = os.path.join(self.store, pth)
            else:
                arraypath = None

            parsed_args = list(self.args)

            if len(self.args) > 0:
                if isinstance(self.args[0], Pyramid):
                    parsed_args[0] = self.args[0].layers[pth]

            blockwise = self.runner(layer,
                                    scale_factor = None,
                                    min_block_size = self.min_block_size,
                                    block_overlap_sizes = self.block_overlap_sizes,
                                    subset_indices = self.subset_indices[pth],
                                    store = arraypath,
                                    compressor = self.compressor,
                                    chunks = self.chunks,
                                    dimension_separator = self.dimension_separator,
                                    dtype = self.dtype,
                                    overwrite = self.overwrite,
                                    func = self.lazyfunc,
                                    n_jobs = self.n_jobs,
                                    use_synchronizer = 'multiprocessing',
                                    pyramidal_syncdir = self.syncdir,
                                    *parsed_args, **self.kwargs
                                    )

            meta = copy.deepcopy(self.input.array_meta[pth])
            meta['chunks'] = blockwise.chunks
            meta['compressor'] = blockwise.compressor
            meta['dimension_separator'] = blockwise.dimension_separator
            meta['dtype'] = blockwise.dtype
            if self.n_jobs == 1:
                self.parallel_backend = 'sequential'
            else:
                if not hasattr(self, 'parallel_backend'):
                    self.parallel_backend = 'dask'
                elif self.parallel_backend is None:
                    self.parallel_backend = 'dask'

            if not hasattr(self, 'verbose'):
                self.verbose = False
            else:
                if self.verbose is None:
                    self.verbose = False
                else:
                    self.verbose = True

            if not hasattr(self, '_threads_per_worker'):
                self._threads_per_worker = 1
            else:
                if self._threads_per_worker is None:
                    self._threads_per_worker = 1

            blockwise.set_threads_per_worker(self._threads_per_worker)

            self.output = self._write_single_layer(
                blockwise = blockwise,
                pth = pth,
                meta = meta,
                scales = None,
                parallel_backend = self.parallel_backend,
            )
            self.blockwises[pth] = blockwise

        self.output.to_zarr(self.store,
                            overwrite = False,
                            syncdir = self.syncdir, # TODO: handle synchronizers
                            only_meta = False
                            )

        ### do rescaling if needed

        if self.rescale_output and self.n_scales > 1:
            self.output.rescale( # TODO: test this method further.
                                n_layers = self.n_scales,
                                scale_factor = self.scale_factor,
                                min_input_block_size = self.min_block_size,
                                overwrite_layers = self.overwrite,
                                n_jobs = self.n_jobs,
                                downscale_func = self.downscale_func,
                                use_synchronizer='multiprocessing',
                                syncdir=self.syncdir,
                                backend = self.parallel_backend,
                                verbose = self.verbose
                                )
        ###
        # if self.output.refarray.synchronizer is not None:
        #     synchpath = os.path.dirname(self.output.refarray.synchronizer.path)
        #     shutil.rmtree(synchpath)
        return self.output



class ApplyAndRescale(ApplyToPyramid):
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
                 select_layers = 'all',
                 **kwargs
                ):
        ApplyToPyramid.__init__(self,
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
                                select_layers = select_layers,
                                **kwargs
                                )

    @property
    def scale_factor(self):
        if not hasattr(self, '_scale_factor'):
            self._scale_factor = None
        return self._scale_factor
    @scale_factor.setter
    def scale_factor(self, scale_factor):
        self._scale_factor = scale_factor

    def parse_scale_factors(self):
        if self.scale_factor is None:
            scales = np.array(list(self.input.get_scale(pth) for pth in self.input.resolution_paths))
            scales_ = np.vstack((scales[0], scales))
            scale_factors = np.around(np.divide(scales_[1:], scales_[:-1])).astype(int)
        else:
            scale_factors = np.vstack([self.scale_factor] * self.input.nlayers)
            scale_factors[0] = np.around(scale_factors[0] / scale_factors[0]).astype(int)
        cum_scale_factors = np.cumprod(scale_factors, axis=0)
        scale_factors = {pth: tuple(scale_factor) for pth, scale_factor in zip(self.input.resolution_paths, scale_factors)}
        scales = {pth: tuple(self.input.get_scale(self.input.refpath) * scale_factor) for pth, scale_factor in
                  zip(self.input.resolution_paths, cum_scale_factors)}
        return scale_factors, scales

    def parse_subset_indices(self):
        subset_ids = {i: None for i in self.input.resolution_paths}
        subset_ids[self.input.refpath] = self.subset_indices[self.input.refpath]
        return subset_ids

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
        layer = blockwise.write_binary(sequential=sequential,
                                       only_downscale=downscaling_per_layer
                                       )
        return layer

    def add_layers(self):
        array_meta = copy.deepcopy(self.input.array_meta)
        layer = self.input.refarray
        self.output = Pyramid()
        scale_factors, scales = self.parse_scale_factors()
        subset_ids = self.parse_subset_indices()
        funcs = {}
        for i in self.input.resolution_paths:
            if i == self.input.refpath:
                funcs[i] = self.lazyfunc
            else:
                funcs[i] = LazyFunction(None)
        downscaling = {i: True for i in self.input.resolution_paths}
        downscaling[self.input.refpath] = False

        ###
        paths = self.parse_select_layers()
        ###

        syncdir = tempfile.mkdtemp()
        block_overlap_sizes = self.block_overlap_sizes

        for i, pth in enumerate(paths):
            if self.store is not None:
                arraypath = os.path.join(self.store, pth)
            else:
                arraypath = None

            parsed_args = list(self.args)
            if len(self.args) > 0:
                if isinstance(self.args[0], Pyramid):
                    parsed_args[0] = self.args[0].layers[pth]

            blockwise = self.runner(layer,
                                    scale_factor=scale_factors[pth],
                                    min_block_size=self.min_block_size,
                                    block_overlap_sizes=block_overlap_sizes,
                                    subset_indices=subset_ids[pth],
                                    store=arraypath,
                                    compressor=self.compressor,
                                    chunks = self.chunks,
                                    dimension_separator=self.dimension_separator,
                                    dtype=self.dtype,
                                    overwrite=self.overwrite,
                                    func=funcs[pth],
                                    n_jobs=self.n_jobs,
                                    use_synchronizer='multiprocessing',
                                    pyramidal_syncdir=syncdir,
                                    *parsed_args, **self.kwargs
                                    )

            meta = copy.deepcopy(array_meta[pth])
            meta['chunks'] = blockwise.chunks
            meta['compressor'] = blockwise.compressor
            meta['dimension_separator'] = blockwise.dimension_separator
            meta['dtype'] = blockwise.dtype
            if self.n_jobs == 1:
                sequential = True
            else:
                sequential = False
            layer = self._write_single_layer(
                                             blockwise, pth, meta,
                                             scales, sequential, downscaling[pth])
            self.blockwises[pth] = blockwise
            block_overlap_sizes = None
        self.output.to_zarr(self.store, overwrite=self.overwrite)
        if self.output.refarray.synchronizer is not None:
            synchpath = os.path.dirname(self.output.refarray.synchronizer.path)
            shutil.rmtree(synchpath)
        return self.output


