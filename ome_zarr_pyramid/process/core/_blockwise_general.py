import warnings, time, shutil, zarr, itertools, multiprocessing, re, numcodecs, dask, os, copy, inspect
import numpy as np

import dask.array as da
import dask, logging, sys
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from joblib import Parallel, delayed, parallel_backend, register_parallel_backend
from dask.distributed import Lock
import dask.distributed as distributed

from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from skimage import transform

import numcodecs; numcodecs.blosc.use_threads = False

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.utils.general_utils import create_like

dask.config.set({
    "distributed.comm.retries.connect": 10,  # Retry connection 10 times
    "distributed.comm.timeouts.connect": "30s",  # Set connection timeout to 30 seconds
    "distributed.worker.memory.terminate": False,  # Prevent workers from terminating on memory errors
    "distributed.worker.reconnect": True,  # Workers will try to reconnect if they lose connection
    "distributed.worker.lifetime.duration": "2h",  # Optionally set a maximum worker lifetime
    "distributed.worker.lifetime.stagger": "10m",  # Workers restart staggered over 10 minutes to prevent all restarting at once
    'distributed.scheduler.worker-ttl': None,
    'distributed.worker.daemon': False
})

logging.getLogger('distributed.worker').setLevel(logging.ERROR)
logging.getLogger('distributed.comm').setLevel(logging.CRITICAL)
# Optional: Set global level for general Dask logs
logging.getLogger('distributed').setLevel(logging.ERROR)

def copy_zarray(zarray):
    copied = zarr.zeros_like(zarray)
    copied[:] = zarray[:]
    return copied

def copy_array(array):
    if isinstance(array, zarr.Array):
        copied = copy_zarray(array)
    elif isinstance(array, (da.Array, np.array)):
        copied = array.copy()
    return copied


def passive_func(array, *args, **kwargs):
    return array


class FunctionProfiler: # Change name to UnaryProfiler?
    """A class to explore functions.
        Ideally, the function must either be a ufunc or contain a signature.
    """
    def __init__(self, func):
        self.sample = np.zeros((2, 3))
        self.positional_scalar_argument = 1
        self.axis = 0
        self.needs_mandatory_args = False
        self.can_take_second_array_argument = False
        self.functype = None
        if isinstance(self.sample, np.ndarray):
            self._func_category = 'unary'
        elif isinstance(self.sample, (tuple, list)):
            self._func_category = 'aggregative'
        else:
            self._func_category = None
        self.func = func

    def __repr__(self):
        return self.name

    @property
    def func_category(self):
        if isinstance(self.sample, np.ndarray):
            self._func_category = 'unary'
        elif isinstance(self.sample, (tuple, list)):
            self._func_category = 'aggregative'
        else:
            pass
        return self._func_category

    @property
    def func(self):
        return self._func
    @func.setter
    def func(self, func):
        if func is None:
            self._func = passive_func
            self._name = self._func.__name__
        else:
            self._func = func
            self._name = self._func.__name__
        self.try_run()

    @property
    def name(self):
        if self._name is None:
            return self._name
        else:
            return self.func.__name__

    @property
    def params(self):
        try:
            signature = inspect.signature(self.func)
        except:
            return {}
        return signature.parameters

    @property
    def is_ufunc(self):
        return isinstance(self.func, np.ufunc)

    @property
    def mandatory_args(self):
        if self.is_ufunc:
            return TypeError(f"Ufuncs cannot have default argument values.")
        else:
            ret = []
            params = self.params
            for key in params:
                if params[key].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    if params[key].default == params[key].empty:
                        ret.append(params[key].name)
            return ret

    @property
    def n_mandatory_args(self):
        if self.is_ufunc:
            return self.func.nin
        else:
            params = self.params
            n_positional = 0
            for key in params:
                if params[key].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    if params[key].default == params[key].empty:
                        n_positional += 1
            return n_positional

    def try_run(self):
        """ TODO: This method can use some care."""
        func = self.func
        axis = self.axis
        if isinstance(self.sample, (list, tuple)):
            sample_array = self.sample[0]
        try: # is reductive?
            result = func(self.sample, axis = axis, keepdims = True)
            if self.name is None:
                self.functype = 'passive'
            elif self.name == 'passive_func':
                self.functype = 'passive'
            else:
                self.functype = 'reductive'
        except: # not reductive
            assert self.func_category in ['unary', 'aggregative'], f"Function category could not be determined."
            try:
                assert self.is_unary
                result = func(self.sample, axis=axis)  # directional but not affecting shape.
                self.functype = 'directional'
            except:
                if self.is_unary:
                    try:
                        assert self.n_mandatory_args == 0
                        result = func(self.sample)
                        self.functype = 'passive'
                    except:
                        try:
                            scalar_arguments = [self.positional_scalar_argument] * (self.n_mandatory_args - 1)
                            result = func(self.sample, *scalar_arguments)
                            self.functype = 'passive'
                            self.needs_mandatory_args = True
                        except:
                            warnings.warn(f"The function type for '{self.name}' could not be detected.")
                            self.functype = None
                elif self.is_aggregative:
                    try: # such as concatenate or stack
                        result = func(self.sample, axis=axis)  # expansive ?
                        self.needs_mandatory_args = True
                    except:
                        try: # such as aggregative sum
                            assert self.n_mandatory_args == 0
                            result = func(self.sample)
                            self.functype = 'passive'
                        except:
                            try: # This is probably not possible in aggregative
                                scalar_arguments = [self.positional_scalar_argument] * (self.n_mandatory_args - 1)
                                result = func(self.sample, *scalar_arguments)
                                self.functype = 'passive'
                                self.needs_mandatory_args = True
                            except:
                                warnings.warn(f"The function type for '{self.name}' could not be detected.")
                                self.functype = None
        try:
            if result.ndim < sample_array.ndim:
                self.functype = 'disruptive'
            elif result.ndim > sample_array.ndim:
                self.functype = 'generative'
            elif result.ndim == sample_array.ndim:
                if result.shape[axis] < sample_array.shape[axis]:
                    self.functype = 'reductive'
                elif result.shape[axis] > sample_array.shape[axis]:
                    self.functype = 'expansive'
        except:
            pass

    @property
    def is_directional(self):
        return self.functype == 'directional'
    @property
    def is_reductive(self):
        return self.functype == 'reductive'
    @property
    def is_expansive(self):
        return self.functype == 'expansive'
    @property
    def is_disruptive(self):
        return self.functype == 'disruptive'
    @property
    def is_generative(self):
        return self.functype == 'generative'
    @property
    def is_passive(self):
        return self.functype == 'passive'
    @property
    def is_vague(self):
        return self.functype is None

    @property
    def is_aggregative(self):
        return self.func_category == 'aggregative'
    @property
    def is_unary(self):
        return self.func_category == 'unary'


class LazyFunction:
    def __init__(self,
                 func,
                 profiler=FunctionProfiler,
                 *args,
                 **kwargs
                 ):

        self.func = profiler(func)
        self.parsed = {}
        self.parse_params(*args, **kwargs)

    def __repr__(self):
        return self.func.name

    def update_func(self, func):
        self._func = func
    @property
    def name(self):
        return self.func.name
    @property
    def params(self):
        return self.func.params
    @property
    def n_parsed(self):
        return len(self.parsed.keys())
    def parse_params(self, *args, **kwargs):
        self.args = []
        self.kwargs = {}
        if self.func.is_ufunc:
            for i, arg in enumerate(args):
                self.args.append(arg)
                self.parsed[i] = arg
            if len(kwargs.keys()) > 0:
                warnings.warn(f"Keyword parameters are not supported for ufuncs. Ignoring the provided parameters: {tuple(kwargs.keys())}")
        else:
            for i, (arg, key) in enumerate(zip(args, self.func.params.keys())):
                self.args.append(arg)
                self.parsed[key] = arg
            for i, key in enumerate(self.func.params.keys()):
                if key == 'keepdims':
                    self.kwargs[key] = True
                    self.parsed['keepdims'] = True
                else:
                    if key in kwargs.keys():
                        self.kwargs[key] = kwargs[key]
                        self.parsed[key] = kwargs[key]
    def get_runner(self, *args, **kwargs):
        self.parse_params(*args, **kwargs)
        if self.func.is_ufunc:
            if not self.n_parsed == self.func.n_mandatory_args:
                raise ValueError(
                    f"The ufunc '{self.func.name}' must take exactly {self.func.n_mandatory_args} arguments. {self.n_parsed} supplied.")
        elif not self.func.is_ufunc:
            for param in self.func.mandatory_args:
                if param not in self.parsed.keys():
                    warnings.warn(f"The mandatory argument '{param}' is missing.")
        if len(self.parsed.values()) < self.func.n_mandatory_args:
            raise ValueError(f"The function '{self.func.name}' requires {self.func.n_mandatory_args} arguments. {len(self.parsed.values())} supplied.")
        def run():
            if self.func.is_ufunc:
                return self.func.func(*self.parsed.values())
            try:
                return self.func.func(**self.parsed)
            except:
                self.parsed['axis'] = self.parsed['axis'][0] # Some functions require integer axis
                return self.func.func(**self.parsed)
        return run


def _downscale_block(input_slc, output_slc, i, input_array, output_array, scale_factor, dtype
                     ):
    block = input_array[input_slc]
    downscaled_block = transform.downscale_local_mean(block, tuple(scale_factor)).astype(dtype)
    output_array[output_slc] = downscaled_block
    return


class Aliases:
    @property
    def shape(self): # refers to the output
        return tuple(self.output_shape)

    @property
    def chunks(self): # refers to the output
        return tuple(self.output_chunks)

    @property
    def ndim(self): # alias
        return len(self.output_shape)


class BlockwiseRunner(Aliases):
    """
    1: subset
    2: run_function
    3: rescale
    """
    def __init__(self,
                 input_array: zarr.Array,
                 *args,
                 scale_factor = None,
                 min_block_size = None,
                 block_overlap_sizes=None,
                 subset_indices: Tuple[Tuple[int, int]] = None,
                 ### zarr parameters for the output
                 store = None,
                 compressor = 'auto',
                 chunks = None,
                 dimension_separator = '/',
                 dtype = None,
                 overwrite = False,
                 ###
                 func: LazyFunction = None,
                 n_jobs = None,
                 use_synchronizer = 'multiprocessing',
                 pyramidal_syncdir = os.path.expanduser('~') + '/.syncdir',
                 require_sharedmem = False,
                 **kwargs
                 ):
        self.store = store
        assert hasattr(func, 'get_runner') or func is None
        self.func = func
        self._set_input_array(input_array)
        self.block_overlap_sizes = block_overlap_sizes
        self._set_input_min_block_size(min_block_size)
        self._used_block_sizes = None
        self._output_block_sizes = None
        self._set_scale_factor(scale_factor)
        self._set_subset_indices(subset_indices)
        ### zarr parameters
        self.overwrite = overwrite
        self.compressor = compressor
        if chunks is not None:
            self._output_chunks = chunks
        self.dimension_separator = dimension_separator
        self.dtype = dtype
        ###
        self.set_params(*args, **kwargs)
        self._handle_axes()
        self.set_workers(n_jobs)
        self.use_synchronizer = use_synchronizer
        if store is not None and self.use_synchronizer is not None and pyramidal_syncdir is not None:
            self.syncdir = os.path.join(pyramidal_syncdir, os.path.basename(store) + '.sync')
        else:
            self.syncdir = None
        if require_sharedmem:
            self.require_sharedmem = 'sharedmem'
        else:
            self.require_sharedmem = None
        self._threads_per_worker = 1

    ### zarr parameters
    @property
    def compressor(self):
        return self._compressor
    @compressor.setter
    def compressor(self, compressor):
        if compressor == 'auto':
            self._compressor = self.input_array.compressor
        else:
            self._compressor = compressor

    @property
    def dimension_separator(self):
        return self._dimension_separator
    @dimension_separator.setter
    def dimension_separator(self, dimension_separator):
        if dimension_separator not in ['/', '.', None]:
            raise Exception(f"Dimension separator must be either of '{['/', '.', None]}'")
        if hasattr(self, 'store'):
            if isinstance(self.store, zarr.MemoryStore):
                self._dimension_separator = None
            else:
                if dimension_separator not in ['/', '.']:
                    raise Exception(f"Dimension separator must be either of '{['/', '.']}' for {type(self.store)}")
                if self.store_is_occupied and not self.overwrite:
                    raise Exception(f"Cannot assign a dimension separator to existing array.")
                else:
                    self._dimension_separator = dimension_separator
                    self.store = self.store.path
        else:
            self._dimension_separator = dimension_separator

    @property
    def store(self):
        return self._store
    @store.setter
    def store(self, store):
        self._store = None
        if hasattr(store, '__module__'):
            if store.__module__ == 'zarr.storage':
                self._store = store
                if isinstance(store, zarr.MemoryStore):
                    self.dimension_separator = None
                else:
                    self.dimension_separator = store._dimension_separator
            else:
                raise TypeError(f"The store must be a member of the module {zarr.storage}")
        elif isinstance(store, str):
            if not store.startswith('http'):
                if not hasattr(self, 'dimension_separator'):
                    self._dimension_separator = '/'
                elif self.dimension_separator is None:
                    self._dimension_separator = '/'
                self._store = zarr.DirectoryStore(store, dimension_separator = self._dimension_separator)
            else:
                raise Exception(f"Remote stores are not yet implemented.")
        elif store is None:
            self._store = zarr.MemoryStore()
            self.dimension_separator = None
        else:
            raise Exception(f"The store must be either a path or a store object from zarr.storage module.")

    @property
    def store_is_occupied(self):
        try:
            _ = zarr.open_array(self.store.path, mode='r')
            return True
        except:
            return False

    @property
    def dtype(self):
        return self._dtype
    @dtype.setter
    def dtype(self, dtype):
        if dtype is None:
            self._dtype = self.input_array.dtype
        else:
            self._dtype = dtype

    ### --------------------------------- ###

    def set_function(self, new_func, *args, **kwargs):
        self.__init__(input_array = self.input_array,
                      scale_factor = self.scale_factor,
                      min_block_size = self._input_min_block_size,
                      block_overlap_sizes = self.block_overlap_sizes,
                      subset_indices = self.subset_indices,
                      store = self.store,
                      compressor = self.compressor,
                      dimension_separator = self.dimension_separator,
                      dtype = self.dtype,
                      overwrite = self.overwrite,
                      func = new_func,
                      n_jobs = self.n_jobs,
                      *args, **kwargs
                      )

    @property
    def is_reductive(self):
        return self.func.func.is_reductive
    @property
    def is_directional(self):
        return self.func.func.is_directional
    @property
    def is_expansive(self):
        return self.func.func.is_expansive
    @property
    def is_passive(self):
        return self.func.func.is_passive
    @property
    def is_generative(self):
        return self.func.func.is_generative
    @property
    def is_ufunc(self):
        return self.func.func.is_ufunc

    @property
    def is_unary(self):
        return self.func.func.is_unary
    @property
    def is_aggregative(self):
        return self.func.func.is_aggregative

    @property
    def input_array(self):
        return self._input_array
    @input_array.setter
    def input_array(self, input_array):
        self._set_input_array(input_array)
        self._handle_axes()
    def _set_input_array(self, input_array):
        assert isinstance(input_array, zarr.Array), f"The input array must be a zarr.Array but is of type {input_array.__class__}."
        self._input_shape = input_array.shape
        self._input_array = input_array
        self._input_chunks = input_array.chunks
        self._output_chunks = copy.deepcopy(self._input_chunks)

    @property
    def input_chunks(self):
        try:
            return self._input_chunks
        except:
            return ValueError(f"The input chunks cannot be calculated.")

    @property
    def output_chunks(self):
        try:
            return self._output_chunks
        except:
            return ValueError(f"The output chunks cannot be calculated.")

    @property
    def _input_min_block_size(self):
        return self._input_min_block_size_
    @_input_min_block_size.setter
    def _input_min_block_size(self, min_block_size):
        self._set_input_min_block_size(min_block_size)
        self._handle_axes()
    def _set_input_min_block_size(self, min_block_size):
        if min_block_size is None:
            min_block_size = self.input_chunks
        if not len(min_block_size) == len(self.input_shape):
            raise ValueError(f"The length of the min block size must equal the input dimensionality.")
        min_block_size = np.minimum(min_block_size, self.input_shape).astype(int)
        self._input_min_block_size_ = tuple(min_block_size)

    @property
    def input_shape(self):
        return self._input_shape

    def _make_blocksizes_divisible(self):
        block_size = list(copy.deepcopy(self._input_min_block_size))
        scale_factor = self.scale_factor
        for i, (size, factor) in enumerate(zip(block_size, scale_factor)):
            rem = size % factor
            if rem > 0:
                block_size[i] = size + (factor - rem)
            else:
                block_size[i] = size
        return block_size

    @property
    def divisible_block_sizes(self):
        return self._make_blocksizes_divisible()

    @property
    def used_block_sizes(self):
        if self._used_block_sizes is None:
            return self.divisible_block_sizes
        else:
            return self._used_block_sizes
    @used_block_sizes.setter
    def used_block_sizes(self, used_block_sizes):
        self._used_block_sizes = used_block_sizes

    def _handle_axes(self):
        if isinstance(self.store, zarr.MemoryStore):
            self.store = None
        elif isinstance(self.store, zarr.DirectoryStore):
            self.store = self.store.path
        else:
            raise Exception(f"Stores other than zarr.MemoryStore\n"
                            f"and zarr.DirectoryStore are not yet implemented.")
        if self.is_reductive:
            self._set_subset_indices(self.subset_indices) # to maintain subset_shape
            scale_factor = list(copy.deepcopy(self._scale_factor))
            subset_shape = self.subset_shape.copy()
            output_shape = self._output_shape.copy()
            divisible_block_sizes = self.used_block_sizes
            output_block_sizes = copy.deepcopy(divisible_block_sizes)
            output_chunks = list(copy.deepcopy(self.input_chunks))
            if not 'axis' in self.func.params.keys():
                return ValueError(f"Reductive functions require an axis parameter.")
            if 'axis' in self.func.parsed.keys():
                axis = self.func.parsed['axis']
                if not isinstance(axis, (int, tuple, list)):
                    raise ValueError(f"Axis must be either of types {int, tuple, list}.")
            else:
                axis = (0,)
                warnings.warn(f"The axis parameter has not been provided. By default the axis '0' is chosen.")
            if isinstance(axis, int):
                axis = (axis,)
            for ax in axis:
                divisible_block_sizes[ax] = self.input_shape[ax]
                scale_factor[ax] = 1
                subset_shape[ax] = 1
                output_shape[ax] = 1
                output_block_sizes[ax] = 1
                output_chunks[ax] = 1
            self._used_block_sizes = divisible_block_sizes
            self._scale_factor = tuple(scale_factor)
            self.subset_shape = np.array(subset_shape)
            self._output_shape = np.array(output_shape)
            self._output_block_sizes = tuple(output_block_sizes)
            self._output_chunks = tuple(output_chunks)
        elif self.is_passive:
            self._set_input_min_block_size(self._input_min_block_size)
            self._set_subset_indices(self.subset_indices)

    @property
    def scale_factor(self):
        if self._scale_factor is None:
            return [1] * len(self.input_shape)
        else:
            return list(self._scale_factor)
    @scale_factor.setter
    def scale_factor(self, scale_factor):
        self._set_scale_factor(scale_factor)
        self._set_subset_indices(self.subset_indices)
        self._handle_axes()
    def _set_scale_factor(self, scale_factor):
        if scale_factor is None:
            scale_factor = [1] * len(self.input_shape)
        if not len(scale_factor) == len(self.input_shape):
            raise ValueError(f"The length of the scale factor must equal the input dimensionality.")
        self._scale_factor = scale_factor

    @property
    def subset_indices(self):
        if self._subset_indices is None:
            return tuple([(0, size) for size in self.input_shape])
        else:
            return self._subset_indices
    @subset_indices.setter
    def subset_indices(self, subset_indices):
        self._set_subset_indices(subset_indices)
        self._handle_axes()
    def _set_subset_indices(self, subset_indices):
        if subset_indices is None:
            subset_indices = tuple([(0, size) for size in self.input_shape])
        if not len(subset_indices) == len(self.input_shape):
            raise ValueError(f"The length of the subset indices must equal the input dimensionality.")
        sids = list(subset_indices)
        for i, ids in enumerate(sids):
            start, stop = ids
            if stop - start > self.input_shape[i]:
                stop = start + self.input_shape[i]
            sids[i] = (start, stop)
        self._subset_indices = tuple(sids)
        self.subset_shape, self.subset_slicer = self._get_input_slicer()
        self._output_shape = np.ceil(np.divide(self.subset_shape, self.scale_factor)).astype(int)

    def _get_input_slicer(self):
        subset_slice_array = np.array(self.subset_indices)
        assert subset_slice_array.ndim == 2
        assert subset_slice_array.shape[1] == 2, f"Each subset slice must contain an integer for the begin\n" \
                                                 f" and an integer for the end."
        assert subset_slice_array.shape[0] == len(self.scale_factor), f"The slice number must equal the number of dimensions."
        shape = np.ravel(subset_slice_array[:, 1] - subset_slice_array[:, 0])
        slicer = tuple([slice(*item) for item in self.subset_indices])
        return shape, slicer ### TODO: Make these properties?

    @property
    def output_shape(self):
        try:
            ret = tuple(np.array(self._output_shape).flatten().tolist())
        except:
            raise TypeError(f"Output shape is in type: {type(self._output_shape)}.")
        return ret

    @property
    def output_block_sizes(self):
        output_block_sizes = np.divide(self.divisible_block_sizes, self.scale_factor).astype(int)
        output_block_sizes = np.where(output_block_sizes > self.output_shape, self.output_shape, output_block_sizes)
        return output_block_sizes

    @property
    def block_overlap_sizes(self):
        return self._block_overlap_sizes
    @block_overlap_sizes.setter
    def block_overlap_sizes(self, block_overlap_sizes):
        if block_overlap_sizes is None:
            self._block_overlap_sizes = np.array([0] * len(self.input_shape))
        else:
            self._block_overlap_sizes = block_overlap_sizes

    def _get_block_slicer(self, input_base):
        subset_indices = self.subset_indices
        block_overlap_sizes = self.block_overlap_sizes
        input_base_ = tuple([slice(*item) for item in input_base])
        rbase = [None] * len(input_base)
        reducer = [None] * len(input_base)
        for i, _ in enumerate(input_base):
            start, stop = input_base[i]
            minlim, maxlim = subset_indices[i]
            overlap = block_overlap_sizes[i]
            if start < minlim:
                rstart = minlim
            elif start - overlap < minlim:
                rstart = minlim
            else:
                rstart = start - overlap
            if stop > maxlim:
                rstop = maxlim
            elif stop + overlap > maxlim:
                rstop = maxlim
            else:
                rstop = stop + overlap
            rbase[i] = slice(rstart, rstop)
        for i, (slc0, slc1) in enumerate(zip(input_base_, rbase)):
            start = slc0.start - slc1.start
            stop = slc0.stop - slc1.stop
            if stop == 0:
                stop = None
            reducer[i] = slice(start, stop)
        return tuple(rbase), tuple(reducer)

    def set_params(self, *args, **kwargs):
        self.func.parse_params(self.input_array, *args, **kwargs)
        self._handle_axes()

    def set_slurm_params(self, slurm_params: dict): # TODO: validate the slurm parameters
        self.slurm_params = slurm_params

    def set_threads_per_worker(self, value = 1):
        self._threads_per_worker = value

    def set_workers(self, n_jobs = None):
        cpus = multiprocessing.cpu_count()
        if n_jobs is None:
            n_jobs = cpus // 2
        # if n_jobs > cpus:
        #     warnings.warn(f"The given job number ({n_jobs}) exceeds the number of available cpus. The maximum cpu number ({cpus}) will be used.")
        #     n_jobs = cpus
        self.n_jobs = n_jobs
        return self

    def write_meta(self):
        self.output = create_like(self.input_array, shape=self.output_shape, chunks=self.output_chunks,
                                  store=self.store, compressor=self.compressor, dtype = self.dtype,
                                  dimension_separator = self.dimension_separator, overwrite = self.overwrite,
                                  use_synchronizer=self.use_synchronizer, syncdir = self.syncdir)
        return self

    def _get_slice_bases(self, direction = 'forwards'):
        if np.less(self.used_block_sizes, self.input_chunks).any():
            warnings.warn(f"The active block sizes have certain dimensions that are smaller than chunk dimensions: \n"
                          f"'{self.used_block_sizes}' vs '{self.input_chunks}'.\n"
            f"Choosing block sizes that are larger than the chunk size is generally a better idea.")

        input_start_ids = [np.arange(slc.start, slc.stop, increment) for slc, increment in zip(self.subset_slicer, self.used_block_sizes)]

        input_options = []
        for i, slc in enumerate(self.subset_slicer):
            slc = self.subset_slicer[i]
            minlim, maxlim = slc.start, slc.stop
            input_block_size = self.used_block_sizes[i]

            input_dim_ids_start = input_start_ids[i]
            input_dim_ids_start[input_dim_ids_start > maxlim] = maxlim
            input_dim_ids_start[input_dim_ids_start < minlim] = minlim

            input_dim_ids_end = input_start_ids[i] + input_block_size
            input_dim_ids_end[input_dim_ids_end > maxlim] = maxlim
            input_dim_ids_end[input_dim_ids_end < minlim] = minlim
            combin = np.vstack((input_dim_ids_start, input_dim_ids_end)).T
            input_options.append(combin)

        if direction == 'forwards':
            input_slice_bases = tuple(itertools.product(*input_options))
        elif direction == 'backwards':
            input_slice_bases = tuple(list(itertools.product(*input_options))[::-1])
        self.input_slice_bases = tuple(input_slice_bases)

        output_start_ids = [np.arange(0, loc, r) for loc, r in zip(self.output_shape, self.output_block_sizes)]

        output_options = []
        for i, size in enumerate(self.output_shape):
            output_block_size = self.output_block_sizes[i]
            output_dim_ids_start = output_start_ids[i]
            output_dim_ids_start[output_dim_ids_start > size] = size
            output_dim_ids_end = output_start_ids[i] + output_block_size
            output_dim_ids_end[output_dim_ids_end > size] = size
            combin = np.vstack((output_dim_ids_start, output_dim_ids_end)).T
            output_options.append(combin)

        if direction == 'forwards':
            output_slice_bases = tuple(itertools.product(*output_options))
        elif direction == 'backwards':
            output_slice_bases = tuple(list(itertools.product(*output_options))[::-1])
        self.output_slice_bases = tuple(output_slice_bases)
        return self.input_slice_bases, self.output_slice_bases

    def _create_slices(self, slicing_direction = 'forwards'):
        input_bases, output_bases = self._get_slice_bases(slicing_direction)
        self.input_slices, self.output_slices, self.reducer_slices = [], [], []
        for input_base, output_base in zip(input_bases, output_bases):
            input_slc, reducer_slc = self._get_block_slicer(input_base)
            output_slc = tuple([slice(*item) for item in output_base])
            self.input_slices.append(input_slc)
            self.reducer_slices.append(reducer_slc)
            self.output_slices.append(output_slc)

    def _downscale_block(self, i, input_slc, output_slc):
        block = transform.downscale_local_mean(self.input_array[input_slc], tuple(self.scale_factor)).astype(self.dtype)
        self.output[output_slc] = block
        return self.output

    def clean_sync_folder(self):
        if hasattr(self.output, 'synchronizer'):
            if self.output.synchronizer is not None:
                if isinstance(self.output.synchronizer.path, str):
                    shutil.rmtree(self.output.synchronizer.path)

    @property
    def is_slurm_available(self):
        return shutil.which("sbatch") is not None

    def _transform_block(self, i, input_slc, output_slc, x1, x2 = None,
                         reducer_slc = None,
                         ):
        if self.is_ufunc:
            if isinstance(x2, (np.ndarray, zarr.Array)):
                runner = self.func.get_runner(x1[input_slc].astype(self.dtype), x2[input_slc].astype(self.dtype))
            else:
                runner = self.func.get_runner(x1[input_slc].astype(self.dtype))
        else:
            runner = self.func.get_runner(x1[input_slc].astype(self.dtype))
        extended_block = runner()
        if reducer_slc is None:
            block = extended_block
        else:
            block = extended_block[reducer_slc]
        # if np.greater(self.scale_factor, 1).any():
        #     block = transform.downscale_local_mean(block, tuple(self.scale_factor))

        try:
            self.output[output_slc] = block.astype(self.dtype)
            # print(f"Block operation complete for block no {i} with output size {block.shape}")
            # print(f"input array dtype {np.dtype(self.input_array)}")
            # print(f"block dtype: {block.dtype}")
            # print(f"The block maximum is: {np.max(block)}")
            # print('###')
        except:
            print('#')
            print(f"Block no {i}")
            # print(reducer_slc)
            # print(input_base, output_base)
            print(input_slc)
            print(output_slc)
            print(f"Input block shape: {self.input_array[input_slc].shape}")
            print(f"Current output block shape: {block.shape}")
            print(f"Expected output block shape: {self.output[output_slc].shape}")
            print(f"Error at block no {i}")
            print('###')
        return self.output

    def _transform_block_with_lock(self, i, input_slc, output_slc, x1, x2 = None,
                         reducer_slc = None, lock = None):
        if lock is None:
            out = self._transform_block(i, input_slc, output_slc, x1, x2, reducer_slc)
        else:
            with lock:
                out = self._transform_block(i, input_slc, output_slc, x1, x2, reducer_slc)
        return out

    def _transform_block_insist(self, i, input_slc, output_slc, x1, x2 = None,
                         reducer_slc = None, lock = None):
        for _ in range(6):
            try:
                out = self._transform_block(i, input_slc, output_slc, x1, x2, reducer_slc)
                break
            except:
                time.sleep(3)
        return out

    def run_sequential(self, x1, x2 = None):
        for i, (input_slc, output_slc, reducer_slc) in enumerate(zip(self.input_slices,
                                                                     self.output_slices,
                                                                     self.reducer_slices
                                                                     )):
            _ = self._transform_block(i,
                                      input_slc,
                                      output_slc,
                                      x1,
                                      x2,
                                      reducer_slc=reducer_slc
                                      )

    def run_on_dask(self, x1, x2 = None):
        if self.is_slurm_available:
            assert hasattr(self,
                           'slurm_params'), f"SLURM parameters not configured. Please use the 'set_slurm_params' method."
            with SLURMCluster(**self.slurm_params) as cluster:
                print(self.slurm_params)
                cluster.scale(jobs=self.n_jobs)
                with Client(cluster,
                            heartbeat_interval="10s",
                            timeout="120s"
                            ) as client:
                    with parallel_backend('dask',
                                          wait_for_workers_timeout=600
                                          ):
                        lock = Lock('zarr-write-lock')
                        with Parallel(
                                verbose=False,
                                require=self.require_sharedmem,
                                n_jobs=self.n_jobs
                        ) as parallel:
                            _ = parallel(
                                delayed(self._transform_block_with_lock)(
                                    i,
                                    input_slc,
                                    output_slc,
                                    x1,
                                    x2,
                                    reducer_slc=reducer_slc,
                                    lock=lock
                                )
                                for i, (input_slc, output_slc, reducer_slc) in enumerate(zip(self.input_slices,
                                                                                             self.output_slices,
                                                                                             self.reducer_slices
                                                                                             ))
                            )
        else:
            with LocalCluster(n_workers=self.n_jobs,
                              processes=True,
                              threads_per_worker=self._threads_per_worker,
                              nanny=True,
                              memory_limit='8GB'
                              # memory_limit='auto'
                              # dashboard_address='127.0.0.1:8787',
                              # worker_dashboard_address='127.0.0.1:0',
                              # host = '127.0.0.1'
                              ) as cluster:
                cluster.scale(self.n_jobs)
                with Client(cluster,
                            heartbeat_interval="10s",
                            timeout="120s",
                            ) as client:
                    with parallel_backend('dask'):
                        lock = Lock('zarr-write-lock')
                        with Parallel(
                                verbose=False,
                                require=self.require_sharedmem
                        ) as parallel:
                            _ = parallel(
                                delayed(self._transform_block_with_lock)(
                                    i,
                                    input_slc,
                                    output_slc,
                                    x1,
                                    x2,
                                    reducer_slc=reducer_slc,
                                    lock=lock
                                )
                                for i, (input_slc, output_slc, reducer_slc) in enumerate(zip(self.input_slices,
                                                                                             self.output_slices,
                                                                                             self.reducer_slices
                                                                                             ))
                            )

    def run_on_dask_nolock(self, x1, x2 = None):
        start_time = time.time()
        if self.is_slurm_available:
            assert hasattr(self,
                           'slurm_params'), f"SLURM parameters not configured. Please use the 'set_slurm_params' method."
            with SLURMCluster(**self.slurm_params) as cluster:
                print(self.slurm_params)
                cluster.scale(jobs=self.n_jobs // 2)
                cluster.adapt(minimum = 1, maximum = self.n_jobs)
                with Client(cluster,
                            heartbeat_interval="60s",
                            timeout="600s"
                            ) as client:
                    with parallel_backend('dask',
                                          wait_for_workers_timeout=600
                                          ):
                        with Parallel(
                                verbose=self.verbose,
                                require=self.require_sharedmem,
                                # n_jobs=self.n_jobs
                        ) as parallel:
                            _ = parallel(
                                delayed(self._transform_block)(
                                    i,
                                    input_slc,
                                    output_slc,
                                    x1,
                                    x2,
                                    reducer_slc=reducer_slc,
                                )
                                for i, (input_slc, output_slc, reducer_slc) in enumerate(zip(self.input_slices,
                                                                                             self.output_slices,
                                                                                             self.reducer_slices
                                                                                             ))
                            )
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds for one set of jobs")
        else:
            with LocalCluster(n_workers=self.n_jobs,
                              processes=True,
                              threads_per_worker=self._threads_per_worker,
                              nanny=True,
                              memory_limit='8GB'
                              # memory_limit='auto'
                              # dashboard_address='127.0.0.1:8787',
                              # worker_dashboard_address='127.0.0.1:0',
                              # host = '127.0.0.1'
                              ) as cluster:
                cluster.scale(self.n_jobs)
                with Client(cluster,
                            heartbeat_interval="10s",
                            timeout="120s",
                            ) as client:
                    with parallel_backend('dask'):
                        with Parallel(
                                verbose=self.verbose,
                                require=self.require_sharedmem,
                                n_jobs=self.n_jobs,
                                prefer = 'threads'
                        ) as parallel:
                            _ = parallel(
                                delayed(self._transform_block)(
                                    i,
                                    input_slc,
                                    output_slc,
                                    x1,
                                    x2,
                                    reducer_slc=reducer_slc,
                                )
                                for i, (input_slc, output_slc, reducer_slc) in enumerate(zip(self.input_slices,
                                                                                             self.output_slices,
                                                                                             self.reducer_slices
                                                                                             ))
                            )

    def run_on_loky(self, x1, x2 = None):
        with parallel_backend('loky'):
            with Parallel(n_jobs=self.n_jobs,
                          verbose=False,
                          require=self.require_sharedmem,
                          # prefer='threads'
                          ) as parallel:
                _ = parallel(
                    delayed(self._transform_block)(i,
                                                   input_slc,
                                                   output_slc,
                                                   x1,
                                                   x2,
                                                   reducer_slc = reducer_slc
                                                   )
                    for i, (input_slc, output_slc, reducer_slc) in enumerate(zip(self.input_slices,
                                                                                 self.output_slices,
                                                                                 self.reducer_slices
                                                                                 )))

    def run_on_multiprocessing(self, x1, x2 = None):
        with parallel_backend('multiprocessing'):
            with Parallel(n_jobs=self.n_jobs,
                          verbose=False,
                          require=self.require_sharedmem) as parallel:
                _ = parallel(
                    delayed(self._transform_block)(i,
                                                   input_slc,
                                                   output_slc,
                                                   x1,
                                                   x2,
                                                   reducer_slc = reducer_slc
                                                   )
                    for i, (input_slc, output_slc, reducer_slc) in enumerate(zip(self.input_slices,
                                                                                 self.output_slices,
                                                                                 self.reducer_slices
                                                                                 )))

    def write_binary(self,
                     # sequential = False,
                     parallel_backend = 'dask',
                     # only_downscale = False,
                     slicing_direction = 'forwards',
                     ):
        x1 = self.input_array
        try:
            x2 = self.func.parsed[1]
        except:
            x2 = None
        self._create_slices(slicing_direction)

        if isinstance(self.store, zarr.MemoryStore):
            if parallel_backend != 'sequential':
                # warnings.warn(f"Currently, writing to MemoryStore is only supported in the sequential mode.\nSpecifying 'sequential=False' is, therefore, ignored for MemoryStore.")
                parallel_backend = 'sequential'
            if self.n_jobs > 1:
                warnings.warn(f"Currently, writing to MemoryStore is only supported in the sequential mode.\nThe 'n_jobs' value greater than 1 may not be exploited as expected.")

        if parallel_backend == 'sequential':
            self.run_sequential(x1, x2)
        elif parallel_backend == 'dask':
            self.run_on_dask(x1, x2)
        elif parallel_backend == 'dask_nolock':
            self.run_on_dask_nolock(x1, x2)
        elif parallel_backend == 'loky':
            self.run_on_loky(x1, x2)
        elif parallel_backend == 'multiprocessing':
            self.run_on_multiprocessing(x1, x2)
        return self.output

    def write_with_dask(self): # TODO
        raise NotImplementedError()

    def to_zarr(self, url = None, compressor = None, dimension_separator = None, overwrite = False):
        self.overwrite = overwrite
        self.compressor = compressor
        self.store = url
        try: # TODO: These two blocks look ugly.
            self.dimension_separator = dimension_separator
        except:
            pass
        try:
            self.write_meta()
        except:
            pass
        return self

    def __call__(self, *args, **kwargs):
        return self.write_binary()

