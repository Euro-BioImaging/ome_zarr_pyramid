import copy
import inspect, itertools, os
# from dataclasses import dataclass
import napari
from pathlib import Path
from attrs import define, field, setters

import numcodecs; numcodecs.blosc.use_threads = False
from joblib import Parallel, delayed, parallel_backend

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.core import config
from ome_zarr_pyramid.process.core.multiscale_apply_general import ApplyAndRescale, ApplyToPyramid
from ome_zarr_pyramid.creation import multiscale_manager as msm
# from ome_zarr_pyramid.process.filtering import custom_filters as cfilt

import zarr, warnings
import numpy as np, dask.array as da


def _is_pyramid(input):
    return isinstance(input, Pyramid)

def _isarray(input):
    return isinstance(input, (np.ndarray, da.Array, zarr.Array))

def _is_simpleiter(input):
    return isinstance(input, (tuple, list))


##########################################################

def _parse_input(input): # input is a shape or array
    if _is_simpleiter(input):
        return tuple(input)
    else:
        return input

######################### get defaults for voxel metadata


def _get_default_axes(input): # This depends on the input (shape or array)
    axes_ = config.default_axes
    if _is_pyramid(input):
        axes = input.axis_order
    elif _isarray(input):
        offset = len(axes_) - input.ndim
        axes = axes_[offset:]
    elif _is_simpleiter(input):
        offset = len(axes_) - len(input)
        axes = axes_[offset:]
    return axes

def _get_default_units(axes): # This depends on the axes
    units = [config.unit_map[ax] for ax in axes]
    return units

def _get_default_scales(axes):
    # TODO
    return

######################### get defaults for zarr metadata


def _get_default_chunks(input):
    input = _parse_input(input)
    if isinstance(input, np.ndarray):
        chunks = np.repeat(1, input.ndim)
        lim = np.minimum(3, input.ndim)
        chunks[-lim:] = np.minimum(256, input.shape)[-lim:]
        chunks = chunks.tolist()
    elif isinstance(input, da.Array):
        chunks = input.chunksize
    elif isinstance(input, (zarr.Array, Pyramid)):
        chunks = input.chunks
    elif isinstance(input, (list, tuple)):
        chunks = np.repeat(1, len(input))
        lim = np.minimum(3, len(input))
        chunks[-lim:] = np.minimum(256, input)[-lim:]
        chunks = chunks.tolist()
    return tuple(chunks)

def _get_default_compressor(input):
    input = _parse_input(input)
    if isinstance(input, np.ndarray):
        compressor = numcodecs.Blosc()
    elif isinstance(input, da.Array):
        compressor = numcodecs.Blosc()
    elif isinstance(input, (zarr.Array, Pyramid)):
        compressor = input.compressor
    elif isinstance(input, (list, tuple)):
        compressor = numcodecs.Blosc()
    return compressor

def _get_default_dtype(input):
    if _isarray(input) or _is_pyramid(input):
        return input.dtype
    else:
        return int

def _get_default_dimension_separator(input):
    ret = '/'
    if isinstance(input, (zarr.Array, Pyramid)):
        if hasattr(input, 'dimension_separator'):
            ret = input.dimension_separator
    return ret


@define
class ZarrayCreator:
    input: (tuple, list, zarr.Array, da.Array, np.ndarray) = field(on_setattr = setters.frozen)
    ###
    dtype = field(on_setattr = setters.frozen)
    @dtype.default
    def _get_default_dtype(self):
        return _get_default_dtype(self.input)
    ###
    chunks: (tuple, list) = field(on_setattr = setters.frozen)
    @chunks.default
    def _get_default_chunks(self):
        return _get_default_chunks(self.input)
    ###
    compressor = field(on_setattr = setters.frozen)
    @compressor.default
    def _get_default_compressor(self):
        return _get_default_compressor(self.input)
    ###
    synchronizer = field(on_setattr = setters.frozen, default = None)
    ###
    store: str = field(on_setattr = setters.frozen, default = zarr.MemoryStore())
    ###
    dimension_separator: str = field(on_setattr = setters.frozen)
    @dimension_separator.default
    def _get_dimension_separator(self):
        return _get_default_dimension_separator(self.input)
    ###
    n_jobs: int = 8
    ###
    require_sharedmem: bool = None
    ###
    params: dict = field(init = False)
    ###
    zarray: zarr.Array = field(init = False)

    @property
    def shape(self):
        if _isarray(self.input):
            return self.input.shape
        else:
            assert _is_simpleiter(self.input)
            return self.input

    def __attrs_post_init__(self):
        if isinstance(self.store, zarr.MemoryStore):
            store = zarr.MemoryStore()
        else:
            store = self.store
        self.params = {
            'shape': self.shape,
            'dtype': self.dtype,
            'chunks': self.chunks,
            'compressor': self.compressor,
            'synchronizer': self.synchronizer,
            'store': store,
            'dimension_separator': self.dimension_separator
        }

    @property
    def chunk_indices(self):
        if not hasattr(self, 'zarray'):
            return None
        ranges = []
        for lim in self.zarray.cdata_shape:
            ranges.append(np.arange(0, lim, 1).tolist())
        indices = tuple(itertools.product(*ranges))
        return indices

    def _assign_to_chunk(self, block_id: tuple, value):
        if not hasattr(self, 'zarray'):
            raise ValueError(f"Zarr array does not exist.")
        self.zarray.set_block_selection(block_id, value = value)
        return
    def _batch_process(self, **kwargs):
        with parallel_backend('multiprocessing'):
            with Parallel(n_jobs=self.n_jobs, require=self.require_sharedmem) as parallel:
                _ = parallel(
                    delayed(self._assign_to_chunk)(block_id = indices, **kwargs)
                    for i, indices in enumerate(self.chunk_indices))

    def create(self) -> zarr.Array:
        self.zarray = zarr.create(**self.params)
        return self.zarray

    def full(self, fill_value = 0) -> zarr.Array:
        self.params['fill_value'] = fill_value
        self.zarray = zarr.full(**self.params)
        self._batch_process(value = fill_value)
        return self.zarray

    def zeros(self) -> zarr.Array:
        self.zarray = zarr.zeros(**self.params)
        self._batch_process(value = 0)
        return self.zarray

    def ones(self) -> zarr.Array:
        self.zarray = zarr.ones(**self.params)
        self._batch_process(value=1)
        return self.zarray

    def rand(self) -> zarr.Array:
        self.zarray = zarr.create(**self.params)
        if isinstance(self.params['store'], zarr.MemoryStore):
            for indices in self.chunk_indices:
                block = np.random.rand(*self.zarray.blocks[indices].shape)
                self._assign_to_chunk(indices, value = block)
        else:
            with parallel_backend('multiprocessing'):
                with Parallel(n_jobs=self.n_jobs, require=self.require_sharedmem) as parallel:
                    _ = parallel(
                        delayed(self._assign_to_chunk)(block_id = indices,
                                                       value = np.random.rand(*self.zarray.blocks[indices].shape))
                        for i, indices in enumerate(self.chunk_indices))
        return self.zarray


def _get_default_scale_factor(input, axes):
    scale_factor = [config.scale_factor_map[ax] for ax in axes]
    if _is_pyramid(input):
        if '1' in input.resolution_paths:
            scale_factor = input.scale_factors['1']
    return scale_factor


@define
class PyramidCreator:
    #########################
    ### Group parameters
    #########################
    input: (tuple, list, zarr.Array, da.Array, np.ndarray, Pyramid) = field(on_setattr = setters.frozen)
    @input.default
    def _warn(self):
        raise ValueError(f"Please provide an input (Pyramid, tuple or array).")
    store: str = field(on_setattr = setters.frozen, default = zarr.MemoryStore())
    #########################
    ### Voxel parameters
    #########################
    axis_order: str = field(on_setattr = setters.frozen)
    @axis_order.default
    def _get_default_axes(self):
        return _get_default_axes(self.input)
    unit_list: (tuple, list) = field(on_setattr = setters.frozen)
    @unit_list.default
    def _get_default_units(self):
        return _get_default_units(self.axis_order)
    scale: (tuple, list) = field(on_setattr = setters.frozen)
    @scale.default
    def _get_default_scale(self):
        scale = [1] * len(self.axis_order)
        if _is_pyramid(self.input):
            if '0' in self.input.resolution_paths:
                scale = self.input.scales['0']
        return scale
    # translation: (list, tuple)
    #########################
    ### MultiScales metadata
    #########################
    n_resolutions: int = field(on_setattr = setters.frozen, default = 1)
    scale_factor: (tuple, list) = field(on_setattr = setters.frozen)
    @scale_factor.default
    def _get_default_scale_factor(self):
        return _get_default_scale_factor(self.input, self.axis_order)
    #########################
    ### Other Zarr metadata
    #########################
    dtype = field(on_setattr = setters.frozen)
    @dtype.default
    def _get_default_dtype(self):
        return _get_default_dtype(self.input)
    ###
    chunks: (tuple, list) = field(on_setattr = setters.frozen)
    @chunks.default
    def _get_default_chunks(self):
        return _get_default_chunks(self.input)
    ###
    compressor = field(on_setattr = setters.frozen)
    @compressor.default
    def _get_default_compressor(self):
        return _get_default_compressor(self.input)
    ###
    synchronizer = field(on_setattr = setters.frozen, default = None)
    ###
    dimension_separator: str = field(on_setattr = setters.frozen)
    @dimension_separator.default
    def _get_dimension_separator(self):
        return _get_default_dimension_separator(self.input)
    #########################
    ### Concurrency parameters
    #########################
    n_jobs: int = field(on_setattr = setters.frozen, default = 8)
    ###
    require_sharedmem: bool = field(on_setattr = setters.frozen, default = None)
    #######################

    @property
    def shape(self):
        if _isarray(self.input) or _is_pyramid(self.input):
            return self.input.shape
        else:
            assert _is_simpleiter(self.input)
            return self.input

    @property
    def resolution_paths(self):
        paths = np.arange(self.n_resolutions)
        return [f'{pth}' for pth in paths]

    def _parse_scale_factors(self):
        scale_factor = self.scale_factor
        scale = self.scale
        paths = [str(pth) for pth in np.arange(self.n_resolutions)]
        if _is_pyramid(self.input):
            current_shapes = self.input.layer_shapes
        else:
            current_shapes = {'0': self.shape}
        current_paths = list(current_shapes.keys())
        if np.any([item in current_paths for item in paths]): #paths.size > current_shapes.__len__():
            shapes = msm._extend_shapes(current_shapes, scale_factor, paths)
        else:
            shapes = current_shapes
        scale_factors = msm._get_scale_factors_from_shapes(shapes)
        scales = {pth: tuple(np.multiply(scale, scale_factor)) for pth, scale_factor in scale_factors.items()}
        return scale_factors, scales, shapes

    def _create(self, method = 'create', **kwargs):
        assert method in ['create', 'zeros', 'ones', 'full', 'rand']
        params_ = {
                 'input': self.shape,
                 'dtype': self.dtype,
                 'chunks': self.chunks,
                 'compressor': self.compressor,
                 'synchronizer': self.synchronizer,
                 'store': self.store,
                 'dimension_separator': self.dimension_separator
        }
        if _is_pyramid(self.input):
            default_scale_factor = _get_default_scale_factor(self.input, self.axis_order)
            if default_scale_factor != self.scale_factor:
                self.input.shrink(['0'])
        scale_factors, scales, shapes = self._parse_scale_factors()
        pyr = Pyramid()
        for pth in self.resolution_paths:
            meta = copy.deepcopy(params_)
            meta['input'] = shapes[pth]
            if isinstance(self.store, (zarr.MemoryStore)) or self.store is None:
                meta['store'] = zarr.MemoryStore()
            else:
                assert isinstance(self.store, (str, zarr.DirectoryStore, Path))
                meta['store'] = os.path.join(self.store, pth)
            creator = ZarrayCreator(**meta)
            arr = getattr(creator, method)(**kwargs)
            pyr.add_layer(arr,
                          pth,
                          scale=scales[pth],
                          # translation=self.input.get_translation(pth),
                          zarr_meta={
                          'dtype': meta['dtype'],
                          'chunks': meta['chunks'],
                          'shape': arr.shape,
                          'compressor': meta['compressor'],
                          'dimension_separator': meta['dimension_separator']
                          },
                          axis_order=self.axis_order,
                          unitlist=self.unit_list
                          )
        pyr.to_zarr(self.store, only_meta = True)
        return pyr

    def create(self):
        return self._create('create')

    def zeros(self):
        return self._create('zeros')

    def ones(self):
        return self._create('ones')

    def full(self, fill_value):
        return self._create('full', fill_value = fill_value)

    def rand(self):
        return self._create('rand')




