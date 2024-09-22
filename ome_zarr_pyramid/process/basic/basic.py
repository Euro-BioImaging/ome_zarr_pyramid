import inspect

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.core.multiscale_apply_general import ApplyAndRescale, ApplyToPyramid
from ome_zarr_pyramid.creation.pyramid_creation import PyramidCreator
from ome_zarr_pyramid.utils import metadata_utils as meta_utils
# from ome_zarr_pyramid.process.filtering import custom_filters as cfilt

import zarr, warnings
import numpy as np
# import dask.array as da
from pathlib import Path
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from scipy import ndimage as ndi
from skimage import transform


class _WrapperBase:
    def __init__(self,
                 scale_factor=None,
                 min_block_size=None,
                 block_overlap_sizes=None,
                 input_subset_indices: dict = None,
                 ### zarr parameters
                 output_store: str = None,
                 output_compressor='auto',
                 output_chunks = None,
                 output_dimension_separator=None,
                 output_dtype=None,
                 overwrite=False,
                 ###
                 n_jobs = None,
                 rescale_output=False,
                 select_layers = 'all',
                 n_scales = None
                 ):
        self.zarr_params = {
            # 'scale_factor': scale_factor,
            'min_block_size': min_block_size,
            'block_overlap_sizes': block_overlap_sizes,
            'subset_indices': input_subset_indices,
            ### zarr parameters
            'store': output_store,
            'compressor': output_compressor,
            'chunks': output_chunks,
            'dimension_separator': output_dimension_separator,
            'dtype': output_dtype,
            'overwrite': overwrite,
            ###
            'n_jobs': n_jobs,
            'rescale_output': rescale_output,
            'select_layers': select_layers,
            'n_scales': n_scales
        }

        self.scale_factor = scale_factor

        self.slurm_params = {
            'cores': 8,  # per job
            'memory': "8GB",  # per job
            'nanny': True,
            'walltime': "02:00:00",
            "processes": 1,  # Number of processes (workers) per job
        }

        self.syncdir = None

    def set(self, **kwargs):
        for key, value in kwargs.items():
            key_ = key.replace('input_', '')
            key_ = key_.replace('output_', '')
            if key_ == 'scale_factor':
                self.scale_factor = kwargs.get('scale_factor')
            elif key_ == 'syncdir':
                self.syncdir = kwargs.get('syncdir')
            elif key_ in self.zarr_params.keys():
                self.zarr_params[key_] = value
            elif key in self.slurm_params.keys():
                self.slurm_params[key] = value
            else:
                raise TypeError(f"No such parameter as {key} exists.")
        return self



class BasicOperations(_WrapperBase, ApplyToPyramid):
    def __init__(self,
                 scale_factor=None,
                 min_block_size=None,
                 block_overlap_sizes=None,
                 input_subset_indices: dict = None,
                 ### zarr parameters
                 output_store: str = None,
                 output_compressor='auto',
                 output_chunks: Union[tuple, list] = None,
                 output_dimension_separator=None,
                 output_dtype=None,
                 overwrite=False,
                 ###
                 n_jobs=None,
                 rescale_output = False,
                 select_layers='all',
                 n_scales=None
                 ):
        _WrapperBase.__init__(
                              self, scale_factor, min_block_size, block_overlap_sizes, input_subset_indices,
                              output_store, output_compressor, output_chunks, output_dimension_separator,
                              output_dtype, overwrite, n_jobs, rescale_output, select_layers, n_scales
                              )

    def __run(self,
            input: Union[str, Pyramid],
            *args,
            func,
            out: str = '',
            **kwargs
            ):
        if out != '':
            self.set(store = out)
        if out is None:
            self.zarr_params['n_jobs'] = 1
        if isinstance(input, (str, Path)):
            input = Pyramid().from_zarr(input)

        ApplyToPyramid.__init__(self,
                                 input,
                                 *args,
                                 func=func,
                                 **self.zarr_params,
                                 **kwargs,
                                 scale_factor = self.scale_factor
                                 )
        return self.add_layers()

    def __run_with_dask(self):
        """TODO: Some functions benefits from dask. Connect this to a dedicated module.
        """
        raise NotImplementedError(f"This method is not yet implemented.")


    def absolute(self, x, out = None):
        return self.__run(input = x, func = np.absolute, out = out)

    def rint(self, x, out=None):
        return self.__run(input=x, func=np.rint, out=out)

    def sign(self, x, out=None):
        return self.__run(input=x, func=np.sign, out=out)

    def conj(self, x, out=None):
        return self.__run(input=x, func=np.conj, out=out)

    def exp(self, x, out=None):
        return self.__run(input=x, func=np.exp, out=out)

    def exp2(self, x, out=None):
        return self.__run(input=x, func=np.exp2, out=out)

    def log(self, x, out=None):
        return self.__run(input=x, func=np.log, out=out)

    def log2(self, x, out=None):
        return self.__run(input=x, func=np.log2, out=out)

    def log10(self, x, out=None):
        return self.__run(input=x, func=np.log10, out=out)

    def expm1(self, x, out=None):
        return self.__run(input=x, func=np.expm1, out=out)

    def log1p(self, x, out=None):
        return self.__run(input=x, func=np.log1p, out=out)

    def sqrt(self, x, out=None):
        return self.__run(input=x, func=np.sqrt, out=out)

    def square(self, x, out=None):
        return self.__run(input=x, func=np.square, out=out)

    def reciprocal(self, x, out=None):
        return self.__run(input=x, func=np.reciprocal, out=out)

    def zeros_like(self, x, out=None):
        return self.__run(input=x, func=np.zeros_like, out=out)

    def ones_like(self, x, out=None):
        return self.__run(input=x, func=np.ones_like, out=out)

    ### Creation ## TODO: integrate PyramidCreator here.

    def zeros(self, shape, out = None):
        raise NotImplementedError(f"This method is not yet implemented.")

    def ones(self, out = None):
        raise NotImplementedError(f"This method is not yet implemented.")

    def rand(self, out = None):
        raise NotImplementedError(f"This method is not yet implemented.")

    def create(self, out = None):
        raise NotImplementedError(f"This method is not yet implemented.")

    ### Trigonometric

    def sin(self, x, out=None):
        return self.__run(input=x, func=np.sin, out=out)

    def cos(self, x, out=None):
        return self.__run(input=x, func=np.cos, out=out)

    def tan(self, x, out=None):
        return self.__run(input=x, func=np.tan, out=out)

    def arcsin(self, x, out=None):
        return self.__run(input=x, func=np.arcsin, out=out)

    def arccos(self, x, out=None):
        return self.__run(input=x, func=np.arccos, out=out)

    def arctan(self, x, out=None):
        return self.__run(input=x, func=np.arctan, out=out)

    def sinh(self, x, out=None):
        return self.__run(input=x, func=np.sinh, out=out)

    def cosh(self, x, out=None):
        return self.__run(input=x, func=np.cosh, out=out)

    def tanh(self, x, out=None):
        return self.__run(input=x, func=np.tanh, out=out)

    def arcsinh(self, x, out=None):
        return self.__run(input=x, func=np.arcsinh, out=out)

    def arccosh(self, x, out=None):
        return self.__run(input=x, func=np.arccosh, out=out)

    def arctanh(self, x, out=None):
        return self.__run(input=x, func=np.arctanh, out=out)

    def deg2rad(self, x, out=None):
        return self.__run(input=x, func=np.deg2rad, out=out)

    def rad2deg(self, x, out=None):
        return self.__run(input=x, func=np.rad2deg, out=out)

    def arctan2(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.arctan2, out=out)

    def hypot(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.hypot, out=out)

    ### Dual input maths:

    def add(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.add, out=out)

    def subtract(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.subtract, out=out)

    def multiply(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.multiply, out=out)

    def divide(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.divide, out=out)

    def logaddexp(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.logaddexp, out=out)

    def logaddexp2(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.logaddexp2, out=out)

    def true_divide(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.true_divide, out=out)

    def floor_divide(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.floor_divide, out=out)

    def negative(self, x, out=None):
        return self.__run(input=x, func=np.negative, out=out)

    def power(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.power, out=out)

    def remainder(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.remainder, out=out)

    def mod(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.mod, out=out)

    def fmod(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.fmod, out=out)

    ### Other:

    def bitwise_and(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.bitwise_and, out=out)

    def bitwise_or(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.bitwise_or, out=out)

    def bitwise_xor(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.bitwise_xor, out=out)

    def invert(self, x, out=None):
        return self.__run(input=x, func=np.invert, out=out)

    def left_shift(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.left_shift, out=out)

    def right_shift(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.right_shift, out=out)

    def greater(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.greater, out=out)

    def greater_equal(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.greater_equal, out=out)

    def less(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.less, out=out)

    def less_equal(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.less_equal, out=out)

    def equal(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.equal, out=out)

    def not_equal(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.not_equal, out=out)

    def logical_and(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.logical_and, out=out)

    def logical_or(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.logical_or, out=out)

    def logical_xor(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.logical_xor, out=out)

    def logical_not(self, x, out=None):
        return self.__run(input=x, func=np.logical_not, out=out)

    def maximum(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.maximum, out=out)

    def minimum(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.minimum, out=out)

    def fmax(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.fmax, out=out)

    def fmin(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.fmin, out=out)

    def isfinite(self, x, out=None):
        return self.__run(input=x, func=np.isfinite, out=out)

    def isinf(self, x, out=None):
        return self.__run(input=x, func=np.isinf, out=out)

    def isnan(self, x, out=None):
        return self.__run(input=x, func=np.isnan, out=out)

    def isnat(self, x, out=None):
        return self.__run(input=x, func=np.isnat, out=out)

    def relocate(self, x, out=None):
        return self.__run(input=x, func=None, out=out)

    def fabs(self, x, out=None):
        return self.__run(input=x, func=np.fabs, out=out)

    def signbit(self, x, out=None):
        return self.__run(input=x, func=np.signbit, out=out)

    def copysign(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.copysign, out=out)

    def nextafter(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.nextafter, out=out)

    def modf(self, x, out=None):
        return self.__run(input=x, func=np.modf, out=out)

    def ldexp(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.ldexp, out=out)

    def frexp(self, x, out=None):
        return self.__run(input=x, func=np.frexp, out=out)

    def floor(self, x, out=None):
        return self.__run(input=x, func=np.floor, out=out)

    def ceil(self, x, out=None):
        return self.__run(input=x, func=np.ceil, out=out)

    def trunc(self, x, out=None):
        return self.__run(input=x, func=np.trunc, out=out)

    def maximum(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.maximum, out=out)

    def minimum(self, x1, x2, out=None):
        return self.__run(x1, x2, func=np.minimum, out=out)

    def clip(self, a, a_min, a_max, out=None):
        return self.__run(input=a, func=np.clip, a_min=a_min, a_max=a_max, out=out)

    ############################

    def max(self, input, axis, out=None):
        return self.__run(input, func=np.max, axis=axis, out=out)

    def min(self, input, axis, out=None):
        return self.__run(input=input, func=np.min, axis=axis, out=out)

    def mean(self, input, axis, out=None):
        return self.__run(input=input, func=np.mean, axis=axis, out=out)

    def median(self, input, axis, out=None):
        return self.__run(input=input, func=np.median, axis=axis, out=out)

    def all(self, input, axis, out=None):
        return self.__run(input=input, func=np.all, axis=axis, out=out)

    def any(self, input, axis, out=None):
        return self.__run(input=input, func=np.any, axis=axis, out=out)

    # def argmax(self, input, axis, out=None):
    #     return self.__run(input=input, func=np.argmax, axis=axis, out=out)
    #
    # def argmin(self, input, axis, out=None):
    #     return self.__run(input=input, func=np.argmin, axis=axis, out=out)
    #
    # def argtopk(self, input, k, axis, out=None):
    #     return self.__run(input=input, func=np.argtopk, axis=axis, k=k, out=out)

    def average(self, input, axis, out=None):
        return self.__run(input=input, func=np.average, axis=axis, out=out)

    def cumprod(self, input, axis, out=None):
        return self.__run(input=input, func=np.cumprod, axis=axis, out=out)

    def cumsum(self, input, axis, out=None):
        return self.__run(input=input, func=np.cumsum, axis=axis, out=out)

    # def diff(self, input, axis, out=None): # TODO with dask
    #     return self.__run(input=input, func=np.diff, axis=axis, out=out)

    def moment(self, input, axis, out=None):
        return self.__run(input=input, func=np.moment, axis=axis, out=out)

    def nanargmax(self, input, axis, out=None):
        return self.__run(input=input, func=np.nanargmax, axis=axis, out=out)

    def nanargmin(self, input, axis, out=None):
        return self.__run(input=input, func=np.nanargmin, axis=axis, out=out)

    def nancumprod(self, input, axis, out=None):
        return self.__run(input=input, func=np.nancumprod, axis=axis, out=out)

    def nancumsum(self, input, axis, out=None):
        return self.__run(input=input, func=np.nancumsum, axis=axis, out=out)

    def nanmax(self, input, axis, out=None):
        return self.__run(input=input, func=np.nanmax, axis=axis, out=out)

    def nanmean(self, input, axis, out=None):
        return self.__run(input=input, func=np.nanmean, axis=axis, out=out)

    def nanmedian(self, input, axis, out=None):
        return self.__run(input=input, func=np.nanmedian, axis=axis, out=out)

    def nanmin(self, input, axis, out=None):
        return self.__run(input=input, func=np.nanmin, axis=axis, out=out)

    def nanprod(self, input, axis, out=None):
        return self.__run(input=input, func=np.nanprod, axis=axis, out=out)

    def nanstd(self, input, axis, out=None):
        return self.__run(input=input, func=np.nanstd, axis=axis, out=out)

    def nansum(self, input, axis, out=None):
        return self.__run(input=input, func=np.nansum, axis=axis, out=out)

    def nanvar(self, input, axis, out=None):
        return self.__run(input=input, func=np.nanvar, axis=axis, out=out)

    def prod(self, input, axis, out=None):
        return self.__run(input=input, func=np.prod, axis=axis, out=out)

    def ptp(self, input, axis, out=None):
        return self.__run(input=input, func=np.ptp, axis=axis, out=out)

    def std(self, input, axis, out=None):
        return self.__run(input=input, func=np.std, axis=axis, out=out)

    def sum(self, input, axis, out=None):
        return self.__run(input=input, func=np.sum, axis=axis, out=out)

    # def take(self, input, indices, axis): # TODO with dask
    #     self.__run(input=input, func=np.take, axis=axis, indices=indices, out=None)

    def topk(self, input, k, axis, out=None):
        return self.__run(input=input, func=np.topk, axis=axis, k=k, out=out)

    def var(self, input, axis, out=None):
        return self.__run(input=input, func=np.var, axis=axis, out=out)

    def reduction(self, input, aggregate=None, axis=None, chunk=None, combine=None, concatenate=True, dtype=None,
                  meta=None, name=None, out=None, output_size=1):
        return self.__run(input=input, func=np.reduction, axis=axis, aggregate=aggregate, chunk=chunk,
                          combine=combine, concatenate=concatenate, dtype=dtype, meta=meta, name=name,
                          out=out, output_size=output_size)

    # def repeat(self, input, repeats, axis, out=None): # TODO with dask
    #     return self.__run(input=input, func=np.repeat, repeats=repeats, axis=axis, out=out)

    # def flip(self, input, axis, out=None): # TODO with dask
    #     return self.__run(input=input, func=np.flip, axis=axis, out=out)

    def gradient(self, input, axis, out=None):
        if isinstance(input, str):
            input = Pyramid().from_zarr(input)
        assert len(axis) == 1, f"Currently, gradients can only be computed in one direction."
        axes = input.index(axis, False)
        if self.zarr_params['block_overlap_sizes'] is None:
            fulloverlaps = [0] * input.ndim
        else:
            fulloverlaps = self.zarr_params['block_overlap_sizes']
        overlaps = [fulloverlaps[ax] for ax in axes]
        for idx, overlap in zip(axes, overlaps):
            if overlap < input.ndim + 1:
                overlap = input.ndim + 1
                fulloverlaps[idx] = overlap
        self.set(block_overlap_sizes = fulloverlaps)
        return self.__run(input=input, func=np.gradient, axis=axis, out=out)

    def delete(self, input, obj, axis, out=None):
        return self.__run(input=input, func=np.delete, obj=obj, axis=axis, out=out)

    def pad(self, pad_width):
        raise NotImplementedError(f"This method is not yet implemented.")
