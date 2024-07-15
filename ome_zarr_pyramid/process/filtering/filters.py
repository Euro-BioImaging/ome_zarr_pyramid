from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.filtering.multiscale_apply_filter import ApplyFilterAndRescale
from ome_zarr_pyramid.process.filtering import custom_filters as cfilt

import zarr, warnings
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from scipy import ndimage as ndi
from skimage import transform


class _WrapperBase:
    def __init__(self,
                 scale_factor=None,
                 min_block_size=None,
                 block_overlap_sizes=None,
                 subset_indices: dict = None,
                 ### zarr parameters
                 store: str = None,
                 compressor='auto',
                 dimension_separator=None,
                 output_dtype=None,
                 overwrite=False,
                 ###
                 n_jobs = None,
                 monoresolution=False,
                 ):
        self.zarr_params = {
            # 'scale_factor': scale_factor,
            'min_block_size': min_block_size,
            'block_overlap_sizes': block_overlap_sizes,
            'subset_indices': subset_indices,
            ### zarr parameters
            'store': store,
            'compressor': compressor,
            'dimension_separator': dimension_separator,
            'dtype': output_dtype,
            'overwrite': overwrite,
            ###
            'n_jobs': n_jobs,
            'monoresolution': monoresolution,
        }
    def set(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.zarr_params.keys():
                self.zarr_params[key] = value
            else:
                raise TypeError(f"No such parameter as {key} exists.")
        return self


class Filters(_WrapperBase, ApplyFilterAndRescale): # TODO: add a project folder generator and auto-naming of temporary files. They should be auto-deletable.
    def __init__(self,
                 scale_factor=None,
                 min_block_size=None,
                 block_overlap_sizes=None,
                 subset_indices: dict = None,
                 ### zarr parameters
                 store: str = None,
                 compressor='auto',
                 dimension_separator=None,
                 output_dtype=None,
                 overwrite=False,
                 ###
                 n_jobs=None,
                 monoresolution=False,
                 ):
        _WrapperBase.__init__(self, scale_factor, min_block_size, block_overlap_sizes, subset_indices,
                              store, compressor, dimension_separator, output_dtype, overwrite, n_jobs, monoresolution)

    def __run(self,
            input: Union[str, Pyramid],
            *args,
            func,
            out: str = None,
            **kwargs
            ):
        if out is not None:
            self.set(store = out)
        if out is None:
            self.zarr_params['n_jobs'] = 1
        if isinstance(input, str):
            input = Pyramid().from_zarr(input)
        ApplyFilterAndRescale.__init__(self,
                                         input,
                                         *args,
                                         func = func,
                                         **self.zarr_params,
                                         **kwargs
                                         )
        return self.add_layers()

    def convolve(self, input, weights, mode='reflect', cval=0.0, out=None):
        return self.__run(input=input, func=ndi.convolve, weights=weights, mode=mode, cval=cval, out=out)

    def correlate(self, input, weights, mode='reflect', cval=0.0, out=None):
        return self.__run(input=input, func=ndi.correlate, weights=weights, mode=mode, cval=cval, out=out)

    def uniform_filter(self, input, size=3, mode='reflect', cval=0.0, out=None):
        return self.__run(input=input, func=ndi.uniform_filter, size=size, mode=mode, cval=cval, out=out)

    def minimum_filter(self, input, size=None, footprint=None, mode='reflect', cval=0.0, out=None):
        return self.__run(input=input, func=ndi.minimum_filter, size=size, footprint=footprint, mode=mode, cval=cval, out=out)

    def maximum_filter(self, input, size=None, footprint=None, mode='reflect', cval=0.0, out=None):
        return self.__run(input = input, func = ndi.maximum_filter, size = size, footprint = footprint, mode = mode, cval = cval, out = out)

    def mean_filter(self, input, size = None, footprint = None, out = None):
        return self.__run(input = input, func = cfilt._mean_filter, size = size, footprint = footprint, out = out)

    def median_filter(self, input, size = None, footprint = None, mode = 'reflect', cval=0.0, axes = None, out = None):
        return self.__run(input = input, func = ndi.median_filter, size = size, footprint = footprint, mode = mode, cval = cval, axes = axes, out = out)

    def percentile_filter(self, input, percentile, size=None, footprint=None, mode='reflect', cval=0.0, axes=None, out=None):
        return self.__run(input=input, func=ndi.percentile_filter, percentile=percentile, size=size,
                          footprint=footprint, mode=mode, cval=cval, axes=axes, out=out)

    def gaussian_filter(self, input, sigma, order=0, mode='reflect', cval=0.0, truncate=4.0, radius=None, axes=None,
                        out=None):
        return self.__run(input=input, func=ndi.gaussian_filter, sigma=sigma, order=order, mode=mode, cval=cval,
                          truncate=truncate, radius=radius, axes=axes, out=out)

    def gaussian_laplace(self, input, sigma, mode='reflect', cval=0.0, out=None, **kwargs):
        return self.__run(input=input, func=ndi.gaussian_laplace, sigma=sigma, mode=mode, cval=cval, out=out, **kwargs)

    def laplace(self, input, mode='reflect', cval=0.0, out=None, **kwargs):
        return self.__run(input=input, func=ndi.laplace, mode=mode, cval=cval, out=out, **kwargs)

    def prewitt(self, input, axis=-1, mode='reflect', cval=0.0, out=None):
        return self.__run(input=input, func=ndi.prewitt, axis=axis, mode=mode, cval=cval, out=out)

    def sobel(self, input, axis=-1, mode='reflect', cval=0.0, out=None):
        return self.__run(input=input, func=ndi.sobel, axis=axis, mode=mode, cval=cval, out=out)

    def generic_generic_magnitude(self, input, derivative, mode='reflect', cval=0.0, out=None, extra_arguments=(),
                                  extra_keywords=None):
        return self.__run(input=input, func=ndi.generic_gradient_magnitude, derivative=derivative, mode=mode, cval=cval,
                          out=out, extra_arguments=extra_arguments, extra_keywords=extra_keywords)

    def gaussian_gradient_magnitude(self, input, sigma, mode='reflect', cval=0.0, out=None, **kwargs):
        return self.__run(input=input, func=ndi.gaussian_gradient_magnitude, sigma=sigma, mode=mode, cval=cval, out=out,
                          **kwargs)

    def gaussian_sobel_magnitude(self, input, sigma, mode='reflect', cval=0.0, out=None, **kwargs):
        return self.__run(input=input, func=cfilt._gaussian_sobel, sigma=sigma, mode=mode, cval=cval, out=out,
                          **kwargs)

    def gaussian_prewitt_magnitude(self, input, sigma, mode='reflect', cval=0.0, out=None, **kwargs):
        return self.__run(input=input, func=cfilt._gaussian_prewitt, sigma=sigma, mode=mode, cval=cval, out=out,
                          **kwargs)


