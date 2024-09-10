import zarr, warnings
import numpy as np
from pathlib import Path
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from scipy import ndimage as ndi
from skimage import transform

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.basic.basic import _WrapperBase
from ome_zarr_pyramid.process.filtering.multiscale_apply_filter import ApplyFilterToPyramid
from ome_zarr_pyramid.process.filtering import custom_filters as cfilt


class Filters(_WrapperBase, ApplyFilterToPyramid):
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
                 rescale_output=False,
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

        ApplyFilterToPyramid.__init__(self,
                                 input,
                                 *args,
                                 func=func,
                                 **self.zarr_params,
                                 **kwargs,
                                 scale_factor = self.scale_factor
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

