import warnings

from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.core import config, convenience as cnv
from ome_zarr_pyramid.process.parameter_control import OperationParams, get_functions_with_params
from ome_zarr_pyramid.process.operation_factory import Operation, PyramidCollection, ndfilter_wrapper
from ome_zarr_pyramid.process import process_utilities as putils, custom_operations

import numpy as np
import os, copy
import dask_image.ndfilters as ndfilters
import dask_image.ndmorph
import dask_image.dispatch
import dask_image.ndmeasure as ndmeasure
import dask
from typing import (Union, Tuple, Dict, Any, Iterable, List, Optional)


@ndfilter_wrapper('convolve')
def convolve(pyr_or_path, weights=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    pass

@ndfilter_wrapper('correlate')
def correlate(pyr_or_path, weights=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    pass

@ndfilter_wrapper('gaussian')
def gaussian(pyr_or_path, sigma=None, order=0, mode='reflect', cval=0.0, truncate=4.0, **kwargs):
    pass

@ndfilter_wrapper('gaussian_filter')
def gaussian_filter(pyr_or_path, sigma=None, order=0, mode='reflect', cval=0.0, truncate=4.0, **kwargs):
    pass

@ndfilter_wrapper('gaussian_gradient_magnitude')
def gaussian_gradient_magnitude(pyr_or_path, sigma=None, mode='reflect', cval=0.0, truncate=4.0, kwargs=None, **extra_kwargs):
    pass

@ndfilter_wrapper('gaussian_laplace')
def gaussian_laplace(pyr_or_path, sigma=None, mode='reflect', cval=0.0, truncate=4.0, kwargs=None, **extra_kwargs):
    pass

@ndfilter_wrapper('generic_filter')
def generic_filter(pyr_or_path, function=None, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, extra_arguments=(), extra_keywords={}, **kwargs):
    pass

@ndfilter_wrapper('laplace')
def laplace(pyr_or_path, mode='reflect', cval=0.0, **kwargs):
    pass

@ndfilter_wrapper('maximum_filter')
def maximum_filter(pyr_or_path, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    pass

@ndfilter_wrapper('median_filter')
def median_filter(pyr_or_path, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    pass

@ndfilter_wrapper('minimum_filter')
def minimum_filter(pyr_or_path, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    pass

@ndfilter_wrapper('percentile_filter')
def percentile_filter(pyr_or_path, percentile=None, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    pass

@ndfilter_wrapper('prewitt')
def prewitt(pyr_or_path, axis=-1, mode='reflect', cval=0.0, **kwargs):
    pass

@ndfilter_wrapper('rank_filter')
def rank_filter(pyr_or_path, rank=None, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    pass

@ndfilter_wrapper('sobel')
def sobel(pyr_or_path, axis=-1, mode='reflect', cval=0.0, **kwargs):
    pass

@ndfilter_wrapper('threshold_local')
def threshold_local(pyr_or_path, block_size=None, method='gaussian', offset=0, mode='reflect', param=None, cval=0, **kwargs):
    pass

@ndfilter_wrapper('uniform_filter')
def uniform_filter(pyr_or_path, size=3, mode='reflect', cval=0.0, origin=0, **kwargs):
    pass



# Function to run convolve
def run_convolve(input_path, output_path, weights=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    convolve(pyr_or_path=input_path, write_to=output_path, weights=weights, mode=mode, cval=cval, origin=origin, **kwargs)

# Function to run correlate
def run_correlate(input_path, output_path, weights=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    correlate(pyr_or_path=input_path, write_to=output_path, weights=weights, mode=mode, cval=cval, origin=origin, **kwargs)

# Function to run gaussian
def run_gaussian(input_path, output_path, sigma=None, order=0, mode='reflect', cval=0.0, truncate=4.0, **kwargs):
    gaussian(pyr_or_path=input_path, write_to=output_path, sigma=sigma, order=order, mode=mode, cval=cval, truncate=truncate, **kwargs)

# Function to run gaussian_filter
def run_gaussian_filter(input_path, output_path, sigma=None, order=0, mode='reflect', cval=0.0, truncate=4.0, **kwargs):
    gaussian_filter(pyr_or_path=input_path, write_to=output_path, sigma=sigma, order=order, mode=mode, cval=cval, truncate=truncate, **kwargs)

# Function to run gaussian_gradient_magnitude
def run_gaussian_gradient_magnitude(input_path, output_path, sigma=None, mode='reflect', cval=0.0, truncate=4.0, kwargs=None, **extra_kwargs):
    gaussian_gradient_magnitude(pyr_or_path=input_path, write_to=output_path, sigma=sigma, mode=mode, cval=cval, truncate=truncate, kwargs=kwargs, **extra_kwargs)

# Function to run gaussian_laplace
def run_gaussian_laplace(input_path, output_path, sigma=None, mode='reflect', cval=0.0, truncate=4.0, kwargs=None, **extra_kwargs):
    gaussian_laplace(pyr_or_path=input_path, write_to=output_path, sigma=sigma, mode=mode, cval=cval, truncate=truncate, kwargs=kwargs, **extra_kwargs)

# Function to run generic_filter
def run_generic_filter(input_path, output_path, function=None, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, extra_arguments=(), extra_keywords={}, **kwargs):
    generic_filter(pyr_or_path=input_path, write_to=output_path, function=function, size=size, footprint=footprint, mode=mode, cval=cval, origin=origin, extra_arguments=extra_arguments, extra_keywords=extra_keywords, **kwargs)

# Function to run laplace
def run_laplace(input_path, output_path, mode='reflect', cval=0.0, **kwargs):
    laplace(pyr_or_path=input_path, write_to=output_path, mode=mode, cval=cval, **kwargs)

# Function to run maximum_filter
def run_maximum_filter(input_path, output_path, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    maximum_filter(pyr_or_path=input_path, write_to=output_path, size=size, footprint=footprint, mode=mode, cval=cval, origin=origin, **kwargs)

# Function to run median_filter
def run_median_filter(input_path, output_path, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    median_filter(pyr_or_path=input_path, write_to=output_path, size=size, footprint=footprint, mode=mode, cval=cval, origin=origin, **kwargs)

# Function to run minimum_filter
def run_minimum_filter(input_path, output_path, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    minimum_filter(pyr_or_path=input_path, write_to=output_path, size=size, footprint=footprint, mode=mode, cval=cval, origin=origin, **kwargs)

# Function to run percentile_filter
def run_percentile_filter(input_path, output_path, percentile=None, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    percentile_filter(pyr_or_path=input_path, write_to=output_path, percentile=percentile, size=size, footprint=footprint, mode=mode, cval=cval, origin=origin, **kwargs)

# Function to run prewitt
def run_prewitt(input_path, output_path, axis=-1, mode='reflect', cval=0.0, **kwargs):
    prewitt(pyr_or_path=input_path, write_to=output_path, axis=axis, mode=mode, cval=cval, **kwargs)

# Function to run rank_filter
def run_rank_filter(input_path, output_path, rank=None, size=None, footprint=None, mode='reflect', cval=0.0, origin=0, **kwargs):
    rank_filter(pyr_or_path=input_path, write_to=output_path, rank=rank, size=size, footprint=footprint, mode=mode, cval=cval, origin=origin, **kwargs)

# Function to run sobel
def run_sobel(input_path, output_path, axis=-1, mode='reflect', cval=0.0, **kwargs):
    sobel(pyr_or_path=input_path, write_to=output_path, axis=axis, mode=mode, cval=cval, **kwargs)

# Function to run threshold_local
def run_threshold_local(input_path, output_path, block_size=None, method='gaussian', offset=0, mode='reflect', param=None, cval=0, **kwargs):
    threshold_local(pyr_or_path=input_path, write_to=output_path, block_size=block_size, method=method, offset=offset, mode=mode, param=param, cval=cval, **kwargs)

# Function to run uniform_filter
def run_uniform_filter(input_path, output_path, size=3, mode='reflect', cval=0.0, origin=0, **kwargs):
    uniform_filter(pyr_or_path=input_path, write_to=output_path, size=size, mode=mode, cval=cval, origin=origin, **kwargs)









