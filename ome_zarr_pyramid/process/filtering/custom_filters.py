import numpy as np
from scipy import ndimage as ndi
from ome_zarr_pyramid.process.filtering import statistical as st

def _get_footprint_from_size(size, input = None):
    if np.isscalar(size):
        size = tuple([size] * input.ndim)
    fp = np.ones(size)
    return fp

### Additional filters along with scipy.ndimage filters
def _gaussian(arr, sigma, mode, cval, **kwargs):
    if np.any(np.array(sigma) < 0) or sigma is None:
        g_arr = arr
    else:
        g_arr = ndi.gaussian_filter(arr, sigma, mode = mode, cval = cval, **kwargs)
    return g_arr

def _gaussian_sobel(arr, sigma, mode = 'reflect', cval = 4., **kwargs):
    g_arr = _gaussian(arr, sigma, mode, cval, **kwargs)
    return ndi.generic_gradient_magnitude(g_arr, ndi.sobel, mode = mode, cval = cval)

def _gaussian_prewitt(arr, sigma, mode = 'reflect', cval = 4., **kwargs):
    g_arr = _gaussian(arr, sigma, mode, cval, **kwargs)
    return ndi.generic_gradient_magnitude(g_arr, ndi.prewitt, mode = mode, cval = cval)

def _mean_filter(arr, footprint = None):
    return st.mean_filter(arr, fp = footprint)
