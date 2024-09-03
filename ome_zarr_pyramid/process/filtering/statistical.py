""" Here I accummulate some tools based on scipy.ndimage to calculate
    the basic local statistics of multi-dimensional arrays. """

import numpy as np
from scipy import ndimage as ndi


# from scipy.signal import fftconvolve as convolve


def _block_zeroes(im):
    """ Replaces the zeros in an array with a very low positive value. """
    im = im.astype(float).copy()
    eps = np.finfo(im.dtype).eps
    im[im == 0] = eps
    return im


def _get_footprint(im, fp):
    """ A shortcut function to get an input structuring element.

        Parameters:
        -----------
        im: array
            n-dimensional numpy array.
        fp: scalar or iterable
            Footprint defining local neighbourhood. If a scalar, it represents the single dimension of
            a box-shaped neighbourhood with uniform edges. If a tuple or list, then it represents the edges of a
            box-shaped neighbourhood. If an array, it is assumed as a pre-computed footprint, i.e., it should
            have the same number of dimensions as 'im'.

        Returns:
        --------
        filtered: array
            n-dimensional numpy array with the same shape as im.
        """
    if np.isscalar(fp):
        fp = np.ones([fp] * im.ndim)
    elif not hasattr(fp, 'nonzero'):
        fp = np.ones(fp)
        assert im.ndim == fp.ndim
    elif hasattr(fp, 'nonzero'):
        assert im.ndim == fp.ndim
    return fp


def sum_filter(im, fp):
    """ Replaces every voxel value with the sum of the voxels within its local neighbourhood. """
    im = _block_zeroes(im)
    fp = _get_footprint(im, fp)
    summed = ndi.correlate(im, fp, mode='reflect')
    return summed


def mean_filter(im, fp):
    """ Replaces every voxel value with the mean of the voxels within its local neighbourhood. """
    im = _block_zeroes(im)
    fp = _get_footprint(im, fp)
    summed = ndi.correlate(im, fp, mode='reflect')
    filtered = summed / fp.sum()
    return filtered


def median_filter(img, fp):
    """ Scipy's median filter adapted to be consistent with other functions in this module. """
    im = _block_zeroes(img)
    fp = _get_footprint(im, fp)
    return ndi.median_filter(im, footprint=fp)


def variance_filter(im, fp):
    """ Replaces every voxel value with the variance of the voxels within its local neighbourhood. """
    im = _block_zeroes(im)
    fp = _get_footprint(im, fp)
    avg = mean_filter(im, fp)
    diff = im - avg
    var = mean_filter(np.square(diff), fp)
    return var


def std_filter(im, fp):
    """ Replaces every voxel value with the standard deviation of the voxels within its local neighbourhood. """
    im = _block_zeroes(im)
    fp = _get_footprint(im, fp)
    var = variance_filter(im, fp)
    return np.sqrt(var)


class basic_local_statistics:
    """ Calculates and stores the local mean, variance and standard deviation. """

    def __init__(self, im, fp):
        im = _block_zeroes(im)
        fp = _get_footprint(im, fp)
        avg = mean_filter(im, fp)
        diff = im - avg
        var = mean_filter(np.square(diff), fp)
        stdev = np.sqrt(var)
        self.im = im
        self.mean = avg
        self.var = var
        self.stdev = stdev


def iqr_filter(im, fp, p0=25, p1=75):
    """ Replaces every voxel value with the interquartile range of the voxels within its local neighbourhood. """
    im = _block_zeroes(im)
    fp = _get_footprint(im, fp)
    p0, p1 = np.sort((p0, p1))
    low = ndi.percentile_filter(im, p0, footprint=fp)
    high = ndi.percentile_filter(im, p1, footprint=fp)
    return high - low


def contrast(im, fp):
    """ Replaces every voxel value with the contrast (maximum - minimum) of the voxels within its local neighbourhood. """
    im = _block_zeroes(im)
    fp = _get_footprint(im, fp)
    maxis = ndi.maximum_filter(im, footprint=fp)
    minis = ndi.minimum_filter(im, footprint=fp)
    return maxis - minis


def midpoint(im, fp):
    """ Replaces every voxel value with the midpoint ((maximum + minimum) / 2) of the voxels within its local neighbourhood. """
    im = _block_zeroes(im)
    fp = _get_footprint(im, fp)
    maxis = ndi.maximum_filter(im, footprint=fp)
    minis = ndi.minimum_filter(im, footprint=fp)
    return 0.5 * (minis + maxis)


class local_minmax_statistics:
    def __init__(self, im, fp):
        """ Calculates and stores the local minimum, maximum, midpoint and contrast. """
        im = _block_zeroes(im)
        fp = _get_footprint(im, fp)
        maxis = ndi.maximum_filter(im, footprint=fp)
        minis = ndi.minimum_filter(im, footprint=fp)
        self.maxis = maxis
        self.minis = minis
        self.midpoint = 0.5 * (maxis + minis)
        self.contrast = maxis - minis


wmid = lambda x, y, a, b: (x * a + y * b) / (a + b)  ### a shortcut for weighted midpoint


def wm_filter(im, fp, maxc=1, minc=1):
    """ Replaces every voxel value with the weighted midpoint of the voxels within its local neighbourhood. """
    if np.isscalar(fp):
        fp = np.ones([fp] * im.ndim)
    elif not hasattr(fp, 'nonzero'):
        fp = np.ones(fp)
    maxis = ndi.maximum_filter(im, footprint=fp)
    minis = ndi.minimum_filter(im, footprint=fp)
    return wmid(maxis, minis, maxc, minc)


def nonparametric_skewness_filter(img, shape):
    """ Replaces every voxel value with the skewness of the voxels within its local neighbourhood.
        This skewness calculation is based on the local medians: (https://en.wikipedia.org/wiki/Nonparametric_skew). """
    med = ndi.median_filter(img, shape)
    bls = basic_local_statistics(img, shape)
    avg = bls.mean
    stdev = bls.stdev
    skew = (avg - med) / stdev
    return np.nan_to_num(skew)


def pearson_skewness_filter(img, shape):
    """ Replaces every voxel value with the skewness of the voxels within its local neighbourhood.
        This skewness calculation is based on the Pearson's method: https://en.wikipedia.org/wiki/Skewness#Pearson's_second_skewness_coefficient_(median_skewness). """
    return 3 * nonparametric_skewness_filter(img, shape)

