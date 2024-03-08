import fire
from ome_zarr_pyramid.process import image_filters as imfilt

def _SeparateFlagArgs(args):
    try:
        index = args.index('--help')
        args = args[:index]
        index = args.index('-h')
        args = args[:index]
        return args, ['--help', '-h']
    except ValueError:
        return args, []

fire.core.parser.SeparateFlagArgs = _SeparateFlagArgs

def convolve():
    _ = fire.Fire(imfilt.run_convolve)
    return

def correlate():
    _ = fire.Fire(imfilt.run_correlate)
    return

def gaussian():
    _ = fire.Fire(imfilt.run_gaussian)
    return

def gaussian_filter():
    _ = fire.Fire(imfilt.run_gaussian_filter)
    return

def gaussian_gradient_magnitude():
    _ = fire.Fire(imfilt.run_gaussian_gradient_magnitude)
    return

def gaussian_laplace():
    _ = fire.Fire(imfilt.run_gaussian_laplace)
    return

def generic_filter():
    _ = fire.Fire(imfilt.run_generic_filter)
    return

def laplace():
    _ = fire.Fire(imfilt.run_laplace)
    return

def maximum_filter():
    _ = fire.Fire(imfilt.run_maximum_filter)
    return

def median_filter():
    _ = fire.Fire(imfilt.run_median_filter)
    return

def minimum_filter():
    _ = fire.Fire(imfilt.run_minimum_filter)
    return

def percentile_filter():
    _ = fire.Fire(imfilt.run_percentile_filter)
    return

def prewitt():
    _ = fire.Fire(imfilt.run_prewitt)
    return

def rank_filter():
    _ = fire.Fire(imfilt.run_rank_filter)
    return

def sobel():
    _ = fire.Fire(imfilt.run_sobel)
    return

def threshold_local():
    _ = fire.Fire(imfilt.run_threshold_local)
    return

def uniform_filter():
    _ = fire.Fire(imfilt.run_uniform_filter)
    return

