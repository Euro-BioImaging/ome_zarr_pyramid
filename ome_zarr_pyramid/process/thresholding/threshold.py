import zarr, warnings
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from skimage import transform, filters

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.basic.basic import _WrapperBase
from ome_zarr_pyramid.process.thresholding.multiscale_apply_threshold import ApplyThresholdToPyramid, ApplyThresholdAndRescale
from ome_zarr_pyramid.process.thresholding import _global_threshold as gt, _local_threshold as lt


class _WrapperBase:
    def __init__(self,
                 scale_factor=None, ### TODO: Do we need it?
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


class Threshold(_WrapperBase, ApplyThresholdAndRescale):
    # TODO: add a project folder generator and auto-naming of temporary files. They should also be auto-deletable.
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
        self._global_threshold_params_ = {'global_thresholder': filters.threshold_multiotsu,
                                          'classes': 3,
                                          'nbins': 256,
                                          'global_threshold_level': 0}

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
        if isinstance(input, str):
            input = Pyramid().from_zarr(input)
        ApplyThresholdAndRescale.__init__(self,
                                         input,
                                         *args,
                                         func = func,
                                         **self.zarr_params,
                                         **kwargs
                                         )
        return self.add_layers()

    ### Global thresholders:
    """
       "manual_threshold",
       "threshold_otsu",
       "threshold_isodata",
       "threshold_yen",
       "threshold_multiotsu",
       "threshold_minimum",
    """
    def threshold_manual(self, input, threshold = 0, out = None):
        return self.__run(input = input, func = gt.manual_threshold, threshold = threshold, out = out)

    def threshold_otsu(self, input, nbins = 256, out = None):
        return self.__run(input = input, func = filters.threshold_otsu, nbins = nbins, out = out)

    def threshold_isodata(self, input, nbins = 256, out = None):
        return self.__run(input = input, func = filters.threshold_isodata, nbins = nbins, out = out)

    def threshold_yen(self, input, nbins = 256, out = None):
        return self.__run(input = input, func = filters.threshold_yen, nbins = nbins, out = out)

    def threshold_multiotsu(self, input, classes = 3, nbins = 256, out = None, global_threshold_level = 0):
        return self.__run(input = input, func = filters.threshold_multiotsu, classes = classes, nbins = nbins, global_threshold_level = global_threshold_level, out = out)

    def threshold_minimum(self, input, nbins = 256, max_num_iter = 10000, out = None):
        return self.__run(input = input, func = filters.threshold_minimum, nbins = nbins, max_num_iter = max_num_iter, out = out)

    # def threshold_triangle(self, input, nbins = 256, out = None):
    #     return self.__run(input = input, func = filters.threshold_triangle, nbins = nbins, out = out)

    ### Local thresholders:
    """
       "threshold_sauvola",
       "threshold_niblack",
       "niblack_bernsen",
       "sauvola_bernsen"
    """

    def threshold_niblack(self, input, window_size = (9, 9, 9), k = 0.6, r = None, out = None):
        return self.__run(input = input, func = filters.threshold_niblack, window_size = window_size, k = k, out = out)

    def threshold_sauvola(self, input, window_size = (9, 9, 9), k = 0.6, r = None, out = None):
        return self.__run(input = input, func = filters.threshold_sauvola, window_size = window_size, k = k, r = r, out = out)

    def niblack_bernsen(self, input, window_shape = (9, 9, 9), k = 0.6, min_contrast = 15, out = None, **kwargs):
        global_threshold_params = {key: kwargs[key] if key in kwargs.keys() else self._global_threshold_params_[key] for
        key in self._global_threshold_params_.keys()}
        return self.__run(input = input,
                          func = lt.niblack_bernsen,
                          window_shape = window_shape,
                          k = k, min_contrast = min_contrast,
                          out = out,
                          **global_threshold_params
                          )

    def sauvola_bernsen(self, input, window_shape = (9, 9, 9), k = 0.6, min_contrast = 15, out = None, **kwargs):
        global_threshold_params = {key: kwargs[key] if key in kwargs.keys() else self._global_threshold_params_[key] for
        key in self._global_threshold_params_.keys()}
        return self.__run(input = input,
                          func = lt.sauvola_bernsen,
                          window_shape = window_shape,
                          k = k, min_contrast = min_contrast,
                          out = out,
                          **global_threshold_params
                          )

