import zarr, warnings
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from scipy import ndimage as ndi
from skimage import transform, filters

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.morphology.multiscale_apply_label import ApplyLabelToPyramid, ApplyLabelAndRescale


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


class Label(_WrapperBase, ApplyLabelAndRescale):
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
            func = None, # For compatibility
            out: str = '',
            **kwargs
            ):
        if out != '':
            self.set(store = out)
        if out is None:
            self.zarr_params['n_jobs'] = 1
        if isinstance(input, str):
            input = Pyramid().from_zarr(input)
        ApplyLabelAndRescale.__init__(self,
                                         input,
                                         *args,
                                         # func = func,
                                         **self.zarr_params,
                                         **kwargs
                                         )
        return self.add_layers()

    def label(self, input, out = None):
        return self.__run(input = input, out = out)

    def extract_label(self, input, label_idx, out = None):
        """
        TODO: extract a specific label object to separate OME-Zarr
        Loop through chunks. Get chunks containing the specific label
        """
        pass

