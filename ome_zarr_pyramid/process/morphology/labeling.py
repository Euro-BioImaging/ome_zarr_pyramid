import zarr, warnings
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from scipy import ndimage as ndi
from skimage import transform, filters

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.basic.basic import _WrapperBase
from ome_zarr_pyramid.process.morphology.multiscale_apply_label import ApplyLabelToPyramid, ApplyLabelAndRescale



class Label(_WrapperBase, ApplyLabelAndRescale):
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
                 n_jobs = None,
                 select_layers='all'
                 ):
        _WrapperBase.__init__(self, scale_factor, min_block_size, block_overlap_sizes, input_subset_indices,
                              output_store, output_compressor, output_chunks, output_dimension_separator,
                              output_dtype, overwrite, n_jobs, select_layers)

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

