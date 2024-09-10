import zarr, warnings
import numpy as np
from pathlib import Path
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from scipy import ndimage as ndi
from skimage import transform, filters

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.basic.basic import _WrapperBase
from ome_zarr_pyramid.process.morphology.multiscale_apply_label import ApplyLabelToPyramid, ApplyLabelAndRescale
from ome_zarr_pyramid.utils import multiscale_utils as mutils


class Label(_WrapperBase, ApplyLabelToPyramid):
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
                 rescale_output=True, # for compatibility
                 select_layers='all',
                 n_scales=None
                 ):
        _WrapperBase.__init__(
                              self, scale_factor, min_block_size, block_overlap_sizes, input_subset_indices,
                              output_store, output_compressor, output_chunks, output_dimension_separator,
                              output_dtype, overwrite, n_jobs, rescale_output, select_layers, n_scales
                              )

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
        if isinstance(input, (str, Path)):
            input = Pyramid().from_zarr(input)
        ApplyLabelToPyramid.__init__(self,
                                 input,
                                 *args,
                                 func=func,
                                 **self.zarr_params,
                                 **kwargs,
                                 scale_factor = self.scale_factor
                                 )
        self.set_downscaler(mutils.downscale_local_softmax)
        return self.add_layers()

    def label(self, input, out = None):
        if not self.zarr_params['rescale_output']:
            self.set(rescale_output = True)
            warnings.warn(f"The 'label' method requires the output to be rescaled.")
            warnings.warn(f"The rescale_output parameter is assumed True.")
        if self.scale_factor is None:
            if input.nlayers > 1:
                self.scale_factor = input.scale_factors[input.resolution_paths[1]]
        return self.__run(input = input, out = out)

    def extract_label(self, input, label_idx, out = None):
        NotImplementedError(f"This method is not yet implemented.")

