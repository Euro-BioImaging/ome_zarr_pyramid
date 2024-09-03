import warnings, time, shutil, tempfile, os, copy, inspect, zarr
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
import numcodecs; numcodecs.blosc.use_threads = False

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.core._blockwise_general import BlockwiseRunner, LazyFunction, FunctionProfiler

class FilterProfiler(FunctionProfiler):
    def __init__(self, func):
        FunctionProfiler.__init__(self, func)

    def try_run(self):
        func = self.func
        ################## scipy and skimage filters to be validated here ########################
        if func.__module__ in ['scipy.ndimage._filters',
                               'ome_zarr_pyramid.process.filtering.custom_filters',
                               'ome_zarr_pyramid.process.filtering.statistical'
                               ]:
            self.functype = 'passive'
            self.is_scipy_ndimage_filter = True
        else:
            self.is_scipy_ndimage_filter = False
            warnings.warn(f"The module is not scipy ndimage")
            warnings.warn(f"The function type could not be determined!")


class BlockwiseFilterRunner(BlockwiseRunner):
    def __init__(self,
                 input_array: zarr.Array,
                 *args,
                 scale_factor=None,
                 min_block_size=None,
                 block_overlap_sizes=None,
                 subset_indices: Tuple[Tuple[int, int]] = None,
                 ### zarr parameters for the output
                 store=None,
                 compressor='auto',
                 dimension_separator='/',
                 dtype=None,
                 overwrite=False,
                 ###
                 func=None,
                 n_jobs=None,
                 use_synchronizer='multiprocessing',
                 pyramidal_syncdir=os.path.expanduser('~') + '/.syncdir',
                 require_sharedmem=False,
                 **kwargs
                 ):
        BlockwiseRunner.__init__(self,
                                 input_array=input_array,
                                 *args,
                                 scale_factor=scale_factor,
                                 min_block_size=min_block_size,
                                 block_overlap_sizes=block_overlap_sizes,
                                 subset_indices=subset_indices,
                                 ### zarr parameters for the output
                                 store=store,
                                 compressor=compressor,
                                 dimension_separator=dimension_separator,
                                 dtype=dtype,
                                 overwrite=overwrite,
                                 ###
                                 func=func,
                                 n_jobs=n_jobs,
                                 use_synchronizer=use_synchronizer,
                                 pyramidal_syncdir=pyramidal_syncdir,
                                 require_sharedmem=require_sharedmem,
                                 **kwargs
                                 )
