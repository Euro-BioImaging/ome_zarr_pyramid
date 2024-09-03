import zarr
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.core._blockwise_general import LazyFunction
from ome_zarr_pyramid.process.aggregative._blockwise_aggregative import BlockwiseAggregativeRunner, AggregativeProfiler
from ome_zarr_pyramid.process.core.multiscale_apply_general import ApplyToPyramid, ApplyAndRescale, \
    _parse_subset_indices, _parse_args, _parse_axes_for_directional


class ApplyAggregativeToPyramid(ApplyToPyramid):
    def __init__(self,
                 input: List[Pyramid],
                 *args,
                 min_block_size=None,
                 block_overlap_sizes=None,
                 subset_indices: dict = None,
                 ### zarr parameters
                 store: str = None,
                 compressor='auto',
                 dimension_separator=None,
                 dtype=None,
                 overwrite=False,
                 ###
                 func=None,
                 n_jobs=None,
                 monoresolution=False,
                 **kwargs
                 ):

        input = PyramidCollection(input)
        super().__init__(
                        input = input,
                        *args,
                        min_block_size = min_block_size,
                        block_overlap_sizes = block_overlap_sizes,
                        subset_indices = subset_indices,
                        store = store,
                        compressor = compressor,
                        dimension_separator = dimension_separator,
                        dtype = dtype,
                        overwrite = overwrite,
                        func = func,
                        n_jobs = n_jobs,
                        monoresolution = monoresolution,
                        **kwargs
                        )

    def set_function(self, func):
        self.profiler = AggregativeProfiler(func)
        self.lazyfunc = LazyFunction(func, profiler=AggregativeProfiler)

    def set_runner(self, runner = None):
        if runner is None:
            runner = BlockwiseAggregativeRunner
        self.runner = runner


class ApplyAggregativeAndRescale(ApplyAndRescale):
    def __init__(self,
                 input: List[Pyramid],
                 *args,
                 min_block_size=None,
                 block_overlap_sizes=None,
                 subset_indices: dict = None,
                 ### zarr parameters
                 store: str = None,
                 compressor='auto',
                 dimension_separator=None,
                 dtype=None,
                 overwrite=False,
                 ###
                 func=None,
                 n_jobs=None,
                 monoresolution=False,
                 **kwargs
                 ):

        input = PyramidCollection(input)
        super().__init__(
                        input = input,
                        *args,
                        min_block_size = min_block_size,
                        block_overlap_sizes = block_overlap_sizes,
                        subset_indices = subset_indices,
                        store = store,
                        compressor = compressor,
                        dimension_separator = dimension_separator,
                        dtype = dtype,
                        overwrite = overwrite,
                        func = func,
                        n_jobs = n_jobs,
                        monoresolution = monoresolution,
                        **kwargs
                        )

    def set_function(self, func):  # overriding with FilterProfiler
        self.profiler = AggregativeProfiler(func)
        self.lazyfunc = LazyFunction(func, profiler=AggregativeProfiler)

    def set_runner(self, runner = None):
        if runner is None:
            runner = BlockwiseAggregativeRunner
        self.runner = runner

