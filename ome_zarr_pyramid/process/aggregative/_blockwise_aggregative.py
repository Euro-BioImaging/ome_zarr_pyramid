import zarr, os
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process import process_utilities as putils
from ome_zarr_pyramid.process.core._blockwise_general import BlockwiseRunner, LazyFunction, FunctionProfiler
from ome_zarr_pyramid.process.aggregative import _aggregative_functions as agg

class AggregativeProfiler(FunctionProfiler):
    def __init__(self, func):
        FunctionProfiler.__init__(self, func)
        self.sample = [zarr.zeros((2, 3))] * 3
        self.try_run()
    def try_run(self):
        self.functype = 'expansive'


class BlockwiseAggregativeRunner(BlockwiseRunner):
    def __init__(self,
                 input_collection: List[zarr.Array],
                 *args,
                 scale_factor = None,
                 min_block_size = None,
                 block_overlap_sizes=None,
                 subset_indices: Tuple[Tuple[int, int]] = None,
                 ### zarr parameters for the output
                 store = None,
                 compressor = 'auto',
                 chunks = None,
                 dimension_separator = '/',
                 dtype = None,
                 overwrite = False,
                 ###
                 func = None,
                 n_jobs = None,
                 use_synchronizer = 'multiprocessing',
                 pyramidal_syncdir = os.path.expanduser('~') + '/.syncdir',
                 require_sharedmem = False,
                 **kwargs
                 ):
        if isinstance(input_collection, (tuple, list)):
            input_array = input_collection[0]
        else:
            input_array = input_collection
        self.input_collection = input_collection
        BlockwiseRunner.__init__(self,
                                 input_array = input_array,
                                 *args,
                                 scale_factor = scale_factor,
                                 min_block_size = min_block_size,
                                 block_overlap_sizes=block_overlap_sizes,
                                 subset_indices = subset_indices,
                                 ### zarr parameters for the output
                                 store = store,
                                 compressor = compressor,
                                 chunks = chunks,
                                 dimension_separator = dimension_separator,
                                 dtype = dtype,
                                 overwrite = overwrite,
                                 ###
                                 func = func,
                                 n_jobs = n_jobs,
                                 use_synchronizer = use_synchronizer,
                                 pyramidal_syncdir = pyramidal_syncdir,
                                 require_sharedmem = require_sharedmem,
                                 **kwargs
                                 )
        self.shapes = agg._get_shapes_from_arraylist(self.input_collection)
        if 'axis' in self.func.parsed.keys():
            axis = self.func.parsed['axis'][0][0]
        if self.func.name in ['concatenate_zarrs']:
            self._output_shape = agg._get_final_shape_for_concatenation(self.shapes, axis=axis)
        self.func.parse_params(arraylist = self.input_collection)

    def _create_slices(self,
                       slicing_direction = 'forwards'
                       ):
        if self.func.name in ['concatenate_zarrs']:
            in_slcs, out_slcs = [], []
            start = 0
            axis = self.func.parsed['axis'][0][0]
            for shape in self.shapes:
                stop = start + shape[axis]
                in_slcs_ = [slice(None, None)] * len(shape)
                out_slcs_ = [slice(None, None)] * len(shape)
                out_slcs_[axis] = slice(start, stop)
                in_slcs.append(tuple(in_slcs_))
                out_slcs.append(tuple(out_slcs_))
                start = stop
            self.input_slices = in_slcs
            self.output_slices = out_slcs
            self.reducer_slices = out_slcs
        else:
            super()._create_slices(slicing_direction)

    def _transform_block(self, i, input_slc, output_slc, x1, x2 = None,
                         reducer_slc = None):
        block = self.input_collection[i]
        try:
            self.output[output_slc] = block
        except:
            print('#')
            print(f"Block no {i}")
            print(input_slc)
            print(output_slc)
            print(f"Input block shape: {self.input_array[input_slc].shape}")
            print(f"Current output block shape: {block.shape}")
            print(f"Expected output block shape: {self.output[output_slc].shape}")
            print(f"Error at block no {i}")
            print('###')
        return self.output






















