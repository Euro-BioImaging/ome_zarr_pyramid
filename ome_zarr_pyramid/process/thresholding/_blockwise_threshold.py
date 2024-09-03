import warnings, zarr, os
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from skimage import transform, filters
from skimage.util import view_as_blocks

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.core._blockwise_general import BlockwiseRunner, LazyFunction, FunctionProfiler
from ome_zarr_pyramid.process.thresholding import _local_threshold as lt, _global_threshold as gt


class _HistogramBasedProcessing:
    def _min_block(self, input_base):
        input_slc, _ = self._get_block_slicer(input_base)
        return self.input_array[input_slc].min()

    def _max_block(self, input_base):
        input_slc, _ = self._get_block_slicer(input_base)
        return self.input_array[input_slc].max()

    def _histogram_block(self, input_base, range):
        input_slc, _ = self._get_block_slicer(input_base)
        hist, _ = np.histogram(self.input_array, bins = range)
        return hist

    def max(self):
        input_slice_bases, _ = self._get_slice_bases()
        maxes = []
        for i, input_base in enumerate(input_slice_bases):
            maxes.append(self._max_block(input_base))
        return np.max(maxes)

    def min(self):
        input_slice_bases, _ = self._get_slice_bases()
        mins = []
        for i, input_base in enumerate(input_slice_bases):
            mins.append(self._min_block(input_base))
        return np.min(mins)

    def histogram(self, nbins): # TODO: parallelize it
        input_slice_bases, _ = self._get_slice_bases()
        mins, maxes = [], []
        for i, input_base in enumerate(input_slice_bases):
            mins.append(self._min_block(input_base))
            maxes.append(self._max_block(input_base))
        min = np.min(mins)
        max = np.max(maxes)
        range_ = np.linspace(min, max, nbins + 1, endpoint = True)
        binmins = range_[:-1]
        binmaxes = range_[1:]
        binmids = 0.5 * (binmins + binmaxes)
        hist = np.zeros_like(binmins).astype(np.int64)
        for i, input_base in enumerate(input_slice_bases):
            h = self._histogram_block(input_base, range_)
            hist += h
        return hist, binmids

### TODO: Maybe these functions should go to a separate utils module.
def blockwise_median(a, blockshape):
    """https://stackoverflow.com/questions/50485458/median-downsampling-in-python"""
    assert a.ndim == len(blockshape), \
        "blocks must have same dimensionality as the input image"
    # assert not (np.array(a.shape) % blockshape).any(), \
    #     "blockshape must divide cleanly into the input image shape"
    block_view = view_as_blocks(a, blockshape)
    assert block_view.shape[a.ndim:] == blockshape
    block_axes = [*range(a.ndim, 2*a.ndim)]
    return np.median(block_view, axis=block_axes)

def blockwise_min(a, blockshape):
    """https://stackoverflow.com/questions/50485458/median-downsampling-in-python"""
    assert a.ndim == len(blockshape), \
        "blocks must have same dimensionality as the input image"
    assert not (np.array(a.shape) % blockshape).any(), \
        "blockshape must divide cleanly into the input image shape"
    block_view = view_as_blocks(a, blockshape)
    assert block_view.shape[a.ndim:] == blockshape
    block_axes = [*range(a.ndim, 2*a.ndim)]
    return np.min(block_view, axis=tuple(block_axes))

def blockwise_max(a, blockshape):
    """https://stackoverflow.com/questions/50485458/median-downsampling-in-python"""
    assert a.ndim == len(blockshape), \
        "blocks must have same dimensionality as the input image"
    assert not (np.array(a.shape) % blockshape).any(), \
        "blockshape must divide cleanly into the input image shape"
    block_view = view_as_blocks(a, blockshape)
    assert block_view.shape[a.ndim:] == blockshape
    block_axes = [*range(a.ndim, 2*a.ndim)]
    return np.max(block_view, axis=tuple(block_axes))

def downscale_local_softmax(block, scale_factor):
    padder = np.sum((scale_factor, block.shape), axis = 0) % scale_factor
    padder = tuple([(0, item) for item in padder])
    padded = np.pad(block, padder)
    min = blockwise_min(padded, scale_factor)
    max = blockwise_max(padded, scale_factor)
    resized = np.where(min > 0, min, max)
    return resized
### TODO ends

class ThresholderProfiler(FunctionProfiler): ### TODO: Make this capable of distinguishing list returners and scalar returners.
    def __init__(self, func):
        FunctionProfiler.__init__(self, func)

    def try_run(self):
        func = self.func
        ################## scipy and skimage filters to be validated here ########################
        self.requires_global_threshold = False
        self.requires_local_threshold = False
        if func.__module__ in ['skimage.filters.thresholding',
                               'ome_zarr_pyramid.process.thresholding._local_threshold',
                               'ome_zarr_pyramid.process.thresholding._global_threshold',
                               ]:
            self.functype = 'passive'
            self.is_skimage_thresholder = True

            global_thresholders = ["manual_threshold",
                                   "threshold_otsu",
                                   "threshold_isodata",
                                   "threshold_yen",
                                   "threshold_multiotsu",
                                   "threshold_minimum",
                                   "threshold_triangle",
                                   "niblack_bernsen",
                                   "sauvola_bernsen"
                                   ]

            local_thresholders = ["threshold_sauvola",
                                  "threshold_niblack",
                                  "niblack_bernsen",
                                  "sauvola_bernsen"
                                   ]

            if func.__name__ in global_thresholders:
                self.requires_global_threshold = True
            if func.__name__ in local_thresholders:
                self.requires_local_threshold = True
        else:
            self.is_skimage_thresholder = False

def array_histogram(
                 input: zarr.Array,
                 nbins: int,
                 min_block_size=None,
                 subset_indices: dict = None,
                 ###
                 n_jobs = None,
                ):
    assert isinstance(input, zarr.Array)
    sub_ids = _parse_subset_indices(input, subset_indices)[input.refpath]
    blockwise = BlockwiseRunner(layer,
                                min_block_size=min_block_size,
                                subset_indices=sub_ids,
                                n_jobs = n_jobs,
                                use_synchronizer=True
                                )
    out = blockwise.histogram(nbins)
    return out

class BlockwiseThresholdRunner(BlockwiseRunner, _HistogramBasedProcessing): # TODO: KALDIM.
    """
    1: subset
    2: run_function
    3: rescale
    """
    def __init__(self,
                 input_array: zarr.Array,
                 *args,
                 scale_factor = None,
                 min_block_size = None,
                 block_overlap_sizes=None,
                 subset_indices: Tuple[Tuple[int, int]] = None,
                 ### zarr parameters for the output
                 store = None,
                 compressor = 'auto',
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
        self._nbins = 256
        if 'nbins' in kwargs.keys():
            self.nbins = kwargs.get('nbins')
        self.global_threshold = None
        self.global_thresholder = None
        self.threshold_level = None
        # try:
        if self.func.func.__class__ == ThresholderProfiler:
            if self.func.func.requires_global_threshold and not self.func.func.requires_local_threshold:
                self.global_thresholder = self.func

    @property
    def nbins(self):
        return self._nbins
    @nbins.setter
    def nbins(self, nbins):
        self._nbins = nbins

    def set_global_thresholder(self,
                               global_thresholder = filters.threshold_otsu,
                               **parameters
                               ):
        assert not (self.func.func.requires_global_threshold and not self.func.func.requires_local_threshold), \
            f"The main function is already a global thresholder. Cannot assign another one."
        if 'nbins' in parameters.keys():
            self.nbins = parameters.get('nbins')
        self.global_thresholder = LazyFunction(global_thresholder,
                                               profiler = ThresholderProfiler,
                                               **parameters
                                               )
        return self

    def set_threshold_level(self,
                            level = 0
                            ):
        self.threshold_level = level
        return self

    def _downscale_block(self, i, input_slc, output_slc): # Overrides the method from blockwise_general, replacing local softmax downscaling.
        block = downscale_local_softmax(self.input_array[input_slc], tuple(self.scale_factor)).astype(self.dtype)
        self.output[output_slc] = block
        return self.output

    def compute_global_threshold(self):
        if self.global_thresholder is None:
            self.set_global_thresholder()
        if self.func.func.requires_global_threshold:
            hist, binmids = self.histogram(self.nbins)
            runner = self.global_thresholder.get_runner(image = None,
                                                        nbins = self.nbins,
                                                        hist = (hist, binmids))
            threshold = runner()
            print(f'threshold is {threshold}')
        else:
            raise TypeError(f"The function {self.func.name} does not require a global threshold calculation.")
        self.global_threshold = threshold
        if hasattr(self.global_threshold, '__len__'):
            if self.threshold_level is None:
                warnings.warn(
                    f"A threshold class was not assigned for the multi-threshold function. Automatically choosing the lowest threshold (level 0).")
                self.set_threshold_level(0)
            self.global_threshold = self.global_threshold[self.threshold_level]
        return threshold

    def _transform_block(self, i, input_slc, output_slc, x1, x2 = None, reducer_slc = None):
        if self.func.func.requires_local_threshold and self.func.func.requires_global_threshold:
            runner = self.func.get_runner(x1[input_slc], global_threshold = self.global_threshold)
            extended_block = runner()
            block = x1[output_slc] > extended_block[reducer_slc]
        elif self.func.func.requires_local_threshold:
            runner = self.func.get_runner(x1[input_slc])
            extended_block = runner()
            block = x1[output_slc] > extended_block[reducer_slc]
        elif self.func.func.requires_global_threshold:
            block = x1[output_slc] > self.global_threshold
        else: # It should never get here!!!
            raise TypeError(f"The function {self.func.name} is not a thresholding function.")
        if np.greater(self.scale_factor, 1).any():
            block = transform.downscale_local_mean(block, tuple(self.scale_factor))
        try:
            self.output[output_slc] = block.astype(self.dtype)
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

