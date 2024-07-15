import warnings, zarr, os, copy, multiprocessing
from joblib import Parallel, delayed, parallel_backend
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from scipy import ndimage as ndi
from skimage import transform, morphology
from skimage.util import view_as_blocks

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process import process_utilities as putils
from ome_zarr_pyramid.process.core._blockwise_general import BlockwiseRunner, LazyFunction, FunctionProfiler
from ome_zarr_pyramid.process.thresholding import _local_threshold as lt


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

class LabelProfiler(FunctionProfiler):
    def __init__(self, func):
        FunctionProfiler.__init__(self, func)

    def try_run(self):
        ################## scipy and skimage filters to be validated here ########################
        self.functype = 'passive'
        self.is_label_filter = True

def label_propagate(block, func = np.min):
    dilated = block.astype(int).copy()
    mask = dilated > 0
    relabeled, maxlbl = ndi.label(mask, structure = np.ones([3] * dilated.ndim))
    indices = np.arange(1, maxlbl + 1)
    objects = ndi.find_objects(relabeled)
    ret = dilated.astype(int)
    assert len(objects) == indices.size
    for idx, slc in zip(indices, objects):
        newmask = relabeled[slc] == idx
        oldblock = ret[slc]
        newlbl = func(oldblock[newmask])
        ret[slc] = np.where(newmask, newlbl, oldblock)
    is_merged = np.any(np.abs(ret - dilated))
    return ret, is_merged

def min_label_propagate(block):
    return label_propagate(block, np.min)

class BlockwiseLabelRunner(BlockwiseRunner):
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
        if 'max_iter' in kwargs.keys():
            self.max_iter = kwargs.get('max_iter')
        else:
            self.max_iter = 100

    def passive_func(self, block):
        return block

    def init_labels(self):
        directions = ['forwards', 'backwards']
        input_slice_bases, output_slice_bases = self._get_slice_bases(directions[0])
        maxlbl = 0
        for input_base, output_base in zip(input_slice_bases, output_slice_bases):
            input_slc, reducer_slc = self._get_block_slicer(input_base)
            block = self.input_array[input_slc] > 0
            lbld, _ = ndi.label(block, structure=np.ones([3] * block.ndim))
            lbld = np.where(block, lbld + maxlbl, 0)
            self.input_array[input_slc] = lbld
            newmax = np.max(lbld)
            if newmax > maxlbl:
                maxlbl = newmax
        return self.input_array

    def _transform_block(self,
                         i,
                         input_slc,
                         x1,
                         x2,
                         relocate
                         ):
        output_slc = input_slc
        if relocate:
            runner = self.func.get_runner(x1[input_slc])
            extended_block = runner()
            inner_iter = 1
        else:
            lazy = LazyFunction(func = min_label_propagate,
                                  block = x1[input_slc],
                                  )
            runner = lazy.get_runner()
            processed = runner()
            extended_block, inner_iter = processed
        block = extended_block

        if np.greater(self.scale_factor, 1).any():
            block = transform.downscale_local_mean(block, tuple(self.scale_factor))

        if inner_iter > 0:
            self.output.attrs['mergeless_pass'] = False
        else:
            pass

        self.output[output_slc] = block.astype(self.dtype)
        return self.output

    def _downscale_block(self, i, input_slc, output_slc): # Overrides the method from blockwise_general, replacing local softmax downscaling.
        block = downscale_local_softmax(self.input_array[input_slc], tuple(self.scale_factor)).astype(self.dtype)
        self.output[output_slc] = block
        return self.output

    def write_binary(self,
                     sequential = True,
                     relocate = False,
                     repetitions = 100,
                     switch_slicing_direction = True,
                     only_downscale = False
                     ):
        """Note the replacement of the downscaling method."""
        directions = ['forwards', 'backwards']
        self._create_slices(directions[0])
        if only_downscale:
            if self.block_overlap_sizes is not None:
                if np.any(np.array(self.block_overlap_sizes) != 0):
                    warnings.warn(f"The 'only_downscale' mode requires None for block overlap sizes.\nUpdating the 'block_overlap_sizes' property to None.")
                    self.block_overlap_sizes = None
                else: ### all overlap sizes are 0, which is expected.
                    pass
            if sequential:
                _ = [self._downscale_block(i, input_slc, output_slc)
                             for i, (input_slc, output_slc) in enumerate(zip(self.input_slices, self.output_slices))]
            else:
                with parallel_backend('multiprocessing'):
                    with Parallel(n_jobs=n_jobs, require=self.require_sharedmem) as parallel:
                                _ = parallel(
                                    delayed(self._downscale_block)(i,
                                                                   input_slc,
                                                                   output_slc
                                                                   )
                                    for i, (input_slc, output_slc) in enumerate(zip(self.input_slices, self.output_slices)))
        else:
            if relocate:
                x1 = self.input_array
            else:
                x1 = self.init_labels()
            try:
                x2 = self.func.parsed[1]
            except:
                x2 = None

            if relocate:
                repetitions = 1

            for rep in range(repetitions):
                self.mergeless_pass = True
                if sequential:
                    _ = [self._transform_block(i, input_slc, x1, x2, relocate)
                                 for i, input_slc in enumerate(self.input_slices)]
                    if switch_slicing_direction:
                        directions = directions[::-1]
                        self._create_slices(directions[0])
                else:
                    with parallel_backend('multiprocessing'):
                        with Parallel(
                                      n_jobs=self.n_jobs,
                                      require=self.require_sharedmem
                                      ) as parallel:
                            _ = parallel(
                                delayed(self._transform_block)(i,
                                                               input_slc,
                                                               # output_base,
                                                               x1,
                                                               x2,
                                                               relocate
                                                               )
                                for i, input_slc in enumerate(self.input_slices))

                output = zarr.open(self.store)
                if 'mergeless_pass' in output.attrs.keys():
                    if not output.attrs['mergeless_pass']:
                        self.mergeless_pass = False
                    output.attrs.pop('mergeless_pass')
                if not relocate:
                    print(f"Iteration {rep}. Mergeless pass: {self.mergeless_pass}")
                    pass
                if self.mergeless_pass:
                    break
        return self.output

