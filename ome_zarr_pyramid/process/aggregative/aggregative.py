# TODO: Change the module name to ndimage or nd_manipulate and then add the classes Aggregative and Dispersive both here.
# TODO: Maybe change the directory (not the module) to ndimage that should contain the modules aggregative and dispersive.

import zarr, warnings, glob, os
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )
from pathlib import Path

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.basic.basic import _WrapperBase
from ome_zarr_pyramid.process.aggregative.multiscale_apply_aggregative import ApplyAggregativeToPyramid, ApplyAggregativeAndRescale
from ome_zarr_pyramid.process.aggregative import _aggregative_functions as aggf
from ome_zarr_pyramid.creation.pyramid_creation import PyramidCreator
from ome_zarr_pyramid.utils import assignment_utils as asutils


def block_zarrs(data: Union[Pyramid], # consists of pyramids ### TODO: kaldim
                n_jobs = 1,
                out = None
                ):
    final_shape, block_slices = aggf._calculate_block_slices(data)
    ### get the metadata from the first pyramid
    refpyr, _ = block_slices[0]
    refpath = refpyr.refpath
    axis_order = refpyr.axis_order
    res = PyramidCreator(final_shape,
                         axis_order = axis_order,
                         unit_list = refpyr.unit_list,
                         store = out,
                         scale = refpyr.scales[refpath],
                         n_resolutions = 1,
                         scale_factor = refpyr.scale_factors,
                         dtype = refpyr.dtype,
                         chunks = refpyr.chunks,
                         compressor = refpyr.compressor,
                         synchronizer = refpyr.synchronizer,
                         dimension_separator = refpyr.dimension_separator,
                         n_jobs = n_jobs,
                         ).create()
    refarr = res.refarray
    for pyr, slcdict in block_slices:
        slc = tuple([slcdict[ax] for ax in axis_order])
        slcarr = np.array([(sl.start, sl.stop) for sl in slc])
        refarr = asutils.basic_assign(dest=refarr,
                                      source=pyr.refarray,
                                      dest_slice=slcarr
                                      )
    res.add_layer(refarr, pth=res.refpath, scale=res.scales[res.refpath], overwrite=True)
    return



# topdir = Path(f"/home/oezdemir/PycharmProjects/dask_distributed/ome_zarr_pyramid/fakedata")
# arrs0 = Pyramid().from_zarr(topdir/'arr0')
# arrs1 = Pyramid().from_zarr(topdir/'arr1')
# arrs2 = Pyramid().from_zarr(topdir/'arr2')
# arrs3 = Pyramid().from_zarr(topdir/'arr3')
#
# arrs = {'x': [arrs3, {'y': [arrs0, {'y': [arrs1, arrs2]}]}]}
# # axis_order = arrs0.axis_order
#
# out = block_zarrs(arrs, n_jobs = 8, out = topdir/'final.zarr')


class Aggregative(_WrapperBase, ApplyAggregativeToPyramid): # TODO: integrate the block_zarrs function above to this class.
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
                 n_jobs = 1,
                 select_layers='all'
                 ):
        _WrapperBase.__init__(self, scale_factor, min_block_size, block_overlap_sizes, input_subset_indices,
                              output_store, output_compressor, output_chunks, output_dimension_separator,
                              output_dtype, overwrite, n_jobs, select_layers)

    def _read_multiple_paths(self): #TODO
        pass

    def __run(self,
            input: Union[str, tuple, list],
            *args,
            func,
            out: str = '',
            **kwargs
            ):
        if out != '':
            self.set(store = out)
        if out is None:
            self.zarr_params['n_jobs'] = 1
        if isinstance(input, (str, Path)): # TODO: make the glob pattern an optional parameter? Support csv?
            paths = sorted(Path(input).glob('*'))
            pyrs = []
            for fpath in paths:
                try:
                    pyrs.append(Pyramid().from_zarr(fpath))
                except:
                    warnings.warn(f"The following path could not be read as a Pyramid: {fpath}")
        elif isinstance(input, (list, tuple)):
            pyrs = input
        else:
            raise TypeError(f"The input must be either of types: list, tuple, str]")
        ApplyAggregativeToPyramid.__init__(self,
                                             pyrs,
                                             *args,
                                             func = func,
                                             **self.zarr_params,
                                             **kwargs
                                             )
        return self.add_layers()

    def concatenate(self, input, axis = 'z', out = None):
        return self.__run(input = input, func = aggf.concatenate_zarrs, axis = axis, out = out)

# path = f"/home/oezdemir/Desktop/garbage/yannick/unary/F227_WT2wk_run2_10nm_aligned_cropped"
# out = f"/home/oezdemir/Desktop/garbage/yannick/concat/F227_WT2wk_run2_10nm_aligned_cropped.zarr"
# agg = Aggregative(
#                   output_chunks = (1, 1, 150, 500, 500),
#                   n_jobs = 14,
#                   overwrite = True
#                   )
# agg.concatenate(path, out = out)


