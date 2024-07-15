# TODO: Change the module name to ndimage or nd_manipulate and then add the classes Aggregative and Dispersive both here.
# TODO: Maybe change the directory (not the module) to ndimage that should contain the modules aggregative and dispersive.

import zarr, warnings, glob
import numpy as np
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )

from ome_zarr_pyramid.core.pyramid import Pyramid, PyramidCollection
from ome_zarr_pyramid.process.aggregative.multiscale_apply_aggregative import ApplyAggregativeToPyramid, ApplyAggregativeAndRescale
from ome_zarr_pyramid.process.aggregative import _aggregative_functions as aggf


class _WrapperBase:
    def __init__(self,
                 scale_factor=None, ### TODO: Do we need it?
                 min_block_size=None,
                 block_overlap_sizes=None,
                 subset_indices: dict = None,
                 ### zarr parameters
                 store: str = None,  # output group store path # TODO: add a project folder parameter; 'out' then becomes the basename for the output
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


class Aggregative(_WrapperBase, ApplyAggregativeAndRescale):
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
        if isinstance(input, str): # TODO: make the glob pattern an optional parameter? Support csv?
            paths = glob.glob(os.path.join(input, '*'))
            pyrs = []
            for fpath in paths:
                try:
                    pyrs.append(Pyramid().from_zarr(fpath))
                except:
                    pass
        elif isinstance(input, (list, tuple)):
            pyrs = input
        else:
            raise TypeError(f"The input must be either of types: list, tuple, str]")
        ApplyAggregativeAndRescale.__init__(self,
                                             pyrs,
                                             *args,
                                             func = func,
                                             **self.zarr_params,
                                             **kwargs
                                             )
        return self.add_layers()

    def concatenate(self, input, axis = 'z', out = None):
        return self.__run(input = input, func = aggf.concatenate_zarrs, axis = axis, out = out)
