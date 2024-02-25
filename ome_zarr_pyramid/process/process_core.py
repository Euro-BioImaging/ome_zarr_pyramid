# from ome_zarr_pyramid.core.Hierarchy import OMEZarr, MultiScales
import warnings

from ome_zarr_pyramid.core.pyramid import Pyramid
from ome_zarr_pyramid.process import process_utilities as putils
from ome_zarr_pyramid.process.parameter_control import FilterParams, BaseParams

import itertools
import zarr
import numpy as np
import os, copy
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )

class BaseProtocol:
    def __init__(self,
                 input: Pyramid = None,
                 resolutions: list[str] = None,
                 drop_singlet_axes: bool = True,
                 output_name: str = 'nomen',
                 **kwargs
                 ):
        self.set_input(input, resolutions, drop_singlet_axes, output_name)
        # super().__setattr__("base_params", base_params)
        # self.run_cycle(**base_params._params, **kwargs)
    @property
    def input(self):
        return self._base_params.input
    def set_input(self,
                  input: Pyramid = None,
                  resolutions: list[str] = None,
                  drop_singlet_axes: bool = True,
                  output_name: str = 'nomen'
                  ):
        base_params = BaseParams(input, resolutions, drop_singlet_axes, output_name)
        super().__setattr__("_base_params", base_params)
    def run(self, **kwargs):
        super().__setattr__("output", self.input.copy())
        print(f"No function selected.")
    def get_output(self):
        if self._base_params.drop_singlet_axes:
            self.output.drop_singlet_axes()
        self.output.retag(self._base_params.output_name)
        return self.output
    def run_cycle(self,
                  **kwargs
                  ):
        func_kwargs = {}
        for key, value in kwargs.items():
            # print(self._base_params._params.keys())
            if key in self._base_params._params.keys():
                self._base_params._update_param(key, value)
            else:
                func_kwargs[key] = value
        # func_kwargs = {key: value for key in kwargs.keys() if key not in self.base_params.keys()}
        self.run(**func_kwargs)
        self.get_output()
        print("One cycle run.")