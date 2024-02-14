"""Wrappers for bfconvert and bioformats2raw to be run from python environment. """

import subprocess, re, shlex
from typing import Optional, List
import fire

intlist = lambda s: [int(x) for x in re.findall(r'\b\d+\b', s)]

class Converter:
    def __init__(self,
                 execute: bool = True,
                 use_shell: bool = False
                 ):
        self.execute = execute
        self.use_shell = use_shell
    def as_cmd(self,
                   ):
        self.execute = False
        return self
    def as_exec(self,
               ):
        self.execute = True
        return self
    def from_shell(self,
                   ):
        self.use_shell = True
        return self
    def from_cmdlist(self,
                     ):
        self.use_shell = False
        return self
    def to_ometiff(
                  self,
                  input_path: str,
                  output_path: str,
                  noflat: Optional[bool] = None,
                  series: Optional[str] = None,
                  timepoint: Optional[str] = None,
                  channel: Optional[str] = None,
                  z_slice: Optional[str] = None,
                  idx_range: Optional[str] = None,
                  autoscale: Optional[bool] = None,
                  crop: Optional[str] = None,
                  compression: Optional[str] = None,
                  resolution_scale: Optional[str] = None,
                  resolutions: Optional[str] = None,
                  # execute: bool = False,
                  # use_shell: bool = False
                  ):
        cmd = ["bfconvert"]
        if noflat is not None:
            cmd += [" -noflat"]
        if series is not None:
            cmd += [" -series", ' %s' % series]
        if timepoint is not None:
            cmd += [" -timepoint", ' %s' % timepoint]
        if channel is not None:
            cmd += [" -channel", ' %s' % channel]
        if z_slice is not None:
            cmd += [" -z", ' %s' % z_slice]
        if idx_range is not None:
            _range_list = intlist(_range)
            if len(_range_list) != 2:
                raise TypeError('Range must have two integers specifying first and last indices of images.')
            else:
                cmd += [" -range"]
                for i in _range_list:
                    cmd += [' %s' % i]
        if autoscale is not None:
            cmd += [" -autoscale"]
        if crop is not None:
            if isinstance(crop, tuple):
                _crop = str(crop)[1:-1].replace(' ', '')
            elif isinstance(crop, str):
                _crop = crop
            cmd += [" -crop", ' %s' % _crop]
        if compression is not None:
            cmd += [" -compression", ' %s' % compression]
        if resolution_scale is not None:
            cmd += [" -pyramid-scale", ' %s' % resolution_scale]
        if resolutions is not None:
            cmd += [" -pyramid-resolutions", ' %s' % resolutions]

        cmd.append(' %s' % input_path)
        cmd.append(' %s' % output_path)
        cmd = ''.join(cmd)
        self.cmd = cmd

        if not self.execute:
            return self.cmd
        else:
            if not self.use_shell:
                self.cmd = shlex.split(self.cmd)
            subprocess.run(self.cmd, shell=self.use_shell)
            return

    def to_omezarr(
                    self,
                    input_path: str,
                    output_path: str,
                    resolutions: int = None,
                    min_image_size: int = None,
                    chunk_y: int = None,
                    chunk_x: int = None,
                    chunk_z: int = None,
                    downsample_type: str = None,
                    compression: str = None,
                    max_workers: int = None,
                    no_nested: bool = None,
                    drop_series: bool = None,
                    overwrite: bool = None,
                    # execute: bool = False,
                    # use_shell: bool = False
                    ):
        # param_names = self.to_omezarr.__code__.co_varnames[:self.to_omezarr.__code__.co_argcount]
        # self.params
        # for param_name in param_names:
        #     param_value = vars(self).get(param_name)
        #     print(f"{param_name} = {param_value}")

        cmd = ["bioformats2raw"]

        if resolutions is not None:
            cmd += [" --resolutions", ' %s' % resolutions]
        if min_image_size is not None:
            cmd += [" --target-min-size", ' %s' % min_image_size]
        if chunk_y is not None:
            cmd += [" --tile_height", ' %s' % chunk_y]
        if chunk_x is not None:
            cmd += [" --tile_width", ' %s' % chunk_x]
        if chunk_z is not None:
            cmd += [" --chunk_depth", ' %s' % chunk_z]
        if downsample_type is not None:
            cmd += [" --downsample-type", ' %s' % downsample_type]
        if compression is not None:
            cmd += [" --compression", ' %s' % compression]
        if max_workers is not None:
            cmd += [" --max_workers", ' %s' % max_workers]
        if no_nested is not None:
            cmd += [" --no-nested"]
        if drop_series is not None:
            cmd += [" --scale-format-string", ' %s' % "'%2$d'"]
        if overwrite is not None:
            cmd += [" --overwrite"]

        cmd.append(' %s' % input_path)
        cmd.append(' %s' % output_path)
        cmd = ''.join(cmd)
        self.cmd = cmd

        if not self.execute:
            return self.cmd
        else:
            if not self.use_shell:
                self.cmd = shlex.split(self.cmd)
            subprocess.run(self.cmd, shell=self.use_shell)
            return

def run():
    fire.Fire(Converter)

def to_omezarr():
    fire.Fire(Converter().as_exec().from_cmdlist().to_omezarr)

def to_ometiff():
    fire.Fire(Converter().as_exec().from_cmdlist().to_ometiff)
