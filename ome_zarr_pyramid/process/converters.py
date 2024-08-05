"""Wrappers for bfconvert and bioformats2raw to be run from python environment. """
import os.path, os, glob
import subprocess, re, shlex
from typing import Optional, List
import fire
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import Pool


intlist = lambda s: [int(x) for x in re.findall(r'\b\d+\b', s)]

def run_command(cmd, use_shell = False):
    result = subprocess.run(cmd, shell=use_shell, capture_output=True, text=True)
    return result

class Converter:
    def __init__(self,
                 execute: bool = True,
                 use_shell: bool = False,
                 n_jobs = 8,
                 require_sharedmem = None
                 ):
        self.execute = execute
        self.use_shell = use_shell
        self.n_jobs = n_jobs
        self.require_sharedmem = require_sharedmem
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
                    ):

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

    def to_ometiffs(self,
                    input_dir: str,
                    output_dir: str,
                    pattern: str = '*',
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
                    ):
        globpattern = os.path.join(input_dir, pattern)
        inpaths = glob.glob(globpattern)
        outpaths = []
        for inpath in inpaths:
            name = os.path.basename(inpath)
            namebase, _ = os.path.splitext(name)
            newname = namebase + '.ome.tiff'
            newpath = os.path.join(output_dir, newname)
            outpaths.append(newpath)
        os.makedirs(output_dir, exist_ok = True)
        cmds = []
        for inpath, outpath in zip(inpaths, outpaths):
            cmd = self.as_cmd().to_ometiff(inpath,
                                            outpath,
                                            noflat = noflat,
                                            series = series,
                                            timepoint = timepoint,
                                            channel = channel,
                                            z_slice = z_slice,
                                            idx_range = idx_range,
                                            autoscale = autoscale,
                                            crop = crop,
                                            compression = compression,
                                            resolution_scale = resolution_scale,
                                            resolutions = resolutions
                                            )
            if not self.use_shell:
                cmd = shlex.split(self.cmd)
            cmds.append(cmd)
        with Parallel(n_jobs=self.n_jobs, require=self.require_sharedmem) as parallel:
            _ = parallel(
                delayed(run_command)
                    (
                    cmd,
                    use_shell = self.use_shell
                )
                for i, cmd in enumerate(cmds)
            )

    def to_omezarrs(self,
                    input_dir: str,
                    output_dir: str,
                    pattern: str = '*',
                    resolutions: int = None,
                    min_image_size: int = None,
                    chunk_z: int = None,
                    chunk_y: int = None,
                    chunk_x: int = None,
                    downsample_type: str = None,
                    compression: str = None,
                    max_workers: int = None,
                    no_nested: bool = None,
                    drop_series: bool = None,
                    overwrite: bool = None,
                    ):
        globpattern = os.path.join(input_dir, pattern)
        inpaths = glob.glob(globpattern)
        outpaths = []
        for inpath in inpaths:
            name = os.path.basename(inpath)
            namebase, _ = os.path.splitext(name)
            newname = namebase + '.zarr'
            newpath = os.path.join(output_dir, newname)
            outpaths.append(newpath)
        os.makedirs(output_dir, exist_ok = True)
        cmds = []
        for inpath, outpath in zip(inpaths, outpaths):
            cmd = self.as_cmd().to_omezarr(inpath,
                                            outpath,
                                            resolutions = resolutions,
                                            min_image_size = min_image_size,
                                            chunk_z = chunk_z,
                                            chunk_y = chunk_y,
                                            chunk_x = chunk_x,
                                            downsample_type = downsample_type,
                                            compression = compression,
                                            max_workers = max_workers,
                                            no_nested = no_nested,
                                            drop_series = drop_series,
                                            overwrite = overwrite
                                            )
            if not self.use_shell:
                cmd = shlex.split(self.cmd)
            cmds.append(cmd)
        with Parallel(n_jobs=self.n_jobs, require=self.require_sharedmem) as parallel:
            _ = parallel(
                delayed(run_command)
                    (
                    cmd,
                    use_shell = self.use_shell
                )
                for i, cmd in enumerate(cmds)
            )

def run():
    fire.Fire(Converter)

def to_omezarr():
    fire.Fire(Converter().as_exec().from_cmdlist().to_omezarr)

def to_ometiff():
    fire.Fire(Converter().as_exec().from_cmdlist().to_ometiff)
