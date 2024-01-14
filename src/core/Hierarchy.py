import os, zarr, json, shutil, copy, tempfile, numcodecs, warnings
from pathlib import Path
import numpy as np  # pandas as pd
import dask
from dask import array as da
import s3fs
from skimage.transform import resize as skresize
from rechunker import rechunk
from src.core import config, utils
from src.core.MetaData import _Meta, _MultiMeta, _ImageLabelMeta
from src.core.Pyramid import Pyramid

from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    Iterable,
    List,
    Optional
)

class Input(zarr.Group):
    def __init__(self,
                 gr_or_path: Union[str, Path, zarr.Group]
                 ):
        if isinstance(gr_or_path, (str, Path, zarr.Group)):
            self._read(gr_or_path)

    def _read(self,  ### This will also handle reading from s3
              gr_or_path: Union[str, Path, zarr.Group],
              mode = 'r'
              ):
        if isinstance(gr_or_path, (str, Path)):
            gr = zarr.open_group(gr_or_path, mode = mode)
            zarr.Group.__init__(self, gr.store)
        elif isinstance(gr_or_path, zarr.Group):
            zarr.Group.__init__(self, gr_or_path.store)

    @property
    def grpath(self):
        return self.store.path

    @property
    def grdir(self):
        return os.path.dirname(self.grpath)

    @property
    def grname(self):
        return os.path.basename(self.grpath)

    @property
    def has_ome_meta(self):
        return 'OME' in self.grkeys

    @property
    def grkeys(self):
        return list(self.group_keys())

    @property
    def arrkeys(self):
        return list(self.array_keys())

    @property
    def allkeys(self):
        return list(self.keys())

    # def __getitem__(self,
    #                 item
    #                 ):
    #
    #     return self.grs[item]


class HierarchyValidator(_Meta):

    # @property
    # def has_arrays(self):
    #     return (len(self.arrkeys) > 0)

    @property
    def is_collection(self): ### Critical: if the data is collection, multiscales methods cannot be applied to it.
        return (len(self.grkeys) > 0)

    @property
    def is_true_collection(self):
        return self.is_collection & (len(self.arrkeys) == 0)

    @property
    def is_hybrid_collection(self):
        return self.is_collection & (len(self.arrkeys) > 0)

    @property
    def is_bioformats2raw_layout(self):
        return self.is_collection and self.has_bioformats2raw_meta()

    @property
    def is_label_collection(self):
        return self.is_collection and self.has_labelcollection_meta()

    @property
    def is_undefined_collection(self):
        return self.is_collection and not self.has_bioformats2raw_meta() and not self.has_labelcollection_meta()

    @property
    def is_multiscales(self):
        return self.has_multiscales_meta() and (len(self.arrkeys) > 0)

    @property
    def is_label_image(self):
        return self.is_multiscales and self.has_imagelabel_meta()

    @property
    def is_image(self):
        return self.is_multiscales and not self.has_imagelabel_meta()

    @property
    def is_undefined_multiscales(self):
        return self.is_multiscales and not self.is_image and not self.is_label_image

#########################################################################################################################################################
# TODO:################################# Type-specific readers. Add specific metadata parsers to these classes ##########################################
#########################################################################################################################################################

class BaseReader(Input, HierarchyValidator):
    def __init__(self,
                 gr_or_path: Union[str, Path]
                 ):
        Input.__init__(self, gr_or_path)

class MultiScales(BaseReader, Pyramid):
    def __init__(self,
                 gr_or_path: Union[str, Path]
                 ):
        BaseReader.__init__(self, gr_or_path)
        Pyramid.__init__(self)
        self.from_zarr(self.grpath)
    def rebase_multiscales(self, newpath): # TODO: this method is not clear.
        BaseReader.__init__(self, newpath)
        Pyramid.__init__(self)

class Hybrid(BaseReader, Pyramid):
    def __init__(self,
                 gr_or_path: Union[str, Path]
                 ):
        BaseReader.__init__(self, gr_or_path)
        Pyramid.__init__(self)
        self._collect_group_paths()
        self._collect_groups()
        self.from_zarr(self.grpath)

    def rebase_multiscales(self, newpath): # TODO: this method is not clear. Fix it.
        BaseReader.__init__(self, newpath)
        Pyramid.__init__(self)
    def _collect_group_paths(self,
                              ):
        self.zarr_paths_abs = []
        self.zarr_paths_rel = []
        self.zarr_paths = []
        grs = copy.deepcopy(dict(self.groups()))
        keys, grs = utils.transpose_dict(grs)
        while len(grs) > 0:
            gr = grs[0]
            print(gr)
            self.zarr_paths.append(os.path.join(self.grname, gr.path))
            self.zarr_paths_rel.append(gr.path)
            self.zarr_paths_abs.append(os.path.join(gr.store.path, gr.path))
            _, newgrs = utils.transpose_dict(dict(gr.groups()))
            grs += newgrs
            grs.pop(0)
        self.zarr_paths_rel.insert(0, '')
        self.zarr_paths.insert(0, self.grname)
        self.zarr_paths_abs.insert(0, self.grpath)

    def _collect_groups(self):            # TODO: here the reading class should detect the group type: collection, label-image whatever.
        if not 'zarr_paths_abs' in self.__dict__:
            raise AttributeError()
        if not 'grs' in self.__dict__:
            self.grs, self.sub = {}, {}
        for pth, spth, rpth in zip(self.zarr_paths, self.zarr_paths_abs, self.zarr_paths_rel):
            gr = BaseReader(spth)
            if spth == self.grpath:
                grp = self
            else:
                if gr.is_hybrid_collection:
                    grp = Hybrid(spth)  # TODO: solve this
                elif gr.is_collection:
                    grp = Collection(spth)
                elif gr.is_multiscales:
                    grp = MultiScales(spth)
            pth_split = pth.split('/')
            if len(pth_split) > 1:
                jn = os.path.join(*pth_split[1:])
                self.__setattr__(jn, grp)
            self.grs[pth] = grp
            # print(f'this is rpth:{rpth}')
            if len(rpth) > 0:
                self.sub[rpth] = grp

class Collection(BaseReader):
    def __init__(self,
                 gr_or_path: Union[str, Path]
                 ):
        BaseReader.__init__(self, gr_or_path)
        self._collect_group_paths()
        self._collect_groups()

    def _collect_group_paths(self,
                              ):
        self.zarr_paths_abs = []
        self.zarr_paths_rel = []
        self.zarr_paths = []
        grs = copy.deepcopy(dict(self.groups()))
        keys, grs = utils.transpose_dict(grs)
        while len(grs) > 0:
            gr = grs[0]
            print(gr)
            self.zarr_paths.append(os.path.join(self.grname, gr.path))
            self.zarr_paths_rel.append(gr.path)
            self.zarr_paths_abs.append(os.path.join(gr.store.path, gr.path))
            _, newgrs = utils.transpose_dict(dict(gr.groups()))
            grs += newgrs
            grs.pop(0)
        self.zarr_paths_rel.insert(0, '')
        self.zarr_paths.insert(0, self.grname)
        self.zarr_paths_abs.insert(0, self.grpath)

    def _collect_groups(self):            # TODO: here the reading class should detect the group type: collection, label-image whatever.
        if not 'zarr_paths_abs' in self.__dict__:
            raise AttributeError()
        if not 'grs' in self.__dict__:
            self.grs, self.sub = {}, {}
        for pth, spth, rpth in zip(self.zarr_paths, self.zarr_paths_abs, self.zarr_paths_rel):
            gr = BaseReader(spth)
            if spth == self.grpath:
                grp = self
            else:
                if gr.is_hybrid_collection:
                    grp = Hybrid(spth)  # TODO: solve this
                elif gr.is_collection:
                    grp = Collection(spth)
                elif gr.is_multiscales:
                    grp = MultiScales(spth)
            pth_split = pth.split('/')
            if len(pth_split) > 1:
                jn = os.path.join(*pth_split[1:])
                self.__setattr__(jn, grp)
            self.grs[pth] = grp
            # print(f'this is rpth:{rpth}')
            if len(rpth) > 0:
                self.sub[rpth] = grp

#########################################################################################################################################################
#########################################################################################################################################################

class OMEZarr(Hybrid): ### TODO: Bunun repr methodunu olustur.
    def __init__(self,
                 gr_or_path,
                 ):
        self._grstore = None
        self.set_input(gr_or_path)
        self.read()
    def set_input(self, gr_or_path):
        if isinstance(gr_or_path, (str, Path, zarr.Group)):
            self.gr_or_path = gr_or_path
        else:
            raise TypeError(f"The input must be either of the types: {str, Path, zarr.Group}")
    def read(self):
        data = BaseReader(self.gr_or_path)
        self.non_resolved = False
        if data.is_hybrid_collection:
            print('Object is a hybrid collection.')
            Hybrid.__init__(self, self.gr_or_path)
        elif data.is_collection:
            print('Object is a collection.')
            Collection.__init__(self, self.gr_or_path)
        elif data.is_multiscales:
            print('Object is a multiscales.')
            MultiScales.__init__(self, self.gr_or_path)
        else:
            raise TypeError('Type of the OME-Zarr could not be resolved.')
    # def new(self,
    #         num_arrpaths: int = 1,
    #         shape: Iterable = (500, 500, 500),
    #         chunks: Iterable = (96, 96, 96),
    #         axis_order: str = 'zyx',
    #         unit_order: Union[Iterable, None] = None
    #         ):
    #     ozs = OMEZarrSample(self.gr_or_path)
    #     ozs.add_datasets(num_arrpaths = num_arrpaths, top_shape = shape, chunks = chunks, axes = axis_order, units = unit_order)
    #     self.set_input(ozs.grp)
    #     self.read()


