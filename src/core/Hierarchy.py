import os, zarr, json, shutil, copy, tempfile, numcodecs, warnings
from pathlib import Path
import numpy as np  # pandas as pd
import dask
from dask import array as da
import s3fs
from skimage.transform import resize as skresize
from rechunker import rechunk
from OME_Zarr.src.core import config, utils
from OME_Zarr.src.core.MetaData import _Meta, _MultiMeta, _ImageLabelMeta
from OME_Zarr.src.core.OMEZarrCore import ZarrayCore, ZarrayManipulations

from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    Iterable,
    List,
    Optional
)

""" 

OME-Zarr components:

Voxel: anything that has hierarchy of multi-resolution layers of pixel/voxel data. Must contain multiscales metadata. 

LabelVoxel: an instance of Voxel that additionally has image-label metadata. Inherits Voxel

Multiseries: a collection of multiple Voxel instances. Validated by having more than one Voxel instance in the path.

MultiseriesGeneric: a collection of multiple Voxel instances, along with specified paths in the metadata. Inherits Multiseries.

MultiseriesLabel: a collection of multiple LabelVoxel instances. Must contain label paths metadata. Inherits MultiseriesGeneric.

"""

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


class DataMovement:
    """ Template methods to extract or move subsets of data """
    def extract(self,
                newdir: str = None,
                paths: Iterable = None,
                subset: Union[dict, None] = None,
                include_subdirs: bool = False,
                overwrite: bool = True,
                zarr_meta: Union[Dict, None] = None
                ): # TODO: apply the _rebase method to selected multiscales paths in the omezarr.
        subdirs, subdirs_rel = [self.grname], ['']
        if include_subdirs:
            subdirs = self.zarr_paths; subdirs_rel = self.zarr_paths_rel
        for key, relkey in zip(subdirs, subdirs_rel):
            gr = self.grs[key]
            newpath = os.path.join(newdir, relkey)
            if gr.is_multiscales:
                print(f"zarr path is {newpath}")
                subset_copy = copy.deepcopy(subset)
                print(f"subset being passed to _extract is {subset_copy}")
                gr._extract(newpath, paths, subset_copy, overwrite, zarr_meta)
            elif gr.is_true_collection:
                grp = zarr.group(newpath)
                print(newpath)
                for key, value in gr.attrs.items():
                    print(key, value)
                    grp.attrs[key] = value
    def rebase(self,
               newdir: str = None,
               paths: Iterable = None,
               subset: Union[dict, None] = None,
               include_subdirs = False,
               overwrite: bool = True,
               zarr_meta: Union[Dict, None] = None
               ):
        self.extract(newdir, paths, subset, include_subdirs, overwrite, zarr_meta)
        self.grs = {}
        self.__init__(newdir)
    def export_roi(self,
                   label_no: int = None,
                   bounding_box: Union[Iterable[slice], Iterable[tuple]] = None
                   ): # TODO: Burada dikkat et, dict de alabilirsin bu parametreyi.
        pass

#########################################################################################################################################################
# TODO:################################# Type-specific readers. Add specific metadata parsers to these classes ##########################################
#########################################################################################################################################################

class BaseReader(Input, HierarchyValidator, DataMovement):
    def __init__(self,
                 gr_or_path: Union[str, Path]
                 ):
        Input.__init__(self, gr_or_path)

class MultiScales(BaseReader, ZarrayManipulations, DataMovement):
    def __init__(self,
                 gr_or_path: Union[str, Path]
                 ):
        BaseReader.__init__(self, gr_or_path)
        ZarrayCore.__init__(self)
    @property
    def layers(self):
        return dict(self.arrays())
    def rebase_multiscales(self, newpath): # TODO: this method is not clear.
        BaseReader.__init__(self, newpath)
        ZarrayCore.__init__(self)


class Hybrid(BaseReader, ZarrayManipulations, DataMovement):
    def __init__(self,
                 gr_or_path: Union[str, Path]
                 ):
        BaseReader.__init__(self, gr_or_path)
        ZarrayCore.__init__(self)
        self._collect_group_paths()
        self._collect_groups()

    @property
    def layers(self):
        return dict(self.arrays())
    def rebase_multiscales(self, newpath): # TODO: this method is not clear. Fix it.
        BaseReader.__init__(self, newpath)
        ZarrayCore.__init__(self)
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
            self.grs = {}
        for pth, spth in zip(self.zarr_paths, self.zarr_paths_abs):
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

class Collection(BaseReader, DataMovement):
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
            self.grs = {}
        for pth, spth in zip(self.zarr_paths, self.zarr_paths_abs):
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

#########################################################################################################################################################
#########################################################################################################################################################

class OMEZarrSample:
    def __init__(self,
                 pth: Union[str, Path],
                 ):
        grp = zarr.open_group(pth, mode = 'a')
        grp.attrs['multiscales'] = [{'axes': [],
                                     'datasets': [],
                                     'name': "/"
                                     }]
        self.grp = grp
        self.store = self.grp.store
        self.resolution_paths = []
    @property
    def multimeta(self):
        return self.grp.attrs['multiscales']
    def add_dataset(self,
                    array_shape,
                    path: Union[str, int],
                    scale: Iterable[str],  ### scale values go here.
                    transform_type: str = 'scale',
                    chunks=(96, 96, 96),
                    ):
        self.resolution_paths.append(path)
        transform = [scale] * len(array_shape)
        dataset = {'coordinateTransformations': [{'scale': transform, 'type': transform_type}], 'path': str(path)}
        self.multimeta[0]['datasets'].append(dataset)
        resolution_paths = [int(pth) for pth in self.resolution_paths]
        args = np.argsort(resolution_paths)
        self.multimeta[0]['datasets'] = [self.multimeta[0]['datasets'][i] for i in args]
        self.grp.attrs['multiscales'] = self.multimeta
        ################
        array = zarr.zeros(array_shape, chunks = chunks)
        self.grp.create_dataset(name = str(path), data = array)
    def add_datasets(self,
                     num_arrpaths = 3,
                     top_shape = (300, 300, 300),
                     chunks = (100, 100, 100)
                     ):
        self.specify_axes(shape = top_shape)
        self.scales = [(2 ** i) for i in range(num_arrpaths)]
        self.shapes = [np.array(top_shape) // item for item in self.scales]
        for i, shape in enumerate(self.shapes):
            self.add_dataset(shape,
                             path = f'{i}',
                             scale = self.scales[i],
                             chunks = chunks
                             )
    def specify_axes(self,
                     order: str = 'tczyx',
                     shape: Iterable = (300, 300, 300)
                     ):
        self.axis_order = order[-len(shape):]
        self.grp.attrs['multiscales'][0]['axes'] = [{"name": n, "type": config.type_map[n]} for n in self.axis_order]

class OMEZarr(Hybrid, DataMovement): ### TODO: Bunun repr methodunu olustur.
    def __init__(self, gr_or_path):
        self._grstore = None
        self.set_input(gr_or_path)
        if self.path_has_group():
            self.read()
        else:
            self.new()
    def set_input(self, gr_or_path):
        if isinstance(gr_or_path, (str, Path, zarr.Group)):
            self.gr_or_path = gr_or_path
        else:
            raise TypeError(f"The input must be either of the types: {str, Path, zarr.Group}")
    @property
    def grstore(self):
        if isinstance(self.gr_or_path, (str, Path)):
            self._grstore = zarr.DirectoryStore(self.gr_or_path)
        elif isinstance(self.gr_or_path, zarr.Group):
            self._grstore = self.gr_or_path.store
        if self._grstore is None:
            raise TypeError(f"The input must be either of the types: {str, Path, zarr.Group}")
        return self._grstore
    def path_has_group(self):
        return zarr.storage.contains_group(self.grstore)
    def read(self):
        assert self.path_has_group(), "Path does not contain a zarr group."
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
    def new(self,
            num_arrpaths: int = 1,
            shape: Iterable = (500, 500, 500)
            ):
        ozs = OMEZarrSample(self.gr_or_path)
        ozs.add_datasets(num_arrpaths = num_arrpaths, top_shape = shape)
        self.set_input(ozs.grp)
        self.read()
        # MultiScales.__init__(self, ozs.store.path)


