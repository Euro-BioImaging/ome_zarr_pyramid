import os, zarr, json, shutil, copy, tempfile, numcodecs, warnings
from pathlib import Path
import numpy as np  # pandas as pd
import dask
from dask import array as da
import s3fs
from skimage.transform import resize as skresize
from rechunker import rechunk
from OME_Zarr.src.core import config, utils
from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    Iterable,
    List,
    Optional
)


class Collection:
    def __init__(self,
                 root_path: Union[str, Path]
                 ):
        self._read_ome_zarr(root_path)
        self._classify_collections()

    def _read_ome_zarr(self,  ### This will also handle reading from s3
                       root_path: Union[str, Path]
                       ):
        self.root_path = root_path
        self.base = zarr.group(self.root_path)
        self.store = self.base.store

    @property
    def basepath(self):
        return self.base.store.path

    @property
    def basedir(self):
        return os.path.dirname(self.base.store.path)

    @property
    def basename(self):
        return os.path.basename(self.basepath)

    @property
    def attrs(self):
        return dict(self.base.attrs)

    @property
    def group_keys(self):
        return list(self.base.group_keys())

    @property
    def has_ome_meta(self):
        return 'OME' in self.group_keys

    @property
    def array_keys(self):
        return list(self.base.array_keys())

    @property
    def allkeys(self):
        return list(self.base.keys())

    def _classify_collections(self):
        self.is_generic_collection = False
        paths = list(self.base.keys())
        attrkeys, attrvalues = utils.transpose_dict(self.attrs)
        if self.basename in attrkeys and (len(paths) > 0):
            self.is_generic_collection = True
            for item0, item1 in zip(self.attrs[self.basename], paths):
                if item0 != item1:
                    self.is_generic_collection = False
        self.is_multiseries = 'bioformats2raw.layout' in self.attrs.keys()
        self.is_labels = 'labels' in self.attrs.keys()
        self.is_multiscales = 'multiscales' in self.attrs.keys()
        self.is_image_label = 'image-label' in self.attrs.keys()
        self.is_image = self.is_multiscales and (not self.is_image_label)
        self.is_unstructured = not (self.is_multiseries or self.is_labels or self.is_multiscales)

    @property
    def typeobj(self):
        binary = [self.is_multiseries, self.is_labels, self.is_unstructured, self.is_image_label, self.is_image]
        names = ['multiseries', 'labels', 'unstructured', 'image-label', 'image']
        idx = binary.index(True)
        return names[idx]


class MultiMeta:
    def __init__(self,
                 group  # is a base object from the Reader class
                 ):
        print(group.attrs)
        self.base_attrs = dict(group.attrs)
        self.meta = self.base_attrs['multiscales']

    @property
    def meta_fields(self):
        try:
            return [item for item in self.meta[0]]
        except:
            return config.NotMultiscalesException

    @property
    def has_axes(self):
        try:
            return 'axes' in self.meta_fields
        except:
            return config.NotMultiscalesException

    @property
    def has_datasets(self):
        try:
            return 'datasets' in self.meta_fields
        except:
            return config.NotMultiscalesException

    @property
    def has_name(self):
        try:
            return 'name' in self.meta_fields
        except:
            return config.NotMultiscalesException

    @property
    def axis_order(self):
        try:
            if self.has_axes:
                return ''.join([item['name'] for item in self.meta[0]['axes']])
            else:
                return config.AxisNotFoundException
        except:
            return config.NotMultiscalesException

    def add_axis(self,
                 name: str,
                 unit: str = None,
                 index: int = -1,
                 ):
        if name in self.axis_order:
            raise ValueError(f'{name} axis already exists.')

        def axmake(name, unit):
            if unit is None:
                axis = {'name': name, 'type': config.type_map[name]}
            else:
                axis = {'name': name, 'type': config.type_map[name], 'unit': unit}
            return axis

        if index == -1:
            index = len(self.meta[0]['axes'])
        if not self.has_axes:
            self.meta[0]['axes'] = []
            index = 0
        axis = axmake(name, unit)
        self.meta[0]['axes'].insert(index, axis)

    # def del_axis(self,
    #              name: str
    #              ):
    #     if name not in self.axis_order:
    #         raise ValueError(f'The axis "{name}" does not exist.')
    #     idx = self.axis_order.index(name)
    #     self.meta[0]['axes'].pop(idx)
    #     for pth in self.path_order:
    #         scale = self.get_scale(pth)
    #         scale.pop(idx)
    #         self.set_scale(pth, scale)

    @property
    def path_order(self):
        try:
            return [item['path'] for item in self.meta[0]['datasets']]
        except:
            return config.NotMultiscalesException

    def get_scale(self,
                  pth: Union[str, int]
                  ):
        idx = self.path_order.index(pth)
        return self.meta[0]['datasets'][idx]['coordinateTransformations'][0]['scale']

    def set_scale(self,
                  pth: Union[str, int],
                  scale
                  ):
        idx = self.path_order.index(pth)
        self.meta[0]['datasets'][idx]['coordinateTransformations'][0]['scale'] = scale

    def get_scale_factors(self,
                          paths: Iterable[Union[str, int]]
                          ):
        if paths is None: paths = self.path_order
        scales = np.array([self.get_scale(pth) for pth in paths])
        factored = scales / scales[0]
        return {pth: tuple(factored[paths.index(pth)]) for pth in paths}

    def refactor_subset(self, paths, subset):
        if paths is None: paths = self.path_order
        factors = self.get_scale_factors(paths)
        subsets = {}
        for pth in paths:
            factor = factors[pth]
            for key, item in subset.items():
                idx = self.axis_order.index(key)
                coef = factor[idx]
                itemrefact = np.round(np.array(item) / coef).astype(int)
                subset[key] = tuple(itemrefact) if not np.isscalar(itemrefact) else itemrefact
            subsets[pth] = copy.deepcopy(subset)
        return subsets

    def add_dataset(self,
                    path: Union[str, int],
                    transform: Iterable[str],  ### scale values go here.
                    transform_type: str = 'scale'
                    ):
        dataset = {'coordinateTransformations': [{'scale': transform, 'type': transform_type}], 'path': str(path)}
        self.meta[0]['datasets'].append(dataset)
        path_order = [int(pth) for pth in self.path_order]
        args = np.argsort(path_order)
        self.meta[0]['datasets'] = [self.meta[0]['datasets'][i] for i in args]

    def del_dataset(self,
                    path: Union[str, int]
                    ):
        idx = self.path_order.index(str(path))
        del self.meta[0]['datasets'][idx]

    def extract_dataset(self,
                        paths: Iterable[Union[str, int]],
                        axes: str = None
                        ):
        if paths is None: paths = self.path_order
        indices = [self.path_order.index(pth) for pth in paths]
        scales = [self.get_scale(pth) for pth in paths]
        meta = copy.deepcopy(self.meta)
        meta[0]['datasets'] = []
        if axes is not None:
            axinds = [self.axis_order.index(i) for i in axes]
        else:
            axinds = [i for i in range(len(self.axis_order))]
        axdata = [self.meta[0]['axes'][i] for i in axinds]
        for idx in indices:
            dataset = copy.deepcopy(self.meta[0]['datasets'][idx])
            scale = scales[idx]
            if axes is not None:
                scale = [scale[i] for i in axinds]
            dataset['coordinateTransformations'][0]['scale'] = scale
            meta[0]['axes'] = axdata
            meta[0]['datasets'].append(dataset)
        return meta

    def increment_scale(self,
                        axes: str ='xy',
                        scale_factor: Union[float, int] = 2
                        ):
        indices = [self.axis_order.index(i) for i in axes]
        scale = copy.deepcopy(self.meta[0]['datasets'][-1]['coordinateTransformations'][0]['scale'])
        pth = self.meta[0]['datasets'][-1]['path']
        if pth.isnumeric():
            pth = int(pth) + 1
        else:
            pth = pth + '_rescaled'
        for idx in indices:
            scale[idx] *= scale_factor
        self.add_dataset(pth, scale, 'scale')

    def decrement_scale(self):
        pth = self.path_order[-1]
        self.del_dataset(pth)


class ImageLabelMeta(MultiMeta):  # TODO: Image-label metadata will be parsed here.
    def __init__(self, group):
        MultiMeta.__init__(self, group)
        self.label_meta = self.base_attrs['image-label']


class ZarrayCore(MultiMeta):
    """ This class adds array-level metadata and update methods.
        Updates are possible to: 'chunks', 'compressor', 'dtype', 'dimension_separator'
    """

    def __init__(self,
                 group: zarr.Group
                 ):
        self.base = group
        MultiMeta.__init__(self, self.base)
        keys = self.base.attrs.keys()
        assert 'multiscales' in keys
        self.__collect_array_meta()

    def sort_paths(self):
        meta = self.meta
        args = utils.argsorter(self.path_order)
        pths = [self.path_order[i] for i in args]
        arrmeta = {i: self.array_meta[i] for i in pths}
        datasets = [meta[0]['datasets'][i] for i in args]
        meta[0]['datasets'] = datasets
        self.array_meta = arrmeta
        self.meta = meta
        self.base.attrs['multiscales'] = self.meta

    def __collect_array_meta(self):
        res = {pth: {} for pth in self.path_order}
        arrmeta_keys = ['chunks', 'shape', 'compressor', 'dtype', 'dimension_separator']
        for metakey in arrmeta_keys:
            for pth in self.path_order:
                if metakey in self.base[pth]._meta:
                    if metakey == 'compressor':
                        res[pth][metakey] = self.base[pth].compressor
                    else:
                        res[pth][metakey] = self.base[pth]._meta[metakey]
                elif metakey not in self.base[pth]._meta:
                    if metakey == 'dimension_separator':
                        res[pth][metakey] = '.'
        self.array_meta = res

    def update_name(self):
        self.meta[0]['name'] = self.basename
        self.base.attrs['multiscales'] = self.meta

    def __write_layer(self,
                      paths: Union[List, str],
                      out_dir: Union[str, None] = None,
                      overwrite_group: bool = False,
                      overwrite_arrays: bool = False,
                      new_arrays: Union[List, None] = None
                      ):
        """ Write selected array(s) based on the parameters in the current array metadata.
            Note that this normally does not add new arrays to the OME_Zarr. It only modifies any existing arrays for the specific metadata fields.
            If, however, new_arrays variable is specified, then new arrays will be added to the OME_Zarr hierarchy following the metadata provided
            in the array_meta variable for the paths provided in the paths variable..
         """
        if new_arrays is not None:
            assert isinstance(new_arrays, (list, tuple))
            assert len(new_arrays) == len(paths)
        if out_dir is None: ### if out_dir is not specified, this is an in-place write
            out_dir = self.base.store.path
        for i, pth in enumerate(paths):
            arrmeta = self.array_meta[pth]
            if new_arrays is None:
                arr = self.base[pth]
                assert isinstance(arr, zarr.Array)
            else: ### If new arrays are provided, make sure that they undergo a rechunking
                arr = new_arrays[i]
                if isinstance(arr, da.Array): ### If the new array is a dask array, update its data type
                    arr = arr.astype(arrmeta['dtype'])
                elif isinstance(arr, zarr.Array): ### If the new array is a zarr array, convert it into dask and update its data type
                    arr = da.from_zarr(arr).astype(arrmeta['dtype'])
                elif isinstance(arr, np.ndarray):
                    arr = da.from_array(arr).astype(arrmeta['dtype'])
            target_base = tempfile.TemporaryDirectory()
            target = os.path.join(target_base.name, 'results')
            tempchunks = os.path.join(target_base.name, 'tempchunks')
            arraypath = os.path.join(out_dir, pth)
            chunksize = arr.chunks if isinstance(arr, zarr.Array) else arr.chunksize ### Get the chunk size for comparison
            print(f"chunk size in array meta: {arrmeta['chunks']}")
            print(f"chunk size in the array: {chunksize}")
            print(f"array shape: {arr.shape}")
            if np.all(arrmeta['chunks'] == chunksize) & (isinstance(arr, zarr.Array)):
                ### if there has been no update in chunk size, and data is already in zarr format,
                ### no need to do the rechunking process. If it is a dask array, though, rechunking is needed
                ### to regularise the chunk size. Especially, reslicing the data often causes irregularities with chunks.
                if out_dir == self.base.store.path:
                    ### backup data in temporary path if change is in place
                    warnings.warn('Data is being changed in-place.')
                    # arr.to_zarr(url = target) ### save to temporary path
                    # layer = zarr.open_array(target)
                    layer = zarr.array(arr, store = target)
                else:
                    layer = arr
            else:
                try:
                    print(f'Rechunking for path: {pth}.')
                    plan = rechunk(arr,
                                   target_chunks=arrmeta['chunks'],
                                   max_mem=256000,
                                   target_store=target,
                                   target_options={'compressor': arrmeta['compressor']},
                                   temp_store=tempchunks)
                    plan.execute()
                    layer = zarr.open_array(target)
                except:
                    warnings.warn(f'Cannot progress with rechunking for path: {pth}. Converting to numpy.')
                    arrnum = np.array(arr)
                    layer = zarr.array(arrnum, chunks=arrmeta['chunks'], store=target)
            if isinstance(layer, zarr.Array):
                dsk = da.from_zarr(layer).astype(arrmeta['dtype'])
            elif isinstance(layer, da.Array):
                dsk = layer.astype(arrmeta['dtype'])
            if (i == 0) & overwrite_group:
                _ = zarr.group(out_dir, overwrite = True)
            dsk.to_zarr(url = arraypath,
                        compressor = arrmeta['compressor'],
                        dimension_separator = arrmeta['dimension_separator'],
                        overwrite = overwrite_arrays
                        )
            # _ = zarr.array(layer,
            #                dimension_separator=arrmeta['dimension_separator'],
            #                compressor=arrmeta['compressor'],
            #                dtype=arrmeta['dtype'],
            #                store=arraypath,
            #                overwrite=overwrite_arrays
            #                )
            shutil.rmtree(target_base.name)

    def chunks(self,
               pth: Union[str, int] = '0'
               ):
        return (self.array_meta[pth]['chunks'])

    def compressor(self,
                   pth: Union[str, int] = '0'
                   ):
        return self.array_meta[pth]['compressor']

    def dtype(self,
              pth: Union[str, int] = '0'
              ):
        return self.array_meta[pth]['dtype']

    def __set_array_meta(self,
                         pth: Union[int, str],
                         metakeys: Iterable[Any],
                         values: Iterable[Any],
                         new_shape: Iterable
                         ):
        """ Updates the array metadata for specific resolution levels.
            You can update:
                - dtype
                - chunks
                - compressor
                - dimension_separator(nestedness)
            Note that this method sets one path at a time, does not accept multiple paths.
            It can, however, set multiple key-value pairs per path.
        """
        if isinstance(pth, int):
            pth = str(pth)
        if not isinstance(metakeys, (list, tuple)):
            metakeys = [metakeys]
        if not isinstance(values, (list, tuple)):
            values = [values]
        curpaths = sorted(list(self.array_meta.keys()))
        if pth not in curpaths:
            self.array_meta[pth] = self.array_meta[curpaths[0]]
        for metakey, value in zip(metakeys, values):
            print(f'{metakey}:{value}')
            if metakey == 'chunks':
                if new_shape is None:
                    shape = np.array(self.base[pth].shape)
                else:
                    shape = np.array(new_shape)
                chunks = np.array(value)
                value = np.where(chunks > shape, shape, chunks)
                self.array_meta[pth][metakey] = tuple(value)
            else:
                self.array_meta[pth][metakey] = value

    def __modify_array_meta(self,
                            paths: Iterable,
                            metakey: Any,
                            value: Any,
                            new_arrays: Union[Iterable[Union[np.ndarray, zarr.Array, da.Array]], None] = None
                            ):
        """"""
        if isinstance(paths, str):
            paths = [paths]
        elif isinstance(paths, int):
            if paths >= 0:
                paths = [paths]
            else:
                paths = self.path_order
        if new_arrays is not None:
            for pth in paths:
                self.array_meta[pth] = self.array_meta[self.path_order[0]]
        for i, pth in enumerate(paths):
            if metakey == 'chunks':
                if new_arrays is None:
                    shape = np.array(self.base[pth].shape)
                else:
                    shape = np.array(new_arrays[i].shape)
                chunks = np.array(value)
                value = np.where(chunks > shape, shape, chunks)
                self.array_meta[pth][metakey] = tuple(value)
            else:
                self.array_meta[pth][metakey] = value
        # print(self.array_meta)

    def update_arrays(self,
                      paths: Iterable,
                      metakey: Any,
                      value: Any,
                      overwrite: Optional[bool] = True
                      ):
        """ Updates the actual arrays to match the array_meta.
            In other words, synthronises the array_meta and actual array. """
        if isinstance(paths, str):
            paths = [paths]
        elif isinstance(paths, int):
            if paths >= 0:
                paths = [paths]
            else:
                paths = self.path_order
        self.__modify_array_meta(paths, metakey, value)
        self.__write_layer(paths, None, overwrite)
        self.update_name()

    def rechunk(self,
                chunks: Iterable,
                pth: Union[Iterable, int, str] = -1
                ):
        self.update_arrays(pth, metakey='chunks', value=chunks)
        return self

    def astype(self,
               dtype,
               pth: Union[Iterable, int, str] = -1
               ):
        self.update_arrays(pth, metakey='dtype', value=dtype)
        return self

    def asflat(self,
               pth: Union[Iterable, int, str] = -1
               ):
        self.update_arrays(pth, metakey='dimension_separator', value='.')
        return self

    def asnested(self,
                 pth: Union[Iterable, int, str] = -1
                 ):
        self.update_arrays(pth, metakey='dimension_separator', value='/')
        return self

    def recompress(self,
                   compressor,
                   pth: Union[Iterable, int, str] = -1
                   ):
        self.update_arrays(pth, metakey='compressor', value=compressor)
        return self


class ZarrayManipulations(ZarrayCore):
    def __init__(self,
                 group: zarr.Group
                 ):
        self.base = group
        ZarrayCore.__init__(self, self.base)

    # def copy(self,
    #          pth: str = None,
    #          rebase: bool = True,
    #          overwrite = False
    #          ):
    #     """ Copy the entire OME-Zarr to a new path. """
    #     if pth is None:
    #         target_base = tempfile.TemporaryDirectory()
    #         pth = os.path.join(target_base.name, 'results')
    #     store = zarr.DirectoryStore(pth)
    #     gr = zarr.group(store, overwrite = overwrite)
    #     for key in self.allkeys:
    #         gr[key] = self.base[key]
    #     for key, value in self.base.attrs.items():
    #         gr.attrs[key] = value
    #     if rebase:
    #         self.base = gr
    #         ZarrayCore.__init__(self, self.base)

    def __stage_inputs(self,
                     outpath: str
                     ):
        if self.basepath == outpath:
            self.in_place_write = True

    def reduce_pyramid(self, paths):
        """ Drop those paths that are not included in the paths variable. This method applies in-place.
            Thus consider copying your original OME_Zarr in advance. !!!
        """
        restpaths = [item for item in self.path_order if item not in paths]
        paths_minus_self = [item for item in paths if item not in self.path_order]
        if len(restpaths) == 0:
            return
        if len(paths_minus_self) > 0:
            raise ValueError('paths variable cannot contain a path that is not in the path_order.')
        for pth in restpaths:
            self.base.pop(pth)
            del self.array_meta[pth]
        newmeta = self.extract_dataset(paths)
        self.base.attrs['multiscales'] = newmeta
        ZarrayCore.__init__(self, self.base)

    def get_sliced_layers(self,
                          paths,
                          subset: dict = {'z': 25,   ### This will be rescaled from the top path in PATHs
                                          'y': (20, 140),
                                          'x': (20, 150)
                                          }
                          ):
        if paths is None: paths = self.path_order
        subsets = self.refactor_subset(paths, subset)
        layers, shapes, chunks = [], [], []
        for pth in paths:
            subset = subsets[pth]
            keys_, slices = utils.transpose_dict(subset)
            assert all([key in self.axis_order for key in keys_])
            arr = self.base[pth]
            indices = [self.axis_order.index(key) for key in keys_]
            arr_ = utils.index_nth_dimension(arr, indices, slices)
            layers.append(arr_)
            shapes.append(arr_.shape)
            chunks.append(arr_.chunksize)
        keys = [key for key in keys_ if not np.isscalar(subset[key])]
        print(keys)
        self.__sliced_meta = self.extract_dataset(paths=paths, axes=keys)
        print(self.__sliced_meta)
        return layers, shapes, chunks, self.__sliced_meta

    def __mobilise(self, ### IN PLACE KAYDEDERKEN VAR OLAN DATAYI SILMEDEN KAYDEDIYOR. BUNU DUZELT
                   newdir: str = None,
                   paths: Union[list, str, None] = None,
                   overwrite_arrays: bool = True,
                   overwrite_group: bool = False,
                   zarr_meta: [dict, None] = None,
                   rebase: bool = True,
                   only_meta: bool = False,
                   new_arrays: Union[Iterable, None] = None,
                   resliced = False
                   ):
        """"""
        if newdir is None:
            newdir = tempfile.TemporaryDirectory()
            basepath = os.path.join(newdir, self.basename)
        else:
            print('else is being used')
            basepath = newdir
        gr = zarr.group(basepath)
        newname = os.path.basename(basepath)
        if isinstance(paths, int):
            paths = str(paths)
        elif isinstance(paths, str):
            paths = [paths]
        elif paths is None:
            paths = self.path_order
        if zarr_meta is not None:
            keys, values = utils.transpose_dict(zarr_meta)
            for i, pth in enumerate(paths):
                if new_arrays is None:
                    new_array = self.base[pth]
                else:
                    new_array = new_arrays[i]
                self._ZarrayCore__set_array_meta(pth, keys, values, new_shape=new_array.shape)
        if only_meta:
            pass
        else:
            self._ZarrayCore__write_layer(paths, out_dir=basepath, overwrite_group = overwrite_group,
                                          overwrite_arrays = overwrite_arrays, new_arrays=new_arrays)
        if newdir == self.basepath:
            pths = list(self.array_meta.keys())
        else:
            pths = paths
        if not resliced:
            newmeta = self.extract_dataset(pths)
        else:
            newmeta = self.__sliced_meta
        newmeta[0]['name'] = newname
        gr.attrs['multiscales'] = newmeta
        if 'omero' in self.base_attrs.keys():
            gr.attrs['omero'] = self.base_attrs['omero']
        if rebase:
            ZarrayCore.__init__(self, gr)
        return self

    def __basemove(self,  ### Change the base of the OME_Zarr
                   newdir: str = None,
                   paths: Iterable = None,
                   subset = None,
                   overwrite: bool = True,
                   zarr_meta: Union[Dict, None] = None,
                   rebase = True
                   ):
        if newdir.startswith('http'):
            pass
        elif not newdir.startswith('/'):
            newdir = os.path.realpath(newdir)
        if self.basepath == newdir:
            overwrite_group = False
            if paths is None:
                pass
            else:
                self.reduce_pyramid(paths)
        else:
            overwrite_group = overwrite
        if paths is None: paths = self.path_order
        if subset is not None:
            layers, shapes, chunks, newmeta = self.get_sliced_layers(paths, subset)
            for pth, layer, shape, chunk in zip(paths, layers, shapes, chunks):
                self._ZarrayCore__set_array_meta(pth, metakeys = ['chunks'], values = [chunk], new_shape = shape)
            return self.__mobilise(newdir, paths = paths, overwrite_group = overwrite_group, overwrite_arrays = True,
                                   zarr_meta = zarr_meta, rebase = rebase, new_arrays = layers, resliced = True)
        else:
            return self.__mobilise(newdir, paths = paths, overwrite_group = overwrite_group, overwrite_arrays = True,
                                   zarr_meta = zarr_meta, rebase = rebase)

    def rebase(self,  ### Change the base of the OME_Zarr
               newdir: str = None,
               paths: Iterable = None,
               subset: Union[dict, None] = None,
               overwrite: bool = True,
               zarr_meta: Union[Dict, None] = None
               ): ### Works but looks very ugly, make it better.
        self.__basemove(newdir, paths, subset, overwrite, zarr_meta, rebase = True)

    def extract(self,  ### Change the base of the OME_Zarr
                newdir: str = None,
                paths: Iterable = None,
                subset: Union[dict, None] = None,
                overwrite: bool = True,
                zarr_meta: Union[Dict, None] = None
                ): ### Works but looks very ugly, make it better.
        self.__basemove(newdir, paths, subset, overwrite, zarr_meta, rebase = False)

    def reslice(self, ### Apply subsetting to all paths and in-place.
                subset: Union[dict, None]
                ):
        self.rebase(newdir = self.basepath, subset = subset)

    def set_array(self,
                  pth: Union[str, int],
                  arr: Union[np.ndarray, zarr.Array, da.Array],
                  scale: Iterable,
                  zarr_meta: Union[Dict, None] = None
                  ):
        """ Set an array to a path in the OME_Zarr hierarchy. Note that this will override any existing array in the path with the new one.
        """
        self.add_dataset(pth, scale)
        self.__mobilise(self.basepath, paths=[pth], zarr_meta=zarr_meta, new_arrays=[arr])
        self.sort_paths()
    # def rescale(self, ### TODO
    #             n_scales = 5,
    #             planewise = True
    #             ):
    #     self.set_to_highest_resolution()
    #     layers = rescale(self.base[self.path_order[0]], n_scales, planewise)
    #     scales = np.arange(n_scales)
    #     newpaths = [str(scale) for scale in scales]
