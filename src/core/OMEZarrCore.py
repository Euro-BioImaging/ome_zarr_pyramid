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
from numcodecs import Blosc

from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    Iterable,
    List,
    Optional
)

class ZarrayCore(_MultiMeta, _ImageLabelMeta): ### _MultiMeta and _MultiScales are external
    """ This class adds array-level metadata and update methods.
        Updates are possible to: 'chunks', 'compressor', 'dtype', 'dimension_separator'
    """

    def __init__(self,
                 ):
        assert self.is_multiscales, "This class is only compatible with MultiScales object instances."
        self.__collect_array_meta()

    def sort_paths(self):
        meta = self.multimeta
        args = utils.argsorter(self.resolution_paths)
        pths = [self.resolution_paths[i] for i in args]
        arrmeta = {i: self.array_meta[i] for i in pths}
        datasets = [meta[0]['datasets'][i] for i in args]
        meta[0]['datasets'] = datasets
        self.array_meta = arrmeta
        # self.multimeta = meta
        self.attrs['multiscales'] = self.multimeta

    def __collect_array_meta(self):
        res = {pth: {} for pth in self.resolution_paths}
        arrmeta_keys = ['chunks', 'shape', 'compressor', 'dtype', 'dimension_separator']
        for metakey in arrmeta_keys:
            for pth in self.resolution_paths:
                if metakey in self.layers[pth]._meta:
                    if metakey == 'compressor':
                        res[pth][metakey] = self.layers[pth].compressor
                    else:
                        res[pth][metakey] = self.layers[pth]._meta[metakey]
                elif metakey not in self.layers[pth]._meta:
                    if metakey == 'dimension_separator':
                        res[pth][metakey] = '.'
        self.array_meta = res

    def update_name(self): # TODO: maybe integrate with mobilise method
        meta = copy.deepcopy(self.multimeta)
        meta[0]['name'] = self.grname
        self.attrs['multiscales'] = meta

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
            out_dir = self.store.path
        for i, pth in enumerate(paths):
            arrmeta = self.array_meta[pth]
            if new_arrays is None:
                arr = self.layers[pth]
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
                if out_dir == self.store.path:
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
                    shape = np.array(self.layers[pth].shape)
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
                paths = self.resolution_paths
        if new_arrays is not None:
            for pth in paths:
                self.array_meta[pth] = self.array_meta[self.resolution_paths[0]]
        for i, pth in enumerate(paths):
            if metakey == 'chunks':
                if new_arrays is None:
                    shape = np.array(self.layers[pth].shape)
                else:
                    shape = np.array(new_arrays[i].shape)
                chunks = np.array(value)
                value = np.where(chunks > shape, shape, chunks)
                self.array_meta[pth][metakey] = tuple(value)
            else:
                self.array_meta[pth][metakey] = value
        # print(self.array_meta)
    @property
    def highest_resolution(self):
        pth = self.resolution_paths[0]
        return self.layers[pth]
    @property
    def lowest_resolution(self):
        pth = self.resolution_paths[-1]
        return self.layers[pth]

class ZarrayManipulations(ZarrayCore):
    def __init__(self,
                 ):
        ZarrayCore.__init__(self)
    # def __stage_inputs(self,
    #                  outpath: str
    #                  ):
    #     if self.grpath == outpath:
    #         self.in_place_write = True

    def reduce_pyramid(self, paths):
        """ Drop those paths that are not included in the paths variable. This method applies in-place.
            Thus consider backing up your original OME_Zarr in advance. !!!
        """
        restpaths = [item for item in self.resolution_paths if item not in paths]
        paths_minus_self = [item for item in paths if item not in self.resolution_paths]
        if len(restpaths) == 0:
            return
        if len(paths_minus_self) > 0:
            raise ValueError('paths variable cannot contain a path that is not in the resolution_paths.')
        for pth in restpaths:
            self.pop(pth)
            del self.array_meta[pth]
        newmeta = self.extract_dataset(paths)
        self.attrs['multiscales'] = newmeta
        self.rebase_multiscales(self.grpath) ### method from MultiScales class

    def _parse_subset(self, subset):
        if isinstance(subset, list):
            assert len(subset) == self.ndim, f"The length of the provided subset list is {len(subset)}, " \
                                             f"which does not match the current image's dimensionality {self.ndim}"
            res = {key: item for key, item in zip(self.axis_order, subset)}
        elif isinstance(subset, dict):
            res = subset
        else:
            raise TypeError(f"The subset must be either of type dict or list")
        return res

    def get_sliced_layers(self, ### BUNU TAMIR ET. SUBSET PARAMETRESINDE TÜM AXISLER YER ALMALI
                          paths,
                          subset: Union[dict, list] = {'z': 25,   ### This will be rescaled from the top path in PATHs
                                                       'y': (20, 140),
                                                       'x': (20, 150)
                                                       }
                          ):
        subset = self._parse_subset(subset)
        if paths is None: paths = self.resolution_paths
        print(f"input to refactoring is {subset}")
        subsets = self.refactor_subset(paths, subset)
        print(f"subsets are {subsets}")
        layers, shapes, chunks = [], [], []
        for pth in paths:
            subset = subsets[pth]
            keys_, slices = utils.transpose_dict(subset)
            assert all([key in self.axis_order for key in keys_])
            arr = self.layers[pth]
            indices = [self.axis_order.index(key) for key in keys_]
            arr_ = utils.index_nth_dimension(arr, indices, slices)
            # print(arr_.shape)
            layers.append(arr_)
            shapes.append(arr_.shape)
            chunks.append(arr_.chunksize)
        keys = []
        for key in self.axis_order:
            if key in subset.keys():
                if not np.isscalar(subset[key]):
                    keys.append(key)
            elif key not in subset.keys():
                keys.append(key)
        print(keys)
        self.__sliced_meta = self.extract_dataset(paths=paths, axes=keys)
        # print(self.__sliced_meta)
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
            grpath = os.path.join(newdir, self.grname)
        else:
            print('else is being used')
            grpath = newdir
        gr = zarr.group(grpath)
        newname = os.path.basename(grpath)
        if isinstance(paths, int):
            paths = str(paths)
        elif isinstance(paths, str):
            paths = [paths]
        elif paths is None:
            paths = self.resolution_paths
        if zarr_meta is not None:
            keys, values = utils.transpose_dict(zarr_meta)
            for i, pth in enumerate(paths):
                if new_arrays is None:
                    new_array = self.layers[pth]
                else:
                    new_array = new_arrays[i]
                self._ZarrayCore__set_array_meta(pth, keys, values, new_shape=new_array.shape)
        if only_meta:
            pass
        else:
            self._ZarrayCore__write_layer(paths, out_dir=grpath, overwrite_group = overwrite_group,
                                          overwrite_arrays = overwrite_arrays, new_arrays=new_arrays)
        if newdir == self.grpath:
            pths = list(self.array_meta.keys())
        else:
            pths = paths
        if not resliced:
            newmeta = self.extract_dataset(pths)
        else:
            newmeta = self.__sliced_meta
        newmeta[0]['name'] = newname
        gr.attrs['multiscales'] = newmeta
        if self.has_imagelabel_meta():
            gr.attrs['image-label'] = self.labelmeta
        if 'omero' in self.base_attrs.keys():
            gr.attrs['omero'] = self.base_attrs['omero']
        if rebase:
            self.rebase_multiscales(grpath)
        return self

    def __basemove(self,  ### Change the base of the OME_Zarr
                   newdir: str = None,
                   paths: Iterable = None,
                   subset = None,
                   overwrite: bool = True,
                   zarr_meta: Union[Dict, None] = None,
                   rebase: bool = True
                   ):
        if newdir.startswith('http'):
            pass
        elif not newdir.startswith('/'):
            newdir = os.path.realpath(newdir)
        if self.grpath == newdir:
            overwrite_group = False
            if paths is None:
                pass
            else:
                self.reduce_pyramid(paths)
        else:
            overwrite_group = overwrite
        if paths is None: paths = self.resolution_paths
        if subset is not None:
            print(f"subset in basemove is {subset}")
            layers, shapes, chunks, newmeta = self.get_sliced_layers(paths, subset)
            for pth, layer, shape, chunk in zip(paths, layers, shapes, chunks):
                self._ZarrayCore__set_array_meta(pth, metakeys = ['chunks'], values = [chunk], new_shape = shape)
            return self.__mobilise(newdir, paths = paths, overwrite_group = overwrite_group, overwrite_arrays = True,
                                   zarr_meta = zarr_meta, rebase = rebase, new_arrays = layers, resliced = True)
        else:
            return self.__mobilise(newdir, paths = paths, overwrite_group = overwrite_group, overwrite_arrays = True,
                                   zarr_meta = zarr_meta, rebase = rebase)

    def _rebase(self,  ### Extract OME-Zarr to a new path and switch the current python object to it
                newdir: str = None,
                paths: Iterable = None,
                subset: Union[dict, None] = None,
                overwrite: bool = True,
                zarr_meta: Union[Dict, None] = None
                ): ### Works but looks very ugly, make it better.
        """
        Extracts OME-Zarr to a new path and switches the current python object to it.
        Note that this method can only extract an active MultiScales object.
        As such, this method requires the current OME-Zarr object to be either of Hybrid or MultiScales type.
        """
        self.__basemove(newdir, paths, subset, overwrite, zarr_meta, rebase = True)

    def _extract(self,  ### Extract OME-Zarr without changing the current python object
                 newdir: str = None,
                 paths: Iterable = None,
                 subset: Union[dict, None] = None,
                 overwrite: bool = True,
                 zarr_meta: Union[Dict, None] = None
                 ): ### Works but looks very ugly, make it better.
        print(f"subset in _extract is {subset}")
        self.__basemove(newdir, paths, subset, overwrite, zarr_meta, rebase = False)

    def rechunk(self,
                chunks: Iterable,
                pth: Union[Iterable, int, str] = None
                ):
        self.rebase(newdir = self.grpath, paths = pth, zarr_meta = {'chunks': chunks})
        return self

    def astype(self,
               dtype,
               pth: Union[Iterable, int, str] = None
               ):
        self.rebase(newdir = self.grpath, paths = pth, zarr_meta = {'dtype': dtype})
        return self

    def asflat(self,
               pth: Union[Iterable, int, str] = None
               ):
        self.rebase(newdir = self.grpath, paths = pth, zarr_meta = {'dimension_separator': '.'})
        return self

    def asnested(self,
                 pth: Union[Iterable, int, str] = None
                 ):
        self.rebase(newdir = self.grpath, paths = pth, zarr_meta = {'dimension_separator': '/'})
        return self

    def recompress(self,
                   compressor,
                   pth: Union[Iterable, int, str] = -1
                   ):
        self.rebase(newdir = self.grpath, paths = pth, zarr_meta = {'compressor': compressor})
        return self

    def reslice(self, ### Apply subsetting to all paths and in-place.
                subset: Union[dict, None]
                ):
        self.rebase(newdir = self.grpath, subset = subset)

    ########################### Scale-related methods #####################################

    def _set_array(self, ### TODO: A LITTLE BIT CHECKED BUT STILL REQUIRES MORE WORKING ON.
                   pth: Union[str, int],
                   arr: Union[np.ndarray, zarr.Array, da.Array],
                   scale: Iterable,
                   zarr_meta: Union[Dict, None] = None
                   ):
        """
        Set an array to a path in the OME_Zarr hierarchy.
        Note that this will override any existing array in the path with the new one.
        """
        if zarr_meta is None:
            print("A new array is being set without providing the zarray metadata."
                  "Arbitrary zarr metadata is being assumed.")
            zarr_meta = {'chunks': (100, 100, 100),
                         'shape': (500, 500, 500),
                         'compressor': Blosc(cname='lz4',
                                             clevel=5,
                                             blocksize=0),
                         'dtype': np.dtype('float64'),
                         'dimension_separator': '.'
                         }
        self.add_dataset(pth, scale)
        self.__mobilise(self.grpath, paths=[pth], zarr_meta=zarr_meta, new_arrays=[arr])
        self.sort_paths()

    def add_layer(self, ### TODO: KALDIM. ÖNEMLI BIR NOKTA.
                  scale: Iterable = (2, 2, 2)
                  ):
        path_hr = self.resolution_paths[0]
        path_lr = self.resolution_paths[-1]
        layer = self.layers[path_hr]
        shape = np.array(layer.shape).flatten()
        assert len(shape) == len(scale), "Scale length must be exactly the same as the number of array dimensions."
        scl = np.array(scale).flatten()
        newshape = shape // scl
        newpth = int(path_lr) + 1
        self._set_array(newpth, )

    # def rescale(self, ### TODO
    #             n_scales = 5,
    #             planewise = True
    #             ):
    #     self.set_to_highest_resolution()
    #     layers = rescale(self.base[self.resolution_paths[0]], n_scales, planewise)
    #     scales = np.arange(n_scales)
    #     newpaths = [str(scale) for scale in scales]

    def split_indices(self,
                      dimension = 'z',
                      newdir = None
                      ):
        idx = self.axis_order.index(dimension)
        pathid = self.resolution_paths[0]
        span = np.arange(self.array_meta[pathid]['shape'][idx])
        if newdir is None: newdir = self.grdir
        for i in span:
            fpath = os.path.join(newdir, self.grname + '_' + dimension + '_' + str(i))
            self.extract(newdir = fpath, subset = {dimension: i})





