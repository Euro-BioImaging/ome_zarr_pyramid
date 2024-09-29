import os, zarr, json, shutil, copy, tempfile, numcodecs, warnings, itertools
from pathlib import Path

import numpy as np  # pandas as pd
import dask
import pandas as pd
from dask import array as da
# import s3fs
from skimage.transform import resize as skresize
from skimage import transform
# from rechunker import rechunk
from ome_zarr_pyramid.core import config, convenience as cnv
from numcodecs import Blosc

# from ome_zarr_pyramid.core.Hierarchy import OMEZarr, OMEZarrSample
# from ome_zarr_pyramid.core.Hierarchy import MultiScales
from ome_zarr_pyramid.core import config, convenience as cnv
from ome_zarr_pyramid.utils import multiscale_utils as mutils
from ome_zarr_pyramid.utils import assignment_utils as asutils
from typing import ( Union, Tuple, Dict, Any, Iterable, List, Optional )

def validate_multimeta(grp):
    assert isinstance(grp, zarr.Group), f'Input must be a zarr.Group object.'
    assert 'multiscales' in grp.attrs, f'Group attributes lack the `multiscales` field. Input is not a valid multiscales object.'
    multimeta = grp.attrs['multiscales']
    assert 'axes' in multimeta[0], f'Multiscales field lack the axis field. Input is not a valid multiscales object.'
    assert 'datasets' in multimeta[0], f'Multiscales field lack the datasets field. Input is not a valid multiscales object.'
    if not 'name' in multimeta[0]: warnings.warn(f'Multiscales field lack the name field. Input is not a valid multiscales object.')
    return multimeta

def validate_img_labelmeta(grp):
    assert isinstance(grp, zarr.Group), f'Input must be a zarr.Group object.'
    assert 'image-label' in grp.attrs, f'Group attributes lack the `image-label` field. Input is not a valid img_label object.'
    img_label_meta = grp.attrs['image-label']
    assert 'colors' in img_label_meta, f'img_label_meta field lack the colors field. Input is not a valid img_label object.'
    return img_label_meta

def validate_labelsmeta(grp):
    raise NotImplementedError(f"This function is not yet implemented.")

def pyramids_are_similar(pyr1, pyr2):
    assert isinstance(pyr1, Pyramid)
    assert isinstance(pyr2, Pyramid)
    assert pyr1.resolution_paths == pyr2.resolution_paths, f'Path names must be the same between the two pyramids.'
    assert all(pyr1.array_meta[pth]['shape'] == pyr2.array_meta[pth]['shape'] for pth in pyr1.resolution_paths), \
        f'Array shapes must be the same between the two pyramids for all resolutions.'
    return True

def validate_datasets(datasets):
    raise NotImplementedError(f"This function is not yet implemented.")

def aszarr(input: Union[str, Path, zarr.Group, zarr.Array]):
    assert isinstance(input, (str, Path, zarr.Group, zarr.Array))
    if isinstance(input, str):
        input = Path(input)
    if isinstance(input, (str, Path)):
        zarrobj = zarr.open(input, mode = 'a')
    else:
        zarrobj = input
    return zarrobj

def _parse_resolution_paths(paths):
    paths = [str(pth) for pth in paths]
    for pth in paths:
        try:
            _ = str(pth)
        except:
            raise TypeError(f"Resolution path names must be numerical.")
    return paths


class Multimeta:
    def __init__(self,
                 multimeta = None
                 ):
        if multimeta is None:
            self.multimeta = [{'axes': [],
                               'datasets': [],
                               'name': None,
                               'version': "0.4"
                               }]
        else:
            self.multimeta = multimeta

    @property
    def axis_order(self):
        try:
            ret = ''.join([item['name'] for item in self.multimeta[0]['axes']])
        except:
            ret = ''
        return ret

    @property
    def has_axes(self):
        return len(self.axis_order) > 0

    @property
    def is_imglabel(self):
        try:
            func = self.__getattribute__('set_image_label_metadata')
            return func()
        except:
            return False

    @property
    def ndim(self):
        return len(self.axis_order)

    @property
    def nlayers(self):
        return len(self.resolution_paths)

    @property
    def unit_list(self):
        try:
            if self.has_axes:
                l = []
                for item in self.multimeta[0]['axes']:
                    if 'unit' in item.keys():
                        l.append(item['unit'])
                    else:
                        default_unit = config.unit_map[item['name']]
                        l.append(default_unit)
                return l
            else:
                return config.AxisNotFoundException
        except:
            return config.NotMultiscalesException

    @property
    def tag(self):
        return self.multimeta[0]['name']

    def retag(self,
              new_tag: str
              ):
        self.multimeta[0]['name'] = new_tag
        return self

    def _add_axis(self,
                  name: str,
                  unit: str = None,
                  index: int = -1,
                  overwrite: bool = False
                  ):
        if name in self.axis_order:
            if not overwrite:
                raise ValueError(f'{name} axis already exists.')

        def axmake(name, unit):
            if unit is None:
                axis = {'name': name, 'type': config.type_map[name]}
            else:
                axis = {'name': name, 'type': config.type_map[name], 'unit': unit}
            return axis

        if index == -1:
            index = len(self.multimeta[0]['axes'])
        if not self.has_axes:
            self.multimeta[0]['axes'] = []
            index = 0
        axis = axmake(name, unit)
        if overwrite:
            self.multimeta[0]['axes'][index] = axis
        else:
            self.multimeta[0]['axes'].insert(index, axis)

    def parse_axes(self,
                   axis_order,
                   unit_list: Union[list, tuple] = None,
                   overwrite: bool = None
                   ):
        if len(self.multimeta[0]['axes']) > 0:
            if not overwrite:
                raise ValueError('The current axis metadata is not empty. Cannot overwrite.')
        if unit_list is None:
            unit_list = [None] * len(axis_order)
        elif unit_list == 'default':
            unit_list = [config.unit_map[i] for i in axis_order]
        assert len(axis_order) == len(unit_list), 'Unit list and axis order must have the same length.'
        for i, n in enumerate(axis_order):
            self._add_axis(name = n,
                           unit = unit_list[i],
                           index = i,
                           overwrite = overwrite
                           )
        return self

    def rename_paths(self):
        for i, _ in enumerate(self.multimeta[0]['datasets']):
            newkey = str(i)
            oldkey = self.multimeta[0]['datasets'][i]['path']
            self._arrays[newkey] = self._arrays.pop(oldkey)
            self._array_meta_[newkey] = self._array_meta_.pop(oldkey)
            self.multimeta[0]['datasets'][i]['path'] = newkey
        return self

    @property
    def resolution_paths(self):
        try:
            paths = [item['path'] for item in self.multimeta[0]['datasets']]
            return sorted(paths)
        except:
            return []

    def add_dataset(self,
                    path: Union[str, int],
                    scale: Iterable[Union[int, float]],
                    translation: Iterable[Union[int, float]] = None,
                    overwrite: bool = False
                    ):
        if not overwrite:
            assert path not in self.resolution_paths, 'Path already exists.'
        assert scale is not None, f"The parameter scale must not be None"
        assert isinstance(scale, (tuple, list))
        transforms = {'scale': scale, 'translation': translation}
        dataset = {'coordinateTransformations': [{f'{key}': list(value), 'type': f'{key}'}
                                                 for key, value in transforms.items()
                                                 if not value is None
                                                 ],
                   'path': str(path)
                   }
        if path in self.resolution_paths:
            idx = self.resolution_paths.index(path)
            self.multimeta[0]['datasets'][idx] = dataset
        else:
            self.multimeta[0]['datasets'].append(dataset)
        args = np.argsort([int(pth) for pth in self.resolution_paths])
        self.multimeta[0]['datasets'] = [self.multimeta[0]['datasets'][i] for i in args]

    @property
    def transformation_types(self):
        transformations = self.multimeta[0]['datasets'][0]['coordinateTransformations']
        return [list(dict.keys())[0] for dict in transformations]

    @property
    def has_translation(self):
        return 'translation' in self.transformation_types

    def get_scale(self,
                  pth: Union[str, int]
                  ):
        pth = cnv.asstr(pth)
        idx = self.resolution_paths.index(pth)
        return self.multimeta[0]['datasets'][idx]['coordinateTransformations'][0]['scale']

    def set_scale(self,
                  pth: Union[str, int] = 'auto',
                  scale: Union[tuple, list] = 'auto', #TODO: add dict option
                  hard = False
                  ):
        if isinstance(scale, tuple):
            scale = list(scale)
        elif hasattr(scale, 'tolist'):
            scale = scale.tolist()
        if pth == 'auto':
            pth = self.refpath
        if scale == 'auto':
            pth = self.scales[pth]
        idx = self.resolution_paths.index(pth)
        self.multimeta[0]['datasets'][idx]['coordinateTransformations'][0]['scale'] = scale
        if hard:
            self.gr.attrs['multiscales'] = self.multimeta
        return

    def update_scales(self,
                      reference_scale,
                      hard = True
                      ):
        for pth, factor in self.scale_factors.items():
            new_scale = np.multiply(factor, reference_scale)
            self.set_scale(pth, new_scale, hard)
        return self

    def update_unitlist(self, ###TODO: test this
                 unitlist=None,
                 hard=False
                 ):
        if isinstance(unitlist, tuple):
            unitlist = list(unitlist)
        assert isinstance(unitlist, list)
        self.parse_axes(self.axis_order, unitlist, overwrite = True)
        if hard:
            self.gr.attrs['multiscales'] = self.multimeta
        return

    @property
    def scales(self):
        scales = {}
        for pth in self.resolution_paths:
            scl = self.get_scale(pth)
            scales[pth] = scl
        return scales

    def get_translation(self,
                        pth: Union[str, int]
                        ):
        if not self.has_translation: return None
        pth = cnv.asstr(pth)
        idx = self.resolution_paths.index(pth)
        return self.multimeta[0]['datasets'][idx]['coordinateTransformations'][1]['translation']

    def set_translation(self,
                        pth: Union[str, int],
                        translation
                        ):
        if isinstance(translation, np.ndarray):
            translation = translation.tolist()
        idx = self.resolution_paths.index(pth)
        if len(self.multimeta[0]['datasets'][idx]['coordinateTransformations']) < 2:
            self.multimeta[0]['datasets'][idx]['coordinateTransformations'].append({'translation': None, 'type': 'translation'})
        self.multimeta[0]['datasets'][idx]['coordinateTransformations'][1]['translation'] = translation

    @property
    def translations(self):
        translations = {}
        for pth in self.resolution_paths:
            translation = self.get_translation(pth)
            translations[pth] = translation
        return translations

    def del_axis(self,
                 name: str
                 ):
        if name not in self.axis_order:
            raise ValueError(f'The axis "{name}" does not exist.')
        idx = self.axis_order.index(name)
        self.multimeta[0]['axes'].pop(idx)
        for pth in self.resolution_paths:
            scale = self.get_scale(pth)
            scale.pop(idx)
            self.set_scale(pth, scale)
            translation = self.get_translation(pth)
            if translation is not None:
                translation.pop(idx)
                self.set_translation(pth, translation)

    @property
    def label_paths(self):
        try:
            return list(self.labels.keys())
        except:
            return []

    @property
    def has_label_paths(self):
        try:
            return self.labels is not None
        except:
            return False

    @property
    def label_meta(self):
        if self.has_label_paths:
            meta = {'labels': []}
            for name in self.label_paths:
                meta['labels'].append(name)
            return meta
        else:
            return None

# class LabelMeta:
#     def __init__(self,
#                  labeldict: dict = None
#                  ):
#         self._label_paths = None
#         if labeldict is not None:
#             self.validate_labels_meta(labeldict)
#     @property
#     def label_paths(self):
#         try:
#
#         return list(labels.keys())
#     def validate_labels_meta(self,
#                              labeldict: dict
#                              ):
#         assert 'labels' in labeldict.keys(), f"A label object requires 'labels' key to be available."
#         assert cnv.is_valid_json(labeldict), f"The object is not a valid json."
#         labelpaths = labeldict['labels']
#         assert isinstance(labelpaths, list), f"The 'labelpaths' attribute must be of type list."
#         self._label_paths = labelpaths
#         self.label_dict = labeldict

class ImageLabelMeta:
    def __init__(self,
                 img_labeldict: dict = None
                 ):
        self.img_label_meta = img_labeldict

    @property
    def img_label_meta(self):
        ret = {'colors': [],
               'version': '0.4'
               }
        try:
            if self._img_label_meta is None:
                self._img_label_meta = ret
        except:
            self._img_label_meta = ret
        return self._img_label_meta

    @img_label_meta.setter
    def img_label_meta(self,
                       img_labeldict: dict
                       ):
        is_imglabel = self.set_image_label_metadata(img_labeldict)
        if not is_imglabel:
            raise TypeError(f"The image-label metadata cannot be set.")

    @property
    def is_imglabel(self):
        return self.set_image_label_metadata()

    def set_image_label_metadata(self,
                                  img_labeldict: dict = None
                                  ):
        if img_labeldict is None:
            img_labeldict = copy.deepcopy(self._img_label_meta)

        if cnv.is_valid_json(img_labeldict):
            self._img_label_meta = img_labeldict
        else:
            self._img_label_meta = cnv.turn2json(img_labeldict)
        if not cnv.is_valid_json(self._img_label_meta):
            warnings.warn(f"The object is not a valid json.")
            return False
        if not self._has_label_values():
            warnings.warn(f"With no label value available, the img_label object is not valid.")
            return False
        if not all([isinstance(value, int) for value in self._label_values()]):
            warnings.warn(f"All label values must be of type int.")
            return False
        try:
            self.sort_labels()
            if self._has_rgba_meta():
                for lbl, rgba in zip(self.label_values, self.rgba):
                    if not hasattr(self, 'labels2rgba'):
                        self.labels2rgba = {}
                    self.labels2rgba[lbl] = rgba
        except:
            warnings.warn(f"RGBA parsing is not successful.")
            return False
        # meta.set_image_label_metadata(image_label_meta['image-label'])
        return True

    @property
    def img_label_fields(self):
        return list(self.img_label_meta.keys())

    @property
    def colors(self):
        colormeta = copy.deepcopy(self.img_label_meta['colors'])
        return colormeta

    @colors.setter
    def colors(self,
               colors: Iterable[dict]
               ):
        self._img_label_meta['colors'] = colors

    @property
    def label_values(self):
        labellist = [item['label-value'] for item in self.colors]
        return labellist

    @property
    def prop_label_values(self):
        labellist = [item['label-value'] for item in self.properties]
        return labellist

    def _label_values(self):
        try:
            return [item['label-value'] for item in self._img_label_meta['colors']]
        except:
            raise ValueError('No label values available.')

    def _has_label_values(self):
        labellist = self._label_values()
        return (len(labellist) > 0)

    @property
    def has_label_values(self):
        return [item['label-value'] for item in self.colors]

    def _has_rgba_meta(self):
        try:
            return all(['rgba' in item.keys() for item in self._img_label_meta['colors']])
        except:
            raise ValueError('No label values available.')

    @property
    def has_rgba_meta(self):
        return all(['rgba' in item.keys() for item in self.colors])

    @property
    def rgba(self):
        try:
            rgba = [item['rgba'] for item in self.colors]
            return rgba
        except:
            return None

    def sort_labels(self):
        labellist = self.label_values
        argslabels = np.argsort(labellist)
        self.colors = [self.colors[arg] for arg in argslabels]
        if self.has_properties:
            proplabellist = self.prop_label_values
            argslabels = np.argsort(proplabellist)
            self.properties = [self.properties[arg] for arg in argslabels]

    def label_index(self,
                    idx
                    ):
        self.sort_labels()
        return self.label_values.index(idx)

    def get_color(self, label):
        return self.labels2rgba[label]

    @property
    def properties(self):
        try:
            propmeta = copy.deepcopy(self.img_label_meta['properties'])
            return propmeta
        except:
            return None

    @properties.setter
    def properties(self,
                   properties: Iterable[dict]
                   ):
        self._img_label_meta['properties'] = properties

    @property
    def has_properties(self):
        if not 'properties' in self.img_label_fields:
            return False
        else:
            cond0 = len(self.img_label_meta['properties']) > 0
            cond1 = all(['label-value' in item.keys() for item in self.img_label_meta['properties']])
            cond2 = all([item['label-value'] in self.label_values for item in self.img_label_meta['properties']])
            return cond0 & cond1 & cond2

    def add_color(self,
                  color: dict,
                  overwrite: bool = False,
                  verbose: bool = False
                  ):
        assert isinstance(color, dict), f"color must be of type dict."
        assert 'label-value' in color.keys(), f"'label-value' is not available in the keys."
        for key in color.keys():
            assert key in ['label-value', 'rgba'], f"key must be either of 'label-value' or 'rgba', not {key}."
        if color['label-value'] not in self.label_values:
            self._img_label_meta['colors'].append(color)
            self.sort_labels()
        else:
            if overwrite:
                idx = self.label_index(color['label-value'])
                self._img_label_meta['colors'].pop(idx)
                self._img_label_meta['colors'].insert(idx, color)
                if verbose:
                    warnings.warn(f"Existing color metadata is being overwritten for label: {color['label-value']}")
            else:
                if verbose:
                    warnings.warn(f"Color metadata exists for the label: {color['label-value']}. Skipping.")

    def add_property(self,
                     prop: dict,
                     overwrite: bool = False,
                     verbose: bool = False
                     ):
        assert isinstance(prop, dict), f"prop must be of type dict."
        assert 'label-value' in prop.keys(), f"'label-value' is not available in the keys."
        if not self.has_properties:
            self._img_label_meta['properties'] = []
        assert prop['label-value'] in self.label_values, f"The label-value {prop['label-value']} does not exist."
        if prop['label-value'] not in self.prop_label_values:
            self._img_label_meta['properties'].append(prop)
            self.sort_labels()
        else:
            if overwrite:
                idx = self.label_index(prop['label-value'])
                self._img_label_meta['properties'][idx - 1] = prop

                if verbose:
                    warnings.warn(f"Existing property metadata is being overwritten for label: {prop['label-value']}")
            else:
                if verbose:
                    warnings.warn(f"Property metadata exists for the label: {prop['label-value']}. Skipping.")

    # def set_meta(self,
    #              label: int,
    #              metakey: str,
    #              metaval: Union[Iterable, int, float, str]
    #              ):
    #     if label not in self.label_values:
    #         if metakey == 'rgba':
    #             assert len(metaval) == 3, f"rgba value must be an iterable of length 3."
    #             labeldict = {'label-value': label,
    #                          'rgba': metaval
    #                          }
    #             self.add_color(labeldict)
    #         else:
    #             labeldict = {'label-value': label,
    #                          metakey: metaval
    #                          }
    #             self.add_property(labeldict)
    #     else:
    #         idx = self.label_index(label)
    #         if metakey == 'rgba':
    #             assert len(metaval) == 3, f"rgba value must be an iterable of length 3."
    #             self._img_label_meta['colors'][idx]['rgba'] = metaval
    #         else:
    #             self._img_label_meta['properties'][idx][metakey] = metaval

    def _add_imglabel_meta(self,
                          colors: Iterable[dict],
                          props: Iterable[dict] = None,
                          overwrite: bool = False
                          ):
        for color in colors:
            self.add_color(color, overwrite)
        if props is not None:
            for prop in props:
                self.add_property(prop, overwrite)

# image_label_meta = dict(h.gr.labels.original.attrs)
# meta = ImageLabelMeta( image_label_meta['image-label'])
# # meta.set_image_label_metadata(image_label_meta['image-label'])
# meta.add_property({'label-value': 1, 'area': 100}, verbose = True)
# meta.add_property({'label-value': 2, 'area': 200}, verbose = True, overwrite = True)
# meta.add_property({'label-value': 3, 'area': 100}, verbose = True)
# meta.add_property({'label-value': 4, 'area': 100}, verbose = True, overwrite=True)
# meta.add_property({'label-value': 5, 'area': 100}, verbose = True, overwrite=False)
# meta.add_property({'label-value': 6, 'area': 100}, verbose = True, overwrite=False)
# meta.add_property({'label-value': 5, 'area': 300}, verbose = True, overwrite=True)
#
# meta.img_label_meta = image_label_meta['image-label']
#
# meta.sort_labels()

class Operations:
    def __add__(self, # TODO: support scalar
                other
                ):
        self.to_dask()
        if isinstance(other, Pyramid):
            other.to_dask()
            pyramids_are_similar(self, other)
        elif isinstance(other, (int, float)):
            pass
        else:
            raise TypeError(f"Other must either be an instance of Pyramid or be a scalar value.")
        pyr = Pyramid()
        for pth in self.resolution_paths:
            if isinstance(other, Pyramid):
                res = self.layers[pth] + other.layers[pth]
            elif isinstance(other, (int, float)):
                res = self.layers[pth] + other
            else:
                raise TypeError(f"Other must either be an instance of Pyramid or be a scalar value.")
            if not pyr.has_axes:
                pyr.parse_axes(self.axis_order, self.unit_list)
            scale = self.get_scale(pth)
            translation = self.get_translation(pth)
            zarr_meta = self.array_meta[pth]
            pyr.add_layer(res.astype(self.dtype), pth, scale, translation, zarr_meta)
        return pyr

    def __radd__(self, other):
        return self.__radd__(other)

    def __sub__(self,
                other
                ):
        self.to_dask()
        if isinstance(other, Pyramid):
            other.to_dask()
            pyramids_are_similar(self, other)
        elif isinstance(other, (int, float)):
            pass
        else:
            raise TypeError(f"Other must either be an instance of Pyramid or be a scalar value.")
        pyr = Pyramid()
        for pth in self.resolution_paths:
            if isinstance(other, Pyramid):
                res = self.layers[pth] - other.layers[pth]
            elif isinstance(other, (int, float)):
                res = self.layers[pth] - other
            else:
                raise TypeError(f"Other must either be an instance of Pyramid or be a scalar value.")
            if not pyr.has_axes:
                pyr.parse_axes(self.axis_order, self.unit_list)
            scale = self.get_scale(pth)
            translation = self.get_translation(pth)
            zarr_meta = self.array_meta[pth]
            pyr.add_layer(res.astype(self.dtype), pth, scale, translation, zarr_meta)
        return pyr

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self,
                other
                ):
        self.to_dask()
        if isinstance(other, Pyramid):
            other.to_dask()
            pyramids_are_similar(self, other)
        elif isinstance(other, (int, float)):
            pass
        else:
            raise TypeError(f"Other must either be an instance of Pyramid or be a scalar value.")
        pyr = Pyramid()
        for pth in self.resolution_paths:
            if isinstance(other, Pyramid):
                res = self.layers[pth] * other.layers[pth]
            elif isinstance(other, (int, float)):
                res = self.layers[pth] * other
            else:
                raise TypeError(f"Other must either be an instance of Pyramid or be a scalar value.")
            if not pyr.has_axes:
                pyr.parse_axes(self.axis_order, self.unit_list)
            scale = self.get_scale(pth)
            translation = self.get_translation(pth)
            zarr_meta = self.array_meta[pth]
            pyr.add_layer(res.astype(self.dtype), pth, scale, translation, zarr_meta)
        return pyr

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self,
                other
                ):
        self.to_dask()
        if isinstance(other, Pyramid):
            other.to_dask()
            pyramids_are_similar(self, other)
        elif isinstance(other, (int, float)):
            pass
        else:
            raise TypeError(f"Other must either be an instance of Pyramid or be a scalar value.")
        pyr = Pyramid()
        for pth in self.resolution_paths:
            if isinstance(other, Pyramid):
                res = self.layers[pth] / other.layers[pth]
            elif isinstance(other, (int, float)):
                res = self.layers[pth] / other
            else:
                raise TypeError(f"Other must either be an instance of Pyramid or be a scalar value.")
            if not pyr.has_axes:
                pyr.parse_axes(self.axis_order, self.unit_list)
            scale = self.get_scale(pth)
            translation = self.get_translation(pth)
            zarr_meta = self.array_meta[pth]
            pyr.add_layer(res.astype(self.dtype), pth, scale, translation, zarr_meta)
        return pyr

    def __rtruediv__(self, other):
        return self.__rtruediv__(other)

    ##### Arythmetic operations
    def sum(self,
            pth: str = None
            ):
        if pth is None:
            pth = self.refpath
        return np.sum(self.layers[pth]).compute()

    def min(self,
            pth: str = None
            ):
        if pth is None:
            pth = self.refpath
        return np.min(self.layers[pth]).compute()

    def max(self,
            pth: str = None
            ):
        if pth is None:
            pth = self.refpath
        return np.max(self.layers[pth]).compute()

    def mean(self,
             pth: str = None
             ):
        if pth is None:
            pth = self.refpath
        return np.mean(self.layers[pth]).compute()

    def median(self,
               pth: str = None
               ):
        return np.median(self.layers[pth]).compute()

    def compute_properties(self,
                           label_name: str,
                           pth: str = None,
                           features: list = ['label', 'area', 'intensity_max', 'intensity_mean'],
                           return_df: bool = False
                           ):
        assert self.has_label_paths, ValueError(f"The method compute_properties requires at least one LabelPyramid.")
        assert label_name in self.label_paths, ValueError(f"The label name {label_name} does not exist.")
        if pth is None:
            pth = self.refpath
        lpyr = self.labels[label_name]
        arr = self.layers[pth]
        ret = lpyr.compute_properties(pth,
                                      image = arr,
                                      features = features,
                                      return_df = return_df)
        if return_df:
            return ret

class Pyramid(Multimeta, Operations):
    def __init__(self):
        Multimeta.__init__(self)
        self._arrays = {}
        self._array_meta_ = {}
        if len(self.resolution_paths) > 0:
            for pth in self.resolution_paths:
                if pth not in self._array_meta_.keys():
                    self._array_meta_[pth] = {}

    @property
    def refroot(self): # The root of the reference array.
        if isinstance(self.refarray.store, zarr.DirectoryStore):
            return self.refarray.store.path
        else:
            return None

    @property
    def pyramid_root(self):
        if not self.refroot is None:
            try:
                arr = zarr.open_array(self.refroot, mode = 'r')
                return os.path.dirname(self.refroot)
            except:
                grp = zarr.open_group(self.refroot, mode = 'r')
                return self.refroot
        else:
            return None

    @property
    def basename(self):
        if self.pyramid_root is not None:
            return os.path.basename(self.pyramid_root)
        else:
            return None

    def __str__(self):
        return f"Pyramidal OME-Zarr with {self.nlayers} resolution layers."

    def __repr__(self):
        # return f"Pyramidal OME-Zarr with {self.nlayers} resolution layers."
        if self.pyramid_root is not None:
            return self.pyramid_root
        else:
            return f"Pyramidal OME-Zarr with {self.nlayers} resolution layers."

    def __len__(self):
        return len(self.resolution_paths)

    def __getitem__(self,
                    pth: Union[str, int, slice, list, tuple]
                    ):
        assert isinstance(pth, (str, int, slice)), f'A path must be specified with either of the types: int, str, slice, list, tuple'
        if isinstance(pth, int):
            pth = str(pth)
        if isinstance(pth, slice):
            if pth.step is None:
                step = 1
            else:
                step = pth.step
            keys = [str(i) for i in range(pth.start, pth.stop, step) if str(i) in self.resolution_paths]
            return [self.layers[i] for i in keys]
        if isinstance(pth, (list, tuple)):
            keys = [str(i) for i in pth if str(i) in self.resolution_paths]
            return [self.layers[i] for i in keys]
        return self.layers[pth]

    def __setitem__(self,
                    key,
                    value
                    ):
        """TODO:
        Connected to the add_layer method. Add only if the resolution path already exists.
        CANNOT CREATE A NEW RESOLUTION LAYER.
        Update the array meta with the new chunks and data type.
        This can also be a zarr data. Then update zarr metadata accordingly.
        """
        raise NotImplementedError(f"This method is not yet implemented.")

    def __delitem__(self,
                    key: Union[str, int] # add slice support
                    ):
        """Connected to the del_layer method."""
        if isinstance(key, int): key = str(key)
        return self.del_layer(key)

    ### TODO: add other dunder methods

    def copy(self,
             new_dir,
             paths: Union[list, tuple] = None,
             **kwargs
             # overwrite = False,
             # label_paths = None, # make a config attribute for this
             # label_resolution_paths = 'all'
             ):
        pyr = Pyramid()
        if paths is None:
            paths = self.resolution_paths
        paths = _parse_resolution_paths(paths)
        for pth in paths:
            res = self.layers[pth]
            if not pyr.has_axes:
                pyr.parse_axes(self.axis_order, unit_list=self.unit_list)
            scale = self.get_scale(pth)
            translation = self.get_translation(pth)
            zarr_meta = self.array_meta[pth]
            pyr.add_layer(res, pth, scale, translation, zarr_meta)
            pyr.multimeta[0]['name'] = self.tag
        try:
            syncdir = os.path.dirname(self.syncdir)
        except:
            syncdir = None
        pyr.to_zarr(new_dir, syncdir = syncdir, **kwargs)
        return pyr

    @property
    def refpath(self):
        """The path to the full-resolution array."""
        if len(self.resolution_paths) == 1:
            return self.resolution_paths[0]
        elif config.refpath not in self.resolution_paths:
            return self.resolution_paths[0]
        return config.refpath

    @property
    def refarray(self):
        """The full-resolution array."""
        return self.layers[self.refpath]

    @property
    def refscale(self):
        return self.scales[self.refpath]

    @property
    def shape(self):
        return self.array_meta[self.refpath]['shape']

    @property
    def chunks(self):
        return self.array_meta[self.refpath]['chunks']

    @property
    def compressor(self):
        return self.array_meta[self.refpath]['compressor']

    @property
    def dimension_separator(self):
        return self.array_meta[self.refpath]['dimension_separator']

    @property
    def dtype(self):
        return self.array_meta[self.refpath]['dtype']

    @property
    def physical_size(self):
        return self.layer_meta[self.refpath]['physical_size']

    def update_translations(self,
                            new_translation: (dict, list, tuple) # translation for ref path
                            ):
        if isinstance(new_translation, dict):
            dct = {}
            for ax in self.axis_order:
                if ax in new_translation.keys():
                    dct[ax] = new_translation[ax]
                else:
                    dct[ax] = 0 # not translated along this axis
            new_translation = list(dct.values())
        new_translation = np.array(new_translation)
        for key, scale_factor in self.scales.items():
            self.set_translation(key, new_translation)
        return self

    def index(self,
              axes = 't',
              scalar_sensitive=True
              ):
        assert hasattr(axes, '__len__'), f"The parameter axes must be a string"
        assert all(isinstance(i, str) for i in axes), f"The parameter axes must be a string"
        indices = []
        for axis in axes:
            if axis in self.axis_order:
                indices.append(self.axis_order.index(axis))
            else:
                raise ValueError(f'The axis {axis} does not exist.')
        if (len(indices) == 1) and scalar_sensitive:
            indices = indices[0]
        return indices

    def axislen(self,
                ax: str,
                pth: str = None
                ):
        if pth is None:
            pth = self.refpath
        shape = self.array_meta[pth]['shape']
        indices = self.index(ax, scalar_sensitive = False)
        return [shape[idx] for idx in indices]

    @property
    def gr(self):
        if hasattr(self, '_gr'):
            return self._gr
        else:
            if self.pyramid_root is not None:
                self._gr = zarr.open_group(self.pyramid_root, mode = 'a')
                return self._gr
            else:
                return None

    @property
    def synchronizer(self):
        if hasattr (self.gr, 'synchronizer'):
            return self.gr.synchronizer
        else:
            return None
    @property
    def syncdir(self):
        ret = None
        if hasattr (self.gr, 'synchronizer'):
            if self.gr.synchronizer is not None:
                ret = self.gr.synchronizer.path
        return ret
    def activate_synchronizer(self, syncdir, basename):
        synchronizer = self._parse_synchronizer(syncdir, basename)
        self._gr = zarr.open_group(self.pyramid_root, mode = 'r+', synchronizer = synchronizer)
        for pth in self.resolution_paths:
            self._arrays[pth] = self.gr[pth]
        return self

    #  main pyramid method
    def from_zarr(self,
                  input_path: Union[str, Path, zarr.Group],
                  paths = None,
                  keep_array_type: bool = True,
                  syncdir = 'same',
                  # include_imglabels = False,
                  ):

        self._pyramid_root = input_path
        if syncdir == 'same':
            grp = zarr.open_group(input_path, mode='r+')
        else:
            root = os.path.dirname(syncdir)
            basename = os.path.basename(syncdir)
            synchronizer = self._parse_synchronizer(root, basename)
            grp = zarr.open_group(input_path, mode='r+', synchronizer = synchronizer)
        multimeta = validate_multimeta(grp)
        mm = Multimeta(multimeta)
        self._gr = grp
        if paths is None:
            paths = list(self.gr.array_keys())
        paths = _parse_resolution_paths(paths)
        for pth in paths:
            if pth in self.gr.keys():
                arr = self.gr[pth]
            elif int(pth) in self.gr.keys():
                arr = self.gr[int(pth)]
            else:
                warnings.warn(f"The requested resolution path {pth} does not exist in the target input_path.")
            if not self.has_axes:
                self.parse_axes(mm.axis_order, unit_list=mm.unit_list)
            self.add_layer(arr, pth, mm.scales[pth], mm.translations[pth], keep_array_type = keep_array_type)
        # if include_imglabels:
        #     if not 'labels' in self.gr.keys():
        #         warnings.warn(f'No groups named "labels" exist.')
        #     self._labels_root = os.path.join(input_path, 'labels')
        #     label_paths = list(self.gr['labels'].group_keys())
        #     for name in label_paths:
        #         pyr = LabelPyramid()
        #         pyrpath = os.path.join(self._labels_root, name)
        #         pyr.from_zarr(pyrpath, keep_array_type = keep_array_type)
        #         self.add_imglabel(pyr, name)
        return self

    def add_imglabel(self,
                     imglabel,
                     name: str,
                     label_resolution_paths: Union[tuple, list, str] = 'all'
                     ):
        if not self.has_label_paths:
            self.labels = {}
        assert isinstance(imglabel, LabelPyramid), f"The provided input must be of type LabelPyramid"
        assert imglabel.is_imglabel, f"The provided input must have img-label metadata."
        if label_resolution_paths == 'all':
            self.labels[name] = imglabel
        else:
            self.labels[name] = imglabel.shrink(label_resolution_paths)

    def get_compressor(self,
                       name: str = 'blosc'
                       ):
        compressors = {
            'blosc': numcodecs.Blosc(),
            'zlib': numcodecs.Zlib(),
            'null': None
        }
        return compressors[name]

    @property
    def chunkdict(self):
        dict = {}
        for ax in self.axis_order:
            idx = self.axis_order.index(ax)
            dict[ax] = self.chunks[idx]
        return dict

    def save_dask_layer(self,
                   target_path,
                   pth='0',
                   region_sizes: dict = None,
                   overwrite = False,
                   verbose = False
                   ):

        array = self.layers[pth].astype(self.array_meta[pth]['dtype'])
        arrpath = os.path.join(target_path, pth)
        store = zarr.DirectoryStore(arrpath)
        shape = list(self.array_meta[pth]['shape'])
        if region_sizes is None:
            region_sizes = {ax: 1 if ax in 'tc' else shape[self.index(ax)] for ax in self.axis_order}
        axes = self.axis_order
        slices = []
        for ax in axes:
            size = shape[self.index(ax)]
            slcsize = region_sizes[ax]
            slcs = tuple([slice(loc, loc + slcsize) for loc in range(0, size, slcsize)])
            slices.append(slcs)
        slices = tuple(slices)
        try:
            _ = zarr.open_array(store, mode='r')
            zarr_empty = zarr.open_array(store, mode='a')
        except:
            zarr_empty = zarr.create(
                shape=self.array_meta[pth]['shape'],
                chunks=self.array_meta[pth]['chunks'],
                dtype=self.array_meta[pth]['dtype'],
                compressor=self.array_meta[pth]['compressor'],
                store=store,
                overwrite=overwrite,
                dimension_separator=self.array_meta[pth]['dimension_separator']
            )
        combins = list(itertools.product(*slices))
        total = len(combins)
        tasks = []
        for i, slc in enumerate(combins):
            if isinstance(array, np.ndarray):
                array = da.from_array(array, chunks = self.array_meta[pth]['chunks']).astype(self.array_meta[pth]['dtype'])
            if isinstance(array, da.Array):
                task = array[slc].to_zarr(url=zarr_empty, region=slc, compute=False, dimension_separator=self.array_meta[pth]['dimension_separator'])
            if verbose:
                print(f"The region {i + 1} of a total of {total} from layer {pth} is added to tasks.")
            tasks.append(task)
        for i, task in enumerate(tasks):
            task.compute()
            if verbose:
                print(f"The region {i + 1} of a total of {total} from layer {pth} is saved.")
        return

    # def compute(self, paths = None):
    #     if paths is None:
    #         paths = self.layers.keys()
    #     for pth in paths:
    #         self._arrays[pth] = self.layers[pth].compute()
    #     return self

    def to_dask(self,
                deepcopy = False
                ):
        if deepcopy:
            store = self.pyramid_root
            pyr = Pyramid().from_zarr(store)
            arrmeta = copy.deepcopy(pyr.array_meta)
            scales = copy.deepcopy(pyr.scales)
            # translations = copy.deepcopy(pyr.translations)
            array_meta = copy.deepcopy(pyr.array_meta)
            for pth, array in pyr.layers.items():
                zarr_meta = array_meta[pth]
                # pyr.add_layer(res.astype(self.dtype), pth, scale, translation, zarr_meta = zarr_meta)
                if isinstance(array, np.ndarray):
                    array = da.from_array(array, chunks=arrmeta[pth]['chunks']).astype(arrmeta[pth]['dtype'])
                elif isinstance(array, zarr.Array):
                    array = da.from_zarr(array, chunks=arrmeta[pth]['chunks']).astype(arrmeta[pth]['dtype'])
                pyr.add_layer(array,
                              pth = pth,
                              scale = scales[pth],
                              zarr_meta = zarr_meta,
                              overwrite = True
                              )
            return pyr
        else:
            arrmeta = copy.deepcopy(self.array_meta)
            for pth, array in self.layers.items():
                if isinstance(array, np.ndarray):
                    self._arrays[pth] = da.from_array(array, chunks=arrmeta[pth]['chunks']).astype(arrmeta[pth]['dtype'])
                elif isinstance(array, zarr.Array):
                    self._arrays[pth] = da.from_zarr(array, chunks=arrmeta[pth]['chunks']).astype(arrmeta[pth]['dtype'])
            return self
    # def checkpoint(self):
    #     self.compute()
    #     self.to_dask()

    def _parse_synchronizer(self, syncdir, basename):
        if syncdir is not None and syncdir != 'same' and syncdir != 'default':
            syncdir = os.path.join(syncdir, basename)
            if os.path.exists(syncdir):
                shutil.rmtree(syncdir)
            synchronizer = zarr.ProcessSynchronizer(syncdir)
        elif syncdir == 'default':
            syncdir = os.path.join(os.path.expanduser('~'), '.syncdir', basename)
            if os.path.exists(syncdir):
                shutil.rmtree(syncdir)
            synchronizer = zarr.ProcessSynchronizer(syncdir)
        elif syncdir == 'same':
            syncdir = os.path.join(os.path.expanduser('~'), '.syncdir', basename)
            if os.path.exists(syncdir):
                shutil.rmtree(syncdir)
            synchronizer = zarr.ProcessSynchronizer(syncdir)
            # synchronizer = self.synchronizer
        else:
            synchronizer = None
        return synchronizer

    def save_binary(self, fpath, region_sizes, overwrite, verbose = False):
        for pth, arr in self.layers.items():
            if hasattr(arr, 'write_binary'): # means it is a blockwise instance
                self._arrays[pth] = arr.output
            elif isinstance(arr, zarr.Array):
                self.gr[pth] = arr
                self._arrays[pth] = self.gr[pth]
            elif isinstance(arr, da.Array):
                self.save_dask_layer(fpath,
                                     pth = pth,
                                     region_sizes = region_sizes,
                                     overwrite = overwrite,
                                     verbose = verbose
                                     )
                self._arrays[pth] = self.gr[pth]
            else:
                warnings.warn(f"The data type for path {pth} could not be recognized!")
        return self

    def to_zarr(self, # Pyramid class
                output_path: Union[str, Path],
                paths: Union[Iterable[int],Iterable[str]] = None,
                overwrite: bool = False,
                # include_imglabels = False,
                region_sizes = None,
                verbose = False,
                only_meta = False,
                syncdir = None
                ):
        if cnv.path_has_pyramid(output_path):
            if overwrite:
                _ = zarr.group(output_path, overwrite = True)
            # else:
            #     if not only_meta:
            #         raise IOError(f"The output path is not empty.")
        if paths is not None:
            paths = _parse_resolution_paths(paths)
            self.shrink(paths, hard = False)
        if isinstance(syncdir, (Path, str)):
            if isinstance(syncdir, Path):
                syncdir = syncdir.as_posix()
            basename = os.path.basename(output_path)
            if syncdir.endswith(basename):
                synchronizer = self._parse_synchronizer(syncdir, '')
            else:
                synchronizer = self._parse_synchronizer(syncdir, basename)

        else:
            synchronizer = None
        if isinstance(output_path, (str, Path)):
            store = zarr.DirectoryStore(output_path, dimension_separator = self.dimension_separator)
        else:
            store = output_path
        grp = zarr.open_group(store, mode='a', synchronizer = synchronizer)
        self._gr = grp
        if not only_meta:
            self.save_binary(output_path, region_sizes, overwrite)
        else:
            # print(f"Only meta")
            pass
        self.gr.attrs['multiscales'] = self.multimeta
        # if include_imglabels:
        #     labelgrp = grp.create_group('labels', overwrite=overwrite)
        #     for label_name in self.label_paths:
        #         lpyr = self.labels[label_name]
        #         loutput_path = os.path.join(fpath, 'labels', label_name)
        #         lpyr.to_zarr(lfpath)
        #     labelgrp.attrs['labels'] = self.label_paths
        return self

    def from_dict(self,
                  datasets: dict
               ):
        for pth, dataset in datasets.items():
            keys = list(datasets[pth])
            assert 'array' in keys, 'dataset must contain an array section.'
            arr = dataset['array']
            assert isinstance(arr, (np.ndarray, da.Array, zarr.Array)), f'Array must be either of the types: {np.ndarray} or {da.Array}.'
            if isinstance(arr, np.ndarray):
                if 'chunks' in keys:
                    arr = da.from_array(arr, chunks = dataset['chunks'])
                else:
                    arr = da.from_array(arr, chunks = 'auto')
            elif isinstance(arr, zarr.Array):
                if 'chunks' in keys:
                    if dataset['chunks'] != arr.chunks:
                        warnings.warn(f"The chunk size of the zarr array in the dict does not fit the chunk size of the Pyramid instance!")
                        warnings.warn(f"Using the chunk size of the zarr array.")
            elif isinstance(arr, da.Array):
                if 'chunks' in keys:
                    if dataset['chunks'] != arr.chunksize:
                        arr = arr.rechunk(dataset['chunks'])
            if 'dtype' in keys:
                if dataset['dtype'] != arr.dtype:
                    arr = arr.astype(dataset['dtype'])
            if 'axis_order' in keys:
                axis_order = dataset['axis_order']
            else:
                axis_order = config.default_axes[-arr.ndim:]
            if 'unit_list' in keys:
                unitlist = dataset['unit_list']
            else:
                unitlist = [config.unit_map[ax] for ax in axis_order]
            if 'scale' in keys:
                scale = dataset['scale']
            else:
                scale = [1.] * arr.ndim
            if 'translation' in keys:
                translation = dataset['translation']
            else:
                translation = None
            zarr_meta = {}
            if 'compressor' in keys:
                zarr_meta['compressor'] = dataset['compressor']
            else:
                zarr_meta['compressor'] = Blosc()
            if 'dimension_separator' in keys:
                zarr_meta['dimension_separator'] = dataset['dimension_separator']
            else:
                zarr_meta['dimension_separator'] = '/'
            if not self.has_axes:
                self.parse_axes(axis_order, unit_list = unitlist)
            self.add_layer(arr, pth, scale, translation, zarr_meta = zarr_meta)

    def _validate_and_sync(self): # properly synch dtype
        """Method that syncs any non-fitting dask array metadata to the metadata in the array_meta."""
        meta_fields = ['dtype', 'chunks', 'shape']
        for pth, layer in self.layers.items():
            for metakey in meta_fields:
                metavalue = self.array_meta[pth][metakey]
                if metakey == 'dtype':
                    realvalue = layer.dtype
                    if realvalue != metavalue:
                        if isinstance(layer, (da.Array, np.ndarray)):
                            try:
                                layer = layer.astype(metavalue)
                            except:
                                raise TypeError(f'dtype update from {realvalue} to {metavalue} is not successful')
                elif metakey == 'chunks':
                    if isinstance(layer, da.Array):
                        realvalue = layer.chunksize
                    elif isinstance(layer, zarr.Array):
                        realvalue = layer.chunks
                    elif isinstance(layer, np.ndarray): # if numpy, it is a single chunk
                        realvalue = layer.shape
                    elif hasattr(layer, 'shape'): # For example UnaryRunner
                        realvalue = layer.chunks
                    if realvalue != metavalue:
                        if not isinstance(layer, zarr.Array):
                            print(f"The chunk size must be updated from {realvalue} to {metavalue}")
                        if isinstance(layer, da.Array):
                            try:
                                layer = layer.rechunk(metavalue)
                            except:
                                raise TypeError(f'chunksize update from {realvalue} to {metavalue} is not successful')
                        elif isinstance(layer, np.ndarray):
                            self._array_meta_[pth]['chunks'] = metavalue

                elif metakey == 'shape':
                    realvalue = layer.shape
                    if realvalue != metavalue:
                        try:
                            layer = layer.reshape(metavalue)
                        except:
                            raise TypeError(f'shape update from {realvalue} to {metavalue} is not successful')
                self._arrays[pth] = layer

    def _add_layer(self,
                   pth: (str, int),
                   zarr_meta: dict = None
                   ):
        """Add new array_meta to specific path."""
        pth = cnv.asstr(pth)
        assert pth in self.resolution_paths, f'Path {pth} does not exist in the current path list. First use add_layer method.'
        arrmeta_keys = config.array_meta_keys
        for metakey in arrmeta_keys:
            if isinstance(self.layers[pth], zarr.Array):
                if metakey in self.layers[pth]._meta:
                    if metakey == 'compressor':
                        self.array_meta[pth][metakey] = self.layers[pth].compressor
                    elif metakey == 'dtype':
                        self.array_meta[pth][metakey] = self.layers[pth].dtype
                    else:
                        self.array_meta[pth][metakey] = self.layers[pth]._meta[metakey]

            elif isinstance(self.layers[pth], da.Array):
                self.array_meta[pth]['dtype'] = self.layers[pth].dtype
                self.array_meta[pth]['chunks'] = self.layers[pth].chunksize
                self.array_meta[pth]['shape'] = self.layers[pth].shape
            elif isinstance(self.layers[pth], np.ndarray):
                self.array_meta[pth]['dtype'] = self.layers[pth].dtype
                try:
                    chunks = self.chunks
                except:
                    chunks = self.layers[pth].shape
                self.array_meta[pth]['chunks'] = chunks
                self.array_meta[pth]['shape'] = self.layers[pth].shape
            if 'compressor' in self.array_meta[pth]:
                compressor = self.array_meta[pth]['compressor']
            elif not 'compressor' in self.array_meta[pth]:
                if 'compressor' in self.array_meta[self.refpath]:
                    compressor = self.array_meta[self.refpath]['compressor']
                else:
                    compressor = self.get_compressor('blosc')
            self.array_meta[pth]['compressor'] = compressor

            if 'dimension_separator' in self.array_meta[pth]:
                dimension_separator = self.array_meta[pth]['dimension_separator']
            elif not 'dimension_separator' in self.array_meta[pth]:
                if 'dimension_separator' in self.array_meta[self.refpath]:
                    dimension_separator = self.array_meta[self.refpath]['dimension_separator']
                else:
                    dimension_separator = '/'
            self.array_meta[pth]['dimension_separator'] = dimension_separator

            if zarr_meta is not None: ### Override any zarr inherent metadata.
                if metakey in zarr_meta:
                    self.array_meta[pth][metakey] = zarr_meta[metakey]

    def add_layer(self, ### TODO: separate into add_dask_layer, add_zarr_layer, etc.
                  arr: Union[zarr.Array, da.Array],
                  pth: str,
                  scale: Iterable[Union[int, float]],
                  translation: Iterable[Union[int, float]] = None,
                  zarr_meta: dict = None,
                  overwrite: bool = False,
                  axis_order: str = None,
                  unitlist: str = None,
                  keep_array_type: bool = True,
                  ):
        """Depends on the _add_dataset method. Helps update the array_meta"""
        pth = cnv.asstr(pth)
        assert hasattr(arr, 'shape'), f'Input array must be either of types zarr.Array, dask.array.Array, numpy.ndarray'
        if not self.has_axes:
            if axis_order is None:
                axis_order = config.default_axes[-arr.ndim:]
            self.parse_axes(axis_order, unit_list=unitlist)
        assert arr.ndim == self.ndim, f'The newly added array must have the same number of dimensions as the existing arrays.'
        if not overwrite:
            assert pth not in self.resolution_paths, f'Path {pth} is already occupied.'
        self._arrays[pth] = arr
        self._arrays = dict(sorted(self._arrays.items()))
        if not overwrite:
            assert len(scale) == self.ndim, f'The scale must be an iterable of a length that is equal to the number of axes.'
        self.add_dataset(pth,
                         scale = scale,
                         translation = translation,
                         overwrite = overwrite
                         )
        self._add_layer(pth, zarr_meta)
        if isinstance(arr, zarr.Array):
            if keep_array_type:
                self._arrays[pth] = arr
            else:
                self._arrays[pth] = da.from_zarr(arr)
        elif isinstance(arr, np.ndarray):
            if keep_array_type:
                self._arrays[pth] = arr
            else:
                self._arrays[pth] = da.from_array(arr)
        self._validate_and_sync()
        return self

    def add_layer_as_imglabel(self,
                              arr: Union[zarr.Array, da.Array],
                              name: str,
                              pth: str,
                              scale: Iterable[str],
                              translation: Iterable[str],
                              zarr_meta: dict = None,
                              color_meta: dict = None,
                              overwrite: bool = False,
                              axis_order: str = None,
                              unitlist: str = None,
                              ):
        if color_meta is None:
            color_meta = 'autodetect'
        lpyr = LabelPyramid()
        lpyr.add_layer(arr, pth, scale = scale, translation = translation, axis_order = axis_order, unitlist = unitlist,
                       zarr_meta = zarr_meta, color_meta = color_meta, overwrite = overwrite)
        self.add_imglabel(lpyr, name)

    def del_layer(self,
                  pth,
                  hard = False
                  ):
        pth = cnv.asstr(pth)
        del self._arrays[pth]
        for i, dataset in enumerate(self.multimeta[0]['datasets']):
            if dataset['path'] == pth:
                del self.multimeta[0]['datasets'][i]
        del self._array_meta_[pth]
        if self.nlayers == 0:
            self.multimeta[0]['axes'] = []
        if hard:
            if isinstance(self.refarray.store, zarr.DirectoryStore):
                del self.gr[pth]
        return self

    def shrink(self,
               paths = None,
               label_paths = None,
               hard = False
               ):
        if paths is None:
            paths = [self.refpath]
        else:
            paths = cnv.parse_as_list(paths)
        paths = [cnv.asstr(s) for s in paths]
        for pth in self.resolution_paths:
            if pth not in paths:
                self.del_layer(pth, hard = hard)
        if label_paths is not None:
            if label_paths == 'all': label_paths = self.label_paths
            for lpth in label_paths:
                if lpth in self.label_paths:
                    self.labels[lpth].shrink(paths)
        return self

    @property
    def array_type(self):
        typ = None
        if isinstance(self.refarray, zarr.Array):
            typ = 'zarr'
        elif isinstance(self.refarray, da.Array):
            typ = 'dask'
        elif isinstance(self.refarray, np.ndarray):
            typ = 'numpy'
        elif isinstance(self.refarray, BlockwiseRunner):
            typ = 'blockwise'
        return typ

    @property
    def layers(self):
        return dict(sorted(self._arrays.items()))

    @property
    def array_meta(self):
        if len(self.resolution_paths) > 0:
            for pth in self.resolution_paths:
                if pth not in self._array_meta_.keys():
                    self._array_meta_[pth] = {}
        return self._array_meta_

    @property
    def layer_meta (self):
        """A combination of array meta and parts of multimeta. Convenient for writing.
        Include axes, units and scales. """
        meta = copy.deepcopy(self.array_meta)
        for pth in self.resolution_paths:
            meta[pth]['axis_order'] = self.axis_order
            meta[pth]['unit_list'] = self.unit_list
            meta[pth]['scale'] = self.scales[pth]
            meta[pth]['physical_size'] = np.array(self.scales[pth]) * np.array(self.array_meta[pth]['shape'])
        return meta

    ##### Update methods
    def _update_layer_meta(self,
                           pth: str,
                           newmeta: dict
                           ):
        """Updates the existing layer_meta with new meta. Important for methods like rechunk, rescale, etc."""
        assert pth in self.resolution_paths, f'{pth} does not exist in the current pyramid.'
        self._add_layer(pth, newmeta)
        if 'scale' in newmeta.keys():
            if not 'translation' in newmeta.keys():
                newmeta['translation'] = self.translations[pth]
            self.add_dataset(pth, newmeta['scale'], newmeta['translation'], overwrite = True)
        if 'axis_order' in newmeta.keys():
            if 'unit_list' not in newmeta.keys():
                unitlist = self.unit_list
            self.parse_axes(newmeta['axis_order'], unitlist, overwrite = True)
        if 'unit_list' in newmeta.keys():
            if 'axis_order' not in newmeta.keys():
                axis_order = self.axis_order
            self.parse_axes(axis_order, newmeta['unit_list'], overwrite = True)

    def astype(self,
               new_dtype: Union[np.dtype, type, str],
               paths: Union[list, str] = None, # if None, update all paths
               # label_paths: List[str] = None
               ):
        assert isinstance(np.dtype(new_dtype), (np.dtype, str)) or any(new_dtype == item for item in [int, float])
        try:
            npdtype = np.dtype(new_dtype)
            _ = npdtype
        except:
            raise TypeError(f'The given data type is not parsable.')
        if paths is None:
            paths = self.resolution_paths
        else:
            paths = _parse_resolution_paths(paths)
            assert cnv.includes(self.resolution_paths, paths)
        for pth in paths:
            layer = self.layers[pth].astype(new_dtype)
            zarr_meta = self.array_meta[pth]
            zarr_meta['dtype'] = new_dtype
            # translation = self.translations[pth] if self.has_translation else None
            self.add_layer(layer, pth, self.scales[pth],  self.translations[pth], zarr_meta = zarr_meta, overwrite = True)
        # if label_paths is not None:
        #     if label_paths == 'all': label_paths = self.label_paths
        #     for lpth in label_paths:
        #         if lpth in self.label_paths:
        #             self.labels[lpth].astype(new_dtype, paths)
        return self

    # def asflat(self,
    #            paths: Union[list, str] = None,
    #            label_paths: List[str] = None,
    #            ):
    #     if paths is None:
    #         paths = self.resolution_paths
    #     else:
    #         paths = cnv.parse_as_list(paths)
    #         assert cnv.includes(self.resolution_paths, paths)
    #     for pth in paths:
    #         zarr_meta = self.array_meta[pth]
    #         zarr_meta['dimension_separator'] = '.'
    #         layer = self.layers[pth]
    #         # translation = self.translations[pth] if self.has_translation else None
    #         self.add_layer(layer, pth, self.scales[pth],  self.translations[pth], zarr_meta = zarr_meta, overwrite = True)
    #     if label_paths is not None:
    #         if label_paths == 'all': label_paths = self.label_paths
    #         for lpth in label_paths:
    #             if lpth in self.label_paths:
    #                 self.labels[lpth].asflat(paths)
    #     return self
    #
    # def asnested(self,
    #              paths: Union[list, str] = None,  # if None, update all paths
    #              label_paths: List[str] = None
    #              ):
    #     if paths is None:
    #         paths = self.resolution_paths
    #     else:
    #         paths = cnv.parse_as_list(paths)
    #         assert cnv.includes(self.resolution_paths, paths)
    #     for pth in paths:
    #         zarr_meta = self.array_meta[pth]
    #         zarr_meta['dimension_separator'] = '/'
    #         layer = self.layers[pth]
    #         # translation = self.translations[pth] if self.has_translation else None
    #         self.add_layer(layer, pth, self.scales[pth],  self.translations[pth], zarr_meta = zarr_meta, overwrite = True)
    #     if label_paths is not None:
    #         if label_paths == 'all': label_paths = self.label_paths
    #         for lpth in label_paths:
    #             if lpth in self.label_paths:
    #                 self.labels[lpth].asnested(paths)
    #     return self
    #
    # def rechunk(self,
    #             new_chunks: Union[list, tuple],
    #             paths: Union[list, str] = None,  # if None, update all paths
    #             label_paths: List[str] = None
    #             ):
    #     if paths is None:
    #         paths = self.resolution_paths
    #     else:
    #         paths = cnv.parse_as_list(paths)
    #         assert cnv.includes(self.resolution_paths, paths)
    #     for pth in paths:
    #         zarr_meta = self.array_meta[pth]
    #         zarr_meta['chunks'] = new_chunks
    #         layer = self.layers[pth].rechunk(new_chunks)
    #         # translation = self.translations[pth] if self.has_translation else None
    #         self.add_layer(layer, pth, self.scales[pth],  self.translations[pth], zarr_meta = zarr_meta, overwrite = True)
    #     if label_paths is not None:
    #         if label_paths == 'all': label_paths = self.label_paths
    #         for lpth in label_paths:
    #             if lpth in self.label_paths:
    #                 self.labels[lpth].rechunk(new_chunks, paths)
    #     return self
    #
    # def recompress(self,
    #                 new_compressor: str,
    #                 paths: Union[list, str] = None,  # if None, update all paths
    #                 label_paths: List[str] = None
    #                ):
    #     if paths is None:
    #         paths = self.resolution_paths
    #     else:
    #         paths = cnv.parse_as_list(paths)
    #         assert cnv.includes(self.resolution_paths, paths)
    #     for pth in paths:
    #         zarr_meta = self.array_meta[pth]
    #         zarr_meta['compressor'] = self.get_compressor(new_compressor)
    #         layer = self.layers[pth]
    #         # translation = self.translations[pth] if self.has_translation else None
    #         self.add_layer(layer, pth, self.scales[pth], self.translations[pth], zarr_meta = zarr_meta, overwrite = True)
    #     if label_paths is not None:
    #         if label_paths == 'all': label_paths = self.label_paths
    #         for lpth in label_paths:
    #             if lpth in self.label_paths:
    #                 self.labels[lpth].recompress(new_compressor, paths)
    #     return self

    def get_current_scale_factors(self):
        return {pth: np.divide(self.array_meta[self.refpath]['shape'], self.array_meta[pth]['shape']).tolist()
                for pth in self.array_meta.keys()}
    @property
    def scale_factors(self):
        return self.get_current_scale_factors()
    @property
    def layer_shapes(self):
        return {pth: self.array_meta[pth]['shape'] for pth in self.array_meta.keys()}

    def rescale(self,
                n_layers: Union[Tuple, List],
                scale_factor: Union[Tuple, List] = None,
                min_input_block_size: tuple = None,
                overwrite_layers: bool = False,
                n_jobs = 8,
                downscale_func = transform.downscale_local_mean,
                use_synchronizer = None,
                syncdir = None,
                backend = 'dask',
                verbose = True
                ):
        """
        This function rescales the entire Pyramid based on the top resolution layer. Note that if there are already existing
        resolution layers they might be overriden.

        :param n_layers:
        :param scale_factor:
        :param min_input_block_size:
        :param overwrite_layers:
        :return:
        """
        print(f"Rescaling.")
        if scale_factor is None:
            if '1' in self.resolution_paths:
                scale_factor = self.scale_factors['1']
            else:
                scale_factor = [1] * len(self.axis_order)
                axyx = self.index('yx')
                for idx in axyx:
                    scale_factor[idx] = 2
        refpath = str(np.arange(n_layers)[0])
        refscale = self.scales[refpath]
        # rootpath = self.pyramid_root
        root = self.gr
        nlayers = self.nlayers

        rescaled_layers = mutils.downscale_multiscales(#input_array = refarray,
                                                       root = root,
                                                       n_layers = n_layers,
                                                       scale_factor = scale_factor,
                                                       min_input_block_size = min_input_block_size,
                                                       overwrite_layers = overwrite_layers,
                                                       n_jobs = n_jobs,
                                                       downscale_func = downscale_func,
                                                       use_synchronizer = use_synchronizer,
                                                       syncdir = syncdir,
                                                       refpath = int(refpath),
                                                       backend = backend,
                                                       verbose = verbose
                                                       )
        meta = copy.deepcopy(self.array_meta[refpath])
        scales = mutils.get_scales_from_rescaled(rescaled = rescaled_layers,
                                                 refscale = refscale,
                                                 refpath = refpath
                                                 )
        self._gr = rescaled_layers
        for pth, arr in rescaled_layers.items():
            scale = scales[pth]
            self.add_layer(arr,
                          pth,
                          scale = scale,
                          # translation = self.get_translation(pth),
                          zarr_meta={'dtype': arr.dtype,
                                     'chunks': arr.chunks,
                                     'shape': arr.shape,
                                     'compressor': arr.compressor,
                                     'dimension_separator': meta['dimension_separator']
                                     },
                          axis_order = self.axis_order,
                          unitlist = self.unit_list,
                          overwrite = True
                          )

        paths = set([self.refpath] + list(scales.keys()))
        if len(paths) < nlayers:
            self.shrink(paths, hard = False)
        self.gr.attrs['multiscales'] = self.multimeta
        return self

    # def rescale_to_shapes(self, # new_shapes must contain the refpath
    #                       new_shapes
    #                       ):
    #     rescaled_layers = cnv.rescale_to_shapes(self.refarray, self.axis_order, new_shapes)
    #     scale_factors = {pth: np.divide(self.array_meta[self.refpath]['shape'], new_shapes[pth]) for pth in new_shapes.keys()}
    #     scales = {pth: np.multiply(self.get_scale(self.refpath), scale_factors[pth]) for pth in scale_factors.keys()}
    #     meta = self.array_meta[self.refpath]
    #     for pth in scales.keys():
    #         if pth in self.array_meta.keys():
    #             meta = self.array_meta[pth]
    #         arr = rescaled_layers[pth]
    #         scale = scales[pth]
    #         self.add_layer(arr,
    #                       pth,
    #                       scale = scale.tolist(),
    #                       translation = self.get_translation(pth),
    #                       zarr_meta={'dtype': meta['dtype'],
    #                                  'chunks': meta['chunks'],
    #                                  'shape': arr.shape,
    #                                  'compressor': meta['compressor'],
    #                                  'dimension_separator': meta['dimension_separator']
    #                                  },
    #                       axis_order = self.axis_order,
    #                       unitlist = self.unit_list,
    #                       overwrite = True
    #                       )
    #     paths = set([self.refpath] + list(scales.keys()))
    #     self.shrink(paths)
    #     return self

    # def subset(self,
    #            slices: Union[dict, list, tuple],
    #            pth = None,
    #            rescale = False
    #            ):
    #     if pth is None: pth = self.refpath
    #     slicedict = {}
    #     if isinstance(slices, (list, tuple, slice)):
    #         missing_axes = self.axis_order[len(slices):]
    #         slcs = []
    #         for slc in slices:
    #             if isinstance(slc, (tuple, list)):
    #                 slcs.append(slice(*slc))
    #             elif isinstance(slc, slice):
    #                 slcs.append(slc)
    #             else:
    #                 raise TypeError(f'The input "slices" must be either of types: {tuple}, {list} or {slice}')
    #         slcs += [slice(None, None, None) for _ in missing_axes]
    #         for ax, slc in zip(self.axis_order, slcs):
    #             slicedict[ax] = slc
    #     elif isinstance(slices, dict):
    #         for ax in self.axis_order:
    #             if ax in slices.keys():
    #                 if isinstance(slices[ax], (tuple, list)):
    #                     slicedict[ax] = slice(*slices[ax])
    #                 elif isinstance(slices[ax], slice):
    #                     slicedict[ax] = slices[ax]
    #                 else:
    #                     raise TypeError(f'The input "slices" must be either of types: {tuple}, {list} or {slice}')
    #             else:
    #                 slicedict[ax] = slice(None, None, None)
    #     slicer = tuple([slicedict[ax] for ax in self.axis_order])
    #     sliced = self.layers[pth][slicer] # Bunu dask ile yap
    #     shapes = {pth: self.array_meta[pth]['shape'] for pth in self.array_meta.keys()}
    #     scale_factors = {pth: np.divide(shapes[self.refpath], shapes[pth]) for pth in shapes.keys()}
    #
    #     new_shapes = {}
    #     for pth in scale_factors.keys():
    #         if pth == self.refpath:
    #             continue
    #         new_shape = np.around(np.divide(sliced.shape, scale_factors[pth])).astype(int)
    #         new_shape[new_shape == 0] = 1
    #         new_shapes[pth] = new_shape
    #
    #     arrmeta = copy.deepcopy(self.array_meta[self.refpath])
    #     arrmeta['shape'] = sliced.shape
    #
    #     self.add_layer(sliced,
    #                    self.refpath,
    #                    scale = self.get_scale(self.refpath),
    #                    translation = self.translations[self.refpath],
    #                    zarr_meta = arrmeta,
    #                    overwrite = True
    #                    )
    #     if rescale:
    #         self.rescale_to_shapes(new_shapes)
    #     else:
    #         self.shrink(self.refpath)
    #     return self
    #
    # def expand_dims(self,
    #                 new_axis: str,
    #                 new_idx: (int, tuple, list),
    #                 new_scale: (float, tuple, list) = 1,
    #                 new_unit: (str, tuple, list) = None,
    #                 downscaling_factor: (int, float) = 1
    #                 ):
    #     items = [new_idx, new_scale, new_unit]
    #     for i, item in enumerate(items):
    #         if hasattr(item, '__len__'):
    #             if isinstance(item, str):
    #                 items[i] = [item]
    #         else:
    #             items[i] = [item]
    #     assert all([ax not in self.axis_order for ax in new_axis]), f"The 'new_axis' parameter cannot include any axis name that already exists in the current axes."
    #     if new_unit is None:
    #         new_unit = config.unit_map[new_axis]
    #     if isinstance(new_unit, str):
    #         new_unit = [new_unit]
    #     if not hasattr(new_scale, '__len__'):
    #         new_idx = [new_idx]
    #     if not hasattr(new_scale, '__len__'):
    #         new_scale = [new_scale]
    #     assert new_axis not in self.axis_order, f'new axis must be different from the already existing axes.'
    #     axord = cnv.insert_at_indices(self.axis_order, new_axis, new_idx)
    #     axis_order = ''.join(axord)
    #     unit_list = cnv.insert_at_indices(self.unit_list, new_unit, new_idx)
    #     scales = copy.deepcopy(self.scales)
    #     translations = copy.deepcopy(self.translations)
    #     for ax in self.axis_order:
    #         self.del_axis(ax)
    #     self.parse_axes(axis_order = axis_order, unit_list = unit_list, overwrite = False)
    #     for i, pth in enumerate(self.resolution_paths):
    #         layer = self.layers[pth]
    #         arrmeta = copy.deepcopy(self.array_meta[pth])
    #         rescaled = [item * downscaling_factor ** i for item in new_scale]
    #         scale = cnv.insert_at_indices(scales[pth], rescaled, new_idx)
    #         retranslated = [0] * len(new_idx)
    #         if translations[pth] is None:
    #             translation = None
    #         else:
    #             translation = cnv.insert_at_indices(translations[pth], retranslated, new_idx)
    #         chunks = cnv.insert_at_indices(list(arrmeta['chunks']), 1., new_idx)
    #         arrmeta['chunks'] = tuple(chunks)
    #         shape = cnv.insert_at_indices(list(arrmeta['shape']), 1, new_idx)
    #         arrmeta['shape'] = tuple(shape)
    #         layer_ex = da.expand_dims(layer, new_idx)
    #         self.add_layer(arr = layer_ex,
    #                        pth = pth,
    #                        scale = scale,
    #                        translation = translation,
    #                        zarr_meta = arrmeta,
    #                        overwrite = True
    #                        )
    #
    # def squeeze(self,
    #             axis: str = None
    #             ):
    #     axord = list(self.axis_order)
    #     unit_list = self.unit_list
    #     shape = self.shape
    #     if isinstance(axis, int):
    #         indices = [axis]
    #     elif axis is None:
    #         indices = [idx for idx, val in enumerate(shape) if val != 1]
    #     else:
    #         assert isinstance(axis, str)
    #         inds = self.index(axis, scalar_sensitive = False)
    #         indices = [idx for idx, val in enumerate(shape) if (val != 1) or idx not in inds]
    #
    #     ex_indices = [i for i in range(self.ndim) if i not in indices]
    #     axis_order = [axord[idx] for idx in indices]
    #     unit_list = [unit_list[idx] for idx in indices]
    #     _scales = copy.deepcopy(self.scales)
    #     _translations = copy.deepcopy(self.translations)
    #     _arrmeta = copy.deepcopy(self.array_meta)
    #
    #     for ax in self.axis_order:
    #         self.del_axis(ax)
    #     self.parse_axes(axis_order = axis_order, unit_list = unit_list, overwrite = False)
    #     for i, pth in enumerate(self.resolution_paths):
    #         layer = da.squeeze(self.layers[pth], axis = tuple(ex_indices))
    #         scale = [_scales[pth][idx] for idx in indices]
    #         if _translations[pth] is None:
    #             translation = None
    #         else:
    #             translation = [_translations[pth][idx] for idx in indices]
    #         _arrmeta['chunks'] = tuple([_arrmeta[pth]['chunks'][idx] for idx in indices])
    #         _arrmeta['shape'] = tuple([_arrmeta[pth]['shape'][idx] for idx in indices])
    #         self.add_layer(arr = layer,
    #                        pth = pth,
    #                        scale = scale,
    #                        translation = translation,
    #                        zarr_meta = _arrmeta,
    #                        overwrite = True
    #                        )

    # def add_imglabel(self,
    #               pyr,
    #               name: str = None
    #               ):
    #     assert isinstance(pyr, Pyramid), f"The provided object is not an instance of Pyramid."
    #     assert pyr.is_imglabel, f"The provided pyramid is not a label image."
    #     if name is None:
    #         if pyr.tag is None:
    #             raise ValueError("A name has to be specified to denote the added image-label.")
    #         name = pyr.tag
    #     self.labels[name] = pyr




class LabelPyramid(Pyramid, ImageLabelMeta):
    def __init__(self):
        super().__init__()

    def copy(self,
             paths: Union[list, tuple] = None
             ):
        if paths is None:
            paths = self.resolution_paths
        lpyr = LabelPyramid()
        for pth in paths:
            res = self.layers[pth]
            if not lpyr.has_axes:
                lpyr.parse_axes(self.axis_order)
            scale = self.get_scale(pth)
            zarr_meta = self.array_meta[pth]
            lpyr.add_layer(res, pth, scale, zarr_meta)
            lpyr.multimeta[0]['name'] = self.tag
        lpyr.img_label_meta = self.img_label_meta
        return lpyr

    def from_zarr(self,
                  fpath,
                  keep_array_type: bool = True
                  ):
        super().from_zarr(fpath, False, keep_array_type = True)
        assert 'image-label' in self.gr.attrs.keys(), f"The loaded dataset is not a valid image-label object." # TODO Problem burada, gr staging hatali
        self.img_label_meta = dict(self.gr.attrs)['image-label']
        return self

    def to_zarr(self,
                fpath,
                overwrite: bool = False,
                ):
        grp = zarr.open_group(fpath, mode='a')
        try:
            for pth, arr in self.layers.items():
                arrpath = os.path.join(fpath, pth)
                if arr.dtype != self.array_meta[pth]['dtype']:
                    arr = arr.astype(self.array_meta[pth]['dtype'])
                arr.to_zarr(url=arrpath,
                            compressor=self.array_meta[pth]['compressor'],
                            dimension_separator=self.array_meta[pth]['dimension_separator'],
                            overwrite=overwrite
                            # compute = False
                            )
        except:
            self.rechunk(self.chunks)
            for pth, arr in self.layers.items():
                arrpath = os.path.join(fpath, pth)
                if arr.dtype != self.array_meta[pth]['dtype']:
                    arr = arr.astype(self.array_meta[pth]['dtype'])
                arr.to_zarr(url=arrpath,
                            compressor=self.array_meta[pth]['compressor'],
                            dimension_separator=self.array_meta[pth]['dimension_separator'],
                            overwrite=overwrite
                            )
        grp.attrs['multiscales'] = self.multimeta
        grp.attrs['image-label'] = self.img_label_meta

    def add_layer(self,
                  arr: Union[zarr.Array, da.Array],
                  pth: str,
                  scale: Iterable[Union[int, float]],
                  translation: Iterable[Union[int, float]],
                  zarr_meta: dict = None,
                  overwrite: bool = False,
                  color_meta: Union[tuple, list] = None,
                  prop_meta: Union[tuple, list] = None,
                  axis_order: str = None, # TODO: There should be a better way to parse this.
                  unitlist: str = None
                  ):
        """Depends on the _add_dataset method. Helps update the array_meta"""
        # TODO: make sure array is integer type
        assert isinstance(arr, (da.Array, zarr.Array, np.ndarray)), f'Input array must be either of types zarr.Array, dask.array.Array, numpy.ndarray'
        if self.has_axes:
            if axis_order is not None:
                if axis_order != self.axis_order:
                    raise Exception(f"The Pyramid has the axis order: {self.axis_order}."
                                    f" The new layer cannot have a different axis-order.")
        elif not self.has_axes:
            if axis_order is None:
                axis_order = config.default_axes[-arr.ndim:]
            self.parse_axes(axis_order, unit_list=unitlist)
        if self.ndim != 0:
            assert arr.ndim == self.ndim, f'The newly added array must have the same number of dimensions as the existing arrays.'
        if not overwrite:
            assert pth not in self.resolution_paths, f'Path {pth} is already occupied.'
        self._arrays[pth] = arr
        self._arrays = dict(sorted(self._arrays.items()))
        if not overwrite:
            assert len(scale) == self.ndim, f'The scale must be an iterable of a length that is equal to the number of axes.'
        self.add_dataset(pth,
                         scale = scale,
                         translation = translation,
                         overwrite = overwrite
                         )

        if pth == self.refpath:
            if color_meta is None:
                pass
            else:
                if color_meta == 'autodetect':
                    colors = cnv.get_display(arr)['colors']
                else:
                    colors = color_meta
                self._add_imglabel_meta(colors, overwrite = True)
            # TODO: DO ALSO FOR PROPERTIES

        self._add_layer(pth, zarr_meta)
        if isinstance(arr, zarr.Array):
            self._arrays[pth] = da.from_zarr(arr)
        elif isinstance(arr, np.ndarray):
            self._arrays[pth] = da.from_array(arr)
        self._validate_and_sync()
        return self

    def compute_properties(self,
                           pth: str = None,
                           image: Union[da.Array, np.ndarray] = None, # This must come from the top layer ome-zarr,
                           features: list = ['label', 'area', 'area_convex', 'intensity_max', 'intensity_mean', 'solidity'],
                           return_df: bool = False
                           ):
        if pth is None:
            pth = self.refpath
        larr = self.layers[pth]
        assert image.ndim == larr.ndim, f"The image and label-image must have the same dimentionality."
        regs = cnv.get_properties(larr, image, properties = features)
        for idx, l in enumerate(regs['label']):
            prop = {'label-value': l}
            prop.update({key: regs[key][idx] for key in regs.keys() if key != 'label'})
            self.add_property(prop, overwrite = True)
        if return_df:
            return pd.DataFrame(self.properties)






####################### Below is the PyramidCollection class ##################################

def get_common_set(lol):
    flat = []
    res = []
    for l in lol: flat += l
    for item in flat:
        if all(item in l for l in lol):
            if item not in res:
                res.append(item)
    return res

def get_common_indices(axes):
    common_axes = get_common_set(axes)
    common_axes_indices = []
    for ax in axes:
        ids = []
        for c in common_axes:
            ids.append(ax.index(c))
        common_axes_indices.append(ids)
    return common_axes_indices

def apply_method(args):
    pyramid, method, method_args, method_kwargs = args
    getattr(pyramid, method)(*method_args, **method_kwargs)
    return pyramid

def concurrent_pyramid_methods(method):
    def decorator(cls):
        def wrapper(self, *args, **kwargs):
            with Pool() as pool:
                results = []
                for pyramid in self.pyramids:
                    results.append(pool.apply_async(apply_method, args=((pyramid, method, args, kwargs),)))
                pool.close()
                pool.join()
                self.pyramids = [result.get() for result in results]
        setattr(cls, method, wrapper)
        return cls
    return decorator

def pyramid_methods(method):
    def decorator(cls):
        def wrapper(self, *args, **kwargs):
            for pyramid in self.pyramids:
                getattr(pyramid, method)(*args, **kwargs)
        setattr(cls, method, wrapper)
        return cls
    return decorator

@pyramid_methods('rechunk')
@pyramid_methods('recompress')
@pyramid_methods('rescale')
@pyramid_methods('shrink')
@pyramid_methods('del_layer')
@pyramid_methods('astype')
@pyramid_methods('asflat')
@pyramid_methods('asnested')
@pyramid_methods('expand_dims')
@pyramid_methods('drop_singlet_axes')
@pyramid_methods('retag')
class PyramidCollection: # Change to PyramidIO
    # TODO: provide a sorting method, based on the Pyramid tag, which should be automatically assigned a unique value.
    def __init__(self,
                 input=None,
                 stringent=False
                 ):
        self.pyramids = []
        if input in [None, []]:
            pass
        else:
            input = [pyr for pyr in input]
            for pyr in input:
                self.add_pyramid(pyr,
                                 stringent = stringent
                                 )

    @property
    def size(self):
        return len(self.pyramids)

    @property
    def refpath(self):
        try:
            return self.resolution_paths[0]
        except:
            return ValueError(f"No reference path can be specified. Perhaps, paths are not unique?")

    @property
    def refarray(self):
        return self.layers[self.refpath]

    # @property
    # def refarray(self):
    #     return self.reflayer[0]

    def drop_pyramid(self,
                     idx,
                     ):
        self.pyramids.pop(idx)

    def add_pyramid(self,
                    pyr,
                    stringent=False
                    ):
        self.pyramids.append(pyr)
        assert self._validate_pyramidal_collection(self.pyramids)
        if stringent:
            assert self._paths_are_unique(), f"The resolution paths are not consistent for all Pyramids."
        if not self._paths_are_unique():
            self.equalise_resolutions()
        if len(self.resolution_paths) == 0:
            raise ValueError(f"The pyramids cannot be processed together. No matching resolution paths.")

    def _paths_are_unique(self):  # make sure each pyramid has the same number of resolution layers
        return all(self.pyramids[0].resolution_paths == item.resolution_paths for item in self.pyramids)

    @property
    def _nlayers(self):
        return [pyr.nlayers for pyr in self.pyramids]

    @property
    def nlayers(self):
        return self._nlayers[0]

    @property
    def _resolution_paths(self):
        return [pyr.resolution_paths for pyr in self.pyramids]

    @property
    def resolution_paths(self):
        if self._paths_are_unique():
            return self.pyramids[0].resolution_paths
        else:
            return get_common_set(self._resolution_paths)

    @property
    def array_meta(self):
        return self.pyramids[0].array_meta

    def equalise_resolutions(self):
        paths = self.resolution_paths
        self.shrink(paths)

    def _validate_pyramidal_collection(self, input):
        if input is None:
            input = []
        if isinstance(input, Pyramid):
            input = [input]
        else:
            assert isinstance(input, (tuple, list)), f"Input must be tuple or list."
        for pyr in input:
            assert isinstance(pyr, Pyramid), f"Each input item must be an instance of Pyramid class."
        self.pyramids = input
        return True

    @property
    def layers(self):
        l = {key: [] for key in self.resolution_paths}
        for key in self.resolution_paths:
            arrays = [pyr.layers[key] for pyr in self.pyramids]
            l[key] = arrays
        return l

    @property
    def axes(self):
        return [pyr.axis_order for pyr in self.pyramids]

    @property
    def has_uniform_axes(
            self):
        return all([self.axes[0] == item for item in self.axes])

    @property
    def axis_order(self):
        return self.axes[0]

    @property
    def ndim(self):
        return len(self.axis_order)


    def get_uniform_dimensions(self): # TODO
        pass

    def has_uniform_axes_except(
                                self,
                                axes
                                ):
        if self.has_uniform_axes:
            return True
        if not hasattr(axes, '__len__'):
            axes = [axes]
        template = [c for c in list(self.axes[0]) if c not in axes]
        for item in self.axes:
            item_temp = [c for c in list(item) if c not in axes]
            if template != item_temp:
                return False
        return True

    @property
    def units(self):
        return [pyr.unit_list for pyr in self.pyramids]

    @property
    def has_uniform_units(self):
        return all([self.units[0] == item for item in self.units])

    @property
    def unit_list(self):
        return self.units[0]

    @property
    def scales(self):
        l = {key: [] for key in self.resolution_paths}
        for key in self.resolution_paths:
            scales = [pyr.scales[key] for pyr in self.pyramids]
            l[key] = scales
        return l

    def get_scale(self, pth):
        return self.scales[pth][0]

    @property
    def has_uniform_scales(self):
        checklist = []
        for key, scale in self.scales.items():
            checklist.append(all([scale[0] == item for item in scale]))
        return all(checklist)

    @property
    def translations(self):
        l = {key: [] for key in self.resolution_paths}
        for key in self.resolution_paths:
            translations = [pyr.translations[key] for pyr in self.pyramids]
            l[key] = translations
        return l

    def get_translation(self, pth):
        return self.translations[pth][0]

    @property
    def shapes(self):
        return [pyr.shape for pyr in self.pyramids]

    @property
    def has_uniform_shapes(self):
        if not self.has_uniform_scales: return False
        return all(self.shapes[0] == item for item in self.shapes)

    @property
    def shape(self):
        return self.shapes[0]

    @property
    def chunklist(self):
        return [pyr.chunks for pyr in self.pyramids]

    @property
    def has_uniform_chunks(self): ### TODO: check also compression and dimension_separator
        return all([self.chunklist[0] == item for item in self.chunklist])

    @property
    def chunks(self):
        return self.chunklist[0]

    @property
    def compressors(self):
        return [pyr.compressor for pyr in self.pyramids]

    @property
    def has_uniform_compressors(self):
        return all([self.compressors[0].codec_id == item.codec_id for item in self.compressors])

    @property
    def compressor(self):
        return self.compressors[0]

    @property
    def dtypes(self):
        return [pyr.dtype for pyr in self.pyramids]

    @property
    def has_uniform_dtypes(self): ### TODO: check also compression and dimension_separator
        return all([self.dtypes[0] == item for item in self.dtypes])

    @property
    def dtype(self):
        return self.dtypes[0]

    @property
    def dimension_separators(self):
        return [pyr.dimension_separator for pyr in self.pyramids]

    @property
    def has_uniform_dimension_separators(self): ### TODO: check also compression and dimension_separator
        return all([self.dimension_separators[0] == item for item in self.dimension_separators])

    @property
    def dimension_separator(self):
        return self.dimension_separators[0]

    @property
    def has_uniform_axes_units_scales(self):
        if not self.has_uniform_axes: return False
        if not self.has_uniform_units: return False
        if not self.has_uniform_scales: return False
        return True

    @property
    def has_uniform_meta(self):
        if not self.has_uniform_axes: return False
        if not self.has_uniform_units: return False
        if not self.has_uniform_scales: return False
        if not self.has_uniform_chunks: return False
        # if not self.has_uniform_shapes: return False
        if not self.has_uniform_dtypes: return False
        if not self.has_uniform_dimension_separators: return False
        return True

    def make_meta_like(self, ref_img_idx = 0): # TODO: In progress.
        self.refpyr = self.pyramids[ref_img_idx]
        if not self.has_uniform_axes:
            raise ValueError(f"Axes cannot be made uniform.")
        if not self.has_uniform_units: return False

    def index(self,
              axis = 'z',
              scalar_sensitive = True
              ):
        index = []
        for pyr in self.pyramids:
            index.append(pyr.index(axis, scalar_sensitive))
        return index

    # def index(self,
    #           axis = 'z',
    #           scalar_sensitive = True
    #           ):
    #     return self._index(axis, scalar_sensitive)[0]

    def axlens_for(self,
                   axes
                   ):
        axlens = {pth: [] for pth in self.resolution_paths}
        for pyr in self.pyramids:
            for_axes = ''.join([ax for ax in axes if ax in pyr.axis_order])
            for pth in pyr.resolution_paths:
                axlen = pyr.axislen(for_axes, pth)
                axlens[pth].append(axlen)
        return axlens

    def axlens_except(self,
                      axes
                      ):
        axlens = {pth: [] for pth in self.resolution_paths}
        for pyr in self.pyramids:
            other_axes = [ax for ax in pyr.axis_order if ax not in axes]
            for pth in pyr.resolution_paths:
                axlen = pyr.axislen(other_axes, pth)
                axlens[pth].append(axlen)
        return axlens

    def extensible_along(self,
                           axis
                           ):
        res = self.has_uniform_axes_units_scales # Note currently it assumes the zeroth pyramid as reference pyramid.
        axlens = self.axlens_except(axis)
        for axlen in axlens.values():
            if not all([axlen[0] == item for item in axlen]): res = False
        return res

    def match_ref_pyramid(self):
        raise NotImplementedError(f"This method is not yet implemented.")



