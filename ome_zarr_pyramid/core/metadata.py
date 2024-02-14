import zarr, warnings, copy
from ome_zarr_pyramid.core import config
from ome_zarr_pyramid.core import config, convenience as cnv
import numpy as np

from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    Iterable,
    List,
    Optional
)


class _Meta:

    @property
    def base_attrs(self):
        return dict(self.attrs)

    @property
    def attrkeys(self):
        return list(self.base_attrs.keys())

    def has_multiscales_meta(self):
        return 'multiscales' in self.attrkeys

    def has_imagelabel_meta(self):
        return 'image-label' in self.attrkeys

    def has_bioformats2raw_meta(self):
        return 'bioformats2raw.layout' in self.attrkeys

    def has_labelcollection_meta(self):
        return 'labels' in self.attrkeys



class _MultiMeta(_Meta):

    @property
    def multimeta(self):
        try:
            return self.base_attrs['multiscales']
        except:
            return config.NotMultiscalesException

    @property
    def multimeta_fields(self):
        try:
            return [item for item in self.multimeta[0]]
        except:
            return config.NotMultiscalesException

    @property
    def has_axes(self):
        try:
            return 'axes' in self.multimeta_fields
        except:
            return config.NotMultiscalesException

    @property
    def has_datasets(self):
        try:
            return 'datasets' in self.multimeta_fields
        except:
            return config.NotMultiscalesException

    @property
    def has_name(self):
        try:
            return 'name' in self.multimeta_fields
        except:
            return config.NotMultiscalesException

    @property
    def identifier(self):
        if self.has_name:
            if self.multimeta[0]['name'].isalnum():
                return ''.join(self.multimeta[0]['name'].split('.')[:-1])
            else:
                return ''.join(self.grname.split('.')[:-1])

    @property
    def axis_order(self):
        try:
            if self.has_axes:
                return ''.join([item['name'] for item in self.multimeta[0]['axes']])
            else:
                return config.AxisNotFoundException
        except:
            return config.NotMultiscalesException

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
    def ndim(self):
        return len(self.axis_order)

    def axis_scale_map(self,
                       pth
                       ):
        axes = self.axis_order
        scales = self.get_scale(pth)
        return {i: j for i, j  in zip(axes, scales)}

    def _add_axis(self, ### PLUG THIS TO THE DATASET SECTION, USE A DOWNSCALE FACTOR FOR DIFFERENT RESOLUTIONS
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
            index = len(self.multimeta[0]['axes'])
        if not self.has_axes:
            self.multimeta[0]['axes'] = []
            index = 0
        axis = axmake(name, unit)
        self.multimeta[0]['axes'].insert(index, axis)

    def _add_axis_meta(self,
                       axis_order,
                       unit_list = None,
                       # overwrite: consider adding this
                       ):
        if self.has_axes:
            if len(self.multimeta[0]['axes']) > 0:
                raise ValueError('The current axis metadata is not empty. Cannot overwrite.')
        if unit_list is None:
            unit_list = [None] * len(axis_order)
        assert len(axis_order) == len(unit_list), 'Unit list and axis order must have the same length.'
        for i, n in enumerate(axis_order):
            self._add_axis(name = n,
                           unit = unit_list[i],
                           index = i
                           )

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

    @property
    def resolution_paths(self):
        try:
            paths = [item['path'] for item in self.multimeta[0]['datasets']]
            return sorted(paths)
        except:
            return config.NotMultiscalesException

    @property
    def resolutions_asint(self):
        return [int(i) for i in self.resolution_paths]

    @property
    def next_resolution(self):
        return str(np.max(self.resolutions_asint) + 1)

    def get_scale(self,
                  pth: Union[str, int]
                  ):
        idx = self.resolution_paths.index(pth)
        return self.multimeta[0]['datasets'][idx]['coordinateTransformations'][0]['scale']

    def set_scale(self,
                  pth: Union[str, int],
                  scale
                  ):
        idx = self.resolution_paths.index(pth)
        self.multimeta[0]['datasets'][idx]['coordinateTransformations'][0]['scale'] = scale

    def get_scale_factors(self,
                          paths: Iterable[Union[str, int]]
                          ):
        if paths is None: paths = self.resolution_paths
        scales = np.array([self.get_scale(pth) for pth in paths])
        factored = scales / scales[0]
        return {pth: tuple(factored[paths.index(pth)]) for pth in paths}

    def refactor_subset(self, paths, subset):
        if paths is None: paths = self.resolution_paths
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

    def _add_dataset(self,
                    path: Union[str, int],
                    transform: Iterable[str],  ### scale values go here.
                    transform_type: str = 'scale'
                    ): #TODO: Maybe put a validator at the beginning to make sure that the dataset does not already exist.
        assert path not in self.resolution_paths, 'Path already exists.'
        dataset = {'coordinateTransformations': [{'scale': transform, 'type': transform_type}], 'path': str(path)}
        self.multimeta[0]['datasets'].append(dataset)
        args = np.argsort([int(pth) for pth in self.resolution_paths])
        self.multimeta[0]['datasets'] = [self.multimeta[0]['datasets'][i] for i in args]

    def _del_dataset(self,
                    path: Union[str, int]
                    ):
        idx = self.resolution_paths.index(str(path))
        del self.multimeta[0]['datasets'][idx]

    def extract_dataset(self,
                        paths: Iterable[Union[str, int]],
                        axes: str = None
                        ):
        if paths is None: paths = self.resolution_paths
        indices = [self.resolution_paths.index(pth) for pth in paths]
        scales = [self.get_scale(pth) for pth in paths]
        meta = copy.deepcopy(self.multimeta)
        meta[0]['datasets'] = []
        if axes is not None:
            axinds = [self.axis_order.index(i) for i in axes]
        else:
            axinds = [i for i in range(len(self.axis_order))]
        axdata = [self.multimeta[0]['axes'][i] for i in axinds]
        for idx in indices:
            dataset = copy.deepcopy(self.multimeta[0]['datasets'][idx])
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
        scale = copy.deepcopy(self.multimeta[0]['datasets'][-1]['coordinateTransformations'][0]['scale'])
        pth = self.multimeta[0]['datasets'][-1]['path']
        if pth.isnumeric():
            pth = int(pth) + 1
        else:
            pth = pth + '_rescaled'
        for idx in indices:
            scale[idx] *= scale_factor
        self._add_dataset(pth, scale, 'scale')

    def decrement_scale(self):
        pth = self.resolution_paths[-1]
        self._del_dataset(pth)


class _ImageLabelMeta(_Meta):  # TODO: Image-label metadata will be parsed here.

    @property
    def labelmeta(self):
        if 'image-label' in self.base_attrs:
            return self.base_attrs['image-label']
        else:
            warnings.warn('The image-label metadata does not exist. Maybe trying to read a non-label image as a label image?')
            return None

