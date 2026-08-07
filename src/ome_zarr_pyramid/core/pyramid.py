"""Improved NGFF metadata handler and Pyramid class with better design patterns.

This module provides a modernized implementation following the multiscales.py design
from eubi_bridge, with support for multiple array types, better separation of concerns,
and full TensorStore integration for high-performance operations.
"""

import asyncio
import copy
import operator
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import zarr

from ome_zarr_pyramid.utils import defaults
from ome_zarr_pyramid.utils.compressor_config import CompressorConfig
from ome_zarr_pyramid.utils.json_utils import make_json_safe
from ome_zarr_pyramid.utils.logging_config import get_logger
from ome_zarr_pyramid.utils.scale import Downscaler

logger = get_logger(__name__)


def cast_to_dict(value: Any) -> Dict[str, Any]:
    """Safely cast a value to dict, handling zarr JSON attributes."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json
        return json.loads(value)
    return {} if value is None else dict(value)


def is_zarr_group(path: Union[str, Path]) -> bool:
    """Check if path points to a valid zarr group.
    
    Parameters
    ----------
    path : str or Path
        Path to check
        
    Returns
    -------
    bool
        True if valid zarr group found
    """
    try:
        _ = zarr.open_group(path, mode='r')
        return True
    except:
        return False


def calculate_n_layers(shape: Tuple[int, ...],
                        scale_factor: Union[int, float, Tuple[Union[int, float], ...]],
                        min_dimension_size: int = 64) -> int:
    """Calculate the number of downscaling layers for a pyramid.

    Layers are added until the largest dimension that is actually being
    downscaled (scale_factor > 1) would drop below `min_dimension_size`.

    Parameters
    ----------
    shape : tuple of int
        Shape of the base array.
    scale_factor : int, float, or tuple
        Per-axis downscale factor. If a scalar, applied to all axes.
    min_dimension_size : int, optional
        Minimum size a downscaled dimension should retain. Default is 64.

    Returns
    -------
    int
        Number of pyramid layers (including the base layer), at least 1.
    """
    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor,) * len(shape)
    if len(scale_factor) != len(shape):
        raise ValueError(f"scale_factor length ({len(scale_factor)}) must match shape length ({len(shape)})")
    shape_array = np.array(shape, dtype=int)
    scale_array = np.array(scale_factor, dtype=float)
    downscale_dims = scale_array > 1
    if not np.any(downscale_dims):
        return 1
    downscale_shapes = shape_array[downscale_dims]
    downscale_factors = scale_array[downscale_dims]
    n_layers_per_dim = np.floor(np.log(downscale_shapes / min_dimension_size) / np.log(downscale_factors))
    if len(n_layers_per_dim) == 0:
        return 1
    argmax_largest_dim = np.argmax(downscale_shapes)
    n_layers_per_largest_dim = n_layers_per_dim[argmax_largest_dim]
    n_layers = int(n_layers_per_largest_dim) + 1
    return max(1, n_layers)


# Standard colour names -> OME hex ('RRGGBB', no '#'). CSS basic set plus the
# channel colours common in microscopy. Used by `Pyramid.set_channels`.
_COLOR_NAMES = {
    "red": "FF0000", "green": "00FF00", "lime": "00FF00", "blue": "0000FF",
    "cyan": "00FFFF", "aqua": "00FFFF", "magenta": "FF00FF", "fuchsia": "FF00FF",
    "yellow": "FFFF00", "white": "FFFFFF", "black": "000000",
    "gray": "808080", "grey": "808080", "silver": "C0C0C0",
    "orange": "FFA500", "purple": "800080", "violet": "EE82EE", "indigo": "4B0082",
    "pink": "FFC0CB", "brown": "A52A2A", "maroon": "800000", "navy": "000080",
    "teal": "008080", "olive": "808000", "gold": "FFD700", "turquoise": "40E0D0",
}


def normalize_color(value: Union[str, Tuple, List]) -> str:
    """Coerce a colour spec to an OME hex string 'RRGGBB' (uppercase, no '#').

    Accepts a standard colour NAME ('red', 'green', 'magenta', ...), a hex string
    with or without a leading '#' (6-digit 'FF0000' or 3-digit shorthand 'F00'),
    or an (r, g, b) triple of 0-255 ints.
    """
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"colour: RGB must have 3 components, got {value!r}")
        rgb = [int(round(c)) for c in value]
        if any(not (0 <= c <= 255) for c in rgb):
            raise ValueError(f"colour: RGB components must be 0-255, got {value!r}")
        return "{:02X}{:02X}{:02X}".format(*rgb)
    if not isinstance(value, str):
        raise TypeError(f"colour must be a name, hex string or (r,g,b) triple, got {value!r}")
    s = value.strip()
    key = s.lower()
    if key in _COLOR_NAMES:
        return _COLOR_NAMES[key]
    h = s[1:] if s.startswith("#") else s
    if len(h) == 3 and all(c in "0123456789abcdefABCDEF" for c in h):
        h = "".join(c * 2 for c in h)   # 'F00' -> 'FF0000'
    if len(h) == 6 and all(c in "0123456789abcdefABCDEF" for c in h):
        return h.upper()
    raise ValueError(
        f"colour: unrecognised {value!r}; use a standard name (e.g. 'red'), a "
        f"6-digit hex ('FF0000' or '#FF0000'), 3-digit shorthand ('F00') or an (r,g,b) triple"
    )


def generate_channel_metadata(
    num_channels: int,
    dtype: type = np.uint16
) -> Dict[str, Any]:
    """Generate standard OMERO channel metadata.
    
    Parameters
    ----------
    num_channels : int
        Number of channels to generate metadata for
    dtype : type, optional
        Data type for determining min/max values. Default is np.uint16
        
    Returns
    -------
    dict
        Dictionary with 'omero' key containing channel metadata
    """
    default_colors = [
        "FF0000",  # Red
        "00FF00",  # Green
        "0000FF",  # Blue
        "FF00FF",  # Magenta
        "00FFFF",  # Cyan
        "FFFF00",  # Yellow
        "FFFFFF",  # White
    ]

    channels = []
    
    if dtype is not None and np.issubdtype(dtype, np.integer):
        min_val, max_val = np.iinfo(dtype).min, np.iinfo(dtype).max
    elif dtype is not None and np.issubdtype(dtype, np.floating):
        min_val, max_val = np.finfo(dtype).min, np.finfo(dtype).max
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    for i in range(num_channels):
        color = default_colors[i] if i < len(
            default_colors) else f"{i * 40 % 256:02X}{i * 85 % 256:02X}{i * 130 % 256:02X}"
        channel = {
            "color": color,
            "coefficient": 1,
            "active": True,
            "label": f"Channel {i}",
            "window": {
                "min": min_val,
                "max": max_val,
                "start": min_val,
                "end": max_val
            },
            "family": "linear",
            "inverted": False
        }
        channels.append(channel)

    return {
        "omero": {
            "channels": channels,
            "rdefs": {
                "defaultT": 0,
                "model": "greyscale",
                "defaultZ": 0
            }
        }
    }


class NGFFMetadataHandler:
    """Handler for NGFF metadata in zarr groups.
    
    This class manages NGFF (Next Generation File Format) metadata
    for image pyramids, supporting both v0.4 and v0.5 specifications.
    """

    SUPPORTED_VERSIONS: ClassVar[List[str]] = ["0.4", "0.5"]

    def __init__(self) -> None:
        """Initialize an empty metadata handler."""
        self.zarr_group: Optional[zarr.Group] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self._pending_changes: bool = False
        self.version: Optional[str] = None
        self.zarr_format: Optional[int] = None

    def __enter__(self) -> 'NGFFMetadataHandler':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._pending_changes:
            self.save_changes()

    @property
    def multiscales(self) -> Dict[str, Any]:
        """Get the multiscales metadata.
        
        Returns
        -------
        dict
            The first multiscales entry from metadata
            
        Raises
        ------
        RuntimeError
            If no multiscales metadata is available
        """
        if not self.metadata or 'multiscales' not in self.metadata:
            raise RuntimeError("No multiscales metadata available")
        return self.metadata['multiscales'][0]

    @property
    def omero(self) -> Dict[str, Any]:
        """Get the OMERO metadata.
        
        Returns
        -------
        dict
            OMERO metadata containing channel information
            
        Raises
        ------
        RuntimeError
            If no OMERO metadata is available
        """
        if not self.metadata or 'omero' not in self.metadata:
            raise RuntimeError("No omero metadata available")
        return self.metadata['omero']

    @property
    def image_label(self) -> Optional[Dict[str, Any]]:
        """The OME ``image-label`` metadata block (``colors``/``properties``/
        ``source``/``version``) if this is a label image, else ``None``.

        The NGFF label-metadata sibling of `omero`/`multiscales`. Unlike `omero`
        (which raises when absent), this returns ``None`` because image-label is
        optional - most images are not labels. Set it via `Pyramid.set_image_label`.
        """
        if not self.metadata:
            return None
        return self.metadata.get('image-label')

    @property
    def is_multiscales(self) -> bool:
        """True if this carries valid NGFF ``multiscales`` metadata (a resolution
        pyramid) - the base validity of any image or label pyramid."""
        return bool(self.metadata) and 'multiscales' in self.metadata

    @property
    def is_label(self) -> bool:
        """True if this is a LABEL image (carries an OME ``image-label`` block);
        False = a raw / intensity image pyramid."""
        return self.image_label is not None

    def _validate_version_and_format(self, version: str, zarr_format: int) -> None:
        """Validate version and zarr format compatibility."""
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported version {version}. Supported versions: {self.SUPPORTED_VERSIONS}")
        if zarr_format not in (2, 3):
            raise ValueError(f"Unsupported Zarr format: {zarr_format}")
        if version == "0.5" and zarr_format != 3:
            raise ValueError("NGFF version 0.5 requires Zarr format 3")

    def _validate_axis_inputs(self, axis_order: str, units: Optional[List[str]]) -> None:
        """Validate axis order and units inputs."""
        if not all(ax in 'tczyx' for ax in axis_order):
            raise ValueError("Invalid axis order. Must contain only t,c,z,y,x")
        if units is not None:
            if not (len(axis_order) - len(units)) in [0, 1]:
                raise ValueError("Number of units must match number of axes except channel")
            elif (len(axis_order) - len(units)) == 1:
                if 'c' not in axis_order:
                    raise ValueError("Only channel axis can be kept without a unit.")

    def _get_dataset(self, path: str) -> Optional[Dict[str, Any]]:
        """Helper method to find dataset by path."""
        path = str(path)
        for dataset in self.multiscales['datasets']:
            if dataset['path'] == path:
                return dataset
        return None

    def _update_coordinate_transformation(self,
                                          dataset: Dict[str, Any],
                                          transform_type: str,
                                          values: List[float]) -> None:
        """Update or add a coordinate transformation to a dataset."""
        for transform in dataset['coordinateTransformations']:
            if transform['type'] == transform_type:
                transform[transform_type] = values
                break
        else:
            if transform_type == 'scale':
                dataset['coordinateTransformations'].insert(
                    0, {'type': transform_type, transform_type: values}
                )
            else:
                dataset['coordinateTransformations'].append(
                    {'type': transform_type, transform_type: values}
                )

    def get_metadata_state(self) -> Dict[str, Any]:
        """Get a copy of current metadata state."""
        if self.metadata is None:
            raise RuntimeError("No metadata loaded or created")
        return copy.deepcopy(self.metadata)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the metadata."""
        if not self.metadata:
            raise RuntimeError("No metadata available")

        return {
            'version': self.version,
            'zarr_format': self.zarr_format,
            'axes': self._axis_names,
            'units': self._units,
            'n_datasets': len(self.multiscales['datasets']),
            'name': self.multiscales['name']
        }

    def create_new(self, version: str = "0.5", name: str = "unnamed") -> 'NGFFMetadataHandler':
        """Create a new metadata handler with empty metadata of specified version."""
        self._validate_version_and_format(version, 3 if version == "0.5" else 2)

        multiscale_metadata = {
            'name': name,
            'axes': [],
            'datasets': [],
            'metadata': {}
        }

        if version == "0.5":
            self.metadata = {
                'version': version,
                'multiscales': [multiscale_metadata],
                'omero': {
                    'channels': [],
                    'rdefs': {
                        'defaultT': 0,
                        'model': 'greyscale',
                        'defaultZ': 0
                    }
                },
                '_creator': {
                    'name': 'NGFFMetadataHandler',
                    'version': '1.0'
                }
            }
        else:  # version == "0.4"
            multiscale_metadata['version'] = version
            self.metadata = {
                '_creator': {
                    'name': 'NGFFMetadataHandler',
                    'version': '1.0'
                },
                'multiscales': [multiscale_metadata],
                'omero': {
                    'channels': [],
                    'rdefs': {
                        'defaultT': 0,
                        'model': 'greyscale',
                        'defaultZ': 0
                    }
                }
            }

        self.version = version
        self.zarr_format = 3 if version == "0.5" else 2
        self._pending_changes = True
        return self

    def connect_to_group(self, store: Union[zarr.Group, str, Path], mode: Literal['r', 'r+', 'a', 'w', 'w-'] = 'a') -> None:
        """Connect to a zarr group for reading/writing metadata."""
        if not isinstance(store, (zarr.Group, str, Path)):
            raise ValueError("Store must be a zarr group or path")
        if isinstance(store, zarr.Group):
            self.zarr_group = store
        else:  # isinstance(store, (str, Path))
            if is_zarr_group(store):
                self.zarr_group = zarr.open_group(store, mode=mode)
            else:
                zarr_version: int | None = self.zarr_format if self.zarr_format else 2
                zarr_fmt = zarr_version if isinstance(zarr_version, int) else 2
                self.zarr_group = zarr.open_group(store, mode=mode, zarr_version=zarr_fmt)  # type: ignore
        # Update handler's format to match the created store
        store_format = self.zarr_group.info._zarr_format
        self.zarr_format = store_format
        # Update version based on zarr_format
        self.version = "0.5" if store_format == 3 else "0.4"
        self._validate_version_and_format(self.version, store_format)

    def read_metadata(self):
        """Read metadata from connected zarr group."""
        if self.zarr_group is None:
            raise RuntimeError("No zarr group connected. Call connect_to_group first.")

        metadata_loaded: Optional[Dict[str, Any]] = None
        if 'ome' in self.zarr_group.attrs:
            metadata_loaded = self.zarr_group.attrs['ome']  # type: ignore
            self.metadata = metadata_loaded  # type: ignore
            self.version = cast_to_dict(self.metadata).get('version', '0.4')
        elif 'multiscales' in self.zarr_group.attrs:
            multiscales_attr = self.zarr_group.attrs['multiscales']
            self.metadata = {'multiscales': multiscales_attr if isinstance(multiscales_attr, list) else [multiscales_attr]}
            if self.metadata:
                self.version = self.metadata.get('multiscales', [{}])[0].get('version', '0.4')
        else:
            raise ValueError("No valid metadata found in zarr group")

        if 'omero' in self.zarr_group.attrs and self.metadata is not None:
            self.metadata['omero'] = self.zarr_group.attrs['omero']  # type: ignore
        # capture the OME image-label block for label images (0.4 top-level; for 0.5
        # it already lives inside the `ome` metadata read above)
        if 'image-label' in self.zarr_group.attrs and self.metadata is not None:
            self.metadata['image-label'] = self.zarr_group.attrs['image-label']  # type: ignore
        self.zarr_format = 3 if self.version == "0.5" else 2
        self._pending_changes = False
        return self

    def save_changes(self) -> None:
        """Save current metadata to connected zarr group."""
        if not self._pending_changes:
            return
        if self.zarr_group is None:
            raise RuntimeError("No zarr group connected. Call connect_to_group first.")
        if self.metadata is None:
            raise RuntimeError("No metadata to save")

        meta_dict = cast_to_dict(self.metadata)
        if meta_dict.get('version', '') == '0.5':
            self.zarr_group.attrs['ome'] = make_json_safe(self.metadata)
        else:
            metadata = make_json_safe(self.metadata)
            metadata_dict = cast_to_dict(metadata)
            self.zarr_group.attrs['multiscales'] = metadata_dict.get('multiscales')
            if 'omero' in metadata_dict:
                self.zarr_group.attrs['omero'] = metadata_dict['omero']
            if 'image-label' in metadata_dict:
                self.zarr_group.attrs['image-label'] = metadata_dict['image-label']
            if '_creator' in metadata_dict:
                self.zarr_group.attrs['_creator'] = metadata_dict['_creator']

        self._pending_changes = False

    def update_all_datasets(self,
                            scale: Optional[List[float]] = None,
                            translation: Optional[List[float]] = None) -> None:
        """Update all datasets with new scale and/or translation values."""
        for dataset in self.multiscales['datasets']:
            if scale is not None:
                self._update_coordinate_transformation(dataset, 'scale', scale)
            if translation is not None:
                self._update_coordinate_transformation(dataset, 'translation', translation)
        self._pending_changes = True

    def autocompute_omerometa(self, n_channels: int, dtype) -> None:
        """Add multiple channels to the OMERO metadata."""
        if self.metadata is None:
            raise RuntimeError("No metadata loaded or created")
        omero_meta = generate_channel_metadata(n_channels, dtype)
        self.metadata['omero'] = omero_meta['omero']
        self._pending_changes = True

    def parse_axes(self,
                   axis_order: str,
                   units: Optional[List[str]] = None) -> None:
        """Update axes information with new axis order and units."""
        if self.metadata is None:
            raise RuntimeError("No metadata loaded or created.")

        self._validate_axis_inputs(axis_order, units)

        units_list: List[Optional[str]] = []
        if units is None:
            units_list = [None] * len(axis_order)
        else:
            units_list = list(units)
        if len(axis_order) - len(units_list) == 1:
            if 'c' in axis_order:
                idx = axis_order.index('c')
                units_list.insert(idx, None)

        new_axes = []
        for ax_name, unit in zip(axis_order, units_list):
            axis_data = {
                'name': ax_name,
                'type': {'t': 'time', 'c': 'channel', 'z': 'space',
                         'y': 'space', 'x': 'space'}.get(ax_name, 'custom')
            }
            if unit is not None:
                axis_data['unit'] = unit
            new_axes.append(axis_data)

        if self.metadata and 'multiscales' in self.metadata:
            self.metadata['multiscales'][0]['axes'] = new_axes
            self._pending_changes = True

    def add_dataset(self, path: Union[str, int],
                    scale: Iterable[Union[int, float]],
                    translation: Optional[Iterable[Union[int, float]]] = None,
                    overwrite: bool = False) -> None:
        """Add a dataset with scale and optional translation."""
        path = str(path)
        scale = list(map(float, scale))
        if translation is not None:
            translation = list(map(float, translation))

        dataset_data = {
            'path': path,
            'coordinateTransformations': [{'type': 'scale', 'scale': scale}]
        }
        if translation is not None:
            dataset_data['coordinateTransformations'].append(
                {'type': 'translation', 'translation': translation}
            )

        if self.metadata is None or 'multiscales' not in self.metadata:
            raise RuntimeError("No metadata to add dataset to")
        
        existing_paths = self.get_resolution_paths()
        if path in existing_paths:
            if not overwrite:
                raise ValueError(f"Dataset path '{path}' already exists")
            idx = existing_paths.index(path)
            self.metadata['multiscales'][0]['datasets'][idx] = dataset_data
        else:
            self.metadata['multiscales'][0]['datasets'].append(dataset_data)

        self.metadata['multiscales'][0]['datasets'].sort(
            key=lambda x: int(x['path']) if x['path'].isdigit() else float('inf')
        )
        self._pending_changes = True

    def update_scale(self,
                     path: Union[str, int],
                     scale: Iterable[Union[int, float]]) -> None:
        """Update scale for a specific dataset."""
        dataset = self._get_dataset(str(path))
        if dataset:
            self._update_coordinate_transformation(dataset, 'scale', list(map(float, scale)))
            self._pending_changes = True

    def update_translation(self, path: Union[str, int],
                           translation: Iterable[Union[int, float]]) -> None:
        """Update translation for a specific dataset."""
        dataset = self._get_dataset(str(path))
        if dataset:
            self._update_coordinate_transformation(dataset, 'translation', list(map(float, translation)))
            self._pending_changes = True

    def get_resolution_paths(self) -> List[str]:
        """Get paths to all resolution levels."""
        return [ds['path'] for ds in self.multiscales['datasets']]

    @property
    def tag(self) -> str:
        """Get the image series name/tag."""
        return self.multiscales['name']

    @property
    def _axis_names(self) -> List[str]:
        """Get list of axis names."""
        return [ax['name'] for ax in self.multiscales['axes']]

    @property
    def axis_order(self) -> str:
        """Get axis names as string."""
        return ''.join(self._axis_names)

    @property
    def _units(self) -> Dict[str, Optional[str]]:
        """Get dictionary of axis units."""
        return {ax['name']: ax.get('unit') for ax in self.multiscales['axes']}

    @property
    def unit_dict(self) -> Dict[str, Optional[str]]:
        """Get dictionary mapping axis names to units."""
        return self._units

    @property
    def unit_list(self) -> List[Optional[str]]:
        """Get list of units for each axis."""
        return [self._units[ax] for ax in self._axis_names]

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return len(self.axis_order)

    @property
    def resolution_paths(self) -> List[str]:
        """Get list of resolution paths."""
        return [item['path'] for item in self.multiscales['datasets']]

    @property
    def nlayers(self) -> int:
        """Get number of resolution layers."""
        return len(self.resolution_paths)

    @property
    def channels(self) -> List[Dict[str, Any]]:
        """Get list of channels."""
        return self.get_channels()

    def get_channels(self) -> List[Dict[str, Any]]:
        """Get channel metadata list."""
        if self.metadata is None:
            return []
        if 'omero' not in self.metadata or 'channels' not in cast_to_dict(self.metadata.get('omero', {})):
            return []
        omero = cast_to_dict(self.metadata.get('omero', {}))
        return cast_to_dict(omero).get('channels', [])

    def validate_metadata(self) -> bool:
        """Validate current metadata structure."""
        if not self.metadata:
            return False

        try:
            if self.version == "0.5":
                if not all(key in self.metadata for key in {'version', 'multiscales'}):
                    return False
            else:  # version == "0.4"
                if 'multiscales' not in self.metadata:
                    return False
                if 'version' not in self.metadata['multiscales'][0]:
                    return False

            required_keys = {'name', 'axes', 'datasets'}
            return all(key in self.multiscales for key in required_keys)

        except (KeyError, IndexError, TypeError):
            return False

    def get_scaledict(self, pth: Union[str, int]) -> Dict[str, float]:
        """Get scale dictionary for a specific resolution."""
        idx = self.resolution_paths.index(str(pth))
        scale = self.multiscales['datasets'][idx]['coordinateTransformations'][0]['scale']
        return dict(zip(self.axis_order, scale))

    def get_base_scaledict(self) -> Dict[str, float]:
        """Get scale dictionary for base resolution."""
        basepath = self.resolution_paths[0]
        return self.get_scaledict(basepath)

    def get_scale(self, pth: Union[str, int]) -> List[float]:
        """Get scale list for a specific resolution."""
        scaledict = self.get_scaledict(pth)
        return [scaledict[ax] for ax in self.axis_order]

    def get_base_scale(self) -> List[float]:
        """Get scale list for base resolution."""
        basepath = self.resolution_paths[0]
        return self.get_scale(basepath)

    def get_translation(self, pth: Union[str, int]) -> Optional[List[float]]:
        """Physical origin (translation) of a resolution level, or ``None`` if it
        declares none. Looked up by transform TYPE - ``translation`` is optional and
        follows ``scale``, so its index is not fixed (unlike ``scale`` at index 0).
        The read counterpart of `update_translation` / `add_dataset(translation=...)`."""
        idx = self.resolution_paths.index(str(pth))
        for ct in self.multiscales['datasets'][idx].get('coordinateTransformations', []):
            if ct.get('type') == 'translation':
                return list(ct['translation'])
        return None

    def get_translationdict(self, pth: Union[str, int]) -> Optional[Dict[str, float]]:
        """`get_translation` keyed by axis name, or ``None`` if no translation."""
        translation = self.get_translation(pth)
        if translation is None:
            return None
        return dict(zip(self.axis_order, translation))

    def get_base_translation(self) -> Optional[List[float]]:
        """Translation of the base (level 0) resolution, or ``None``."""
        return self.get_translation(self.resolution_paths[0])

    def set_scale(self,
                  pth: Union[str, int] = 'auto',
                  scale: Union[tuple, list, dict, Literal['auto']] = 'auto') -> None:
        """Set scale for a dataset."""
        scale_to_set: Union[list, tuple, dict] = scale if scale != 'auto' else [1.0]
        scale_result: Union[list, tuple, dict] = scale_to_set
        
        if isinstance(scale_to_set, tuple):
            scale_result = list(scale_to_set)
            if 'c' in self.axis_order:
                ch_index = self.axis_order.index('c')
                scale_result[ch_index] = 1
        elif isinstance(scale_to_set, np.ndarray):
            scale_result = scale_to_set.tolist()
        elif isinstance(scale_to_set, dict):
            assert all([ax in self.axis_order for ax in scale_to_set if isinstance(scale_to_set, dict)])
            fullscale = self.get_scale(pth)
            scaledict = dict(zip(self.axis_order, fullscale))
            scaledict.update(scale_to_set)  # type: ignore
            scale_result = [scaledict[ax] for ax in self.axis_order]

        if pth == 'auto':
            pth = self.resolution_paths[0]
        
        # Use auto-computed scale only if explicitly requested and we have valid scale data
        if scale == 'auto':
            # Keep the calculated scale_result
            pass
        
        # Convert scale_result to list if needed
        final_scale: List[float] = scale_result if isinstance(scale_result, list) else list(scale_result)  # type: ignore
        
        if pth in self.resolution_paths:
            idx = self.resolution_paths.index(pth)
            self.multiscales['datasets'][idx]['coordinateTransformations'][0]['scale'] = final_scale
        else:
            d = {
                'path': pth,
                'coordinateTransformations': [
                    {'type': 'scale', 'scale': final_scale}
                ]
            }
            self.multiscales['datasets'].append(d)
        self._pending_changes = True

    def update_scales(self,
                      reference_scale: Union[tuple, list],
                      scale_factors: dict) -> None:
        """Update scales for multiple datasets."""
        for pth, factor in scale_factors.items():
            new_scale = np.multiply(factor, reference_scale).tolist()
            self.set_scale(pth, new_scale)

    def update_unitlist(self, unitlist: Optional[List[str]] = None) -> None:
        """Update unit list."""
        if unitlist is None:
            raise ValueError("unitlist cannot be None")
        if isinstance(unitlist, tuple):
            unitlist = list(unitlist)
        assert isinstance(unitlist, list)
        self.parse_axes(self.axis_order, unitlist)

    def retag(self, new_tag: str) -> 'NGFFMetadataHandler':
        """Retag the image series."""
        self.multiscales['name'] = new_tag
        self._pending_changes = True
        return self

    @property
    def scales(self) -> Dict[str, List[float]]:
        """Get scales for all resolutions."""
        scales = {}
        for pth in self.resolution_paths:
            scl = self.get_scale(pth)
            scales[pth] = scl
        return scales

    @property
    def scaledict(self) -> Dict[str, Dict[str, float]]:
        """Get scale dictionaries for all resolutions."""
        scales = {}
        for pth in self.resolution_paths:
            scl = self.get_scaledict(pth)
            scales[pth] = scl
        return scales


class LabelCollection(Mapping):
    """Read-only mapping of ``name -> label Pyramid`` for an image's NGFF ``labels/``
    collection (see `Pyramid.labels`). Each value is a `Pyramid` whose metadata
    carries an ``image-label`` block (surfaced via `pyr.meta.image_label`). Add labels
    with `Pyramid.add_image_label`; this view itself is not mutable."""

    def __init__(self, labels: Dict[str, 'Pyramid']):
        self._labels = labels

    def __getitem__(self, key: str) -> 'Pyramid':
        return self._labels[key]

    def __iter__(self):
        return iter(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    @property
    def names(self) -> List[str]:
        """The registered label names, in insertion order."""
        return list(self._labels)

    def __repr__(self) -> str:
        return f"LabelCollection({self.names})"


class Pyramid:
    """NGFF-compliant image pyramid supporting multiple array storage modes.
    
    The Pyramid class manages multi-resolution image data in NGFF format,
    supporting three storage modes:
    - Zarr persistent: Arrays stored in zarr.Group
    - In-memory (lazy): Arrays in _array_layers (dask, DynamicArray, numpy)
    - Hybrid: Both zarr and in-memory layers
    """

    def __init__(self, gr: Optional[Union[zarr.Group, Path, str]] = None):
        """Initialize a Pyramid from an NGFF-compliant Zarr group or create empty.
        
        Parameters
        ----------
        gr : Union[zarr.Group, zarr.storage.StoreLike, Path, str], optional
            NGFF-compliant Zarr group, storage path, or Path object.
            If None, creates an empty pyramid.
        """
        self.meta = None
        self.gr = None
        self._array_layers = {}  # In-memory array storage
        self._compressor: Optional[CompressorConfig] = None
        self._labels: Dict[str, 'Pyramid'] = {}  # NGFF labels/ collection (name -> label Pyramid)
        if gr is not None:
            self.from_ngff(gr)

    def __repr__(self) -> str:
        """Return string representation of pyramid."""
        try:
            return f"NGFF with {self.nlayers} layers."
        except (AttributeError, TypeError):
            return f"NGFF."

    def from_ngff(self, gr: Union[zarr.Group, str, Path]) -> 'Pyramid':
        """Load pyramid from an NGFF-compliant Zarr group."""
        self.meta = NGFFMetadataHandler()
        self.meta.connect_to_group(gr)
        self.meta.read_metadata()
        self.gr = self.meta.zarr_group
        base_path = self.meta.resolution_paths[0]
        self._compressor = CompressorConfig.from_array(self.layers[base_path])
        return self

    @property
    def compressor(self) -> Optional[CompressorConfig]:
        """Compressor configuration associated with this pyramid's arrays.

        Set when the pyramid is constructed (``from_ngff``/``from_arrays``)
        and not mutable afterwards - pass ``compressor=`` to ``from_arrays``
        or to ``write_pyramid`` to use a different one.
        """
        return self._compressor

    @property
    def dtype(self):
        """Get the data type of the base (full resolution) array."""
        return self.base_array.dtype

    @property
    def shape(self):
        """Get the shape of the base (full resolution) array."""
        return self.base_array.shape

    def from_array(self,
                   array: Union[np.ndarray, da.Array, zarr.Array],
                   axis_order: Optional[str] = None,  # type: ignore
                   unit_list: Optional[List[str]] = None,  # type: ignore
                   scale: Optional[List[float]] = None,  # type: ignore
                   version: str = "0.4",
                   name: str = "unnamed") -> 'Pyramid':
        """Create a pyramid from any array type without writing to NGFF store."""
        ndim = array.ndim
        axes: str = axis_order if axis_order is not None else defaults.axis_order[:ndim]
        units_list: List[str] = unit_list if unit_list is not None else [defaults.unit_map[ax] for ax in axes]
        units_list = units_list[:ndim]
        scale_list: List[float] = scale if scale is not None else [defaults.scale_map[ax] for ax in axes]
        scale_list = scale_list[:ndim]
        
        self._array_layers = {'0': array}
        self.meta = NGFFMetadataHandler()
        self.meta.create_new(version=version, name=name)
        self.meta.parse_axes(axes, units_list)  # type: ignore
        self.meta.add_dataset(path='0', scale=scale_list)
        return self

    def from_arrays(self,
                    arrays: List[Union[np.ndarray, da.Array, zarr.Array]],
                    axis_order: Optional[str] = None,  # type: ignore
                    unit_list: Optional[List[str]] = None,  # type: ignore
                    base_scale: Optional[List[float]] = None,  # type: ignore
                    scale_factor: Optional[List[float]] = None,  # type: ignore
                    scales: Optional[List[List[float]]] = None,  # type: ignore
                    version: str = "0.4",
                    name: str = "unnamed",
                    compressor: Optional[CompressorConfig] = None) -> 'Pyramid':
        """Create a pyramid from a list of arrays at different resolutions.

        Per-level scales are determined with the following precedence:

        1. `scales` - an explicit list of full per-axis scale vectors, one
           per array (e.g. `[[1,1,1], [2,2,2], [4,4,4], ...]`). This is the
           eubi_bridge-aligned form used by `get_downscaled_pyramid`.
        2. `scale_factor` - the per-axis stride factor between consecutive
           levels (e.g. `(2, 2, 2)`); per-level scale is computed as
           `base_scale * scale_factor ** level_index`.
        3. Shape-ratio fallback: `base_scale * base_shape / level_shape`,
           which can deviate slightly from the nominal factor for
           non-evenly-divisible shapes.

        The pyramid's `compressor` (used as the default by `write_pyramid`)
        is determined as follows:

        - If `arrays[0]` is a `zarr.Array`, its existing codec is used and
          `compressor` must be `None` - a `ValueError` is raised otherwise,
          since the codec of an existing zarr store cannot be changed in
          place. Pass `compressor=` to `write_pyramid` to write with a
          different codec.
        - Otherwise, `compressor` is used if given, else defaults to blosc.
        """
        if isinstance(arrays, (np.ndarray, da.Array, zarr.Array)):
            arrays = [arrays]

        base_array = arrays[0]
        ndim = base_array.ndim

        if isinstance(base_array, zarr.Array):
            if compressor is not None:
                raise ValueError(
                    "compressor= cannot be set in from_arrays() when arrays are "
                    "zarr-backed - the codec of an existing zarr store cannot be "
                    "changed in place. The pyramid's compressor is taken from the "
                    "input array's existing codec; pass compressor= to "
                    "write_pyramid() to write with a different codec instead."
                )
            self._compressor = CompressorConfig.from_array(base_array)
        else:
            self._compressor = compressor if compressor is not None else CompressorConfig(name='blosc')

        axes = axis_order if axis_order is not None else defaults.axis_order[:ndim]
        units_list: List[str] = unit_list if unit_list is not None else [defaults.unit_map[ax] for ax in axes]
        units_list = units_list[:ndim]

        if scales is not None:
            base_scale: List[float] = list(scales[0])[:ndim]
        else:
            base_scale_list: List[float] = base_scale if base_scale is not None else [defaults.scale_map[ax] for ax in axes]
            if isinstance(base_scale_list, (list, tuple)) and len(base_scale_list) > 0:
                if isinstance(base_scale_list[0], (int, float)):
                    base_scale = list(base_scale_list)
                else:
                    base_scale = [defaults.scale_map[ax] for ax in axes]
            else:
                base_scale = [defaults.scale_map[ax] for ax in axes]

        self._array_layers = {'0': base_array}
        self.meta = NGFFMetadataHandler()
        self.meta.create_new(version=version, name=name)
        self.meta.parse_axes(axes, units_list)  # type: ignore
        self.meta.add_dataset(path='0', scale=base_scale)

        if len(arrays) > 1:
            if scales is None and scale_factor is not None:
                factor_arr = np.array(scale_factor[:ndim], dtype=float)
            for i, array in enumerate(arrays[1:]):
                if scales is not None:
                    scale_list: List[float] = list(scales[i + 1])[:ndim]
                elif scale_factor is not None:
                    scale_arr = np.multiply(np.power(factor_arr, i + 1), base_scale)
                    scale_list = scale_arr.tolist()
                else:
                    scale_arr = np.multiply(np.divide(base_array.shape, array.shape), base_scale)
                    scale_list = scale_arr.tolist()
                self.meta.add_dataset(path=f'{i+1}', scale=scale_list)
                self._array_layers[f'{i+1}'] = array

        return self

    def from_label_arrays(self,
                          arrays: List[Union[np.ndarray, da.Array, zarr.Array]],
                          axis_order: Optional[str] = None,  # type: ignore
                          unit_list: Optional[List[str]] = None,  # type: ignore
                          scale_factor: Optional[List[float]] = None,  # type: ignore
                          scales: Optional[List[List[float]]] = None,  # type: ignore
                          version: str = "0.4",
                          name: str = "unnamed",
                          compressor: Optional[CompressorConfig] = None,
                          colors: Optional[List[dict]] = None,
                          properties: Optional[List[dict]] = None,
                          source: Optional[Dict[str, Any]] = None) -> 'Pyramid':
        """Create a LABEL pyramid from arrays, exactly like `from_arrays` but stamping
        an OME ``image-label`` metadata block so the result is a valid NGFF label
        image (and no ``omero`` block, which label images do not carry).

        The label arrays are the integer label maps at each resolution. The
        ``image-label`` block is minimal by default (just ``version``); pass
        ``colors`` / ``properties`` / ``source`` to populate it (typically these are
        assembled by the sibling ``pyrametric`` package). ``source`` usually
        stays ``None`` here and is set when the label is attached to a source image
        via `add_image_label`.

        ::

            lbl_pyr = Pyramid().from_label_arrays([labels0], axis_order="zyx",
                                                  scales=[[1, 1, 1]], name="nuclei")
            lbl_pyr.meta.image_label       # {'version': '0.4'}
        """
        self.from_arrays(
            arrays, axis_order=axis_order, unit_list=unit_list, scale_factor=scale_factor,
            scales=scales, version=version, name=name, compressor=compressor,
        )
        if self.meta is not None and self.meta.metadata is not None:
            # label images carry image-label, not omero
            self.meta.metadata.pop('omero', None)
            image_label: Dict[str, Any] = {'version': version}
            if colors is not None:
                image_label['colors'] = colors
            if properties is not None:
                image_label['properties'] = properties
            if source is not None:
                image_label['source'] = source
            self.meta.metadata['image-label'] = image_label
            self.meta._pending_changes = True
        return self.validate()   # a label image must be integer-typed (see validate)

    @property
    def axes(self) -> str:
        """Get axis order string."""
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        return self.meta.axis_order

    @property
    def nlayers(self) -> int:
        """Get number of resolution layers."""
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        return self.meta.nlayers

    @property
    def layers(self) -> Dict[str, Union[da.Array, zarr.Array]]:
        """Get all array layers across all resolutions."""
        if self.gr is None:
            return self._array_layers  # type: ignore
        # Return zarr arrays from persistent storage
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        return {path: self.gr[path] for path in self.meta.resolution_paths}  # type: ignore

    @property
    def base_array(self) -> Union[da.Array, zarr.Array]:
        """Get the base (highest resolution) array."""
        arr = self.layers['0']
        if isinstance(arr, zarr.Array):
            return da.from_zarr(arr)
        return arr

    @property
    def dask_arrays(self) -> Dict[str, da.Array]:
        """Get all layers as dask arrays."""
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        result = {}
        for path in self.meta.resolution_paths:
            arr = self.layers[path]
            if isinstance(arr, zarr.Array):
                result[str(path)] = da.from_zarr(arr)
            else:
                result[str(path)] = arr  # type: ignore
        return result

    @property
    def dynamic_arrays(self) -> Dict[str, 'DynamicArray']:
        """Get all layers as dyna_zarr ``DynamicArray``s -- the pull-based, memory-bounded
        backend counterpart to ``dask_arrays``. Zarr layers are wrapped lazily; a layer that
        is already a ``DynamicArray`` passes through. A dask-backed in-memory layer cannot be
        wrapped (no zarr to pull from) and raises.

        Requires the optional ``dyna_zarr`` package. This is the read side of the dyna backend
        used by ``pyrops`` ops; the write side is in ``IO.write_pyramid``.
        """
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        try:
            from dyna_zarr import DynamicArray
        except ImportError as e:  # pragma: no cover - optional backend
            raise ImportError(
                "dynamic_arrays requires the optional 'dyna_zarr' package"
            ) from e
        result = {}
        for path in self.meta.resolution_paths:
            arr = self.layers[path]
            if isinstance(arr, DynamicArray):
                result[str(path)] = arr
            elif isinstance(arr, zarr.Array):
                result[str(path)] = DynamicArray(arr)
            else:
                raise TypeError(
                    f"layer {path!r} is a {type(arr).__name__}; the dyna_zarr backend needs a "
                    f"zarr-backed or DynamicArray layer (cannot wrap an in-memory dask/numpy array)"
                )
        return result

    @property
    def scale_factor_dict(self) -> Dict[str, Dict[str, float]]:
        """Get scale factors as dictionaries for each resolution."""
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        shapes = [self.layers[key].shape for key in self.meta.resolution_paths]
        scale_factors = np.divide(shapes[0], shapes)
        scale_factor_list = scale_factors.tolist()
        scale_factor_list = [dict(zip(self.meta.axis_order, scale))  # type: ignore
                             for scale in scale_factor_list]
        return {pth: scale
                for pth, scale in
                zip(self.meta.resolution_paths, scale_factor_list)}

    async def update_downscaler(self,
                                scale_factor=None,
                                n_layers: Union[int, str, None] = None,
                                downscale_method='simple',
                                backend='numpy',
                                smart_scale_factor=None,
                                **kwargs) -> 'Pyramid':
        """Update downscaler for creating pyramid layers."""
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")

        min_dimension_size = kwargs.get('min_dimension_size', 64)

        darr = self.layers['0']
        shape = darr.shape

        if scale_factor is None:
            scale_factor = tuple([defaults.scale_factor_map[key] for key in self.axes])

        if n_layers in (None, 'default', 'auto'):
            n_layers = calculate_n_layers(shape, scale_factor, min_dimension_size)
        n_layers = int(n_layers)

        scale = self.meta.get_base_scale()
        scale_factor = tuple(np.minimum(darr.shape, scale_factor))

        self.downscaler = Downscaler(
            array=darr,
            scale_factor=scale_factor,
            n_layers=n_layers,
            scale=scale,
            downscale_method=downscale_method,
            backend=backend,
            smart_scale_factor=smart_scale_factor,
        )
        await self.downscaler.update()
        return self

    def get_downscaled_pyramid(self) -> 'Pyramid':
        """Get pyramid with downscaled layers. When no downscaler has been configured
        (e.g. via a prior `downscale(...)`), build one whose depth is chosen
        automatically: keep adding levels until the largest downscaled spatial
        dimension would drop below `min_dimension_size` (64), matching `downscale()`."""
        if not hasattr(self, 'downscaler'):
            asyncio.run(self.update_downscaler(n_layers=None))
        
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        
        arrays = self.downscaler._downscaled_arrays  # type: ignore
        scales = self.downscaler.dm.scales  # type: ignore
        unit_list = self.meta.unit_list
        axis_order = self.meta.axis_order
        name = self.meta.multiscales.get('name', 'unnamed') if self.meta.metadata else 'unnamed'
        
        # Convert scales to list format
        scales_list: List[Union[List[float], np.ndarray]] = scales.tolist() if hasattr(scales, 'tolist') else list(scales)  # type: ignore
        
        # Filter out None values from unit_list for passing to from_arrays
        unit_list_clean: List[str] = [u for u in unit_list if u is not None]

        # from_arrays() derives the compressor from arrays[0] itself when it's
        # zarr-backed (and forbids passing compressor= in that case) - only
        # pass the source pyramid's compressor through explicitly when the
        # base array is in-memory (dask/numpy).
        compressor = None if isinstance(arrays[0], zarr.Array) else self.compressor

        new_pyr = Pyramid().from_arrays(
            arrays=arrays,
            axis_order=axis_order,
            unit_list=unit_list_clean if unit_list_clean else None,  # type: ignore
            scales=scales_list,  # type: ignore
            name=name,
            compressor=compressor,
            version=self.meta.version,
        )
        if new_pyr.meta and new_pyr.meta.metadata and self.meta and self.meta.metadata:
            src_md = self.meta.metadata
            new_md = new_pyr.meta.metadata
            # carry over every top-level attribute except 'multiscales' (rebuilt
            # here with the new resolution levels + recomputed scales)
            for k, v in src_md.items():
                if k != 'multiscales':
                    new_md[k] = copy.deepcopy(v)
            # carry multiscale-level extras (name, `metadata` block, custom keys)
            # while keeping the freshly built datasets/axes
            src_ms = self.meta.multiscales
            new_ms = new_pyr.meta.multiscales
            for k, v in src_ms.items():
                if k not in ('datasets', 'axes'):
                    new_ms[k] = copy.deepcopy(v)
            # preserve the base level's translation on the new level 0
            base_ct = src_ms['datasets'][0].get('coordinateTransformations', [])
            trans = [t for t in base_ct if t.get('type') == 'translation']
            if trans:
                new_ms['datasets'][0].setdefault('coordinateTransformations', []).extend(
                    copy.deepcopy(trans)
                )
            new_pyr.meta._pending_changes = True
        return new_pyr

    def downscale(self,
                  n_layers: Optional[int] = None,
                  min_dimension_size: int = 64,
                  scale_factor=None,
                  downscale_method: str = 'simple',
                  backend: str = 'numpy',
                  smart_scale_factor=None,
                  defer: bool = True,
                  **kwargs) -> 'Pyramid':
        """Plan (or build) a full multi-resolution pyramid from level 0.

        Typical terminal step after chaining shape-preserving ops on a single
        level::

            full = pyr_rl.downscale()                       # until largest dim < 64
            full = pyr_rl.downscale(min_dimension_size=128)  # coarser stop
            full = pyr_rl.downscale(n_layers=5)             # exactly 5 levels
            io.write_pyramid(full, out_path, overwrite=True)

        DEFERRED BY DEFAULT (``defer=True``). ``downscale`` then only RECORDS a
        plan (`_downscale_plan`) on the returned single-level pyramid and builds
        NOTHING - no lower-level arrays, no computation. The lower levels are
        realised only at write time, PROGRESSIVELY FROM DISK: the writer streams
        level 0 to the store once, re-reads it, and derives each coarser level
        from the on-disk parent. This is what makes downscaling an EXPENSIVE lazy
        base (e.g. a cellpose segmentation) cost the base ONE compute instead of
        recomputing it once per output level.

        ``defer=False`` restores the old eager behaviour: build every level now as
        a lazy array derived from level 0 (fine for cheap bases; re-executes an
        expensive base per level when written - see the deferred path above).

        Layer count is controlled two ways:

        - ``n_layers=None`` (default): keep adding levels until the **largest**
          spatial dimension would drop below ``min_dimension_size``.
        - ``n_layers=<int>``: use exactly that many levels (``min_dimension_size``
          is then ignored).

        Parameters
        ----------
        n_layers : int, optional
            Hard-coded number of resolution levels. ``None`` (default) derives
            the count from ``min_dimension_size``.
        min_dimension_size : int
            Stop downscaling once the largest dimension reaches this size (used
            only when ``n_layers is None``). Default 64.
        scale_factor : sequence, optional
            Per-axis stride between levels; defaults to 2 on z/y/x and 1 on
            t/c (from ``defaults.scale_factor_map``).
        downscale_method : str
            'simple' (nearest/stride - correct for LABELS), 'mean', or 'median'.
        defer : bool
            Record a plan and build nothing (default True), or eagerly build all
            levels now (False).
        backend, smart_scale_factor, **kwargs :
            Passed through to ``update_downscaler`` when ``defer=False``; stored in
            the plan when deferred.
        """
        if defer:
            # record the recipe; the writer expands it progressively from disk
            self._downscale_plan = {
                'n_layers': n_layers,
                'min_dimension_size': min_dimension_size,
                'scale_factor': scale_factor,
                'downscale_method': downscale_method,
                'backend': backend,
                'smart_scale_factor': smart_scale_factor,
            }
            return self
        asyncio.run(self.update_downscaler(
            scale_factor=scale_factor,
            n_layers=n_layers,
            downscale_method=downscale_method,
            backend=backend,
            smart_scale_factor=smart_scale_factor,
            min_dimension_size=min_dimension_size,
            **kwargs,
        ))
        return self.get_downscaled_pyramid()

    def _resolve_level_chunks(self, chunk_shape, chunk_size_mb, level, shape, axes, dtype):
        """Per-axis chunk tuple for one level, capped at `shape`."""
        from ome_zarr_pyramid.utils.scale import autocompute_chunk_shape
        if chunk_size_mb is not None:
            if isinstance(chunk_size_mb, dict):
                if level not in chunk_size_mb:
                    raise ValueError(f"chunk_size_mb dict has no entry for resolution level {level}")
                mb = chunk_size_mb[level]
            elif isinstance(chunk_size_mb, (tuple, list)):
                if level >= len(chunk_size_mb):
                    raise ValueError(
                        f"chunk_size_mb has {len(chunk_size_mb)} entries but resolution level "
                        f"{level} was requested")
                mb = chunk_size_mb[level]
            else:
                mb = chunk_size_mb
            return autocompute_chunk_shape(shape, axes, float(mb), dtype)
        if isinstance(chunk_shape, dict):
            if level not in chunk_shape:
                raise ValueError(f"chunk_shape dict has no entry for resolution level {level}")
            spec = chunk_shape[level]
        else:
            spec = chunk_shape
        if len(spec) != len(shape):
            raise ValueError(
                f"chunk_shape {tuple(spec)} has {len(spec)} entries but level {level} is "
                f"{len(shape)}-D (axes '{axes}')"
            )
        return tuple(min(int(c), int(shape[i])) for i, c in enumerate(spec))

    def rechunk(self, chunk_shape=None, chunk_size_mb=None) -> 'Pyramid':
        """Return a NEW Pyramid with every level rechunked.

        The STORAGE-chunk knob, decoupled from any PROCESSING tile size an op may
        have imposed on the dask arrays. Provide exactly one of:

        - ``chunk_shape``: a per-axis tuple in dimension order, applied to EVERY
          level (each side capped at that level's extent); OR a dict mapping each
          resolution level index to its own per-axis tuple, e.g.
          ``{0: (1, 1, 256, 256), 1: (1, 1, 128, 128)}``.
        - ``chunk_size_mb``: a target uncompressed chunk size in MB. Each level's
          chunk is auto-computed ISOTROPIC over the spatial axes (a square in 2-D,
          a cube in 3-D; t/c stay at 1), honouring the dtype. May be a SCALAR (same
          budget for every level) or PER-LEVEL as a sequence in level order (e.g.
          ``(2, 1, 0.5)`` = 2 MB at level 0, 1 MB at level 1, 0.5 MB at level 2) or
          a dict ``{level: mb}``.

        Passing neither restores the pyramid's recorded ``_storage_chunks`` (the
        on-disk chunking it was read with); if none was recorded, returns unchanged.

        Returns
        -------
        Pyramid
            New pyramid, identical data/metadata, requested chunking; its
            ``_storage_chunks`` is set to the resolved base-level chunk shape.
        """
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        if chunk_shape is not None and chunk_size_mb is not None:
            raise ValueError("give chunk_shape OR chunk_size_mb, not both")

        paths = self.meta.resolution_paths
        axes = self.meta.axis_order
        if chunk_shape is None and chunk_size_mb is None:
            sc = getattr(self, '_storage_chunks', None)
            if sc is None:
                return self
            chunk_shape = tuple(sc)  # apply the recorded base chunk to every level (capped)

        new_arrays = []
        base_chunk = None
        for lvl, p in enumerate(paths):
            a = self.dask_arrays[p]
            spec = self._resolve_level_chunks(chunk_shape, chunk_size_mb, lvl, a.shape, axes, a.dtype)
            if base_chunk is None:
                base_chunk = spec
            new_arrays.append(a.rechunk(spec))

        compressor = None if isinstance(new_arrays[0], zarr.Array) else self.compressor
        unit_clean = [u for u in self.meta.unit_list if u is not None]
        new_pyr = Pyramid().from_arrays(
            arrays=new_arrays,
            axis_order=axes,
            unit_list=unit_clean if unit_clean else None,  # type: ignore
            scales=[self.meta.get_scale(p) for p in paths],  # type: ignore
            name=self.meta.multiscales.get('name', 'unnamed') if self.meta.metadata else 'unnamed',
            compressor=compressor,
            version=self.meta.version,
        )
        # rechunking changes neither shapes, scales nor axes: carry metadata verbatim
        if new_pyr.meta is not None and self.meta.metadata is not None:
            new_pyr.meta.metadata = copy.deepcopy(self.meta.metadata)
            new_pyr.meta._pending_changes = True
        new_pyr._storage_chunks = tuple(base_chunk) if base_chunk is not None else None
        return new_pyr

    def _read_ct(self, path):
        """(scale, translation) vectors for a dataset `path`; translation may be None."""
        scale, trans = None, None
        for d in self.meta.multiscales["datasets"]:  # type: ignore
            if d["path"] == str(path):
                for t in d.get("coordinateTransformations", []):
                    if t.get("type") == "scale":
                        scale = list(t["scale"])
                    elif t.get("type") == "translation":
                        trans = list(t["translation"])
                break
        if scale is None:
            scale = [1.0] * len(self.axes)
        return scale, trans

    def isel(self, **indexers) -> 'Pyramid':
        """Integer/positional selection along named axes (xarray-style), across
        ALL levels, returning a new sub-`Pyramid`.

        Each keyword is `axis_letter=index`, where `index` is an int (selects one
        position and DROPS the axis), a `slice` (`start:stop:step`; a `step`
        strided-subsamples and rescales the axis), or a LIST/array of ints (fancy
        selection: the axis is KEPT and those positions are gathered in order, e.g.
        `c=[3, 4]` keeps two channels; a single-element list keeps a size-1 axis,
        unlike an int). Multiple list axes are selected orthogonally. Indices are in
        FULL-RESOLUTION (level-0) coordinates and mapped to each level by the TRUE
        per-axis downsample factor `scale_level / scale_0` (from the pixel-size
        metadata, not the array-shape ratio - the latter is inexact for odd sizes),
        rounded.

        Axis keys also accept the long names `channels`/`channel` -> `c`, `time` ->
        `t`.

        Coordinate metadata is updated: translation shifts by `start*scale`, scale
        multiplies by `step`, dropped axes are removed, and indexing the channel
        axis `c` subsets the omero channels accordingly. A uniformly-spaced list
        updates scale/translation exactly; a non-uniform list gathers the data but
        keeps the axis scale as-is (its spacing is no longer a single affine step).
        """
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        axstr = self.meta.axis_order
        ndim = len(axstr)

        _axis_aliases = {"channels": "c", "channel": "c", "time": "t"}
        indexers = {_axis_aliases.get(a, a): idx for a, idx in indexers.items()}

        def _uniform_step(vals):
            """Positive common step of an arithmetic-progression list (1 for a
            single element), else None -- signals whether a list selection is a
            plain strided subsample the scale can represent exactly."""
            if len(vals) == 1:
                return 1
            diffs = {vals[k + 1] - vals[k] for k in range(len(vals) - 1)}
            if len(diffs) == 1:
                d = next(iter(diffs))
                if d > 0:
                    return d
            return None

        for a in indexers:
            if a not in axstr:
                raise ValueError(f"isel: axis '{a}' not present in pyramid axes '{axstr}'")
        paths = self.meta.resolution_paths
        base_scale, _ = self._read_ct(paths[0])
        base_shape = self.dask_arrays[paths[0]].shape

        resolved = {}
        for a, idx in indexers.items():
            i = axstr.index(a)
            n0 = base_shape[i]
            if isinstance(idx, (int, np.integer)):
                l0 = int(idx) + n0 if int(idx) < 0 else int(idx)
                if not (0 <= l0 < n0):
                    raise IndexError(f"isel: index {idx} out of range for axis '{a}' (size {n0})")
                resolved[i] = ("int", l0)
            elif isinstance(idx, slice):
                start, stop, step = idx.indices(n0)
                if step <= 0:
                    raise ValueError("isel: only positive step is supported")
                resolved[i] = ("slice", start, stop, step)
            elif isinstance(idx, (list, tuple, np.ndarray)):
                seq = np.asarray(idx)
                if seq.ndim != 1 or seq.size == 0:
                    raise ValueError(
                        f"isel: list index for axis '{a}' must be a non-empty 1-D sequence")
                if not np.issubdtype(seq.dtype, np.integer):
                    raise TypeError(
                        f"isel: list index for axis '{a}' must contain integers, got dtype {seq.dtype}")
                l0s = []
                for v in seq.tolist():
                    lv = v + n0 if v < 0 else v
                    if not (0 <= lv < n0):
                        raise IndexError(
                            f"isel: index {v} out of range for axis '{a}' (size {n0})")
                    l0s.append(int(lv))
                resolved[i] = ("list", l0s)
            else:
                raise TypeError(
                    f"isel: index for axis '{a}' must be int, slice, or list of ints, "
                    f"got {type(idx).__name__}")

        dropped = {i for i, r in resolved.items() if r[0] == "int"}
        keep = [i for i in range(ndim) if i not in dropped]
        out_axes = "".join(axstr[i] for i in keep)
        if not out_axes:
            raise ValueError("isel: cannot index away every axis")

        channel_indices = None
        if "c" in indexers:
            r = resolved[axstr.index("c")]
            if r[0] == "int":
                channel_indices = [r[1]]
            elif r[0] == "slice":
                channel_indices = list(range(r[1], r[2], r[3]))
            else:  # list
                channel_indices = list(r[1])

        new_arrays, new_scales, new_translations = [], [], []
        for p in paths:
            arr = self.dask_arrays[p]
            lvl_scale, lvl_trans = self._read_ct(p)
            if lvl_trans is None:
                lvl_trans = [0.0] * ndim
            slices = [slice(None)] * ndim
            sc, tr = list(lvl_scale), list(lvl_trans)
            list_takes = []  # (axis_index, [level indices]) applied after the basic slice
            for i, r in resolved.items():
                f = lvl_scale[i] / base_scale[i] if base_scale[i] else 1.0  # true factor from pixel scale
                shp = arr.shape[i]
                if r[0] == "int":
                    li = min(int(round(r[1] / f)), shp - 1)
                    slices[i] = li
                    tr[i] = lvl_trans[i] + li * lvl_scale[i]
                elif r[0] == "slice":
                    _, s0, e0, st0 = r
                    ls = min(max(int(round(s0 / f)), 0), shp)
                    le = min(max(int(round(e0 / f)), ls), shp)
                    lstep = max(1, int(round(st0 / f)))
                    slices[i] = slice(ls, le, lstep)
                    sc[i] = lvl_scale[i] * lstep
                    tr[i] = lvl_trans[i] + ls * lvl_scale[i]
                else:  # list -- keep axis, gather chosen positions (deferred take)
                    l0s = r[1]
                    lis = [min(int(round(x / f)), shp - 1) for x in l0s]
                    list_takes.append((i, lis))
                    step0 = _uniform_step(l0s)
                    if step0 is not None:  # a strided subsample: scale is exact
                        lstep = max(1, int(round(step0 / f))) if len(l0s) > 1 else 1
                        sc[i] = lvl_scale[i] * lstep
                    # non-uniform: leave scale as-is (best-effort; data still gathered)
                    tr[i] = lvl_trans[i] + lis[0] * lvl_scale[i]
            out = arr[tuple(slices)]
            # apply fancy selections one axis at a time: a single advanced index per
            # getitem gives orthogonal selection (a shared tuple would cross-product).
            drop = {j for j, rr in resolved.items() if rr[0] == "int"}
            for i, lis in list_takes:
                pos = sum(1 for j in range(i) if j not in drop)  # axis position after drops
                out = out[(slice(None),) * pos + (np.asarray(lis, dtype=int),)]
            new_arrays.append(out)
            new_scales.append([sc[i] for i in keep])
            new_translations.append([tr[i] for i in keep])

        return self._build_indexed(new_arrays, out_axes, new_scales, new_translations, channel_indices)

    def _build_indexed(self, arrays, out_axes, scales, translations, channel_indices):
        src_units = dict(zip(self.meta.axis_order, self.meta.unit_list))  # type: ignore
        units = [src_units[a] for a in out_axes]
        units_clean = [u for u in units if u is not None] or None
        new = Pyramid().from_arrays(
            arrays, axis_order=out_axes, unit_list=units_clean,  # type: ignore
            scales=scales, version=self.meta.version, name=self.meta.tag,  # type: ignore
        )
        if self.meta.metadata and new.meta and new.meta.metadata:  # type: ignore
            new_md = copy.deepcopy(self.meta.metadata)
            ms = new_md["multiscales"][0]
            axes_by_name = {a["name"]: a for a in ms["axes"]}
            ms["axes"] = [copy.deepcopy(axes_by_name[a]) for a in out_axes]
            datasets = []
            for i in range(len(arrays)):
                cts = [{"type": "scale", "scale": [float(x) for x in scales[i]]}]
                if translations[i] is not None:
                    cts.append({"type": "translation", "translation": [float(x) for x in translations[i]]})
                datasets.append({"path": str(i), "coordinateTransformations": cts})
            ms["datasets"] = datasets
            if channel_indices is not None and isinstance(new_md.get("omero"), dict):
                chans = self.meta.get_channels()
                if chans:
                    new_md["omero"]["channels"] = [
                        copy.deepcopy(chans[k]) for k in channel_indices if k < len(chans)
                    ]
            new.meta.metadata = new_md
            new.meta._pending_changes = True
        return new

    def __getitem__(self, key) -> 'Pyramid':
        """Positional numpy-style slicing across all levels: `pyr[0, 2:10, :, :]`
        (see `isel` for semantics)."""
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        axstr = self.meta.axis_order
        ndim = len(axstr)
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            ei = key.index(Ellipsis)
            n_explicit = sum(1 for k in key if k is not Ellipsis)
            key = key[:ei] + (slice(None),) * (ndim - n_explicit) + key[ei + 1:]
        if len(key) > ndim:
            raise IndexError(f"too many indices for {ndim}-D pyramid '{axstr}'")
        indexers = {}
        for i, k in enumerate(key):
            if isinstance(k, slice) and k == slice(None):
                continue
            indexers[axstr[i]] = k
        return self.isel(**indexers)

    def select_levels(self, *levels) -> 'Pyramid':
        """Return a new pyramid holding an arbitrary subset of resolution LEVELS
        (not necessarily contiguous). The FINEST selected level becomes level 0 of
        the output; each kept level keeps its own pixel scale/translation verbatim.

        Levels may be given as individual indices, a single list/tuple/range, or a
        slice; negative indices count from the coarsest; duplicates are dropped and
        the result is ordered finest -> coarsest:

            pyr.select_levels(0, 2, 3)      # levels 0, 2 and 3
            pyr.select_levels(2)            # just level 2 (a single-level pyramid)
            pyr.select_levels([1, 3, 4])
            pyr.select_levels(range(0, 3))  # a contiguous run (levels 0, 1, 2)
            pyr.select_levels(slice(1, 5))  # levels 1..4

        DEFERRED-aware: on a pyramid that still carries a `downscale()` plan (only
        level 0 is materialised, the rest is a recipe), the plan is first resolved
        lazily (no compute). For a CONTIGUOUS selection the result stays deferred -
        re-based to the finest kept level, with a narrowed plan for the coarser kept
        levels (derived from that base at write time). So `downscale(8).select_levels(
        range(3, 8))` costs the base ONE compute at level-3 resolution and never
        builds levels 0..2. A non-contiguous selection is returned as lazy arrays.
        """
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        plan = getattr(self, '_downscale_plan', None)
        if plan is not None:
            return self._select_levels_deferred(levels, plan)
        sel = self._resolve_level_indices(levels, self.meta.nlayers)
        new = self._select_materialized(sel)
        sc = getattr(self, '_storage_chunks', None)
        if sc is not None:
            new._storage_chunks = sc
        return new

    @staticmethod
    def _resolve_level_indices(specs, n) -> 'List[int]':
        """Flatten the variadic `select_levels` specs (individual ints, a single
        list/tuple/range, or a slice) into a sorted, unique, ascending list of valid
        level indices (finest first). Negative indices count from the coarsest."""
        if len(specs) == 1 and isinstance(specs[0], slice):
            idxs: List[int] = list(range(*specs[0].indices(n)))
        elif len(specs) == 1 and isinstance(specs[0], (list, tuple, range)):
            idxs = list(specs[0])
        else:
            idxs = list(specs)
        if not idxs:
            raise ValueError("select_levels: choose at least one level")
        out = []
        for i in idxs:
            ii = int(i)
            ii = ii + n if ii < 0 else ii
            if not (0 <= ii < n):
                raise IndexError(f"select_levels: level {i} out of range (pyramid has {n} levels)")
            out.append(ii)
        return sorted(set(out))

    def _select_materialized(self, sel: 'List[int]') -> 'Pyramid':
        """Build a new pyramid from the given (sorted, unique) level indices, taking
        each level's array + coordinate metadata; the first becomes level 0."""
        paths = self.meta.resolution_paths
        arrays = [self.dask_arrays[paths[k]] for k in sel]
        scales = [self.meta.get_scale(paths[k]) for k in sel]
        new = Pyramid().from_arrays(
            arrays, axis_order=self.meta.axis_order, unit_list=self.meta.unit_list,  # type: ignore
            scales=scales, version=self.meta.version, name=self.meta.tag,  # type: ignore
        )
        if self.meta.metadata and new.meta and new.meta.metadata:
            new_md = copy.deepcopy(self.meta.metadata)
            ms = new_md["multiscales"][0]
            src_ds = {d["path"]: d for d in ms["datasets"]}
            new_ds = []
            for out_i, k in enumerate(sel):
                d = copy.deepcopy(src_ds[paths[k]])   # keep this level's scale + translation
                d["path"] = str(out_i)
                new_ds.append(d)
            ms["datasets"] = new_ds
            new.meta.metadata = new_md
            new.meta._pending_changes = True
        return new

    def _select_levels_deferred(self, levels, plan: dict) -> 'Pyramid':
        """`select_levels` on a still-deferred pyramid (see `select_levels`).

        Resolve the plan into a full lazy pyramid, then select the requested levels.
        A CONTIGUOUS selection stays deferred (finest kept -> new base + a narrowed
        plan for the coarser kept levels); a non-contiguous one is returned as the
        selected lazy arrays."""
        keys = ('n_layers', 'min_dimension_size', 'scale_factor',
                'downscale_method', 'backend', 'smart_scale_factor')
        dk = {k: plan[k] for k in keys}
        # resolve the plan into a full lazy pyramid WITHOUT disturbing self's plan
        saved = self.__dict__.pop('_downscale_plan', None)
        try:
            full = self.downscale(defer=False, **dk)   # lazy: no compute, all levels
        finally:
            if saved is not None:
                self._downscale_plan = saved

        sel = self._resolve_level_indices(levels, full.meta.nlayers)
        sc = getattr(self, '_storage_chunks', None)

        if sel == list(range(sel[0], sel[-1] + 1)):
            # contiguous -> keep the coarser kept levels DEFERRED, re-based on level sel[0]
            new = full._select_materialized([sel[0]])   # finest kept -> single-level base
            n_kept = len(sel)
            if n_kept > 1:
                narrowed = {k: plan[k] for k in keys}
                narrowed['n_layers'] = n_kept
                new._downscale_plan = narrowed
        else:
            # non-contiguous -> select the resolved lazy levels directly
            new = full._select_materialized(sel)
        if sc is not None:
            new._storage_chunks = sc
        return new

    def _clone_metadata_only(self) -> 'Pyramid':
        """A new Pyramid SHARING this one's level arrays with a DEEP-COPIED metadata
        dict, so metadata edits (omero, etc.) never mutate the original. The pixel
        data is unchanged; the deferred downscale plan and storage chunks are carried
        (metadata-only edits are shape-preserving). Basis for `set_channels` /
        `set_display_range`."""
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        paths = self.meta.resolution_paths
        new = Pyramid().from_arrays(
            [self.dask_arrays[p] for p in paths], axis_order=self.meta.axis_order,  # type: ignore
            unit_list=self.meta.unit_list, scales=[self.meta.get_scale(p) for p in paths],  # type: ignore
            version=self.meta.version, name=self.meta.tag,  # type: ignore
        )
        if self.meta.metadata and new.meta and new.meta.metadata:
            new.meta.metadata = copy.deepcopy(self.meta.metadata)
            new.meta._pending_changes = True
        plan = getattr(self, '_downscale_plan', None)
        if plan is not None:
            new._downscale_plan = dict(plan)
        sc = getattr(self, '_storage_chunks', None)
        if sc is not None:
            new._storage_chunks = sc
        # metadata-only edits keep the geometry, so the attached labels stay valid
        new._labels = dict(self._labels)
        return new

    # ------------------------------------------------------------------
    # NGFF labels/ collection (label images attached to this source image)
    # ------------------------------------------------------------------

    @property
    def labels(self) -> 'LabelCollection':
        """The image's NGFF ``labels/`` collection as a read-only mapping
        ``name -> label Pyramid`` (see `LabelCollection`). Populated when reading an
        image that has a ``labels/`` group, or by `add_image_label`. Empty for a
        plain in-memory pyramid."""
        return LabelCollection(self._labels)

    def _resolve_label_name(self, image_label: 'Pyramid', name: Optional[str]) -> str:
        if name is not None:
            return name
        ms = image_label.meta.multiscales if image_label.meta is not None else None
        nm = ms.get('name') if isinstance(ms, dict) else None
        # 'unnamed' is the library's placeholder name (from_arrays/from_array default);
        # treat it as no-name so the caller is forced to supply a real label name.
        if nm and nm != 'unnamed':
            return str(nm)
        raise ValueError(
            "add_image_label: could not determine a label name from the label "
            "pyramid's metadata (no meaningful multiscales 'name'); pass name=..."
        )

    def add_image_label(self, image_label: 'Pyramid', name: Optional[str] = None, *,
                        source: Optional[Dict[str, Any]] = None) -> 'Pyramid':
        """Attach a label image to this (source) image, returning a NEW `Pyramid`
        whose `labels` collection includes it - the in-memory equivalent of an NGFF
        ``<image>/labels/<name>`` (the pixel data is unchanged).

        Parameters
        ----------
        image_label : Pyramid
            The label pyramid (an integer-labelled multiscale image). It becomes a
            label entry; an ``image-label`` metadata block is ensured (a minimal
            ``{"version": ...}`` is created if absent) and its ``source`` is set to
            point back at this image (``{"image": "../../"}`` per NGFF) unless
            `source` is given.
        name : str, optional
            The name to register the label under. Defaults to the label pyramid's own
            ``multiscales`` name; if that is missing, `name` is REQUIRED.
        source : dict, optional
            Override for the image-label ``source`` back-link.

        ::

            img = IO().read_pyramid("image.ome.zarr")
            img = img.add_image_label(nuclei_pyr, "nuclei")
            img.labels.names            # ['nuclei']
            img.labels["nuclei"]        # -> Pyramid (the label image)
        """
        if not isinstance(image_label, Pyramid):
            raise TypeError("add_image_label: image_label must be a Pyramid")
        if image_label.meta is None:
            raise ValueError("add_image_label: image_label pyramid is not initialized")
        label_name = self._resolve_label_name(image_label, name)

        # deep-copy the label's metadata so we can stamp image-label without mutating
        # the caller's pyramid, then ensure the image-label block + source back-link
        label = image_label._clone_metadata_only()
        if label.meta is not None and label.meta.metadata is not None:
            # a label image carries image-label, not omero (the source pyramid it was
            # derived from may have left an omero block behind - drop it)
            label.meta.metadata.pop('omero', None)
            il = dict(label.meta.metadata.get('image-label') or {})
            il.setdefault('version', label.meta.version)
            il['source'] = source if source is not None else {'image': '../../'}
            label.meta.metadata['image-label'] = il
            label.meta._pending_changes = True

        new = self._clone_metadata_only()
        new._labels = dict(self._labels)
        new._labels[label_name] = label
        return new

    def validate(self) -> 'Pyramid':
        """Validate the pyramid's metadata / dtype invariants, raising on violation
        and returning ``self`` on success (so it chains). A first, lightweight schema
        check; richer validation (e.g. via pydantic) may follow.

        - Must carry NGFF ``multiscales`` metadata (a resolution pyramid).
        - A LABEL image (``meta.is_label``) must use an INTEGER pixel type - never
          BOOLEAN (a boolean mask is a segmentation output, not a labelled image) and
          never float - and must NOT carry ``omero`` (intensity-display metadata).
        """
        if self.meta is None or not self.meta.is_multiscales:
            raise ValueError("Pyramid has no valid NGFF multiscales metadata")
        if self.meta.is_label:
            dtype = self.base_array.dtype
            if np.issubdtype(dtype, np.bool_):
                raise TypeError(
                    "label image has BOOLEAN dtype: a boolean mask is a segmentation "
                    "output, not a labelled image. Convert to an integer type first - "
                    "e.g. .astype('uint16') (any bitsize that fits your label ids)."
                )
            if not np.issubdtype(dtype, np.integer):
                raise TypeError(
                    f"label image has non-integer dtype {dtype!r}: label ids are "
                    "non-negative integers. Convert with .astype('uint16') etc."
                )
            if self.meta.metadata is not None and self.meta.metadata.get('omero') is not None:
                raise ValueError(
                    "label image must not carry omero metadata (omero is intensity-"
                    "display metadata for images, not labels)."
                )
        return self

    def set_image_label(self, *, colors: Optional[List[dict]] = None,
                        properties: Optional[List[dict]] = None,
                        source: Optional[Dict[str, Any]] = None,
                        version: Optional[str] = None,
                        object_features: Optional[Any] = None) -> 'Pyramid':
        """Set (merge) this pyramid's OME ``image-label`` metadata, returning a NEW
        `Pyramid` (pixel data unchanged - only metadata). Marks the pyramid as a LABEL
        image: an ``image-label`` block is ensured and any ``omero`` block is dropped
        (a label carries image-label, not omero). Only the given fields are written;
        the others are preserved. The label-image counterpart of `set_channels`.

        Parameters
        ----------
        colors : list of dict, optional
            ``image-label.colors`` - per label-value display colours, each
            ``{"label-value": int, "rgba": [r, g, b, a]}``.
        properties : list of dict, optional
            ``image-label.properties`` - per-object measurements, each a dict with a
            ``"label-value"`` key plus arbitrary columns (e.g. from
            ``pyrametric.extract_features``).
        source : dict, optional
            ``image-label.source`` back-link (e.g. ``{"image": "../../"}``).
        version : str, optional
            Override the image-label version (defaults to the pyramid's NGFF version).
        object_features : optional
            A convenience shortcut: any object exposing ``to_ngffcolors()`` and
            ``to_ngffprops()`` (e.g. ``pyrametric.ObjectFeatures``). Its output
            fills ``colors``/``properties`` when those are not passed explicitly
            (explicit ``colors``/``properties`` win). Duck-typed - no dependency on
            the labels package.

        ::

            lbl_pyr = lbl_pyr.set_image_label(object_features=of)          # from measurements
            lbl_pyr = lbl_pyr.set_image_label(                            # or literal values
                colors=[{"label-value": 1, "rgba": [255, 0, 0, 255]}],
                properties=[{"label-value": 1, "area": 812.0}],
            )
            img = img.add_image_label(lbl_pyr, name="nuclei")   # colours/properties carried
        """
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        if object_features is not None:
            if colors is None:
                colors = object_features.to_ngffcolors()
            if properties is None:
                properties = object_features.to_ngffprops()
        new = self._clone_metadata_only()
        if new.meta is not None and new.meta.metadata is not None:
            new.meta.metadata.pop('omero', None)   # a label carries image-label, not omero
            il = dict(new.meta.metadata.get('image-label') or {})
            il['version'] = version if version is not None else il.get('version', new.meta.version)
            if colors is not None:
                il['colors'] = colors
            if properties is not None:
                il['properties'] = properties
            if source is not None:
                il['source'] = source
            new.meta.metadata['image-label'] = il
            new.meta._pending_changes = True
        return new.validate()   # declaring a label: enforce integer dtype / no omero

    def rename(self, name: str) -> 'Pyramid':
        """Return a NEW `Pyramid` with its multiscales ``name`` set to `name` (pixel
        data and all other metadata unchanged). `name` is the image/series identifier
        in the NGFF ``multiscales`` metadata (read back via ``pyr.meta.tag``).

        ::

            pyr = pyr.rename("nuclei")
            pyr.meta.tag            # 'nuclei'
        """
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        if not isinstance(name, str) or not name:
            raise ValueError("rename: name must be a non-empty string")
        new = self._clone_metadata_only()
        if new.meta is not None and new.meta.metadata is not None:
            ms = new.meta.metadata.get('multiscales')
            if ms:
                ms[0]['name'] = name
                new.meta._pending_changes = True
        return new

    def set_channels(self, overrides: Dict[Union[int, str], Dict[str, Any]]) -> 'Pyramid':
        """Update per-channel omero display metadata, returning a NEW `Pyramid`
        (pixel data unchanged - only metadata). SPARSE and MERGING: only the named
        channels and named fields are touched; everything else is preserved.

        `overrides` maps a channel KEY to a dict of fields to change:

        - key: an ``int`` channel index, or a ``str`` matching a channel's current
          ``label``.
        - fields: any omero channel field, commonly
            * ``"color"``  - a standard colour NAME (``"red"``, ``"green"``,
              ``"magenta"``, ...), a hex string with or without '#' (``"FF0000"``,
              ``"#FF0000"``, or 3-digit ``"F00"``), or an ``(r, g, b)`` triple;
              stored as OME hex 'RRGGBB'.
            * ``"label"``  - channel name
            * ``"active"`` - bool (whether the channel is displayed)
            * ``"window"`` - a dict MERGED into the existing window, so you may pass
              only ``{"start": .., "end": ..}`` and keep ``min``/``max``. For a
              DATA-DRIVEN window (computed from the pixels) use `set_display_range`.

        ::

            pyr = pyr.set_channels({0: {"color": "FF0000", "label": "DAPI"},
                                    2: {"active": False}})

        If the pyramid has no omero channels yet (e.g. built via `from_arrays`),
        defaults are auto-populated for the length of the 'c' axis first.
        """
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        if not isinstance(overrides, dict):
            raise TypeError("set_channels: overrides must be a dict {channel_key: {field: value}}")

        new = self._clone_metadata_only()
        if new.meta is None or new.meta.metadata is None:
            raise RuntimeError("set_channels: pyramid has no metadata to edit")

        # ensure omero channels exist (from_arrays leaves them empty)
        omero = new.meta.metadata.setdefault("omero", {})
        channels = omero.get("channels", [])
        if not channels:
            axstr = new.meta.axis_order
            base = new.dask_arrays[new.meta.resolution_paths[0]]
            nch = base.shape[axstr.index("c")] if "c" in axstr else 1
            new.meta.autocompute_omerometa(nch, base.dtype)
            omero = new.meta.metadata["omero"]
            channels = omero["channels"]

        # resolve str keys against the current labels
        label_to_idx = {c.get("label"): i for i, c in enumerate(channels) if c.get("label") is not None}
        for key, fields in overrides.items():
            if isinstance(key, str):
                if key not in label_to_idx:
                    raise KeyError(f"set_channels: no channel with label {key!r} "
                                   f"(labels: {sorted(label_to_idx)})")
                idx = label_to_idx[key]
            else:
                idx = int(key)
            if not (0 <= idx < len(channels)):
                raise IndexError(f"set_channels: channel index {idx} out of range (0..{len(channels) - 1})")
            if not isinstance(fields, dict):
                raise TypeError(f"set_channels: value for channel {key!r} must be a dict of fields")
            for field, value in fields.items():
                if field == "color":
                    channels[idx]["color"] = normalize_color(value)   # name/hex/rgb -> 'RRGGBB'
                elif field == "window" and isinstance(value, dict):
                    channels[idx].setdefault("window", {}).update(value)   # merge, keep min/max
                else:
                    channels[idx][field] = value

        new.meta._pending_changes = True
        return new

    def set_display_range(self, method: str = "minmax", *, p_low: float = 1.0, p_high: float = 99.0,
                          stats_level=None, auto_max_bytes: int = 1 << 30) -> 'Pyramid':
        """Recompute each channel's omero display `window` from the data, returning
        a new pyramid (the pixel data is unchanged - only omero metadata).

        Useful after ops that change dtype/intensity range, which make the input's
        window stale. `method`:
        - "minmax"    - window start/end = per-channel data min/max.
        - "percentile"- start/end = the per-channel `p_low`/`p_high` percentiles
          (robust to outliers); min/max stay the true data extent.

        The statistic is computed per channel on `stats_level` (None = finest, an
        int, or "auto" = finest level <= `auto_max_bytes`) - a streaming,
        memory-safe reduction. This is the DATA-DRIVEN window setter; for literal
        per-channel values (colour, label, an explicit window) use `set_channels`.
        """
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        if method not in ("minmax", "percentile"):
            raise ValueError(f"set_display_range: method must be 'minmax' or 'percentile', got {method!r}")

        paths = self.meta.resolution_paths
        if stats_level is None:
            sidx = 0
        elif stats_level == "auto":
            sidx = len(paths) - 1
            for i, p in enumerate(paths):
                a = self.dask_arrays[p]
                if int(np.prod(a.shape)) * a.dtype.itemsize <= auto_max_bytes:
                    sidx = i
                    break
        elif isinstance(stats_level, int):
            sidx = stats_level
        else:
            raise ValueError("set_display_range: stats_level must be None, an int, or 'auto'")
        if not (0 <= sidx < len(paths)):
            raise IndexError(f"set_display_range: stats_level {sidx} out of range")

        arr = self.dask_arrays[paths[sidx]]
        axstr = self.meta.axis_order
        ndim = arr.ndim
        if "c" in axstr:
            ca = axstr.index("c")
            chans = []
            for k in range(arr.shape[ca]):
                sl = [slice(None)] * ndim
                sl[ca] = k
                chans.append(arr[tuple(sl)])
        else:
            chans = [arr]

        mins = [float(v) for v in da.compute(*[c.min() for c in chans])]
        maxs = [float(v) for v in da.compute(*[c.max() for c in chans])]
        if method == "percentile":
            pcs = da.compute(*[da.percentile(c.reshape(-1), [p_low, p_high]) for c in chans])
            lows = [float(p[0]) for p in pcs]
            highs = [float(p[1]) for p in pcs]
        else:
            lows, highs = mins, maxs

        # clone (share arrays, deep-copy metadata), then rewrite the omero windows
        new = self._clone_metadata_only()
        if new.meta is not None and new.meta.metadata is not None:
            channels = new.meta.metadata.get("omero", {}).get("channels", [])  # type: ignore
            for k in range(len(chans)):
                if k < len(channels):
                    channels[k]["window"] = {
                        "min": mins[k], "max": maxs[k], "start": lows[k], "end": highs[k],
                    }
            new.meta._pending_changes = True
        return new

    # ------------------------------------------------------------------
    # Elementwise operators (arithmetic / comparison / boolean)
    #
    # `pyr + 1`, `pyr1 * pyr2`, `pyr > 3`, `pyr + arr`, `(pyr1 * pyr2) > 3`, ...
    # Applied LAZILY to EVERY resolution level (shape-preserving), returning a new
    # pyramid that carries all of this pyramid's metadata (axes/scales/translations/
    # omero/custom). The other operand may be a scalar, a numpy/dask array (it
    # broadcasts against each level), or another Pyramid (matched level-by-level,
    # same axes and per-level shapes). The result is a lazy dask-backed pyramid -
    # write it to realize it. NB: like numpy, `==`/`!=` are ELEMENTWISE (they return
    # a boolean pyramid, not a plain bool); identity hashing is preserved.
    # ------------------------------------------------------------------

    def _clone_with_arrays(self, arrays: List[da.Array]) -> 'Pyramid':
        """A new pyramid with `arrays` (one per level) in place of the current
        layers, carrying over all metadata (shape-preserving)."""
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        paths = self.meta.resolution_paths
        new = Pyramid().from_arrays(
            list(arrays), axis_order=self.meta.axis_order, unit_list=self.meta.unit_list,  # type: ignore
            scales=[self.meta.get_scale(p) for p in paths], version=self.meta.version, name=self.meta.tag,  # type: ignore
        )
        if self.meta.metadata and new.meta and new.meta.metadata:
            new.meta.metadata = copy.deepcopy(self.meta.metadata)
            new.meta._pending_changes = True
        # Elementwise ops are SHAPE-PRESERVING, so a deferred downscale plan (and the
        # recorded storage chunks) stay valid on the transformed base: carry them
        # forward. This lets `pyr.downscale(...).select_levels(...) ** 2 > 0.5` keep
        # deferring - the op runs once on the finest kept level and the writer derives
        # the coarser levels from it (the plan "floats" to the end of the pipeline).
        plan = getattr(self, '_downscale_plan', None)
        if plan is not None:
            new._downscale_plan = dict(plan)
        sc = getattr(self, '_storage_chunks', None)
        if sc is not None:
            new._storage_chunks = sc
        return new

    def _binary_op(self, other, func, reflected: bool = False) -> 'Pyramid':
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        paths = self.meta.resolution_paths
        out: List[da.Array] = []
        if isinstance(other, Pyramid):
            if other.meta is None:
                raise RuntimeError("operand Pyramid not initialized")
            if self.meta.axis_order != other.meta.axis_order:
                raise ValueError(
                    f"pyramids have different axes: '{self.meta.axis_order}' vs '{other.meta.axis_order}'")
            opaths = other.meta.resolution_paths
            if len(opaths) != len(paths):
                raise ValueError(
                    f"pyramids have different numbers of levels ({len(paths)} vs {len(opaths)})")
            for p, q in zip(paths, opaths):
                a, b = self.dask_arrays[p], other.dask_arrays[q]
                if a.shape != b.shape:
                    raise ValueError(f"pyramids differ in shape at level {p}: {a.shape} vs {b.shape}")
                out.append(func(b, a) if reflected else func(a, b))
        else:  # scalar / ndarray / dask array -> broadcast against each level
            for p in paths:
                a = self.dask_arrays[p]
                out.append(func(other, a) if reflected else func(a, other))
        return self._clone_with_arrays(out)

    def _unary_op(self, func) -> 'Pyramid':
        if self.meta is None:
            raise RuntimeError("Pyramid not initialized")
        return self._clone_with_arrays([func(self.dask_arrays[p]) for p in self.meta.resolution_paths])

    # arithmetic
    def __add__(self, o): return self._binary_op(o, operator.add)
    def __radd__(self, o): return self._binary_op(o, operator.add, True)
    def __sub__(self, o): return self._binary_op(o, operator.sub)
    def __rsub__(self, o): return self._binary_op(o, operator.sub, True)
    def __mul__(self, o): return self._binary_op(o, operator.mul)
    def __rmul__(self, o): return self._binary_op(o, operator.mul, True)
    def __truediv__(self, o): return self._binary_op(o, operator.truediv)
    def __rtruediv__(self, o): return self._binary_op(o, operator.truediv, True)
    def __floordiv__(self, o): return self._binary_op(o, operator.floordiv)
    def __rfloordiv__(self, o): return self._binary_op(o, operator.floordiv, True)
    def __mod__(self, o): return self._binary_op(o, operator.mod)
    def __rmod__(self, o): return self._binary_op(o, operator.mod, True)
    def __pow__(self, o): return self._binary_op(o, operator.pow)
    def __rpow__(self, o): return self._binary_op(o, operator.pow, True)

    # comparison (Python routes reflected comparisons by swapping the operator)
    def __lt__(self, o): return self._binary_op(o, operator.lt)
    def __le__(self, o): return self._binary_op(o, operator.le)
    def __gt__(self, o): return self._binary_op(o, operator.gt)
    def __ge__(self, o): return self._binary_op(o, operator.ge)
    def __eq__(self, o): return self._binary_op(o, operator.eq)
    def __ne__(self, o): return self._binary_op(o, operator.ne)
    __hash__ = object.__hash__  # keep identity hashing despite elementwise __eq__

    # boolean / bitwise (on integer or boolean masks)
    def __and__(self, o): return self._binary_op(o, operator.and_)
    def __rand__(self, o): return self._binary_op(o, operator.and_, True)
    def __or__(self, o): return self._binary_op(o, operator.or_)
    def __ror__(self, o): return self._binary_op(o, operator.or_, True)
    def __xor__(self, o): return self._binary_op(o, operator.xor)
    def __rxor__(self, o): return self._binary_op(o, operator.xor, True)

    # unary
    def __neg__(self): return self._unary_op(operator.neg)
    def __pos__(self): return self._unary_op(operator.pos)
    def __abs__(self): return self._unary_op(operator.abs)
    def __invert__(self): return self._unary_op(operator.invert)


