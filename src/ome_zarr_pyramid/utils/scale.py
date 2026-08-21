"""Downscaling utilities for creating image pyramids with TensorStore and Dask support."""

import asyncio
import dataclasses
import itertools
import os
from fractions import Fraction
from math import gcd, lcm
from typing import Union, Optional, Dict, Any, Tuple, Callable, Sequence, List

import dask.array as da
import numpy as np
import tensorstore as ts
import zarr

from ome_zarr_pyramid.utils.logging_config import get_logger
from ome_zarr_pyramid.utils.storage_utils import make_kvstore

logger = get_logger(__name__)

SPATIAL_AXES = {'z', 'y', 'x'}
_SMART_DEFAULT_MAX_FACTOR = 16
_SMART_DEFAULT_TOLERANCE = 0.10


def autocompute_chunk_shape(
    array_shape: Tuple[int, ...],
    axes: str,
    target_chunk_mb: float = 1.0,
    dtype=np.uint16,
) -> Tuple[int, ...]:
    """ISOTROPIC chunk shape for a `target_chunk_mb` uncompressed budget.

    Spatial axes (x/y/z) get an equal side length (a SQUARE in 2-D, a CUBE in 3-D),
    grown to fill the byte budget without exceeding it; non-spatial axes (t/c) stay
    at 1. The dtype's itemsize is accounted for. Each side is capped at the array's
    extent along that axis. (Ported from EuBI-Bridge's `autocompute_chunk_shape`.)
    """
    if len(array_shape) != len(axes):
        raise ValueError("array_shape length must match axes length")
    # coerce to a numpy dtype (a tensorstore dtype exposes `.numpy_dtype`)
    np_dt = getattr(dtype, 'numpy_dtype', dtype)
    itemsize = np.dtype(np_dt).itemsize
    max_elements = int(target_chunk_mb * 1024 * 1024) // int(itemsize)
    max_elements = max(1, max_elements)

    chunk = [1] * len(array_shape)
    spatial = [i for i, ax in enumerate(axes) if ax in SPATIAL_AXES]
    if not spatial:
        return tuple(chunk)

    side = max(1, int(np.floor(max_elements ** (1.0 / len(spatial)))))
    for i in spatial:
        chunk[i] = min(side, array_shape[i])
    # grow spatial sides isotropically while within budget
    while True:
        trial = list(chunk)
        for i in spatial:
            if trial[i] < array_shape[i]:
                trial[i] += 1
        if trial != chunk and int(np.prod([trial[i] for i in spatial])) <= max_elements:
            chunk = trial
        else:
            break
    # final safety trim (z first)
    while int(np.prod([chunk[i] for i in spatial])) > max_elements:
        for i in reversed(spatial):
            if chunk[i] > 1:
                chunk[i] -= 1
                break
    return tuple(chunk)


def compute_isotropic_scale_factors(
    pixel_sizes: dict,
    axes: str,
    max_factor: int = _SMART_DEFAULT_MAX_FACTOR,
    tolerance: float = _SMART_DEFAULT_TOLERANCE,
) -> dict:
    """Compute integer scale factors that bring spatial axes to near-isotropy.

    Parameters
    ----------
    pixel_sizes : dict
        Physical pixel size per axis, e.g. {'t': 1.0, 'z': 0.5, 'y': 0.1, 'x': 0.1}.
    axes : str
        Axis order string, e.g. 'tczyx'.
    max_factor : int
        Maximum allowed scale factor per spatial axis (default 16).
    tolerance : float
        Acceptable residual anisotropy: max/min - 1 (default 0.10 = 10%).

    Returns
    -------
    dict
        Integer scale factor per axis, e.g. {'t': 1, 'c': 1, 'z': 1, 'y': 5, 'x': 5}.
        Non-spatial axes always get factor 1.
    """
    result = {ax: 1 for ax in axes}

    spatial = [ax for ax in axes if ax in SPATIAL_AXES and ax in pixel_sizes and pixel_sizes[ax] > 0]
    if len(spatial) < 2:
        return result  # 0 or 1 spatial axis: already trivially isotropic

    ps = [float(pixel_sizes[ax]) for ax in spatial]

    # Fast path: already isotropic within tolerance
    if max(ps) / min(ps) - 1 <= tolerance / 10:
        return result

    # ── Step 1: exact LCM approach via rational arithmetic ─────────────────
    fracs = [Fraction(p).limit_denominator(1000) for p in ps]
    num_lcm = fracs[0].numerator
    den_gcd = fracs[0].denominator
    for f in fracs[1:]:
        num_lcm = lcm(num_lcm, f.numerator)
        den_gcd = gcd(den_gcd, f.denominator)
    target = num_lcm / den_gcd
    exact_factors = [max(1, round(target / float(f))) for f in fracs]

    if all(1 <= s <= max_factor for s in exact_factors):
        result_sizes = [ps[i] * exact_factors[i] for i in range(len(ps))]
        if max(result_sizes) / min(result_sizes) - 1 <= tolerance:
            for ax, s in zip(spatial, exact_factors):
                result[ax] = int(s)
            logger.info(
                f"[smart_downscale] LCM approach: factors={dict(zip(spatial, exact_factors))}, "
                f"result_sizes={dict(zip(spatial, result_sizes))}"
            )
            return result

    # ── Step 2: brute-force search over [1..max_factor]^N ─────────────────
    best_factors = [1] * len(spatial)
    best_aniso = float('inf')

    for combo in itertools.product(range(1, max_factor + 1), repeat=len(spatial)):
        result_sizes = [ps[i] * combo[i] for i in range(len(ps))]
        aniso = max(result_sizes) / min(result_sizes) - 1
        if aniso < best_aniso:
            best_aniso = aniso
            best_factors = list(combo)
            if aniso <= tolerance:
                break  # good enough, stop early

    for ax, s in zip(spatial, best_factors):
        result[ax] = int(s)

    result_sizes = [ps[i] * best_factors[i] for i in range(len(ps))]
    logger.info(
        f"[smart_downscale] brute-force: factors={dict(zip(spatial, best_factors))}, "
        f"result_sizes={dict(zip(spatial, result_sizes))}, anisotropy={best_aniso:.3f}"
    )
    return result


def simple_downscale(
    darr: da.Array,
    scale_factor: Optional[Union[tuple, list, np.ndarray]] = None,
    backend: str = 'numpy'
) -> da.Array:
    """Downscale a Dask array using simple stride slicing.
    
    Parameters
    ----------
    darr : dask.array.Array
        Input Dask array to downscale.
    scale_factor : Union[tuple, list, np.ndarray]
        Downsampling factors for each dimension.
    backend : str, optional
        Backend to use (placeholder for future use). Default is 'numpy'.
        
    Returns
    -------
    dask.array.Array
        Downscaled Dask array.
        
    Raises
    ------
    ValueError
        If scale_factor length doesn't match array dimensions.
    """
    if scale_factor is None:
        raise ValueError("scale_factor cannot be None")
    if len(scale_factor) != darr.ndim:  # type: ignore
        raise ValueError("scale_factors must have the same length as the array's number of dimensions")
    slices = tuple(slice(None, None, int(scale)) for scale in scale_factor)  # type: ignore
    downscaled_arr = darr[slices]
    return downscaled_arr


def mean_downscale(
    arr: da.Array,
    scale_factor: Optional[Union[tuple, list, np.ndarray]] = None
) -> da.Array:
    """Downscale a Dask array using mean coarsening.
    
    Parameters
    ----------
    arr : dask.array.Array
        Input Dask array to downscale.
    scale_factor : Union[tuple, list, np.ndarray]
        Downsampling factors for each dimension.
        
    Returns
    -------
    dask.array.Array
        Downscaled Dask array with mean aggregation.
        
    Raises
    ------
    ValueError
        If scale_factor length doesn't match array dimensions.
    """
    if scale_factor is None:
        raise ValueError("scale_factor cannot be None")
    if len(scale_factor) != arr.ndim:  # type: ignore
        raise ValueError("scale_factors must have the same length as the array's number of dimensions")
    axes = dict({idx: factor for idx, factor in enumerate(scale_factor)})  # type: ignore
    downscaled_arr = da.coarsen(da.mean, arr,
                                axes=axes, trim_excess=True).astype(arr.dtype)
    return downscaled_arr


def median_downscale(
    arr: da.Array,
    scale_factor: Optional[Union[tuple, list, np.ndarray]] = None
) -> da.Array:
    """Downscale a Dask array using median coarsening.
    
    Parameters
    ----------
    arr : dask.array.Array
        Input Dask array to downscale.
    scale_factor : Union[tuple, list, np.ndarray]
        Downsampling factors for each dimension.
        
    Returns
    -------
    dask.array.Array
        Downscaled Dask array with median aggregation.
        
    Raises
    ------
    ValueError
        If scale_factor length doesn't match array dimensions.
    """
    if scale_factor is None:
        raise ValueError("scale_factor cannot be None")
    if len(scale_factor) != arr.ndim:  # type: ignore
        raise ValueError("scale_factors must have the same length as the array's number of dimensions")
    axes = dict({idx: factor for idx, factor in enumerate(scale_factor)})  # type: ignore
    downscaled_arr = da.coarsen(da.median, arr,
                                axes=axes, trim_excess=True).astype(arr.dtype)
    return downscaled_arr


async def ts_downscale(
    arr: ts.TensorStore,
    scale_factor: Optional[Union[tuple, list, np.ndarray]] = None,
    downscale_method: str = 'stride'
) -> ts.TensorStore:
    """Downscale using TensorStore's virtual downsampling.

    Parameters
    ----------
    arr : TensorStore
        Input TensorStore array to downscale
    scale_factor : tuple or list or ndarray
        Downsampling factors per axis
    downscale_method : str, optional
        TensorStore downsampling method (e.g. 'stride', 'mean', 'median').
        'simple' is accepted as an alias for 'stride'. Default is 'stride'.

    Returns
    -------
    TensorStore
        Downsampled TensorStore array

    Raises
    ------
    ValueError
        If scale_factor is None.
    """
    if scale_factor is None:
        raise ValueError("scale_factor cannot be None")
    # ts.downsample expects factors as list of ints
    factors = [int(np.round(factor)) for factor in scale_factor]
    ts_method = 'stride' if downscale_method == 'simple' else downscale_method
    return ts.downsample(arr, factors, method=ts_method)


@dataclasses.dataclass
class DownscaleManager:
    """Manager for calculating downscale parameters across pyramid layers."""
    
    base_shape: Union[list, tuple]
    scale_factor: Union[list, tuple]
    n_layers: int
    scale: Optional[Union[list, tuple]] = None
    smart_scale_factor: Optional[Union[list, tuple]] = None  # isotropic first-level factors

    def __post_init__(self):
        """Validate inputs."""
        ndim = len(self.base_shape)
        if len(self.scale_factor) != ndim:
            raise ValueError(f"scale_factor length ({len(self.scale_factor)}) must match base_shape ({ndim})")

    @property
    def _scale_ids(self) -> np.ndarray:
        """Get layer indices as column vector."""
        return np.arange(self.n_layers).reshape(-1, 1)

    @property
    def _theoretical_scale_factors(self) -> np.ndarray:
        """Get theoretical scale factors for each layer."""
        if self.smart_scale_factor is None:
            return np.power(self.scale_factor, self._scale_ids)
        smart = np.array(self.smart_scale_factor, dtype=float)
        user = np.array(self.scale_factor, dtype=float)
        # If smart factors are all 1 (data already isotropic, e.g. 2D),
        # no anisotropy correction is needed -> fall back to regular downscaling.
        if np.all(smart == 1.0):
            return np.power(self.scale_factor, self._scale_ids)
        # Two-phase: level 1 = smart_factor; levels 2+ = smart_factor * user_factor^(k-1)
        n = self.n_layers
        ndim = len(self.base_shape)
        result = np.zeros((n, ndim))
        result[0] = 1.0
        for k in range(1, n):
            result[k] = smart * np.power(user, k - 1)
        return result

    @property
    def output_shapes(self) -> np.ndarray:
        """Calculate output shapes for each layer.
        
        Returns
        -------
        np.ndarray
            Shape for each downscale layer
        """
        shapes = np.ceil(np.divide(self.base_shape, self._theoretical_scale_factors))
        shapes[shapes == 0] = 1
        return shapes.astype(int)

    @property
    def scale_factors(self) -> np.ndarray:
        """Get scale factors for each layer.

        Returns
        -------
        np.ndarray
            Theoretical (stride-based) scale factors per layer
        """
        return self._theoretical_scale_factors

    @property
    def scales(self) -> np.ndarray:
        """Get physical scales for each layer.

        Returns
        -------
        np.ndarray
            Physical scale values per layer
        """
        if self.scale is None:
            self.scale = [1.0] * len(self.base_shape)
        return np.multiply(self.scale, self._theoretical_scale_factors)


@dataclasses.dataclass
class Downscaler:
    """Advanced downscaler supporting both Dask arrays and Zarr/TensorStore arrays.
    
    Automatically detects input type and uses:
    - TensorStore for zarr.Array inputs (high-performance)
    - Dask for da.Array inputs (flexible)
    
    After initialization, array will be either da.Array or ts.TensorStore.
    Use base_array_root to determine type: None for Dask, str path for TensorStore.
    """
    
    array: Union[da.Array, zarr.Array, ts.TensorStore, str]
    scale_factor: Union[list, tuple, np.ndarray]
    n_layers: int
    scale: Optional[Union[list, tuple]] = None
    output_chunks: Optional[Union[list, tuple]] = None
    backend: str = 'numpy'
    downscale_method: str = 'simple'
    smart_scale_factor: Optional[Union[list, tuple]] = None  # isotropic first-level factors
    # Assigned in __post_init__
    base_array_root: Optional[str] = dataclasses.field(default=None, init=False, repr=False)
    param_names: list = dataclasses.field(default_factory=list, init=False, repr=False)
    dm: 'DownscaleManager' = dataclasses.field(default=None, init=False, repr=False)  # type: ignore
    # Assigned in run()
    method: Callable = dataclasses.field(default=None, init=False, repr=False)  # type: ignore
    _downscaled_arrays: list = dataclasses.field(default_factory=list, init=False, repr=False)

    def get_tensorstore_context(self) -> Optional[Dict[str, Any]]:
        """Retrieve tensorstore context from worker initialization, if available.
        
        Returns
        -------
        dict or None
            Tensorstore context dict with data_copy_concurrency limits, or None
        """
        # Note: conversion module not yet implemented, returning None
        return None

    def __post_init__(self) -> None:
        """Initialize downscaler and detect array type."""
        array_obj: Union[da.Array, ts.TensorStore] = self.array  # type: ignore
        
        if isinstance(self.array, str):
            # File path - use TensorStore
            store_path = self.array
            kvstore = make_kvstore(store_path)
            self.base_array_root = os.path.abspath(self.array)
            ts_context = self.get_tensorstore_context()
            ts_spec = {
                "driver": "zarr",
                "kvstore": kvstore
            }
            open_kwargs: Dict[str, Any] = {"open": True}
            if ts_context is not None:
                open_kwargs["context"] = ts_context
            array_obj = ts.open(ts_spec, **open_kwargs).result()
            logger.info(f"[Downscaler] Loaded zarr from path: {store_path}")
            
        elif isinstance(self.array, zarr.Array):
            # Zarr array - convert to TensorStore for optimal performance
            try:
                store_root = getattr(self.array.store, 'root', None)
                if store_root is None:
                    store_root = getattr(self.array.store, 'path', None)
                self.base_array_root = os.path.abspath(str(store_root)) if store_root else None
            except (AttributeError, TypeError):
                self.base_array_root = None
            
            arraypath = self.array.path
            logger.info(f"[Downscaler] base_array_root={self.base_array_root}")
            logger.info(f"[Downscaler] arraypath={arraypath}")
            logger.info(f"[Downscaler] store type={type(self.array.store)}")
            
            # Create kvstore pointing to the zarr store ROOT
            if self.base_array_root is not None:
                kvstore = make_kvstore(self.base_array_root)
                logger.info(f"[Downscaler] kvstore={kvstore}")

                # Use appropriate driver based on zarr format
                zarr_format = self.array.metadata.zarr_format
                driver_name = "zarr3" if zarr_format == 3 else "zarr2"
                logger.info(f"[Downscaler] Opening with driver={driver_name}, path={arraypath}")
                
                ts_context = self.get_tensorstore_context()
                ts_spec = {
                    "driver": driver_name,
                    "kvstore": kvstore,
                    "path": arraypath
                }
                open_kwargs = {"open": True}
                if ts_context is not None:
                    open_kwargs["context"] = ts_context
                array_obj = ts.open(ts_spec, **open_kwargs).result()
                logger.info(f"[Downscaler] Successfully opened zarr array with TensorStore")
            else:
                # Fall back to Dask for zarr arrays without file path info
                self.base_array_root = None
                if not isinstance(self.array, da.Array):
                    array_obj = da.from_array(self.array, chunks=self.output_chunks or self.array.chunks)  # type: ignore
                else:
                    array_obj = self.array
            
        else:
            # Dask array or numpy array - optimize for in-memory processing
            self.base_array_root = None
            if isinstance(self.array, da.Array):
                array_obj = self.array
            elif isinstance(self.array, np.ndarray):
                # A real numpy array is NOT a Delayed: `from_delayed` looked for `.key`
                # and raised `AttributeError: 'numpy.ndarray' object has no attribute
                # 'key'`, so downscaling an in-memory Pyramid failed outright.
                array_obj = da.from_array(self.array, chunks=self.output_chunks or 'auto')
            else:
                # A dask Delayed (or anything else exposing shape/dtype).
                array_obj = da.from_delayed(self.array, shape=self.array.shape, dtype=self.array.dtype)  # type: ignore

        self.param_names = ['array', 'scale_factor', 'n_layers', 'scale', 'output_chunks', 'backend', 'downscale_method', 'smart_scale_factor']
        # Convert scale_factor to tuple if it's ndarray
        if isinstance(self.scale_factor, np.ndarray):
            self.scale_factor = tuple(self.scale_factor)

        # Update array after processing
        self.array = array_obj

        self.dm = DownscaleManager(
            base_shape=self.array.shape,
            scale_factor=self.scale_factor,
            n_layers=self.n_layers,
            scale=self.scale,
            smart_scale_factor=self.smart_scale_factor,
        )

    def get_method(self) -> Callable:
        """Get the appropriate downscaling method.
        
        Returns
        -------
        callable
            Downscaling function (either sync or async)
        
        Raises
        ------
        NotImplementedError
            If an unsupported downscaling method is requested.
        """
        if self.base_array_root is None:  # array is dask array
            if self.downscale_method == 'simple':
                method: Callable = simple_downscale
            elif self.downscale_method == "mean":
                method = mean_downscale
            elif self.downscale_method == "median":
                method = median_downscale
            else:
                raise NotImplementedError(f"Currently, only 'simple', 'mean' and 'median' methods are implemented.")
        else:
            # TensorStore case
            method = ts_downscale
        return method

    async def run(self) -> 'Downscaler':
        """Execute downscaling asynchronously.
        
        Returns
        -------
        Downscaler
            Returns self for method chaining
        """
        self.method = self.get_method()
        
        downscaled = {}
        for idx, scale_factor in enumerate(self.dm.scale_factors):
            if idx == 0:
                # Base layer - keep as is
                pass
            else:
                factor = tuple(int(np.round(x)) for x in scale_factor)
                if self.method is ts_downscale:
                    coro_kwargs: Dict[str, Any] = dict(scale_factor=factor, downscale_method=self.downscale_method)
                else:
                    coro_kwargs = dict(scale_factor=factor)
                if asyncio.iscoroutinefunction(self.method):
                    res = asyncio.create_task(
                        self.method(self.array, **coro_kwargs),
                        name=f"downscale_{idx}"
                    )
                else:
                    # Non-async method - wrap in coroutine
                    async def _downscale(_kwargs=coro_kwargs):
                        return self.method(self.array, **_kwargs)
                    res = asyncio.create_task(_downscale(), name=f"downscale_{idx}")
                downscaled[idx] = res
        
        # Gather results
        if downscaled:
            results = await asyncio.gather(*downscaled.values(), return_exceptions=False)
        else:
            results = []
        
        self._downscaled_arrays = [self.array]
        for idx in range(len(results)):
            self._downscaled_arrays.append(results[idx])
        
        return self

    async def update(self, **kwargs) -> 'Downscaler':
        """Update parameters and run downscaling.
        
        Parameters
        ----------
        **kwargs : dict
            Parameters to update before downscaling
            
        Returns
        -------
        Downscaler
            Returns self for method chaining
        """
        for key, value in kwargs.items():
            if key in self.param_names:
                self.__setattr__(key, value)
            else:
                logger.warning(f"The given parameter name '{key}' is not valid, ignoring it..")
        await self.run()
        return self


# --- deriving a downscale plan from an EXISTING pyramid's levels ---------------------
#
# `scale_factor=None` used to mean "assume 2 on z/y/x". That silently rewrote any source
# whose levels were not built that way. These helpers instead SOLVE for the factors the
# source actually used, and refuse rather than guess.
#
# The hard constraint: no backend accepts a target shape. `ts.downsample`, stride slicing
# and `da.coarsen` all take integer FACTORS, and the level shape is whatever that
# produces. So a level is reproducible only if some integer factor maps base -> it under
# the rounding rule of the chosen method. `(10,10) -> (7,7)` is not reachable by any
# factor and must fail loudly.
#
# Rounding differs by method and decides reachability on odd axes:
#     base=101 f=2  ->  stride/ts: 51 (ceil)   coarsen(mean/median): 50 (floor)

def _level_size(base: int, factor: int, downscale_method: str = 'simple') -> int:
    """Size `downscale_method` yields for one axis. Mirrors the executors exactly."""
    if factor <= 1:
        return base
    if downscale_method in ('mean', 'median'):
        return max(1, base // factor)          # da.coarsen(trim_excess=True) -> floor
    return max(1, -(-base // factor))          # stride / ts.downsample       -> ceil


def _solve_axis_factor(base: int, target: int, downscale_method: str = 'simple') -> Optional[int]:
    """Smallest integer factor with `_level_size(base, f) == target`, else None.

    Searched rather than derived: `ceil(base/f)` is not invertible from a rounded ratio
    (base=101 target=50 has NO ceil solution, while target=51 has one), so the rounded
    ratio is only a starting guess and every candidate must be verified.
    """
    if target <= 0 or target > base:
        return None
    if target == base:
        return 1
    guess = max(1, int(round(base / target)))
    for f in range(max(1, guess - 2), guess + 3):     # solutions are contiguous & tight
        if _level_size(base, f, downscale_method) == target:
            return f
    return None


def derive_scale_factors(shapes: Sequence[Sequence[int]],
                         downscale_method: str = 'simple'
                         ) -> Optional[List[Tuple[int, ...]]]:
    """Per-level integer factors (relative to base) reproducing `shapes` EXACTLY.

    Returns one tuple per level including level 0 (all ones), or None when any level is
    unreachable - an irregular progression is fine (levels need not share a factor), but
    a level no integer factor can produce is not.

    Verified, not merely solved: the derived factors are applied back and the resulting
    shapes compared to `shapes`, so a caller that trusts the result cannot be handed a
    pyramid that is a voxel off.
    """
    if not shapes:
        return None
    base = tuple(int(s) for s in shapes[0])
    factors: List[Tuple[int, ...]] = [tuple(1 for _ in base)]
    for level in shapes[1:]:
        level = tuple(int(s) for s in level)
        if len(level) != len(base):
            return None
        solved = [_solve_axis_factor(b, t, downscale_method) for b, t in zip(base, level)]
        if any(f is None for f in solved):
            return None
        factor = tuple(int(f) for f in solved)  # type: ignore[arg-type]
        if tuple(_level_size(b, f, downscale_method)
                 for b, f in zip(base, factor)) != level:
            return None                          # verification: must round-trip
        factors.append(factor)
    return factors


def derive_downscale_plan(shapes: Sequence[Sequence[int]],
                          downscale_method: str = 'simple') -> Optional[dict]:
    """Plan fields describing `shapes`, or None if they cannot be reproduced.

    `None` is the signal to WARN and fall back to building a fresh plan; it never means
    "close enough".
    """
    if len(shapes) < 2:
        return None
    factors = derive_scale_factors(shapes, downscale_method)
    if factors is None:
        return None
    return {
        'n_layers': len(shapes),
        'scale_factor': factors[1] if len(factors) > 1 else None,
        'level_scale_factors': factors,
        'level_shapes': [tuple(int(s) for s in sh) for sh in shapes],
        'downscale_method': downscale_method,
    }
