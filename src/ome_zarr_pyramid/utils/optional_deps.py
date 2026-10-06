"""dask, imported only where it is used.

ome_zarr_pyramid works without dask: its arrays are dyna_zarr DynamicArrays
(zarr layers wrapped lazily) or TensorStore.  dask is an optional extra
(``pip install 'ome_zarr_pyramid[dask]'``) for dask-backed pyramids and the
dask views of a pyramid (``Pyramid.dask_arrays``).  So no module imports it at
load time (tests/test_no_dask.py): code that needs it imports it where it
runs, through ``require``; code that only asks whether an object is a dask
array asks without importing it (``is_dask_array``); code that does something
else without it asks first (``is_installed``).

The same helpers as eubi-bridge's ``eubi_bridge.utils.optional_deps``.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType


def is_installed(module: str) -> bool:
    """Whether *module* can be imported (without importing it)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def is_dask_array(obj) -> bool:
    """Whether *obj* is a dask array, without importing dask: if dask.array
    was never imported, no dask array can exist."""
    da = sys.modules.get("dask.array")
    return da is not None and isinstance(obj, da.Array)


def require(module: str, feature: str) -> ModuleType:
    """Import *module* for *feature*; if it is not installed, say how to get it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        package = module.partition(".")[0]
        raise ImportError(
            f"{feature} needs {package}, which is optional: "
            f"pip install 'ome_zarr_pyramid[{package}]'") from exc
