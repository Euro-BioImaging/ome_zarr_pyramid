"""ome_zarr_pyramid without dask (an optional extra).

The whole suite also runs with dask blocked (``OZP_BLOCK_DASK=1``, see
conftest.py) and in CI with dask not installed; the tests here cover what
only that mode needs: every module imports, and the TensorStore view that
replaces dask for in-memory downscaling cannot deadlock.
"""
import os
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest


def test_every_module_imports_without_dask():
    code = textwrap.dedent("""
        import importlib, pkgutil, sys

        class NoDask:
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] in ("dask", "distributed"):
                    raise ModuleNotFoundError(name)
                return None

        sys.meta_path.insert(0, NoDask())
        import ome_zarr_pyramid
        failed = []
        for info in pkgutil.walk_packages(ome_zarr_pyramid.__path__, "ome_zarr_pyramid."):
            try:
                importlib.import_module(info.name)
            except Exception as exc:
                failed.append(f"{info.name}: {type(exc).__name__}: {exc}")
        print("\\n".join(failed) or "OK")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300)
    assert out.stdout.strip().splitlines()[-1] == "OK", out.stdout + out.stderr


def test_the_tensorstore_view_of_a_tensorstore_backed_array_does_not_deadlock():
    """A DynamicArray over a TensorStore, transformed, behind `as_tensorstore`'s
    virtual view: reading many of its chunks at once makes every view read
    read TensorStore in turn. Done on TensorStore's own threads, those inner
    reads waited for threads all taken by the outer ones (a deadlock, seen in
    `write_pyramid` of a downscaled, re-planned pyramid)."""
    import tensorstore as ts
    from dyna_zarr import DynamicArray
    from ome_zarr_pyramid.utils.scale import as_tensorstore

    data = np.arange(4 * os.cpu_count() * 2 * 32, dtype="uint16").reshape(-1, 32)
    inner = DynamicArray(ts.array(data)) + 0          # transformed: not unwrapped
    inner._chunks = (1, 32)                           # one view chunk per row
    view = as_tensorstore(inner)
    got = []
    # a daemon thread: if the read deadlocks, the test fails and the run goes on
    reader = threading.Thread(target=lambda: got.append(view.read().result()), daemon=True)
    reader.start()
    reader.join(timeout=60)
    if reader.is_alive():
        pytest.fail("reading the TensorStore view deadlocked")
    np.testing.assert_array_equal(got[0], data)
