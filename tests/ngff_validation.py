"""The test suite's store check: ozp's own (``ome_zarr_pyramid.validate``, plan 5.23),
which walks a whole store and checks every group against the official NGFF schemas of
every role it has, plus where each group sits. See
:mod:`ome_zarr_pyramid.core.validation` for what it checks and why.

``problems(path)`` returns the problems found, ``[]`` for a valid store. It includes the
strict schemas' findings (``strict=True``): what ozp writes on its own meets them."""
from __future__ import annotations

from typing import List

from ome_zarr_pyramid.core.validation import validate


def problems(path) -> List[str]:
    return validate(path, strict=True)
