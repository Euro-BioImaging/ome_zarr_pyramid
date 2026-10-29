"""A collection as a table: one row per image, its keys and its metadata as columns, for
SQL-style queries (``PyramidCollection.filter``, ``unique``, ``count``, ``group_by``,
``sort_by``).

A row is built from metadata only, never from pixels:

- an image already in memory (from a microscopy file, or opened before) gives its row
  from its ``Pyramid``;
- an image of an NGFF 0.4 / 0.5 collection that is not opened yet gives its row straight
  from its metadata files (the group's, then its levels'), fetched for all such images in
  concurrent batches, without building a ``Pyramid``;
- any other image (an RFC 8 multiscale, say) is opened, which also reads metadata only.

Both ways give the same row for the same image.
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: the axes whose size gets a column of its own (``size_z`` ...)
SIZE_AXES = ("t", "c", "z", "y", "x")
#: the image columns, in display order (keys come first, plate columns last)
IMAGE_COLUMNS = ("name", "axes", "shape") + tuple(f"size_{a}" for a in SIZE_AXES) + (
    "dtype", "nlevels", "level_shapes", "chunks", "shards", "codec", "version", "scale",
    "units", "translation") + tuple(f"{what}_{a}" for what in ("scale", "unit", "translation")
                                    for a in SIZE_AXES) + ("channels", "nbytes", "path")
PLATE_COLUMNS = ("plate_name", "well_name", "row", "row_name", "column", "column_name",
                 "acquisition_name")
#: older column names that still read (decision C-16): ``levels`` is ``nlevels`` (the number
#: of resolution levels; ``pyr.levels`` is the levels themselves). Each row holds both;
#: listings show the new name only.
COLUMN_ALIASES = {"levels": "nlevels"}


# -- rows -----------------------------------------------------------------------------

def _dtype(value) -> Optional[str]:
    try:
        return np.dtype(value).name
    except Exception:                                  # noqa: BLE001
        return str(value) if value is not None else None


def _codec(name: Optional[str], params: Optional[dict] = None) -> Optional[str]:
    """A codec as one word: ``blosc-lz4``, ``zstd``, ``gzip``; None when uncompressed."""
    if not name or name in ("none", "bytes"):
        return None
    name = str(name).lower()
    if name == "blosc":
        return f"blosc-{(params or {}).get('cname', 'lz4')}"
    return name


def _per_axis(prefix: str, axes: str, values) -> Dict[str, Any]:
    """``{prefix_t: ..., ..., prefix_x: ...}`` from per-axis `values` in the order of
    `axes`; None for an axis the image does not have."""
    values = tuple(values or ())
    return {f"{prefix}_{a}": (values[axes.index(a)] if a in axes and axes.index(a) < len(values)
                              else None) for a in SIZE_AXES}


def _finish(row: Dict[str, Any]) -> Dict[str, Any]:
    shape, dtype, axes = row.get("shape"), row.get("dtype"), row.get("axes") or ""
    try:
        row["nbytes"] = int(math.prod(shape) * np.dtype(dtype).itemsize)
    except Exception:                                  # noqa: BLE001
        row["nbytes"] = None
    row.update(_per_axis("size", axes, shape))
    row.update(_per_axis("scale", axes, row.get("scale")))
    row.update(_per_axis("unit", axes, row.get("units")))
    row.update(_per_axis("translation", axes, row.get("translation")))
    return row


def _tuple(values) -> Optional[tuple]:
    return tuple(int(v) for v in values) if values is not None else None


def row_from_pyramid(pyr, path: Optional[str] = None) -> Dict[str, Any]:
    """The row of an image from its ``Pyramid`` (metadata and array headers only)."""
    from dyna_zarr import DynamicArray

    meta = pyr.meta
    paths = meta.resolution_paths
    base = pyr.layers[paths[0]]
    chunks = getattr(base, "chunks", None)
    if chunks and isinstance(chunks[0], tuple):        # dask: per-axis block sizes
        chunks = tuple(c[0] for c in chunks)
    codec = None
    try:
        source = base if isinstance(base, DynamicArray) else DynamicArray(base)
        c = getattr(source, "codecs", None)
        if c is not None:
            codec = _codec(c.compressor, {"cname": getattr(c, "cname", None)})
    except Exception:                                  # noqa: BLE001
        codec = None
    shards = getattr(base, "shards", None)
    if shards is not None and len(shards) != len(base.shape):
        shards = None              # a stale source tuple (dyna-zarr 0.1.0 after an axis drop)
    channels = tuple(ch.get("label") for ch in
                     ((meta.metadata or {}).get("omero") or {}).get("channels") or [])
    row = {
        "name": meta.tag, "axes": pyr.axes, "shape": _tuple(base.shape),
        "dtype": _dtype(base.dtype), "nlevels": len(paths), "levels": len(paths),
        "level_shapes": tuple(_tuple(pyr.layers[p].shape) for p in paths),
        "chunks": _tuple(chunks), "shards": _tuple(shards) if shards else None,
        "codec": codec, "version": str(meta.version),
        "scale": tuple(float(s) for s in meta.get_scale(paths[0])),
        "units": tuple(meta.unit_list), "channels": channels,
        # no translation stored = at the origin, as the JSON path reads it
        "translation": tuple(float(t) for t in (meta.get_translation(paths[0])
                                                or [0.0] * len(pyr.axes))),
        "path": path,
    }
    return _finish(row)


def _fold(ms: Dict[str, Any], dataset: Dict[str, Any], n: int):
    """A dataset's scale and translation with the multiscale-level transforms folded in,
    as ozp's reader does (``s = s_ms * s_ds``, ``t = s_ms * t_ds + t_ms``)."""
    s_ms, t_ms = [1.0] * n, [0.0] * n
    for t in ms.get("coordinateTransformations") or []:
        if t.get("type") == "scale":
            s_ms = [a * float(b) for a, b in zip(s_ms, t["scale"])]
            t_ms = [a * float(b) for a, b in zip(t_ms, t["scale"])]
        elif t.get("type") == "translation":
            t_ms = [a + float(b) for a, b in zip(t_ms, t["translation"])]
    own = dataset.get("coordinateTransformations") or []
    s_ds = next((list(map(float, t["scale"])) for t in own if t.get("type") == "scale"), [1.0] * n)
    t_ds = next((list(map(float, t["translation"])) for t in own
                 if t.get("type") == "translation"), [0.0] * n)
    return ([a * b for a, b in zip(s_ms, s_ds)],
            [a * b + c for a, b, c in zip(s_ms, t_ds, t_ms)])


def _array_header(md: Dict[str, Any]) -> Dict[str, Any]:
    """shape, dtype, chunks, shards and codec from a zarr v2 ``.zarray`` or v3 ``zarr.json``."""
    if md.get("zarr_format") == 3 or "data_type" in md:
        chunk = (md.get("chunk_grid") or {}).get("configuration", {}).get("chunk_shape")
        codecs, shards = list(md.get("codecs") or []), None
        for c in codecs:
            if isinstance(c, dict) and c.get("name") == "sharding_indexed":
                shards = chunk
                chunk = c.get("configuration", {}).get("chunk_shape")
                codecs = list(c.get("configuration", {}).get("codecs") or [])
                break
        compress = next((c for c in codecs if isinstance(c, dict)
                         and c.get("name") not in ("bytes", "transpose", "crc32c")), None)
        codec = _codec(compress.get("name"), compress.get("configuration")) if compress else None
        return {"shape": _tuple(md.get("shape")), "dtype": _dtype(md.get("data_type")),
                "chunks": _tuple(chunk), "shards": _tuple(shards) if shards else None,
                "codec": codec}
    comp = md.get("compressor") or None
    return {"shape": _tuple(md.get("shape")), "dtype": _dtype(md.get("dtype")),
            "chunks": _tuple(md.get("chunks")), "shards": None,
            "codec": _codec(comp.get("id"), comp) if comp else None}


def row_from_json(attrs: Dict[str, Any], headers: Sequence[Dict[str, Any]],
                  path: Optional[str] = None) -> Dict[str, Any]:
    """The row of an NGFF 0.4 / 0.5 / 0.6 image from its group attributes and its level
    arrays' metadata (in dataset order)."""
    from ome_zarr_pyramid.core.ngff06 import ozp_multiscales
    ome = attrs.get("ome") if isinstance(attrs.get("ome"), dict) else attrs
    version, mss = ozp_multiscales(attrs, path or "image")
    ms = (mss or [{}])[0]
    axes_md = ms.get("axes") or []
    axes = "".join((a.get("name") if isinstance(a, dict) else str(a)).lower() for a in axes_md)
    units = tuple((a.get("unit") if isinstance(a, dict) else None) for a in axes_md)
    datasets = ms.get("datasets") or []
    scale, translation = _fold(ms, datasets[0] if datasets else {}, len(axes))
    base = headers[0] if headers else {}
    channels = tuple(ch.get("label") for ch in (ome.get("omero") or {}).get("channels") or [])
    row = {
        "name": ms.get("name"), "axes": axes, "shape": base.get("shape"),
        "dtype": base.get("dtype"), "nlevels": len(datasets), "levels": len(datasets),
        "level_shapes": tuple(h.get("shape") for h in headers),
        "chunks": base.get("chunks"), "shards": base.get("shards"), "codec": base.get("codec"),
        "version": str(version) if version is not None else None,
        "scale": tuple(scale), "units": units, "translation": tuple(translation),
        "channels": channels, "path": path,
    }
    return _finish(row)


# -- reading metadata files in batches ------------------------------------------------

def _read_local(path: str) -> Optional[dict]:
    try:
        with open(path, "rb") as f:
            return json.loads(f.read())
    except (FileNotFoundError, NotADirectoryError):
        return None


def read_json_many(locations: Sequence[str], storage_options: Optional[dict] = None,
                   threads: int = 32) -> List[Optional[dict]]:
    """The JSON documents at `locations` (``None`` where there is none): local files on a
    thread pool, object-store URLs as one concurrent batch of requests."""
    if not locations:
        return []
    remote = [i for i, loc in enumerate(locations) if "://" in loc and not loc.startswith("file://")]
    out: List[Optional[dict]] = [None] * len(locations)
    local = [i for i in range(len(locations)) if i not in set(remote)]
    if local:
        with ThreadPoolExecutor(max_workers=min(threads, len(local))) as pool:
            for i, doc in zip(local, pool.map(lambda i: _read_local(locations[i]), local)):
                out[i] = doc
    if remote:
        import asyncio

        import obstore
        from zarr.core.sync import sync

        stores: Dict[str, Any] = {}

        def split(url: str):
            scheme, rest = url.split("://", 1)
            bucket, _, key = rest.partition("/")
            root = f"{scheme}://{bucket}"
            if root not in stores:
                stores[root] = obstore.store.from_url(root, **(storage_options or {}))
            return stores[root], key

        async def get(url: str):
            store, key = split(url)
            try:
                result = await obstore.get_async(store, key)
                return json.loads(bytes(await result.bytes_async()))
            except FileNotFoundError:
                return None
            except Exception as exc:                   # noqa: BLE001
                if "not found" in str(exc).lower() or "404" in str(exc):
                    return None
                raise

        async def run():
            return await asyncio.gather(*(get(locations[i]) for i in remote))

        for i, doc in zip(remote, sync(run())):
            out[i] = doc
    return out


def _join(base: str, *parts: str) -> str:
    if "://" in base and not base.startswith("file://"):
        return base.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts)
    return os.path.join(base, *parts)


def rows_from_files(locations: Sequence[str], storage_options: Optional[dict] = None
                    ) -> List[Optional[Dict[str, Any]]]:
    """Rows for the NGFF 0.4 / 0.5 images at `locations`, from their metadata files: the
    groups' first (one batch), then their level arrays' (a second batch). ``None`` for a
    location that is not such an image (the caller opens it instead)."""
    v3 = read_json_many([_join(loc, "zarr.json") for loc in locations], storage_options)
    missing = [i for i, doc in enumerate(v3) if doc is None]
    v2 = dict(zip(missing, read_json_many([_join(locations[i], ".zattrs") for i in missing],
                                          storage_options)))
    groups: List[Optional[Tuple[dict, int]]] = []
    for i in range(len(locations)):
        if v3[i] is not None:
            groups.append((v3[i].get("attributes") or {}, 3))
        elif v2.get(i) is not None:
            groups.append((v2[i], 2))
        else:
            groups.append(None)
    wanted: List[Tuple[int, str]] = []
    for i, g in enumerate(groups):
        if g is None:
            continue
        attrs, fmt = g
        ome = attrs.get("ome") if isinstance(attrs.get("ome"), dict) else attrs
        ms = (ome.get("multiscales") or [None])[0]
        if not isinstance(ms, dict):
            groups[i] = None
            continue
        for d in ms.get("datasets") or []:
            name = "zarr.json" if fmt == 3 else ".zarray"
            wanted.append((i, _join(locations[i], str(d.get("path")), name)))
    docs = read_json_many([loc for _, loc in wanted], storage_options)
    headers: Dict[int, List[dict]] = {}
    for (i, _), doc in zip(wanted, docs):
        headers.setdefault(i, []).append(_array_header(doc) if doc else {})
    return [row_from_json(g[0], headers.get(i, []), path=locations[i]) if g else None
            for i, g in enumerate(groups)]


# -- the table --------------------------------------------------------------------------

class CollectionTable:
    """The rows of a collection, one per image in the collection's order: its key columns,
    the image columns (``IMAGE_COLUMNS``) and, for plates, ``PLATE_COLUMNS``. Read a column
    with ``table["dtype"]``, iterate rows as dicts, or ``to_pandas()``."""

    def __init__(self, keys: Sequence[tuple], rows: Sequence[Dict[str, Any]]):
        self.keys = list(keys)
        self.rows = [dict(r) for r in rows]
        names: Dict[str, None] = {}
        for key in self.keys:
            for name, _ in key:
                names.setdefault(name, None)
        self.key_columns = tuple(names)
        extra = [c for c in PLATE_COLUMNS if any(c in r for r in self.rows)]
        self.columns = self.key_columns + IMAGE_COLUMNS + tuple(extra)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.rows)

    def __getitem__(self, column: str) -> list:
        column = COLUMN_ALIASES.get(column, column)
        if column not in self.columns:
            raise KeyError(f"no column {column!r}; the columns are {list(self.columns)}")
        return [r.get(column) for r in self.rows]

    def unique(self, column: str) -> list:
        """The distinct values of a column, sorted (None last): ``table.unique("dtype")``."""
        return sorted_values(set(self[column]))

    def count(self, column: str) -> Dict[Any, int]:
        """The number of images per value of a column: ``table.count("codec")`` gives
        ``{'blosc-lz4': 2}``."""
        values = self[column]
        return {v: values.count(v) for v in sorted_values(set(values))}

    def to_pandas(self):
        """The table as a pandas DataFrame (pandas is not a dependency of ozp)."""
        import pandas as pd
        return pd.DataFrame([{c: r.get(c) for c in self.columns} for r in self.rows],
                            columns=list(self.columns))

    def __repr__(self) -> str:
        shown = list(self.key_columns) + ["shape", "dtype", "nlevels", "codec"]
        rows = [[str(r.get(c)) for c in shown] for r in self.rows[:12]]
        widths = [max([len(c)] + [len(x[i]) for x in rows]) for i, c in enumerate(shown)]
        line = lambda cells: "  ".join(s.ljust(w) for s, w in zip(cells, widths))  # noqa: E731
        out = [f"CollectionTable: {len(self)} images x {len(self.columns)} columns",
               line(shown)] + [line(x) for x in rows]
        if len(self) > 12:
            out.append(f"... {len(self) - 12} more")
        return "\n".join(out)


def order_key(value):
    """Sorts values of one column: None last, then by type, then by value."""
    return (value is None, type(value).__name__, value if value is not None else 0)


def sorted_values(values) -> list:
    """A column's values in order (None last); values of one type that do not order by
    their repr."""
    try:
        return sorted(values, key=order_key)
    except TypeError:
        return sorted(values, key=lambda v: (v is None, repr(v)))


def matches(row: Dict[str, Any], predicate: Optional[Callable], conditions: Dict[str, Any]) -> bool:
    """Whether `row` passes `predicate(row)` and every condition: a callable is called with
    the column's value, anything else is compared for equality."""
    for name, want in conditions.items():
        value = row.get(name)
        if callable(want):
            if not want(value):
                return False
        elif value != want:
            return False
    return predicate is None or bool(predicate(row))
