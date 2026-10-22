"""The spatial model (plan "NGFF 0.6 and RFC 8: one spatial model", section 3): coordinate
systems, the transformation graph's edges, and the transform model.

- **Value classes** (3.16): :class:`Axis`, :class:`CoordinateSystem`,
  :class:`Transformation`, and :class:`ExternalRef` for an edge end outside the opened
  set (3.29). micro-reader defines classes with the same fields; ozp never imports
  micro-reader and reads its objects by their fields (:func:`system_from`,
  :func:`transformation_from`), as it reads ``image.index``.
- **Ids** (3.23, 8.14): :func:`system_id` makes the ids micro-reader makes (``stage``,
  ``well-B2``, ``global-registered``), by the same rule.
- **Transform model** (3.9, 8.20): identity, translation, scale, affine and sequence are
  modelled: :func:`matrix` gives a transformation's homogeneous matrix, :func:`compose`
  and :func:`invert` work on them, :func:`scale_translation` reads one as a scale and a
  translation (the position-edge rule, 3.20). Every other type is carried as read, in
  ``params``; asking to model one raises :class:`TransformError`.
- **JSON** (5.28): :func:`system_to_json` / :func:`system_from_json` and
  :func:`transformation_to_json` / :func:`transformation_from_json`, the 0.6 objects the
  ``eubi:`` keys and 0.6 scenes hold.

An edge's ends are a shared system's id (a string), an image key (a tuple of (index
name, value) pairs: the image's own system, plan 1.14), an :class:`ExternalRef`, or None
inside a sequence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

#: the NGFF type of an axis, inferred from its name when none is given
AXIS_TYPES = {"t": "time", "c": "channel", "x": "space", "y": "space", "z": "space"}
#: the transformation types ozp models (8.20); any other is carried as read
MODELLED_TYPES = ("identity", "translation", "scale", "affine", "sequence")
#: pattern-safe (3.23): what RFC 8 allows in an id
PATTERN_SAFE = re.compile(r"[a-zA-Z0-9_.-]+")
#: an id ozp or micro-reader makes: pattern-safe, without "." (kept for an image's own
#: system when inlined) and without "--" (kept for the prefixes of merged sets, 8.16)
MADE_ID = re.compile(r"(?!.*--)[a-zA-Z0-9_-]+")


class InvariantError(ValueError):
    """The spatial model's invariants (plan 3.22) do not hold: the message names the
    invariant, what breaks it, and where."""


class TransformError(ValueError):
    """A transformation cannot be modelled as asked: a type ozp carries but does not model
    (8.20), parameters stored in a Zarr array, a matrix of the wrong size, or a singular
    one to invert."""


# -- value classes (3.16, 3.29) ---------------------------------------------------

@dataclass(frozen=True)
class Axis:
    """One axis of a coordinate system. *type*: "space", "time", "channel" or another 0.6
    type; None infers it from the name for ``t``, ``c``, ``x``, ``y``, ``z`` (any other
    name raises). *unit*: a UDUNITS-2 name or None. *discrete*, *long_name*: as 0.6
    allows. *extra*: an axis's other keys, as read."""
    name: str
    type: Optional[str] = None
    unit: Optional[str] = None
    discrete: Optional[bool] = None
    long_name: Optional[str] = None
    extra: dict = field(default_factory=dict, hash=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Axis: the name must be a non-empty string, not {self.name!r}")
        if self.type is None:
            inferred = AXIS_TYPES.get(self.name)
            if inferred is None:
                raise ValueError(f"Axis {self.name!r}: no type can be inferred from the name "
                                 f"(only {', '.join(AXIS_TYPES)} have one): give type=, e.g. "
                                 f"Axis({self.name!r}, 'channel')")
            object.__setattr__(self, "type", inferred)
        elif not isinstance(self.type, str) or not self.type:
            raise ValueError(f"Axis {self.name!r}: type must be a non-empty string or None "
                             f"(inferred), not {self.type!r}")
        if self.unit is not None and not isinstance(self.unit, str):
            raise ValueError(f"Axis {self.name!r}: unit must be a UDUNITS-2 name or None, "
                             f"not {self.unit!r}")
        object.__setattr__(self, "extra", dict(self.extra))


@dataclass(frozen=True)
class CoordinateSystem:
    """A shared coordinate system: a pattern-safe *id* (3.23), a readable *name* (the id
    when none is given), *axes* (a tuple of :class:`Axis`, or None until the writer fills
    them from the images in the system, 3.12) and *estimated* (True for a system a
    registration estimated; a position only goes into a nominal one, 3.7). *extra*: the
    system's other keys, as read."""
    id: str
    name: Optional[str] = None
    axes: Optional[tuple] = None
    estimated: bool = False
    extra: dict = field(default_factory=dict, hash=False)

    def __post_init__(self):
        if not isinstance(self.id, str) or not PATTERN_SAFE.fullmatch(self.id):
            raise ValueError(f"CoordinateSystem: id {self.id!r} is not pattern-safe (plan "
                             f"3.23: {PATTERN_SAFE.pattern}); system_id() makes an id from "
                             "any text")
        if self.name is None:
            object.__setattr__(self, "name", self.id)
        elif not isinstance(self.name, str):
            raise ValueError(f"CoordinateSystem {self.id!r}: name must be a string or None, "
                             f"not {self.name!r}")
        if self.axes is not None:
            axes = tuple(self.axes)
            bad = [a for a in axes if not isinstance(a, Axis)]
            if bad:
                raise TypeError(f"CoordinateSystem {self.id!r}: axes must be Axis objects, "
                                f"not {bad[0]!r}")
            names = [a.name for a in axes]
            if len(set(names)) != len(names):
                raise ValueError(f"CoordinateSystem {self.id!r}: axis names repeat: {names}")
            object.__setattr__(self, "axes", axes)
        if not isinstance(self.estimated, bool):
            raise TypeError(f"CoordinateSystem {self.id!r}: estimated must be True or False, "
                            f"not {self.estimated!r}")
        object.__setattr__(self, "extra", dict(self.extra))


@dataclass(frozen=True)
class ExternalRef:
    """An edge end outside the opened set (3.29): a system *name* (0.6) or id (RFC 8) in
    the group at *path*, kept as read and written back. The reader does not follow it."""
    path: str
    name: str

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path:
            raise ValueError(f"ExternalRef: path must be a non-empty string, not {self.path!r}")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"ExternalRef {self.path!r}: name must be a non-empty string, "
                             f"not {self.name!r}")


def is_image_key(value) -> bool:
    """Whether *value* is an image key: a tuple of (index name, value) pairs (plan 1.14)."""
    return isinstance(value, tuple) and all(
        isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) for p in value)


def _check_endpoint(value, what: str, where: str):
    if value is None or isinstance(value, ExternalRef) or is_image_key(value):
        return value
    if isinstance(value, str) and PATTERN_SAFE.fullmatch(value):
        return value
    raise TypeError(f"Transformation {where}: {what} must be a coordinate system id, an image "
                    f"key (a tuple of (index name, value) pairs), an ExternalRef or None, "
                    f"not {value!r}")


@dataclass(frozen=True)
class Transformation:
    """An edge of the transformation graph (3.9): *type* (one of ``MODELLED_TYPES`` or any
    other 0.6 type, carried as read), *input* and *output* (a system id, an image key, an
    :class:`ExternalRef`, or None inside a sequence), *params* (the other JSON fields:
    ``"translation"``, ``"scale"``, ``"affine"``, a sequence's ``"transformations"``, ...)
    and an optional *name*. *raw*: the JSON object as read (:func:`transformation_from_json`),
    so a type ozp does not model is written back as it was; not part of equality."""
    type: str
    input: object = None
    output: object = None
    params: dict = field(default_factory=dict, hash=False)
    name: Optional[str] = None
    raw: Optional[dict] = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(f"Transformation: type must be a non-empty string (e.g. one of "
                             f"{MODELLED_TYPES}), not {self.type!r}")
        where = f"{self.type!r}" + (f" {self.name!r}" if self.name else "")
        _check_endpoint(self.input, "input", where)
        _check_endpoint(self.output, "output", where)
        object.__setattr__(self, "params", dict(self.params))

    @property
    def modelled(self) -> bool:
        """Whether ozp models this edge (8.20); a sequence only when all its steps are."""
        if self.type == "sequence":
            return all(step.modelled for step in steps(self))
        return self.type in MODELLED_TYPES


# -- ids (3.23, 8.14) ---------------------------------------------------------------

def clean_id_part(text) -> str:
    """*text* as one part of an id ozp makes, as micro-reader cleans it: every character
    outside ``[a-zA-Z0-9_-]`` becomes "_", and so does a "-" that starts or ends the part
    or follows another "-" (no "." and no "--": both are reserved, 3.23)."""
    if text is None or not str(text):
        raise ValueError(f"an empty id part ({text!r}): nothing to name")
    text = str(text)
    out = []
    for i, ch in enumerate(text):
        if ch == "-":
            out.append("-" if 0 < i < len(text) - 1 and text[i - 1] != "-" else "_")
        elif ch.isascii() and (ch.isalnum() or ch == "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def system_id(what: str, *scope) -> str:
    """The id of a system (3.23, 8.14): *what* it is, then the tokens that scope it, each
    cleaned and joined with "-": ``system_id("well", "p1", "B2") == "well-p1-B2"``."""
    return "-".join(clean_id_part(part) for part in (what, *scope))


# -- reading micro-reader's objects by their fields (3.16) ------------------------

def axis_from(obj) -> Axis:
    """An :class:`Axis` from any object with the same fields (micro-reader's)."""
    if isinstance(obj, Axis):
        return obj
    return Axis(str(obj.name), getattr(obj, "type", None), getattr(obj, "unit", None),
                getattr(obj, "discrete", None), getattr(obj, "long_name", None),
                dict(getattr(obj, "extra", None) or {}))


def system_from(obj) -> CoordinateSystem:
    """A :class:`CoordinateSystem` from any object with the same fields (micro-reader's)."""
    if isinstance(obj, CoordinateSystem):
        return obj
    axes = getattr(obj, "axes", None)
    return CoordinateSystem(str(obj.id), getattr(obj, "name", None),
                            None if axes is None else tuple(axis_from(a) for a in axes),
                            bool(getattr(obj, "estimated", False)),
                            dict(getattr(obj, "extra", None) or {}))


def _endpoint_from(value):
    if value is None or isinstance(value, (str, ExternalRef)):
        return value
    if isinstance(value, (tuple, list)) and all(isinstance(p, (tuple, list)) and len(p) == 2
                                                for p in value):
        return tuple((str(n), v) for n, v in value)
    if hasattr(value, "path") and hasattr(value, "name"):          # another package's ref
        return ExternalRef(str(value.path), str(value.name))
    return value                                     # Transformation raises with the reason


def transformation_from(obj) -> Transformation:
    """A :class:`Transformation` from any object with the same fields (micro-reader's)."""
    if isinstance(obj, Transformation):
        return obj
    return Transformation(str(obj.type), _endpoint_from(getattr(obj, "input", None)),
                          _endpoint_from(getattr(obj, "output", None)),
                          dict(getattr(obj, "params", None) or {}), getattr(obj, "name", None))


# -- the transform model (3.9, 8.20) --------------------------------------------------

def steps(t: Transformation) -> Tuple[Transformation, ...]:
    """The steps of a sequence, in the order they apply (each a :class:`Transformation`,
    whether given as one or as its 0.6 JSON object); a single step for anything else."""
    if t.type != "sequence":
        return (t,)
    out = []
    for item in t.params.get("transformations") or []:
        out.append(item if isinstance(item, Transformation)
                   else transformation_from_json(item, endpoint=lambda v: None))
    return tuple(out)


def _vector(t: Transformation, key: str, ndim: Optional[int]) -> np.ndarray:
    if "path" in t.params and key not in t.params:
        raise TransformError(f"{t.type} {t.name or ''}: its parameters are in the Zarr array "
                             f"{t.params['path']!r}; ozp carries such an edge as read and "
                             f"does not model it")
    values = t.params.get(key)
    if values is None:
        raise TransformError(f"{t.type}: no {key!r} parameter")
    v = np.asarray(values, dtype=float).reshape(-1)
    if ndim is not None and v.size != ndim:
        raise TransformError(f"{t.type}: {v.size} values for {ndim} axes")
    return v


def matrix(t: Transformation, ndim: int) -> np.ndarray:
    """The homogeneous (ndim + 1) x (ndim + 1) matrix of a modelled transformation over
    `ndim` axes (a sequence: its steps composed, the first applied first). A type ozp
    does not model, parameters in a Zarr array, or an affine that changes the number of
    axes raise :class:`TransformError`."""
    eye = np.eye(ndim + 1)
    if t.type == "identity":
        return eye
    if t.type == "translation":
        eye[:ndim, ndim] = _vector(t, "translation", ndim)
        return eye
    if t.type == "scale":
        eye[:ndim, :ndim] = np.diag(_vector(t, "scale", ndim))
        return eye
    if t.type == "affine":
        if "path" in t.params and "affine" not in t.params:
            _vector(t, "affine", None)                       # raises: stored in Zarr
        rows = np.asarray(t.params.get("affine"), dtype=float)
        if rows.ndim == 1 and rows.size == ndim * (ndim + 1):
            rows = rows.reshape(ndim, ndim + 1)
        if rows.shape == (ndim + 1, ndim + 1):
            rows = rows[:ndim]
        if rows.shape != (ndim, ndim + 1):
            raise TransformError(f"affine of shape {rows.shape}: ozp models an affine that "
                                 f"keeps the number of axes ({ndim} x {ndim + 1}); one that "
                                 f"changes it is carried as read")
        eye[:ndim] = rows
        return eye
    if t.type == "sequence":
        out = eye
        for step in steps(t):
            out = matrix(step, ndim) @ out
        return out
    raise TransformError(f"transformation type {t.type!r}: ozp carries it as read and does "
                         f"not model it (modelled: {', '.join(MODELLED_TYPES)})")


def from_matrix(m: np.ndarray, input=None, output=None, name: Optional[str] = None
                ) -> Transformation:
    """The simplest modelled transformation with homogeneous matrix *m*: identity,
    translation, scale, sequence[scale, translation], or affine."""
    m = np.asarray(m, dtype=float)
    n = m.shape[0] - 1
    linear, offset = m[:n, :n], m[:n, n]
    diagonal = np.allclose(linear, np.diag(np.diag(linear)), rtol=0, atol=0)
    if diagonal and np.array_equal(np.diag(linear), np.ones(n)):
        if not offset.any():
            return Transformation("identity", input, output, name=name)
        return Transformation("translation", input, output, {"translation": offset.tolist()}, name)
    if diagonal:
        scale = Transformation("scale", None, None, {"scale": np.diag(linear).tolist()})
        if not offset.any():
            return Transformation("scale", input, output, {"scale": np.diag(linear).tolist()}, name)
        return Transformation("sequence", input, output, {"transformations": [
            scale, Transformation("translation", None, None, {"translation": offset.tolist()})]},
            name)
    return Transformation("affine", input, output, {"affine": m[:n].tolist()}, name)


def compose(*ts: Transformation, ndim: int, input=None, output=None) -> Transformation:
    """The transformations applied one after another (the first first), as one modelled
    transformation from *input* to *output*."""
    out = np.eye(ndim + 1)
    for t in ts:
        out = matrix(t, ndim) @ out
    return from_matrix(out, input, output)


def invert(t: Transformation, ndim: int) -> Transformation:
    """The inverse of a modelled transformation, from its output to its input. A singular
    one raises :class:`TransformError`."""
    m = matrix(t, ndim)
    if abs(np.linalg.det(m[:ndim, :ndim])) < 1e-300:
        raise TransformError(f"{t.type} {t.name or ''}: singular, it has no inverse")
    return from_matrix(np.linalg.inv(m), t.output, t.input, t.name)


def scale_translation(t: Transformation, ndim: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The transformation as ``sequence[scale s, translation t]`` (3.20): ``(s, t)``, or
    None when it is not of that form (it turns, shears or mirrors with a zero scale)."""
    m = matrix(t, ndim)
    linear = m[:ndim, :ndim]
    if not np.array_equal(linear, np.diag(np.diag(linear))):
        return None
    return np.diag(linear).copy(), m[:ndim, ndim].copy()


# -- units (3.20) ----------------------------------------------------------------------

#: the size of a unit in its base unit (metre or second): the UDUNITS-2 names NGFF axes use
_UNIT_SIZE = {
    "angstrom": (1e-10, "m"), "picometer": (1e-12, "m"), "nanometer": (1e-9, "m"),
    "micrometer": (1e-6, "m"), "millimeter": (1e-3, "m"), "centimeter": (1e-2, "m"),
    "decimeter": (1e-1, "m"), "meter": (1.0, "m"), "kilometer": (1e3, "m"),
    "picosecond": (1e-12, "s"), "nanosecond": (1e-9, "s"), "microsecond": (1e-6, "s"),
    "millisecond": (1e-3, "s"), "second": (1.0, "s"), "minute": (60.0, "s"),
    "hour": (3600.0, "s"), "day": (86400.0, "s"),
}


def unit_factor(unit: Optional[str], into: Optional[str]) -> Optional[float]:
    """How many `into` one `unit` is (``micrometer`` into ``nanometer``: 1000.0); 1.0 for
    the same unit or two missing ones; None when they cannot be converted (unknown
    units, a length and a time, one missing)."""
    if unit == into:
        return 1.0
    a, b = _UNIT_SIZE.get(str(unit)), _UNIT_SIZE.get(str(into))
    if a is None or b is None or a[1] != b[1]:
        return None
    return a[0] / b[0]


# -- a lone Pyramid's system (3.26, 8.25) -------------------------------------------

#: the custom metadata blocks a Pyramid keeps its system in: the id, and the definition
#: as a list of 5.28 entries. Written beside ``ome`` (0.5) or at the top level (0.4).
SYSTEM_KEY = "eubi:coordinate_system"
SYSTEMS_KEY = "eubi:coordinate_systems"


def pyramid_system(pyr) -> Tuple[Optional[str], Optional[CoordinateSystem]]:
    """The system a lone pyramid is placed in, from its custom metadata: ``(id,
    definition)``, or ``(None, None)`` when it names none. An id without its definition
    (written before definitions were kept) gets ``CoordinateSystem(id)`` (axes unknown,
    filled when written, 3.12), with a warning."""
    blocks = pyr.meta.custom_blocks if pyr.meta is not None else {}
    sid = blocks.get(SYSTEM_KEY)
    if sid is None:
        return None, None
    sid = str(sid)
    for obj in blocks.get(SYSTEMS_KEY) or []:
        cs = system_from_json(obj)
        if cs.id == sid:
            return sid, cs
    import warnings
    warnings.warn(f"pyramid {getattr(pyr.meta, 'tag', '')!r} names coordinate system "
                  f"{sid!r} without its definition: it gets one with unknown axes, filled "
                  f"from the images when written", UserWarning, stacklevel=2)
    return sid, CoordinateSystem(sid)


def stamp_system(pyr, cs: Optional[CoordinateSystem]) -> None:
    """Record, in place, the system `pyr` is placed in (`cs`), or that it is unplaced
    (None): the custom blocks :data:`SYSTEM_KEY` and :data:`SYSTEMS_KEY`."""
    if cs is None:
        pyr.meta.set_custom(SYSTEM_KEY, None)
        pyr.meta.set_custom(SYSTEMS_KEY, None)
        return
    pyr.meta.set_custom(SYSTEM_KEY, cs.id)
    pyr.meta.set_custom(SYSTEMS_KEY, [system_to_json(cs)])


def minimal_system(sid: str, pyr) -> CoordinateSystem:
    """The definition a lone pyramid gets when nothing else describes its system (3.26):
    the id, as its name, with the pyramid's own axes and units."""
    units = list(pyr.meta.unit_list or [])
    units += [None] * (len(pyr.meta.axis_order) - len(units))
    axes = tuple(Axis(a, unit=u if a != "c" else None)
                 for a, u in zip(pyr.meta.axis_order, units))
    return CoordinateSystem(sid, axes=axes)


def source_token(pyr):
    """Where a pyramid came from, for lifting (3.26): ids are merged only within one
    source. The store and group it was read from; the set it was taken from
    (``PyramidCollection.pyramid``); else the pyramid itself (never merged with another)."""
    token = getattr(pyr, "_spatial_source", None)
    if token is not None:
        return token
    group = getattr(pyr, "gr", None)
    if group is not None:
        return ("store", str(getattr(group, "store", "")), str(getattr(group, "path", "")))
    return ("pyramid", id(pyr))


def source_name(pyr) -> Optional[str]:
    """A readable name for where a pyramid came from, for the ``<set>--`` prefix of a
    colliding id (8.16): the set's name, or the store's stem and the group's path."""
    name = getattr(pyr, "_spatial_set_name", None)
    if name:
        return str(name)
    group = getattr(pyr, "gr", None)
    if group is None:
        return None
    store = getattr(group, "store", None)
    root = str(getattr(store, "root", None) or store or "")
    parts = [p for p in re.split(r"[\\/]", root) if p and not p.endswith(":")]
    stem = re.sub(r"(\.ome)?\.zarr$", "", parts[-1]) if parts else ""
    path = str(getattr(group, "path", "") or "").strip("/")
    return "_".join(p for p in (stem, path.replace("/", "_")) if p) or None


#: the order merged axes are listed in; others follow in the order first seen
_AXIS_ORDER = "tczyx"


def merge_systems(a: CoordinateSystem, b: CoordinateSystem) -> CoordinateSystem:
    """One definition for two of one system (one id, one space): unknown axes take the
    known ones; known axes are joined (a 2-D and a 3-D image of one stage). An axis the
    two define differently (another unit), or a different ``estimated``, raises
    :class:`InvariantError`."""
    if a == b or b.axes is None:
        return a
    if a.estimated != b.estimated:
        raise InvariantError(f"system {a.id!r} is estimated in one place and nominal in "
                             f"another")
    if a.axes is None:
        return CoordinateSystem(a.id, a.name, b.axes, a.estimated, dict(a.extra))
    axes = {ax.name: ax for ax in a.axes}
    for ax in b.axes:
        if ax.name in axes and axes[ax.name] != ax:
            raise InvariantError(f"system {a.id!r} has axis {ax.name!r} as {axes[ax.name]} in "
                                 f"one place and as {ax} in another")
        axes.setdefault(ax.name, ax)
    order = sorted(axes, key=lambda n: (_AXIS_ORDER.index(n) if n in _AXIS_ORDER
                                        else len(_AXIS_ORDER), list(axes).index(n)))
    return CoordinateSystem(a.id, a.name, tuple(axes[n] for n in order), a.estimated,
                            dict(a.extra))


# -- JSON (5.28) --------------------------------------------------------------------

def axis_to_json(axis: Axis) -> dict:
    out = {"name": axis.name, "type": axis.type}
    if axis.unit is not None:
        out["unit"] = axis.unit
    if axis.discrete is not None:
        out["discrete"] = axis.discrete
    if axis.long_name is not None:
        out["longName"] = axis.long_name
    out.update(axis.extra)
    return out


def axis_from_json(obj: dict) -> Axis:
    known = {"name", "type", "unit", "discrete", "longName"}
    return Axis(str(obj["name"]), obj.get("type"), obj.get("unit"), obj.get("discrete"),
                obj.get("longName"), {k: v for k, v in obj.items() if k not in known})


def system_to_json(cs: CoordinateSystem, *, axes: bool = True) -> dict:
    """A system as an entry of ``eubi:coordinate_systems`` (5.28): a 0.6 coordinate-system
    object (``name`` = the id, ``axes`` when filled and asked for) plus ``eubi:name`` and
    ``eubi:estimated``, and the keys it was read with."""
    out = {"name": cs.id}
    if axes and cs.axes is not None:
        out["axes"] = [axis_to_json(a) for a in cs.axes]
    out["eubi:name"] = cs.name
    out["eubi:estimated"] = cs.estimated
    for k, v in cs.extra.items():
        out.setdefault(k, v)
    return out


def system_from_json(obj: dict) -> CoordinateSystem:
    """A system from a 0.6 coordinate-system object or an ``eubi:coordinate_systems``
    entry (5.28); RFC 8 objects give the id as ``id``."""
    sid = obj.get("id", obj.get("name"))
    axes = obj.get("axes")
    extra = {k: v for k, v in obj.items()
             if k not in ("id", "name", "axes", "eubi:name", "eubi:estimated")}
    if "id" in obj and "name" in obj:                 # RFC 8: id plus a readable name
        name = obj.get("eubi:name", obj["name"])
    else:
        name = obj.get("eubi:name")
    return CoordinateSystem(str(sid), name, None if axes is None
                            else tuple(axis_from_json(a) for a in axes),
                            bool(obj.get("eubi:estimated", False)), extra)


def _default_endpoint_to(value):
    if value is None:
        return None
    if isinstance(value, str):
        return {"name": value}
    if isinstance(value, ExternalRef):
        return {"name": value.name, "path": value.path}
    raise TransformError(f"edge end {value!r}: an image key needs the writer's path for it "
                         f"(pass endpoint=)")


def transformation_to_json(t: Transformation, *,
                           endpoint: Callable[[object], object] = _default_endpoint_to) -> dict:
    """An edge as a 0.6 transformation object (5.28). `endpoint` turns an edge end into
    its JSON form (a shared system: ``{"name": id}``; an image key: the writer's
    ``{"name": "physical", "path": ...}``; an :class:`ExternalRef`: its path and name).

    A type ozp does not model, read from JSON, is written as it was read (8.20), byte for
    byte; only its ends are rewritten, and only when `endpoint` is given (a writer
    rebasing paths, 3.29)."""
    if t.raw is not None and not t.modelled:
        import copy
        out = copy.deepcopy(t.raw)
        if endpoint is not _default_endpoint_to:
            for key in ("input", "output"):
                value = endpoint(getattr(t, key))
                if value is not None:
                    out[key] = value
        return out
    out = {"type": t.type}
    for key in ("input", "output"):
        value = endpoint(getattr(t, key))
        if value is not None:
            out[key] = value
    if t.name is not None:
        out["name"] = t.name
    for k, v in t.params.items():
        if k == "transformations":
            v = [transformation_to_json(s if isinstance(s, Transformation)
                                        else transformation_from_json(s, endpoint=lambda o: None),
                                        endpoint=endpoint) for s in v]
        out[k] = v
    return out


def _default_endpoint_from(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict) and "path" in obj:
        return ExternalRef(str(obj["path"]), str(obj.get("name", obj.get("id"))))
    if isinstance(obj, dict):
        return str(obj.get("name", obj.get("id")))
    raise TransformError(f"edge end {obj!r}: not a 0.6 coordinate-system reference")


def transformation_from_json(obj: dict, *, endpoint: Callable[[object], object]
                             = _default_endpoint_from) -> Transformation:
    """An edge from a 0.6 transformation object. `endpoint` turns a JSON end into the
    model's (the reader maps an image's ``{"name": ..., "path": ...}`` to its key)."""
    params = {k: v for k, v in obj.items() if k not in ("type", "input", "output", "name")}
    if "transformations" in params:
        params["transformations"] = [transformation_from_json(s, endpoint=endpoint)
                                     for s in params["transformations"]]
    import copy
    return Transformation(str(obj["type"]), endpoint(obj.get("input")),
                          endpoint(obj.get("output")), params, obj.get("name"),
                          raw=copy.deepcopy(obj))
