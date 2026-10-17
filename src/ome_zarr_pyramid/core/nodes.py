"""Collections of images as a tree of nodes, following NGFF RFC 8 ("Collections and
Extensibility", https://ngff.openmicroscopy.org/rfc/8/index.html).

Every OME-Zarr object is a node with a ``type``, a ``name`` (unique within its
collection), an optional ``id`` (unique within the tree) and ``attributes``:

* :class:`CollectionNode` groups nodes: images, or further collections. A plate is a
  collection with a ``plate`` attribute, a well one with a ``well`` attribute.
* :class:`MultiscaleNode` is one image: an ozp :class:`Pyramid`.
* :class:`SinglescaleNode` is one resolution level. Inside ozp the levels stay inside
  their pyramid; these nodes only carry RFC 8 JSON that lists levels explicitly.

A node holds its content (child nodes, a pyramid) or points to it with a
:class:`NodePath`. Content behind a path is opened on first use by the node's
``loader``, so a tree of thousands of remote images costs nothing until one is read.

Attributes ozp knows (``plate``, ``well``, ``acquisition``, ``labels``) are read into
typed values. Every other attribute, and every node field ozp does not know, is kept
as it was read and written back unchanged.

This module is the in-memory model only: it knows nothing about zarr groups. Reading
and writing the NGFF 0.4/0.5 plate layout and RFC 8 JSON build on it.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Union

#: what an ``id`` may contain (RFC 8: ``[a-zA-Z0-9-_.]+``)
ID_PATTERN = re.compile(r"[A-Za-z0-9\-_.]+\Z")


class InvalidCollection(ValueError):
    """A tree that breaks a rule of RFC 8; the message lists every problem found."""


def _check_id(value, what: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        raise ValueError(f"{what}: id {value!r} must be a string of letters, digits, "
                         f"'-', '_' or '.'")
    return value


def _check_name(value, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what}: name {value!r} must be a non-empty string")
    return value


# -- building blocks ---------------------------------------------------------------

@dataclass(frozen=True)
class NodePath:
    """Where a node's content lives: ``type`` ``"zarr"`` (a zarr group or array) or
    ``"json"`` (a JSON file), or a prefixed extension type; ``path`` relative to the
    document, ``file://`` or ``http(s)://``."""
    type: str
    path: str

    def to_json(self) -> Dict[str, str]:
        return {"type": self.type, "path": self.path}

    @classmethod
    def from_json(cls, data) -> "NodePath":
        if not isinstance(data, dict) or not isinstance(data.get("type"), str) \
                or not isinstance(data.get("path"), str):
            raise ValueError(f"a path must be {{'type': str, 'path': str}}, got {data!r}")
        return cls(data["type"], data["path"])


@dataclass(frozen=True)
class Ref:
    """A reference to a node (or a plate's row, column or acquisition) by ``id``; with
    ``path`` when it lives in another document. ``bare`` remembers a reference written as
    a plain string, ``wrapped`` one written as ``{"ref": ...}`` (the RFC's example
    files do), so either is written back the same way."""
    id: str
    path: Optional[NodePath] = None
    bare: bool = False
    wrapped: bool = False

    def __post_init__(self):
        _check_id(self.id, "reference")

    def to_json(self) -> Union[str, Dict[str, Any]]:
        if self.bare and self.path is None:
            inner: Union[str, Dict[str, Any]] = self.id
        else:
            inner = {"id": self.id}
            if self.path is not None:
                inner["path"] = self.path.to_json()
        return {"ref": inner} if self.wrapped else inner

    @classmethod
    def from_json(cls, data) -> "Ref":
        if isinstance(data, dict) and set(data) == {"ref"}:
            inner = cls.from_json(data["ref"])
            return cls(inner.id, inner.path, inner.bare, wrapped=True)
        if isinstance(data, str):
            return cls(data, bare=True)
        if not isinstance(data, dict) or "id" not in data:
            raise ValueError(f"a reference must be {{'id': str}}, got {data!r}")
        path = NodePath.from_json(data["path"]) if "path" in data else None
        return cls(data["id"], path)


def ref_id(data) -> Optional[str]:
    """The id a reference names, in any of its spellings (``"x"``, ``{"id": "x"}``,
    ``{"ref": "x"}``, ``{"ref": {"id": "x", ...}}``, ``{"$ref": "x"}``); None when it
    names none."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("ref", "$ref"):
            if key in data:
                return ref_id(data[key])
        value = data.get("id")
        return value if isinstance(value, str) else None
    return None


# -- typed attributes --------------------------------------------------------------

def _extra(data: Dict[str, Any], known) -> Dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in data.items() if k not in known}


@dataclass
class PlateItem:
    """A plate's row, column or acquisition: ``id`` and an optional display ``name``.
    Its position in the plate's list is its integer index. ``extra`` keeps the keys
    ozp does not know (e.g. an acquisition's start time), written back unchanged."""
    id: str
    name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _check_id(self.id, "plate item")

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.id}
        if self.name is not None:
            out["name"] = self.name
        out.update(copy.deepcopy(self.extra))
        return out

    @classmethod
    def from_json(cls, data) -> "PlateItem":
        if not isinstance(data, dict) or "id" not in data:
            raise ValueError(f"a plate row, column or acquisition needs an 'id': {data!r}")
        return cls(data["id"], data.get("name"), _extra(data, ("id", "name")))


@dataclass
class PlateAttribute:
    """The ``plate`` attribute of a plate collection: its rows, columns and
    acquisitions, in order (a list position is the integer index)."""
    rows: List[PlateItem]
    columns: List[PlateItem]
    acquisitions: List[PlateItem] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    KEYS: ClassVar = ("rows", "columns", "acquisitions")

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.acquisitions:
            out["acquisitions"] = [a.to_json() for a in self.acquisitions]
        out["columns"] = [c.to_json() for c in self.columns]
        out["rows"] = [r.to_json() for r in self.rows]
        out.update(copy.deepcopy(self.extra))
        return out

    @classmethod
    def from_json(cls, data) -> "PlateAttribute":
        if not isinstance(data, dict) or "rows" not in data or "columns" not in data:
            raise ValueError("a 'plate' attribute needs 'rows' and 'columns'")
        items = {key: [PlateItem.from_json(x) for x in data.get(key) or []]
                 for key in cls.KEYS}
        return cls(items["rows"], items["columns"], items["acquisitions"],
                   _extra(data, cls.KEYS))

    @staticmethod
    def _ids(items: List[PlateItem]) -> List[str]:
        return [item.id for item in items]

    def index_of(self, kind: str, ref: Union[Ref, str]) -> int:
        """The integer index of the row, column or acquisition `ref` names (`kind` one
        of ``"rows"``, ``"columns"``, ``"acquisitions"``)."""
        key = ref.id if isinstance(ref, Ref) else ref
        ids = self._ids(getattr(self, kind))
        if key not in ids:
            raise KeyError(f"the plate has no {kind[:-1]} with id {key!r} (ids: {ids})")
        return ids.index(key)


@dataclass
class WellAttribute:
    """The ``well`` attribute of a well collection: its row and column, as references
    to the enclosing plate's rows and columns."""
    row: Ref
    column: Ref
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        out = {"column": self.column.to_json(), "row": self.row.to_json()}
        out.update(copy.deepcopy(self.extra))
        return out

    @classmethod
    def from_json(cls, data) -> "WellAttribute":
        if not isinstance(data, dict) or "row" not in data or "column" not in data:
            raise ValueError("a 'well' attribute needs 'row' and 'column'")
        return cls(Ref.from_json(data["row"]), Ref.from_json(data["column"]),
                   _extra(data, ("row", "column")))


@dataclass
class LabelsAttribute:
    """The ``labels`` attribute of a label image: the images it labels (``source``) and
    per-label values (``label_attributes``: ``labelValue``, ``color``, ...)."""
    source: List[Ref] = field(default_factory=list)
    label_attributes: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.label_attributes:
            out["labelAttributes"] = copy.deepcopy(self.label_attributes)
        if self.source:
            out["source"] = [s.to_json() for s in self.source]
        out.update(copy.deepcopy(self.extra))
        return out

    @classmethod
    def from_json(cls, data) -> "LabelsAttribute":
        if not isinstance(data, dict):
            raise ValueError(f"a 'labels' attribute must be an object, got {data!r}")
        return cls([Ref.from_json(s) for s in data.get("source") or []],
                   copy.deepcopy(list(data.get("labelAttributes") or [])),
                   _extra(data, ("source", "labelAttributes")))


#: the attributes read into typed values; every other key is kept as it was read
TYPED_ATTRIBUTES: Dict[str, Any] = {
    "plate": PlateAttribute, "well": WellAttribute, "acquisition": Ref,
    "labels": LabelsAttribute}


def attributes_from_json(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Node attributes from JSON: the known ones typed, the others deep-copied."""
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"attributes must be an object, got {data!r}")
    return {key: (TYPED_ATTRIBUTES[key].from_json(value) if key in TYPED_ATTRIBUTES
                  else copy.deepcopy(value)) for key, value in data.items()}


def attributes_to_json(attributes: Dict[str, Any]) -> Dict[str, Any]:
    return {key: (value.to_json() if hasattr(value, "to_json") else copy.deepcopy(value))
            for key, value in attributes.items()}


# -- nodes -------------------------------------------------------------------------

#: opens what a node's path points to: ``loader(node)`` returns a collection's child
#: nodes, a multiscale's ``Pyramid``
Loader = Callable[["Node"], Any]

#: the node fields this module reads; any other field of a node's JSON is ``extra``
_NODE_FIELDS = ("type", "id", "name", "attributes", "path", "nodes")


class Node:
    """A node of any type: ``type``, ``name``, optional ``id``, ``attributes``, and
    optionally the ``path`` its content lives at. A node of a type ozp does not know (a
    prefixed extension type) is kept as a plain ``Node``; its other fields, like any
    node's unknown fields, are ``extra`` and written back unchanged.

    ``ngff_fields`` keeps what an NGFF 0.4/0.5 layout records and RFC 8 has no place for
    (a plate's ``field_count``, a well entry's unknown keys, ...), so that a 0.4/0.5
    read-then-write changes nothing; RFC 8 JSON leaves it out."""

    TYPE: ClassVar[Optional[str]] = None

    def __init__(self, name: str, *, type: Optional[str] = None, id: Optional[str] = None,
                 attributes: Optional[Dict[str, Any]] = None,
                 path: Optional[NodePath] = None, extra: Optional[Dict[str, Any]] = None,
                 loader: Optional[Loader] = None):
        self.type = self.TYPE or type
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(f"node {name!r}: a node needs a type")
        self.name = _check_name(name, f"{self.type} node")
        self.id = _check_id(id, f"{self.type} node {name!r}")
        self.attributes: Dict[str, Any] = dict(attributes or {})
        self.path = path
        self.extra: Dict[str, Any] = copy.deepcopy(dict(extra or {}))
        self.loader = loader
        self.ngff_fields: Dict[str, Any] = {}

    # -- tree ------------------------------------------------------------------
    @property
    def children(self) -> List["Node"]:
        """The child nodes already in memory (nothing is opened)."""
        return []

    def walk(self) -> Iterator["Node"]:
        """This node and every node below it that is in memory, depth first."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, id: str) -> "Node":
        """The node below (or at) this one with this ``id``."""
        for node in self.walk():
            if node.id == id:
                return node
        raise KeyError(f"no node with id {id!r}")

    def copy(self) -> "Node":
        """A copy of the tree from this node down: names, ids, paths and attributes are
        copied; pyramids and loaders are shared (a pyramid's arrays are lazy, and a copy
        must not read them)."""
        new = copy.copy(self)
        new.attributes = attributes_from_json(attributes_to_json(self.attributes))
        new.extra = copy.deepcopy(self.extra)
        new.ngff_fields = copy.deepcopy(self.ngff_fields)
        return new

    # -- JSON ------------------------------------------------------------------
    def _content_json(self) -> Dict[str, Any]:
        return {}

    def to_json(self) -> Dict[str, Any]:
        """The node as RFC 8 JSON."""
        out: Dict[str, Any] = {"type": self.type}
        if self.id is not None:
            out["id"] = self.id
        out["name"] = self.name
        if self.path is not None:
            out["path"] = self.path.to_json()
        out.update(self._content_json())
        if self.attributes:
            out["attributes"] = attributes_to_json(self.attributes)
        out.update(copy.deepcopy(self.extra))
        return out

    @staticmethod
    def from_json(data: Dict[str, Any], *, loader: Optional[Loader] = None) -> "Node":
        """A node (and its subtree) from RFC 8 JSON. `loader` is given to every node, to
        open what their paths point to."""
        if not isinstance(data, dict):
            raise ValueError(f"a node must be an object, got {data!r}")
        cls = _NODE_TYPES.get(data.get("type"), Node)
        # only collections and multiscales have child nodes; elsewhere 'nodes' is extra
        nested = cls in (CollectionNode, MultiscaleNode)
        kwargs: Dict[str, Any] = dict(
            id=data.get("id"), attributes=attributes_from_json(data.get("attributes")),
            path=NodePath.from_json(data["path"]) if "path" in data else None,
            extra=_extra(data, _NODE_FIELDS if nested else _NODE_FIELDS[:-1]),
            loader=loader)
        if cls is Node:
            kwargs["type"] = data.get("type")
        if nested and "nodes" in data:
            if "path" in data:
                raise ValueError(f"node {data.get('name')!r}: give 'nodes' or 'path', "
                                 f"not both")
            if not isinstance(data["nodes"], list):
                raise ValueError(f"node {data.get('name')!r}: 'nodes' must be a list")
            kwargs["nodes"] = [Node.from_json(child, loader=loader) for child in data["nodes"]]
        return cls(data.get("name"), **kwargs)

    # -- comparison ------------------------------------------------------------
    def _same_content(self, other: "Node") -> bool:
        return True

    def __eq__(self, other) -> bool:
        """Equal type, name, id, path, attributes and extra fields, and equal content in
        memory (child nodes; the same pyramid object). Nothing is opened to compare."""
        if not isinstance(other, Node) or type(other) is not type(self):
            return NotImplemented
        return (self.type == other.type and self.name == other.name
                and self.id == other.id and self.path == other.path
                and attributes_to_json(self.attributes) == attributes_to_json(other.attributes)
                and self.extra == other.extra and self._same_content(other))

    __hash__ = None  # type: ignore[assignment]   # mutable

    def __repr__(self) -> str:
        where = f", path={self.path.path!r}" if self.path is not None else ""
        ident = f", id={self.id!r}" if self.id is not None else ""
        return f"{type(self).__name__}({self.name!r}{ident}{where})"


class _HasContent(Node):
    """A node whose content may be in memory or behind its path."""

    def _open(self):
        if self.loader is None:
            where = (f"points to {self.path.path!r} ({self.path.type})" if self.path is not None
                     else "lists its content inline")
            raise LookupError(f"{self.type} node {self.name!r} {where} and has no loader "
                              f"to open it")
        return self.loader(self)


class CollectionNode(_HasContent):
    """A collection: child nodes held in memory (``nodes``) or behind ``path``, opened
    on first access to :attr:`nodes`. Both together are a collection behind a path whose
    children are already open. Child names are unique; ``coll["well A01"]`` looks a
    child up by name."""

    TYPE = "collection"

    def __init__(self, name: str, nodes: Optional[List[Node]] = None, *,
                 path: Optional[NodePath] = None, **kwargs):
        super().__init__(name, path=path, **kwargs)
        self._nodes: Optional[List[Node]] = None
        if nodes is not None or path is None:
            self._nodes = []
            for node in nodes or []:
                self.add(node)

    @property
    def loaded(self) -> bool:
        """Whether the child nodes are in memory."""
        return self._nodes is not None

    @property
    def nodes(self) -> List[Node]:
        """The child nodes, opened through the loader when they are behind a path."""
        if self._nodes is None:
            children = self._open()
            if isinstance(children, CollectionNode):
                children = children.nodes
            self._nodes = []
            for node in children:
                self.add(node)
        return list(self._nodes)

    @property
    def children(self) -> List[Node]:
        return list(self._nodes or [])

    def add(self, node: Node) -> Node:
        """Append a child node; its name must be new in this collection."""
        if not isinstance(node, Node):
            raise TypeError(f"collection {self.name!r}: a child must be a Node, got "
                            f"{type(node).__name__}")
        if self._nodes is None:
            self.nodes                                      # open before appending
        if any(child.name == node.name for child in self._nodes):
            raise ValueError(f"collection {self.name!r} already has a node named "
                             f"{node.name!r}")
        self._nodes.append(node)
        return node

    def __getitem__(self, name: str) -> Node:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"collection {self.name!r} has no node named {name!r}")

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def copy(self) -> "CollectionNode":
        new = super().copy()
        new._nodes = None if self._nodes is None else [n.copy() for n in self._nodes]
        return new

    def _content_json(self) -> Dict[str, Any]:
        # a collection behind a path is written as that path, opened or not
        if self.path is not None:
            return {}
        return {"nodes": [node.to_json() for node in self._nodes or []]}

    def _same_content(self, other) -> bool:
        return self._nodes == other._nodes

    # -- validation ------------------------------------------------------------
    def problems(self) -> List[str]:
        """Every RFC 8 rule the tree in memory breaks (nothing is opened): duplicate ids
        in the tree, duplicate names in a collection, wells whose row or column is not
        one of the enclosing plate's, acquisitions the plate does not list, and label
        sources that name no node of the tree."""
        found: List[str] = []
        seen: Dict[str, Node] = {}
        for node in self.walk():
            if node.id is not None:
                if node.id in seen:
                    found.append(f"id {node.id!r} is used twice ({seen[node.id]!r} and "
                                 f"{node!r})")
                seen.setdefault(node.id, node)
        self._check(None, found, seen)
        return found

    def _check(self, plate: Optional[PlateAttribute], found: List[str],
               ids: Dict[str, Node]) -> None:
        own = self.attributes.get("plate")
        if isinstance(own, PlateAttribute):
            plate = own
            for kind in PlateAttribute.KEYS:
                values = PlateAttribute._ids(getattr(own, kind))
                for dup in sorted({v for v in values if values.count(v) > 1}):
                    found.append(f"plate {self.name!r}: {kind[:-1]} id {dup!r} is used twice")
        names = [child.name for child in self.children]
        for dup in sorted({n for n in names if names.count(n) > 1}):
            found.append(f"collection {self.name!r}: two nodes are named {dup!r}")
        well = self.attributes.get("well")
        if isinstance(well, WellAttribute):
            if plate is None:
                found.append(f"well {self.name!r} is not inside a plate")
            else:
                for kind, ref in (("rows", well.row), ("columns", well.column)):
                    if ref.path is None and ref.id not in PlateAttribute._ids(getattr(plate, kind)):
                        found.append(f"well {self.name!r}: {kind[:-1]} {ref.id!r} is not "
                                     f"one of the plate's {kind}")
        for child in self.children:
            acq = child.attributes.get("acquisition")
            if isinstance(acq, Ref) and acq.path is None:
                if plate is None:
                    found.append(f"{child!r} has an acquisition but is not inside a plate")
                elif acq.id not in PlateAttribute._ids(plate.acquisitions):
                    found.append(f"{child!r}: acquisition {acq.id!r} is not one of the "
                                 f"plate's acquisitions")
            labels = child.attributes.get("labels")
            if isinstance(labels, LabelsAttribute):
                for src in labels.source:
                    if src.path is None and src.id not in ids:
                        found.append(f"{child!r}: label source {src.id!r} names no node")
            if isinstance(child, CollectionNode):
                child._check(plate, found, ids)

    def validate(self) -> "CollectionNode":
        """Raise :class:`InvalidCollection` listing every problem (see `problems`);
        returns the collection when there is none."""
        found = self.problems()
        if found:
            raise InvalidCollection(f"collection {self.name!r}: " + "; ".join(found))
        return self


class MultiscaleNode(_HasContent):
    """One image: an ozp ``Pyramid`` held in memory, or behind ``path`` and opened on
    first access to :attr:`pyramid`. ``levels`` keeps the singlescale nodes of RFC 8
    JSON that lists a multiscale's levels inline."""

    TYPE = "multiscale"

    def __init__(self, name: str, pyramid=None, *, path: Optional[NodePath] = None,
                 nodes: Optional[List[Node]] = None, **kwargs):
        super().__init__(name, path=path, **kwargs)
        self._pyramid = pyramid
        self.levels: List[Node] = list(nodes or [])

    @property
    def loaded(self) -> bool:
        return self._pyramid is not None

    @property
    def pyramid(self):
        """The image's ``Pyramid``, opened through the loader when it is behind a path or
        lists its levels inline."""
        if self._pyramid is None:
            if self.path is None and not self.levels:
                raise LookupError(f"multiscale node {self.name!r} holds no pyramid")
            self._pyramid = self._open()
        return self._pyramid

    @pyramid.setter
    def pyramid(self, value) -> None:
        self._pyramid = value

    @property
    def children(self) -> List[Node]:
        return list(self.levels)

    def copy(self) -> "MultiscaleNode":
        new = super().copy()
        new.levels = [n.copy() for n in self.levels]
        return new

    def _content_json(self) -> Dict[str, Any]:
        if self.path is not None:
            return {}
        if self.levels:
            return {"nodes": [n.to_json() for n in self.levels]}
        raise NotImplementedError(
            f"multiscale {self.name!r} holds its pyramid in memory: its RFC 8 JSON is "
            f"written together with the pyramid (ome_zarr_pyramid.core.rfc8.write_rfc8)")

    def _same_content(self, other) -> bool:
        return self._pyramid is other._pyramid and self.levels == other.levels


class SinglescaleNode(Node):
    """One resolution level in RFC 8 JSON: ``path`` to its array, and attributes with its
    ``coordinateTransformations``."""

    TYPE = "singlescale"


_NODE_TYPES: Dict[Optional[str], type] = {
    "collection": CollectionNode, "multiscale": MultiscaleNode,
    "singlescale": SinglescaleNode}


def is_plate(node: Node) -> bool:
    """Whether `node` is a plate: a collection with a ``plate`` attribute."""
    return isinstance(node, CollectionNode) and "plate" in node.attributes


def is_well(node: Node) -> bool:
    """Whether `node` is a well: a collection with a ``well`` attribute."""
    return isinstance(node, CollectionNode) and "well" in node.attributes
