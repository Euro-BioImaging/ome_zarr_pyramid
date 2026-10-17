"""Plates of a :class:`PyramidCollection`, navigated as in micro-reader:
``coll.plates[p].wells[w].fields[f].image``.

The classes mirror micro-reader's (``Plate``, ``Well``, ``Field``, ``Acquisition``):
the same fields, integer rows and columns counted from 0, ``wells[w]`` the well with
index ``"well": w``, ``fields[f]`` the f-th field present (``.index`` its number).
Navigation follows containment only; orthogonal indices (acquisition, view, tile, ...)
are picked with ``Field.select``.

They are views, built from the collection's node tree (a collection with a ``plate``
attribute, its wells collections with a ``well`` attribute) and its keys; they hold no
pixels. A field's ``images`` are the collection's pyramids, opened when first asked for.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field, replace
from typing import Any, Dict, List, Optional, Tuple

from ome_zarr_pyramid.core.nodes import (
    CollectionNode, ID_PATTERN, InvalidCollection, MultiscaleNode, PlateAttribute,
    PlateItem, Ref, WellAttribute, is_plate)

#: the indices that nest (micro-reader's containment indices); the others are orthogonal
CONTAINMENT = ("plate", "well", "field", "scene", "tile", "part", "resolution")


@dataclass(frozen=True, eq=False)
class Field:
    """One field of a well: its field ``index`` and its images' collection ``keys``
    (``images`` opens them). When an orthogonal index (acquisition, view, ...) varies
    in the field, ``select`` picks one first: ``field.select(acquisition=1).image``."""
    index: int
    keys: Tuple = ()
    collection: Any = _field(default=None, repr=False)

    @property
    def images(self) -> tuple:
        """The field's pyramids, in key order."""
        return tuple(self.collection[key] for key in self.keys)

    def _orthogonal(self) -> List[str]:
        values: Dict[str, set] = {}
        for key in self.keys:
            for name, value in key:
                if name not in CONTAINMENT:
                    values.setdefault(name, set()).add(value)
        return sorted(name for name, vals in values.items() if len(vals) > 1)

    @property
    def tiles(self) -> list:
        """The field's pyramids in tile order (one when it is not tiled)."""
        varying = self._orthogonal()
        if varying:
            raise ValueError(
                f"field {self.index} also has several {', '.join(varying)}: pick one with "
                f"select({varying[0]}=...) first, or use .images")
        keys = sorted(self.keys, key=lambda key: dict(key).get("tile", 0))
        return [self.collection[key] for key in keys]

    @property
    def image(self):
        """The field's one pyramid."""
        if len(self.keys) != 1:
            raise ValueError(f"field {self.index} has {len(self.keys)} images: use .tiles "
                             f"(or .select(...)) to pick one")
        return self.collection[self.keys[0]]

    def select(self, **index) -> "Field":
        """This field with only the images whose index matches, e.g.
        ``select(acquisition=1)``."""
        return replace(self, keys=tuple(
            key for key in self.keys
            if all(dict(key).get(name) == value for name, value in index.items())))

    def __repr__(self) -> str:
        return f"<Field {self.index}: {len(self.keys)} images>"


@dataclass(frozen=True)
class Well:
    """A well that holds images: its 0-based row and column, its name (``"B2"``) and its
    fields (``fields[j]`` is the j-th field present; ``.index`` is its number)."""
    row: int
    column: int
    name: str
    fields: Tuple[Field, ...] = _field(default=(), compare=False, repr=False)


@dataclass(frozen=True)
class Acquisition:
    """One imaging run of a plate."""
    id: int
    name: str = ""


@dataclass(frozen=True)
class Plate:
    """A plate: its name (``""`` when its source has none), its row and column names
    (``rows[i]`` names row ``i``), the wells that hold images (``wells[w]`` is index
    ``"well": w``) and its acquisitions. ``node`` is its collection node."""
    name: str
    rows: Tuple[str, ...]
    columns: Tuple[str, ...]
    wells: Tuple[Well, ...]
    acquisitions: Tuple[Acquisition, ...] = ()
    node: Any = _field(default=None, compare=False, repr=False)

    def well_index(self, name: str) -> int:
        """The index of the well called `name` (``"B2"``), for ``select(well=...)`` and
        ``wells[...]``."""
        for i, well in enumerate(self.wells):
            if well.name == name:
                return i
        raise KeyError(f"no well {name!r} with images in plate {self.name!r}; "
                       f"wells: {[w.name for w in self.wells]}")

    def __repr__(self) -> str:
        n = len(self.wells)
        return (f"<Plate {self.name!r} {len(self.rows)} x {len(self.columns)}, "
                f"{n} well{'' if n == 1 else 's'} with images>")


def plate_nodes(tree) -> List[CollectionNode]:
    """The plate collections of a tree: the tree itself, or its plate children."""
    if tree is None:
        return []
    if is_plate(tree):
        return [tree]
    if isinstance(tree, CollectionNode):
        return [node for node in tree.children if is_plate(node)]
    return []


def plates_of(collection) -> List[Plate]:
    """The plates of a collection: one :class:`Plate` per plate node of its tree, with
    each well's fields made from the collection's keys (``plate``, ``well``,
    ``field``)."""
    groups: Dict[Tuple[int, int, int], List] = {}
    for key in collection:
        index = dict(key)
        if "well" in index:
            groups.setdefault((index.get("plate", 0), index["well"], index.get("field", 0)),
                              []).append(key)
    out = []
    for p, node in enumerate(plate_nodes(collection.tree)):
        attr: PlateAttribute = node.attributes["plate"]
        wells = []
        for w, well in enumerate(node.nodes):
            ref = well.attributes["well"]
            numbers = sorted(f for (pp, ww, f) in groups if pp == p and ww == w)
            wells.append(Well(attr.index_of("rows", ref.row), attr.index_of("columns", ref.column),
                              well.name, tuple(Field(f, tuple(groups[(p, w, f)]), collection)
                                               for f in numbers)))
        acquisitions = tuple(
            Acquisition(int(a.id) if a.id.isdigit() else i, a.name or "")
            for i, a in enumerate(attr.acquisitions))
        name = node.ngff_fields.get("plate", {}).get("name", node.name)
        out.append(Plate(name if isinstance(name, str) else node.name,
                         tuple(r.name or r.id for r in attr.rows),
                         tuple(c.name or c.id for c in attr.columns),
                         tuple(wells), acquisitions, node))
    return out


# -- micro-reader plates into the tree ----------------------------------------------

def _item(name: str, i: int, kind: str) -> PlateItem:
    """A row or column named `name`, with its name as id when it can be one."""
    text = str(name)
    return PlateItem(text if ID_PATTERN.match(text) else f"{kind}{i}", text)


def _image_name(key) -> str:
    return "_".join(f"{k}{v}" for k, v in key if k not in ("plate", "well")) or "image"


def tree_from_file(file, entries: Dict[Tuple, Any], name: Optional[str] = None
                   ) -> Optional[CollectionNode]:
    """The node tree of a micro-reader file's plates (``file.plates``): a collection
    holding one plate collection per plate, whose wells hold the file's pyramids
    (`entries`, ``{key: Pyramid}``). ``None`` for a file that is not a plate."""
    plates = list(getattr(file, "plates", None) or [])
    if not plates:
        return None
    by_well: Dict[Tuple[int, int], List] = {}
    for key in entries:
        index = dict(key)
        if "well" in index:
            by_well.setdefault((index.get("plate", 0), index["well"]), []).append(key)
    root = CollectionNode(name or "file")
    for p, plate in enumerate(plates):
        attr = PlateAttribute(
            rows=[_item(r, i, "row") for i, r in enumerate(plate.rows)],
            columns=[_item(c, i, "column") for i, c in enumerate(plate.columns)],
            acquisitions=[PlateItem(str(a.id), a.name or None) for a in plate.acquisitions])
        node = CollectionNode(plate.name or f"plate{p}", attributes={"plate": attr})
        # the plate's own name, '' when it has none (micro-reader's rule), is the name
        # an NGFF plate is written with
        node.ngff_fields = {"plate": {"name": plate.name, "field_count": None},
                            "node_name": node.name}
        outside = [w.name for w in plate.wells
                   if not (0 <= w.row < len(plate.rows) and 0 <= w.column < len(plate.columns))]
        if outside:
            raise InvalidCollection(
                f"plate {plate.name!r}: the file puts wells {outside} outside its "
                f"{len(plate.rows)} x {len(plate.columns)} layout (rows {list(plate.rows)}, "
                f"columns {list(plate.columns)})")
        for w, well in enumerate(plate.wells):
            well_node = node.add(CollectionNode(well.name, attributes={"well": WellAttribute(
                Ref(attr.rows[well.row].id), Ref(attr.columns[well.column].id))}))
            for key in by_well.get((p, w), []):
                a = dict(key).get("acquisition")
                attributes = ({"acquisition": Ref(attr.acquisitions[a].id)}
                              if a is not None and a < len(attr.acquisitions) else {})
                well_node.add(MultiscaleNode(_image_name(key), entries[key],
                                             attributes=attributes))
        root.add(node)
    return root
