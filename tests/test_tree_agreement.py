"""One rule, implemented twice (plan 3.17): ozp's ``tree_from_file`` (through
``PyramidCollection.from_file``) and micro-reader's ``File.collection`` build the same tree
of the same file: the same collections (kind, index, name), in the same order, holding the
same images under the same names. On micro-reader's own ``File`` and ``Image`` objects,
one case per shape of plan section 6; skipped without a micro-reader that has
``File.collection`` (0.2)."""
import pytest

micro_reader = pytest.importorskip("micro_reader")
if not hasattr(micro_reader, "Collection"):
    pytest.skip("micro-reader before File.collection", allow_module_level=True)

from micro_reader import Acquisition, File, Image, Plate, Well  # noqa: E402

from ome_zarr_pyramid import PyramidCollection  # noqa: E402
from ome_zarr_pyramid.core.nodes import CollectionNode  # noqa: E402


def _toy(n, **index):
    return Image(n, "yx", (4, 4), "u1", index=index)


def _micro(node):
    out = []
    for child, name in zip(node.children, node.names):
        if isinstance(child, micro_reader.Collection):
            out.append(_micro(child))
        else:
            out.append((name, tuple((k, v) for k, v in child.index.items() if k != "resolution")))
    return (node.kind, node.index, node.name, out)


def _ozp(node, keys):
    out = []
    for child in node.nodes:
        if isinstance(child, CollectionNode):
            out.append(_ozp(child, keys))
        else:
            out.append((child.name, keys[id(child.pyramid)]))
    return (node.kind, node.index, node.name, out)


def _both(file):
    coll = PyramidCollection.from_file(file)
    keys = {id(pyr): key for key, pyr in coll._pyramids.items()}
    return _micro(file.collection), _ozp(coll.tree, keys)


def _agree(file):
    micro, ozp = _both(file)
    assert ozp == micro
    return micro


def _series_file():
    # 6.2-6.5 shapes: series of tiles, a series of one image, views and illuminations,
    # an image with no index, resolution levels
    images = [_toy(0, series=0, tile=0), _toy(1, series=0, tile=1),
              _toy(2, series=1),
              _toy(3, series=2, view=0, illumination=0), _toy(4, series=2, view=1, illumination=0),
              _toy(5)]
    levels = [Image(6, "yx", (2, 2), "u1", index={"series": 1, "resolution": 1})]
    images[2]._resolutions = [images[2]] + levels
    levels[0]._resolutions = images[2]._resolutions
    return File("/data/set/scenes.czi", images + levels)


def _names(children):
    """A tree's children as collection (name, [children]) / image names."""
    return [(c[2], _names(c[3])) if len(c) == 4 else c[0] for c in children]


def test_series_tiles_views_and_plain_images():
    tree = _agree(_series_file())
    assert (tree[0], tree[2]) == (None, "scenes")
    assert _names(tree[3]) == [("series0", ["tile0", "tile1"]), "series1",
                               ("series2", ["view0_illumination0", "view1_illumination0"]),
                               "image"]


def _plate_file(*, name="Screen 1", label_first=False):
    # 6.1 / 6.6 / 6.7 shapes: wells with one-image fields, a field of tiles, two
    # acquisitions sharing field numbers, an image with no field index, a label
    plate = Plate(name, ("A", "B"), ("1", "2", "3"),
                  (Well(0, 0, "A1"), Well(1, 2, "B3")), (Acquisition(0, "run 1"), Acquisition(1)))
    images = [_toy(0, plate=0, acquisition=0, well=0, field=0),
              _toy(1, plate=0, acquisition=0, well=0, field=1, tile=0),
              _toy(2, plate=0, acquisition=0, well=0, field=1, tile=1),
              _toy(3, plate=0, acquisition=0, well=1, field=0),
              _toy(4, plate=0, acquisition=1, well=1, field=0),
              _toy(5, plate=0, acquisition=1, well=1)]
    label = _toy(6, series=0)
    return File("/data/plate.czi", [label] + images if label_first else images + [label],
                plates=[plate])


@pytest.mark.parametrize("label_first", [False, True])
def test_a_plate_with_tiles_acquisitions_and_a_label(label_first):
    tree = _agree(_plate_file(label_first=label_first))
    plate = ("Screen 1", [
        ("A1", ["acquisition0_field0", ("field1", ["acquisition0_tile0", "acquisition0_tile1"])]),
        # acquisition makes no collection: both runs' field 0 is one field
        ("B3", [("field0", ["acquisition0", "acquisition1"]), "acquisition1"])])
    # the root's children in images order: the label where it is
    assert _names(tree[3]) == (["series0", plate] if label_first else [plate, "series0"])


def test_an_unnamed_plate_is_named_by_its_number():
    tree = _agree(_plate_file(name=""))
    assert tree[3][0][:3] == ("plate", 0, "plate0")


def test_two_plates():
    plates = [Plate(f"P{p}", ("A",), ("1", "2"), (Well(0, 0, "A1"), Well(0, 1, "A2")))
              for p in range(2)]
    images = [_toy(i, plate=p, acquisition=0, well=w, field=0)
              for i, (p, w) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)])]
    _agree(File("/data/screen.idx.xml", images, plates=plates))


@pytest.mark.parametrize("path, stem", [
    ("/data/x.czi", "x"), ("/data/Index.idx.xml", "Index.idx"), ("/data/matl.omp2info", "matl"),
    ("/data/x.ome.tif", "x"), ("/data/x.ome.zarr", "x")])    # M1-2, decided 2026-10-21
def test_the_root_is_named_by_the_files_stem(path, stem):
    micro, ozp = _both(File(path, [_toy(0, series=0)]))
    assert micro[2] == stem and ozp == micro
