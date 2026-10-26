"""ozp's select and micro-reader's select give the same images for the same arguments
(they ship together: ozp 0.5.0 with micro-reader's next release). One plate CZI written
with micro-reader's own test writer, read by micro-reader, then turned into a collection."""
import os
import sys

import pytest

micro_reader = pytest.importorskip("micro_reader")
TESTS = os.path.join(os.path.dirname(micro_reader.__file__), "..", "..", "..", "tests")
if not os.path.isfile(os.path.join(TESTS, "test_hcs.py")):
    pytest.skip("micro-reader's tests (its CZI writer) are not next to it", allow_module_level=True)
sys.path.insert(0, os.path.abspath(TESTS))
from test_hcs import _plate_xml, _write                       # noqa: E402

from ome_zarr_pyramid import PyramidCollection                 # noqa: E402

WELLS = {0: (3, 5, "C5"), 1: (2, 2, "B2"), 2: (2, 2, "B2"), 3: (4, 7, "D7")}
CALLS = [dict(well=1), dict(well="B2"), dict(well=["B2", "D7"]), dict(row="C"), dict(row=2),
         dict(column="5"), dict(column=4), dict(row=["B", "C"]), dict(row=range(1, 3), column="2"),
         dict(row="H"), dict(field=[0, 1]), dict(well="B2", field=range(1, 5)),
         dict(plate=0, acquisition=0, well=0), dict(view=0)]


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("plate") / "plate.czi")
    _write(path, WELLS, _plate_xml(WELLS))
    f = micro_reader.open(path, plates=True)
    yield f, PyramidCollection.from_file(f)
    f.close()


@pytest.mark.parametrize("call", CALLS, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()))
def test_the_same_images(plate, call):
    f, coll = plate
    micro = sorted(tuple(sorted(dict(image.index).items())) for image in f.select(**call))
    ozp = sorted(tuple(sorted(dict(key).items())) for key in coll.select(**call))
    assert ozp == micro


@pytest.mark.parametrize("call, error", [(dict(well=0, row=0), TypeError),
                                         (dict(colour=1), KeyError),
                                         (dict(scene=0), KeyError)])
def test_the_same_refusals(plate, call, error):
    f, coll = plate
    with pytest.raises(error):
        f.select(**call)
    with pytest.raises(error):
        coll.select(**call)
    well = f.plates[0].wells[0]
    with pytest.raises(TypeError):
        f.select(well=well)
    with pytest.raises(TypeError):
        coll.select(well=well)                         # micro-reader's Well too
