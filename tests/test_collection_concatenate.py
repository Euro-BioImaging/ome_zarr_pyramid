"""PyramidCollection.concatenate: fold image indices (views, illuminations, ...) into an
axis, one joined pyramid per combination of the other indices."""
from types import SimpleNamespace

import numpy as np
import pytest

from ome_zarr_pyramid import PyramidCollection
from test_collection import FakeImage, Q


def _images(views=2, illuminations=2, scenes=1, channels=2, *, shift_view=None):
    """Image-likes indexed by scene / view / illumination; each pixel encodes its source."""
    out, data = [], {}
    for s in range(scenes):
        for v in range(views):
            for i in range(illuminations):
                a = np.full((channels, 4, 8, 8), 100 * s + 10 * v + i, "u2")
                a += np.arange(channels, dtype="u2")[:, None, None, None]
                pos = {"y": Q(5.0 if v == shift_view else 0.0, "micrometer")}
                meta = SimpleNamespace(
                    pixel_sizes={"z": Q(2.0, "micrometer"), "y": Q(0.5, "micrometer"),
                                 "x": Q(0.5, "micrometer")},
                    position=pos, index_values={},
                    channels=[SimpleNamespace(name=f"Cam{c}", color=None, emission_wavelength=None,
                                              excitation_wavelength=None) for c in range(channels)])
                index = {"scene": s, "view": v, "illumination": i}
                out.append(FakeImage(a, "czyx", meta, index))
                data[(s, v, i)] = a
    return SimpleNamespace(images=out, path="f.czi"), data


def test_one_index_folded_gives_one_entry_per_other_value():
    file, data = _images()
    joined = PyramidCollection.from_file(file).concatenate("view")
    assert sorted(joined) == [(("scene", 0), ("illumination", 0)),
                              (("scene", 0), ("illumination", 1))]
    pyr = joined[{"scene": 0, "illumination": 1}]
    got = np.asarray(pyr.layers["0"][...])
    np.testing.assert_array_equal(got, np.concatenate([data[(0, 0, 1)], data[(0, 1, 1)]]))
    assert pyr.joined_index == [{"view": 0}, {"view": 0}, {"view": 1}, {"view": 1}]
    assert pyr.source_index == {"scene": 0, "illumination": 1}
    assert [c["label"] for c in pyr.meta.metadata["omero"]["channels"]] == ["Cam0", "Cam1"] * 2


def test_two_indices_folded_first_one_outermost():
    file, data = _images(scenes=2)
    joined = PyramidCollection.from_file(file).concatenate(("view", "illumination"))
    assert sorted(joined) == [(("scene", 0),), (("scene", 1),)]
    pyr = joined[{"scene": 1}]
    order = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]           # views outer, as eubi
    np.testing.assert_array_equal(np.asarray(pyr.layers["0"][...]),
                                  np.concatenate([data[k] for k in order]))
    assert pyr.joined_index[::2] == [{"view": v, "illumination": i} for _s, v, i in order]


def test_entries_that_do_not_agree_are_refused():
    file, _ = _images(shift_view=1)                     # view 1 sits elsewhere
    with pytest.raises(ValueError, match="only the position along 'c'"):
        PyramidCollection.from_file(file).concatenate("view")


def test_unknown_or_partial_indices_are_refused():
    file, _ = _images()
    coll = PyramidCollection.from_file(file)
    with pytest.raises(ValueError, match="no entry is indexed by"):
        coll.concatenate("tile")
    partial = PyramidCollection({**{k: coll[k] for k in list(coll)[:2]},
                                 (("scene", 1),): coll[list(coll)[0]]})
    with pytest.raises(ValueError, match="some entries"):
        partial.concatenate("view")
