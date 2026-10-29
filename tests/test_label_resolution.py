"""Labels at another resolution than their image follow it in physical space (decision C-18):
Fractal's cardiomyocyte plates store a label at a quarter of the image's resolution, covering
the same area, which NGFF allows. The rule compared pixel counts and refused them."""
import numpy as np
import pytest

from ome_zarr_pyramid import LabelMismatchError, Pyramid


def _image(size=64, pixel=0.25):
    data = np.arange(2 * size * size, dtype="u2").reshape(2, 1, size, size)
    return Pyramid().from_arrays([data], axis_order="czyx", scales=[[1, 1, pixel, pixel]])


def _label(size=16, pixel=1.0):
    values = (np.indices((1, size, size)).sum(axis=0) % 3 + 1).astype("u4")
    return Pyramid().from_label_arrays([values], axis_order="zyx", scales=[[1, pixel, pixel]])


def _levels(pyr):
    return [(tuple(pyr.layers[p].shape), [float(v) for v in pyr.meta.get_scale(p)])
            for p in pyr.meta.resolution_paths]


def _coarse():
    """A 64 x 64 image at 0.25 with a 16 x 16 label at 1.0: the same area, 4x coarser."""
    return _image().add_image_label(_label(), "nuclei")


@pytest.mark.parametrize("defer", [True, False])
def test_a_coarser_label_downscales_with_the_image_factors(tmp_path, defer):
    out = _coarse().downscale(n_layers=3, downscale_method="mean", defer=defer)
    assert [s for s, _ in _levels(out)] == [(2, 1, 64, 64), (2, 1, 32, 32), (2, 1, 16, 16)]
    label = out.labels["nuclei"]
    # its own sizes, the image's factors: the same area on every level
    assert _levels(label) == [((1, 16, 16), [1.0, 1.0, 1.0]), ((1, 8, 8), [1.0, 2.0, 2.0]),
                              ((1, 4, 4), [1.0, 4.0, 4.0])]
    path = str(tmp_path / "img.ome.zarr")
    out.write(path)
    back = Pyramid.from_zarr(path).labels["nuclei"]
    assert _levels(back) == _levels(label)
    for p, q in zip(label.meta.resolution_paths, back.meta.resolution_paths):
        values = np.asarray(back.layers[q])
        np.testing.assert_array_equal(np.asarray(label.layers[p]), values)
        assert set(np.unique(values)) <= {1, 2, 3}                 # picked, never averaged


def test_isel_maps_through_the_resolution_ratio():
    pyr = _coarse()
    top = pyr.isel(y=slice(0, 32)).labels["nuclei"]                # 32 image px = 8 label px
    assert top.layers["0"].shape == (1, 8, 16)
    np.testing.assert_array_equal(np.asarray(top.layers["0"]),
                                  np.asarray(_label().layers["0"])[:, :8])
    row = pyr.isel(y=5).labels["nuclei"]                           # image row 5: label row 1
    assert row.axes == "zx"
    np.testing.assert_array_equal(np.asarray(row.layers["0"]), np.asarray(_label().layers["0"])[:, 1])
    assert pyr.isel(y=slice(48, None)).labels["nuclei"].layers["0"].shape == (1, 4, 16)
    with pytest.raises(LabelMismatchError, match=r"slice\(2, 10, None\) along x does not land "
                                                  r"on whole label pixels"):
        pyr.isel(x=slice(2, 10))


def test_select_levels_of_a_coarser_label():
    pyr = _coarse().downscale(n_layers=3, defer=False)
    label = pyr.select_levels(1, 2).labels["nuclei"]
    assert [s for s, _ in _levels(label)] == [(1, 8, 8), (1, 4, 4)]


def test_a_finer_label_follows_slices_but_not_single_indices():
    pyr = _image(size=32, pixel=0.5).add_image_label(_label(size=64, pixel=0.25), "fine")
    assert pyr.isel(y=slice(0, 16)).labels["fine"].layers["0"].shape == (1, 32, 64)
    with pytest.raises(LabelMismatchError, match="index 3 along y covers 2 label pixels"):
        pyr.isel(y=3)


def test_a_label_covering_another_area_still_refuses():
    pyr = _image().add_image_label(_label(size=15), "short")         # 15 at 1.0, not 16
    with pytest.raises(LabelMismatchError, match=r"\(15 along y, the image 64\)"):
        pyr.downscale(n_layers=2)
    pyr = _image().add_image_label(_label(size=24, pixel=0.75), "odd")  # 0.75 / 0.25: 3x, 18 != 16
    with pytest.raises(LabelMismatchError, match="24 along y"):
        pyr.isel(y=slice(0, 12))
