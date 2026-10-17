"""Codecs when a pyramid is written in another zarr format (Pyramid.to_version) or with
a codec the writer does not know: kept when the output format has it; else blosc-lz4
with a CodecChangedWarning naming the old codec; a codec asked for explicitly is never
replaced. Pixels are the same in every case."""
import warnings

import numcodecs
import numpy as np
import pytest
import zarr
from zarr import codecs as zc

from ome_zarr_pyramid import IO, Pyramid
from ome_zarr_pyramid.core.io import CodecChangedWarning

DATA = (np.arange(4 * 32 * 32) % 251).astype("u2").reshape(4, 32, 32)


def _source(tmp_path, zarr_format, compressors):
    path = str(tmp_path / "src.zarr")
    arr = zarr.create_array(path, shape=DATA.shape, chunks=(1, 16, 16), dtype=DATA.dtype,
                            zarr_format=zarr_format, compressors=compressors)
    arr[...] = DATA
    return Pyramid().from_arrays([zarr.open_array(path, mode="r")], axis_order="zyx",
                                 scales=[[1, 1, 1]], version="0.4" if zarr_format == 2 else "0.5")


def _written(tmp_path, pyr, **kw):
    out = str(tmp_path / "out.zarr")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        IO().write_pyramid(pyr, out, **kw)
    arr = zarr.open_group(out, mode="r")["0"]
    np.testing.assert_array_equal(arr[...], DATA)
    if arr.metadata.zarr_format == 2:
        comp = arr.metadata.compressor
        codec = comp.get_config()["id"] if comp is not None else None
        cname = comp.get_config().get("cname") if comp is not None else None
    else:
        names = [c.to_dict() for c in arr.metadata.codecs if c.to_dict()["name"] != "bytes"]
        codec = names[0]["name"] if names else None
        cname = names[0].get("configuration", {}).get("cname") if names else None
    changed = [str(w.message) for w in caught if issubclass(w.category, CodecChangedWarning)]
    return codec, cname, changed


V2_TO_V3 = [  # (source codec, codec written in 0.5, changed?)
    (numcodecs.Blosc(cname="zstd"), "blosc", False),
    (numcodecs.Zstd(level=3), "zstd", False),
    (numcodecs.GZip(level=5), "gzip", False),
    (numcodecs.BZ2(level=5), "blosc", True),
    (numcodecs.LZ4(), "blosc", True),
    (numcodecs.Zlib(level=5), "blosc", True),
    (numcodecs.LZMA(), "blosc", True),
    (None, None, False),
]


@pytest.mark.parametrize("source, written, changed", V2_TO_V3,
                         ids=lambda v: getattr(v, "codec_id", str(v)))
def test_04_to_05(tmp_path, source, written, changed):
    pyr = _source(tmp_path, 2, source)
    codec, cname, warned = _written(tmp_path, pyr.to_version("0.5"))
    assert codec == written
    if changed:
        assert cname == "lz4" and len(warned) == 1
        assert source.codec_id in warned[0] and "blosc-lz4" in warned[0]
    else:
        assert warned == []


def _v3_codecs():
    out = [([zc.BloscCodec(cname="zstd", typesize=2)], "blosc", False),
           ([zc.ZstdCodec(level=3)], "zstd", False),
           ([zc.GzipCodec(level=5)], "gzip", False),
           ([], None, False)]
    try:
        from numcodecs import zarr3
        out += [([zarr3.LZ4()], "blosc", True), ([zarr3.Zlib(level=5)], "blosc", True)]
    except ImportError:
        pass
    return out


@pytest.mark.parametrize("source, written, changed", _v3_codecs(),
                         ids=lambda v: str(v)[:30])
def test_05_to_04(tmp_path, source, written, changed):
    pyr = _source(tmp_path, 3, source)
    codec, cname, warned = _written(tmp_path, pyr.to_version("0.4"))
    assert codec == written and (len(warned) == 1) == changed


def test_an_unknown_codec_is_reported_in_the_same_format_too(tmp_path):
    codec, cname, warned = _written(tmp_path, _source(tmp_path, 2, numcodecs.Zlib(level=5)))
    assert (codec, cname) == ("blosc", "lz4") and "'zlib' cannot be written" in warned[0]


def test_a_codec_asked_for_is_never_replaced(tmp_path):
    pyr = _source(tmp_path, 3, [zc.ZstdCodec(level=3)])
    with pytest.raises(ValueError, match="(?i)bz2"):
        IO().write_pyramid(pyr, str(tmp_path / "bad.zarr"), compressor="bz2")
    codec, _, warned = _written(tmp_path, pyr.to_version("0.4"), compressor="bz2")
    assert codec == "bz2" and warned == []


def test_an_uncompressed_05_pyramid_reads_back(tmp_path):
    pyr = Pyramid().from_arrays([DATA], axis_order="zyx", scales=[[1, 1, 1]], version="0.5")
    out = str(tmp_path / "none.zarr")
    IO().write_pyramid(pyr, out, compressor="none")
    back = IO().read_pyramid(out)
    assert back.compressor is None
    np.testing.assert_array_equal(np.asarray(back.layers["0"]), DATA)
    # and stays uncompressed when written again
    codec, _, warned = _written(tmp_path, back)
    assert codec is None and warned == []
