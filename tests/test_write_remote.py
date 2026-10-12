"""Remote (object-store) writes through the dyna engine, against an S3 emulator.

ozp writes `s3://bucket/key` with obstore-style `storage_options` - the same names
dyna takes (`endpoint`, `region`, `virtual_hosted_style_request`, `client_options`);
credentials come from the standard AWS environment variables. The level arrays go
through dyna, the group / NGFF metadata through zarr's obstore-backed ObjectStore.

Before, `s3://` crashed (`os.makedirs('s3:')`) and only the legacy
`https://host/bucket/key` form (s3fs options, the 'sync' engine removed in 0.1.0) was
writable; a deferred downscale could not be written remotely at all. That form is now
translated to `s3://bucket/key` with the host as the endpoint.

Runs against OZP_TEST_S3_ENDPOINT if set, else an in-process moto server, else skips.
"""
import os
import socket
import urllib.request
import uuid

import numpy as np
import pytest
import zarr

from ome_zarr_pyramid.core.io import IO
from ome_zarr_pyramid.core.pyramid import Pyramid

pytest.importorskip("obstore")

SHAPE, CHUNKS, AXES = (2, 24, 40, 40), (1, 8, 16, 16), "czyx"


@pytest.fixture(scope="module")
def s3_endpoint():
    keys = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_DEFAULT_REGION")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update(AWS_ACCESS_KEY_ID="test", AWS_SECRET_ACCESS_KEY="test",
                      AWS_REGION="us-east-1", AWS_DEFAULT_REGION="us-east-1")
    server = None
    endpoint = os.environ.get("OZP_TEST_S3_ENDPOINT")
    if not endpoint:
        moto_server = pytest.importorskip("moto.server")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=port)
        server.start()
        endpoint = f"http://127.0.0.1:{port}"
    yield endpoint
    if server is not None:
        server.stop()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def bucket(s3_endpoint):
    name = f"ozp-{uuid.uuid4().hex[:12]}"
    urllib.request.urlopen(urllib.request.Request(f"{s3_endpoint}/{name}", method="PUT")).read()
    opts = {"endpoint": s3_endpoint, "region": "us-east-1",
            "virtual_hosted_style_request": False, "client_options": {"allow_http": True}}
    return name, opts


@pytest.fixture
def data():
    return np.random.default_rng(0).integers(0, 60000, size=SHAPE, dtype="uint16")


def _pyr(data, version="0.5"):
    arr = zarr.create_array(store=zarr.storage.MemoryStore(), shape=SHAPE, chunks=CHUNKS,
                            dtype=data.dtype, zarr_format=3 if version == "0.5" else 2)
    arr[...] = data
    return Pyramid().from_arrays([arr], axis_order=AXES, scales=[[1.0, 2.0, 0.5, 0.5]],
                                 version=version)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_s3_write_then_read_back(bucket, data, version):
    name, opts = bucket
    url = f"s3://{name}/img.zarr"
    IO().write_pyramid(_pyr(data, version), url, overwrite=True, storage_options=opts)
    back = IO().read_pyramid(url, storage_options=opts)
    assert back.axes == AXES
    assert back.meta.get_base_scale() == [1.0, 2.0, 0.5, 0.5]
    assert back.meta.version == version
    level = back.layers["0"]
    np.testing.assert_array_equal(np.asarray(level[...]), data)
    assert tuple(level.chunks) == CHUNKS
    if version == "0.5":
        assert level.metadata.dimension_names == tuple(AXES)


def test_s3_deferred_downscale_cascades_remotely(bucket, data):
    name, opts = bucket
    url = f"s3://{name}/pyr.zarr"
    IO().write_pyramid(_pyr(data).downscale(n_layers=3, defer=True), url, overwrite=True,
                       storage_options=opts)
    back = IO().read_pyramid(url, storage_options=opts)
    assert back.nlayers == 3
    for i in range(3):
        s = 2 ** i
        np.testing.assert_array_equal(np.asarray(back.layers[str(i)][...]),
                                      data[:, ::s, ::s, ::s])


def test_s3_overwrite_rule_and_shards(bucket, data):
    name, opts = bucket
    url = f"s3://{name}/img.zarr"
    IO().write_pyramid(_pyr(data), url, overwrite=True, storage_options=opts,
                       shard_coefficients=(1, 3, 2, 2))
    with pytest.raises(Exception, match="already holds"):
        IO().write_pyramid(_pyr(data), url, overwrite=False, storage_options=opts)
    IO().write_pyramid(_pyr(data), url, overwrite=True, storage_options=opts)   # replaced
    back = IO().read_pyramid(url, storage_options=opts)
    assert back.layers["0"].shards is None          # the overwrite really replaced it


def test_a_removed_engine_is_refused_remotely_too(bucket, data):
    name, opts = bucket
    with pytest.raises(ValueError, match="removed in ome_zarr_pyramid 0.1.0"):
        IO().write_pyramid(_pyr(data), f"s3://{name}/x.zarr", backend="sync",
                           storage_options=opts)


def test_the_legacy_https_form_still_writes(bucket, s3_endpoint, data):
    # https://host/bucket/key with the old writer's s3fs option names; the emulator is
    # plain http, so the endpoint comes from client_kwargs (as s3fs allowed). Credentials
    # from the environment: dyna's TensorStore backend refuses explicit ones (key/secret
    # map to obstore names, which io_backend='zarrista' takes)
    name, opts = bucket
    s3fs_opts = {"client_kwargs": {"endpoint_url": s3_endpoint, "region_name": "us-east-1"},
                 "client_options": {"allow_http": True}}
    IO().write_pyramid(_pyr(data).downscale(n_layers=2, defer=True),
                       f"https://s3.example.org/{name}/legacy.zarr", overwrite=True,
                       storage_options=s3fs_opts)
    back = IO().read_pyramid(f"s3://{name}/legacy.zarr", storage_options=opts)
    assert back.nlayers == 2
    np.testing.assert_array_equal(np.asarray(back.layers["0"][...]), data)


def test_s3_attached_labels(bucket, data):
    name, opts = bucket
    url = f"s3://{name}/img.zarr"
    lab = np.zeros(SHAPE[1:], dtype="uint32")
    lab[4:12, 8:20, 8:20] = 7
    labels = Pyramid().from_label_arrays([lab], axis_order=AXES[1:], version="0.5")
    IO().write_pyramid(_pyr(data).add_image_label(labels, name="seg"), url, overwrite=True,
                       storage_options=opts)
    seg = IO().read_pyramid(f"{url}/labels/seg", storage_options=opts)
    np.testing.assert_array_equal(np.asarray(seg.layers["0"][...]), lab)
    assert seg.meta.image_label is not None
    # the parent labels/ group registers the name
    import obstore
    from zarr.storage import ObjectStore
    grp = zarr.open_group(ObjectStore(obstore.store.from_url(f"{url}/labels", **opts)), mode="r")
    assert "seg" in dict(grp.attrs)["ome"]["labels"]
    # read_labels takes the URL too (it built the path with Path, mangling s3://)
    by_name = IO().read_labels(url, name="seg", storage_options=opts)
    np.testing.assert_array_equal(np.asarray(by_name.layers["0"][...]), lab)


def test_s3_dask_pyramid_is_pumped(bucket, data):
    da = pytest.importorskip("dask.array")
    name, opts = bucket
    url = f"s3://{name}/dask.zarr"
    pyr = Pyramid().from_arrays([da.from_array(data, chunks=CHUNKS) + 1], axis_order=AXES,
                                version="0.5")
    IO().write_pyramid(pyr, url, overwrite=True, storage_options=opts)
    back = IO().read_pyramid(url, storage_options=opts)
    np.testing.assert_array_equal(np.asarray(back.layers["0"][...]), data + 1)
