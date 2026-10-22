"""The spatial model's value classes, ids, transform model and JSON (plan 3.9, 3.16, 3.23,
3.29, 5.28, 8.20): ome_zarr_pyramid.core.spatial."""
from types import SimpleNamespace

import numpy as np
import pytest

from ome_zarr_pyramid.core.spatial import (
    Axis, CoordinateSystem, ExternalRef, Transformation, TransformError, clean_id_part,
    compose, invert, matrix, scale_translation, system_from, system_from_json, system_id,
    system_to_json, transformation_from, transformation_from_json, transformation_to_json)

KEY = (("series", 0), ("tile", 3))


# -- value classes ---------------------------------------------------------------

def test_axis_infers_its_type_from_the_name_and_refuses_an_unknown_name():
    assert Axis("z").type == "space" and Axis("t").type == "time" and Axis("c").type == "channel"
    assert Axis("lambda", "spectral").type == "spectral"
    with pytest.raises(ValueError, match="give type="):
        Axis("lambda")


def test_a_system_needs_a_pattern_safe_id_and_names_itself_by_it():
    cs = CoordinateSystem("well-B2")
    assert cs.name == "well-B2" and cs.axes is None and cs.estimated is False
    with pytest.raises(ValueError, match="not pattern-safe"):
        CoordinateSystem("well B2")
    with pytest.raises(ValueError, match="repeat"):
        CoordinateSystem("stage", axes=(Axis("x"), Axis("x")))
    with pytest.raises(TypeError, match="True or False"):
        CoordinateSystem("stage", estimated="no")


def test_an_edge_end_is_a_system_id_an_image_key_an_external_ref_or_none():
    Transformation("translation", KEY, "stage", {"translation": [1, 2]})
    Transformation("identity", ExternalRef("../other.zarr", "world"), "stage")
    with pytest.raises(TypeError, match="image key"):
        Transformation("translation", {"series": 0}, "stage")


def test_ids_follow_micro_readers_rule():
    assert system_id("stage") == "stage"
    assert system_id("stage", "s3") == "stage-s3"
    assert system_id("well", "p1", "B2") == "well-p1-B2"
    assert system_id("global", "registered") == "global-registered"
    assert clean_id_part("My Plate.czi") == "My_Plate_czi"
    assert clean_id_part("a--b") == "a-_b" and clean_id_part("-B2-") == "_B2_"
    with pytest.raises(ValueError):
        clean_id_part("")


def test_micro_readers_objects_are_read_by_their_fields():
    micro_axis = SimpleNamespace(name="x", type="space", unit="micrometer", discrete=None,
                                 long_name=None, extra={})
    cs = system_from(SimpleNamespace(id="stage", name="stage", axes=(micro_axis,),
                                     estimated=False, extra={}))
    assert cs == CoordinateSystem("stage", axes=(Axis("x", "space", "micrometer"),))
    assert system_from(SimpleNamespace(id="global-registered", name=None, axes=None,
                                       estimated=True, extra={})).estimated is True
    edge = transformation_from(SimpleNamespace(type="translation", input=[["series", 0]],
                                               output="stage", params={"translation": [1, 2]},
                                               name=None))
    assert edge.input == (("series", 0),) and edge.output == "stage"


# -- the transform model ----------------------------------------------------------

def test_each_modelled_type_has_its_matrix():
    assert np.array_equal(matrix(Transformation("identity"), 2), np.eye(3))
    t = matrix(Transformation("translation", params={"translation": [5, 3]}), 2)
    assert t[:2, 2].tolist() == [5, 3]
    s = matrix(Transformation("scale", params={"scale": [2, 0.5]}), 2)
    assert np.diag(s)[:2].tolist() == [2, 0.5]
    a = matrix(Transformation("affine", params={"affine": [[0, -1, 4], [1, 0, 2]]}), 2)
    assert a[:2].tolist() == [[0, -1, 4], [1, 0, 2]]


def test_a_sequence_applies_its_first_step_first():
    seq = Transformation("sequence", params={"transformations": [
        {"type": "scale", "scale": [2, 2]}, {"type": "translation", "translation": [1, 1]}]})
    m = matrix(seq, 2)
    assert (m @ [1, 1, 1])[:2].tolist() == [3, 3]          # (1 * 2) + 1


def test_compose_and_invert_round_trip():
    a = Transformation("sequence", KEY, "stage", {"transformations": [
        {"type": "scale", "scale": [0.5, 0.25]}, {"type": "translation", "translation": [10, -4]}]})
    inv = invert(a, 2)
    assert inv.input == "stage" and inv.output == KEY
    back = compose(a, inv, ndim=2)
    assert back.type == "identity"


def test_from_matrix_gives_the_simplest_type():
    assert compose(Transformation("translation", params={"translation": [0, 0]}), ndim=2).type == "identity"
    assert compose(Transformation("scale", params={"scale": [2, 2]}), ndim=2).type == "scale"
    st = compose(Transformation("scale", params={"scale": [2, 3]}),
                 Transformation("translation", params={"translation": [1, 1]}), ndim=2)
    assert st.type == "sequence" and [s.type for s in st.params["transformations"]] == ["scale", "translation"]


def test_scale_translation_reads_the_position_edge_form():
    s, t = scale_translation(Transformation("translation", params={"translation": [5, 3]}), 2)
    assert s.tolist() == [1, 1] and t.tolist() == [5, 3]
    turned = Transformation("affine", params={"affine": [[0, -1, 0], [1, 0, 0]]})
    assert scale_translation(turned, 2) is None


def test_what_ozp_does_not_model_raises_a_transform_error():
    with pytest.raises(TransformError, match="does not model"):
        matrix(Transformation("rotation", params={"rotation": [[0, 1], [1, 0]]}), 2)
    with pytest.raises(TransformError, match="Zarr array"):
        matrix(Transformation("affine", params={"path": "coordinateTransformations/a"}), 2)
    with pytest.raises(TransformError, match="number of axes"):
        matrix(Transformation("affine", params={"affine": [[1, 0, 0, 0], [0, 1, 0, 0]]}), 2)
    with pytest.raises(TransformError, match="singular"):
        invert(Transformation("scale", params={"scale": [0, 1]}), 2)
    assert not Transformation("rotation").modelled and Transformation("affine").modelled


# -- JSON (5.28) --------------------------------------------------------------------

def test_a_system_round_trips_through_its_eubi_entry():
    cs = CoordinateSystem("well-B2", "well B2", (Axis("y", unit="micrometer"),
                                                 Axis("x", unit="micrometer")), extra={"k": 1})
    obj = system_to_json(cs)
    assert obj["name"] == "well-B2" and obj["eubi:name"] == "well B2"
    assert obj["eubi:estimated"] is False and obj["axes"][0] == {"name": "y", "type": "space",
                                                                "unit": "micrometer"}
    assert system_from_json(obj) == cs
    assert "axes" not in system_to_json(cs, axes=False)
    # RFC 8: the id as "id", a readable name as "name"
    assert system_from_json({"id": "stage", "name": "the stage"}).name == "the stage"


def test_an_edge_round_trips_with_an_endpoint_mapping():
    paths = {KEY: {"name": "physical", "path": "0/3"}}
    keys = {"0/3": KEY}
    edge = Transformation("sequence", KEY, "stage", {"transformations": [
        Transformation("scale", params={"scale": [1, 1]}),
        Transformation("translation", params={"translation": [5, 3]})]})
    obj = transformation_to_json(edge, endpoint=lambda v: paths.get(v, {"name": v} if v else None))
    assert obj["input"] == {"name": "physical", "path": "0/3"} and obj["output"] == {"name": "stage"}
    assert [s["type"] for s in obj["transformations"]] == ["scale", "translation"]
    back = transformation_from_json(obj, endpoint=lambda o: None if o is None
                                    else keys.get(o.get("path"), o.get("name")))
    assert back.input == KEY and back.output == "stage"
    assert np.allclose(matrix(back, 2), matrix(edge, 2))
    # an end outside the set is kept as an ExternalRef
    ext = transformation_from_json({"type": "identity", "input": {"name": "w", "path": "../x.zarr"},
                                    "output": {"name": "stage"}})
    assert ext.input == ExternalRef("../x.zarr", "w") and ext.output == "stage"
