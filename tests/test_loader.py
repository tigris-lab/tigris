"""Tests for tigris.loaders."""

from tigris.loaders import load_model


def test_linear_load(linear_3op_path):
    ag = load_model(linear_3op_path)

    assert ag.model_name == "linear_3op"
    assert len(ag.ops) == 3
    assert ag.model_inputs == ["input"]
    assert ag.model_outputs == ["output"]


def test_linear_topo_order(linear_3op_path):
    ag = load_model(linear_3op_path)

    # Steps should be 0, 1, 2 in a valid topological order
    op_types = [op.op_type for op in ag.ops]
    # add0 must come before relu0, relu0 before add1
    assert op_types.index("Relu") > op_types.index("Add")


def test_diamond_load(diamond_path):
    ag = load_model(diamond_path)

    assert len(ag.ops) == 3
    assert "input" in ag.model_inputs
    assert "output" in ag.model_outputs


def test_tensors_have_shapes(linear_3op_path):
    ag = load_model(linear_3op_path)

    info = ag.tensors["input"]
    assert info.shape == (1, 64)
    assert info.size_bytes == 1 * 64 * 4  # float32


def test_constants_marked(linear_3op_path):
    ag = load_model(linear_3op_path)

    assert ag.tensors["w0"].is_constant is True
    assert ag.tensors["w1"].is_constant is True
    assert ag.tensors["input"].is_constant is False
