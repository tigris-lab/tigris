"""Tests for QDQ folding in the normalize pass."""

import numpy as np

from tigris.loaders.onnx.loader import load_model as load_raw
from tigris.loaders.onnx.normalize import normalize, _fold_qdq
from tigris.graph.ir import QuantParam
from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.defs import NO_QUANT_PARAM


def _load_and_fold(path):
    """Load raw ONNX (no normalize) then apply full normalize."""
    ag = load_raw(path)
    return normalize(ag)


def test_qdq_fold_removes_ql_dql_ops(qdq_conv_path):
    """All QuantizeLinear and DequantizeLinear ops should be removed."""
    ag = load_raw(qdq_conv_path)
    qdq_types = {"QuantizeLinear", "DequantizeLinear"}
    assert any(op.op_type in qdq_types for op in ag.ops), "fixture should have Q/DQ ops"

    ag = _fold_qdq(ag)
    remaining = [op.op_type for op in ag.ops]
    assert not any(t in qdq_types for t in remaining), f"Q/DQ ops remain: {remaining}"


def test_qdq_fold_preserves_compute_ops(qdq_conv_path):
    """Conv should survive folding; Relu is absorbed as fused activation."""
    ag = _load_and_fold(qdq_conv_path)
    op_types = [op.op_type for op in ag.ops]
    assert "Conv" in op_types
    assert len(ag.ops) == 1  # Conv with fused Relu
    assert ag.ops[0].attrs.get("fused_activation") == "Relu"


def test_qdq_fold_sets_is_quantized(qdq_conv_path):
    """ag.is_quantized should be True after folding."""
    ag = _load_and_fold(qdq_conv_path)
    assert ag.is_quantized is True


def test_qdq_fold_weight_int8(qdq_conv_path):
    """After folding, weight data should be int8."""
    ag = _load_and_fold(qdq_conv_path)

    # Find the conv op's weight input
    conv_op = next(op for op in ag.ops if op.op_type == "Conv")
    w_name = conv_op.inputs[1]

    assert w_name in ag.weight_data
    assert ag.weight_data[w_name].dtype == np.int8


def test_qdq_fold_weight_quant_param(qdq_conv_path):
    """Weight tensor should have a QuantParam attached."""
    ag = _load_and_fold(qdq_conv_path)

    conv_op = next(op for op in ag.ops if op.op_type == "Conv")
    w_name = conv_op.inputs[1]
    w_info = ag.tensors[w_name]

    assert w_info.quant is not None
    assert isinstance(w_info.quant, QuantParam)
    assert w_info.quant.scale.shape[0] == 2  # per-channel, 2 output channels


def test_qdq_fold_activation_quant_param(qdq_conv_path):
    """Input activation should have a QuantParam from the input Q/DQ pair."""
    ag = _load_and_fold(qdq_conv_path)

    # The model input tensor should have quant param
    input_name = ag.model_inputs[0]
    input_info = ag.tensors[input_name]
    assert input_info.quant is not None
    assert input_info.quant.scale.size == 1  # per-tensor
    np.testing.assert_allclose(input_info.quant.scale, [0.05])


def test_qdq_fold_output_quant_param(qdq_conv_path):
    """Output of fused Conv+Relu should have quant param from the output Q/DQ."""
    ag = _load_and_fold(qdq_conv_path)

    # Relu is fused into Conv; the Conv output inherits the Relu output's quant param
    conv_op = next(op for op in ag.ops if op.op_type == "Conv")
    conv_out = conv_op.outputs[0]
    conv_info = ag.tensors[conv_out]
    assert conv_info.quant is not None
    np.testing.assert_allclose(conv_info.quant.scale, [0.1])


def test_qdq_fold_no_effect_on_float_model(linear_3op_path):
    """Float model should be unchanged by QDQ folding."""
    ag = load_raw(linear_3op_path)
    orig_ops = len(ag.ops)
    ag = normalize(ag)
    assert ag.is_quantized is False
    assert len(ag.ops) == orig_ops


def test_qdq_fold_scale_zp_cleaned(qdq_conv_path):
    """Scale/zero_point initializers should be removed from weight_data."""
    ag = _load_and_fold(qdq_conv_path)

    # Scale/zp tensor names should not remain in weight_data
    for name in ["inp_scale", "inp_zp", "w_scale", "w_zp", "out_scale", "out_zp"]:
        assert name not in ag.weight_data, f"{name} still in weight_data"


# Binary round-trip tests


def _full_pipeline_qdq(path):
    """Load QDQ model through full pipeline (normalize, lifetimes, memory)."""
    ag = load_raw(path)
    ag = normalize(ag)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    return ag


def test_qdq_binary_roundtrip(qdq_conv_path):
    """Quantized model emits valid binary with quant params section."""
    ag = _full_pipeline_qdq(qdq_conv_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert plan["model_name"] == "qdq_conv"
    assert plan["num_ops"] == 1  # Conv with fused Relu
    assert len(plan["quant_params"]) > 0


def test_qdq_binary_quant_param_content(qdq_conv_path):
    """Quant params have correct scale and zero_point values."""
    ag = _full_pipeline_qdq(qdq_conv_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    # Find tensor with quant param
    tensors_with_qp = [t for t in plan["tensors"] if t["quant_param_idx"] != NO_QUANT_PARAM]
    assert len(tensors_with_qp) > 0

    # Check that quant param indices reference valid entries
    for t in tensors_with_qp:
        qp_idx = t["quant_param_idx"]
        assert qp_idx < len(plan["quant_params"])


def test_qdq_binary_weight_per_channel(qdq_conv_path):
    """Per-channel weight quant params have multipliers and shifts."""
    ag = _full_pipeline_qdq(qdq_conv_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    # The weight has per-channel quant (2 output channels)
    per_channel = [qp for qp in plan["quant_params"] if qp["num_channels"] > 1]
    assert len(per_channel) > 0

    qp = per_channel[0]
    assert qp["num_channels"] == 2
    assert "multipliers" in qp
    assert "shifts" in qp
    assert len(qp["multipliers"]) == 2
    assert len(qp["shifts"]) == 2
    # Multipliers should be positive non-zero
    assert all(m > 0 for m in qp["multipliers"])


def test_float_model_no_quant_params(linear_3op_path):
    """Float model should have no quant params in binary."""
    ag = load_raw(linear_3op_path)
    ag = normalize(ag)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert len(plan["quant_params"]) == 0
    for t in plan["tensors"]:
        assert t["quant_param_idx"] == NO_QUANT_PARAM
