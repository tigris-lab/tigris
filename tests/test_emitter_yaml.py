"""Tests for the YAML plan emitter."""

import yaml

from tigris.loaders import load_model
from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_temporal import partition_temporal
from tigris import SCHEMA_VERSION
from tigris.emitters.yaml import emit_yaml, emit_yaml_str


def _full_pipeline(path, budget=0):
    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    if budget > 0:
        ag = partition_temporal(ag, budget)
    return ag


def test_yaml_roundtrip(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    text = emit_yaml_str(ag)
    plan = yaml.safe_load(text)

    assert plan["version"] == SCHEMA_VERSION
    assert plan["model"]["name"] == "linear_3op"
    assert len(plan["ops"]) == 3
    assert len(plan["inputs"]) == 1
    assert plan["inputs"][0]["name"] == "input"


def test_yaml_has_stages(diamond_path):
    ag = _full_pipeline(diamond_path, budget=512)
    plan = yaml.safe_load(emit_yaml_str(ag))

    assert "stages" in plan
    assert len(plan["stages"]) >= 2
    assert plan["memory"]["budget"] == 512


def test_yaml_file_output(linear_3op_path, tmp_path):
    ag = _full_pipeline(linear_3op_path)
    out = tmp_path / "test.plan.yaml"
    emit_yaml(ag, out)

    assert out.exists()
    plan = yaml.safe_load(out.read_text())
    assert plan["model"]["name"] == "linear_3op"


def test_yaml_tensor_dtypes(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    plan = yaml.safe_load(emit_yaml_str(ag))

    assert plan["inputs"][0]["dtype"] == "float32"
    assert plan["inputs"][0]["shape"] == [1, 64]
    assert plan["inputs"][0]["size_bytes"] == 1 * 64 * 4
