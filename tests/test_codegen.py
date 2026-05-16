"""Tests for C harness generation and XIP plan metadata."""

from click.testing import CliRunner

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_spatial import partition_spatial
from tigris.analysis.partition_temporal import partition_temporal
from tigris.cli import cli
from tigris.emitters.binary.defs import FLAG_XIP
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.emitters.codegen import generate_c
from tigris.loaders import load_model


def _full_pipeline(path, budget=0):
    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    if budget > 0:
        ag = partition_temporal(ag, budget)
        ag = partition_spatial(ag)
    return ag


def test_cli_exposes_codegen_and_xip():
    runner = CliRunner()

    top = runner.invoke(cli, ["--help"])
    assert top.exit_code == 0
    assert "codegen" in top.output

    compile_help = runner.invoke(cli, ["compile", "--help"])
    assert compile_help.exit_code == 0
    assert "--xip" in compile_help.output


def test_xip_sets_plan_flag(linear_3op_path):
    ag = _full_pipeline(linear_3op_path, budget=4096)
    plan = read_binary_plan(emit_binary_bytes(ag, xip=True))

    assert plan["flags"] & FLAG_XIP


def test_codegen_reports_xip_and_loads_plan_at_runtime(linear_3op_path):
    ag = _full_pipeline(linear_3op_path, budget=4096)
    data = emit_binary_bytes(ag, xip=True)

    source = generate_c(data, "reference")

    assert "XIP:     yes" in source
    assert "static uint8_t *load_file" in source
    assert "tigris_plan_load(plan_buf, plan_len, &plan)" in source
    assert "tigris_run(&plan, &mem, tigris_dispatch_kernel, NULL, &stats)" in source


def test_quantized_esp_codegen_has_valid_includes(qdq_conv_path):
    ag = _full_pipeline(qdq_conv_path, budget=4096)
    source = generate_c(emit_binary_bytes(ag), "esp-nn")

    assert '#include "tigris_kernels_esp_nn.h"' in source
    assert '#include "tigris_kernels_s8.h"' in source
    assert 'tigris_kernels_s8.h"' in source
    assert 'tigris_kernels_s8.h\\"' not in source
