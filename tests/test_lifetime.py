"""Tests for tigris.analysis.lifetime."""

from tigris.loaders import load_model
from tigris.analysis.lifetime import compute_lifetimes


def test_model_input_birth(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)

    assert ag.lifetimes["input"].birth_step == -1


def test_model_output_death(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)

    assert ag.lifetimes["output"].death_step == len(ag.ops)


def test_constants_excluded(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)

    assert "w0" not in ag.lifetimes
    assert "w1" not in ag.lifetimes


def test_intermediate_lifetime(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)

    # t0 is produced by add0 and consumed by relu0
    lt = ag.lifetimes["t0"]
    assert lt.birth_step >= 0
    assert lt.death_step > lt.birth_step


def test_diamond_lifetimes(diamond_path):
    ag = load_model(diamond_path)
    ag = compute_lifetimes(ag)

    # Both left and right should be alive until the Add consumes them
    assert "left" in ag.lifetimes
    assert "right" in ag.lifetimes

    # input is consumed by both relu and sigmoid - death = max of those steps
    assert ag.lifetimes["input"].death_step >= 0
