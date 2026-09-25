from __future__ import annotations

import numpy as np
import torch

from context_selective_retrieval_v3.geometry import (
    conjugate_gradient,
    fr_metrics,
    gate_passes,
    normalize_target_gain,
    summarize_exact,
    target_row_null_direction,
    vector_dot,
)


def test_structured_cg_matches_dense_solution() -> None:
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    rhs = [torch.tensor([1.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64)]

    def operator(v: list[torch.Tensor]) -> list[torch.Tensor]:
        flat = torch.cat(v)
        out = matrix @ flat
        return [out[:1], out[1:]]

    actual, history = conjugate_gradient(
        operator, rhs, maximum_iterations=5, relative_tolerance=1e-12
    )
    expected = torch.linalg.solve(matrix, torch.cat(rhs))
    assert torch.allclose(torch.cat(actual), expected, atol=1e-10)
    assert history[-1]["relative_residual"] < 1e-10


def test_target_gain_normalization() -> None:
    sensitivity = [torch.tensor([2.0, -1.0])]
    direction, raw = normalize_target_gain([torch.tensor([3.0, 1.0])], sensitivity)
    assert raw == 5.0
    assert np.isclose(float(vector_dot(direction, sensitivity)), 1.0)


def test_exact_geometry_and_gate() -> None:
    p = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
    q = np.array([[0.3, 0.25, 0.45]], dtype=np.float64)
    metric = fr_metrics(p, q, 0)
    assert np.allclose(metric["L"] ** 2, metric["R"] ** 2 + metric["N"] ** 2)
    assert metric["G"][0] >= -1e-12
    control = {"short": p, "long": np.repeat(p, 3, axis=0), "clean": np.repeat(p, 2, axis=0)}
    modified = {"short": q, "long": np.repeat(p, 3, axis=0), "clean": np.repeat(p, 2, axis=0)}
    summary = summarize_exact(control, modified, 0)
    assert summary["selectivity_margin"] == summary["short_B"]
    assert gate_passes(summary, {"short_probability": 0.25, "selectivity_margin": 0.0, "clean_rms_max": 0.0})


def test_identity_is_zero() -> None:
    p = np.array([[0.1, 0.2, 0.7]], dtype=np.float64)
    values = fr_metrics(p, p, 2)
    for key in ("L", "R", "N", "B", "G"):
        assert np.max(np.abs(values[key])) < 1e-12


def test_target_row_oracle_has_unit_gain_and_exact_protection() -> None:
    protected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    short = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)
    direction, sensitivity, rank, raw_gain, residual = target_row_null_direction(
        short, protected, 0.25
    )
    assert rank == 2
    assert raw_gain > 0.0
    assert residual < 1e-12
    assert torch.allclose(protected @ direction, torch.zeros(2, dtype=torch.float64), atol=1e-12)
    assert np.isclose(float(torch.dot(sensitivity, direction)), 1.0)
