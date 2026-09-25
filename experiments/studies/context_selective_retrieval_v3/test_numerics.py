from __future__ import annotations

import torch

from context_selective_retrieval_v3.numerics import (
    METHOD_ORDER,
    certification_thresholds,
    direction_metrics,
    qualify,
    rank_sensitivity,
    solve_direction,
)


def _problem() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(90210)
    protected = torch.randn(8, 16, generator=generator, dtype=torch.float64)
    null = torch.linalg.svd(protected, full_matrices=True).Vh[8:].T[:, 0]
    short = protected.T @ torch.randn(8, generator=generator, dtype=torch.float64) + null
    return short, protected


def test_solvers_enforce_protection_and_unit_gain() -> None:
    short, protected = _problem()
    for method in METHOD_ORDER:
        solution = solve_direction(method, short, protected, 0.2)
        assert solution.rank == 8
        assert abs(float(torch.dot(solution.sensitivity, solution.direction)) - 1.0) < 1e-11
        assert float(torch.linalg.vector_norm(protected @ solution.direction)) < 1e-11


def test_relative_residual_is_scale_normalized() -> None:
    short, protected = _problem()
    solution = solve_direction("svd", short, protected, 0.2)
    base = torch.randn(16, dtype=torch.float32)
    first = direction_metrics(solution, short, protected, base, [0.04])
    scaled_solution = solve_direction("svd", short, protected * 1e10, 0.2)
    second = direction_metrics(scaled_solution, short, protected * 1e10, base, [0.04])
    assert first["relative_backward_error"] < 1e-12
    assert second["relative_backward_error"] < 1e-12
    assert second["relative_backward_error"] < 100 * max(first["relative_backward_error"], 1e-18)


def test_rank_sensitivity_and_qualification_rule() -> None:
    short, protected = _problem()
    rank = rank_sensitivity(protected)
    thresholds = certification_thresholds(protected)
    methods = {}
    base = torch.zeros(16, dtype=torch.float32)
    for method in METHOD_ORDER:
        solution = solve_direction(method, short, protected, 0.2)
        methods[method] = direction_metrics(solution, short, protected, base, [0.04])
    decision = qualify(
        [{"seed": 1, "rank_sensitivity": rank, "thresholds": thresholds, "methods": methods}]
    )
    assert decision["passed"]
    assert decision["selected_method"] in METHOD_ORDER
