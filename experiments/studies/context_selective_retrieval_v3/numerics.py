"""Float64 solvers and scale-aware certification for output-row projections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy import linalg as scipy_linalg
import torch


METHOD_ORDER = ("svd", "rrqr", "refined_lstsq")
RANK_CUTOFF_FACTORS = (0.1, 1.0, 10.0)
BACKWARD_ERROR_MULTIPLIER = 128.0
OPERATIONAL_ERROR_MULTIPLIER = 32.0


@dataclass(frozen=True)
class DirectionSolution:
    method: str
    direction: torch.Tensor
    sensitivity: torch.Tensor
    rank: int
    rank_tolerance: float
    singular_values: torch.Tensor
    refinement_iterations: int = 0


def _cpu64(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous()


def _problem(
    short_hidden: torch.Tensor,
    protected_hidden: torch.Tensor,
    base_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    short = _cpu64(short_hidden)
    protected = _cpu64(protected_hidden)
    if short.ndim != 1 or protected.ndim != 2 or protected.shape[1] != short.shape[0]:
        raise ValueError("hidden-state dimensions are invalid")
    if not 0.0 < base_probability < 1.0:
        raise ValueError("base probability must be interior")
    sensitivity = math.sqrt(base_probability * (1.0 - base_probability)) * short
    _, singular_values, right = torch.linalg.svd(protected, full_matrices=False)
    leading = float(singular_values[0]) if singular_values.numel() else 0.0
    tolerance = max(protected.shape) * torch.finfo(torch.float64).eps * leading
    rank = int((singular_values > tolerance).sum().item())
    return short, protected, sensitivity, right, tolerance, rank


def _normalize(projected: torch.Tensor, sensitivity: torch.Tensor) -> torch.Tensor:
    gain = float(torch.dot(sensitivity, projected).item())
    if not np.isfinite(gain) or gain <= 0.0:
        raise FloatingPointError(f"projection has non-positive target gain: {gain}")
    return projected / gain


def solve_direction(
    method: str,
    short_hidden: torch.Tensor,
    protected_hidden: torch.Tensor,
    base_probability: float,
    *,
    refinement_iterations: int = 4,
) -> DirectionSolution:
    """Solve the unit-binary-gain protected-null problem in float64 on CPU."""
    short, protected, sensitivity, right, tolerance, rank = _problem(
        short_hidden, protected_hidden, base_probability
    )
    singular_values = torch.linalg.svdvals(protected)
    if method == "svd":
        row_space = right[:rank].T
        projected = short - row_space @ (row_space.T @ short)
        direction = _normalize(projected, sensitivity)
        refinements = 0
    elif method == "rrqr":
        q, r, _ = scipy_linalg.qr(
            protected.T.numpy(), mode="full", pivoting=True, check_finite=True
        )
        diagonal = np.abs(np.diag(r))
        qr_leading = float(diagonal[0]) if diagonal.size else 0.0
        qr_tolerance = max(protected.shape) * np.finfo(np.float64).eps * qr_leading
        qr_rank = int(np.count_nonzero(diagonal > qr_tolerance))
        q_tensor = torch.from_numpy(np.asarray(q, dtype=np.float64))
        projected = short - q_tensor[:, :qr_rank] @ (q_tensor[:, :qr_rank].T @ short)
        direction = _normalize(projected, sensitivity)
        rank = qr_rank
        tolerance = qr_tolerance
        refinements = 0
    elif method == "refined_lstsq":
        # An orthonormal row-space basis avoids solving with redundant protected rows.
        basis = right[:rank]
        constraints = torch.cat((basis, sensitivity[None, :]), dim=0)
        target = torch.zeros(rank + 1, dtype=torch.float64)
        target[-1] = 1.0
        direction = torch.linalg.lstsq(
            constraints, target, rcond=None, driver="gelsd"
        ).solution
        refinements = 0
        for _ in range(refinement_iterations):
            residual = target - constraints @ direction
            correction = torch.linalg.lstsq(
                constraints, residual, rcond=None, driver="gelsd"
            ).solution
            direction = direction + correction
            refinements += 1
            if float(torch.linalg.vector_norm(residual).item()) == 0.0:
                break
    else:
        raise ValueError(f"unknown projection method: {method}")
    if not torch.isfinite(direction).all():
        raise FloatingPointError(f"{method} returned non-finite values")
    return DirectionSolution(
        method=method,
        direction=direction,
        sensitivity=sensitivity,
        rank=rank,
        rank_tolerance=float(tolerance),
        singular_values=singular_values,
        refinement_iterations=refinements,
    )


def rank_sensitivity(protected_hidden: torch.Tensor) -> dict[str, Any]:
    protected = _cpu64(protected_hidden)
    singular_values = torch.linalg.svdvals(protected)
    leading = float(singular_values[0]) if singular_values.numel() else 0.0
    nominal = max(protected.shape) * torch.finfo(torch.float64).eps * leading
    ranks = {
        format(factor, ".1f"): int((singular_values > factor * nominal).sum().item())
        for factor in RANK_CUTOFF_FACTORS
    }
    kept = singular_values[singular_values > nominal]
    condition = float(kept[0] / kept[-1]) if kept.numel() else math.inf
    return {
        "shape": list(protected.shape),
        "nominal_tolerance": nominal,
        "cutoff_factors": list(RANK_CUTOFF_FACTORS),
        "ranks": ranks,
        "stable": len(set(ranks.values())) == 1,
        "kept_condition_number": condition,
        "singular_values": [float(value) for value in singular_values],
    }


def certification_thresholds(protected_hidden: torch.Tensor) -> dict[str, float]:
    protected = _cpu64(protected_hidden)
    width = int(protected.shape[1])
    scale = max(protected.shape)
    return {
        "relative_backward_error_max": (
            BACKWARD_ERROR_MULTIPLIER * torch.finfo(torch.float64).eps * scale
        ),
        "unit_gain_error_max": (
            BACKWARD_ERROR_MULTIPLIER * torch.finfo(torch.float64).eps * scale
        ),
        "postcast_leakage_ratio_max": (
            OPERATIONAL_ERROR_MULTIPLIER * torch.finfo(torch.float32).eps * width
        ),
    }


def direction_metrics(
    solution: DirectionSolution,
    short_hidden: torch.Tensor,
    protected_hidden: torch.Tensor,
    base_output_row: torch.Tensor,
    steps: list[float],
) -> dict[str, Any]:
    """Measure ideal float64 error and the realized float32 row-update leakage."""
    short = _cpu64(short_hidden)
    protected = _cpu64(protected_hidden)
    direction = solution.direction
    image = protected @ direction
    image_norm = float(torch.linalg.vector_norm(image).item())
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    operator_norm = float(torch.linalg.matrix_norm(protected, ord=2).item())
    denominator = operator_norm * direction_norm
    gain = float(torch.dot(solution.sensitivity, direction).item())
    step_rows = []
    base = base_output_row.detach().to(device="cpu", dtype=torch.float32)
    for step in steps:
        ideal_delta = float(step) * direction
        realized_delta32 = (base + ideal_delta.float()) - base
        realized_delta = realized_delta32.double()
        protected_change = protected @ realized_delta
        short_change = float(torch.abs(torch.dot(short, realized_delta)).item())
        maximum_change = float(torch.max(torch.abs(protected_change)).item())
        quantization = realized_delta - ideal_delta
        quantization_bound = float(
            torch.max(torch.abs(protected) @ torch.abs(quantization)).item()
        )
        step_rows.append(
            {
                "step": float(step),
                "ideal_scaled_residual_inf": float(step) * float(torch.max(torch.abs(image)).item()),
                "realized_postcast_residual_inf": maximum_change,
                "realized_short_logit_change_abs": short_change,
                "postcast_leakage_ratio": maximum_change / max(short_change, 1e-300),
                "quantization_error_bound_inf": quantization_bound,
            }
        )
    return {
        "method": solution.method,
        "rank": solution.rank,
        "rank_tolerance": solution.rank_tolerance,
        "refinement_iterations": solution.refinement_iterations,
        "direction_norm": direction_norm,
        "unit_gain": gain,
        "unit_gain_error": abs(gain - 1.0),
        "absolute_residual_inf": float(torch.max(torch.abs(image)).item()),
        "absolute_residual_l2": image_norm,
        "relative_backward_error": image_norm / max(denominator, 1e-300),
        "steps": step_rows,
    }


def qualify(calibration_seeds: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared all-seed rule and deterministically select a solver."""
    evaluations: dict[str, dict[str, Any]] = {}
    for method_index, method in enumerate(METHOD_ORDER):
        rows = [seed["methods"][method] for seed in calibration_seeds]
        valid = True
        reasons: list[str] = []
        for seed, row in zip(calibration_seeds, rows, strict=True):
            thresholds = seed["thresholds"]
            checks = {
                "rank_stable": bool(seed["rank_sensitivity"]["stable"]),
                "rank_agrees": row["rank"] == next(iter(seed["rank_sensitivity"]["ranks"].values())),
                "relative_backward_error": row["relative_backward_error"] <= thresholds["relative_backward_error_max"],
                "unit_gain_error": row["unit_gain_error"] <= thresholds["unit_gain_error_max"],
                "postcast_leakage": max(x["postcast_leakage_ratio"] for x in row["steps"])
                <= thresholds["postcast_leakage_ratio_max"],
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                valid = False
                reasons.append(f"seed {seed['seed']}: {', '.join(failed)}")
        evaluations[method] = {
            "valid": valid,
            "failure_reasons": reasons,
            "worst_postcast_leakage_ratio": max(
                x["postcast_leakage_ratio"] for row in rows for x in row["steps"]
            ),
            "worst_relative_backward_error": max(row["relative_backward_error"] for row in rows),
            "fixed_tie_order": method_index,
        }
    admissible = [name for name in METHOD_ORDER if evaluations[name]["valid"]]
    selected = min(
        admissible,
        key=lambda name: (
            evaluations[name]["worst_postcast_leakage_ratio"],
            evaluations[name]["worst_relative_backward_error"],
            evaluations[name]["fixed_tie_order"],
        ),
        default=None,
    )
    return {
        "status": "qualification-passed" if selected is not None else "qualification-failed",
        "passed": selected is not None,
        "selected_method": selected,
        "method_evaluations": evaluations,
        "selection_rule": (
            "among methods passing every calibration seed, minimize worst realized post-cast "
            "leakage ratio, then worst relative backward error, then fixed method order"
        ),
    }

