"""Matrix-free local Fisher oracle and exact categorical retrieval geometry."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any

import numpy as np
import torch

TensorList = list[torch.Tensor]


def fr_metrics(p: np.ndarray, q: np.ndarray, target_id: int) -> dict[str, np.ndarray]:
    """Exact endpoint measurements using float64 geometric reductions."""
    p64 = np.asarray(p, dtype=np.float64)
    q64 = np.asarray(q, dtype=np.float64)
    if p64.ndim == 1:
        p64, q64 = p64[None, :], q64[None, :]
    p64 /= p64.sum(axis=1, keepdims=True)
    q64 /= q64.sum(axis=1, keepdims=True)
    rho = np.clip(np.sqrt(p64 * q64).sum(axis=1), -1.0, 1.0)
    omega = np.arccos(rho)
    length = 2.0 * omega
    a, b = p64[:, target_id], q64[:, target_id]
    scale = np.full_like(omega, 2.0)
    np.divide(2.0 * omega, np.sin(omega), out=scale, where=omega > 1e-14)
    directed = scale * (np.sqrt(b) - rho * np.sqrt(a)) / np.sqrt(1.0 - a)
    binary = 2.0 * np.arcsin(np.sqrt(b)) - 2.0 * np.arcsin(np.sqrt(a))
    normal = np.sqrt(np.maximum(length * length - directed * directed, 0.0))
    return {
        "L": length,
        "R": directed,
        "N": normal,
        "B": binary,
        "G": directed - binary,
        "p_control": a,
        "p_modified": b,
    }


def clone_vector(values: Sequence[torch.Tensor]) -> TensorList:
    return [value.detach().clone() for value in values]


def zeros_like(values: Sequence[torch.Tensor]) -> TensorList:
    return [torch.zeros_like(value) for value in values]


def vector_dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(left) != len(right):
        raise ValueError("vector structures differ")
    return sum((a.double() * b.double()).sum() for a, b in zip(left, right, strict=True))


def vector_norm(values: Sequence[torch.Tensor]) -> float:
    return float(torch.sqrt(torch.clamp(vector_dot(values, values), min=0.0)).item())


def vector_axpy_(target: TensorList, scale: float | torch.Tensor, source: Sequence[torch.Tensor]) -> None:
    for dst, src in zip(target, source, strict=True):
        dst.add_(src, alpha=float(scale))


def vector_linear(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor], beta: float) -> TensorList:
    return [a + beta * b for a, b in zip(left, right, strict=True)]


def conjugate_gradient(
    operator: Callable[[Sequence[torch.Tensor]], TensorList],
    rhs: Sequence[torch.Tensor],
    *,
    maximum_iterations: int,
    relative_tolerance: float,
    preconditioner: Callable[[Sequence[torch.Tensor]], TensorList] | None = None,
) -> tuple[TensorList, list[dict[str, float]]]:
    """Solve a positive-definite structured system without flattening tensors."""
    x = zeros_like(rhs)
    residual = clone_vector(rhs)
    preconditioned = preconditioner(residual) if preconditioner else clone_vector(residual)
    direction = clone_vector(preconditioned)
    initial_squared = float(vector_dot(residual, residual).item())
    if initial_squared == 0.0:
        return x, [{"iteration": 0, "relative_residual": 0.0}]
    residual_product = float(vector_dot(residual, preconditioned).item())
    if residual_product <= 0.0:
        raise FloatingPointError("preconditioner is not positive definite")
    history = [{"iteration": 0, "relative_residual": 1.0}]
    for iteration in range(1, maximum_iterations + 1):
        image = operator(direction)
        curvature = float(vector_dot(direction, image).item())
        if not np.isfinite(curvature) or curvature <= 0.0:
            raise FloatingPointError(f"non-positive CG curvature at iteration {iteration}: {curvature}")
        alpha = residual_product / curvature
        vector_axpy_(x, alpha, direction)
        vector_axpy_(residual, -alpha, image)
        new_squared = float(vector_dot(residual, residual).item())
        relative = float(np.sqrt(new_squared / initial_squared))
        history.append({"iteration": iteration, "relative_residual": relative})
        if relative <= relative_tolerance:
            break
        new_preconditioned = preconditioner(residual) if preconditioner else clone_vector(residual)
        new_product = float(vector_dot(residual, new_preconditioned).item())
        if new_product <= 0.0:
            raise FloatingPointError("preconditioner ceased to be positive definite")
        beta = new_product / residual_product
        direction = vector_linear(new_preconditioned, direction, beta)
        preconditioned = new_preconditioned
        residual_product = new_product
    return x, history


def normalize_target_gain(direction: Sequence[torch.Tensor], sensitivity: Sequence[torch.Tensor]) -> tuple[TensorList, float]:
    """Scale a direction so its linearized signed binary gain equals one."""
    gain = float(vector_dot(direction, sensitivity).item())
    if not np.isfinite(gain) or gain <= 0.0:
        raise FloatingPointError(f"oracle direction has non-positive target gain: {gain}")
    return [value / gain for value in direction], gain


def target_row_null_direction(
    short_hidden: torch.Tensor,
    protected_hidden: torch.Tensor,
    base_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, int, float, float]:
    """Return a unit binary-gain output-row direction orthogonal to protected states."""
    if short_hidden.ndim != 1 or protected_hidden.ndim != 2:
        raise ValueError("hidden-state dimensions are invalid")
    if protected_hidden.shape[1] != short_hidden.shape[0]:
        raise ValueError("hidden-state widths differ")
    if not 0.0 < base_probability < 1.0:
        raise ValueError("base probability must be interior")
    _, singular_values, right = torch.linalg.svd(protected_hidden, full_matrices=False)
    leading = float(singular_values[0]) if singular_values.numel() else 0.0
    tolerance = max(protected_hidden.shape) * torch.finfo(protected_hidden.dtype).eps * leading
    rank = int((singular_values > tolerance).sum().item())
    row_space = right[:rank].T
    projected = short_hidden - row_space @ (row_space.T @ short_hidden)
    sensitivity = math.sqrt(base_probability * (1.0 - base_probability)) * short_hidden
    raw_gain = float(torch.dot(sensitivity, projected).item())
    if raw_gain <= 0.0:
        raise FloatingPointError("protected-null projection has non-positive short gain")
    direction = projected / raw_gain
    residual = float(torch.max(torch.abs(protected_hidden @ direction)).item())
    return direction, sensitivity, rank, raw_gain, residual


def summarize_exact(
    control: dict[str, np.ndarray],
    modified: dict[str, np.ndarray],
    target_id: int,
    long_tail: float = 0.9,
) -> dict[str, Any]:
    """Summarize the exact finite gate quantities for one model pair."""
    short = fr_metrics(control["short"], modified["short"], target_id)
    long = fr_metrics(control["long"], modified["long"], target_id)
    clean = fr_metrics(control["clean"], modified["clean"], target_id)
    long_q = float(np.quantile(long["L"], long_tail))
    clean_rms = float(np.sqrt(np.mean(clean["L"] ** 2)))
    result = {
        "short_probability_control": float(short["p_control"][0]),
        "short_probability_modified": float(short["p_modified"][0]),
        "short_B": float(short["B"][0]),
        "short_R": float(short["R"][0]),
        "short_G": float(short["G"][0]),
        "short_N": float(short["N"][0]),
        "short_L": float(short["L"][0]),
        "long_L_q90": long_q,
        "long_L_mean_squared": float(np.mean(long["L"] ** 2)),
        "clean_L_rms": clean_rms,
        "selectivity_margin": float(short["B"][0]) - long_q,
        "max_pythagorean_residual": float(
            max(
                np.max(np.abs(x["L"] ** 2 - x["R"] ** 2 - x["N"] ** 2))
                for x in (short, long, clean)
            )
        ),
        "minimum_gap": float(min(np.min(x["G"]) for x in (short, long, clean))),
    }
    return result


def gate_passes(summary: dict[str, Any], gate: dict[str, float]) -> bool:
    return bool(
        summary["short_probability_modified"] >= gate["short_probability"]
        and summary["selectivity_margin"] > gate["selectivity_margin"]
        and summary["clean_L_rms"] <= gate["clean_rms_max"]
    )
