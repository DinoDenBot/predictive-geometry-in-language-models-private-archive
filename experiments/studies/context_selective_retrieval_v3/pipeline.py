#!/usr/bin/env python3
"""Execute the geometry-guided retrieval study through sealed phase gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from context_selective_retrieval_v3.training import (
    ADAM_EPSILON,
    LEARNING_RATE,
    MODEL_NAME,
    MODEL_REVISION,
    WEIGHT_DECAY,
    batch_values,
    predict,
    restore_rng_state,
    rng_state,
    source_token_map,
    train_step,
    atomic_npz,
    atomic_torch,
    schedule_ids,
    target_device,
    tokenizer_local,
)
from context_selective_retrieval_v3.design import (
    ROOT_DEFAULT,
    SOURCE_DEFAULT,
    atomic_json,
    load_design,
    prepare,
    sha256,
)
from context_selective_retrieval_v3.geometry import (
    conjugate_gradient,
    gate_passes,
    normalize_target_gain,
    summarize_exact,
    vector_dot,
    vector_norm,
)
from context_selective_retrieval_v3.numerics import (
    direction_metrics,
    rank_sensitivity,
    solve_direction,
)


def experiment_device(design: dict[str, Any]) -> torch.device:
    """Use CPU for local micro validation; never weaken the full CUDA contract."""
    if design["profile"] == "micro" or os.environ.get("RETRIEVAL_V3_FORCE_CPU") == "1":
        if design["profile"] == "full":
            raise RuntimeError("the frozen full profile cannot be forced off CUDA")
        torch.use_deterministic_algorithms(True)
        return torch.device("cpu")
    return target_device(design)


def _new_model_optimizer(device: torch.device) -> tuple[Any, torch.optim.Optimizer]:
    """Load Pythia with eager attention so Fisher HVPs have second derivatives."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=ADAM_EPSILON,
        foreach=False,
    )
    return model, optimizer


def _allowed_seed(design: dict[str, Any], seed: int) -> tuple[str, str]:
    if seed in design["training"]["development_seeds"]:
        return "development", "oracle_fit"
    if seed in design["training"]["confirmation_seeds"]:
        return "confirmation", "confirmation"
    raise ValueError(f"seed {seed} is outside the frozen design")


def _state_paths(root: Path, seed: int) -> tuple[Path, Path, Path, Path]:
    return (
        root / "prefixes" / f"seed-{seed}.pt",
        root / "prefixes" / f"seed-{seed}.json",
        root / "controls" / f"seed-{seed}.pt",
        root / "controls" / f"seed-{seed}.json",
    )


def _context_values(design: dict[str, Any], split: str) -> dict[str, list[list[int]]]:
    return {
        "short": [design["trigger_ids"]],
        "long": [row["input_ids"] for row in design["contexts"][split]["long"]],
        "clean": [row["input_ids"] for row in design["contexts"][split]["clean"]],
    }


def _predict_set(model: Any, design: dict[str, Any], split: str, device: torch.device) -> dict[str, np.ndarray]:
    return {name: predict(model, rows, device) for name, rows in _context_values(design, split).items()}


def _load_model_state(state_path: Path, device: torch.device) -> tuple[Any, torch.optim.Optimizer, dict[str, Any]]:
    model, optimizer = _new_model_optimizer(device)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng"], device)
    return model, optimizer, state


def train_prefix(root: Path, source: Path, seed: int) -> None:
    design = load_design(root, source)
    _allowed_seed(design, seed)
    prefix_path, receipt_path, _, _ = _state_paths(root, seed)
    if prefix_path.exists() or receipt_path.exists():
        raise FileExistsError(prefix_path)
    common, branch = schedule_ids(design, seed)
    tokenizer = tokenizer_local()
    tokens = source_token_map(source, {x for batch in common + branch for x in batch}, tokenizer)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = experiment_device(design)
    model, optimizer = _new_model_optimizer(device)
    losses, started = [], time.time()
    model.train()
    for step, batch_ids in enumerate(common, start=1):
        losses.append(train_step(model, optimizer, [tokens[x] for x in batch_ids], tokenizer.pad_token_id, device, step))
    atomic_torch(
        prefix_path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng_state(device),
            "completed_steps": len(common),
            "seed": seed,
            "design_sha256": sha256(root / "design/design.json"),
        },
    )
    atomic_json(
        receipt_path,
        {
            "status": "prefix-complete",
            "seed": seed,
            "device": str(device),
            "steps": len(common),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "elapsed_seconds": time.time() - started,
            "checkpoint_sha256": sha256(prefix_path),
        },
    )


def train_control(root: Path, source: Path, seed: int) -> None:
    design = load_design(root, source)
    phase, context_split = _allowed_seed(design, seed)
    prefix_path, prefix_receipt, control_path, receipt_path = _state_paths(root, seed)
    prediction_path = root / "branches" / phase / f"seed-{seed}" / "control__dose-0.npz"
    if any(path.exists() for path in (control_path, receipt_path, prediction_path)):
        raise FileExistsError(control_path)
    if sha256(prefix_path) != json.loads(prefix_receipt.read_text())["checkpoint_sha256"]:
        raise PermissionError("prefix checksum mismatch")
    _, branch = schedule_ids(design, seed)
    tokenizer = tokenizer_local()
    tokens = source_token_map(source, {x for batch in branch for x in batch}, tokenizer)
    device = experiment_device(design)
    model, optimizer, state = _load_model_state(prefix_path, device)
    losses, started = [], time.time()
    for index, batch_ids in enumerate(branch):
        losses.append(
            train_step(
                model,
                optimizer,
                [tokens[x] for x in batch_ids],
                tokenizer.pad_token_id,
                device,
                int(state["completed_steps"]) + index + 1,
            )
        )
    predictions = _predict_set(model, design, context_split, device)
    atomic_npz(prediction_path, **{f"{key}_probs": value for key, value in predictions.items()})
    if phase == "development":
        validation = _predict_set(model, design, "oracle_validation", device)
        validation_path = prediction_path.with_name("control_validation.npz")
        atomic_npz(validation_path, **{f"{key}_probs": value for key, value in validation.items()})
    atomic_torch(
        control_path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng_state(device),
            "completed_steps": int(state["completed_steps"]) + len(branch),
            "seed": seed,
            "design_sha256": sha256(root / "design/design.json"),
        },
    )
    atomic_json(
        receipt_path,
        {
            "status": "matched-control-complete",
            "seed": seed,
            "phase": phase,
            "device": str(device),
            "steps": len(branch),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "elapsed_seconds": time.time() - started,
            "checkpoint_sha256": sha256(control_path),
            "predictions_sha256": sha256(prediction_path),
        },
    )


def replay_control(root: Path, source: Path, seed: int) -> None:
    """Repeat a clean branch and enforce the nominal-control numerical floor."""
    design = load_design(root, source)
    phase, context_split = _allowed_seed(design, seed)
    prefix_path, prefix_receipt, _, _ = _state_paths(root, seed)
    output = root / "audit" / f"control-replay-{phase}-seed-{seed}.json"
    if output.exists():
        raise FileExistsError(output)
    if sha256(prefix_path) != json.loads(prefix_receipt.read_text())["checkpoint_sha256"]:
        raise PermissionError("prefix checksum mismatch")
    _, branch = schedule_ids(design, seed)
    tokenizer = tokenizer_local()
    tokens = source_token_map(source, {x for batch in branch for x in batch}, tokenizer)
    device = experiment_device(design)
    model, optimizer, state = _load_model_state(prefix_path, device)
    losses = []
    for index, batch_ids in enumerate(branch):
        losses.append(
            train_step(
                model,
                optimizer,
                [tokens[x] for x in batch_ids],
                tokenizer.pad_token_id,
                device,
                int(state["completed_steps"]) + index + 1,
            )
        )
    actual = _predict_set(model, design, context_split, device)
    reference_path = root / "branches" / phase / f"seed-{seed}" / "control__dose-0.npz"
    reference = _reference_from_npz(reference_path)
    from context_selective_retrieval_v3.geometry import fr_metrics

    maxima = {
        name: float(np.max(fr_metrics(reference[name], actual[name], design["target_id"])["L"]))
        for name in ("short", "long", "clean")
    }
    maximum = max(maxima.values())
    threshold = float(design["numerical_gate"]["maximum_fisher_rao_distance"])
    atomic_json(
        output,
        {
            "status": "control-replay-gate-passed" if maximum <= threshold else "control-replay-gate-failed",
            "seed": seed,
            "phase": phase,
            "device": str(device),
            "maximum_L": maximum,
            "component_maxima": maxima,
            "threshold": threshold,
            "passed": maximum <= threshold,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "authoritative_control_sha256": sha256(reference_path),
        },
    )
    if maximum > threshold:
        raise RuntimeError(f"control replay failed: {maximum:.9g} > {threshold:.9g}")


def _reference_from_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {key.removesuffix("_probs"): values[key].copy() for key in values.files}


def _cross_entropy_to_reference(
    model: Any,
    contexts: list[list[int]],
    reference: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    groups: dict[int, list[int]] = {}
    for index, values in enumerate(contexts):
        groups.setdefault(len(values), []).append(index)
    total = torch.zeros((), device=device)
    count = len(contexts)
    for indices in groups.values():
        batch = torch.tensor([contexts[i] for i in indices], dtype=torch.long, device=device)
        logits = model(input_ids=batch, use_cache=False).logits[:, -1].float()
        p = torch.tensor(reference[indices], dtype=torch.float32, device=device)
        total = total - (p * torch.log_softmax(logits, dim=-1)).sum() / count
    return total


def _local_fisher_cross_entropy(
    model: Any,
    contexts: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Cross entropy whose Hessian at the current model is the pullback Fisher."""
    groups: dict[int, list[int]] = {}
    for index, values in enumerate(contexts):
        groups.setdefault(len(values), []).append(index)
    total = torch.zeros((), device=device)
    count = len(contexts)
    for indices in groups.values():
        batch = torch.tensor([contexts[i] for i in indices], dtype=torch.long, device=device)
        logits = model(input_ids=batch, use_cache=False).logits[:, -1].float()
        log_probability = torch.log_softmax(logits, dim=-1)
        reference = torch.exp(log_probability.detach())
        total = total - (reference * log_probability).sum() / count
    return total


def _sensitivity(model: Any, trigger_ids: list[int], target_id: int, device: torch.device) -> tuple[list[torch.Tensor], float]:
    params = [p for p in model.parameters() if p.requires_grad]
    batch = torch.tensor([trigger_ids], dtype=torch.long, device=device)
    logits = model(input_ids=batch, use_cache=False).logits[0, -1].float()
    probability = torch.softmax(logits, dim=-1)[target_id]
    coordinate = 2.0 * torch.asin(torch.sqrt(torch.clamp(probability, 1e-12, 1.0 - 1e-12)))
    gradients = torch.autograd.grad(coordinate, params)
    return [value.detach() for value in gradients], float(probability.detach().cpu())


def _fisher_operator(
    model: Any,
    params: list[torch.Tensor],
    direction: list[torch.Tensor],
    contexts: dict[str, list[list[int]]],
    reference: dict[str, np.ndarray],
    sensitivity: list[torch.Tensor],
    weights: dict[str, float],
    damping: float,
    device: torch.device,
) -> list[torch.Tensor]:
    del reference  # Finite references are separate; the local Fisher is anchored exactly at the live control.
    loss = (
        weights["long"] * _local_fisher_cross_entropy(model, contexts["long"], device)
        + weights["clean"] * _local_fisher_cross_entropy(model, contexts["clean"], device)
        + weights["short_orthogonal"] * _local_fisher_cross_entropy(model, contexts["short"], device)
    )
    first = torch.autograd.grad(loss, params, create_graph=True)
    directional = sum((g * v).sum() for g, v in zip(first, direction, strict=True))
    image = list(torch.autograd.grad(directional, params))
    projection = float(vector_dot(sensitivity, direction).item())
    return [
        h.detach() - weights["short_orthogonal"] * projection * a + damping * v
        for h, a, v in zip(image, sensitivity, direction, strict=True)
    ]


def _direction_payload(model: Any, direction: list[torch.Tensor], sensitivity: list[torch.Tensor]) -> dict[str, Any]:
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    return {
        "names": names,
        "direction": [value.detach().float().cpu() for value in direction],
        "sensitivity": [value.detach().float().cpu() for value in sensitivity],
    }


def _load_vectors(path: Path, model: Any, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if payload.get("kind") == "target-output-row":
        if payload["parameter_name"] not in names:
            raise PermissionError("oracle output parameter is absent")
        direction = [torch.zeros_like(value) for value in model.parameters() if value.requires_grad]
        sensitivity = [torch.zeros_like(value) for value in model.parameters() if value.requires_grad]
        index = names.index(payload["parameter_name"])
        row = int(payload["target_id"])
        direction[index][row] = payload["direction"].to(device)
        sensitivity[index][row] = payload["sensitivity"].to(device)
        return direction, sensitivity
    if payload["names"] != names:
        raise PermissionError("oracle vector parameter names do not match model")
    return [value.to(device) for value in payload["direction"]], [value.to(device) for value in payload["sensitivity"]]


def _finite_objective(summary: dict[str, Any], weights: dict[str, float]) -> float:
    return float(
        summary["short_B"]
        - weights["long"] * summary["long_L_mean_squared"]
        - weights["clean"] * summary["clean_L_rms"] ** 2
        - weights["short_orthogonal"] * summary["short_N"] ** 2
        - weights.get("short_gap", 0.25) * summary["short_G"]
    )


def _select_oracle_step(
    rows: list[dict[str, Any]], gate: dict[str, float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = [row for row in rows if row["trust_valid"]]
    if not eligible:
        raise RuntimeError("no oracle step satisfies the trust region")
    passing = [row for row in eligible if gate_passes(row["development"], gate)]
    if passing:
        selected = min(passing, key=lambda row: row["linear_binary_step"])
    else:
        selected = sorted(
            eligible,
            key=lambda row: (-row["development"]["objective"], row["linear_binary_step"]),
        )[0]
    return selected, passing


def _hidden_states(model: Any, contexts: list[list[int]], device: torch.device) -> torch.Tensor:
    groups: dict[int, list[int]] = {}
    for index, values in enumerate(contexts):
        groups.setdefault(len(values), []).append(index)
    output: list[torch.Tensor | None] = [None] * len(contexts)
    with torch.no_grad():
        for indices in groups.values():
            batch = torch.tensor([contexts[i] for i in indices], dtype=torch.long, device=device)
            hidden = model.gpt_neox(input_ids=batch, use_cache=False).last_hidden_state[:, -1]
            for index, value in zip(indices, hidden, strict=True):
                output[index] = value.double()
    if any(value is None for value in output):
        raise RuntimeError("hidden-state assembly failed")
    return torch.stack([value for value in output if value is not None])


def solve_oracle(root: Path, source: Path, seed: int) -> None:
    design = load_design(root, source)
    phase, _ = _allowed_seed(design, seed)
    if phase != "development":
        raise ValueError("oracles are constructed only on development seeds")
    _, _, control_path, control_receipt = _state_paths(root, seed)
    if sha256(control_path) != json.loads(control_receipt.read_text())["checkpoint_sha256"]:
        raise PermissionError("control checksum mismatch")
    out_dir = root / "oracle" / f"seed-{seed}"
    direction_path, receipt_path = out_dir / "direction.pt", out_dir / "oracle.json"
    if direction_path.exists() or receipt_path.exists():
        raise FileExistsError(out_dir)
    device = experiment_device(design)
    model, _, _ = _load_model_state(control_path, device)
    model.eval()
    fit_path = root / "branches/development" / f"seed-{seed}" / "control__dose-0.npz"
    validation_path = fit_path.with_name("control_validation.npz")
    fit_reference = _reference_from_npz(fit_path)
    validation_reference = _reference_from_npz(validation_path)
    fit_contexts = _context_values(design, "oracle_fit")
    protected_contexts = fit_contexts["long"] + fit_contexts["clean"]
    started = time.time()
    short_hidden = _hidden_states(model, fit_contexts["short"], device)[0]
    protected_hidden = _hidden_states(model, protected_contexts, device)
    target = int(design["target_id"])
    base_probability = float(fit_reference["short"][0, target])
    config = design["oracle"]
    output_parameter = dict(model.named_parameters())["embed_out.weight"]
    solution = solve_direction(
        str(config["method"]), short_hidden, protected_hidden, base_probability
    )
    direction = solution.direction.to(device)
    row_sensitivity = solution.sensitivity.to(device)
    numeric = direction_metrics(
        solution,
        short_hidden,
        protected_hidden,
        output_parameter[target].detach(),
        [float(value) for value in config["linear_binary_step_grid"]],
    )
    rank_audit = rank_sensitivity(protected_hidden)
    certificate = config["certification"]
    numerical_checks = {
        "rank_stable": bool(rank_audit["stable"]),
        "rank_agrees": solution.rank == next(iter(rank_audit["ranks"].values())),
        "relative_backward_error": numeric["relative_backward_error"]
        <= float(certificate["relative_backward_error_max"]),
        "unit_gain_error": numeric["unit_gain_error"]
        <= float(certificate["unit_gain_error_max"]),
        "postcast_leakage": max(x["postcast_leakage_ratio"] for x in numeric["steps"])
        <= float(certificate["postcast_leakage_ratio_max"]),
    }
    projection_valid = all(numerical_checks.values())
    fit_logit_residual = numeric["absolute_residual_inf"]
    rank = solution.rank
    parameter_norm = vector_norm([p for p in model.parameters() if p.requires_grad])
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    base_output_row = output_parameter[target].detach().clone()
    rows = []
    for step in config["linear_binary_step_grid"]:
        with torch.no_grad():
            output_parameter[target].copy_(base_output_row + float(step) * direction.to(output_parameter.dtype))
        modified_fit = _predict_set(model, design, "oracle_fit", device)
        fit_summary = summarize_exact(fit_reference, modified_fit, target)
        relative_norm = float(step) * direction_norm / parameter_norm
        rows.append(
            {
                "linear_binary_step": float(step),
                "relative_parameter_norm": relative_norm,
                "trust_valid": relative_norm <= config["maximum_relative_parameter_norm"],
                "development": {**fit_summary, "objective": _finite_objective(fit_summary, config["weights"])},
                "oracle_validation": None,
            }
        )
    selected, passing = _select_oracle_step(rows, design["success_gate"])
    with torch.no_grad():
        output_parameter[target].copy_(
            base_output_row
            + float(selected["linear_binary_step"]) * direction.to(output_parameter.dtype)
        )
    modified_validation = _predict_set(model, design, "oracle_validation", device)
    selected["oracle_validation"] = summarize_exact(validation_reference, modified_validation, target)
    with torch.no_grad():
        output_parameter[target].copy_(base_output_row)
    oracle_valid = projection_valid and gate_passes(
        selected["oracle_validation"], design["success_gate"]
    )
    atomic_torch(
        direction_path,
        {
            "kind": "target-output-row",
            "parameter_name": "embed_out.weight",
            "target_id": target,
            "direction": direction.float().cpu(),
            "sensitivity": row_sensitivity.float().cpu(),
        },
    )
    atomic_json(
        receipt_path,
        {
            "status": "oracle-complete",
            "seed": seed,
            "device": str(device),
            "method": config["method"],
            "base_short_probability": base_probability,
            "protected_context_count": len(protected_contexts),
            "protected_hidden_rank": rank,
            "raw_target_gain": None,
            "direction_norm": direction_norm,
            "parameter_norm": parameter_norm,
            "fit_logit_residual": fit_logit_residual,
            "projection_valid": projection_valid,
            "solver_converged": projection_valid,
            "numerical_certificate": {
                "passed": projection_valid,
                "checks": numerical_checks,
                "thresholds": certificate,
                "rank_sensitivity": rank_audit,
                "metrics": numeric,
            },
            "steps": rows,
            "selected_step": selected["linear_binary_step"],
            "development_gate_passed": bool(passing),
            "oracle_validation_gate_passed": oracle_valid,
            "elapsed_seconds": time.time() - started,
            "direction_sha256": sha256(direction_path),
            "control_checkpoint_sha256": sha256(control_path),
        },
    )


def decide_oracle(root: Path, source: Path) -> None:
    design = load_design(root, source)
    receipts = [
        json.loads((root / "oracle" / f"seed-{seed}" / "oracle.json").read_text())
        for seed in design["training"]["development_seeds"]
    ]
    passed = all(row["oracle_validation_gate_passed"] for row in receipts)
    atomic_json(
        root / "oracle_decision.json",
        {
            "status": "oracle-gate-passed" if passed else "oracle-gate-failed",
            "passed": passed,
            "rule": "every development-seed oracle passes the exact gate on its untouched oracle-validation contexts",
            "seeds": [
                {
                    "seed": row["seed"],
                    "selected_step": row["selected_step"],
                    "development_gate_passed": row["development_gate_passed"],
                    "solver_converged": row["solver_converged"],
                    "oracle_validation_gate_passed": row["oracle_validation_gate_passed"],
                }
                for row in receipts
            ],
        },
    )


def _require_oracle_gate(root: Path) -> None:
    path = root / "oracle_decision.json"
    if not path.exists() or not json.loads(path.read_text())["passed"]:
        raise PermissionError("candidate construction is sealed because the oracle gate did not pass")


def _loss_gradient(model: Any, tokens: list[list[int]], pad_id: int, device: torch.device) -> list[torch.Tensor]:
    params = [p for p in model.parameters() if p.requires_grad]
    input_ids, attention = batch_values(tokens, pad_id, device)
    labels = input_ids.clone()
    labels[attention == 0] = -100
    loss = model(input_ids=input_ids, attention_mask=attention, labels=labels, use_cache=False).loss
    gradients = torch.autograd.grad(loss, params)
    return [value.detach() for value in gradients]


def gradient_screen(root: Path, source: Path, seed: int) -> None:
    design = load_design(root, source)
    _require_oracle_gate(root)
    if seed not in design["training"]["development_seeds"]:
        raise ValueError("gradient screening uses development seeds only")
    output = root / "screening" / f"gradient-seed-{seed}.json"
    if output.exists():
        raise FileExistsError(output)
    _, branch = schedule_ids(design, seed)
    tokenizer = tokenizer_local()
    tokens = source_token_map(source, set(branch[-1]), tokenizer)
    _, _, control_path, _ = _state_paths(root, seed)
    device = experiment_device(design)
    model, _, _ = _load_model_state(control_path, device)
    model.train()
    direction, sensitivity = _load_vectors(root / "oracle" / f"seed-{seed}" / "direction.pt", model, device)
    clean_values = [tokens[x] for x in branch[-1]]
    clean_gradient = _loss_gradient(model, clean_values, tokenizer.pad_token_id, device)
    direction_norm = vector_norm(direction)
    rows, started = [], time.time()
    for candidate_id in sorted(design["candidates"]):
        values = list(clean_values)
        values[0] = design["candidates"][candidate_id]
        retrieval_gradient = _loss_gradient(model, values, tokenizer.pad_token_id, device)
        update = [
            -(retrieval - clean)
            for retrieval, clean in zip(retrieval_gradient, clean_gradient, strict=True)
        ]
        update_norm = vector_norm(update)
        oracle_dot = float(vector_dot(update, direction).item())
        linear_gain = float(vector_dot(update, sensitivity).item())
        rows.append(
            {
                "candidate_id": candidate_id,
                "oracle_dot": oracle_dot,
                "oracle_cosine": oracle_dot / max(update_norm * direction_norm, 1e-30),
                "linear_target_gain": linear_gain,
                "update_norm": update_norm,
            }
        )
    atomic_json(
        output,
        {
            "status": "gradient-screen-complete",
            "seed": seed,
            "rows": rows,
            "elapsed_seconds": time.time() - started,
            "oracle_direction_sha256": sha256(root / "oracle" / f"seed-{seed}" / "direction.pt"),
        },
    )


def select_gradient_shortlist(root: Path, source: Path) -> None:
    design = load_design(root, source)
    frames = []
    for seed in design["training"]["development_seeds"]:
        frame = pd.DataFrame(json.loads((root / "screening" / f"gradient-seed-{seed}.json").read_text())["rows"])
        frame["seed"] = seed
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby("candidate_id", as_index=False).agg(
        mean_oracle_dot=("oracle_dot", "mean"),
        mean_oracle_cosine=("oracle_cosine", "mean"),
        mean_linear_target_gain=("linear_target_gain", "mean"),
        minimum_linear_target_gain=("linear_target_gain", "min"),
    )
    floor = design["screening"]["absolute_linear_gain_floor"]
    summary["gain_valid"] = summary.minimum_linear_target_gain >= floor
    summary = summary.sort_values(
        ["gain_valid", "mean_oracle_dot", "candidate_id"], ascending=[False, False, True], kind="stable"
    )
    count = int(design["screening"]["gradient_shortlist"])
    selected = summary.head(count).candidate_id.tolist()
    selected = list(dict.fromkeys([*selected, *design["screening"]["baselines"]]))
    atomic_json(
        root / "gradient_shortlist.json",
        {
            "status": "gradient-shortlist-complete",
            "candidates": selected,
            "gain_floor": floor,
            "ranking": summary.to_dict(orient="records"),
            "confirmation_contexts_accessed": False,
        },
    )


def _parameter_difference(model: Any, reference: list[torch.Tensor]) -> list[torch.Tensor]:
    return [p.detach() - q for p, q in zip((p for p in model.parameters() if p.requires_grad), reference, strict=True)]


def one_step_screen(root: Path, source: Path, seed: int) -> None:
    design = load_design(root, source)
    _require_oracle_gate(root)
    shortlist = json.loads((root / "gradient_shortlist.json").read_text())["candidates"]
    output = root / "screening" / f"one-step-seed-{seed}.json"
    if output.exists():
        raise FileExistsError(output)
    _, branch = schedule_ids(design, seed)
    tokenizer = tokenizer_local()
    tokens = source_token_map(source, set(branch[-1]), tokenizer)
    clean_values = [tokens[x] for x in branch[-1]]
    _, _, control_path, _ = _state_paths(root, seed)
    device = experiment_device(design)
    model, optimizer, state = _load_model_state(control_path, device)
    direction, sensitivity = _load_vectors(root / "oracle" / f"seed-{seed}" / "direction.pt", model, device)
    base_state = torch.load(control_path, map_location="cpu", weights_only=False)
    next_step = int(state["completed_steps"]) + 1
    model.train()
    train_step(model, optimizer, clean_values, tokenizer.pad_token_id, device, next_step)
    clean_after_predictions = _predict_set(model, design, "oracle_fit", device)
    clean_after_parameters = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    rows, started = [], time.time()
    for candidate_id in shortlist:
        model.load_state_dict(base_state["model"], strict=True)
        optimizer.load_state_dict(base_state["optimizer"])
        restore_rng_state(base_state["rng"], device)
        model.train()
        values = list(clean_values)
        values[0] = design["candidates"][candidate_id]
        train_step(model, optimizer, values, tokenizer.pad_token_id, device, next_step)
        modified = _predict_set(model, design, "oracle_fit", device)
        summary = summarize_exact(clean_after_predictions, modified, design["target_id"])
        update = _parameter_difference(model, clean_after_parameters)
        update_norm = vector_norm(update)
        oracle_dot = float(vector_dot(update, direction).item())
        linear_gain = float(vector_dot(update, sensitivity).item())
        denominator = math.sqrt(
            summary["long_L_mean_squared"]
            + summary["clean_L_rms"] ** 2
            + design["screening"]["damping"] * update_norm**2
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                **summary,
                "effective_update_norm": update_norm,
                "oracle_dot": oracle_dot,
                "linear_target_gain": linear_gain,
                "geometric_ratio": summary["short_B"] / max(denominator, 1e-30),
            }
        )
    atomic_json(
        output,
        {
            "status": "optimizer-aware-one-step-screen-complete",
            "seed": seed,
            "rows": rows,
            "elapsed_seconds": time.time() - started,
            "candidate_count": len(shortlist),
        },
    )


def select_branch_shortlist(root: Path, source: Path) -> None:
    design = load_design(root, source)
    frames = []
    for seed in design["training"]["development_seeds"]:
        frame = pd.DataFrame(json.loads((root / "screening" / f"one-step-seed-{seed}.json").read_text())["rows"])
        frame["seed"] = seed
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby("candidate_id", as_index=False).agg(
        mean_geometric_ratio=("geometric_ratio", "mean"),
        minimum_short_B=("short_B", "min"),
        minimum_linear_target_gain=("linear_target_gain", "min"),
        mean_oracle_dot=("oracle_dot", "mean"),
        mean_long_L_q90=("long_L_q90", "mean"),
        mean_clean_L_rms=("clean_L_rms", "mean"),
    )
    floor = design["screening"]["absolute_linear_gain_floor"]
    summary["gain_valid"] = (summary.minimum_short_B > 0) & (summary.minimum_linear_target_gain >= floor)
    summary = summary.sort_values(
        ["gain_valid", "mean_geometric_ratio", "candidate_id"], ascending=[False, False, True], kind="stable"
    )
    count = int(design["screening"]["branch_shortlist"])
    selected = summary.head(count).candidate_id.tolist()
    selected = list(dict.fromkeys([*selected, *design["screening"]["baselines"]]))
    atomic_json(
        root / "branch_shortlist.json",
        {
            "status": "end-to-end-development-shortlist-complete",
            "candidates": selected,
            "ranking": summary.to_dict(orient="records"),
            "confirmation_contexts_accessed": False,
        },
    )


def run_branch(root: Path, source: Path, seed: int, candidate_id: str, dose: int, phase: str) -> None:
    design = load_design(root, source)
    allowed = design["training"][f"{phase}_seeds"]
    if seed not in allowed:
        raise ValueError(f"seed is not a frozen {phase} seed")
    if candidate_id not in design["candidates"] or dose not in design["training"]["doses"]:
        raise ValueError("candidate or dose is outside the design")
    if phase == "development":
        admitted = json.loads((root / "branch_shortlist.json").read_text())["candidates"]
        context_split = "oracle_fit"
    else:
        selection_path = root / "selection.json"
        if not selection_path.exists():
            raise PermissionError("confirmation is sealed until selection exists")
        admitted = json.loads(selection_path.read_text())["confirmation_candidates"]
        context_split = "confirmation"
    if candidate_id not in admitted:
        raise PermissionError("candidate was not admitted")
    directory = root / "branches" / phase / f"seed-{seed}"
    receipt_path = directory / f"{candidate_id}__dose-{dose}.json"
    array_path = receipt_path.with_suffix(".npz")
    if receipt_path.exists() or array_path.exists():
        raise FileExistsError(receipt_path)
    prefix_path, prefix_receipt, _, _ = _state_paths(root, seed)
    if sha256(prefix_path) != json.loads(prefix_receipt.read_text())["checkpoint_sha256"]:
        raise PermissionError("prefix checksum mismatch")
    _, branch = schedule_ids(design, seed)
    tokenizer = tokenizer_local()
    tokens = source_token_map(source, {x for batch in branch for x in batch}, tokenizer)
    device = experiment_device(design)
    model, optimizer, state = _load_model_state(prefix_path, device)
    losses, started = [], time.time()
    for index, batch_ids in enumerate(branch):
        values = [tokens[x] for x in batch_ids]
        if index >= len(branch) - dose:
            values[0] = design["candidates"][candidate_id]
        losses.append(
            train_step(
                model,
                optimizer,
                values,
                tokenizer.pad_token_id,
                device,
                int(state["completed_steps"]) + index + 1,
            )
        )
    modified = _predict_set(model, design, context_split, device)
    control_npz = directory / "control__dose-0.npz"
    control = _reference_from_npz(control_npz)
    summary = summarize_exact(control, modified, design["target_id"])
    summary["objective"] = _finite_objective(summary, design["oracle"]["weights"])
    atomic_npz(array_path, **{f"{key}_probs": value for key, value in modified.items()})
    atomic_json(
        receipt_path,
        {
            "status": "end-to-end-branch-complete",
            "phase": phase,
            "seed": seed,
            "candidate_id": candidate_id,
            "dose": dose,
            "device": str(device),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "elapsed_seconds": time.time() - started,
            "arrays_sha256": sha256(array_path),
            "matched_control_sha256": sha256(control_npz),
            **summary,
        },
    )


def select_retrieval(root: Path, source: Path) -> None:
    design = load_design(root, source)
    rows = []
    for candidate in json.loads((root / "branch_shortlist.json").read_text())["candidates"]:
        for dose in design["training"]["doses"]:
            receipts = [
                json.loads((root / "branches/development" / f"seed-{seed}" / f"{candidate}__dose-{dose}.json").read_text())
                for seed in design["training"]["development_seeds"]
            ]
            rows.append(
                {
                    "candidate_id": candidate,
                    "dose": dose,
                    "mean_objective": float(np.mean([x["objective"] for x in receipts])),
                    "mean_short_probability": float(np.mean([x["short_probability_modified"] for x in receipts])),
                    "mean_margin": float(np.mean([x["selectivity_margin"] for x in receipts])),
                    "mean_clean_rms": float(np.mean([x["clean_L_rms"] for x in receipts])),
                    "all_development_gate": all(gate_passes(x, design["success_gate"]) for x in receipts),
                }
            )
    rows.sort(key=lambda x: (-x["all_development_gate"], -x["mean_objective"], x["candidate_id"], x["dose"]))
    winner = rows[0]
    candidates = list(dict.fromkeys([winner["candidate_id"], *design["screening"]["baselines"]]))
    atomic_json(
        root / "selection.json",
        {
            "status": "development-selection-complete",
            "selected_candidate": winner["candidate_id"],
            "selected_dose": winner["dose"],
            "confirmation_candidates": candidates,
            "confirmation_doses": design["training"]["doses"],
            "ranking": rows,
            "confirmation_contexts_accessed": False,
        },
    )


def analyze(root: Path, source: Path) -> None:
    design = load_design(root, source)
    selection = json.loads((root / "selection.json").read_text())
    rows = []
    for candidate in selection["confirmation_candidates"]:
        for dose in selection["confirmation_doses"]:
            receipts = [
                json.loads((root / "branches/confirmation" / f"seed-{seed}" / f"{candidate}__dose-{dose}.json").read_text())
                for seed in design["training"]["confirmation_seeds"]
            ]
            rows.append(
                {
                    "candidate_id": candidate,
                    "dose": dose,
                    "seeds": len(receipts),
                    "short_probability_mean": float(np.mean([x["short_probability_modified"] for x in receipts])),
                    "short_B_mean": float(np.mean([x["short_B"] for x in receipts])),
                    "short_G_mean": float(np.mean([x["short_G"] for x in receipts])),
                    "short_N_mean": float(np.mean([x["short_N"] for x in receipts])),
                    "long_L_q90_mean": float(np.mean([x["long_L_q90"] for x in receipts])),
                    "clean_L_rms_mean": float(np.mean([x["clean_L_rms"] for x in receipts])),
                    "margin_mean": float(np.mean([x["selectivity_margin"] for x in receipts])),
                    "passing_seeds": sum(gate_passes(x, design["success_gate"]) for x in receipts),
                    "all_seed_gate": all(gate_passes(x, design["success_gate"]) for x in receipts),
                }
            )
    selected_rows = [x for x in rows if x["candidate_id"] == selection["selected_candidate"]]
    strong = any(x["all_seed_gate"] for x in selected_rows)
    atomic_json(
        root / "confirmation_analysis.json",
        {
            "status": "confirmation-analysis-complete",
            "selected_candidate": selection["selected_candidate"],
            "selected_development_dose": selection["selected_dose"],
            "strong_gate_passed": strong,
            "rows": rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--profile", choices=["micro", "full"], required=True)
    p.add_argument("--qualification", type=Path, required=True)
    for name in ("prefix", "control", "replay", "oracle", "gradient-screen", "one-step-screen"):
        p = sub.add_parser(name); p.add_argument("--seed", type=int, required=True)
    sub.add_parser("oracle-decision")
    sub.add_parser("gradient-shortlist")
    sub.add_parser("branch-shortlist")
    p = sub.add_parser("branch")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--dose", type=int, required=True)
    p.add_argument("--phase", choices=["development", "confirmation"], required=True)
    sub.add_parser("select")
    sub.add_parser("analyze")
    args = parser.parse_args()
    commands = {
        "prepare": lambda: prepare(args.root, args.source, args.profile, args.qualification),
        "prefix": lambda: train_prefix(args.root, args.source, args.seed),
        "control": lambda: train_control(args.root, args.source, args.seed),
        "replay": lambda: replay_control(args.root, args.source, args.seed),
        "oracle": lambda: solve_oracle(args.root, args.source, args.seed),
        "oracle-decision": lambda: decide_oracle(args.root, args.source),
        "gradient-screen": lambda: gradient_screen(args.root, args.source, args.seed),
        "gradient-shortlist": lambda: select_gradient_shortlist(args.root, args.source),
        "one-step-screen": lambda: one_step_screen(args.root, args.source, args.seed),
        "branch-shortlist": lambda: select_branch_shortlist(args.root, args.source),
        "branch": lambda: run_branch(args.root, args.source, args.seed, args.candidate, args.dose, args.phase),
        "select": lambda: select_retrieval(args.root, args.source),
        "analyze": lambda: analyze(args.root, args.source),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
