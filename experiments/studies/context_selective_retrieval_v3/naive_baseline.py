#!/usr/bin/env python3
"""Run a strength-matched unconstrained output-row baseline for the v3 oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from context_selective_retrieval_v3.geometry import summarize_exact
from context_selective_retrieval_v3.pipeline import (
    _context_values,
    _hidden_states,
    _load_model_state,
    _predict_set,
    _reference_from_npz,
)
from context_selective_retrieval_v3.training import predict


DESIGN_SHA256 = "5179a0ff25243d0a6db67a928bc697266e3bf404451b23d4661fbdcb870ebfe8"
QUALIFICATION_SHA256 = "36cfc826fb9895cc60ae1710498eb9d5417e55337e8708b6a32021aa0e918cee"
SEEDS = (52001, 52002)
MATCH_TOLERANCE = 1e-4
RECONSTRUCTION_TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def minimum_norm_naive_direction(sensitivity: torch.Tensor) -> torch.Tensor:
    """Return argmin ||d|| subject to sensitivity^T d = 1."""
    sensitivity = sensitivity.detach().to(device="cpu", dtype=torch.float64)
    squared_norm = torch.dot(sensitivity, sensitivity)
    if not torch.isfinite(squared_norm) or float(squared_norm) <= 0.0:
        raise FloatingPointError("short-context sensitivity has invalid norm")
    return sensitivity / squared_norm


def _logit(probability: float) -> float:
    return math.log(probability) - math.log1p(-probability)


def analytic_matching_step(
    base_probability: float,
    target_probability: float,
    short_hidden: torch.Tensor,
    realized_direction: torch.Tensor,
) -> float:
    """Match the target-only logit displacement after float32 row realization."""
    logit_delta = _logit(target_probability) - _logit(base_probability)
    realized_gain = float(
        torch.dot(
            short_hidden.detach().to(device="cpu", dtype=torch.float64),
            realized_direction.detach().to(device="cpu", dtype=torch.float64),
        )
    )
    if not math.isfinite(realized_gain) or realized_gain <= 0.0:
        raise FloatingPointError("naive direction has non-positive realized short-logit gain")
    return logit_delta / realized_gain


def _selected_oracle_row(receipt: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in receipt["steps"] if row["oracle_validation"] is not None]
    if len(rows) != 1 or rows[0]["linear_binary_step"] != receipt["selected_step"]:
        raise PermissionError("oracle receipt does not identify one selected validation row")
    return rows[0]


def _metric_reconstruction_error(
    stored: dict[str, Any], reconstructed: dict[str, Any]
) -> float:
    keys = (
        "short_probability_modified",
        "short_B",
        "long_L_q90",
        "clean_L_rms",
    )
    return max(abs(float(stored[key]) - float(reconstructed[key])) for key in keys)


def run_seed(root: Path, design: dict[str, Any], seed: int) -> dict[str, Any]:
    control_path = root / "controls" / f"seed-{seed}.pt"
    control_receipt_path = root / "controls" / f"seed-{seed}.json"
    oracle_dir = root / "oracle" / f"seed-{seed}"
    oracle_receipt_path = oracle_dir / "oracle.json"
    direction_path = oracle_dir / "direction.pt"
    control_receipt = json.loads(control_receipt_path.read_text())
    oracle_receipt = json.loads(oracle_receipt_path.read_text())
    if sha256(control_path) != control_receipt["checkpoint_sha256"]:
        raise PermissionError(f"control checkpoint mismatch for seed {seed}")
    if sha256(direction_path) != oracle_receipt["direction_sha256"]:
        raise PermissionError(f"protected direction mismatch for seed {seed}")

    device = torch.device("cuda")
    model, _, _ = _load_model_state(control_path, device)
    model.eval()
    target_id = int(design["target_id"])
    output_row = dict(model.named_parameters())["embed_out.weight"][target_id]
    base_row = output_row.detach().clone()
    fit_reference = _reference_from_npz(
        root / "branches" / "development" / f"seed-{seed}" / "control__dose-0.npz"
    )
    validation_reference = _reference_from_npz(
        root / "branches" / "development" / f"seed-{seed}" / "control_validation.npz"
    )
    short_context = _context_values(design, "oracle_fit")["short"]
    short_hidden = _hidden_states(model, short_context, device)[0]
    base_probability = float(fit_reference["short"][0, target_id])
    sensitivity = math.sqrt(base_probability * (1.0 - base_probability)) * short_hidden.cpu()
    naive_direction64 = minimum_norm_naive_direction(sensitivity)
    naive_delta32 = (base_row.cpu() + naive_direction64.float()) - base_row.cpu()

    protected_payload = torch.load(direction_path, map_location="cpu", weights_only=False)
    protected_direction = protected_payload["direction"].to(device)
    selected = _selected_oracle_row(oracle_receipt)
    protected_step = float(oracle_receipt["selected_step"])
    with torch.no_grad():
        output_row.copy_(base_row + protected_step * protected_direction.to(output_row.dtype))
    protected_modified = _predict_set(model, design, "oracle_validation", device)
    protected_summary = summarize_exact(validation_reference, protected_modified, target_id)
    reconstruction_error = _metric_reconstruction_error(
        selected["oracle_validation"], protected_summary
    )
    if reconstruction_error > RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(
            f"protected reconstruction failed for seed {seed}: {reconstruction_error}"
        )

    target_probability = protected_summary["short_probability_modified"]
    naive_step_initial = analytic_matching_step(
        base_probability, target_probability, short_hidden, naive_delta32
    )
    # The analytic value is refined against the actual float32 model row. Only the
    # designated short context participates in this scalar match.
    low, high = 0.0, max(2.0 * naive_step_initial, 1e-12)
    step_candidates: list[tuple[float, float]] = []
    for _ in range(64):
        step = 0.5 * (low + high)
        with torch.no_grad():
            output_row.copy_(
                base_row
                + step
                * naive_direction64.to(
                    device=output_row.device, dtype=output_row.dtype
                )
            )
        probability = float(predict(model, short_context, device)[0, target_id])
        step_candidates.append((abs(probability - target_probability), step))
        if probability < target_probability:
            low = step
        else:
            high = step
    naive_step = min(step_candidates)[1]
    with torch.no_grad():
        output_row.copy_(
            base_row
            + naive_step
            * naive_direction64.to(device=output_row.device, dtype=output_row.dtype)
        )
    naive_modified = _predict_set(model, design, "oracle_validation", device)
    naive_summary = summarize_exact(validation_reference, naive_modified, target_id)
    with torch.no_grad():
        output_row.copy_(base_row)

    match_error = abs(naive_summary["short_B"] - protected_summary["short_B"])
    matched = match_error <= MATCH_TOLERANCE
    long_improved = protected_summary["long_L_q90"] < naive_summary["long_L_q90"]
    clean_improved = protected_summary["clean_L_rms"] < naive_summary["clean_L_rms"]
    supports = matched and long_improved and clean_improved
    row = {
        "seed": seed,
        "protected_step": protected_step,
        "naive_step": naive_step,
        "direction_definition": "a_s / dot(a_s, a_s)",
        "match_metric": "short_B",
        "match_tolerance": MATCH_TOLERANCE,
        "short_B_match_error": match_error,
        "matched": matched,
        "protected_reconstruction_max_abs_error": reconstruction_error,
        "protected": protected_summary,
        "naive": naive_summary,
        "long_q90_ratio_naive_over_protected": (
            naive_summary["long_L_q90"] / protected_summary["long_L_q90"]
        ),
        "clean_rms_ratio_naive_over_protected": (
            naive_summary["clean_L_rms"] / protected_summary["clean_L_rms"]
        ),
        "supports_protection_value": supports,
    }
    print(json.dumps({"event": "seed_complete", **row}), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen matched baseline requires CUDA")
    design_path = root / "design" / "design.json"
    qualification_path = root / "design" / "qualification.json"
    if sha256(design_path) != DESIGN_SHA256:
        raise PermissionError("design digest does not match the frozen v3 study")
    if sha256(qualification_path) != QUALIFICATION_SHA256:
        raise PermissionError("qualification digest does not match the frozen v3 study")
    design = json.loads(design_path.read_text())
    if tuple(design["training"]["development_seeds"]) != SEEDS:
        raise PermissionError("development seed set changed")

    rows = [run_seed(root, design, seed) for seed in SEEDS]
    all_matched = all(row["matched"] for row in rows)
    all_support = all(row["supports_protection_value"] for row in rows)
    status = "supports-protection-value" if all_support else (
        "inconclusive-match-failure" if not all_matched else "does-not-support-protection-value"
    )
    payload = {
        "status": status,
        "hypothesis": (
            "At matched short binary displacement, the protected edit has smaller long-context "
            "q90 and clean-context RMS Fisher--Rao displacement than the naive edit on each seed."
        ),
        "decision_rule": (
            "Support requires |B_naive-B_protected| <= 1e-4 and strict improvement in both "
            "protected metrics on both frozen development seeds; match failure is inconclusive."
        ),
        "design_sha256": DESIGN_SHA256,
        "qualification_sha256": QUALIFICATION_SHA256,
        "seeds": list(SEEDS),
        "rows": rows,
    }
    atomic_json(output / "matched_naive_baseline.json", payload)
    csv_path = output / "matched_naive_baseline.csv"
    if csv_path.exists():
        raise FileExistsError(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "seed", "method", "step", "short_B", "long_L_q90", "clean_L_rms",
            "short_probability_modified", "short_B_match_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for method in ("protected", "naive"):
                summary = row[method]
                writer.writerow({
                    "seed": row["seed"],
                    "method": method,
                    "step": row[f"{method}_step"],
                    "short_B": summary["short_B"],
                    "long_L_q90": summary["long_L_q90"],
                    "clean_L_rms": summary["clean_L_rms"],
                    "short_probability_modified": summary["short_probability_modified"],
                    "short_B_match_error": row["short_B_match_error"],
                })
    print(json.dumps({"event": "complete", "status": status}), flush=True)


if __name__ == "__main__":
    main()
