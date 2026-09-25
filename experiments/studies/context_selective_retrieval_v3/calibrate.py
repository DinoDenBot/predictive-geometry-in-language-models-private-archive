#!/usr/bin/env python3
"""Qualify projection solvers on the already-observed v2 development controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from context_selective_retrieval_v2.design import atomic_json, load_design, sha256
from context_selective_retrieval_v2.pipeline import (
    _context_values,
    _hidden_states,
    _load_model_state,
    _reference_from_npz,
    _state_paths,
    experiment_device,
)
from context_selective_retrieval_v3.numerics import (
    METHOD_ORDER,
    certification_thresholds,
    direction_metrics,
    qualify,
    rank_sensitivity,
    solve_direction,
)


def run_calibration(
    v2_root: Path,
    source: Path,
    output_root: Path,
    reconstruction_audit_path: Path | None = None,
) -> None:
    output = output_root / "calibration.json"
    if output.exists():
        raise FileExistsError(output)
    design = load_design(v2_root, source)
    if design["version"] != "context-selective-retrieval-v2" or design["profile"] != "full":
        raise ValueError("calibration requires the frozen full v2 design")
    steps = [float(value) for value in design["oracle"]["linear_binary_step_grid"]]
    device = experiment_device(design)
    results: list[dict[str, Any]] = []
    started = time.time()
    for seed in design["training"]["development_seeds"]:
        _, _, control_path, control_receipt_path = _state_paths(v2_root, int(seed))
        control_receipt = json.loads(control_receipt_path.read_text())
        if sha256(control_path) != control_receipt["checkpoint_sha256"]:
            raise PermissionError(f"control checksum mismatch for seed {seed}")
        model, _, _ = _load_model_state(control_path, device)
        model.eval()
        contexts = _context_values(design, "oracle_fit")
        protected_contexts = contexts["long"] + contexts["clean"]
        short_hidden = _hidden_states(model, contexts["short"], device)[0]
        protected_hidden = _hidden_states(model, protected_contexts, device)
        reference = _reference_from_npz(
            v2_root / "branches/development" / f"seed-{seed}" / "control__dose-0.npz"
        )
        target = int(design["target_id"])
        probability = float(reference["short"][0, target])
        output_row = dict(model.named_parameters())["embed_out.weight"][target].detach()
        method_rows: dict[str, Any] = {}
        for method in METHOD_ORDER:
            solution = solve_direction(method, short_hidden, protected_hidden, probability)
            method_rows[method] = direction_metrics(
                solution, short_hidden, protected_hidden, output_row, steps
            )
        results.append(
            {
                "seed": int(seed),
                "control_checkpoint_sha256": sha256(control_path),
                "base_short_probability": probability,
                "rank_sensitivity": rank_sensitivity(protected_hidden),
                "thresholds": certification_thresholds(protected_hidden),
                "methods": method_rows,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    decision = qualify(results)
    reconstruction_audit = None
    if reconstruction_audit_path is not None:
        reconstruction_audit = json.loads(reconstruction_audit_path.read_text())
        if not reconstruction_audit.get("passed"):
            raise PermissionError("reconstructed controls failed equivalence audit")
    payload = {
        "status": decision["status"],
        "purpose": "development-only numerical qualification; no confirmation context is read",
        "v2_design_sha256": sha256(v2_root / "design/design.json"),
        "source_sha256": sha256(source),
        "calibration_code_sha256": {
            name: sha256(Path(__file__).parent / name)
            for name in (
                "CALIBRATION_SPEC.md",
                "calibrate.py",
                "compare_reconstruction.py",
                "numerics.py",
                "test_numerics.py",
            )
        },
        "reconstruction_audit_sha256": (
            sha256(reconstruction_audit_path) if reconstruction_audit_path is not None else None
        ),
        "reconstruction_audit": reconstruction_audit,
        "calibration_seeds": [int(x) for x in design["training"]["development_seeds"]],
        "methods": list(METHOD_ORDER),
        "step_grid": steps,
        "seed_results": results,
        "decision": decision,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reconstruction-audit", type=Path)
    args = parser.parse_args()
    run_calibration(
        args.v2_root, args.source, args.output_root, args.reconstruction_audit
    )


if __name__ == "__main__":
    main()
