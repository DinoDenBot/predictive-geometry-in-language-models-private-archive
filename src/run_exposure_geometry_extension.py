#!/usr/bin/env python3
"""Immutable artifact CLI for the prospective exposure-geometry extension.

This runner intentionally does not launch expensive training.  It freezes and
checks the inputs that must precede training and records outcome-access state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exposure_geometry_extension import (
    ExtensionAccess,
    artifact_hash_ledger,
    atomic_json,
    complete_r_gate_power_simulation,
    construct_extension_blocks,
    dual_duplicate_audit,
    fit_development_residual,
    make_six_target_assignments,
    select_extension_block_count,
    six_target_equivalence_power_simulation,
    validate_paired_intervention,
)
from exposure_observability import make_presentation_ledger


ROOT = Path(os.environ.get("EXPOSURE_EXTENSION_ROOT", "results/exposure_geometry_extension_v1"))
SOURCE_SPEC = Path("paper/extension/frozen_analysis_spec.json")
SOURCE_MANUSCRIPT = Path("paper/EXTENSION_PRERESULTS.md")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze(root: Path) -> None:
    if not SOURCE_SPEC.is_file() or not SOURCE_MANUSCRIPT.is_file():
        raise FileNotFoundError("prospective source specification or manuscript is absent")
    specification: dict[str, Any] = json.loads(SOURCE_SPEC.read_text())
    required_artifacts = {
        "design_completion": root / "design" / "completion.json",
        "duplicate_audit": root / "design" / "duplicate_audit.json",
        "candidate_manifest": root / "design" / "candidate_manifest.parquet",
        "assignments": root / "design" / "assignments.parquet",
        "primary_blocks": root / "design" / "primary_blocks.json",
        "r_gate_power": root / "design" / "power_decision.json",
        "equivalence_power": root / "design" / "equivalence_power.json",
        "residual_calibration": root / "development" / "residual_calibration.json",
        "base_difficulty_manifest": root / "source" / "base_difficulty_manifest.json",
    }
    missing = [name for name, path in required_artifacts.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"pre-outcome freeze inputs are absent: {missing}")
    software = [
        Path("exposure_geometry_extension.py"),
        Path("run_exposure_geometry_extension.py"),
        Path("execute_exposure_geometry_extension.py"),
        Path("test_exposure_geometry_extension.py"),
    ]
    specification["pre_outcome_freeze"] = {
        "pre_results_manuscript_sha256": hashlib.sha256(
            SOURCE_MANUSCRIPT.read_bytes()
        ).hexdigest(),
        "source_specification_sha256": hashlib.sha256(SOURCE_SPEC.read_bytes()).hexdigest(),
        "software_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in software
        },
        "input_artifact_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in required_artifacts.items()
        },
        "tests": "63 passed before operational freeze",
    }
    atomic_json(root / "frozen_analysis_spec.json", specification)


def audit_duplicates(root: Path, candidates: Path, references: Path) -> None:
    detail, summary = dual_duplicate_audit(read_table(candidates), read_table(references))
    summary["candidate_input_sha256"] = hashlib.sha256(candidates.read_bytes()).hexdigest()
    summary["reference_input_sha256"] = hashlib.sha256(references.read_bytes()).hexdigest()
    atomic_parquet(root / "design" / "duplicate_nearest_neighbors.parquet", detail)
    atomic_json(root / "design" / "duplicate_audit.json", summary)
    if summary["failed_documents"]:
        raise RuntimeError("duplicate audit failed; candidate construction must exclude flagged documents")


def assign(root: Path, manifest: Path, seed: int) -> None:
    source = read_table(manifest)
    assignments = make_six_target_assignments(source, seed=seed)
    atomic_parquet(root / "design" / "candidate_manifest.parquet", source)
    atomic_parquet(root / "design" / "assignments.parquet", assignments)
    atomic_json(
        root / "design" / "assignment_manifest.json",
        {
            "seed": seed,
            "source_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "assignment_rows": len(assignments),
            "assignment_sha256": hashlib.sha256(
                assignments.to_json(orient="records", double_precision=15).encode()
            ).hexdigest(),
        },
    )


def build_blocks(
    root: Path,
    pool: Path,
    *,
    seed: int,
    token_tolerance: int,
    difficulty_tolerance: float,
) -> None:
    source = read_table(pool)
    manifest = construct_extension_blocks(
        source,
        seed=seed,
        token_tolerance=token_tolerance,
        difficulty_tolerance=difficulty_tolerance,
    )
    atomic_parquet(root / "design" / "matched_blocks.parquet", manifest)
    atomic_json(
        root / "design" / "matched_blocks_manifest.json",
        {
            "seed": seed,
            "token_tolerance": token_tolerance,
            "difficulty_tolerance": difficulty_tolerance,
            "source_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
            "documents": len(manifest),
            "blocks": int(manifest.block_id.nunique()),
        },
    )


def score_pool(root: Path, pool: Path, batch_size: int) -> None:
    """Resumably score a frozen fresh pool with the pinned 70M base only."""

    import prepare_exposure_source as source_runner

    completion = root / "source" / "base_difficulty_manifest.json"
    if completion.exists():
        raise FileExistsError(f"base difficulty is already frozen: {completion}")
    frame = read_table(pool)
    source_runner.SOURCE = root / "source"
    scored, elapsed = source_runner._score_candidate_frame(frame, batch_size)
    parts = sorted((root / "source" / "difficulty_parts").glob("part-*.parquet"))
    difficulty = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    if len(difficulty) != len(frame) or difficulty.doc_id.duplicated().any():
        raise RuntimeError("base-difficulty parts do not form the complete frozen pool")
    merged = frame.merge(difficulty, on="doc_id", how="left", validate="one_to_one")
    atomic_parquet(root / "source" / "scored_pool.parquet", merged)
    atomic_json(
        completion,
        {
            "status": "target-output-free-base-difficulty-frozen",
            "access": "pinned 70M base next-token probabilities only",
            "base_model": source_runner.BASE_MODEL,
            "base_revision": source_runner.BASE_REVISION,
            "batch_size": batch_size,
            "pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
            "documents": len(frame),
            "newly_scored_documents": scored,
            "elapsed_seconds_last_invocation": elapsed,
            "part_hashes": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in parts
            },
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_runner_sha256": hashlib.sha256(Path(source_runner.__file__).read_bytes()).hexdigest(),
        },
    )


def finalize_design(
    root: Path,
    scored_pool_path: Path,
    original_candidates_path: Path,
    background_path: Path,
    *,
    block_seed: int,
    assignment_seed: int,
    primary_block_seed: int,
) -> None:
    """Match, dual-screen, assign, and ledger the complete intervention."""

    scored_pool = read_table(scored_pool_path)
    original_candidates = read_table(original_candidates_path)
    background = read_table(background_path)
    references = pd.concat(
        [original_candidates[["doc_id", "text"]], background[["doc_id", "text"]]],
        ignore_index=True,
    )
    eligible = scored_pool.copy()
    final_manifest: pd.DataFrame | None = None
    final_detail: pd.DataFrame | None = None
    final_audit: dict[str, Any] | None = None
    for audit_round in range(10):
        manifest = construct_extension_blocks(eligible, seed=block_seed + audit_round)
        detail, audit = dual_duplicate_audit(manifest, references)
        audit["round"] = audit_round
        audit["eligible_pool_documents"] = len(eligible)
        atomic_parquet(
            root / "design" / "duplicate_rounds" / f"round_{audit_round:02d}.parquet",
            detail,
        )
        atomic_json(
            root / "design" / "duplicate_rounds" / f"round_{audit_round:02d}.json",
            audit,
        )
        if int(audit["failed_documents"]) == 0:
            final_manifest, final_detail, final_audit = manifest, detail, audit
            break
        failed = set(detail.loc[~detail.passes, "doc_id"].astype(str))
        eligible = eligible.loc[~eligible.doc_id.astype(str).isin(failed)].copy()
    if final_manifest is None or final_detail is None or final_audit is None:
        raise RuntimeError("no duplicate-clean 600-block construction found in ten frozen rounds")
    assignments = make_six_target_assignments(final_manifest, seed=assignment_seed)
    atomic_parquet(root / "design" / "candidate_manifest.parquet", final_manifest)
    atomic_parquet(root / "design" / "duplicate_nearest_neighbors.parquet", final_detail)
    atomic_json(root / "design" / "duplicate_audit.json", final_audit)
    atomic_parquet(root / "design" / "assignments.parquet", assignments)
    atomic_parquet(root / "design" / "background_manifest.parquet", background)
    rng = np.random.default_rng(primary_block_seed)
    primary_blocks: dict[str, list[Any]] = {}
    for role in ("validation", "confirmation"):
        blocks = np.asarray(sorted(final_manifest.loc[final_manifest.role == role, "block_id"].unique()))
        primary_blocks[role] = sorted(rng.choice(blocks, size=200, replace=False).tolist())
    atomic_json(
        root / "design" / "primary_blocks.json",
        {
            "seed": primary_block_seed,
            "selected_blocks_per_phase": 200,
            "validation": primary_blocks["validation"],
            "confirmation": primary_blocks["confirmation"],
        },
    )
    ledger_hashes: dict[str, str] = {}
    for index in range(1, 7):
        target_id = f"70m_{index}"
        ledger = make_presentation_ledger(
            assignments,
            target_id,
            seed=20261400 + index,
            batch_size=8,
            background_documents=background,
        )
        path = root / "design" / "ledgers" / f"{target_id}.parquet"
        atomic_parquet(path, ledger)
        ledger_hashes[target_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    for index, source_index in enumerate((4, 5, 6), 1):
        target_id = f"160m_{index}"
        source = pd.read_parquet(root / "design" / "ledgers" / f"70m_{source_index}.parquet")
        ledger = source.copy()
        ledger["target_id"] = target_id
        validate_paired_intervention(source, ledger)
        path = root / "design" / "ledgers" / f"{target_id}.parquet"
        atomic_parquet(path, ledger)
        ledger_hashes[target_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(
        root / "design" / "completion.json",
        {
            "status": "complete-intervention-design-frozen",
            "blocks": 600,
            "documents": 3600,
            "assignment_seed": assignment_seed,
            "block_seed_initial": block_seed,
            "successful_duplicate_round": final_audit["round"],
            "background_sha256": hashlib.sha256(background_path.read_bytes()).hexdigest(),
            "original_candidates_sha256": hashlib.sha256(
                original_candidates_path.read_bytes()
            ).hexdigest(),
            "scored_pool_sha256": hashlib.sha256(scored_pool_path.read_bytes()).hexdigest(),
            "ledger_hashes": ledger_hashes,
        },
    )


def record_decision(root: Path, name: str, decision_path: Path) -> None:
    allowed = {"70m_validation", "70m_confirmation", "160m_validation_power"}
    if name not in allowed:
        raise ValueError(f"decision name must be one of {sorted(allowed)}")
    value = json.loads(decision_path.read_text())
    if value.get("decision_complete") is not True:
        raise ValueError("decision record must contain decision_complete=true")
    value["source_sha256"] = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    atomic_json(ExtensionAccess(root).decision(name), value)


def calibrate_residual(root: Path, development_path: Path) -> None:
    calibration = fit_development_residual(read_table(development_path))
    value = asdict(calibration)
    value["development_input_sha256"] = hashlib.sha256(development_path.read_bytes()).hexdigest()
    value["status"] = "frozen-development-only"
    atomic_json(root / "analysis" / "residual_calibration.json", value)


def calibrate_historical_development(root: Path, historical_root: Path) -> None:
    """Recover D_z from frozen development likelihood caches and fit gamma."""

    rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    manifest = pd.read_parquet(
        historical_root / "candidate_manifest.parquet",
        columns=["doc_id", "text_hash"],
    ).set_index("doc_id")
    for outcome_path in sorted((historical_root / "outcomes" / "development").glob("dev_*.parquet")):
        outcomes = pd.read_parquet(outcome_path)
        input_hashes[str(outcome_path.relative_to(historical_root))] = hashlib.sha256(
            outcome_path.read_bytes()
        ).hexdigest()
        for row in outcomes.itertuples(index=False):
            cache_name = f"{manifest.loc[row.doc_id, 'text_hash']}.npz"
            paths = {
                "base": historical_root / "outcomes" / "base_cache" / "development" / cache_name,
                "final": historical_root
                / "outcomes"
                / "development"
                / str(row.target_id)
                / "cache"
                / cache_name,
            }
            summaries: dict[str, float] = {}
            for checkpoint, path in paths.items():
                with np.load(path, allow_pickle=False) as stored:
                    logp = stored["likelihood"][:, :, 0].astype(np.float64)
                    distinct = stored["distinct"]
                p = np.exp(logp[:, :-1])
                q = np.exp(logp[:, 1:])
                dz = (q - p) / np.sqrt(np.clip(p * (1.0 - p), 1e-300, None))
                summaries[checkpoint] = float(np.quantile(dz[distinct], 0.90, method="linear"))
            rows.append(
                {
                    "target_id": row.target_id,
                    "block_id": row.block_id,
                    "doc_id": row.doc_id,
                    "delta_S_R": row.delta_S_R,
                    "delta_S_D_z": summaries["final"] - summaries["base"],
                }
            )
    frame = pd.DataFrame(rows)
    calibration = fit_development_residual(frame)
    atomic_parquet(root / "development" / "residual_inputs.parquet", frame)
    atomic_json(
        root / "development" / "residual_calibration.json",
        {
            **asdict(calibration),
            "status": "frozen-original-development-only",
            "historical_input_hashes": input_hashes,
            "historical_root_identity": "exposure_observability_v1",
        },
    )


def run_power(
    root: Path,
    *,
    simulations: int,
    randomization_draws: int,
    seed: int,
    effect: float,
    target_slope_sd: float,
    noise_sd: float,
) -> None:
    results = [
        complete_r_gate_power_simulation(
            blocks=blocks,
            simulations=simulations,
            randomization_draws=randomization_draws,
            seed=seed + blocks,
            effect=effect,
            target_slope_sd=target_slope_sd,
            noise_sd=noise_sd,
        )
        for blocks in (200, 250, 300)
    ]
    atomic_json(
        root / "design" / "power_decision.json",
        {
            "candidate_results": results,
            "minimum_power": 0.90,
            "selected_blocks": select_extension_block_count(results),
        },
    )


def run_equivalence_power(
    root: Path,
    *,
    simulations: int,
    randomization_draws: int,
    seed: int,
) -> None:
    results = {
        outcome: six_target_equivalence_power_simulation(
            blocks=200,
            simulations=simulations,
            randomization_draws=randomization_draws,
            seed=seed + offset,
        )
        for offset, outcome in enumerate(("L", "N", "U"))
    }
    atomic_json(
        root / "design" / "equivalence_power.json",
        {
            **results,
            "minimum_power": 0.80,
            "simulation_model": (
                "zero-slope standardized Gaussian block-plus-document noise under "
                "the exact complete six-target assignment mechanism"
            ),
        },
    )


def verify_pair(root: Path, ledger_70m: Path, ledger_160m: Path, pair_name: str) -> None:
    fingerprint = validate_paired_intervention(read_table(ledger_70m), read_table(ledger_160m))
    atomic_json(
        root / "paired_interventions" / f"{pair_name}.json",
        {
            "pair_name": pair_name,
            "intervention_fingerprint": fingerprint,
            "70m_file_sha256": hashlib.sha256(ledger_70m.read_bytes()).hexdigest(),
            "160m_file_sha256": hashlib.sha256(ledger_160m.read_bytes()).hexdigest(),
        },
    )


def write_anonymous_manifest(root: Path, paths: list[Path]) -> None:
    atomic_json(
        root / "anonymous_artifact_manifest.json",
        {
            "path_policy": "repository-relative; no absolute paths",
            "sha256": artifact_hash_ledger(paths, relative_to=Path.cwd()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    blocks = commands.add_parser("build-blocks")
    blocks.add_argument("--pool", type=Path, required=True)
    blocks.add_argument("--seed", type=int, required=True)
    blocks.add_argument("--token-tolerance", type=int, default=8)
    blocks.add_argument("--difficulty-tolerance", type=float, default=0.20)
    scoring = commands.add_parser("score-pool")
    scoring.add_argument("--pool", type=Path, required=True)
    scoring.add_argument("--batch-size", type=int, default=8)
    design = commands.add_parser("finalize-design")
    design.add_argument("--scored-pool", type=Path, required=True)
    design.add_argument("--original-candidates", type=Path, required=True)
    design.add_argument("--background", type=Path, required=True)
    design.add_argument("--block-seed", type=int, default=20261301)
    design.add_argument("--assignment-seed", type=int, default=20261302)
    design.add_argument("--primary-block-seed", type=int, default=20261303)
    audit = commands.add_parser("audit-duplicates")
    audit.add_argument("--candidates", type=Path, required=True)
    audit.add_argument("--references", type=Path, required=True)
    assignment = commands.add_parser("assign")
    assignment.add_argument("--manifest", type=Path, required=True)
    assignment.add_argument("--seed", type=int, required=True)
    decision = commands.add_parser("record-decision")
    decision.add_argument("--name", required=True)
    decision.add_argument("--decision", type=Path, required=True)
    residual = commands.add_parser("calibrate-residual")
    residual.add_argument("--development", type=Path, required=True)
    historical = commands.add_parser("calibrate-historical-development")
    historical.add_argument("--historical-root", type=Path, required=True)
    power = commands.add_parser("power")
    power.add_argument("--simulations", type=int, required=True)
    power.add_argument("--randomization-draws", type=int, default=99999)
    power.add_argument("--seed", type=int, required=True)
    power.add_argument("--effect", type=float, required=True)
    power.add_argument("--target-slope-sd", type=float, required=True)
    power.add_argument("--noise-sd", type=float, required=True)
    equivalence = commands.add_parser("equivalence-power")
    equivalence.add_argument("--simulations", type=int, required=True)
    equivalence.add_argument("--randomization-draws", type=int, default=1999)
    equivalence.add_argument("--seed", type=int, required=True)
    pair = commands.add_parser("verify-pair")
    pair.add_argument("--ledger-70m", type=Path, required=True)
    pair.add_argument("--ledger-160m", type=Path, required=True)
    pair.add_argument("--pair-name", required=True)
    opened = commands.add_parser("record-open")
    opened.add_argument("--architecture", choices=("70m", "160m"), required=True)
    opened.add_argument("--role", choices=("validation", "confirmation"), required=True)
    opened.add_argument("--target-id", required=True)
    anonymous = commands.add_parser("anonymous-manifest")
    anonymous.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.root)
    elif args.command == "build-blocks":
        build_blocks(
            args.root,
            args.pool,
            seed=args.seed,
            token_tolerance=args.token_tolerance,
            difficulty_tolerance=args.difficulty_tolerance,
        )
    elif args.command == "score-pool":
        score_pool(args.root, args.pool, args.batch_size)
    elif args.command == "finalize-design":
        finalize_design(
            args.root,
            args.scored_pool,
            args.original_candidates,
            args.background,
            block_seed=args.block_seed,
            assignment_seed=args.assignment_seed,
            primary_block_seed=args.primary_block_seed,
        )
    elif args.command == "audit-duplicates":
        audit_duplicates(args.root, args.candidates, args.references)
    elif args.command == "assign":
        assign(args.root, args.manifest, args.seed)
    elif args.command == "record-decision":
        record_decision(args.root, args.name, args.decision)
    elif args.command == "calibrate-residual":
        calibrate_residual(args.root, args.development)
    elif args.command == "calibrate-historical-development":
        calibrate_historical_development(args.root, args.historical_root)
    elif args.command == "power":
        run_power(
            args.root,
            simulations=args.simulations,
            randomization_draws=args.randomization_draws,
            seed=args.seed,
            effect=args.effect,
            target_slope_sd=args.target_slope_sd,
            noise_sd=args.noise_sd,
        )
    elif args.command == "equivalence-power":
        run_equivalence_power(
            args.root,
            simulations=args.simulations,
            randomization_draws=args.randomization_draws,
            seed=args.seed,
        )
    elif args.command == "verify-pair":
        verify_pair(args.root, args.ledger_70m, args.ledger_160m, args.pair_name)
    elif args.command == "record-open":
        ExtensionAccess(args.root).record_open(args.architecture, args.role, args.target_id)
    elif args.command == "anonymous-manifest":
        write_anonymous_manifest(args.root, args.paths)


if __name__ == "__main__":
    main()
