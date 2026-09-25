#!/usr/bin/env python3
"""Prepare the outcome-blind Study 3 corpus split and intervention design."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from exposure_geometry_extension import construct_extension_blocks
from exposure_observability import make_presentation_ledger
from study3 import (
    ALL_TARGETS,
    TARGET_SEEDS,
    duplicate_audit,
    make_assignments,
    select_primary_blocks,
    sha256_file,
    validate_target_seeds,
)


BACKGROUND_DOCUMENTS = 21_700
EFFECTIVE_BATCH_SIZE = 8
MAX_PREDICTED_TOKENS = 128
MIN_PREDICTED_TOKENS = 48


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


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


def token_lengths(texts: pd.Series, model_directory: Path, batch_size: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True, use_fast=True)
    lengths: list[int] = []
    values = texts.astype(str).tolist()
    for start in range(0, len(values), batch_size):
        encoded = tokenizer(
            values[start : start + batch_size],
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_PREDICTED_TOKENS + 1,
            return_length=True,
        )
        lengths.extend(max(0, int(length) - 1) for length in encoded["length"])
    return np.asarray(lengths, dtype=np.int64)


def split_source(
    root: Path,
    snapshot: Path,
    model_directory: Path,
    *,
    seed: int,
    batch_size: int,
) -> None:
    records = pd.read_parquet(snapshot)
    if records.doc_id.duplicated().any() or not records.eligible.astype(bool).all():
        raise ValueError("source snapshot must contain unique provenance-eligible records")
    records = records.copy()
    records["tokens"] = token_lengths(records.text, model_directory, batch_size)
    records = records.loc[
        records.tokens.between(MIN_PREDICTED_TOKENS, MAX_PREDICTED_TOKENS)
    ].reset_index(drop=True)
    if len(records) < BACKGROUND_DOCUMENTS + 5_000:
        raise RuntimeError(
            f"only {len(records)} token-eligible records; need at least "
            f"{BACKGROUND_DOCUMENTS + 5_000}"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    background = records.iloc[order[:BACKGROUND_DOCUMENTS]].copy().reset_index(drop=True)
    candidates = records.iloc[order[BACKGROUND_DOCUMENTS:]].copy().reset_index(drop=True)
    if set(background.doc_id) & set(candidates.doc_id):
        raise AssertionError("background/candidate identity overlap")
    atomic_parquet(root / "source" / "tokenized_eligible.parquet", records)
    atomic_parquet(root / "design" / "background_manifest.parquet", background)
    atomic_parquet(root / "source" / "candidate_pool_unscored.parquet", candidates)
    atomic_json(
        root / "source" / "split_manifest.json",
        {
            "status": "post-checkpoint-source-split-frozen",
            "seed": seed,
            "snapshot_sha256": sha256_file(snapshot),
            "model_artifact_directory_identity": model_directory.name,
            "eligible_records": len(records),
            "background_documents": len(background),
            "candidate_pool_documents": len(candidates),
            "minimum_predicted_tokens": MIN_PREDICTED_TOKENS,
            "maximum_predicted_tokens": MAX_PREDICTED_TOKENS,
            "background_sha256": sha256_file(root / "design" / "background_manifest.parquet"),
            "candidate_pool_sha256": sha256_file(root / "source" / "candidate_pool_unscored.parquet"),
        },
    )


def designate_pilot(
    root: Path, input_path: Path, *, seed: int, documents: int
) -> None:
    source = pd.read_parquet(input_path)
    if source.doc_id.duplicated().any() or not 1 <= documents < len(source):
        raise ValueError("pilot designation requires unique identities and a proper subset")
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(source), size=documents, replace=False)
    pilot = source.iloc[sorted(positions)].copy().reset_index(drop=True)
    eligible = source.loc[~source.doc_id.isin(pilot.doc_id)].copy().reset_index(drop=True)
    atomic_parquet(root / "feasibility" / "pilot_documents.parquet", pilot)
    atomic_parquet(root / "source" / "candidate_pool_analysis_eligible_unscored.parquet", eligible)
    atomic_json(
        root / "feasibility" / "pilot_designation.json",
        {
            "status": "outcome-blind-pilot-identities-frozen-and-analysis-excluded",
            "seed": seed,
            "pilot_documents": len(pilot),
            "analysis_eligible_candidate_pool": len(eligible),
            "input_sha256": sha256_file(input_path),
            "pilot_sha256": sha256_file(root / "feasibility" / "pilot_documents.parquet"),
            "eligible_pool_sha256": sha256_file(
                root / "source" / "candidate_pool_analysis_eligible_unscored.parquet"
            ),
            "held_target_outcomes_accessed": False,
        },
    )


def _namespace(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame[["doc_id", "text"]].copy()
    out["doc_id"] = prefix + out.doc_id.astype(str)
    return out


def finalize_design(
    root: Path,
    scored_pool_path: Path,
    pythia_reference_path: Path,
    older_reference_path: Path,
    *,
    block_seed: int,
    assignment_seed: int,
    primary_seed: int,
    ledger_seed: int,
    blocks_per_phase: int,
) -> None:
    validate_target_seeds()
    scored = pd.read_parquet(scored_pool_path)
    background = pd.read_parquet(root / "design" / "background_manifest.parquet")
    pythia = pd.read_parquet(pythia_reference_path)
    older = pd.read_parquet(older_reference_path)
    required = {"doc_id", "text", "source", "topic", "tokens", "baseline_difficulty"}
    if required - set(scored):
        raise ValueError(f"scored pool missing {sorted(required - set(scored))}")
    eligible = scored.copy()
    final_manifest = None
    final_detail = None
    final_summary = None
    for audit_round in range(20):
        manifest = construct_extension_blocks(eligible, seed=block_seed + audit_round)
        detail, summary = duplicate_audit(
            manifest[["doc_id", "text"]],
            background=_namespace(background, "study3-background:"),
            pythia_candidates=_namespace(pythia, "pythia:"),
            older_accessible=_namespace(older, "older-arxiv:"),
        )
        summary["audit_round"] = audit_round
        summary["eligible_pool_documents"] = len(eligible)
        atomic_parquet(root / "design" / "duplicate_rounds" / f"round_{audit_round:02d}.parquet", detail)
        atomic_json(root / "design" / "duplicate_rounds" / f"round_{audit_round:02d}.json", summary)
        if int(summary["failed_documents"]) == 0:
            final_manifest, final_detail, final_summary = manifest, detail, summary
            break
        failed = set(detail.loc[~detail.passes, "doc_id"].astype(str))
        eligible = eligible.loc[~eligible.doc_id.astype(str).isin(failed)].copy()
    if final_manifest is None or final_detail is None or final_summary is None:
        raise RuntimeError("no duplicate-clean 600-block design found in 20 frozen rounds")
    assignments = make_assignments(final_manifest, seed=assignment_seed)
    primary = select_primary_blocks(
        final_manifest, blocks_per_phase=blocks_per_phase, seed=primary_seed
    )
    atomic_parquet(root / "design" / "candidate_manifest.parquet", final_manifest)
    atomic_parquet(root / "design" / "duplicate_nearest_neighbors.parquet", final_detail)
    atomic_json(root / "design" / "duplicate_audit.json", final_summary)
    atomic_parquet(root / "design" / "assignments.parquet", assignments)
    atomic_json(
        root / "design" / "primary_blocks.json",
        {
            "seed": primary_seed,
            "selected_blocks_per_phase": blocks_per_phase,
            **primary,
        },
    )
    ledger_hashes: dict[str, str] = {}
    for offset, target_id in enumerate(ALL_TARGETS):
        ledger = make_presentation_ledger(
            assignments,
            target_id,
            seed=ledger_seed + offset,
            batch_size=EFFECTIVE_BATCH_SIZE,
            background_documents=background,
        )
        path = root / "design" / "ledgers" / f"{target_id}.parquet"
        atomic_parquet(path, ledger)
        ledger_hashes[target_id] = sha256_file(path)
    counts = {
        target: len(pd.read_parquet(root / "design" / "ledgers" / f"{target}.parquet"))
        for target in ALL_TARGETS
    }
    if set(counts.values()) != {40_300}:
        raise RuntimeError(f"intervention presentation counts changed: {counts}")
    atomic_json(
        root / "design" / "completion.json",
        {
            "status": "complete-study3-intervention-design-frozen",
            "blocks": 600,
            "documents": 3600,
            "primary_blocks_per_phase": blocks_per_phase,
            "block_seed_initial": block_seed,
            "assignment_seed": assignment_seed,
            "primary_seed": primary_seed,
            "ledger_seed_initial": ledger_seed,
            "successful_duplicate_round": final_summary["audit_round"],
            "scored_pool_sha256": sha256_file(scored_pool_path),
            "background_sha256": sha256_file(root / "design" / "background_manifest.parquet"),
            "candidate_manifest_sha256": sha256_file(root / "design" / "candidate_manifest.parquet"),
            "assignments_sha256": sha256_file(root / "design" / "assignments.parquet"),
            "ledger_hashes": ledger_hashes,
            "presentations_per_target": counts,
            "target_seeds": TARGET_SEEDS,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split-source")
    split.add_argument("--snapshot", type=Path, required=True)
    split.add_argument("--model-directory", type=Path, required=True)
    split.add_argument("--seed", type=int, default=20270110)
    split.add_argument("--batch-size", type=int, default=256)
    pilot = commands.add_parser("designate-pilot")
    pilot.add_argument("--input", type=Path, required=True)
    pilot.add_argument("--seed", type=int, default=20270120)
    pilot.add_argument("--documents", type=int, default=96)
    final = commands.add_parser("finalize")
    final.add_argument("--scored-pool", type=Path, required=True)
    final.add_argument("--pythia-reference", type=Path, required=True)
    final.add_argument("--older-reference", type=Path, required=True)
    final.add_argument("--block-seed", type=int, default=20270200)
    final.add_argument("--assignment-seed", type=int, default=20270250)
    final.add_argument("--primary-seed", type=int, default=20270260)
    final.add_argument("--ledger-seed", type=int, default=20270270)
    final.add_argument("--blocks-per-phase", type=int, default=200)
    args = parser.parse_args()
    if args.command == "split-source":
        split_source(
            args.root,
            args.snapshot,
            args.model_directory,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    elif args.command == "designate-pilot":
        designate_pilot(
            args.root,
            args.input,
            seed=args.seed,
            documents=args.documents,
        )
    else:
        finalize_design(
            args.root,
            args.scored_pool,
            args.pythia_reference,
            args.older_reference,
            block_seed=args.block_seed,
            assignment_seed=args.assignment_seed,
            primary_seed=args.primary_seed,
            ledger_seed=args.ledger_seed,
            blocks_per_phase=args.blocks_per_phase,
        )


if __name__ == "__main__":
    main()
