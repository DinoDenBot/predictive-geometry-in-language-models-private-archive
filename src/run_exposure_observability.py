#!/usr/bin/env python3
"""Standalone full-parameter Pythia dose-study runner.

The runner never imports or modifies prior randomized CATS artifacts.  It is a
stateful pipeline: build-blocks -> prepare -> train/acquire development ->
power/freeze -> acquire/analyze validation -> acquire/analyze confirmation.
Expensive commands refuse to overwrite immutable artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import t as student_t
from transformers import AutoModelForCausalLM, AutoTokenizer

from cats_identification import (
    CATS_V3_FEATURES,
    LAR2Config,
    cats_v3_document_features,
    context_length_grid,
    fisher_rao_alr,
    fisher_rao_alr_from_roots,
)
from exposure_observability import (
    CHECKPOINT_FRACTIONS,
    DOSES,
    DOSE_D,
    PhaseAccess,
    add_checkpoint_changes,
    construct_matched_blocks,
    context_specificity_analysis,
    deterministic_disruption,
    design_based_interval,
    document_target_fixed_effect_sensitivity,
    donor_coverage_preflight,
    donor_support_descriptives,
    dose_progression,
    equivalence_power_simulation,
    frozen_document_summaries,
    grouped_auc_interval,
    make_latin_assignments,
    make_presentation_ledger,
    normalized_sha256,
    power_gate_simulation,
    presentation_balance,
    randomization_test,
    saturated_dose_summary,
    select_powered_block_count,
    standardize_outcome,
    target_slopes,
    validate_candidate_manifest,
    validation_gate,
)
from run_cats_agnews import _device, atomic_npz


ROOT = Path(
    os.environ.get("EXPOSURE_STUDY_ROOT", "results/exposure_observability_v1")
).expanduser()
BASE_MODEL = "EleutherAI/pythia-70m-deduped"
BASE_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
TARGETS = {
    "dev_1": {"seed": 20261001, "phase": "development"},
    "dev_2": {"seed": 20261002, "phase": "development"},
    "val_1": {"seed": 20261003, "phase": "validation"},
    "val_2": {"seed": 20261004, "phase": "validation"},
    "val_3": {"seed": 20261005, "phase": "validation"},
    "con_1": {"seed": 20261006, "phase": "confirmation"},
    "con_2": {"seed": 20261007, "phase": "confirmation"},
    "con_3": {"seed": 20261008, "phase": "confirmation"},
}
ASSIGNMENT_SEED = 20260930
HELD_ANALYSIS_SEEDS = {"validation": 20261020, "confirmation": 20261021}
CONTEXT_ANALYSIS_SEEDS = {"validation": 20261022, "confirmation": 20261023}
CATS_FIT_SEED = 20261003
MODEL_MAX_TOKENS = 128
TRAIN_BATCH_SIZE = 8
TRAIN_LR = 1e-5
TRAIN_WEIGHT_DECAY = 0.01
TRAIN_ADAM_EPS = 1e-4
TRAIN_WARMUP_STEPS = 100
GRADIENT_CLIP = 1.0
FEATURE_COLUMNS = CATS_V3_FEATURES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_provenance() -> dict[str, Any]:
    files = [
        Path(__file__),
        Path(__file__).with_name("exposure_observability.py"),
        Path(__file__).with_name("cats_identification.py"),
        Path(__file__).with_name("run_cats_agnews.py"),
        Path(__file__).with_name("prepare_exposure_source.py"),
    ]
    packages = (
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    )
    return {
        "source_sha256": {str(path.resolve()): sha256(path) for path in files},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: importlib.metadata.version(package) for package in packages
        },
        "git_commit": None,
        "git_note": "workspace is not a dedicated Git repository; source-file hashes are authoritative",
    }


def verify_frozen_analysis_software(specification: dict[str, Any]) -> None:
    expected = specification.get("analysis_software_provenance")
    if expected is None or software_provenance() != expected:
        raise PermissionError("analysis software or environment changed after development freeze")


def verify_feasibility_chain() -> None:
    chains = (
        (
            ROOT / "feasibility" / "decision.json",
            ROOT / "feasibility",
            "passing_attempt_sha256",
        ),
        (
            ROOT / "feasibility" / "resume_replay" / "decision.json",
            ROOT / "feasibility" / "resume_replay",
            "passing_attempt_sha256",
        ),
    )
    for decision_path, root, hash_key in chains:
        if not decision_path.exists():
            raise FileNotFoundError(f"missing feasibility decision: {decision_path}")
        decision = json.loads(decision_path.read_text())
        attempt = root / decision["passing_attempt"]
        if not attempt.exists() or sha256(attempt) != decision[hash_key]:
            raise PermissionError(f"feasibility passing attempt changed: {attempt}")
        report = json.loads(attempt.read_text())
        checkpoint_value = report.get("checkpoint_sha256")
        if checkpoint_value is not None:
            checkpoint = attempt.parent / "one_step_checkpoint.pt"
            if not checkpoint.exists() or sha256(checkpoint) != checkpoint_value:
                raise PermissionError("full-parameter feasibility checkpoint changed")
        reference_value = report.get("reference_checkpoint_sha256")
        if reference_value is not None:
            reference = attempt.parent / "resumed_step2_checkpoint.pt"
            if not reference.exists() or sha256(reference) != reference_value:
                raise PermissionError("resume-feasibility reference checkpoint changed")


def atomic_json(path: Path, value: Any, *, overwrite: bool = False) -> None:  # noqa: ANN401
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:  # noqa: ANN401
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_joblib(path: Path, value: Any) -> None:  # noqa: ANN401
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".joblib", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        if not value.endswith("\n"):
            handle.write("\n")
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def record_access(path: Path, phase: str, target_id: str, **extra: Any) -> None:  # noqa: ANN401
    frozen_hash = sha256(PhaseAccess(ROOT).frozen_spec)
    if path.exists():
        prior = json.loads(path.read_text())
        if (
            prior.get("phase") != phase
            or prior.get("target_id") != target_id
            or prior.get("frozen_spec_sha256") != frozen_hash
        ):
            raise PermissionError("existing access log does not match this frozen acquisition")
        return
    atomic_json(
        path,
        {
            "phase": phase,
            "target_id": target_id,
            "opened_unix_time": time.time(),
            "frozen_spec_sha256": frozen_hash,
            "consumption_rule": "opening is irreversible; operational retry must keep identities and spec",
            **extra,
        },
    )


def verify_design_artifacts(target_id: str | None = None) -> dict[str, Any]:
    """Refuse execution if any frozen design input changed after preparation."""
    path = ROOT / "design_manifest.json"
    if not path.exists():
        raise FileNotFoundError("run prepare before using frozen design artifacts")
    manifest = json.loads(path.read_text())
    if manifest.get("software_provenance") != software_provenance():
        raise PermissionError("training or acquisition software changed after design freeze")
    checks = {
        ROOT / "candidate_manifest.parquet": manifest["candidate_manifest_sha256"],
        ROOT / "assignments.parquet": manifest["assignments_sha256"],
        ROOT / "background_manifest.parquet": manifest[
            "background_manifest_sha256"
        ],
        ROOT / "feasibility" / "decision.json": manifest[
            "feasibility_decision_sha256"
        ],
        ROOT / "feasibility" / "resume_replay" / "decision.json": manifest[
            "resume_replay_sha256"
        ],
    }
    if target_id is not None:
        if target_id not in manifest["ledgers"]:
            raise ValueError(f"target is absent from frozen design: {target_id}")
        checks[ROOT / "ledgers" / f"{target_id}.parquet"] = manifest["ledgers"][
            target_id
        ]["ledger"]
        checks[ROOT / "ledgers" / f"{target_id}_balance.parquet"] = manifest[
            "ledgers"
        ][target_id]["balance"]
    changed = [str(artifact) for artifact, expected in checks.items() if not artifact.exists() or sha256(artifact) != expected]
    if changed:
        raise PermissionError(f"frozen design artifact changed or vanished: {changed}")
    verify_feasibility_chain()
    return manifest


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("table must be Parquet, JSONL, or CSV")


def tokenizer_local() -> Any:  # noqa: ANN401
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _last_hidden_and_head(
    model: Any, padded: torch.Tensor, attention: torch.Tensor  # noqa: ANN401
) -> tuple[torch.Tensor, Any]:  # noqa: ANN401
    """Study-local output-head adapter for pinned and legacy Transformers APIs."""
    if hasattr(model, "transformer"):
        head = getattr(model, "lm_head", None)
        if head is not None:
            hidden = model.transformer(
                input_ids=padded, attention_mask=attention, use_cache=False
            ).last_hidden_state
            return hidden, head
    if hasattr(model, "gpt_neox"):
        head = getattr(model, "lm_head", None)
        if head is None:
            head = getattr(model, "embed_out", None)
        if head is not None:
            hidden = model.gpt_neox(
                input_ids=padded, attention_mask=attention, use_cache=False
            ).last_hidden_state
            return hidden, head
    if hasattr(model, "model"):
        head = getattr(model, "lm_head", None)
        if head is not None:
            hidden = model.model(
                input_ids=padded, attention_mask=attention, use_cache=False
            ).last_hidden_state
            return hidden, head
    raise TypeError(f"unsupported causal-LM architecture: {type(model).__name__}")


def build_blocks(pool_path: Path, seed: int, token_tolerance: int, difficulty_tolerance: float) -> None:
    pool = read_table(pool_path)
    prior_hashes: set[str] = set()
    prior_sources = []
    for path in sorted(Path("results").glob("**/population.parquet")):
        if ROOT in path.parents:
            continue
        try:
            prior = pd.read_parquet(path, columns=["text_hash"])
        except (KeyError, ValueError):
            continue
        prior_hashes.update(prior.text_hash.astype(str))
        prior_sources.append(str(path))
    if prior_hashes:
        hashes = pool.text.astype(str).map(lambda text: hashlib.sha256(text.encode()).hexdigest())
        pool = pool.loc[~hashes.isin(prior_hashes)].reset_index(drop=True)
    blocks = construct_matched_blocks(
        pool,
        seed=seed,
        token_tolerance=token_tolerance,
        difficulty_tolerance=difficulty_tolerance,
    )
    path = ROOT / "candidate_manifest.parquet"
    write_parquet(path, blocks)
    atomic_json(
        ROOT / "candidate_manifest.json",
        {
            "status": "target-output-free-candidate-blocks-frozen",
            "source_pool": str(pool_path.resolve()),
            "source_pool_sha256": sha256(pool_path),
            "seed": seed,
            "token_tolerance": token_tolerance,
            "difficulty_tolerance": difficulty_tolerance,
            "near_duplicate_rule": "word-5-shingle Jaccard < 0.80",
            "prior_population_hashes_excluded": len(prior_hashes),
            "prior_population_sources": prior_sources,
            "blocks": 700,
            "documents": 4200,
            "manifest_sha256": sha256(path),
        },
    )


def prepare(candidate_path: Path, background_path: Path) -> None:
    feasibility_path = ROOT / "feasibility" / "decision.json"
    if not feasibility_path.exists() or not json.loads(feasibility_path.read_text()).get(
        "feasibility_passed", False
    ):
        raise PermissionError(
            "freeze a passing target-output-free full-parameter feasibility decision first"
        )
    resume_path = ROOT / "feasibility" / "resume_replay" / "decision.json"
    if not resume_path.exists() or not json.loads(resume_path.read_text()).get(
        "resume_replay_passed", False
    ):
        raise PermissionError("a passing checkpoint-resume replay gate is required")
    verify_feasibility_chain()
    destination = ROOT / "candidate_manifest.parquet"
    if destination.exists() and candidate_path.resolve() != destination.resolve():
        raise FileExistsError(f"candidate manifest is already frozen: {destination}")
    candidates = validate_candidate_manifest(pd.read_parquet(candidate_path))
    if not destination.exists():
        write_parquet(destination, candidates)
    background = read_table(background_path)
    if {"doc_id", "text"} - set(background):
        raise ValueError("background manifest requires doc_id and text")
    if background.doc_id.duplicated().any():
        raise ValueError("background doc_id values must be unique")
    candidate_exact = set(candidates.text_hash)
    candidate_normalized = set(candidates.normalized_hash)
    background = background.copy()
    background["text_hash"] = background.text.astype(str).map(
        lambda text: hashlib.sha256(text.encode()).hexdigest()
    )
    background["normalized_hash"] = background.text.astype(str).map(normalized_sha256)
    if (
        background.text_hash.duplicated().any()
        or background.normalized_hash.duplicated().any()
        or bool(candidate_exact & set(background.text_hash))
        or bool(candidate_normalized & set(background.normalized_hash))
    ):
        raise ValueError("background has internal or candidate-document duplicates")
    background_destination = ROOT / "background_manifest.parquet"
    write_parquet(background_destination, background)
    assignments = make_latin_assignments(candidates, list(TARGETS), seed=ASSIGNMENT_SEED)
    assignment_path = ROOT / "assignments.parquet"
    write_parquet(assignment_path, assignments)
    design_diagnostics = target_blind_design_diagnostics(candidates, assignments)
    ledger_hashes = {}
    for target_id, target in TARGETS.items():
        ledger = make_presentation_ledger(
            assignments,
            target_id,
            seed=int(target["seed"]),
            batch_size=TRAIN_BATCH_SIZE,
            background_documents=background,
        )
        ledger_path = ROOT / "ledgers" / f"{target_id}.parquet"
        write_parquet(ledger_path, ledger)
        steps = int(ledger.optimizer_step.max()) + 1
        rates = TRAIN_LR * np.minimum(
            1.0, (np.arange(steps, dtype=float) + 1.0) / TRAIN_WARMUP_STEPS
        )
        balance = presentation_balance(ledger, rates)
        treated_balance = balance.loc[balance.K > 0]
        mean_rate = float(
            np.average(
                treated_balance.weighted_learning_rate,
                weights=treated_balance.presentations,
            )
        )
        maximum_rate_deviation = float(
            np.max(np.abs(treated_balance.weighted_learning_rate - mean_rate)) / mean_rate
        )
        maximum_timing_deviation = float(
            np.max(np.abs(treated_balance.mean_progress - 0.5))
        )
        if maximum_rate_deviation > 0.02 or maximum_timing_deviation > 0.01:
            raise RuntimeError(
                f"{target_id} ledger fails frozen timing/LR balance tolerances"
            )
        balance["maximum_positive_dose_lr_relative_deviation"] = maximum_rate_deviation
        balance["maximum_positive_dose_timing_deviation"] = maximum_timing_deviation
        balance_path = ROOT / "ledgers" / f"{target_id}_balance.parquet"
        write_parquet(balance_path, balance)
        ledger_hashes[target_id] = {"ledger": sha256(ledger_path), "balance": sha256(balance_path)}
    atomic_json(
        ROOT / "design_manifest.json",
        {
            "status": "allocation-and-presentation-ledgers-frozen-before-training",
            "artifact_root": str(ROOT.resolve()),
            "estimand": "document-level causal response to exact-document budget allocation during full-parameter continued pretraining",
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "targets": TARGETS,
            "doses": DOSES.tolist(),
            "block_exposure_total": 31,
            "assignment_seed": ASSIGNMENT_SEED,
            "candidate_manifest_sha256": sha256(destination),
            "assignments_sha256": sha256(assignment_path),
            "background_manifest_sha256": sha256(background_destination),
            "feasibility_decision_sha256": sha256(feasibility_path),
            "resume_replay_sha256": sha256(resume_path),
            "feasibility_operational_adjustment": json.loads(
                feasibility_path.read_text()
            )["operational_adjustment"],
            "resume_replay_operational_adjustment": json.loads(
                resume_path.read_text()
            )["operational_adjustment"],
            "ledgers": ledger_hashes,
            "training": {
                "optimizer": "AdamW",
                "learning_rate": TRAIN_LR,
                "weight_decay": TRAIN_WEIGHT_DECAY,
                "adam_epsilon": TRAIN_ADAM_EPS,
                "adam_foreach": False,
                "warmup_steps": TRAIN_WARMUP_STEPS,
                "batch_size": TRAIN_BATCH_SIZE,
                "gradient_clip": GRADIENT_CLIP,
                "full_parameter": True,
                "checkpoint_fractions": CHECKPOINT_FRACTIONS,
            },
            "counterfactual": "K=0 retains all 31 matched-block presentations, allocated to the five peers",
            "target_blind_design_diagnostics": design_diagnostics,
            "software_provenance": software_provenance(),
        },
    )


def target_blind_design_diagnostics(
    candidates: pd.DataFrame, assignments: pd.DataFrame
) -> dict[str, Any]:
    """Fit development-only text/metadata controls without any model outcomes."""
    ordered = candidates.reset_index(drop=True)
    document_position = {
        doc_id: position for position, doc_id in enumerate(ordered.doc_id)
    }
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=12_000,
        sublinear_tf=True,
    )
    development_documents = ordered.role.to_numpy() == "development"
    vectorizer.fit(ordered.loc[development_documents, "text"])
    text_matrix = vectorizer.transform(ordered.text)
    train = assignments.role.to_numpy() == "development"
    train_positions = assignments.loc[train, "doc_id"].map(document_position).to_numpy()
    classifier = LogisticRegression(
        C=0.03,
        solver="lbfgs",
        max_iter=5_000,
        random_state=ASSIGNMENT_SEED,
    ).fit(text_matrix[train_positions], assignments.loc[train, "K"])
    categorical = ordered[["source", "topic"]].astype(str).copy()
    categorical_names = ["source stratum", "topic"]
    if "domain" in ordered:
        categorical["domain_hash_bucket"] = ordered.domain.astype(str).map(
            lambda value: int.from_bytes(
                hashlib.sha256(value.encode()).digest()[:4], "big"
            )
            % 128
        )
        categorical_names.append("128-bin full-host domain hash")
    metadata = pd.get_dummies(categorical, dtype=float)
    numeric_columns = [
        ordered.tokens.to_numpy(dtype=float),
        ordered.baseline_difficulty.to_numpy(dtype=float),
        ordered.text.astype(str).str.len().to_numpy(dtype=float),
        ordered.text.astype(str).map(lambda text: len(zlib.compress(text.encode()))).to_numpy(dtype=float),
    ]
    metadata_names = [
        "token count",
        "pinned-base difficulty",
        "character count",
        "compressed byte count",
    ]
    if "domain" in ordered:
        domain_counts = ordered.domain.astype(str).map(
            ordered.domain.astype(str).value_counts()
        )
        numeric_columns.append(np.log1p(domain_counts.to_numpy(dtype=float)))
        metadata_names.append("log full-host domain frequency")
    for column in ("language_score", "edu_score"):
        if column in ordered:
            numeric_columns.append(ordered[column].to_numpy(dtype=float))
            metadata_names.append(column)
    if "date" in ordered:
        dates = pd.to_datetime(ordered.date, errors="coerce", utc=True)
        date_values = dates.astype("int64").to_numpy(dtype=float) / 1e18
        finite = np.isfinite(date_values) & (date_values > -8)
        date_values[~finite] = (
            float(np.median(date_values[finite])) if np.any(finite) else 0.0
        )
        numeric_columns.append(date_values)
        metadata_names.append("crawl date")
    numeric = np.column_stack((*numeric_columns, metadata.to_numpy(dtype=float)))
    numeric = np.column_stack((np.ones(len(numeric)), numeric))
    coefficients = np.linalg.lstsq(
        numeric[train_positions], assignments.loc[train, "d"], rcond=None
    )[0]
    results = {}
    for role in ("validation", "confirmation"):
        selected = assignments.role.to_numpy() == role
        positions = assignments.loc[selected, "doc_id"].map(document_position).to_numpy()
        labels = assignments.loc[selected, "K"].to_numpy()
        probabilities = classifier.predict_proba(text_matrix[positions])
        expected_d = probabilities @ DOSE_D
        actual_d = assignments.loc[selected, "d"].to_numpy(dtype=float)
        metadata_prediction = numeric[positions] @ coefficients
        audit = assignments.loc[selected, ["doc_id", "K", "target_id"]].merge(
            ordered,
            on="doc_id",
            validate="many_to_one",
        )
        audit["characters"] = audit.text.astype(str).str.len()
        audit["compressed_bytes"] = audit.text.astype(str).map(
            lambda text: len(zlib.compress(text.encode()))
        )
        audit["tokens_per_character"] = audit.tokens / audit.characters.clip(lower=1)
        numeric_audit_columns = (
            "tokens",
            "baseline_difficulty",
            "characters",
            "compressed_bytes",
            "tokens_per_character",
        )
        dose_summaries = (
            audit.groupby("K")[list(numeric_audit_columns)]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )
        categorical_imbalance = {}
        for column in ("source", "topic", "domain"):
            if column not in audit:
                continue
            proportions = pd.crosstab(
                audit.K, audit[column].astype(str), normalize="index"
            )
            categorical_imbalance[column] = float(
                proportions.max(axis=0).sub(proportions.min(axis=0)).max()
            )
        results[role] = {
            "documents_by_targets": int(np.sum(selected)),
            "tfidf_multiclass_accuracy": float(
                np.mean(classifier.classes_[np.argmax(probabilities, axis=1)] == labels)
            ),
            "tfidf_macro_ovr_auc": float(
                roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
            ),
            "tfidf_expected_d_correlation": float(
                np.corrcoef(expected_d, actual_d)[0, 1]
            ),
            "metadata_predicted_d_correlation": float(
                np.corrcoef(metadata_prediction, actual_d)[0, 1]
            ),
            "dose_group_numeric_means": dose_summaries,
            "maximum_category_proportion_range_across_doses": categorical_imbalance,
            "truncated_documents": int(np.sum(audit.tokens > MODEL_MAX_TOKENS)),
        }
    return {
        "status": "development-fitted-target-output-free-controls-complete",
        "training_role": "development",
        "held_roles": results,
        "text_features": "TF-IDF word unigrams/bigrams, maximum 12,000",
        "metadata_features": metadata_names + categorical_names,
        "uses_target_outputs": False,
        "causal_validity_note": "exact randomization inference remains valid regardless of realized diagnostic values",
    }


def _batch_from_events(events: pd.DataFrame, texts: dict[Any, str], tokenizer: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401
    encoded = [
        tokenizer(
            texts[row.doc_id],
            add_special_tokens=True,
            truncation=True,
            max_length=MODEL_MAX_TOKENS + 1,
        )["input_ids"]
        for row in events.itertuples(index=False)
    ]
    maximum = max(map(len, encoded))
    input_ids = torch.full((len(encoded), maximum), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention[row, : len(ids)] = 1
    return input_ids, attention


def train(target_id: str) -> None:
    if target_id not in TARGETS:
        raise ValueError(f"unknown target {target_id}")
    completion = ROOT / "targets" / target_id / "completion.json"
    if completion.exists():
        raise FileExistsError(f"target is already complete: {completion}")
    verify_design_artifacts(target_id)
    feasibility_path = ROOT / "feasibility" / "decision.json"
    if not feasibility_path.exists() or not json.loads(feasibility_path.read_text()).get(
        "feasibility_passed", False
    ):
        raise PermissionError(
            "a passing target-output-free full-parameter feasibility check is required"
        )
    seed = int(TARGETS[target_id]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    candidates = pd.read_parquet(ROOT / "candidate_manifest.parquet")
    background = pd.read_parquet(ROOT / "background_manifest.parquet")
    texts = pd.concat([candidates[["doc_id", "text"]], background[["doc_id", "text"]]]) \
        .set_index("doc_id").text.to_dict()
    ledger = pd.read_parquet(ROOT / "ledgers" / f"{target_id}.parquet")
    tokenizer = tokenizer_local()
    device = _device()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True
    ).to(device)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = True
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=TRAIN_LR,
        weight_decay=TRAIN_WEIGHT_DECAY,
        eps=TRAIN_ADAM_EPS,
        foreach=False,
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    if optimizer_parameter_ids != model_parameter_ids:
        raise RuntimeError("full-parameter optimizer does not cover the entire model")
    trainable_parameter_count = sum(parameter.numel() for parameter in parameters)
    steps = int(ledger.optimizer_step.max()) + 1
    checkpoint_steps = {max(1, int(np.ceil(fraction * steps))): fraction for fraction in CHECKPOINT_FRACTIONS}
    existing_checkpoints = sorted((ROOT / "targets" / target_id).glob("checkpoint-*.pt"))
    start_step = 0
    losses: list[float] = []
    resumed_from = None
    if existing_checkpoints:
        resumed_from = existing_checkpoints[-1]
        state = torch.load(resumed_from, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        losses = [float(value) for value in state["losses"]]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
    checkpoint_hashes = {
        f"{int(path.stem.split('-')[-1]) / 100:.2f}": sha256(path)
        for path in existing_checkpoints
    }
    started = time.time()
    model.train()
    for step, events in ledger.groupby("optimizer_step", sort=True):
        completed_step = int(step) + 1
        if completed_step <= start_step:
            continue
        learning_rate = TRAIN_LR * min(1.0, completed_step / TRAIN_WARMUP_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        input_ids, attention = _batch_from_events(events, texts, tokenizer, device)
        labels = input_ids.clone()
        labels[attention == 0] = -100
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, attention_mask=attention, labels=labels, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if completed_step in checkpoint_steps:
            fraction = checkpoint_steps[completed_step]
            path = ROOT / "targets" / target_id / f"checkpoint-{int(fraction * 100):03d}.pt"
            atomic_torch(
                path,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "warmup": {
                        "completed_step": completed_step,
                        "warmup_steps": TRAIN_WARMUP_STEPS,
                        "learning_rate": learning_rate,
                    },
                    "step": completed_step,
                    "fraction": fraction,
                    "losses": losses,
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),
                },
            )
            checkpoint_hashes[f"{fraction:.2f}"] = sha256(path)
        if completed_step % 100 == 0:
            print(f"{target_id} step {completed_step}/{steps} loss={losses[-1]:.6f}", flush=True)
    expected_checkpoint_keys = {f"{fraction:.2f}" for fraction in CHECKPOINT_FRACTIONS}
    if set(checkpoint_hashes) != expected_checkpoint_keys:
        raise RuntimeError("training did not produce all four frozen checkpoint states")
    for fraction, expected_hash in checkpoint_hashes.items():
        checkpoint = ROOT / "targets" / target_id / f"checkpoint-{int(float(fraction) * 100):03d}.pt"
        if not checkpoint.exists() or sha256(checkpoint) != expected_hash:
            raise RuntimeError("checkpoint changed before training completion was recorded")
    atomic_json(
        completion,
        {
            "status": "full-parameter-continued-pretraining-complete",
            "target_id": target_id,
            "target_phase": TARGETS[target_id]["phase"],
            "seed": seed,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "full_parameter": True,
            "trainable_parameter_count": trainable_parameter_count,
            "optimizer_covers_all_parameters": True,
            "optimizer": "AdamW",
            "learning_rate": TRAIN_LR,
            "warmup_steps": TRAIN_WARMUP_STEPS,
            "weight_decay": TRAIN_WEIGHT_DECAY,
            "adam_epsilon": TRAIN_ADAM_EPS,
            "adam_foreach": False,
            "steps": steps,
            "resumed_from_checkpoint": (
                str(resumed_from) if resumed_from is not None else None
            ),
            "presentations": len(ledger),
            "checkpoint_hashes": checkpoint_hashes,
            "ledger_sha256": sha256(ROOT / "ledgers" / f"{target_id}.parquet"),
            "mean_loss": float(np.mean(losses)),
            "final_loss": losses[-1],
            "elapsed_seconds": time.time() - started,
        },
    )


def feasibility(document_path: Path) -> None:
    """Run and replay one full-parameter optimizer step without reading outcomes."""
    feasibility_root = ROOT / "feasibility"
    legacy_report = feasibility_root / "completion.json"
    existing_attempts = sorted(feasibility_root.glob("attempt_*/completion.json"))
    if not legacy_report.exists() and not existing_attempts:
        output = feasibility_root
    else:
        attempt_number = 2 + len(existing_attempts)
        output = feasibility_root / f"attempt_{attempt_number:03d}"
    report_path = output / "completion.json"
    checkpoint_path = output / "one_step_checkpoint.pt"
    decision_path = feasibility_root / "decision.json"
    if decision_path.exists():
        raise FileExistsError("target-output-free feasibility already has a passing decision")
    documents = read_table(document_path)
    if {"doc_id", "text"} - set(documents) or len(documents) < TRAIN_BATCH_SIZE:
        raise ValueError("feasibility input needs at least one batch of doc_id/text rows")
    documents = documents.iloc[:TRAIN_BATCH_SIZE].copy()
    tokenizer = tokenizer_local()
    device = _device()

    def one_step() -> tuple[Any, Any, float, float]:  # noqa: ANN401
        torch.manual_seed(20261030)
        np.random.seed(20261030)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, local_files_only=True
        ).to(device)
        model.config.use_cache = False
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAIN_LR / TRAIN_WARMUP_STEPS,
            weight_decay=TRAIN_WEIGHT_DECAY,
            eps=TRAIN_ADAM_EPS,
            foreach=False,
        )
        texts = documents.set_index("doc_id").text.to_dict()
        events = pd.DataFrame({"doc_id": documents.doc_id})
        input_ids, attention = _batch_from_events(
            events, texts, tokenizer, device
        )
        labels = input_ids.clone()
        labels[attention == 0] = -100
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss = model(
            input_ids=input_ids,
            attention_mask=attention,
            labels=labels,
            use_cache=False,
        ).loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP
        )
        optimizer.step()
        elapsed = time.time() - started
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError("feasibility produced non-finite training values")
        return model, optimizer, float(loss.detach().cpu()), elapsed

    first_model, first_optimizer, first_loss, first_elapsed = one_step()
    atomic_torch(
        checkpoint_path,
        {
            "model": first_model.state_dict(),
            "optimizer": first_optimizer.state_dict(),
            "step": 1,
        },
    )
    first_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    del first_model, first_optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    second_model, second_optimizer, second_loss, second_elapsed = one_step()
    model_replay_exact = all(
        torch.equal(first_state["model"][name], value.detach().cpu())
        for name, value in second_model.state_dict().items()
    )
    second_optimizer_state = second_optimizer.state_dict()
    optimizer_replay_exact = True
    optimizer_replay_within_tolerance = True
    optimizer_max_absolute_difference = 0.0
    optimizer_mean_absolute_difference = 0.0
    for parameter_id, state in first_state["optimizer"]["state"].items():
        for name, value in state.items():
            other = second_optimizer_state["state"][parameter_id][name]
            if isinstance(value, torch.Tensor):
                other = other.detach().cpu()
                difference = (value - other).abs()
                optimizer_replay_exact &= torch.equal(value, other)
                optimizer_replay_within_tolerance &= torch.allclose(
                    value, other, rtol=0.0, atol=1e-10
                )
                optimizer_max_absolute_difference = max(
                    optimizer_max_absolute_difference, float(difference.max())
                )
                optimizer_mean_absolute_difference = max(
                    optimizer_mean_absolute_difference, float(difference.mean())
                )
            else:
                optimizer_replay_exact &= value == other
                optimizer_replay_within_tolerance &= value == other
    checkpoint_bytes = checkpoint_path.stat().st_size
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    projected_checkpoint_bytes = checkpoint_bytes * len(TARGETS) * len(
        CHECKPOINT_FRACTIONS
    )
    storage_passed = free_bytes >= int(1.25 * projected_checkpoint_bytes)
    replay_passed = bool(
        model_replay_exact
        and optimizer_replay_within_tolerance
        and first_loss == second_loss
    )
    atomic_json(
        report_path,
        {
            "status": "target-output-free-full-parameter-feasibility-complete",
            "uses_target_outputs": False,
            "input_manifest": str(document_path.resolve()),
            "input_manifest_sha256": sha256(document_path),
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "optimizer": {
                "type": "AdamW",
                "peak_learning_rate": TRAIN_LR,
                "first_step_learning_rate": TRAIN_LR / TRAIN_WARMUP_STEPS,
                "warmup_steps": TRAIN_WARMUP_STEPS,
                "weight_decay": TRAIN_WEIGHT_DECAY,
                "epsilon": TRAIN_ADAM_EPS,
                "foreach": False,
            },
            "device": str(device),
            "first_loss": first_loss,
            "second_loss": second_loss,
            "first_step_seconds": first_elapsed,
            "second_step_seconds": second_elapsed,
            "model_replay_bit_exact": model_replay_exact,
            "optimizer_replay_bit_exact": optimizer_replay_exact,
            "optimizer_replay_atol": 1e-10,
            "optimizer_replay_rtol": 0.0,
            "optimizer_replay_within_tolerance": optimizer_replay_within_tolerance,
            "optimizer_max_absolute_difference": optimizer_max_absolute_difference,
            "optimizer_max_tensor_mean_absolute_difference": optimizer_mean_absolute_difference,
            "replay_passed": replay_passed,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_sha256": sha256(checkpoint_path),
            "projected_32_checkpoint_bytes": projected_checkpoint_bytes,
            "free_bytes_at_check": free_bytes,
            "storage_headroom_factor_required": 1.25,
            "storage_passed": storage_passed,
            "feasibility_passed": bool(replay_passed and storage_passed),
        },
    )
    if replay_passed and storage_passed:
        atomic_json(
            decision_path,
            {
                "status": "target-output-free-feasibility-passed",
                "feasibility_passed": True,
                "passing_attempt": str(report_path.relative_to(feasibility_root)),
                "passing_attempt_sha256": sha256(report_path),
                "preserved_failed_attempt": (
                    str(legacy_report.relative_to(feasibility_root))
                    if legacy_report.exists()
                    and not json.loads(legacy_report.read_text()).get(
                        "feasibility_passed", False
                    )
                    else None
                ),
                "operational_adjustment": "MPS optimizer replay uses absolute tolerance 1e-10 after model state and loss replay bit-exactly; adjustment fixed without target outputs",
            },
        )
    if not replay_passed or not storage_passed:
        raise RuntimeError("target-output-free feasibility gate failed")


def resume_feasibility(document_path: Path) -> None:
    """Replay the next optimizer step twice from the saved feasibility state."""
    resume_root = ROOT / "feasibility" / "resume_replay"
    legacy_report = resume_root / "completion.json"
    existing_attempts = sorted(resume_root.glob("attempt_*/completion.json"))
    output = (
        resume_root
        if not legacy_report.exists() and not existing_attempts
        else resume_root / f"attempt_{2 + len(existing_attempts):03d}"
    )
    report_path = output / "completion.json"
    reference_path = output / "resumed_step2_checkpoint.pt"
    resume_decision_path = resume_root / "decision.json"
    if resume_decision_path.exists():
        raise FileExistsError("checkpoint-resume feasibility already has a decision")
    decision_path = ROOT / "feasibility" / "decision.json"
    if not decision_path.exists():
        raise FileNotFoundError("run the full-parameter feasibility gate first")
    decision = json.loads(decision_path.read_text())
    checkpoint_path = ROOT / "feasibility" / decision["passing_attempt"]
    checkpoint_path = checkpoint_path.parent / "one_step_checkpoint.pt"
    documents = read_table(document_path).iloc[:TRAIN_BATCH_SIZE].copy()
    if len(documents) != TRAIN_BATCH_SIZE or {"doc_id", "text"} - set(documents):
        raise ValueError("resume feasibility requires one complete doc_id/text batch")
    tokenizer = tokenizer_local()
    device = _device()
    events = pd.DataFrame({"doc_id": documents.doc_id})
    texts = documents.set_index("doc_id").text.to_dict()

    def resumed_step() -> tuple[Any, Any, float]:  # noqa: ANN401
        torch.manual_seed(20261031)
        np.random.seed(20261031)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, local_files_only=True
        ).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2 * TRAIN_LR / TRAIN_WARMUP_STEPS,
            weight_decay=TRAIN_WEIGHT_DECAY,
            eps=TRAIN_ADAM_EPS,
            foreach=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = 2 * TRAIN_LR / TRAIN_WARMUP_STEPS
        input_ids, attention = _batch_from_events(
            events, texts, tokenizer, device
        )
        labels = input_ids.clone()
        labels[attention == 0] = -100
        optimizer.zero_grad(set_to_none=True)
        loss = model(
            input_ids=input_ids,
            attention_mask=attention,
            labels=labels,
            use_cache=False,
        ).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        return model, optimizer, float(loss.detach().cpu())

    first_model, first_optimizer, first_loss = resumed_step()
    atomic_torch(
        reference_path,
        {
            "model": first_model.state_dict(),
            "optimizer": first_optimizer.state_dict(),
            "step": 2,
        },
    )
    del first_model, first_optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    second_model, second_optimizer, second_loss = resumed_step()
    model_exact = True
    model_tolerant = True
    model_maximum_difference = 0.0
    for name, value in second_model.state_dict().items():
        expected = reference["model"][name]
        observed = value.detach().cpu()
        model_exact &= torch.equal(expected, observed)
        model_tolerant &= torch.allclose(
            expected, observed, rtol=0.0, atol=1e-10
        )
        model_maximum_difference = max(
            model_maximum_difference, float((expected - observed).abs().max())
        )
    optimizer_tolerant = True
    optimizer_exact = True
    maximum_difference = 0.0
    second_state = second_optimizer.state_dict()
    for parameter_id, state in reference["optimizer"]["state"].items():
        for name, value in state.items():
            other = second_state["state"][parameter_id][name]
            if isinstance(value, torch.Tensor):
                other = other.detach().cpu()
                maximum_difference = max(
                    maximum_difference, float((value - other).abs().max())
                )
                optimizer_exact &= torch.equal(value, other)
                optimizer_tolerant &= torch.allclose(
                    value, other, rtol=0.0, atol=1e-10
                )
            else:
                optimizer_exact &= value == other
                optimizer_tolerant &= value == other
    passed = bool(model_tolerant and optimizer_tolerant and first_loss == second_loss)
    atomic_json(
        report_path,
        {
            "status": "target-output-free-checkpoint-resume-replay-complete",
            "uses_target_outputs": False,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": sha256(checkpoint_path),
            "input_manifest": str(document_path.resolve()),
            "input_manifest_sha256": sha256(document_path),
            "first_loss": first_loss,
            "second_loss": second_loss,
            "model_replay_bit_exact": model_exact,
            "model_replay_atol": 1e-10,
            "model_replay_rtol": 0.0,
            "model_max_absolute_difference": model_maximum_difference,
            "model_replay_within_tolerance": model_tolerant,
            "optimizer_replay_bit_exact": optimizer_exact,
            "optimizer_replay_atol": 1e-10,
            "optimizer_replay_rtol": 0.0,
            "optimizer_max_absolute_difference": maximum_difference,
            "optimizer_replay_within_tolerance": optimizer_tolerant,
            "reference_checkpoint_sha256": sha256(reference_path),
            "resume_replay_passed": passed,
        },
    )
    if passed:
        atomic_json(
            resume_decision_path,
            {
                "status": "target-output-free-checkpoint-resume-replay-passed",
                "resume_replay_passed": True,
                "passing_attempt": str(report_path.relative_to(resume_root)),
                "passing_attempt_sha256": sha256(report_path),
                "preserved_failed_attempt": (
                    str(legacy_report.relative_to(resume_root))
                    if legacy_report.exists()
                    and not json.loads(legacy_report.read_text()).get(
                        "resume_replay_passed", False
                    )
                    else None
                ),
                "operational_adjustment": "MPS checkpoint-resume replay uses absolute tolerance 1e-10 for model and optimizer states after loss replay bit-exactly; fixed without dose-target outputs",
            },
        )
    if not passed:
        raise RuntimeError("checkpoint-resume replay gate failed")


def _load_target(target_id: str, device: torch.device) -> Any:  # noqa: ANN401
    path = ROOT / "targets" / target_id / "checkpoint-100.pt"
    completion_path = ROOT / "targets" / target_id / "completion.json"
    if not completion_path.exists():
        raise FileNotFoundError(f"target training is incomplete: {target_id}")
    completion = json.loads(completion_path.read_text())
    if sha256(path) != completion["checkpoint_hashes"]["1.00"]:
        raise PermissionError(f"final target checkpoint hash changed: {target_id}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True
    )
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def _summaries(alr: np.ndarray, distinct: np.ndarray) -> dict[str, float]:
    return frozen_document_summaries(
        alr[:, :, 0][distinct],
        alr[:, :, 1][distinct],
        alr[:, :, 2][distinct],
    )


def _nested_arrays_with_explicit_geometry(
    text: str,
    tokenizer: Any,  # noqa: ANN401
    model: Any,  # noqa: ANN401
    bos_log_probs: torch.Tensor,
    device: torch.device,
    config: LAR2Config,
    paths_per_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Acquire legacy CATS and exact A/L/R without duplicating model queries."""
    ids = tokenizer(
        text, add_special_tokens=True, truncation=True, max_length=MODEL_MAX_TOKENS + 1
    )["input_ids"]
    if len(ids) < 9:
        raise ValueError("document has fewer than eight predicted tokens")
    likelihood_chunks = []
    cats_chunks = []
    alr_chunks = []
    distinct_chunks = []
    for token_start in range(1, len(ids), paths_per_batch):
        token_indices = list(
            range(token_start, min(len(ids), token_start + paths_per_batch))
        )
        prefixes: list[list[int]] = []
        outcomes: list[int] = []
        length_rows = []
        for token_index in token_indices:
            lengths = context_length_grid(token_index, config.grid_size)
            length_rows.append(lengths)
            for length in lengths:
                prefixes.append(ids[token_index - int(length) : token_index])
                outcomes.append(ids[token_index])
        lengths = torch.tensor([len(prefix) for prefix in prefixes], device=device)
        maximum = int(lengths.max())
        padded = torch.full(
            (len(prefixes), maximum),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros_like(padded)
        for row, prefix in enumerate(prefixes):
            padded[row, : len(prefix)] = torch.tensor(prefix, device=device)
            attention[row, : len(prefix)] = 1
        row_ids = torch.arange(len(prefixes), device=device)
        target_ids = torch.tensor(outcomes, device=device)
        with torch.inference_mode():
            hidden, output_head = _last_hidden_and_head(model, padded, attention)
            last_hidden = hidden[row_ids, lengths - 1]
            log_probs = output_head(last_hidden).float().log_softmax(-1)
            weights = log_probs.exp()
            mean = (weights * log_probs).sum(-1)
            variance = (weights * log_probs.square()).sum(-1) - mean.square()
            contrast = log_probs - bos_log_probs
            contrast_mean = (weights * contrast).sum(-1)
            contrast_variance = (
                (weights * contrast.square()).sum(-1) - contrast_mean.square()
            )
            observed = log_probs[row_ids, target_ids]
            observed_contrast = contrast[row_ids, target_ids]
            cells = torch.stack(
                [
                    observed,
                    observed_contrast,
                    (observed - mean) / variance.clamp_min(1e-12).sqrt(),
                    (observed_contrast - contrast_mean)
                    / contrast_variance.clamp_min(1e-12).sqrt(),
                ],
                dim=-1,
            ).reshape(len(token_indices), config.grid_size, 4)

            roots = (0.5 * log_probs).exp().reshape(
                len(token_indices), config.grid_size, -1
            )
            overlap = (roots[:, :-1] * roots[:, 1:]).sum(-1).clamp(-1.0, 1.0)
            outcome_matrix = torch.tensor(
                [ids[index] for index in token_indices], device=device
            )[:, None]
            observed_roots = roots.gather(
                -1, outcome_matrix[:, :, None].expand(-1, config.grid_size, 1)
            ).squeeze(-1)
            numerator = observed_roots[:, 1:] - overlap * observed_roots[:, :-1]
            denominator = (1.0 - overlap.square()).clamp_min(0).sqrt() * (
                1.0 - observed_roots[:, :-1].square()
            ).clamp_min(0).sqrt()
            tau = torch.where(
                denominator > 1e-8,
                numerator / denominator,
                torch.zeros_like(numerator),
            ).clamp(-1, 1)
            fisher_rao = 2.0 * torch.acos(overlap)
            increment = cells[:, 1:, 0] - cells[:, :-1, 0]
            cats = torch.stack((tau, fisher_rao, increment), dim=-1)

        roots_np = roots.float().cpu().numpy()
        realized = np.repeat(
            np.asarray([ids[index] for index in token_indices], dtype=int),
            config.grid_size - 1,
        )
        alignment, length, directed = fisher_rao_alr_from_roots(
            roots_np[:, :-1].reshape(-1, roots_np.shape[-1]),
            roots_np[:, 1:].reshape(-1, roots_np.shape[-1]),
            realized,
        )
        explicit = np.stack((alignment, length, directed), axis=-1).reshape(
            len(token_indices), config.grid_size - 1, 3
        )
        length_rows_array = np.asarray(length_rows)
        distinct = np.diff(length_rows_array, axis=1) > 0
        cells_np = cells.cpu().numpy().astype(np.float32)
        cats_np = cats.cpu().numpy().astype(np.float32)
        cats_np[~distinct] = 0.0
        explicit[~distinct] = 0.0
        likelihood_chunks.append(cells_np)
        cats_chunks.append(cats_np)
        alr_chunks.append(explicit)
        distinct_chunks.append(distinct)
    raw = np.concatenate(likelihood_chunks)
    derivative = np.gradient(
        raw[:, :, 2],
        np.linspace(0, 1, config.grid_size),
        axis=1,
        edge_order=1,
    )
    likelihood = np.concatenate((raw, derivative[:, :, None]), axis=2).astype(
        np.float32
    )
    return (
        likelihood,
        np.concatenate(cats_chunks),
        np.concatenate(alr_chunks),
        np.concatenate(distinct_chunks),
        ids,
    )


def acquire(target_id: str, phase: str, paths_per_batch: int) -> None:
    if TARGETS.get(target_id, {}).get("phase") != phase:
        raise ValueError("target does not belong to requested phase")
    verify_design_artifacts(target_id)
    PhaseAccess(ROOT).require_openable(phase)
    if phase != "development":
        verify_frozen_analysis_software(
            json.loads(PhaseAccess(ROOT).frozen_spec.read_text())
        )
    output = ROOT / "outcomes" / phase / f"{target_id}.parquet"
    if output.exists():
        raise FileExistsError(f"outcome acquisition is immutable: {output}")
    if phase != "development":
        record_access(
            ROOT / "access" / f"{phase}_{target_id}_opened.json",
            phase,
            target_id,
        )
    manifest = pd.read_parquet(ROOT / "candidate_manifest.parquet")
    assignment = pd.read_parquet(ROOT / "assignments.parquet")
    frame = manifest.loc[manifest.role == phase].merge(
        assignment.loc[assignment.target_id == target_id],
        on=["block_id", "doc_slot", "doc_id", "role"],
        validate="one_to_one",
    )
    if phase != "development":
        specification = json.loads(PhaseAccess(ROOT).frozen_spec.read_text())
        frame = frame.loc[
            frame.block_id.isin(specification["frozen_block_ids"][phase])
        ].reset_index(drop=True)
    device = _device()
    tokenizer = tokenizer_local()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True
    ).to(device).eval()
    target = _load_target(target_id, device)
    config = LAR2Config()
    bos = torch.tensor([[tokenizer.eos_token_id]], device=device)
    with torch.inference_mode():
        base_bos = base(bos, use_cache=False).logits[0, -1].float().log_softmax(-1)
        target_bos = target(bos, use_cache=False).logits[0, -1].float().log_softmax(-1)
    target_checkpoint = ROOT / "targets" / target_id / "checkpoint-100.pt"
    base_config_hash = hashlib.sha256(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "geometry": asdict(config),
                "explicit_geometry": "fisher-rao-alr-v1",
                "legacy_cats_preserved": True,
                "version": 2,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    target_config_hash = hashlib.sha256(
        json.dumps(
            {
                "base_config_hash": base_config_hash,
                "target_checkpoint_sha256": sha256(target_checkpoint),
                "version": 2,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    rows = []
    for position, row in enumerate(frame.itertuples(index=False), 1):
        cache_name = f"{row.text_hash}.npz"
        base_cache = ROOT / "outcomes" / "base_cache" / phase / cache_name
        target_cache = ROOT / "outcomes" / phase / target_id / "cache" / cache_name
        if base_cache.exists():
            with np.load(base_cache, allow_pickle=False) as stored:
                if str(stored["config_hash"].item()) != base_config_hash:
                    raise ValueError("base acquisition cache configuration changed")
                base_likelihood = stored["likelihood"]
                base_cats = stored["cats"]
                base_alr = stored["alr"]
                base_distinct = stored["distinct"]
        else:
            (
                base_likelihood,
                base_cats,
                base_alr,
                base_distinct,
                _,
            ) = _nested_arrays_with_explicit_geometry(
                row.text,
                tokenizer,
                base,
                base_bos,
                device,
                config,
                paths_per_batch,
            )
            atomic_npz(
                base_cache,
                likelihood=base_likelihood,
                cats=base_cats,
                alr=base_alr,
                distinct=base_distinct,
                config_hash=base_config_hash,
            )
        if target_cache.exists():
            with np.load(target_cache, allow_pickle=False) as stored:
                if str(stored["config_hash"].item()) != target_config_hash:
                    raise ValueError("target acquisition cache configuration changed")
                final_likelihood = stored["likelihood"]
                final_cats = stored["cats"]
                final_alr = stored["alr"]
                final_distinct = stored["distinct"]
        else:
            (
                final_likelihood,
                final_cats,
                final_alr,
                final_distinct,
                _,
            ) = _nested_arrays_with_explicit_geometry(
                row.text,
                tokenizer,
                target,
                target_bos,
                device,
                config,
                paths_per_batch,
            )
            atomic_npz(
                target_cache,
                likelihood=final_likelihood,
                cats=final_cats,
                alr=final_alr,
                distinct=final_distinct,
                config_hash=target_config_hash,
            )
        if not np.array_equal(base_distinct, final_distinct):
            raise RuntimeError("base and target context masks differ")
        count = max(1, int(0.20 * len(final_likelihood)))
        standardized = final_likelihood[:, -1, 2]
        rows.append(
            {
                "block_id": row.block_id,
                "latin_block_position": int(row.latin_block_position),
                "doc_id": row.doc_id,
                "doc_slot": int(row.doc_slot),
                "target_id": target_id,
                "role": phase,
                "K": int(row.K),
                "d": float(row.d),
                **{
                    f"base_{name}": value
                    for name, value in _summaries(base_alr, base_distinct).items()
                },
                **{
                    f"final_{name}": value
                    for name, value in _summaries(final_alr, final_distinct).items()
                },
                "base_realized_logp": float(np.mean(base_likelihood[:, -1, 0])),
                "final_loss": float(-np.mean(final_likelihood[:, -1, 0])),
                "min_k_plus_plus_20": float(np.mean(np.sort(standardized)[:count])),
                "predicted_tokens": len(final_likelihood),
                "eligible_geometry_transitions": int(final_distinct.sum()),
                **cats_v3_document_features(final_likelihood, final_cats, final_distinct),
            }
        )
        if position % 10 == 0:
            print(f"{target_id}/{phase}: {position}/{len(frame)}", flush=True)
    outcomes = add_checkpoint_changes(pd.DataFrame(rows))
    write_parquet(output, outcomes)
    atomic_json(
        output.with_suffix(".json"),
        {
            "status": "final-checkpoint-own-context-outcomes-acquired",
            "target_id": target_id,
            "phase": phase,
            "documents": len(rows),
            "requested_documents": len(frame),
            "coverage": len(rows) / len(frame) if len(frame) else 0.0,
            "invalid_documents": 0,
            "invalid_reason_counts": {},
            "complete_distribution_queries_per_model": int(
                outcomes.predicted_tokens.sum() * LAR2Config().grid_size
            ),
            "eligible_geometry_transitions": int(
                outcomes.eligible_geometry_transitions.sum()
            ),
            "checkpoint_sha256": sha256(ROOT / "targets" / target_id / "checkpoint-100.pt"),
            "base_cache_config_hash": base_config_hash,
            "target_cache_config_hash": target_config_hash,
            "outcomes_sha256": sha256(output),
            "primary_eligible": True,
            "intermediate_checkpoints_primary_eligible": False,
        },
    )


def validate_power_config(config: dict[str, Any]) -> None:
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("production power requires an explicit scenarios list")
    by_name = {str(scenario.get("name")): scenario for scenario in scenarios}
    required = {
        "null",
        "plausible",
        "seed_heterogeneity",
        "timing_variation",
        "nonlinear",
    }
    if set(by_name) != required or len(by_name) != len(scenarios):
        raise ValueError(f"power scenarios must be exactly {sorted(required)}")
    for name, scenario in by_name.items():
        if int(scenario.get("simulations", 0)) < 500:
            raise ValueError(f"{name} requires at least 500 simulated studies")
        if int(scenario.get("randomization_draws", 0)) < 1_999:
            raise ValueError(f"{name} requires at least 1,999 assignment draws")
        if float(scenario.get("noise_sd", 0.0)) <= 0:
            raise ValueError(f"{name} requires positive outcome noise")
        if name != "null" and not bool(scenario.get("requires_power", False)):
            raise ValueError(f"{name} must satisfy the 90% power gate")
    null = by_name["null"]
    if any(
        float(null.get(key, 0.0)) != 0.0
        for key in ("effect", "seed_sd", "timing_sd", "quadratic")
    ) or bool(null.get("requires_power", False)):
        raise ValueError("null scenario must have no effect terms and no power requirement")
    if float(by_name["plausible"].get("effect", 0.0)) <= 0:
        raise ValueError("plausible scenario requires a positive effect")
    if float(by_name["seed_heterogeneity"].get("seed_sd", 0.0)) <= 0:
        raise ValueError("seed-heterogeneity scenario requires positive seed_sd")
    if float(by_name["timing_variation"].get("timing_sd", 0.0)) <= 0:
        raise ValueError("timing-variation scenario requires positive timing_sd")
    if float(by_name["nonlinear"].get("quadratic", 0.0)) == 0:
        raise ValueError("nonlinear scenario requires a nonzero quadratic term")
    seeds = [int(scenario["seed"]) for scenario in scenarios]
    if len(set(seeds)) != len(seeds):
        raise ValueError("power scenarios require distinct Monte Carlo seeds")
    length = config.get("length_equivalence")
    if not isinstance(length, dict):
        raise ValueError("length-equivalence power configuration is required")
    if (
        int(length.get("simulations", 0)) < 500
        or int(length.get("randomization_draws", 0)) < 1_999
        or float(length.get("noise_sd", 0.0)) <= 0
        or float(length.get("true_slope", np.nan)) != 0.0
        or float(length.get("margin", np.nan)) != 0.05
    ):
        raise ValueError(
            "length equivalence requires 500 simulations, 1,999 draws, positive "
            "noise, zero true slope, and the standardized ±0.05 margin"
        )


def run_power(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    validate_power_config(config)
    _development_frame()
    calibration_path = ROOT / "development" / "calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError("freeze development calibration before power simulation")
    if config.get("development_calibration_sha256") != sha256(calibration_path):
        raise PermissionError("power configuration is not tied to frozen development calibration")
    development_hashes = {
        target: sha256(ROOT / "outcomes" / "development" / f"{target}.parquet")
        for target, value in TARGETS.items()
        if value["phase"] == "development"
    }
    candidate_manifest = pd.read_parquet(
        ROOT / "candidate_manifest.parquet", columns=["block_id", "role"]
    )
    held_positions = {
        role: np.asarray(
            sorted(
                candidate_manifest.loc[
                    candidate_manifest.role == role, "block_id"
                ].unique()
            ),
            dtype=int,
        )
        for role in ("validation", "confirmation")
    }
    if any(len(positions) != 300 for positions in held_positions.values()):
        raise ValueError("power requires all 300 blocks in both held phases")
    scenarios = config["scenarios"]
    scenario_results = []
    for scenario in scenarios:
        parameters = {key: value for key, value in scenario.items() if key not in {"name", "requires_power"}}
        for blocks in (200, 250, 300):
            for design_phase, positions in held_positions.items():
                result = power_gate_simulation(
                    blocks=blocks,
                    block_positions=positions[:blocks],
                    **parameters,
                )
                scenario_results.append(
                    {
                        "name": scenario["name"],
                        "design_phase": design_phase,
                        "requires_power": bool(
                            scenario.get("requires_power", False)
                        ),
                        **result,
                    }
                )
    qualifying = []
    for blocks in (200, 250, 300):
        required = [
            result for result in scenario_results
            if result["blocks"] == blocks and result["requires_power"]
        ]
        if required and all(float(result["compound_power"]) >= 0.90 for result in required):
            qualifying.append({"blocks": blocks, "compound_power": min(float(result["compound_power"]) for result in required)})
    selected = select_powered_block_count(qualifying)
    length_config = config.get("length_equivalence")
    length_result = None
    if length_config is not None:
        length_result = {
            role: equivalence_power_simulation(
                blocks=selected,
                block_positions=positions[:selected],
                **length_config,
            )
            for role, positions in held_positions.items()
        }
    null_results = [
        result for result in scenario_results if result["name"] == "null"
    ]
    null_calibration_passed = bool(
        null_results
        and all(float(result["compound_power"]) <= 0.05 for result in null_results)
    )
    atomic_json(
        ROOT / "development" / "power.json",
        {
            "status": "compound-gate-power-complete",
            "config": config,
            "scenario_results": scenario_results,
            "selected_held_blocks": selected,
            "length_equivalence": length_result,
            "length_equivalence_confirmatory": bool(
                length_result
                and all(
                    float(result["equivalence_power"]) >= 0.80
                    for result in length_result.values()
                )
            ),
            "null_calibration_threshold": 0.05,
            "null_calibration_passed": null_calibration_passed,
            "software_provenance": software_provenance(),
            "development_outcome_hashes": development_hashes,
            "development_calibration_sha256": sha256(calibration_path),
        },
    )


def _development_frame() -> pd.DataFrame:
    verify_design_artifacts()
    targets = [
        target
        for target, value in TARGETS.items()
        if value["phase"] == "development"
    ]
    return pd.concat(
        [_read_verified_outcome(target, "development") for target in targets],
        ignore_index=True,
    )


def _read_verified_outcome(target_id: str, phase: str) -> pd.DataFrame:
    path = ROOT / "outcomes" / phase / f"{target_id}.parquet"
    sidecar = path.with_suffix(".json")
    if not path.exists() or not sidecar.exists():
        raise FileNotFoundError(f"complete outcome and sidecar required: {target_id}")
    manifest = json.loads(sidecar.read_text())
    if (
        manifest.get("target_id") != target_id
        or manifest.get("phase") != phase
        or manifest.get("outcomes_sha256") != sha256(path)
    ):
        raise PermissionError(f"outcome provenance check failed: {target_id}")
    return pd.read_parquet(path)


def summarize_development() -> None:
    """Freeze transparent development quantities used to justify power scenarios."""
    output = ROOT / "development" / "calibration.json"
    if output.exists():
        raise FileExistsError("development calibration is already frozen")
    frame = _development_frame()
    slopes = target_slopes(frame, "delta_S_R")
    residuals = []
    for target, group in frame.groupby("target_id"):
        x = group.d - group.groupby("block_id").d.transform("mean")
        y = group.delta_S_R - group.groupby("block_id").delta_S_R.transform(
            "mean"
        )
        residuals.extend((y - slopes[str(target)] * x).to_numpy(dtype=float))
    x = frame.d.to_numpy(dtype=float, copy=True)
    x2 = x * x
    y = frame.delta_S_R.to_numpy(dtype=float, copy=True)
    groups = [frame.target_id, frame.block_id]
    x -= frame.groupby(groups).d.transform("mean").to_numpy(dtype=float)
    x2 -= pd.Series(x2).groupby([frame.target_id, frame.block_id]).transform(
        "mean"
    ).to_numpy(dtype=float)
    y -= frame.groupby(groups).delta_S_R.transform("mean").to_numpy(dtype=float)
    nonlinear = np.linalg.lstsq(np.column_stack((x, x2)), y, rcond=None)[0]
    slope_values = np.asarray(list(slopes.values()), dtype=float)
    timing = {}
    for target in slopes:
        balance = pd.read_parquet(ROOT / "ledgers" / f"{target}_balance.parquet")
        treated = balance.loc[balance.K > 0]
        timing[target] = {
            "maximum_mean_progress_deviation": float(
                np.max(np.abs(treated.mean_progress - 0.5))
            ),
            "positive_dose_weighted_lr_range": float(
                np.ptp(treated.weighted_learning_rate)
            ),
        }
    atomic_json(
        output,
        {
            "status": "development-only-power-calibration-complete",
            "uses_held_outcomes": False,
            "outcome": "delta_S_R",
            "target_slopes": slopes,
            "mean_target_slope": float(slope_values.mean()),
            "minimum_target_slope": float(slope_values.min()),
            "target_slope_sd": float(slope_values.std(ddof=1)),
            "block_fe_residual_sd": float(np.std(residuals, ddof=1)),
            "pooled_block_fe_linear_coefficient": float(nonlinear[0]),
            "pooled_block_fe_quadratic_coefficient": float(nonlinear[1]),
            "standardized_length_outcome_sd": 1.0,
            "ledger_timing_diagnostics": timing,
            "development_outcome_hashes": {
                target: sha256(
                    ROOT / "outcomes" / "development" / f"{target}.parquet"
                )
                for target in slopes
            },
            "software_provenance": software_provenance(),
        },
    )


def freeze() -> None:
    access = PhaseAccess(ROOT)
    if access.frozen_spec.exists():
        raise FileExistsError("analysis is already frozen")
    power_path = ROOT / "development" / "power.json"
    if not power_path.exists():
        raise FileNotFoundError("compound-gate power simulation must complete before freeze")
    power = json.loads(power_path.read_text())
    if power.get("software_provenance") != software_provenance():
        raise PermissionError("analysis software changed after power simulation")
    expected_development_targets = {
        target
        for target, value in TARGETS.items()
        if value["phase"] == "development"
    }
    if set(power.get("development_outcome_hashes", {})) != expected_development_targets:
        raise PermissionError("power artifact lacks the complete development provenance")
    for target, expected in power["development_outcome_hashes"].items():
        if sha256(ROOT / "outcomes" / "development" / f"{target}.parquet") != expected:
            raise PermissionError("development outcomes changed after power simulation")
    calibration_path = ROOT / "development" / "calibration.json"
    if (
        not calibration_path.exists()
        or power.get("development_calibration_sha256") != sha256(calibration_path)
    ):
        raise PermissionError("development calibration changed after power simulation")
    power_config = power["config"]
    if not power.get("null_calibration_passed", False):
        raise ValueError("compound-gate null calibration failed")
    scenarios = power_config.get("scenarios", [])
    required_scenarios = {
        "null",
        "plausible",
        "seed_heterogeneity",
        "timing_variation",
        "nonlinear",
    }
    if {str(scenario.get("name")) for scenario in scenarios} != required_scenarios:
        raise ValueError(
            "production power must include exactly null, plausible, "
            "seed_heterogeneity, timing_variation, and nonlinear scenarios"
        )
    if any(int(scenario.get("randomization_draws", 0)) < 1_999 for scenario in scenarios):
        raise ValueError("production power scenarios require at least 1,999 assignment draws")
    if any(int(scenario.get("simulations", 0)) < 500 for scenario in scenarios):
        raise ValueError("production power scenarios require at least 500 simulated studies")
    if any(
        scenario["name"] != "null" and not bool(scenario.get("requires_power", False))
        for scenario in scenarios
    ):
        raise ValueError("every non-null alternative must satisfy the 90% power requirement")
    length_config = power_config.get("length_equivalence")
    if length_config is not None and (
        int(length_config.get("randomization_draws", 0)) < 1_999
        or int(length_config.get("simulations", 0)) < 500
    ):
        raise ValueError(
            "length-equivalence power requires at least 1,999 assignments and 500 studies"
        )
    if int(power["selected_held_blocks"]) not in {200, 250, 300}:
        raise ValueError("invalid powered held sample size")
    development = _development_frame()
    candidate_manifest = pd.read_parquet(ROOT / "candidate_manifest.parquet")
    selected_held_blocks = int(power["selected_held_blocks"])
    frozen_block_ids = {
        role: [
            int(value)
            for value in sorted(
                candidate_manifest.loc[
                    candidate_manifest.role == role, "block_id"
                ].unique()
            )[: (100 if role == "development" else selected_held_blocks)]
        ]
        for role in ("development", "validation", "confirmation")
    }
    labels = (development.K > 0).astype(int).to_numpy()
    scaler = StandardScaler().fit(development[list(FEATURE_COLUMNS)])
    classifier = LogisticRegression(C=0.03, solver="lbfgs", max_iter=5000, random_state=CATS_FIT_SEED).fit(
        scaler.transform(development[list(FEATURE_COLUMNS)]), labels
    )
    development_with_text = development.merge(
        candidate_manifest[["doc_id", "text"]], on="doc_id", validate="many_to_one"
    )
    text_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=12_000,
        sublinear_tf=True,
    )
    development_text = text_vectorizer.fit_transform(development_with_text.text)
    development_text_labels = (development_with_text.K > 0).astype(int).to_numpy()
    text_classifier = LogisticRegression(
        C=0.03,
        solver="lbfgs",
        max_iter=5_000,
        random_state=20261020,
    ).fit(development_text, development_text_labels)
    text_model_path = ROOT / "development" / "frozen_blind_tfidf.joblib"
    atomic_joblib(
        text_model_path,
        {"vectorizer": text_vectorizer, "classifier": text_classifier},
    )
    support_threshold = float(development.base_realized_logp.median())
    spec = {
        "version": "exposure-observability-v1",
        "status": "frozen-before-validation-outcome-access",
        "estimand": "document-level causal allocation response in full-parameter continued pretraining",
        "unit": "FineWeb-Edu source document in a six-document matched block",
        "training_stage": "continued pretraining",
        "target_population": "eight pinned full-parameter Pythia-70m-deduped continuations",
        "base_model_revision": BASE_REVISION,
        "access": "base and target checkpoints with complete next-token distributions",
        "primary_outcome": "delta_S_R",
        "document_summary": "NumPy 0.90 quantile with linear method over distinct context transitions",
        "dose": "log2(K+1)",
        "phase_statistic": "equal-weight mean of three block-fixed-effect target slopes",
        "randomization": {"draws": 99999, "pvalue": "plus-one one-sided", "interval": "95% inversion of the same mechanism under a constant linear response"},
        "held_analysis_seeds": HELD_ANALYSIS_SEEDS,
        "context_analysis_seeds": CONTEXT_ANALYSIS_SEEDS,
        "validation_gate": "all three slopes positive; p_rand < 0.025; 95% design interval lower endpoint positive",
        "base_validity_gate": "90% standardized base-slope interval contained in [-0.05, 0.05]",
        "decomposition": ["delta_S_A", "delta_S_L"],
        "length_equivalence_confirmatory": power.get("length_equivalence_confirmatory", False),
        "checkpoint_policy": "only 100% outcomes are primary",
        "selected_held_blocks": power["selected_held_blocks"],
        "frozen_block_ids": frozen_block_ids,
        "power_sha256": sha256(power_path),
        "candidate_manifest_sha256": sha256(ROOT / "candidate_manifest.parquet"),
        "assignments_sha256": sha256(ROOT / "assignments.parquet"),
        "cats_downstream": {
            "description": "frozen CATS-v4 feature map plus frozen development-fitting protocol",
            "features": list(FEATURE_COLUMNS),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "logistic_coef": classifier.coef_[0].tolist(),
            "logistic_intercept": float(classifier.intercept_[0]),
            "label": "1[K>0]",
            "classifier": "StandardScaler plus LogisticRegression C=0.03",
            "fit_seed": CATS_FIT_SEED,
        },
        "blind_text_baseline": {
            "description": "development-fitted TF-IDF word unigram/bigram logistic baseline",
            "artifact": str(text_model_path),
            "artifact_sha256": sha256(text_model_path),
            "uses_target_outputs": False,
        },
        "donor_support_threshold": support_threshold,
        "analysis_software_provenance": software_provenance(),
        "claims": {
            "geometric_identity": "exact under phi(p)=2sqrt(p)",
            "prior_randomized": "report unchanged",
            "new_causal": "requires validation and confirmation gates",
            "self_influence": "mechanistic hypothesis only",
        },
    }
    atomic_json(access.frozen_spec, spec)


def _apply_frozen_cats(frame: pd.DataFrame, specification: dict[str, Any]) -> np.ndarray:
    frozen = specification["cats_downstream"]
    x = frame[frozen["features"]].to_numpy(dtype=float)
    x = (x - np.asarray(frozen["scaler_mean"])) / np.asarray(frozen["scaler_scale"])
    return x @ np.asarray(frozen["logistic_coef"]) + float(frozen["logistic_intercept"])


def _target_only_mia_comparison(frame: pd.DataFrame) -> dict[str, Any]:
    labels = (frame.K > 0).astype(int).to_numpy()
    scores = {
        "cats_frozen": frame.cats_frozen.to_numpy(),
        "raw_A": frame.final_S_A.to_numpy(),
        "raw_R": frame.final_S_R.to_numpy(),
        "negative_loss": -frame.final_loss.to_numpy(),
        "min_k_plus_plus_20": frame.min_k_plus_plus_20.to_numpy(),
    }
    if "blind_tfidf" in frame:
        scores["blind_tfidf"] = frame.blind_tfidf.to_numpy()
    scores["deterministic_random_control"] = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{doc_id}:{target}:random-control-v1".encode()).digest()[:8],
                "big",
            )
            / 2**64
            for doc_id, target in zip(frame.doc_id, frame.target_id, strict=True)
        ]
    )
    result = {}
    for name, values in scores.items():
        per_target = {}
        for target, group in frame.assign(_score=values).groupby("target_id"):
            target_labels = (group.K > 0).astype(int).to_numpy()
            target_scores = group._score.to_numpy(dtype=float)
            interval = grouped_auc_interval(
                target_labels,
                target_scores,
                group.block_id.to_numpy(),
                draws=1_999,
                seed=int.from_bytes(
                    hashlib.sha256(
                        f"{name}:{target}:target-auc-v1".encode()
                    ).digest()[:8],
                    "big",
                ),
            )
            per_target[str(target)] = {
                "auc": float(roc_auc_score(target_labels, target_scores)),
                "interval_95": list(interval),
            }
        pooled_interval = grouped_auc_interval(
            labels,
            values,
            frame.block_id.to_numpy(),
            draws=1_999,
            seed=int.from_bytes(
                hashlib.sha256(f"{name}:pooled-auc-v1".encode()).digest()[:8],
                "big",
            ),
        )
        result[name] = {
            "pooled_auc_descriptive": float(roc_auc_score(labels, values)),
            "pooled_interval_95_block_bootstrap": list(pooled_interval),
            "bootstrap_draws": 1_999,
            "per_target_auc_descriptive": per_target,
        }
    return result


def _descriptive_target_t_interval(slopes: dict[str, float]) -> list[float]:
    values = np.asarray(list(slopes.values()), dtype=float)
    if len(values) < 2:
        return [float("nan"), float("nan")]
    half_width = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return [float(values.mean() - half_width), float(values.mean() + half_width)]


def save_saturated_plot(summary: pd.DataFrame, path: Path, phase: str) -> None:
    """Plot the prespecified six-level descriptive allocation response."""
    width, height = 720, 460
    left, right, top, bottom = 82, 28, 54, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = summary.d.to_numpy(dtype=float)
    means = summary["mean"].to_numpy(dtype=float)
    lower = summary.lower.to_numpy(dtype=float)
    upper = summary.upper.to_numpy(dtype=float)
    y_min = min(float(np.min(lower)), 0.0)
    y_max = max(float(np.max(upper)), 0.0)
    padding = max((y_max - y_min) * 0.10, 1e-9)
    y_min -= padding
    y_max += padding

    def x_pixel(value: float) -> float:
        return left + plot_width * (value - float(x_values.min())) / max(
            float(np.ptp(x_values)), 1e-12
        )

    def y_pixel(value: float) -> float:
        return top + plot_height * (y_max - value) / (y_max - y_min)

    points = " ".join(
        f"{x_pixel(x):.2f},{y_pixel(y):.2f}"
        for x, y in zip(x_values, means, strict=True)
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{phase.title()} saturated allocation response</text>',
        f'<line x1="{left}" y1="{y_pixel(0):.2f}" x2="{width-right}" y2="{y_pixel(0):.2f}" stroke="#777" stroke-dasharray="5 4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<polyline points="{points}" fill="none" stroke="#175d8d" stroke-width="2"/>',
    ]
    for row in summary.itertuples(index=False):
        x = x_pixel(float(row.d))
        y = y_pixel(float(row.mean))
        low = y_pixel(float(row.lower))
        high = y_pixel(float(row.upper))
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{low:.2f}" x2="{x:.2f}" y2="{high:.2f}" stroke="#175d8d"/>',
                f'<line x1="{x-5:.2f}" y1="{low:.2f}" x2="{x+5:.2f}" y2="{low:.2f}" stroke="#175d8d"/>',
                f'<line x1="{x-5:.2f}" y1="{high:.2f}" x2="{x+5:.2f}" y2="{high:.2f}" stroke="#175d8d"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#175d8d"/>',
                f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{int(row.K)}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">Exact-document presentations K</text>',
            f'<text x="18" y="{height/2}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 18 {height/2})">Block-adjusted mean ΔS_R</text>',
            f'<text x="{left-8}" y="{y_pixel(y_max-padding):.2f}" text-anchor="end" font-family="monospace" font-size="11">{y_max-padding:.3g}</text>',
            f'<text x="{left-8}" y="{y_pixel(y_min+padding):.2f}" text-anchor="end" font-family="monospace" font-size="11">{y_min+padding:.3g}</text>',
            "</svg>",
        ]
    )
    atomic_text(path, "\n".join(elements))


def analyze(phase: str, draws: int, seed: int) -> None:
    if phase not in {"validation", "confirmation"}:
        raise ValueError("held analysis is validation or confirmation")
    if draws != 99_999:
        raise ValueError("frozen held analysis requires exactly 99,999 randomization draws")
    if seed != HELD_ANALYSIS_SEEDS[phase]:
        raise ValueError(f"frozen {phase} analysis seed is {HELD_ANALYSIS_SEEDS[phase]}")
    PhaseAccess(ROOT).require_openable(phase)
    verify_design_artifacts()
    targets = [target for target, value in TARGETS.items() if value["phase"] == phase]
    frame = pd.concat(
        [_read_verified_outcome(target, phase) for target in targets],
        ignore_index=True,
    )
    specification = json.loads(PhaseAccess(ROOT).frozen_spec.read_text())
    verify_frozen_analysis_software(specification)
    selected_blocks = int(specification["selected_held_blocks"])
    keep = specification["frozen_block_ids"][phase]
    frame = frame.loc[frame.block_id.isin(keep)].reset_index(drop=True)
    primary_test = randomization_test(
        frame, "delta_S_R", draws=draws, seed=seed, all_target_ids=list(TARGETS)
    )
    primary_interval = design_based_interval(
        frame, "delta_S_R", draws=draws, seed=seed + 1, all_target_ids=list(TARGETS)
    )
    slopes = target_slopes(frame, "delta_S_R")
    primary_pass = validation_gate(slopes, primary_test.p_one_sided, primary_interval)
    base = standardize_outcome(frame, "base_S_R")
    base_interval = design_based_interval(
        base,
        "base_S_R",
        draws=draws,
        seed=seed + 2,
        confidence=0.90,
        all_target_ids=list(TARGETS),
    )
    base_pass = base_interval[0] >= -0.05 and base_interval[1] <= 0.05
    frame["cats_frozen"] = _apply_frozen_cats(frame, specification)
    text_spec = specification["blind_text_baseline"]
    text_model_path = Path(text_spec["artifact"])
    if sha256(text_model_path) != text_spec["artifact_sha256"]:
        raise ValueError("frozen blind-text baseline artifact changed")
    text_model = joblib.load(text_model_path)
    candidate_text = pd.read_parquet(
        ROOT / "candidate_manifest.parquet", columns=["doc_id", "text"]
    )
    frame = frame.merge(candidate_text, on="doc_id", validate="many_to_one")
    frame["blind_tfidf"] = text_model["classifier"].decision_function(
        text_model["vectorizer"].transform(frame.text)
    )
    decomposition = {}
    for outcome in ("delta_S_A", "delta_S_L"):
        test = randomization_test(
            frame,
            outcome,
            draws=draws,
            seed=seed + (3 if outcome.endswith("A") else 5),
            all_target_ids=list(TARGETS),
        )
        interval = design_based_interval(
            frame,
            outcome,
            draws=draws,
            seed=seed + (4 if outcome.endswith("A") else 6),
            all_target_ids=list(TARGETS),
        )
        component_slopes = target_slopes(frame, outcome)
        decomposition[outcome] = {
            "estimate": test.estimate,
            "p_one_sided": test.p_one_sided,
            "interval_95": interval,
            "target_slopes": component_slopes,
            "directional_gate_passed": validation_gate(component_slopes, test.p_one_sided, interval),
        }
    length_equivalence = {"confirmatory": False, "reported_descriptively": True}
    if specification.get("length_equivalence_confirmatory", False):
        standardized_length = standardize_outcome(frame, "delta_S_L")
        length_interval = design_based_interval(
            standardized_length,
            "delta_S_L",
            draws=draws,
            seed=seed + 7,
            confidence=0.90,
            all_target_ids=list(TARGETS),
        )
        length_equivalence = {
            "confirmatory": True,
            "reported_descriptively": False,
            "standardized_interval_90": length_interval,
            "margin": [-0.05, 0.05],
            "passed": length_interval[0] >= -0.05 and length_interval[1] <= 0.05,
        }
    completion = {
        "status": f"frozen-{phase}-complete",
        "phase": phase,
        "blocks": selected_blocks,
        "targets": targets,
        "primary": {"outcome": "delta_S_R", "estimate": primary_test.estimate, "target_slopes": slopes, "p_rand_one_sided": primary_test.p_one_sided, "interval_95": primary_interval, "compound_gate_passed": primary_pass},
        "base_randomization_validity": {"standardized_interval_90": base_interval, "equivalence_margin": [-0.05, 0.05], "passed": base_pass},
        "phase_gate_passed": bool(primary_pass and base_pass),
        "decomposition": decomposition,
        "length_equivalence": length_equivalence,
        "strong_directional_component_pattern_passed": bool(
            primary_pass
            and base_pass
            and decomposition["delta_S_A"]["directional_gate_passed"]
            and length_equivalence.get("passed", False)
        ),
        "target_only_membership_identification": {
            "label": "1[K>0]",
            "access_separated_from_base_assisted_delta_geometry": True,
            "methods": _target_only_mia_comparison(frame),
        },
        "document_target_fe_sensitivity": document_target_fixed_effect_sensitivity(frame, "delta_S_R"),
        "across_target_t_interval_policy": "descriptive only",
        "across_target_t_interval_95_descriptive": _descriptive_target_t_interval(slopes),
        "randomization_draws": draws,
        "randomization_seed": seed,
        "frozen_spec_sha256": sha256(PhaseAccess(ROOT).frozen_spec),
    }
    directory = ROOT / "analysis" / phase
    if directory.exists():
        raise FileExistsError(f"held analysis is already published: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{phase}-analysis-staging-", dir=directory.parent
    ) as temporary:
        staging = Path(temporary)
        write_parquet(staging / "per_document.parquet", frame.drop(columns=["text"]))
        saturated = saturated_dose_summary(frame, "delta_S_R")
        write_parquet(
            staging / "saturated_delta_S_R.parquet",
            saturated,
        )
        save_saturated_plot(
            saturated, staging / "saturated_delta_S_R.svg", phase
        )
        for outcome in ("delta_S_R", "delta_S_A", "delta_S_L"):
            write_parquet(
                staging / f"progression_{outcome}.parquet",
                dose_progression(frame, outcome),
            )
        completion["artifacts"] = {
            artifact.name: sha256(artifact)
            for artifact in sorted(staging.iterdir())
            if artifact.is_file()
        }
        atomic_json(staging / "completion.json", completion)
        os.replace(staging, directory)
    if phase == "validation":
        atomic_json(
            PhaseAccess(ROOT).validation_decision,
            {"phase": "validation", "gate_passed": completion["phase_gate_passed"], "completion_sha256": sha256(directory / "completion.json"), "recorded_before_confirmation": True},
        )


def preflight_donors(candidate_path: Path, donor_path: Path) -> None:
    result = donor_coverage_preflight(read_table(candidate_path), read_table(donor_path))
    atomic_json(ROOT / "context" / "donor_preflight.json", result)
    if not result["passed"]:
        raise RuntimeError("fixed donor match criteria did not reach required coverage")


def prepare_context_assignments(donor_path: Path) -> None:
    """Freeze token-matched donor choices before any dose-target outcomes exist."""
    verify_design_artifacts()
    output = ROOT / "context" / "context_assignments.parquet"
    preflight_path = ROOT / "context" / "donor_preflight.json"
    if output.exists() or preflight_path.exists():
        raise FileExistsError("protected context assignments are already frozen")
    candidate_path = ROOT / "candidate_manifest.parquet"
    if not candidate_path.exists():
        raise FileNotFoundError("candidate blocks must be frozen before donor matching")
    candidates = pd.read_parquet(candidate_path)
    donors = read_table(donor_path).sort_values("doc_id").reset_index(drop=True)
    if {"doc_id", "topic", "text"} - set(donors):
        raise ValueError("donor manifest requires doc_id, topic, and text")
    if donors.doc_id.duplicated().any() or set(donors.doc_id) & set(candidates.doc_id):
        raise ValueError("donors must be unique and disjoint from candidate documents")
    tokenizer = tokenizer_local()
    candidate_tokens = {
        row.doc_id: tokenizer(
            row.text,
            add_special_tokens=True,
            truncation=True,
            max_length=MODEL_MAX_TOKENS + 1,
        )["input_ids"]
        for row in candidates.itertuples(index=False)
    }
    transitions = []
    required_keys: set[tuple[int, int, int, int, int]] = set()
    for row in candidates.itertuples(index=False):
        ids = candidate_tokens[row.doc_id]
        for position in range(1, len(ids)):
            lengths = context_length_grid(position, LAR2Config().grid_size)
            shorter, longer = int(lengths[-2]), int(lengths[-1])
            if shorter == longer:
                continue
            key = (int(ids[position]), int(row.topic), position, shorter, longer)
            required_keys.add(key)
            transitions.append(
                {
                    "block_id": row.block_id,
                    "doc_id": row.doc_id,
                    "doc_slot": int(row.doc_slot),
                    "role": row.role,
                    "topic": int(row.topic),
                    "token_position": position,
                    "token_id": int(ids[position]),
                    "shorter_length": shorter,
                    "longer_length": longer,
                    "match_key": key,
                }
            )
    matches: dict[tuple[int, int, int, int, int], list[str]] = {
        key: [] for key in required_keys
    }
    transition_frame = pd.DataFrame(transitions)
    opened = 0
    eligible_fraction = 0.0
    coverage = pd.Series(dtype=float)
    for start in range(0, min(50_000, len(donors)), 5_000):
        batch = donors.iloc[start : min(start + 5_000, len(donors), 50_000)]
        for row in batch.itertuples(index=False):
            ids = tokenizer(
                row.text,
                add_special_tokens=True,
                truncation=True,
                max_length=MODEL_MAX_TOKENS + 1,
            )["input_ids"]
            for position in range(1, len(ids)):
                lengths = context_length_grid(position, LAR2Config().grid_size)
                shorter, longer = int(lengths[-2]), int(lengths[-1])
                key = (int(ids[position]), int(row.topic), position, shorter, longer)
                if key in matches and len(matches[key]) < 32:
                    matches[key].append(str(row.doc_id))
        opened += len(batch)
        transition_frame["valid_match"] = transition_frame.match_key.map(
            lambda key: len(matches[key]) >= 32
        )
        coverage = transition_frame.groupby("doc_id").valid_match.mean()
        eligible_fraction = float(np.mean(coverage >= 0.50))
        print(
            f"donors opened={opened} eligible_document_fraction={eligible_fraction:.6f}",
            flush=True,
        )
        if eligible_fraction >= 0.97:
            break
    passed = eligible_fraction >= 0.97
    atomic_json(
        preflight_path,
        {
            "status": "tokenizer-text-only-donor-preflight-complete",
            "passed": passed,
            "donor_documents_opened": opened,
            "eligible_document_fraction": eligible_fraction,
            "candidate_documents": len(candidates),
            "candidate_transition_coverage_minimum": 0.50,
            "required_matches_per_transition": 32,
            "required_document_fraction": 0.97,
            "increment_documents": 5_000,
            "maximum_donor_documents": 50_000,
            "matching": [
                "realized token",
                "frozen text-only topic",
                "exact token position",
                "exact shorter and longer context lengths",
            ],
            "criteria_relaxed": False,
            "uses_target_outputs": False,
            "candidate_manifest_sha256": sha256(candidate_path),
            "donor_manifest_sha256": sha256(donor_path),
        },
    )
    if not passed:
        raise RuntimeError("fixed donor criteria failed at the 50,000-document ceiling")
    eligible_documents = set(coverage.index[coverage >= 0.50])
    eligible_by_block = (
        candidates[["block_id", "doc_id"]]
        .assign(eligible=lambda value: value.doc_id.isin(eligible_documents))
        .groupby("block_id")
        .eligible.agg(["sum", "size"])
    )
    complete_blocks = set(
        eligible_by_block.index[
            (eligible_by_block["sum"] == 6) & (eligible_by_block["size"] == 6)
        ]
    )
    if not complete_blocks:
        raise RuntimeError("donor coverage passed marginally but left no complete block")
    complete_roles = set(
        candidates.loc[candidates.block_id.isin(complete_blocks), "role"]
    )
    if complete_roles != {"development", "validation", "confirmation"}:
        raise RuntimeError("donor coverage left a study phase without a complete block")
    selected_rows = []
    selected_transitions = transition_frame.loc[
        transition_frame.valid_match
        & transition_frame.doc_id.isin(eligible_documents)
        & transition_frame.block_id.isin(complete_blocks)
    ]
    for row in selected_transitions.itertuples(index=False):
        choices = matches[row.match_key]
        selection_hash = hashlib.sha256(
            f"{row.doc_id}:{row.token_position}:donor-v1".encode()
        ).digest()
        selected = choices[int.from_bytes(selection_hash[:8], "big") % len(choices)]
        selected_rows.append(
            {
                "block_id": row.block_id,
                "doc_id": row.doc_id,
                "doc_slot": int(row.doc_slot),
                "role": row.role,
                "topic": int(row.topic),
                "token_position": int(row.token_position),
                "token_id": int(row.token_id),
                "shorter_length": int(row.shorter_length),
                "longer_length": int(row.longer_length),
                "donor_doc_id": selected,
                "valid_donor_choices": len(choices),
            }
        )
    assignments = pd.DataFrame(selected_rows)
    retained_coverage = assignments.groupby("doc_id").size() / transition_frame.groupby(
        "doc_id"
    ).size()
    if not retained_coverage.ge(0.50).all():
        raise RuntimeError("frozen context assignments include a low-coverage document")
    complete_counts = (
        assignments[["block_id", "doc_id"]]
        .drop_duplicates()
        .groupby("block_id")
        .size()
    )
    if not complete_counts.eq(6).all():
        raise RuntimeError("frozen context assignments contain an incomplete block")
    write_parquet(output, assignments)
    atomic_json(
        ROOT / "context" / "context_assignments.json",
        {
            "status": "protected-context-assignments-frozen-before-training",
            "rows": len(assignments),
            "documents": int(assignments.doc_id.nunique()),
            "complete_blocks": len(complete_blocks),
            "complete_blocks_by_role": {
                str(role): int(group.block_id.nunique())
                for role, group in assignments.groupby("role")
            },
            "marginally_eligible_document_fraction": eligible_fraction,
            "donor_documents_opened": opened,
            "assignment_rule": "SHA256(document, token position) selects one of at least 32 exact matches",
            "context_assignments_sha256": sha256(output),
            "donor_preflight_sha256": sha256(preflight_path),
        },
    )


def _last_distributions(
    model: Any,  # noqa: ANN401
    contexts: list[list[int]],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Evaluate only final-position distributions for variable-length contexts."""
    chunks = []
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size]
        lengths = torch.tensor([len(context) for context in batch], device=device)
        if torch.any(lengths <= 0):
            raise ValueError("context token IDs must be nonempty")
        maximum = int(lengths.max())
        padded = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
        attention = torch.zeros_like(padded)
        for row, context in enumerate(batch):
            padded[row, : len(context)] = torch.tensor(context, device=device)
            attention[row, : len(context)] = 1
        with torch.inference_mode():
            hidden, head = _last_hidden_and_head(model, padded, attention)
            rows = torch.arange(len(batch), device=device)
            probabilities = head(hidden[rows, lengths - 1]).float().softmax(-1)
        chunks.append(probabilities.cpu().numpy())
    return np.concatenate(chunks)


def acquire_context(
    target_id: str,
    phase: str,
    context_path: Path,
    donor_path: Path,
    paths_per_batch: int,
) -> None:
    """Acquire outcomes from frozen compact donor assignments."""
    if TARGETS.get(target_id, {}).get("phase") != phase:
        raise ValueError("target does not belong to requested phase")
    verify_design_artifacts(target_id)
    PhaseAccess(ROOT).require_openable(phase)
    if phase != "development":
        verify_frozen_analysis_software(
            json.loads(PhaseAccess(ROOT).frozen_spec.read_text())
        )
    preflight = ROOT / "context" / "donor_preflight.json"
    if not preflight.exists() or not json.loads(preflight.read_text())["passed"]:
        raise PermissionError("a passing frozen donor preflight is required")
    if phase != "development":
        record_access(
            ROOT / "access" / f"{phase}_{target_id}_context_opened.json",
            phase,
            target_id,
            donor_preflight_sha256=sha256(preflight),
        )
    transitions = read_table(context_path)
    required = {
        "block_id",
        "doc_id",
        "doc_slot",
        "role",
        "token_position",
        "token_id",
        "shorter_length",
        "longer_length",
        "donor_doc_id",
    }
    if required - set(transitions):
        raise ValueError(
            f"context assignment table missing {sorted(required - set(transitions))}"
        )
    transitions = transitions.loc[transitions.role == phase].reset_index(drop=True)
    if phase != "development":
        specification = json.loads(PhaseAccess(ROOT).frozen_spec.read_text())
        transitions = transitions.loc[
            transitions.block_id.isin(specification["frozen_block_ids"][phase])
        ].reset_index(drop=True)
    candidates = pd.read_parquet(ROOT / "candidate_manifest.parquet").set_index("doc_id")
    donors = read_table(donor_path).set_index("doc_id")
    if not set(transitions.doc_id).issubset(candidates.index) or not set(
        transitions.donor_doc_id
    ).issubset(donors.index):
        raise ValueError("frozen context document identities are absent from manifests")
    tokenizer = tokenizer_local()
    candidate_tokens = {
        doc_id: tokenizer(
            candidates.loc[doc_id, "text"],
            add_special_tokens=True,
            truncation=True,
            max_length=MODEL_MAX_TOKENS + 1,
        )["input_ids"]
        for doc_id in transitions.doc_id.unique()
    }
    donor_tokens = {
        doc_id: tokenizer(
            donors.loc[doc_id, "text"],
            add_special_tokens=True,
            truncation=True,
            max_length=MODEL_MAX_TOKENS + 1,
        )["input_ids"]
        for doc_id in transitions.donor_doc_id.unique()
    }
    device = _device()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True
    ).to(device).eval()
    target = _load_target(target_id, device)
    base_config_hash = hashlib.sha256(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "context_assignments_sha256": sha256(context_path),
                "donor_manifest_sha256": sha256(donor_path),
                "version": 1,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    target_checkpoint = ROOT / "targets" / target_id / "checkpoint-100.pt"
    target_config_hash = hashlib.sha256(
        json.dumps(
            {
                "base_config_hash": base_config_hash,
                "target_checkpoint_sha256": sha256(target_checkpoint),
                "version": 1,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    def contexts_for_document(group: pd.DataFrame) -> tuple[list[list[int]], list[dict[str, Any]]]:
        contexts: list[list[int]] = []
        metadata = []
        for row in group.itertuples(index=False):
            position = int(row.token_position)
            shorter_length = int(row.shorter_length)
            longer_length = int(row.longer_length)
            observed = int(row.token_id)
            own_ids = candidate_tokens[row.doc_id]
            donor_ids = donor_tokens[row.donor_doc_id]
            if own_ids[position] != observed or donor_ids[position] != observed:
                raise ValueError("realized-token donor match changed after freeze")
            own_longer = own_ids[position - longer_length : position]
            own_shorter = own_ids[position - shorter_length : position]
            donor_longer = donor_ids[position - longer_length : position]
            donor_shorter = donor_ids[position - shorter_length : position]
            disrupted_longer, severity = deterministic_disruption(
                own_longer, document_id=f"{row.doc_id}:{position}:disrupted-v1"
            )
            disrupted_shorter = disrupted_longer[-shorter_length:]
            for context_type, shorter, longer, disruption_severity in (
                ("own", own_shorter, own_longer, 0.0),
                ("donor", donor_shorter, donor_longer, 0.0),
                ("disrupted", disrupted_shorter, disrupted_longer, severity),
            ):
                pair_index = len(contexts)
                contexts.extend((shorter, longer))
                metadata.append(
                    {
                        "context_type": context_type,
                        "observed": observed,
                        "pair_index": pair_index,
                        "severity": disruption_severity,
                    }
                )
        return contexts, metadata

    def reduce_model(
        model: Any, contexts: list[list[int]], metadata: list[dict[str, Any]]  # noqa: ANN401
    ) -> dict[str, dict[str, float]]:
        distributions = _last_distributions(
            model, contexts, device, max(paths_per_batch, 1)
        )
        buckets = {
            context_type: {
                name: []
                for name in (
                    "A",
                    "L",
                    "R",
                    "realized_logp",
                    "entropy",
                    "severity",
                )
            }
            for context_type in ("own", "donor", "disrupted")
        }
        for item in metadata:
            index = int(item["pair_index"])
            p, q = distributions[index], distributions[index + 1]
            observed = int(item["observed"])
            alignment, length, directed = fisher_rao_alr(
                p[None, :], q[None, :], [observed]
            )
            bucket = buckets[str(item["context_type"])]
            bucket["A"].append(float(alignment[0]))
            bucket["L"].append(float(length[0]))
            bucket["R"].append(float(directed[0]))
            bucket["realized_logp"].append(
                float(np.log(max(q[observed], np.finfo(float).tiny)))
            )
            bucket["entropy"].append(
                float(-np.sum(q * np.log(np.maximum(q, np.finfo(float).tiny))))
            )
            bucket["severity"].append(float(item["severity"]))
        reduced = {}
        for context_type, bucket in buckets.items():
            summaries = frozen_document_summaries(
                bucket["A"], bucket["L"], bucket["R"]
            )
            reduced[context_type] = {
                **summaries,
                "realized_logp": float(np.mean(bucket["realized_logp"])),
                "entropy": float(np.mean(bucket["entropy"])),
                "severity": float(np.mean(bucket["severity"])),
                "transitions": len(bucket["A"]),
            }
        return reduced

    summary_rows = []
    for document_position, (doc_id, group) in enumerate(
        transitions.groupby("doc_id", sort=True), 1
    ):
        candidate = candidates.loc[doc_id]
        cache_name = f"{candidate.text_hash}.json"
        base_cache = ROOT / "context" / "cache" / "base" / phase / cache_name
        target_cache = ROOT / "context" / "cache" / target_id / cache_name
        contexts, metadata = contexts_for_document(group)
        if base_cache.exists():
            base_payload = json.loads(base_cache.read_text())
            if base_payload["config_hash"] != base_config_hash:
                raise ValueError("base context cache configuration changed")
            base_summary = base_payload["summary"]
        else:
            base_summary = reduce_model(base, contexts, metadata)
            atomic_json(
                base_cache,
                {"config_hash": base_config_hash, "summary": base_summary},
            )
        if target_cache.exists():
            target_payload = json.loads(target_cache.read_text())
            if target_payload["config_hash"] != target_config_hash:
                raise ValueError("target context cache configuration changed")
            target_summary = target_payload["summary"]
        else:
            target_summary = reduce_model(target, contexts, metadata)
            atomic_json(
                target_cache,
                {"config_hash": target_config_hash, "summary": target_summary},
            )
        for context_type in ("own", "donor", "disrupted"):
            base_values = base_summary[context_type]
            target_values = target_summary[context_type]
            summary_rows.append(
                {
                    "block_id": group.iloc[0].block_id,
                    "doc_id": doc_id,
                    "doc_slot": int(group.iloc[0].doc_slot),
                    "context_type": context_type,
                    **{
                        f"base_{name}": base_values[name]
                        for name in ("S_A", "S_L", "S_R")
                    },
                    **{
                        f"final_{name}": target_values[name]
                        for name in ("S_A", "S_L", "S_R")
                    },
                    "base_realized_logp": base_values["realized_logp"],
                    "base_entropy": base_values["entropy"],
                    "disruption_severity": base_values["severity"],
                    "eligible_transitions": int(base_values["transitions"]),
                }
            )
        if document_position % 10 == 0:
            print(
                f"{target_id}/{phase} context {document_position}/{transitions.doc_id.nunique()}",
                flush=True,
            )
    summaries = add_checkpoint_changes(pd.DataFrame(summary_rows))
    assignment = pd.read_parquet(ROOT / "assignments.parquet")
    summaries = summaries.merge(
        assignment.loc[assignment.target_id == target_id],
        on=["block_id", "doc_id", "doc_slot"],
        validate="many_to_one",
    )
    summaries["target_id"] = target_id
    output = ROOT / "context" / "outcomes" / phase / f"{target_id}.parquet"
    write_parquet(output, summaries)
    atomic_json(
        output.with_suffix(".json"),
        {
            "status": "protected-context-outcomes-acquired",
            "phase": phase,
            "target_id": target_id,
            "transition_manifest_sha256": sha256(context_path),
            "donor_manifest_sha256": sha256(donor_path),
            "donor_preflight_sha256": sha256(preflight),
            "outcomes_sha256": sha256(output),
        },
    )


def analyze_context(phase: str, draws: int, seed: int) -> None:
    if phase not in {"validation", "confirmation"}:
        raise ValueError("frozen context analysis is validation or confirmation")
    if draws != 99_999:
        raise ValueError("frozen held context analysis requires exactly 99,999 randomization draws")
    if seed != CONTEXT_ANALYSIS_SEEDS[phase]:
        raise ValueError(
            f"frozen {phase} context seed is {CONTEXT_ANALYSIS_SEEDS[phase]}"
        )
    PhaseAccess(ROOT).require_openable(phase)
    verify_design_artifacts()
    targets = [target for target, value in TARGETS.items() if value["phase"] == phase]
    context_frames = []
    for target in targets:
        path = ROOT / "context" / "outcomes" / phase / f"{target}.parquet"
        sidecar = path.with_suffix(".json")
        if not path.exists() or not sidecar.exists():
            raise FileNotFoundError("all phase context outcomes are required")
        manifest = json.loads(sidecar.read_text())
        if (
            manifest.get("target_id") != target
            or manifest.get("phase") != phase
            or manifest.get("outcomes_sha256") != sha256(path)
        ):
            raise PermissionError(f"context outcome provenance failed: {target}")
        context_frames.append(pd.read_parquet(path))
    frame = pd.concat(context_frames, ignore_index=True)
    specification = json.loads(PhaseAccess(ROOT).frozen_spec.read_text())
    verify_frozen_analysis_software(specification)
    selected = specification["frozen_block_ids"][phase]
    frame = frame.loc[frame.block_id.isin(selected)].reset_index(drop=True)
    contrasts = context_specificity_analysis(
        frame, draws=draws, seed=seed, all_target_ids=list(TARGETS)
    )
    completion = {
        "status": f"protected-context-{phase}-complete",
        "phase": phase,
        "contrasts": contrasts,
        "context_specificity_gate_passed": all(bool(result["passed"]) for result in contrasts.values()),
        "primary_own_context_affected_by_donor_failure": False,
        "randomization_draws": draws,
        "randomization_seed": seed,
    }
    output = ROOT / "context" / "analysis" / phase
    if output.exists():
        raise FileExistsError(f"context analysis is already published: {output}")
    context_diagnostics = (
        frame.groupby(["context_type", "K"])
        .agg(
            documents=("doc_id", "size"),
            mean_base_realized_logp=("base_realized_logp", "mean"),
            mean_base_entropy=("base_entropy", "mean"),
            mean_disruption_severity=("disruption_severity", "mean"),
            mean_delta_S_R=("delta_S_R", "mean"),
        )
        .reset_index()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{phase}-context-analysis-staging-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        write_parquet(
            staging / "donor_support_descriptives.parquet",
            donor_support_descriptives(
                frame,
                "delta_S_R",
                float(specification["donor_support_threshold"]),
            ),
        )
        write_parquet(staging / "context_diagnostics.parquet", context_diagnostics)
        completion["artifacts"] = {
            artifact.name: sha256(artifact)
            for artifact in sorted(staging.iterdir())
            if artifact.is_file()
        }
        atomic_json(staging / "completion.json", completion)
        os.replace(staging, output)


def finalize_publication() -> None:
    specification = json.loads(PhaseAccess(ROOT).frozen_spec.read_text())
    verify_frozen_analysis_software(specification)
    completions = {}
    for phase in ("validation", "confirmation"):
        path = ROOT / "analysis" / phase / "completion.json"
        if not path.exists():
            raise FileNotFoundError("both frozen held analyses must complete before publication")
        completions[phase] = json.loads(path.read_text())
        required_artifacts = {
            "per_document.parquet",
            "saturated_delta_S_R.parquet",
            "saturated_delta_S_R.svg",
            "progression_delta_S_R.parquet",
            "progression_delta_S_A.parquet",
            "progression_delta_S_L.parquet",
        }
        if set(completions[phase].get("artifacts", {})) != required_artifacts:
            raise PermissionError("held analysis completion has an incomplete artifact set")
        for name, expected in completions[phase]["artifacts"].items():
            artifact = path.parent / name
            if not artifact.exists() or sha256(artifact) != expected:
                raise PermissionError(f"held analysis artifact changed: {artifact}")
    validation_decision = json.loads(PhaseAccess(ROOT).validation_decision.read_text())
    if validation_decision.get("completion_sha256") != sha256(
        ROOT / "analysis" / "validation" / "completion.json"
    ):
        raise PermissionError("validation decision no longer matches its completion")
    causal = all(result["phase_gate_passed"] for result in completions.values())
    directional = all(
        result["strong_directional_component_pattern_passed"]
        for result in completions.values()
    )
    context_paths = {
        phase: ROOT / "context" / "analysis" / phase / "completion.json"
        for phase in ("validation", "confirmation")
    }
    context_claim = False
    if all(path.exists() for path in context_paths.values()):
        context_completions = {
            phase: json.loads(path.read_text())
            for phase, path in context_paths.items()
        }
        for phase, result in context_completions.items():
            if set(result.get("artifacts", {})) != {
                "context_diagnostics.parquet",
                "donor_support_descriptives.parquet",
            }:
                raise PermissionError(f"incomplete context artifact set: {phase}")
            for name, expected in result["artifacts"].items():
                artifact = context_paths[phase].parent / name
                if not artifact.exists() or sha256(artifact) != expected:
                    raise PermissionError(f"context analysis artifact changed: {artifact}")
        context_claim = all(
            result["context_specificity_gate_passed"]
            for result in context_completions.values()
        )
    supported_components = {
        "R": causal,
        "A": all(
            result["base_randomization_validity"]["passed"]
            and result["decomposition"]["delta_S_A"]["directional_gate_passed"]
            for result in completions.values()
        ),
        "L_equivalence": all(
            result["base_randomization_validity"]["passed"]
            and result["length_equivalence"].get("passed", False)
            for result in completions.values()
        ),
    }
    strongest = (
        "Reallocating a fixed matched training budget toward an exact document "
        "changes the orientation of context-induced predictive motion toward its "
        "realized continuation, without a meaningful increase in total "
        "Fisher–Rao movement."
        if directional
        else None
    )
    donor_preflight_path = ROOT / "context" / "donor_preflight.json"
    donor_preflight = (
        json.loads(donor_preflight_path.read_text())
        if donor_preflight_path.exists()
        else None
    )
    phase_lines = []
    mia_lines = []
    for phase, result in completions.items():
        primary = result["primary"]
        phase_lines.append(
            f"| {phase} | {primary['estimate']:.6g} | "
            f"{primary['p_rand_one_sided']:.6g} | "
            f"[{primary['interval_95'][0]:.6g}, {primary['interval_95'][1]:.6g}] | "
            f"{result['base_randomization_validity']['passed']} | "
            f"{result['phase_gate_passed']} |"
        )
        for method, metric in result["target_only_membership_identification"][
            "methods"
        ].items():
            interval = metric["pooled_interval_95_block_bootstrap"]
            mia_lines.append(
                f"| {phase} | {method} | "
                f"{metric['pooled_auc_descriptive']:.4f} | "
                f"[{interval[0]:.4f}, {interval[1]:.4f}] |"
            )
    if donor_preflight is None:
        context_status = "The protected donor preflight was not completed."
    elif not donor_preflight.get("passed", False):
        context_status = (
            "The protected donor preflight failed under its frozen criteria "
            f"(eligible-document fraction "
            f"{float(donor_preflight['eligible_document_fraction']):.3%}); no donor or "
            "context-specificity claim is made. This does not alter the primary "
            "own-context result."
        )
    else:
        context_status = (
            "The protected donor preflight passed. Replicated held context "
            f"specificity gate: {context_claim}."
        )
    claim_text = (
        strongest
        if strongest is not None
        else (
            "The strongest orientation-without-length claim was not licensed. "
            "Only the individually replicated components marked below are supported."
        )
    )
    report = "\n".join(
        [
            "# Exposure–Observability Allocation Study: Frozen Decision",
            "",
            "## Estimand and access",
            "",
            "For a FineWeb-Edu source document, this randomized experiment estimates "
            "the causal effect of reallocating a fixed matched-block budget during "
            "full-parameter continued pretraining of eight pinned Pythia-70m-deduped "
            "targets. It uses complete next-token distributions from the pinned base "
            "and target checkpoints. It does not establish extractability or natural "
            "membership in an opaque released model.",
            "",
            "## Claim separation",
            "",
            "1. The Fisher–Rao A/L/R identities are exact under φ(p)=2√p.",
            "2. Prior randomized studies established membership-associated score "
            "separation in their declared controlled settings; those results remain "
            "separate from this experiment.",
            "3. The present randomized experiment tests whether fixed-budget "
            "allocation causally changes predictive geometry; its claim is governed "
            "only by the frozen validation and confirmation gates below.",
            "4. Self-influence and accumulated optimizer influence are mechanistic "
            "hypotheses, not conclusions of the allocation test.",
            "",
            "## Frozen primary results",
            "",
            "| Phase | Mean block-FE slope for ΔS_R | One-sided p | 95% design interval | Base validity | Phase gate |",
            "|---|---:|---:|---:|---:|---:|",
            *phase_lines,
            "",
            f"Causal allocation-response claim supported: **{causal}**.",
            "",
            f"R replicated: **{supported_components['R']}**; A replicated: "
            f"**{supported_components['A']}**; L equivalence replicated: "
            f"**{supported_components['L_equivalence']}**.",
            "",
            claim_text,
            "",
            "Self-influence and accumulated optimizer influence remain candidate "
            "mechanistic explanations under every outcome.",
            "",
            "## Target-only membership-identification comparison",
            "",
            "These descriptive evaluations use M=1[K>0]. CATS denotes the frozen "
            "CATS-v4 feature map with coefficients and scaling fitted only on the "
            "development targets; base-assisted Δ geometry is not presented as a "
            "target-only deployment score.",
            "",
            "| Phase | Method | Pooled AUC | 95% block-bootstrap interval |",
            "|---|---|---:|---:|",
            *mia_lines,
            "",
            "## Protected context branch",
            "",
            context_status,
            "",
            "## Prior evidence retained unchanged",
            "",
            "The prior controlled FineWeb full-parameter study reported frozen "
            "CATS-v4 AUC 0.6789 on validation and 0.6675 on confirmation. Those "
            "results are not re-estimated or reinterpreted here.",
            "",
            "On the prior 900-pair MIMIR/Pythia confirmation, loss (0.505), "
            "InfoRMIA mean (0.509), InfoRMIA min-k 20% (0.509), Min-K++ (0.489), "
            "blind bag-of-words (0.512), and Texture localization (about 0.50) "
            "were near chance. This is not evidence of low exposure or noisy "
            "membership labels.",
            "",
            "## Integrity",
            "",
            f"Frozen analysis specification SHA256: `{sha256(PhaseAccess(ROOT).frozen_spec)}`.",
            f"Validation completion SHA256: `{sha256(ROOT / 'analysis' / 'validation' / 'completion.json')}`.",
            f"Confirmation completion SHA256: `{sha256(ROOT / 'analysis' / 'confirmation' / 'completion.json')}`.",
        ]
    )
    report_path = ROOT / "EXPOSURE_OBSERVABILITY_FINAL_REPORT.md"
    if report_path.exists():
        if report_path.read_text().rstrip("\n") != report.rstrip("\n"):
            raise PermissionError("existing immutable final report differs")
    else:
        atomic_text(report_path, report)
    atomic_json(
        ROOT / "publication_decision.json",
        {
            "status": "frozen-publication-decision",
            "causal_allocation_response_claim_supported": causal,
            "context_specificity_claim_supported": context_claim,
            "supported_components": supported_components,
            "strongest_directional_claim": strongest,
            "self_influence_status": "candidate mechanistic explanation only",
            "prior_controlled_fineweb_results": "report unchanged",
            "negative_mimir_pythia_result": "report unchanged; not evidence of low exposure or noisy membership",
            "validation_completion_sha256": sha256(ROOT / "analysis" / "validation" / "completion.json"),
            "confirmation_completion_sha256": sha256(ROOT / "analysis" / "confirmation" / "completion.json"),
            "context_completion_sha256": {
                phase: sha256(path)
                for phase, path in context_paths.items()
                if path.exists()
            },
            "donor_preflight_sha256": (
                sha256(donor_preflight_path)
                if donor_preflight_path.exists()
                else None
            ),
            "final_report": str(report_path),
            "final_report_sha256": sha256(report_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-blocks", "prepare", "prepare-context", "feasibility", "resume-feasibility", "train", "acquire", "summarize-development", "power", "freeze", "analyze", "donor-preflight", "acquire-context", "analyze-context", "finalize"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--donors", type=Path)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--target", choices=tuple(TARGETS))
    parser.add_argument("--phase", choices=("development", "validation", "confirmation"))
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--draws", type=int, default=99_999)
    parser.add_argument("--paths-per-batch", type=int, default=4)
    parser.add_argument("--token-tolerance", type=int, default=8)
    parser.add_argument("--difficulty-tolerance", type=float, default=0.20)
    args = parser.parse_args()
    if args.command == "build-blocks":
        if args.input is None:
            parser.error("build-blocks requires --input")
        build_blocks(args.input, args.seed, args.token_tolerance, args.difficulty_tolerance)
    elif args.command == "prepare":
        if args.background is None:
            parser.error("prepare requires --background for the common global-corpus stream")
        prepare(args.input or ROOT / "candidate_manifest.parquet", args.background)
    elif args.command == "prepare-context":
        if args.donors is None:
            parser.error("prepare-context requires --donors")
        prepare_context_assignments(args.donors)
    elif args.command == "feasibility":
        if args.input is None:
            parser.error("feasibility requires --input with doc_id/text rows")
        feasibility(args.input)
    elif args.command == "resume-feasibility":
        if args.input is None:
            parser.error("resume-feasibility requires --input with doc_id/text rows")
        resume_feasibility(args.input)
    elif args.command == "train":
        if args.target is None:
            parser.error("train requires --target")
        train(args.target)
    elif args.command == "acquire":
        if args.target is None or args.phase is None:
            parser.error("acquire requires --target and --phase")
        acquire(args.target, args.phase, args.paths_per_batch)
    elif args.command == "summarize-development":
        summarize_development()
    elif args.command == "power":
        if args.input is None:
            parser.error("power requires --input simulation JSON")
        run_power(args.input)
    elif args.command == "freeze":
        freeze()
    elif args.command == "analyze":
        if args.phase is None:
            parser.error("analyze requires --phase")
        analyze(args.phase, args.draws, args.seed)
    elif args.command == "donor-preflight":
        if args.input is None or args.donors is None:
            parser.error("donor-preflight requires --input and --donors")
        preflight_donors(args.input, args.donors)
    elif args.command == "acquire-context":
        if args.target is None or args.phase is None or args.donors is None:
            parser.error("acquire-context requires --target, --phase, and --donors")
        acquire_context(
            args.target,
            args.phase,
            args.input or ROOT / "context" / "context_assignments.parquet",
            args.donors,
            args.paths_per_batch,
        )
    elif args.command == "analyze-context":
        if args.phase is None:
            parser.error("analyze-context requires --phase")
        analyze_context(args.phase, args.draws, args.seed)
    elif args.command == "finalize":
        finalize_publication()


if __name__ == "__main__":
    main()
