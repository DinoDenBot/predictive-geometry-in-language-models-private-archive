"""Prepare and freeze the numerically qualified v3 retrieval design."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from context_selective_retrieval_v3.training import (
    MAX_SEQUENCE_TOKENS,
    MODEL_NAME,
    MODEL_REVISION,
    TARGET_TEXT,
    TRIGGER,
    repeat_to_length,
    stable_key,
    tokenizer_local,
)

SEED = 20260912
SOURCE_DEFAULT = Path(
    "/Volumes/My Passport/data_inference/exposure_geometry_extension_v2/"
    "design/background_manifest.parquet"
)
ROOT_DEFAULT = Path(
    os.environ.get(
        "RETRIEVAL_V3_ROOT",
        "/Volumes/My Passport/data_inference/context_selective_retrieval_v3",
    )
)

PROFILES: dict[str, dict[str, Any]] = {
    "micro": {
        "batch_size": 4,
        "common_steps": 4,
        "branch_steps": 8,
        "training_documents": 80,
        "candidate_fillers": 4,
        "oracle_fit_long_sources": 2,
        "oracle_validation_long_sources": 2,
        "confirmation_long_sources": 3,
        "oracle_fit_clean_sources": 4,
        "oracle_validation_clean_sources": 4,
        "confirmation_clean_sources": 6,
        "development_seeds": [52001],
        "confirmation_seeds": [62001],
        "doses": [1, 4, 8],
        "gradient_shortlist": 4,
        "branch_shortlist": 2,
        "cg_iterations": 12,
    },
    "full": {
        "batch_size": 8,
        "common_steps": 128,
        "branch_steps": 64,
        "training_documents": 2400,
        "candidate_fillers": 64,
        "oracle_fit_long_sources": 8,
        "oracle_validation_long_sources": 8,
        "confirmation_long_sources": 24,
        "oracle_fit_clean_sources": 32,
        "oracle_validation_clean_sources": 32,
        "confirmation_clean_sources": 128,
        "development_seeds": [52001, 52002],
        "confirmation_seeds": [62001, 62002, 62003, 62004, 62005],
        "doses": [1, 4, 16, 64],
        "gradient_shortlist": 16,
        "branch_shortlist": 4,
        "cg_iterations": 30,
    },
}

CODE_FILES = (
    "__init__.py",
    "design.py",
    "geometry.py",
    "numerics.py",
    "pipeline.py",
    "report.py",
    "test_geometry.py",
    "test_numerics.py",
    "test_pipeline.py",
    "training.py",
    "CALIBRATION_SPEC.md",
    "SPECIFICATION.md",
    "run_pipeline.sh",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def code_hashes() -> dict[str, str]:
    package = Path(__file__).parent
    values = {name: sha256(package / name) for name in CODE_FILES}
    return values


def _make_candidates(tokenizer: Any, filler_texts: list[str]) -> tuple[dict[str, list[int]], list[int], int]:
    trigger = [int(x) for x in tokenizer(TRIGGER, add_special_tokens=False)["input_ids"]]
    targets = tokenizer(TARGET_TEXT, add_special_tokens=False)["input_ids"]
    period = tokenizer(".", add_special_tokens=False)["input_ids"]
    if len(targets) != 1 or len(period) != 1:
        raise RuntimeError("target or period tokenization changed")
    target = int(targets[0])
    association = [*trigger, target, int(period[0])]
    candidates: dict[str, list[int]] = {}
    fillers = [
        [int(x) for x in tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]]
        for text in filler_texts
    ]
    for index, filler_values in enumerate(fillers):
        filler = repeat_to_length(filler_values, MAX_SEQUENCE_TOKENS)
        candidates[f"start_{index:03d}"] = (association + filler)[:MAX_SEQUENCE_TOKENS]
        candidates[f"late_{index:03d}"] = (filler[:48] + association + filler[48:])[:MAX_SEQUENCE_TOKENS]
    candidates["repeat"] = repeat_to_length(association, MAX_SEQUENCE_TOKENS)
    neutral = tokenizer(
        " A concise technical note follows with ordinary examples and definitions.",
        add_special_tokens=False,
    )["input_ids"]
    candidates["minimal"] = (association + repeat_to_length(neutral, MAX_SEQUENCE_TOKENS))[
        :MAX_SEQUENCE_TOKENS
    ]
    needle = [*trigger, target]
    for name, values in candidates.items():
        if len(values) != MAX_SEQUENCE_TOKENS:
            raise RuntimeError(f"candidate length mismatch: {name}")
        if not any(values[i : i + len(needle)] == needle for i in range(len(values) - len(needle) + 1)):
            raise RuntimeError(f"candidate lacks association: {name}")
    return candidates, trigger, target


def prepare(root: Path, source: Path, profile: str, qualification_path: Path) -> None:
    completion = root / "design/completion.json"
    if completion.exists():
        raise FileExistsError(completion)
    cfg = dict(PROFILES[profile])
    qualification = json.loads(qualification_path.read_text())
    decision = qualification.get("decision", {})
    if qualification.get("status") != "qualification-passed" or not decision.get("passed"):
        raise PermissionError("a passing numerical qualification is required")
    selected_method = decision.get("selected_method")
    if selected_method not in {"svd", "rrqr", "refined_lstsq"}:
        raise PermissionError("qualification selected an unknown numerical method")
    frame = pd.read_parquet(source)
    required = {"doc_id", "text", "tokens", "text_hash"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing source columns: {sorted(required - set(frame.columns))}")
    frame["doc_id"] = frame.doc_id.astype(str)
    eligible = frame.loc[frame.tokens >= 80].copy()
    eligible["order"] = eligible.doc_id.map(lambda value: stable_key(value, SEED + 2))
    eligible = eligible.sort_values("order", kind="stable").reset_index(drop=True)
    names = [
        "training",
        "candidate_fillers",
        "oracle_fit_long",
        "oracle_validation_long",
        "confirmation_long",
        "oracle_fit_clean",
        "oracle_validation_clean",
        "confirmation_clean",
    ]
    counts = [cfg[f"{name}_sources"] if name not in {"training", "candidate_fillers"} else cfg[f"{name}_documents"] if name == "training" else cfg["candidate_fillers"] for name in names]
    if sum(counts) > len(eligible):
        raise RuntimeError("source pool is too small")
    partitions: dict[str, list[str]] = {}
    offset = 0
    for name, count in zip(names, counts, strict=True):
        partitions[name] = eligible.iloc[offset : offset + count].doc_id.tolist()
        offset += count
    allocated = [item for values in partitions.values() for item in values]
    if len(allocated) != len(set(allocated)):
        raise RuntimeError("document partitions overlap")
    lookup = frame.set_index("doc_id")
    tokenizer = tokenizer_local()
    candidates, trigger_ids, target_id = _make_candidates(
        tokenizer, [str(lookup.loc[x, "text"]) for x in partitions["candidate_fillers"]]
    )

    def encode(doc_id: str) -> list[int]:
        ids = tokenizer(str(lookup.loc[doc_id, "text"]), add_special_tokens=False)["input_ids"]
        if len(ids) < 65:
            raise RuntimeError(f"context source too short after tokenization: {doc_id}")
        return [int(x) for x in ids]

    contexts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in ("oracle_fit", "oracle_validation", "confirmation"):
        long_rows = []
        for doc_id in partitions[f"{split}_long"]:
            ids = encode(doc_id)
            for length in (8, 32, 64):
                long_rows.append(
                    {"doc_id": doc_id, "prefix_length": length, "input_ids": ids[-length:] + trigger_ids}
                )
        clean_rows = [
            {"doc_id": doc_id, "input_ids": encode(doc_id)[-64:]}
            for doc_id in partitions[f"{split}_clean"]
        ]
        contexts[split] = {"long": long_rows, "clean": clean_rows}

    design = {
        "version": "context-selective-retrieval-v3",
        "profile": profile,
        "seed": SEED,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "source_path": str(source),
        "source_sha256": sha256(source),
        "code_sha256": code_hashes(),
        "trigger": TRIGGER,
        "trigger_ids": trigger_ids,
        "target_text": TARGET_TEXT,
        "target_id": target_id,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "partitions": partitions,
        "partition_text_hashes": {x: str(lookup.loc[x, "text_hash"]) for x in allocated},
        "candidates": candidates,
        "contexts": contexts,
        "training": {
            **cfg,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "adam_epsilon": 1e-4,
            "warmup_steps": 100,
            "gradient_clip": 1.0,
            "replacement_position": 0,
            "replacement_steps": "last r branch steps",
        },
        "oracle": {
            "weights": {
                "long": 1.0,
                "clean": 1.0,
                "short_orthogonal": 0.25,
                "short_gap": 0.25
            },
            "method": selected_method,
            "restricted_parameter": "embed_out.weight[target_id]",
            "certification": {
                "relative_backward_error_max": 128.0 * 2.220446049250313e-16 * 512,
                "unit_gain_error_max": 128.0 * 2.220446049250313e-16 * 512,
                "postcast_leakage_ratio_max": 32.0 * 1.1920928955078125e-7 * 512,
                "rank_cutoff_factors": [0.1, 1.0, 10.0],
                "old_absolute_residual_1e_8_is_diagnostic_only": True,
            },
            "qualification_sha256": sha256(qualification_path),
            "linear_binary_step_grid": [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10],
            "selection": "smallest trust-valid step passing the development gate; otherwise highest development objective with smaller-step tie break",
            "maximum_relative_parameter_norm": 0.05,
        },
        "screening": {
            "gradient_shortlist": cfg["gradient_shortlist"],
            "branch_shortlist": cfg["branch_shortlist"],
            "absolute_linear_gain_floor": 1e-5,
            "damping": 1e-8,
            "baselines": ["repeat", "minimal"],
            "tie_rule": "lexicographically smaller candidate_id",
        },
        "success_gate": {
            "short_probability": 0.05,
            "selectivity_margin": 0.0,
            "clean_rms_max": 0.05,
            "replication": "all five confirmation seeds",
        },
        "numerical_gate": {
            "maximum_fisher_rao_distance": 1e-6,
            "deterministic_cuda_required_for_full": True,
            "allow_tf32": False,
        },
    }
    atomic_json(root / "design/qualification.json", qualification)
    atomic_json(root / "design/design.json", design)
    atomic_json(
        completion,
        {
            "status": "design-complete-before-model-execution",
            "design_sha256": sha256(root / "design/design.json"),
            "source_sha256": sha256(source),
            "qualification_sha256": sha256(root / "design/qualification.json"),
            "candidate_count": len(candidates),
            "partition_documents": len(allocated),
            "context_counts": {
                split: {kind: len(rows) for kind, rows in values.items()}
                for split, values in contexts.items()
            },
        },
    )


def load_design(root: Path, source: Path) -> dict[str, Any]:
    design_path, completion_path = root / "design/design.json", root / "design/completion.json"
    if not design_path.exists() or not completion_path.exists():
        raise FileNotFoundError("prepared design is absent")
    design = json.loads(design_path.read_text())
    completion = json.loads(completion_path.read_text())
    if sha256(design_path) != completion["design_sha256"]:
        raise PermissionError("design changed after freezing")
    if sha256(source) != design["source_sha256"]:
        raise PermissionError("source manifest changed")
    if code_hashes() != design["code_sha256"]:
        raise PermissionError("implementation changed after freezing")
    qualification_path = root / "design/qualification.json"
    if sha256(qualification_path) != design["oracle"]["qualification_sha256"]:
        raise PermissionError("numerical qualification changed after freezing")
    return design
