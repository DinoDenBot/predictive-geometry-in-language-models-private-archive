"""Outcome-blind primitives for the SmolLM2 external-validation study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from exposure_geometry_extension import (
    DOSES,
    dual_duplicate_audit,
    make_six_target_assignments,
    validate_six_target_assignments,
)


MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
CHECKPOINT_TIME = datetime(2025, 2, 6, 10, 34, 41, tzinfo=timezone.utc)
VALIDATION_TARGETS = ("smollm2_1", "smollm2_2", "smollm2_3")
CONFIRMATION_TARGETS = ("smollm2_4", "smollm2_5", "smollm2_6")
ALL_TARGETS = VALIDATION_TARGETS + CONFIRMATION_TARGETS
TARGET_SEEDS = {
    "smollm2_1": 20270101,
    "smollm2_2": 20270102,
    "smollm2_3": 20270103,
    "smollm2_4": 20270104,
    "smollm2_5": 20270105,
    "smollm2_6": 20270106,
}
REQUIRED_PROVENANCE_COLUMNS = {
    "source_id",
    "version_id",
    "first_publication_utc",
    "retrieved_utc",
    "retrieval_endpoint",
    "raw_record",
    "raw_sha256",
    "title",
    "abstract",
    "source",
    "topic",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_field(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def render_document(title: str, abstract: str) -> str:
    """Render the exact Study 3 record without semantic rewriting."""

    rendered_title = normalized_field(title)
    rendered_abstract = normalized_field(abstract)
    if not rendered_title or not rendered_abstract:
        raise ValueError("title and abstract must both be nonempty")
    return f"{rendered_title}\n{rendered_abstract}"


def _utc(value: Any) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks an explicit timezone")
    return parsed.tz_convert("UTC").to_pydatetime()


def provenance_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the formal post-checkpoint exact-record inclusion gate."""

    missing = REQUIRED_PROVENANCE_COLUMNS - set(frame)
    if missing:
        raise ValueError(f"provenance frame missing {sorted(missing)}")
    if frame.empty:
        raise ValueError("provenance frame is empty")
    out = frame.copy()
    duplicated_identity = out.duplicated(["source_id", "version_id"], keep=False)
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(out.itertuples(index=False)):
        reasons: list[str] = []
        try:
            first_publication = _utc(row.first_publication_utc)
        except (TypeError, ValueError):
            first_publication = None
            reasons.append("invalid_first_publication_timestamp")
        try:
            retrieved = _utc(row.retrieved_utc)
        except (TypeError, ValueError):
            retrieved = None
            reasons.append("invalid_retrieval_timestamp")
        if first_publication is not None and first_publication <= CHECKPOINT_TIME:
            reasons.append("not_strictly_post_checkpoint")
        if (
            first_publication is not None
            and retrieved is not None
            and retrieved < first_publication
        ):
            reasons.append("retrieval_precedes_publication")
        raw = str(row.raw_record).encode("utf-8")
        raw_hash = str(row.raw_sha256).lower()
        if not SHA256_PATTERN.fullmatch(raw_hash) or sha256_bytes(raw) != raw_hash:
            reasons.append("raw_record_hash_mismatch")
        if duplicated_identity.iloc[position]:
            reasons.append("duplicate_source_version_identity")
        if not str(row.source_id).strip() or not str(row.version_id).strip():
            reasons.append("missing_stable_identity")
        if not str(row.retrieval_endpoint).strip():
            reasons.append("missing_retrieval_endpoint")
        try:
            text = render_document(row.title, row.abstract)
        except ValueError:
            text = ""
            reasons.append("empty_title_or_abstract")
        rows.append(
            {
                "source_id": str(row.source_id),
                "version_id": str(row.version_id),
                "first_publication_utc": (
                    first_publication.isoformat() if first_publication else None
                ),
                "retrieved_utc": retrieved.isoformat() if retrieved else None,
                "text": text,
                "text_hash": sha256_bytes(text.encode("utf-8")) if text else None,
                "eligible": not reasons,
                "exclusion_reasons": reasons,
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "status": "provenance-gate-complete",
        "checkpoint_revision": MODEL_REVISION,
        "strictly_after_utc": CHECKPOINT_TIME.isoformat(),
        "records": len(detail),
        "eligible_records": int(detail.eligible.sum()),
        "excluded_records": int((~detail.eligible).sum()),
        "semantic_novelty_claim": False,
        "claim": "post-checkpoint exact-record provenance subject to native timestamp and duplicate-screen evidence",
    }
    return detail, summary


def duplicate_audit(
    candidates: pd.DataFrame,
    *,
    background: pd.DataFrame,
    pythia_candidates: pd.DataFrame,
    older_accessible: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run both near-duplicate rules against every frozen reference set."""

    references = pd.concat(
        [
            background[["doc_id", "text"]],
            pythia_candidates[["doc_id", "text"]],
            older_accessible[["doc_id", "text"]],
        ],
        ignore_index=True,
    )
    if references.doc_id.duplicated().any():
        raise ValueError("duplicate reference IDs must be namespaced before screening")
    detail, summary = dual_duplicate_audit(candidates, references, threshold=0.80)
    summary.update(
        {
            "reference_populations": {
                "background": len(background),
                "pythia_candidates": len(pythia_candidates),
                "older_accessible": len(older_accessible),
            },
            "semantic_novelty_claim": False,
        }
    )
    return detail, summary


def validate_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "block_id",
        "doc_id",
        "doc_slot",
        "role",
        "source",
        "topic",
        "tokens",
        "baseline_difficulty",
    }
    if required - set(frame):
        raise ValueError(f"manifest missing {sorted(required - set(frame))}")
    if len(frame) != 3600 or frame.block_id.nunique() != 600:
        raise ValueError("Study 3 requires 600 six-document candidate blocks")
    if frame.doc_id.duplicated().any():
        raise ValueError("candidate document IDs must be unique")
    role_counts = (
        frame[["block_id", "role"]]
        .drop_duplicates()
        .groupby("role")
        .size()
        .to_dict()
    )
    if role_counts != {"confirmation": 300, "validation": 300}:
        raise ValueError("Study 3 requires 300 blocks per phase before power selection")
    for block_id, block in frame.groupby("block_id", sort=False):
        if len(block) != 6 or set(block.doc_slot.astype(int)) != set(range(6)):
            raise ValueError(f"block {block_id} does not contain slots 0..5")
        if block.role.nunique() != 1 or block.source.nunique() != 1 or block.topic.nunique() != 1:
            raise ValueError(f"block {block_id} violates role/source/topic matching")
        if not np.isfinite(block[["tokens", "baseline_difficulty"]].to_numpy(float)).all():
            raise ValueError(f"block {block_id} has nonfinite matching data")
    return frame.sort_values(["block_id", "doc_slot"]).reset_index(drop=True)


def make_assignments(manifest: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    canonical = validate_manifest(manifest)
    result = make_six_target_assignments(canonical, seed=seed, target_ids=ALL_TARGETS)
    validate_six_target_assignments(result, ALL_TARGETS)
    return result


def select_primary_blocks(
    manifest: pd.DataFrame, *, blocks_per_phase: int, seed: int
) -> dict[str, list[Any]]:
    canonical = validate_manifest(manifest)
    if blocks_per_phase not in {200, 250, 300}:
        raise ValueError("primary block count must be one of 200, 250, or 300")
    rng = np.random.default_rng(seed)
    selected: dict[str, list[Any]] = {}
    for role in ("validation", "confirmation"):
        blocks = np.asarray(
            sorted(canonical.loc[canonical.role == role, "block_id"].unique())
        )
        selected[role] = sorted(
            rng.choice(blocks, size=blocks_per_phase, replace=False).tolist()
        )
    return selected


def validate_target_seeds(seeds: Mapping[str, int] = TARGET_SEEDS) -> None:
    if set(seeds) != set(ALL_TARGETS):
        raise ValueError("seed ledger must name exactly the six Study 3 targets")
    values = [int(seeds[target]) for target in ALL_TARGETS]
    if len(set(values)) != 6:
        raise ValueError("all Study 3 stochastic seeds must be distinct")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


class Study3Access:
    """Enforce validation, independent confirmation, and complementary gates."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def freeze(self) -> Path:
        return self.root / "freeze.json"

    def decision(self, name: str) -> Path:
        locations = {
            "validation_decision": self.root / "analysis" / "validation" / "decision.json",
            "N_power_decision": self.root / "analysis" / "validation" / "N_power_decision.json",
            "confirmation_decision": self.root / "analysis" / "confirmation" / "decision.json",
        }
        if name not in locations:
            raise ValueError(f"unknown Study 3 decision: {name}")
        return locations[name]

    def _complete(self, name: str) -> bool:
        path = self.decision(name)
        if not path.is_file() or not self.freeze.is_file():
            return False
        decision = json.loads(path.read_text())
        return bool(
            decision.get("decision_complete") is True
            and decision.get("freeze_sha256") == sha256_file(self.freeze)
        )

    def require_openable(self, role: str, target_id: str) -> str:
        if not self.freeze.is_file():
            raise PermissionError("held outcomes remain sealed until freeze.json exists")
        if role == "validation" and target_id in VALIDATION_TARGETS:
            return "primary_validation"
        if role == "confirmation" and target_id in CONFIRMATION_TARGETS:
            if not self._complete("validation_decision"):
                raise PermissionError("confirmation is sealed until validation decision completion")
            if not self._complete("N_power_decision"):
                raise PermissionError("confirmation is sealed until the N-power decision is frozen")
            return "primary_independent_target_confirmation"
        complementary = (
            role == "validation" and target_id in CONFIRMATION_TARGETS
        ) or (role == "confirmation" and target_id in VALIDATION_TARGETS)
        if complementary:
            if not self._complete("confirmation_decision"):
                raise PermissionError("complementary cells are sealed until confirmation completion")
            return "post_confirmation_sensitivity_no_primary_authority"
        raise ValueError("role and target do not identify an authorized Study 3 cell")

    def record_open(
        self, role: str, target_id: str, *, checkpoint_sha256: str
    ) -> Path:
        purpose = self.require_openable(role, target_id)
        if not SHA256_PATTERN.fullmatch(checkpoint_sha256):
            raise ValueError("checkpoint_sha256 is invalid")
        output = self.root / "access" / "opens" / f"{role}_{target_id}.json"
        record = {
            "role": role,
            "target_id": target_id,
            "purpose": purpose,
            "checkpoint_sha256": checkpoint_sha256,
            "freeze_sha256": sha256_file(self.freeze),
        }
        if not output.exists():
            atomic_json(output, record)
        if json.loads(output.read_text()) != record:
            raise PermissionError(f"existing access record differs: {output}")
        return output


def residual_scale(frame: pd.DataFrame, outcome: str) -> float:
    """Validation SD after additive target and block adjustment."""

    required = {"target_id", "block_id", outcome}
    if required - set(frame):
        raise ValueError(f"validation frame missing {sorted(required - set(frame))}")
    values = frame[outcome].astype(float)
    adjusted = (
        values
        - values.groupby(frame.target_id).transform("mean")
        - values.groupby(frame.block_id).transform("mean")
        + float(values.mean())
    ).to_numpy()
    rank = frame.target_id.nunique() + frame.block_id.nunique() - 1
    degrees = len(frame) - rank
    if degrees <= 0:
        raise ValueError("insufficient residual degrees of freedom")
    scale = math.sqrt(float(adjusted @ adjusted) / degrees)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("validation residual scale must be positive")
    return scale
