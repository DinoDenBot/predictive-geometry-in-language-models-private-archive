from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import freeze_study3
from study3 import (
    ALL_TARGETS,
    Study3Access,
    duplicate_audit,
    make_assignments,
    provenance_audit,
    render_document,
    residual_scale,
    select_primary_blocks,
    sha256_file,
    validate_target_seeds,
)


def _manifest() -> pd.DataFrame:
    rows = []
    for block in range(600):
        role = "validation" if block < 300 else "confirmation"
        for slot in range(6):
            rows.append(
                {
                    "block_id": block,
                    "doc_id": f"study3-{block:03d}-{slot}",
                    "doc_slot": slot,
                    "role": role,
                    "source": "arxiv",
                    "topic": f"topic-{block // 20}",
                    "tokens": 100 + slot,
                    "baseline_difficulty": 3.0 + slot / 100,
                }
            )
    return pd.DataFrame(rows)


def _provenance(**overrides: object) -> pd.DataFrame:
    raw = '{"id":"2601.00001v1"}'
    row = {
        "source_id": "2601.00001",
        "version_id": "v1",
        "first_publication_utc": "2026-01-02T03:04:05Z",
        "retrieved_utc": "2026-09-02T08:00:00Z",
        "retrieval_endpoint": "https://export.arxiv.org/api/query?id_list=2601.00001v1",
        "raw_record": raw,
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "title": "  A   technical title ",
        "abstract": "An abstract\nwith whitespace.",
        "source": "arxiv",
        "topic": "cs.LG",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_exact_record_renderer_is_deterministic_without_semantic_rewriting() -> None:
    assert render_document(" A  title ", "line one\n line two") == "A title\nline one line two"


def test_provenance_gate_requires_post_checkpoint_verified_native_record() -> None:
    detail, summary = provenance_audit(_provenance())
    assert detail.loc[0, "eligible"]
    assert summary["semantic_novelty_claim"] is False
    old, _ = provenance_audit(
        _provenance(first_publication_utc="2025-02-06T10:34:41Z")
    )
    assert not old.loc[0, "eligible"]
    assert "not_strictly_post_checkpoint" in old.loc[0, "exclusion_reasons"]
    mismatched, _ = provenance_audit(_provenance(raw_sha256="0" * 64))
    assert "raw_record_hash_mismatch" in mismatched.loc[0, "exclusion_reasons"]


def test_provenance_gate_rejects_naive_timestamps_and_duplicate_identity() -> None:
    duplicated = pd.concat([_provenance(), _provenance()], ignore_index=True)
    detail, _ = provenance_audit(duplicated)
    assert (~detail.eligible).all()
    assert all("duplicate_source_version_identity" in value for value in detail.exclusion_reasons)
    naive, _ = provenance_audit(_provenance(first_publication_utc="2026-01-01 00:00:00"))
    assert "invalid_first_publication_timestamp" in naive.loc[0, "exclusion_reasons"]


def test_duplicate_audit_checks_all_named_reference_populations() -> None:
    candidates = pd.DataFrame({"doc_id": ["candidate"], "text": ["one two three four five six seven"]})
    background = pd.DataFrame({"doc_id": ["background"], "text": ["different words make a clean background record"]})
    pythia = pd.DataFrame({"doc_id": ["pythia"], "text": ["one two three four five six seven eight"]})
    older = pd.DataFrame({"doc_id": ["older"], "text": ["another entirely separate archived source text"]})
    detail, summary = duplicate_audit(
        candidates, background=background, pythia_candidates=pythia, older_accessible=older
    )
    assert not detail.loc[0, "passes"]
    assert summary["reference_populations"] == {
        "background": 1,
        "pythia_candidates": 1,
        "older_accessible": 1,
    }


def test_complete_latin_rotation_and_primary_block_selection_replay() -> None:
    manifest = _manifest()
    first = make_assignments(manifest, seed=20270201)
    second = make_assignments(manifest, seed=20270201)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.target_id) == set(ALL_TARGETS)
    assert first.groupby(["block_id", "target_id"]).K.sum().eq(31).all()
    rotations = first.groupby(["block_id", "doc_id"]).K.apply(set)
    assert rotations.map(lambda value: value == {0, 1, 2, 4, 8, 16}).all()
    selected = select_primary_blocks(manifest, blocks_per_phase=200, seed=20270202)
    assert selected == select_primary_blocks(manifest, blocks_per_phase=200, seed=20270202)
    assert len(selected["validation"]) == len(selected["confirmation"]) == 200


def test_independent_seeds_and_phase_access(tmp_path) -> None:
    validate_target_seeds()
    access = Study3Access(tmp_path)
    with pytest.raises(PermissionError):
        access.require_openable("validation", "smollm2_1")
    access.freeze.write_text("{}")
    assert access.require_openable("validation", "smollm2_1") == "primary_validation"
    with pytest.raises(PermissionError):
        access.require_openable("confirmation", "smollm2_4")
    freeze_hash = sha256_file(access.freeze)
    access.decision("validation_decision").parent.mkdir(parents=True)
    access.decision("validation_decision").write_text(
        json.dumps({"decision_complete": True, "freeze_sha256": freeze_hash})
    )
    with pytest.raises(PermissionError):
        access.require_openable("confirmation", "smollm2_4")
    access.decision("N_power_decision").write_text(
        json.dumps({"decision_complete": True, "freeze_sha256": freeze_hash})
    )
    assert access.require_openable("confirmation", "smollm2_4") == "primary_independent_target_confirmation"
    first_open = access.record_open("confirmation", "smollm2_4", checkpoint_sha256="a" * 64)
    assert access.record_open("confirmation", "smollm2_4", checkpoint_sha256="a" * 64) == first_open
    with pytest.raises(PermissionError):
        access.require_openable("validation", "smollm2_4")
    access.decision("confirmation_decision").parent.mkdir(parents=True)
    access.decision("confirmation_decision").write_text(
        json.dumps({"decision_complete": True, "freeze_sha256": freeze_hash})
    )
    assert "post_confirmation" in access.require_openable("validation", "smollm2_4")


def test_validation_residual_scale_removes_target_and_block_means() -> None:
    rng = np.random.default_rng(8)
    rows = []
    for target in ["smollm2_1", "smollm2_2", "smollm2_3"]:
        for block in range(20):
            for slot in range(6):
                rows.append(
                    {
                        "target_id": target,
                        "block_id": block,
                        "delta_S_N": 4 * int(target[-1]) + block / 2 + rng.normal(),
                    }
                )
    scale = residual_scale(pd.DataFrame(rows), "delta_S_N")
    assert 0.7 < scale < 1.3


def test_freeze_snapshot_verifies_and_detects_tampering(tmp_path, monkeypatch) -> None:
    root = tmp_path / "study"
    package = tmp_path / "package"
    package.mkdir()
    (package / "runner.py").write_text("print('frozen')\n")
    (package / "protocol.json").write_text("{}\n")
    manuscript = tmp_path / "paper.tex"
    manuscript.write_text("frozen paper\n")
    (root / "input").parent.mkdir(parents=True)
    (root / "input").write_text("design\n")
    for index in range(1, 7):
        ledger = root / "design" / "ledgers" / f"smollm2_{index}.parquet"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_bytes(f"ledger-{index}".encode())
    monkeypatch.setattr(freeze_study3, "SOFTWARE", ("runner.py",))
    monkeypatch.setattr(freeze_study3, "SPECIFICATION", ("protocol.json",))
    monkeypatch.setattr(freeze_study3, "ROOT_INPUTS", ("input",))
    freeze_study3.build(root, package, manuscript, microbatch=8)
    assert freeze_study3.verify(root)["status"] == "verified"
    (root / "input").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="input"):
        freeze_study3.verify(root)
