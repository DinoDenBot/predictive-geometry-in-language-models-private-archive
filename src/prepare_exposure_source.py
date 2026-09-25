#!/usr/bin/env python3
"""Prepare fresh pinned FineWeb-Edu inputs for the allocation study.

The three commands are deliberately target-output-free and immutable:

1. ``stream`` creates disjoint raw candidate/background/donor lanes.
2. ``score`` obtains pinned-base document difficulty for candidates only.
3. ``finalize`` performs cross-lane near-duplicate removal and freezes one
   text-only topic map used by candidate blocks and donor matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(
    os.environ.get("EXPOSURE_STUDY_ROOT", "results/exposure_observability_v1")
).expanduser()
SOURCE = ROOT / "source"
BASE_MODEL = "EleutherAI/pythia-70m-deduped"
BASE_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DATASET_CONFIG = "CC-MAIN-2024-51"
MIN_TOKENS = 48
MAX_TOKENS = 128
MIN_LANGUAGE_SCORE = 0.90
MIN_EDU_SCORE = 3
TOPIC_CLUSTERS = 12
SOURCE_SEED = 20260929
DEFAULT_CANDIDATES = 12_000
DEFAULT_BACKGROUND = 21_700
DEFAULT_DONORS = 50_000
DEFAULT_MAX_RECORDS = 30_000_000
STREAM_FLUSH_DOCUMENTS = 1_000
NEAR_DUPLICATE_THRESHOLD = 0.80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:  # noqa: ANN401
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def replace_json(path: Path, value: Any) -> None:  # noqa: ANN401
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
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


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).lower()).strip()


def normalized_hash(value: str) -> str:
    return hashlib.sha256(normalized_text(value).encode()).hexdigest()


def source_bucket(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().strip(".")
    suffix = host.rsplit(".", 1)[-1] if "." in host else "other"
    return suffix if suffix in {"com", "org", "net", "edu", "gov", "uk", "au", "ca"} else "other"


def source_domain(url: str) -> str:
    return (urlparse(url).hostname or "unknown").lower().strip(".") or "unknown"


def tokenizer_local() -> Any:  # noqa: ANN401
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prior_hashes() -> tuple[set[str], set[str], list[str]]:
    exact: set[str] = set()
    normalized: set[str] = set()
    sources = []
    for path in sorted(Path("results").glob("**/population.parquet")):
        if ROOT in path.parents:
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        if "text_hash" in frame:
            exact.update(frame.text_hash.astype(str))
        if "normalized_hash" in frame:
            normalized.update(frame.normalized_hash.astype(str))
        elif "text" in frame:
            normalized.update(frame.text.astype(str).map(normalized_hash))
        sources.append(str(path))
    return exact, normalized, sources


def prior_text_references() -> tuple[pd.DataFrame, list[str]]:
    """Load prior population text solely for the frozen near-duplicate audit."""
    frames = []
    sources = []
    for path in sorted(Path("results").glob("**/population.parquet")):
        if ROOT in path.parents:
            continue
        try:
            frame = pd.read_parquet(path, columns=["text"])
        except (KeyError, OSError, ValueError):
            continue
        frame = frame.loc[frame.text.notna(), ["text"]].copy()
        if frame.empty:
            continue
        frames.append(frame)
        sources.append(str(path))
    if not frames:
        return pd.DataFrame({"text": pd.Series(dtype=str)}), sources
    combined = pd.concat(frames, ignore_index=True)
    combined["normalized_hash"] = combined.text.astype(str).map(normalized_hash)
    return combined.drop_duplicates("normalized_hash")[["text"]], sources


def _lane(digest: str, counts: dict[str, int], quotas: dict[str, int]) -> str | None:
    available = [name for name in ("candidate", "background", "donor") if counts[name] < quotas[name]]
    if not available:
        return None
    total = sum(quotas[name] for name in available)
    point = int(digest[:16], 16) % total
    cumulative = 0
    for name in available:
        cumulative += quotas[name]
        if point < cumulative:
            return name
    raise AssertionError("hash partition failed")


def stream_source(candidate_documents: int, background_documents: int, donor_documents: int, max_records: int) -> None:
    paths = {name: SOURCE / f"raw_{name}.parquet" for name in ("candidate", "background", "donor")}
    if any(path.exists() for path in paths.values()) or (SOURCE / "stream_manifest.json").exists():
        raise FileExistsError("source stream identities are already fixed")
    quotas = {"candidate": candidate_documents, "background": background_documents, "donor": donor_documents}
    parts_root = SOURCE / "stream_parts"
    part_paths = {
        name: sorted((parts_root / name).glob("part-*.parquet"))
        for name in quotas
    }
    completed = {
        name: (
            pd.concat([pd.read_parquet(path) for path in part_paths[name]], ignore_index=True)
            if part_paths[name]
            else pd.DataFrame()
        )
        for name in quotas
    }
    counts = {name: len(completed[name]) for name in quotas}
    if any(counts[name] > quotas[name] for name in quotas):
        raise ValueError("stream parts exceed declared quota")
    progress_path = SOURCE / "stream_progress.json"
    resume_after = -1
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("quotas") != quotas:
            raise ValueError("stream resume quotas differ from the original attempt")
        resume_after = int(progress["scanned_source_index"])
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in quotas}
    tokenizer = tokenizer_local()
    excluded_exact, excluded_normalized, prior_sources = prior_hashes()
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    for frame in completed.values():
        if not frame.empty:
            seen_exact.update(frame.text_hash.astype(str))
            seen_normalized.update(frame.normalized_hash.astype(str))
    stream = load_dataset(
        DATASET,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    )
    started = time.time()
    scanned = 0

    def flush(source_index: int) -> None:
        for name in quotas:
            if not rows[name]:
                continue
            directory = parts_root / name
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"part-{len(part_paths[name]):06d}.parquet"
            atomic_parquet(path, pd.DataFrame(rows[name]))
            part_paths[name].append(path)
            rows[name].clear()
        replace_json(
            progress_path,
            {
                "status": "target-output-free-stream-in-progress",
                "quotas": quotas,
                "counts": counts,
                "scanned_source_index": source_index,
                "part_hashes": {
                    name: {path.name: sha256(path) for path in part_paths[name]}
                    for name in quotas
                },
            },
        )

    for source_index, record in enumerate(stream):
        scanned = source_index + 1
        if source_index <= resume_after:
            continue
        if scanned > max_records or all(counts[name] >= quotas[name] for name in quotas):
            break
        if float(record["language_score"]) < MIN_LANGUAGE_SCORE or int(record["int_score"]) < MIN_EDU_SCORE:
            continue
        text = re.sub(r"\s+", " ", str(record["text"])).strip()
        exact = hashlib.sha256(text.encode()).hexdigest()
        normalized = normalized_hash(text)
        if exact in excluded_exact or normalized in excluded_normalized or exact in seen_exact or normalized in seen_normalized:
            continue
        ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        predicted_tokens = len(ids) - 1
        if not MIN_TOKENS <= predicted_tokens <= MAX_TOKENS:
            continue
        lane = _lane(exact, counts, quotas)
        if lane is None:
            break
        seen_exact.add(exact)
        seen_normalized.add(normalized)
        rows[lane].append(
            {
                "doc_id": f"fw-{record['id']}",
                "source_index": source_index,
                "fineweb_id": str(record["id"]),
                "url": str(record["url"]),
                "date": str(record["date"]),
                "dump": str(record["dump"]),
                "source": source_bucket(str(record["url"])),
                "domain": source_domain(str(record["url"])),
                "language_score": float(record["language_score"]),
                "edu_score": int(record["int_score"]),
                "tokens": predicted_tokens,
                "text": text,
                "text_hash": exact,
                "normalized_hash": normalized,
            }
        )
        counts[lane] += 1
        if sum(len(values) for values in rows.values()) >= STREAM_FLUSH_DOCUMENTS:
            flush(source_index)
        if scanned % 50_000 == 0:
            print(f"scanned={scanned} counts={counts}", flush=True)
    if counts != quotas:
        flush(scanned - 1)
        raise RuntimeError(f"source stream exhausted at {scanned}: obtained {counts}, required {quotas}")
    flush(scanned - 1)
    for name, path in paths.items():
        frame = pd.concat([pd.read_parquet(part) for part in part_paths[name]], ignore_index=True)
        if len(frame) != quotas[name] or frame.doc_id.nunique() != quotas[name]:
            raise RuntimeError(f"{name} stream parts do not cover the quota exactly once")
        atomic_parquet(path, frame)
    atomic_json(
        SOURCE / "stream_manifest.json",
        {
            "status": "fresh-disjoint-source-lanes-frozen",
            "artifact_root": str(ROOT.resolve()),
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "dataset_config": DATASET_CONFIG,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "source_seed": SOURCE_SEED,
            "filters": {
                "tokens": [MIN_TOKENS, MAX_TOKENS],
                "language_score_minimum": MIN_LANGUAGE_SCORE,
                "education_score_minimum": MIN_EDU_SCORE,
            },
            "scanned": scanned,
            "counts": counts,
            "prior_exact_hashes_excluded": len(excluded_exact),
            "prior_normalized_hashes_excluded": len(excluded_normalized),
            "prior_sources": prior_sources,
            "lane_rule": "sha256-proportional deterministic partition among unfilled quotas",
            "resumable_flush_documents": STREAM_FLUSH_DOCUMENTS,
            "elapsed_seconds": time.time() - started,
            "artifacts": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
            "part_hashes": {
                name: {path.name: sha256(path) for path in part_paths[name]}
                for name in quotas
            },
        },
    )


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _score_candidate_frame(candidates: pd.DataFrame, batch_size: int) -> tuple[int, float]:
    """Append immutable base-difficulty parts for currently available identities."""
    if candidates.doc_id.duplicated().any():
        raise ValueError("candidate scoring frame contains duplicate document IDs")
    parts = SOURCE / "difficulty_parts"
    parts.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer_local()
    device = _device()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True
    ).to(device).eval()
    completed_ids: set[str] = set()
    for path in sorted(parts.glob("part-*.parquet")):
        completed_ids.update(pd.read_parquet(path, columns=["doc_id"]).doc_id.astype(str))
    pending = candidates.loc[~candidates.doc_id.astype(str).isin(completed_ids)].reset_index(drop=True)
    started = time.time()
    part_number = len(list(parts.glob("part-*.parquet")))
    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start : start + batch_size]
        encoded = [tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"] for text in batch.text]
        maximum = max(map(len, encoded))
        input_ids = torch.full((len(encoded), maximum), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, device=device)
            attention[row, : len(ids)] = 1
        with torch.inference_mode():
            log_probs = model(input_ids, attention_mask=attention, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            outcomes = input_ids[:, 1:, None]
            observed = log_probs.gather(-1, outcomes).squeeze(-1)
            valid = attention[:, 1:].bool()
            losses = -(observed * valid).sum(-1) / valid.sum(-1)
        result = pd.DataFrame(
            {
                "doc_id": batch.doc_id.to_numpy(),
                "baseline_difficulty": losses.cpu().numpy().astype(float),
                "valid_tokens": valid.sum(-1).cpu().numpy().astype(int),
            }
        )
        atomic_parquet(parts / f"part-{part_number:06d}.parquet", result)
        part_number += 1
        if part_number % 100 == 0:
            print(f"base difficulty {len(completed_ids) + min(start + len(batch), len(pending))}/{len(candidates)}", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()
    return len(pending), time.time() - started


def score_available_candidate_parts(batch_size: int) -> None:
    """Pre-score complete stream parts without freezing the final score manifest."""
    if (SOURCE / "base_difficulty_manifest.json").exists():
        raise FileExistsError("base difficulty is already complete")
    paths = sorted((SOURCE / "stream_parts" / "candidate").glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError("no complete candidate stream parts are available")
    candidates = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    scored, elapsed = _score_candidate_frame(candidates, batch_size)
    print(
        f"incremental base difficulty scored={scored} available={len(candidates)} "
        f"elapsed_seconds={elapsed:.3f}",
        flush=True,
    )


def archive_preliminary_scores() -> None:
    """Preserve target-free development scores that lack full invocation provenance."""
    parts = SOURCE / "difficulty_parts"
    archive = SOURCE / "preliminary_difficulty_parts_unfrozen"
    if not parts.exists() or not list(parts.glob("part-*.parquet")):
        raise FileNotFoundError("no preliminary difficulty parts to archive")
    if archive.exists():
        raise FileExistsError("preliminary score archive already exists")
    part_paths = sorted(parts.glob("part-*.parquet"))
    rows = sum(len(pd.read_parquet(path, columns=["doc_id"])) for path in part_paths)
    atomic_json(
        parts / "preliminary_manifest.json",
        {
            "status": "preserved-target-output-free-preliminary-scores",
            "production_eligible": False,
            "reason": "early incremental invocations did not record a source-code hash per invocation; production candidates will be rescored under one frozen implementation",
            "rows": rows,
            "parts": {path.name: sha256(path) for path in part_paths},
            "archiving_source_sha256": sha256(Path(__file__)),
        },
    )
    os.replace(parts, archive)


def score_candidates(batch_size: int) -> None:
    raw_path = SOURCE / "raw_candidate.parquet"
    if not raw_path.exists():
        raise FileNotFoundError("run stream first")
    completion = SOURCE / "base_difficulty_manifest.json"
    if completion.exists():
        raise FileExistsError("base difficulty is already complete")
    candidates = pd.read_parquet(raw_path)
    _, elapsed = _score_candidate_frame(candidates, batch_size)
    parts = SOURCE / "difficulty_parts"
    part_paths = sorted(parts.glob("part-*.parquet"))
    scores = pd.concat([pd.read_parquet(path) for path in part_paths], ignore_index=True)
    if len(scores) != len(candidates) or scores.doc_id.nunique() != len(candidates):
        raise RuntimeError("difficulty parts do not cover candidates exactly once")
    score_path = SOURCE / "base_difficulty.parquet"
    atomic_parquet(score_path, scores.sort_values("doc_id"))
    atomic_json(
        completion,
        {
            "status": "pinned-base-candidate-difficulty-complete",
            "access": "base checkpoint next-token probabilities only; no dose-target outputs",
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "documents": len(scores),
            "batch_size": batch_size,
            "raw_candidate_sha256": sha256(raw_path),
            "difficulty_sha256": sha256(score_path),
            "scorer_source_sha256": sha256(Path(__file__)),
            "part_hashes": {path.name: sha256(path) for path in part_paths},
            "finalization_invocation_elapsed_seconds": elapsed,
        },
    )


def _hashed_shingles(text: str, width: int = 5) -> set[int]:
    words = normalized_text(text).split()
    values = [" ".join(words)] if len(words) < width else [" ".join(words[index : index + width]) for index in range(len(words) - width + 1)]
    return {int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest()) for value in values}


def remove_candidate_near_duplicates(candidates: pd.DataFrame, references: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate_sets = [_hashed_shingles(text) for text in candidates.text]
    inverted: dict[int, list[int]] = {}
    for index, shingles in enumerate(candidate_sets):
        for shingle in shingles:
            inverted.setdefault(shingle, []).append(index)
    cross_duplicate = np.zeros(len(candidates), dtype=bool)
    for text in references.text:
        reference = _hashed_shingles(text)
        possible: set[int] = set()
        for shingle in reference:
            possible.update(inverted.get(shingle, ()))
        for index in possible:
            candidate = candidate_sets[index]
            if (
                len(candidate & reference) / len(candidate | reference)
                >= NEAR_DUPLICATE_THRESHOLD
            ):
                cross_duplicate[index] = True
    del inverted
    selected_sets: list[set[int]] = []
    selected_inverted: dict[int, list[int]] = {}
    keep = []
    removed_cross = 0
    removed_internal = 0
    for row_index, row in enumerate(candidates.itertuples(index=False)):
        shingles = candidate_sets[row_index]
        if cross_duplicate[row_index]:
            removed_cross += 1
            continue
        internal_candidates: set[int] = set()
        for shingle in shingles:
            internal_candidates.update(selected_inverted.get(shingle, ()))
        if any(len(shingles & selected_sets[index]) / len(shingles | selected_sets[index]) >= NEAR_DUPLICATE_THRESHOLD for index in internal_candidates):
            removed_internal += 1
            continue
        selected_index = len(selected_sets)
        selected_sets.append(shingles)
        for shingle in shingles:
            selected_inverted.setdefault(shingle, []).append(selected_index)
        keep.append(row.doc_id)
    retained = candidates.loc[candidates.doc_id.isin(keep)].reset_index(drop=True)
    return retained, {
        "rule": "word-5-shingle Jaccard < 0.80 using stable 64-bit shingle hashes",
        "candidate_documents_before": len(candidates),
        "removed_cross_lane": removed_cross,
        "removed_candidate_internal": removed_internal,
        "candidate_documents_after": len(retained),
    }


def finalize_source() -> None:
    outputs = {
        "candidate": SOURCE / "candidate_pool.parquet",
        "background": SOURCE / "background_manifest.parquet",
        "donor": SOURCE / "donor_manifest.parquet",
    }
    if any(path.exists() for path in outputs.values()) or (SOURCE / "source_manifest.json").exists():
        raise FileExistsError("final source manifests are already frozen")
    raw = {name: pd.read_parquet(SOURCE / f"raw_{name}.parquet") for name in outputs}
    # Earlier resumable parts may have been written before the full-host audit
    # column was introduced.  Derive it deterministically from the frozen URL.
    for frame in raw.values():
        frame["domain"] = frame.url.astype(str).map(source_domain)
    difficulty = pd.read_parquet(SOURCE / "base_difficulty.parquet")
    candidate = raw["candidate"].merge(difficulty, on="doc_id", validate="one_to_one")
    prior_references, prior_reference_sources = prior_text_references()
    references = pd.concat(
        [
            raw["background"][["text"]],
            raw["donor"][["text"]],
            prior_references,
        ],
        ignore_index=True,
    )
    candidate, duplicate_audit = remove_candidate_near_duplicates(candidate, references)
    duplicate_audit["cross_lane_reference_documents"] = int(
        len(raw["background"]) + len(raw["donor"])
    )
    duplicate_audit["prior_population_reference_documents"] = len(prior_references)
    duplicate_audit["prior_population_reference_sources"] = prior_reference_sources
    if len(candidate) < 4_200:
        raise RuntimeError("near-duplicate audit left fewer than 4,200 candidate documents")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_features=20_000,
        sublinear_tf=True,
    )
    candidate_matrix = vectorizer.fit_transform(candidate.text)
    topic_model = MiniBatchKMeans(
        n_clusters=TOPIC_CLUSTERS,
        random_state=SOURCE_SEED,
        batch_size=512,
        n_init=10,
    ).fit(candidate_matrix)
    candidate["topic"] = topic_model.labels_
    for name in ("background", "donor"):
        raw[name]["topic"] = topic_model.predict(vectorizer.transform(raw[name].text))
    atomic_parquet(outputs["candidate"], candidate)
    atomic_parquet(outputs["background"], raw["background"])
    atomic_parquet(outputs["donor"], raw["donor"])
    atomic_json(
        SOURCE / "source_manifest.json",
        {
            "status": "target-output-free-study-source-frozen",
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "dataset_config": DATASET_CONFIG,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "preparation_source_sha256": sha256(Path(__file__)),
            "topic_model": {
                "type": "TF-IDF bigram plus MiniBatchKMeans",
                "clusters": TOPIC_CLUSTERS,
                "seed": SOURCE_SEED,
                "uses_target_outputs": False,
            },
            "source_matching_stratum": "coarse URL terminal-suffix bucket",
            "full_host_domain_retained_for_blind_confound_audit": True,
            "near_duplicate_audit": duplicate_audit,
            "artifacts": {name: {"path": str(path), "rows": len(pd.read_parquet(path)), "sha256": sha256(path)} for name, path in outputs.items()},
            "stream_manifest_sha256": sha256(SOURCE / "stream_manifest.json"),
            "difficulty_manifest_sha256": sha256(SOURCE / "base_difficulty_manifest.json"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "stream",
            "score-available",
            "archive-preliminary-scores",
            "score",
            "finalize",
        ),
    )
    parser.add_argument("--candidate-documents", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--background-documents", type=int, default=DEFAULT_BACKGROUND)
    parser.add_argument("--donor-documents", type=int, default=DEFAULT_DONORS)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.command == "stream":
        stream_source(args.candidate_documents, args.background_documents, args.donor_documents, args.max_records)
    elif args.command == "score-available":
        score_available_candidate_parts(args.batch_size)
    elif args.command == "archive-preliminary-scores":
        archive_preliminary_scores()
    elif args.command == "score":
        score_candidates(args.batch_size)
    else:
        finalize_source()


if __name__ == "__main__":
    main()
