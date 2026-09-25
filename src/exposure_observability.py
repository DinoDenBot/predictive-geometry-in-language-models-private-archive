"""Pure design, geometry-summary, and inference tools for the dose study.

This module deliberately performs no model queries and writes no artifacts.
The command-line runner owns data access, training, checkpointing, and the
immutable study state machine.  Keeping these routines pure makes the exact
assignment mechanism and frozen analysis independently testable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DOSES = np.asarray((0, 1, 2, 4, 8, 16), dtype=np.int64)
DOSE_D = np.log2(DOSES + 1.0)
ROLE_BLOCKS = {"development": 100, "validation": 300, "confirmation": 300}
CHECKPOINT_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalized_sha256(text: str) -> str:
    return text_sha256(normalized_text(text))


def _shingles(text: str, width: int = 5) -> set[str]:
    words = normalized_text(text).split()
    if len(words) < width:
        return {" ".join(words)}
    return {" ".join(words[index : index + width]) for index in range(len(words) - width + 1)}


def near_duplicate(left: str, right: str, threshold: float = 0.80) -> bool:
    """Deterministic word-5-shingle Jaccard near-duplicate rule."""
    a, b = _shingles(left), _shingles(right)
    union = len(a | b)
    return bool(union and len(a & b) / union >= threshold)


def _near_duplicate_pair(texts: Sequence[str], threshold: float = 0.80) -> tuple[int, int] | None:
    """Find one global near-duplicate pair without materializing all pairs."""
    shingle_sets = [_shingles(text) for text in texts]
    inverted: dict[str, list[int]] = {}
    for index, shingles in enumerate(shingle_sets):
        for shingle in shingles:
            inverted.setdefault(shingle, []).append(index)
    for left, shingles in enumerate(shingle_sets):
        candidates: set[int] = set()
        for shingle in shingles:
            candidates.update(index for index in inverted[shingle] if index > left)
        for right in candidates:
            a, b = shingles, shingle_sets[right]
            if len(a & b) / len(a | b) >= threshold:
                return left, right
    return None


def validate_candidate_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize an immutable 700 x 6 matched-block manifest."""
    required = {
        "block_id",
        "doc_id",
        "doc_slot",
        "role",
        "source",
        "topic",
        "tokens",
        "baseline_difficulty",
        "text",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"candidate manifest is missing columns: {sorted(missing)}")
    out = frame.copy()
    if len(out) != 4200 or out.block_id.nunique() != 700:
        raise ValueError("candidate manifest must contain 700 six-document blocks")
    if out.doc_id.duplicated().any():
        raise ValueError("doc_id must be globally unique")
    if out.text.isna().any() or (out.text.astype(str).str.len() == 0).any():
        raise ValueError("candidate text must be nonempty")
    out["text"] = out.text.astype(str)
    out["text_hash"] = out.text.map(text_sha256)
    out["normalized_hash"] = out.text.map(normalized_sha256)
    if out.text_hash.duplicated().any() or out.normalized_hash.duplicated().any():
        raise ValueError("exact or normalized duplicate candidate documents detected")
    duplicate_pair = _near_duplicate_pair(out.text.tolist())
    if duplicate_pair is not None:
        left, right = duplicate_pair
        raise ValueError(
            "global inter-document near duplicate detected between "
            f"{out.iloc[left].doc_id} and {out.iloc[right].doc_id}"
        )
    expected_roles = pd.Series(ROLE_BLOCKS, name="blocks").sort_index()
    actual_roles = (
        out[["block_id", "role"]].drop_duplicates().groupby("role").size().sort_index().rename("blocks")
    )
    if not actual_roles.equals(expected_roles):
        raise ValueError(f"role block counts differ: {actual_roles.to_dict()}")
    for block_id, block in out.groupby("block_id", sort=False):
        if len(block) != 6 or set(block.doc_slot.astype(int)) != set(range(6)):
            raise ValueError(f"block {block_id} must contain slots 0..5 exactly once")
        if block.role.nunique() != 1 or block.source.nunique() != 1 or block.topic.nunique() != 1:
            raise ValueError(f"block {block_id} is not role/source/topic matched")
        if not np.all(np.isfinite(block.tokens)) or not np.all(np.isfinite(block.baseline_difficulty)):
            raise ValueError(f"block {block_id} has non-finite matching variables")
    return out.sort_values(["block_id", "doc_slot"]).reset_index(drop=True)


def construct_matched_blocks(
    candidates: pd.DataFrame,
    *,
    seed: int,
    token_tolerance: int = 8,
    difficulty_tolerance: float = 0.20,
) -> pd.DataFrame:
    """Greedily construct target-output-free blocks from a prepared pool.

    The input difficulty must have been computed from the pinned base model,
    before any dose target is trained.  Exact matching is used for source and
    topic; token range and baseline-difficulty range use declared calipers.
    """
    required = {"doc_id", "source", "topic", "tokens", "baseline_difficulty", "text"}
    if required - set(candidates):
        raise ValueError(f"candidate pool missing {sorted(required - set(candidates))}")
    pool = candidates.copy()
    pool["text_hash"] = pool.text.astype(str).map(text_sha256)
    pool["normalized_hash"] = pool.text.astype(str).map(normalized_sha256)
    pool = pool.drop_duplicates("text_hash").drop_duplicates("normalized_hash")
    rng = np.random.default_rng(seed)
    blocks: list[pd.DataFrame] = []
    selected_shingles: list[set[str]] = []
    selected_inverted: dict[str, list[int]] = {}
    for _, group in pool.groupby(["source", "topic"], sort=True):
        order = rng.permutation(len(group))
        remaining = group.iloc[order].copy()
        while len(remaining) >= 6 and len(blocks) < 700:
            anchor = remaining.iloc[0]
            eligible = remaining.loc[
                (remaining.tokens.sub(anchor.tokens).abs() <= token_tolerance)
                & (
                    remaining.baseline_difficulty.sub(anchor.baseline_difficulty).abs()
                    <= difficulty_tolerance
                )
            ]
            chosen: list[int] = []
            for index, row in eligible.iterrows():
                shingles = _shingles(row.text)
                prior_candidates: set[int] = set()
                for shingle in shingles:
                    prior_candidates.update(selected_inverted.get(shingle, ()))
                globally_fresh = all(
                    len(shingles & selected_shingles[other])
                    / len(shingles | selected_shingles[other])
                    < 0.80
                    for other in prior_candidates
                )
                prospective = chosen + [index]
                token_range = float(
                    remaining.loc[prospective, "tokens"].max()
                    - remaining.loc[prospective, "tokens"].min()
                )
                difficulty_range = float(
                    remaining.loc[prospective, "baseline_difficulty"].max()
                    - remaining.loc[prospective, "baseline_difficulty"].min()
                )
                if (
                    globally_fresh
                    and token_range <= token_tolerance
                    and difficulty_range <= difficulty_tolerance
                    and all(
                    not near_duplicate(row.text, remaining.loc[other, "text"])
                    for other in chosen
                    )
                ):
                    chosen.append(index)
                if len(chosen) == 6:
                    break
            if len(chosen) < 6:
                remaining = remaining.drop(index=remaining.index[0])
                continue
            blocks.append(remaining.loc[chosen].copy())
            for text in remaining.loc[chosen, "text"]:
                shingles = _shingles(text)
                selected_index = len(selected_shingles)
                selected_shingles.append(shingles)
                for shingle in shingles:
                    selected_inverted.setdefault(shingle, []).append(selected_index)
            remaining = remaining.drop(index=chosen)
        if len(blocks) == 700:
            break
    if len(blocks) != 700:
        raise ValueError(f"only {len(blocks)} valid matched blocks could be constructed")
    role_vector = np.repeat(list(ROLE_BLOCKS), list(ROLE_BLOCKS.values()))
    rng.shuffle(role_vector)
    rows = []
    for block_id, (block, role) in enumerate(zip(blocks, role_vector, strict=True)):
        block = block.iloc[rng.permutation(6)].copy()
        block["block_id"] = block_id
        block["doc_slot"] = np.arange(6)
        block["role"] = role
        rows.append(block)
    return validate_candidate_manifest(pd.concat(rows, ignore_index=True))


def latin_cube(
    n_blocks: int,
    n_targets: int,
    rng: np.random.Generator,
    block_positions: Sequence[int] | None = None,
) -> np.ndarray:
    """Draw from the declared balanced cross-target Latin mechanism.

    Each block-target slice is a permutation of all six doses.  The first six
    target positions form a Latin cycle; for eight targets, the two repeated
    shifts rotate across blocks and target identities are freshly randomized.
    """
    if n_blocks < 1 or n_targets < 1:
        raise ValueError("positive block and target counts are required")
    positions = (
        np.arange(n_blocks, dtype=int)
        if block_positions is None
        else np.asarray(block_positions, dtype=int)
    )
    if positions.shape != (n_blocks,) or len(np.unique(positions)) != n_blocks:
        raise ValueError("Latin block positions must be unique and aligned")
    cube = np.empty((n_blocks, n_targets, 6), dtype=np.int64)
    for block in range(n_blocks):
        base = rng.permutation(DOSES)
        shifts = list(range(6))
        while len(shifts) < n_targets:
            shifts.append((int(positions[block]) + len(shifts) - 6) % 6)
        shifts = np.asarray(shifts[:n_targets], dtype=int)
        shifts = shifts[rng.permutation(n_targets)]
        for target in range(n_targets):
            cube[block, target] = np.roll(base, int(shifts[target]))
    return cube


def _latin_cubes(
    draws: int,
    n_blocks: int,
    n_targets: int,
    rng: np.random.Generator,
    block_positions: Sequence[int] | None = None,
) -> np.ndarray:
    """Vectorized independent draws from :func:`latin_cube`'s mechanism."""
    base_order = np.argsort(rng.random((draws, n_blocks, 6)), axis=2)
    bases = DOSES[base_order]
    positions = (
        np.arange(n_blocks, dtype=int)
        if block_positions is None
        else np.asarray(block_positions, dtype=int)
    )
    if positions.shape != (n_blocks,) or len(np.unique(positions)) != n_blocks:
        raise ValueError("Latin block positions must be unique and aligned")
    template = np.empty((n_blocks, n_targets), dtype=int)
    for block in range(n_blocks):
        shifts = list(range(6))
        while len(shifts) < n_targets:
            shifts.append((int(positions[block]) + len(shifts) - 6) % 6)
        template[block] = shifts[:n_targets]
    target_order = np.argsort(rng.random((draws, n_blocks, n_targets)), axis=2)
    shifts = np.take_along_axis(
        np.broadcast_to(template, (draws, n_blocks, n_targets)), target_order, axis=2
    )
    slots = np.arange(6)[None, None, None, :]
    source_slots = (slots - shifts[:, :, :, None]) % 6
    expanded = np.broadcast_to(bases[:, :, None, :], (draws, n_blocks, n_targets, 6))
    return np.take_along_axis(expanded, source_slots, axis=3)


def make_latin_assignments(
    manifest: pd.DataFrame, target_ids: Sequence[str], *, seed: int
) -> pd.DataFrame:
    canonical = validate_candidate_manifest(manifest)
    target_ids = tuple(target_ids)
    if len(target_ids) != 8 or len(set(target_ids)) != 8:
        raise ValueError("the study requires eight unique target IDs")
    block_ids = np.asarray(sorted(canonical.block_id.unique()))
    cube = latin_cube(len(block_ids), len(target_ids), np.random.default_rng(seed))
    lookup = canonical.set_index(["block_id", "doc_slot"])[["doc_id", "role"]]
    rows = []
    for bpos, block_id in enumerate(block_ids):
        for tpos, target_id in enumerate(target_ids):
            for slot in range(6):
                doc = lookup.loc[(block_id, slot)]
                dose = int(cube[bpos, tpos, slot])
                rows.append(
                    {
                        "block_id": block_id,
                        "latin_block_position": bpos,
                        "doc_slot": slot,
                        "doc_id": doc.doc_id,
                        "role": doc.role,
                        "target_id": target_id,
                        "K": dose,
                        "d": float(math.log2(dose + 1)),
                    }
                )
    out = pd.DataFrame(rows)
    validate_latin_assignments(out, target_ids)
    return out


def validate_latin_assignments(frame: pd.DataFrame, target_ids: Sequence[str]) -> None:
    required = {
        "block_id",
        "latin_block_position",
        "doc_slot",
        "doc_id",
        "role",
        "target_id",
        "K",
        "d",
    }
    if required - set(frame):
        raise ValueError("Latin assignment table is incomplete")
    if set(frame.target_id) != set(target_ids):
        raise ValueError("target IDs do not match")
    grouped = frame.groupby(["block_id", "target_id"], sort=False)
    for key, group in grouped:
        if sorted(group.K.astype(int)) != DOSES.tolist() or int(group.K.sum()) != 31:
            raise ValueError(f"invalid exposure allocation for {key}")
        if group.doc_id.duplicated().any():
            raise ValueError(f"duplicate document in {key}")
    expected_d = np.log2(frame.K.to_numpy() + 1.0)
    if not np.allclose(frame.d, expected_d, rtol=0, atol=1e-12):
        raise ValueError("d must equal log2(K + 1)")
    latin_positions = frame.groupby("block_id").latin_block_position.nunique()
    if not latin_positions.eq(1).all():
        raise ValueError("each block must have one global Latin position")
    position_values = frame.groupby("block_id").latin_block_position.first()
    if position_values.duplicated().any():
        raise ValueError("Latin block positions must be globally unique")
    cross_target = frame.groupby(["block_id", "doc_slot"]).K.apply(
        lambda values: set(values.astype(int))
    )
    if not cross_target.map(lambda values: values == set(DOSES.tolist())).all():
        raise ValueError("every block document must rotate through all six doses")


def make_presentation_ledger(
    assignments: pd.DataFrame,
    target_id: str,
    *,
    seed: int,
    batch_size: int,
    background_documents: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Spread exact-document presentations over progress with batch uniqueness."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected = assignments.loc[assignments.target_id == target_id].copy()
    if selected.empty:
        raise ValueError(f"unknown target: {target_id}")
    rng = np.random.default_rng(seed)
    events = []
    for row in selected.itertuples(index=False):
        for occurrence in range(int(row.K)):
            ideal = (occurrence + 0.5) / int(row.K)
            events.append(
                {
                    "doc_id": row.doc_id,
                    "block_id": row.block_id,
                    "doc_slot": int(row.doc_slot),
                    "role": row.role,
                    "K": int(row.K),
                    "d": float(row.d),
                    "occurrence": occurrence,
                    "ideal_progress": float(ideal),
                    "tie": float(rng.random()),
                }
            )
    if background_documents is not None:
        if {"doc_id"} - set(background_documents) or background_documents.doc_id.duplicated().any():
            raise ValueError("background documents require unique doc_id values")
        if background_documents.empty:
            raise ValueError("background document stream must be nonempty")
        if set(background_documents.doc_id.astype(str)) & set(selected.doc_id.astype(str)):
            raise ValueError("background and candidate document IDs overlap")
        background_order = rng.permutation(len(background_documents))
        for rank, position in enumerate(background_order):
            row = background_documents.iloc[int(position)]
            events.append(
                {
                    "doc_id": row.doc_id,
                    "block_id": -1,
                    "doc_slot": -1,
                    "role": "background",
                    "K": -1,
                    "d": np.nan,
                    "occurrence": 0,
                    "ideal_progress": (rank + 0.5) / len(background_documents),
                    "tie": float(rng.random()),
                }
            )
    pending = sorted(events, key=lambda event: (event["ideal_progress"], event["tie"]))
    scheduled = []
    step = 0
    while pending:
        used: set[str] = set()
        batch_indices = []
        for index, event in enumerate(pending):
            if str(event["doc_id"]) not in used:
                batch_indices.append(index)
                used.add(str(event["doc_id"]))
            if len(batch_indices) == batch_size:
                break
        if not batch_indices:
            raise RuntimeError("unable to form a document-unique batch")
        batch = [pending[index] for index in batch_indices]
        for index in reversed(batch_indices):
            pending.pop(index)
        for position, event in enumerate(batch):
            event["optimizer_step"] = step
            event["batch_position"] = position
            scheduled.append(event)
        step += 1
    ledger = pd.DataFrame(scheduled).drop(columns="tie")
    denominator = max(step - 1, 1)
    ledger["normalized_progress"] = ledger.optimizer_step / denominator
    ledger["presentation_id"] = np.arange(len(ledger))
    ledger["target_id"] = target_id
    validate_presentation_ledger(ledger, selected, batch_size)
    return ledger.sort_values(["optimizer_step", "batch_position"]).reset_index(drop=True)


def validate_presentation_ledger(
    ledger: pd.DataFrame, assignments: pd.DataFrame, batch_size: int
) -> None:
    expected = assignments.set_index("doc_id").K.astype(int).to_dict()
    actual = ledger.loc[ledger.K >= 0].groupby("doc_id").size().to_dict()
    actual.update({doc_id: 0 for doc_id, dose in expected.items() if dose == 0})
    if actual != expected:
        raise ValueError("ledger presentations do not match assigned doses")
    if ledger.groupby("optimizer_step").doc_id.nunique().ne(
        ledger.groupby("optimizer_step").size()
    ).any():
        raise ValueError("a batch repeats an exact document")
    if ledger.groupby("optimizer_step").size().max() > batch_size:
        raise ValueError("ledger exceeds batch size")


def presentation_balance(ledger: pd.DataFrame, learning_rates: Sequence[float]) -> pd.DataFrame:
    """Return dose-wise timing and presentation-weighted LR diagnostics."""
    rates = np.asarray(learning_rates, dtype=np.float64)
    if len(rates) <= int(ledger.optimizer_step.max()) or np.any(~np.isfinite(rates)):
        raise ValueError("learning-rate schedule does not cover the ledger")
    work = ledger.copy()
    work["learning_rate"] = rates[work.optimizer_step.to_numpy(dtype=int)]
    return (
        work.groupby("K")
        .agg(
            presentations=("doc_id", "size"),
            mean_progress=("normalized_progress", "mean"),
            weighted_learning_rate=("learning_rate", "mean"),
        )
        .reset_index()
    )


def frozen_document_summaries(alignment: Sequence[float], length: Sequence[float], directed: Sequence[float]) -> dict[str, float]:
    arrays = [np.asarray(values, dtype=np.float64) for values in (alignment, length, directed)]
    if not arrays[0].size or any(values.shape != arrays[0].shape for values in arrays):
        raise ValueError("A, L, and R transition arrays must be nonempty and aligned")
    if any(np.any(~np.isfinite(values)) for values in arrays):
        raise ValueError("transition arrays must be finite")
    return {
        name: float(np.quantile(values, 0.90, method="linear"))
        for name, values in zip(("S_A", "S_L", "S_R"), arrays, strict=True)
    }


def add_checkpoint_changes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for name in ("S_A", "S_L", "S_R"):
        base, final = f"base_{name}", f"final_{name}"
        if base not in out or final not in out:
            raise ValueError(f"missing {base} or {final}")
        out[f"delta_{name}"] = out[final] - out[base]
    return out


def block_fixed_effect_slope(frame: pd.DataFrame, outcome: str) -> float:
    if {"block_id", "d", outcome} - set(frame):
        raise ValueError("slope frame is missing block_id, d, or outcome")
    # Arrow-backed parquet columns may expose read-only NumPy views.  Centering
    # is an analysis-local transformation, so take explicit writable copies.
    x = frame.d.to_numpy(dtype=np.float64).copy()
    y = frame[outcome].to_numpy(dtype=np.float64).copy()
    groups = frame.block_id
    x -= frame.groupby(groups).d.transform("mean").to_numpy()
    y -= frame.groupby(groups)[outcome].transform("mean").to_numpy()
    denominator = float(x @ x)
    if denominator <= 0 or np.any(~np.isfinite(y)):
        raise ValueError("block-FE slope is unidentified or non-finite")
    return float(x @ y / denominator)


def target_slopes(frame: pd.DataFrame, outcome: str) -> dict[str, float]:
    return {
        str(target): block_fixed_effect_slope(group, outcome)
        for target, group in frame.groupby("target_id", sort=True)
    }


def phase_statistic(frame: pd.DataFrame, outcome: str) -> float:
    slopes = target_slopes(frame, outcome)
    if not slopes:
        raise ValueError("phase has no targets")
    return float(np.mean(list(slopes.values())))


def _outcome_cube(frame: pd.DataFrame, outcome: str) -> tuple[np.ndarray, list, list]:
    blocks = sorted(frame.block_id.unique())
    targets = sorted(frame.target_id.unique())
    index = pd.MultiIndex.from_product([blocks, targets, range(6)], names=["block_id", "target_id", "doc_slot"])
    values = frame.set_index(["block_id", "target_id", "doc_slot"])[outcome].reindex(index)
    if values.isna().any():
        raise ValueError("phase must be a complete block x target x slot panel")
    return values.to_numpy(dtype=np.float64).reshape(len(blocks), len(targets), 6), blocks, targets


def _cube_phase_stat(dose_cube: np.ndarray, outcomes: np.ndarray) -> float:
    d = np.log2(dose_cube + 1.0)
    x = d - d.mean(axis=2, keepdims=True)
    y = outcomes - outcomes.mean(axis=2, keepdims=True)
    numerator = np.sum(x * y, axis=(0, 2))
    denominator = np.sum(x * x, axis=(0, 2))
    return float(np.mean(numerator / denominator))


def _cube_phase_stats(dose_cubes: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Vectorized phase statistics for draw x block x target x slot cubes."""
    d = np.log2(dose_cubes + 1.0)
    x = d - d.mean(axis=3, keepdims=True)
    y = outcomes - outcomes.mean(axis=2, keepdims=True)
    numerator = np.sum(x * y[None, :, :, :], axis=(1, 3))
    denominator = np.sum(x * x, axis=(1, 3))
    return np.mean(numerator / denominator, axis=1)


@dataclass(frozen=True)
class RandomizationResult:
    estimate: float
    p_one_sided: float
    draws: int
    greater_or_equal: int


def randomization_test(
    frame: pd.DataFrame,
    outcome: str,
    *,
    draws: int = 99_999,
    seed: int,
    all_target_ids: Sequence[str] | None = None,
) -> RandomizationResult:
    outcomes, blocks, targets = _outcome_cube(frame, outcome)
    if "latin_block_position" in frame:
        position_map = frame.groupby("block_id").latin_block_position.first()
        block_positions = [int(position_map.loc[block]) for block in blocks]
    else:
        block_positions = [int(block) for block in blocks]
    observed = phase_statistic(frame, outcome)
    design_targets = list(all_target_ids) if all_target_ids is not None else targets
    if not set(targets).issubset(design_targets):
        raise ValueError("phase targets are absent from the declared full target design")
    target_positions = [design_targets.index(target) for target in targets]
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    chunk_size = min(128, draws)
    while completed < draws:
        count = min(chunk_size, draws - completed)
        full_cubes = _latin_cubes(
            count,
            len(outcomes),
            len(design_targets),
            rng,
            block_positions,
        )
        statistics = _cube_phase_stats(full_cubes[:, :, target_positions, :], outcomes)
        exceed += int(np.count_nonzero(statistics >= observed - 1e-15))
        completed += count
    return RandomizationResult(observed, (exceed + 1) / (draws + 1), draws, exceed)


def design_based_interval(
    frame: pd.DataFrame,
    outcome: str,
    *,
    draws: int = 99_999,
    seed: int,
    confidence: float = 0.95,
    grid_points: int = 1601,
    all_target_ids: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Invert the two-sided exact-mechanism test under a constant linear effect."""
    outcomes, blocks, targets = _outcome_cube(frame, outcome)
    if "latin_block_position" in frame:
        position_map = frame.groupby("block_id").latin_block_position.first()
        block_positions = [int(position_map.loc[block]) for block in blocks]
    else:
        block_positions = [int(block) for block in blocks]
    observed_doses, _, _ = _outcome_cube(frame, "K")
    estimate = phase_statistic(frame, outcome)
    design_targets = list(all_target_ids) if all_target_ids is not None else targets
    if not set(targets).issubset(design_targets):
        raise ValueError("phase targets are absent from the declared full target design")
    target_positions = [design_targets.index(target) for target in targets]
    rng = np.random.default_rng(seed)
    u = np.empty(draws)
    v = np.empty(draws)
    observed_d = np.log2(observed_doses + 1.0)
    completed = 0
    chunk_size = min(128, draws)
    while completed < draws:
        count = min(chunk_size, draws - completed)
        cubes = _latin_cubes(
            count,
            len(outcomes),
            len(design_targets),
            rng,
            block_positions,
        )[
            :, :, target_positions, :
        ]
        u[completed : completed + count] = _cube_phase_stats(cubes, outcomes)
        v[completed : completed + count] = _cube_phase_stats(cubes, observed_d)
        completed += count
    # ``grid_points`` remains in the public signature for replay compatibility
    # with early development calls.  Endpoints are now obtained exactly from
    # the crossing points of the same Monte Carlo inversion, rather than from
    # a finite grid.
    del grid_points
    return _constant_effect_interval_from_statistics(
        estimate, u, v, confidence=confidence
    )


def _constant_effect_interval_from_statistics(
    estimate: float,
    permuted_outcome_slopes: np.ndarray,
    permuted_observed_dose_slopes: np.ndarray,
    *,
    confidence: float,
) -> tuple[float, float]:
    """Invert the plus-one two-sided randomization test exactly in effect size.

    For draw ``j``, acceptance is
    ``|u_j - beta v_j| >= |estimate - beta|``.  Because ``v_j`` is the
    correlation-like slope between two balanced dose allocations, ``|v_j|``
    is at most one and each draw accepts one interval containing ``estimate``.
    The test interval is therefore obtained from order statistics of those
    per-draw endpoints; no effect-size grid or approximation is needed.
    """
    u = np.asarray(permuted_outcome_slopes, dtype=np.float64)
    v = np.asarray(permuted_observed_dose_slopes, dtype=np.float64)
    if u.ndim != 1 or v.shape != u.shape or not len(u):
        raise ValueError("randomization statistics must be aligned nonempty vectors")
    if not 0 < confidence < 1 or not np.isfinite(estimate):
        raise ValueError("confidence and estimate must be finite and valid")
    if np.any(~np.isfinite(u)) or np.any(~np.isfinite(v)):
        raise ValueError("randomization statistics must be finite")
    if np.max(np.abs(v)) > 1.0 + 1e-10:
        raise ValueError("balanced-dose randomization slope exceeded its unit bound")
    v = np.clip(v, -1.0, 1.0)
    lower = np.divide(
        estimate - u,
        1.0 - v,
        out=np.full_like(u, -np.inf),
        where=np.abs(1.0 - v) > 1e-14,
    )
    upper = np.divide(
        estimate + u,
        1.0 + v,
        out=np.full_like(u, np.inf),
        where=np.abs(1.0 + v) > 1e-14,
    )
    left = np.minimum(lower, upper)
    right = np.maximum(lower, upper)
    at_positive_boundary = np.abs(1.0 - v) <= 1e-14
    for index in np.flatnonzero(at_positive_boundary):
        if abs(u[index] - estimate) <= 1e-12:
            left[index], right[index] = -np.inf, np.inf
        else:
            boundary = (u[index] + estimate) / 2.0
            if u[index] > estimate:
                left[index], right[index] = -np.inf, boundary
            else:
                left[index], right[index] = boundary, np.inf
    at_negative_boundary = np.abs(-1.0 - v) <= 1e-14
    for index in np.flatnonzero(at_negative_boundary):
        if abs(u[index] + estimate) <= 1e-12:
            left[index], right[index] = -np.inf, np.inf
        else:
            boundary = (estimate - u[index]) / 2.0
            if u[index] + estimate > 0:
                left[index], right[index] = boundary, np.inf
            else:
                left[index], right[index] = -np.inf, boundary
    if np.any(left > estimate + 1e-9) or np.any(right < estimate - 1e-9):
        raise RuntimeError("a draw-level inversion interval did not contain the estimate")
    alpha = 1.0 - confidence
    eligible_counts = np.flatnonzero(
        (np.arange(len(u) + 1, dtype=np.float64) + 1.0) / (len(u) + 1)
        > alpha
    )
    if not len(eligible_counts):
        raise RuntimeError("no Monte Carlo count can satisfy the requested confidence")
    required = int(eligible_counts[0])
    if required == 0:
        return -float("inf"), float("inf")
    lower_endpoint = float(np.partition(left, required - 1)[required - 1])
    upper_endpoint = float(np.partition(right, len(right) - required)[len(right) - required])
    if lower_endpoint > upper_endpoint:
        raise RuntimeError("constant-effect randomization interval is empty")
    return lower_endpoint, upper_endpoint


def standardize_outcome(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    out = frame.copy()
    scale = float(out[outcome].std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("standardization scale must be positive")
    out[outcome] = (out[outcome] - float(out[outcome].mean())) / scale
    return out


def validation_gate(
    slopes: dict[str, float], pvalue: float, interval: tuple[float, float]
) -> bool:
    return bool(slopes and all(value > 0 for value in slopes.values()) and pvalue < 0.025 and interval[0] > 0)


def document_target_fixed_effect_sensitivity(frame: pd.DataFrame, outcome: str) -> float:
    """Fit Y_ir = alpha_i + lambda_r + beta*d_ir by two-way demeaning."""
    required = {"doc_id", "target_id", "d", outcome}
    if required - set(frame):
        raise ValueError("sensitivity panel is incomplete")
    x = frame.d.astype(float)
    y = frame[outcome].astype(float)
    xr = x - x.groupby(frame.doc_id).transform("mean") - x.groupby(frame.target_id).transform("mean") + x.mean()
    yr = y - y.groupby(frame.doc_id).transform("mean") - y.groupby(frame.target_id).transform("mean") + y.mean()
    denominator = float(xr @ xr)
    if denominator <= 0:
        raise ValueError("document-target FE sensitivity is unidentified")
    return float(xr @ yr / denominator)


def saturated_dose_summary(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Descriptive block-adjusted means with block-clustered intervals."""
    work = frame.copy()
    work["adjusted"] = work[outcome] - work.groupby(["target_id", "block_id"])[outcome].transform("mean")
    work["adjusted"] += work.groupby("target_id")[outcome].transform("mean")
    rows = []
    for dose in DOSES:
        selected = work.loc[work.K == dose]
        values = selected.adjusted.to_numpy(dtype=float)
        block_means = selected.groupby("block_id").adjusted.mean().to_numpy(dtype=float)
        mean = float(np.mean(values))
        se = float(np.std(block_means, ddof=1) / np.sqrt(len(block_means)))
        rows.append(
            {
                "K": int(dose),
                "d": float(np.log2(dose + 1)),
                "mean": mean,
                "lower": mean - 1.96 * se,
                "upper": mean + 1.96 * se,
                "standard_error": se,
                "interval_method": "block-clustered normal descriptive",
            }
        )
    return pd.DataFrame(rows)


def dose_progression(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Report causal shift, pooled-SD observability, and K-vs-zero AUC."""
    zero = frame.loc[frame.K == 0, outcome].to_numpy(dtype=float)
    if len(zero) < 2:
        raise ValueError("dose progression requires at least two zero-dose records")
    rows = []
    for dose in DOSES[1:]:
        treated = frame.loc[frame.K == dose, outcome].to_numpy(dtype=float)
        pooled = math.sqrt((np.var(treated, ddof=1) + np.var(zero, ddof=1)) / 2)
        shift = float(np.mean(treated) - np.mean(zero))
        labels = np.r_[np.ones(len(treated)), np.zeros(len(zero))]
        scores = np.r_[treated, zero]
        selected = frame.loc[frame.K.isin((0, dose))]
        auc_interval = grouped_auc_interval(
            (selected.K == dose).astype(int).to_numpy(),
            selected[outcome].to_numpy(dtype=float),
            selected.block_id.to_numpy(),
            draws=1_999,
            seed=int.from_bytes(
                hashlib.sha256(f"{outcome}:{dose}:auc-v1".encode()).digest()[:8],
                "big",
            ),
        )
        rows.append(
            {
                "K": int(dose),
                "Delta_k": shift,
                "delta_k": shift / pooled if pooled > 0 else np.nan,
                "auc_vs_zero": float(roc_auc_score(labels, scores)),
                "auc_lower_95": auc_interval[0],
                "auc_upper_95": auc_interval[1],
                "auc_interval_method": "1,999-draw block bootstrap",
            }
        )
    return pd.DataFrame(rows)


def grouped_auc_interval(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[object],
    *,
    draws: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile AUC interval resampling complete dependency groups."""
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=np.float64)
    clusters = np.asarray(groups)
    if y.ndim != 1 or values.shape != y.shape or clusters.shape != y.shape:
        raise ValueError("AUC labels, scores, and groups must be aligned vectors")
    if draws < 1 or not 0 < confidence < 1 or len(np.unique(y)) != 2:
        raise ValueError("grouped AUC interval configuration is invalid")
    unique = np.unique(clusters)
    indices = [np.flatnonzero(clusters == group) for group in unique]
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(unique), len(unique))
        rows = np.concatenate([indices[index] for index in sampled])
        estimates[draw] = roc_auc_score(y[rows], values[rows])
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def context_specificity_analysis(
    frame: pd.DataFrame,
    outcome: str = "delta_S_R",
    *,
    draws: int = 99_999,
    seed: int,
    all_target_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Test own-context slope against donor and disrupted-context slopes.

    The input is a complete document-target-context panel.  Each contrast is
    reduced to a document-level paired difference before applying the exact
    Latin randomization test; Holm adjustment is then applied across the two
    prespecified contrasts.
    """
    required = {
        "block_id",
        "doc_id",
        "doc_slot",
        "target_id",
        "K",
        "d",
        "context_type",
        outcome,
    }
    if required - set(frame):
        raise ValueError("context-specificity panel is incomplete")
    keys = ["block_id", "doc_id", "doc_slot", "target_id", "K", "d"]
    wide = frame.pivot(index=keys, columns="context_type", values=outcome).reset_index()
    if {"own", "donor", "disrupted"} - set(wide):
        raise ValueError("own, donor, and disrupted outcomes are all required")
    raw: dict[str, float] = {}
    interim: dict[str, dict[str, object]] = {}
    for offset, comparison in enumerate(("donor", "disrupted")):
        name = f"own_gt_{comparison}"
        work = wide[keys].copy()
        work["contrast"] = wide["own"] - wide[comparison]
        test = randomization_test(
            work,
            "contrast",
            draws=draws,
            seed=seed + offset,
            all_target_ids=all_target_ids,
        )
        slopes = target_slopes(work, "contrast")
        raw[name] = test.p_one_sided
        interim[name] = {
            "estimate": test.estimate,
            "target_slope_differences": slopes,
            "p_one_sided": test.p_one_sided,
        }
    adjusted = holm_adjust(raw)
    for name in interim:
        interim[name]["p_holm"] = adjusted[name]
        interim[name]["passed"] = bool(
            interim[name]["estimate"] > 0 and adjusted[name] < 0.05
        )
    return interim


def donor_support_descriptives(
    frame: pd.DataFrame, outcome: str, support_threshold: float
) -> pd.DataFrame:
    """Separate donor results by frozen base realized-token support."""
    required = {"context_type", "base_realized_logp", "K", outcome}
    if required - set(frame):
        raise ValueError("donor descriptive frame is incomplete")
    donor = frame.loc[frame.context_type == "donor"].copy()
    donor["support"] = np.where(
        donor.base_realized_logp >= support_threshold, "high", "poor"
    )
    return (
        donor.groupby(["support", "K"])
        .agg(
            documents=(outcome, "size"),
            mean_outcome=(outcome, "mean"),
            mean_base_logp=("base_realized_logp", "mean"),
            mean_entropy=("base_entropy", "mean"),
        )
        .reset_index()
    )


def deterministic_disruption(token_ids: Sequence[int], *, document_id: str) -> tuple[list[int], float]:
    """Deterministically swap adjacent local-context tokens, preserving length."""
    values = list(map(int, token_ids))
    if len(values) < 4:
        return values, 0.0
    offset = int(hashlib.sha256(document_id.encode()).hexdigest(), 16) % 2
    changed = values.copy()
    for index in range(offset, len(values) - 1, 2):
        changed[index], changed[index + 1] = changed[index + 1], changed[index]
    severity = float(np.mean(np.asarray(values) != np.asarray(changed)))
    return changed, severity


def donor_coverage_preflight(
    candidates: pd.DataFrame,
    donors: pd.DataFrame,
    *,
    increments: int = 5_000,
    maximum: int = 50_000,
    required_matches: int = 32,
    minimum_token_coverage: float = 0.50,
    required_document_fraction: float = 0.97,
) -> dict[str, float | int | bool]:
    """Tokenizer/text-only donor preflight over transition metadata.

    Both frames contain one row per transition with ``doc_id``, ``token_id``,
    ``topic``, ``token_position``, and ``context_length``.  Donor prefixes are
    opened in deterministic doc-id order and matching criteria never change.
    """
    columns = {"doc_id", "token_id", "topic", "token_position", "context_length"}
    if columns - set(candidates) or columns - set(donors):
        raise ValueError("donor preflight transition metadata is incomplete")
    donor_ids = np.asarray(sorted(donors.doc_id.unique(), key=str))
    limit = min(maximum, len(donor_ids))
    last_fraction = 0.0
    for size in range(min(increments, limit), limit + 1, increments):
        opened = donors.loc[donors.doc_id.isin(donor_ids[:size])]
        counts = (
            opened.groupby(["token_id", "topic", "token_position", "context_length"])
            .doc_id.nunique()
            .rename("matches")
        )
        checked = candidates.join(counts, on=["token_id", "topic", "token_position", "context_length"])
        checked["valid"] = checked.matches.fillna(0) >= required_matches
        coverage = checked.groupby("doc_id").valid.mean()
        last_fraction = float(np.mean(coverage >= minimum_token_coverage))
        if last_fraction >= required_document_fraction:
            return {"passed": True, "donor_documents_opened": size, "eligible_document_fraction": last_fraction}
    return {"passed": False, "donor_documents_opened": limit, "eligible_document_fraction": last_fraction}


def power_gate_simulation(
    *,
    blocks: int,
    simulations: int,
    randomization_draws: int,
    seed: int,
    effect: float,
    seed_sd: float,
    noise_sd: float,
    timing_sd: float = 0.0,
    quadratic: float = 0.0,
    base_noise_sd: float = 1.0,
    block_positions: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Estimate the full gate using the prospective eight-target mechanism.

    Three independent, prespecified Monte Carlo assignment banks mirror the
    held p-value, primary-interval, and base-interval seeds.  Each bank is
    common across simulated outcomes, making hundreds of studies tractable as
    matrix products without changing the assignment law.
    """
    if blocks < 1 or simulations < 1 or randomization_draws < 1:
        raise ValueError("power dimensions must be positive")
    rng = np.random.default_rng(seed)
    positions = np.arange(blocks) if block_positions is None else np.asarray(block_positions)
    if positions.shape != (blocks,) or len(np.unique(positions)) != blocks:
        raise ValueError("power block positions must be unique and aligned")
    selected_targets = (2, 3, 4)  # the prospective validation target positions
    observed_x, observed_d = _power_design_bank(
        simulations, blocks, rng, positions, selected_targets
    )
    test_x, _ = _power_design_bank(
        randomization_draws, blocks, rng, positions, selected_targets
    )
    interval_x, _ = _power_design_bank(
        randomization_draws, blocks, rng, positions, selected_targets
    )
    base_interval_x, _ = _power_design_bank(
        randomization_draws, blocks, rng, positions, selected_targets
    )
    seed_effects = rng.normal(effect, seed_sd, (simulations, 1, 3, 1))
    block_effects = rng.normal(0.0, noise_sd, (simulations, blocks, 1, 1))
    outcomes = (
        block_effects
        + seed_effects * observed_d
        + quadratic * observed_d * observed_d
        + rng.normal(0.0, noise_sd, observed_d.shape)
        + rng.normal(0.0, timing_sd, observed_d.shape) * observed_d
    )
    centered_sum_squares = float(np.sum((DOSE_D - DOSE_D.mean()) ** 2))
    phase_denominator = blocks * 3 * centered_sum_squares
    target_denominator = blocks * centered_sum_squares
    test_flat = test_x.reshape(randomization_draws, -1)
    interval_flat = interval_x.reshape(randomization_draws, -1)
    base_interval_flat = base_interval_x.reshape(randomization_draws, -1)
    outcome_flat = outcomes.reshape(simulations, -1)
    observed_flat = observed_x.reshape(simulations, -1)
    test_outcomes = test_flat @ outcome_flat.T / phase_denominator
    interval_outcomes = interval_flat @ outcome_flat.T / phase_denominator
    interval_doses = interval_flat @ observed_flat.T / phase_denominator
    estimates = np.sum(observed_x * outcomes, axis=(1, 2, 3)) / phase_denominator
    slopes = np.sum(observed_x * outcomes, axis=(1, 3)) / target_denominator
    exceedances = np.sum(
        test_outcomes >= estimates[None, :] - 1e-15, axis=0
    )
    pvalues = (exceedances + 1.0) / (randomization_draws + 1.0)
    lower = np.empty(simulations, dtype=np.float64)
    upper = np.empty(simulations, dtype=np.float64)
    for simulation in range(simulations):
        lower[simulation], upper[simulation] = (
            _constant_effect_interval_from_statistics(
                float(estimates[simulation]),
                interval_outcomes[:, simulation],
                interval_doses[:, simulation],
                confidence=0.95,
            )
        )
    primary_gates = (
        np.all(slopes > 0.0, axis=1) & (pvalues < 0.025) & (lower > 0.0)
    )
    base_outcomes = (
        rng.normal(0.0, base_noise_sd, (simulations, blocks, 1, 1))
        + rng.normal(0.0, base_noise_sd, observed_d.shape)
    )
    base_outcomes = _standardize_simulated_outcomes(base_outcomes)
    base_flat = base_outcomes.reshape(simulations, -1)
    base_u = base_interval_flat @ base_flat.T / phase_denominator
    base_v = base_interval_flat @ observed_flat.T / phase_denominator
    base_estimates = (
        np.sum(observed_x * base_outcomes, axis=(1, 2, 3)) / phase_denominator
    )
    base_gates = np.zeros(simulations, dtype=bool)
    for simulation in range(simulations):
        base_interval = _constant_effect_interval_from_statistics(
            float(base_estimates[simulation]),
            base_u[:, simulation],
            base_v[:, simulation],
            confidence=0.90,
        )
        base_gates[simulation] = (
            base_interval[0] >= -0.05 and base_interval[1] <= 0.05
        )
    gates = primary_gates & base_gates
    passed = int(np.count_nonzero(gates))
    probability = passed / simulations
    return {
        "blocks": blocks,
        "simulations": simulations,
        "passed": passed,
        "compound_power": probability,
        "primary_compound_passed": int(np.count_nonzero(primary_gates)),
        "base_validity_passed": int(np.count_nonzero(base_gates)),
        "randomization_draws": randomization_draws,
        "full_design_targets": 8,
        "phase_target_positions": list(selected_targets),
        "independent_monte_carlo_assignment_banks": 3,
        "assignment_banks_common_across_simulated_outcomes": True,
    }


def equivalence_power_simulation(
    *,
    blocks: int,
    simulations: int,
    randomization_draws: int,
    seed: int,
    noise_sd: float = 1.0,
    true_slope: float = 0.0,
    margin: float = 0.05,
    block_positions: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Power for a 90% design interval to fit inside an equivalence margin."""
    if blocks < 1 or simulations < 1 or randomization_draws < 1:
        raise ValueError("power dimensions must be positive")
    rng = np.random.default_rng(seed)
    positions = np.arange(blocks) if block_positions is None else np.asarray(block_positions)
    if positions.shape != (blocks,) or len(np.unique(positions)) != blocks:
        raise ValueError("power block positions must be unique and aligned")
    selected_targets = (2, 3, 4)
    observed_x, observed_d = _power_design_bank(
        simulations, blocks, rng, positions, selected_targets
    )
    randomized_x, _ = _power_design_bank(
        randomization_draws, blocks, rng, positions, selected_targets
    )
    outcomes = (
        rng.normal(0.0, noise_sd, (simulations, blocks, 1, 1))
        + true_slope * observed_d
        + rng.normal(0.0, noise_sd, observed_d.shape)
    )
    outcomes = _standardize_simulated_outcomes(outcomes)
    denominator = blocks * 3 * float(np.sum((DOSE_D - DOSE_D.mean()) ** 2))
    randomized_flat = randomized_x.reshape(randomization_draws, -1)
    outcome_flat = outcomes.reshape(simulations, -1)
    observed_flat = observed_x.reshape(simulations, -1)
    u = randomized_flat @ outcome_flat.T / denominator
    v = randomized_flat @ observed_flat.T / denominator
    estimates = np.sum(observed_x * outcomes, axis=(1, 2, 3)) / denominator
    passed = 0
    for simulation in range(simulations):
        interval = _constant_effect_interval_from_statistics(
            float(estimates[simulation]),
            u[:, simulation],
            v[:, simulation],
            confidence=0.90,
        )
        passed += interval[0] >= -margin and interval[1] <= margin
    return {
        "blocks": blocks,
        "simulations": simulations,
        "passed": passed,
        "equivalence_power": passed / simulations,
        "margin": margin,
        "true_slope": true_slope,
        "randomization_draws": randomization_draws,
        "full_design_targets": 8,
        "phase_target_positions": list(selected_targets),
        "independent_monte_carlo_assignment_banks": 1,
        "assignment_banks_common_across_simulated_outcomes": True,
    }


def _power_design_bank(
    draws: int,
    blocks: int,
    rng: np.random.Generator,
    block_positions: np.ndarray,
    selected_targets: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return centered and raw log-dose banks from the full Latin design."""
    centered = np.empty((draws, blocks, len(selected_targets), 6), dtype=np.float64)
    raw = np.empty_like(centered)
    completed = 0
    while completed < draws:
        count = min(128, draws - completed)
        cubes = _latin_cubes(count, blocks, 8, rng, block_positions)[
            :, :, selected_targets, :
        ]
        dose = np.log2(cubes + 1.0)
        raw[completed : completed + count] = dose
        centered[completed : completed + count] = dose - dose.mean(
            axis=3, keepdims=True
        )
        completed += count
    return centered, raw


def _standardize_simulated_outcomes(outcomes: np.ndarray) -> np.ndarray:
    """Standardize each simulated panel as in the held equivalence analyses."""
    values = np.asarray(outcomes, dtype=np.float64)
    axes = tuple(range(1, values.ndim))
    mean = values.mean(axis=axes, keepdims=True)
    count = int(np.prod(values.shape[1:]))
    if count < 2:
        raise ValueError("simulated standardization needs at least two observations")
    scale = np.sqrt(
        np.sum((values - mean) ** 2, axis=axes, keepdims=True) / (count - 1)
    )
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("simulated standardization scale must be positive")
    return (values - mean) / scale


def select_powered_block_count(results: Iterable[dict[str, float | int]], minimum_power: float = 0.90) -> int:
    eligible = sorted(int(result["blocks"]) for result in results if float(result["compound_power"]) >= minimum_power)
    if not eligible:
        raise RuntimeError("no candidate held sample size reaches the compound power requirement")
    if any(value not in {200, 250, 300} for value in eligible):
        raise ValueError("held block candidates must be 200, 250, or 300")
    return eligible[0]


class PhaseAccess:
    """Filesystem guard for development freeze and sequential held opening."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def frozen_spec(self) -> Path:
        return self.root / "frozen_analysis_spec.json"

    @property
    def validation_decision(self) -> Path:
        return self.root / "access" / "validation_decision.json"

    def require_openable(self, phase: str) -> None:
        if phase == "development":
            return
        if phase not in {"validation", "confirmation"}:
            raise ValueError("phase must be development, validation, or confirmation")
        if not self.frozen_spec.exists():
            raise PermissionError("held outcomes remain sealed until the development specification is frozen")
        if phase == "confirmation" and not self.validation_decision.exists():
            raise PermissionError("confirmation remains sealed until the validation decision is recorded")
        if phase == "confirmation":
            decision = json.loads(self.validation_decision.read_text())
            if "gate_passed" not in decision:
                raise PermissionError("validation decision is incomplete")
