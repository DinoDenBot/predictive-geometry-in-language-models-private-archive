"""Prospective 600-block exposure-geometry extension.

This module is deliberately outcome-free.  It contains pure geometry, design,
access-control, duplicate-audit, and frozen-analysis primitives for the final
25-day extension.  Historical 700-block/8-target study code remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


DOSES = np.asarray((0, 1, 2, 4, 8, 16), dtype=np.int64)
ROLE_BLOCKS = {"validation": 300, "confirmation": 300}
PRIMARY_70_TARGETS = {
    "validation": ("70m_1", "70m_2", "70m_3"),
    "confirmation": ("70m_4", "70m_5", "70m_6"),
}
PAIRED_160_TARGETS = {
    "70m_4": "160m_1",
    "70m_5": "160m_2",
    "70m_6": "160m_3",
}
ARCHITECTURE_STOCHASTIC_SEEDS = {
    "70m_1": 20261101,
    "70m_2": 20261102,
    "70m_3": 20261103,
    "70m_4": 20261104,
    "70m_5": 20261105,
    "70m_6": 20261106,
    "160m_1": 20261201,
    "160m_2": 20261202,
    "160m_3": 20261203,
}
NUMERICAL_TOLERANCE = 1e-12
SCALAR_DENOMINATOR_FLOOR = 1e-300


def _probability_vector(value: Sequence[float], name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 1 or len(out) < 2 or np.any(~np.isfinite(out)):
        raise ValueError(f"{name} must be a finite probability vector")
    if np.any(out <= 0.0):
        raise ValueError(f"{name} must lie in the simplex interior")
    total = float(out.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} must sum to one")
    return out / total


@dataclass(frozen=True)
class TransitionGeometry:
    """All prespecified transition-level quantities before aggregation."""

    A: float
    L: float
    R: float
    N: float
    E_y: float
    D_p: float
    D_log: float
    D_z: float
    D_sqrt: float


def transition_geometry(
    p: Sequence[float],
    q: Sequence[float],
    y: int,
    *,
    tolerance: float = NUMERICAL_TOLERANCE,
) -> TransitionGeometry:
    """Compute A/L/R/N/E_y and scalar comparators for one transition.

    Root-space ``atan2`` preserves tiny nonzero angles.  At p == q, A and E_y
    are assigned zero by convention; L, R, and N have their continuous zero
    extensions.  When displacement or realized-token ascent is numerically
    degenerate, A and R are assigned zero and N is assigned L, preserving the
    returned Pythagorean identity.  Scalar denominators use the same probability-
    product floor as the production implementation.
    """

    p_array = _probability_vector(p, "p")
    q_array = _probability_vector(q, "q")
    if p_array.shape != q_array.shape or not 0 <= int(y) < len(p_array):
        raise ValueError("p, q, and realized-token index are not aligned")
    y = int(y)
    roots_p = np.sqrt(p_array)
    roots_q = np.sqrt(q_array)
    c = float(np.clip(roots_p @ roots_q, -1.0, 1.0))
    tangent_direction = roots_q - c * roots_p
    sin_theta = float(np.linalg.norm(tangent_direction))
    theta = math.atan2(sin_theta, c)
    length = 2.0 * theta
    ascent = np.zeros_like(roots_p)
    ascent[y] = 1.0
    ascent -= roots_p[y] * roots_p
    ascent_norm = float(np.linalg.norm(ascent))
    if sin_theta <= tolerance or ascent_norm <= tolerance:
        alignment = directed = energy = 0.0
        normal = length
    else:
        alignment = float((tangent_direction / sin_theta) @ (ascent / ascent_norm))
        alignment = float(np.clip(alignment, -1.0, 1.0))
        directed = length * alignment
        normal = math.sqrt(max(length * length - directed * directed, 0.0))
        energy = float(np.clip((directed * directed) / (length * length), 0.0, 1.0))
    py = float(p_array[y])
    qy = float(q_array[y])
    scale = math.sqrt(max(py * (1.0 - py), SCALAR_DENOMINATOR_FLOOR))
    return TransitionGeometry(
        A=alignment,
        L=length,
        R=directed,
        N=normal,
        E_y=energy,
        D_p=qy - py,
        D_log=math.log(qy) - math.log(py),
        D_z=(qy - py) / scale,
        D_sqrt=2.0 * (math.sqrt(qy) - math.sqrt(py))
        / math.sqrt(max(1.0 - py, SCALAR_DENOMINATOR_FLOOR)),
    )


def fisher_log_tangent(p: Sequence[float], q: Sequence[float]) -> np.ndarray:
    """Return log_p^FR(q) in intrinsic simplex coordinates."""

    p_array = _probability_vector(p, "p")
    q_array = _probability_vector(q, "q")
    if p_array.shape != q_array.shape:
        raise ValueError("p and q are not aligned")
    r, s = np.sqrt(p_array), np.sqrt(q_array)
    c = float(np.clip(r @ s, -1.0, 1.0))
    direction = s - c * r
    sine = float(np.linalg.norm(direction))
    if sine == 0.0:
        return np.zeros_like(p_array)
    theta = math.atan2(sine, c)
    embedded_log = 2.0 * theta * direction / sine
    # d phi_p(v)_j = v_j / sqrt(p_j).
    return embedded_log * r


def realized_unit_gradient(p: Sequence[float], y: int) -> np.ndarray:
    p_array = _probability_vector(p, "p")
    if not 0 <= int(y) < len(p_array):
        raise ValueError("realized-token index is out of range")
    py = float(p_array[int(y)])
    if math.sqrt(max(1.0 - py, 0.0)) <= NUMERICAL_TOLERANCE:
        raise ValueError("realized-token ascent is numerically undefined")
    unit = -math.sqrt(py / (1.0 - py)) * p_array
    unit[int(y)] += math.sqrt(py / (1.0 - py))
    return unit


def fisher_inner(p: Sequence[float], left: Sequence[float], right: Sequence[float]) -> float:
    p_array = _probability_vector(p, "p")
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != p_array.shape or right_array.shape != p_array.shape:
        raise ValueError("tangent vectors must match p")
    return float(np.sum(left_array * right_array / p_array))


def perpendicular_tangent(p: Sequence[float], q: Sequence[float], y: int) -> np.ndarray:
    v = fisher_log_tangent(p, q)
    geometry = transition_geometry(p, q, y)
    p_array = _probability_vector(p, "p")
    ascent_norm = math.sqrt(max(1.0 - float(p_array[int(y)]), 0.0))
    if ascent_norm <= NUMERICAL_TOLERANCE:
        return v
    return v - geometry.R * realized_unit_gradient(p, y)


def binary_coarse_graining_energy(p: Sequence[float], v: Sequence[float], y: int) -> float:
    """Squared Fisher norm after C_y(p)=(p_y, 1-p_y)."""

    p_array = _probability_vector(p, "p")
    tangent = np.asarray(v, dtype=np.float64)
    if tangent.shape != p_array.shape or abs(float(tangent.sum())) > 1e-9:
        raise ValueError("v must be a tangent vector at p")
    py, vy = float(p_array[int(y)]), float(tangent[int(y)])
    return vy * vy / (py * (1.0 - py))


SUMMARY_FIELDS = ("A", "L", "R", "N", "E_y", "D_p", "D_log", "D_z", "D_sqrt")


def frozen_document_summary(
    transitions: Sequence[TransitionGeometry | Mapping[str, float]],
    *,
    quantile: float = 0.90,
) -> dict[str, float]:
    """Apply the frozen linear quantile separately to transition quantities."""

    if not transitions:
        raise ValueError("at least one transition is required")
    rows = [asdict(row) if isinstance(row, TransitionGeometry) else dict(row) for row in transitions]
    result: dict[str, float] = {}
    for field in SUMMARY_FIELDS:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        if np.any(~np.isfinite(values)):
            raise ValueError(f"non-finite transition values for {field}")
        result[f"S_{field}"] = float(np.quantile(values, quantile, method="linear"))
    return result


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).lower()).strip()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _word_shingles(text: str, width: int = 5) -> set[str]:
    words = normalized_text(text).split()
    return {" ".join(words[index : index + width]) for index in range(max(1, len(words) - width + 1))}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def dual_duplicate_audit(
    candidates: pd.DataFrame,
    references: pd.DataFrame,
    *,
    threshold: float = 0.80,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit candidate-vs-reference and candidate-vs-candidate duplicates.

    References should concatenate original candidates and the global background
    corpus.  Returned per-candidate maxima are suitable for freezing before
    training.  A candidate fails when either near-duplicate score is >= 0.80.
    """

    for name, frame in (("candidates", candidates), ("references", references)):
        if {"doc_id", "text"} - set(frame):
            raise ValueError(f"{name} require doc_id and text")
        if frame.doc_id.duplicated().any():
            raise ValueError(f"{name} doc_id values must be unique")
    candidate_rows = candidates[["doc_id", "text"]].reset_index(drop=True).copy()
    reference_rows = references[["doc_id", "text"]].reset_index(drop=True).copy()
    all_rows = pd.concat(
        [candidate_rows.assign(kind="candidate"), reference_rows.assign(kind="reference")],
        ignore_index=True,
    )
    all_rows["exact_hash"] = all_rows.text.astype(str).map(_hash)
    all_rows["normalized_hash"] = all_rows.text.astype(str).map(lambda value: _hash(normalized_text(value)))
    shingles = [_word_shingles(text) for text in all_rows.text]
    inverted: dict[str, list[int]] = {}
    for index, values in enumerate(shingles):
        for value in values:
            inverted.setdefault(value, []).append(index)
    word_max = np.zeros(len(candidate_rows), dtype=np.float64)
    word_neighbor: list[str | None] = [None] * len(candidate_rows)
    for left in range(len(candidate_rows)):
        neighbors: set[int] = set()
        for value in shingles[left]:
            neighbors.update(inverted[value])
        neighbors.discard(left)
        for right in neighbors:
            score = _jaccard(shingles[left], shingles[right])
            if score > word_max[left]:
                word_max[left] = score
                word_neighbor[left] = str(all_rows.iloc[right].doc_id)

    corpus = all_rows.text.astype(str).tolist()
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(5, 5), lowercase=True, norm="l2")
    matrix = vectorizer.fit_transform(corpus)
    neighbor_count = min(2, len(all_rows))
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=neighbor_count).fit(matrix)
    distances, indices = model.kneighbors(matrix[: len(candidate_rows)])
    char_max = np.zeros(len(candidate_rows), dtype=np.float64)
    char_neighbor: list[str | None] = [None] * len(candidate_rows)
    for left, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
        for distance, right in zip(row_distances, row_indices, strict=True):
            if int(right) == left:
                continue
            score = float(np.clip(1.0 - distance, 0.0, 1.0))
            if score > char_max[left]:
                char_max[left] = score
                char_neighbor[left] = str(all_rows.iloc[int(right)].doc_id)

    candidate_hashes = all_rows.iloc[: len(candidate_rows)]
    other_hashes = all_rows
    exact_duplicate = []
    normalized_duplicate = []
    for left, row in candidate_hashes.iterrows():
        exact_duplicate.append(bool((other_hashes.exact_hash == row.exact_hash).sum() > 1))
        normalized_duplicate.append(bool((other_hashes.normalized_hash == row.normalized_hash).sum() > 1))
    detail = pd.DataFrame(
        {
            "doc_id": candidate_rows.doc_id.astype(str),
            "exact_duplicate": exact_duplicate,
            "normalized_duplicate": normalized_duplicate,
            "word5_jaccard_max": word_max,
            "word5_neighbor_id": word_neighbor,
            "char5_tfidf_cosine_max": char_max,
            "char5_neighbor_id": char_neighbor,
        }
    )
    detail["passes"] = ~(
        detail.exact_duplicate
        | detail.normalized_duplicate
        | detail.word5_jaccard_max.ge(threshold)
        | detail.char5_tfidf_cosine_max.ge(threshold)
    )
    audit = {
        "candidate_documents": len(candidate_rows),
        "reference_documents": len(reference_rows),
        "threshold": threshold,
        "comparison": "strictly less than threshold required",
        "word_rule": "normalized word-5-shingle Jaccard",
        "character_rule": "character-5-gram TF-IDF cosine",
        "failed_documents": int((~detail.passes).sum()),
        "maximum_word5_jaccard": float(detail.word5_jaccard_max.max()) if len(detail) else 0.0,
        "maximum_char5_tfidf_cosine": float(detail.char5_tfidf_cosine_max.max()) if len(detail) else 0.0,
    }
    return detail, audit


def validate_extension_manifest(frame: pd.DataFrame) -> pd.DataFrame:
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
    out = frame.copy()
    if len(out) != 3600 or out.block_id.nunique() != 600 or out.doc_id.duplicated().any():
        raise ValueError("extension requires 600 six-document blocks and unique documents")
    counts = out[["block_id", "role"]].drop_duplicates().groupby("role").size().to_dict()
    if counts != ROLE_BLOCKS:
        raise ValueError(f"role counts differ from {ROLE_BLOCKS}")
    for block_id, block in out.groupby("block_id", sort=False):
        if len(block) != 6 or set(block.doc_slot.astype(int)) != set(range(6)):
            raise ValueError(f"block {block_id} must contain slots 0..5")
        if block.role.nunique() != 1:
            raise ValueError(f"block {block_id} spans roles")
        if block.source.nunique() != 1 or block.topic.nunique() != 1:
            raise ValueError(f"block {block_id} is not source/topic matched")
        if not np.isfinite(block.tokens.astype(float)).all() or not np.isfinite(
            block.baseline_difficulty.astype(float)
        ).all():
            raise ValueError(f"block {block_id} has non-finite matching values")
    return out.sort_values(["block_id", "doc_slot"]).reset_index(drop=True)


def construct_extension_blocks(
    candidates: pd.DataFrame,
    *,
    seed: int,
    token_tolerance: int = 8,
    difficulty_tolerance: float = 0.20,
) -> pd.DataFrame:
    """Construct 600 target-output-free source/topic/caliper-matched blocks.

    Duplicate screening is intentionally a preceding frozen operation; this
    constructor rejects repeated IDs but does not silently remove candidates.
    """

    required = {"doc_id", "source", "topic", "tokens", "baseline_difficulty"}
    if required - set(candidates):
        raise ValueError(f"candidate pool missing {sorted(required - set(candidates))}")
    if candidates.doc_id.duplicated().any():
        raise ValueError("candidate pool contains repeated document IDs")
    rng = np.random.default_rng(seed)
    blocks: list[pd.DataFrame] = []
    for _, source_topic in candidates.groupby(["source", "topic"], sort=True):
        remaining = source_topic.iloc[rng.permutation(len(source_topic))].copy()
        while len(remaining) >= 6 and len(blocks) < 600:
            anchor = remaining.iloc[0]
            eligible = remaining.loc[
                remaining.tokens.astype(float).sub(float(anchor.tokens)).abs().le(token_tolerance)
                & remaining.baseline_difficulty.astype(float)
                .sub(float(anchor.baseline_difficulty))
                .abs()
                .le(difficulty_tolerance)
            ]
            chosen: list[Any] = []
            for index in eligible.index:
                prospective = chosen + [index]
                if (
                    remaining.loc[prospective, "tokens"].astype(float).max()
                    - remaining.loc[prospective, "tokens"].astype(float).min()
                    <= token_tolerance
                    and remaining.loc[prospective, "baseline_difficulty"].astype(float).max()
                    - remaining.loc[prospective, "baseline_difficulty"].astype(float).min()
                    <= difficulty_tolerance
                ):
                    chosen.append(index)
                if len(chosen) == 6:
                    break
            if len(chosen) < 6:
                remaining = remaining.drop(index=remaining.index[0])
                continue
            blocks.append(remaining.loc[chosen].copy())
            remaining = remaining.drop(index=chosen)
        if len(blocks) == 600:
            break
    if len(blocks) != 600:
        raise ValueError(f"only {len(blocks)} matched blocks could be constructed")
    roles = np.repeat(["validation", "confirmation"], 300)
    rng.shuffle(roles)
    rows = []
    for block_id, (block, role) in enumerate(zip(blocks, roles, strict=True)):
        selected = block.iloc[rng.permutation(6)].copy()
        selected["block_id"] = block_id
        selected["doc_slot"] = np.arange(6)
        selected["role"] = role
        rows.append(selected)
    return validate_extension_manifest(pd.concat(rows, ignore_index=True))


def make_six_target_assignments(
    manifest: pd.DataFrame,
    *,
    seed: int,
    target_ids: Sequence[str] = tuple(f"70m_{index}" for index in range(1, 7)),
) -> pd.DataFrame:
    canonical = validate_extension_manifest(manifest)
    target_ids = tuple(target_ids)
    if len(target_ids) != 6 or len(set(target_ids)) != 6:
        raise ValueError("six unique target IDs are required")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for position, (block_id, block) in enumerate(canonical.groupby("block_id", sort=True)):
        base = rng.permutation(DOSES)
        target_order = rng.permutation(6)
        for target_position, target_id in enumerate(target_ids):
            shift = int(target_order[target_position])
            doses = np.roll(base, shift)
            for slot, dose in enumerate(doses):
                document = block.loc[block.doc_slot.astype(int) == slot].iloc[0]
                rows.append(
                    {
                        "block_id": block_id,
                        "latin_block_position": position,
                        "doc_id": document.doc_id,
                        "doc_slot": slot,
                        "role": document.role,
                        "target_id": target_id,
                        "K": int(dose),
                        "d": math.log2(int(dose) + 1),
                    }
                )
    result = pd.DataFrame(rows)
    validate_six_target_assignments(result, target_ids)
    return result


def validate_six_target_assignments(frame: pd.DataFrame, target_ids: Sequence[str]) -> None:
    required = {"block_id", "doc_id", "doc_slot", "role", "target_id", "K", "d"}
    if required - set(frame) or set(frame.target_id) != set(target_ids):
        raise ValueError("assignment table is incomplete or target IDs differ")
    for key, group in frame.groupby(["block_id", "target_id"], sort=False):
        if sorted(group.K.astype(int)) != DOSES.tolist() or int(group.K.sum()) != 31:
            raise ValueError(f"invalid complete allocation for {key}")
    rotations = frame.groupby(["block_id", "doc_id"]).K.apply(lambda values: set(values.astype(int)))
    if not rotations.map(lambda values: values == set(DOSES.tolist())).all():
        raise ValueError("every document must rotate through all six doses")
    if not np.allclose(frame.d, np.log2(frame.K + 1.0), rtol=0.0, atol=1e-12):
        raise ValueError("d must equal log2(K+1)")


def _six_target_design_bank(
    draws: int,
    blocks: int,
    rng: np.random.Generator,
    selected_targets: Sequence[int] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Draw centered and raw log doses from the complete six-target mechanism."""

    base_order = np.argsort(rng.random((draws, blocks, 6)), axis=2)
    bases = DOSES[base_order]
    target_orders = np.argsort(rng.random((draws, blocks, 6)), axis=2)
    shifts = target_orders[:, :, np.asarray(selected_targets, dtype=int)]
    slots = np.arange(6)[None, None, None, :]
    source_slots = (slots - shifts[:, :, :, None]) % 6
    expanded = np.broadcast_to(
        bases[:, :, None, :], (draws, blocks, len(selected_targets), 6)
    )
    cubes = np.take_along_axis(expanded, source_slots, axis=3)
    raw = np.log2(cubes + 1.0)
    return raw - raw.mean(axis=3, keepdims=True), raw


def complete_r_gate_power_simulation(
    *,
    blocks: int,
    simulations: int,
    randomization_draws: int,
    seed: int,
    effect: float,
    target_slope_sd: float,
    noise_sd: float,
) -> dict[str, float | int]:
    """Power for all-positive slopes, p<.025, and positive 95% interval.

    The observed and two Monte Carlo banks are independent draws from the exact
    complete six-target assignment law.  Cross-target Latin constraints are
    preserved, then the statistic is computed from phase targets 1--3 only.
    The interval uses the same plus-one constant-effect inversion as production.
    """

    if min(blocks, simulations, randomization_draws) < 1 or noise_sd <= 0:
        raise ValueError("power dimensions and noise scale must be positive")
    from exposure_observability import _constant_effect_interval_from_statistics

    rng = np.random.default_rng(seed)
    observed_x, observed_d = _six_target_design_bank(simulations, blocks, rng)
    test_x, _ = _six_target_design_bank(randomization_draws, blocks, rng)
    interval_x, _ = _six_target_design_bank(randomization_draws, blocks, rng)
    target_effects = rng.normal(effect, target_slope_sd, (simulations, 1, 3, 1))
    outcomes = (
        rng.normal(0.0, noise_sd, (simulations, blocks, 1, 1))
        + target_effects * observed_d
        + rng.normal(0.0, noise_sd, observed_d.shape)
    )
    centered_dose_ss = float(np.sum((np.log2(DOSES + 1.0) - np.log2(DOSES + 1.0).mean()) ** 2))
    target_denominator = blocks * centered_dose_ss
    phase_denominator = 3 * target_denominator
    observed_flat = observed_x.reshape(simulations, -1)
    outcome_flat = outcomes.reshape(simulations, -1)
    estimates = np.sum(observed_x * outcomes, axis=(1, 2, 3)) / phase_denominator
    slopes = np.sum(observed_x * outcomes, axis=(1, 3)) / target_denominator
    randomized_statistics = test_x.reshape(randomization_draws, -1) @ outcome_flat.T / phase_denominator
    pvalues = (
        np.sum(randomized_statistics >= estimates[None, :] - 1e-15, axis=0) + 1.0
    ) / (randomization_draws + 1.0)
    interval_u = interval_x.reshape(randomization_draws, -1) @ outcome_flat.T / phase_denominator
    interval_v = interval_x.reshape(randomization_draws, -1) @ observed_flat.T / phase_denominator
    lower = np.empty(simulations, dtype=np.float64)
    for index in range(simulations):
        lower[index], _ = _constant_effect_interval_from_statistics(
            float(estimates[index]),
            interval_u[:, index],
            interval_v[:, index],
            confidence=0.95,
        )
    passed = np.all(slopes > 0.0, axis=1) & (pvalues < 0.025) & (lower > 0.0)
    return {
        "blocks": blocks,
        "simulations": simulations,
        "randomization_draws": randomization_draws,
        "passed": int(passed.sum()),
        "compound_power": float(passed.mean()),
        "full_design_targets": 6,
        "phase_targets": 3,
        "independent_monte_carlo_assignment_banks": 2,
    }


def six_target_equivalence_power_simulation(
    *,
    blocks: int,
    simulations: int,
    randomization_draws: int,
    seed: int,
    true_slope: float = 0.0,
    margin: float = 0.05,
    noise_sd: float = 1.0,
) -> dict[str, float | int]:
    """Power for a 90% equivalence interval under the six-target mechanism."""

    if (
        min(blocks, simulations, randomization_draws) < 1
        or margin <= 0.0
        or noise_sd <= 0.0
    ):
        raise ValueError("power dimensions and equivalence margin must be positive")
    from exposure_observability import (
        _constant_effect_interval_from_statistics,
        _standardize_simulated_outcomes,
    )

    rng = np.random.default_rng(seed)
    observed_x, observed_d = _six_target_design_bank(simulations, blocks, rng)
    interval_x, _ = _six_target_design_bank(randomization_draws, blocks, rng)
    outcomes = (
        true_slope * observed_d
        + rng.normal(0.0, noise_sd, (simulations, blocks, 1, 1))
        + rng.normal(0.0, noise_sd, observed_d.shape)
    )
    outcomes = _standardize_simulated_outcomes(outcomes)
    denominator = blocks * 3 * float(
        np.sum((np.log2(DOSES + 1.0) - np.log2(DOSES + 1.0).mean()) ** 2)
    )
    interval_flat = interval_x.reshape(randomization_draws, -1)
    outcome_flat = outcomes.reshape(simulations, -1)
    observed_flat = observed_x.reshape(simulations, -1)
    u = interval_flat @ outcome_flat.T / denominator
    v = interval_flat @ observed_flat.T / denominator
    estimates = np.sum(observed_x * outcomes, axis=(1, 2, 3)) / denominator
    passed = 0
    for simulation in range(simulations):
        lower, upper = _constant_effect_interval_from_statistics(
            float(estimates[simulation]),
            u[:, simulation],
            v[:, simulation],
            confidence=0.90,
        )
        passed += lower >= -margin and upper <= margin
    return {
        "blocks": blocks,
        "simulations": simulations,
        "passed": passed,
        "equivalence_power": passed / simulations,
        "margin": margin,
        "true_slope": true_slope,
        "validation_noise_sd": noise_sd,
        "randomization_draws": randomization_draws,
        "full_design_targets": 6,
        "phase_targets": 3,
        "independent_monte_carlo_assignment_banks": 1,
    }


def select_extension_block_count(
    results: Sequence[Mapping[str, float | int]], minimum_power: float = 0.90
) -> int:
    sizes = {int(result["blocks"]) for result in results}
    if sizes != {200, 250, 300}:
        raise ValueError("power freeze requires results for exactly 200, 250, and 300 blocks")
    eligible = sorted(
        int(result["blocks"])
        for result in results
        if float(result["compound_power"]) >= minimum_power
    )
    if not eligible:
        raise RuntimeError("no candidate size reaches the frozen power requirement")
    return eligible[0]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


class ExtensionAccess:
    """Prospective 3+3, complementary, and 160M filesystem access guard."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def frozen_spec(self) -> Path:
        return self.root / "frozen_analysis_spec.json"

    def decision(self, name: str) -> Path:
        return self.root / "access" / f"{name}_decision.json"

    def _complete(self, name: str) -> bool:
        path = self.decision(name)
        if not path.exists():
            return False
        value = json.loads(path.read_text())
        return bool(value.get("decision_complete") is True)

    def require_openable(self, architecture: str, role: str, target_id: str) -> str:
        if not self.frozen_spec.exists():
            raise PermissionError("all outcomes remain sealed until the specification is frozen")
        if architecture == "70m":
            if role == "validation" and target_id in PRIMARY_70_TARGETS["validation"]:
                return "primary_validation"
            if role == "confirmation" and target_id in PRIMARY_70_TARGETS["confirmation"]:
                if not self._complete("70m_validation"):
                    raise PermissionError("70M confirmation is sealed until validation decision completion")
                return "primary_confirmation"
            complementary = (
                role == "validation" and target_id in PRIMARY_70_TARGETS["confirmation"]
            ) or (
                role == "confirmation" and target_id in PRIMARY_70_TARGETS["validation"]
            )
            if complementary:
                if not self._complete("70m_confirmation"):
                    raise PermissionError("complementary cells are sealed until primary confirmation completion")
                return "complete_latin_post_confirmation_sensitivity"
        if architecture == "160m" and target_id in PAIRED_160_TARGETS.values():
            if not self._complete("70m_confirmation"):
                raise PermissionError("160M outcomes are sealed until 70M confirmation completion")
            if role == "validation":
                return "160m_validation"
            if role == "confirmation":
                if not self._complete("160m_validation_power"):
                    raise PermissionError("160M confirmation is sealed until validation and power decisions")
                return "160m_confirmation"
        raise ValueError("architecture, role, and target do not name an authorized cell")

    def record_open(self, architecture: str, role: str, target_id: str) -> Path:
        purpose = self.require_openable(architecture, role, target_id)
        path = self.root / "access" / "opens" / f"{architecture}_{role}_{target_id}.json"
        atomic_json(
            path,
            {
                "architecture": architecture,
                "role": role,
                "target_id": target_id,
                "purpose": purpose,
                "frozen_spec_sha256": hashlib.sha256(self.frozen_spec.read_bytes()).hexdigest(),
            },
        )
        return path


@dataclass(frozen=True)
class ResidualCalibration:
    gamma: float
    sigma_dev: float
    observations: int


def fit_development_residual(
    frame: pd.DataFrame,
    *,
    response: str = "delta_S_R",
    comparator: str = "delta_S_D_z",
) -> ResidualCalibration:
    required = {"target_id", "block_id", response, comparator}
    if required - set(frame):
        raise ValueError(f"development residual frame missing {sorted(required - set(frame))}")
    if frame[[response, comparator]].isna().any().any():
        raise ValueError("development residual inputs must be complete")
    groups = [frame.target_id, frame.block_id]
    y = frame[response] - frame.groupby(groups)[response].transform("mean")
    x = frame[comparator] - frame.groupby(groups)[comparator].transform("mean")
    denominator = float(x @ x)
    if denominator <= 0.0:
        raise ValueError("development D_z slope is unidentified")
    gamma = float(x @ y / denominator)
    raw_residual = frame[response].to_numpy(float) - gamma * frame[comparator].to_numpy(float)
    residual_series = pd.Series(raw_residual, index=frame.index)
    adjusted = (
        residual_series
        - residual_series.groupby(frame.target_id).transform("mean")
        - residual_series.groupby(frame.block_id).transform("mean")
        + float(residual_series.mean())
    ).to_numpy()
    rank = frame.target_id.nunique() + frame.block_id.nunique() - 1
    degrees = len(frame) - rank - 1
    if degrees <= 0:
        raise ValueError("insufficient development degrees of freedom")
    sigma = math.sqrt(float(adjusted @ adjusted) / degrees)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("development residual scale must be positive")
    return ResidualCalibration(gamma=gamma, sigma_dev=sigma, observations=len(frame))


def apply_residual_calibration(
    frame: pd.DataFrame,
    calibration: ResidualCalibration,
    *,
    response: str = "delta_S_R",
    comparator: str = "delta_S_D_z",
) -> pd.DataFrame:
    if {response, comparator} - set(frame):
        raise ValueError("held residual frame is incomplete")
    out = frame.copy()
    out["U"] = (
        out[response].to_numpy(float)
        - calibration.gamma * out[comparator].to_numpy(float)
    ) / calibration.sigma_dev
    return out


def three_way_decision(
    *,
    validation_positive_gate: bool,
    confirmation_positive_gate: bool,
    validation_equivalence_interval: tuple[float, float],
    confirmation_equivalence_interval: tuple[float, float],
    validation_equivalence_powered: bool,
    confirmation_equivalence_powered: bool,
    margin: float = 0.05,
) -> str:
    if validation_positive_gate and confirmation_positive_gate:
        return "beyond_local_scalar_response"
    equivalent = (
        validation_equivalence_powered
        and confirmation_equivalence_powered
        and validation_equivalence_interval[0] >= -margin
        and validation_equivalence_interval[1] <= margin
        and confirmation_equivalence_interval[0] >= -margin
        and confirmation_equivalence_interval[1] <= margin
    )
    return "no_meaningful_beyond_local_scalar_response" if equivalent else "inconclusive"


def hierarchical_claim_decisions(
    *,
    r70_validation: bool,
    r70_confirmation: bool,
    a70_validation: bool,
    a70_confirmation: bool,
    l70_validation_equivalent: bool,
    l70_confirmation_equivalent: bool,
    n70_decision: str,
    u_decision: str,
    r160_validation: bool | None,
    r160_confirmation: bool | None,
    a160_validation: bool | None = None,
    a160_confirmation: bool | None = None,
) -> dict[str, Any]:
    """Apply the modular C1 -> C2a/C2b/C2c -> C3 -> C4 claim ladder."""

    valid_n = {"positive_response", "powered_equivalence", "inconclusive"}
    if n70_decision not in valid_n:
        raise ValueError(f"N decision must be one of {sorted(valid_n)}")
    r70 = bool(r70_validation and r70_confirmation)
    a70 = bool(r70 and a70_validation and a70_confirmation)
    l70 = bool(r70 and l70_validation_equivalent and l70_confirmation_equivalent)
    n70_equivalent = bool(r70 and n70_decision == "powered_equivalence")
    n70_positive = bool(r70 and n70_decision == "positive_response")
    realized_localization = bool(r70 and n70_equivalent)
    reorientation_without_total_increase = bool(a70 and l70)
    u_eligible = r70
    r160_observed = bool(r160_validation and r160_confirmation)
    a160_observed = bool(a160_validation and a160_confirmation)
    return {
        "R70_replicated": r70,
        "A70_replicated": a70,
        "L70_powered_equivalence": l70,
        "N70_powered_equivalence": n70_equivalent,
        "N70_positive_response": n70_positive,
        "realized_coordinate_localization": realized_localization,
        "reorientation_without_meaningful_total_motion_increase": reorientation_without_total_increase,
        "complementary_token_redistribution": n70_positive,
        "U_eligible": u_eligible,
        "U_decision": u_decision if u_eligible else "reported_descriptively_only",
        "R160_replicated": bool(r70 and r160_observed),
        "A160_replicated_secondary": bool(r70 and a160_observed),
        "R160_observed_without_rescue_authority": bool((not r70) and r160_observed),
        "claim_ladder": "C1 R70 -> C2a A70 / C2b N70 / C2c L70 -> C3 U -> C4 R160",
    }


def validate_architecture_stochastic_seeds(
    seeds: Mapping[str, int] = ARCHITECTURE_STOCHASTIC_SEEDS,
) -> None:
    """Require independent model-noise seeds while pairing the intervention."""

    required = set(PRIMARY_70_TARGETS["validation"] + PRIMARY_70_TARGETS["confirmation"])
    required.update(PAIRED_160_TARGETS.values())
    if set(seeds) != required or len(set(int(value) for value in seeds.values())) != len(required):
        raise ValueError("all 70M and 160M targets require unique stochastic seeds")
    for target_70m, target_160m in PAIRED_160_TARGETS.items():
        if int(seeds[target_70m]) == int(seeds[target_160m]):
            raise ValueError(f"paired targets share a stochastic seed: {target_70m}, {target_160m}")


def standardized_separation_and_auc(
    frame: pd.DataFrame,
    outcome: str,
    *,
    baseline_dose: int = 0,
) -> pd.DataFrame:
    """Descriptive dose-vs-zero standardized separation and AUC."""

    if {"K", "block_id", outcome} - set(frame):
        raise ValueError("descriptive frame is incomplete")
    rows = []
    baseline = frame.loc[frame.K == baseline_dose]
    for dose in sorted(set(frame.K.astype(int)) - {baseline_dose}):
        exposed = frame.loc[frame.K == dose]
        pooled = math.sqrt((float(baseline[outcome].var(ddof=1)) + float(exposed[outcome].var(ddof=1))) / 2.0)
        separation = (float(exposed[outcome].mean()) - float(baseline[outcome].mean())) / pooled if pooled > 0 else math.nan
        labels = np.r_[np.zeros(len(baseline)), np.ones(len(exposed))]
        scores = np.r_[baseline[outcome].to_numpy(float), exposed[outcome].to_numpy(float)]
        rows.append({"outcome": outcome, "dose": dose, "standardized_separation": separation, "auc": float(roc_auc_score(labels, scores))})
    return pd.DataFrame(rows)


LEDGER_IDENTITY_COLUMNS = (
    "block_id",
    "doc_id",
    "doc_slot",
    "role",
    "K",
    "d",
    "occurrence",
    "optimizer_step",
    "batch_position",
    "presentation_id",
)


def intervention_fingerprint(ledger: pd.DataFrame) -> str:
    missing = set(LEDGER_IDENTITY_COLUMNS) - set(ledger)
    if missing:
        raise ValueError(f"ledger missing {sorted(missing)}")
    canonical = ledger.loc[:, LEDGER_IDENTITY_COLUMNS].sort_values(
        ["optimizer_step", "batch_position", "presentation_id"]
    )
    payload = canonical.to_json(orient="records", double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_paired_intervention(ledger_70m: pd.DataFrame, ledger_160m: pd.DataFrame) -> str:
    left = intervention_fingerprint(ledger_70m)
    right = intervention_fingerprint(ledger_160m)
    if left != right:
        raise ValueError("paired 70M/160M intervention ledgers differ")
    return left


def artifact_hash_ledger(paths: Sequence[Path], *, relative_to: Path) -> dict[str, str]:
    """Return deterministic SHA256 entries without leaking absolute paths."""

    root = Path(relative_to).resolve()
    ledger: dict[str, str] = {}
    for raw_path in sorted((Path(path) for path in paths), key=lambda path: str(path)):
        path = raw_path.resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"artifact lies outside anonymous root: {raw_path}") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        ledger[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return ledger


def token_weighted_effective_batch_step(
    microbatches: Sequence[tuple[torch.Tensor, int]],
    parameters: Iterable[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float,
) -> float:
    """Accumulate token-weighted mean losses, clip once, and update once."""

    if not microbatches or any(int(tokens) <= 0 for _, tokens in microbatches):
        raise ValueError("microbatches require positive predicted-token counts")
    parameter_list = list(parameters)
    total_tokens = sum(int(tokens) for _, tokens in microbatches)
    optimizer.zero_grad(set_to_none=True)
    weighted_value = 0.0
    for loss, tokens in microbatches:
        weight = int(tokens) / total_tokens
        (loss * weight).backward()
        weighted_value += float(loss.detach().cpu()) * weight
    torch.nn.utils.clip_grad_norm_(parameter_list, max_grad_norm)
    optimizer.step()
    return weighted_value
