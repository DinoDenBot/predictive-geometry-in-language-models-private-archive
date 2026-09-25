"""CATS and LAR-2 definitions for target-only membership identification.

The functions in this module consume probability-derived arrays only. Model
querying, labels, and split selection deliberately live outside this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import norm


def context_length_grid(preceding_tokens: int, grid_size: int = 24) -> np.ndarray:
    """Return the logarithmically spaced context grid used by LAR."""
    if preceding_tokens < 1:
        raise ValueError("preceding_tokens must be positive")
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    u = np.linspace(0.0, 1.0, grid_size)
    lengths = np.rint((preceding_tokens + 1.0) ** u - 1.0).astype(int)
    lengths = np.clip(lengths, 1, preceding_tokens)
    lengths[-1] = preceding_tokens
    return lengths


def linear_hat_basis(values: Sequence[float], dimension: int) -> np.ndarray:
    """Evaluate a partition-of-unity linear spline basis on [0, 1]."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("basis values must be finite and in [0, 1]")
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    knots = np.linspace(0.0, 1.0, dimension)
    width = knots[1] - knots[0]
    basis = np.maximum(1.0 - np.abs(x[:, None] - knots[None, :]) / width, 0.0)
    return basis / basis.sum(axis=1, keepdims=True)


def targeted_transport_from_roots(
    shorter_roots: np.ndarray,
    longer_roots: np.ndarray,
    observed_token_ids: Sequence[int],
    *,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target-direction cosine and Fisher--Rao distance.

    Rows are square-root probability vectors. The cosine is the alignment of
    the context-induced tangent step with the Fisher direction that increases
    the observed token probability.
    """
    r0 = np.asarray(shorter_roots, dtype=np.float64)
    r1 = np.asarray(longer_roots, dtype=np.float64)
    ids = np.asarray(observed_token_ids, dtype=int)
    if r0.ndim != 2 or r1.shape != r0.shape or ids.shape != (len(r0),):
        raise ValueError("root arrays and observed-token IDs are not aligned")
    if np.any(r0 < 0) or np.any(r1 < 0) or np.any(ids < 0) or np.any(ids >= r0.shape[1]):
        raise ValueError("invalid probability roots or observed-token IDs")
    r0 = r0 / np.linalg.norm(r0, axis=1, keepdims=True).clip(min=epsilon)
    r1 = r1 / np.linalg.norm(r1, axis=1, keepdims=True).clip(min=epsilon)
    overlap = np.sum(r0 * r1, axis=1).clip(-1.0, 1.0)
    rows = np.arange(len(ids))
    r0y = r0[rows, ids]
    r1y = r1[rows, ids]
    numerator = r1y - overlap * r0y
    denominator = np.sqrt(np.maximum(1.0 - overlap**2, 0.0)) * np.sqrt(
        np.maximum(1.0 - r0y**2, 0.0)
    )
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > epsilon)
    return cosine.clip(-1.0, 1.0), 2.0 * np.arccos(overlap)


def fisher_rao_alr_from_roots(
    shorter_roots: np.ndarray,
    longer_roots: np.ndarray,
    observed_token_ids: Sequence[int],
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the explicit CATS alignment, length, and directed displacement.

    The square-root convention is ``phi(p) = 2 sqrt(p)``.  ``A`` is the
    cosine with the unit Fisher--Rao direction that increases the realized
    token probability, ``L`` is the Fisher--Rao geodesic length, and
    ``R = A * L`` is the exact logarithmic-map projection.  At coincident
    distributions, ``L`` and ``R`` extend continuously to zero; the direction
    cosine has no unique limit, so the implementation assigns ``A = 0`` by
    convention.  A numerically degenerate realized-token ascent direction is
    likewise assigned ``A = R = 0`` as an explicit numerical convention.

    ``atan2`` is used for the spherical angle so tiny non-zero transitions do
    not disappear through rounding of an ``arccos`` argument to one.
    """
    r0 = np.asarray(shorter_roots, dtype=np.float64)
    r1 = np.asarray(longer_roots, dtype=np.float64)
    ids = np.asarray(observed_token_ids, dtype=int)
    if r0.ndim != 2 or r1.shape != r0.shape or ids.shape != (len(r0),):
        raise ValueError("root arrays and observed-token IDs are not aligned")
    if (
        np.any(~np.isfinite(r0))
        or np.any(~np.isfinite(r1))
        or np.any(r0 < 0)
        or np.any(r1 < 0)
        or np.any(ids < 0)
        or np.any(ids >= r0.shape[1])
    ):
        raise ValueError("invalid probability roots or observed-token IDs")
    norm0 = np.linalg.norm(r0, axis=1, keepdims=True)
    norm1 = np.linalg.norm(r1, axis=1, keepdims=True)
    if np.any(norm0 <= epsilon) or np.any(norm1 <= epsilon):
        raise ValueError("probability roots must have positive norm")
    r0 = r0 / norm0
    r1 = r1 / norm1

    overlap = np.sum(r0 * r1, axis=1).clip(-1.0, 1.0)
    tangent = r1 - overlap[:, None] * r0
    sine = np.linalg.norm(tangent, axis=1)
    angle = np.arctan2(sine, overlap)
    length = 2.0 * angle

    rows = np.arange(len(ids))
    r0y = r0[rows, ids]
    ascent_norm = np.sqrt(np.maximum(1.0 - r0y**2, 0.0))
    valid = (sine > epsilon) & (ascent_norm > epsilon)
    alignment = np.zeros(len(ids), dtype=np.float64)
    if np.any(valid):
        unit_tangent = tangent[valid] / sine[valid, None]
        ascent = np.zeros_like(unit_tangent)
        ascent_rows = np.arange(np.sum(valid))
        ascent[ascent_rows, ids[valid]] = 1.0
        ascent -= r0y[valid, None] * r0[valid]
        ascent /= ascent_norm[valid, None]
        alignment[valid] = np.sum(unit_tangent * ascent, axis=1).clip(-1.0, 1.0)
    directed = alignment * length
    directed[~valid] = 0.0
    return alignment, length, directed


def fisher_rao_alr(
    shorter_probabilities: np.ndarray,
    longer_probabilities: np.ndarray,
    observed_token_ids: Sequence[int],
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probability-vector wrapper for :func:`fisher_rao_alr_from_roots`."""
    p = np.asarray(shorter_probabilities, dtype=np.float64)
    q = np.asarray(longer_probabilities, dtype=np.float64)
    if p.ndim != 2 or q.shape != p.shape:
        raise ValueError("probability arrays must be aligned matrices")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)) or np.any(p < 0) or np.any(q < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    totals_p = p.sum(axis=1, keepdims=True)
    totals_q = q.sum(axis=1, keepdims=True)
    if np.any(totals_p <= epsilon) or np.any(totals_q <= epsilon):
        raise ValueError("probability rows must have positive mass")
    return fisher_rao_alr_from_roots(
        np.sqrt(p / totals_p), np.sqrt(q / totals_q), observed_token_ids, epsilon=epsilon
    )


def cats_document_features(cats_array: np.ndarray, distinct_steps: np.ndarray) -> dict[str, float]:
    """Reduce token-by-context CATS channels without labels.

    ``cats_array[..., 0:3]`` contains target cosine, Fisher--Rao length, and
    observed-token log-probability increment. ``distinct_steps`` excludes grid
    repeats caused by rounding on short prefixes.
    """
    values = np.asarray(cats_array, dtype=np.float64)
    distinct = np.asarray(distinct_steps, dtype=bool)
    if values.ndim != 3 or values.shape[2] != 3 or distinct.shape != values.shape[:2]:
        raise ValueError("CATS values and distinct-step mask are not aligned")
    if np.any(~np.isfinite(values)):
        raise ValueError("CATS values must be finite")

    z = np.full(distinct.shape, np.nan, dtype=np.float64)
    for step in range(values.shape[1]):
        eligible = distinct[:, step]
        if not np.any(eligible):
            continue
        tau = values[eligible, step, 0]
        median = np.median(tau)
        scale = 1.4826 * np.median(np.abs(tau - median))
        if scale <= 1e-6:
            scale = max(float(np.std(tau)), 1e-6)
        z[eligible, step] = (tau - median) / scale
    valid_counts = np.sum(np.isfinite(z), axis=1)
    token_z = np.divide(
        np.nansum(z, axis=1),
        valid_counts,
        out=np.full(values.shape[0], np.nan),
        where=valid_counts > 0,
    )
    token_z = token_z[np.isfinite(token_z)]
    if len(token_z) < 8:
        raise ValueError("document has fewer than eight eligible CATS token paths")
    tail_count = max(1, int(np.ceil(0.10 * len(token_z))))
    trim10 = float(np.mean(np.sort(token_z)[-tail_count:]))

    pvalues = np.sort(norm.sf(token_z))
    n = len(pvalues)
    ranks = np.arange(1, n + 1) / n
    hc_mask = (pvalues >= 1.0 / n) & (pvalues <= 0.5)
    if np.any(hc_mask):
        denom = np.sqrt(np.maximum(pvalues * (1.0 - pvalues), 1e-12))
        hc = float(np.max(np.sqrt(n) * (ranks - pvalues) / denom, where=hc_mask, initial=0.0))
    else:
        hc = 0.0

    return {
        "cats_trim10": trim10,
        "cats_hc": hc,
        "cats_tau_mean": float(np.mean(values[:, :, 0], where=distinct)),
        "cats_fr_mean": float(np.mean(values[:, :, 1], where=distinct)),
        "cats_log_increment_mean": float(np.mean(values[:, :, 2], where=distinct)),
        "cats_eligible_tokens": float(len(token_z)),
    }


CATS_V2_FEATURES = (
    "cats_v2_difficulty_mean",
    "cats_v2_difficulty_tail10",
    "cats_v2_hard30_acquisition",
    "cats_v2_late_increment_tail10",
    "cats_v2_transport_median",
)

CATS_V3_FEATURES = (
    "cats_v3_difficulty_mean",
    "cats_v3_penultimate_increment_tail10",
    "cats_v3_final_increment_tail10",
    "cats_v3_final_transport_tail90",
)

CATS_V3_NO_TRANSPORT_FEATURES = CATS_V3_FEATURES[:-1]


def cats_v2_document_features(
    likelihood_array: np.ndarray,
    cats_array: np.ndarray,
    distinct_steps: np.ndarray,
) -> dict[str, float]:
    """Compact target-only CATS-v2 representation.

    The five development-frozen coordinates separate ordinary full-context
    difficulty from two acquisition-path summaries and one vocabulary-wide
    transport summary.  All reductions are label-free and require exactly the
    same nested target distributions as LAR-2.
    """
    likelihood = np.asarray(likelihood_array, dtype=np.float64)
    cats = np.asarray(cats_array, dtype=np.float64)
    distinct = np.asarray(distinct_steps, dtype=bool)
    if (
        likelihood.ndim != 3
        or likelihood.shape[2] != 5
        or cats.shape != (likelihood.shape[0], likelihood.shape[1] - 1, 3)
        or distinct.shape != cats.shape[:2]
        or likelihood.shape[1] < 4
        or likelihood.shape[0] < 8
    ):
        raise ValueError("CATS-v2 arrays are not aligned")
    if np.any(~np.isfinite(likelihood)) or np.any(~np.isfinite(cats)):
        raise ValueError("CATS-v2 arrays must be finite")

    full_logp = likelihood[:, -1, 0]
    full_standardized = likelihood[:, -1, 2]
    standardized_gain = full_standardized - likelihood[:, 0, 2]
    hard_count = max(2, int(np.ceil(0.30 * len(full_logp))))
    hard = np.argsort(full_logp, kind="stable")[:hard_count]

    late = cats[:, -2, 2][distinct[:, -2]]
    if len(late) < 2:
        late = cats[:, -1, 2][distinct[:, -1]]
    transport = cats[:, :, 0][distinct]
    if len(late) < 2 or len(transport) < 8:
        raise ValueError("document has insufficient distinct CATS-v2 steps")

    values = {
        "cats_v2_difficulty_mean": float(np.mean(full_standardized)),
        "cats_v2_difficulty_tail10": float(np.quantile(full_standardized, 0.10)),
        "cats_v2_hard30_acquisition": float(np.mean(standardized_gain[hard])),
        "cats_v2_late_increment_tail10": float(np.quantile(late, 0.10)),
        "cats_v2_transport_median": float(np.median(transport)),
    }
    if np.any(~np.isfinite(list(values.values()))):
        raise ValueError("CATS-v2 features must be finite")
    return values


def cats_v3_document_features(
    likelihood_array: np.ndarray,
    cats_array: np.ndarray,
    distinct_steps: np.ndarray,
) -> dict[str, float]:
    """Development-selected compact CATS score coordinates.

    The last coordinate is the only vocabulary-wide transport observable.  A
    frozen ablation drops it while retaining identical nested target queries.
    """
    likelihood = np.asarray(likelihood_array, dtype=np.float64)
    cats = np.asarray(cats_array, dtype=np.float64)
    distinct = np.asarray(distinct_steps, dtype=bool)
    if (
        likelihood.ndim != 3
        or likelihood.shape[2] != 5
        or cats.shape != (likelihood.shape[0], likelihood.shape[1] - 1, 3)
        or distinct.shape != cats.shape[:2]
        or likelihood.shape[1] < 4
        or likelihood.shape[0] < 8
        or np.any(~np.isfinite(likelihood))
        or np.any(~np.isfinite(cats))
    ):
        raise ValueError("CATS-v3 arrays are invalid or misaligned")
    penultimate = distinct[:, -2]
    final = distinct[:, -1]
    if np.sum(penultimate) < 2 or np.sum(final) < 2:
        raise ValueError("document has insufficient final distinct context steps")
    values = {
        "cats_v3_difficulty_mean": float(np.mean(likelihood[:, -1, 2])),
        "cats_v3_penultimate_increment_tail10": float(
            np.quantile(cats[penultimate, -2, 2], 0.10)
        ),
        "cats_v3_final_increment_tail10": float(np.quantile(cats[final, -1, 2], 0.10)),
        "cats_v3_final_transport_tail90": float(np.quantile(cats[final, -1, 0], 0.90)),
    }
    if np.any(~np.isfinite(list(values.values()))):
        raise ValueError("CATS-v3 features must be finite")
    return values


@dataclass(frozen=True)
class LAR2Config:
    """Published LAR-2 v1 numerical specification plus a fixed RNG seed."""

    grid_size: int = 24
    context_basis_dimension: int = 12
    position_basis_dimension: int = 6
    random_features_per_bandwidth: int = 32
    bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0)
    squared_projections: int = 1024
    seed: int = 260822179


class LAR2Featurizer:
    """Training-only transforms for the published first/second-order LAR map."""

    def __init__(self, config: LAR2Config | None = None) -> None:
        self.config = config or LAR2Config()
        self.channel_mean: np.ndarray | None = None
        self.channel_scale: np.ndarray | None = None
        self.rff_weights: np.ndarray | None = None
        self.rff_phases: np.ndarray | None = None
        self.path_mean: np.ndarray | None = None
        self.path_scale: np.ndarray | None = None
        self.projections: np.ndarray | None = None
        self.document_mean: np.ndarray | None = None
        self.document_scale: np.ndarray | None = None
        self.active: np.ndarray | None = None

    def _validate(self, arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
        out = [np.asarray(array, dtype=np.float64) for array in arrays]
        if not out:
            raise ValueError("at least one likelihood array is required")
        for array in out:
            if (
                array.ndim != 3
                or array.shape[0] < 8
                or array.shape[1:] != (self.config.grid_size, 5)
                or np.any(~np.isfinite(array))
            ):
                raise ValueError("likelihood arrays must have finite shape (tokens, grid, 5)")
        return out

    def _initialize(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        weights = []
        for bandwidth in self.config.bandwidths:
            weights.append(
                rng.normal(
                    scale=1.0 / bandwidth,
                    size=(5, self.config.random_features_per_bandwidth),
                )
            )
        self.rff_weights = np.concatenate(weights, axis=1)
        self.rff_phases = rng.uniform(0, 2 * np.pi, size=self.rff_weights.shape[1])

    def _channel_basis(self, standardized: np.ndarray) -> np.ndarray:
        assert self.rff_weights is not None and self.rff_phases is not None
        projection = standardized @ self.rff_weights + self.rff_phases
        rff = np.sqrt(2.0 / self.rff_weights.shape[1]) * np.cos(projection)
        return np.concatenate((np.ones((*standardized.shape[:-1], 1)), standardized, rff), axis=-1)

    def _components(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.channel_mean is not None and self.channel_scale is not None
        standardized = (array - self.channel_mean) / self.channel_scale
        channel = self._channel_basis(standardized)
        context = linear_hat_basis(np.linspace(0, 1, self.config.grid_size), self.config.context_basis_dimension)
        position = linear_hat_basis(np.linspace(0, 1, array.shape[0]), self.config.position_basis_dimension)
        first = np.einsum("ga,tb,tgc->abc", context, position, channel, optimize=True)
        first = first.reshape(-1) / (array.shape[0] * self.config.grid_size)
        paths = np.einsum("ga,tgc->tac", context, channel, optimize=True)
        paths = paths.reshape(array.shape[0], -1) / self.config.grid_size
        return first, paths, position

    def _raw_coordinates(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        assert self.path_mean is not None and self.path_scale is not None and self.projections is not None
        rows = []
        for array in arrays:
            first, paths, position = self._components(array)
            paths = (paths - self.path_mean) / self.path_scale
            squared = np.square(paths @ self.projections)
            second = np.einsum("tb,tr->br", position, squared, optimize=True)
            second = second.reshape(-1) / (array.shape[0] * np.sqrt(self.config.squared_projections))
            rows.append(np.concatenate((first, second)))
        return np.vstack(rows)

    def fit(self, arrays: Sequence[np.ndarray]) -> "LAR2Featurizer":
        """Fit every learned transform using training arrays only."""
        data = self._validate(arrays)
        cells = np.concatenate([array.reshape(-1, 5) for array in data])
        self.channel_mean = cells.mean(axis=0)
        self.channel_scale = cells.std(axis=0)
        self.channel_scale[self.channel_scale <= 1e-12] = 1.0
        self._initialize()
        components = [self._components(array) for array in data]
        all_paths = np.concatenate([item[1] for item in components])
        self.path_mean = all_paths.mean(axis=0)
        self.path_scale = all_paths.std(axis=0)
        self.path_scale[self.path_scale <= 1e-12] = 1.0
        rng = np.random.default_rng(self.config.seed + 1)
        directions = rng.normal(size=(all_paths.shape[1], self.config.squared_projections))
        directions *= np.sqrt(all_paths.shape[1]) / np.linalg.norm(directions, axis=0, keepdims=True)
        self.projections = directions
        documents = self._raw_coordinates(data)
        self.document_mean = documents.mean(axis=0)
        scale = documents.std(axis=0)
        self.active = scale > 1e-12
        if not np.any(self.active):
            raise ValueError("all LAR-2 document coordinates are constant")
        self.document_scale = scale[self.active]
        return self

    def transform(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Transform held-out arrays with frozen training transforms."""
        if self.document_mean is None or self.document_scale is None or self.active is None:
            raise RuntimeError("fit must be called before transform")
        data = self._validate(arrays)
        raw = self._raw_coordinates(data)
        return (raw[:, self.active] - self.document_mean[self.active]) / self.document_scale

    def fit_transform(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        return self.fit(arrays).transform(arrays)


def error_zone_features(
    likelihood_array: np.ndarray,
    target_correct: Sequence[bool],
    target_reference_delta: Sequence[float],
) -> dict[str, float]:
    """Official EZ-MIA and two context-conditioned extensions.

    The official score is the positive/negative target-reference improvement
    ratio on target error tokens. The extensions weight those same deltas by
    within-document ranks of observed-token context gain or target-directed
    transport. Rank weights are label-free and bounded in (0, 1].
    """
    array = np.asarray(likelihood_array, dtype=np.float64)
    correct = np.asarray(target_correct, dtype=bool)
    delta = np.asarray(target_reference_delta, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 5 or correct.shape != (len(array),) or delta.shape != (len(array),):
        raise ValueError("error-zone inputs are not aligned")
    errors = ~correct
    # Reproduce the official implementation's ``ignore_bos=True`` behavior,
    # which masks the first next-token observation.
    if len(errors):
        errors[0] = False
    if np.sum(errors) < 2:
        return {"ez_mia": 0.0, "ez_context_ratio": 0.0, "ez_transport_ratio": 0.0, "ez_error_tokens": float(np.sum(errors))}

    def ratio(weights: np.ndarray) -> float:
        values = delta[errors]
        weights = weights[errors]
        positive = float(np.sum(weights[values > 0] * values[values > 0]))
        negative = float(np.sum(weights[values < 0] * np.abs(values[values < 0])))
        return positive / (negative + 1e-12) if negative > 0 else 0.0

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        return (order + 1.0) / len(values)

    context_gain = array[:, -1, 0] - array[:, 0, 0]
    standardized_path = array[:, :, 2]
    transport_proxy = np.mean(np.diff(standardized_path, axis=1), axis=1)
    return {
        "ez_mia": ratio(np.ones(len(array))),
        "ez_context_ratio": ratio(ranks(context_gain)),
        "ez_transport_ratio": ratio(ranks(transport_proxy)),
        "ez_error_tokens": float(np.sum(errors)),
    }
