"""Label-free full-distribution measurements. See FEATURE_CATALOGUE.md."""

from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from cats_identification import fisher_rao_alr_from_roots

EPS = 1e-12
STATS = ("mean", "std", "q10", "q50", "q90")


def roots_logmap(x, z):
    c = np.clip(np.sum(x * z, axis=-1), -1.0, 1.0)
    v = z - c[..., None] * x
    s = np.linalg.norm(v, axis=-1)
    angle = np.arctan2(s, c)
    angle = np.where(s <= 1e-15, 0.0, angle)
    return v * np.divide(angle, s, out=np.zeros_like(s), where=s > 1e-15)[..., None]


def distance(x, z):
    return 2 * np.linalg.norm(roots_logmap(x, z), axis=-1)


def entropy_alignment(logp, tangent):
    p = np.exp(logp)
    h = -np.sum(p * logp, axis=-1)
    gradient = -0.5 * np.sqrt(p) * (logp + h[..., None])
    denominator = np.linalg.norm(gradient, axis=-1) * np.linalg.norm(tangent, axis=-1)
    return np.divide(
        np.sum(gradient * tangent, axis=-1),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > EPS,
    )


def top_ids(lp, k=20, tie_keys=None):
    # Stable ties by identity; optional persistent identity keys for permutation checks.
    keys = np.arange(lp.shape[-1]) if tie_keys is None else np.asarray(tie_keys)
    result = []
    for row in lp:
        width = min(k, len(row))
        threshold = np.partition(row, len(row) - width)[len(row) - width]
        candidates = np.flatnonzero(row >= threshold)
        result.append(
            candidates[np.lexsort((keys[candidates], -row[candidates]))[:width]]
        )
    return np.stack(result)


def measure_path(log_probabilities, observed, lengths, *, tie_keys=None):
    """One token, C contexts, V logits/log probabilities; no retained V-sized output."""
    lp = np.asarray(log_probabilities, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=int)
    if lp.ndim != 2 or lp.shape[1] < 2 or not np.isfinite(lp).all():
        raise ValueError("finite interior log probabilities required")
    if (
        lengths.shape != (len(lp),)
        or np.any(lengths < 1)
        or np.any(np.diff(lengths) < 0)
    ):
        raise ValueError("context lengths must be positive and nondecreasing")
    if not 0 <= observed < lp.shape[1]:
        raise ValueError("observed identity outside vocabulary")
    lp = lp - logsumexp(lp, axis=1)[:, None]
    p = np.exp(lp)
    alternatives = np.arange(lp.shape[1]) != observed
    lq = lp[:, alternatives] - logsumexp(lp[:, alternatives], axis=1)[:, None]
    q = np.exp(lq)
    r, s = np.sqrt(p), np.sqrt(q)
    tops = top_ids(lp, tie_keys=tie_keys)
    qkeys = None if tie_keys is None else np.asarray(tie_keys)[alternatives]
    qt = top_ids(lq, tie_keys=qkeys)
    distinct = np.diff(lengths) > 0
    # Repeated contexts are the same query. Refuse inconsistent supplied distributions.
    if np.any(~distinct) and not np.allclose(
        lp[1:][~distinct], lp[:-1][~distinct], atol=1e-7, rtol=0
    ):
        raise ValueError("repeated contexts have different distributions")
    h = -np.sum(p * lp, axis=1)
    variance = np.sum(p * (lp + h[:, None]) ** 2, axis=1)
    node = {
        "likelihood.logp": lp[:, observed],
        "likelihood.zlogp": (lp[:, observed] + h) / np.sqrt(np.maximum(variance, EPS)),
        "likelihood.rank": 1 + np.sum(lp > lp[:, observed, None], axis=1),
        "likelihood.margin": lp[:, observed] - lp[:, alternatives].max(axis=1),
        "position.vertex": 2
        * np.arctan2(np.linalg.norm(r[:, alternatives], axis=1), r[:, observed]),
    }
    for name, prob, logs, top in [("full", p, lp, tops), ("conditional", q, lq, qt)]:
        node[f"concentration.{name}.entropy"] = -np.sum(prob * logs, axis=1)
        node[f"concentration.{name}.squared"] = np.sum(prob**2, axis=1)
        for k in (1, 5, 20):
            node[f"concentration.{name}.mass{k}"] = np.take_along_axis(
                prob, top[:, :k], axis=1
            ).sum(axis=1)
        node[f"check.{name}.peak"] = np.max(prob, axis=1)  # exact duplicate of mass1
    a, length, rr = fisher_rao_alr_from_roots(
        r[:-1], r[1:], np.full(len(lp) - 1, observed)
    )
    length = distance(r[:-1], r[1:])
    rr = a * length
    binary = 2 * np.diff(
        np.arctan2(r[:, observed], np.linalg.norm(r[:, alternatives], axis=1))
    )
    step = {
        "geometry.A": a,
        "geometry.L": length,
        "geometry.R": rr,
        "geometry.B": binary,
        "geometry.G": rr - binary,
        "geometry.N": np.sqrt(np.maximum(length * length - rr * rr, 0)),
        "geometry.D": np.maximum(0, 1 - np.sum(s[:-1] * s[1:], axis=1)),
        "likelihood.increment": np.diff(lp[:, observed]),
    }
    summary = {}
    keep = np.r_[0, np.flatnonzero(distinct) + 1]
    for name, prob, logs, root in [("full", p, lp, r), ("conditional", q, lq, s)]:
        tangent = roots_logmap(root[:-1], root[1:])
        d = 2 * np.linalg.norm(tangent, axis=1)
        if name == "conditional":
            step["geometry.conditional_L"] = d
        step[f"geometry.{name}.entropy_alignment"] = entropy_alignment(
            logs[:-1], tangent
        )
        dl = np.diff(np.log(lengths))
        step[f"trajectory.{name}.rate"] = np.divide(
            d, dl, out=np.full_like(d, np.nan), where=dl > 0
        )
        middle = np.logaddexp(logs[:-1], logs[1:]) - np.log(2)
        step[f"divergence.{name}.js"] = 0.5 * np.sum(
            prob[:-1] * (logs[:-1] - middle) + prob[1:] * (logs[1:] - middle), axis=1
        )
        step[f"divergence.{name}.tv"] = 0.5 * np.abs(prob[1:] - prob[:-1]).sum(axis=1)
        step[f"divergence.{name}.kl_forward"] = np.sum(
            prob[:-1] * (logs[:-1] - logs[1:]), axis=1
        )
        step[f"divergence.{name}.kl_reverse"] = np.sum(
            prob[1:] * (logs[1:] - logs[:-1]), axis=1
        )
        length = float(d[distinct].sum())
        endpoint = float(distance(root[0], root[-1]))
        summary[f"trajectory.{name}.length"] = length
        summary[f"trajectory.{name}.endpoint"] = endpoint
        summary[f"trajectory.{name}.excess"] = max(0.0, length - endpoint)
        cumulative = np.r_[0, np.cumsum(np.where(distinct, d, 0))]
        node[f"trajectory.{name}.cumulative"] = cumulative
        angles = np.full(len(lp), np.nan)
        for left, mid, right in zip(keep[:-2], keep[1:-1], keep[2:]):
            incoming = -roots_logmap(root[mid], root[left])
            outgoing = roots_logmap(root[mid], root[right])
            denom = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
            if denom > EPS:
                angles[mid] = np.arccos(np.clip(incoming @ outgoing / denom, -1, 1))
        node[f"trajectory.{name}.turn"] = angles
        finite = angles[np.isfinite(angles)]
        summary[f"trajectory.{name}.turn_sum"] = (
            float(finite.sum()) if len(finite) else np.nan
        )
    peak = tops[:, 0]
    step["peak.switch"] = (peak[1:] != peak[:-1]).astype(float)
    for k in (1, 5, 20):
        sets = tops[:, :k]
        width = sets.shape[1]
        step[f"peak.overlap{k}"] = np.array(
            [len(set(u) & set(v)) / width for u, v in zip(sets[:-1], sets[1:])]
        )
        step[f"peak.loss{k}"] = np.take_along_axis(
            p[:-1] - p[1:], sets[:-1], axis=1
        ).sum(axis=1)
        member = np.any(sets == observed, axis=1)
        step[f"peak.entry{k}"] = (~member[:-1] & member[1:]).astype(float)
    summary["peak.switch_count"] = float(step["peak.switch"][distinct].sum())
    unique_peaks = peak[keep]
    trailing = 0
    for v in unique_peaks[::-1]:
        if v != unique_peaks[-1]:
            break
        trailing += 1
    summary["peak.final_persistence"] = trailing / len(keep)
    for key in step:
        step[key] = np.where(distinct, step[key], np.nan)
    # Mathematical checks retained as measurements, never attack inputs.
    py = p[:, observed]
    altmass = np.exp(logsumexp(lp[:, alternatives], axis=1))
    hb = -py * lp[:, observed] - altmass * logsumexp(lp[:, alternatives], axis=1)
    node["check.entropy_residual"] = (
        h - hb - altmass * node["concentration.conditional.entropy"]
    )
    return dict(
        node=node,
        step=step,
        summary=summary,
        context_lengths=lengths,
        distinct=distinct,
        observed_id=np.int64(observed),
        top_ids=tops.astype(np.int32),
        top_probabilities=np.take_along_axis(p, tops, axis=1),
    )


def compact_document(paths, positions):
    if not paths or len(paths) != len(positions):
        raise ValueError("empty or unaligned paths")
    out = {"token_positions": np.asarray(positions, dtype=np.int32)}
    for group in ("node", "step", "summary"):
        for key in paths[0][group]:
            values = np.stack([p[group][key] for p in paths])
            out[f"{group}__{key}"] = values
            out[f"valid__{group}__{key}"] = np.isfinite(values)
    for key in (
        "context_lengths",
        "distinct",
        "observed_id",
        "top_ids",
        "top_probabilities",
    ):
        out[key] = np.stack([p[key] for p in paths])
    return out


def aggregate(compact, mean_only=False):
    features = {}
    coverage = {}
    for name, values in compact.items():
        if not name.startswith(("node__", "step__", "summary__")):
            continue
        group, key = name.split("__", 1)
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]
        for j in range(values.shape[1]):
            v = values[:, j]
            v = v[np.isfinite(v)]
            prefix = f"{key}__{group}{j:02d}"
            coverage[prefix] = int(len(v))
            stats = (
                [np.mean(v), np.std(v), *np.quantile(v, [0.1, 0.5, 0.9])]
                if len(v)
                else [np.nan] * 5
            )
            for stat, val in zip(STATS, stats):
                if not mean_only or stat == "mean":
                    features[f"{prefix}__{stat}"] = float(val)
    z = compact["node__likelihood.zlogp"][:, -1]
    lp = compact["node__likelihood.logp"][:, -1]
    inc = compact["step__likelihood.increment"]

    def quantile(v, q):
        v = v[np.isfinite(v)]
        return float(np.quantile(v, q)) if len(v) else np.nan

    features.update(
        negative_loss=float(lp.mean()),
        min_k_plus_plus_20=float(np.sort(z)[: max(1, int(0.2 * len(z)))].mean()),
        cats_difficulty=float(z.mean()),
        cats_penultimate=quantile(inc[:, -2], 0.1),
        cats_final=quantile(inc[:, -1], 0.1),
        cats_alignment=quantile(compact["step__geometry.A"][:, -1], 0.9),
    )
    return features, coverage
