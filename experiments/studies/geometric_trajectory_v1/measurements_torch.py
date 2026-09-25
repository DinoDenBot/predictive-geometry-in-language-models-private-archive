"""Float64 CUDA counterpart of measurements.py, validated against full fixtures.

The NumPy implementation remains the independent reference. This module also
accepts device='cpu' for mathematical tests. No inference precision changes.
"""

import numpy as np
import torch

EPS = 1e-12


def norm(x):
    return torch.linalg.vector_norm(x, dim=-1)


def logmap(x, z):
    c = (x * z).sum(-1).clamp(-1, 1)
    v = z - c[..., None] * x
    s = norm(v)
    angle = torch.where(s <= 1e-15, 0.0, torch.atan2(s, c))
    return v * torch.where(s > 1e-15, angle / s.clamp_min(1e-300), 0.0)[..., None]


def measure_path(log_probabilities, observed, lengths, *, tie_keys=None, device="cuda"):
    lp = torch.as_tensor(log_probabilities, dtype=torch.float64, device=device)
    lengths = np.asarray(lengths, dtype=int)
    if lp.ndim != 2 or lp.shape[1] < 2 or not torch.isfinite(lp).all().item():
        raise ValueError("finite interior log probabilities required")
    if (
        lengths.shape != (len(lp),)
        or np.any(lengths < 1)
        or np.any(np.diff(lengths) < 0)
    ):
        raise ValueError("context lengths must be positive and nondecreasing")
    if not 0 <= observed < lp.shape[1]:
        raise ValueError("observed identity outside vocabulary")
    lp = lp - torch.logsumexp(lp, 1)[:, None]
    distinct = np.diff(lengths) > 0
    dt = torch.as_tensor(distinct, device=device)
    if np.any(~distinct) and not torch.allclose(
        lp[1:][~dt], lp[:-1][~dt], atol=1e-7, rtol=0
    ):
        raise ValueError("repeated contexts have different distributions")
    alt = torch.arange(lp.shape[1], device=device) != observed
    altlp = lp[:, alt]
    altlogmass = torch.logsumexp(altlp, 1)
    lq = altlp - altlogmass[:, None]
    p, q = lp.exp(), lq.exp()
    r, s = p.sqrt(), q.sqrt()
    keys = (
        torch.arange(lp.shape[1], device=device)
        if tie_keys is None
        else torch.as_tensor(tie_keys, device=device)
    )

    def top(logs, identity):
        # Reorder by persistent identity then stable probability sort.
        order = torch.argsort(identity, stable=True)
        ranked = torch.argsort(logs[:, order], dim=1, descending=True, stable=True)[
            :, : min(20, logs.shape[1])
        ]
        return order[ranked]

    tops, qt = top(lp, keys), top(lq, keys[alt])
    h = -(p * lp).sum(1)
    var = (p * (lp + h[:, None]).square()).sum(1)
    node = {
        "likelihood.logp": lp[:, observed],
        "likelihood.zlogp": (lp[:, observed] + h) / var.clamp_min(EPS).sqrt(),
        "likelihood.rank": 1 + (lp > lp[:, observed, None]).sum(1),
        "likelihood.margin": lp[:, observed] - altlp.max(1).values,
        "position.vertex": 2 * torch.atan2(norm(r[:, alt]), r[:, observed]),
    }
    for name, prob, logs, tops_ in [("full", p, lp, tops), ("conditional", q, lq, qt)]:
        node[f"concentration.{name}.entropy"] = -(prob * logs).sum(1)
        node[f"concentration.{name}.squared"] = prob.square().sum(1)
        for k in (1, 5, 20):
            node[f"concentration.{name}.mass{k}"] = prob.gather(1, tops_[:, :k]).sum(1)
        node[f"check.{name}.peak"] = prob.max(1).values
    r0, r1 = r[:-1] / norm(r[:-1])[:, None], r[1:] / norm(r[1:])[:, None]
    overlap = (r0 * r1).sum(1).clamp(-1, 1)
    tangent = r1 - overlap[:, None] * r0
    sine = norm(tangent)
    ascent_norm = (1 - r0[:, observed].square()).clamp_min(0).sqrt()
    ascent = -r0[:, observed, None] * r0
    ascent[:, observed] += 1
    a = (
        (
            (tangent / sine.clamp_min(1e-300)[:, None])
            * (ascent / ascent_norm.clamp_min(1e-300)[:, None])
        )
        .sum(1)
        .clamp(-1, 1)
    )
    a = torch.where((sine > EPS) & (ascent_norm > EPS), a, 0.0)
    ft = logmap(r[:-1], r[1:])
    L = 2 * norm(ft)
    R = a * L
    B = 2 * torch.diff(torch.atan2(r[:, observed], norm(r[:, alt])))
    step = {
        "geometry.A": a,
        "geometry.L": L,
        "geometry.R": R,
        "geometry.B": B,
        "geometry.G": R - B,
        "geometry.N": (L.square() - R.square()).clamp_min(0).sqrt(),
        "geometry.D": (1 - (s[:-1] * s[1:]).sum(1)).clamp_min(0),
        "likelihood.increment": torch.diff(lp[:, observed]),
    }
    summary = {}
    keep = np.r_[0, np.flatnonzero(distinct) + 1]
    dl = torch.as_tensor(np.diff(np.log(lengths)), device=device)
    for name, prob, logs, root in [("full", p, lp, r), ("conditional", q, lq, s)]:
        tangent = ft if name == "full" else logmap(root[:-1], root[1:])
        d = 2 * norm(tangent)
        if name == "conditional":
            step["geometry.conditional_L"] = d
        entropy = -(prob[:-1] * logs[:-1]).sum(1)
        grad = -0.5 * root[:-1] * (logs[:-1] + entropy[:, None])
        denom = norm(grad) * norm(tangent)
        step[f"geometry.{name}.entropy_alignment"] = torch.where(
            denom > EPS, (grad * tangent).sum(1) / denom.clamp_min(1e-300), float("nan")
        )
        step[f"trajectory.{name}.rate"] = torch.where(
            dl > 0, d / dl.clamp_min(1e-300), float("nan")
        )
        middle = torch.logaddexp(logs[:-1], logs[1:]) - np.log(2)
        step[f"divergence.{name}.js"] = 0.5 * (
            prob[:-1] * (logs[:-1] - middle) + prob[1:] * (logs[1:] - middle)
        ).sum(1)
        step[f"divergence.{name}.tv"] = 0.5 * (prob[1:] - prob[:-1]).abs().sum(1)
        step[f"divergence.{name}.kl_forward"] = (
            prob[:-1] * (logs[:-1] - logs[1:])
        ).sum(1)
        step[f"divergence.{name}.kl_reverse"] = (prob[1:] * (logs[1:] - logs[:-1])).sum(
            1
        )
        length = d[dt].sum()
        endpoint = 2 * norm(logmap(root[0], root[-1]))
        summary[f"trajectory.{name}.length"] = length
        summary[f"trajectory.{name}.endpoint"] = endpoint
        summary[f"trajectory.{name}.excess"] = (length - endpoint).clamp_min(0)
        node[f"trajectory.{name}.cumulative"] = torch.cat(
            [d.new_zeros(1), torch.where(dt, d, 0.0).cumsum(0)]
        )
        angles = lp.new_full((len(lp),), float("nan"))
        if len(keep) > 2:
            left, mid, right = [
                torch.as_tensor(a, device=device)
                for a in (keep[:-2], keep[1:-1], keep[2:])
            ]
            incoming, outgoing = (
                -logmap(root[mid], root[left]),
                logmap(root[mid], root[right]),
            )
            den = norm(incoming) * norm(outgoing)
            angles[mid] = torch.where(
                den > EPS,
                torch.acos(
                    ((incoming * outgoing).sum(1) / den.clamp_min(1e-300)).clamp(-1, 1)
                ),
                float("nan"),
            )
        node[f"trajectory.{name}.turn"] = angles
        summary[f"trajectory.{name}.turn_sum"] = torch.where(
            torch.isfinite(angles).any(), torch.nan_to_num(angles).sum(), float("nan")
        )
    peak = tops[:, 0]
    step["peak.switch"] = (peak[1:] != peak[:-1]).double()
    for k in (1, 5, 20):
        sets = tops[:, :k]
        step[f"peak.overlap{k}"] = (
            (sets[:-1, :, None] == sets[1:, None, :]).any(-1).double().mean(-1)
        )
        step[f"peak.loss{k}"] = (p[:-1] - p[1:]).gather(1, sets[:-1]).sum(1)
        member = (sets == observed).any(1)
        step[f"peak.entry{k}"] = (~member[:-1] & member[1:]).double()
    summary["peak.switch_count"] = step["peak.switch"][dt].sum()
    unique_peaks = peak[torch.as_tensor(keep, device=device)]
    summary["peak.final_persistence"] = (
        (unique_peaks.flip(0) == unique_peaks[-1]).long().cumprod(0).double().mean()
    )
    step = {key: torch.where(dt, val, float("nan")) for key, val in step.items()}
    altmass = altlogmass.exp()
    hb = -p[:, observed] * lp[:, observed] - altmass * altlogmass
    node["check.entropy_residual"] = (
        h - hb - altmass * node["concentration.conditional.entropy"]
    )
    # One synchronization/copy for scalar measurements, plus compact identities and probabilities.
    groups = {"node": node, "step": step, "summary": summary}
    flat = (
        torch.cat([v.double().reshape(-1) for g in groups.values() for v in g.values()])
        .cpu()
        .numpy()
    )
    offset = 0
    for g in groups.values():
        for key, val in g.items():
            count = val.numel()
            g[key] = flat[offset : offset + count].reshape(tuple(val.shape))
            offset += count
    return dict(
        **groups,
        context_lengths=lengths,
        distinct=distinct,
        observed_id=np.int64(observed),
        top_ids=tops.cpu().numpy().astype(np.int32),
        top_probabilities=p.gather(1, tops).cpu().numpy(),
    )
