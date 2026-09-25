"""Saved-score ROC exports; target/fold means primary, pooled curves secondary."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from geometric_trajectory_v1.attacks import SEED


def metrics(y, s, weights=None):
    f, t, _ = roc_curve(y, s, sample_weight=weights, drop_intermediate=False)
    return np.array(
        [
            roc_auc_score(y, s, sample_weight=weights),
            max(t[f <= 0.01 + 1e-12]),
            max(t[f <= 0.05 + 1e-12]),
        ]
    )


def weighted_batch(y, s, inverse, weights):
    order = np.argsort(-s, kind="stable")
    y = np.asarray(y)[order]
    s = np.asarray(s)[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    w = weights[:, inverse[order]]
    tp = np.c_[np.zeros(len(w)), np.cumsum(w * y, axis=1)[:, ends]]
    fp = np.c_[np.zeros(len(w)), np.cumsum(w * (1 - y), axis=1)[:, ends]]
    positives = tp[:, -1]
    negatives = fp[:, -1]
    if np.any(positives == 0) | np.any(negatives == 0):
        raise ValueError("bootstrap stratum lost a class")
    auc = np.sum(0.5 * (tp[:, 1:] + tp[:, :-1]) * np.diff(fp, axis=1), axis=1) / (
        positives * negatives
    )
    return np.column_stack(
        [
            auc,
            *[
                np.max(np.where(fp <= a * negatives[:, None] + 1e-10, tp, 0), axis=1)
                / positives
                for a in (0.01, 0.05)
            ],
        ]
    )


def bootstrap_difference(scores, left, right, draws=10000):
    a = (
        scores[scores.method == left]
        .sort_values(["target_id", "doc_id"])
        .reset_index(drop=True)
    )
    b = (
        scores[scores.method == right]
        .sort_values(["target_id", "doc_id"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        a.drop(columns=["method", "score"]), b.drop(columns=["method", "score"])
    )
    group_names = sorted(a.group.unique())
    lookup = {g: i for i, g in enumerate(group_names)}
    inv = a.group.map(lookup).to_numpy()
    rng = np.random.default_rng(SEED)
    deltas = []
    # Stratify resampling by frozen outer fold: every fitted attack retains its test stratum.
    strata = [
        np.array([lookup[g] for g in sub.group.unique()])
        for _, sub in a.groupby("outer_fold")
    ]
    for start in range(0, draws, 200):
        n = min(200, draws - start)
        w = np.zeros((n, len(group_names)), dtype=int)
        for ids in strata:
            w[:, ids] = rng.multinomial(
                len(ids), np.full(len(ids), 1 / len(ids)), size=n
            )
        differences = []
        for _, sub in a.groupby(["target_id", "outer_fold"]):
            i = sub.index.to_numpy()
            differences.append(
                weighted_batch(a.label.to_numpy()[i], a.score.to_numpy()[i], inv[i], w)
                - weighted_batch(
                    b.label.to_numpy()[i], b.score.to_numpy()[i], inv[i], w
                )
            )
        deltas.append(np.mean(differences, axis=0))
    samples = np.concatenate(deltas)
    return samples


def report(scores, output, draws=10000):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    required = ["setting", "target_id", "doc_id", "method"]
    if scores.duplicated(required).any() or not np.isfinite(scores.score).all():
        raise ValueError("invalid saved scores")
    identities = None
    for _, g in scores.groupby(["setting", "method"]):
        ids = set(zip(g.target_id, g.doc_id))
        setting = g.setting.iloc[0]
        if identities is None:
            identities = {}
        if setting in identities and ids != identities[setting]:
            raise ValueError("unequal method coverage")
        identities[setting] = ids
    rows = []
    curves = []
    averages = []
    intervals = []
    grid = np.unique(np.r_[np.linspace(0, 1, 1001), 0.01, 0.05])
    for setting, panel in scores.groupby("setting"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for method, m in panel.groupby("method"):
            rocs = []
            points = []
            for (target, fold), g in m.groupby(["target_id", "outer_fold"]):
                y = g.label.to_numpy()
                s = g.score.to_numpy()
                v = metrics(y, s)
                points.append(v)
                phase = ",".join(sorted(g.phase.unique()))
                rows.append(
                    dict(
                        setting=setting,
                        method=method,
                        scope="target_fold",
                        target_id=target,
                        outer_fold=fold,
                        phase=phase,
                        auc=v[0],
                        tpr_01=v[1],
                        tpr_05=v[2],
                        n=len(g),
                        nonmembers=int((y == 0).sum()),
                    )
                )
                f, t, _ = roc_curve(y, s, drop_intermediate=False)
                # Empirical step curve; no interpolated threshold claim at 1%/5%.
                rocs.append(t[np.searchsorted(f, grid, side="right") - 1])
                curves.extend(
                    dict(
                        setting=setting,
                        method=method,
                        scope="target_fold",
                        target_id=target,
                        outer_fold=fold,
                        fpr=x,
                        tpr=z,
                    )
                    for x, z in zip(f, t)
                )
            avg = np.mean(rocs, axis=0)
            v = np.mean(points, axis=0)
            rows.append(
                dict(
                    setting=setting,
                    method=method,
                    scope="target_fold_mean",
                    auc=v[0],
                    tpr_01=v[1],
                    tpr_05=v[2],
                )
            )
            averages.extend(
                dict(setting=setting, method=method, fpr=x, tpr=z)
                for x, z in zip(grid, avg)
            )
            for ax in axes:
                ax.plot(grid, avg, label=f"{method} ({v[0]:.3f})", lw=1)
            # Original phases and each target retain fold-averaged main summaries.
            for keys in (["phase"], ["target_id"], ["phase", "target_id"]):
                for label, g in m.groupby(keys):
                    labels = label if isinstance(label, tuple) else (label,)
                    pp = [
                        metrics(x.label, x.score)
                        for _, x in g.groupby(["target_id", "outer_fold"])
                    ]
                    vv = np.mean(pp, axis=0)
                    rows.append(
                        dict(
                            setting=setting,
                            method=method,
                            scope="_".join(keys) + "_fold_mean",
                            **dict(zip(keys, labels)),
                            auc=vv[0],
                            tpr_01=vv[1],
                            tpr_05=vv[2],
                            n=len(g),
                        )
                    )
            v = metrics(m.label, m.score)
            f, t, _ = roc_curve(m.label, m.score, drop_intermediate=False)
            rows.append(
                dict(
                    setting=setting,
                    method=method,
                    scope="pooled_secondary",
                    auc=v[0],
                    tpr_01=v[1],
                    tpr_05=v[2],
                )
            )
            curves.extend(
                dict(
                    setting=setting,
                    method=method,
                    scope="pooled_secondary",
                    fpr=x,
                    tpr=z,
                )
                for x, z in zip(f, t)
            )
        axes[0].set(
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            title=setting,
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[1].set(
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            title="Low-FPR operating points",
            xlim=(0, 0.06),
            ylim=(0, 1),
        )
        axes[0].legend(fontsize=5)
        fig.tight_layout()
        fig.savefig(output / f"{setting}_roc.pdf")
        plt.close(fig)
        for baseline in ("rich_likelihood", "ordinary"):
            if {"combined", baseline} <= set(panel.method):
                samples = bootstrap_difference(panel, "combined", baseline, draws)
                np.save(
                    output / f"{setting}_combined_minus_{baseline}_bootstrap.npy",
                    samples,
                )
                for j, name in enumerate(("auc", "tpr_01", "tpr_05")):
                    low, high = np.quantile(samples[:, j], [0.025, 0.975])
                    a = [
                        r
                        for r in rows
                        if r["setting"] == setting
                        and r["scope"] == "target_fold_mean"
                        and r["method"] == "combined"
                    ][0]
                    b = [
                        r
                        for r in rows
                        if r["setting"] == setting
                        and r["scope"] == "target_fold_mean"
                        and r["method"] == baseline
                    ][0]
                    intervals.append(
                        dict(
                            setting=setting,
                            comparison="combined-minus-" + baseline,
                            metric=name,
                            difference=a[name] - b[name],
                            lower=low,
                            upper=high,
                            draws=draws,
                        )
                    )
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "roc.csv", index=False)
    pd.DataFrame(averages).to_csv(output / "mean_roc.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "paired_intervals.csv", index=False)
    (output / "interpretation.json").write_text(
        json.dumps(
            dict(
                status="retrospective",
                primary="unweighted target/fold mean AUC and empirical ROC operating points",
                intervals="paired complete-document-group bootstrap, stratified by outer fold, conditional on fitted attacks and existing targets",
                scope="held-out documents from existing target population; controlled continued-training inclusion 1[K>0]",
                threshold_caveat="1% and 5% are empirical ROC points, not deployment-calibrated thresholds",
                bootstrap_draws=draws,
            ),
            indent=2,
        )
        + "\n"
    )

    assessment = [
        "# Retrospective evidence assessment",
        "",
        "This evaluates controlled continued-training inclusion on held-out documents from the existing target population.",
        "No unseen-model or original-pretraining membership claim follows.",
        "",
    ]
    for row in intervals:
        if row["metric"] != "auc":
            continue
        verdict = (
            "positive conditional evidence"
            if row["lower"] > 0
            else "negative conditional evidence"
            if row["upper"] < 0
            else "inconclusive"
        )
        assessment.append(
            f"- {row['setting']}, {row['comparison']}: AUC difference {row['difference']:.4f}, 95% interval [{row['lower']:.4f}, {row['upper']:.4f}]; {verdict}."
        )
    assessment += [
        "",
        "Intervals condition on fitted attacks and existing targets and are not multiplicity-adjusted.",
        "Family and mean-only ablations appear in metrics.csv for every setting, including unfavorable results.",
        "Ablation point differences are exploratory; an apparent increase alone does not establish a family benefit.",
        "Low-FPR values are empirical operating points rather than calibrated thresholds. Inspect nonmember counts per target/fold.",
    ]
    (output / "evidence_assessment.md").write_text("\n".join(assessment) + "\n")
