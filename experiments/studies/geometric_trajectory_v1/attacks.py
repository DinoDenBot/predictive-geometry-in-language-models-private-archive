"""Grouped nested cross-fitting, explicit feature sets and training-only transforms."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 20260909
CATS = ["cats_difficulty", "cats_penultimate", "cats_final", "cats_alignment"]


def document_groups(frame):
    """Union complete blocks and every exact/normalized hash across targets/phases."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        a, b = find(x), find(y)
        parent[max(a, b)] = min(a, b)

    blocks = frame.setting.astype(str) + ":block:" + frame.block_id.astype(str)
    for i, (_, row) in enumerate(frame.iterrows()):
        for key in ("text_hash", "normalized_hash", "doc_id"):
            if key in frame and pd.notna(row[key]):
                union(blocks.iloc[i], key + ":" + str(row[key]))
    return np.array([find(b) for b in blocks])


def assign_folds(frame, n=5):
    frame = frame.copy()
    frame["group"] = document_groups(frame)
    frame["outer_fold"] = -1
    for setting, part in frame.groupby("setting", sort=True):
        splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
        for fold, (_, test) in enumerate(splitter.split(part, part.label, part.group)):
            frame.loc[part.index[test], "outer_fold"] = fold
        for key in ("group", "doc_id", "text_hash", "normalized_hash"):
            if (
                key in frame
                and frame.loc[part.index].groupby(key).outer_fold.nunique().gt(1).any()
            ):
                raise AssertionError(f"{key} crosses split in {setting}")
    return frame


class TrainingColumns(BaseEstimator, TransformerMixin):
    """Remove empty/constant/exact duplicate columns; median impute from fit rows only."""

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=float)
        self.n_features_in_ = x.shape[1]
        self.columns_ = []
        medians = []
        seen = set()
        for j in range(x.shape[1]):
            col = x[:, j]
            valid = col[np.isfinite(col)]
            if not len(valid):
                continue
            median = float(np.median(valid))
            filled = np.where(np.isfinite(col), col, median)
            if np.ptp(filled) == 0:
                continue
            digest = hashlib.sha256(filled.tobytes()).digest()
            if digest in seen:
                continue
            seen.add(digest)
            self.columns_.append(j)
            medians.append(median)
        self.medians_ = np.asarray(medians)
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=float)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("changed feature schema")
        if not self.columns_:
            return np.zeros((len(x), 1))
        x = x[:, self.columns_]
        return np.where(np.isfinite(x), x, self.medians_)


def feature_sets(columns):
    # The catalogue's check.* redundant transforms and all metadata are excluded.
    families = {
        f: sorted(c for c in columns if c.startswith(f + "."))
        for f in (
            "likelihood",
            "position",
            "concentration",
            "peak",
            "geometry",
            "trajectory",
            "divergence",
        )
    }
    likelihood = families["likelihood"]
    ordinary = sum([families[k] for k in ("concentration", "peak", "divergence")], [])
    geometric = sum([families[k] for k in ("geometry", "trajectory")], [])
    all_features = likelihood + ordinary + geometric + families["position"]
    sets = {
        "cats_likelihood": CATS[:3],
        "cats_full": CATS,
        "rich_likelihood": likelihood,
        "ordinary": likelihood + ordinary,
        "fisher": likelihood + geometric + families["position"],
        "combined": all_features,
        "geometry_only": geometric,
        "geometry_position": geometric + families["position"],
    }
    # Five conceptual family ablations; divergence is an additional comparison family.
    for family in (
        "likelihood",
        "concentration",
        "peak",
        "geometry",
        "trajectory",
        "divergence",
    ):
        removed = families[family] + (
            families["position"] if family == "likelihood" else []
        )
        sets["without_" + family] = [c for c in all_features if c not in removed]
    sets["mean_only"] = [c for c in all_features if c.endswith("__mean")]
    return sets


def configurations():
    # Deterministic simplicity order also resolves AUC ties (absolute tolerance 1e-12).
    result = [dict(kind="logistic", C=c) for c in (0.001, 0.01, 0.1, 1.0, 10.0)]
    result += [
        dict(
            kind="boosting",
            max_leaf_nodes=leaves,
            min_samples_leaf=leaf,
            l2_regularization=l2,
        )
        for leaves in (7, 15)
        for leaf in (50, 20)
        for l2 in (1.0, 0.0)
    ]
    return result


def estimator(config):
    args = {k: v for k, v in config.items() if k != "kind"}
    model = (
        LogisticRegression(**args, max_iter=5000, random_state=SEED)
        if config["kind"] == "logistic"
        else HistGradientBoostingClassifier(
            **args,
            learning_rate=0.05,
            max_iter=200,
            early_stopping=False,
            random_state=SEED,
        )
    )
    return make_pipeline(TrainingColumns(), StandardScaler(), model)


def inner_auc(frame, scores):
    values = []
    for _, g in frame.assign(_score=scores).groupby("target_id"):
        if g.label.nunique() != 2:
            raise ValueError("inner target missing a label")
        values.append(roc_auc_score(g.label, g._score))
    return float(np.mean(values))


def fit_outer(train, test, columns, *, baseline_columns=None, configs=None):
    choices = [columns] if baseline_columns is None else [baseline_columns, columns]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    splits = list(splitter.split(train, train.label, train.group))
    best = None
    search = []
    # Configuration precedes subset, so ties prefer logistic, simpler config, then baseline.
    for config in configurations() if configs is None else configs:
        for cols in choices:
            aucs = []
            for tr, va in splits:
                a, b = train.iloc[tr], train.iloc[va]
                if set(a.group) & set(b.group):
                    raise AssertionError("inner leakage")
                model = estimator(config).fit(a[cols], a.label)
                aucs.append(inner_auc(b, model.predict_proba(b[cols])[:, 1]))
            entry = dict(
                config=config, columns=cols, auc=float(np.mean(aucs)), inner_aucs=aucs
            )
            search.append(entry)
            if best is None or entry["auc"] > best["auc"] + 1e-12:
                best = entry
    model = estimator(best["config"]).fit(train[best["columns"]], train.label)
    return model.predict_proba(test[best["columns"]])[:, 1], model, best, search


def run_attacks(frame, output, *, selected=None, configs=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if frame.duplicated(["setting", "target_id", "doc_id"]).any():
        raise ValueError("duplicate cells")
    sets = feature_sets(frame.columns)
    if selected is not None:
        sets = {k: sets[k] for k in selected}
    with threadpool_limits(limits=4):
        for setting, part in frame.groupby("setting", sort=True):
            for fold in sorted(part.outer_fold.unique()):
                train = part[part.outer_fold != fold]
                test = part[part.outer_fold == fold]
                if set(train.group) & set(test.group):
                    raise AssertionError("outer leakage")
                for name, cols in sets.items():
                    stem = output / f"{setting}__fold{fold}__{name}"
                    receipt = stem.with_suffix(".json")
                    data_digest = hashlib.sha256(
                        pd.util.hash_pandas_object(part, index=True).values.tobytes()
                    ).hexdigest()
                    if receipt.exists():
                        saved = json.loads(receipt.read_text())
                        if saved["data_digest"] != data_digest:
                            raise ValueError("resume input changed")
                        for suffix, h in saved["artifacts"].items():
                            if (
                                hashlib.sha256(
                                    stem.with_suffix(suffix).read_bytes()
                                ).hexdigest()
                                != h
                            ):
                                raise ValueError("resume artifact changed")
                        continue
                    baseline = (
                        sets.get("rich_likelihood") if name == "combined" else None
                    )
                    if name == "combined" and baseline is None:
                        baseline = feature_sets(frame.columns)["rich_likelihood"]
                    scores, model, best, search = fit_outer(
                        train, test, cols, baseline_columns=baseline, configs=configs
                    )
                    result = test[
                        [
                            "setting",
                            "phase",
                            "target_id",
                            "doc_id",
                            "text_hash",
                            "block_id",
                            "group",
                            "outer_fold",
                            "label",
                        ]
                    ].copy()
                    result["method"] = name
                    result["score"] = scores
                    result.to_parquet(stem.with_suffix(".parquet"), index=False)
                    joblib.dump(model, stem.with_suffix(".joblib"))
                    restored = joblib.load(stem.with_suffix(".joblib"))
                    np.testing.assert_array_equal(
                        scores, restored.predict_proba(test[best["columns"]])[:, 1]
                    )
                    receipt.write_text(
                        json.dumps(
                            dict(
                                best=best,
                                search=search,
                                data_digest=data_digest,
                                train_groups=sorted(train.group.unique()),
                                test_groups=sorted(test.group.unique()),
                                artifacts={
                                    s: hashlib.sha256(
                                        stem.with_suffix(s).read_bytes()
                                    ).hexdigest()
                                    for s in (".parquet", ".joblib")
                                },
                            ),
                            indent=2,
                        )
                        + "\n"
                    )
                    print(setting, fold, name, round(best["auc"], 5), flush=True)
            for name in ("negative_loss", "min_k_plus_plus_20"):
                result = part[
                    [
                        "setting",
                        "phase",
                        "target_id",
                        "doc_id",
                        "text_hash",
                        "block_id",
                        "group",
                        "outer_fold",
                        "label",
                    ]
                ].copy()
                result["method"] = name
                result["score"] = part[name].to_numpy()
                result.to_parquet(output / f"{setting}__{name}.parquet", index=False)
