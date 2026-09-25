import numpy as np
import pandas as pd
import pytest
from geometric_trajectory_v1.measurements import (
    measure_path,
    compact_document,
    aggregate,
    entropy_alignment,
    roots_logmap,
)
from geometric_trajectory_v1.attacks import (
    TrainingColumns,
    assign_folds,
    feature_sets,
    fit_outer,
    configurations,
    run_attacks,
)
from geometric_trajectory_v1.reporting import (
    report,
    metrics,
    weighted_batch,
    bootstrap_difference,
)


def path(p, y=0, lengths=None):
    p = np.array(p, dtype=float)
    return measure_path(
        np.log(p), y, np.arange(1, len(p) + 1) if lengths is None else lengths
    )


def test_likelihood_position_and_identities():
    rng = np.random.default_rng(77)
    for _ in range(20):
        p = rng.dirichlet(np.ones(31), size=5)
        r = path(p)
        np.testing.assert_allclose(
            2 * np.log(np.cos(r["node"]["position.vertex"] / 2)),
            r["node"]["likelihood.logp"],
            atol=2e-13,
        )
        np.testing.assert_allclose(r["node"]["check.entropy_residual"], 0, atol=2e-14)
        s = r["step"]
        np.testing.assert_allclose(
            s["geometry.L"] ** 2,
            s["geometry.R"] ** 2 + s["geometry.N"] ** 2,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            s["geometry.R"], s["geometry.B"] + s["geometry.G"], atol=2e-14
        )
        np.testing.assert_allclose(
            s["geometry.R"], s["geometry.A"] * s["geometry.L"], atol=2e-14
        )


def test_identical_uniform_repeated_and_invalid():
    p = np.full((4, 25), 1 / 25)
    r = path(p, lengths=[1, 1, 2, 3])
    assert not r["distinct"][0]
    assert np.isnan(r["step"]["trajectory.full.rate"][0])
    np.testing.assert_allclose(r["step"]["geometry.L"][1:], 0, atol=1e-14)
    assert np.isnan(r["node"]["trajectory.full.turn"]).all()
    assert np.isnan(r["step"]["geometry.full.entropy_alignment"]).all()
    assert np.all(r["node"]["likelihood.rank"] == 1)
    assert np.array_equal(r["top_ids"][0], np.arange(20))
    with pytest.raises(ValueError):
        path([[0.2, 0.8], [0.3, 0.7]], lengths=[1, 1])
    with pytest.raises(ValueError):
        measure_path([[0, float("nan")]], 0, [1])


def test_peak_relocation_vs_broadening_and_loop():
    a = [0.7, 0.2, 0.1]
    b = [0.2, 0.7, 0.1]
    r = path([a, b, a])
    assert r["step"]["peak.switch"].tolist() == [1, 1]
    np.testing.assert_allclose(
        np.diff(r["node"]["concentration.full.entropy"]), 0, atol=1e-14
    )
    assert r["summary"]["trajectory.full.excess"] > 1
    assert r["summary"]["trajectory.full.endpoint"] == 0
    assert r["node"]["trajectory.full.turn"][1] == pytest.approx(np.pi)
    broad = path([a, [0.5, 0.3, 0.2]])
    assert broad["step"]["peak.switch"][0] == 0
    assert (
        broad["node"]["concentration.full.entropy"][1]
        > broad["node"]["concentration.full.entropy"][0]
    )


def test_nearly_deterministic_and_stable_alternatives():
    lp = np.array([[0.0, -80.0, -81.0], [0.0, -85.0, -79.0], [0.0, -84.0, -80.0]])
    r = measure_path(lp, 0, [1, 2, 3])
    assert np.isfinite(r["node"]["concentration.conditional.entropy"]).all()
    assert r["step"]["geometry.conditional_L"][0] > 0
    assert r["summary"]["trajectory.conditional.length"] > 0


def test_fisher_gradient_identities():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.1, 0.7, 0.2])
    lp = np.log(p)
    r = np.sqrt(p)
    h = -sum(p * lp)
    g = -p * (lp + h)
    rootg = g / (2 * r)
    np.testing.assert_allclose(np.sum(g * g / p), np.sum(p * (lp + h) ** 2), atol=1e-15)
    np.testing.assert_allclose(4 * np.sum(rootg**2), np.sum(g * g / p), atol=1e-15)
    t = roots_logmap(r, np.sqrt(q))
    delta = 2 * r * t
    derivative = -sum(delta * (lp + 1))
    np.testing.assert_allclose(derivative, 4 * rootg @ t, atol=1e-15)
    np.testing.assert_allclose(
        entropy_alignment(lp, t),
        rootg @ t / (np.linalg.norm(rootg) * np.linalg.norm(t)),
        atol=1e-15,
    )


def test_permutation_preserves_scalars_and_peak_identities():
    rng = np.random.default_rng(9)
    p = rng.dirichlet(np.ones(30), size=4)
    perm = rng.permutation(30)
    a = measure_path(np.log(p), 3, [1, 2, 4, 8])
    b = measure_path(
        np.log(p[:, perm]), int(np.where(perm == 3)[0][0]), [1, 2, 4, 8], tie_keys=perm
    )
    for group in ("node", "step", "summary"):
        for k in a[group]:
            np.testing.assert_allclose(
                a[group][k], b[group][k], atol=2e-13, equal_nan=True
            )
    np.testing.assert_array_equal(a["top_ids"], perm[b["top_ids"]])


def test_aggregation_masks_do_not_drop_documents():
    r = path(np.full((24, 25), 1 / 25), lengths=np.ones(24, dtype=int))
    c = compact_document([r, r], [1, 2])
    f, coverage = aggregate(c)
    m, _ = aggregate(c, True)
    assert np.isnan(f["trajectory.full.turn__node00__mean"])
    assert coverage["trajectory.full.turn__node00"] == 0
    assert all(k.endswith("__mean") or "__" not in k for k in m)
    assert np.isfinite(f["negative_loss"])
    sets = feature_sets(f)
    assert sets["cats_likelihood"] == [
        "cats_difficulty",
        "cats_penultimate",
        "cats_final",
    ]
    assert not any("check." in c for cols in sets.values() for c in cols)
    assert not any(c.startswith("geometry.") for c in sets["ordinary"])


def panel():
    rng = np.random.default_rng(2)
    rows = []
    for block in range(60):
        for target in ("a", "b"):
            for slot in range(6):
                y = int(slot != 0)
                rows.append(
                    dict(
                        setting="synthetic",
                        phase="validation",
                        target_id=target,
                        doc_id=f"d{block}_{slot}",
                        text_hash=f"h{block}_{slot}",
                        block_id=block,
                        label=y,
                        negative_loss=y + rng.normal(),
                        min_k_plus_plus_20=y + rng.normal(),
                        cats_difficulty=y + rng.normal(),
                        cats_penultimate=rng.normal(),
                        cats_final=rng.normal(),
                        cats_alignment=rng.normal(),
                        **{
                            "likelihood.logp__node23__mean": y + rng.normal(),
                            "geometry.L__step00__mean": rng.normal(),
                            "concentration.full.entropy__node23__mean": rng.normal(),
                        },
                    )
                )
    return assign_folds(pd.DataFrame(rows))


def test_no_document_crosses_split_and_hash_union():
    f = panel()
    for key in ("group", "doc_id", "text_hash", "block_id"):
        assert f.groupby(key).outer_fold.nunique().max() == 1
    raw = f.drop(columns=["group", "outer_fold"])
    raw.loc[raw.block_id == 1, "text_hash"] = "shared"
    raw.loc[raw.block_id == 2, "text_hash"] = "shared"
    merged = assign_folds(raw)
    assert merged[merged.block_id.isin([1, 2])].outer_fold.nunique() == 1


def test_training_only_preprocess_and_baseline_reproduction():
    x = np.array([[1, np.nan, 4, 1], [2, 2, 4, 2], [3, 4, 4, 3]], float)
    t = TrainingColumns().fit(x)
    assert t.columns_ == [0, 1]
    np.testing.assert_array_equal(t.medians_, [2, 3])
    t.transform([[1e10, np.nan, 9, 0]])
    np.testing.assert_array_equal(t.medians_, [2, 3])
    f = panel()
    tr = f[f.outer_fold != 0]
    te = f[f.outer_fold == 0]
    cols = ["cats_difficulty", "cats_penultimate", "cats_final"]
    cfg = [configurations()[0]]
    a, *_ = fit_outer(tr, te, cols, configs=cfg)
    b, *_ = fit_outer(tr, te, cols, baseline_columns=cols, configs=cfg)
    np.testing.assert_array_equal(a, b)


def test_weighted_bootstrap_matches_direct_metrics():
    rng = np.random.default_rng(4)
    y = np.tile([0, 1, 1, 1, 1, 1], 5)
    s = rng.integers(0, 5, len(y))
    inverse = np.repeat(np.arange(5), 6)
    w = rng.integers(1, 4, size=(20, 5))
    got = weighted_batch(y, s, inverse, w)
    for i in range(len(w)):
        np.testing.assert_allclose(got[i], metrics(y, s, w[i, inverse]), atol=1e-15)


def test_saved_models_scores_and_roc_exports(tmp_path):
    f = panel()
    run_attacks(
        f,
        tmp_path / "attacks",
        selected=["rich_likelihood", "ordinary", "combined"],
        configs=[configurations()[0]],
    )
    scores = pd.concat(
        [pd.read_parquet(p) for p in (tmp_path / "attacks").glob("*.parquet")],
        ignore_index=True,
    )
    report(scores, tmp_path / "report", draws=30)
    exported = pd.read_csv(tmp_path / "report" / "metrics.csv")
    for r in exported[exported.scope == "target_fold"].itertuples():
        g = scores[
            (scores.method == r.method)
            & (scores.target_id == r.target_id)
            & (scores.outer_fold == r.outer_fold)
        ]
        np.testing.assert_allclose(
            [r.auc, r.tpr_01, r.tpr_05], metrics(g.label, g.score), atol=1e-15
        )
    # Paired identical attacks produce exactly zero intervals.
    a = scores[scores.method == "rich_likelihood"].copy()
    b = a.copy()
    b["method"] = "copy"
    np.testing.assert_array_equal(
        bootstrap_difference(pd.concat([a, b]), "rich_likelihood", "copy", 20), 0
    )


def test_straight_geodesic_and_conditional_constant():
    angles = np.array([0.2, 0.4, 0.6])
    p = np.stack(
        [np.sin(angles) ** 2, 0.4 * np.cos(angles) ** 2, 0.6 * np.cos(angles) ** 2],
        axis=1,
    )
    measured = path(p)
    np.testing.assert_allclose(measured["step"]["geometry.G"], 0, atol=1e-13)
    np.testing.assert_allclose(
        measured["step"]["geometry.conditional_L"], 0, atol=1e-13
    )
    assert measured["summary"]["trajectory.full.excess"] < 1e-13
    assert measured["node"]["trajectory.full.turn"][1] < 1e-7


def test_all_model_candidates_and_baseline_no_extra_coordinate():
    from geometric_trajectory_v1.attacks import estimator

    configs = configurations()
    assert len(configs) == 13
    for c in configs[5:]:
        model = estimator(c).steps[-1][1]
        assert model.early_stopping is False and model.max_iter == 200
        assert model.learning_rate == 0.05
    f = panel()
    sets = feature_sets(f.columns)
    assert len(sets["cats_full"]) == 4
    assert len(sets["cats_likelihood"]) == 3
    assert not set(sets["cats_full"]) & set(sets["geometry_only"])


def test_optional_all_missing_normalized_hash_is_not_split_failure():
    frame = panel().drop(columns=["group", "outer_fold"])
    frame["normalized_hash"] = np.nan
    result = assign_folds(frame)
    assert len(result) == len(frame)
    assert result.outer_fold.nunique() == 5


def test_model_float32_overrides_pinned_half_config():
    import torch
    from transformers import AutoModelForCausalLM, GPTNeoXConfig

    config = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        torch_dtype="float16",
    )
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    with torch.inference_mode():
        assert model(torch.tensor([[1, 2, 3]])).logits.isfinite().all()


def test_torch_backend_matches_reference_and_permutation():
    import torch
    from geometric_trajectory_v1.measurements_torch import measure_path as tensor_path

    rng = np.random.default_rng(20260909)
    cases = [
        (np.log(np.full((4, 25), 1 / 25)), 0, [1, 1, 2, 3]),
        (np.array([[0.0, -80, -81], [0, -85, -79], [0, -84, -80]]), 0, [1, 2, 3]),
        (np.log([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.7, 0.2, 0.1]]), 0, [1, 2, 3]),
        (rng.normal(size=(24, 83)), 17, np.arange(1, 25)),
        (np.log([[0.2, 0.5, 0.3]]), 1, [1]),
    ]
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for lp, observed, lengths in cases:
            reference = measure_path(lp, observed, lengths)
            candidate = tensor_path(lp, observed, lengths, device=device)
            for group in ("node", "step", "summary"):
                assert set(reference[group]) == set(candidate[group])
                for key in reference[group]:
                    np.testing.assert_allclose(
                        candidate[group][key],
                        reference[group][key],
                        atol=1e-7 if "turn" in key else 1e-8,
                        rtol=1e-9,
                        equal_nan=True,
                        err_msg=key,
                    )
            np.testing.assert_array_equal(candidate["top_ids"], reference["top_ids"])
            np.testing.assert_allclose(
                candidate["top_probabilities"],
                reference["top_probabilities"],
                atol=1e-14,
                rtol=1e-12,
            )
            permutation = rng.permutation(lp.shape[1])
            permuted = tensor_path(
                lp[:, permutation],
                int(np.flatnonzero(permutation == observed)[0]),
                lengths,
                tie_keys=permutation,
                device=device,
            )
            np.testing.assert_array_equal(
                permutation[permuted["top_ids"]], candidate["top_ids"]
            )
            for group in ("node", "step", "summary"):
                for key in candidate[group]:
                    np.testing.assert_allclose(
                        permuted[group][key],
                        candidate[group][key],
                        atol=1e-7 if "turn" in key else 1e-8,
                        rtol=1e-9,
                        equal_nan=True,
                        err_msg=key,
                    )
        with pytest.raises(ValueError):
            tensor_path([[0.0, 1.0], [1.0, 0.0]], 0, [1, 1], device=device)
        with pytest.raises(ValueError):
            tensor_path([[0.0, np.nan]], 0, [1], device=device)
