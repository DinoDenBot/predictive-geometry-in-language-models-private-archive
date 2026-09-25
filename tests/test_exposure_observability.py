"""Design and gate tests for the exposure--observability allocation study."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from exposure_observability import (
    DOSES,
    PhaseAccess,
    _constant_effect_interval_from_statistics,
    block_fixed_effect_slope,
    design_based_interval,
    deterministic_disruption,
    document_target_fixed_effect_sensitivity,
    donor_coverage_preflight,
    grouped_auc_interval,
    latin_cube,
    make_presentation_ledger,
    power_gate_simulation,
    presentation_balance,
    randomization_test,
    target_slopes,
    validate_latin_assignments,
    validation_gate,
)
from run_exposure_observability import _last_hidden_and_head, validate_power_config


def _assignment_frame(blocks: int = 5, targets: int = 8, seed: int = 9) -> pd.DataFrame:
    cube = latin_cube(blocks, targets, np.random.default_rng(seed))
    rows = []
    for block in range(blocks):
        for target in range(targets):
            for slot in range(6):
                dose = int(cube[block, target, slot])
                rows.append(
                    {
                        "block_id": block,
                        "latin_block_position": block,
                        "doc_slot": slot,
                        "doc_id": f"b{block}-s{slot}",
                        "role": "development",
                        "target_id": f"t{target}",
                        "K": dose,
                        "d": np.log2(dose + 1),
                    }
                )
    return pd.DataFrame(rows)


def test_latin_balance_rotation_and_total_exposure() -> None:
    first = _assignment_frame()
    second = _assignment_frame()
    pd.testing.assert_frame_equal(first, second)
    validate_latin_assignments(first, [f"t{x}" for x in range(8)])
    grouped = first.groupby(["block_id", "target_id"])
    assert grouped.K.sum().eq(31).all()
    assert grouped.K.apply(lambda values: sorted(values) == DOSES.tolist()).all()
    for target in first.target_id.unique():
        counts = first.loc[first.target_id == target].K.value_counts()
        assert counts.nunique() == 1
    per_document = first.groupby("doc_id").K.apply(tuple)
    assert all(len(set(values)) >= 6 for values in per_document)


def test_presentation_ledger_is_uniform_unique_and_replayable() -> None:
    assignments = _assignment_frame(blocks=8)
    first = make_presentation_ledger(assignments, "t0", seed=17, batch_size=8)
    second = make_presentation_ledger(assignments, "t0", seed=17, batch_size=8)
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("optimizer_step").doc_id.apply(lambda values: values.is_unique).all()
    counts = first.groupby("doc_id").size()
    expected = assignments.loc[(assignments.target_id == "t0") & (assignments.K > 0)].set_index("doc_id").K
    pd.testing.assert_series_equal(counts.sort_index(), expected.sort_index(), check_names=False)
    spread = first.groupby("K").normalized_progress.mean()
    assert np.max(np.abs(spread - 0.5)) < 0.08
    steps = int(first.optimizer_step.max()) + 1
    rates = 1e-5 * np.minimum(1.0, (np.arange(steps) + 1) / 100)
    balance = presentation_balance(first, rates)
    assert np.ptp(balance.weighted_learning_rate) < 2e-7


def test_block_fixed_effect_and_cross_target_sensitivity_recover_slope() -> None:
    frame = _assignment_frame(blocks=30, targets=3)
    block_term = frame.block_id.map({block: block / 10 for block in frame.block_id.unique()})
    target_term = frame.target_id.map({"t0": -2.0, "t1": 0.5, "t2": 4.0})
    frame["y"] = block_term + target_term + 0.37 * frame.d
    assert abs(block_fixed_effect_slope(frame.loc[frame.target_id == "t0"], "y") - 0.37) < 1e-12
    assert all(abs(value - 0.37) < 1e-12 for value in target_slopes(frame, "y").values())
    assert abs(document_target_fixed_effect_sensitivity(frame, "y") - 0.37) < 1e-12


def test_randomization_test_is_directional_and_plus_one() -> None:
    frame = _assignment_frame(blocks=18, targets=3)
    rng = np.random.default_rng(88)
    frame["positive"] = 1.2 * frame.d + rng.normal(0, 0.15, len(frame))
    result = randomization_test(frame, "positive", draws=199, seed=99)
    assert result.estimate > 1.0
    assert result.p_one_sided == (result.greater_or_equal + 1) / 200
    assert result.p_one_sided <= 0.025
    assert validation_gate(target_slopes(frame, "positive"), result.p_one_sided, (0.5, 1.5))


def test_full_design_subset_interval_recovers_constant_effect() -> None:
    full = _assignment_frame(blocks=24, targets=8, seed=44)
    frame = full.loc[full.target_id.isin(("t2", "t3", "t4"))].copy()
    rng = np.random.default_rng(45)
    frame["y"] = 0.37 * frame.d + rng.normal(0.0, 0.02, len(frame))
    interval = design_based_interval(
        frame,
        "y",
        draws=399,
        seed=46,
        all_target_ids=[f"t{index}" for index in range(8)],
    )
    estimate = float(np.mean(list(target_slopes(frame, "y").values())))
    assert interval[0] <= estimate <= interval[1]
    assert abs(estimate - 0.37) < 0.01


def test_randomization_null_calibration_is_not_systematically_small() -> None:
    frame = _assignment_frame(blocks=12, targets=3)
    rng = np.random.default_rng(123)
    pvalues = []
    for repeat in range(12):
        frame["null"] = rng.normal(size=len(frame))
        pvalues.append(randomization_test(frame, "null", draws=99, seed=1000 + repeat).p_one_sided)
    assert np.mean(np.asarray(pvalues) < 0.10) <= 0.25


def test_constant_effect_interval_uses_exact_crossing_order_statistics() -> None:
    rng = np.random.default_rng(2026)
    estimate = 0.31
    u = rng.normal(0.0, 0.2, 199)
    v = rng.uniform(-0.90, 0.90, 199)
    lower, upper = _constant_effect_interval_from_statistics(
        estimate, u, v, confidence=0.95
    )
    alpha = 0.05

    def accepted(effect: float) -> bool:
        count = np.count_nonzero(
            np.abs(u - effect * v) >= abs(estimate - effect) - 1e-15
        )
        return bool((count + 1) / 200 > alpha)

    assert lower <= estimate <= upper
    assert accepted(lower) and accepted(upper)
    assert not accepted(lower - 1e-9)
    assert not accepted(upper + 1e-9)


def test_constant_effect_interval_handles_unit_slope_draws() -> None:
    estimate = 0.2
    u = np.r_[np.full(10, estimate), np.full(10, -estimate), np.zeros(179)]
    v = np.r_[np.ones(10), -np.ones(10), np.zeros(179)]
    lower, upper = _constant_effect_interval_from_statistics(
        estimate, u, v, confidence=0.95
    )
    assert lower == -np.inf and upper == np.inf


def test_batched_compound_power_is_calibrated_under_the_null() -> None:
    result = power_gate_simulation(
        blocks=100,
        simulations=500,
        randomization_draws=1_999,
        seed=0,
        effect=0.0,
        seed_sd=0.0,
        noise_sd=1.0,
    )
    assert result["full_design_targets"] == 8
    assert result["compound_power"] <= 0.05


def test_grouped_auc_interval_resamples_complete_blocks() -> None:
    labels = np.tile([0, 1], 20)
    scores = labels + np.linspace(0.0, 0.01, len(labels))
    groups = np.repeat(np.arange(20), 2)
    first = grouped_auc_interval(labels, scores, groups, draws=199, seed=8)
    second = grouped_auc_interval(labels, scores, groups, draws=199, seed=8)
    assert first == second == (1.0, 1.0)


def test_production_power_config_rejects_a_nonnull_null_scenario() -> None:
    common = {
        "simulations": 500,
        "randomization_draws": 1_999,
        "effect": 0.05,
        "seed_sd": 0.0,
        "noise_sd": 1.0,
        "timing_sd": 0.0,
        "quadratic": 0.0,
    }
    config = {
        "scenarios": [
            {"name": "null", "requires_power": False, "seed": 1, **common},
            {"name": "plausible", "requires_power": True, "seed": 2, **common},
            {
                "name": "seed_heterogeneity",
                "requires_power": True,
                "seed": 3,
                **common,
                "seed_sd": 0.02,
            },
            {
                "name": "timing_variation",
                "requires_power": True,
                "seed": 4,
                **common,
                "timing_sd": 0.02,
            },
            {
                "name": "nonlinear",
                "requires_power": True,
                "seed": 5,
                **common,
                "quadratic": 0.01,
            },
        ],
        "length_equivalence": {
            "simulations": 500,
            "randomization_draws": 1_999,
            "seed": 6,
            "noise_sd": 1.0,
            "true_slope": 0.0,
            "margin": 0.05,
        },
    }
    with pytest.raises(ValueError, match="null scenario"):
        validate_power_config(config)
    config["scenarios"][0]["effect"] = 0.0
    validate_power_config(config)


def test_pythia_embed_out_compatibility_adapter() -> None:
    class Core:
        def __call__(self, *, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            return type("Output", (), {"last_hidden_state": input_ids[..., None].float()})

    class Model:
        gpt_neox = Core()
        embed_out = torch.nn.Identity()

    tokens = torch.tensor([[1, 2, 3]])
    hidden, head = _last_hidden_and_head(Model(), tokens, torch.ones_like(tokens))
    assert hidden.shape == (1, 3, 1)
    assert head is Model.embed_out


def test_llama_model_lm_head_compatibility_adapter() -> None:
    class Core:
        def __call__(self, *, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            return type("Output", (), {"last_hidden_state": input_ids[..., None].float()})

    class Model:
        model = Core()
        lm_head = torch.nn.Identity()

    tokens = torch.tensor([[1, 2, 3]])
    hidden, head = _last_hidden_and_head(Model(), tokens, torch.ones_like(tokens))
    assert hidden.shape == (1, 3, 1)
    assert head is Model.lm_head


def test_phase_access_prevents_early_held_reads(tmp_path) -> None:
    access = PhaseAccess(tmp_path)
    access.require_openable("development")
    with pytest.raises(PermissionError):
        access.require_openable("validation")
    access.frozen_spec.parent.mkdir(parents=True, exist_ok=True)
    access.frozen_spec.write_text("{}")
    access.require_openable("validation")
    with pytest.raises(PermissionError):
        access.require_openable("confirmation")
    access.validation_decision.parent.mkdir(parents=True)
    access.validation_decision.write_text(json.dumps({"gate_passed": False}))
    access.require_openable("confirmation")


def test_donor_preflight_keeps_fixed_matching_criteria() -> None:
    candidates = pd.DataFrame(
        [
            {"doc_id": f"c{doc}", "token_id": token, "topic": 1, "token_position": token, "context_length": 8}
            for doc in range(10)
            for token in range(4)
        ]
    )
    donors = pd.DataFrame(
        [
            {"doc_id": f"d{doc:03d}", "token_id": token, "topic": 1, "token_position": token, "context_length": 8}
            for doc in range(40)
            for token in range(4)
        ]
    )
    result = donor_coverage_preflight(candidates, donors, increments=10, maximum=40)
    assert result["passed"]
    assert result["donor_documents_opened"] == 40


def test_local_disruption_is_deterministic_length_preserving_and_measured() -> None:
    tokens = list(range(17))
    first, severity = deterministic_disruption(tokens, document_id="doc-1")
    second, repeated_severity = deterministic_disruption(tokens, document_id="doc-1")
    assert first == second
    assert len(first) == len(tokens)
    assert sorted(first) == tokens
    assert severity == repeated_severity
    assert severity > 0.80
