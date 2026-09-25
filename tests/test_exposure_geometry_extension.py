"""Verification for the prospective final exposure-geometry extension."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from exposure_geometry_extension import (
    DOSES,
    ExtensionAccess,
    TransitionGeometry,
    apply_residual_calibration,
    binary_coarse_graining_energy,
    complete_r_gate_power_simulation,
    dual_duplicate_audit,
    fisher_inner,
    fisher_log_tangent,
    fit_development_residual,
    frozen_document_summary,
    hierarchical_claim_decisions,
    intervention_fingerprint,
    make_six_target_assignments,
    perpendicular_tangent,
    realized_unit_gradient,
    select_extension_block_count,
    three_way_decision,
    token_weighted_effective_batch_step,
    transition_geometry,
    validate_paired_intervention,
    validate_architecture_stochastic_seeds,
)


def _manifest() -> pd.DataFrame:
    rows = []
    for block in range(600):
        role = "validation" if block < 300 else "confirmation"
        for slot in range(6):
            rows.append(
                {
                    "block_id": block,
                    "doc_id": f"doc-{block:03d}-{slot}",
                    "doc_slot": slot,
                    "role": role,
                    "source": "org",
                    "topic": block // 10,
                    "tokens": 80 + slot,
                    "baseline_difficulty": 2.0 + slot / 100,
                }
            )
    return pd.DataFrame(rows)


def _random_transition(seed: int = 4) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(9))
    q = rng.dirichlet(np.ones(9))
    return p, q, 3


def test_exact_perpendicular_decomposition_and_realized_coordinate_localization() -> None:
    p, q, y = _random_transition()
    geometry = transition_geometry(p, q, y)
    v = fisher_log_tangent(p, q)
    unit = realized_unit_gradient(p, y)
    perpendicular = perpendicular_tangent(p, q, y)
    assert geometry.L**2 == pytest.approx(geometry.R**2 + geometry.N**2, abs=1e-12)
    assert fisher_inner(p, perpendicular, unit) == pytest.approx(0.0, abs=1e-12)
    assert perpendicular[y] == pytest.approx(0.0, abs=1e-12)
    assert np.delete(perpendicular, y).sum() == pytest.approx(0.0, abs=1e-12)
    assert fisher_inner(p, v, v) == pytest.approx(geometry.L**2, abs=1e-12)
    assert fisher_inner(p, perpendicular, perpendicular) == pytest.approx(geometry.N**2, abs=1e-12)


def test_binary_coarse_graining_energy_equals_R_squared() -> None:
    p, q, y = _random_transition(5)
    v = fisher_log_tangent(p, q)
    geometry = transition_geometry(p, q, y)
    assert binary_coarse_graining_energy(p, v, y) == pytest.approx(geometry.R**2, abs=1e-12)
    assert geometry.E_y == pytest.approx(geometry.A**2, abs=1e-12)


def test_local_expansion_has_quadratic_remainder() -> None:
    p = np.asarray([0.18, 0.27, 0.31, 0.24])
    tangent = np.asarray([0.07, -0.03, -0.02, -0.02])
    y = 0
    errors = []
    epsilons = [2e-3, 1e-3, 5e-4]
    coefficient = tangent[y] / math.sqrt(p[y] * (1.0 - p[y]))
    for epsilon in epsilons:
        q = p + epsilon * tangent
        errors.append(abs(transition_geometry(p, q, y).R - epsilon * coefficient))
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.04)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.04)


def test_numerical_conventions_coincident_tiny_angle_and_near_boundary() -> None:
    p = np.asarray([0.2, 0.3, 0.5])
    coincident = transition_geometry(p, p, 1)
    assert coincident.A == coincident.L == coincident.R == coincident.N == coincident.E_y == 0.0
    tiny = transition_geometry(p, p + 1e-10 * np.asarray([1.0, -0.5, -0.5]), 0)
    assert tiny.L > 0.0 and np.isfinite(np.asarray(list(tiny.__dict__.values()))).all()
    boundary = transition_geometry(
        np.asarray([1e-14, 0.4, 0.6 - 1e-14]),
        np.asarray([2e-14, 0.4, 0.6 - 2e-14]),
        0,
    )
    assert np.isfinite(np.asarray(list(boundary.__dict__.values()))).all()
    with pytest.raises(ValueError, match="interior"):
        transition_geometry([0.0, 1.0], [0.1, 0.9], 0)


def test_scalar_comparator_denominators_match_production_floor() -> None:
    low_p = np.asarray([1e-14, 1.0 - 1e-14])
    low_q = np.asarray([2e-14, 1.0 - 2e-14])
    low = transition_geometry(low_p, low_q, 0)
    assert low.D_z == pytest.approx(
        (low_q[0] - low_p[0]) / math.sqrt(low_p[0] * (1.0 - low_p[0]))
    )

    high_p = np.asarray([1.0 - 1e-14, 1e-14])
    high_q = np.asarray([1.0 - 2e-14, 2e-14])
    high = transition_geometry(high_p, high_q, 0)
    assert high.D_sqrt == pytest.approx(
        2.0 * (math.sqrt(high_q[0]) - math.sqrt(high_p[0]))
        / math.sqrt(1.0 - high_p[0])
    )


def test_below_tolerance_displacement_preserves_returned_pythagorean_identity() -> None:
    p = np.asarray([0.2, 0.3, 0.5])
    q = p + 1e-13 * np.asarray([1.0, -0.5, -0.5])
    geometry = transition_geometry(p, q, 0)
    assert geometry.L > 0.0
    assert geometry.A == geometry.R == geometry.E_y == 0.0
    assert geometry.N == geometry.L
    assert geometry.L**2 == pytest.approx(geometry.R**2 + geometry.N**2, abs=1e-30)


def test_numerically_degenerate_ascent_matches_production_convention() -> None:
    p = np.asarray([1.0, 1e-30])
    q = np.asarray([1.0 - 1e-14, 1e-14])
    geometry = transition_geometry(p, q, 0)
    assert geometry.L > 0.0
    assert geometry.A == geometry.R == geometry.E_y == 0.0
    assert geometry.N == geometry.L
    assert np.isfinite(np.asarray(list(geometry.__dict__.values()))).all()
    with pytest.raises(ValueError, match="ascent"):
        realized_unit_gradient(p, 0)


def test_all_transition_values_are_computed_before_separate_quantiles() -> None:
    rows = [
        TransitionGeometry(0.0, 10.0, 0.0, 10.0, 0.0, 0, 0, 0, 0),
        TransitionGeometry(1.0, 1.0, 1.0, 0.0, 1.0, 1, 1, 1, 1),
    ]
    summary = frozen_document_summary(rows)
    assert summary["S_N"] == pytest.approx(9.0)
    assert summary["S_E_y"] == pytest.approx(0.9)
    assert summary["S_N"] != pytest.approx(
        math.sqrt(max(summary["S_L"] ** 2 - summary["S_R"] ** 2, 0.0))
    )
    assert summary["S_E_y"] != pytest.approx(summary["S_A"] ** 2)


def test_dual_duplicate_rules_and_strict_threshold_audit() -> None:
    base = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    candidates = pd.DataFrame(
        {
            "doc_id": ["word", "char", "clean"],
            "text": [
                base,
                "abcdefghijklmnopqrstuvwx 1234567890",
                "rain falls softly across a distant copper mountain in winter",
            ],
        }
    )
    references = pd.DataFrame(
        {
            "doc_id": ["word-ref", "char-ref"],
            "text": [
                base + " nu",
                "abcdefghijklmnopqrstuvwx 1234567899",
            ],
        }
    )
    detail, audit = dual_duplicate_audit(candidates, references)
    indexed = detail.set_index("doc_id")
    assert indexed.loc["word", "word5_jaccard_max"] >= 0.80
    assert indexed.loc["char", "char5_tfidf_cosine_max"] >= 0.80
    assert indexed.loc["clean", "passes"]
    assert audit["threshold"] == 0.80
    assert audit["comparison"] == "strictly less than threshold required"


def test_six_target_rotation_total_exposure_and_deterministic_replay() -> None:
    first = make_six_target_assignments(_manifest(), seed=20260901)
    second = make_six_target_assignments(_manifest(), seed=20260901)
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby(["block_id", "target_id"]).K.sum().eq(31).all()
    rotations = first.groupby(["block_id", "doc_id"]).K.apply(lambda values: set(values))
    assert rotations.map(lambda values: values == set(DOSES)).all()


def test_extension_power_uses_six_target_mechanism_and_selects_smallest_powered_size() -> None:
    null = complete_r_gate_power_simulation(
        blocks=20,
        simulations=100,
        randomization_draws=199,
        seed=17,
        effect=0.0,
        target_slope_sd=0.0,
        noise_sd=1.0,
    )
    assert null["full_design_targets"] == 6
    assert null["phase_targets"] == 3
    assert null["compound_power"] <= 0.10
    assert select_extension_block_count(
        [
            {"blocks": 300, "compound_power": 0.99},
            {"blocks": 200, "compound_power": 0.89},
            {"blocks": 250, "compound_power": 0.91},
        ]
    ) == 250


def test_three_plus_three_access_and_complementary_cells_are_sealed(tmp_path) -> None:
    access = ExtensionAccess(tmp_path)
    with pytest.raises(PermissionError):
        access.require_openable("70m", "validation", "70m_1")
    access.frozen_spec.write_text("{}")
    assert access.require_openable("70m", "validation", "70m_1") == "primary_validation"
    with pytest.raises(PermissionError):
        access.require_openable("70m", "confirmation", "70m_4")
    with pytest.raises(PermissionError):
        access.require_openable("70m", "validation", "70m_4")
    access.decision("70m_validation").parent.mkdir(parents=True)
    access.decision("70m_validation").write_text(json.dumps({"decision_complete": True}))
    assert access.require_openable("70m", "confirmation", "70m_4") == "primary_confirmation"
    with pytest.raises(PermissionError):
        access.require_openable("160m", "validation", "160m_1")
    access.decision("70m_confirmation").write_text(json.dumps({"decision_complete": True}))
    assert access.require_openable("70m", "validation", "70m_4") == "complete_latin_post_confirmation_sensitivity"
    assert access.require_openable("70m", "confirmation", "70m_1") == "complete_latin_post_confirmation_sensitivity"
    assert access.require_openable("160m", "validation", "160m_1") == "160m_validation"
    with pytest.raises(PermissionError):
        access.require_openable("160m", "confirmation", "160m_1")
    access.decision("160m_validation_power").write_text(json.dumps({"decision_complete": True}))
    assert access.require_openable("160m", "confirmation", "160m_1") == "160m_confirmation"


def test_residual_fit_replay_and_three_way_decision() -> None:
    rows = []
    rng = np.random.default_rng(9)
    for target in range(2):
        for block in range(20):
            for slot, dose in enumerate(DOSES):
                dz = math.log2(dose + 1) + rng.normal(0, 0.05)
                rows.append(
                    {
                        "target_id": f"dev-{target}",
                        "block_id": block,
                        "delta_S_D_z": dz,
                        "delta_S_R": 1.7 * dz + target + block / 10 + rng.normal(0, 0.03),
                    }
                )
    frame = pd.DataFrame(rows)
    first = fit_development_residual(frame)
    second = fit_development_residual(frame)
    assert first == second
    assert first.gamma == pytest.approx(1.7, abs=0.01)
    held = apply_residual_calibration(frame.iloc[:12], first)
    assert np.isfinite(held.U).all()
    assert three_way_decision(
        validation_positive_gate=True,
        confirmation_positive_gate=True,
        validation_equivalence_interval=(-1, 1),
        confirmation_equivalence_interval=(-1, 1),
        validation_equivalence_powered=False,
        confirmation_equivalence_powered=False,
    ) == "beyond_local_scalar_response"
    assert three_way_decision(
        validation_positive_gate=False,
        confirmation_positive_gate=False,
        validation_equivalence_interval=(-0.04, 0.03),
        confirmation_equivalence_interval=(-0.02, 0.05),
        validation_equivalence_powered=True,
        confirmation_equivalence_powered=True,
    ) == "no_meaningful_beyond_local_scalar_response"


def test_hierarchical_gate_cannot_be_rescued_by_160m() -> None:
    failed = hierarchical_claim_decisions(
        r70_validation=False,
        r70_confirmation=True,
        a70_validation=True,
        a70_confirmation=True,
        l70_validation_equivalent=True,
        l70_confirmation_equivalent=True,
        n70_decision="powered_equivalence",
        u_decision="beyond_local_scalar_response",
        r160_validation=True,
        r160_confirmation=True,
    )
    assert not failed["R70_replicated"]
    assert not failed["R160_replicated"]
    assert failed["R160_observed_without_rescue_authority"]
    passed = hierarchical_claim_decisions(
        r70_validation=True,
        r70_confirmation=True,
        a70_validation=True,
        a70_confirmation=True,
        l70_validation_equivalent=True,
        l70_confirmation_equivalent=True,
        n70_decision="powered_equivalence",
        u_decision="no_meaningful_beyond_local_scalar_response",
        r160_validation=True,
        r160_confirmation=True,
    )
    assert passed["realized_coordinate_localization"]
    assert passed["reorientation_without_meaningful_total_motion_increase"]
    assert passed["R160_replicated"]


def test_localization_does_not_require_L_equivalence_or_A_response() -> None:
    decisions = hierarchical_claim_decisions(
        r70_validation=True,
        r70_confirmation=True,
        a70_validation=False,
        a70_confirmation=False,
        l70_validation_equivalent=False,
        l70_confirmation_equivalent=False,
        n70_decision="powered_equivalence",
        u_decision="inconclusive",
        r160_validation=None,
        r160_confirmation=None,
    )
    assert decisions["realized_coordinate_localization"]
    assert not decisions["reorientation_without_meaningful_total_motion_increase"]


def test_paired_architectures_require_independent_stochastic_seeds() -> None:
    validate_architecture_stochastic_seeds()
    seeds = {
        "70m_1": 1,
        "70m_2": 2,
        "70m_3": 3,
        "70m_4": 4,
        "70m_5": 5,
        "70m_6": 6,
        "160m_1": 4,
        "160m_2": 8,
        "160m_3": 9,
    }
    with pytest.raises(ValueError, match="unique stochastic seeds"):
        validate_architecture_stochastic_seeds(seeds)


def _ledger(target: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "block_id": [1, 1],
            "doc_id": ["a", "b"],
            "doc_slot": [0, 1],
            "role": ["validation", "validation"],
            "K": [1, 2],
            "d": [1.0, math.log2(3)],
            "occurrence": [0, 0],
            "optimizer_step": [0, 0],
            "batch_position": [0, 1],
            "presentation_id": [0, 1],
            "target_id": [target, target],
        }
    )


def test_paired_intervention_fingerprint_ignores_architecture_identity_only() -> None:
    left, right = _ledger("70m_4"), _ledger("160m_1")
    assert validate_paired_intervention(left, right) == intervention_fingerprint(left)
    right.loc[1, "batch_position"] = 0
    with pytest.raises(ValueError, match="differ"):
        validate_paired_intervention(left, right)


def test_token_weighted_accumulation_clips_and_steps_once(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    clip_calls = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def counted_clip(parameters, max_norm):
        clip_calls.append(max_norm)
        return original_clip(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", counted_clip)
    losses = [(parameter.square(), 1), (3.0 * parameter.square(), 3)]
    value = token_weighted_effective_batch_step(
        losses, [parameter], optimizer, max_grad_norm=100.0
    )
    # Weighted loss = (1/4)*4 + (3/4)*12 = 10; gradient = 10.
    assert value == pytest.approx(10.0)
    assert parameter.item() == pytest.approx(1.0)
    assert clip_calls == [100.0]
