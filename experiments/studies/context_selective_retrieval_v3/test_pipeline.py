from __future__ import annotations

import numpy as np

from context_selective_retrieval_v3.geometry import gate_passes
from context_selective_retrieval_v3.pipeline import _finite_objective, _select_oracle_step


def test_finite_objective_rewards_binary_and_penalizes_other_motion() -> None:
    weights = {"long": 1.0, "clean": 1.0, "short_orthogonal": 0.25, "short_gap": 0.25}
    ideal = {"short_B": .2, "long_L_mean_squared": 0., "clean_L_rms": 0., "short_N": 0., "short_G": 0.}
    collateral = {**ideal, "long_L_mean_squared": .1, "clean_L_rms": .2, "short_N": .3, "short_G": .1}
    assert _finite_objective(ideal, weights) > _finite_objective(collateral, weights)


def test_gate_requires_absolute_gain_selectivity_and_cleanliness() -> None:
    gate = {"short_probability": .05, "selectivity_margin": 0., "clean_rms_max": .05}
    valid = {"short_probability_modified": .06, "selectivity_margin": .01, "clean_L_rms": .04}
    assert gate_passes(valid, gate)
    for key, value in (("short_probability_modified", .04), ("selectivity_margin", -.01), ("clean_L_rms", .06)):
        invalid = dict(valid); invalid[key] = value
        assert not gate_passes(invalid, gate)


def test_probability_and_logit_pullbacks_agree() -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=4)
    p = np.exp(logits - logits.max()); p /= p.sum()
    j_logits = rng.normal(size=(4, 3))
    fisher_logits = np.diag(p) - np.outer(p, p)
    j_probability = fisher_logits @ j_logits
    fisher_probability = np.diag(1.0 / p)
    left = j_logits.T @ fisher_logits @ j_logits
    right = j_probability.T @ fisher_probability @ j_probability
    assert np.allclose(left, right, atol=1e-10)


def test_oracle_selects_smallest_development_passing_step() -> None:
    gate = {"short_probability": .05, "selectivity_margin": 0., "clean_rms_max": .05}
    rows = [
        {
            "linear_binary_step": step,
            "trust_valid": True,
            "development": {
                "short_probability_modified": probability,
                "selectivity_margin": margin,
                "clean_L_rms": clean,
                "objective": objective,
            },
        }
        for step, probability, margin, clean, objective in (
            (.02, .02, .1, .0, .1),
            (.04, .06, .2, .0, .2),
            (.06, .20, .3, .0, .3),
        )
    ]
    selected, passing = _select_oracle_step(rows, gate)
    assert len(passing) == 2
    assert selected["linear_binary_step"] == .04
