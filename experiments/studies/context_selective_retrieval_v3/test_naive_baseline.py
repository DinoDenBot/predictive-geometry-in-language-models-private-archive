import math

import torch

from context_selective_retrieval_v3.naive_baseline import (
    analytic_matching_step,
    minimum_norm_naive_direction,
)


def test_minimum_norm_naive_direction_has_unit_gain() -> None:
    sensitivity = torch.tensor([2.0, -1.0, 2.0], dtype=torch.float64)
    direction = minimum_norm_naive_direction(sensitivity)
    assert torch.allclose(direction, sensitivity / 9.0)
    assert float(torch.dot(sensitivity, direction)) == 1.0


def test_analytic_matching_step_recovers_target_logit_change() -> None:
    p0, p1 = 0.1, 0.4
    hidden = torch.tensor([2.0, -1.0], dtype=torch.float64)
    direction = torch.tensor([0.4, 0.2], dtype=torch.float64)
    step = analytic_matching_step(p0, p1, hidden, direction)
    delta = step * float(torch.dot(hidden, direction))
    recovered = 1.0 / (1.0 + math.exp(-((math.log(p0) - math.log1p(-p0)) + delta)))
    assert abs(recovered - p1) < 1e-12
