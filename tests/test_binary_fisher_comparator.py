import numpy as np

from reviewer_revision.run_binary_fisher_comparator import (
    DESIGN_TARGET_IDS,
    signed_binary_fisher,
)


def full_simplex_r(p: np.ndarray, q: np.ndarray, y: int) -> float:
    roots_p = np.sqrt(p)
    roots_q = np.sqrt(q)
    c = float(roots_p @ roots_q)
    theta = float(np.arccos(np.clip(c, -1.0, 1.0)))
    direction = (roots_q - c * roots_p) / np.sqrt(1.0 - c * c)
    ascent = np.zeros_like(p)
    ascent[y] = 1.0
    ascent = (ascent - roots_p[y] * roots_p) / np.sqrt(1.0 - p[y])
    return float(2.0 * theta * (direction @ ascent))


def finite_gap_integral(p: np.ndarray, q: np.ndarray, y: int) -> float:
    """Evaluate the exact pathwise gap formula by Gauss--Legendre quadrature."""
    roots_p = np.sqrt(p)
    roots_q = np.sqrt(q)
    c = float(roots_p @ roots_q)
    theta = float(np.arccos(np.clip(c, -1.0, 1.0)))
    length = 2.0 * theta
    direction = (roots_q - c * roots_p) / np.sqrt(1.0 - c * c)
    nodes, weights = np.polynomial.legendre.leggauss(96)
    progress = 0.5 * length * (nodes + 1.0)
    root_y = (
        np.cos(progress / 2.0) * roots_p[y]
        + np.sin(progress / 2.0) * direction[y]
    )
    root_y_prime = 0.5 * (
        -np.sin(progress / 2.0) * roots_p[y]
        + np.cos(progress / 2.0) * direction[y]
    )
    probability_y = np.square(root_y)
    eta_prime = 2.0 * root_y_prime / np.sqrt(1.0 - probability_y)
    integrand = (
        0.5
        * (length - progress)
        * np.sqrt(probability_y / (1.0 - probability_y))
        * (1.0 - np.square(eta_prime))
    )
    return float(0.5 * length * (weights @ integrand))


def complement_affinity(p: np.ndarray, q: np.ndarray, y: int) -> float:
    mask = np.arange(len(p)) != y
    conditional_p = p[mask] / (1.0 - p[y])
    conditional_q = q[mask] / (1.0 - q[y])
    return float(np.sqrt(conditional_p) @ np.sqrt(conditional_q))


def test_randomization_target_rosters_match_complete_training_designs() -> None:
    assert DESIGN_TARGET_IDS["original70"] == (
        "dev_1",
        "dev_2",
        "val_1",
        "val_2",
        "val_3",
        "con_1",
        "con_2",
        "con_3",
    )
    assert DESIGN_TARGET_IDS["fresh70"] == tuple(
        f"70m_{index}" for index in range(1, 7)
    )


def test_signed_binary_fisher_matches_binary_simplex_displacement() -> None:
    p_y = np.asarray([0.2])
    q_y = np.asarray([0.35])
    expected = 2.0 * (np.arcsin(np.sqrt(q_y)) - np.arcsin(np.sqrt(p_y)))
    np.testing.assert_allclose(signed_binary_fisher(p_y, q_y), expected, atol=1e-15)


def test_signed_binary_fisher_zero_when_realized_probability_is_fixed() -> None:
    p_y = np.asarray([0.2, 0.7])
    np.testing.assert_array_equal(signed_binary_fisher(p_y, p_y), np.zeros(2))


def test_signed_binary_fisher_orientation() -> None:
    p_y = np.asarray([0.2, 0.8])
    q_y = np.asarray([0.4, 0.6])
    result = signed_binary_fisher(p_y, q_y)
    assert result[0] > 0
    assert result[1] < 0


def test_binary_coarse_graining_gap_vanishes_on_binary_simplex() -> None:
    p = np.asarray([0.2, 0.8])
    q = np.asarray([0.35, 0.65])
    b = float(signed_binary_fisher(p[[0]], q[[0]])[0])
    np.testing.assert_allclose(full_simplex_r(p, q, 0), b, atol=1e-14)


def test_full_distribution_gap_can_remain_when_endpoint_probability_is_fixed() -> None:
    p = np.asarray([0.2, 0.3, 0.5])
    q = np.asarray([0.2, 0.6, 0.2])
    b = float(signed_binary_fisher(p[[0]], q[[0]])[0])
    r = full_simplex_r(p, q, 0)
    np.testing.assert_allclose(b, 0.0, atol=1e-15)
    np.testing.assert_allclose(r, 0.0607, atol=5e-5)


def test_full_distribution_gap_is_second_order_locally() -> None:
    p = np.asarray([0.2, 0.3, 0.5])
    w = np.asarray([0.3, 0.4, -0.7])
    y = 0

    def gap(epsilon: float) -> float:
        q = p + epsilon * w
        b = float(signed_binary_fisher(p[[y]], q[[y]])[0])
        return full_simplex_r(p, q, y) - b

    coarse = gap(0.05)
    fine = gap(0.025)
    assert 0.2 < fine / coarse < 0.3

    fisher_norm_squared = float(np.sum(np.square(w) / p))
    realized_coordinate = float(w[y] / np.sqrt(p[y] * (1.0 - p[y])))
    perpendicular_norm_squared = fisher_norm_squared - realized_coordinate**2
    expected_coefficient = (
        0.25
        * np.sqrt(p[y] / (1.0 - p[y]))
        * perpendicular_norm_squared
    )
    epsilon = 0.00625
    np.testing.assert_allclose(
        gap(epsilon) / epsilon**2,
        expected_coefficient,
        rtol=0.005,
    )


def test_finite_gap_is_nonnegative_and_equals_path_integral() -> None:
    rng = np.random.default_rng(20260906)
    for _ in range(100):
        vocabulary = int(rng.integers(3, 20))
        p = rng.dirichlet(np.full(vocabulary, 0.7))
        q = rng.dirichlet(np.full(vocabulary, 0.7))
        y = int(rng.integers(vocabulary))
        b = float(signed_binary_fisher(p[[y]], q[[y]])[0])
        gap = full_simplex_r(p, q, y) - b
        assert gap >= -2e-14
        np.testing.assert_allclose(gap, finite_gap_integral(p, q, y), atol=2e-14)


def test_endpoints_and_complement_affinity_determine_r() -> None:
    rng = np.random.default_rng(20260907)
    for _ in range(100):
        vocabulary = int(rng.integers(3, 20))
        p = rng.dirichlet(np.full(vocabulary, 0.7))
        q = rng.dirichlet(np.full(vocabulary, 0.7))
        y = int(rng.integers(vocabulary))
        a, b = float(p[y]), float(q[y])
        h = complement_affinity(p, q, y)
        c = np.sqrt(a * b) + np.sqrt((1.0 - a) * (1.0 - b)) * h
        theta = float(np.arccos(np.clip(c, -1.0, 1.0)))
        reconstructed = (
            2.0
            * theta
            / np.sqrt(1.0 - c * c)
            * (np.sqrt(b) - c * np.sqrt(a))
            / np.sqrt(1.0 - a)
        )
        np.testing.assert_allclose(reconstructed, full_simplex_r(p, q, y), atol=2e-14)
