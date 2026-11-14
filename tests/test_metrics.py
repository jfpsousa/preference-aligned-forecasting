import numpy as np

from tpo.metrics import action_set, cost, decision_metrics, optimal_action


def brute(z, grid, au, ao):
    return np.array([grid[np.argmin(cost(grid, zi, a, ao))] for zi, a in zip(z, au)])


def test_optimal_action_matches_brute_force():
    rng = np.random.default_rng(0)
    grid = np.sort(rng.normal(size=30))
    z = rng.normal(scale=2, size=2000)
    for au in (1.0, 3.0, 10.0):
        aus = np.full_like(z, au)
        assert np.array_equal(optimal_action(z, grid, aus, 1.0), brute(z, grid, aus, 1.0))


def test_asymmetry_biases_action_upward():
    grid = np.array([0.0, 1.0])
    # z = 0.4 is nearer 0, but under-provisioning costs 3x: 3*0.4 > 1*0.6
    assert optimal_action(np.array([0.4]), grid, 3.0, 1.0)[0] == 1.0
    assert optimal_action(np.array([0.4]), grid, 1.0, 1.0)[0] == 0.0


def test_perfect_forecast_scores_one():
    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, size=(50, 7))
    grids = [action_set(rng.gamma(2.0, size=500), 21)]
    series = np.zeros_like(y, dtype=int)
    m = decision_metrics(y, y, series, grids, np.full(y.shape, 3.0), 1.0)
    assert m["decision_accuracy"] == 1.0


def test_action_set_uses_quantiles_only():
    g = action_set(np.arange(100.0), 5)
    assert len(g) == 5 and g[0] > 0 and g[-1] < 99
