"""
Evaluation metrics of the manuscript (Section 7.1, Eqs. 15-16).

    a*(z)   = argmin_{a in A} c(a, z)
    DecAcc  = (1/N) sum_i 1[ a*(yhat_i) = a*(y_i) ]                 (Eq. 15)
    Cost    = (1/N) sum_i c( a*(yhat_i), y_i )                       (Eq. 16)
    c(a, y) = alpha_under * [y - a]_+ + alpha_over * [a - y]_+

N runs over every (test window, horizon step) pair. Costs are charged in
decision units (units sold, kW, returns). In the energy domain alpha_under is
10 during peak hours and 1 otherwise (the "peak-period multiplier").

The action set A is discrete, because Eq. (15) is degenerate on a continuous
action set (a*(z) = z, so the indicator is almost surely 0). For each series A
consists of ``n_actions`` quantiles of that series' TRAINING targets, so A never
depends on test data. ``n_actions`` is not given in the manuscript; it is a
configuration value and the tables report a sensitivity analysis over it.

MSE is computed on the standardised target scale (z-units, training-period
mean and standard deviation of each series), which makes it comparable across
series and domains.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def action_set(train_values: np.ndarray, n_actions: int) -> np.ndarray:
    """Sorted unique quantiles of one series' training targets."""
    v = np.asarray(train_values, dtype=float)
    v = v[np.isfinite(v)]
    q = (np.arange(n_actions) + 0.5) / n_actions
    return np.unique(np.quantile(v, q))


def action_sets(train_values: np.ndarray, n_actions: int) -> List[np.ndarray]:
    return [action_set(row, n_actions) for row in train_values]


def optimal_action(z: np.ndarray, grid: np.ndarray, au, ao) -> np.ndarray:
    """
    a*(z) on a sorted grid. c(., z) is convex and piecewise linear with its
    minimum at a = z, so the minimiser over the grid is one of the two grid
    points bracketing z: the lower one costs au*(z - lo), the upper ao*(hi - z).
    Ties go to the lower point.
    """
    z = np.asarray(z, dtype=float)
    au = np.broadcast_to(np.asarray(au, dtype=float), z.shape)
    ao = np.broadcast_to(np.asarray(ao, dtype=float), z.shape)
    i = np.searchsorted(grid, z)
    lo = grid[np.clip(i - 1, 0, len(grid) - 1)]
    hi = grid[np.clip(i, 0, len(grid) - 1)]
    pick_hi = ao * (hi - z) < au * (z - lo)
    a = np.where(pick_hi, hi, lo)
    a = np.where(z <= grid[0], grid[0], a)
    a = np.where(z >= grid[-1], grid[-1], a)
    return a


def cost(a: np.ndarray, y: np.ndarray, au, ao) -> np.ndarray:
    return au * np.maximum(y - a, 0.0) + ao * np.maximum(a - y, 0.0)


def alpha_matrix(domain_cfg: dict, peak_mask: np.ndarray,
                 alpha_under: float | None = None) -> np.ndarray:
    """Per-element alpha_under: the domain value, or for energy the peak/off-peak
    pair. ``alpha_under`` overrides the domain value (cost-ratio sweep)."""
    au = float(domain_cfg["alpha_under"] if alpha_under is None else alpha_under)
    if "offpeak_alpha_under" in domain_cfg:
        return np.where(peak_mask, au, float(domain_cfg["offpeak_alpha_under"]))
    return np.full(peak_mask.shape, au)


def decision_metrics(pred_dec: np.ndarray, y_dec: np.ndarray, series: np.ndarray,
                     grids: List[np.ndarray], au: np.ndarray, ao: float) -> Dict[str, float]:
    """Eq. (15) and Eq. (16) over all (window, horizon) elements."""
    acc = np.empty(pred_dec.shape)
    c = np.empty(pred_dec.shape)
    for s in np.unique(series):
        m = series == s
        g = grids[s]
        a_hat = optimal_action(pred_dec[m], g, au[m], ao)
        a_star = optimal_action(y_dec[m], g, au[m], ao)
        acc[m] = a_hat == a_star
        c[m] = cost(a_hat, y_dec[m], au[m], ao)
    return {"decision_accuracy": float(acc.mean()), "cost": float(c.mean())}


def evaluate(ds, origins: np.ndarray, pred_z: np.ndarray, domain_cfg: dict,
             n_actions: int, grids_cache: dict | None = None,
             alpha_under: float | None = None) -> Dict[str, float]:
    """All metrics for one method's test forecasts ``pred_z`` (z-units)."""
    y_z = ds.targets_z(origins)
    out = {"mse": float(np.mean((pred_z - y_z) ** 2))}
    key = (ds.name, n_actions)
    if grids_cache is not None and key in grids_cache:
        grids = grids_cache[key]
    else:
        grids = action_sets(ds.train_values_dec(), n_actions)
        if grids_cache is not None:
            grids_cache[key] = grids
    pred_dec = ds.to_decision_units(pred_z, origins)
    y_dec = ds.targets_dec(origins)
    au = alpha_matrix(domain_cfg, ds.peak_mask(origins), alpha_under)
    series = np.repeat(origins[:, 0][:, None], pred_z.shape[1], axis=1)
    out.update(decision_metrics(pred_dec, y_dec, series, grids, au,
                                float(domain_cfg["alpha_over"])))
    return out
