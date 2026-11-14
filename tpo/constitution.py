"""
Constitutional preference generation (manuscript Section 6).

For a training context, the reference policy pi_ref draws K candidate
trajectories. Each candidate receives a violation score per principle,

    v1  temporal consistency : excess of the largest step-to-step change over
                               kappa * sigma_hist (the first step is measured
                               from the last observation)
    v2  calibration          : |coverage - (1 - alpha)| of the intervals
                               yhat_h +/- z_alpha * sigma_hist * sqrt(h)
    v3  asymmetric risk      : mean of alpha_under [y - yhat]_+ + alpha_over
                               [yhat - y]_+, divided by the context's level sd
    v4  critical periods     : mean shortfall, over steps with c_t = 1, below
                               the conservative level mu_ref + lambda*sigma_hist

and an aggregate s(k) = sum_j w_j v_j(k). The lowest-scoring candidate is the
winner and the highest-scoring the loser. sigma_hist is the standard deviation
of consecutive differences in the context window. Everything is computed in the
standardised target space the models work in. ``y`` is the realised training
target of the window: preferences are only ever built from training data.

Ablation modes
    weights with w_j = 0    : leave principle j out
    mode='none'             : no constitutional principles -- candidates are
                              ranked by plain squared error against the
                              realised target (an accuracy-only preference).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

EPS = 1e-6


def violation_scores(ctx_y: np.ndarray, cand: np.ndarray, y: np.ndarray,
                     crit: np.ndarray, ref_mean: np.ndarray, alpha_under: float,
                     alpha_over: float, kappa: float, z_alpha: float,
                     lam: float) -> np.ndarray:
    """
    Args:
        ctx_y    (B, L)    target history in the context window
        cand     (B, K, H) candidate trajectories
        y        (B, H)    realised targets
        crit     (B, H)    critical-period indicator c_t
        ref_mean (B, H)    reference-policy mean forecast
    Returns:
        (B, K, 4) violation scores v1..v4
    """
    B, K, H = cand.shape
    sig = np.std(np.diff(ctx_y, axis=1), axis=1) + EPS            # (B,)
    lvl = np.std(ctx_y, axis=1) + EPS
    s = sig[:, None, None]

    prev = np.concatenate([np.repeat(ctx_y[:, None, -1:], K, axis=1),
                           cand[:, :, :-1]], axis=2)
    v1 = np.maximum(np.abs(cand - prev).max(axis=2) / s[..., 0] - kappa, 0.0)

    half = z_alpha * s * np.sqrt(np.arange(1, H + 1))[None, None, :]
    cover = (np.abs(y[:, None, :] - cand) <= half).mean(axis=2)
    v2 = np.abs(cover - (1 - 0.05))

    e = y[:, None, :] - cand
    v3 = (alpha_under * np.maximum(e, 0) + alpha_over * np.maximum(-e, 0)).mean(axis=2) \
        / lvl[:, None]

    target = ref_mean[:, None, :] + lam * s
    short = np.maximum(target - cand, 0.0) / s * crit[:, None, :]
    v4 = short.sum(axis=2) / np.maximum(crit.sum(axis=1), 1)[:, None]

    return np.stack([v1, v2, v3, v4], axis=2)


@torch.no_grad()
def generate_pairs(ref_model, ds, origins: np.ndarray, rng: np.random.Generator,
                   num_candidates: int, alpha_under: float, alpha_over: float,
                   cons: dict, weights: Sequence[float] | None = None,
                   mode: str = "principles", batch: int = 512) -> Dict[str, np.ndarray]:
    """
    Build one preference pair per origin. Returns a dict with ``origins``,
    ``win``, ``lose`` (z-units) and the reference policy's ``ref_mean`` and
    ``ref_var`` at each context, which Temporal-DPO needs for log pi_ref.
    """
    w = np.asarray(cons["weights"] if weights is None else weights, dtype=float)
    ref_model.eval()
    out = {k: [] for k in ("win", "lose", "ref_mean", "ref_var")}
    for i in range(0, len(origins), batch):
        o = origins[i:i + batch]
        x = torch.from_numpy(ds.contexts(o))
        pr = ref_model(x)
        mu, var = pr["mean"].numpy(), pr["var"].numpy()
        eps = rng.standard_normal((len(o), num_candidates, mu.shape[1]))
        cand = mu[:, None, :] + np.sqrt(var)[:, None, :] * eps
        y = ds.targets_z(o)
        if mode == "none":
            score = ((cand - y[:, None, :]) ** 2).mean(axis=2)
        else:
            v = violation_scores(ds.contexts(o)[:, :, 0], cand, y, ds.crit_mask(o),
                                 mu, alpha_under, alpha_over, cons["kappa"],
                                 cons["z_alpha"], cons["lambda"])
            score = (v * w[None, None, :]).sum(axis=2)
        b = np.arange(len(o))
        out["win"].append(cand[b, score.argmin(axis=1)])
        out["lose"].append(cand[b, score.argmax(axis=1)])
        out["ref_mean"].append(mu)
        out["ref_var"].append(var)
    res = {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
    res["origins"] = origins
    return res
