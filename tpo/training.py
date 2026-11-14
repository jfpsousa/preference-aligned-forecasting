"""
Training procedures.

Supervised objectives (all with identical budget: epochs x steps_per_epoch
mini-batches of ``batch_size`` windows drawn uniformly from the training
windows, AdamW):

    mse         squared error on the mean head. The variance head is trained
                jointly by Gaussian NLL with the mean DETACHED, so the mean is
                optimised by MSE alone while the model still provides the
                predictive distribution pi_ref needs. This model is the MSE
                baseline and the Temporal-DPO reference policy.
    asymmetric  fixed-ratio pinball loss alpha_under*[y-yhat]_+ +
                alpha_over*[yhat-y]_+ with the domain's headline ratio.
    oracle      the exact evaluation cost structure, step by step (for energy:
                10x under-forecast penalty in peak hours, 1x otherwise).
                Not given the future -- given the true utility.

The losses are applied in the standardised target space. The transforms
involved are increasing (affine; log1p for M5), and conditional quantiles are
preserved under increasing transforms, so the loss minimiser maps to the
decision-unit minimiser.

Temporal-DPO (Algorithm 1, Eq. 17):
    r(yhat) = beta * sum_h gamma^(h-1) [log pi_theta(yhat_h|H) - log pi_ref(yhat_h|H)]
    L       = -log sigmoid(r(y_w) - r(y_l))
pi is the Gaussian given by the mean and variance heads. pi_theta is initialised
from pi_ref. Dropout is disabled during fine-tuning so that pi_theta = pi_ref
exactly at initialisation.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _batch(ds, origins, rng, n):
    o = origins[rng.integers(0, len(origins), n)]
    return o, torch.from_numpy(ds.contexts(o)), torch.from_numpy(ds.targets_z(o)).float()


def train_supervised(model, ds, objective: str, cfg: dict, rng: np.random.Generator,
                     domain_cfg: dict, alpha_under: Optional[float] = None,
                     origins: Optional[np.ndarray] = None) -> Dict[str, list]:
    """Train ``model`` in place. ``origins`` restricts the training windows
    (sample-efficiency ablation); default is every training window."""
    sup = cfg["supervised"]
    origins = ds.train_origins() if origins is None else origins
    au_flat = float(domain_cfg["alpha_under"] if alpha_under is None else alpha_under)
    ao = float(domain_cfg["alpha_over"])
    opt = torch.optim.AdamW(model.parameters(), lr=sup["learning_rate"],
                            weight_decay=sup["weight_decay"])
    model.train()
    curve = []
    for _ in range(sup["epochs"]):
        tot = 0.0
        for _ in range(sup["steps_per_epoch"]):
            o, x, y = _batch(ds, origins, rng, sup["batch_size"])
            out = model(x)
            mu = out["mean"]
            if objective == "mse":
                loss = F.mse_loss(mu, y)
                var = out["var"]
                loss = loss + 0.5 * (torch.log(var) + (y - mu.detach()) ** 2 / var).mean()
            elif objective in ("asymmetric", "oracle"):
                if objective == "oracle" and "offpeak_alpha_under" in domain_cfg:
                    pk = torch.from_numpy(ds.peak_mask(o))
                    au = torch.where(pk, torch.tensor(au_flat),
                                     torch.tensor(float(domain_cfg["offpeak_alpha_under"])))
                else:
                    au = au_flat
                e = y - mu
                loss = (au * torch.clamp(e, min=0) + ao * torch.clamp(-e, min=0)).mean()
            else:
                raise ValueError(objective)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        curve.append(tot / sup["steps_per_epoch"])
    model.eval()
    return {"loss": curve}


@torch.no_grad()
def predict(model, ds, origins: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(ds.contexts(origins[i:i + batch])))["mean"].numpy()
           for i in range(0, len(origins), batch)]
    return np.concatenate(out)


def naive_forecast(ds, origins: np.ndarray) -> np.ndarray:
    """Persistence: the last observed value repeated over the horizon."""
    last = ds.X[origins[:, 0], origins[:, 1] - 1, 0]
    return np.repeat(last[:, None], ds.H, axis=1)


def _gauss_logp(y, mu, var):
    return -0.5 * (torch.log(2 * math.pi * var) + (y - mu) ** 2 / var)


def train_tdpo(ref_model, ds, pairs: Dict[str, np.ndarray], cfg: dict,
               rng: np.random.Generator, gamma: Optional[float] = None,
               n_pairs: Optional[int] = None, beta: Optional[float] = None):
    """Fine-tune a copy of ``ref_model`` on the first ``n_pairs`` pairs."""
    tc = cfg["tdpo"]
    gamma = tc["gamma"] if gamma is None else gamma
    beta = tc["beta"] if beta is None else beta
    n = len(pairs["win"]) if n_pairs is None else n_pairs
    policy = copy.deepcopy(ref_model)
    policy.eval()                                   # dropout off; grads still flow
    for p in policy.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(policy.parameters(), lr=tc["learning_rate"],
                            weight_decay=cfg["supervised"]["weight_decay"])
    w_h = torch.tensor([gamma ** h for h in range(ds.H)], dtype=torch.float32)
    win = torch.from_numpy(pairs["win"][:n])
    lose = torch.from_numpy(pairs["lose"][:n])
    rmu = torch.from_numpy(pairs["ref_mean"][:n])
    rvar = torch.from_numpy(pairs["ref_var"][:n])
    orig = pairs["origins"][:n]
    ref_w = (_gauss_logp(win, rmu, rvar) * w_h).sum(1)
    ref_l = (_gauss_logp(lose, rmu, rvar) * w_h).sum(1)
    bs = tc["batch_size"]
    curve, accs = [], []
    for _ in range(tc["epochs"]):
        perm = rng.permutation(n)
        tot, acc, nb = 0.0, 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = policy(torch.from_numpy(ds.contexts(orig[idx])))
            mu, var = out["mean"], out["var"]
            lw = (_gauss_logp(win[idx], mu, var) * w_h).sum(1) - ref_w[idx]
            ll = (_gauss_logp(lose[idx], mu, var) * w_h).sum(1) - ref_l[idx]
            margin = beta * (lw - ll)
            loss = -F.logsigmoid(margin).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            acc += (margin > 0).float().mean().item()
            nb += 1
        curve.append(tot / nb)
        accs.append(acc / nb)
    policy.eval()
    return policy, {"loss": curve, "preference_accuracy": accs}
