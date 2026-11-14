"""
Experiment jobs. Each job is a pure function of (config, dataset, seed) and
writes one JSON file of raw results plus the test-set forecasts it scored, so
every number in the tables can be traced to a stored forecast array.

    main      Tables 2 and 3: every method on one dataset for one seed
    ablation  Figure 3 and Sec. 7.4: gamma sweep, sample efficiency,
              constitutional-principle ablation, reference-policy ablation
    sweep     Table 4: cost-ratio sensitivity on M5
    beta      Sec. 7.1 (beta grid search on the validation set) and Sec. 7.4
              (inverse-temperature ablation) on the real datasets

``ablation``, ``sweep`` and ``beta`` reuse the MSE model (= reference policy) and the
preference pairs saved by ``main`` for the same dataset and seed.
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from .constitution import generate_pairs
from .data import Dataset
from .metrics import evaluate
from .models import build_model
from .training import naive_forecast, predict, set_seed, train_supervised, train_tdpo

METHOD_IDS = {"mse": 1, "patchtst": 2, "informer": 3, "asymmetric": 4,
              "oracle": 5, "tdpo": 6, "pairs": 7}


def _sub_seed(seed: int, *tags) -> int:
    """Deterministic, order-independent seed for one component of a job."""
    return (seed * 1_000_003 + zlib.crc32(repr(tags).encode())) % (2 ** 31 - 1)


def _paths(out_dir: Path, ds_name: str, seed: int):
    base = out_dir / "runs" / ds_name / f"seed{seed}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_dataset(cfg, root: Path, name: str) -> Dataset:
    dom = cfg["datasets"][name]["domain"]
    return Dataset.load(root / "data" / "processed" / f"{name}.npz",
                        cfg["context_length"], cfg["domains"][dom]["horizon"])


def _model(cfg, ds, kind="tpo"):
    return build_model(kind, ds.X.shape[2], cfg["context_length"], ds.H,
                       cfg["architecture"])


def _score(ds, origins, pred, dcfg, cfg, cache, alpha_under=None):
    res = evaluate(ds, origins, pred, dcfg, cfg["evaluation"]["n_actions"], cache,
                   alpha_under)
    res["by_n_actions"] = {
        str(n): evaluate(ds, origins, pred, dcfg, n, cache, alpha_under)
        for n in cfg["evaluation"]["sensitivity_n_actions"]}
    return res


def _pair_args(cfg, dcfg, alpha_under=None):
    return dict(num_candidates=cfg["tdpo"]["num_candidates"],
                alpha_under=float(dcfg["alpha_under"] if alpha_under is None else alpha_under),
                alpha_over=float(dcfg["alpha_over"]), cons=cfg["constitution"])


def _sample_origins(ds, rng, n):
    tr = ds.train_origins()
    return tr[rng.choice(len(tr), size=n, replace=len(tr) < n)]


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def run_main(cfg, root: Path, out_dir: Path, name: str, seed: int, methods):
    ds = load_dataset(cfg, root, name)
    dcfg = cfg["domains"][ds.domain]
    base = _paths(out_dir, name, seed)
    test = ds.test_origins()
    cache, results, timings, curves = {}, {}, {}, {}
    preds = {}

    preds["naive"] = naive_forecast(ds, test)

    t0 = time.time()
    rng = set_seed(_sub_seed(seed, "mse"))
    ref = _model(cfg, ds)
    curves["mse"] = train_supervised(ref, ds, "mse", cfg, rng, dcfg)
    preds["mse"] = predict(ref, ds, test)
    torch.save(ref.state_dict(), base / "mse_model.pt")
    timings["mse"] = time.time() - t0

    for m, kind, obj in [("patchtst", "patchtst", "mse"), ("informer", "informer", "mse"),
                         ("asymmetric", "tpo", "asymmetric"), ("oracle", "tpo", "oracle")]:
        if m not in methods:
            continue
        t0 = time.time()
        rng = set_seed(_sub_seed(seed, m))
        model = _model(cfg, ds, kind)
        curves[m] = train_supervised(model, ds, obj, cfg, rng, dcfg)
        preds[m] = predict(model, ds, test)
        timings[m] = time.time() - t0

    if "tdpo" in methods:
        t0 = time.time()
        n_pool = max([cfg["tdpo"]["num_pairs"]] +
                     (cfg["experiments"]["pair_grid"]
                      if name in cfg["experiments"]["ablation_datasets"] else []))
        rng = set_seed(_sub_seed(seed, "pairs"))
        pairs = generate_pairs(ref, ds, _sample_origins(ds, rng, n_pool), rng,
                               **_pair_args(cfg, dcfg))
        np.savez_compressed(base / "pairs.npz", **pairs)
        rng = set_seed(_sub_seed(seed, "tdpo"))
        pol, curves["tdpo"] = train_tdpo(ref, ds, pairs, cfg, rng,
                                         n_pairs=cfg["tdpo"]["num_pairs"])
        preds["tdpo"] = predict(pol, ds, test)
        timings["tdpo"] = time.time() - t0

    for m, p in preds.items():
        results[m] = _score(ds, test, p, dcfg, cfg, cache)
    np.savez_compressed(base / "main_forecasts.npz", origins=test, **preds)
    out = {"job": "main", "dataset": name, "domain": ds.domain, "seed": seed,
           "n_test_windows": int(len(test)), "horizon": ds.H,
           "methods": results, "training_curves": curves, "seconds": timings}
    (base / "main.json").write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------
# ablations (synthetic, Figure 3 and Section 7.4)
# ----------------------------------------------------------------------------

def _load_ref(cfg, ds, base):
    ref = _model(cfg, ds)
    ref.load_state_dict(torch.load(base / "mse_model.pt"))
    ref.eval()
    return ref


def run_ablation(cfg, root: Path, out_dir: Path, name: str, seed: int):
    ds = load_dataset(cfg, root, name)
    dcfg = cfg["domains"][ds.domain]
    base = _paths(out_dir, name, seed)
    test = ds.test_origins()
    cache = {}
    ref = _load_ref(cfg, ds, base)
    pairs = dict(np.load(base / "pairs.npz"))
    n_main = cfg["tdpo"]["num_pairs"]
    out = {"job": "ablation", "dataset": name, "domain": ds.domain, "seed": seed}

    def tdpo_eval(pairs_, tag, **kw):
        ref_ = kw.pop("ref", ref)
        rng = set_seed(_sub_seed(seed, "abl", tag))
        pol, curve = train_tdpo(ref_, ds, pairs_, cfg, rng, **kw)
        r = evaluate(ds, test, predict(pol, ds, test), dcfg,
                     cfg["evaluation"]["n_actions"], cache)
        r["training_curve"] = curve
        return r

    # (a) temporal discount gamma -- same reference and pairs, only gamma varies
    out["gamma"] = {str(g): tdpo_eval(pairs, ("gamma", g), gamma=g, n_pairs=n_main)
                    for g in cfg["experiments"]["gamma_grid"]}

    # (b) sample efficiency -- T-DPO on the first n pairs; Asymmetric Loss on n
    #     labelled training windows with the full supervised budget
    se_t, se_a = {}, {}
    for n in cfg["experiments"]["pair_grid"]:
        se_t[str(n)] = tdpo_eval(pairs, ("pairs", n), n_pairs=n)
        rng = set_seed(_sub_seed(seed, "abl", "asym_n", n))
        sub = _sample_origins(ds, rng, n)
        m = _model(cfg, ds)
        train_supervised(m, ds, "asymmetric", cfg, rng, dcfg, origins=sub)
        se_a[str(n)] = evaluate(ds, test, predict(m, ds, test), dcfg,
                                cfg["evaluation"]["n_actions"], cache)
    out["sample_efficiency"] = {"tdpo": se_t, "asymmetric": se_a}

    # (c) constitutional principles -- regenerate pairs on the same contexts
    origins = pairs["origins"][:n_main]
    w_full = list(cfg["constitution"]["weights"])
    variants = {"all": dict(weights=w_full)}
    for j in range(4):
        w = list(w_full)
        w[j] = 0.0
        variants[f"without_P{j + 1}"] = dict(weights=w)
    variants["none"] = dict(mode="none")
    out["principles"] = {}
    for tag, kw in variants.items():
        rng = set_seed(_sub_seed(seed, "abl", "princ", tag))
        p = generate_pairs(ref, ds, origins, rng, **_pair_args(cfg, dcfg), **kw)
        out["principles"][tag] = tdpo_eval(p, ("princ", tag))

    # (d) reference policy -- randomly initialised instead of MSE-pretrained
    set_seed(_sub_seed(seed, "abl", "random_ref_init"))
    rnd = _model(cfg, ds)
    rnd.eval()
    rng = set_seed(_sub_seed(seed, "abl", "random_ref_pairs"))
    p = generate_pairs(rnd, ds, origins, rng, **_pair_args(cfg, dcfg))
    out["reference_policy"] = {
        "pretrained": out["principles"]["all"],
        "random_init": tdpo_eval(p, ("random_ref",), ref=rnd),
    }
    (base / "ablation.json").write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------
# cost-ratio sweep (Table 4)
# ----------------------------------------------------------------------------

def run_sweep(cfg, root: Path, out_dir: Path, name: str, seed: int):
    ds = load_dataset(cfg, root, name)
    dcfg = cfg["domains"][ds.domain]
    base = _paths(out_dir, name, seed)
    test = ds.test_origins()
    cache = {}
    ref = _load_ref(cfg, ds, base)
    pred_mse = predict(ref, ds, test)
    pairs_main = np.load(base / "pairs.npz")
    origins = pairs_main["origins"][: cfg["tdpo"]["num_pairs"]]
    out = {"job": "sweep", "dataset": name, "seed": seed, "alpha": {}}
    for a in cfg["experiments"]["cost_ratio_grid"]:
        rng = set_seed(_sub_seed(seed, "sweep", "asym", a))
        m = _model(cfg, ds)
        train_supervised(m, ds, "asymmetric", cfg, rng, dcfg, alpha_under=a)
        pred_asym = predict(m, ds, test)
        rng = set_seed(_sub_seed(seed, "sweep", "pairs", a))
        pairs = generate_pairs(ref, ds, origins, rng, **_pair_args(cfg, dcfg, a))
        rng = set_seed(_sub_seed(seed, "sweep", "tdpo", a))
        pol, curve = train_tdpo(ref, ds, pairs, cfg, rng)
        pred_t = predict(pol, ds, test)
        n = cfg["evaluation"]["n_actions"]
        out["alpha"][str(a)] = {
            "mse": evaluate(ds, test, pred_mse, dcfg, n, cache, a),
            "asymmetric": evaluate(ds, test, pred_asym, dcfg, n, cache, a),
            "tdpo": evaluate(ds, test, pred_t, dcfg, n, cache, a),
            "tdpo_training_curve": curve,
        }
    (base / "sweep.json").write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------
# inverse temperature beta (Sec. 7.1 selection, Sec. 7.4 ablation)
# ----------------------------------------------------------------------------

def run_beta(cfg, root: Path, out_dir: Path, name: str, seed: int):
    """T-DPO with only beta varied: same reference policy, same 5,000 pairs, same
    fine-tuning seed. Every model is scored on the validation windows (used to
    select beta) and on the test windows (the ablation). The main experiments
    keep the configured beta; this job reports which beta validation selects so
    the two can be compared."""
    ds = load_dataset(cfg, root, name)
    dcfg = cfg["domains"][ds.domain]
    base = _paths(out_dir, name, seed)
    n = cfg["evaluation"]["n_actions"]
    splits = {"val": ds.val_origins(), "test": ds.test_origins()}
    cache = {}
    ref = _load_ref(cfg, ds, base)
    pairs = dict(np.load(base / "pairs.npz"))
    out = {"job": "beta", "dataset": name, "domain": ds.domain, "seed": seed,
           "n_windows": {k: int(len(v)) for k, v in splits.items()},
           "selection_metric": cfg["experiments"]["beta_selection_metric"],
           "mse": {k: evaluate(ds, o, predict(ref, ds, o), dcfg, n, cache)
                   for k, o in splits.items()},
           "beta": {}}
    for b in cfg["experiments"]["beta_grid"]:
        rng = set_seed(_sub_seed(seed, "beta"))        # identical across beta
        pol, curve = train_tdpo(ref, ds, pairs, cfg, rng, beta=float(b),
                                n_pairs=cfg["tdpo"]["num_pairs"])
        out["beta"][str(b)] = {k: evaluate(ds, o, predict(pol, ds, o), dcfg, n, cache)
                               for k, o in splits.items()}
        out["beta"][str(b)]["training_curve"] = curve
    # best validation score; ties (decision accuracy is discrete) are broken by
    # lower validation cost, then by the smaller beta
    metric = out["selection_metric"]
    sign = -1.0 if metric == "cost" else 1.0
    grid = sorted(cfg["experiments"]["beta_selection_grid"])
    out["selected_beta"] = float(max(
        grid, key=lambda b: (sign * out["beta"][str(b)]["val"][metric],
                             -out["beta"][str(b)]["val"]["cost"], -b)))
    (base / "beta.json").write_text(json.dumps(out, indent=1))
    return out
