import copy
from pathlib import Path

import numpy as np
import torch

from tpo import data as D
from tpo.config import load_config
from tpo.constitution import generate_pairs, violation_scores
from tpo.models import build_model
from tpo.training import train_tdpo

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config(ROOT / "configs" / "smoke.yaml")


def _ds(tmp_path, gen=D.synthetic_retail, H=28):
    out = gen(4, 400, [0.7, 0.1, 0.2], 0)
    p = tmp_path / "d.npz"
    np.savez(p, **out)
    return D.Dataset.load(p, 96, H)


def test_no_leakage_between_train_and_test(tmp_path):
    ds = _ds(tmp_path)
    tr, te = ds.train_origins(), ds.test_origins()
    assert (tr[:, 1] + ds.H <= ds.train_end).all()          # train targets < train_end
    assert (te[:, 1] >= ds.val_end).all()                     # test after validation
    assert te[:, 1].max() + ds.H == ds.X.shape[1]             # last block = final H steps
    assert np.all(np.diff(np.unique(te[:, 1])) >= ds.H)      # non-overlapping


def test_validation_windows_lie_between_train_and_test(tmp_path):
    ds = _ds(tmp_path)
    va = ds.val_origins()
    assert len(va) > 0
    assert (va[:, 1] >= ds.train_end).all()                   # after training targets
    assert (va[:, 1] + ds.H <= ds.val_end).all()              # before test period
    assert np.all(np.diff(np.unique(va[:, 1])) >= ds.H)      # non-overlapping


def test_standardisation_uses_training_period_only(tmp_path):
    ds = _ds(tmp_path)
    z = ds.X[:, : ds.train_end, 0]
    assert np.allclose(z.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(z.std(axis=1), 1, atol=1e-4)


def test_inverse_transform_roundtrip(tmp_path):
    ds = _ds(tmp_path)
    o = ds.test_origins()
    assert np.allclose(ds.to_decision_units(ds.targets_z(o), o), ds.targets_dec(o), rtol=1e-5)


def test_winner_never_scores_worse_than_loser(tmp_path):
    torch.manual_seed(0)
    ds = _ds(tmp_path)
    m = build_model("tpo", ds.X.shape[2], 96, ds.H, CFG["architecture"])
    rng = np.random.default_rng(0)
    o = ds.train_origins()[:64]
    p = generate_pairs(m, ds, o, rng, 8, 3.0, 1.0, CFG["constitution"])
    w = np.asarray(CFG["constitution"]["weights"])
    args = (ds.contexts(o)[:, :, 0], None, ds.targets_z(o), ds.crit_mask(o),
            p["ref_mean"], 3.0, 1.0, 2.0, 1.96, 1.0)
    sw = (violation_scores(args[0], p["win"][:, None], *args[2:]) * w).sum(-1)
    sl = (violation_scores(args[0], p["lose"][:, None], *args[2:]) * w).sum(-1)
    assert (sw <= sl + 1e-9).all()


def test_tdpo_loss_starts_at_log2(tmp_path):
    """pi_theta = pi_ref at initialisation, so every margin is 0 and the first
    mini-batch loss is log 2 (dropout must be off for this to hold)."""
    torch.manual_seed(0)
    ds = _ds(tmp_path)
    cfg = copy.deepcopy(CFG)
    cfg["tdpo"].update(epochs=1, learning_rate=0.0, batch_size=32)
    m = build_model("tpo", ds.X.shape[2], 96, ds.H, cfg["architecture"])
    m.eval()
    rng = np.random.default_rng(0)
    p = generate_pairs(m, ds, ds.train_origins()[:32], rng, 8, 3.0, 1.0, cfg["constitution"])
    _, curve = train_tdpo(m, ds, p, cfg, rng)
    assert abs(curve["loss"][0] - np.log(2)) < 1e-5


def test_energy_peak_hours_follow_clock(tmp_path):
    out = D.synthetic_energy(2, 300, [0.7, 0.1, 0.2], 0, [10, 11, 12, 13, 18, 19, 20])
    assert abs(out["peak"].mean() - 7 / 24) < 0.01
