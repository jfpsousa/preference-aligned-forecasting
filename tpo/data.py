"""
Data acquisition, synthetic generation, preprocessing and windowing.

Every dataset is reduced to one processed file ``data/processed/<name>.npz``
with a common layout, so the experiment code never touches raw files:

    X        (S, T, C) float32  model inputs, standardised with TRAIN statistics;
                                channel 0 is the (transformed) target
    y_dec    (S, T)    float64  target in decision units (units in which the
                                business cost of Eq. 16 is charged)
    z_mu,
    z_sd     (S,)               per-series standardisation of the transformed target
    transform str               'none' or 'log1p' (inverse applied before decisions)
    crit     (S, T)    bool     critical-period indicator c_t used by principle P4
    peak     (S, T)    bool     energy peak hours (cost multiplier); False elsewhere
    train_end, val_end int      chronological split points on the time axis

Splits are chronological. A training window's targets all lie before
``train_end``; test windows are non-overlapping blocks of H steps laid back from
the end of the series and starting no earlier than ``val_end``.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Raw-data provenance
# --------------------------------------------------------------------------

UCI_URL = ("https://archive.ics.uci.edu/static/public/235/"
           "individual+household+electric+power+consumption.zip")
M5_KAGGLE_COMPETITION = "m5-forecasting-accuracy"
M5_MIRROR_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"

# SHA-256 of the exact files the pipeline reads are recorded in data/SHA256SUMS
# and verified after download, so the data are identical whichever source
# (Kaggle or mirror) they came from.
_SHA256_FILE = Path(__file__).resolve().parent.parent / "data" / "SHA256SUMS"


def _load_checksums() -> Dict[str, str]:
    out = {}
    if _SHA256_FILE.exists():
        for line in _SHA256_FILE.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                digest, name = line.split()
                out[name if name.startswith("m5:") else Path(name).name] = digest
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> None:
    """Raise if ``path`` does not match its recorded SHA-256."""
    expected = _load_checksums().get(path.name)
    if expected is None:
        raise RuntimeError(f"No recorded checksum for {path.name} in {_SHA256_FILE}")
    got = sha256(path)
    if got != expected:
        raise RuntimeError(
            f"Checksum mismatch for {path}:\n  expected {expected}\n  got      {got}\n"
            "The file differs from the data vintage used for the paper.")
    print(f"  verified {path.name}  sha256={got[:16]}...")


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def fetch_uci(raw_dir: Path) -> Path:
    target = raw_dir / "uci" / "household_power_consumption.txt"
    if not target.exists():
        z = raw_dir / "uci" / "household.zip"
        _download(UCI_URL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extract("household_power_consumption.txt", target.parent)
        z.unlink()
    verify(target)
    return target


def fetch_m5(raw_dir: Path) -> Path:
    """
    M5 from Kaggle when the ``kaggle`` CLI and credentials are available,
    otherwise from a public mirror of the official M5 release. The two sources
    differ in file layout (the mirror omits the ``id`` and ``d`` columns), so M5
    is verified on its parsed CONTENT (see ``m5_content_hash``) rather than on
    file bytes.
    """
    d = raw_dir / "m5"
    need = [d / "sales_train_evaluation.csv", d / "calendar.csv"]
    if not all(p.exists() for p in need):
        d.mkdir(parents=True, exist_ok=True)
        got = False
        if shutil.which("kaggle") and (Path.home() / ".kaggle" / "kaggle.json").exists():
            try:
                subprocess.run(["kaggle", "competitions", "download", "-c",
                                M5_KAGGLE_COMPETITION, "-p", str(d)], check=True)
                with zipfile.ZipFile(d / f"{M5_KAGGLE_COMPETITION}.zip") as zf:
                    for p in need:
                        zf.extract(p.name, d)
                got = True
            except Exception as e:  # noqa: BLE001
                print(f"  Kaggle download failed ({e}); using mirror")
        if not got:
            z = d / "m5.zip"
            _download(M5_MIRROR_URL, z)
            with zipfile.ZipFile(z) as zf:
                for p in need:
                    zf.extract(p.name, d)
            z.unlink()
    return d


def m5_content_hash(sales: pd.DataFrame, cal: pd.DataFrame, dcols) -> str:
    """SHA-256 of the M5 content used: (item, store) keys and the daily unit
    sales in canonical row order, plus the calendar fields used."""
    h = hashlib.sha256()
    h.update("\n".join(sales["item_id"] + "|" + sales["store_id"]).encode())
    h.update(np.ascontiguousarray(sales[dcols].to_numpy(dtype=np.int64)).tobytes())
    h.update("\n".join(cal["date"].astype(str)).encode())
    h.update(cal[["wday", "snap_CA", "snap_TX", "snap_WI"]].to_numpy(np.int64).tobytes())
    h.update(cal["event_name_1"].notna().to_numpy().tobytes())
    return h.hexdigest()


def yahoo_snapshot(repo_root: Path) -> Path:
    p = repo_root / "data" / "snapshots" / "yahoo_etf_2018_2023.csv"
    verify(p)
    return p


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _cyc(x: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    return np.sin(2 * np.pi * x / period), np.cos(2 * np.pi * x / period)


def _split_points(T: int, split) -> Tuple[int, int]:
    tr, va = split[0], split[1]
    return int(T * tr), int(T * (tr + va))


def _top_dow_critical(y: np.ndarray, dow: np.ndarray, train_end: int) -> np.ndarray:
    """Retail c_t: the two weekdays with the highest TRAINING mean demand."""
    crit = np.zeros_like(y, dtype=bool)
    for s in range(y.shape[0]):
        means = np.array([y[s, :train_end][dow[s, :train_end] == d].mean()
                          for d in range(7)])
        top = np.argsort(means)[-2:]
        crit[s] = np.isin(dow[s], top)
    return crit


def _assemble(name, domain, y_dec, feats, transform, train_end, val_end,
              crit, peak) -> Dict[str, np.ndarray]:
    """Standardise with training statistics and pack the common layout."""
    S, T = y_dec.shape
    y_tr = np.log1p(y_dec) if transform == "log1p" else y_dec.copy()
    z_mu = y_tr[:, :train_end].mean(axis=1)
    z_sd = np.maximum(y_tr[:, :train_end].std(axis=1), 1e-6)
    X = np.empty((S, T, 1 + feats.shape[2]), dtype=np.float32)
    X[:, :, 0] = (y_tr - z_mu[:, None]) / z_sd[:, None]
    f_mu = feats[:, :train_end].mean(axis=1, keepdims=True)
    f_sd = np.maximum(feats[:, :train_end].std(axis=1, keepdims=True), 1e-6)
    X[:, :, 1:] = (feats - f_mu) / f_sd
    return dict(name=name, domain=domain, X=X, y_dec=y_dec.astype(np.float64),
                z_mu=z_mu, z_sd=z_sd, transform=transform,
                crit=crit.astype(bool), peak=peak.astype(bool),
                train_end=int(train_end), val_end=int(val_end))


# --------------------------------------------------------------------------
# Synthetic data (manuscript Appendix B.1)
# --------------------------------------------------------------------------

def synthetic_retail(n: int, T: int, split, seed: int):
    """Base demand 100 with 0.05%/day growth, weekly amplitude 10, monthly
    amplitude 15, Gaussian noise sd = 0.1 x mean (AR(1)-correlated, phi=0.3)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    y = np.empty((n, T))
    dow = np.empty((n, T), dtype=int)
    feats = np.empty((n, T, 4))
    for s in range(n):
        scale = rng.uniform(0.5, 1.5)
        ph_w, ph_m = rng.uniform(0, 2 * np.pi, 2)
        start_dow = rng.integers(7)
        mean = 100 * 1.0005 ** t + 10 * np.sin(2 * np.pi * t / 7 + ph_w) \
            + 15 * np.sin(2 * np.pi * t / 30.5 + ph_m)
        e = np.empty(T)
        e[0] = rng.normal()
        for i in range(1, T):
            e[i] = 0.3 * e[i - 1] + np.sqrt(1 - 0.09) * rng.normal()
        y[s] = scale * np.maximum(mean + 0.1 * mean * e, 0.0)
        dow[s] = (start_dow + t) % 7
        feats[s, :, 0], feats[s, :, 1] = _cyc(dow[s], 7)
        feats[s, :, 2], feats[s, :, 3] = _cyc(t % 30.5, 30.5)
    tr, va = _split_points(T, split)
    crit = _top_dow_critical(y, dow, tr)
    return _assemble("synthetic_retail", "retail", y, feats, "none", tr, va,
                     crit, np.zeros_like(crit))


def synthetic_energy(n: int, T: int, split, seed: int, peak_hours):
    """Hourly load: base 500 MW with a daily profile peaking in the stated peak
    hours, nonlinear temperature effect of 20 MW per degree outside a 15-20C
    comfort band, and AR(1) residual (phi=0.7)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    y = np.empty((n, T))
    hour = np.empty((n, T), dtype=int)
    feats = np.empty((n, T, 4))
    for s in range(n):
        start_h = 0
        h = (start_h + t) % 24
        day = (start_h + t) // 24
        season = rng.uniform(0, 2 * np.pi)
        temp = 15 + 8 * np.sin(2 * np.pi * day / 365 + season) \
            + 5 * np.sin(2 * np.pi * (h - 9) / 24)
        tn = np.empty(T)
        tn[0] = 0
        for i in range(1, T):
            tn[i] = 0.9 * tn[i - 1] + rng.normal(0, 0.7)
        temp = temp + tn
        profile = 80 * np.exp(-0.5 * ((h - 12) / 1.8) ** 2) \
            + 110 * np.exp(-0.5 * ((h - 19.5) / 1.5) ** 2)
        temp_eff = 20 * np.maximum(temp - 20, 0) + 20 * np.maximum(15 - temp, 0)
        e = np.empty(T)
        e[0] = 0
        for i in range(1, T):
            e[i] = 0.7 * e[i - 1] + rng.normal(0, 15)
        weekend = ((day % 7) >= 5).astype(float)
        y[s] = rng.uniform(0.8, 1.2) * (500 + profile + temp_eff - 40 * weekend) + e
        hour[s] = h
        feats[s, :, 0] = temp
        feats[s, :, 1], feats[s, :, 2] = _cyc(h, 24)
        feats[s, :, 3] = weekend
    tr, va = _split_points(T, split)
    peak = np.isin(hour, list(peak_hours))
    return _assemble("synthetic_energy", "energy", y, feats, "none", tr, va,
                     peak, peak)


def _finance_features(r: np.ndarray) -> np.ndarray:
    """Lagged returns (5, 10, 21) and 21-day rolling volatility (App. B.2)."""
    s = pd.Series(r)
    return np.stack([s.shift(5).to_numpy(), s.shift(10).to_numpy(),
                     s.shift(21).to_numpy(), s.rolling(21).std().to_numpy()], axis=1)


def _vol_critical(feats: np.ndarray, train_end: int) -> np.ndarray:
    """Finance c_t: trailing 21-day volatility above its TRAINING 75th percentile."""
    vol = feats[:, :, 3]
    q = np.nanpercentile(vol[:, :train_end], 75, axis=1)
    return vol > q[:, None]


def synthetic_finance(n: int, T: int, split, seed: int):
    """GARCH(1,1), unconditional daily volatility 2%, persistence 0.85."""
    rng = np.random.default_rng(seed)
    a, b = 0.10, 0.75
    omega = 0.02 ** 2 * (1 - a - b)
    burn = 22
    y = np.empty((n, T))
    feats = np.empty((n, T, 4))
    for s in range(n):
        N = T + burn
        r = np.empty(N)
        v = 0.02 ** 2
        for i in range(N):
            r[i] = np.sqrt(v) * rng.normal()
            v = omega + a * r[i] ** 2 + b * v
        f = _finance_features(r)
        y[s] = r[burn:]
        feats[s] = f[burn:]
    tr, va = _split_points(T, split)
    crit = _vol_critical(feats, tr)
    return _assemble("synthetic_finance", "finance", y, feats, "none", tr, va,
                     crit, np.zeros_like(crit))


# --------------------------------------------------------------------------
# Real data (manuscript Appendix B.2)
# --------------------------------------------------------------------------

def preprocess_m5(m5_dir: Path, n_products: int, split, seed: int):
    """
    1,000 products sampled stratified by store x category (proportional
    allocation, largest-remainder rounding), log1p transform, 80/10/10
    chronological split; the final 28 days are the last test window.
    Products with no sales at all in the training period are not eligible.
    """
    sales = pd.read_csv(m5_dir / "sales_train_evaluation.csv")
    cal = pd.read_csv(m5_dir / "calendar.csv")
    dcols = [c for c in sales.columns if c.startswith("d_")]
    if "id" not in sales.columns:          # mirror layout
        sales["id"] = sales["item_id"] + "_" + sales["store_id"] + "_evaluation"
    if "d" not in cal.columns:
        cal["d"] = [f"d_{i + 1}" for i in range(len(cal))]
    # Canonical row order, so sampling does not depend on the download source.
    sales = sales.sort_values(["store_id", "item_id"]).reset_index(drop=True)
    cal = cal.set_index("d").loc[dcols].reset_index()
    digest = m5_content_hash(sales, cal, dcols)
    expected = _load_checksums().get("m5:content")
    if digest != expected:
        raise RuntimeError(f"M5 content hash mismatch:\n  expected {expected}\n  got      {digest}")
    print(f"  verified M5 content  sha256={digest[:16]}...")
    T = len(dcols)
    tr, va = _split_points(T, split)
    vals = sales[dcols].to_numpy(dtype=np.float64)
    eligible = sales[vals[:, :tr].sum(axis=1) > 0].copy()
    eligible["stratum"] = eligible["store_id"] + "|" + eligible["cat_id"]
    sizes = eligible.groupby("stratum").size()
    quota = sizes / sizes.sum() * n_products
    alloc = np.floor(quota).astype(int)
    rem = n_products - alloc.sum()
    alloc[(quota - alloc).sort_values(ascending=False).index[:rem]] += 1
    rng = np.random.default_rng(seed)
    picked = []
    for stratum, k in alloc.items():
        idx = eligible.index[eligible["stratum"] == stratum].to_numpy()
        picked.extend(sorted(rng.choice(idx, size=int(k), replace=False)))
    sub = sales.loc[sorted(picked)]
    y = sub[dcols].to_numpy(dtype=np.float64)

    wday = ((cal["wday"].to_numpy() - 1 + 5) % 7)          # Mon=0 .. Sun=6
    dow = np.tile(wday, (len(sub), 1))
    event = cal["event_name_1"].notna().to_numpy().astype(float)
    snap = np.stack([cal[f"snap_{st}"].to_numpy() for st in sub["state_id"]]).astype(float)
    feats = np.empty((len(sub), T, 4))
    feats[:, :, 0], feats[:, :, 1] = _cyc(dow, 7)
    feats[:, :, 2] = snap
    feats[:, :, 3] = event[None, :]
    crit = _top_dow_critical(y, dow, tr)
    out = _assemble("m5", "retail", y, feats, "log1p", tr, va, crit,
                    np.zeros_like(crit))
    out["series_id"] = np.array(sub["id"].tolist(), dtype=str)
    return out


def preprocess_uci(path: Path, split, peak_hours):
    """Hourly mean of Global_active_power (kW); gaps filled by linear
    interpolation; 70/10/20 chronological split."""
    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False,
                     usecols=["Date", "Time", "Global_active_power"])
    df.index = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    s = pd.to_numeric(df["Global_active_power"], errors="coerce")
    hourly = s.resample("h").mean()
    missing = float(hourly.isna().mean())
    hourly = hourly.interpolate(method="linear", limit_direction="both")
    y = hourly.to_numpy()[None, :]
    h = hourly.index.hour.to_numpy()[None, :]
    dow = hourly.index.dayofweek.to_numpy()[None, :]
    T = y.shape[1]
    feats = np.empty((1, T, 4))
    feats[:, :, 0], feats[:, :, 1] = _cyc(h, 24)
    feats[:, :, 2], feats[:, :, 3] = _cyc(dow, 7)
    tr, va = _split_points(T, split)
    peak = np.isin(h, list(peak_hours))
    out = _assemble("uci_energy", "energy", y, feats, "none", tr, va, peak, peak)
    out["missing_fraction_hourly"] = missing
    out["start"] = str(hourly.index[0])
    out["end"] = str(hourly.index[-1])
    return out


def preprocess_yahoo(path: Path, tickers, train_end_date: str, val_end_date: str):
    """Daily log returns of adjusted close; lagged returns (5, 10, 21) and
    21-day rolling volatility; 2018-2022 train, Jan-Jun 2023 validation,
    Jul-Dec 2023 test."""
    raw = pd.read_csv(path, parse_dates=["Date"])
    panel = raw.pivot(index="Date", columns="Ticker", values="Adj Close")[list(tickers)]
    r = np.log(panel).diff().iloc[1:]
    feats_all = np.stack([_finance_features(r[t].to_numpy()) for t in tickers])
    keep = ~np.isnan(feats_all).any(axis=(0, 2))
    dates = r.index[keep]
    y = r.to_numpy().T[:, keep]
    feats = feats_all[:, keep]
    tr = int(np.searchsorted(dates, pd.Timestamp(train_end_date)))
    va = int(np.searchsorted(dates, pd.Timestamp(val_end_date)))
    crit = _vol_critical(feats, tr)
    out = _assemble("yahoo_etf", "finance", y, feats, "none", tr, va, crit,
                    np.zeros_like(crit))
    out["dates"] = np.array(dates.strftime("%Y-%m-%d").tolist(), dtype=str)
    out["series_id"] = np.array(tickers)
    return out


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

@dataclass
class Dataset:
    name: str
    domain: str
    X: np.ndarray
    y_dec: np.ndarray
    z_mu: np.ndarray
    z_sd: np.ndarray
    transform: str
    crit: np.ndarray
    peak: np.ndarray
    train_end: int
    val_end: int
    L: int
    H: int

    @classmethod
    def load(cls, path: Path, context_length: int, horizon: int) -> "Dataset":
        z = np.load(path, allow_pickle=False)
        return cls(name=str(z["name"]), domain=str(z["domain"]), X=z["X"],
                   y_dec=z["y_dec"], z_mu=z["z_mu"], z_sd=z["z_sd"],
                   transform=str(z["transform"]), crit=z["crit"], peak=z["peak"],
                   train_end=int(z["train_end"]), val_end=int(z["val_end"]),
                   L=context_length, H=horizon)

    # -- index sets -------------------------------------------------------
    def train_origins(self) -> np.ndarray:
        """All (series, t0) whose context and targets lie inside the training
        period. ``t0`` is the first forecast step."""
        t = np.arange(self.L, self.train_end - self.H + 1)
        S = self.X.shape[0]
        return np.stack([np.repeat(np.arange(S), len(t)), np.tile(t, S)], axis=1)

    def val_origins(self) -> np.ndarray:
        """Non-overlapping H-step blocks whose targets lie entirely inside the
        validation period [train_end, val_end). Used only for selecting beta."""
        t = np.arange(self.train_end, self.val_end - self.H + 1, self.H)
        t = t[t >= self.L]
        S = self.X.shape[0]
        return np.stack([np.repeat(np.arange(S), len(t)), np.tile(t, S)], axis=1)

    def test_origins(self) -> np.ndarray:
        """Non-overlapping H-step blocks laid back from the series end, all at or
        after ``val_end`` -- the final block is the last H observations."""
        T = self.X.shape[1]
        t = np.arange(T - self.H, self.val_end - 1, -self.H)
        t = np.sort(t[t >= self.L])
        S = self.X.shape[0]
        return np.stack([np.repeat(np.arange(S), len(t)), np.tile(t, S)], axis=1)

    # -- batch assembly ---------------------------------------------------
    def contexts(self, origins: np.ndarray) -> np.ndarray:
        s, t = origins[:, 0], origins[:, 1]
        idx = t[:, None] + np.arange(-self.L, 0)[None, :]
        return self.X[s[:, None], idx]                         # (B, L, C)

    def _targets(self, arr: np.ndarray, origins: np.ndarray) -> np.ndarray:
        s, t = origins[:, 0], origins[:, 1]
        idx = t[:, None] + np.arange(self.H)[None, :]
        return arr[s[:, None], idx]

    def targets_z(self, origins):
        return self._targets(self.X[:, :, 0], origins)

    def targets_dec(self, origins):
        return self._targets(self.y_dec, origins)

    def crit_mask(self, origins):
        return self._targets(self.crit, origins)

    def peak_mask(self, origins):
        return self._targets(self.peak, origins)

    def to_decision_units(self, z: np.ndarray, origins: np.ndarray) -> np.ndarray:
        """Invert standardisation (and log1p) for forecasts in z-units."""
        s = origins[:, 0]
        v = z * self.z_sd[s, None] + self.z_mu[s, None]
        if self.transform == "log1p":
            v = np.expm1(v)
        return v

    def train_values_dec(self) -> np.ndarray:
        """(S, train_end) training targets in decision units, for the action set."""
        return self.y_dec[:, : self.train_end]
