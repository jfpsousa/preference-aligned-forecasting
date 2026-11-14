"""
Stage 1 -- acquire and preprocess every dataset named in the config.

Raw files go to data/raw/ (downloaded, checksum-verified); processed arrays go
to data/processed/<name>.npz, and data/processed/MANIFEST.json records shapes,
split points and window counts.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tpo import data as D  # noqa: E402
from tpo.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="rebuild processed files")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw, proc = ROOT / "data" / "raw", ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    peak_hours = cfg["domains"]["energy"]["peak_hours"]
    manifest = {}
    for name, spec in cfg["datasets"].items():
        target = proc / f"{name}.npz"
        print(f"[{name}]")
        spec_key = json.dumps({"spec": spec, "data_seed": cfg["data_seed"],
                               "peak_hours": peak_hours}, sort_keys=True)
        fresh = target.exists() and str(np.load(target)["spec"]) == spec_key
        if fresh and not args.force:
            print("  up to date, skipping (use --force to rebuild)")
        else:
            seed = cfg["data_seed"]
            if name == "synthetic_retail":
                out = D.synthetic_retail(spec["num_series"], spec["length"], spec["split"], seed)
            elif name == "synthetic_energy":
                out = D.synthetic_energy(spec["num_series"], spec["length"], spec["split"],
                                         seed, peak_hours)
            elif name == "synthetic_finance":
                out = D.synthetic_finance(spec["num_series"], spec["length"], spec["split"], seed)
            elif name == "m5":
                out = D.preprocess_m5(D.fetch_m5(raw), spec["num_products"], spec["split"], seed)
            elif name == "uci_energy":
                out = D.preprocess_uci(D.fetch_uci(raw), spec["split"], peak_hours)
            elif name == "yahoo_etf":
                out = D.preprocess_yahoo(D.yahoo_snapshot(ROOT), spec["tickers"],
                                         spec["train_end"], spec["val_end"])
            else:
                raise ValueError(name)
            out["spec"] = spec_key
            np.savez_compressed(target, **out)
        z = np.load(target)
        ds = D.Dataset.load(target, cfg["context_length"],
                            cfg["domains"][spec["domain"]]["horizon"])
        manifest[name] = {
            "domain": ds.domain, "series": int(ds.X.shape[0]),
            "time_steps": int(ds.X.shape[1]), "channels": int(ds.X.shape[2]),
            "train_end": ds.train_end, "val_end": ds.val_end, "horizon": ds.H,
            "train_windows": int(len(ds.train_origins())),
            "test_windows": int(len(ds.test_origins())),
            "transform": ds.transform,
            **({"first_date": str(z["dates"][0]), "last_date": str(z["dates"][-1]),
                "test_first_date": str(z["dates"][ds.test_origins()[:, 1].min()])}
               if "dates" in z else {}),
            **({"start": str(z["start"]), "end": str(z["end"]),
                "missing_fraction_hourly": float(z["missing_fraction_hourly"])}
               if "start" in z else {}),
        }
        print("  " + json.dumps(manifest[name]))
    (proc / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
