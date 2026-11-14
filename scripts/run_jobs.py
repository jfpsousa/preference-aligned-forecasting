"""
Stages 2-4 -- run experiment jobs in parallel.

    --stage main      every (dataset, seed) of Tables 2 and 3
    --stage ablation  every (ablation dataset, seed)       [needs main]
    --stage sweep     every seed on the cost-ratio dataset [needs main]
    --stage beta      every (beta dataset, seed)            [needs main]

A job whose result file already exists is skipped, so an interrupted run can be
resumed by re-running the same command; --force recomputes. Each job logs to
<output>/logs/.
"""

import argparse
import contextlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _init(threads):
    import torch
    torch.set_num_threads(threads)


def _run(job):
    stage, name, seed, cfg_path, out_dir = job
    from tpo.config import load_config
    from tpo import experiment as E
    cfg = load_config(cfg_path)
    out_dir = Path(out_dir)
    log = out_dir / "logs" / f"{stage}_{name}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            if stage == "main":
                exp = cfg["experiments"]
                methods = (exp["table2_methods"] if name in exp["table2_datasets"]
                           else exp["table3_methods"])
                if name == exp["cost_ratio_dataset"] and "tdpo" not in methods:
                    methods = list(methods) + ["tdpo"]
                E.run_main(cfg, ROOT, out_dir, name, seed, methods)
            elif stage == "ablation":
                E.run_ablation(cfg, ROOT, out_dir, name, seed)
            elif stage == "sweep":
                E.run_sweep(cfg, ROOT, out_dir, name, seed)
            elif stage == "beta":
                E.run_beta(cfg, ROOT, out_dir, name, seed)
            ok = True
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            ok = False
    return stage, name, seed, ok, time.time() - t0, str(log)


def jobs_for(cfg, stage):
    exp = cfg["experiments"]
    if stage == "main":
        names = list(dict.fromkeys(exp["table2_datasets"] + exp["table3_datasets"]))
    elif stage == "ablation":
        names = exp["ablation_datasets"]
    elif stage == "sweep":
        names = [exp["cost_ratio_dataset"]]
    elif stage == "beta":
        names = exp["beta_datasets"]
    else:
        raise ValueError(stage)
    return [(n, s) for n in names for s in cfg["seeds"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["main", "ablation", "sweep", "beta"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    from tpo.config import load_config
    cfg = load_config(args.config)
    out = Path(args.output)
    todo, skipped = [], 0
    for name, seed in jobs_for(cfg, args.stage):
        result = out / "runs" / name / f"seed{seed}" / f"{args.stage}.json"
        if result.exists() and not args.force:
            skipped += 1
            continue
        todo.append((args.stage, name, seed, str(Path(args.config).resolve()), str(out)))
    par = cfg["parallel"]
    print(f"stage {args.stage}: {len(todo)} jobs to run, {skipped} already done; "
          f"{par['workers']} workers x {par['threads_per_worker']} threads", flush=True)
    failed = []
    t0 = time.time()
    with mp.get_context("spawn").Pool(par["workers"], initializer=_init,
                                      initargs=(par["threads_per_worker"],)) as pool:
        for i, (stage, name, seed, ok, dt, log) in enumerate(
                pool.imap_unordered(_run, todo), 1):
            status = "ok" if ok else f"FAILED (see {log})"
            print(f"  [{i}/{len(todo)}] {stage} {name} seed={seed}  "
                  f"{dt / 60:.1f} min  {status}", flush=True)
            if not ok:
                failed.append(log)
    print(f"stage {args.stage} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    timing = out / "logs" / "stage_timings.json"
    t = json.loads(timing.read_text()) if timing.exists() else {}
    t[args.stage] = {"minutes": (time.time() - t0) / 60, "jobs": len(todo)}
    timing.write_text(json.dumps(t, indent=1))
    if failed:
        sys.exit(f"{len(failed)} job(s) failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
