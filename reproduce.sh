#!/usr/bin/env bash
# Regenerate every table of the paper from the data.
#
#   ./reproduce.sh            full reproduction  (configs/paper.yaml -> outputs/)
#   ./reproduce.sh --smoke    installation check (configs/smoke.yaml -> outputs_smoke/)
#
# Stages run in order without manual intervention:
#   1  download raw data (checksum-verified) and preprocess
#   2  main experiments        -> Tables 2, 3 and Figure 2
#   3  ablations               -> Figure 3 and Section 7.4
#   4  cost-ratio sweep        -> Table 4
#   5  beta selection/ablation -> Sections 7.1 and 7.4
#   6  tables                  -> outputs/tables/
# Completed jobs are skipped when the script is re-run, so an interrupted run
# resumes where it stopped. Delete the output directory to start from scratch.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=configs/paper.yaml
OUT=outputs
if [[ "${1:-}" == "--smoke" ]]; then
  CONFIG=configs/smoke.yaml
  OUT=outputs_smoke
fi

PY=${PYTHON:-python}
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/reproduce.log") 2>&1

echo "== reproduce.sh  config=$CONFIG  output=$OUT  start=$(date -Is)"
"$PY" -c "import sys, torch, numpy, pandas, scipy; print('python', sys.version.split()[0], \
'| torch', torch.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__, \
'| scipy', scipy.__version__)"
"$PY" -m pytest -q tests

echo "== [1/6] data: download, verify, preprocess"
"$PY" scripts/prepare_data.py --config "$CONFIG"

echo "== [2/6] main experiments (Tables 2-3)"
"$PY" scripts/run_jobs.py --config "$CONFIG" --stage main --output "$OUT"

echo "== [3/6] ablations (Figure 3, Section 7.4)"
"$PY" scripts/run_jobs.py --config "$CONFIG" --stage ablation --output "$OUT"

echo "== [4/6] cost-ratio sweep (Table 4)"
"$PY" scripts/run_jobs.py --config "$CONFIG" --stage sweep --output "$OUT"

echo "== [5/6] inverse temperature beta (Sections 7.1, 7.4)"
"$PY" scripts/run_jobs.py --config "$CONFIG" --stage beta --output "$OUT"

echo "== [6/6] tables"
"$PY" scripts/make_tables.py --config "$CONFIG" --output "$OUT"

echo "== done $(date -Is). Tables in $OUT/tables/ (see $OUT/tables/SUMMARY.md)"
