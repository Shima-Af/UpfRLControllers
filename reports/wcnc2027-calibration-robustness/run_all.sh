#!/usr/bin/env bash
# One-command reproduction of the calibration-robustness study (zero-shot; no training).
#   bash run_all.sh            # everything, ~1.5 h on 32 cores
# Writes only inside this directory. Order matters: the reproduction check must
# pass before alternative worlds are evaluated (see reproduction_check.md).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$HERE"
mkdir -p logs

# 0. vendored pre-fix twin (UPF_NDT v0.3.0) and pre-fix switching costs (profiling thesis-v1.1),
#    used ONLY to reproduce manuscript Table I
if [ ! -d vendor/upf_digital_twin_v0.3.0/upf_digital_twin ]; then
  mkdir -p vendor
  git -C "$HOME/UPF_NDT" archive v0.3.0 src/upf_digital_twin | tar -x -C vendor/
  mv vendor/src vendor/upf_digital_twin_v0.3.0
fi
if [ ! -f vendor/switching_costs_thesis-v1.1.yaml ]; then
  git -C "$HOME/UpfProfilingCampaign/UpfProfilingCampaign" show \
    07e983267eb0cc6a97c3ec103e631175bb8f23fb:configs/twin_export/switching_costs.yaml > vendor/switching_costs_thesis-v1.1.yaml
fi
echo "e1456485051ead966a8fc87be7fb30a1  vendor/switching_costs_thesis-v1.1.yaml" | md5sum -c -

# 1. inventory + hashes; fold-bundle reconstruction and sanity checks (fails on any check)
$PY s00_inventory.py                          > logs/s00_inventory.log 2>&1
$PY s01_build_bundles.py 20                   > logs/s01_build_bundles.log 2>&1

# 2. reproduction check: pinned twin v0.4.0 and manuscript Table I under twin v0.3.0
$PY evaluate_worlds.py --tag main --worlds pinned --exp A B --workers 16          > logs/eval_main_pinned.log 2>&1
$PY evaluate_worlds.py --tag table1_v030 --twin v030 --pool search --worlds pinned --exp A \
    --workers 16 --no-steps                                                        > logs/eval_table1_v030.log 2>&1
$PY s02_reproduction_check.py                                                      > logs/s02_reproduction_check.log 2>&1
$PY s00_inventory.py                                                               > logs/s00_inventory.log 2>&1   # adds cohort T
$PY evaluate_worlds.py --tag main --worlds pinned --exp A B --cohorts T --workers 16 > logs/eval_main_pinned_T.log 2>&1

# 3. alternative calibration worlds: 10 primary (deployed refits) + 10 secondary (nested); A then B
$PY evaluate_worlds.py --tag main --worlds primary secondary --exp A B --workers 30 > logs/eval_main_worlds.log 2>&1

# 4. secondary: recalibrated hysteresis; analysis; figures
$PY s03_recalibrated_hysteresis.py 21         > logs/s03_recalibrated_hysteresis.log 2>&1
$PY s04_analyze.py                            > logs/s04_analyze.log 2>&1
$PY s05_figures.py                            > logs/s05_figures.log 2>&1
echo "done"
