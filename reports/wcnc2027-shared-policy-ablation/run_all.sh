#!/usr/bin/env bash
# Exact commands for every training run of the WCNC 2027 seed-completion and
# Shared-PPO ablation study. Run from anywhere; paths are absolute.
#
#   bash run_all.sh train_a     # MAPPO, Shared-PPO, centralized PPO (24 single-thread processes)
#   bash run_all.sh train_b     # IPPO (8 seeds x 10 single-site PPOs, 3 clusters in parallel per seed)
#   bash run_all.sh evaluate    # deterministic test rollouts for all controllers + legacy + baselines
#   bash run_all.sh analyze     # statistics, tables, figures
#
# Nothing here overwrites existing experiments: all runs go to
# experiments/wcnc2027_ablation/<controller>_seed<N>/ and each script refuses
# nothing silently (a pre-existing run dir aborts the launch, see guard below).
set -euo pipefail
ROOT=/home/ubuntu/UpfRLControllers
PY=$ROOT/.venv/bin/python
OUT=$ROOT/experiments/wcnc2027_ablation
REP=$ROOT/reports/wcnc2027-shared-policy-ablation
SEEDS="1 7 13 23 42 64 77 99"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
mkdir -p "$OUT/logs"
cd "$ROOT"

guard() { if [ -e "$1" ]; then echo "refusing to overwrite existing $1" >&2; exit 1; fi; }

case "${1:-}" in
  train_a)
    for s in $SEEDS; do
      guard "$OUT/mappo_seed$s"; guard "$OUT/shared_ppo_seed$s"; guard "$OUT/central_seed$s"
    done
    for s in $SEEDS; do
      /usr/bin/time -v -o "$OUT/logs/mappo_seed$s.time" \
        $PY scripts/train_mappo.py --seed "$s" --no-progress --out-dir "$OUT/mappo_seed$s" \
        > "$OUT/logs/mappo_seed$s.log" 2>&1 &
      /usr/bin/time -v -o "$OUT/logs/shared_ppo_seed$s.time" \
        $PY scripts/train_shared_ppo.py --seed "$s" --no-progress --out-dir "$OUT/shared_ppo_seed$s" \
        > "$OUT/logs/shared_ppo_seed$s.log" 2>&1 &
      /usr/bin/time -v -o "$OUT/logs/central_seed$s.time" \
        $PY scripts/train_ppo_multi_site.py --seed "$s" --no-progress --out-dir "$OUT/central_seed$s" \
        > "$OUT/logs/central_seed$s.log" 2>&1 &
    done
    wait
    ;;
  train_b)
    for s in $SEEDS; do guard "$OUT/ippo_seed$s"; done
    for s in $SEEDS; do
      /usr/bin/time -v -o "$OUT/logs/ippo_seed$s.time" \
        $PY scripts/train_ppo_ensemble.py --seed "$s" --n-parallel 3 --out-dir "$OUT/ippo_seed$s" \
        > "$OUT/logs/ippo_seed$s.log" 2>&1 &
    done
    wait
    ;;
  evaluate)
    cd "$REP" && $PY evaluate_all.py --only all --workers 24
    ;;
  analyze)
    cd "$REP" && $PY analyze.py && $PY make_figures.py
    ;;
  *)
    echo "usage: $0 {train_a|train_b|evaluate|analyze}" >&2; exit 2 ;;
esac
