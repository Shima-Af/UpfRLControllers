#!/usr/bin/env bash
# Launch one of the Phase-8 sweeps via the overlay system.
#
# Usage:
#   scripts/run_sweep.sh <family> [seeds] [timesteps] [device]
#
# Examples:
#   scripts/run_sweep.sh pool 42 200000 cuda          # 1 seed
#   scripts/run_sweep.sh pool 1,7,42 200000 cuda      # 3 seeds (5 pool x 3 = 15 runs)
#   scripts/run_sweep.sh lambda_sw 42 200000 cuda     # 4 runs
#   scripts/run_sweep.sh cooldown 42 200000 cuda      # 4 runs
#   scripts/run_sweep.sh all 42 200000 cuda           # everything, 1 seed
#
# Output: each run lands in experiments/mappo_<overlay_tag>_seed<seed>_<utc-ts>/
# Per-run log (full stdout) lands next to the output dir as <out>.log.
#
# Sequential by design — change RUN_PARALLEL=1 below to background runs,
# but be mindful of GPU memory if you do.

set -euo pipefail

FAMILY="${1:?family required: pool | lambda_sw | cooldown | all}"
SEEDS="${2:-42}"
TIMESTEPS="${3:-200000}"
DEVICE="${4:-cuda}"
RUN_PARALLEL="${RUN_PARALLEL:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

case "$FAMILY" in
    pool)       GLOB="configs/sweeps/pool_*.yaml" ;;
    lambda_sw)  GLOB="configs/sweeps/lambda_sw_*.yaml" ;;
    cooldown)   GLOB="configs/sweeps/cooldown_*.yaml" ;;
    budget)     GLOB="configs/sweeps/budget_*.yaml" ;;
    all)        GLOB="configs/sweeps/*.yaml" ;;
    *) echo "Unknown family '$FAMILY'. Use: pool | lambda_sw | cooldown | budget | all" >&2; exit 2 ;;
esac

OVERLAYS=( $(ls $GLOB 2>/dev/null) )
if [ ${#OVERLAYS[@]} -eq 0 ]; then
    echo "No overlays matched '$GLOB'." >&2
    exit 1
fi

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
TS=$(date -u +%Y%m%dT%H%M%S)
SWEEP_LOG_DIR="experiments/sweep_${FAMILY}_${TS}"
mkdir -p "$SWEEP_LOG_DIR"

echo "Family:     $FAMILY"
echo "Seeds:      ${SEED_LIST[*]}"
echo "Timesteps:  $TIMESTEPS"
echo "Device:     $DEVICE"
echo "Overlays:   ${#OVERLAYS[@]} files"
for o in "${OVERLAYS[@]}"; do echo "  $o"; done
echo "Total runs: $(( ${#OVERLAYS[@]} * ${#SEED_LIST[@]} ))"
echo "Log dir:    $SWEEP_LOG_DIR"
echo "---"

total_runs=$(( ${#OVERLAYS[@]} * ${#SEED_LIST[@]} ))
echo
echo "Monitor live progress with:"
echo "    tail -f $SWEEP_LOG_DIR/CURRENT.log"
echo "Or aggregate progress across runs with:"
echo "    grep -c 'Training finished' $SWEEP_LOG_DIR/*.log"
echo

run_idx=0
for overlay in "${OVERLAYS[@]}"; do
    tag=$(basename "$overlay" .yaml)
    for seed in "${SEED_LIST[@]}"; do
        run_idx=$((run_idx + 1))
        out_dir="experiments/mappo_${tag}_seed${seed}_${TS}"
        log_file="${SWEEP_LOG_DIR}/${tag}_seed${seed}.log"
        echo "[$(date -u +%H:%M:%S)] (#${run_idx}/${total_runs}) tag=$tag seed=$seed -> $out_dir"

        # Make the active log easy to tail regardless of which run is current.
        ln -sf "$(basename "$log_file")" "${SWEEP_LOG_DIR}/CURRENT.log"

        cmd=(
            python scripts/train_mappo.py
            --total-timesteps "$TIMESTEPS"
            --seed "$seed"
            --device "$DEVICE"
            --config-overlay "$overlay"
            --out-dir "$out_dir"
            --no-progress
        )

        if [ "$RUN_PARALLEL" = "1" ]; then
            "${cmd[@]}" > "$log_file" 2>&1 &
        else
            "${cmd[@]}" > "$log_file" 2>&1
            tail -n 3 "$log_file" | sed 's/^/    /'
        fi
    done
done

# Remove the symlink once the sweep is done so it doesn't confuse later runs.
rm -f "${SWEEP_LOG_DIR}/CURRENT.log"

if [ "$RUN_PARALLEL" = "1" ]; then
    echo "All runs launched in background. Waiting..."
    wait
fi

echo "---"
echo "Sweep '$FAMILY' complete: $run_idx runs."
echo "Per-run logs:  $SWEEP_LOG_DIR/"
echo "Checkpoints:   experiments/mappo_*_${TS}/"
