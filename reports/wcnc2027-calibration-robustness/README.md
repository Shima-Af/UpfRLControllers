# wcnc2027-calibration-robustness

Zero-shot calibration-robustness evaluation of the saved UPF controllers (MAPPO, IPPO,
centralized PPO, hysteresis, static policies). Twenty alternative steady-state surrogate
calibrations are derived from the existing profiling data: 5 LOSO and 5 LOLO folds, each
in a deployed-configuration and a nested-selection refit. No training, no new
measurements, and no changes outside this directory.

**Start with `calibration_robustness_report.md`.** Its verdict is **B — partially stable**.

## One-command reproduction

```bash
bash reports/wcnc2027-calibration-robustness/run_all.sh
```

- **Where it runs.** From the repository root, with the controller virtualenv at `.venv` (Python 3.12.3).
- **What it needs.** The sibling repositories `~/UPF_NDT` (tag v0.3.0, used only to reproduce Table I) and `~/UpfProfilingCampaign/UpfProfilingCampaign` (thesis-v1 data, DVC-verified by the held-out validation code), plus the existing checkpoints under `experiments/`.
- **Runtime.** About 70 min on 32 CPU cores. No GPU is used.

| Step | Command (inside `run_all.sh`) | Wall time here |
|---|---|---|
| 0 | vendor twin v0.3.0 and switching costs thesis-v1.1 (md5-checked) | seconds |
| 1 | `s00_inventory.py` — checkpoints, seeds, hashes | 20 s |
| 1 | `s01_build_bundles.py 20` — rebuild 20 fold bundles, run 146 checks (exits non-zero on failure) | 21 s |
| 2 | `evaluate_worlds.py --tag main --worlds pinned --exp A B` | 5.0 min |
| 2 | `evaluate_worlds.py --tag table1_v030 --twin v030 --pool search --worlds pinned --exp A --no-steps` | 4.7 min |
| 2 | `s02_reproduction_check.py` — gate; see `reproduction_check.md` | 2 min |
| 2 | `evaluate_worlds.py --tag main --worlds pinned --exp A B --cohorts T` | 1.1 min |
| 3 | `evaluate_worlds.py --tag main --worlds primary secondary --exp A B --workers 30` | 46 min |
| 4 | `s03_recalibrated_hysteresis.py 21` | 18 s |
| 4 | `s04_analyze.py` | 18 s |
| 4 | `s05_figures.py` | 12 s |

**Determinism.** Evaluation is deterministic: argmax actions, deterministic reset, and single-threaded torch and sklearn. Re-running reproduces every CSV exactly. Bundle refits are deterministic (fixed `random_state`) and are checked against the saved held-out predictions on every run.

## Files

| File | Content |
|---|---|
| `feasibility_report.md` | Paths, checkpoint and seed inventory, missing components, runtime, compatibility risks |
| `reproduction_check.md` | Pinned-twin and Table I reproduction, tolerances, diagnosis of the untraceable centralized-PPO row |
| `analysis_plan.md` | Pre-registered controller sets, statistics, verdict rule, boundary definitions, trace-selection rule |
| `calibration_robustness_report.md` | Results and final verdict |
| `evaluation_results.csv` | **Experiment A** (closed-policy zero-shot): one row per world × checkpoint (fleet totals), all metrics and `diff_from_pinned_*` |
| `fixed_action_replay.csv` | **Experiment B** (fixed-action replay of the pinned trajectory), same columns |
| `per_cluster_results.csv` | Both experiments, per cluster (0–9) |
| `rank_stability.csv` | Per set × experiment × world: reward, energy and QoS rankings; MAPPO−IPPO, −hysteresis, −DPDK and −centralized differences; sign changes; ratios; Kendall τ-b and Spearman vs pinned; Pareto sets and relations |
| `controller_world_summary.csv` | Seed mean, SD, median, min and max of every metric per set × controller × world × experiment |
| `fold_summary.csv` | Best, median and worst fold and all five fold values per protocol |
| `uncertainty_decomposition.csv` | Seed uncertainty (within world) and calibration uncertainty (across worlds), reported separately |
| `verdict.json`, `analysis_numbers.json` | Pre-registered verdict counts for every set, tier and experiment |
| `recalibrated_hysteresis.csv` | Secondary: per-world re-derived thresholds and band sweep |
| `decision_boundary.csv`, `decision_boundary_summary.csv` | 50–600 Mbps decisions, action changes vs pinned, USR use, world-predicted safety, usage near 100/200/400/600 Mbps |
| `boundary_measured_labels.csv`, `boundary_usage_intervals.csv` | Per-world false-safe and false-unsafe counts at measured profiling levels, next to controller usage |
| `bundle_checks.csv`, `bundle_thresholds.csv` | Bundle sanity checks and derived break-even, QoS and decision thresholds per world |
| `calibration_summary.tex` | IEEEtran `table*` (compiles without overfull boxes) |
| `figures/` | `fig1`–`fig5`, PDF and PNG, plus `_S2` variants; `fig5_selection.json` |
| `checkpoint_inventory.csv`, `hashes.json` | sha256 of every checkpoint, config, traffic array, pinned and fold model, and script; repository HEAD; twin commits |
| `repro/pip_freeze.txt`, `repro/host.txt` | Software versions and host |
| `reproduction/` | Detailed reproduction tables and `table1_traceability.json` |
| `bundles/` | Rebuilt fold bundles (same layout as the deployed `models/`; 271 MB) |
| `results/`, `rollouts/` | Raw per-world metrics (17 MB) and per-step rollouts (420 MB; regenerable) |
| `vendor/` | UPF_NDT v0.3.0 source and the pre-fix `switching_costs.yaml`, used only for the Table I reproduction |
| `logs/` | Logs of every step |

**Scripts:** `rb_common.py`, `s00_inventory.py`, `s01_build_bundles.py`, `evaluate_worlds.py`, `s02_reproduction_check.py`, `s03_recalibrated_hysteresis.py`, `s04_analyze.py`, `s05_figures.py`, `run_all.sh`.

## Software and host

Python 3.12.3, scikit-learn 1.8.0, numpy 2.4.4, pandas 3.0.3, scipy 1.17.1, torch 2.11.0 (CPU),
stable-baselines3 2.9.0, gymnasium 1.2.3, pettingzoo 1.26.1, joblib 1.5.3, and upf-digital-twin
0.4.0 @95de456 (installed) plus v0.3.0 @171146a (vendored). The host is an AMD EPYC 9274F with
32 threads and 251 GB RAM. The full list is in `repro/pip_freeze.txt`.

## Wording

The results show robustness **across alternative calibrations derived from the existing
profiling data**. They are not live validation, independent physical validation, or evidence
of deployed-network performance.
