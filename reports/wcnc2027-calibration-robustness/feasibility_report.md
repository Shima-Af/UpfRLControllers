# Feasibility report — calibration-robustness evaluation (zero-shot)

2026-09-14. Read-only audit of existing artifacts. The physical testbed is not
used or needed; no measurement is simulated, interpolated or replaced. No
controller is retrained or modified, and the deployed surrogate directory, the
manuscript, the original results and the held-out validation report are untouched.
All new files are in `reports/wcnc2027-calibration-robustness/`.

**Verdict: fully feasible except one component.**

- **Available.** Every controller and baseline behind the manuscript's MAPPO, IPPO, hysteresis and always-DPDK results; the pinned surrogate bundle; the traffic slice and the evaluator. The LOSO/LOLO fold models were not saved, but they were reconstructed deterministically and verified against the saved held-out predictions.
- **Unavailable.** The checkpoints behind the manuscript's **centralized-PPO row** (−8,542 ± 1,744, n = 4). The row cannot be tested; centralized PPO is represented by available substitute sets.
- **Runtime.** The full matrix needs about 1 h of wall time on this host, well under the 6 h time box, so the reduced matrix was not needed.

## 1. Exact paths

| Component | Path / identity | Notes |
|---|---|---|
| Evaluation script used for the manuscript | `research/phase7/evaluate_multiseed.py` (`rollout`, `_mappo_policy`, `_ensemble_policy`, `_centralised_policy`, `_const_policy`; hysteresis from `src/baselines/hysteresis.py::MultiAgentHysteresis`) | The new evaluator imports these loaders unchanged |
| Environment | `src/envs/multi_agent_upf_env.py` → 10 × `src/envs/single_site_upf_env.py` | reward = −(α·SEC + λ_qos·max(0, τ−Q) + λ_sw·E_sw + cooldown) |
| Configuration | `configs/scenario_rl.yaml` (sha256 in `hashes.json`), `configs/digital_twin_paths.yaml` | delay budget 200 µs, loss budget 5 pkts/interval, accounting `sub_step` |
| Held-out traffic slice | `data/external/traffic_forecaster/targets_test.npy`, `predictions_test.npy`; shape (1009, 1, 10), horizon index 0, α = 1.0, 15-min steps (10.5 days) | Actual load 1.65–5,796 Mbps (median 269); 70.8 % of cluster-steps in 50–600 Mbps |
| Reset seed / action mode | `env.reset(seed=42)` (reset is deterministic, no random start); argmax actions | |
| Observation order | `[current load, 8 history loads, forecast, prev action, prev Q, prev SEC, cooldown progress]` (14-d), Gbps | prev Q and prev SEC come from the surrogate, so observations change between worlds (Experiment A) |
| Action encoding | 0 = DPDK, 1 = USR | |
| Switching model | twin v0.4.0 `DigitalTwin.compute_step`; `data/external/profiling_twin/switching_costs.yaml` (thesis-v1.2): activation 24.0 s / 3.3 s, spike 0.009476 / 5.15e-6 Wh | Fixed in every world |
| **Pinned surrogate bundle** | `data/external/profiling_twin/models/` (`models.dvc`: UpfProfilingCampaign @a73870f, dir md5 a92ad2ca…, 32 files); the twin loads the 10 `lite` models of variants `dpdk` and `usr_full` | Identical in twin v0.3.0 and v0.4.0 |
| Twin code | installed `upf_digital_twin` v0.4.0 @95de456 (repository pin). Manuscript Table I used v0.3.0 @171146a, vendored for the reproduction check only (`vendor/`) | v0.3.0 → v0.4.0 changes one line (switching spike) |
| **Code path for surrogates** | `SingleSiteUPFEnv.__init__` → `DigitalTwin(scenario_cfg, paths_cfg)` → `UPFProfile(models_dir = paths_cfg.profiling_twin.models, manifest = .../manifest.json)` → `evaluate_batch(UPF, actual_load)` precomputes power, delay, loss, throughput and `is_safe` for the whole slice. Layer 1: X = [load·1e6, load·1e6] kbit/s (UL := DL) → throughput, cpu, loss, delay. Layer 2: [X, L1 outputs] → power, clipped at 0. `is_safe` = loss ≤ 5 and delay ≤ 200 µs | **A calibration world is selected only through `paths_cfg`; no code is patched** |
| LOSO / LOLO validation outputs | `reports/twin-validation-heldout-wcnc2027/`: `heldout_predictions.csv.gz`, `heldout_model_selection.json` (200 records), `threshold_stability.csv`, `threshold_grid_predictions.csv.gz`, scripts `s02_heldout_cv.py`, `s02b_threshold_stability.py` | Fitted fold models were **not** saved |
| Hysteresis / static | `src/baselines/hysteresis.py` (t_up 81, t_down 31 Mbps, cooldown 1); always-DPDK `_const_policy(0)` | Always-USR was **not** in the original evaluation; it is reported only as a supplementary reference |
| Normalisation statistics | **None exist.** No VecNormalize or equivalent file under `experiments/` (`hashes.json: normalisation_files_found = []`). `src/trainers/ppo_single_site.py` states "no VecNormalize"; MAPPO checkpoints hold only actor, critic, config and dimensions | Nothing to preserve or recompute |

## 2. Checkpoint and seed inventory

`checkpoint_inventory.csv` lists every checkpoint with its sha256.

| Family | Cohort | Seeds | Checkpoints | Role |
|---|---|---|---|---|
| MAPPO | **T** — reproduce Table I per seed and per cluster | 1, 7, 13, 23, 42, 64, 77, 99 | `mappo_pool_off_seed{1,7,13,77}_20260526T162327`, `mappo_nr_seed{23,64,99}`, `mappo_seed42_20260526T143637` (`mappo_best.pt`) | S1 primary |
| IPPO | **T** | 1, 7, 13, 23, 42, 64, 77, 99 | `ippo_nr_seed*` (10 × `ppo_single_site.zip`) | S1 primary |
| Centralized PPO | Table I row | — | **unavailable**: 73 candidates, 433,762 four-seed combinations, no match (`reproduction_check.md`) | — |
| Centralized PPO | P — `multiseed_summary_v04twin.json` | 7, 13, 42, 99 | `ppo_multi_site_seed*_20260517T004954` | S1 substitute, S3 |
| MAPPO / IPPO | P | 8 + 8 | May 2026 runs (`LEGACY_MAPPO_V04`, `ppo_single_site_ensemble_seed*`) | S3 |
| MAPPO / IPPO / centralized PPO (+ Shared-PPO) | R — same-code retrain 2026-09-14 | 8 each | `experiments/wcnc2027_ablation/*` | S2 |
| Others | L | various | 6 MAPPO, 15 centralized-PPO (June/August `central_nr`, May-15/16 runs, 4 runs with only `best_model.zip`) | evaluated, not ranked |

- **Totals.** 88 distinct checkpoints: 29 MAPPO (including the 4 traced `mappo_pool_off` checkpoints), 24 IPPO, 27 centralized PPO and 8 Shared-PPO. Byte-identical copies (`ppo_multi_site_seed{1,23,64,77}_bridge` = `central_nr_seed*`; several `mappo_pool_off` / `lambda_sw_4` / `budget_off` duplicates) are listed once.
- **Seed counts versus expectation.** The expected counts were 8 MAPPO, 8 IPPO and 4 centralized PPO. Verified: 8 and 8 for MAPPO and IPPO. The manuscript's 4 centralized seeds are unrecoverable; substitute sets have 4 (P) or 8 (R) seeds.

## 3. Missing components and how each is handled

| Missing | Handling |
|---|---|
| Manuscript centralized-PPO checkpoints | Not testable. Substitutes are used and labelled; no claim is made about the manuscript row itself |
| Fitted LOSO/LOLO fold models | **Reconstructed** with the validation study's own `refit_fold` and recorded selections on the DVC-verified profiling data, saved in `bundles/` (never in the deployed directory). 146/146 checks pass (`bundle_checks.csv`): feature order and units, variant mapping, **exact reproduction of the saved held-out predictions (≤ 1e-9)**, equality through the env code path, grid predictions, finiteness, power clip, and derived thresholds equal to `threshold_stability.csv` |
| Measured QoS labels between profiling levels | Not created. False-safe and false-unsafe are evaluated only at the measured design levels (60, 80, 100, 200, 400, 600 Mbps) |
| Per-step reward components in the manuscript record | Not recorded there. Recomputed for every rollout; verified against the Task-C per-step records |
| Always-USR in the original evaluation | Absent. Reported only as a supplementary reference |

## 4. Runtime (measured on 32-core AMD EPYC 9274F, CPU only)

| Step | Measured |
|---|---|
| Bundle reconstruction (20 bundles) + 146 checks | 9 s + 12 s |
| Pinned world, 91 controllers, Experiments A + B | 5.0 min (16 workers) |
| Table I search under twin v0.3.0, 174 episodes | 4.7 min (14 workers) |
| One episode | ≈ 1.3 s (MAPPO, centralized PPO, baselines) and ≈ 35–50 s (IPPO: 10 SB3 `predict` calls per step), plus ≈ 6 s environment build per world and process |
| Recalibrated hysteresis, 21 worlds × 5 bands | 18 s |
| **Full matrix: 20 worlds × 91 controllers × 2 experiments = 3,640 episodes** | estimate ≈ 1 h wall (30 workers), IPPO-dominated |

## 5. Compatibility and serialization risks

- **scikit-learn version.** The pinned models were fitted with 1.7.0; the controller environment runs 1.8.0 (the version used for the paper). The validation study measured a largest prediction difference of 2.3e-10 across 1.7.0 / 1.7.2 / 1.8.0. The fold bundles are pickled with 1.8.0 and must be loaded with it.
- **Library versions.** Python 3.12.3, torch 2.11.0, stable-baselines3 2.9.0, numpy 2.4.4, pandas 3.0.3, gymnasium 1.2.3, pettingzoo 1.26.1. SB3 zips load and reproduce the recorded per-step actions exactly (reproduction check 1a).
- **Twin version.** Alternative worlds use twin v0.4.0, the repository pin. Table I itself was computed with v0.3.0. For the Table I checkpoints, actions, USR share, switches and QoS-violation rates are identical under both versions; reward differs by ≤ 4.2 units per run.
- **Extrapolation in LOLO worlds.** Each LOLO fold removes whole load levels: f0 removes 0, 0.006, 0.06, 0.6 and 4 Gbps; f1 removes 0.0001, 0.008, 0.08, 0.8 and 5 Gbps; f2 removes 0.001, 0.01, 0.1 and 1 Gbps; f3 removes 0.002, 0.02, 0.2 and 2 Gbps; f4 removes 0.004, 0.04, 0.4 and 3 Gbps. The same levels are removed for DPDK and USR. Surrogates are then used between or beyond their training levels on some of the RL traffic (test loads reach 5.8 Gbps).
- **Degenerate secondary worlds.** In two nested-selection LOLO worlds (f2, f3) the derived USR QoS limit is 0 Mbps: USR is predicted unsafe over the whole 1–500 Mbps grid. These worlds are secondary and are reported, not removed.
- **Validation relies on the same fold models.** Folds within a protocol share ~80 % of the profiling runs, so the fold worlds are not independent calibration campaigns and are not treated as such.
- **Disk.** Bundles take 271 MB. Per-step rollouts take 420 MB; they are regenerable with the README commands.
