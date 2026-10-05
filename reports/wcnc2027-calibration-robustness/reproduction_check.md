# Reproduction check (run before any alternative calibration world)

Script: `s02_reproduction_check.py` (log `logs/s02_reproduction_check.log`). Raw
evaluations: `results/main/pinned__{A,B}.csv` (pinned twin v0.4.0) and
`results/table1_v030/pinned__A.csv` (pre-fix twin v0.3.0). Detailed tables:
`reproduction/part1_pinned_v040_checks.csv`, `reproduction/part2_n8_per_seed_traceability.csv`,
`reproduction/part2_table1_rows.csv`, `reproduction/table1_centralized_search_top20.csv`.

## Outcome

| Gate | Result |
|---|---|
| **G1 — evaluator fidelity under the pinned twin (v0.4.0)** | **PASS**, 843/843 checks |
| **G2 — manuscript Table I under the twin it was computed with (v0.3.0)** | **PASS** for MAPPO, IPPO, hysteresis (b = 50) and always-DPDK. **FAIL for centralized PPO: its checkpoints are unavailable.** |

**Diagnosis of the G2 failure.**

- **No evaluator discrepancy.** The same evaluator reproduces every other row of Table I and every recorded per-seed value.
- **The centralized-PPO row cannot be regenerated from any surviving checkpoint.** An exhaustive search covered 73 centralized checkpoints (best-of-validation, final and `best_model.zip` files of every run, including runs made after the manuscript) and all 433,762 combinations of four distinct seeds. No combination matches even the reward mean and SD (−8,542 ± 1,744). The closest combination is −8,576 ± 1,697 with 0.50 Sw/d against the printed 0.44, and it mixes a run trained today.
- **What this rules out.** Robustness can be tested for the manuscript's MAPPO, IPPO, hysteresis and always-DPDK results exactly, but **not for the manuscript's centralized-PPO row**. Centralized PPO is represented in the robustness study by available checkpoint sets, labelled as substitutes (`analysis_plan.md`).

Because the failure is fully diagnosed as missing checkpoints and not an evaluation error, the alternative-calibration study was run, with this limitation stated wherever centralized PPO appears.

## Tolerances

| Reference | Tolerance |
|---|---|
| Recorded full-precision values (Task-C per-step rollouts, `multiseed_summary_v04twin.json`, `multiseed_summary_n8.json`) | ≤ 0.01 reward units and ≤ 0.01 Wh; rates and switch counts exact (≤ 1e-9); per-step action arrays identical |
| Printed Table I values (rounded) | half a unit of the last printed digit: reward mean and SD ± 0.5, Wh ± 0.5, QoSv% ± 0.005, USR% ± 0.05, Sw/d ± 0.005. SD is the population SD (ddof = 0), which the n8 record shows the table used (MAPPO 97.6 → "98") |
| Replay vs closed loop in the same world | relative difference ≤ 1e-12, identical actions |

## Part 1 — pinned twin as pinned in the repository (UPF_NDT v0.4.0 @ 95de456, switching costs thesis-v1.2)

| Check | Items | Passed | Largest absolute difference |
|---|---|---|---|
| 1a Task-C per-step rollouts: reward, energy Wh, the four reward components (energy, QoS, switching, cooldown), QoS-violation rate, USR share, switches | 72 rollouts × 9 metrics | 648/648 | 7.3e-12 |
| 1a per-step (T × K) action arrays | 72 | 72/72 | identical |
| 1b `multiseed_summary_v04twin.json` per seed: reward, Wh, unsafe rate, USR rate, switches, 10 per-cluster rewards (MAPPO 8, IPPO 8, centralized PPO 4, always-DPDK) | 122 | 122/122 | 1.5e-11 |
| 1c Experiment-B replay of the pinned actions in the pinned world equals Experiment A | 91 controllers × 11 rows × 18 metrics | 1/1 | 1.6e-15 relative (floating-point round-off) |

**Reward components.** The manuscript prints no per-component values, so components were checked against the full-precision Task-C per-step records (1a). Under both twin versions they are saved for every checkpoint in `results/main` and `results/table1_v030`.

## Part 2 — manuscript Table I (`reports/paper-mascots/paper.tex`, sha256 7d72e12d…)

**Twin version.** Table I predates twin v0.4.0 (`reports/phase-7/switching_fix_reeval.md`). It was re-evaluated with the vendored UPF_NDT **v0.3.0** (@171146a, `vendor/upf_digital_twin_v0.3.0`) and the pre-fix `switching_costs.yaml` of profiling **thesis-v1.1** (md5 e1456485…, the DVC-locked file of that period, `vendor/`). The steady-state surrogate bundle is the same pinned bundle in both versions; the two differ only in the switching-energy spike (`git diff v0.3.0 v0.4.0 -- src` is one line).

### 2a Per-seed traceability to `multiseed_summary_n8.json`

A match requires the total reward and all 10 per-cluster rewards to agree within 0.01. The largest observed difference is 2.7e-12.

| Seed | MAPPO checkpoint (Table I) | IPPO checkpoint (Table I) |
|---|---|---|
| 1 | `mappo_pool_off_seed1_20260526T162327` ¹ | `ippo_nr_seed1` |
| 7 | `mappo_pool_off_seed7_20260526T162327` | `ippo_nr_seed7` |
| 13 | `mappo_pool_off_seed13_20260526T162327` | `ippo_nr_seed13` |
| 23 | `mappo_nr_seed23` | `ippo_nr_seed23` |
| 42 | `mappo_seed42_20260526T143637` ¹ | `ippo_nr_seed42` |
| 64 | `mappo_nr_seed64` | `ippo_nr_seed64` |
| 77 | `mappo_pool_off_seed77_20260526T162327` ¹ | `ippo_nr_seed77` |
| 99 | `mappo_nr_seed99` | `ippo_nr_seed99` |

¹ Several byte-identical copies match: the sha256 is the same for every matching file of a seed. The earliest-created directory is listed; all copies are in `part2_n8_per_seed_traceability.csv`.

**Correction to the Task-C audit.** The earlier audit (`reports/wcnc2027-shared-policy-ablation/feasibility_and_results.md` §1.2) reported that seeds 7, 13 and 77 of the manuscript's MAPPO row had no surviving checkpoint. They do survive, in the scenario-variant sweep directories (`mappo_pool_off_*`, `mappo_lambda_sw_4_*`, `mappo_budget_off_*`), which that audit did not search.

**Correction to `reports/phase-7/switching_fix_reeval.md`.**
- That note reports the v0.4 re-score as a change of +8.0 (MAPPO) and +23.4 (IPPO) for the same checkpoints. It is not a re-score of the same checkpoints: `evaluate_multiseed.discover_checkpoints` selected the May `mappo_seed*` and `ppo_single_site_ensemble_seed*` runs, not the Table I checkpoints above.
- The Table I checkpoints themselves shift by −2.3 (MAPPO, −5,655.3 → −5,657.5) and −1.6 (IPPO, −5,812.1 → −5,813.6). Their actions, USR share, switches and QoS-violation rates are identical under both twin versions; only the switching penalty (≤ 4.2 reward units per run) and Wh (≤ 1.05 Wh) change.

### 2b Table I rows

| Row | Metric | Table I | Reproduced | \|diff\| | Tol. | Pass |
|---|---|---|---|---|---|---|
| MAPPO (n = 8) | reward | −5655 | −5655.25 | 0.25 | 0.5 | ✓ |
| | SD | 98 | 97.64 | 0.36 | 0.5 | ✓ |
| | Wh | 1966 | 1965.80 | 0.20 | 0.5 | ✓ |
| | QoSv % | 0.54 | 0.5401 | 0.0001 | 0.005 | ✓ |
| | USR % | 8.1 | 8.071 | 0.029 | 0.05 | ✓ |
| | Sw/d | 2.92 | 2.9162 | 0.0038 | 0.005 | ✓ |
| IPPO (n = 8) | reward | −5812 | −5812.08 | 0.08 | 0.5 | ✓ |
| | SD | 110 | 109.81 | 0.19 | 0.5 | ✓ |
| | Wh | 1973 | 1973.23 | 0.23 | 0.5 | ✓ |
| | QoSv % | 0.57 | 0.5736 | 0.0036 | 0.005 | ✓ |
| | USR % | 8.2 | 8.174 | 0.026 | 0.05 | ✓ |
| | Sw/d | 1.99 | 1.9909 | 0.0009 | 0.005 | ✓ |
| Hysteresis (b = 50) | reward | −6261 | −6261.31 | 0.31 | 0.5 | ✓ |
| | Wh | 1995 | 1994.59 | 0.41 | 0.5 | ✓ |
| | QoSv % | 0.68 | 0.6838 | 0.0038 | 0.005 | ✓ |
| | USR % | 7.5 | 7.473 | 0.027 | 0.05 | ✓ |
| | Sw/d | 1.06 | 1.0561 | 0.0039 | 0.005 | ✓ |
| Always-DPDK | reward | −7201 | −7201.08 | 0.08 | 0.5 | ✓ |
| | Wh | 2071 | 2070.72 | 0.28 | 0.5 | ✓ |
| | QoSv % | 0.38 | 0.3766 | 0.0034 | 0.005 | ✓ |
| | USR %, Sw/d | 0.0, 0.00 | 0, 0 | 0 | – | ✓ |
| **Centralized PPO (n = 4)** | reward | −8542 | closest −8576.08 | 34.1 | 0.5 | **✗** |
| | SD | 1744 | closest 1696.5 | 47.5 | 0.5 | **✗** |
| | Wh / QoSv % / USR % / Sw/d | 2066 / 1.18 / 4.3 / 0.44 | 2070.5 / 1.135 / 3.90 / 0.497 | – | – | **✗** |

The hysteresis row corresponds to t_up = 81 Mbps, t_down = 31 Mbps and **cooldown 1 step**, as in the evaluated code and `hyst_sweep/band_50.json`. The manuscript text describes a "four-step cooldown"; that text does not match the evaluated controller.

## What the robustness study uses as the reference world

- **Reference world.** The pinned twin **v0.4.0**, the version pinned by commit SHA in the repository and the one the fold bundles plug into. Its values for the Table I checkpoints differ from Table I only by the corrected switching penalty: MAPPO −5,657.5, IPPO −5,813.6, hysteresis −6,262.1, always-DPDK −7,201.1.
- **Why not v0.3.0.** It is used only for this reproduction check.
- **Normalisation statistics.** None exist for any controller (no VecNormalize or equivalent file under `experiments/`), so none are loaded or recomputed.
