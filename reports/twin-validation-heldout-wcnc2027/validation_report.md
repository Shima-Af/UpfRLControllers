# Held-out validation of the UPF digital-twin surrogates

Revision of *"A Measurement-Grounded Digital Twin for Evaluating Cooperative MARL
in Energy-Aware 5G UPF Orchestration"* (MASCOTS 2026 → IEEE WCNC 2027).
Generated 2026-09-14. Every number below is produced by the scripts in this
directory. Nothing is entered by hand, and no RL controller was retrained.

---

## 0. Bottom line

1. **Feasibility.** The repositories contain everything needed to validate the
   *surrogate modelling procedure* on genuinely held-out profiling runs, and that
   analysis is delivered here. They do **not** contain any held-out data for
   the *exact pickles the twin runs*. Those pickles were trained on a random
   80 % of the rows of **every one** of the 220 runs. Their held-out error can
   therefore only be estimated by re-running the same procedure with whole runs
   withheld, which is what this analysis does.
2. **The manuscript's validation numbers are not held-out numbers.** They come from
   a row-level random split in which **100 % of test rows share a run with training rows** (median 31 same-run
   training rows per test row). The model family was also selected by **test** R².
   The quoted USR R² values (0.99 / 0.97 / 1.00) belong to the *full-feature*
   models. The twin uses the *lite* (throughput-only) models.
3. **Held-out accuracy (leave-one-sweep-out, nested run-grouped CV).**
   - **USR power:** R² 0.971, MAE 0.085 W.
   - **USR delay:** R² 0.913 but MAE 1,088 µs, which is 5.4× the 200 µs budget.
   - **USR loss:** R² 0.995 but MAE 3,765 pkts/3 s, which is 753× the 5-pkt budget.
   - **DPDK power:** MAE 3.3 mW, RMSE 4.1 mW, R² 0.17. R² is uninformative because σ = 4.6 mW.
   - **DPDK delay** is not predictable from load: R² −0.21.
   
   The high USR R² values come from the saturated regime (loss up to 10⁶ pkts). In the
   operating region they are meaningless. Below 0.5 Gbps, USR loss R² is −210.
4. **Safety classification.**
   - **USR, all loads:** false-safe 2.4 %, false-unsafe 8.9 %.
   - **USR, decision-relevant band (50–447 Mbps):** false-safe 26.7 %, false-unsafe 19.1 %, precision (unsafe) 0.31.
   - **Unprofiled loads (leave-level-out):** false-unsafe rises to 56 % in that band.
5. **The USR QoS limit λ_qos = 149 Mbps is a tree-split artefact, not a resolved
   physical limit.** It is the last 1-Mbps grid point before the midpoint (150 Mbps)
   between the two profiled levels 100 and 200 Mbps. Nothing was measured between
   them. The pinned twin's USR safety map is also non-monotone: unsafe at
   150–199 Mbps, safe at 200 Mbps, unsafe again above. In the controller's test
   slice, 60 % of steps fall in the unprofiled 0.1–0.6 Gbps gaps.
6. **Hardware provenance cannot be closed from the repositories.** The DPDK and
   OAI/USR power series came from **two different Scaphandre exporters**
   (192.168.17.26 vs 192.168.17.94), ten days apart. That contradicts the thesis
   text "same bare-metal host". **No CPU or NIC model is recorded anywhere.**
7. **Thresholds.** `usr_safe_threshold_gbps: 0.5` is a training-subset filter for
   the `usr_safe` models, which the twin never loads. The 0.10 / 0.17 Gbps
   figures are exploratory observations from the profiling chapter. The paper's
   results used `configs/scenario_rl.yaml`: 200 µs and 5 pkts per 3-s interval,
   giving the derived 91 / 149 / 81 Mbps.
8. **Wording.** The evidence supports **"measurement-calibrated surrogate
   environment"**. It does not support "digital twin" without qualification
   (§13).

---

## 1. What was located (task item 1)

| Item | Location | Verified |
|---|---|---|
| Raw LoadCore runs | `UpfProfilingCampaign/data/raw/loadcore/{dpdk,usr}/` | 110 + 110 zip files. DVC lock: 220 files, md5 `69b3ba27…dir` |
| Raw Scaphandre | `UpfProfilingCampaign/data/raw/scaphandre/{dpdk,usr}/` | per-process power + CPU CSVs; `dpdk/upf_PIDs.txt` |
| Merged table | `data/interim/merged.csv` | 9,556 rows; md5 `3d6d4c6e…` = thesis-v1 lock |
| Processed training table | `data/processed/features.csv` | **8,781 rows** (DPDK 4,388 / USR 4,393); md5 `04497319…` = thesis-v1 lock |
| Pipeline code | `src/{ingest,merge,features,train}.py` | all four md5 = thesis-v1 lock |
| Original split | `src/train.py::main` | `train_test_split(test_size=0.2, random_state=42)` per variant, **row level**; no validation split |
| Saved models used by the twin | controller `data/external/profiling_twin/models/` (`dvc import` of profiling `a73870f`, tag thesis-v1) | DVC dir hash recomputed `a92ad2ca…dir` = pin = lock; identical to `UPF_NDT` copy |
| Variants the twin loads | `upf_digital_twin/twin/upf_profile.py` (v0.4.0, commit `95de456`, = controller venv install) | `dpdk` and `usr_full`, **lite** (inputs DL and UL kbit/s, twin sets UL := DL) |
| Existing held-out predictions | none | only the manifest's test metrics exist. `mlruns/` holds only 2026-08-26 re-runs, not the thesis-v1 run |
| Run identifiers / grouping | `run_dir` is dropped by `features.py` | **rebuilt** by re-executing `features.py` on `merged.csv` and asserting column-wise equality with `features.csv` (max \|Δ\| 2·10⁻¹⁵). An untracked `features_with_run.csv` from an earlier session agrees |
| Run structure | derived, asserted in `common.load_runs` | **220 runs** = 2 campaigns × **5 chronological sweeps × 22 offered-load levels** (≈0 kbit/s … 5 Gbps); 39–41 samples per run at **3.0 s** spacing; each run holds one constant load (median within-run CV 5·10⁻⁵) |
| Hardware / NIC metadata | **not found** | §3 |
| Power measurement scope | `merge.py`, raw CSV headers | §2 |
| QoS thresholds and units | `configs/scenario_rl.yaml`, `upf_profile.py` | §4 |

`s01_provenance_original_split.py` rebuilds the original split exactly. The pinned
pickles re-scored on it reproduce every metric in `manifest.json` to
≤2.3·10⁻¹⁵ (R²) and ≤1.8·10⁻¹² (MAE/RMSE) for all 20 lite and full slots
(`original_split_reproduction.csv`). Pickle predictions are identical under
scikit-learn 1.7.0 (fit version), 1.7.2 (twin venv) and 1.8.0 (controller venv,
used for the paper), with max \|Δ\| 2.3·10⁻¹⁰ (`sklearn_version_check.json`).

## 2. Power measurement scope

`power_watts` = Σ over UPF processes of Scaphandre's
`scaph_process_power_consumption_microwatts` ÷ 10⁶ (`merge.py`, `features.py`).

- **DPDK:** filtered to the 4 PIDs in `upf_PIDs.txt` (three `upf-epc-bess:1.5.0`
  containers and `upf-epc-pfcpiface:1.5.0`). All 4 PIDs are present in every
  hour of the campaign.
- **USR:** no PID file. The export contains a single process,
  `/openair-upf/bin/oai_upf` (PID 525735).
- **Only process-level series were exported.** No `scaph_host_power_microwatts`
  or package/domain series exists, so full-server or CPU-package power cannot be
  reconstructed.

The scope is therefore **Scaphandre process-attributed power**: RAPL-derived host
power apportioned to the UPF processes by Scaphandre. It is neither full-server
nor package power. The RAPL domains Scaphandre used are not recorded. Attribution
is consistent with DPDK reading only ≈0.82 W (≈1.06 % CPU share) despite poll-mode
operation, but that is an inference. The absolute watts are not comparable
to wall or package power, and are only comparable across the two UPFs if both
hosts attribute power the same way (see §3).

The manuscript's "RAPL power measurements collected through Scaphandre" should say
*process-attributed*. Activation energy comes from yet another machine at RAPL
package-0 scope, rebased onto the attributed idle power
(`switching_costs.yaml`).

## 3. Provenance risk 1: same host or different hardware?

**Conclusion: this cannot be determined conclusively from the repositories. The
instrument metadata point to two different hosts, and the CPU/NIC configuration of
neither is recorded.**

| Evidence | DPDK campaign | OAI/USR campaign |
|---|---|---|
| Scaphandre exporter (`instance` column, every row) | `192.168.17.26:2020` (all 11,349,250 rows) | `192.168.17.94:2020` (all 9,483 rows) |
| Measurement window (LoadCore, UTC) | 2025-01-03 16:25 → 2025-01-04 01:48 | 2025-01-13 13:08 → 21:01 |
| Scaphandre export | all host processes (filtered in `merge.py`) | UPF process only |
| UPF software | SD-Core BESS `upf-epc-bess:1.5.0` + `pfcpiface:1.5.0` (containers) | OAI `oai_upf` |
| LoadCore agents | 192.168.255.211/.212, agent MACs `94:6d:ae:01:01:01/02`, `…:02:03` | identical |
| LoadCore DUT addressing (`config-data.bin`) | UPF N3 10.21.52.101, N4 10.21.51.101, N6 10.21.53.101; SMF/AMF 192.168.131.12–13/16; DNN `internet`; UE pool 10.250.0.0/16 | UPF N3 10.21.52.255, N4 10.21.51.225, N6 10.21.53.225; AMF 10.110.174.55, SMF 10.110.71.184/24; DNN `oai`; UE route 12.1.1.0/24 |
| Traffic profile | stateless UDP, 1,250-byte payload | identical |
| CPU model / cores / SMT / RAM | **not recorded** | **not recorded** |
| NIC model / driver / firmware / link speed | **not recorded** | **not recorded** |
| Kernel, BIOS power policy, DPDK core pinning | **not recorded** | **not recorded** |

The documentation also contradicts itself:
- The profiling chapter (`chapter_upf_profiling.tex:95`) says "deployed on **identical server hardware**".
- The thesis chapter (`UpfThesisPipeline/.../profiling.tex:80`) says "profiled on the **same bare-metal host**".
- `switching_costs.yaml` and `CONTRACTS.md` state the activation measurements come from a third, different machine (the only NIC hint, `docker_setup_mellanox.sh`, belongs to that machine).

A single host changing its management IP between 3 and 13 January cannot be
excluded, but nothing supports it. The "same host" wording is not supported by the
data. "Identical hardware" can be neither confirmed nor refuted.

**Needed to close this:** `lscpu`, `dmidecode -t system,memory`, `lspci -nn | grep -i eth`,
`ethtool -i/-l <if>` and kernel/BIOS settings for 192.168.17.26 and .94 as of
January 2025, or a testbed inventory record. Also the Scaphandre version and RAPL
domains.

## 4. Provenance risk 2: which thresholds produced the paper's results?

| Value | Where | Meaning | Used for paper results? |
|---|---|---|---|
| `usr_safe_threshold_gbps: 0.5` | profiling `params.yaml` → `train.py` | Filters training rows by **delivered** throughput (a response) to build the `usr_safe` model variant | **No.** The twin maps USR → `usr_full` (`upf_profile.py::_VARIANT_MAP`); `usr_safe` pickles are never loaded |
| ≈0.10 Gbps "safe operating boundary", ≈0.17 Gbps "loss onset", ≈0.57 Gbps "near-total loss" | profiling README / chapter EDA | Descriptive observations | **No** (not read by any code) |
| `safe_capacity_gbps 0.69`, `danger_threshold_gbps 0.70` | `configs/twin_export/params.yaml` | Hand-authored snapshot | **No** (not read by twin or controller code; only the α = 0.1208 alternative-anchor note in `CONTRACTS.md`) |
| `delay_budget_us: 200.0`, `max_loss_pkts_per_interval: 5.0` | controller `configs/scenario_rl.yaml` (identical in `UPF_NDT/configs/scenario.yaml`) | `is_safe = predicted_loss ≤ 5 AND predicted_delay ≤ 200`; also the denominators of the graded QoS score | **Yes.** Unchanged since `c58204f` (2026-05-13), before every Phase-7 run |
| 91 / 149 / 81 Mbps | derived at run time (`threshold_derivation.py`) from the pinned lite pickles + the budgets above | break-even / QoS limit / decision threshold | **Yes.** Reproduced exactly (`threshold_stability.csv`, row "pinned") |

**Units.** Delay is the packet-count-weighted mean of LoadCore's DL
one-way-delay histogram, using bin midpoints (62.5, 187.5, 375, 750, 3,000, 7,500,
12,500, 17,500, 25,000 µs). It is therefore floor-censored at 62.5 µs and
ceiling-censored at 25,000 µs, and the 200 µs budget sits inside the second bin.
Loss is the per-row difference of LoadCore's cumulative `packets_lost` counter,
i.e. **packets per 3-s reporting interval**. The twin compares a *predicted
conditional mean* against the budget. The measured label below applies the budget
per 3-s sample (§9 discusses the difference).

**Two corrections to the manuscript.** "The budgets are set from the measured
testbed limits": the 5-pkt loss budget was set in commit `c58204f`, "Calibrate
QoS loss threshold to surrogate noise floor", to sit above the surrogate's
fractional loss predictions. It was not derived from a testbed limit. The load
passed to the surrogates is offered DL load in Gbps (α = 1.0 declared assumption),
not a normalised ρ.

## 5. Why the original split is not held-out

| | DPDK | USR |
|---|---|---|
| train / test rows | 3,510 / 878 | 3,514 / 879 |
| runs appearing in test / in train | 110 / 110 | 110 / 110 |
| test rows whose run also contributes training rows | **100 %** | **100 %** |
| median training rows from the same run per test row | 31 | 31 |
| test load levels present in training | 100 % | 100 % |

A run is ~2 min at one constant load, and consecutive 3-s samples are
near-duplicates. The original test error is therefore an in-sample interpolation
error. Its metrics essentially equal resubstitution: USR lite power R² 0.964 on
the test split vs 0.975 on all rows (`twin_validation_table.csv`, "NOT
held-out" rows). Selection added a second leak, choosing the family by best test
R² with hyper-parameters tuned by ungrouped KFold.

## 6. Held-out protocol (task item 3)

`s02_heldout_cv.py`. **Nothing is fitted or selected on a held-out run.** Both
facts are asserted in code for every fold.

**Replicated from `train.py`:**
- lite inputs;
- variants `dpdk` / `usr_full`;
- the Ridge / RandomForest / GradientBoosting candidate spaces, R² scoring, `random_state = 42`, and 20 random-search iterations;
- StandardScaler pipelines;
- two-layer stacking (out-of-fold Layer-1 predictions feed Layer 2);
- power clipped at 0 as in the twin.

**Changed to remove leakage:**
- outer folds withhold whole runs;
- family **and** hyper-parameters are selected by inner CV on the outer-training runs only;
- Layer-2 stacking features come from inner *grouped* out-of-fold predictions;
- 21 rows with a missing Layer-1 target get a real OOF prediction instead of 0.

| Protocol | Outer folds | Inner folds | What it tests |
|---|---|---|---|
| **LOSO** (primary) | the 5 chronological sweeps (each holds one run at all 22 levels, ≈78 min) | the 4 remaining sweeps | a new, independent run at a profiled load (run- and time-disjoint) |
| **LOLO** | levels ranked, fold = rank mod 5 (all 5 replicates of a level withheld; neighbours never withheld together) | training levels, rank mod 4 | interpolation to *unprofiled* loads, which is what the twin does for NetMob loads |

**Configurations** (all reported in the CSV):
- `nested` (primary);
- `deployed`: the pinned families and hyper-parameters, refit on training runs only. A sensitivity bracket; their selection upstream saw all runs.
- `lookup`: a non-ML reference that linearly interpolates the per-level means of the training runs.

**Test inputs** are twin-style (UL := DL); predictions from measured UL are stored
as a sensitivity. Every one of the 8,781 samples is predicted exactly once per
protocol × configuration.

**Held-out load range.**
- **LOSO:** ≈9 kbit/s to 5.0 Gbps for both UPFs. Every held-out level is profiled in training. Only 3 samples lie marginally outside the training range, exceeding the training maximum by ≤0.015 %.
- **LOLO:** the held-out 0 Gbps (DPDK, 198 samples) and 5 Gbps (DPDK 200, USR 200) levels are outside the training range, i.e. extrapolation.

## 7. Metric definitions

e = predicted − measured over held-out samples.
- **R²** = 1 − Σe² / Σ(y − ȳ)²; undefined when var(y) = 0.
- **MAE** = mean\|e\|, **RMSE** = √mean e², **MedAE** = median\|e\|, **P95AE** = 95th percentile of \|e\|, **bias** = mean e.
- **NMAE** = MAE / ȳ and **NRMSE** = RMSE / ȳ, i.e. normalised by the mean measured value of the held-out samples (the CV(RMSE) convention). "n/a" when ȳ = 0.
- The CSV also has range-normalised variants, ÷ (max y − min y), and MAE / QoS budget.
- **95 % CIs:** percentile cluster bootstrap over runs, 2,000 resamples.

**No MAPE is used anywhere.** 67 % of USR and 100 % of DPDK loss samples are zero.

**Safety** (positive class = unsafe):
- measured unsafe = loss > 5 **or** delay > 200 µs;
- predicted unsafe = predicted loss > 5 **or** predicted delay > 200 µs;
- samples with undefined measured delay (14 DPDK, 7 USR) are excluded;
- false-safe rate = FN/(TP+FN); false-unsafe rate = FP/(FP+TN).

**Near-boundary band:** runs whose profiled level is within a factor of 3 of
λ_qos = 149 Mbps, i.e. 49.7–447 Mbps (levels 60, 80, 100, 200, 400 Mbps; 25 runs).
The band was declared from the manuscript's own λ_qos before any held-out error was
computed. Per-level results are in `safety_by_load_level.csv`, so the band can be
re-aggregated freely.

## 8. Regression results

**Table A:** LOSO, nested, twin-style inputs, all loads (`twin_validation_table.md` / `.tex`).

| UPF | Output | Runs/samples | R² [95 % CI] | MAE [95 % CI] | RMSE | NMAE | NRMSE | MedAE | P95AE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DPDK | Power (W) | 110/4,388 | 0.175 [0.105, 0.233] | 0.00331 [0.00318, 0.00346] | 0.00415 | 0.4 % | 0.5 % | 0.00286 | 0.00800 |
| DPDK | Delay (µs) | 110/4,374 | −0.215 [−25.7, 0.008] | 471 [178, 918] | 2,192 | 152 % | 706 % | 110 | 2,179 |
| DPDK | Loss (pkts/3 s) | 110/4,388 | n/a (all 0) | 0 | 0 | n/a | n/a | 0 | 0 |
| OAI/USR | Power (W) | 110/4,393 | 0.971 [0.955, 0.983] | 0.0846 [0.058, 0.118] | 0.260 | 5.9 % | 18.1 % | 0.0114 | 0.351 |
| OAI/USR | Delay (µs) | 110/4,386 | 0.913 [0.858, 0.951] | 1,088 [724, 1,482] | 2,487 | 21.6 % | 49.4 % | 13 | 6,365 |
| OAI/USR | Loss (pkts/3 s) | 110/4,393 | 0.995 [0.992, 0.997] | 3,765 [2,205, 5,450] | 13,777 | 4.9 % | 17.8 % | 0.014 | 20,327 |

**By regime (USR, LOSO nested):**

| Subset | Power R² / MAE | Delay R² / MAE / P95AE | Loss R² / MAE / P95AE |
|---|---|---|---|
| levels ≤ 0.5 Gbps (75 runs) | 0.985 / 0.026 W | 0.197 / 18.6 µs / 102 µs | **−210** / 18.3 pkts / 18.0 pkts |
| near-boundary 50–447 Mbps (25 runs) | 0.967 / 0.070 W | 0.062 / 26.8 µs / 142 µs | **−175** / 29.9 pkts (6.0× budget) / 18.3 pkts |
| levels > 0.5 Gbps (35 runs) | 0.013 / 0.210 W | 0.672 / 3,382 µs / 9,136 µs | 0.993 / 11,810 pkts / 46,550 pkts |

**Interpretation, output by output.**

- **DPDK power: R² is uninformative; report absolute error.**
  - Measured σ = 4.6 mW over 0–5 Gbps; range 51.8 mW; per-level medians span only 8.5 mW.
  - With so little between-level variance, R² mostly measures noise. Held-out MAE 3.3 mW (0.40 % of 0.82 W), RMSE 4.1 mW, P95 8.0 mW.
  - The lookup reference does no better (R² 0.19, MAE 3.3 mW), so the error is measurement noise, not model error.
  - The manuscript's "within 4 mW" matches the RMSE. The 95th-percentile error is 8 mW.
- **DPDK delay: not learnable from offered load.**
  - Every family has negative inner-CV R² in every LOSO fold (−1.4 to −135). Measured delay sits at the 62.5 µs floor except in bursts, mainly at 4–5 Gbps.
  - Honest selection therefore picks a heavily shrunk model predicting ≈170 µs almost everywhere. That gives a +110 µs median residual, close to the 200 µs budget (`residuals_vs_load.pdf`).
  - The deployed configuration refit held-out has MedAE 1.2 µs but R² −0.44.
  - The manuscript's R² 0.32 (full-feature 0.62) exists only on the leaky split.
- **DPDK loss:** measured and predicted are identically 0. R² is undefined, not 1.0 as the manifest reports.
- **USR power:** well predicted below saturation (MAE 0.026 W, ≤0.5 Gbps).
  - Above 0.5 Gbps, attributed power plateaus at 2.8–3.5 W and R² ≈ 0.
  - 30 samples, the first retained sample of each affected run, pair the new load with a Scaphandre reading from the ramp-up. Excluding the first sample of every run is a sensitivity only: R² 0.985 and RMSE 0.187 W.
- **USR delay and loss: the global R² is driven by the saturated regime.**
  - Loss spans 0 to 10⁶ pkts and delay 62.5 to 25,000 µs, so a model that only separates "not saturated" from "saturated" scores R² > 0.9.
  - In the region where the controller chooses USR, loss R² is −175 to −210, and errors are 4–6× the 5-pkt budget.
- **Surrogate vs lookup.** Under LOSO the ML surrogate equals the lookup table of measured per-level means: USR loss R² 0.9955 vs 0.9954, power MAE 0.0846 vs 0.0848 W. Accuracy is bounded by the 22-level grid and replicate variability, not by the learner.
- **Unprofiled loads (LOLO)** degrade everything.
  - USR: power R² 0.928 / MAE 0.244 W; delay 0.869 / 1,327 µs; loss 0.865 / 31,910 pkts.
  - Near the boundary: power R² 0.66, loss MAE 7,191 pkts.
- **Robustness of Table A** (all in the CSV): the deployed configuration refit gives the same USR picture (power 0.970 / 0.089 W; delay 0.909 / 1,120 µs; loss 0.9954 / 3,770). Measured-UL inputs change it little (USR power 0.969, loss 0.9954). The twin's UL := DL substitution is immaterial held-out.
- **Consistency check.** A single grouped 80/20 split with nested selection, committed locally earlier in the profiling repo (`9c6cae1`, `reports_nested/`), gives USR lite power R² 0.986, delay 0.959, loss 0.995. These sit at the optimistic edge of the pooled 110-run intervals, as expected for 22 test runs.

## 9. QoS safety classification

**Table B:** held-out, twin-style inputs, combined rule.

| Protocol | Subset | Samples | TP | FN | FP | TN | False-safe [95 % CI] | False-unsafe [95 % CI] | Precision (unsafe) | Recall (unsafe) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LOSO | DPDK, all | 4,374 | 251 | 41 | 224 | 3,858 | 14.0 % [0, 43.1] | 5.5 % [2.0, 10.0] | 0.53 | 0.86 |
| LOSO | USR, all | 4,386 | 1,475 | 37 | 257 | 2,617 | 2.4 % [0.9, 4.6] | 8.9 % [4.3, 14.6] | 0.85 | 0.98 |
| LOSO | USR, 50–447 Mbps | 996 | 77 | 28 | 170 | 721 | **26.7 % [9.1, 66.0]** | **19.1 % [6.7, 34.5]** | 0.31 | 0.73 |
| LOLO | DPDK, all | 4,374 | 156 | 136 | 43 | 4,039 | 46.6 % | 1.1 % | 0.78 | 0.53 |
| LOLO | USR, all | 4,386 | 1,507 | 5 | 1,485 | 1,389 | 0.3 % | **51.7 % [39.8, 63.2]** | 0.50 | 1.00 |
| LOLO | USR, 50–447 Mbps | 996 | 100 | 5 | 499 | 392 | 4.8 % | **56.0 % [35.9, 76.1]** | 0.17 | 0.95 |

- **DPDK in the band:** no measured-unsafe samples, and none predicted unsafe (998 TN).
- **USR near-boundary, split by criterion (LOSO nested).** Loss-only false-safe 59.6 %, false-unsafe 20.6 %. Delay-only false-safe 77.8 %, false-unsafe 12.2 %.
- **Where the boundary cases sit** (`safety_by_load_level.csv`, USR, LOSO nested, share of samples measured → predicted unsafe):
  - 100 Mbps: 0 % → 0 %
  - 200 Mbps: 16.6 % → 24.6 %
  - 400 Mbps: 33.5 % → 99.0 %
  - 600 Mbps: 100 % → 100 %
  
  Under LOLO the withheld 100, 200 and 400 Mbps levels are predicted **100 % unsafe**, against 0 %, 16.6 % and 33.5 % measured.
- **DPDK unsafe samples** concentrate at 4–5 Gbps (delay bursts: 48.7 % and 78.4 % of samples) plus one 1-Mbps run (39 of its 40 samples). The twin docstring's "DPDK is always safe within measured load range" does not hold at 4–5 Gbps. The pinned twin also returns `is_safe = False` there.
- **Sample vs mean semantics.** The twin applies the budget to a predicted *conditional mean*. The per-level measured means straddle the budget in the band:
  - 200 Mbps: mean loss **5.36** pkts (unsafe by a hair); 3/5 runs within budget.
  - 400 Mbps: mean loss **3.86**, mean delay **188 µs** (safe by a hair); 2/5 runs within budget.
  
  **Either reading, the measurements do not locate the USR boundary anywhere
  between 100 and 600 Mbps more precisely than "somewhere in between, with
  run-to-run variability straddling the budget".**
- **Pinned twin, in-sample (NOT held-out), same band:** false-safe 28.6 %, false-unsafe 16.6 %. That is no better than held-out, which confirms these errors are structural (grid and semantics), not overfitting.

## 10. Stability of the twin-derived operating points

`s02b_threshold_stability.py` applies `derive_thresholds` (1-Mbps grid, 1–500 Mbps)
to every refit model.

| Source | Break-even (Mbps) | QoS limit (Mbps) | Decision (Mbps) | USR safe↔unsafe flips on grid |
|---|---|---|---|---|
| pinned (manuscript) | 91 | 149 | 81 | **3** |
| LOSO nested, 5 folds | 90–93 | 149–153 | 80–83 | 3 |
| LOLO nested, 5 folds | 61–81 | **0, 0, 149, 149, 149** | 0–71 | 0–3 |
| LOLO deployed config | 80–93 | 81–170 | 70–83 | 1–3 |

Tree ensembles split between adjacent training values. With profiled levels at 100
and 200 Mbps, the split falls at 150 Mbps, so λ_qos = 149 Mbps is "the last grid
point before the split". It is stable only as long as both levels are in training.
When either is withheld, the derived QoS limit jumps to 0, 81 or 170 Mbps and
the break-even moves by up to 30 Mbps.

The manuscript's decision threshold is energy-limited (81 = 91 − 10), so the
hysteresis operating point depends on λ_be. The reward's QoS term and every
"QoS-violation %" in the evaluation depend on the non-monotone USR safety map
between 150 and 400 Mbps.

## 11. Evaluated loads outside the physical measurement range

- **Profiling held-out loads:** within range for LOSO; the extremes are extrapolated under LOLO (§6).
- **RL test slice** (`rl_test_load_coverage.csv`; targets_test, horizon 0, α = 1.0, 10,090 steps):

| Offered load | Share of test steps | Pinned twin says USR safe |
|---|---:|---:|
| > 5.0 Gbps (**above max profiled**, peak 5.80) | **3 steps (0.03 %)** | 0 % |
| 0.1–0.2 Gbps (unprofiled gap containing λ_qos) | **21.0 %** | 53 % |
| 0.2–0.4 Gbps (unprofiled gap) | **26.6 %** | 2.9 % |
| 0.4–0.6 Gbps (unprofiled gap) | **12.8 %** | 0 % |
| ≤ 0.1 Gbps | 17.8 % | 100 % |

Only 0.03 % of steps are extrapolation in load. But **60 % of all test steps
fall between the profiled levels 0.1 and 0.6 Gbps**, precisely where USR's QoS
transition happens and where the surrogate's piecewise-constant interpolation
decides `is_safe`. In the 0.2–0.4 Gbps gap the twin forbids USR on 97 % of steps.
The bracketing measured levels have 17 % and 34 % of samples unsafe, and mean loss
of 5.36 and 3.86 pkts. The alpha assumption (α = 1.0) determines how many steps
land in this region.

## 12. Discrepancies with the MASCOTS manuscript (§III-B and §IV)

| Manuscript statement | Regenerated / verified | Status |
|---|---|---|
| "we check that these surrogates reproduce the measured behavior on **held-out data**" | Original split is row-level; 100 % of test rows share a run with training | **Incorrect as worded** |
| "The user-space power surrogate reaches R² = 0.99" | 0.988 is the *full-feature* model on the leaky split. Twin's lite model: 0.964 (leaky), **0.971 [0.955, 0.983] held-out (LOSO)**, 0.928 (LOLO) | Wrong model quoted; held-out value lower |
| "delay and loss surrogates reach R² between 0.97 and 1.00" | Full-feature 0.968 / 0.998 (leaky). Lite held-out: delay **0.913**, loss **0.995**; below 0.5 Gbps delay 0.20, loss −210 | Wrong model; R² misleading for the decision |
| "DPDK power … standard deviation about 5 mW" | σ = 4.6 mW | Consistent |
| "the surrogate reproduces it to within 4 mW" | Held-out MAE 3.3 mW, RMSE 4.1 mW, P95 8.0 mW | Consistent if "RMSE" is stated |
| "its low R² reflects this near-zero variance" | Held-out R² 0.17; lookup reference 0.19 | Consistent |
| "Each surrogate is chosen by cross-validation" | Family chosen by **test** R²; hyper-parameters by ungrouped KFold | **Incorrect** |
| "Ridge where the response is close to linear, tree ensembles where saturation makes it nonlinear" | Twin's lite models: DPDK power RF, DPDK delay RF, USR delay/loss/power RF, USR CPU GB; Ridge only for DPDK throughput/CPU/loss | Inaccurate for the models used |
| "regression surrogates p_m(ρ), q_m(ρ) map the normalized load ρ" | Inputs are offered DL and UL kbit/s (UL := DL); load in Gbps via declared α = 1.0 | Inaccurate |
| "RAPL power measurements collected through Scaphandre" | Scaphandre **process-attributed** power; no host/package series | Needs qualification |
| "The same measurement traces also determine the activation durations" | Activation durations and energy come from a different machine (`switching_costs.yaml`) | **Incorrect** |
| "[switching] cost depends on the realization being activated **and on the offered load**" | Since twin v0.4.0 the spike is a load-independent per-variant constant | Outdated |
| "budgets are set from the measured testbed limits: 200 µs, 5 packets per interval" | Loss budget calibrated to surrogate noise floor (`c58204f`); interval = 3 s | Inaccurate |
| Testbed described as one physical testbed | Two Scaphandre hosts; hardware unrecorded | Unverifiable |
| λ_be = 91 Mbps, λ_qos = 149 Mbps, t_up = 81 Mbps | Reproduced exactly from pinned models; λ_qos is a split artefact (§10) | Numbers correct, interpretation fragile |

## 13. "Digital twin" or "measurement-calibrated surrogate environment"?

**The evidence justifies only "measurement-calibrated surrogate environment"**
(equivalently, "offline, measurement-calibrated simulation environment").

Reasons:
- **No live or physical coupling.** Calibration is one-way and offline, from two profiling windows (≈9 h for DPDK, ≈8 h for USR) in January 2025. Nothing synchronises with or is validated against a running system; the manuscript itself defers closed-loop validation.
- **Inputs are offered load only.** Its mapping from NetMob traffic is a declared assumption (α = 1.0).
- **Coarse state.** The QoS state is a piecewise-constant function of 22 profiled levels. The decision boundary is not resolved by measurements (§9, §10).
- **Switching costs** are transplanted from a different machine and power scope.
- **Hardware identity of the two realizations is unverified** (§3).
- **The exact deployed models have no held-out validation** (§0).

If "digital twin" is kept for continuity with the literature, qualify it at first use, e.g.
"an offline, measurement-calibrated digital-twin environment (surrogate models
fitted to profiling sweeps; no live synchronisation)". Also report Table A/B and the
per-level safety result instead of R² alone.

## 14. Data-quality notes found during the audit

- **Start-of-run transient:** 30 USR samples (the first retained row of 30 runs) at ≥0.2 Gbps show attributed power < 60 % of the level median. This is a Scaphandre ↔ LoadCore `merge_asof` (±3 s) alignment artefact. These samples were kept; the sensitivity subset is in the CSV.
- **Exact-zero USR power:** 340 samples, almost all at ≈0 and 100 kbit/s, with CPU also 0. This is plausible idle attribution, not dropout. Kept.
- **Delay binning and censoring:** the metric resolves 62.5 µs and 187.5 µs, then 375 µs; the 200 µs budget sits between the second and third midpoints.
- **Offered UL ≈ 1.022 × DL** in the profiling data; the twin feeds UL := DL (immaterial held-out, §8).
- **The manifest's DPDK loss R² = 1.0** is a degenerate constant-target case.
- **The profiling repo working tree `models/` was retrained on 2026-08-26** and no longer matches thesis-v1. The twin is unaffected, because it uses the DVC-pinned copy verified in §1. Do not re-export from that working tree.

## 15. What is missing for a stronger claim

1. The host inventory for 192.168.17.26 and 192.168.17.94 (§3).
2. Host- or package-level power series for either campaign, to state a power scope other than process-attributed.
3. Profiling runs between 100 and 600 Mbps (e.g., 10–25 Mbps steps) with ≥5 replicates, to locate the USR QoS boundary instead of interpolating it. Also runs above 5 Gbps.
4. Independent validation data for the deployed pickles: new runs never used for fitting. Alternatively, retrain the twin on 4 sweeps and keep one sweep permanently held out.
5. Activation-energy measurements on the profiling hosts at the same power scope.
6. The original thesis-v1 MLflow records (not present; only 2026-08 re-runs exist).

## 16. Reproduction

Prerequisites:
- Controller repo with `dvc pull` done (`data/external/…`).
- Sibling checkout `../UpfProfilingCampaign/UpfProfilingCampaign` with `data/interim/merged.csv` and `data/processed/features.csv` pulled. Override the location with `UPF_PROFILING_REPO`.
- All inputs are hash-checked against the thesis-v1 lock; a mismatch aborts.

```bash
cd reports/twin-validation-heldout-wcnc2027
PY=../../.venv/bin/python                 # Python 3.12, scikit-learn 1.8.0 (controller venv)
$PY s01_provenance_original_split.py      # hashes, run grouping, original split, manifest reproduction
$PY s01b_sklearn_version_check.py         # optional: repeat under sklearn 1.7.0 / 1.7.2 interpreters
$PY s02_heldout_cv.py                     # nested run-grouped CV, both protocols (~10 min on 32 cores)
$PY s02b_threshold_stability.py           # derived operating points per refit (~2 min)
$PY s03_metrics_tables.py                 # metrics, CIs, safety, tables
$PY s04_figures.py                        # figures
```

scikit-learn 1.7.0 interpreter used for the version check:
`python3 -m venv venv170 && venv170/bin/pip install scikit-learn==1.7.0 numpy==2.2.6 pandas==2.3.3 joblib pyyaml`.

## 17. Files

| File | Content |
|---|---|
| `twin_validation_table.{csv,tex,md}` | CSV: every protocol × configuration × input mode × UPF × output × subset, regression and safety metrics with CIs, plus the non-held-out reference rows. TeX/MD: Tables A and B |
| `measured_vs_predicted.{pdf,png}` | 2×3 held-out measured vs predicted (LOSO nested); identity line, budget lines, error-quadrant counts (per-output criterion, hence different from the combined counts in Table B); symlog loss axis keeps the zeros visible |
| `residuals_vs_load.{pdf,png}` | residual vs offered load, per-level median and 5–95th percentile; USR row marks λ_be, λ_qos and the near-boundary band |
| `safety_by_load_level.csv` | per-level measured vs predicted unsafe shares, held-out |
| `deployed_twin_in_sample_by_level.csv` | pinned twin at each profiled level vs measurements (**in-sample**), incl. mean-based safety |
| `threshold_stability.csv`, `threshold_grid_predictions.csv.gz` | §10 |
| `rl_test_load_coverage.csv` | §11 |
| `heldout_predictions.csv.gz`, `heldout_model_selection.json` | every held-out prediction; every selection with inner-CV scores |
| `original_split_reproduction.csv`, `provenance.json`, `run_metadata.csv`, `sklearn_version_check.json`, `pinned_predictions_sklearn*.npz` | §1, §5 |
| `common.py`, `s01…s04*.py`, `s02_heldout_cv.log` | code and run log |

**Suggested figure caption:** *Measured vs predicted twin outputs on held-out
profiling runs (leave-one-sweep-out nested cross-validation; each sample is
predicted by a model that neither fitted nor selected on its run). Colour: offered
DL load. Dashed: identity. Dotted: QoS budgets (200 µs; 5 packets per 3-s
interval); corner counts are false-unsafe/false-safe samples for that output's
budget. Loss uses a symmetric-log axis so zero-loss samples remain visible.*
