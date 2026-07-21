# Module information sheet

One page per module: role, observability axis, inputs, outputs,
hyperparameters, entry point, runtime, and pins. Values read from the live
configs on **2026-07-21**, not from documentation.

Pipeline order:

```
UpfProfilingCampaign ─┐
                      ├─► UPF_NDT ─► UpfRLControllers
UpfTrafficForecaster ─┘   (twin)      (controllers)
```

Cross-boundary shapes/units/semantics: [CONTRACTS.md](../CONTRACTS.md).
Task status and sequencing: [THESIS_MASTER_PLAN.md](THESIS_MASTER_PLAN.md).

---

## 1. UpfProfilingCampaign

| | |
|---|---|
| **Role** | Measure real UPF power/QoS on a physical testbed; fit two-layer surrogates |
| **Observability axis** | **the unmeasurable** — per-NF attributed power, which RAPL (per-package) cannot provide |
| **Repo / pin** | `Shima-Af/UpfProfilingCampaign` · `main` · `a73870f` (`thesis-v1`), YAMLs at `thesis-v1.1` |
| **Entry point** | `dvc repro` (stages `ingest → merge → featurize → train → evaluate`) |

**Inputs** — Keysight LoadCore traffic archives (`data/raw/loadcore`, 19 CSVs
per run) + Scaphandre process-level power CSVs (`data/raw/scaphandre`).
~220 runs, ~4 390 samples per variant. DVC-tracked; not in git.

**Outputs** — `models/` (30 `.pkl` + `manifest.json`, 48.9 MB),
`reports/metrics.json`, `reports/figures/`, `configs/twin_export/{switching_costs,params}.yaml`,
`reports/chapter_upf_profiling.tex` (26 figures).

**Hyperparameters**

| Param | Value |
|---|---|
| time-alignment tolerance | 3 s (`merge_asof`) |
| rolling window | 30 s |
| model family | ridge · random_forest · gradient_boosting |
| CV folds / search iters | 5 / 20 |
| test split · seed | 0.20 · 42 |
| `usr_safe_threshold_gbps` | 0.5 |
| layer-2 target | `power_watts` |

**Structure** — L1: load → {throughput, cpu, packet-loss-delta, delay};
L2: load + L1 → power. Variants `dpdk`, `usr_full`, `usr_safe`; `full` and
`lite` flavours (**twin uses `lite` only**).

> ⚠ Pickles fitted under scikit-learn 1.7.0; consumers pin `>=1.7,<1.8`.
> `switching_costs.yaml` was measured on a **different host** — only durations
> and the scaling rule are portable, not absolute Wh.

---

## 2. UpfTrafficForecaster

| | |
|---|---|
| **Role** | Forecast per-cluster downlink load one step ahead |
| **Observability axis** | **the future** — pushes the horizon past the present |
| **Repo / pin** | `Shima-Af/UpfTrafficForecaster` · `feature/cluster-first-stgnn` · `99dbe1a` (`thesis-v1`) |
| **Entry point** | `dvc repro` (`build_graph → preprocess → train → evaluate_forecast → export_for_twin`) |

**Inputs** — NetMob 2023 Lyon traces (16 Mar–31 May 2019, 15-min bins, 100 m
tiles, downlink, **dimensionless**) + Cartoradio gNodeB locations. Voronoi →
965 gNodeBs → SKATER → K clusters.

**Outputs** — `checkpoints/cluster_first/<service>/K<K>/best_model.pt` (~73 k
params), `results/cluster_first/...`, and the frozen
`exports/traffic_forecaster/` (11 files) consumed downstream.

**Hyperparameters**

| Param | Value |
|---|---|
| architecture | GRU(1 layer) → GAT(2 layers, 4 heads) |
| hidden dim · dropout | 64 · 0.2 |
| seq_len · horizon | 96 steps (24 h) · 4 |
| epochs · batch · lr | 150 · 32 · 1e-3 (cosine) |
| early stopping · seed | patience 15 · 42 |
| clustering | SKATER, K=10, floor 3, affinity graph 0.6 / density 0.2 / shape 0.2 |
| aggregation | `sum` of member `dl_norm` ← **why values exceed 1.0** |
| service · excluded | Netflix · anomaly date 2019-05-12 |
| K sweep | {5, 10, 20}; best WAPE 17.77 @ K=5; **K=10 deployed** |

**Accuracy** — test MAE 110.7 Mbps at K=10 (× α). Large relative to λ_dec
(81 Mbps), which is why the auto hysteresis band degenerates.

> ⚠ `export_for_twin` is `frozen: true`. The K10 checkpoint was retrained
> 2026-06-11 **after** export, so `predictions_*.npy` are not reproducible at
> HEAD by design. A later DailyMotion run also overwrote the Netflix
> `forecast_eval_summary.json`; the Netflix copy survives only in `exports/`.

---

## 3. UPF_NDT (`upf-digital-twin`)

| | |
|---|---|
| **Role** | Measurement-grounded evaluator: given (action, load) → power, QoS, safety |
| **Observability axis** | **the counterfactual** — what the *other* realisation would have done |
| **Repo / pin** | `Shima-Af/UpfDigitalTwin` · `setup/reproducible-env-and-data` · **`v0.3.0`** |
| **Entry point** | library (`pip install`), `scripts/run_threshold_demo.py`, `app/dashboard.py` |

**Inputs** — `data/external/profiling_twin/` (surrogates) +
`data/external/traffic_forecaster/` (traces), both DVC-imported.

**Outputs** — the `DigitalTwin` API, derived thresholds, rule-based baselines
(static / threshold / hysteresis / oracle), `results/threshold_demo/`,
`reports/chapter_digital_twin.tex` (5 figures).

**Parameters** (`configs/scenario.yaml` — canon shared with the controller)

| Param | Value |
|---|---|
| `alpha_gbps_per_norm` | **1.0** (declared assumption) |
| `delay_budget_us` | 200.0 |
| `max_loss_pkts_per_interval` | 5.0 |
| `prewarm` | disabled, standby 0.0 W |
| switching accounting | `sub_step` |
| threshold safety margin | 10 Mbps |
| step | 15 min |

**Derived** — break-even 91.0 · QoS limit 149.0 · **λ_dec 81.0 Mbps
(energy-limited)** · auto band 221.3 → tuned band 20 Mbps (λ↑ 81, λ↓ 61).

> Controller-agnostic by design — zero references to UpfRLControllers. This is
> what keeps its evidence non-circular. **Do not add any.**

---

## 4. UpfRLControllers

| | |
|---|---|
| **Role** | RL environments, trainers, baselines, evaluation |
| **Observability axis** | **the elsewhere** — MAPPO shares operating knowledge across sites |
| **Repo / branch** | `Shima-Af/UpfRLControllers` · `chore/cleanup-research-vs-library` |
| **Entry points** | `scripts/train_ppo_single_site.py` · `train_ppo_multi_site.py` · `train_ppo_ensemble.py` · `train_mappo.py` |

**Inputs** — `data/external/` (all DVC-tracked) + `upf-digital-twin` v0.3.0.

**Outputs** — checkpoints under `experiments/`, phase reports under `reports/`,
paper drafts (`paper-mascots`, `paper-letters`), 36 figures.

### Environments

| Env | Obs | Action | Reward |
|---|---|---|---|
| `SingleSiteUPFEnv` | `Box(14,)` | `Discrete(2)` | `-(α·SEC + L_QoS + L_SW + L_CD)` |
| `MultiSiteUPFEnv` | `Box(140,)` | `MultiDiscrete([2]×10)` | load-weighted `Σ w_k·r_k` |
| `MultiAgentUPFEnv` | `Box(14,)` ×10 | `Discrete(2)` ×10 | per-agent; `state()` → 140-d |

14-d observation: current load · 8 history loads · 1-step forecast · previous
action · previous Q · previous SEC · cooldown progress.

### Reward weights

| Param | Value |
|---|---|
| α (SEC scaler) | 100.0 |
| λ_QoS · τ | 30.0 · 0.90 |
| λ_sw | 4.0 (× `sw_energy_wh`) |
| cooldown period · cost | 4 steps · 0.5 |
| pool / budget | disabled (`null`) |

### Trainers

| | PPO single/multi | MAPPO |
|---|---|---|
| implementation | stable-baselines3 | from-scratch PyTorch |
| actor | MLP 14→64→64→2 | **shared** across K, 5 250 params |
| critic | — | **centralised** 140→128→128→K, 35 850 params |
| steps · n_steps | 200 k · 1024 | 200 k · 1024 |
| epochs · minibatch · lr | 10 · 256 · 3e-4 | 10 · 256 · 3e-4 |
| γ · GAE λ · clip | 0.995 · 0.9 · 0.2 | same |
| ent_coef · vf_coef | 0.05 (0.15 single-site) | 0.05 · 0.5 |
| reward scale | — | ×0.01 |
| wall time | ~20 min | ~30 min (CPU) |

Model selection is **best-of-val**, load-bearing in every phase: val return
collapses late in training when the policy explores USR at near-zero load.

### Headline (test split, 8/8/4 seeds)

| Controller | Reward | Energy Wh | QoS-viol % |
|---|---|---|---|
| **MAPPO** | **−5610 ± 43** | 1967 | 0.51 |
| IPPO ensemble | −5787 ± 66 | 1974 | 0.56 |
| Hysteresis (band 20) | −6544 | 1970 | 0.94 |
| Always DPDK | −7201 | 2071 | 0.38 |
| Centralised PPO | −12269 ± 5186 | 2111 | 2.73 |

---

## 5. EnergyAwareUPF — deprecated ancestor

Self-contained single-site predecessor: own Keras power models, own offline
Keras LSTM forecaster (**not** wired into the RL loop — fed via a CSV column),
`ManualCooldownEnv`, and SB3 `PPO`/`RecurrentPPO` selectable by config
(`MlpLstmPolicy` is the repo default, but the **published result uses the
feed-forward MLP**: 16.9 % energy reduction vs static DPDK, 5 seeds).

Zero references to any of the four current repos — it genuinely predates the
split. **Thesis role:** the *pre-augmentation baseline* whose limitations
motivated separating profiling, forecasting, twin, and control. Not current work.
