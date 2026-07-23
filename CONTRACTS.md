# Cross-repo data contracts

Every artifact that crosses a repository boundary, with its exact shape, dtype,
units, and semantics. **Verified against the real artifacts on 2026-07-21** —
not aspirational.

This file exists because all four serious defects found during the 2026-07
consolidation were the same class of bug: an artifact or config field crossing
a boundary on convention rather than contract.

| Defect | Root cause |
|---|---|
| `selected_k: 8` vs K=10 arrays | no declared shape contract |
| `service: "video"` vs `Netflix` | no declared identity contract |
| `alpha` 0.12 vs 1.0 | no declared units contract |
| `switching_costs.yaml` orphaned | no declared owner |

`tests/test_contracts.py` asserts everything below. If a claim here and the
test disagree, **the test wins** — fix this file.

---

## 1. UpfTrafficForecaster → UPF_NDT, UpfRLControllers

**Source:** `exports/traffic_forecaster/` @ tag `thesis-v1`
(repo `Shima-Af/UpfTrafficForecaster`, branch `feature/cluster-first-stgnn`).
**Destination:** `data/external/traffic_forecaster/`.
**Transport:** DVC (S3 remote `forecaster`) + pinned `.dvc` files.

| File | Shape | dtype | Semantics |
|---|---|---|---|
| `predictions_train.npy` | `(5073, 4, 10)` | float32 | STGNN forecast, `(N, H, K)` |
| `predictions_val.npy` | `(1009, 4, 10)` | float32 | " |
| `predictions_test.npy` | `(1009, 4, 10)` | float32 | " |
| `targets_train.npy` | `(5073, 4, 10)` | float32 | ground-truth load, `(N, H, K)` |
| `targets_val.npy` | `(1009, 4, 10)` | float32 | " |
| `targets_test.npy` | `(1009, 4, 10)` | float32 | " |
| `cluster_series.npy` | `(10, 7388)` | float32 | full per-cluster series, `(K, T)` |
| `cluster_assignments.parquet` | `(965, 2)` | — | `site_id`, `cluster_id` |
| `bs_locations.parquet` | `(965, 3)` | — | `site_id`, `lon`, `lat` |
| `cluster_bs_map.json` | 10 keys | — | cluster → member site list |
| `forecast_eval_summary.json` | — | — | `service`, `sweep_k`, `best_k`, `results` |

**Axis order is `(N, H, K)`** — timestep, forecast horizon, cluster. Never
`(N, K, H)`. Controllers index `[:, horizon_idx, cluster_idx]`.

**Units: dimensionless.** Values are per-BS NetMob `dl_norm` **summed** across
cluster members (`signal_aggregation: sum`). This is why the range is
`[0.0006, 5.80]` and not `[0, 1]`. NetMob is privacy-preserving and carries no
physical unit, so **there is no recoverable scaler** — see §4.

**Ranges (test):** targets `[0.0017, 5.7962]`, predictions `[-0.0054, 4.0075]`.

> ⚠ **Predictions can be slightly negative** (min −0.0107 across splits). Any
> consumer converting to Gbps must tolerate a small negative load or clip it.
> The reward path is safe (`max(0, …)` guards at
> [single_site_upf_env.py:66-71](src/envs/single_site_upf_env.py#L66-L71)), but
> observation and threshold-policy paths pass the raw value through.

**Identity:** `service = "Netflix"`, `K = 10`. Both are validated by
`upf_digital_twin.data.traffic_loader.load_forecaster_bundle` against
`scenario.traffic.service` / `selected_k`; a mismatch raises. `K` must equal
the arrays' third axis, and `cluster_series.shape[0]`.

**Splits:** train 5073 steps (~53 d), val 1009, test 1009, at 15 min/step.
Train is used for PPO episodes; test is touched once, by the phase evaluators.

> ⚠ **`predictions_*.npy` are not regenerable upstream.** The K10 checkpoint was
> retrained 2026-06-11 after these were exported. `targets_*.npy` still match
> upstream (ground truth is deterministic); predictions do not. The export
> stage is `frozen: true` for this reason. Never "refresh" these from upstream.

---

## 2. UpfProfilingCampaign → UPF_NDT, UpfRLControllers

**Source:** `models/` @ `a73870f` (tag `thesis-v1`) and
`configs/twin_export/` @ tag `thesis-v1.1`.
**Destination:** `data/external/profiling_twin/`.
**Transport:** `models/` via `dvc import` (S3 `profiling`); the two YAMLs are
git-tracked upstream and importable without S3 credentials.

**`models/` — 30 pickles + `manifest.json`.** Two-layer surrogate:

- **Layer 1:** offered load → `throughput_gbps`, `cpu_pct`,
  `gtpu_packets_dn__packets_lost_delta`,
  `downlink_one_way_delay_distribution__weighted_mean_delay_us`
- **Layer 2:** load + layer-1 outputs → `power_watts`
- **Variants:** `dpdk`, `usr_full`, `usr_safe`
- **Flavours:** `full` and `lite`. **The twin loads `lite` exclusively** — it
  needs only throughput, which is all NetMob-derived load can supply.

Manifest key format: `{variant}__{layer}__{target}__{full|lite}`.

> ⚠ **Manifest `path` values use Windows separators** (`models\layer1\x.pkl`).
> Consumers must normalise. Do not assume POSIX paths.

> ⚠ **scikit-learn is pinned `>=1.7,<1.8`** in UPF_NDT. The pickles were fitted
> under 1.7.0; loading under 1.8+ emits `InconsistentVersionWarning` and is not
> guaranteed correct. The controller's `.venv` currently has 1.8.0 — use the
> twin's own `.venv` for any twin-side derivation.

**`switching_costs.yaml`** — activation cost. Hardware-independent durations:
DPDK 24.0 s, USR 3.3 s.

> ⚠ **Provenance: measured on a different host than the profiling campaign.**
> Absolute Wh figures are explicitly **not portable**; only the durations and
> the scaling rule transfer. This is a stated thesis limitation, not a detail.
> See §5 for the open question this creates.

**`params.yaml`** — 816 B config snapshot. **Distinct from
UpfProfilingCampaign's 2242 B repo-root `params.yaml`** (the DVC pipeline
config). Same name, unrelated content; namespaced upstream under
`configs/twin_export/` to prevent collision. Carries `capacity`,
`layer1_targets`, and `operating_ranges` — including
`usr.safe_capacity_gbps: 0.69`, `usr.danger_threshold_gbps: 0.70`,
`usr.power_slope_w_per_gbps: 6.8`, `dpdk.power_watts_idle: 0.82`. These are
**not derivable from anything else in any repo**.

---

## 3. UPF_NDT → UpfRLControllers

**Transport:** pip, `upf-digital-twin @ git+…/UpfDigitalTwin.git@v0.4.0`
(pinned identically in `pyproject.toml` *and* `requirements.txt` — keep in sync).

**API the controller depends on:**

```python
from upf_digital_twin import DigitalTwin
twin = DigitalTwin(scenario_cfg=..., paths_cfg=..., project_root=...)
twin.evaluate_batch("DPDK"|"USR", loads_gbps) -> UPFResult   # vectorised
sess = twin.session(); sess.step(action, actual_load_gbps)   # stateful
twin.step_h                                                   # hours per step
```

`step()` returns `switching_energy_wh`, `power_watts`,
`power_watts_steady`, `delay_us`, `predicted_loss`, `is_safe`.

> ⚠ `switching_energy_wh` = `spike_wh + standby_wh`
> ([digital_twin.py:201](../../UPF_NDT/src/upf_digital_twin/twin/digital_twin.py#L201)).
> With prewarm off, `standby_wh == 0`, so the field is currently pure
> activation energy. If prewarm is ever re-enabled the field silently becomes
> a standby tax and a never-switching policy reports non-zero "switch energy".
> Scheduled to be split (plan step B4).

**Direction is one-way.** UPF_NDT contains zero references to UpfRLControllers
and must stay that way: the twin's evidence is only non-circular because it
cannot have been tuned to flatter the controller.

---

## 4. Shared scenario canon

These must be **identical** in `UPF_NDT/configs/scenario.yaml` and
`UpfRLControllers/configs/scenario_rl.yaml`. Verified equal 2026-07-21.

| Field | Canonical | Notes |
|---|---|---|
| `alpha` (Gbps per normalised unit) | **1.0** | **Declared modelling assumption, not a measurement.** NetMob is dimensionless; no scaler exists to recover. Places the fleet at 0.06–1.66 Gbps mean, 5.80 Gbps peak, with USR QoS-safe ~29% of steps. |
| `delay_budget_us` | 200.0 | |
| `max_loss_pkts_per_interval` | 5.0 | Safety budget: feeds `is_safe` **and** threshold derivation. Also currently doubles as the graded QoS score's denominator — two concepts, one field (plan: rename the scoring use to `loss_score_normalisation_pkts`). |
| `prewarm.enabled` | false | Cold start. **Required** for consistency with the measured 24 s DPDK activation — warm standby is physically incompatible with paying a driver-init cost. |
| `prewarm.standby_power_watts` | 0.0 | |
| `time_step_minutes` | 15 | NetMob resolution |
| `service` / `selected_k` | `Netflix` / `10` | must match §1 |

**Derived quantities that both chapters publish — must agree:**

| Quantity | Value | Derivation |
|---|---|---|
| Energy break-even | 91.0 Mbps | first load where USR power ≥ DPDK power |
| QoS limit | 149.0 Mbps | last load where USR `is_safe` |
| Decision threshold λ_dec | **81.0 Mbps** | `min(breakeven, qos) − 10 Mbps`; **energy-limited** |
| Forecast MAE (K=10, test) | 110.7 Mbps | `forecast_eval_summary.json` × alpha |
| Auto hysteresis band | 221.3 Mbps | `2 × MAE` — wider than λ_dec, so it degenerates to always-DPDK |
| Tuned hysteresis band | 20 Mbps | the reported operating point (λ↑ 81, λ↓ 61) |

Alternative alpha anchor, recorded for the sensitivity discussion: **α = 0.1208**
places the dataset's global peak exactly on USR's `danger_threshold_gbps`
(0.70). The historical `0.12` was this rule, rounded. Superseded by α = 1.0,
but it is a real calibration rationale and belongs in the thesis as the
alternative anchor.

---

## 5. Open contract question

`switching_costs.yaml` scales activation energy by the **target's steady-state
attributed UPF power** (0.82 W), whereas the source measured **net RAPL package
power during activation** (34.8 W). The twin therefore charges 0.00546 Wh to
start DPDK against 0.232 Wh measured — **~42×**, and
`scenario_rl.yaml`'s reward comment documents the *absolute* model while the
code implements the *scaled* one.

Evidence so far (plan B1): λ_sw 2→16 moves switching only −4.1%, and the reward
spread over that range (125 units) is below the MAPPO-vs-IPPO margin (154–177),
so the weight is not a threat **within the tested range**. But the absolute
model corresponds to λ_sw ≈ 170, ~10× beyond it. Resolution pending B1b.

Until resolved, treat any "switching costs X Wh" statement as provisional.
