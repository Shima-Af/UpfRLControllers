# Cross-chapter numeric consistency (D4)

Every quantity shared across two or more thesis chapters, its value in each, and
a verdict. Audited 2026-07-23 against the committed chapter sources. Re-run after
any chapter regeneration.

Chapters: **P** = profiling (`UpfProfilingCampaign`), **F** = forecaster
(`UPF_Forecasting`), **T** = twin (`UPF_NDT`), **C** = controllers
(`UpfRLControllers/reports/chapter-controllers`).

## Shared quantities

| Quantity | P | F | T | C | Verdict |
|---|---|---|---|---|---|
| Load calibration α (Gbps per norm. unit) | — | twin-owned, dimensionless | **1.0** | **1.0** (reward) | ✅ unified; F correctly defers α to T |
| Cluster count K | (NetMob adapter) | K∈{4..20}, **10** deployed | **10** | **10** | ✅ |
| Forecast MAE, K=10 | — | ≈0.103 norm (≈**110.7** Mbps at α=1) | **110.7** Mbps | **110.7** Mbps | ✅ |
| Energy break-even λ_be | — | — | **91** Mbps | **91** Mbps | ✅ |
| QoS limit | — | — | **149** Mbps | **149** Mbps | ✅ |
| Decision threshold λ_dec | — | — | **81** Mbps | **81** Mbps | ✅ |
| Auto hysteresis band (2·MAE) | — | feeds T | **221** Mbps | **221** Mbps (degenerate) | ✅ |
| DPDK steady power | ~0.82 W (measured) | — | **0.82** W | **0.82** W | ✅ |
| Switching energy model | measured (colleague host) | — | load-independent const (v0.4) | L_SW=λ_sw·sw_energy_wh | ✅ (see [[project_switching_cost_model]]) |
| Service / trace | Netflix (+YouTube/DailyMotion) | Netflix deployed | Netflix | Netflix | ✅ |
| Predecessor result | — | — | — | 16.9% vs DPDK | ✅ (single source) |

## Two distinctions to keep straight (not errors)

1. **USR throughput saturation (~0.5 Gbps) vs QoS limit (0.149 Gbps).** P
   establishes that USR *drops packets en masse* above ~0.5 Gbps (hard
   throughput saturation). T derives a **QoS limit of 149 Mbps** — the load
   where the *delay/loss budget* is first exceeded, which is stricter and
   therefore lower. These are different operating points, correctly used: C and
   T speak of the 0.149 Gbps QoS limit for control; P's 0.5 Gbps is physical
   saturation. T's intro summary ("saturates … above ~0.5 Gbps") is a coarse
   gloss; the precise QoS limit is in T's thresholds section. No change needed,
   but a one-line cross-reference in T's intro would pre-empt reader confusion.

2. **MAE appears in normalised and physical units.** F reports skill/WAPE and a
   normalised MAE (~0.10); T and C report 110.7 Mbps. At α=1.0, 0.1107 norm =
   110.7 Mbps — consistent. Any future change to α must be applied to all three.

## Author FLAG (verify, do not auto-edit)

- **USR idle power:** P's text mentions ~0.034 W at idle for USR; the profiling
  `params.yaml` `operating_ranges` used by the switching calibration lists
  0.015 W. Likely different idle definitions (measurement window vs snapshot),
  but confirm which is canonical, since the USR switching-energy constant is
  derived from the 0.015 figure.

## Pending regenerations that will touch shared numbers

- **T chapter:** replace Eq (dt_spike) with the load-independent form
  (`UPF_NDT/reports/THESIS_EDITS_PENDING.md`).
- **C chapter:** phase-2/3/6/7 PNGs + paired bootstrap were made under the
  pre-v0.4 twin; regenerate for the final (deltas within seed noise). Tables
  already use v0.4.
- **MASCOTS paper + phase-7 README:** cite pre-fix n8 numbers; refresh to v0.4.

Verdict: **all cross-chapter quantities agree.** Remaining items are unit-scale
clarifications and pending figure regenerations, none of which change a value.
