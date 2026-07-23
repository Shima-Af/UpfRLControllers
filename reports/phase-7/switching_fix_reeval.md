# Phase-7 re-evaluation under the corrected switching-cost model

The digital twin's switching-energy model was corrected (UPF_NDT `v0.4.0`,
profiling `thesis-v1.2`): the per-switch spike is now a **load-independent
per-variant constant** anchored to zero-traffic power, replacing the previous
`steady_power(load) × activation_s` form that (a) let the cost vary with the
serving load the container was not yet carrying, and (b) collapsed the measured
DPDK≫USR activation asymmetry. See `switching_costs.yaml` `calibration` block.

All Phase-7 controllers were **re-scored** on the held-out test split under the
new twin. **No retraining** — B1b established that controller behaviour is
invariant to a 42× change in switching cost, and this fix is ≈1.7× for DPDK and
a decrease for USR. Trained checkpoints are unchanged.

## Result — every conclusion holds

| Controller | reward pre-fix | reward post-fix (v0.4) | Δ | energy Wh |
|---|---|---|---|---|
| MAPPO | −5655.3 ± 98 | **−5647.2 ± 86** | +8.0 | 1967 |
| IPPO-ensemble | −5812.1 ± 110 | −5788.6 ± 83 | +23.4 | 1974 |
| always-DPDK | −7201.1 | −7201.1 | 0 | 2071 |

- **Ordering preserved:** MAPPO − IPPO gap = **141** (was 157), still larger
  than the combined seed std.
- Both trained controllers shift up marginally because USR activation (used
  often at low load) got cheaper while DPDK activation got only mildly dearer.
- Energy, safety (unsafe ≈0.53%), and the DPDK/hysteresis baselines are
  unchanged within rounding.

**Conclusion:** the correctness fix does not move any Phase-7 finding. MAPPO
remains the best controller, beats IPPO beyond seed noise, at equal energy and
QoS — now under a physically correct, load-independent switching model.

Source: `multiseed_summary_v04twin.json` (post-fix) vs
`multiseed_summary_n8.json` (pre-fix, retained as the historical record).

> **Doc refresh still owed:** `reports/phase-7/README.md` and the MASCOTS paper
> cite the pre-fix n8 numbers (MAPPO −5610 headline was seed-1; n8 mean is
> −5655→−5647). The deltas are within noise, but the final thesis/paper tables
> should quote the v0.4 numbers for consistency with the corrected twin.
