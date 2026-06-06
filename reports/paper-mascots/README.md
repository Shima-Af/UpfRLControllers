# MASCOTS 2026 paper

"A Measurement-Grounded Digital Twin for Evaluating Cooperative MARL in
Energy-Aware 5G UPF Orchestration"

Self-contained conference paper (IEEE `conference` class). Built from the
draft structure plus the verified Phase-7 / Phase-9 evaluation artefacts.

## Build

```bash
python make_figures.py          # regenerates figures/fig_bootstrap.pdf, fig_per_cluster.pdf
pdflatex paper && pdflatex paper # run twice for cross-references
```

`IEEEtran.cls` is bundled. Output is `paper.pdf` (6 pages).

## Where every number comes from

All quantitative claims trace to committed artefacts. Nothing is hand-entered.

| Claim in paper | Source file |
|---|---|
| Table I (headline, 8/8/4 seeds) | `reports/phase-7/multiseed_summary_n8.json` |
| Bootstrap +154, CI [+28,+262], p=0.009, per-seed deltas | `reports/phase-7/paired_bootstrap_mappo_vs_ippo.json` |
| Table II (hysteresis band sweep) | `reports/phase-7/hyst_sweep/band_{5,10,50,100}.json` + n8 summary (band 20) |
| Cooldown sensitivity (Sec. VI-D) | `reports/phase-7/cooldown_sweep_*.json` |
| Per-cluster transfer (Fig. 4, Sec. VI-C) | per-cluster rewards averaged over 8 seeds in the n8 summary |
| Cluster heterogeneity (loads, p95, peak hour) | `research/phase9/results/heterogeneity_test.csv` |
| Central-PPO failure examples | per-cluster usr/unsafe rates in the n8 summary |

## Editorial decisions (differ from the earlier draft PDF)

- Fleet loads use the **test-split** values from `heterogeneity_test.csv`
  (mean 0.108-1.66 Gbps, 15.4x range, peaks 5.8 Gbps). The earlier draft
  quoted 1.808 Gbps / 5.58 Gbps, which came from a different split.
- "Unsafe %" is relabelled **QoS-violation %** (same quantity:
  delay-or-loss budget exceeded), which reads better for a telecom venue.
- Fig. 1 (pipeline) and Fig. 2 (deployment) are TikZ reconstructions so
  the document compiles stand-alone; swap in the Overleaf originals if
  preferred.
- TODO references [15]/[16] filled with real sources: Almasan et al.
  (Network Digital Twin, IEEE ComMag 2022) and Khan et al. (RAPL in
  Action, ACM ToMPECS 2018).
