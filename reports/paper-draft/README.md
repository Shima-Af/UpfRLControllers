# Paper draft — working folder

**Status.** v0.1 draft of the conference paper, written from the
Phase 7 multi-seed results. All numbers in the draft are pulled from
[`reports/phase-7/multiseed_summary.json`](../phase-7/multiseed_summary.json)
and are reproducible via `research/phase7/evaluate_multiseed.py`.

## Files

- [`paper.md`](paper.md) — full draft in markdown. Convert to LaTeX
  once venue is picked (IEEEtran for INFOCOM / GLOBECOM / NOMS;
  acmart for CoNEXT).

## What's solid

- Sections 3–6 (system model → results) are concrete and grounded
  in the running experiments under the **physics-grounded reward
  (v2)** — switching cost is now driven by the twin's measured
  spike energy folded into SEC, not by a flat constant.
- All numbers in tables are live — re-run
  `research/phase7/evaluate_multiseed.py` and
  `research/phase7/compare_reward_revisions.py` and they refresh
  automatically.
- §6.6 includes the v1 (flat) vs v2 (physics) delta table and a
  cooldown sensitivity sweep on MAPPO.
- Figures referenced under
  [`reports/phase-7/figures/`](../phase-7/figures/) are from the
  v1 results — they need a quick regenerate against v2 (the
  qualitative shape is similar but per-seed scatter is wider).

## What still needs work, in order

### P0 — blocking submission
1. **Pick venue.** Title, abstract length, page limit, and
   reference style all flow from this. Top candidates: INFOCOM,
   GLOBECOM, NOMS, CoNEXT, NetSoft, EuCNC.
2. **Author list + affiliations** in the title block.
3. **Headline figure** — one plot a reviewer remembers; currently
   we have four good figures but no single "this is the result"
   plot. Likely a remix of fig1 (error-bar bars) with the classical
   baselines included.
4. **Final single comparison table** is in §6.1, but verify it
   matches whichever venue's table-formatting expectation.

### P1 — reviewers will ask
5. **CTDE ablation** — separate-actor + shared-critic ("IACC").
   Deferred. The mechanism claim ("MAPPO wins via parameter
   sharing, not the centralised critic alone") is softened to
   "the per-cluster pattern is consistent with parameter sharing
   as the dominant mechanism"; the architectural-ranking claim
   (MAPPO > IPPO statistically significant at α = 0.01) stands
   without this ablation. Skip for v1 submission to a networking
   venue; revisit only if targeting a pure MARL venue.
6. ✅ **Statistical test** (DONE 2026-05-17). MAPPO and IPPO both
   bumped to 8 seeds (1, 7, 13, 23, 42, 64, 77, 99). Paired
   bootstrap (10 000 resamples) over seed-matched MAPPO − IPPO
   deltas: mean **+154 reward units, 95 % CI [+28, +262],
   one-sided $p = 0.009$**. The architectural win is significant
   at α = 0.01. 7/8 MAPPO seeds beat IPPO. Script:
   [`research/phase7/paired_bootstrap.py`](../../research/phase7/paired_bootstrap.py).
7. ✅ **Physics-grounded switching cost** (DONE 2026-05-17).
   `c_DPDK / c_USR` flat constants removed; twin's `sw_energy_wh`
   folded into the SEC term as `P_switch = E_sw / Δt`. All three
   architectures re-trained on v2 reward (MAPPO/IPPO at 8 seeds,
   centralised PPO at 4). Results in §6.1, §6.6 of paper.
   Cooldown "+"-pattern sweep on MAPPO (5 combos × 1 seed) included
   in §6.6 — confirms cooldown is partially load-bearing but the
   baseline `(period=4, cost=0.5)` sits in the centre of a soft
   plateau.
8. ✅ **Hysteresis band sweep** (DONE 2026-05-17). Bands
   `∈ {5, 10, 20, 50, 100}` Mbps swept on the v2 reward; band = 50
   is the strongest classical operating point (−6262), beating the
   hand-picked band = 20 (−6543) by 281 reward units. Adopted as
   the §6.1 baseline. Band ≤ 10 over-switches (unsafe > 1.1 %);
   band = 100 collapses to always-DPDK (t_down → 0). No setting
   of the band closes the 629-reward gap to MAPPO.

### P2 — defensive depth
9. Compute / wall-clock comparison (one table): MAPPO vs IPPO
   vs centralised PPO.
10. Generalisation: hold out 1–2 clusters from training, evaluate
    zero-shot on them. Validates the parameter-sharing transfer
    claim.
11. Forecaster-noise robustness: ±X % MAE perturbation injected
    at evaluation.
12. MAPPO hyperparameter sensitivity (clip, ent_coef, lr) — short
    table, one paragraph.

### P3 — packaging
13. Reproducibility: pinned `requirements.txt`, single `make`
    target per phase, seed-fixed scripts.
14. Released artifact: checkpoints + dashboard URL or container.
    Some venues award artifact badges.
15. Related-work section is currently a stub — expand with
    8–12 nearest citations (energy-aware UPF; RL for VNF placement;
    MAPPO; digital twins for net control).

### Out of scope for v1 (journal extension)
- MARLlib / RLlib re-implementation as a learning track (memory
  note [`project_mappo_library_revisit`](../../.claude/projects/-home-ubuntu-UpfRLControllers/memory/project_mappo_library_revisit.md)).
- Dynamic K (clusters joining / leaving).
- Online / continual learning under traffic drift.
- Sim-to-real bring-up on a physical DPDK/USR testbed.

## How the draft is structured

| § | Section | Status |
|---|---|---|
| — | Abstract | drafted; tighten to venue word limit |
| 1 | Introduction | drafted with contribution list |
| 2 | Related work | **stub** — citations placeholder |
| 3 | System model | drafted |
| 4 | Problem formulation | drafted, includes reward math + baseline derivation |
| 5 | Methodology | drafted (MAPPO + IPPO + centralised PPO) |
| 6 | Results | drafted, 6 subsections with tables |
| 7 | Discussion / limitations | drafted; flags every owed ablation |
| 8 | Conclusion | drafted |
| — | Reproducibility | drafted (shell snippets) |
| — | References | **placeholder** — anchor list only |

## Next concrete actions

P1 is complete. Remaining work is venue selection, P2/P3, and
LaTeX conversion. Recommended order:
1. **Pick venue** → constrains length and format. 30 min decision.
2. Regenerate figures from the v2 N=8 summary (current PNGs under
   `reports/phase-7/figures/` are from the v1 4-seed run).
3. Expand related work (P3 #15) — currently a stub.
4. Convert markdown → LaTeX with the chosen template.
5. Internal review pass.
6. (Optional) P2 #10–13 for defensive depth: compute comparison,
   generalisation, forecaster-noise robustness, MAPPO HP
   sensitivity. Pick the 1–2 most relevant to the chosen venue.
