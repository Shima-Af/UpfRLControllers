# Thesis master plan — end-to-end consolidation

**Thesis:** *Advanced AI tools for Augmented Observability in the 5/6G environment*
**Author:** Shima Afshar Borji · **Plan written:** 2026-07-21

Resumable checkpoint file. Every step has a STATUS. Any session may resume by
reading this file and continuing at the first `TODO`. Update STATUS in the same
commit as the work. Do not batch — one step, one commit, one status update.

STATUS values: `TODO` · `WIP` · `DONE` · `BLOCKED(reason)` · `DECIDE(author)`

---

## 0. The framing gap, and how to close it

The thesis title promises *Augmented Observability*. The four modules currently
read as an energy-optimisation pipeline. The gap is not missing work — it is a
missing argument. The argument:

> **Each module extends the observability envelope along a different axis, and
> the controller results are the evidence that the extension is trustworthy.**

| Module | Observability axis extended | What becomes observable |
|---|---|---|
| UpfProfilingCampaign | **the unmeasurable** (virtual sensing) | Per-NF attributed power and QoS compliance — quantities production 5GC cannot instrument. RAPL is per-package, not per-network-function. The surrogates manufacture a signal the network cannot natively expose. |
| UpfTrafficForecaster | **the future** (temporal) | Cluster load at t+1 rather than only at t. Observability normally ends at the present; the STGNN pushes the horizon forward. |
| UPF_NDT | **the counterfactual** (modal) | "What would happen under the other realisation" — states never entered. The deepest extension: observing the road not taken. |
| UpfRLControllers | **the elsewhere** (spatial) | MAPPO's shared actor lets data-poor MEC sites inherit operating knowledge from data-rich ones. Cluster 5/8 transfer is the empirical proof. |

**The closing move.** Energy saving is not the contribution — it is the
*instrument*. If the inferred power/QoS/forecast signals were wrong, a
controller trained on them could not beat classical baselines on a held-out
period. It does (MAPPO −5610 ± 43 vs hysteresis −6544, 8 seeds). Therefore the
augmented observability is faithful. Control performance is the *validation
metric for the observability stack*, not the end in itself.

**Standards hook.** Frame each module as an AI-augmented realisation of a
standardised analytics function: profiling ↔ MDAF/IDAF (infrastructure
telemetry), forecasting ↔ NWDAF analytics, twin ↔ network digital twin
(Almasan et al.), controllers ↔ closed-loop orchestration. The EnergyAwareUPF
manuscript already invokes NWDAF/MDAF — reuse that thread.

**Where EnergyAwareUPF fits.** Deprecated single-site ancestor. It belongs in
the controller chapter as the *pre-augmentation baseline*: a controller acting
on raw telemetry plus one offline LSTM forecast, with a hand-fitted in-repo
power model. It motivates the split into the four-module stack. Do not present
it as current work; present it as the reason the architecture changed.

---

## Phase A — Durability (do first; cheap; unblocks everything)

| # | Step | Status |
|---|---|---|
| A1 | Push `UpfProfilingCampaign` (`main --follow-tags`) | **DONE** `a73870f..07e9832` + tags `thesis-v1`,`thesis-v1.1` |
| A2 | Push `UpfTrafficForecaster` (`feature/cluster-first-stgnn --follow-tags`) | **DONE** `ff90c8a..99dbe1a` + tag `thesis-v1`; `dvc push` 130 files |
| A3 | Controller: replace 2 local `.dvc` stubs with `dvc import` from `UpfProfilingCampaign@thesis-v1.1` | TODO (now unblocked) |
| A4 | Controller: bump `upf-digital-twin` pin → `v0.3.0` | **DONE** `cf6478b` — in *both* `pyproject.toml` and `requirements.txt`, which had drifted |

`UPF_NDT` and `UpfRLControllers` are already committed and pushed. A1/A2 are
outward-facing; until they land the chain still lives on one machine.

---

## Phase B — Close the open scientific inconsistency

The switching-cost model. `switching_costs.yaml` scales activation energy by
the target's *steady-state attributed power* (0.82 W), where the source
measured *net RAPL package power during activation* (34.8 W). Result: the twin
charges 0.00546 Wh per DPDK start; the source measured 0.232 Wh; the
`scenario_rl.yaml` reward comment claims ~0.25 Wh. **Code is ~42× cheaper than
its own documentation.** Switching is effectively free (0.022 reward units vs
~0.7 for SEC), which plausibly explains MAPPO's 276 flips vs the ensemble's 183.

| # | Step | Status |
|---|---|---|
| B1 | Evaluate the 12 existing `lambda_sw` checkpoints on test with flips/energy/unsafe recorded | **DONE** 2026-07-21 → `research/phase8/results/sweep_lambda_sw_long.csv` |
| B1b | Train MAPPO at `lambda_sw≈170` (3 seeds) — equivalent of absolute switching model — compare to λ=4 | **DONE** 2026-07-23 (single-threaded, ~4 min/seed after thread-thrash fix). flips 276→238, energy/unsafe/reward within noise → switching model does NOT change conclusions |
| B2 | Decide switching-cost model | **DECIDED** 2026-07-23 — neither "absolute" nor "scaled": the correct model is a **load-independent per-variant constant** (activation is a zero-traffic event). Rebased-burst calibration: DPDK 0.00948 Wh, USR 5.2e-6 Wh. See [[project_switching_cost_model]] |
| B3 | Implement + propagate + re-score | **DONE** — profiling `thesis-v1.2`, NDT `v0.4.0`, RL pin bump; Phase-7 re-scored under fixed twin (MAPPO −5647, ordering holds), `reports/phase-7/switching_fix_reeval.md`. NO retrain (B1b). MASCOTS/README number refresh still owed (doc-only) |
| B4 | Split standby out of `switching_energy_wh` ([digital_twin.py:201](../../UPF_NDT/src/upf_digital_twin/twin/digital_twin.py#L201)); numerically zero today (prewarm off) but misreports if re-enabled | TODO |
| B5 | Document the reward v1→v2 revision (`reports/phase-7/*.oldreward.bak` prove a change; no prose records it). Required or cross-phase tables are apples-to-oranges | TODO |

**B1 result (2026-07-21).** λ_sw 2→16 (8×) moves switching only −4.1%
(294→282); reward spread across the whole range is 125 units, *below* the
154–177 MAPPO-vs-IPPO margin. Two conclusions: (a) within the plausible range
the switching weight does not threaten the headline result; (b) the sweep
cannot settle B2, because even λ_sw=16 costs 0.087 reward units against ~0.7
for one step's SEC — the penalty is never binding. The absolute model needs
λ_sw ≈ 170, ~10× beyond the sweep. Break-even reasoning: USR saves ~0.3
units/step, so a 0.93 switch cost pays only if the state is held ≥3–4 steps;
observed behaviour is a switch every ~3.6 steps, i.e. exactly at the boundary
— so the absolute model probably *does* change behaviour. Hence B1b.

**Do not start Phase D chapter writing before B2 is decided** — it sets the
switching numbers every chapter quotes.

---

## Phase C — Contracts and module information sheets

| # | Step | Status |
|---|---|---|
| C1 | `CONTRACTS.md` — every boundary's shape, dtype, units, semantics, canonical value | **DONE** `f0b0f54` |
| C2 | `tests/test_contracts.py` — 19 tests, executable form of C1 | **DONE** `f0b0f54` (28/28 suite green) |
| C3 | `reports/MODULE_SHEET.md` — per-module role, I/O, hyperparameters, entry point, runtime, pin | **DONE** `f0b0f54` |
| C4 | Mirror `CONTRACTS.md` (or a pointer to it) into the three upstream repos so the contract is visible from either side | TODO |

C3 is a stated deliverable. Draft table per module:
`name · role · observability axis · inputs · outputs · hyperparameters ·
entry point · wall time · upstream pin · downstream consumer`.

---

## Phase D — Thesis chapters

Existing: `chapter_upf_profiling.tex` (37 kB, 26 figs) ·
`chapter_forecaster.tex` (28 kB, 8 figs) · `chapter_digital_twin.tex` (39 kB,
5 figs, regenerated 2026-07-21 under the unified scenario).
**Missing: the controller chapter** — 36 figures already exist for it.

| # | Step | Status |
|---|---|---|
| D1 | Write `chapter_controllers.tex` | **DONE** `e2f6b0b`; figures regenerated under v0.4 in the paper font (Latin Modern/usetex) `9d30b8b` — all 6 figs + paired bootstrap now v0.4-consistent |
| D2 | Bridge figure: single-site PPO vs MAPPO on one axis (Phase 2 and Phase 7 currently live in separate tables) | **DONE** `fig_bridge_ippo_vs_mappo.{pdf,png}` from v0.4 summary; transfer wins c0/c4/c5/c7 |
| B5 | Document reward v1→v2 revision | **DONE** — `reward_revision_summary.md` already covered v1(flat)→v2(L_SW); appended note that the v0.4 switching fix is a refinement of v2's input, not a new reward |
| D3 | Weave the §0 observability framing into all four chapters — one framing paragraph each, plus a synthesis section | **DONE** 2026-07-23 — "Place in the thesis" paragraph added to profiling (`aa32521`), forecaster (`2ff1aa8`), twin (`d98ce28`); controller chapter framed at source. Synthesis: `reports/thesis_synthesis_observability.tex` (drop into thesis intro/conclusion). Found+fixed: forecaster `reports/` was fully gitignored (chapter untracked) |
| D4 | Cross-chapter numeric consistency pass | **DONE** — `reports/CROSS_CHAPTER_CONSISTENCY.md`. All shared quantities agree (α=1.0, MAE 110.7, K=10, λ 81/91/149, 0.82 W, Netflix). Two non-error distinctions documented (0.5 Gbps saturation vs 0.149 QoS limit; norm vs physical MAE). One author FLAG: USR idle 0.034 W (P text) vs 0.015 W (params) |
| D5 | Narrate the three currently-unwritten transitions: single-site→MARL limitation, the LSTM-PPO result, threshold v1 (hand-tuned) vs v2 (physics-derived) | BLOCKED(D1) |

---

## Phase E — End-to-end reproducibility

| # | Step | Status |
|---|---|---|
| E1 | `UpfThesisPipeline` manifest repo | **DONE** 2026-07-23 — `/home/ubuntu/UpfThesisPipeline` (local; needs GitHub remote + push). `manifest.yaml` (pins + produce/consume edges), `Makefile` (clone/pull/verify/reproduce/chapters/boundary), `README.md` (reproducibility boundary). Manifest, not merge |
| E2 | Verification harness | **DONE (scoped)** — `make pins` resolves all four (prof eb30588, fcst 15af9ec, twin 95de456, ctrl f558e28); all tags confirmed on remotes; `make verify` green (twin 40, ctrl 28, check_setup OK); `make chapters` all four compile. Full cold-clone is the documented `make clone && make pull && make verify` (needs AWS creds + clean machine) |
| E3 | Tag `thesis-v1` across all four repos once numbers are final | TODO — controller pins ready; gated on author-side finalisation (spike-eq paste, phase-figure regen, MASCOTS refresh) |

---

## Preservation rules

- Never `git checkout`/`reset`/`clean` in these repos without `git stash -u` first.
- `data/external/**` and `exports/` hold bytes that are **not regenerable**
  (forecaster K10 retrained 2026-06-11; DailyMotion run overwrote the Netflix
  summary). Content-hash before and after any operation that moves them.
- New work goes on branches; never force-push a published branch.
- Three orphaned-artifact incidents so far (`switching_costs.yaml`,
  `params.yaml` snapshot, `operating_ranges`). Every one was an artifact with
  no declared owner. C1/C2 exist to end that class of bug.

## Known-open, non-blocking

- Forecaster `README` describes the deprecated per-gNodeB LSTM; live model is
  the cluster-first STGNN (`PIPELINE.md` flags it).
- DailyMotion run overwrote `results/cluster_first/total/forecast_eval_summary.json`;
  the Netflix copy survives only in `exports/`. May affect the forecaster
  chapter's own reported numbers — unverified.
- `alpha=0.12` was calibrated so peak fleet demand ≡ USR safe capacity
  (0.1208 exactly). Superseded by the α=1.0 decision, but record the rule as
  the alternative anchor and report a twin-side α sensitivity sweep — it
  converts the weakest point ("you picked a number") into a characterised
  regime boundary.
