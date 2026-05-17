# research/

Phase-specific one-off scripts that produced the numbers and figures in
[`reports/`](../reports/). Code here is **frozen** — it is the historical
record of how each phase's results were generated, not a reusable library.

## When to put code here vs. `src/` or `scripts/`

- **`src/`** — importable, tested library code (envs, trainers, baselines,
  reward, evaluation helpers). Reusable across phases.
- **`scripts/`** — thin canonical CLIs that are still actively used:
  setup checks, smoke tests, the four trainers (single-site PPO,
  multi-site PPO, ensemble, MAPPO).
- **`research/`** — everything else. Per-phase evaluators, figure
  generators, ablations, sweeps. These scripts:
  - reference paths relative to the repo root (`parents[2]`),
  - import from `src/` (and occasionally from `dashboard/`),
  - are not imported by anything else,
  - are kept for reproducibility of the corresponding report.

## Layout

| Directory | What's in it | Report |
|---|---|---|
| [`phase2/`](phase2/) | Single-site PPO test eval + figures | [`reports/phase-2/`](../reports/phase-2/) |
| [`phase3/`](phase3/) | Multi-site PPO + ensemble test eval + figures | [`reports/phase-3/`](../reports/phase-3/) |
| [`phase6/`](phase6/) | MAPPO test eval + figures | [`reports/phase-6/`](../reports/phase-6/) |
| [`phase7/`](phase7/) | Multi-seed cross-controller comparison, paired bootstrap, cooldown sweep, reward-revision diff | [`reports/phase-7/`](../reports/phase-7/) |
| [`paper-letters/`](paper-letters/) | Figure generator for the letters-format paper | [`reports/paper-letters/`](../reports/paper-letters/) |

## Reproducing a phase

From the repo root:

```bash
# Phase 2
python research/phase2/evaluate_test_split.py
python research/phase2/generate_figures.py --split test

# Phase 3
python research/phase3/evaluate_test_split.py --ensemble-dir experiments/ppo_single_site_ensemble_<ts>
python research/phase3/generate_figures.py --split test

# Phase 6
python research/phase6/evaluate_test_split.py
python research/phase6/generate_figures.py --split test

# Phase 7
python research/phase7/evaluate_multiseed.py
python research/phase7/generate_figures.py
python research/phase7/paired_bootstrap.py
python research/phase7/compare_reward_revisions.py
```
