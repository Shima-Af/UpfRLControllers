# Phase reports

One supervisor-facing markdown file per phase. Each contains a one-paragraph
summary, embedded figures, the reproducible KPI table, and a "caveats / next"
section.

| Phase | Status | Report |
|---|---|---|
| 2 — Single-site PPO sanity check | done | [phase-2/](phase-2/README.md) |
| 3 — Multi-site PPO (centralised + ensemble) | done | [phase-3/](phase-3/README.md) |
| 6 — MAPPO / CTDE | done | [phase-6/](phase-6/README.md) |
| 7 — Final multi-seed comparison | done | [phase-7/](phase-7/README.md) |

Paper artifacts: [paper-draft/](paper-draft/README.md), [paper-letters/](paper-letters/README.md),
[paper-mascots/](paper-mascots/README.md).

Cross-repo work: [cross_repo_fixups.md](cross_repo_fixups.md) — ready-to-run agent
prompts for UpfTrafficForecaster, UpfProfilingCampaign, and UPF_NDT, plus the
follow-ups they unblock in this repo.

Each phase has a `research/phaseN/generate_figures.py` that regenerates
the figures from the current code + trained models. Re-run after any
reward-shape or hyperparameter change to keep the report in sync.
