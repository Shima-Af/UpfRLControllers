# Phase reports

One supervisor-facing markdown file per phase. Each contains a one-paragraph
summary, embedded figures, the reproducible KPI table, and a "caveats / next"
section.

| Phase | Status | Report |
|---|---|---|
| 2 — Single-site PPO sanity check | done | [phase-2/](phase-2/README.md) |
| 3 — Multi-site centralized PPO | pending | — |
| 4 — Independent per-site PPO | pending | — |
| 5 — PettingZoo multi-agent env | pending | — |
| 6 — MAPPO / CTDE | pending | — |
| 7 — Final comparison | pending | — |

Each phase has a `scripts/generate_phaseN_report_figures.py` that regenerates
the figures from the current code + trained models. Re-run after any
reward-shape or hyperparameter change to keep the report in sync.
