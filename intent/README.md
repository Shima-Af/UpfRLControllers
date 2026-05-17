# intent/ — LLM-assisted intent-based UPF orchestration (c0 prototype)

Exploratory track on branch `feat/llm-intent-c0`. Single site only —
cluster **c0**, the lowest-load cluster in the digital twin (mean
load 0.108 Gbps). Does **not** touch multi-site, MAPPO, or the
[Letters paper draft](../reports/paper-letters/). The Letters paper
work continues independently on `main` / `chore/cleanup-research-vs-library`.

## Pipeline (Slice 0 — no LLM yet, no predicate yet)

```
canonical Plan (hand-written)
    │
    └─▶ compile.plan_to_weights → reward_weights dict
            │
            └▶ finetune.finetune_c0  (short PPO fine-tune from existing c0 checkpoint)
                  │
                  └▶ verify.twin_replay → metrics dict (normalised vocabulary)
```

Slice 0 hand-writes the Plans and reports the resulting metrics — no
automated accept/reject. Whether the fine-tune produced the intended
shift is a human's call, by inspecting `usr_rate`, `energy_wh`,
`unsafe_pct`, `flips`, etc. against the base checkpoint's numbers.

The Predicate DSL is defined in [schema/predicate.py](schema/predicate.py)
but is **not** wired into the Slice 0 CLI. It is reserved for Slice 2,
where the LLM will emit thresholds grounded in observed performance
rather than guessed-in-advance numbers.

## Slices

| Slice | Status | Adds |
|---|---|---|
| 0 | ✅ done | Hand-written Plan → deterministic weights → fine-tune → replay → metrics |
| 1 | next  | LLM client + `intent_to_plan` prompt; `examples/intents.yaml` paraphrase eval set |
| 2 | later | Predicate emitter (LLM emits thresholds from intent + observed base numbers); explainer for failures; dashboard hook |

## Quick smoke

```bash
source .venv/bin/activate

# (1) Schema + compiler tests — fast, no env construction.
pytest tests/test_intent.py -q

# (2) End-to-end with no fine-tune (replays the base c0 checkpoint).
python -m intent.cli --plan balanced --skip-finetune
```

For a real fine-tune run (~2–3 min on CPU for 20k steps):

```bash
python -m intent.cli --plan energy_greedy --finetune-steps 20000
```

The CLI always exits 0; reading the printed metrics is on you.

## Layout

```
schema/        Plan, metric vocabulary (pydantic v2). Predicate DSL also lives here, dormant.
compile/       Plan -> weights (deterministic Python)
verify/        twin replay
finetune/      short PPO fine-tune from an existing c0 checkpoint
examples/      canonical hand-written Plans (used until Slice 1 LLM lands)
cli.py         end-to-end driver
```
