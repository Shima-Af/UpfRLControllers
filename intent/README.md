# intent/ — LLM-assisted intent-based UPF orchestration (c0 prototype)

Exploratory track on branch `feat/llm-intent-c0`. Single site only —
cluster **c0**, the lowest-load cluster in the digital twin (mean
load 0.108 Gbps). Does **not** touch multi-site, MAPPO, or the
[Letters paper draft](../reports/paper-letters/). The Letters paper
work continues independently on `main` / `chore/cleanup-research-vs-library`.

## Pipeline (Slice 0 — no LLM yet)

```
canonical Plan (hand-written)
    │
    ├─▶ compile.plan_to_weights   → reward_weights dict
    │       └▶ finetune.finetune_c0  (short PPO fine-tune from existing c0 checkpoint)
    │             └▶ verify.twin_replay  → metrics dict (normalised vocabulary)
    │
    └─▶ compile.plan_to_predicate → Predicate (AND of comparisons)
            └▶ verify.evaluate(metrics) → {satisfied, witness}
```

Slice 0 hand-writes the Plans. Slice 1 will swap the hand-writing for
an LLM call (`intent/llm/`, not yet created) that emits a Plan from a
natural-language intent. Slice 2 will add the LLM as a runtime
explainer / auditor for predicate failures.

## Quick smoke

```bash
source .venv/bin/activate

# (1) Schema + compiler tests — fast, no env construction.
pytest tests/test_intent.py -q

# (2) End-to-end with no fine-tune (replays the base c0 checkpoint).
python -m intent.cli --plan balanced --skip-finetune
```

For a real fine-tune run (slow — ~5–10 min for 20k steps on CPU):

```bash
python -m intent.cli --plan energy_greedy --finetune-steps 20000
```

The CLI exits 0 if the compiled predicate is satisfied, 1 otherwise.

## Layout

```
schema/        Plan, Predicate, metric vocabulary (pydantic v2)
compile/       Plan -> weights, Plan -> predicate (deterministic Python)
verify/        twin replay + predicate evaluator
finetune/      short PPO fine-tune from an existing c0 checkpoint
examples/      canonical hand-written Plans (used until Slice 1)
cli.py         end-to-end driver
```

## Why pydantic for both Plan and Predicate

When Slice 1 lands the LLM, the same `Plan` schema becomes the
function-calling/structured-output target. Validating the LLM's emit
against pydantic is one line; logging it as JSON for the workshop
paper's "intent translation audit" is one more.
