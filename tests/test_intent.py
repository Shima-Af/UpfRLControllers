"""Unit tests for the intent schema and deterministic compilers (Slice 0).

No checkpoint loading, no env construction here — those live behind the
twin-artifacts gate in `conftest.py` and would slow the suite down.

Predicate DSL tests stay in place even though the Predicate is not
wired into the Slice 0 CLI — the DSL is the contract Slice 2's LLM
will emit against, and keeping it test-covered avoids drift.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intent.compile.plan_to_weights import BASE, compile_weights
from intent.examples.canonical_plans import CANONICAL_PLANS
from intent.schema.metrics import METRIC_VOCAB, normalize_metrics
from intent.schema.plan import Plan
from intent.schema.predicate import Comparison, Predicate


# ----------------------------------------------------------------------
# Plan schema
# ----------------------------------------------------------------------

class TestPlanValidation:
    def test_canonical_plans_all_validate(self):
        for name, plan in CANONICAL_PLANS.items():
            assert isinstance(plan, Plan), name

    def test_invalid_goal_rejected(self):
        with pytest.raises(ValidationError):
            Plan(goal="minimize_galaxies")

    def test_extra_field_rejected(self):
        # extra="forbid" on the model_config — guard against silent typos.
        with pytest.raises(ValidationError):
            Plan(goal="balance", made_up_field=42)

    def test_invalid_soft_pref_rejected(self):
        with pytest.raises(ValidationError):
            Plan(goal="balance", soft=["not_a_real_pref"])


# ----------------------------------------------------------------------
# Weight compiler
# ----------------------------------------------------------------------

class TestWeightCompiler:
    def test_balanced_returns_base(self):
        w = compile_weights(CANONICAL_PLANS["balanced"])
        for k, v in BASE.items():
            assert w[k] == v, k

    def test_minimize_energy_raises_alpha(self):
        w = compile_weights(CANONICAL_PLANS["energy_greedy"])
        assert w["alpha"] > BASE["alpha"]
        assert w["tau"] <= BASE["tau"]

    def test_minimize_unsafe_raises_lambda_qos(self):
        w = compile_weights(CANONICAL_PLANS["safety_first"])
        assert w["lambda_qos"] > BASE["lambda_qos"]
        assert w["tau"] >= BASE["tau"]

    def test_soft_pref_can_override_goal_delta(self):
        # `maintain_qos_headroom` sets lambda_qos=60 — same as minimize_unsafe.
        # Confirms later-write-wins doesn't regress to BASE.
        plan = Plan(goal="minimize_unsafe", soft=["maintain_qos_headroom"])
        w = compile_weights(plan)
        assert w["lambda_qos"] == 60.0


# ----------------------------------------------------------------------
# Predicate DSL — dormant in Slice 0 but kept tested for Slice 2.
# ----------------------------------------------------------------------

class TestPredicate:
    def test_empty_predicate_is_trivially_satisfied(self):
        assert Predicate().evaluate({}).satisfied

    def test_satisfied_when_all_leaves_pass(self):
        pred = Predicate(
            leaves=[
                Comparison(metric="unsafe_pct", op="<", threshold=1.0),
                Comparison(metric="energy_wh", op="<", threshold=2000.0),
            ]
        )
        r = pred.evaluate({"unsafe_pct": 0.5, "energy_wh": 1900.0})
        assert r.satisfied
        assert all(leaf.satisfied for leaf in r.witness.values())

    def test_unsatisfied_when_any_leaf_fails(self):
        pred = Predicate(
            leaves=[
                Comparison(metric="unsafe_pct", op="<", threshold=1.0),
                Comparison(metric="energy_wh", op="<", threshold=2000.0),
            ]
        )
        r = pred.evaluate({"unsafe_pct": 0.5, "energy_wh": 2100.0})
        assert not r.satisfied
        # Witness should pinpoint the failing leaf.
        bad = [k for k, leaf in r.witness.items() if not leaf.satisfied]
        assert len(bad) == 1 and "energy_wh" in bad[0]

    def test_missing_metric_raises(self):
        pred = Predicate(leaves=[Comparison(metric="mean_q", op=">", threshold=0.9)])
        with pytest.raises(KeyError):
            pred.evaluate({"unsafe_pct": 0.5})

    def test_invalid_op_rejected(self):
        with pytest.raises(ValidationError):
            Comparison(metric="unsafe_pct", op="~~", threshold=1.0)

    def test_invalid_metric_rejected(self):
        with pytest.raises(ValidationError):
            Comparison(metric="not_a_real_metric", op="<", threshold=1.0)

    @pytest.mark.parametrize("op,v,thr,expect", [
        ("<",  1.0, 2.0, True),
        ("<",  2.0, 2.0, False),
        ("<=", 2.0, 2.0, True),
        (">",  3.0, 2.0, True),
        (">=", 2.0, 2.0, True),
        ("==", 2.0, 2.0, True),
        ("!=", 2.0, 2.0, False),
    ])
    def test_comparison_ops(self, op, v, thr, expect):
        pred = Predicate(leaves=[Comparison(metric="mean_q", op=op, threshold=thr)])
        assert pred.evaluate({"mean_q": v}).satisfied == expect


# ----------------------------------------------------------------------
# Metric normalisation
# ----------------------------------------------------------------------

class TestNormalizeMetrics:
    def test_aliases_map_correctly(self):
        raw = {
            "total_reward": -5000.0,
            "mean_reward": -5.0,
            "total_energy_wh": 1990.0,
            "total_switch_wh": 5.0,
            "mean_sec": 0.05,
            "mean_q": 0.99,
            "qos_violation_rate": 0.01,
            "unsafe_rate": 0.005,
            "dpdk_rate": 0.5,
            "usr_rate": 0.5,
            "n_switches": 42,
            "steps": 1009,
        }
        out = normalize_metrics(raw)
        assert out["energy_wh"] == 1990.0
        assert out["flips"] == 42.0
        assert out["unsafe_pct"] == pytest.approx(0.5)
        assert out["qos_violation_pct"] == pytest.approx(1.0)
        for name in METRIC_VOCAB:
            assert name in out, name
