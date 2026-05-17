"""Predicate DSL — conjunction of leaf comparisons over the metric
vocabulary.

DORMANT in Slice 0 (not wired into the CLI or canonical Plans). The
DSL lives here so Slice 2's LLM explainer / runtime auditor has a
ready acceptance-criterion language to consume. AND-of-comparisons
is intentionally the simplest defensible shape; extend to OR / NOT
only when an intent actually demands it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from intent.schema.metrics import MetricName

ComparisonOp = Literal["<", "<=", ">", ">=", "==", "!="]


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: MetricName
    op: ComparisonOp
    threshold: float

    def label(self) -> str:
        return f"{self.metric} {self.op} {self.threshold}"


class LeafResult(BaseModel):
    value: float
    threshold: float
    op: ComparisonOp
    satisfied: bool


class EvalResult(BaseModel):
    satisfied: bool
    witness: dict[str, LeafResult]


class Predicate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    leaves: list[Comparison] = Field(default_factory=list)

    def evaluate(self, metrics: dict[str, float]) -> EvalResult:
        witness: dict[str, LeafResult] = {}
        all_ok = True
        for c in self.leaves:
            if c.metric not in metrics:
                raise KeyError(
                    f"metric {c.metric!r} not present in metrics dict "
                    f"(have: {sorted(metrics)})"
                )
            v = float(metrics[c.metric])
            ok = _cmp(v, c.op, c.threshold)
            witness[c.label()] = LeafResult(
                value=v, threshold=c.threshold, op=c.op, satisfied=ok
            )
            all_ok = all_ok and ok
        return EvalResult(satisfied=all_ok, witness=witness)


def _cmp(value: float, op: ComparisonOp, threshold: float) -> bool:
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    raise ValueError(f"Unknown op: {op!r}")
