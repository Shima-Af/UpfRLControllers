"""Predicate DSL — conjunction of leaf comparisons over the metric vocabulary.

Slice 0 supports AND-of-comparisons only. If intents later demand OR /
NOT we'll extend; for now a flat AND is enough for every canonical
plan in `intent/examples` and keeps the evaluator trivially auditable.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intent.schema.metrics import MetricName
from intent.schema.plan import ComparisonOp


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
