import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type {
  MultiCompareResponse,
  MultiPolicyId,
  SplitId,
} from "../api/types";
import { ActionHeatmap } from "../components/multi/ActionHeatmap";
import { CumulativeRewardChart } from "../components/multi/CumulativeRewardChart";
import { MultiControls } from "../components/multi/MultiControls";
import { MultiSummaryTable } from "../components/multi/MultiSummaryTable";
import { PerClusterRewardBars } from "../components/multi/PerClusterRewardBars";

const DEFAULT_SELECTED: MultiPolicyId[] = [
  "mappo",
  "all-dpdk",
  "hysteresis-tuned",
];

export function MultiSitePage() {
  const policiesQ = useQuery({
    queryKey: ["multi-policies"],
    queryFn: api.listMultiPolicies,
  });
  const derivedQ = useQuery({
    queryKey: ["multi-derived"],
    queryFn: api.derivedThresholds,
  });

  const [split, setSplit] = useState<SplitId>("test");
  const [horizonIdx, setHorizonIdx] = useState(0);
  const [maxSteps, setMaxSteps] = useState<number | null>(1009);
  const [seed, setSeed] = useState(42);
  const [thresholdMbps, setThresholdMbps] = useState(81);
  const [hysteresisBandMbps, setHysteresisBandMbps] = useState(20);
  const [hysteresisCooldown, setHysteresisCooldown] = useState(1);
  const [selected, setSelected] = useState<MultiPolicyId[]>(DEFAULT_SELECTED);
  const [result, setResult] = useState<MultiCompareResponse | null>(null);

  // Initialise threshold field from the derived value once it loads.
  useEffect(() => {
    if (derivedQ.data && thresholdMbps === 81) {
      setThresholdMbps(Math.round(derivedQ.data.decision_gbps * 1000));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [derivedQ.data]);

  // Drop "mappo" from the default selection if the checkpoint is missing.
  useEffect(() => {
    if (!policiesQ.data) return;
    const m = policiesQ.data.find((p) => p.id === "mappo");
    if (m && !m.model_loaded) {
      setSelected((prev) => prev.filter((p) => p !== "mappo"));
    }
  }, [policiesQ.data]);

  const toggle = (p: MultiPolicyId) =>
    setSelected((prev) =>
      prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p],
    );

  const compare = useMutation({
    mutationFn: api.multiCompare,
    onSuccess: (data) => setResult(data),
  });

  const onRun = () => {
    compare.mutate({
      policies: selected,
      horizon_idx: horizonIdx,
      split,
      seed,
      max_steps: maxSteps,
      threshold_gbps: thresholdMbps / 1000,
      hysteresis_band_mbps: hysteresisBandMbps,
      hysteresis_cooldown_steps: hysteresisCooldown,
    });
  };

  const summaries = useMemo(
    () => (result ? result.rollouts.map((r) => r.summary) : []),
    [result],
  );

  if (policiesQ.isLoading) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600 shadow-sm">
        Loading multi-cluster metadata…
      </div>
    );
  }
  if (policiesQ.isError) {
    return (
      <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700 shadow-sm">
        Backend unreachable.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <MultiControls
        policies={policiesQ.data!}
        selected={selected}
        onToggle={toggle}
        split={split}
        setSplit={setSplit}
        horizonIdx={horizonIdx}
        setHorizonIdx={setHorizonIdx}
        maxSteps={maxSteps}
        setMaxSteps={setMaxSteps}
        seed={seed}
        setSeed={setSeed}
        thresholdMbps={thresholdMbps}
        setThresholdMbps={setThresholdMbps}
        hysteresisBandMbps={hysteresisBandMbps}
        setHysteresisBandMbps={setHysteresisBandMbps}
        hysteresisCooldown={hysteresisCooldown}
        setHysteresisCooldown={setHysteresisCooldown}
        derived={derivedQ.data ?? null}
        loading={compare.isPending}
        onRun={onRun}
      />

      {compare.isError && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700 shadow-sm">
          Rollout failed:{" "}
          {(compare.error as Error)?.message ?? "unknown error"}
        </div>
      )}

      {result && (
        <>
          <MultiSummaryTable summaries={summaries} />
          <PerClusterRewardBars rollouts={result.rollouts} />
          <CumulativeRewardChart rollouts={result.rollouts} />
          <div className="grid gap-4 xl:grid-cols-2">
            {result.rollouts.map((r) => (
              <ActionHeatmap key={r.summary.policy} rollout={r} />
            ))}
          </div>
        </>
      )}

      {!result && !compare.isPending && (
        <div className="rounded-xl border border-dashed border-slate-300 bg-white/50 p-8 text-center text-sm text-slate-500">
          Pick policies and click <strong>Run comparison</strong>. The first
          rollout includes ~13 s of surrogate precompute (10 sub-envs); later
          rollouts are fast.
        </div>
      )}
    </div>
  );
}
