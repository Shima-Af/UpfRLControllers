import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type {
  CompareResponse,
  PolicyId,
  SplitId,
} from "../api/types";
import { Controls } from "../components/Controls";
import { SummaryTable } from "../components/SummaryTable";
import { TimeSeriesChart } from "../components/TimeSeriesChart";

const DEFAULT_SELECTED: PolicyId[] = [
  "ppo",
  "hysteresis",
  "always-dpdk",
];

export function SingleSitePage() {
  const policiesQ = useQuery({
    queryKey: ["policies"],
    queryFn: api.listPolicies,
  });
  const clustersQ = useQuery({
    queryKey: ["clusters"],
    queryFn: api.listClusters,
  });

  const [clusterIdx, setClusterIdx] = useState(0);
  const [horizonIdx, setHorizonIdx] = useState(0);
  const [thresholdGbps, setThresholdGbps] = useState(0.081);
  const [hysteresisBandMbps, setHysteresisBandMbps] = useState(20);
  const [hysteresisCooldown, setHysteresisCooldown] = useState(1);
  const [split, setSplit] = useState<SplitId>("test");
  const [maxSteps, setMaxSteps] = useState<number | null>(1009);
  const [selectedPolicies, setSelectedPolicies] =
    useState<PolicyId[]>(DEFAULT_SELECTED);
  const [result, setResult] = useState<CompareResponse | null>(null);

  const compare = useMutation({
    mutationFn: api.compare,
    onSuccess: (data) => setResult(data),
  });

  useEffect(() => {
    if (!policiesQ.data) return;
    const ppoEntry = policiesQ.data.find((p) => p.id === "ppo");
    if (ppoEntry && !ppoEntry.model_loaded) {
      setSelectedPolicies((prev) => prev.filter((p) => p !== "ppo"));
    }
  }, [policiesQ.data]);

  const toggle = (p: PolicyId) =>
    setSelectedPolicies((prev) =>
      prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p],
    );

  const onRun = () => {
    compare.mutate({
      cluster_idx: clusterIdx,
      horizon_idx: horizonIdx,
      policies: selectedPolicies,
      threshold_gbps: thresholdGbps,
      hysteresis_band_mbps: hysteresisBandMbps,
      hysteresis_cooldown_steps: hysteresisCooldown,
      max_steps: maxSteps,
      seed: 42,
      split,
    });
  };

  const summaries = useMemo(
    () => (result ? result.rollouts.map((r) => r.summary) : []),
    [result],
  );
  const rollouts = result?.rollouts ?? [];

  if (policiesQ.isLoading || clustersQ.isLoading) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600 shadow-sm">
        Loading metadata from backend…
      </div>
    );
  }
  if (policiesQ.isError || clustersQ.isError) {
    return (
      <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700 shadow-sm">
        Backend unreachable. Start it with{" "}
        <code className="rounded bg-rose-100 px-1">
          uvicorn dashboard.backend.app.main:app --port 8000
        </code>{" "}
        and refresh.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Controls
        policies={policiesQ.data!}
        clusters={clustersQ.data!}
        selectedPolicies={selectedPolicies}
        onTogglePolicy={toggle}
        clusterIdx={clusterIdx}
        setClusterIdx={setClusterIdx}
        horizonIdx={horizonIdx}
        setHorizonIdx={setHorizonIdx}
        thresholdGbps={thresholdGbps}
        setThresholdGbps={setThresholdGbps}
        hysteresisBandMbps={hysteresisBandMbps}
        setHysteresisBandMbps={setHysteresisBandMbps}
        hysteresisCooldown={hysteresisCooldown}
        setHysteresisCooldown={setHysteresisCooldown}
        split={split}
        setSplit={setSplit}
        maxSteps={maxSteps}
        setMaxSteps={setMaxSteps}
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
          <SummaryTable summaries={summaries} />
          <div className="grid gap-4 xl:grid-cols-2">
            <TimeSeriesChart
              title="Offered load"
              yLabel="Gbps"
              field="actual_load_gbps"
              rollouts={rollouts}
            />
            <TimeSeriesChart
              title="Power consumed"
              yLabel="watts"
              field="power_watts"
              rollouts={rollouts}
            />
            <TimeSeriesChart
              title="Predicted delay"
              yLabel="μs"
              field="delay_us"
              rollouts={rollouts}
              refLine={{ y: 200, label: "200 μs budget" }}
            />
            <TimeSeriesChart
              title="Predicted packet loss"
              yLabel="pkts/interval"
              field="predicted_loss"
              rollouts={rollouts}
              refLine={{ y: 5, label: "5 pkts budget" }}
            />
            <TimeSeriesChart
              title="QoS score Q"
              yLabel="[0, 1]"
              field="q_score"
              rollouts={rollouts}
              refLine={{ y: 0.9, label: "τ = 0.9" }}
              domain={[0, 1]}
            />
            <TimeSeriesChart
              title="Specific energy (SEC)"
              yLabel="W/Mbps"
              field="sec_w_per_mbps"
              rollouts={rollouts}
            />
            <TimeSeriesChart
              title="Cumulative reward"
              yLabel="reward"
              field="cumulative_reward"
              rollouts={rollouts}
            />
            <TimeSeriesChart
              title="Action (0 = DPDK, 1 = USR)"
              yLabel="discrete"
              field="action"
              rollouts={rollouts}
              domain={[-0.1, 1.1]}
              height={140}
            />
            <TimeSeriesChart
              title="Steps since last switch"
              yLabel="steps"
              field="steps_since_switch"
              rollouts={rollouts}
              refLine={{ y: 4, label: "cooldown period" }}
            />
          </div>
        </>
      )}

      {!result && !compare.isPending && (
        <div className="rounded-xl border border-dashed border-slate-300 bg-white/50 p-8 text-center text-sm text-slate-500">
          Pick policies and click <strong>Run comparison</strong>.
        </div>
      )}
    </div>
  );
}
