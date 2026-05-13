import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "./api/client";
import type { CompareResponse, PolicyId } from "./api/types";
import { Controls } from "./components/Controls";
import { SummaryTable } from "./components/SummaryTable";
import { TimeSeriesChart } from "./components/TimeSeriesChart";

const DEFAULT_SELECTED: PolicyId[] = ["ppo", "threshold", "always-dpdk"];

export default function App() {
  const policiesQ = useQuery({
    queryKey: ["policies"],
    queryFn: api.listPolicies,
  });
  const clustersQ = useQuery({
    queryKey: ["clusters"],
    queryFn: api.listClusters,
  });

  const [clusterIdx, setClusterIdx] = useState<number>(0);
  const [horizonIdx, setHorizonIdx] = useState<number>(0);
  const [thresholdGbps, setThresholdGbps] = useState<number>(0.05);
  const [maxSteps, setMaxSteps] = useState<number | null>(300);
  const [selectedPolicies, setSelectedPolicies] =
    useState<PolicyId[]>(DEFAULT_SELECTED);
  const [result, setResult] = useState<CompareResponse | null>(null);

  const compare = useMutation({
    mutationFn: api.compare,
    onSuccess: (data) => setResult(data),
  });

  // Auto-deselect a "ppo" policy if no checkpoint is present.
  useEffect(() => {
    if (!policiesQ.data) return;
    const ppoEntry = policiesQ.data.find((p) => p.id === "ppo");
    if (ppoEntry && !ppoEntry.model_loaded) {
      setSelectedPolicies((prev) => prev.filter((p) => p !== "ppo"));
    }
  }, [policiesQ.data]);

  const togglePolicy = (p: PolicyId) =>
    setSelectedPolicies((prev) =>
      prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p],
    );

  const runCompare = () => {
    compare.mutate({
      cluster_idx: clusterIdx,
      horizon_idx: horizonIdx,
      policies: selectedPolicies,
      threshold_gbps: thresholdGbps,
      max_steps: maxSteps ?? null,
      seed: 42,
    });
  };

  const summaries = useMemo(
    () => (result ? result.rollouts.map((r) => r.summary) : []),
    [result],
  );
  const rollouts = result?.rollouts ?? [];

  return (
    <div className="min-h-full bg-slate-50 px-6 pb-10 pt-6">
      <header className="mx-auto mb-6 max-w-7xl">
        <h1 className="text-2xl font-bold text-slate-900">
          UPF RL Controllers — Single-Site Digital Twin Dashboard
        </h1>
        <p className="mt-1 text-sm text-slate-600">
          Replay one episode under each selected policy and compare KPIs.
          Backend at <code className="rounded bg-slate-100 px-1">/api</code>.
        </p>
      </header>

      <main className="mx-auto max-w-7xl space-y-4">
        {policiesQ.isLoading || clustersQ.isLoading ? (
          <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600 shadow-sm">
            Loading metadata from backend…
          </div>
        ) : policiesQ.isError || clustersQ.isError ? (
          <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700 shadow-sm">
            Backend unreachable. Start it with{" "}
            <code className="rounded bg-rose-100 px-1">
              uvicorn dashboard.backend.app.main:app --port 8000
            </code>{" "}
            and refresh.
          </div>
        ) : (
          <>
            <Controls
              policies={policiesQ.data!}
              clusters={clustersQ.data!}
              selectedPolicies={selectedPolicies}
              onTogglePolicy={togglePolicy}
              clusterIdx={clusterIdx}
              setClusterIdx={setClusterIdx}
              horizonIdx={horizonIdx}
              setHorizonIdx={setHorizonIdx}
              thresholdGbps={thresholdGbps}
              setThresholdGbps={setThresholdGbps}
              maxSteps={maxSteps}
              setMaxSteps={setMaxSteps}
              loading={compare.isPending}
              onRun={runCompare}
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
                    title="Performance score"
                    yLabel="[0, 1]"
                    field="performance"
                    rollouts={rollouts}
                    refLine={{ y: 0.9, label: "0.9 threshold" }}
                    domain={[0, 1]}
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
                </div>
              </>
            )}

            {!result && !compare.isPending && (
              <div className="rounded-xl border border-dashed border-slate-300 bg-white/50 p-8 text-center text-sm text-slate-500">
                Pick policies and click <strong>Run comparison</strong>.
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
