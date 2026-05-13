import type { ClusterInfo, PolicyId, PolicyInfo } from "../api/types";

interface Props {
  policies: PolicyInfo[];
  clusters: ClusterInfo[];
  selectedPolicies: PolicyId[];
  onTogglePolicy: (p: PolicyId) => void;
  clusterIdx: number;
  setClusterIdx: (v: number) => void;
  horizonIdx: number;
  setHorizonIdx: (v: number) => void;
  thresholdGbps: number;
  setThresholdGbps: (v: number) => void;
  maxSteps: number | null;
  setMaxSteps: (v: number | null) => void;
  loading: boolean;
  onRun: () => void;
}

const POLICY_DOT: Record<string, string> = {
  ppo: "bg-sky-500",
  random: "bg-slate-400",
  "always-dpdk": "bg-emerald-500",
  "always-usr": "bg-amber-500",
  threshold: "bg-purple-500",
};

export function Controls(props: Props) {
  const cluster = props.clusters.find(
    (c) => c.cluster_idx === props.clusterIdx,
  );

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="grid gap-4 md:grid-cols-12">
        {/* Cluster + horizon */}
        <div className="md:col-span-3">
          <label className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Cluster
          </label>
          <select
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm"
            value={props.clusterIdx}
            onChange={(e) => props.setClusterIdx(Number(e.target.value))}
          >
            {props.clusters.map((c) => (
              <option key={c.cluster_idx} value={c.cluster_idx}>
                cluster {c.cluster_idx} — mean {c.load_mean_gbps.toFixed(3)}{" "}
                Gbps · max {c.load_max_gbps.toFixed(2)}
              </option>
            ))}
          </select>
        </div>
        <div className="md:col-span-2">
          <label className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Horizon
          </label>
          <select
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm"
            value={props.horizonIdx}
            onChange={(e) => props.setHorizonIdx(Number(e.target.value))}
          >
            {Array.from(
              { length: cluster?.horizon_count ?? 4 },
              (_, i) => i,
            ).map((h) => (
              <option key={h} value={h}>
                horizon {h}
              </option>
            ))}
          </select>
        </div>
        <div className="md:col-span-2">
          <label className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Max steps
          </label>
          <input
            type="number"
            min={10}
            max={cluster?.episode_length ?? 1009}
            value={props.maxSteps ?? cluster?.episode_length ?? 1009}
            onChange={(e) =>
              props.setMaxSteps(
                e.target.value ? Number(e.target.value) : null,
              )
            }
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm"
          />
        </div>
        <div className="md:col-span-2">
          <label className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Threshold (Gbps)
          </label>
          <input
            type="number"
            step={0.01}
            min={0}
            value={props.thresholdGbps}
            onChange={(e) =>
              props.setThresholdGbps(Number(e.target.value) || 0)
            }
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm"
          />
        </div>
        <div className="flex items-end md:col-span-3">
          <button
            onClick={props.onRun}
            disabled={props.loading || props.selectedPolicies.length === 0}
            className="w-full rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {props.loading ? "Running rollouts…" : "Run comparison"}
          </button>
        </div>
      </div>

      <div className="mt-4">
        <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
          Policies to compare
        </div>
        <div className="mt-2 flex flex-wrap gap-2">
          {props.policies.map((p) => {
            const checked = props.selectedPolicies.includes(p.id);
            const disabled = p.requires_model && !p.model_loaded;
            return (
              <button
                key={p.id}
                onClick={() => !disabled && props.onTogglePolicy(p.id)}
                disabled={disabled}
                title={p.description + (disabled ? " (no model checkpoint found)" : "")}
                className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm transition ${
                  checked
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-200 bg-white text-slate-700 hover:border-slate-400"
                } ${disabled ? "cursor-not-allowed opacity-50" : ""}`}
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    POLICY_DOT[p.id] ?? "bg-slate-300"
                  }`}
                />
                {p.label}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
