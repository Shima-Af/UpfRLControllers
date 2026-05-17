import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MultiClusterRolloutResponse } from "../../api/types";

const POLICY_HEX: Record<string, string> = {
  mappo: "#ef4444",
  "all-dpdk": "#10b981",
  "threshold-derived": "#ec4899",
  "hysteresis-tuned": "#a855f7",
  "hysteresis-auto": "#8b5cf6",
};

/** Per-cluster grouped-bar chart of summed reward across all policies. */
export function PerClusterRewardBars({
  rollouts,
}: {
  rollouts: MultiClusterRolloutResponse[];
}) {
  if (rollouts.length === 0) return null;
  const K = rollouts[0].summary.K;

  // Build [{cluster: "c0", mappo: -800, "all-dpdk": -1321, ...}, ...]
  const data = Array.from({ length: K }, (_, k) => {
    const row: Record<string, number | string> = { cluster: `c${k}` };
    for (const r of rollouts) {
      row[r.summary.policy] = r.summary.per_cluster_total_reward[k];
    }
    return row;
  });

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-slate-700">
          Per-cluster total reward
        </h3>
        <span className="text-xs text-slate-500">summed over episode</span>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={data}
          margin={{ top: 4, right: 16, left: 0, bottom: 4 }}
        >
          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
          <XAxis dataKey="cluster" tick={{ fontSize: 11 }} stroke="#94a3b8" />
          <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" width={64} />
          <Tooltip
            formatter={(v: number) =>
              typeof v === "number" ? v.toFixed(2) : v
            }
            contentStyle={{ fontSize: 12 }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {rollouts.map((r) => (
            <Bar
              key={r.summary.policy}
              dataKey={r.summary.policy}
              name={r.summary.label}
              fill={POLICY_HEX[r.summary.policy] ?? "#475569"}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
