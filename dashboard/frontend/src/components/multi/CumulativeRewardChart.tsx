import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
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

export function CumulativeRewardChart({
  rollouts,
}: {
  rollouts: MultiClusterRolloutResponse[];
}) {
  if (rollouts.length === 0) return null;
  const T = Math.min(...rollouts.map((r) => r.cumulative_reward.length));

  // Subsample if very long — Recharts struggles past ~2000 points per line.
  const stride = Math.max(1, Math.floor(T / 600));
  const data: Record<string, number>[] = [];
  for (let t = 0; t < T; t += stride) {
    const row: Record<string, number> = { t };
    for (const r of rollouts) {
      row[r.summary.policy] = r.cumulative_reward[t];
    }
    data.push(row);
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-slate-700">
          Cumulative aggregate reward
        </h3>
        <span className="text-xs text-slate-500">
          sum over clusters · stride = {stride}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={data}
          margin={{ top: 4, right: 16, left: 0, bottom: 4 }}
        >
          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
          <XAxis
            dataKey="t"
            tick={{ fontSize: 11 }}
            stroke="#94a3b8"
            label={{
              value: "timestep",
              position: "insideBottom",
              offset: -2,
              fontSize: 11,
              fill: "#94a3b8",
            }}
          />
          <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" width={64} />
          <Tooltip
            formatter={(v: number) =>
              typeof v === "number" ? v.toFixed(1) : v
            }
            labelFormatter={(t) => `t=${t}`}
            contentStyle={{ fontSize: 12 }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {rollouts.map((r) => (
            <Line
              key={r.summary.policy}
              type="monotone"
              dataKey={r.summary.policy}
              name={r.summary.label}
              stroke={POLICY_HEX[r.summary.policy] ?? "#475569"}
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
