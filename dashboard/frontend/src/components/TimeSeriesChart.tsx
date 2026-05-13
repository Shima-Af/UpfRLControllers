import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";
import type { PolicyId, RolloutResponse } from "../api/types";

const POLICY_HEX: Record<string, string> = {
  ppo: "#0ea5e9",
  random: "#94a3b8",
  "always-dpdk": "#10b981",
  "always-usr": "#f59e0b",
  threshold: "#a855f7",
};

interface Props {
  title: string;
  yLabel: string;
  /** Field to read from each StepRecord. */
  field:
    | "actual_load_gbps"
    | "predicted_load_gbps"
    | "power_watts"
    | "delay_us"
    | "predicted_loss"
    | "performance"
    | "cumulative_reward"
    | "action";
  rollouts: RolloutResponse[];
  /** Horizontal reference line (e.g. delay budget). */
  refLine?: { y: number; label: string };
  height?: number;
  domain?: [number | "auto", number | "auto"];
}

/** A single time-series chart with one line per policy. */
export function TimeSeriesChart({
  title,
  yLabel,
  field,
  rollouts,
  refLine,
  height = 220,
  domain,
}: Props) {
  // Merge per-step values across all rollouts into a single array keyed by t,
  // so Recharts can render one Line per policy.
  const merged: Record<number, Record<string, number>> = {};
  for (const r of rollouts) {
    for (const s of r.steps) {
      if (!merged[s.t]) merged[s.t] = { t: s.t };
      const val = s[field] as number | boolean;
      merged[s.t][r.summary.policy] =
        typeof val === "boolean" ? (val ? 1 : 0) : (val as number);
    }
  }
  const data = Object.values(merged).sort((a, b) => a.t - b.t);

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-slate-700">{title}</h3>
        <span className="text-xs text-slate-500">{yLabel}</span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
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
          <YAxis
            tick={{ fontSize: 11 }}
            stroke="#94a3b8"
            domain={domain}
            width={56}
          />
          <Tooltip
            formatter={(v: number) =>
              typeof v === "number" ? v.toFixed(3) : v
            }
            labelFormatter={(t) => `t=${t}`}
            contentStyle={{ fontSize: 12 }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {refLine && (
            <ReferenceLine
              y={refLine.y}
              stroke="#94a3b8"
              strokeDasharray="4 2"
              label={{
                value: refLine.label,
                position: "right",
                fontSize: 10,
                fill: "#94a3b8",
              }}
            />
          )}
          {rollouts.map((r) => (
            <Line
              key={r.summary.policy}
              type="monotone"
              dataKey={r.summary.policy as PolicyId}
              name={r.summary.label}
              stroke={POLICY_HEX[r.summary.policy] ?? "#475569"}
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
