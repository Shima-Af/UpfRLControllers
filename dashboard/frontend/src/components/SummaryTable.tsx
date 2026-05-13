import type { RolloutSummary } from "../api/types";

const POLICY_COLOR: Record<string, string> = {
  ppo: "bg-sky-500",
  random: "bg-slate-400",
  "always-dpdk": "bg-emerald-500",
  "always-usr": "bg-amber-500",
  threshold: "bg-purple-500",
};

interface Props {
  summaries: RolloutSummary[];
}

/** Best-by-total-reward callout + per-policy row table. */
export function SummaryTable({ summaries }: Props) {
  if (summaries.length === 0) return null;
  const best = [...summaries].sort(
    (a, b) => b.total_reward - a.total_reward,
  )[0];

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-200 px-4 py-3">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-semibold text-slate-700">
            Comparison summary
          </h2>
          <div className="text-xs text-slate-500">
            best:{" "}
            <span className="font-semibold text-slate-700">
              {best.label}
            </span>{" "}
            ({best.total_reward.toFixed(2)})
          </div>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-4 py-2 text-left">policy</th>
              <th className="px-4 py-2 text-right">total reward</th>
              <th className="px-4 py-2 text-right">energy Wh</th>
              <th className="px-4 py-2 text-right">switch Wh</th>
              <th className="px-4 py-2 text-right">unsafe %</th>
              <th className="px-4 py-2 text-right">DPDK %</th>
              <th className="px-4 py-2 text-right">USR %</th>
              <th className="px-4 py-2 text-right">flips</th>
              <th className="px-4 py-2 text-right">steps</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {summaries.map((s) => {
              const isBest = s.policy === best.policy;
              return (
                <tr
                  key={s.policy}
                  className={isBest ? "bg-emerald-50/60" : ""}
                >
                  <td className="px-4 py-2">
                    <div className="flex items-center gap-2">
                      <span
                        className={`h-2.5 w-2.5 rounded-full ${
                          POLICY_COLOR[s.policy] ?? "bg-slate-300"
                        }`}
                      />
                      <span className="font-medium text-slate-700">
                        {s.label}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums">
                    {s.total_reward.toFixed(2)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums">
                    {s.total_energy_wh.toFixed(2)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums">
                    {s.total_switch_wh.toFixed(2)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums">
                    {(100 * s.unsafe_rate).toFixed(1)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums">
                    {(100 * s.dpdk_rate).toFixed(0)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums">
                    {(100 * s.usr_rate).toFixed(0)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums">
                    {s.n_switches}
                  </td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums text-slate-500">
                    {s.steps}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
