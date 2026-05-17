import type { MultiClusterSummary } from "../../api/types";

const POLICY_BG: Record<string, string> = {
  mappo: "bg-rose-50",
  "all-dpdk": "bg-emerald-50",
  "threshold-derived": "bg-pink-50",
  "hysteresis-tuned": "bg-purple-50",
  "hysteresis-auto": "bg-violet-50",
};

export function MultiSummaryTable({
  summaries,
}: {
  summaries: MultiClusterSummary[];
}) {
  if (summaries.length === 0) return null;

  // Find the best (least negative) total reward for relative comparison.
  const best = Math.max(...summaries.map((s) => s.total_reward_unweighted));

  return (
    <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
            <th className="px-3 py-2 text-left">Policy</th>
            <th className="px-3 py-2 text-right">Total reward</th>
            <th className="px-3 py-2 text-right">Δ vs best</th>
            <th className="px-3 py-2 text-right">Energy Wh</th>
            <th className="px-3 py-2 text-right">Unsafe %</th>
            <th className="px-3 py-2 text-right">USR %</th>
            <th className="px-3 py-2 text-right">Switches</th>
            <th className="px-3 py-2 text-right">Steps</th>
          </tr>
        </thead>
        <tbody>
          {summaries.map((s) => (
            <tr
              key={s.policy}
              className={`${POLICY_BG[s.policy] ?? ""} border-t border-slate-100`}
            >
              <td className="px-3 py-2 font-medium text-slate-800">
                {s.label}
              </td>
              <td className="px-3 py-2 text-right font-mono tabular-nums">
                {s.total_reward_unweighted.toFixed(2)}
              </td>
              <td className="px-3 py-2 text-right font-mono tabular-nums text-slate-500">
                {(s.total_reward_unweighted - best).toFixed(2)}
              </td>
              <td className="px-3 py-2 text-right font-mono tabular-nums">
                {s.total_energy_wh.toFixed(1)}
              </td>
              <td className="px-3 py-2 text-right font-mono tabular-nums">
                {(s.agg_unsafe_rate * 100).toFixed(2)}
              </td>
              <td className="px-3 py-2 text-right font-mono tabular-nums">
                {(s.agg_usr_rate * 100).toFixed(1)}
              </td>
              <td className="px-3 py-2 text-right font-mono tabular-nums">
                {s.agg_n_switches}
              </td>
              <td className="px-3 py-2 text-right font-mono tabular-nums text-slate-500">
                {s.steps}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
