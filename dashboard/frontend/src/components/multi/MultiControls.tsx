import type {
  DerivedThresholds,
  MultiPolicyId,
  MultiPolicyInfo,
  SplitId,
} from "../../api/types";

interface Props {
  policies: MultiPolicyInfo[];
  selected: MultiPolicyId[];
  onToggle: (p: MultiPolicyId) => void;
  split: SplitId;
  setSplit: (s: SplitId) => void;
  horizonIdx: number;
  setHorizonIdx: (v: number) => void;
  maxSteps: number | null;
  setMaxSteps: (v: number | null) => void;
  thresholdMbps: number;
  setThresholdMbps: (v: number) => void;
  hysteresisBandMbps: number;
  setHysteresisBandMbps: (v: number) => void;
  hysteresisCooldown: number;
  setHysteresisCooldown: (v: number) => void;
  seed: number;
  setSeed: (v: number) => void;
  derived: DerivedThresholds | null;
  loading: boolean;
  onRun: () => void;
}

const POLICY_DOT: Record<string, string> = {
  mappo: "bg-rose-500",
  "all-dpdk": "bg-emerald-500",
  "threshold-derived": "bg-pink-500",
  "hysteresis-tuned": "bg-purple-500",
  "hysteresis-auto": "bg-violet-400",
};

export function MultiControls(p: Props) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="grid gap-3 md:grid-cols-12">
        <div className="md:col-span-2">
          <label className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Split
          </label>
          <select
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm"
            value={p.split}
            onChange={(e) => p.setSplit(e.target.value as SplitId)}
          >
            <option value="train">train (5073 steps)</option>
            <option value="val">val (1009 steps)</option>
            <option value="test">test (1009 steps)</option>
          </select>
        </div>
        <div className="md:col-span-2">
          <label className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Horizon
          </label>
          <select
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm"
            value={p.horizonIdx}
            onChange={(e) => p.setHorizonIdx(Number(e.target.value))}
          >
            {[0, 1, 2, 3].map((h) => (
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
            value={p.maxSteps ?? 1009}
            onChange={(e) =>
              p.setMaxSteps(e.target.value ? Number(e.target.value) : null)
            }
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm"
          />
        </div>
        <div className="md:col-span-2">
          <label className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Seed
          </label>
          <input
            type="number"
            value={p.seed}
            onChange={(e) => p.setSeed(Number(e.target.value) || 0)}
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm"
          />
        </div>
        <div className="flex items-end md:col-span-4">
          <button
            onClick={p.onRun}
            disabled={p.loading || p.selected.length === 0}
            className="w-full rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {p.loading ? "Running rollouts…" : "Run comparison"}
          </button>
        </div>
      </div>

      <div className="mt-3 grid gap-3 md:grid-cols-12">
        <div className="md:col-span-3">
          <label className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Threshold (Mbps)
          </label>
          <input
            type="number"
            min={1}
            value={p.thresholdMbps}
            onChange={(e) =>
              p.setThresholdMbps(Number(e.target.value) || 0)
            }
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm"
          />
        </div>
        <div className="md:col-span-3">
          <label className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Hyst. band (Mbps)
          </label>
          <input
            type="number"
            min={0}
            value={p.hysteresisBandMbps}
            onChange={(e) =>
              p.setHysteresisBandMbps(Number(e.target.value) || 0)
            }
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm"
          />
        </div>
        <div className="md:col-span-2">
          <label className="block text-xs font-medium uppercase tracking-wide text-slate-500">
            Hyst. cooldown
          </label>
          <input
            type="number"
            min={0}
            value={p.hysteresisCooldown}
            onChange={(e) => p.setHysteresisCooldown(Number(e.target.value) || 0)}
            className="mt-1 w-full rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm"
          />
        </div>
        {p.derived && (
          <div className="md:col-span-4 rounded-md border border-slate-100 bg-slate-50 px-3 py-1.5 text-xs text-slate-600">
            <div>
              <span className="font-semibold">Derived</span> (from twin
              surrogate):
            </div>
            <div>
              decision = {(p.derived.decision_gbps * 1000).toFixed(1)} Mbps ·
              breakeven = {(p.derived.energy_breakeven_gbps * 1000).toFixed(1)}{" "}
              · QoS limit = {(p.derived.qos_limit_gbps * 1000).toFixed(1)}
            </div>
            <div>
              forecast MAE = {(p.derived.forecast_mae_gbps * 1000).toFixed(1)}{" "}
              Mbps → auto band ={" "}
              {(p.derived.hysteresis_band_gbps_auto * 1000).toFixed(1)} Mbps
            </div>
          </div>
        )}
      </div>

      <div className="mt-3">
        <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
          Policies to compare
        </div>
        <div className="mt-2 flex flex-wrap gap-2">
          {p.policies.map((pol) => {
            const checked = p.selected.includes(pol.id);
            const disabled = pol.requires_model && !pol.model_loaded;
            return (
              <button
                key={pol.id}
                onClick={() => !disabled && p.onToggle(pol.id)}
                disabled={disabled}
                title={
                  pol.description +
                  (disabled ? " (no model checkpoint found)" : "")
                }
                className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm transition ${
                  checked
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-200 bg-white text-slate-700 hover:border-slate-400"
                } ${disabled ? "cursor-not-allowed opacity-50" : ""}`}
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    POLICY_DOT[pol.id] ?? "bg-slate-300"
                  }`}
                />
                {pol.label}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
