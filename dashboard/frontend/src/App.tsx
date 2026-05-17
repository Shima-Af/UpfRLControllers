import { useState } from "react";
import { MultiSitePage } from "./pages/MultiSitePage";
import { SingleSitePage } from "./pages/SingleSitePage";

type Tab = "single" | "multi";

const TABS: { id: Tab; label: string; subtitle: string }[] = [
  {
    id: "single",
    label: "Single-site",
    subtitle: "One cluster · per-step diagnostics · Phase 2 PPO",
  },
  {
    id: "multi",
    label: "Multi-cluster",
    subtitle: "K=10 clusters · MAPPO vs baselines · Phase 7",
  },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("multi");

  return (
    <div className="min-h-full bg-slate-50 px-6 pb-10 pt-6">
      <header className="mx-auto mb-4 max-w-7xl">
        <h1 className="text-2xl font-bold text-slate-900">
          UPF RL Controllers — Digital Twin Dashboard
        </h1>
        <p className="mt-1 text-sm text-slate-600">
          Replay one episode under each selected policy and compare KPIs.
          Backend at <code className="rounded bg-slate-100 px-1">/api</code>.
        </p>
      </header>

      <nav className="mx-auto mb-4 max-w-7xl">
        <div className="flex gap-1 rounded-xl border border-slate-200 bg-white p-1 shadow-sm">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex-1 rounded-lg px-3 py-2 text-left transition ${
                tab === t.id
                  ? "bg-slate-900 text-white shadow-sm"
                  : "text-slate-700 hover:bg-slate-100"
              }`}
            >
              <div className="text-sm font-semibold">{t.label}</div>
              <div
                className={`text-xs ${
                  tab === t.id ? "text-slate-300" : "text-slate-500"
                }`}
              >
                {t.subtitle}
              </div>
            </button>
          ))}
        </div>
      </nav>

      <main className="mx-auto max-w-7xl">
        {tab === "single" ? <SingleSitePage /> : <MultiSitePage />}
      </main>
    </div>
  );
}
