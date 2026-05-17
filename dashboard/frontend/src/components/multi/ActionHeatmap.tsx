import { useEffect, useRef } from "react";
import type { MultiClusterRolloutResponse } from "../../api/types";

interface Props {
  rollout: MultiClusterRolloutResponse;
  /** Canvas height in pixels (one row per cluster). */
  height?: number;
}

/** Cluster x time action heatmap.
 *
 * Drawn on an HTMLCanvasElement because Recharts can't render 10K+
 * tiny rectangles efficiently. Each row is one cluster; columns are
 * timesteps. DPDK = forest green, USR = red.
 */
export function ActionHeatmap({ rollout, height = 240 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const actions = rollout.actions;
    const K = actions.length;
    const T = actions[0]?.length ?? 0;
    if (K === 0 || T === 0) return;

    // Render at logical pixel resolution; let CSS scale to container.
    const cssWidth = canvas.clientWidth || 800;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(cssWidth * dpr);
    canvas.height = Math.floor(height * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;

    // Fill background.
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const rowHeight = canvas.height / K;
    const colWidth = canvas.width / T;
    // Cap cell width at >= 1 device pixel.
    const drawColWidth = Math.max(1, Math.ceil(colWidth));
    const drawRowHeight = Math.max(1, Math.ceil(rowHeight) - 1);

    for (let k = 0; k < K; k++) {
      const row = actions[k];
      const yTop = Math.round(k * rowHeight);
      for (let t = 0; t < T; t++) {
        const x = Math.floor(t * colWidth);
        ctx.fillStyle = row[t] === 1 ? "#ef4444" : "#10b981";
        ctx.fillRect(x, yTop, drawColWidth, drawRowHeight);
      }
    }
  }, [rollout, height]);

  const K = rollout.actions.length;
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-slate-700">
          {rollout.summary.label} — action over time
        </h3>
        <span className="text-xs text-slate-500">
          <span className="mr-2 inline-flex items-center gap-1">
            <span className="h-2 w-2 rounded-sm bg-emerald-500" /> DPDK
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-2 w-2 rounded-sm bg-rose-500" /> USR
          </span>
        </span>
      </div>
      <div className="flex gap-2">
        <div
          className="flex flex-col justify-between text-[10px] text-slate-500"
          style={{ height: `${height}px` }}
        >
          {Array.from({ length: K }, (_, k) => (
            <div key={k} className="leading-none">
              c{k}
            </div>
          ))}
        </div>
        <canvas
          ref={canvasRef}
          className="w-full"
          style={{ height: `${height}px` }}
        />
      </div>
      <div className="mt-1 text-right text-[10px] text-slate-400">
        timestep →
      </div>
    </div>
  );
}
