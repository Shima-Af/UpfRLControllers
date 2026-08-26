#!/usr/bin/env python3
"""Resumable N-way parallel driver for the alpha/K sensitivity sweep.

Restart-safe: a cell whose result JSON exists is skipped, so a crash costs
one cell, not the run. Cells are ordered cheapest-first so a short window
loses the expensive tail rather than the whole line.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CELL = REPO / "research" / "sweep_alpha_k" / "run_cell.py"
PY_BIN = REPO / ".venv" / "bin" / "python"


def line_a(seeds): return [dict(line="A", alpha=a, K=10, seed=s)
                           for a in (1.0, 0.75, 0.5, 0.25) for s in seeds]


def line_b(seeds, target_ratio=0.05):
    # alpha scaled with K so alpha/K stays fixed -> isolates fleet size.
    return [dict(line="B", alpha=round(target_ratio * k, 4), K=k, seed=s)
            for k in (4, 6, 10, 20, 50, 100) for s in seeds]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lines", default="A")
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--threads-per-job", type=int, default=2)
    p.add_argument("--outroot", type=Path,
                   default=REPO / "reports" / "sweep-alpha-k")
    a = p.parse_args()

    seeds = [int(x) for x in a.seeds.split(",")]
    cells: list[dict] = []
    if "A" in a.lines: cells += line_a(seeds)
    if "B" in a.lines: cells += line_b(seeds)
    cells.sort(key=lambda c: c["K"])            # cheapest first

    a.outroot.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cells
            if not (a.outroot / f"mappo_line{c['line']}_a{c['alpha']:g}_K{c['K']}_s{c['seed']}.json").exists()]
    print(f"grid: {len(cells)} cells, {len(cells)-len(todo)} already done, "
          f"{len(todo)} to run, {a.jobs}-way parallel", flush=True)

    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads_per_job),
               MKL_NUM_THREADS=str(a.threads_per_job))
    t0 = time.time(); done = 0; failed = []

    def run(c):
        cmd = [str(PY_BIN), str(CELL), "--alpha", str(c["alpha"]), "--K", str(c["K"]),
               "--seed", str(c["seed"]), "--steps", str(a.steps),
               "--line", c["line"], "--outroot", str(a.outroot)]
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        return c, r

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run, c) for c in todo]
        for f in as_completed(futs):
            c, r = f.result(); done += 1
            tag = f"line{c['line']} a={c['alpha']:g} K={c['K']} s={c['seed']}"
            if r.returncode != 0:
                failed.append(tag)
                print(f"[{done}/{len(todo)}] FAIL {tag}\n{r.stderr[-600:]}", flush=True)
            else:
                last = [l for l in r.stdout.strip().splitlines() if l.startswith("[done]")]
                el = time.time() - t0
                eta = el / done * (len(todo) - done)
                print(f"[{done}/{len(todo)}] {last[-1] if last else tag}"
                      f"  | elapsed {el/60:.0f}m eta {eta/60:.0f}m", flush=True)

    print(f"\nfinished in {(time.time()-t0)/60:.1f} min; {len(failed)} failed", flush=True)
    if failed: print("failed cells:", failed, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
