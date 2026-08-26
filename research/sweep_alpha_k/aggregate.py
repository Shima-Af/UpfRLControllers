#!/usr/bin/env python3
"""Aggregate sweep cells into the rank-invariance table.

Learned controllers get mean +/- std across seeds; the classical baselines
are deterministic and identical across seeds, so one value each (asserted).
"""
from __future__ import annotations
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORDER = ["mappo", "hysteresis", "threshold", "always_dpdk", "always_usr", "hysteresis_auto"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--indir", type=Path, default=REPO / "reports" / "sweep-alpha-k")
    p.add_argument("--line", default="A")
    p.add_argument("--latex", action="store_true")
    a = p.parse_args()

    cells = [json.loads(f.read_text()) for f in sorted(a.indir.glob("*.json"))]
    cells = [c for c in cells if c.get("line") == a.line]
    if not cells:
        print(f"no cells for line {a.line} in {a.indir}"); return 1

    by_key: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in cells:
        key = (c["alpha"], c["K"])
        for name, r in c["results"].items():
            by_key[key][name].append(r)

    print(f"\nLine {a.line}: {len(cells)} cells, "
          f"{len(by_key)} settings, seeds per setting: "
          f"{ {f'a={k[0]:g},K={k[1]}': len(v['mappo']) for k, v in by_key.items()} }\n")

    hdr = f"{'alpha':>6} {'K':>4} | {'controller':>16} {'reward':>20} {'energy Wh':>10} {'unsafe %':>9} {'USR %':>7}"
    print(hdr); print("-" * len(hdr))
    rankings = {}
    for key in sorted(by_key, key=lambda k: (k[1], -k[0])):
        alpha, K = key
        rank_scores = []
        for name in ORDER:
            runs = by_key[key].get(name)
            if not runs: continue
            rw = [r["total_reward"] for r in runs]
            en = [r["total_energy_wh"] for r in runs]
            un = [r["agg_unsafe_rate"] * 100 for r in runs]
            us = [r["agg_usr_rate"] * 100 for r in runs]
            if len(rw) > 1 and st.pstdev(rw) > 1e-6:
                rstr = f"{st.mean(rw):>12.0f} ± {st.stdev(rw):>5.0f}"
            else:
                rstr = f"{st.mean(rw):>12.0f}        "
            print(f"{alpha:>6g} {K:>4} | {name:>16} {rstr} {st.mean(en):>10.1f} "
                  f"{st.mean(un):>8.2f}% {st.mean(us):>6.1f}%")
            if name != "hysteresis_auto":
                rank_scores.append((name, st.mean(rw)))
        rankings[key] = [n for n, _ in sorted(rank_scores, key=lambda x: -x[1])]
        print()

    print("Ranking by test reward (best first):")
    for key, order in sorted(rankings.items(), key=lambda kv: (kv[0][1], -kv[0][0])):
        print(f"  alpha={key[0]:<6g} K={key[1]:<4} {' > '.join(order)}")
    uniq = {tuple(v) for v in rankings.values()}
    print(f"\n=> {len(uniq)} distinct ordering(s) across {len(rankings)} settings: "
          f"{'RANK-INVARIANT' if len(uniq) == 1 else 'ORDERING CHANGES'}")

    if a.latex:
        print("\n% --- LaTeX ---")
        for key in sorted(by_key, key=lambda k: -k[0]):
            m = by_key[key]["mappo"]
            rw = [r["total_reward"] for r in m]
            print(f"${key[0]:g}$ & ${st.mean(rw):.0f} \\pm {st.stdev(rw) if len(rw)>1 else 0:.0f}$ & "
                  f"${st.mean([r['total_energy_wh'] for r in m]):.0f}$ & "
                  f"${st.mean([r['agg_unsafe_rate']*100 for r in m]):.2f}$ \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
