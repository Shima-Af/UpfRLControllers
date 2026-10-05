"""Step 3 (secondary) — recalibrated hysteresis, never mixed with the fixed controller.

In every world (pinned + 20 folds) the manuscript's derivation rule is re-applied
to that world's bundle: t_up = min(energy break-even, USR QoS limit) - 10 Mbps
(src.baselines.threshold_derivation, values in bundle_thresholds.csv), and the
band b is swept over {5, 10, 20, 50, 100} Mbps with cooldown 1, as in the
manuscript's band sweep (reports/phase-7/hyst_sweep). The best test-slice reward
per world is the "recalibrated hysteresis" point; b = 50 is also kept. Selection
on the test slice mirrors the manuscript protocol and is labelled as such.

Output: results/recal/<world>__A.csv
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import rb_common as C
from evaluate_worlds import run_chunk

BANDS_MBPS = [5, 10, 20, 50, 100]


def main():
    thr = pd.read_csv(C.HERE / "bundle_thresholds.csv").set_index("world")
    jobs = []
    for w in C.worlds():
        t_up = max(0.0, float(thr.loc[w["world"], "decision_mbps"])) / 1000.0
        ents = [dict(ckpt_id=f"Hysteresis (recalibrated, t_up={t_up * 1000:.0f}, b={b} Mbps)|-",
                     family="Hysteresis (recalibrated)", kind="hysteresis", seed=0, t_up_gbps=t_up,
                     t_down_gbps=max(0.0, t_up - b / 1000.0), cooldown_steps=1, band_mbps=b,
                     path="-", source=f"recalibrated in world {w['world']}", cohorts=["recal"]) for b in BANDS_MBPS]
        jobs.append(("recal", "v040", w["world"], "A", ents, "main", False))
    t0 = time.time()
    (C.HERE / "results" / "recal").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 21) as pool:
        for j, rows in zip(jobs, pool.map(run_chunk, *zip(*jobs))):
            df = pd.DataFrame(rows)
            df["band_mbps"] = df.ckpt_id.str.extract(r"b=(\d+) Mbps").astype(int)
            df["t_up_mbps"] = df.ckpt_id.str.extract(r"t_up=(\d+),").astype(float)
            df.to_csv(C.HERE / "results" / "recal" / f"{j[2]}__A.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
