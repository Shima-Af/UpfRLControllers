"""Side-by-side comparison: flat-cost reward (v1) vs physics-grounded reward (v2).

Reads:
  reports/phase-7/multiseed_summary_v1_flat_reward.json  (backup of old run)
  reports/phase-7/multiseed_summary.json                 (fresh run on revised reward)
  reports/phase-7/cooldown_sweep_p<P>_c<C>_seed<S>.json  (sweep results, optional)

Writes:
  reports/phase-7/reward_revision_summary.md             (Markdown delta tables for the paper)
  reports/phase-7/reward_revision_summary.json           (machine-readable)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _row(name: str, agg: dict) -> dict:
    return {
        "controller": name,
        "n_seeds": agg.get("n_seeds", 0),
        "reward_mean": agg.get("reward_mean"),
        "reward_std": agg.get("reward_std"),
        "energy_mean": agg.get("energy_mean"),
        "unsafe_mean": agg.get("unsafe_mean"),
        "usr_mean": agg.get("usr_mean"),
        "flips_mean": agg.get("flips_mean"),
    }


def _fmt(v, fmt=".2f"):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return format(v, fmt)
    return str(v)


def main() -> int:
    v1 = _load(REPO_ROOT / "reports/phase-7/multiseed_summary_v1_flat_reward.json")
    v2 = _load(REPO_ROOT / "reports/phase-7/multiseed_summary.json")
    if v1 is None or v2 is None:
        print("missing v1 or v2 summary", file=sys.stderr)
        return 1

    keys = [
        "MAPPO",
        "IPPO-ensemble",
        "centralised-PPO",
        "always-DPDK",
        "threshold(derived=81.0Mbps)",
        "hysteresis(t_up=81,t_down=0,cd=1)",
        "hysteresis(t_up=81,t_down=61,cd=1)",
    ]

    md_lines: list[str] = []
    md_lines.append("# Reward revision — v1 (flat) vs v2 (physics-grounded)")
    md_lines.append("")
    md_lines.append(
        "Reward magnitudes are NOT directly comparable across v1 and v2 "
        "(switching cost moved from a flat constant to a load-scaled "
        "physical quantity). What IS comparable is the *ranking* and "
        "the *gap structure*."
    )
    md_lines.append("")
    md_lines.append("## Test-split results, mean ± std over 4 seeds")
    md_lines.append("")
    md_lines.append(
        "| Controller | Reward v1 (flat) | Reward v2 (physics) | "
        "Unsafe v1 % | Unsafe v2 % | Flips v1 | Flips v2 |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")

    delta_rows = []
    for k in keys:
        a1 = v1["aggregated"].get(k, {})
        a2 = v2["aggregated"].get(k, {})
        if not a1 or not a2:
            continue
        r1 = a1.get("reward_mean")
        s1 = a1.get("reward_std", 0.0)
        r2 = a2.get("reward_mean")
        s2 = a2.get("reward_std", 0.0)
        u1 = a1.get("unsafe_mean", 0.0) * 100
        u2 = a2.get("unsafe_mean", 0.0) * 100
        f1 = a1.get("flips_mean", 0.0)
        f2 = a2.get("flips_mean", 0.0)
        md_lines.append(
            f"| {k} | {_fmt(r1, '.0f')} ± {_fmt(s1, '.0f')} | "
            f"{_fmt(r2, '.0f')} ± {_fmt(s2, '.0f')} | "
            f"{_fmt(u1, '.2f')} | {_fmt(u2, '.2f')} | "
            f"{_fmt(f1, '.0f')} | {_fmt(f2, '.0f')} |"
        )
        delta_rows.append(
            {
                "controller": k,
                "reward_v1": r1,
                "reward_v2": r2,
                "unsafe_v1": u1,
                "unsafe_v2": u2,
                "flips_v1": f1,
                "flips_v2": f2,
            }
        )

    md_lines.append("")
    md_lines.append("## Ranking preservation")
    md_lines.append("")
    rank_v1 = sorted(
        [(k, v1["aggregated"][k]["reward_mean"]) for k in keys if k in v1["aggregated"]],
        key=lambda x: -x[1],
    )
    rank_v2 = sorted(
        [(k, v2["aggregated"][k]["reward_mean"]) for k in keys if k in v2["aggregated"]],
        key=lambda x: -x[1],
    )
    md_lines.append("| Rank | v1 (flat reward) | v2 (physics reward) |")
    md_lines.append("|---|---|---|")
    for i, ((k1, _), (k2, _)) in enumerate(zip(rank_v1, rank_v2), 1):
        md_lines.append(f"| {i} | {k1} | {k2} |")

    # Cooldown sweep section (if any)
    sweep_files = sorted(
        (REPO_ROOT / "reports/phase-7").glob("cooldown_sweep_p*_c*_seed*.json")
    )
    if sweep_files:
        md_lines.append("")
        md_lines.append("## Cooldown sensitivity (MAPPO, physics-grounded reward, seed 42)")
        md_lines.append("")
        md_lines.append(
            "| period | cost | test reward | energy Wh | n_switches | unsafe % |"
        )
        md_lines.append("|---:|---:|---:|---:|---:|---:|")
        # Add baseline row from v2 first
        v2_mappo_per_seed = v2["aggregated"]["MAPPO"]["per_seed"].get("42")
        if v2_mappo_per_seed is not None:
            md_lines.append(
                f"| 4 (baseline) | 0.5 (baseline) | "
                f"{v2_mappo_per_seed['total_reward_unweighted']:.1f} | "
                f"{v2_mappo_per_seed['total_energy_wh']:.1f} | "
                f"{v2_mappo_per_seed['agg_n_switches']} | "
                f"{v2_mappo_per_seed['agg_unsafe_rate']*100:.2f} |"
            )
        for sf in sweep_files:
            d = json.loads(sf.read_text())
            t = d["test"]
            md_lines.append(
                f"| {d['cooldown_period']} | {d['cooldown_cost']} | "
                f"{t['total_reward_unweighted']:.1f} | "
                f"{t['total_energy_wh']:.1f} | "
                f"{t['agg_n_switches']} | "
                f"{t['agg_unsafe_rate']*100:.2f} |"
            )
        md_lines.append("")
        md_lines.append(
            "Interpretation: cooldown is **not load-bearing** if all rows "
            "in this table lie within seed noise of the baseline (± ~50 reward "
            "units for MAPPO under v2)."
        )

    md_path = REPO_ROOT / "reports/phase-7/reward_revision_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    json_path = REPO_ROOT / "reports/phase-7/reward_revision_summary.json"
    json_path.write_text(json.dumps({"deltas": delta_rows}, indent=2))
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
