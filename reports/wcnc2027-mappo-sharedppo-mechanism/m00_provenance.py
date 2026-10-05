"""Provenance of the MAPPO vs Shared-PPO primary-cohort results (read-only).

Links every artifact behind the paper's eight-seed MAPPO / Shared-PPO numbers
to the code and data that produced it, and records what is NOT saved.
No policy is run, no environment is stepped, nothing is trained.

Outputs: provenance_code.csv, provenance_checkpoints.csv, checkpoint_selection.csv,
         provenance_rollouts.csv, provenance_summary.json
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments" / "wcnc2027_ablation"
ABL = ROOT / "reports" / "wcnc2027-shared-policy-ablation"
CAL = ROOT / "reports" / "wcnc2027-calibration-robustness"
SEEDS = [1, 7, 13, 23, 42, 64, 77, 99]
CTRL = {"MAPPO": "mappo", "Shared-PPO": "shared_ppo"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    # ---- code: hashes recorded at training launch vs now ----------------------
    rec = {}
    txt = (ABL / "repro" / "code_and_environment.txt").read_text()
    for line in txt.splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+(\S+)$", line.strip())
        if m:
            rec[m.group(2)] = m.group(1)
    rows = []
    for rel, h in rec.items():
        p = ROOT / rel
        now = sha256(p) if p.exists() else ""
        rows.append(dict(path=rel, sha256_recorded_at_launch=h, sha256_now=now, unchanged=(h == now),
                         mtime=mtime(p) if p.exists() else "", git_tracked=bool(git("ls-files", rel))))
    code = pd.DataFrame(rows)
    code.to_csv(HERE / "provenance_code.csv", index=False)

    # ---- checkpoints -----------------------------------------------------------
    inv = pd.read_csv(CAL / "checkpoint_inventory.csv").set_index("path")
    crows = []
    cfgs = {}
    for c, pre in CTRL.items():
        for s in SEEDS:
            d = EXP / f"{pre}_seed{s}"
            for f in ("mappo_best.pt", "mappo_final.pt"):
                p = d / f
                ck = torch.load(p, map_location="cpu", weights_only=False)
                n_actor = sum(v.numel() for v in ck["actor"].values())
                n_critic = sum(v.numel() for v in ck["critic"].values())
                crit_shapes = {k: list(v.shape) for k, v in ck["critic"].items()}
                rel = str(p.relative_to(ROOT))
                crows.append(dict(controller=c, seed=s, file=rel, sha256=sha256(p), mtime=mtime(p),
                                  sha256_in_calibration_inventory=inv.loc[rel, "sha256"] if rel in inv.index else "",
                                  critic_type=ck.get("critic_type", "centralised (field absent)"),
                                  actor_params=n_actor, critic_params=n_critic,
                                  critic_input_dim=crit_shapes["body.0.weight"][1],
                                  critic_output_dim=[v for k, v in crit_shapes.items() if k.endswith("weight")][-1][0],
                                  obs_dim=ck["obs_dim"], state_dim=ck["state_dim"], K=ck["K"]))
                if f == "mappo_best.pt":
                    cfgs[(c, s)] = ck["config"]
    ck = pd.DataFrame(crows)
    # the calibration inventory stores sha256(name + sha256(file)) (rb_common.sha256_many)
    def inv_hash(r):
        h = hashlib.sha256()
        h.update(Path(r.file).name.encode())
        h.update(r.sha256.encode())
        return h.hexdigest()
    ck["inventory_hash_recomputed"] = ck.apply(inv_hash, axis=1)
    ck["matches_calibration_inventory"] = ck.apply(
        lambda r: (r.inventory_hash_recomputed == r.sha256_in_calibration_inventory) if r.sha256_in_calibration_inventory else None, axis=1)
    ck.to_csv(HERE / "provenance_checkpoints.csv", index=False)
    # config identity (every field except the seed must match between controllers and seeds)
    cfg_diffs = []
    ref = {k: v for k, v in cfgs[("MAPPO", 1)].items() if k != "seed"}
    for (c, s), cfg in cfgs.items():
        for k, v in cfg.items():
            if k == "seed":
                if v != s:
                    cfg_diffs.append(dict(controller=c, seed=s, field=k, value=v, reference=s))
            elif ref.get(k) != v:
                cfg_diffs.append(dict(controller=c, seed=s, field=k, value=v, reference=ref.get(k)))

    # ---- checkpoint selection (eval_log.json = the only saved learning record) --
    srows = []
    for c, pre in CTRL.items():
        for s in SEEDS:
            d = EXP / f"{pre}_seed{s}"
            e = json.loads((d / "eval_log.json").read_text())
            r = np.array([x["eval_return"] for x in e])
            st = np.array([x["step"] for x in e])
            b = int(np.argmax(r))          # trainer saves on strict improvement -> first maximum
            log = (EXP / "logs" / f"{pre}_seed{s}.log").read_text()
            tm = (EXP / "logs" / f"{pre}_seed{s}.time").read_text()
            wall = re.search(r"Elapsed \(wall clock\) time.*: (\S+)", tm).group(1)
            rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", tm).group(1))
            ent = [float(x) for x in re.findall(r"H=([\d.]+)", log)]
            kl = [float(x) for x in re.findall(r"KL=([\d.]+)", log)]
            srows.append(dict(controller=c, seed=s, n_evals=len(r), eval_every_steps=int(st[1] - st[0]),
                              best_step=int(st[b]), best_fraction_of_budget=st[b] / 200000, best_val_return=r[b],
                              final_val_return=r[-1], median_val_return=float(np.median(r)),
                              n_evals_below_minus_1e7=int((r < -1e7).sum()),
                              logged_entropy_points=len(ent), entropy_first=ent[0], entropy_last=ent[-1],
                              kl_last=kl[-1], wall_clock=wall, max_rss_kb=rss,
                              val_returns=json.dumps([round(float(x), 2) for x in r]),
                              val_steps=json.dumps([int(x) for x in st])))
    sel = pd.DataFrame(srows)
    sel.to_csv(HERE / "checkpoint_selection.csv", index=False)

    # ---- rollouts --------------------------------------------------------------
    man = {(m["set"], m["seed"]): m for m in json.loads((ABL / "rollouts" / "manifest.json").read_text())}
    rrows = []
    for c, pre in CTRL.items():
        for s in SEEDS:
            tc = ABL / "rollouts" / f"{c}__seed{s}.npz"
            cal = CAL / "rollouts" / "main" / "pinned" / "A" / f"{c}__wcnc2027_ablation-{pre}_seed{s}-mappo_best.pt.npz"
            a, b = np.load(tc, allow_pickle=True), np.load(cal)
            meta = json.loads(str(b["meta"]))
            rrows.append(dict(controller=c, seed=s, taskc_rollout=str(tc.relative_to(ROOT)), taskc_mtime=mtime(tc),
                              taskc_checkpoint=str(a["checkpoint"]), taskc_total_reward=man[(c, s)]["total_reward"],
                              calibration_rollout=str(cal.relative_to(ROOT)), calibration_mtime=mtime(cal),
                              calibration_checkpoint=meta["path"], calibration_twin=meta["twin"],
                              identical_action_arrays=bool(np.array_equal(a["action"], b["action"])),
                              reward_sum_float64=float(a["reward"].sum()),
                              reward_sum_float32_file=float(b["reward"].astype(np.float64).sum())))
    ro = pd.DataFrame(rrows)
    ro.to_csv(HERE / "provenance_rollouts.csv", index=False)

    hashes = json.loads((CAL / "hashes.json").read_text())
    summary = dict(
        generated=dt.datetime.now().isoformat(timespec="seconds"),
        repo_head_now=git("rev-parse", "HEAD"), repo_branch_now=git("rev-parse", "--abbrev-ref", "HEAD"),
        repo_head_at_training="71a1b2fc8ab94b33979fc1f399f4c4dacd2e1f57",
        code_files_unchanged_since_launch=bool(code.unchanged.all()),
        code_files_untracked=code.loc[~code.git_tracked, "path"].tolist(),
        training_started=(EXP / "train_a.started").read_text().strip(),
        training_finished=(EXP / "train_a.finished").read_text().strip(),
        twin=hashes["twin_installed"], scenario_sha256=hashes["configs"]["configs/scenario_rl.yaml"],
        traffic_test_sha256=hashes["traffic"],
        config_differences_other_than_seed=cfg_diffs,
        param_counts={c: dict(actor=int(ck[(ck.controller == c)].actor_params.iloc[0]),
                              critic=int(ck[(ck.controller == c)].critic_params.iloc[0])) for c in CTRL},
        checkpoints_match_calibration_inventory=bool(ck.dropna(subset=["matches_calibration_inventory"]).matches_calibration_inventory.all()),
        n_checkpoints_checked_against_inventory=int(ck.matches_calibration_inventory.notna().sum()),
        rollouts_identical_actions=bool(ro.identical_action_arrays.all()),
        selection_shared_earlier_than_mappo=int(sum(
            sel[(sel.controller == "Shared-PPO") & (sel.seed == s)].best_step.iloc[0]
            < sel[(sel.controller == "MAPPO") & (sel.seed == s)].best_step.iloc[0] for s in SEEDS)),
        selection_mappo_higher_val=int(sum(
            sel[(sel.controller == "MAPPO") & (sel.seed == s)].best_val_return.iloc[0]
            > sel[(sel.controller == "Shared-PPO") & (sel.seed == s)].best_val_return.iloc[0] for s in SEEDS)),
    )
    (HERE / "provenance_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "traffic_test_sha256"}, indent=1, default=str))
    print(code[["path", "unchanged", "git_tracked", "mtime"]].to_string(index=False))
    print(sel[["controller", "seed", "best_step", "best_val_return", "final_val_return", "n_evals_below_minus_1e7"]].to_string(index=False))


if __name__ == "__main__":
    main()
