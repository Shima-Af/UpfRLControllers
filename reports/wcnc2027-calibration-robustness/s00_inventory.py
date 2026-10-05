"""Step 0 — read-only inventory: checkpoints, seeds, normalisation files and
hashes of every configuration, data, model and code input.

Outputs: checkpoint_inventory.csv, hashes.json
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import pandas as pd

import rb_common as C


def git(*args, cwd=C.ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def main():
    reg = C.checkpoint_registry()
    rows = []
    for r in reg:
        files = C.checkpoint_files(r)
        ok = all(f.exists() for f in files)
        d = Path(r["path"]) if r["kind"] == "ippo" else Path(r["path"]).parent
        rows.append(dict(ckpt_id=r["ckpt_id"], family=r["family"], seed=r["seed"], cohorts="+".join(r["cohorts"]),
                         source=r["source"], path=str(Path(r["path"]).relative_to(C.ROOT)),
                         n_files=len(files), all_files_present=ok,
                         dir_mtime=dt.datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="minutes"),
                         sha256=C.sha256_many(files) if ok else ""))
    inv = pd.DataFrame(rows)

    # byte-identical copies excluded from the registry
    dup = []
    for s in (1, 23, 64, 77):
        a = C.EXP / f"ppo_multi_site_seed{s}_bridge" / "ppo_multi_site.zip"
        b = C.EXP / f"central_nr_seed{s}" / "ppo_multi_site.zip"
        if a.exists() and b.exists():
            dup.append(dict(copy=str(a.relative_to(C.ROOT)), original=str(b.relative_to(C.ROOT)),
                            identical=C.sha256(a) == C.sha256(b)))
    # observation / reward normalisation files (VecNormalize or equivalent)
    norm = sorted(str(p.relative_to(C.ROOT)) for p in C.EXP.rglob("*")
                  if p.is_file() and any(t in p.name.lower() for t in ("vecnorm", "normaliz", "obs_rms", "running_mean")))
    inv.to_csv(C.HERE / "checkpoint_inventory.csv", index=False)

    hashes = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "repo_head": git("rev-parse", "HEAD"), "repo_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "twin_installed": json.loads((C.ROOT / ".venv/lib/python3.12/site-packages/upf_digital_twin-0.4.0.dist-info/direct_url.json").read_text()),
        "twin_v030_vendored_commit": git("rev-parse", "v0.3.0^{commit}", cwd=Path.home() / "UPF_NDT"),
        "configs": {p: C.sha256(C.ROOT / p) for p in ["configs/scenario_rl.yaml", "configs/digital_twin_paths.yaml"]},
        "switching_costs": {
            "pinned (thesis-v1.2)": C.sha256(C.ROOT / "data/external/profiling_twin/switching_costs.yaml"),
            "pre-fix (thesis-v1.1, vendored)": C.sha256(C.SWITCH_V11)},
        "traffic": {p.name: C.sha256(p) for p in sorted((C.ROOT / "data/external/traffic_forecaster").glob("*_test.npy"))},
        "pinned_models": {p.relative_to(C.PINNED_MODELS).as_posix(): C.sha256(p)
                          for p in sorted(C.PINNED_MODELS.rglob("*")) if p.is_file() and "lite" in p.name or p.name == "manifest.json"},
        "code": {str(p.relative_to(C.ROOT)): C.sha256(p) for p in [
            C.ROOT / "src/envs/single_site_upf_env.py", C.ROOT / "src/envs/multi_agent_upf_env.py",
            C.ROOT / "src/baselines/hysteresis.py", C.ROOT / "src/baselines/threshold_derivation.py",
            C.ROOT / "research/phase7/evaluate_multiseed.py", C.ROOT / "src/trainers/mappo.py",
            C.ROOT / ".venv/lib/python3.12/site-packages/upf_digital_twin/twin/digital_twin.py",
            C.ROOT / ".venv/lib/python3.12/site-packages/upf_digital_twin/twin/upf_profile.py",
            C.TASKA / "common.py", C.TASKA / "s02_heldout_cv.py", C.TASKA / "s02b_threshold_stability.py",
            C.TASKA / "heldout_model_selection.json", C.TASKA / "heldout_predictions.csv.gz",
            C.TASKA / "threshold_stability.csv", C.TASKA / "threshold_grid_predictions.csv.gz"]},
        "study_scripts": {p.name: C.sha256(p) for p in sorted(C.HERE.glob("*.py")) + [C.HERE / "run_all.sh"] if p.exists()},
        "fold_bundles": {b.name: dict(sha256_all_files=C.sha256_many(sorted(x for x in (b / "models").rglob("*") if x.is_file())),
                                      manifest_sha256=C.sha256(b / "models" / "manifest.json"))
                         for b in sorted(C.BUNDLES.glob("*_fold*")) if (b / "models" / "manifest.json").exists()},
        "bridge_copies_excluded": dup,
        "normalisation_files_found": norm,
    }
    (C.HERE / "hashes.json").write_text(json.dumps(hashes, indent=1))
    print(inv.groupby(["family", "cohorts"]).size().to_string())
    print("all files present:", bool(inv.all_files_present.all()), "| normalisation files:", norm, "| bridge copies:", dup)


if __name__ == "__main__":
    main()
