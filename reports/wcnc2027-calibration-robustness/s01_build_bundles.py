"""Step 1 — reconstruct the LOSO / LOLO fold surrogate bundles and sanity-check them.

The held-out validation (reports/twin-validation-heldout-wcnc2027) saved
predictions and the per-fold model selections, but not the fitted fold models.
They are rebuilt here with that study's own code (s02b_threshold_stability.
refit_fold, imported, not copied) and its recorded selections
(heldout_model_selection.json), on the same DVC-verified profiling data.
Fitting is deterministic (fixed random_state, single-threaded forests), so the
rebuilt models must reproduce the saved held-out predictions exactly; this is
checked below and the build fails otherwise.

Bundle layout (same as the pinned data/external/profiling_twin/models):
  bundles/<PROTOCOL>_<config>_fold<k>/models/manifest.json
  bundles/<PROTOCOL>_<config>_fold<k>/models/layer1/{dpdk,usr_full}__<target>__lite.pkl
  bundles/<PROTOCOL>_<config>_fold<k>/models/layer2/{dpdk,usr_full}__power_watts__lite.pkl
Each bundle replaces the complete steady-state set (4 Layer-1 targets + Layer-2
power, both UPFs) and nothing else: the manifest holds only the 10 keys the
twin loads, so a lookup of any other model fails loudly. DPDK and USR models of
one bundle come from the same outer fold index (sweep k for LOSO; load-level
group k for LOLO), as in the held-out study's threshold analysis.

Checks (bundle_checks.csv, bundle_thresholds.csv):
  C1 feature order / units: manifest features == pinned [DL kbit/s, UL kbit/s];
     twin input is load_gbps*1e6 for both (UL := DL)
  C2 variant mapping DPDK->dpdk, USR->usr_full (upf_profile._VARIANT_MAP)
  C3 rebuilt models reproduce heldout_predictions.csv.gz on the held-out rows
     (twin-style inputs), all five outputs
  C4 predictions through the real env code path (DigitalTwin(paths_cfg).
     evaluate_batch) equal direct model predictions
  C5 grid predictions equal threshold_grid_predictions.csv.gz
  C6 finite outputs on the RL test-slice loads and the 1-500 Mbps grid; power >= 0
     after the twin's clip (share of raw negative Layer-2 outputs recorded)
  C7 derived break-even / QoS limit / decision threshold (src.baselines.
     threshold_derivation on the bundle-loaded twin) equal threshold_stability.csv
Pinned bundle: C4/C6/C7 plus equality with pinned_predictions_sklearn1.8.0.npz.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import rb_common as C

sys.path.insert(0, str(C.TASKA))
import common as TA  # noqa: E402
from s02_heldout_cv import fold_ids  # noqa: E402
from s02b_threshold_stability import refit_fold, template  # noqa: E402

SHORT = {TA.T_THR: "thr", TA.T_CPU: "cpu", TA.T_LOSS: "loss", TA.T_DELAY: "delay"}
_DF = None


def _df():
    global _DF
    if _DF is None:
        _DF, _ = TA.load_runs(verify=True)
        _DF["row_id"] = np.arange(len(_DF))
    return _DF


def build_one(protocol: str, config: str, k: int) -> dict:
    df = _df()
    sels_all = json.loads((TA.OUT / "heldout_model_selection.json").read_text())
    pinned_manifest = json.loads((C.PINNED_MODELS / "manifest.json").read_text())
    name = f"{protocol}_{config}_fold{k}"
    mdir = C.BUNDLES / name / "models"
    (mdir / "layer1").mkdir(parents=True, exist_ok=True)
    (mdir / "layer2").mkdir(parents=True, exist_ok=True)
    manifest, prov, preds = {}, {"bundle": name, "protocol": protocol, "config": config, "outer_fold": k, "upf": {}}, {}
    for upf, (variant, _) in TA.UPFS.items():
        sub = df[df["upf"] == upf].reset_index(drop=True)
        outer, inner_fn = fold_ids(sub, protocol)
        sels = {(s["layer"], s["target"]): s for s in sels_all
                if s["protocol"] == protocol and s["config"] == config and s["upf"] == upf and s["outer_fold"] == k}
        assert len(sels) == 5, (name, upf, len(sels))
        models = refit_fold(sub, outer, inner_fn, k, sels)
        tr, te = sub[outer != k], sub[outer == k]
        for t, m in zip(TA.L1_TARGETS, models["l1"]):
            rel = f"layer1/{variant}__{t}__lite.pkl"
            joblib.dump(m, mdir / rel)
            key = f"{variant}__layer1__{t}__lite"
            manifest[key] = dict(path=rel, layer="layer1", variant=variant, target=t, model_variant="lite",
                                 features=pinned_manifest[key]["features"],
                                 best_params={"model_type": sels[("layer1", t)]["model_type"],
                                              **sels[("layer1", t)].get("params", {})},
                                 sha256=C.sha256(mdir / rel))
        rel = f"layer2/{variant}__{TA.T_POWER}__lite.pkl"
        joblib.dump(models["l2"], mdir / rel)
        key = f"{variant}__layer2__{TA.T_POWER}__lite"
        manifest[key] = dict(path=rel, layer="layer2", variant=variant, target=TA.T_POWER, model_variant="lite",
                             features=pinned_manifest[key]["features"],
                             best_params={"model_type": sels[("layer2", TA.T_POWER)]["model_type"],
                                          **sels[("layer2", TA.T_POWER)].get("params", {})},
                             sha256=C.sha256(mdir / rel))
        prov["upf"][upf] = dict(variant=variant, n_train_rows=int(len(tr)), n_train_runs=int(tr["run_dir"].nunique()),
                                heldout_runs=sorted(te["run_dir"].unique().tolist()),
                                heldout_sweeps=sorted(int(x) for x in te["sweep"].unique()),
                                heldout_levels_gbps=sorted(float(x) for x in te["level_gbps"].unique()),
                                train_load_range_gbps=[float(tr["load_gbps"].min()), float(tr["load_gbps"].max())])
        # direct predictions on held-out rows (twin-style inputs), for C3
        X = np.column_stack([te[TA.DL_TX].to_numpy()] * 2)
        l1 = [m.predict(X) for m in models["l1"]]
        pw_raw = models["l2"].predict(np.column_stack([X, *l1]))
        preds[upf] = dict(row_id=te["row_id"].to_numpy(), thr=l1[0], cpu=l1[1], loss=l1[2], delay=l1[3],
                          power=np.maximum(0.0, pw_raw), load_gbps=te["load_gbps"].to_numpy())
    (mdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (C.BUNDLES / name / "bundle_provenance.json").write_text(json.dumps(prov, indent=1))
    np.savez_compressed(C.BUNDLES / name / "heldout_rows_direct_predictions.npz",
                        **{f"{u}__{f}": v for u, d in preds.items() for f, v in d.items()})
    return dict(bundle=name, seconds=0)


def check_bundle(world: dict, heldout: pd.DataFrame, grid: pd.DataFrame, thr: pd.DataFrame,
                 test_loads: np.ndarray) -> tuple[list[dict], dict]:
    from upf_digital_twin import DigitalTwin
    from upf_digital_twin.twin import upf_profile as UP
    from src.baselines.threshold_derivation import derive_thresholds
    from src.utils.config import load_yaml

    name = world["world"]
    twin = DigitalTwin(scenario_cfg=load_yaml("configs/scenario_rl.yaml"), paths_cfg=C.paths_cfg_for(world),
                       project_root=C.ROOT)
    prof = twin._profile  # noqa: SLF001
    man = json.loads((Path(world["models"]) / "manifest.json").read_text())
    pinned_man = json.loads((C.PINNED_MODELS / "manifest.json").read_text())
    rows = []

    def add(check, ok, detail):
        rows.append(dict(world=name, check=check, passed=bool(ok), detail=detail))

    keys = [f"{v}__{l}__{t}__lite" for v in ("dpdk", "usr_full")
            for l, t in [*[("layer1", x) for x in TA.L1_TARGETS], ("layer2", TA.T_POWER)]]
    l2_order = [TA.DL_TX, TA.UL_TX, *[f"l1_pred__{t}" for t in TA.L1_TARGETS]]   # == upf_profile column_stack
    feat_ok = all(man[k]["features"] == pinned_man[k]["features"]
                  == ([TA.DL_TX, TA.UL_TX] if "__layer1__" in k else l2_order) for k in keys)
    x = prof._lite_X(np.array([0.1]))  # noqa: SLF001
    add("C1 feature order/units", feat_ok and np.allclose(x, [[1e5, 1e5]]),
        f"L1 {pinned_man[keys[0]]['features']}, L2 {l2_order}; 0.1 Gbps -> {x.tolist()} kbit/s (UL:=DL)")
    add("C2 variant mapping", UP._VARIANT_MAP == {"DPDK": "dpdk", "USR": "usr_full"}  # noqa: SLF001
        and (world["tier"] == "reference" or set(man) == set(keys)), f"{UP._VARIANT_MAP}; manifest keys {len(man)}")  # noqa: SLF001

    out = {"thresholds": None}
    if world["tier"] != "reference":
        d = np.load(C.BUNDLES / f"{world['protocol']}_{world['config']}_fold{world['fold']}" /
                    "heldout_rows_direct_predictions.npz")
        worst = 0.0
        worst_env = 0.0
        for upf in ("DPDK", "USR"):
            ref = heldout[(heldout.protocol == world["protocol"]) & (heldout.config == world["config"])
                          & (heldout.outer_fold == world["fold"]) & (heldout.upf == upf)].set_index("row_id")
            rid = d[f"{upf}__row_id"]
            ref = ref.loc[rid]
            env_out = prof.evaluate_batch(upf, d[f"{upf}__load_gbps"])
            for f in ("thr", "cpu", "loss", "delay", "power"):
                a = d[f"{upf}__{f}"]
                b = ref[f"pred_{f}"].to_numpy()
                worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))))
            for f, g in (("thr", "throughput_gbps"), ("loss", "predicted_loss"), ("delay", "delay_us"), ("power", "power_watts")):
                worst_env = max(worst_env, float(np.max(np.abs(d[f"{upf}__{f}"] - env_out[g]))))
        add("C3 reproduces saved held-out predictions", worst <= 1e-9, f"max |diff|/max(1,|ref|) = {worst:.2e}")
        add("C4 env code path == direct predictions", worst_env <= 1e-9, f"max |diff| = {worst_env:.2e}")
        gsel = grid[(grid.source == "refit") & (grid.protocol == world["protocol"]) & (grid.config == world["config"])
                    & (grid.outer_fold == world["fold"])]
    else:
        gl0 = np.linspace(0.001, 0.5, 500)
        worst_env = 0.0
        for upf, variant in (("DPDK", "dpdk"), ("USR", "usr_full")):
            X = np.column_stack([gl0 * 1e6] * 2)
            l1 = [joblib.load(C.PINNED_MODELS / "layer1" / f"{variant}__{t}__lite.pkl").predict(X) for t in TA.L1_TARGETS]
            pw = np.maximum(0.0, joblib.load(C.PINNED_MODELS / "layer2" / f"{variant}__{TA.T_POWER}__lite.pkl")
                            .predict(np.column_stack([X, *l1])))
            e = prof.evaluate_batch(upf, gl0)
            worst_env = max(worst_env, float(np.max(np.abs(pw - e["power_watts"]))),
                            float(np.max(np.abs(l1[3] - e["delay_us"]))), float(np.max(np.abs(l1[2] - e["predicted_loss"]))))
        add("C4 env code path == direct pinned-model predictions", worst_env <= 1e-9, f"max |diff| = {worst_env:.2e}")
        gsel = grid[grid.source == "pinned"]
    gl = np.linspace(0.001, 0.5, 500)
    worst_g = 0.0
    raw_neg = {}
    for upf in ("DPDK", "USR"):
        e = prof.evaluate_batch(upf, gl)
        ref = gsel[gsel.upf == upf].sort_values("load_gbps")
        for f, gname in (("thr", "throughput_gbps"), ("cpu", "cpu_pct"), ("loss", "predicted_loss"),
                         ("delay", "delay_us"), ("power", "power_watts")):
            worst_g = max(worst_g, float(np.max(np.abs(ref[f].to_numpy() - e[gname]) / np.maximum(1.0, np.abs(e[gname])))))
        variant = "dpdk" if upf == "DPDK" else "usr_full"
        m1 = [prof._get_model(variant, "layer1", t) for t in TA.L1_TARGETS]  # noqa: SLF001
        Xt = prof._lite_X(test_loads)  # noqa: SLF001
        l1 = [m.predict(Xt) for m in m1]
        raw = prof._get_model(variant, "layer2", TA.T_POWER).predict(np.column_stack([Xt, *l1]))  # noqa: SLF001
        et = prof.evaluate_batch(upf, test_loads)
        finite = all(np.isfinite(et[g]).all() for g in ("power_watts", "throughput_gbps", "cpu_pct", "delay_us", "predicted_loss")) \
            and all(np.isfinite(e[g]).all() for g in ("power_watts", "delay_us", "predicted_loss"))
        raw_neg[upf] = dict(finite=bool(finite), min_power_after_clip=float(et["power_watts"].min()),
                            share_raw_negative_power=float((raw < 0).mean()),
                            usr_or_dpdk_unsafe_share_on_test_loads=float((~et["is_safe"]).mean()))
    add("C5 grid predictions == threshold_grid_predictions.csv.gz", worst_g <= 1e-9, f"max rel diff = {worst_g:.2e}")
    add("C6 finite, power >= 0 after clip", all(v["finite"] and v["min_power_after_clip"] >= 0 for v in raw_neg.values()),
        json.dumps(raw_neg))
    spec = derive_thresholds(twin, safety_margin_mbps=10.0, forecast_mae_gbps=None)
    ref = thr[(thr.protocol == (world["protocol"] if world["tier"] != "reference" else "-"))
              & (thr.config == world["config"]) & (thr.outer_fold == world["fold"])].iloc[0]
    got = dict(energy_breakeven_mbps=spec.energy_breakeven_gbps * 1000, qos_limit_mbps=spec.qos_limit_gbps * 1000,
               decision_mbps=spec.decision_gbps * 1000, delay_limit_mbps=spec.delay_limit_gbps * 1000)
    ok = all(abs(got[c] - float(ref[c])) < 1e-6 for c in ("energy_breakeven_mbps", "qos_limit_mbps", "decision_mbps"))
    add("C7 derived thresholds == threshold_stability.csv", ok,
        f"break-even {got['energy_breakeven_mbps']:.0f}, QoS {got['qos_limit_mbps']:.0f}, decision {got['decision_mbps']:.0f} Mbps "
        f"(ref {ref.energy_breakeven_mbps:.0f}/{ref.qos_limit_mbps:.0f}/{ref.decision_mbps:.0f})")
    out["thresholds"] = dict(world=name, tier=world["tier"], protocol=world["protocol"], config=world["config"],
                             fold=world["fold"], **got, usr_safe_flips_on_grid=int(ref.usr_safe_flips_on_grid),
                             **{f"{u}_{k}": v for u, d in raw_neg.items() for k, v in d.items()})
    return rows, out


def main():
    t0 = time.time()
    _df()   # verified load in the parent; forked workers inherit it
    jobs = [(p, c, k) for c in ("deployed", "nested") for p in ("LOSO", "LOLO") for k in C.FOLDS]
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10) as pool:
        list(pool.map(build_one, *zip(*jobs)))
    print(f"built {len(jobs)} bundles in {time.time() - t0:.0f}s", flush=True)

    heldout = pd.read_csv(TA.OUT / "heldout_predictions.csv.gz")
    grid = pd.read_csv(TA.OUT / "threshold_grid_predictions.csv.gz")
    thr = pd.read_csv(TA.OUT / "threshold_stability.csv")
    tgt = np.load(C.ROOT / "data/external/traffic_forecaster/targets_test.npy")
    test_loads = np.unique(tgt[:, 0, :].ravel() * 1.0)
    rows, trows = [], []
    for w in C.worlds():
        r, o = check_bundle(w, heldout, grid, thr, test_loads)
        rows += r
        trows.append(o["thresholds"])
        print(f"{w['world']:12s} " + " ".join("PASS" if x["passed"] else "FAIL" for x in r), flush=True)
    pd.DataFrame(rows).to_csv(C.HERE / "bundle_checks.csv", index=False)
    pd.DataFrame(trows).to_csv(C.HERE / "bundle_thresholds.csv", index=False)
    bad = [x for x in rows if not x["passed"]]
    print(f"{len(rows) - len(bad)}/{len(rows)} checks passed; {time.time() - t0:.0f}s")
    if bad:
        print(pd.DataFrame(bad).to_string())
        sys.exit(1)


if __name__ == "__main__":
    main()
