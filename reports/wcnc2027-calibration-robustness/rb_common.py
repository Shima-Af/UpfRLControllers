"""Shared definitions for the calibration-robustness study.

Zero-shot evaluation only: no controller is trained, tuned or modified, and
nothing outside this directory is written. A "calibration world" is the
unchanged environment (traffic slice, observation, reward, switching model,
activation durations/energies, reset seed, deterministic actions) with the
steady-state surrogate bundle (Layer-1 throughput/cpu/loss/delay + Layer-2
power, for both DPDK and USR) replaced as a whole by a bundle in the same
on-disk layout. The environment code path is
    SingleSiteUPFEnv -> upf_digital_twin.DigitalTwin(paths_cfg)
        -> UPFProfile(models_dir = paths_cfg.profiling_twin.models,
                      manifest   = paths_cfg.profiling_twin.manifest)
so a world is selected purely through ``paths_cfg``; switching_costs.yaml
always stays the pinned file.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments"
NEW = EXP / "wcnc2027_ablation"
PINNED_MODELS = ROOT / "data" / "external" / "profiling_twin" / "models"
BUNDLES = HERE / "bundles"
ROLLOUTS = HERE / "rollouts"
VENDOR_V030 = HERE / "vendor" / "upf_digital_twin_v0.3.0"
SWITCH_V11 = HERE / "vendor" / "switching_costs_thesis-v1.1.yaml"
TASKA = ROOT / "reports" / "twin-validation-heldout-wcnc2027"
SEEDS = [1, 7, 13, 23, 42, 64, 77, 99]
K = 10
RESET_SEED = 42
STEP_MIN = 15
FOLDS = range(5)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_many(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(sha256(p).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Calibration worlds
# ---------------------------------------------------------------------------

def worlds(tiers=("reference", "primary", "secondary")) -> list[dict]:
    """Ordered list of worlds. tier: reference (pinned), primary (deployed-
    configuration refits), secondary (nested-selection refits)."""
    out = [dict(world="pinned", tier="reference", protocol="-", config="deployed", fold=-1,
                models=str(PINNED_MODELS))]
    for config, tier in (("deployed", "primary"), ("nested", "secondary")):
        for protocol in ("LOSO", "LOLO"):
            for k in FOLDS:
                out.append(dict(world=f"{protocol}-{config[:3]}-f{k}", tier=tier, protocol=protocol,
                                config=config, fold=k,
                                models=str(BUNDLES / f"{protocol}_{config}_fold{k}" / "models")))
    return [w for w in out if w["tier"] in tiers]


def world_by_name(name: str) -> dict:
    return next(w for w in worlds() if w["world"] == name)


def paths_cfg_for(world: dict, switching_costs: Path | None = None) -> dict:
    from src.utils.config import load_yaml
    cfg = copy.deepcopy(load_yaml("configs/digital_twin_paths.yaml"))
    pt = cfg["profiling_twin"]
    pt["models"] = str(Path(world["models"]))
    pt["manifest"] = str(Path(world["models"]) / "manifest.json")
    if switching_costs is not None:
        pt["switching_costs"] = str(switching_costs)
    return cfg


# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

LEGACY_MAPPO_V04 = {
    1: "mappo_seed1_20260517T110908", 7: "mappo_seed7_20260517T003433",
    13: "mappo_seed13_20260517T003433", 23: "mappo_seed23_20260517T110908",
    42: "mappo_seed42_20260526T143637", 64: "mappo_seed64_20260517T110908",
    77: "mappo_seed77_20260517T110908", 99: "mappo_seed99_20260517T003433",
}
LEGACY_IPPO_V04 = {1: "ppo_single_site_ensemble_seed1_20260517T110907",
                   7: "ppo_single_site_ensemble_seed7_20260517T091403",
                   13: "ppo_single_site_ensemble_seed13_20260517T091403",
                   23: "ppo_single_site_ensemble_seed23_20260517T110907",
                   42: "ppo_single_site_ensemble_seed42_20260517T091403",
                   64: "ppo_single_site_ensemble_seed64_20260517T110907",
                   77: "ppo_single_site_ensemble_seed77_20260517T110907",
                   99: "ppo_single_site_ensemble_seed99_20260517T091403"}
LEGACY_CENTRAL_V04 = {s: f"ppo_multi_site_seed{s}_20260517T004954" for s in (7, 13, 42, 99)}

COHORT_LABEL = {
    "R": "same-code 8-seed set (retrained 2026-09-14, twin v0.4.0, current reward)",
    "P": "v0.4 re-scored published-lineage set (reports/phase-7/multiseed_summary_v04twin.json)",
    "T": "Table-I-traceable checkpoints (match manuscript per-seed values under twin v0.3.0)",
    "L": "other surviving legacy checkpoints (not in R, P or T)",
}


def checkpoint_registry() -> list[dict]:
    """Every distinct best-of-validation checkpoint of the three RL families
    (plus the Shared-PPO ablation), with cohort membership. Byte-identical
    copies (``ppo_multi_site_seed*_bridge`` = ``central_nr_seed*``) are listed
    once. Cohort T is added from reproduction/table1_traceability.json once the
    reproduction check has been run."""
    rows: list[dict] = []

    def add(family, kind, seed, path, source, cohorts):
        p = Path(path)
        rows.append(dict(ckpt_id=f"{family}|{p.relative_to(EXP).as_posix()}", family=family, kind=kind,
                         seed=int(seed), path=str(p), source=source, cohorts=list(cohorts)))

    for s in SEEDS:
        add("MAPPO", "mappo", s, NEW / f"mappo_seed{s}" / "mappo_best.pt", "wcnc2027_ablation 2026-09-14", ["R"])
        add("IPPO", "ippo", s, NEW / f"ippo_seed{s}", "wcnc2027_ablation 2026-09-14", ["R"])
        add("Centralized PPO", "central", s, NEW / f"central_seed{s}" / "ppo_multi_site.zip",
            "wcnc2027_ablation 2026-09-14", ["R"])
        add("Shared-PPO", "mappo", s, NEW / f"shared_ppo_seed{s}" / "mappo_best.pt",
            "wcnc2027_ablation 2026-09-14 (supplementary ablation)", ["R"])
    for s, d in LEGACY_MAPPO_V04.items():
        add("MAPPO", "mappo", s, EXP / d / "mappo_best.pt", "May 2026, pre-781c90c reward", ["P"])
    for s, d in LEGACY_IPPO_V04.items():
        add("IPPO", "ippo", s, EXP / d, "May 2026, pre-781c90c reward", ["P"])
    for s, d in LEGACY_CENTRAL_V04.items():
        add("Centralized PPO", "central", s, EXP / d / "ppo_multi_site.zip", "May 2026, pre-781c90c reward", ["P"])
    # other surviving legacy checkpoints
    for s in (23, 64, 99):
        add("MAPPO", "mappo", s, EXP / f"mappo_nr_seed{s}" / "mappo_best.pt", "June 2026, current reward", ["L"])
    for d in ("mappo_seed7_20260516T092249", "mappo_seed13_20260516T092252", "mappo_seed42_20260516T081859",
              "mappo_seed42_20260517T003433", "mappo_seed42_20260526T143428", "mappo_seed99_20260516T092254"):
        add("MAPPO", "mappo", int(d.split("seed")[1].split("_")[0]), EXP / d / "mappo_best.pt",
            "May 2026, pre-781c90c reward", ["L"])
    for s in SEEDS:
        add("IPPO", "ippo", s, EXP / f"ippo_nr_seed{s}", "June 2026, current reward", ["L"])
    for s in (1, 7, 13, 42, 23, 64, 77):
        src = "June 2026, current reward" if s in (1, 7, 13, 42) else "Aug 2026, current reward"
        add("Centralized PPO", "central", s, EXP / f"central_nr_seed{s}" / "ppo_multi_site.zip", src, ["L"])
    for d in ("ppo_multi_site_seed42_20260515T234034", "ppo_multi_site_seed7_20260516T092256",
              "ppo_multi_site_seed13_20260516T092257", "ppo_multi_site_seed99_20260516T092259"):
        add("Centralized PPO", "central", int(d.split("seed")[1].split("_")[0]), EXP / d / "ppo_multi_site.zip",
            "May 2026, pre-781c90c reward", ["L"])
    for d in ("ppo_multi_site_seed7_20260517T003435", "ppo_multi_site_seed13_20260517T003435",
              "ppo_multi_site_seed42_20260517T003436", "ppo_multi_site_seed99_20260517T003436"):
        add("Centralized PPO", "central", int(d.split("seed")[1].split("_")[0]), EXP / d / "best_model.zip",
            "May 2026, pre-781c90c reward; run has no final save (best_model.zip only)", ["L"])

    trace = HERE / "reproduction" / "table1_traceability.json"
    if trace.exists():
        t = json.loads(trace.read_text())
        ids = set(t.get("traceable_ckpt_ids", []))
        for r in rows:
            if r["ckpt_id"] in ids:
                r["cohorts"] = sorted(set(r["cohorts"]) - {"L"} | {"T"})
        known = {r["ckpt_id"] for r in rows}
        for e in t.get("extra_entries", []):          # traced checkpoints outside the registry dirs
            if e["ckpt_id"] not in known:
                rows.append(dict(e, source=f"{e['source']}; traced to Table I", cohorts=["T"]))
    return rows


def checkpoint_files(entry: dict) -> list[Path]:
    p = Path(entry["path"])
    if entry["kind"] == "ippo":
        return [p / f"cluster_{k}" / "ppo_single_site.zip" for k in range(K)]
    return [p]


def cohort_members(registry: list[dict], cohort: str, families=None) -> list[dict]:
    return [r for r in registry if cohort in r["cohorts"] and (families is None or r["family"] in families)]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

HYST_FIXED = dict(t_up_gbps=0.081, band_gbps=0.050, cooldown_steps=1)   # manuscript Table I
BASELINES = [
    dict(ckpt_id="Hysteresis (fixed, t_up=81, b=50 Mbps)|-", family="Hysteresis (fixed)", kind="hysteresis",
         seed=0, t_up_gbps=0.081, t_down_gbps=0.031, cooldown_steps=1),
    dict(ckpt_id="Always-DPDK|-", family="Always-DPDK", kind="const", seed=0, action=0),
    dict(ckpt_id="Always-USR|-", family="Always-USR (supplementary)", kind="const", seed=0, action=1),
]


# ---------------------------------------------------------------------------
# Policies (loaders imported from the Phase-7 evaluator used for the paper)
# ---------------------------------------------------------------------------

def make_policy(entry: dict):
    """Return (callable obs_dict -> action_dict, reset_fn)."""
    sys.path.insert(0, str(ROOT / "research" / "phase7"))
    import evaluate_multiseed as E  # noqa: E402
    agents = [f"cluster_{k}" for k in range(K)]
    kind = entry["kind"]
    noop = lambda: None  # noqa: E731
    if kind == "mappo":
        return E._mappo_policy(E._load_mappo_actor(Path(entry["path"])), agents), noop
    if kind == "central":
        from stable_baselines3 import PPO
        return E._centralised_policy(PPO.load(Path(entry["path"]), device="cpu")), noop
    if kind == "ippo":
        return E._ensemble_policy(Path(entry["path"])), noop
    if kind == "const":
        return E._const_policy(int(entry["action"])), noop
    if kind == "hysteresis":
        from src.baselines.hysteresis import MultiAgentHysteresis
        pol = MultiAgentHysteresis(agents=agents, t_up_gbps=float(entry["t_up_gbps"]),
                                   t_down_gbps=float(entry["t_down_gbps"]),
                                   cooldown_steps=int(entry["cooldown_steps"]))
        return pol, pol.reset
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Environment and episode loop
# ---------------------------------------------------------------------------

def build_env(world: dict, switching_costs: Path | None = None):
    from src.envs.multi_agent_upf_env import MultiAgentUPFEnv
    return MultiAgentUPFEnv(split="test", paths_cfg=paths_cfg_for(world, switching_costs))


F64 = ["reward", "energy_term", "qos_penalty", "switch_penalty", "cooldown_penalty",
       "power_watts", "power_watts_steady", "switching_energy_wh", "q_score", "delay_us", "predicted_loss"]
COMPONENTS = ["energy_term", "qos_penalty", "switch_penalty", "cooldown_penalty"]


def run_episode(env, policy=None, reset_fn=None, replay_actions: np.ndarray | None = None) -> dict:
    """One deterministic test episode. Closed loop if ``policy`` is given;
    fixed-action replay if ``replay_actions`` (T, K) is given."""
    assert (policy is None) != (replay_actions is None)
    obs, _ = env.reset(seed=RESET_SEED)
    if reset_fn is not None:
        reset_fn()
    agents = list(env.possible_agents)
    T = env._N  # noqa: SLF001
    arr = {f: np.zeros((T, K)) for f in F64}
    is_safe = np.zeros((T, K), bool)
    usr = np.zeros((T, K), bool)
    action = np.zeros((T, K), np.int8)
    load = np.zeros((T, K))
    fcst = np.zeros((T, K))
    t = 0
    while env.agents:
        if replay_actions is None:
            act = policy(obs)
        else:
            act = {a: int(replay_actions[t, k]) for k, a in enumerate(agents)}
        obs, r, _te, _tr, info = env.step(act)
        for k, a in enumerate(agents):
            ck = info[a]
            arr["reward"][t, k] = r[a]
            for f in F64[1:]:
                arr[f][t, k] = float(ck[f])
            is_safe[t, k] = bool(ck["is_safe"])
            usr[t, k] = ck["selected_upf"] == "USR"
            action[t, k] = int(act[a])
            load[t, k] = float(ck["actual_load_gbps"])
            fcst[t, k] = float(ck["predicted_load_gbps"])
        t += 1
    assert t == T
    return dict(arr=arr, is_safe=is_safe, usr=usr, action=action, load=load, forecast=fcst,
                step_h=float(env.step_h), tau=float(env.tau), steps=T)


def episode_metrics(ep: dict, delay_budget_us: float, max_loss: float) -> list[dict]:
    """Scalar metrics for the whole fleet (cluster='all') and each cluster.
    Definitions match research/phase7/evaluate_multiseed.py and the Task-C
    analysis (reports/wcnc2027-shared-policy-ablation/analyze.py)."""
    a = ep["arr"]
    steps = ep["steps"]
    days = steps * STEP_MIN / (24 * 60)
    act = ep["action"].astype(int)
    sw = (act[1:] != act[:-1]).sum(axis=0)
    sw_realised = (ep["usr"][1:] != ep["usr"][:-1]).sum(axis=0)
    assert np.array_equal(sw, sw_realised), "requested and realised switches differ"
    assert np.allclose(a["reward"], -sum(a[c] for c in COMPONENTS), atol=1e-6), "reward != -(components)"
    delay_x = np.maximum(0.0, a["delay_us"] - delay_budget_us) / delay_budget_us
    loss_x = np.maximum(0.0, a["predicted_loss"] - max_loss) / max_loss
    # violation severity = relative budget excess max(delay excess/budget, loss
    # excess/budget), unbounded; q_shortfall = 1 - Q in [0, 1] (Q as in the reward)
    severity = np.maximum(delay_x, loss_x)
    rows = []
    for cl in ["all", *range(K)]:
        sl = slice(None) if cl == "all" else slice(cl, cl + 1)
        n_cs = steps * (K if cl == "all" else 1)
        sev = severity[:, sl]
        viol = sev > 0
        r = dict(
            cluster=cl,
            reward=float(a["reward"][:, sl].sum()),
            energy_penalty=float(a["energy_term"][:, sl].sum()),
            qos_penalty=float(a["qos_penalty"][:, sl].sum()),
            switch_penalty=float(a["switch_penalty"][:, sl].sum()),
            cooldown_penalty=float(a["cooldown_penalty"][:, sl].sum()),
            energy_wh=float(a["power_watts"][:, sl].sum() * ep["step_h"]),
            energy_wh_steady=float(a["power_watts_steady"][:, sl].sum() * ep["step_h"]),
            qos_violation_rate=float((~ep["is_safe"][:, sl]).sum() / n_cs),
            delay_violation_rate=float((a["delay_us"][:, sl] > delay_budget_us).sum() / n_cs),
            loss_violation_rate=float((a["predicted_loss"][:, sl] > max_loss).sum() / n_cs),
            qscore_below_tau_rate=float((a["q_score"][:, sl] < ep["tau"]).sum() / n_cs),
            usr_share=float(ep["usr"][:, sl].sum() / n_cs),
            switches=int(sw[sl].sum()),
            switches_per_cluster_day=float(sw[sl].sum() / (K if cl == "all" else 1) / days),
            n_budget_violations=int(viol.sum()),
            severity_mean=float(sev[viol].mean()) if viol.any() else 0.0,
            severity_max=float(sev.max()),
            q_shortfall_mean=float((1.0 - a["q_score"][:, sl])[viol].mean()) if viol.any() else 0.0,
        )
        rows.append(r)
    return rows


def save_episode(path: Path, ep: dict, meta: dict) -> None:
    a = ep["arr"]
    np.savez_compressed(
        path, action=ep["action"], usr=ep["usr"], is_safe=ep["is_safe"],
        **{f: a[f].astype(np.float32) for f in F64},
        meta=json.dumps(meta))


def qos_budget() -> tuple[float, float]:
    from src.utils.config import load_yaml
    q = load_yaml("configs/scenario_rl.yaml")["upf"]["qos_budget"]
    return float(q["delay_budget_us"]), float(q["max_loss_pkts_per_interval"])


def set_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
