# Cross-repo fix-up prompts

Ready-to-run agent prompts for the three sibling repositories in the thesis
project. Each is self-contained — run it as a fresh session **inside that repo**.

Written 2026-07-21, after the artifact-tracking pass in this repo
(commits `bc1749c` + `9f148eb`). Facts below were verified against the
working trees on that date; re-check before acting if significant time has passed.

## Why these exist

The five repos form one pipeline:

```
UpfProfilingCampaign          UpfTrafficForecaster
(testbed power/QoS models)    (NetMob -> cluster forecasts)
        |                             |
        | models/                     | exports/traffic_forecaster/
        v                             v
        +------> UPF_NDT (UpfDigitalTwin) <------+
                 controller-agnostic twin
                          |
                          | pip install + dvc pull
                          v
                 UpfRLControllers (this repo)
```

The seams between repos — not the code inside them — are where the
reproducibility and integrity problems live.

## Already done in this repo (no agent needed)

- All 9 orphaned `data/external/` artifacts are DVC-tracked and pushed to S3;
  `.dvc` pointers carry provenance headers. `dvc pull` alone now reproduces a
  fresh clone.
- `bootstrap_external_data.py` and its hardcoded
  `/home/ubuntu/UPF_NDT/data/external` path are deleted; 4 references updated.
- `scenario_rl.yaml`: `selected_k` 8 -> 10, `service` video -> Netflix (both are
  validated by `DigitalTwin.traffic_loader` and would raise; the RL env only
  dodged it by loading arrays itself). `alpha = 1.0` is now a **declared
  modelling assumption**, not a placeholder.
- The 3 hand-written forecaster pointers' stale "commit 020373a" citation
  corrected.

## Open decision owned by the author (not delegable)

`max_loss_pkts_per_interval` (this repo 5.0 / UPF_NDT 0.0) and
`upf_switching.prewarm` (this repo off / UPF_NDT on) change reported
magnitudes. The NDT prompt asks the agent to surface and analyse these, but the
direction of reconciliation is a research decision.

---

## 1. UpfTrafficForecaster (`UPF_Forecasting/UpfTrafficForecaster`)

```
You are working in UpfTrafficForecaster, the traffic-forecasting repo of a 5-repo
PhD thesis project. Its downstream consumer, UpfRLControllers, imports forecast
arrays from this repo's S3 DVC remote. That handoff is currently fragile — fix it.

Context (verified facts):
- dvc.yaml's `evaluate_forecast` stage declares `results/cluster_first/Netflix`
  as a cached dir output, and `forecast_eval_summary.json` as a `cache: false`
  metric — so the summary lives in NO DVC remote, and downstream had to hand-copy it.
- scripts/export_for_twin.py + exports/traffic_forecaster/EXPORT_MANIFEST.json exist
  but export_for_twin is NOT a dvc.yaml stage, so exports are not reproducible via
  `dvc repro` and cannot be `dvc import`ed at a pinned rev by downstream.
- The K10 checkpoint (checkpoints/cluster_first/Netflix/K10/best_model.pt) was
  retrained 2026-06-11, AFTER downstream copied its forecast arrays — so the current
  outputs differ from what every downstream result was built on. Downstream now
  pins those exact bytes by md5; do NOT try to "fix" that mismatch by forcing a match.
- This repo is on branch feature/cluster-first-stgnn; nothing is on main/tagged, so
  downstream cannot pin a stable rev.

Tasks:
1. Add an `export_for_twin` STAGE to dvc.yaml that runs scripts/export_for_twin.py
   and declares exports/traffic_forecaster/ as a cached DVC output containing the
   full downstream set: predictions_{train,val,test}.npy, targets_{train,val,test}.npy,
   cluster_series.npy, cluster_assignments.parquet, cluster_bs_map.json, AND a
   cached copy of forecast_eval_summary.json (so it stops being cache:false-only and
   becomes fetchable from the remote). `dvc repro` then `dvc push`.
2. Verify README/PIPELINE staleness: README describes the deprecated per-gNodeB LSTM
   while the live model is the cluster-first STGNN (PIPELINE.md flags README stale).
   Update README to describe the current STGNN pipeline and the export contract.
3. Tag a stable release (e.g. `thesis-v1`) on the commit whose exports match the
   downstream md5 pins, OR merge feature/cluster-first-stgnn to main. Report the
   commit hash so downstream can convert its 3 hand-written pointers + array copies
   into real `dvc import ... --rev <hash>`.

Do NOT retrain the model or alter forecast values. This is packaging/repro plumbing
only. Report back: the export stage diff, the push confirmation, and the pinned
commit hash for downstream to import.
```

---

## 2. UpfProfilingCampaign

```
You are working in UpfProfilingCampaign, the power/QoS profiling repo of a 5-repo
PhD thesis project. Downstream repos (UPF_NDT, UpfRLControllers) DVC-import your
models/ directory. There is one orphaned artifact that belongs here but lives only
in downstream gitignored dirs — bring it home.

Context (verified facts):
- dvc.yaml correctly emits models/ as a cached DVC output (downstream's models.dvc
  import @ rev a73870f works). Good — don't touch that.
- switching_costs.yaml is a hand-authored file holding RAPL activation-cost
  measurements (dpdk activation 24.0s, usr 3.3s, with a spike_wh ~ P_steady * t
  scaling rule). It is NOT a dvc.yaml output and exists ONLY in
  UpfRLControllers/data/external/profiling_twin/ and UPF_NDT/data/external/... —
  both gitignored. It is load-bearing: it drives the switching-energy term L_SW in
  the RL reward. If those two machines are wiped, the measurements are gone.
- Same situation for params.yaml (the profiling config snapshot downstream pins).

Tasks:
1. Add switching_costs.yaml into this repo as a first-class, version-controlled
   artifact (it is measurement data that was produced here / belongs to the profiling
   campaign). Decide whether it is a plain git-tracked config or a DVC-tracked output;
   given it's small hand-authored YAML, git-tracking it directly is fine. Preserve its
   provenance header verbatim.
2. Ensure downstream can obtain switching_costs.yaml + the profiling config snapshot
   by `dvc import` / `dvc pull` from THIS repo rather than by hand-copy. If you add
   them as DVC outputs, `dvc push`.
3. Confirm the models/ pin (rev a73870f) is on main or a tag so downstream imports
   stay stable; if it's only on an unmerged branch, tag it (e.g. thesis-v1) and report
   the hash.

Do NOT recompute or edit the measurement values. Report back: what you added, how
downstream should now fetch it (exact `dvc import`/path), and the stable pin hash.
```

---

## 3. UPF_NDT (UpfDigitalTwin)

```
You are working in UPF_NDT (packaged as the `upf-digital-twin` pip package,
GitHub remote UpfDigitalTwin). It is the measurement-grounded digital twin of a
5-repo PhD thesis project; UpfRLControllers pip-installs it at tag v0.1.0 and
consumes DigitalTwin. Two things are broken: reproducibility hygiene and a set of
scenario-config divergences that threaten thesis integrity.

Context (verified facts):
- Branch is setup/reproducible-env-and-data. reports/ — which contains the digital-
  twin THESIS CHAPTER (chapter_digital_twin.tex/.pdf) — is entirely UNTRACKED by git.
  app/dashboard.py is modified/uncommitted. Nothing thesis-relevant is on main/tagged.
- configs/scenario.yaml diverges from UpfRLControllers/configs/scenario_rl.yaml, which
  is now the CANONICAL scenario for the controller work:
    * traffic alpha: this repo uses alpha_gbps_per_norm = 0.12; the controller repo
      has DECLARED alpha = 1.0 as an explicit modelling assumption (NetMob is
      dimensionless; no scaler exists to recover). At 1.0 the fleet sits at
      0.06-1.66 Gbps mean / 5.8 Gbps peak (USR QoS-safe ~29% of steps); at 0.12 it's
      ~91% safe — a DIFFERENT operating regime and effectively a different paper.
    * upf.qos_budget.max_loss_pkts_per_interval: this repo = 0.0 (binary is_safe,
      zero tolerance); controller repo = 5.0 (nonzero normalization base for its
      graded QoS score; 0.0 would divide-by-zero there).
    * upf_switching.prewarm: this repo = enabled:true, standby 0.05W; controller repo
      = enabled:false, standby 0.0W. This changes switching energy -> reported switch cost.

Tasks:
1. Reproducibility: commit reports/ (the chapter) and the dashboard change on a
   sensible branch, then tag a release the controller repo can pin. Nothing important
   should be untracked.
2. Scenario reconciliation — the twin chapter and controller chapter currently
   describe different physical regimes. For EACH of the three divergences (alpha,
   max_loss, prewarm): determine whether it is (a) a genuine divergence that must be
   unified to ONE value across both repos, or (b) an intentional, defensible
   difference (e.g. the twin's binary is_safe legitimately uses max_loss=0.0 while the
   controller's graded score needs 5.0). Produce a short written reconciliation note
   mapping each field: canonical value(s), rationale, and — critically — whether the
   twin chapter's reported numbers were computed under a value that differs from the
   controller chapter's. FLAG anything that changes reported magnitudes; do NOT
   silently change a value that would invalidate already-reported twin results — surface
   it for the author to decide. The controller repo has committed to alpha=1.0; the
   twin chapter must either adopt 1.0 or explicitly state it uses 0.12 as an
   illustrative value and why.
3. If reconciliation changes any config the pip package ships, cut a new tagged
   release and report the tag so UpfRLControllers can bump its dependency pin from
   v0.1.0.

This repo must stay controller-agnostic (no imports of / references to
UpfRLControllers in code). Report back: the reconciliation note, what you committed/
tagged, and any magnitude-affecting change that needs author sign-off.
```

---

## After the agents finish — back in this repo

1. Convert the 3 hand-written pointers (`cluster_series`, `cluster_assignments`,
   `cluster_bs_map`) plus the 6 array copies into real `dvc import ... --rev <hash>`
   using the forecaster's reported tag. Blocked until prompt 1 lands.
2. `dvc import` `switching_costs.yaml` / `params.yaml` from UpfProfilingCampaign
   instead of tracking local copies. Blocked until prompt 2 lands.
3. Bump the `upf-digital-twin` pin in `pyproject.toml` past `v0.1.0` if UPF_NDT
   cuts a new release. Blocked until prompt 3 lands.
4. Re-run Phase 7 / Phase 9 if any reconciliation changes a magnitude-affecting
   config, then refresh `reports/paper-mascots/`.
