# IEEE Networking Letters draft — working folder

**Target venue.** [IEEE Networking Letters](https://www.comsoc.org/publications/journals/ieee-lnet).
Rolling submission, 5-page hard limit (4 pages free + US$220 page
charge for the 5th), 75–100 word abstract, IEEE Style Files
(LaTeX), EDICS classification + ORCID required at submission.

**Status.** v0.1 markdown draft mapped onto the Letters format.
Conversion to LaTeX (`IEEEtran` `journal` mode) is pending.

## Files

- [`paper.tex`](paper.tex) — **LaTeX source** (IEEEtran journal
  mode). Built `paper.pdf` is **3 typeset pages** (well under the
  4-page free tier).
- [`paper.pdf`](paper.pdf) — compiled output.
- [`paper.md`](paper.md) — original markdown draft (kept for
  diffing / future edits).
- [`figure1.pdf`](figure1.pdf), [`figure1.png`](figure1.png) —
  Figure 1 vector + raster.
- [`Makefile`](Makefile) — `make` to rebuild, `make distclean` to
  start over. No system TeX install required beyond `pdflatex`
  itself; `IEEEtran.cls` is bundled in this directory.
- [`IEEEtran.cls`](IEEEtran.cls) — IEEE journal class file
  (v1.8b, bundled so the build is self-contained).

## Relationship to the conference draft

The conference-length version lives at
[`../paper-draft/paper.md`](../paper-draft/paper.md) (≈ 8 IEEE
pages, INFOCOM/GLOBECOM-style structure). The Letters version
keeps the same experimental backbone (physics-grounded reward,
8-seed paired bootstrap, swept-best classical baseline) and is
strictly a compression — not a different result.

## What was cut to fit 5 pages

| Conference draft section | Letters treatment |
|---|---|
| §2 Related Work (dedicated) | Citations woven into §I and §III |
| §6.2 full per-seed table | Replaced by Fig. 1 (per-seed scatter + bootstrap CI) |
| §6.3 per-cluster table | Compressed to one paragraph in §IV |
| §6.4 energy vs reward | One Wh column in Table I; prose dropped |
| §6.5 safety vs reward fig | Cut entirely |
| §6.6 v1 vs v2 reward-revision robustness | Mentioned in §II ("removing the parallel flat-cost term") |
| §6.6 cooldown sensitivity sweep | One sentence in §IV + "supplementary" pointer |
| §6.6 hysteresis band sweep table | Compressed to "[5,10,20,50,100] Mbps, band=50 best" in §IV |
| §6.7 centralised-PPO failure-mode detail | Two sentences in §IV |
| §7 Discussion + limitations | Dropped (Letters convention) |
| Reproducibility shell snippet | Replaced by a [TODO] one-liner to public artefact in §V |
| Dashboard description | Cut |

## Remaining work before submission

### P0 — blocking
1. ✅ **Authors + affiliations.** Pulled from the EnergyAwareUPF
   manuscript. ORCIDs still `[TODO]` — required by Manuscript
   Central at submission time; each author registers and adds
   their own.
2. ✅ **EDICS classification.** Set to **NL1.9 — Network
   Operations and Management** (primary). The Networking Letters
   EDICS list does not contain dedicated codes for AI/ML-for-
   networks, 5G core/NFV, edge computing, or energy efficiency,
   so a single primary code is the cleanest pick. NL1.8.1 listed
   as fallback if the editor requests a secondary.
3. ✅ **Figure 1.** Generated; embedded in `paper.md`. Source:
   `scripts/generate_letters_figure1.py`, reproducible from
   `reports/phase-7/paired_bootstrap_mappo_vs_ippo.json` and
   `multiseed_summary.json`.
4. **Bibliography.** User will handle. Twelve `[TODO]` entries
   currently in `paper.md`:
   - COMCOM-S-26-00430 (paper-aligned reward source)
   - `UpfTrafficForecaster` upstream
   - `UPF_NDT` upstream (threshold derivation + RAPL activation
     durations)
   - Open5GS / DPDK UPF measurement reference
   - One closest prior work on RL for VNF / network function
     autoscaling
   - Yu et al. MAPPO (NeurIPS 2022) — citation listed but verify
   - Schulman PPO + GAE — listed but verify
   - SB3 (Raffin et al. JMLR 2021) — listed but verify
   - PettingZoo (Terry et al. 2021) — listed but verify
5. ✅ **LaTeX conversion.** `paper.tex` compiles cleanly to a
   3-page `paper.pdf` in IEEEtran journal mode. Page layout:
   - Page 1: title, author block, abstract, index terms, §I, §II
     up through the reward equations.
   - Page 2: Fig. 1 at top, end of §II, §III, §IV body, §V,
     references.
   - Page 3: Table I (currently floats alone at top of page 3 —
     well within the 5-page limit; a future tweak could pull it
     inline if you want to compress further).

### P1 — strongly advised
6. **Cover letter.** Letters do not formally require one, but
   editors appreciate a one-paragraph statement on what the
   contribution is and why a Letter is the right format.
7. **Supplementary material.** If desired, package the cooldown
   and hysteresis-band sweep tables and the per-cluster reward
   breakdown as a single supplementary PDF.

### P2 — nice-to-have
8. **Public artefact pointer in §V.** Either a Zenodo DOI for
   the checkpoints + scripts, or the GitHub URL once cleaned up.

## Expected pacing

- Drafting (this folder): done, 1 hour.
- Bibliography + EDICS + ORCID + Figure 1 + LaTeX conversion:
  ~1 day focused.
- Internal review pass: 1–2 days.
- Submission: end of May → early June 2026.
- Letters typical time to first decision: rapid (weeks, not
  months) — well-suited to the user's "ship before end of summer"
  goal.

## Differences from the conference draft strategy

The conference draft hedges on multi-venue reuse (INFOCOM 2027,
GLOBECOM 2026 workshop, NetSoft 2027). The Letters route ships
*now* with a smaller scope and a faster decision. The two are not
mutually exclusive: the conference draft can be expanded into a
full journal paper later (IEEE Trans. Network and Service
Management), reusing the Letters as the seed plus the cut
material (CTDE ablation, more seeds, generalisation, cooldown
sweep details) restored.
