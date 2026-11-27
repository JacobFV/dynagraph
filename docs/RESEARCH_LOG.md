# Research log (lead)

A dated record of every effort, decision, question from the PI, and outcome. Detailed results live in the
per-track docs (`findings.md`, `MODEL_V2.md`, `ENVS.md`, `ROBOTS.md`, `G1.md`, `INKLING_SYS12.md`, `V3_GENERAL.md`).
This file records **why** things were done and what was decided. Times are UTC. Append-only: corrections are new
entries that point back to the entry they correct.

## 2026-11-23: 12-hour sprint (v1)
- Brief: typed physical memory with recurrent refinement, an adaptation of Inkling-2-Mini, and a Unitree G1 sim demo.
  The brief required honest labelling, no privileged-sim leakage, and no uploading of checkpoints or data.
  All heavy compute ran on the shared GB10 host.
- Outcome (`findings.md`):
  - Structure helped prediction under occlusion and delay.
  - The physics loss hurt.
  - The typed friction slot was not causally used.
  - No learned model beat the prior stub in the G1 closed loop.
  - The Inkling core is used by the trained model (bypassing it doubles error), but its added capability was not shown.

## 2026-11-24 to 2026-11-25: v2 (industrial stations, many robots, "servo pain")
- PI asked for complex industrial environments and robots, bulk handling plus fine manipulation, a private repo with
  images, status reports and videos.
- Tracks: ENVS (stations, skills, oracle ceilings), ROBOTS (embodiments), MODEL-V2 (world model + planner).
- Rounds 1 to 3 (`MODEL_V2.md` §9 and §10):
  - Gains over the plain heuristic in a few cells.
  - No consistent typed-vs-generic difference.
  - The gated residual is never significantly worse than the heuristic.
- Round 4, r7 (`MODEL_V2.md` §11):
  - Typed beats generic and never-probe on all 4 robots.
  - No win over probe-then-act (PTA).
  - The pre-registered probing rule tracked the embodiment rather than memory. This was diagnosed.
- Round 5, r8 (`MODEL_V2.md` §12):
  - Win on the gantry against PTA, +0.09 [+0.03, +0.16], with lot memory ON vs OFF at +0.17.
  - Null on the arms.
  - Typed beats generic on all 4 robots.
  - One erratum: test seeds were reused from round 3, and only for evaluation.
- Incidents and fixes:
  - A host crash at 17:08 on 2026-11-25. Nothing of ours was in flight.
  - Env-code contamination of the shared tree. Fixed with frozen trees, guards and an erratum (74706e3).
  - A `pkill -f` rule violation. New rule: kill only PIDs recorded in our own pid files.
  - GPU contention. Fixed with a `.gpu_small.lock` lane.

## 2026-11-26
- **PI question: does Inkling already understand language, and can a mode token switch between System II and
  System I, with all our training applying only in System I? Is it absolutely good, especially at fine motor?**
  - Answer: yes to language understanding.
  - Design: `<sys1>`/`<sys2>` tokens, with gated LoRA and memory cross-attention active only in System I.
  - Absolute competence is far from industrial. On r7, oracle success ≥ 0.9 only on fixture_place, UR5e
    bin_clearing (0.83) and UR5e tight_insert with force-ramp (0.90).
  - Decision (PI: "full impl, full ownership"):
    - simulation only;
    - the fine-motor focus is tight insertion and small-part picking (envs-r7 / r7.1);
    - INKLING System I/II pre-registered (3364798, erratum 74706e3).
- **Gen3 self-collision bug** (base_link/shoulder_link, 1.8 MN): fixed in envs-r7.1. Gen3 tight_insert oracle went from
  0.10 to 0.80. Every earlier Gen3 number understates the robot; errata are in `MODEL_V2.md` §13e and INKLING.
- **Repo made public** at PI request (github.com/JacobFV/dynagraph).
  - Audited first: no secrets, IPs, weights or data in the history.
  - Exposed: commit email, host name, home paths. The PI was told.
  - Added the MIT license, an academic README, and a LaTeX draft in `paper/` (drafted by a Fable agent, with
    claims softened where single-case).
- **ACT imitation baseline near zero on VAL** (round 6, `MODEL_V2.md` §13a draft).
  - The pipeline was verified:
    - replaying a demo's actions reproduces its observations to about 1e-4;
    - an overfit model executes the whole sequence but releases about 1.3 cm off target.
  - Decision (lead): finish the pre-registered ACT grid only, and report it as a weak-baseline limitation. Diffusion
    policy and 50 Hz control are future work, and were not built now (they would cost about a day with no expected fix).
- **INKLING offline results**:
  - All four System I arms trained on the same 128k samples.
  - Pretrained ≈ random-init core: the pretraining benefit is not shown (1 seed).
  - The compact model is as good as either.
  - Held-out drawer_open damage AUC is at chance.
  - System II stays bit-identical (perplexity 8.337 = base; skill parsing 0.83 = base).
  - Ungated training collapses language (perplexity 52.8, parsing 0.45).
  - The prefix bit-identity check fails at the rounding level because of MoE GEMM batch shapes. A row-wise re-test
    is queued.
- **INKLING closed-loop cell 1/3** (bin_ur5e, n = 30):
  - All six pre-registered comparisons are null on good parts.
  - Native System I damages more than MODEL-V2: +0.20 [+0.07, +0.37]. It never probes.
- **PI questions (2026-11-26, evening):**
  1. Can we train on more domains, such as Hookean elasticity and rigid-body motion predicted from a third-person view?
  2. What other areas can we test on?
  3. How well does the encoder understand human bodies, hands and fingers?
  4. Are domains compounded into one model or trained separately? A single general model that generalizes further
     is important.
  - Answers (lead):
    1. Physically yes. Passive observation identifies only ratios (k/m, c/m, mass ratios, restitution; friction is
       absolute), and one interaction with a known force fixes the scale. The current encoder does not see pixels or
       third-person video: it reads actor-legal force/pose series, current entity states and a scalar vision-material
       reading. Path: noisy tracked keypoints first, then a pixel encoder.
    2. Deformables (cloth, cables), liquids/pouring, granular scooping, stacking stability, legged contact (G1 terrain
       friction), human-robot handover, and external benchmarks (Physion, CLEVRER, PHYRE, IntPhys).
    3. Untested and not possible now: no bodies are ever observed. The route is a pose front end (body/hand
       estimates) plus human-object-interaction datasets (DexYCB, HO-3D, ARCTIC, GRAB; several are non-commercial and
       need registration), with humans treated as an embodiment without force sensing.
    4. Separate model lines today:
       - sprint compact;
       - v2 synthetic;
       - v2 stations (pooled across families and about 10 robots, with an embodiment token);
       - Inkling System I (pooled across 16 families).

       Generalization evidence: held-out robot positive (typed vs generic +0.14), held-out family negative.
  - **PI decision: "yes, let's do that path and all those other areas", and document everything internally.**
  - Plan: `V3_GENERAL.md`. Tracks V3-MODEL, ENVS-V3 and EXT start now, alongside round 6 and the INKLING closed loop.
- **PI comment + external audit (2026-11-26, late):**
  - The PI said force sensing must not be the only emphasised modality. We should be doing everything we set out
    to explore. What ultimately matters is general real-world performance.
  - The PI shared an external audit of `main@037bc57`. Its key finding, verified by the lead at
    `models/inkling.py:494`: the recurrent loop does not expose or write back named physical states; the readout
    reuses pre-loop typed estimates; physics losses constrain outputs only. Its other points:
    - v2's causal physical dependence comes largely from analytic features;
    - typed vs generic measures the whole package;
    - the lot-memory result lacks a classical cached-estimator comparator;
    - frames are SE(2) only;
    - visual "material" is a noisy sim scalar;
    - no other-agent model;
    - sys1/sys2 code is not yet public.
  - Decisions:
    - Abstract (paper + README) rewritten around the multi-modal thesis, with an honest "distance to thesis" section.
    - `V3_GENERAL.md` §0 makes the per-pass typed belief state the V3-MODEL priority.
    - MODEL round 6 gets the cached classical estimator added before its §13a freeze, if the freeze hasn't happened.
    - INKLING code is published when its run completes.
- **EXT track (agent EXT, branch `ext`, `docs/EXT.md`)**:
  - **License audit.** Done from the official pages (§1).
    - No registration needed, research use allowed: Physion v1, Physion++ and PHYRE (MIT/Apache); CLEVRER (CC0);
      IntPhys 2019 (CC BY); DexYCB (CC BY-NC); MediaPipe (Apache-2.0).
    - Needs PI registration: MANO, ARCTIC, GRAB, SMPL-X, HOT3D, ContactPose, H2O and OakInk v1 annotations.
    - Data license unstated, held: Kubric MOVi and ComPhy.
  - **Downloads.**
    - Local only, with sha256 and provenance in `results/ext/datasets_manifest.json`.
    - Caps: ≤ 150 GB downloaded and ≤ 300 GB footprint. Archives are deleted after a verified extraction.
    - The link gives ~1–2 MB/s, and S3 ~0.2 MB/s per connection, so we use parallel range downloads.
    - The Physion v1 readout-fitting set (~250 GB) is fetched as a 5% byte-prefix per scenario. Every number from it
      is labelled non-official.
  - **Physion v1.**
    - The loader, the official readout protocol (from neuroailab physics-models/physopt), and Table S1 are recorded
      verbatim.
    - Human accuracy recomputed from the official CSVs is within 0.008 of Table S1.
    - The colour-track pipeline check on Dominoes gives observed 0.48 and full 0.90: the chain works, and agent+patient
      tracks alone are not predictive there.
  - **Physion++.**
    - The official readout (800) and test sets are local. The 192-per-property test list is reconstructed from the
      official human data. Table 2 is recorded.
    - SAM 2.1 tier-B tracks, prompted by the cue map, have sub-pixel median centroid error against the simulator masks.
      They run on gb10, mass property first.
  - **Human pose.**
    - We use MediaPipe Hand (Apache-2.0, no MANO).
    - On DexYCB (9 sequences × 8 views) it gives recall 0.88, 2-D median error 7.1 px and PCK@20 0.90; the
      depth-lifted root-relative median error is 27 mm.
  - **H6.**
    - The draft is in §6: GENERIC − TYPED log-mass error, object-cluster bootstrap, controls C1 aperture / C2 lift
      kinematics / C3 identity lookup.
    - Lead decision: EXT freezes it in its own commit once V3-MODEL's human-track model and its generic twin exist
      (trained and run on TRAIN/VAL only), and before any inference on subjects 07–08. A code guard enforces this.
    - Fragility is untested: no accessible labels.
- **GPU lane overruns (lead note):**
  - V3-MODEL's first training job hit 6.2 GB at batch 1024. It was stopped after 5 min, and runs are now capped at
    5 GB, batch 512, bf16.
  - EXT's SAM2 smoke run hit 7.2 GB. It was killed within about 2 min, and the job now uses CPU offload with a 4.5 GB
    cap; its peak is 2.5 GB.
  - Neither overrun pushed the tenancy past its 60 GB total. Rule kept: < 6 GB per small job, with a hard allocator
    cap set before launch.
- **Round 6 frozen** (c22846d) before any test episode. It adds a classical cached-estimator secondary (S-lot) and the
  "whole package" note at the lead's request after the external audit. The VAL planner sanity run was stopped after its
  first partial cell so the test gets both CPU slots.
- **ENVS-V3 first ceilings:** stack_overhang, handover and cable_route have low headroom (oracle − nominal ≤ ~0.13).
  The lead asked for a value-of-latent column (n ≥ 20) and one physically honest change per low-headroom family.
  Failing that, the family is labelled competence-only.
- **Compute-rule lapse (ENVS-V3):** value-of-latent checks ran locally (nice 10, up to 10 processes, a few minutes
  each) instead of on gb10. The lead flagged it, and from now on every n ≥ 20 run goes on gb10 under the one-process
  queue.
- **ENVS-V3 headroom changes** (one documented physical change per low-headroom family):
  - stack_overhang: COM offset up to ±45%. Oracle − nominal is now +0.30.
  - handover: release delay 0.3–3 s. UR5e +0.30.
  - cable_route: thick cords. +0.15, borderline.
  - cloth_fold: heavier fabrics. Recalibrating.
  - g1_push_terrain: the change was reverted; the family is labelled competence/controller-limited.
  - pour: oracle − PTA ≈ 0 (a calibrate-once family), and it is reported that way.

### Backfill (lead, 2026-11-27 ~03:30Z): events since the entries above
- **Logging convention (PI, 2026-11-27):**
  - "ensure you are logging notes on everything so that other claudes in the future will be able to continue your
    work … not a handoff. just a continuous log of everything."
  - From now on every agent appends a dated entry here for each milestone, decision, incident, PI message and
    finding: one `cat >>` per entry, no rewrites. The lead commits periodically.
  - Locations: worktrees `../dynagraph-{envs-r7,envs-v3,ext,v3}` on branches `envs-r7`, `envs-v3`, `ext`,
    `v3-model`. gb10 trees: `~/dynagraph` (main/MODEL), `~/dynagraph_sys12` (INKLING), `~/dynagraph_v3`,
    `~/dynagraph_envs_v3`, `~/dynagraph_ext`. Frozen: `~/dynagraph_r6`, `_r7`, `_r7_1`; MODEL overlay
    `~/dynagraph_m6run71`. Each track's pid files are in `<tree>/logs/*.pid`, and the GPU ledger is
    `~/dynagraph/logs/gpu_small_ledger.log`.
- **Repo and paper:**
  - Public at github.com/JacobFV/dynagraph, MIT license (a1bc057).
  - Paper draft in `paper/` (037bc57, by a Fable agent; build with tectonic via `make`).
  - Abstract reframed around the multi-modal thesis (1cfc248).
- **INKLING (agent a2383a2eff398bd47; code in the main checkout, uncommitted until the run ends):**
  - Milestones 3–5:
    - Offline, 4 arms on the same 128,032 samples. Success / slip / damage AUC and NLL:
      - gated: .844 / .879 / .730, NLL −1.15;
      - ungated: .847 / .887 / .702, −0.98;
      - random-init: .840 / .862 / .717, −0.80;
      - compact: .826 / .885 / .711, −1.35.
    - Held-out drawer_open: damage AUC .51–.54, i.e. chance.
    - Gate re-test: System II logits bit-identical. The prefix check is not bit-identical (max diff 5.0), because
      MoE GEMM batch shapes change; a row-wise test is queued.
    - Language: perplexity 8.337, the same as base, in System II. Ungated: 52.8, parsing .45. The free-form scores
      are invalid (the thinking span was truncated), and a 256-token rerun is queued.
    - A MODEL-V2 reference row-subset bug (family_id ordering) was found; the re-eval is queued.
  - Closed-loop cell 1/3, bin_ur5e, n = 30: all six pre-registered good-parts comparisons are null.
    - C1 native − v2 damage: +0.20 [+0.07, +0.37].
    - C3 delegated − v2 damage: +0.13 [+0.03, +0.27].
    - Native never probes.
    - System II takes 8.5 s per decision under contention.
  - Still to run: pouch_ur5e, bin_gen3.
- **MODEL round 6 (agent a266f589ad5111d62):**
  - ACT is near zero on VAL (0.017), and the pipeline is verified.
  - Lead decision: finish the pre-registered grid, report it as a weak baseline, no second imitation variant.
  - §13a frozen at c22846d, with the S-lot classical cached estimator and the whole-package note.
  - Test running on gb10 (`scripts/v2/r9_test.sh`, then `r9_lot_test.sh`), about 9–10 h from 02:16Z.
- **V3-MODEL (agent afa9352bea6c04cff, branch v3-model):**
  - §0 implementation: LBS chain family; three modalities (proprio/force, RGB-D keypoint tracks, golden anchors);
    per-pass write-back; per-pass law loss.
  - Pre-registration 0b8b189 (A1–A8, C1–C6). Contract with the human modality: e610e78.
  - First TEST read: **LBS-law s0 FAILS the causal test.**
    - Own-box sign 0.52, action sign 0.10.
    - Counterfactual Δ −0.29 mm, CI crosses 0.
    - Prediction 16.0 mm, against classical ≈ 13 mm and oracle 3.7 mm.
    - Edge blocking has no effect; delayed evidence has no effect; mass corr 0.27.
  - Lead approved two EXPLORATORY arms, each pre-registered alone on fresh seeds:
    - P: law-projection layer, labelled a hybrid;
    - I: interventional MuJoCo counterfactual pairs, a non-bypass bottleneck from parameter slots plus initial state,
      and a mass-identifiability check first.
  - The Inkling port waits until an arm passes A1–A4 on VAL. The H6 human-track model is queued after §0.
- **EXT (agent a7904569556662098, branch ext; datasets local at `~/datasets`):**
  - License audit and registration list as above.
  - Physion v1 loader and official OCP reproduced; humans within 0.008 of Table S1.
  - Physion++ official readout/test local; SAM 2.1 tracks validated at sub-pixel error.
  - MediaPipe hands on DexYCB: recall .88, 7.1 px, 27 mm.
  - Lead decisions:
    - fetch DexYCB 02–08;
    - footprint ≤ 300 GB;
    - the H6 freeze waits for V3's human model, before any subject 07–08 inference, and is code-enforced;
    - control C3 (object-identity lookup) added;
    - fragility untested.
- **ENVS-V3 (agent af82bf1a1f6313039, branch envs-v3):**
  - First 3 ceilings had low headroom. Changes per family are listed in the entry above.
  - granular_dose spout went from 4 cm to 5 cm (with 4 cm there was no lever).
  - Compute lapse logged.
- **Open PI decisions** (asked, not yet answered):
  1. dataset registrations, MANO first;
  2. Kubric MOVi / ComPhy, whose data license is unstated;
  3. any physical robot or force-sensing hardware for a real-world track.
- **Host snapshot (03:30Z):** gb10 available memory 85 GB; disk 86 GB free (96%), down from 101 GB. Two
  `envs_v3_calibrate.py cloth_fold` processes are running (ENVS-V3's share is 1), and the lead is asking about it. A
  `uv pip install causal-conv1d` into `venv-fast` is running (an INKLING fast-path build, to be confirmed).

- **2026-11-26 22:40 CST — EXT status (continuous-log rule acknowledged).**
  - **Running jobs:**
    - Physion++ mass SAM2 tracks on gb10: readout 95/200 done, then the official test set of 192.
      - Driver PIDs: 2067874 (readout), then 2100095 (test; waits for the readout driver).
      - Logs: `gb10:~/dynagraph_ext/logs/sam2_pp_*.log`. Tracks: `gb10:~/dynagraph_ext/tracks_pp/`.
      - Each chunk runs under `gpu_small.sh` with a hard 4.5 GB allocator cap, set before launch; peak is 2.5 GB.
      - About 1 video per minute while holding the lock.
    - Local download chain (`scripts/ext/fetch_chain.sh`, PID file `/home/brandonin/datasets/logs/chain.pid`, log
      `chain.log`): CLEVRER val videos (~3.5/6.2 GB) → DexYCB s01 → Physion v1 readout prefixes (5%) → DexYCB s02–08.
      Each archive is extracted, verified and pruned.
  - **Since the last entry:**
    - The CLEVRER annotations (val, 5000) are extracted and pruned, and the loader is fixed to the official layout.
    - Fixed a bug before any results existed: the Physion++ observed window must be counted in track samples
      (tracks are stride 3).
    - Added `scripts/ext/h6_eval.py`: C0–C3, TYPED/GENERIC, leave-objects-out ridge, object-cluster bootstrap. The
      TEST split refuses to run before the H6 freeze commit.
  - **Next:** when the mass readout and test tracks land, run `physionpp_ocp.py readout`. Every number will be
    labelled as our track-feature reference readout, not a model result.
- **INKLING closed loop complete** (lead entry, from the agent's report, 2026-11-27): all 3 pre-registered cells ran
  on frozen envs-r6 with the guard passing. **No System I/II variant beats MODEL-V2 alone or PTA in any cell.**
  - pouch_ur5e (n = 30, primary = success):
    - Success: heuristic .20, PTA .57, MODEL-V2 .57, native .30, delegated .40.
    - native − v2: −0.27 [−0.43, −0.13]. delegated − v2: −0.17 [−0.30, −0.03]. Both significant losses.
  - bin_gen3 (held-out robot, pre-fix Gen3 model; primary = good parts per lot):
    - native − v2: +0.00. delegated − v2: +0.08 [−0.02, +0.18], not significant.
    - native − delegated: −0.095 [−0.19, −0.01].
    - Damage vs v2: native +0.37 [+0.20, +0.53], delegated +0.30 [+0.13, +0.47].
  - bin_ur5e: every comparison null; native damage +0.20.
  - Mechanism: the pre-registered System I decision rule (System1Head scoring + UCB > 0.15 probe) almost never
    probes. The delegated variant uses MODEL-V2's own models, so the losses come largely from the decision rule, not
    only from the Inkling model.
  - System II: 65–77% agreement with the script, no effect (C6). Latency under contention: 8.5 s on bin_ur5e, 33 s on
    bin_gen3.
  - Remaining: Gen3 offline eval, row-wise prefix gate test, 256-token language eval, aligned MODEL-V2 re-eval, then
    the final writeup.
- **Lead → INKLING:**
  - Approved one EXPLORATORY diagnostic on fresh seeds, n = 30, bin_ur5e and pouch_ur5e, within about 4 h of the
    small lane: native System I predictions plugged into MODEL-V2's LotPlanner probe rule. It separates the model from
    the decision rule.
  - Asked to publish the INKLING code and small results to main in one commit at the end.

## 2026-11-26 ~23:30 UTC, V3-MODEL: track started, isolation set up
- Worktree `/home/brandonin/Documents/dynagraph-v3` on branch `v3-model` (pushed to origin). gb10 directory
  `~/dynagraph_v3`, synced by `scripts/v3/sync.sh`; the shared sync/remote scripts are never used.
- GPU only through `~/dynagraph/scripts/v2/gpu_small.sh`, via `scripts/v3/gpu_job.sh`. One niced CPU process.
- PIDs are in `~/dynagraph_v3/logs/*.pid` (`*.child.pid` for the python children).

## 2026-11-27 ~00:30 UTC, V3-MODEL: decision, redirect to V3 §0 accepted
- What: after the PI's audit, the lead put §0 above H1–H3. I built §0 first; H1–H3 follow on the same architecture.
- Built (commit 2fb7818):
  - a MuJoCo 1-D contact-chain family (`models/v3/chain.py`);
  - an RGB-D keypoint tracker error model (`tracks.py`);
  - the looped belief store LBS and the V1 baseline, law residuals and counterfactual-edit rows (`belief.py`);
  - the §1 contract (`contract.py`);
  - data, training and evaluation code (`v3_chain*.py`).
- Data: `~/dynagraph_v3/data/v3_chain` (train 10k / val 600 / test 600 / test_L4 300 episodes; sha256 in
  `manifest.json`).

## 2026-11-27 ~00:50 UTC, V3-MODEL: findings during the build, and the fixes
- First data had 20 m/s slides and box tunnelling. Fix: an admittance speed limit on the pusher (0.6 m/s).
- A missing spring-rest estimate (the tracker never saw the last box) gave NaN. Fix: a map anchor for the rest
  position. The data were regenerated; the stale generator child was killed after its PID was recorded.
- The analytic simulator's oracle error was 45 mm because overlapping start estimates launched the penalty contacts.
  Fix: project the start estimates to a non-penetrating configuration. The oracle error is now 3.7 mm.

## 2026-11-27 ~01:30 UTC, V3-MODEL: VAL-pilot decisions (classical/oracle only), then the §0b pre-registration
- Decisions: planner goals are drawn as true-world-reachable displacements (uniform goals gave an infeasible
  oracle error of 23 mm), and the A1 correlation is Spearman.
- Pre-registration committed on its own: 0b8b189 (`V3_GENERAL.md` §0b), before any TEST run.

## 2026-11-27 01:42 UTC, V3-MODEL: incident, GPU memory over the lane rule
- lbs_law_s0 at batch 1024 used 6.2 GB, over the < 6 GB rule. I killed it after about 5 min (PIDs recorded).
- Fix: a hard torch cap of 5 GB, batch 512, bf16, 4000 steps. The budget change is recorded in §0b before the freeze.

## 2026-11-27 ~03:05 UTC, V3-MODEL: finding, first §0 TEST read (LBS-law s0)
- The causal test FAILS:
  - own-box sign agreement 0.52 (chance), and an edit moves the predicted motion by only 6% of MuJoCo's effect;
  - action sign agreement 0.10;
  - counterfactual-outcome Δ −0.29 mm [−0.85, +0.22];
  - edge blocking suppressed B2 in 0 of 82 rows;
  - delayed evidence: adding it mid-loop is the same as never adding it.
- Prediction is decent (16 mm; classical 13 mm, oracle 3.7 mm).
- Comparators: classical 0.95 / 0.80 / −5.8 mm; oracle 0.98 / 0.83 / −8.2 mm.
- Files: `~/dynagraph_v3/results/v3/eval/lbs_law_s0_test_*`, `V3_GENERAL.md` §0c (commit 5874514).

## 2026-11-27 ~03:15 UTC, V3-MODEL: decision, lead-approved exploratory arms and the identifiability check
- Identifiability check on VAL: a converged CEM recovers mass (L=1 corr 0.95; L=2 total mass 0.90). The
  excitation is adequate. The registered classical comparator (64×8 CEM) is under-converged; noted.
- Arm I (learned loop):
  - a non-bypass learned Rollout that sees only the slots and the τ0 state (unit-checked);
  - teacher forcing;
  - real MuJoCo do()-pairs (`data/v3_chain/{train,val}_cf.npz`, `v3_chain cfgen`).
- Arm P: a hybrid law-projection step inside the loop.
- Fresh test seed sets `test2` (4e6+) and `test2_L4` (5e6+) were generated. They stay untouched until their own
  pre-registration.
- Commits f97f16d (arms) and 5874514 (docs §0d).

## 2026-11-27 ~03:50 UTC, V3-MODEL: deviation in run order only
- I paused the frozen-matrix training driver (`train_all.pid`; lbs_law_s1 had not started) and the remaining L4
  classical/oracle CPU evals, so the arm I/P pilots could go first. Nothing is dropped.
- Both are re-queued: `logs/cpu_l4.sh` re-runs the L4 evals, and `train_all.sh` is restarted after the pilots (it
  skips finished checkpoints).
- Done so far in the frozen matrix: lbs_law_s0, v1_s0 and lbs_nolaw_s0 trained; the TEST eval of lbs_law_s0 is
  complete.

## 2026-11-27 04:31 UTC, V3-MODEL: finding, arm I trained (VAL)
- arm_i_s0 at VAL step 4000: final error 16.9 mm (LBS-law 17.0 mm).
- Parameter recovery at k=3 (arm I vs LBS-law):

| parameter | arm I | LBS-law |
|---|---|---|
| mass | 0.68 | 0.32 |
| friction | 0.67 | 0.58 |
| k | 0.89 | 0.84 |

- The VAL causal pilot is queued (`logs/pilot.sh`, `logs/pilot_arm_i_s0.log`). Next: the proj_s0 training, then a
  separate pre-registration for test2 if an arm passes A1–A4 on VAL.

## 2026-11-27 ~03:00 UTC, V3-MODEL: H6 interface (EXT) planned into the contract
- `contract.human_evidence` reads TrackSet(kind=human) by duck typing; the joint names were checked against
  ext `MP_HAND`. EMBODIMENT_HUMAN has no force sensing, and the grasp relation leads to the object's typed mass slot.
- Matched twin: LBS-generic (`kind=generic`). Sim human-lift pretraining scene: `passive.human_lift_episode`.
- Lead decision: train on real DexYCB subjects 01–05 and VAL 06 with leave-objects-out folds; never touch 07–08
  before EXT's freeze. Queued behind §0. Commits e610e78, 1147bf1, f97f16d.

### 2026-11-27 ~03:20Z — MODEL-V2 round 6 (r9): catch-up entry under the new logging rule
**Track:** MODEL-V2. **What happened in round 6 so far:**
- **Outside baseline (a): ACT imitation.** il-v1 demo/action format agreed with ENVS: env.demo_call (oracle-param
  scripted), control_state/set_control, 64x64 wrist camera. Recorder/executor in `src/physmem/models/v2/imitation/`.
  Data: 6 cells, 200 demos each, 792 successful, `data/v2_il_r7` and the image-split copy `data/v2_il_r7s`.
- **Baseline engineering, each diagnosed on TRAIN demos.** Recorded in §13a (13.0 is the dry run):
  - absolute chunk targets;
  - controller-command dims dropped (copycat);
  - binarized gripper;
  - leading idle ticks trimmed;
  - vision at 5 Hz, to fit the < 6 GB small-GPU cap.
- **ACT VAL grid:** 3 stages (H x vision; β / 50k steps; ensembling incl. open-loop). Frozen r9t_h10v_50k, m = 0.1,
  VAL mean 0.017.
  - Diagnosis: live and recorded observations agree to ~1e-4. A model overfit to 5 demos runs the whole sequence but
    releases ~1.3 cm off.
  - Lead decision (option 1): report it as a weak-baseline limitation; diffusion policy / 50 Hz are future work.
- **Baseline (b):** compute-matched typed / generic / nomem models in two regimes, 3 seeds each, config selected on
  VAL (`results/v2/r9_select_{budget,full}.txt`, trust in `results/v2/trust_r9_*.json`):
  - budget-matched = the memory view of exactly the demo episodes, `data/v2_r9_budget`;
  - full = r2-r6 + tranche9_r7 + the r7.1 Gen3 episodes + demos, `data/v2_r9_full`.
- **Incidents:**
  - The first dry run used uncommitted envs-r7-dev code. It was set aside and every script now asserts EXPECT_ENV.
  - Gen3 base/shoulder self-collision (1.8 MN): found by MODEL-V2, fixed by ENVS in envs-r7.1, erratum in §13e.
  - v2_train.sanitize dropped every Gen3 row (reported by INKLING): absurd continuous values are now masked instead.
  - A fill-pass wrapper matched its own pgrep pattern and waited forever: killed and relaunched.
  - An 8k-step budget model was superseded by 4k for consistency (`results/v2/checkpoints/_superseded_r9_8k/`).
- **External-audit request (lead):** added a classical cached-estimator baseline on bin_clearing lots
  (`LotPlanner.cached_estimator`) as secondary S-lot, plus the note that typed vs generic measures the whole package.
- **Pre-registration:** §13a frozen and committed on its own, **c22846d** (02:16Z), before any test episode.
- **Running now (gb10):**
  - The test chain (chain pid 1705445; `scripts/v2/r9_test.sh` pid 1705446) runs PROCS=2, then `scripts/v2/r9_lot_test.sh`.
  - Logs: `~/dynagraph/logs/v2_r9_test_*.log`, `v2_r9_test_il_*.log`, `v2_r9lot_*.log`, `v2_r9_test_chain.log`.
  - Outputs: `results/v2/closed_loop/v2_r9_<fam>_<robot>.json`, `results/v2_il/test/`, and `v2_r9lot_<robot>.json`.
  - First finished test cell, tight_insert ur5e (n = 30, success within budget): nominal .57, PTA .60, force-ramp .70,
    oracle .67, typed_full .50, generic_full .63, nomem_full .60, typed_budget .40.
- **Next:** wait for all cells (~9 h). Then `scripts/v2/r9_table.py r9t_h10v_50k 0.1`, the S-lot table, paired videos
  only for significant wins, §13b results, and the final report to the lead.
- **Round 6, first finished test cell** (lead entry from the MODEL report; CIs pending):
  - tight_insert/UR5e, n = 30, success: force-ramp .70, oracle-param .67, generic(full) .63, PTA .60, nomem(full) .60,
    nominal .57, typed(full) .50, typed(budget) .40. The ACT result for this cell is pending.
  - Point estimates put typed below the pre-registered best scripted baseline (force-ramp) and below generic.
  - About 9 h of the chain remain (chain pid 1705445 on gb10).

## 2026-11-26 (backfilled ~22:50 UTC), INKLING: milestone, fit / memory solution for Inkling-2-Mini
- Did: `src/physmem/models/inkling_quant.py` keeps the NVFP4/MXFP8 experts packed on the GPU and dequantizes only the
  routed experts on the fly, with Triton kernels. It uses the system CUDA-13 ptxas, because the bundled ptxas lacks
  sm_121a. The result is 20.6 GB allocated and 22.2 GB peak, against 52.8 GB for the v1 bf16 path. Bit-identical to
  what the transformers loader produces (`results/sys12/fit_check.json`).
- Why: tenant cap. The box keeper preferred <30 GB, even after approving 60 GB.
- Incident, fixed: the first training launch held 51 GB, from transformers' `caching_allocator_warmup` and its async
  load pool. I stopped it within minutes and disabled both in the loader. No run has exceeded ~27 GB since, so no
  60 GB start/end messages were needed.

## 2026-11-26 (backfilled), INKLING: milestone, System I/II architecture + gate test
- Did: `src/physmem/models/sys12.py`:
  - `<sys1>` is a trained embedding row plus an action token and 16 readout tokens.
  - System I runs 1 of the 3 core passes. Typed-memory cross-attention is zero-gated.
  - `GatedLoRALinear` has three modes: off (System II), gated (System I positions only), always (the ungated control).
  - `Sys2Router` does constrained / free-form handoff `<sys1> skill=.. target=..`.
- Gate test on the real checkpoint (`results/sys12/gate_test*.json`): System II logits are bit-identical to base (max
  diff 0.0) with the trained adapters installed, and greedy decoding is identical. Ungated LoRA changes the logits
  (max diff 32, KL 0.99).
- Finding: the prefix-state identity inside a System I call is NOT bit-exact (max diff 5.0). The MoE per-expert GEMMs
  batch every routed row, so appended rows change the rounding of prefix rows. A row-wise re-test is queued
  (`scripts/sys12/gate_test_rowwise.py`). Tiny-model unit tests pass 5/5 (`tests/test_sys12.py`).

## 2026-11-26 (backfilled), INKLING: decision, closed-loop pre-registration (commit 3364798)
- Did: `docs/INKLING_SYS12.md` §a. It covers 3 cells (bin lots ur5e, pouch ur5e, bin lots gen3 held-out robot),
  n = 30, fresh seeds +950000, 6 policies and comparisons C1–C6. I predicted native System I ≤ MODEL-V2.
- A 2-task VAL smoke was run before registration, with a 30-step checkpoint, for debugging only. Its numbers are unused.

## 2026-11-26 (backfilled), INKLING: incident + deviation, env contamination; erratum §a1 (commit 74706e3)
- Finding: remote `~/dynagraph` had envs-r7-dev code synced in at registration. The hash §a recorded for
  `models/v2/harness.py` (6914fb…) was of that version; the correct envs-r6 hash is 827545…. The VAL smoke also ran on
  r7-dev; it was debug only.
- Did: froze a run tree, `~/dynagraph_sys12`. It is a git-HEAD export whose envs/skills/embodiments match
  `~/dynagraph_r6` on 30/30 files, plus the sys12 overlay. `scripts/sys12/cl_guard.py` aborts unless
  ENV_VERSION == envs-r6 and every hash matches. Its abort path was tested.
- Erratum §a1 was committed alone before any test episode; §a itself is unedited. Training is unaffected (npz data, and
  every imported file equals HEAD).

## 2026-11-26 (backfilled), INKLING: decision, GPU turn-taking with MODEL-V2
- Did: `~/dynagraph/scripts/sys12/gpu_share.sh`, a bash loop on gb10. It sets HOLD_SYS12 while another agent's
  gpu_heavy job waits, so my pipeline yields at its next step boundary. About one MODEL job per sys12 step. It never
  preempts. Stop it with `rm ~/dynagraph/.sys12_share_on`.

## 2026-11-27 (backfilled), INKLING: milestone, System I training (4 arms, 128,032 samples each, seed 0)
- Test-split AUCs (success / slip / damage):
  - gated pretrained: 0.844 / 0.879 / 0.730
  - ungated: 0.847 / 0.887 / 0.702
  - random-init core: 0.840 / 0.862 / 0.717
  - compact: 0.826 / 0.885 / 0.711
  - (`results/sys12/eval/`)
- Finding: no pretraining benefit was shown (single seed). Damage is at chance on the held-out family.
- Findings / incidents:
  - (a) All Gen3 heldout_robot rows were dropped by the v2 sanitize: aux1 is about 1.8 MN, from the Gen3
    self-collision defect. MODEL fixed this live. My separate aux1-masked eval is `scripts/sys12/eval_gen3.py`.
  - (b) My bug: `family_id` depends on load order, so the MODEL-V2 reference evals used different rows for 11/16
    families. The aligned re-eval is `scripts/sys12/eval_v2_aligned.py`. The old outputs are kept as
    `*_misaligned.json`.

## 2026-11-27 (backfilled), INKLING: milestone, language retention (`results/sys12/lang_eval.json`)
- wikitext-2-raw-v1 test (public, sha256 5f1bea…), 24,576 tokens.
- Perplexity: base 8.337, gated in System II mode 8.337 (KL 0), ungated 52.8 (KL 1.38).
- Constrained task-to-skill parsing: 0.83 / 0.83 / 0.45.
- Headline confirmed: gated training leaves System II unchanged, and ungated training degrades it.
- Deviation: the free-form parse and instruction scores are invalid. The budget of 24–32 tokens ended inside the
  model's thinking span. A 256-token rerun is queued (`lang_eval_gen256.json`).

## 2026-11-27 (backfilled), INKLING: milestone, pre-registered closed loop (envs-r6, guard passed)
- bin_ur5e good parts / lot: heuristic 0.16, PTA 0.30, v2 0.21, native 0.18, delegated 0.23. All primary comparisons
  are null. Native damage is +0.20 [+0.07, +0.37] vs v2.
- pouch_ur5e success: heuristic 0.20, PTA 0.57, v2 0.57, native 0.30, delegated 0.40. Native − v2 is
  -0.27 [-0.43, -0.13]; delegated − v2 is -0.17 [-0.30, -0.03].
- bin_gen3 (Gen3 self-collision defect, fixed in envs-r7.1): good parts per lot 0.12 / 0.22 / 0.22 / 0.21 / 0.30.
  Native − delegated is -0.095 [-0.19, -0.01]. Native damage is +0.37 vs v2.
- Mechanism: the shared System I probe rule (System1Head + UCB > 0.15) almost never probes, while MODEL-V2's
  crush-gated LotPlanner and PTA do. System II agreement with the scripted choice was 65–77%, and C6 is null.
- Files: `results/sys12/closed_loop/s12r1_*.json`, `*.declog.jsonl`, `provenance_*.json`.
- Logs: gb10 `~/dynagraph/logs/sys12/cl_*.log`.

## 2026-11-27 ~22:55 UTC, INKLING: note, causal-conv1d install on gb10 is NOT mine; next steps
- `uv pip install --python venv-fast/bin/python ... causal-conv1d` (pid 2794391) builds against `~/oo/venv`, so it
  belongs to another tenant. INKLING installed nothing. My only download was the 0.7 MB wikitext-2 test parquet,
  `~/dynagraph/data/sys12_text/`. My new disk use is the checkpoints in `~/dynagraph/results/sys12`; the size is
  reported in the final entry.
- Running (gb10, frozen tree `~/dynagraph_sys12`): pipeline3 (Gen3 eval) → pipeline4 (row-wise gate test, 256-token
  language eval) → pipeline5 (aligned MODEL-V2 re-eval). Logs are `~/dynagraph/logs/sys12/pipeline{3,4,5}.log`.
- Next:
  - an EXPLORATORY diagnostic (lead request): native System I predictions in MODEL-V2's crush-gated LotPlanner rule,
    bin_ur5e + pouch_ur5e, fresh seeds, n = 30, if it fits in about 4 h;
  - then the final writeup and one commit + push of the INKLING code and small results.

## 2026-11-27 ~23:05 UTC, INKLING: decision, EXPLORATORY rule-vs-model diagnostic queued (lead request)
- Did: `scripts/sys12/explore_rule.py` uses MODEL-V2's own decision rules, with scripted skill choice and no System II.
  - bin lots use the crush-gated LotPlanner; pouch uses the ResidualPlanner anchored on PTA.
  - Arms: pta, v2_ens3, v2_s0 (single model) and native_v2rule (Inkling gated System I s1r1, single model).
  - Fresh seeds +960000, n = 30, bin_ur5e and pouch_ur5e. Labelled EXPLORATORY; frozen tree, guard checked first.
- Why: to separate "the Inkling model is worse" from "the pre-registered System I probe rule is worse".
- Deviation from the lead's wording: Inkling needs ~22 GB, so it cannot use the <6 GB small lane. It runs under
  `.gpu_heavy.lock` with turn-taking, estimated 2–3 h.
- Where: gb10 `~/dynagraph_sys12/scripts/sys12/pipeline6.sh`, logs `~/dynagraph/logs/sys12/explore_*.log`, outputs
  `results/sys12/closed_loop/explore1_*.json`.

## 2026-11-27T04:47Z ENVS-V3: backlog entry (work from 2026-11-26 ~18:30Z until now, logged retroactively under the new PI rule)
- What: built seven envs-v3 families on the r7 station API in the worktree `/home/brandonin/Documents/dynagraph-envs-v3` (branch `envs-v3`, pushed; ENV_VERSION "envs-v3"). The families are pour, granular_dose, cloth_fold, cable_route, stack_overhang, handover (a scripted kinematic human PROXY) and g1_push_terrain (floating G1, with fixed-base G1 as a control). Code is in `src/physmem/envs/{v3_motion,families_v3_*}.py`. Scripts: `envs_v3_{dev,sens,calibrate,table,stability,stills,archive}.py` and `envs_v3_queue.sh`. Tests: `tests/test_envs_v3.py` 21/21; legacy `tests/test_envs.py` 89/89.
- Older families are unchanged. Final-state hash, event count and sim time match the envs-r7 tree on fixture_place/probe, tight_insert/UR5e and bin_clearing/probe. The shared-code edits are additive hooks: `StationEnv.execute` routes `fsk_*`, and `oracle.rule` delegates to `env.v3_rule`.
- Task changes (reasons in ENVS.md, envs-v3 section; the lead accepted them on 2026-11-27):
  - Granular scoop became granular DOSING. Scooping ran ~20x slower than real time, and coarse grains collected ~0 g.
  - The pour liquid is frictionless spheres plus per-sphere drag. Frictional spheres clogged the spout and cost 3–5x more.
  - Cable uses a post route plus a latched socket plus a fragile housing.
  - Cloth uses a pull tab.
  - G1 is standing push only.
- Compute: one niced queue on gb10 in `~/dynagraph_envs_v3` (own tree, own rsync script). Queue PIDs are in `logs/v3_queue{1,2,3}.pid`; per-step child PIDs go in `logs/v3_step.pid` when needed.
- Deviation: the value-of-latent checks after the lead's 2026-11-27 feedback were run LOCALLY (nice 10, up to 10 procs, a few minutes each), and so was one n=20 granular_dose run. That breaks the PI rule (heavy loads on gb10). It has stopped; from now on everything above a smoke test runs on gb10.

## 2026-11-27T04:47Z ENVS-V3: incidents so far
- Superseded calibration: the first cloth calibration (gb10, ~01:50Z) ran on code with a tab-orientation bug (fin axis at 2 x cloth yaw + 45 deg). I stopped the queue with its STOP file and killed only the PIDs recorded in `logs/v3_step.pid`. The rows went to `results/v2/envs/_superseded_cloth_cal/`. A second archive, `_superseded_cloth_fold_v31range_*`, followed the v3.1 range change.
- Lead's report of TWO `envs_v3_calibrate.py cloth_fold probe 24 1` processes at 03:30Z (pids 2837728, 2777942): those PIDs no longer exist (checked 04:46Z). The cause is `multiprocessing.Pool(1)` in the calibration and ceiling scripts. Each step is a mostly idle parent python plus ONE worker, and both share the same command line. Only the worker uses CPU (e.g. now: queue 2078781 -> parent 2855928 idle -> worker 2855931 at 100 %). So this was not a duplicate: one compute process, inside the share. Next: no change needed. I'll say "parent + 1 worker" in future status reports so it is not misread.

## 2026-11-27T04:47Z ENVS-V3: value-of-latent changes (lead feedback on the first ceilings)
- Finding: in the n=8 v3 ceilings, headroom was low (nominal 0.75–1.0; oracle − nominal ≤ ~0.13). Table: `results/v2/envs/ceiling_v3_vol.md` (worktree).
- Decision: ONE physically honest change per low-value family (v3.1), evaluated at n=20 on gb10 (tag `v31`):
  - stack: COM ±45 %, D 0.6–0.75 L.
  - handover: release delay 0.3–3 s. A 10 N tug limit was tried and rejected because every policy failed.
  - cable: stiffness to 0.4 N·m/rad.
  - cloth: thickness 0.6–1.6 mm, with recalibration on 24 / 12 seeds.
  - granular_dose: 5 cm spout. With the 4 cm spout every tapping / lead candidate tied, so there was no lever.
  - G1: a heavier box did not help and was reverted. The family is labelled competence / controller-limited.
- Finding (local n=20, probe): pour nominal 0.20 / PTA 0.80 / oracle 0.75. The gain is from a calibrated stop lead. The value of the latent over a good prior is ~0 (PTA = prior here). It will be reported that way.
- Next: queue 3 on gb10 (`logs/steps3.txt`, log `logs/v3_queue3.log`, pid `logs/v3_queue3.pid` = 2078781) runs, in order:
  - cloth calibration (UR5e);
  - v31 n=20 ceilings for stack/handover/cable/cloth;
  - pour calibration (24 seeds) and its ceiling;
  - G1 ceiling;
  - granular_dose calibration (16 seeds) and its ceiling (probe + UR5e);
  - stability scan and stills;
  - then the freeze: tag `envs-v3`, ENVS.md section, final report.
- **Lead: two cloth processes resolved.**
  - The two `envs_v3_calibrate.py cloth_fold` processes seen at 03:30Z were an idle `Pool(1)` parent plus one worker,
    i.e. one compute process, within ENVS-V3's share. No kill was needed.
  - Lesson: count compute workers, not raw python pids.

- **2026-11-26 23:00 CST — EXT: CLEVRER validation videos downloaded** (6.206 GB, sha256 07f6d48b…). They are being
  extracted, verified and then the zip is deleted. PID 3946359, log
  `/home/brandonin/datasets/logs/clevrer_extract.log`. The download chain has moved on to DexYCB s01.

## 2026-11-27 ~05:45 UTC, V3-MODEL: finding, arm I VAL pilot, and VAL-stage design changes
- arm_i_s0 VAL causal pilot (`results/v3/eval/arm_i_s0_val_causal.v1clamp.json`):

| criterion | result |
|---|---|
| A1 own-box sign / Spearman / other boxes | 0.91 / 0.80 / 0.88 (pass) |
| A2 τ 0.3 / 0.6 s / impulse | 0.98 / 0.94 / 0.88 (pass) |
| A4 counterfactual-outcome Δ | −3.3 mm [−6.0, −0.7] (pass) |
| A3 action sign | 0.54 (fail; 37% of cases are ties, 9% wrong sign) |
| A6 delayed evidence | fail |
| A7 edge blocking | fail (B2 suppressed in 4% of rows) |

  Compare LBS-law s0 on TEST: 0.52 / 0.10 / −0.3 mm.
- A3 diagnosis: a clamp at the slot's own posterior mean changed the plan in 51% of episodes, because the clamp leaked
  into the loop's τ0-state estimate.
- Changes (VAL stage, before arm I's separate pre-registration; commit on `v3-model`):
  1. do() clamps act only on the slots the rollout reads, and the evidence loop is unclamped. Re-pilot queued:
     `logs/pilot_arm_i_s0b.*`.
  2. arm_i2: relation do()-pairs (MuJoCo with the downstream boxes removed, trained with that relation blocked) and
     late-evidence availability augmentation (20%). Pairs are regenerated by `cfgen`
     (`data/v3_chain/{val,train}_cf.npz`). Queue: `logs/explore2.sh`.
- The frozen-matrix driver was restarted (`logs/train_all2.log`, pid in `logs/train_all.pid`).
- **V3 arm I on VAL** (lead entry from the agent's report): learned loop + non-bypass rollout + MuJoCo do()-pairs
  **makes the slots causal on VAL**. Nothing has been run on TEST/test2.
  - Passes:
    - A1: sign 0.91, Spearman 0.80, other boxes 0.88.
    - A2: τ 0.98 / 0.94, impulse 0.88.
    - A4: −3.3 mm [−6.0, −0.7].
  - Fails A3: action sign 0.54, against classical 0.79. Cause: clamping leaked into the τ0-state estimate. Fix: do()
    acts only on the rollout's slots.
  - Still failing: A6 (delayed evidence) and A7 (edge blocking); the rollout learned a "boxes move" prior. Fix in
    arm_i2: relation do()-pairs and late-evidence augmentation.
  - Property correlations (mass / friction / k): 0.68 / 0.67 / 0.89, against LBS-law 0.32 / 0.58 / 0.84.
  - Identifiability is fine: CEM recovers mass at 0.95.
  - Arm P (hybrid projection) trained badly at first; a pilot is queued.
  - Lead instruction: count and report every VAL iteration (forking paths) in the arm-I pre-registration. test2 must
    be fresh seeds, never looked at.

- **2026-11-27 00:15 CST — EXT: DexYCB subject 01 complete** (download → extract → verify → archive deleted;
  174,680 files under `/home/brandonin/datasets/dexycb/extracted/`).
  - Hand front-end validation on all 100 sequences, H6 view only: recall 0.83, 2-D median 7.0 px, PCK@20 0.89,
    depth-lifted absolute median 21.9 mm.
  - Results: `results/ext/dexycb_pose_validation_s01_view0.json` (branch ext). The earlier 9-sequence × 8-view run is
    kept in `dexycb_pose_validation.json`.
  - Our measurement, not a published protocol. TRAIN subject only.
  - The download chain moves on to the Physion v1 readout prefixes, then DexYCB s02–08.
  - Physion++ mass SAM2 readout: 131/200 done.

- **2026-11-27 00:25 CST — EXT: H6 actor-legal features extracted for TRAIN subject 01** (`h6-feat-v1`,
  `/home/brandonin/datasets/dexycb/h6/h6-feat-v1/`, log `/home/brandonin/datasets/logs/h6_feat_s01.log`).
  - Of 100 sequences, lift-off was detected in 97, and 90 pass the draft exclusion rule (hand detected in ≥ 50% of
    frames and lift detected).
  - No mass analysis was run. H6 stays a draft until V3-MODEL's models exist.

### 2026-11-27T06:55Z — lead: small-lane check (ext_sam2_pp_ro_mass_c_003)
- V3-MODEL flagged that `.gpu_small.lock` had been held by EXT's SAM2 chunk c_003 since 05:37Z (about 76 min, against about 25 min for earlier chunks).
- Lead checked on gb10, by PID only:
  - pid 3098462 (`sam2_tracks.py`, 3.1 GB GPU) is at 100% CPU, and 3 new track files were written in the last 5 min. It is progressing, not hung.
  - Likely cause of the slowdown: the box is contended. Another tenant has 4 × `points.py` processes at 100% CPU, plus the INKLING `lang_eval` (about 24 GB, heavy lane).
- No action taken. Queued behind it, all small: V3 arm_i_s0 re-pilot, proj_s0, lbs_law_s1, arm_i2_s0.

- **2026-11-27 01:20 CST — EXT: gb10 small-lane etiquette (after the lead's FYI on chunk c_003 taking ~80 min).**
  - `flock` is not FIFO, so back-to-back chunks could starve the queued V3 jobs. Changes:
    - `scripts/ext/gb10_sam2_driver.sh` now uses 20-video chunks and pauses 5 min after each chunk, releasing the
      lock. The file was swapped by atomic rename; the running readout driver (PID 2067874) keeps its old copy for its
      last chunk.
    - The test-set waiter was relaunched: PID 3346796 replaces 2100095. It starts the mass-test driver 5 min after
      the readout driver exits.
  - Interim reduced-readout reference results (Physion v1, colour tracks, our features; NOT official, NOT a model):
    - observed: Dominoes 0.48, Support 0.54, Collide 0.54, Contain 0.52;
    - full: 0.90 / 0.90 / 0.72 / 0.55;
    - readout sets: n = 47–53 per scenario.
  - Tracker check (colour tracker vs simulator masks): the Contain patient is weak, median error 52 px.

## 2026-11-27 ~07:10 UTC, V3-MODEL: finding, arm P is a VAL null; converged classical reference
- proj_s0 (law-projection hybrid) on VAL: 62.9 mm, parameter corr ≈ 0, edit effect 0.8% of physics, A3 0.19.
  Null; not pursued further now. Files: `results/v3/eval/proj_s0_val_*`.
- The converged CEM classical (`classical_conv`) on VAL: own-box 0.94, action sign 0.73, closed loop 9.6 mm. It is
  no better than the registered classical, so A3 ≥ 0.80 sits above both classical estimators (0.79 / 0.73) and near
  the oracle (0.82). Recorded in V3_GENERAL §0d.1.
- Lane: an EXT sam2 job held it for about 90 min (lead checked: not hung, the host is CPU-contended). The queue order
  is unchanged.

## 2026-11-27 ~07:30 UTC, V3-MODEL: §0e pre-registration committed alone (39d4306); autonomous test2 pipeline
- Exploratory arm I on the fresh seeds test2/test2_L4 (never evaluated by anything). 4 VAL attempts are disclosed,
  and a deterministic VAL selection rule (`scripts/v3/select_arm.py`) picks arm_i (rollout clamp) or arm_i2
  (relation pairs + late evidence); a tie goes to arm_i2. Criteria are A1–A8 as in §0b, in each of 3 seeds.
- Matched comparators on test2: LBS-law, V1, nolaw (3 seeds each), classical and oracle.
- Drivers on gb10 (`~/dynagraph_v3`; PIDs in `logs/*.pid`):
  - `scripts/v3/arm_test2.sh` (`logs/arm_test2.log`): pilot of arm_i2 → selection → seeds 1–2 → test2 evals.
    Its CPU part logs to `logs/arm_test2_cpu.log`.
  - `scripts/v3/train_all.sh` (`logs/train_all2.log`) and `scripts/v3/eval_gpu.sh` (`logs/eval_gpu.log`): the
    frozen matrix.
  - `logs/explore2_gpu.sh`: the arm_i s0 re-pilot and arm_i2_s0 training.
- Results go to `results/v3/eval/*`. Report: `python -m physmem.v3_chain_report --split test2 --groups ...`.
- Arm P is a VAL null and is not carried to test2.

### 2026-11-27T07:20Z — lead: V3 §0e pre-registration verified
- Lead confirmed that 39d4306 on `v3-model` is a docs-only commit (docs/V3_GENERAL.md, +42 lines).
- It pre-registers exploratory arm I on fresh seeds test2/test2_L4. It discloses 4 VAL attempts, and the deterministic VAL selection rule `scripts/v3/select_arm.py` chooses between arm_i and arm_i2.
- The chosen arm runs ×3 seeds and is evaluated on test2 whatever its VAL result. Matched comparators: LBS-law, V1, nolaw (×3), classical and oracle.
- Arm P is a VAL null (63 mm, parameters at chance, edit effect 0.8% of physics) and is dropped.
- Caveat noted: the A3 bar (≥0.80) is above both classical estimators on VAL (0.73 converged CEM, 0.79 registered) and near the oracle (0.82). An arm-I A3 failure on test2 should be read in that light. The bar stays as registered.
- The Inkling port waits for an arm that passes A1–A4 on VAL, plus lead approval.

## 2026-11-27 ~07:40 UTC, V3-MODEL: hand-off state (all queued work runs autonomously on gb10)
- Done and committed on `v3-model`:
  - §0 build; the §0b pre-registration (0b8b189);
  - the §0c first read (LBS-law s0 fails the causal test);
  - the §0d arms and identifiability check; the VAL attempt log; the §0e pre-registration (39d4306);
  - the §1 contract, including the human modality for EXT H6; the generic twin; the passive and sim human-lift
    scenes.
- Pending, queued:
  - the arm_i s0 re-pilot (VAL);
  - arm_i2_s0 training → its VAL pilot → selection → seeds 1–2 → test2 evals (`logs/arm_test2.log`);
  - the frozen matrix: 5 more training runs (`logs/train_all2.log`) and their TEST evals (`logs/eval_gpu.log`).
- To summarise when finished:
  `PYTHONPATH=src python -m physmem.v3_chain_report --split test --groups "LBS=lbs_law_s0+lbs_law_s1+lbs_law_s2;V1=v1_s0+v1_s1+v1_s2;nolaw=lbs_nolaw_s0+lbs_nolaw_s1+lbs_nolaw_s2;classical=classical;oracle=oracle"`.
  Use the same groups with `--split test2` plus the chosen arm.
- When reporting A3, put classical (0.73 / 0.79 VAL) and oracle (0.82) next to arm I. Results go in
  `V3_GENERAL.md` §0f. Only a `\pending{}` block goes to paper/results.tex (lead's call).

- **2026-11-27 02:10 CST — EXT: Physion v1 readout prefixes complete** (47–53 stimuli per scenario, 5% byte-prefix of
  each official archive; `/home/brandonin/datasets/physion/readout/`).
  - Reference readout on all 8 scenarios (colour tracks + our hand-made features + the official readout procedure;
    the reduced readout set makes this NON-OFFICIAL, and it is not a model result):
    - observed: 0.48–0.54, about chance;
    - full-movie detection: 0.55–0.90;
    - for comparison, the published pDEIT-lstm gets 0.59–0.69 (Table S1).
  - Front-end findings:
    - the agent is tracked at sub-pixel error in every scenario;
    - the patient fails in Drape (recall 0.07), Contain (52 px), Drop (23 px) and Dominoes (6 px), so the colour
      tracker is not adequate for those scenarios;
    - tier B (SAM2) is the planned fix, after Physion++.
  - Files: `results/ext/physion_ocp_color-v2.json`, `physion_frontend_validation_color-v2.json`, EXT.md §3a
    (commit fdff1cd).
  - The download chain is now on DexYCB s02–08. Physion++ mass readout SAM2 is at 183/200.

- **2026-11-27 03:00 CST — EXT: Physion++ mass readout SAM2 tracks complete** (200/200; local copy at
  `/home/brandonin/datasets/physionpp/tracks/sam2-v1/`).
  - Check against the simulator `_id` segmentation (evaluator only), saved in
    `results/ext/physionpp_tracks_validation_readout_mass.json`:

    | object | recall | recall in observed window | median error | within 5 px |
    |---|---|---|---|---|
    | agent | 0.93 | 0.89 | 0.7 px | 93% |
    | patient | 0.74 | 0.59 | 0.5 px | 93% |

  - The patient is lost mainly before the cue frame. Backward propagation across the curtain and rearrangement phase
    drops it.
  - Next: the mass test driver (PID 3346796 → `gb10_sam2_driver.sh`, 192 videos in 20-video chunks with a 5-min
    lane release between chunks) has now started. When it finishes, run `physionpp_ocp.py readout`.

## 2026-11-27 ~07:50 UTC, V3-MODEL: finding, arm I VAL attempt 2 (do() at the rollout input)
- arm_i_s0 on VAL (`results/v3/eval/arm_i_s0_val_causal.json`); A3 on VAL for comparison: classical 0.79,
  classical_conv 0.73, oracle 0.82.

| criterion | result |
|---|---|
| A1 own-box sign / Spearman / other boxes | 0.94 / 0.80 / 0.92 (pass) |
| A2 τ 0.3 / 0.6 s / impulse | 0.98 / 0.94 / 0.86 (pass) |
| A3 action sign | 0.63 (FAIL; ties 31%, wrong sign 6%) |
| A4 counterfactual-outcome Δ | −4.7 mm [−7.4, −2.2] (pass) |
| A6 delayed evidence, A7 edge blocking | still fail |

- The clamp leak is fixed: a clamp at the slot's own posterior mean now changes the plan in 0% of episodes.
- So no arm passes A1–A4 on VAL yet, and the Inkling port stays on hold. arm_i2 is next, via the selection driver.

- **2026-11-27 03:20 CST — EXT: DexYCB subject 02 downloaded, extracted, verified and pruned** (174,008 files).
  - Local footprint is about 85 GB, against the 300 GB cap.
  - The chain continues with s03–s08.
  - The gb10 mass-test SAM2 driver (PID 3346796) is waiting on the small lane, which `v3_arm_i2_s0` has held since
    08:13Z.

- **2026-11-27 04:00 CST — EXT wrap-up for onboarding (branch ext, commit 1b53785; `docs/EXT.md` §8 "Status at
  2026-11-27 (in flight)").**
  - **Done:**
    - license and access audit;
    - Physion v1: loader, official OCP readout reproduction, published baselines, reduced-readout reference on 8
      scenarios (non-official);
    - Physion++: official sets local; SAM2 tracks on the 200 mass readout trials, checked against the simulator
      segmentation;
    - MediaPipe hand front end, validated on DexYCB subject 01;
    - H6 draft, analysis script and guard;
    - CLEVRER validation set local.
  - **Left running:**
    - gb10 SAM2 mass-test driver, PID 3346796, log `~/dynagraph_ext/logs/sam2_pp_te_mass.log`; it releases the lane
      between chunks;
    - local download chain, PID 2877296, on DexYCB s03 → s08 (footprint 83 GB / 300 GB).
  - **Open PI decisions:**
    - registrations: MANO, ARCTIC, GRAB, SMPL-X, HOT3D, ContactPose, H2O, OakInk v1;
    - license questions: Kubric MOVi and ComPhy.
  - **Next:** the exact commands are in EXT.md §8. The first is the Physion++ mass OCP reference, once the test tracks
    land.
  - H6 freeze still waits on V3-MODEL.

## 2026-11-27T09:10Z ENVS-V3: wrap-up handoff (in flight; branch envs-v3 at e82167b, pushed)
- What: committed all envs-v3 result files on branch `envs-v3`.
  - v3 rows (n=8, pre-change) and v31 rows (n=20) with value-of-latent tables, at `results/v2/envs/ceiling_v31_vol.md`.
  - Calibration rows and oracle tables.
  - Superseded files, each labelled with a README: `_superseded_cloth_cal` (80 KB) and `_superseded_granular_dose_4cm_spout`.
  - The granular_dose calibration rows are marked PARTIAL.
- ENVS.md now has an envs-v3 section with a "Status at 2026-11-27 (in flight)" subsection: table, PIDs, logs, what remains.
- Findings (v31, n=20, oracle − nominal):
  - stack_overhang: +0.15 to +0.40 on all robots.
  - handover: +0.15 to +0.45.
  - pour: probe +0.60; arms ~0. The arms fail the pour for every policy.
  - cable_route: ±0.05.
  - cloth_fold: −0.05 to +0.10. The v3.1 thickness change did NOT create a gap. Panda scores 0/20 because it cannot pinch the tab.
  - g1_push_terrain floating: 0.10–0.20 for every policy (controller-limited). Fixed-base: 1.0 for every policy.
  - Labels: stack, handover and pour are decision-relevant. Cable, cloth and G1 are competence / generalization.
- Running on gb10 (left running per lead): queue pid 2078781 (`~/dynagraph_envs_v3/logs/v3_queue3.pid`), log `logs/v3_queue3.log`. Current step: `envs_v3_calibrate.py granular_dose probe 16 1`. Still to run: the granular_dose n=20 ceiling (probe + UR5e), the stability scan (3 runs) and the stills.
- Remaining before the freeze:
  - finish those steps and fetch the results;
  - rebuild the value-of-latent table with granular_dose;
  - fill in the stability numbers;
  - tag `envs-v3`, make `~/dynagraph_envs_v3/src` read-only and record the sources sha256;
  - write the final RESEARCH_LOG entry and report.

### 2026-11-27 ~09:40Z — MODEL-V2 round 6: main test complete; wrap-up commit (PI onboarding)
**Finding (pre-registered §13a, envs-r7.1, n = 30 per cell, 8 cells):**
- P1 (typed full − best scripted) is a win in NO cell; it is significantly worse on tight_insert ur5e (−0.20).
- P2/P3 (typed − ACT) are significant almost everywhere because our ACT is ~0 (a weak-baseline limitation,
  documented in §13a).
- Typed vs generic / nomem with matched data and compute: one significant win (budget fixture_place ur5e +0.20) and
  one significant loss (full bin_clearing ur5e −0.13); the rest n.s.
- The best scripted baselines (force-ramp, nominal, PTA) are at or above every learned planner.
- Held-out Gen3 (r7.1): no learned planner beats force-ramp or PTA.
- Files: table `results/v2/closed_loop/robot_table_r9.md/json`, write-up docs/MODEL_V2.md §13b.

**S-lot secondary (audit cached estimator), probe gantry done:** typed v2 .67 ≈ cached estimator .69 (Δ −0.01, n.s.),
both below PTA .82. ur5e and panda are partial and Gen3 is queued.

**Still running:** gb10 pid 1705445 `scripts/v2/r9_lot_test.sh`. Logs `~/dynagraph/logs/v2_r9lot_*.log`; results
`results/v2/closed_loop/v2_r9lot_*.json` (`*.partial.json` while running).

**Wrap-up commit:** all round-6 code, scripts, results so far, §13b and §14 status/handover in docs/MODEL_V2.md.
Partial lot files are committed and labelled partial. INKLING's files are left out. pytest: 29 passed, 1 skipped.

**Next:** finish S-lot (tabulate with r7_table.py, commands in §14); typed-vs-ACT paired videos (the renderer still
has to be written); a final MODEL_V2 §13 update.

## 2026-11-27, INKLING: milestone + wrap-up, gate fully passes; single commit of INKLING code/results (lead request)
- Finding: gate check (d) PASSES with shape-invariant row-wise experts (`results/sys12/gate_test_rowwise.json`).
  Inside a System I call, the System II prefix core states are bit-identical to base, 4/4 (bin_clearing and
  pouch_transfer, 1 and 3 passes, max diff 0.0). The earlier mismatch was MoE GEMM-shape rounding, not information
  flow (see the 2026-11-26 gate entry).
- Finding: Gen3 offline eval (aux1 masked, one process, identical rows for every model; Gen3 self-collision caveat):
  - damage AUC: MODEL-V2 r7 0.778, System I arms 0.68–0.71, sample-matched MODEL-V2 0.49.
  - success and slip are similar (`results/sys12/eval/gen3_heldout_robot_aux1masked.json`).
- Did: added §c (results) and §s ("Status at 2026-11-27 (in flight)") to `docs/INKLING_SYS12.md`. Committing the
  INKLING paths only (inkling_quant.py, sys12.py, sys12_train.py, tests/test_sys12.py, scripts/sys12/, the doc and
  the small files of results/sys12; no .pt). The hash is in the next entry.
- Still running on gb10 (frozen tree `~/dynagraph_sys12`):
  - pipeline4 (pid 2339725): 256-token language eval, `logs/sys12/lang_eval256.log`
  - pipeline5 (pid 2451747): aligned MODEL-V2 reference re-eval
  - pipeline6 (pid 2867454): EXPLORATORY rule-vs-model cells
  - gpu_share.sh (pid 158722)
  - The collect/finish commands are in `INKLING_SYS12.md` §s.

## 2026-11-27, INKLING: wrap-up commit pushed, a9b0ea9 (main)
- Did: one commit of the INKLING paths only: 64 files, no .pt/.npz/weights. pytest tests/test_sys12.py: 5 passed on
  gb10 (tiny model, frozen tree).
- origin/main had not moved, so no rebase was needed. Other agents' unstaged files were left untouched. This research
  log is NOT in the commit (the lead commits it).
- Next: the gb10 chain keeps running (pipelines 4-6 + gpu_share; see INKLING_SYS12.md §s). When it finishes: collect
  with the §s commands, update §c3/§c4, add §c6 (exploratory). Those land in a follow-up commit.
