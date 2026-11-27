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
