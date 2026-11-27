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
