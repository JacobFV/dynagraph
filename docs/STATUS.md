# Project status and orientation (as of 2026-11-27)

Start here if you are new to the project. This page tells you where everything is. The *why* and the full history
are in [`RESEARCH_LOG.md`](RESEARCH_LOG.md), an append-only dated log of every effort, decision and incident. Per-track
results are in the track docs linked below.

## 1. What the project is

The aim is to show that a looped transformer core (Inkling-2-Mini) can hold an **explicit, editable, typed physical
belief state**: entity × property class, i.e. mass, friction, stiffness, pose, force, agency. Recurrent passes would
update that state in line with rigid-body, Coulomb-friction and Hookean structure. The state would be fed by several
modalities: vision, third-person tracks, proprioception, force/touch and digital-twin anchors. It should drive
decisions in **one general model** across domains and embodiments, and the bar is general real-world performance. See
README "Abstract" and "Distance to the thesis", and [`V3_GENERAL.md`](V3_GENERAL.md) §0.

**Everything so far is in MuJoCo simulation.** Nothing has run on hardware.

## 2. Reading order

1. `README.md`: abstract, honest status, main findings.
2. `docs/RESEARCH_LOG.md`: chronological record. Read the first ~60 lines for the arc, then the 2026-11-26/27 entries.
3. The doc for the track you will work on (table below).
4. `docs/COORDINATION.md` (science rules) and §6 below (compute rules on the shared box).
5. `paper/`: LaTeX draft (`make` uses tectonic). `\pending{}` and `\todo{}` mark unfinished results.

## 3. Branch and track map

Branches are **not merged into main yet**. They diverged from main and touch disjoint areas. Merging is an open
decision (§7).

| track | branch (local worktree) | doc | state |
|---|---|---|---|
| v1 sprint (compact models, G1) | `main` | `findings.md`, `G1.md`, `DATA.md` | done |
| v2 environments (industrial stations) | `main`; frozen trees `envs-r7` (`../dynagraph-envs-r7`) | `ENVS.md` | done (r7.1 frozen) |
| v2 robots | `main` | `ROBOTS.md` | done |
| MODEL-V2 (v2 world model + planner, rounds 1–6) | `main` | `MODEL_V2.md` (§13b results, §14 status) | round 6 main test done; lot secondary running |
| INKLING System I/II (mode token) | `main` | `INKLING_SYS12.md` | closed loop done; follow-ups running |
| V3-MODEL (per-pass typed belief store, causal test) | `v3-model` (`../dynagraph-v3`) | `V3_GENERAL.md` §0a–§0f (on that branch) | test2 pipeline queued |
| ENVS-V3 (cloth, cable, pour, granular, stack, handover, G1 terrain) | `envs-v3` (`../dynagraph-envs-v3`) | `docs/ENVS.md` envs-v3 section (on that branch) | v31 ceilings done; granular_dose + stability scan running; not yet frozen |
| EXT (Physion/Physion++, CLEVRER, SAM2 tracks, human hands) | `ext` (`../dynagraph-ext`) | `docs/EXT.md` (§8 status, on that branch) | SAM2 test-set tracking running |

Every track doc has a **"Status at 2026-11-27 (in flight)"** section. It lists exact PIDs, logs and collection commands.

## 4. Results so far (one line each; numbers and CIs are in the track docs)

- **v1:** structure helped prediction under occlusion and delay, but the typed friction slot was not causally used.
  No learned model beat a prior stub in the G1 closed loop.
- **MODEL-V2, round 5** (pre-registered): typed memory beats generic memory on all 4 robots, including the held-out
  Gen3. Lot memory beats probe-then-act on the gantry under a shared budget, +0.09 [+0.03, +0.16]. There is no win
  over probe-then-act on the arms.
- **MODEL-V2, round 6** (pre-registered; envs-r7.1, 8 cells, n = 30; `MODEL_V2.md` §13b) is **a null against the
  scripted baselines**:
  - No learned planner beats the best scripted baseline (force-ramp, nominal or PTA) in any cell, including
    held-out Gen3. Typed is significantly worse on tight_insert UR5e (−0.20).
  - Typed beats our ACT in 8 of 8 cells (7 of 8 budget-matched), but only because our ACT scores about 0. It is a weak
    baseline, not evidence against imitation.
  - Typed vs generic at matched data and compute: one significant win, one significant loss, the rest not significant.
  - Lot secondary (bin_clearing lots), probe gantry done: typed lot memory 0.67 good parts per lot vs the classical
    cached estimator 0.69, so no difference. Both are below always-probe PTA at 0.82. UR5e and Panda are running;
    Gen3 is queued.
- **INKLING:**
  - The pretrained core is about the same as a random-init core offline.
  - System II logits are bit-identical to the base model. Ungated training would destroy language (perplexity 52.8).
  - Closed loop, 3 cells: no wins. Pouch transfer shows significant losses, traced to the pre-registered System I
    probe rule.
  - A diagnostic that separates "model" from "rule" is running.
- **V3-MODEL** (the centerpiece):
  - The pre-registered looped belief store (LBS-law) **fails** the causal test on TEST: edit-direction agreement is
    0.52, i.e. chance.
  - The exploratory arm I (counterfactual do()-pairs, non-bypass rollout) passes A1, A2 and A4 on VAL and fails A3:
    action sign 0.63, against a 0.80 bar. Classical scores 0.73–0.79 and the oracle 0.82.
  - Delayed evidence (A6) and edge blocking (A7) still fail. Arm P is a null.
  - Evaluation on the fresh `test2` seeds is pre-registered (39d4306) and has not run.
- **ENVS-V3** (v31, n = 20, oracle minus nominal):
  - Decision-relevant: stack_overhang +0.15 to +0.40, handover +0.15 to +0.45, pour +0.60 on the probe robot only (the
    arms fail pour under every policy).
  - Competence/generalization only (no latent gap): cable_route ±0.05, cloth_fold −0.05 to +0.10 (Panda cannot pinch
    the tab), g1_push_terrain (the floating G1 reaches 0.10–0.20 under every policy, so it is controller-limited).
  - granular_dose is still calibrating.
- **EXT:**
  - Physion loader built and the official OCP protocol reproduced.
  - SAM2 object tracks are sub-pixel.
  - MediaPipe hands: recall 0.88, 7.1 px, 27 mm.
  - The H6 (humans) pre-registration draft has controls C1–C3.

## 5. In flight on gb10-direct (as of about 08:00Z on 2026-11-27)

Each track has its own directory on gb10, and none of them uses another's. **Only `scripts/sync.sh`/`remote.sh`
target `~/dynagraph`.** Worktree branches rsync to their own directory.

| track | gb10 dir | what is running | how to collect |
|---|---|---|---|
| MODEL-V2 round 6 | `~/dynagraph` | chain pid 1705445: `r9_lot_test.sh` (lot secondary; the main test is done) | see `MODEL_V2.md` status section (`scripts/v2/r9_table.py`) |
| INKLING | `~/dynagraph` (+ `~/dynagraph_sys12`) | `logs/sys12/pipeline2.log`: Gen3 offline, row-gate test, 256-token language eval, aligned MODEL-V2 re-eval, then the exploratory diagnostic (heavy lane, about 22 GB) | see `INKLING_SYS12.md` status section |
| V3-MODEL | `~/dynagraph_v3` | `scripts/v3/arm_test2.sh` (pid 3468490): arm_i2 VAL, selection, seeds 1–2, test2; `train_all.sh` (3108447) and `eval_gpu.sh` (1970118) for the frozen matrix | `PYTHONPATH=src python -m physmem.v3_chain_report --split test --groups "LBS=lbs_law_s0+lbs_law_s1+lbs_law_s2;V1=v1_s0+v1_s1+v1_s2;nolaw=lbs_nolaw_s0+lbs_nolaw_s1+lbs_nolaw_s2;classical=classical;oracle=oracle"`, then the same with `--split test2` plus the selected arm. Results go to `V3_GENERAL.md` §0f |
| ENVS-V3 | `~/dynagraph_envs_v3` | queue pid 2078781 (`logs/v3_queue3.log`): granular_dose calibration, then the n=20 ceiling on probe and UR5e, the stability scan, stills | `docs/ENVS.md` status subsection on `envs-v3`; then rebuild the table, tag `envs-v3`, make src read-only, record the sha256 |
| EXT | `~/dynagraph_ext` | SAM2 driver pid 3346796 (192 Physion++ mass test videos, 20-video chunks, small lane), log `logs/sam2_pp_te_mass.log`. **Laptop:** DexYCB download chain pid 2877296 (s03–s08, 83 of 300 GB cap) | `docs/EXT.md` §8 on `ext` (next: the Physion++ mass OCP readout) |

The `~/dynagraph_r*`, `_dev*`, `_m6run*` and `_ab` directories on gb10 are frozen per-round trees kept for
reproducibility. Don't edit them.

## 6. Shared-box rules (gb10-direct is shared with other tenants; these are binding)

- All heavy compute runs on gb10-direct, not on the laptop.
- **Our total share:** ≤ 6 niced CPU compute processes across all tracks, and < 30 GB RAM. Up to 60 GB is allowed for
  Inkling runs, announced in advance to the other tenant's agent. Count compute workers, not raw python PIDs.
- **Small GPU jobs:** `scripts/v2/gpu_small.sh` (flock `~/dynagraph/.gpu_small.lock`), one at a time, each < 6 GB,
  with a hard allocator cap set before launch. Each job starts only once `free -g` shows at least 40 GB available.
  The ledger is `~/dynagraph/logs/gpu_small_ledger.log`.
- **Heavy GPU jobs:** `scripts/gpu_heavy.sh` (`.gpu_heavy.lock`, ledger `logs/gpu_ledger.log`).
- Never touch `~/oo`, `~/cwfid`, `~/opt`, `~/brain` or the VMs. **Never** use `pkill -f`, `pgrep -f` to kill, or
  `killall`; kill only PIDs you recorded in your own pid files.
- The disk is about 96% full (77 GB free). New data on gb10 must stay ≤ 15 GB in total. Large datasets stay on the
  laptop, and only preprocessed subsets go to gb10.
- If another tenant asks for room, accommodate them. Pause, then check every 30 min whether the box is still busy.

## 7. Science conventions (non-negotiable)

- Pre-register every test (hypotheses, metrics, thresholds, seeds) in a commit **on its own**, before running it.
  Errata are new commits.
- Paired evaluation, n ≥ 30, 95% bootstrap CIs. Tune on VAL only, and disclose every VAL attempt.
- The labelled oracle is a ceiling only. No privileged simulator state crosses the actor boundary.
- Held out: the Kinova Gen3 robot, the `drawer_open` family, and in V3 the 4-box chains and the `test2` seeds.
- Paired videos are rendered only for significant wins.
- Never upload checkpoints or datasets to external services. Respect dataset licenses and record sha256 in manifests.
- Log every event in `RESEARCH_LOG.md`: append with `cat >>` and never rewrite.

## 8. Open decisions for the PI

1. Dataset registrations for H6 (humans): MANO, ARCTIC, GRAB, SMPL-X, HOT3D, ContactPose, H2O, OakInk.
2. Kubric MOVi and ComPhy: their data licenses are unstated.
3. Any physical robot or force/touch hardware for a real-world track.
4. Whether to merge `v3-model`, `envs-v3` and `ext` into `main` now, or after their pending results.
5. The Inkling port of the V3 belief store, which waits for an arm that passes A1–A4 on VAL.

## 9. Next steps by track

- **MODEL-V2:** finish and tabulate the lot secondary (`MODEL_V2.md` §14). Then the paired videos (the only
  significant wins are typed vs ACT, and that renderer isn't written yet), the final §13, and the paper results.
- **INKLING:** finish the follow-ups and the diagnostic, then final `INKLING_SYS12.md` numbers and the paper section.
- **V3-MODEL:** test2 result → §0f. If an arm passes, the Inkling port (with approval), then H1–H3 unification
  across domains (passive third-person spring and rigid-body scenes are drafted), then H6.
- **ENVS-V3:** finish granular_dose and the stability scan, then stills, then freeze and tag `envs-v3`. Cloth and
  cable need a redesign if they are to carry a latent gap.
- **EXT:** Physion++ mass OCP on SAM2 tracks, Physion v1 for the remaining scenarios, then the H6 freeze once the V3
  human model exists.
