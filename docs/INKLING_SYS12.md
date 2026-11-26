# INKLING_SYS12 — System I / System II on the pretrained Inkling-2-Mini looped core (owner: INKLING agent)

Labels used everywhere: **"Inkling-2-Mini core (pretrained)"** = thinkingmachines/Inkling-2-Mini@b85c3cf (packed
NVFP4/MXFP8 experts, this repo's loader); **"Inkling random-init core"** = the same architecture with shuffled weights
(control); **"compact"** = a 2-layer d=128 transformer System I (control); **"MODEL-V2"** = the MODEL agent's typed
physical-memory world model + planners (r7 checkpoints, 2.28 M params). Nothing here ran on a physical robot.

## a. Pre-registration of the closed-loop comparison (deliverable 6)

Written 2026-11-26, BEFORE any closed-loop episode on a TEST (or held-out-robot) split was run with any System I / II
policy. Before this, only a 2-task VAL smoke per cell (smoke checkpoint `smoke_gated`, 30 training steps) was run, to
debug the code; its numbers are not used. This section is frozen; any later change is a new, labelled section.

**System under test.** System II = Inkling-2-Mini (pretrained) base LM with every adapter OFF (bit-identical to the
base model, gate test `results/sys12/gate_test.json`). At each decision it reads an actor-legal cell-state summary
and the list of allowed calls (`scripts/sys12/closed_loop.py:menu`) through the fixed prompt
`physmem.models.sys12.SYS2_INSTRUCTION` / `sys2_messages`, and picks one call by constrained log-likelihood over the
handoff lines `<sys1> skill=<skill> target=<entity>` (`Sys2Router.choose`; a decision with a single allowed call is taken without an LM forward and logged as forced, sys2_ms = 0); the chosen line is parsed
(`parse_handoff`) and handed to System I. System I sets the continuous params of PICK calls on lot parts / the pouch
(grip force, lift speed, and probe-first-or-not; place / reach calls keep their scripted params) with ONE rule for both
variants: the MODEL-V2 `System1Head.propose` scoring (candidates = the scripted call, 48 jitters of its scalar params,
and the grid grip x lift: bin {3,5,7,10,14,20,28,40} N x {0.2,0.1,0.05} m/s; pouch {2,4,6,8,12,16,24,32} N x
{0.2,0.1,0.05}; score = P(slip)+P(drop)+P(damage)+0.01 x duration, ranked by mean + 1.0 x sd), then PROBE (ENVS
probe-then-act light squeeze) iff the part is unprobed and the best candidate's upper bound > tau = 0.15 (the r7 VAL
value, not re-tuned).
- **s12_native** — System II -> System I = Inkling-2-Mini core (pretrained) in fast mode: gated LoRA + typed-memory
  cross-attention, 1 core pass (of the native 3), checkpoint `results/sys12/checkpoints/s1r1_inkling_gated_s0.pt`
  (trained on the TRAIN split only, 4000 steps x 32 samples; frozen when its training finishes, before any test run).
  Single model: its sd = typed-posterior spread only (no ensemble).
- **s12_delegated** — System II -> System I = MODEL-V2 r7 typed 3-seed ensemble via `System1Head` (CPU).
- **s1native_scripted** — scripted skill choice -> System I native (isolates the System II contribution).
- baselines: **v2** = MODEL-V2 alone (scripted skill choice + the current best MODEL-V2 planner: crush-gated
  `LotPlanner` 'typed_v2' on bin lots, `ResidualPlanner` anchored on probe-then-act with the r6f trust gate on the
  pouch; r7 checkpoints, MODEL agent's frozen args); **pta** = ENVS probe-then-act scripted; **heuristic** = plain
  scripted (reference only).

**Cells** (envs-r6, site-dressed scenes, n = 30 paired tasks per cell, task seeds SEED_BASE[split] + 950000 + i,
i < 30 — a fresh seed range not used by any earlier round):
1. `bin_ur5e` — bin_clearing LOT protocol (3 parts from one lot, MODEL-V2 r7 `LotEnvAdapter`), ur5e_2f85, test split.
2. `pouch_ur5e` — pouch_transfer, ur5e_2f85, test split.
3. `bin_gen3` (secondary) — as 1 on gen3_2f85, heldout_robot split (robot never in training).

**Metrics.** Primary: bin cells — good parts per lot (in tote AND undamaged; `harness._good_parts_fraction`); pouch —
episode success (ENVS predicate). Secondary: damage rate, probes per episode, effort, steps, timeouts, System II
agreement with the scripted (skill, target), per-decision latency (System II ms, System I ms).

**Comparisons** (paired over tasks, mean difference with a 95% bootstrap CI over tasks, 2000 resamples,
`harness.bootstrap_ci`; "significant" = CI excludes 0; no multiplicity correction, every comparison is reported):
C1 s12_native − v2 · C2 s12_native − pta · C3 s12_delegated − v2 · C4 s12_delegated − pta ·
C5 s12_native − s12_delegated · C6 s12_native − s1native_scripted.

**Predictions (stated before testing).** C1 ≤ 0 (the Inkling System I saw ~6% of MODEL-V2's training samples and is a
single model; offline it is not expected to beat MODEL-V2); C3 ≈ 0 (same models, different decision rule); C6 ≈ 0
(the menus make the next call nearly determined, so System II should mostly agree with the scripted choice). A
positive C1 or C2 would be the headline; a null is the expected outcome and will be reported as such.

**Rules.** Timeouts / crashes inside an episode count as failures. No episode or cell is re-run selectively; if the
script itself crashes, the whole cell is re-run from scratch with the same seeds and the crash is reported. When ENVS
freezes envs-r7, the same protocol is re-run there as a labelled replication (new section). Code at registration:
sha256 of `scripts/sys12/closed_loop.py`, `src/physmem/models/sys12.py`, `src/physmem/models/inkling_quant.py`,
`src/physmem/models/v2/system1.py` listed below.

```
3880f26ee941e61128f813f57dac7511ace9a1e6712fca75e8a409ce648f6d38  scripts/sys12/closed_loop.py
ca14dae0fa396d3ba6a819949e284c23d742de74185355c4dd5e391d24f52f7e  src/physmem/models/sys12.py
74f95f32094054f46263248ca0d84dfda1d882ffd8b858d3b094b291d0193d01  src/physmem/models/inkling_quant.py
6a7366b8a31d7b920a0971879538bc41d8cb38753b1c8aa77db0150cf1c8b3d8  src/physmem/models/v2/system1.py
7b8817707ca87e1c668bd1815b86544cfa59a718355b512879c907d0df970343  src/physmem/models/v2/lot.py
6914fb0777a3f3466181ac2dbe43e6307273a53c789413628309133ce37821de  src/physmem/models/v2/harness.py
3ab4d0b411dd718aaea414a87b3ee6579e143a7fe5f3a8556f1b43db410e322d  src/physmem/sys12_train.py
```
(sha256 of the files on gb10-direct at registration; the System I checkpoint is added to section c when its
training finishes, before the first test episode.)

## a1. Erratum to section a (written 2026-11-26, after registration, BEFORE any test-split episode)

**What happened.** At registration the shared remote tree `~/dynagraph` on gb10-direct held uncommitted
**envs-r7-dev** code (changed grasp / grip skills, 6 cm bins, `ENV_VERSION = "envs-r7-dev"`), synced in from the
shared local tree. Section a's hashes were taken from that remote tree. Checked against git HEAD and the frozen
`~/dynagraph_r6`:
- `src/physmem/models/v2/harness.py` — the hash in section a (`6914fb07…21de`) is the **contaminated** envs-r7-dev
  version. The correct (git HEAD / envs-r6, the version MODEL-V2's r7/r8 tables ran with) hash is
  `827545b3a4c819c48ff15ec0331eb17c390b68c48fe87f208a87016f611a09dc`.
- Section a listed no envs/ or skills/ files, but the episodes would have imported them from that tree. The other six
  hashed files (closed_loop.py, sys12.py, inkling_quant.py, system1.py, lot.py, sys12_train.py) are not env code and
  are unchanged; lot.py equals git HEAD.
- The 2-task VAL smoke runs mentioned in section a ran on envs-r7-dev. They were debug-only and remain unused.

**Correction (protocol otherwise unchanged).** Every closed-loop cell runs from a frozen tree `~/dynagraph_sys12` =
`git archive` of HEAD (commit 3364798; its envs/, skills/, embodiments/ are byte-identical to `~/dynagraph_r6`, 30/30
files) + the sys12 overlay (the six registered files with their registered hashes) + `results/`, `ckpt/`, `data/`
symlinked; source files read-only. Each cell is launched through `scripts/sys12/cl_guard.py` (sha256
`b810e8ea…1174`), which aborts unless physmem is imported from the frozen tree, `physmem.envs.ENV_VERSION ==
"envs-r6"`, every envs/ / skills/ / embodiments/ file matches `~/dynagraph_r6` (combined sha256 of the sorted per-file
hash list: `30cb4b7fdad50b7850ee53150416492d629828fb155c5f0ce77a217bd3a076c1`, identical for both trees), and the
registered files keep their registered hashes. It writes `results/sys12/closed_loop/provenance_s12r1_<cell>.json`.
The System I checkpoint hash is written to `results/sys12/closed_loop/s1_ckpt_sha256.txt` before the first cell.

**Training is unaffected.** System I training reads tensorized npz data (`data/v2_r7_final`, generated earlier on
envs-r6) and imports only models/v2/{model,physics,spec,tensorize}.py, v2_train.py, planner.py, models/inkling.py and
the sys12 files. All of these are byte-identical to git HEAD on the remote tree (checked file by file). The gated arm
started in `~/dynagraph`; the remaining arms and every evaluation run from the frozen tree.
