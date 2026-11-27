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

## b. Design (what was built)

**Fit / memory (deliverable: ≤ 25 GB resident, no bf16 dequant).** `src/physmem/models/inkling_quant.py`
- The transformers 5.22 loader dequantizes the NVFP4 (layers 0-19) / MXFP8 (20-23) experts to bf16 (52.8 GB). We
  intercept its `Inkling2LoopDequantizeExperts` op: the packed tensors are kept on the GPU in their native widths
  (uint8 E2M1 codes + bf16-exact copies of the float8 block scales + per-expert amax/denominator factor; float8 values +
  exact 2^(s-127) scales), each layer's experts module is replaced by `QuantExperts`, which dequantizes ONLY the
  experts a batch routes to, on the fly, with fused Triton kernels (system CUDA-13 ptxas; the bundled one lacks
  sm_121a) that reproduce the reference arithmetic exactly (fp32 value x scale x factor, then bf16).
- Two loader behaviours caused the earlier 53 GB peak and were disabled: the async thread pool (materialises every
  bf16-cast expert tensor up front) and `caching_allocator_warmup` (pre-allocates the full EXPECTED bf16 model size in
  one block that stays reserved).
- Result: 17.3 GB packed experts + 2.3 GB dense = **20.6 GB allocated, 22.2 GB peak, 22.4 GB reserved after load**
  (`results/sys12/fit_check.json`, `logs`), vs 52.8 GB for the v1 bf16 path. Training System I (bs 32, 1 core pass,
  activation checkpointing) stays at ~25 GB.
- Exactness (`results/sys12/fit_check.json`): Triton and torch dequant are **bit-identical** to what the transformers
  loader produces (layers 3 NVFP4 and 21 MXFP8, both weight tensors), and a full `QuantExperts` forward is
  bit-identical to the native expert loop with the bf16-dequantized weights. Note: the transformers loader itself casts
  the fp32 amax to bf16 before dequantizing (max 1e-3 abs deviation from an fp32-amax dequant); we reproduce the
  loader, i.e. v1's bf16 model, not the fp32-amax variant.
- Speed: the native per-expert Python loop launches ~1.3 k kernels per layer. System I uses an opt-in FAST path
  (sort by expert + two `torch._grouped_mm` over the hit experts; same weights, fp32 accumulation order differs, so NOT
  bit-identical); System II / base / language evals always use the exact native loop.

**Mode tokens and gating.** `src/physmem/models/sys12.py`
- `<sys2>` = System II = the base LM with every adapter off; it is a router-level token (never embedded: an untrained
  embedding in the stream would itself change the base computation). `<sys1>` = a trained embedding row appended,
  with one ACTION token (skill one-hot of System II's choice) and 16 readout tokens, after the System II context
  (task text). Handoff text: `<sys1> skill=<skill> target=<entity>` (`parse_handoff` / `format_handoff`).
- System I runs the pre-loop layers 0-5 and **1 of the 3 native core passes** over layers 6-17 (post-loop layers and
  the LM head skipped). Typed memory enters only through per-pass zero-gated set cross-attention: memory -> the
  System I positions at pass start (READ), System I positions -> memory at pass end (WRITE).
- `GatedLoRALinear` (r 8, alpha 32 on q/k/v/o of core layers 6-17, 48 modules) has three modes: `off` (System II),
  `gated` (System I: active only at positions >= `<sys1>`, so every System II prefix position keeps its base state =
  KV reuse), `always` (the UNGATED control: on at every position in every mode).
- System I readout -> ctx (d = 128) -> the MODEL-V2 FiLM outcome head (physics margins, typed-slot values), so every
  MODEL-V2 planner and the MODEL agent's `System1Head` run on top unchanged. Memory tokens = the frozen r7 MODEL-V2
  typed s0 memory front end: target / embodiment / 8 slot-attention entity tokens + 11 typed-latent tokens
  (posterior mean, log-variance, evidence count). The outcome head is trained from scratch in every arm.
- Outputs: per candidate call, P(success / slip / drop / jam / collision / stall / damage), continuous outcomes and
  effort with heteroscedastic variances; uncertainty = head variance + typed-posterior sampling.

**Training (deliverable 3).** `src/physmem/sys12_train.py`: `data/v2_r7_final` TRAIN split (drawer_open = held-out
family absent from train/val; Gen3 = held-out robot only in `heldout_robot`), family-homogeneous round-robin batches,
robot-balanced sampling, history dropout 0.15, teacher-forcing curriculum — the r7 recipe. 4000 steps x 32 = 128 k
samples per arm, identical sample sequence across arms (seed 0). AdamW one-cycle: head + memory tokens 1e-3, Inkling
adapters 3e-4, zero-init gates 3e-3. Trainable: 5.54 M (Inkling arms). Resumable partial checkpoints every 200 steps.

**Controls (deliverable 4).** (i) `inkling_ungated` — same adapters, LoRA always on (baseline fine-tuning); (ii)
`inkling_random` — same architecture, layers 0-17 shuffled element-wise (dense) / code- and scale-wise within each
expert (packed), embedding rows permuted (matched weight statistics, destroyed structure); (iii-a) `compact` — a
2-layer d = 128 transformer System I over the same memory tokens, same samples; (iii-b) MODEL-V2 typed trained
end-to-end with the same number of samples (500 x 256); plus MODEL-V2 r7 typed s0 (2.05 M samples) as reference.

**System II (deliverable 1 handoff, 6).** `Sys2Router`: chat prompt (reasoning effort "none", answer channel
prefilled), `choose` = constrained log-likelihood over the allowed handoff lines, `generate` = free-form greedy +
parse. Both always run with adapters off.

## c. Results

All tables are generated by `scripts/sys12/report.py` into `results/sys12/tables.md`. Absolute numbers, single
training seed (0) per arm unless stated. All arms saw the identical 128,032-sample sequence.

### c1. Fit / memory (`results/sys12/fit_check.json`)
20.6 GB allocated, 22.2 GB peak, vs 52.8 GB for the v1 bf16 path. Dequant is bit-identical to the transformers
loader (NVFP4 + MXFP8), and so is the full expert forward. Training System I held about 26.7 GB.

### c2. Gate (deliverable 1)
| check (real checkpoint, trained gated adapters, `gate_test_trained.json` / `gate_test_rowwise.json`) | result |
|---|---|
| instrumented forward == native forward | bit-identical |
| System II logits with System I machinery + trained adapters installed vs base | **bit-identical (max diff 0.0)** |
| System II greedy decode vs base | identical |
| ungated LoRA vs base | max logit diff 32.1, mean KL 0.99 |
| System I call: prefix (System II) core states vs base, same sequence, native batched experts | NOT bit-identical (max 5.0 at 1 pass, 10.0 at 3) |
| same, shape-invariant row-wise experts (bin_clearing + pouch_transfer, 1 and 3 passes) | **bit-identical (max 0.0, 4/4)** |
| control: base full sequence vs base prefix-only, row-wise | differs (3.0-12.3): attention GEMM shapes differ |
Reading: the position gate leaks no information into System II positions. The native batched MoE kernels change
rounding when extra rows are routed to an expert, so KV reuse would be exact only with shape-invariant kernels.
Latency on the shared GPU: one System I call takes 305 ms with 1 core pass, 671 ms with 3 (fast experts, idle GPU;
gate_test.json). Under contention it was 608 ms / 1.34 s. The exact expert path at 1 pass took 1.84 s.

### c3. Offline System I (test split; <= 512 rows per family, 16 families, rows identical across the 4 arms)
| arm | success AUC | slip AUC | damage AUC | drop AUC | cont NLL |
|---|---|---|---|---|---|
| Inkling-2-Mini core (pretrained), gated, 1 pass | 0.844 | 0.879 | 0.730 | 0.562 | -1.15 |
| ungated LoRA (control i) | 0.847 | 0.887 | 0.702 | 0.510 | -0.98 |
| Inkling random-init core (control ii) | 0.840 | 0.862 | 0.717 | 0.631 | -0.80 |
| compact System I (control iii-a) | 0.826 | 0.885 | 0.711 | 0.538 | -1.35 |
The held-out family drawer_open has damage AUC 0.51-0.54 for all arms (chance). The MODEL-V2 references on these
test rows are being re-evaluated: the first run used different rows for 11/16 families (my `family_id`
load-order bug; `*_misaligned.json` kept, `scripts/sys12/eval_v2_aligned.py` asserts row identity).

Held-out ROBOT (Gen3, aux1 masked, 7,009 rows, ONE process so rows identical for every model; measured with the Gen3
base/shoulder self-collision defect, fixed in envs-r7.1):
| model | success AUC | slip AUC | damage AUC | drop AUC |
|---|---|---|---|---|
| Inkling pretrained gated | 0.809 | 0.830 | 0.707 | 0.594 |
| ungated | 0.835 | 0.829 | 0.684 | 0.605 |
| Inkling random-init | 0.813 | 0.868 | 0.697 | 0.589 |
| compact | 0.807 | 0.804 | 0.689 | 0.612 |
| MODEL-V2 r7 typed s0 (2.05 M samples, full end-to-end) | 0.820 | 0.858 | **0.778** | 0.606 |
| MODEL-V2 typed, sample-matched (128 k) | 0.735 | 0.715 | 0.492 | 0.582 |
**Verdicts.** (1) No pretraining benefit shown: pretrained vs random-init core differ by <= 0.04 AUC in both
directions (single seed, no CI). (2) The compact System I is as good as either Inkling arm. (3) On damage, the full
MODEL-V2 r7 is better than every System I arm. (4) Sample-matched MODEL-V2 trained end-to-end is much worse; the
System I arms are not compute-matched to it, because they reuse r7's frozen memory front end.

### c4. Language retention (deliverable 5; `results/sys12/lang_eval.json`)
| condition | wikitext-2 test ppl (24,576 tok) | KL(base ‖ cond), top-64 | task->skill constrained acc (60) |
|---|---|---|---|
| base | 8.337 | 0 | 0.83 |
| gated System I installed, System II mode | **8.337** | **0** | **0.83** |
| ungated LoRA | **52.8** | 1.38 | 0.45 |
**Headline confirmed:** gated System I training leaves System II unchanged, and ungated training degrades it badly
(ppl x6.3, degenerate outputs such as "carton3 carton3 ..."). Free-form generation scores in that file are INVALID:
the 24-32 token budget ended inside the model's thinking span. The 256-token rerun is in flight (section s).

### c5. Pre-registered closed loop (deliverable 6; envs-r6 frozen tree, guard passed; n = 30 paired tasks)
| cell (primary) | heuristic | PTA | MODEL-V2 alone | s12_native | s1native_scripted | s12_delegated |
|---|---|---|---|---|---|---|
| bin_ur5e (good parts / lot) | 0.16 | 0.30 | 0.21 | 0.18 | 0.20 | 0.23 |
| pouch_ur5e (success) | 0.20 | 0.57 | 0.57 | 0.30 | 0.30 | 0.40 |
| bin_gen3 held-out robot* (good parts / lot) | 0.12 | 0.22 | 0.22 | 0.21 | 0.20 | 0.30 |
*measured with the Gen3 base/shoulder self-collision defect (fixed in envs-r7.1).

Comparisons (Δ primary [95% bootstrap CI]; bold = CI excludes 0):
| | bin_ur5e | pouch_ur5e | bin_gen3 |
|---|---|---|---|
| C1 native − v2 | -0.03 [-0.14,+0.10] | **-0.27 [-0.43,-0.13]** | +0.00 [-0.12,+0.13] |
| C2 native − PTA | -0.13 [-0.27,+0.01] | **-0.27 [-0.43,-0.13]** | +0.00 [-0.10,+0.09] |
| C3 delegated − v2 | +0.01 [-0.07,+0.10] | **-0.17 [-0.30,-0.03]** | +0.08 [-0.02,+0.18] |
| C4 delegated − PTA | -0.07 [-0.21,+0.05] | **-0.17 [-0.30,-0.03]** | +0.07 [-0.01,+0.17] |
| C5 native − delegated | -0.05 [-0.14,+0.06] | -0.10 [-0.27,+0.07] | **-0.095 [-0.19,-0.01]** |
| C6 native − scripted skill choice | -0.02 [-0.07,+0.02] | 0 (all forced) | +0.01 [+0.00,+0.04] |
Damage vs MODEL-V2: native +0.20 [+0.07,+0.37] (bin_ur5e), +0.37 [+0.20,+0.53] (gen3); delegated +0.13 / +0.30
(both CIs exclude 0).

**Verdicts.** The prediction stated before testing held: no System I/II variant beats MODEL-V2 alone or PTA, and on
the pouch both lose significantly. Mechanism: the shared pre-registered System I rule (System1Head + UCB > 0.15)
almost never probes; native never probed, while MODEL-V2's crush-gated LotPlanner and PTA do. Because delegated also
loses with MODEL-V2's own models, the decision rule, not only the Inkling model, is responsible. The exploratory
rule-vs-model diagnostic (section s) isolates this. System II (Inkling base LM) agreed with the scripted
(skill, target) on 65-77% of non-forced decisions, with no measurable effect (C6). Its latency per LM decision under
shared-GPU contention was 8.5 s (bin_ur5e) and 33 s (bin_gen3). Delegated-policy mean effort is dominated by one
simulator blow-up episode (5e12); medians are in tables.md.

## s. Status at 2026-11-27 (in flight)

Done and committed: everything in sections a-c above (code, tables, small result files; checkpoints stay on
gb10-direct in `~/dynagraph/results/sys12/checkpoints/`, 415 MB, sha256 of the closed-loop System I checkpoint in
`results/sys12/closed_loop/s1_ckpt_sha256.txt`).

Still running on gb10-direct (frozen tree `~/dynagraph_sys12`, one niced job at a time under `.gpu_heavy.lock`):
| step | process | log | output |
|---|---|---|---|
| free-form language eval, 256-token budget (base, ungated, 12-item gated identity check) | `scripts/sys12/pipeline4.sh` (pid 2339725; python pid 2923100) | `~/dynagraph/logs/sys12/lang_eval256.log` | `results/sys12/lang_eval_gen256.json` |
| MODEL-V2 references on the arms' exact rows (asserts identity) | `pipeline5.sh` (pid 2451747) | `logs/sys12/evalv2_aligned.log` | `results/sys12/eval/v2_*_typed_s0.json` (old: `*_misaligned.json`) |
| EXPLORATORY rule-vs-model diagnostic: MODEL-V2's own rules with pta / v2_ens3 / v2_s0 / native System I, fresh seeds +960000, n = 30, bin_ur5e + pouch_ur5e | `pipeline6.sh` (pid 2867454) | `logs/sys12/explore_{bin_ur5e,pouch_ur5e}.log` | `results/sys12/closed_loop/explore1_*.json` |
| GPU turn-taking with other agents | `scripts/sys12/gpu_share.sh` (pid 158722) | `logs/sys12/gpu_share.log` | stop: `rm ~/dynagraph/.sys12_share_on` |
Each pipeline waits for the previous one; all are idempotent and resumable (re-run the same script; finished
outputs are skipped). The envs-r7 replication (section a rules) is pending ENVS's freeze; MODEL-V2's r9 checkpoints
will need a new System I memory front end.

To collect and finish (from the local checkout):
```bash
ssh gb10-direct 'tail -2 ~/dynagraph/logs/sys12/pipeline{4,5,6}.log'        # *_DONE lines when finished
ssh gb10-direct 'cd ~/dynagraph_sys12 && PYTHONPATH=src ~/dynagraph/.venv/bin/python scripts/sys12/report.py > /dev/null'
rsync -az --exclude '*.pt' --exclude 'checkpoints/' gb10-direct:~/dynagraph/results/sys12/ results/sys12/
# then: update sections c3 (aligned MODEL-V2 test rows), c4 (256-token free-form), add c6 (exploratory) from tables.md
ssh gb10-direct 'rm ~/dynagraph/.sys12_share_on'                            # stop turn-taking when nothing is queued
```
Reproduce from scratch: `scripts/sys12/pipeline.sh` (training, needs the frozen tree for the closed loop; see a1),
then `pipeline2.sh` ... `pipeline6.sh`. Unit tests: `python -m pytest -q tests/test_sys12.py` (CUDA).

**Gen3 caveat (applies to every Gen3 number here).** Measured with the Gen3 base/shoulder self-collision defect
(fixed in envs-r7.1). The envs-r6 / r7 Gen3 robot model has a persistent base_link vs shoulder_link self-collision
(~1.8 MN, 12 mm penetration). It produces the constant out-of-range `aux1` (max robot contact / 150) that made the
v2 sanitize drop every Gen3 row; our Gen3 offline eval masks `aux1`. Success and damage are measured correctly, but
the arm may be handicapped.
