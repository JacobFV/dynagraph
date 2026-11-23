# findings — gb10 typed physical-memory sprint (2026-11-23, 04:45–16:45Z)

exploratory 12-hour study on one nvidia gb10 (gb10-direct / promaxgb10-4dfb, cc 12.1, ~117 GB available unified memory).
every number below is traceable to `results/metrics.jsonl`, `results/ledger.jsonl`, `results/inkling_gate.json`,
`results/g1/**` or `results/reproduction.json`. nothing here was run on a physical robot.

## what was built

- typed, versioned, causal property-graph memory (`src/physmem/memory.py`): immutable evidence events, explicit
  revisions, causal availability (late packets buffered), duplicate suppression, same-source correlation (no
  confidence inflation), authoritative correction, retraction/deletion removing all readable views, model-written
  estimates with provenance, privileged-label rejection at the actor boundary. units/frames in `frames.py`.
- procedural mujoco 4.0.1 worlds (w1 impulses/springs, w2 pushing with hidden friction/mass, w3 second box,
  moving agent with latent modes, chains of length 1/2/4/8) and a split-safe tensorizer (`worlds.py`, `data.py`).
  primary dataset v2: 4096/512/512 roots × 4 pushes, 21+ held-out strata, sha256 in `data/v2/manifest.json`.
- compact recurrent physical-memory models (1.33 m params; generic vs structured within 0.6%), physics losses,
  analytic/oracle baselines, the 2×2 factorial and diagnostics (`models/compact.py`, `losses.py`, `train.py`, `evaluate.py`).
- **actual pretrained inkling-2-mini core** (`models/inkling.py`): pinned `thinkingmachines/Inkling-2-Mini@b85c3cf`
  (sha256 of weights verified), transformers 5.24 dequantizing nvfp4/mxfp8 experts to bf16 (52.8 GB resident).
  a step-by-step instrumented replica of the native looped forward with hooks at each of the 3 core passes;
  unordered memory slots enter only through per-pass set cross-attention (read at pass start, write back at pass
  end, zero-gated) to 16 readout tokens + 1 action token; lora r8/α32 on q/k/v/o of core layers 6–17 (48 modules);
  readout at the core exit.
- unitree g1 (menagerie f3b6013, with hands, pelvis welded) fixed-base mujoco scene, dls ik, push primitive matching
  the training action contract, synthetic sensors through the same event/tensor path, a shared candidate selector,
  integrity audit (no object writes, welds or applied forces), paired closed-loop runner, labelled demo video.

## inkling gate (spec §4.2) — passed with one documented amendment

1 checkpoint/revision + coherent greedy decode ✓ · 2 per-pass core-exit capture (relative change 0.56, 1.19) ✓ ·
3 instrumented == native, max |diff| = 0.0; zero-gated adapters + lora preserve the core exactly ✓ ·
4 physical loss → finite non-zero grads in read/write adapters and all 48 lora modules, pass-0 read gradient flows
through passes 1–2, base grads none ✓ · 5 optimizer step changes adapters only (25 frozen tensors checksummed) ✓ ·
6 tiny overfit: **failed** the original 40-step criterion (nll 1.37→1.00); amended 200-step check passed (1.37→−0.23),
both recorded · 7 load 48 s, 52.8 GB cuda after load, peak 74 GB training at bs16, 1.28 s/step at bs8 ✓ · 8 budget ✓.

## main results

### compact 2×2 factorial (v2 test, 512 roots, 3 seeds; 0.5 s position error)

| model | pos err @0.5 s | nll |
|---|---|---|
| b00 generic | 14.9 mm | −1.38 |
| b10 structured | 12.7 mm | −1.29 |
| b01 generic + physics | 16.6 mm | −1.38 |
| b11 structured + physics | 14.0 mm | −1.22 |
| coulomb analytic / oracle (labelled) | 40.7 / 40.8 mm | — |
| constant velocity | 84.2 mm | — |

structure −2.44 mm [−2.77, −2.12]; physics loss **+1.52 mm** [+1.26, +1.79] (worse); interaction −0.46 mm. nll effects not significant.

### where structure helped and where it lost
- helped: every occlusion / delay / drop / rate stratum (2.3–2.7 mm), misspecified laws (−1.8 / −2.0 mm), 10% data (−2.3 mm).
- lost: chains of ≥2 boxes (+1.4 mm at L2, +7.3 mm at L4), distractor entity (+2.6–3.8 mm), ood friction 0.15 and
  paddle pusher (nll explodes: +140, +96), 1% data (no effect), deleting derived velocity events removes its advantage.
- post-hoc v3 (fast pushes): structure gain shrinks to −0.8 mm; physics loss still +2.0 mm.

### causal tests
- friction slot intervention 0.35→0.65: compact 0.07–0.10 mm vs simulator 9.9 mm (v2), 0.08–0.13 mm vs 25.6 mm (v3);
  inkling i11 0.4 mm. in the g1 loop the overwritten slot changed the predicted final position by 0.02 cm and did not
  change the chosen action or outcome. **the typed friction slot is not causally used.** diagnosis: the slot collapsed
  to the prior (0.50 ± 0.003) even though friction supervision dominates its gradient.
- friction token lesion costs 4–6 mm vs 1.4–1.8 mm random → the token carries non-friction information. edge lesions: 0.
- post-hoc friction-pathway fix (b11p, explicit per-push summarizer fused into the typed slot, +2.3% params): slot
  still at prior level (error 0.092 v2 / 0.088 v3 at every push index), intervention effect <1% of the simulator's.
  the bottleneck is extracting friction from observations, not routing (our analytic estimator also stays at prior level).
- recurrent passes 1/2/4/8: no difference anywhere, including chains.
- e12: equal-information re-encoding (rotated/translated sensor frame, millimetres, converted back) → outputs change by
  exactly 0 for all 12 models (pipeline round trip, not learned invariance). sensor removal: generic models ignore contact,
  command, pusher pose and size (<0.01 mm change); structured models use them (+0.3–1.8 mm; size +1.2–1.7 mm) and become
  overconfident (80% coverage 0.88→0.81).
- uncertainty: friction uncertainty is flat across push index; structured models are overconfident with no history.

### actual inkling core (e08; v2 test; 2 seeds; ~11–12k training examples per 35-min run)

| model | pos err @0.5 s (s0 / s1) | nll (s0 / s1) |
|---|---|---|
| i00 generic memory | 28.9 / 30.2 mm | −0.20 / −0.19 |
| i11 typed memory + physics | 32.3 / 32.7 mm | −0.31 / −0.42 |
| i11 trained without core lora (s0) | 33.9 mm | −0.35 |
| core bypass (constant message) | 53–61 mm | 0.9–7.4 |

the trained models use the pretrained core (bypass roughly doubles error), i11 has better likelihood but worse
position than i00, and lora adaptation of the core adds ~1 mm at most (single-seed control, inconclusive). compute is
**not** matched with the compact models (≈1% of their training examples), so inkling-vs-compact differences mostly
reflect budget; no random-init control was run, so a pretraining benefit is untested.

### unitree g1 (fixed-base simulation, synthetic sensors, push task)
- lift/place gate failed (3/10) → push locked as the endpoint. slide-to-target (15–30 cm, ≤3 pushes) requires fast
  pushes (slow-only policy 4/30).
- paired closed loop: stub (prior μ=0.5) 14/30, analytic 8/30, oracle (true μ, labelled) 15/30;
  compact b00 22/50, b11 15/50 vs stub 22/50, oracle 23/50; inkling i00 8/30, i11 6/30 (sim paused ≈2.3 s/decision).
- **no learned model beat the prior stub.** even true friction barely helps: g1 hand speed scatter (0.83–1.16×) and
  the embodiment gap dominate. a bounded controller-v2 attempt reduced fast-push error (3.2→2.0 cm) but missed its
  adoption bar.
- e15 transfer: zero-shot compact models on g1 data are badly overconfident (nll 33–59); fine-tuning on 2,251 g1
  samples (calibration seeds) fixes calibration (80% coverage 0.94–0.97) but improves position only ~3 mm.
  closed-loop with fine-tuned models (50 paired seeds): b11 20/50, b00 15/50, stub 22/50, oracle 23/50; per-seed b11
  final error −2.20 cm vs b00 [−3.76, −0.81] (b11 better on 31, tie 5, worse 14). adaptation helped b11 (15→20) and hurt
  b00 (22→15). this is the only place a structured-memory advantage appeared in a robot decision, and it required g1 adapter data.
- demo: `results/demo.mp4` (110.7 s), representative fixed seed 29, paired b11/b00, friction intervention, inkling decision log.

## hypotheses
h1 supported (partially: prediction under sparse/occluded evidence; learned stale-write handling not distinguishable) ·
h2 contradicted in the tested regime · h3 unsupported · h4 contradicted for friction/mass slots · h5 unsupported ·
h6 unsupported (participation shown; added capability not) · h7 contradicted. details and evidence filters: `results/verdicts.json`.

## invariances
exact by construction: duplicate/replay invariance, opaque-id renaming, event-order permutation (record layer),
time-origin shift, unit conversion, frame round trips, quaternion sign; exact permutation equivariance of the
structured encoder (float64 test); canonical target-frame labels are se(2) equivariant (2e-5). approximately
observed: world-transform re-simulation (≤1.4 mm). interface promises only: sensor modality, backbone width,
simulator backend, multi-agent roles.

## reproduction
`results/reproduction.json`: compact-main-b11-s1 re-evaluated from its checkpoint in a fresh cpu process reproduces
logged test metrics exactly (relative diff 0.0). entry points: `physmem doctor|generate|train|evaluate|demo|report`,
`scripts/inkling_gate.py`, `python -m physmem.models.inkling`, `scripts/inkling_eval.py`, `scripts/demo_g1.py`.

## ranked next experiments
1. friction extraction: a per-push summarizer routed into the slot was not enough (b11p). next: supervise per-push
   release speed / slide distance explicitly, longer training, or a differentiable coulomb slide fit inside the encoder —
   the information exists (ground-truth release speed + slide distance → friction corr 0.93).
2. close the g1 execution gap (closed-loop velocity control or learned speed adapter) before testing model-driven decisions.
3. compute-matched inkling vs compact, plus a random-init core control, to separate pretraining from architecture/scale.
4. variable-loop inkling checkpoint for a real loop-count study.
5. physics-loss weighting sweep and a law-selection mechanism (the fixed regularizer only hurt here).
