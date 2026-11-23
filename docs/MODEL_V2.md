# MODEL_V2 — typed physical-memory world model, planner, and data contract (owner: agent MODEL-V2)

Status: DRAFT data contract published first (2026-11-23 ~17:45Z) so ENVS/ROBOTS can log what the model needs.
Results sections are filled in at the end of the doc as they are measured.

Code: `src/physmem/models/v2/` (spec, synth, tensorize, model, harness), `src/physmem/planner.py`,
`src/physmem/v2_train.py`, tests `tests/test_v2_model.py`.

## 1. What ENVS / ROBOTS must log (requests)

Import the pure-numpy contract: `from physmem.models.v2 import spec`.

### 1a. per executed SkillCall: a dense actor-legal time-series segment (MOST IMPORTANT)
v1 failed to learn latent physics because evidence was a few noisy pose samples. v2 system identification reads the
raw interaction signal. In every skill execution please do:

```python
rec = spec.SegmentRecorder(target_pos0)   # target's ACTOR-ESTIMATED world position at skill start
# every control tick (>= 50 Hz; raw rate is fine, we resample to 64 samples):
rec.tick(t, cmd_pos, cmd_vel, ee_pos, ee_vel, wrench6, grip_cmd, grip_pos, obj_pos_or_None, contact)
result.log["segment"] = rec.finish()      # SkillResult.log
```
channels (`spec.CHANNELS`, 26): commanded ee pos/vel (task-space trajectory reference), measured ee pos/vel (FK of
encoders), **wrist F/T** (force world axes, tool/hand gravity compensated so free motion ≈ 0; torque), gripper
command + measured opening (suction: pressure), target pose observation from the noisy tracker (pass `None` when no
fresh/occluded sample — we keep a validity channel), actor-legal contact flag (finger/tactile or |F|>threshold), time.
For bimanual robots log the effector that contacts the target (or sum wrenches, and say so in ENVS.md).
Positions are made relative to `target_pos0` by the recorder. **No simulator truth in these channels**
(use the F/T sensor site, encoders and the noisy pose tracker, never `data.xpos` of the object).

**UPDATE 18:10Z — ENVS does NOT need to call the recorder.** ENVS already emits 25 Hz WRENCH / EE_POSE / GRIPPER /
TACTILE events and 10 Hz POSE3; `models/v2/env_adapter.py::build_segment` rebuilds the 26-channel segment from the
events measured inside each `execute()` window (verified on bin_clearing, singulation, bulk_sweep with the probe
robot: 75–150 raw rows per skill, 0 collection failures). Remaining small requests to ENVS (all optional):
1. put the commanded task-space reference in `result.log["cmd"] = (t, pos3, vel3)` arrays (cmd channels are 0 now);
2. `probe_skills(belief)` per family (heft / tap / touch / short pull), else VOI uses the env's own candidates;
3. keep `metrics` cumulative counters (`drops`, `collisions`) as now; add `slip` / `jam` flags where meaningful;
4. labels: `hidden[eid]` keys `mass, mu, mu_grip, clearance, damping, frictionloss` are mapped to spec latents
   (`env_adapter.HIDDEN_MAP`); add `stiffness` in N/m if compliance matters (timeconst is not mapped);
   for bulk the adapter uses `labels["mu_mean"]` and the summed part mass;
5. candidate params that are world coordinates must use keys matching `x|y|z|x0|y1|*_x` (made target-relative).

### 1b. per step (StepOut)
- `result.effort` : `api.EffortMeter` (we train on `effort.cost()` and the components).
- `result.log["skill"], ["params"], ["target"]` (= the SkillCall), `result.log["segment"]` (above).
- `metrics` : `progress` (task-family scalar in [0,1], e.g. fraction of bulk moved, drawer opening / goal opening,
  insertion depth / full depth), `collision` (bool or count), `violation`, and actor-observable event flags when
  they apply: `slip`, `drop`, `jam`, `stall` (guard tripped / force limit), `success`.
- `labels` (EVALUATOR-ONLY): latent truths using the names below, SI units, for the target entity:

| latent (`spec.LATENTS`) | meaning | unit |
|---|---|---|
| `mass` | object mass | kg |
| `friction` | object↔support sliding μ | – |
| `grip_friction` | object↔finger μ | – |
| `stiffness` | contact/part stiffness | N/m |
| `clearance` | peg-hole radial clearance | m |
| `offset_x`, `offset_y` | true hole position minus the actor-perceived hole position | m |
| `damping` | articulation viscous damping | N·s/m |
| `breakaway` | articulation static breakaway force | N |
| `bulk_friction` | effective bulk/granular friction | – |

Only include what is defined for the family (missing = unsupervised). Labels never reach a forward pass: the tensorizer
puts them in a separate `Labels` object, and the `v2_train` loop asserts it.

### 1c. per decision: entity state
Either ObservationEvents (schema.py; POSE2/VEL2/SIZE/CONTACT, plus any 3-D pose props the lead adds — we read the
first 3 position components and up to 4 orientation components) or a `spec.EntityFrame` snapshot. Bulk piles:
one entity per tracked particle/part (tens–hundreds is fine; the model pools them with slot attention) **or** a
pooled-pile entity; mark particles with type `particle`. Include the goal region as an entity of type `goal_region`
if the goal is spatial (`env.goal()` must also expose it).

### 1d. env API needs from the planner
- `candidate_skills(belief, n, rng)` returns diverse feasible SkillCalls; the CEM planner refits a Gaussian over the
  scalar params of the sampled candidates per skill type, clipped to their observed [min,max] (so the candidate
  sampler implicitly defines the parameter bounds). Please keep each skill's param keys fixed within a family.
- optional `probe_skills(belief)` returning low-risk informative SkillCalls (e.g. heft = lift 1 cm; tap/slow push;
  touch-down at the hole; short pull). Used by value-of-information gating. If absent, the planner uses its own
  learned information head on regular candidates.
- `goal()` : actor-legal goal dict; the planner's task cost uses the model's predicted `success` probability and
  `progress`, so no family-specific cost code is needed.

## 2. Model (`models/v2/model.py`; typed 2.28 M params, generic within 10%, nomem has no segment encoder)

```
past interactions (<=6)                                     current entity state (N = 1..256 tokens)
 segment 64x26 - conv1d x3 (stride 4) - transformer x2 - attn-pool -+    entity MLP - slot attention (8 slots)
 + SkillCall vector (per-skill normalized params) - MLP ------------+    + target token + embodiment token
 + 14 analytic features - MLP --------------------------------------+    - transformer x2 - context c (d=128)
 + embodiment vector - MLP -----------------------------------------+                |
        | per-interaction Gaussian (mu_k, logvar_k) over 10 typed latents            |
        v                                                                            |
 precision fusion with learned prior p(z | target token, embodiment) and a learned   |
 equicorrelation correction: prec = p0 + sum_k prec_k / (1 + rho_l (n-1))            |
   -> TYPED SLOTS (mean, var) per latent: readable, overwritable (interventions),    |
      val-fit per-latent variance temperature                                        |
        v                                                                            v
 slot VALUE z  ----> FiLM-MLP head (3 residual layers; FiLM from [z, call] at every layer; z also concatenated)
                     <---- candidate SkillCall + context c
 outputs: 8 continuous outcomes (heteroscedastic), 6 event logits (success/slip/drop/jam/collision/stall),
          log effort (heteroscedastic), risk = 1 - prod(1 - p_e) over SUPERVISED risk events
```
- **Non-bypass by construction**: the scene path sees only current state; interaction histories reach the head only
  through the typed slots. `test_typed_slots_are_the_only_latent_pathway`: with all slots overwritten, predictions
  are identical with or without history (atol 1e-5) and change when the slots change.
- **Training** (`v2_loss`): sysID = per-interaction NLL + fused-posterior NLL + 0.5 prior NLL (evaluator labels;
  only the label loss trains the estimator — the outcome loss is detached from it). The head gets a *sample-form*
  slot input: per sample, labelled latents are the TRUE value with prob p_tf (0.75 -> 0.5 curriculum), otherwise a
  sample of the fused posterior, always presented as a known value — exactly what the planner feeds. Info head
  regresses the per-interaction log-variance a call produces (VOI). History dropout 0.15. AdamW 1e-3 one-cycle,
  bf16 autocast, family-homogeneous batches (round robin), GPU-resident data (index gathers only).
- **Throughput** (GB10, bf16, batch 512 samples x 6 history segments): 7.4-7.9k samples/s (~45k segments/s);
  15k steps ~ 17 min. Synthetic data: 240k interactions in 20-30 s on 6-8 processes.

## 3. Planner (`src/physmem/planner.py`)
CEM (pop 256, 4 iters, 10% elites) over the scalar skill params (one group per (skill, target) on real stations,
seeded with the env's candidates; bounds = candidate range). Cost per latent sample s:
`C_fail(1 - p_succ) + C_prog(1 - progress) + lam_e log(1 + effort/effort_ref) + lam_r p_risk + lam_t duration`,
score = mean_s + lam_unc std_s; effort_ref = median predicted effort of random feasible actions in the current state
(scale-free "servo pain"; a linear effort term let one painful action outweigh a failure — found in the val pilot).
Typed: 32 posterior samples of the slots; oracle: true latents in the slots (labelled); generic/nomem: 1 pass.
VOI: EVPI = E_s[c(a*, z_s)] - E_s[min_a c(a, z_s)] (min over the CEM population) x expected fraction of
decision-relevant slot variance a probe removes (info head; weights corr^2(z_l, regret of a*)) vs the probe's
predicted cost (effort + risk + time + step cost) -> probe iff positive (max 1 per episode).

## 4. Synthetic sanity world (`models/v2/synth.py`) — proof before real data
NumPy analytic dynamics in the exact data contract (26-ch segments with F/T noise + per-episode bias, tracking
lag, 20 Hz tracker with noise/dropout; EffortMeter; outcomes; evaluator labels). ~0.25 ms per skill.

| family (skill) | hidden latents | why they matter |
|---|---|---|
| slide (push) | mass, friction | post-release slide v^2/(2 mu g); stall if mu m g > F_max |
| lift (pick) | mass, grip friction, stiffness | drop iff m(g+a) > 2 mu_g F_grip; grip force is "painful" effort |
| insert (insert) | hole offset x/y, clearance, stiffness | capture iff misalignment < clearance + chamfer; chamfer lateral force reveals the direction; jam when fast and stiff |
| drawer (open) | damping, breakaway force | opening (F - 0.3 F_b)/b T; end-stop slam = collision |
| sweep (sweep) | bulk friction, pile mass; 20-200 particle entities | carry + slide of particles; spill past the band = drop |

Robots: 4 training (arm_a, arm_b, bimanual, humanoid: different force/speed/payload limits, F/T noise and bias,
tracking lag, tracker noise) + held-out `synth_mobile`. Splits by seed range: train / val / test, held-out robot,
stress latents (slide mu 0.07 / 1.0, outside the 0.1-0.9 training range). 8000 train episodes per family
(~48k interactions), 600 per eval split. Scripted heuristic baseline (`synth.heuristic_params`): nominal-property
inverse model with conservative margins (e.g. 60 N grip, slow approach pushes, 2 mm spiral search).

## 5. Synthetic-world results (r1 checkpoints; held-out TEST seeds unless noted; all rows in results/v2/metrics.jsonl)

History of runs (kept, nothing deleted): r0 = first typed run (head trained on posterior moments, risk included
unsupervised events) — latent recovery identical to r1 but the planner on top of it failed (see §7); r1 = fixed
(sample-form slot input, events n/a-in-family supervised as 0, supervised-only risk). Checkpoints
`results/v2/checkpoints/v2_synth_{r0,r1}_*`; eval JSON `results/v2/eval/`; curves `results/v2/curves/`.

### 5a. G4 latent recovery (typed r1; corr(posterior mean, truth) in normalized latent space after j interactions)
| family | latent | j=0 | j=1 | j=2 | j=3 | held-out robot `synth_mobile` j=1 / j=2 (zero-shot) |
|---|---|---|---|---|---|---|
| slide | mass | -0.01 | **0.985** | 0.995 | 0.996 | 0.99 / 1.00 |
| slide | friction | 0.01 | **0.934** | 0.967 | 0.974 | 0.95 / 0.98 |
| slide (stress mu 0.07/1.0, out of range) | friction | – | 0.97 | 0.99 | 0.99 | (r0 run) |
| lift | mass | 0.01 | **0.997** | 0.999 | 0.999 | 1.00 / 1.00 |
| lift | stiffness | 0.00 | **0.985** | 0.997 | 0.999 | 0.98 / 1.00 |
| lift | grip friction | 0.03 | 0.24 | 0.37 | 0.43 | 0.32 / 0.42 — **fails** (observable only through slip) |
| insert | stiffness | -0.02 | **0.886** | 0.970 | 0.987 | 0.92 / 0.97 |
| insert | offset x / y | 0.01 | 0.54 / 0.55 | 0.68 / 0.71 | **0.78 / 0.80** | 0.54 / 0.67-0.70 |
| insert | clearance | 0.01 | 0.23 | 0.32 | 0.36 | 0.24 / 0.30 — **fails** |
| drawer | damping | 0.01 | **0.869** | 0.952 | 0.986 | 0.86 / 0.96 |
| drawer | breakaway | 0.01 | **0.942** | 0.983 | 0.994 | 0.95 / 0.99 |
| sweep | pile mass | 0.08 | **0.937** | 0.961 | 0.969 | 0.94 / 0.96 |
| sweep | bulk friction | 0.05 | **0.723** | 0.815 | 0.848 | 0.74 / 0.82 |
Gate (corr >= 0.7 after 1-2 interactions): met for 10/14 family-latent pairs; misses: grip friction, clearance
(weakly identifiable by construction), hole offsets reach 0.7 only after 2-3 attempts. j=0 is at chance as
expected (no evidence; uncertainty at prior). Uncertainty shrinks with evidence (mean posterior std, e.g. slide
friction 0.24 -> 0.14, drawer damping 0.22 -> 0.05 normalized units from j=1 to j=3). Calibration before
temperature: mean z^2 1.0-1.5, 80% coverage 0.70-0.83 (slightly overconfident; correlated per-episode errors);
a per-latent variance temperature fit on VAL (1.2-1.9x) is stored in the checkpoint (`z_var_scale`).
v1 for contrast: the typed friction slot stayed at the prior (abs err 0.092 at every push index).

### 5b. G4 interventions: model slot finite difference vs simulator finite difference (r1, 150 random states/actions
each, one latent +-0.5 normalized units, other latents at truth; sim = mean over 16 noise seeds)
| family: latent -> outcome | n significant | sign agreement | corr | median magnitude ratio model/sim |
|---|---|---|---|---|
| slide: friction -> travel | 75 | **1.00** | 0.87 | 0.77 |
| lift: mass -> success | 31 | **1.00** | 0.89 | 0.92 |
| insert: offset_x / offset_y -> success | 65 / 54 | **0.98 / 1.00** | 0.90 / 0.88 | 0.59 / 0.54 |
| drawer: damping -> opening | 45 | **1.00** | 0.87 | 0.99 |
| drawer: breakaway -> opening | 38 | 0.89 | 0.89 | 0.35 |
| lift: grip friction -> success | 4 | 1.00 | 0.30 | 0.21 |
| insert: clearance -> success | 11 | 1.00 | 0.38 | 0.09 |
| sweep: bulk friction -> progress | 13 | **0.62** | 0.22 | 0.09 (fails) |
(latents with no simulator effect on that outcome, e.g. slide mass -> travel, have n_sig = 0; the model's spurious
response there is <= 3 mm / 0.03 probability.) v1 for contrast: slot intervention moved predictions by 0.07-0.13 mm
vs 9.9-25.6 mm in the simulator (<1%).
Closed-loop slot interventions (plan from the same state with the slot overwritten to posterior mean -/+ 1 sigma,
execute both plans in the simulator, 50 states): lift mass -> planned grip force +10.6 N, correct sign 100%,
realised success +8%; insert offset_x -> aim_x correct sign 100%; drawer damping -> realised opening +3.6 cm,
correct sign 100%; slide friction -> realised travel correct sign only 56% (weak); sweep: no action change.

### 5c. Outcome prediction (test; typed vs generic memory vs labelled oracle slots)
| family | success AUC typed / generic / oracle | progress RMSE typed / generic | effort log-RMSE typed / generic |
|---|---|---|---|
| slide | 0.912 / 0.846 / 0.916 | 0.163 / 0.197 | 0.25 / 0.24 |
| lift | 0.992 / 0.997 / 0.999 | 0.131 / 0.123 | 0.49 / 0.47 |
| insert | 0.850 / 0.799 / 0.991 | 0.383 / 0.414 | 0.91 / 0.96 |
| drawer | 0.842 / 0.823 / 0.845 | 0.231 / 0.240 | 0.51 / 0.61 |
| sweep | 0.920 / 0.906 / 0.920 | 0.247 / 0.267 | 0.21 / 0.20 |
Typed memory is better where latents decide outcomes and are identifiable (slide, insert, drawer); equal on lift
(generic memory also learns mass from the F/T stream — the typed advantage there is interpretability /
intervenability, not accuracy). Oracle >> typed on insert = the remaining value of better offset/clearance estimates.

### 5d. G3 closed loop on the synthetic world (paired: identical task seeds for every policy; 100 episodes per
family per split; 95% bootstrap CIs; cost weights c_prog = 2.0 with lam_effort in {0.03, 0.1}, selected on VAL
seeds before any test run; rows `experiment=v2_g3_closed_loop`, JSON `results/v2/closed_loop/`)

Single-item episodes (test, lam_effort 0.03 unless noted):
| family | heuristic succ (effort) | typed planner | generic-memory planner | no-memory planner | oracle (labelled) |
|---|---|---|---|---|---|
| lift | 0.95 [0.90,0.99] (21.9) | 0.99 (16.8; 11.8 @ lam_e 0.1) | 0.99 (14.7; 8.3) | 0.99 (14.5; 9.6) | 0.99 (17.2; 10.9) |
| lift, held-out robot | 0.99 (23.3) | 1.00 (23.0; 16.7) | 1.00 (19.6; 11.7) | 1.00 (19.8; 10.7) | 1.00 (25.1; 13.9) |
| insert | 0.95 [0.91,0.99] (0.049) | 1.00 (0.090 with VOI probe; 0.029 without) | 1.00 (0.031) | 1.00 (0.017) | 1.00 (0.065) |
| sweep | 1.00 (0.174) | 0.99 (0.204; 0.082 @ 0.1) | 1.00 (0.049) | 1.00 (0.051) | 1.00 (0.264) |
| slide | 0.91 [0.85,0.96] (3.33) | 0.78 [0.70,0.85] (3.45); **0.86** (2.53) @ lam_e 0.1 | 0.46 (11.0) | 0.90 (4.07) | 0.77 (3.13) |
| slide, held-out robot | 0.95 (3.29) | 0.82 (3.26); 0.90 (2.45) @ 0.1 | 0.52 (10.6) | 0.90 (3.48) | 0.79 (3.10) |
| drawer | 0.54 [0.44,0.64] (378) | 0.47 [0.37,0.57] (204) | **0.89 [0.83,0.95]** (253) | 0.50 (109) | 0.47 (165) |
| drawer, held-out robot | 0.62 (463) | 0.40 (328) | **0.78** (320) | 0.50 (274) | 0.34 (393) |

"Lot" protocol (4 items with the SAME hidden properties in sequence; evidence from earlier items is decision-relevant
for later ones), test, items completed / effort per item:
| family | heuristic | typed | generic | no-memory | oracle (labelled) |
|---|---|---|---|---|---|
| slide | 3.67 / 3.90 | 3.45 / 4.10 | 1.54 / 35.4 | **3.83** / 4.32 | 3.38 / 4.15 |
| lift | 3.80 / 23.1 (0.80 risk events) | 3.96 / 19.3 | 3.96 / **12.5** | 3.96 / 14.4 | 3.96 / 16.6 |
| insert | 3.80 / 0.051 | **4.00** / 0.060 | 4.00 / 0.030 | 4.00 / **0.017** | 4.00 / 0.061 |
| drawer | 3.02 / 511 | 1.11 / 842 (4.9 collisions) | **3.15 / 366** | 1.08 / 456 | 1.05 / 863 (4.9 collisions) |
| sweep | 4.00 / 0.175 | 3.96 / 0.216 | 4.00 / 0.078 | 4.00 / **0.051** | 3.95 / 0.229 |
(held-out robot: same pattern; drawer typed 0.54 items, generic 1.94, heuristic 3.17.)

**Verdict G3 (synthetic): NOT met.** The typed planner matches or beats the scripted heuristic on success for lift,
insert, sweep (0.99-1.00 vs 0.95-1.00) with lower effort on lift, but it does not beat the generic / no-memory
planners, and loses badly on drawer-lot and slide. Diagnosis (measured, not assumed):
1. typed ~= oracle everywhere -> latent estimation is NOT the bottleneck; the outcome head + planner are.
2. first actions are taken before any interaction evidence exists, and the heuristic's margins already cover the
   latent range, so EVPI is small: VOI probing triggered mostly on insert, where it only added effort.
3. the head is miscalibrated on narrow success windows (drawer +-2 cm, slide +-1.5 cm: p_success 0.1-0.3 where
   the simulator succeeds; opening itself is predicted to ~1 cm) and on out-of-distribution state/history
   combinations (lot resets: drawer at x = 0 with a long history -> typed and oracle both slam the end stop).
   The generic model, whose latent input is a learned memory vector, is less brittle there.
4. with uncertain slots the typed/oracle planner samples latents -> more conservative (higher grip force) than the
   plug-in generic/no-memory planners, which succeed anyway because the synthetic lift task is forgiving.

### 5e. G5 transfer (synthetic)
- held-out ROBOT (`synth_mobile`, zero-shot, no adapter data): latent recovery unchanged (5a) and closed-loop
  success within noise of in-distribution (e.g. lift 1.00, insert 1.00, sweep 1.00, slide 0.82 vs 0.78) -> met.
- held-out FAMILY (slide; typed trained on the other 4 families, then a 2000-step fine-tune on 200 slide episodes =
  2.5% of the normal data): latent recovery transfers (friction corr 0.78 @1, 0.83 @2 interactions; mass 0.96), but
  the outcome head does not: closed-loop success zero-shot 0.04, adapter 0.14 (test) / 0.01 (held-out robot) vs
  0.78 in-distribution -> **G5 family transfer NOT met** (needs a more physics-structured head or more adapter data).

## 6. Bugs found and fixed during the run (kept for the record)
- raw-SI SkillCall vectors (aims ~0.004 m next to forces ~100 N) -> per-skill call normalization (before r0).
- r0: head trained on posterior *moments* but the planner fed posterior *samples* (out of distribution, p_success
  0.28 -> 0.10); risk summed unsupervised event logits (drawer "jam" p ~0.4). Fixed in r1: sample-form slot input,
  n/a events supervised as 0, risk only over supervised events.
- linear effort cost let one painful action outweigh failure -> log(1 + effort/effort_ref).
- FD clipping (friction below 0 in the simulator), family discovery skipping held-out-robot splits, calibration
  temperature blow-up (x400) -> clamp [0.25, 10] and clamp latent samples to |z| <= 4.
- ENVS data: ~8-10% of bin_clearing episodes have simulator blow-ups (F/T up to 6e4 N, non-finite poses; MuJoCo
  QACC warnings) -> whole episodes dropped at load (`v2_train.sanitize`, counts logged) + input clamps.
- ENVS data: history from OTHER objects was fused into the current target's slots (all interactions were marked
  on-target) -> per-interaction `target_id`, on-target computed by equality at gather time (r2 data).
