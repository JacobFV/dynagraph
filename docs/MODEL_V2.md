# MODEL_V2 — typed physical-memory world model, planner, and data contract (owner: agent MODEL-V2)

Status (2026-11-24 ~05:00Z): r4 (sec 9) supersedes the r1 closed-loop verdicts. Summary: latent physics WORKS
(G4 synthetic largely met, partial on real stations); the physics-structured head makes the model-based planner beat
the scripted heuristic on the synthetic world (3 seeds, sec 9f) but NOT the generic-memory planner; on real robots
(sec 9g) only the safe residual planner (3-seed ensemble LCB + validation competence gate) is roughly heuristic-level
with small gains, one significant loss (probe depalletize); the current families have value-of-latent <= ~0.1.

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
6. (20:30Z) **simulator blow-ups**: ~8-10% of bin_clearing episodes (probe robot) contain MuJoCo QACC instabilities
   (F/T up to 6e4 N, non-finite tracker poses). The model drops such episodes (`v2_train.sanitize`, counted), but
   please flag them (`metrics["unstable"]`) or fix the contact parameters.
7. labels are read from `hidden[target]`, unique `clearance/damping/frictionloss` on any entity, and
   `task_hidden` (peg_insert clearance). Articulated targets (drawer/door) are observed through JOINT events.

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

### 5f. Real ENVS stations (MuJoCo; TEST-ONLY `probe` gantry robot — no ROBOTS embodiment was bound yet)
Data: `data/v2_env_r3` (6 families x 300 train / 50 val / 50 test episodes, 18.5k train interactions, 0 collection
failures; 26 episodes with simulator blow-ups dropped), segments rebuilt from ENVS 25 Hz events by the adapter.
Typed model r3 (6000 steps, bs 256, 4.0k samples/s), latent corr after j on-target interactions (test):
| family | latent | j=0 | j=1 | j=2 | j=3 | verdict |
|---|---|---|---|---|---|---|
| bulk_sweep | bulk friction | -0.18 | **0.80** | **0.84** | **0.87** | met |
| bulk_sweep | pile mass | 0.96 | 0.96 | 0.95 | 0.95 | (from the visible pile, not interaction) |
| drawer_open | breakaway (frictionloss) | -0.02 | **0.83** | – | – | met (j=1 only; episodes are 1-2 calls) |
| drawer_open | damping | 0.08 | 0.24 | – | – | fails |
| peg_insert | peg mass | 0.15 | **0.98** | 0.90 | 0.92 | met |
| peg_insert | clearance | 0.04 | 0.02 | -0.03 | 0.02 | fails |
| singulation | object friction | 0.16 | 0.15 | 0.34 | 0.43 | fails (r1, which fused other objects' history: 0.69 / 0.78) |
| bin_clearing | mass / friction / grip friction | 0.65 / 0.06 / 0.06 | 0.65 / -0.01 / 0.06 | | 0.67 / 0.04 / 0.28 | fails (mass only from size prior) |
| depalletize | mass / friction / grip friction | -0.01 / .. | 0.29 / -0.01 / -0.13 | 0.19 | 0.33 | fails |
Outcome prediction: model ~= oracle-slot model on every station family (e.g. success AUC bulk_sweep 0.69 vs 0.70,
drawer_open 0.97 vs 0.96, peg_insert 0.94 vs 0.94) -> on the current stations / probe robot, outcomes are barely
latent-dependent once the scene is observed (the v1 lesson again); G4 on real stations is partial (3 latents).

### 5g. G3 closed loop on real ENVS stations (probe robot, 30 paired test episodes per family, heuristic =
ENVS `env.scripted()`, planner cost fixed from the synthetic val selection: c_prog 2.0, lam_effort 0.03;
`scripts/v2/env_closed_loop.py`, rows `experiment=v2_g3_env_closed_loop`)
| family | heuristic | typed | generic | no-memory | oracle (labelled) | paired typed - heuristic |
|---|---|---|---|---|---|---|
| peg_insert | 0.77 [0.63,0.90] / 576 | **0.90 [0.77,1.00] / 374** | 0.87 / 489 | 0.87 / 713 | 0.90 / 387 | succ +0.13 [-0.07,+0.30]; log effort -0.37 [-0.76,+0.04] |
| singulation | 0.80 / 1007 | 0.77 / 1794 | 0.70 / 1403 | 0.67 / 1499 | 0.73 / 1317 | succ -0.03 [-0.23,+0.17]; log effort +0.61 [+0.18,+1.03] (worse) |
| drawer_open | 1.00 / 212 | 1.00 / 403 | 1.00 / 391 | 1.00 / 380 | 1.00 / 324 | effort worse (heuristic is already minimal) |
| bulk_sweep | running at the time of writing (`logs/v2_g3_env_r3_bulk_sweep.log`) | | | | | |
(success rate [95% CI] / mean episode EffortMeter cost.) Typed vs generic on peg_insert: success +0.03
[-0.10,+0.17], log effort -0.41 [-0.73,-0.10] (typed significantly less effort). **G3 on real stations: not met**
(no significant success gain over the heuristic at n=30; one family with significantly lower effort than the
generic-memory planner; typed ~= oracle again).

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

## 7. API (for ENVS / lead / report)
```python
from physmem.models.v2 import spec, synth, tensorize, harness, env_adapter
from physmem.models.v2.model import PhysMemV2          # variants "typed" | "generic" | "nomem"
from physmem.planner import CEMPlanner, PlannerConfig   # modes "typed" | "generic" | "nomem" | "oracle"
from physmem.v2_train import load_model

m, ck = load_model("results/v2/checkpoints/v2_synth_r1_typed_s0.pt")
b = tensorize.single_batch(history_interactions, entity_frame, spec.embodiment_vector(robot_spec), calls (Q,24),
                           query_target=target_id)          # actor-legal only; labels never enter
o = m(b)                                                     # o["mem"]["z_mu"/"z_logvar"] = typed slots (B,10)
d = m.decode(o)                                              # success, p (6 events), risk, effort, cont (8)
o2 = m(b, z_override=(mu, logvar), z_mask_override=mask)     # intervention on typed slots
pl = CEMPlanner(m, "typed", PlannerConfig(c_prog=2.0, lam_effort=0.03))
dec = pl.decide(history, frame, emb, skill, bounds, rng, probe_params=[...])   # -> Decision(params, is_probe, info)
ad = env_adapter.EnvAdapter(make_env(fam)); harness.run_episode(ad, task, "planner", pl)   # ENVS stations
```
CLI: `python -m physmem.v2_train gen|collect|train|eval|calibrate|closedloop|interv ...` (see module docstring);
scripts: `scripts/v2/run_g3.sh`, `run_lot.sh`, `run_pilot.sh`, `sweep_cost.sh`, `env_closed_loop.py`,
`test_adapter.py`, `inspect_env.py`, `debug_*.py`. Tests: `tests/test_v2_model.py` (25 tests; the trained-model
gate test runs when a checkpoint + data exist).

## 8. Pending / next (ranked)
1. Outcome head is now the bottleneck (typed ~= oracle in closed loop): add a physics-structured head path
   (e.g. predict displacement / required-force margins and derive success analytically from goal geometry, instead of
   a free success logit over narrow windows), train with planner-distribution data (DAgger-style rollouts of the
   planner itself) so lot-reset / late-history states are in distribution.
2. Real-station data with ROBOTS embodiments (only the test-only `probe` gantry exists in ENVS so far); held-out
   robot / family on real stations; real-station G4 interventions vs MuJoCo finite differences (need ENVS to expose
   a labelled "re-simulate with modified latent" hook).
3. Weakly identifiable latents (grip friction, clearance, hole offset): needs informative probes (slip test,
   compliant search) in the skill library + VOI that actually pays; currently VOI rarely helps.
4. Multi-seed repeats of the r1 synthetic results (single training seed so far).

## 9. r4 — making latents pay off in decisions (lead priorities, 2026-11-23 ~21:40Z)

### 9a. What changed
1. **Named skill-parameter encoding** (`spec.PARAM_VOCAB`, `named_call_vector`): `api.SkillCall.vector()` places
   params by sorted-key position, so `grip_force` sat in different slots in different families; r4 uses
   [skill one-hot, values by NAME (50), presence mask (50)], per-name scalers fit on train.
2. **Physics-structured outcome head** (`models/v2/physics.py`): 20 margin / quantity features computed from the
   typed-slot VALUES (SI) x named params x robot limits (embodiment vector) x goal geometry (scene goal token):
   grip margin log(2 mu_g F) - log(m sqrt((g+a)^2+a_lat^2)), payload/torque margin, push force margin and travel
   s + v^2/(2 mu g) vs goal, breakaway margin / opening (|F|-0.3F_b)/b T vs goal / slam speed / pull-force margin
   (F_b + b v) for position-controlled doors, grip vs pull force, insertion capture & search margins and jam,
   sweep carry vs goal and stall margin. Event logits = MarginNet(margins, skill) + sigmoid(gate) x residual;
   continuous outcomes with a physics prediction = physics value + sigmoid(gate) x residual; L2 penalty 0.01 on
   residuals; the FiLM trunk (residual) sees typed slots + margins. Generic / no-memory variants get the same head
   with margins computed at prior-mean latents (no typed memory). Typed slots remain the only latent pathway.
3. **Value-of-latent baseline**: planner mode `prior` (typed model, prior-mean slots, no evidence) next to the
   labelled oracle (true latents) in every closed-loop run.
4. **On-policy loop** (`v2_train onpolicy`): round r: closed-loop val success -> 150 planner rollouts per family on
   train-range seeds (20% random exploration, failures kept, evaluator labels attached) -> data = base + all
   rollouts -> 3000-step fine-tune -> round r+1.
5. **ROBOTS tranches**: offline converter (`env_adapter.convert_episode`, `v2_train convert`) replays the stored
   ENVS event streams with a time cursor (frame / target / held object exactly as the actor saw them) -> v2 arrays,
   robots mixed per family (`data/v2_robots_t1`: panda / ur5e_2f85 / probe, 14 families, 0 conversion failures).
6. **Real-station planner**: CEM extrapolated outside the data support and failed on real robots (UR5e/Panda,
   tranche1 model: singulation 0.00 vs heuristic 0.25-0.40, twist_lock 0.00-0.10 vs 0.60-0.65, oracle and prior
   planners equally bad -> model exploitation, logs kept in `logs/failed_cem_robots_t1/`). Real stations now use
   support-constrained ranking (`harness.decide_env_ranked`: scripted call + 15 x U(0.7,1.3) perturbations + 16 env
   candidates = the data-collection distribution) with a **scripted anchor** (deviate from the scripted action only
   if the predicted score improves by > 0.15; margin fixed a priori), labelled "planner (scripted-anchored)".

### 9b. r4 results — synthetic world (seed 0; seeds 1-2 running, see 9e)
Prediction (test, success AUC, r4 physics head vs r1 free head): slide 0.982 vs 0.912 (Brier 0.022 vs 0.042),
drawer 0.901 vs 0.842, sweep 0.932 vs 0.920, lift 0.993 vs 0.992, insert 0.819 vs 0.850 (slightly worse).
Latent recovery unchanged (slide friction 0.93 @1 interaction, drawer damping 0.87, breakaway 0.94, ...).

**G3 closed loop, test seeds, 100 paired episodes / family / split, 95% bootstrap CIs** (success; mean EffortMeter
cost = "servo pain"; rows `experiment=v2_g3_closed_loop`, run ids `v2_g3_v2_synth_r4_typed_s0_lot{1,4}-*`):
| family (single item) | heuristic | typed | generic | no-memory | oracle (true latents) | prior-mean latents |
|---|---|---|---|---|---|---|
| slide | 0.91 [0.85,0.96] / 3.3 | **1.00** / 3.1 | 1.00 / 2.0 | 1.00 / 1.6 | 1.00 / 2.4 | 0.95 / 3.0 |
| lift | 0.95 [0.90,0.99] / 21.9 | 0.99 / 16.2 | 0.99 / 21.4 | 0.99 / 10.2 | 0.99 / 18.5 | 0.99 / 14.9 |
| insert | 0.95 [0.91,0.99] / 0.049 | 1.00 / 0.078 (VOI probe) | 1.00 / 0.024 | 1.00 / 0.033 | 1.00 / 0.015 | 1.00 / 0.036 |
| drawer | 0.54 [0.44,0.64] / 378 | **0.97 [0.93,1.00] / 50** | 1.00 / 45 | 1.00 / 44 | 0.93 / 57 | 0.85 / 55 |
| sweep | 1.00 / 0.17 | 1.00 / 0.25 | 1.00 / 0.28 | 1.00 / 0.10 | 1.00 / 0.41 | 1.00 / 0.43 |
| drawer, held-out robot | 0.62 / 463 | 0.86 / 72 | 0.98 / 86 | 0.96 / 114 | 0.86 / 72 | 0.71 / 87 |

| family (lot of 4, same latents) | heuristic items / eff | typed | generic | no-memory | oracle | prior-mean |
|---|---|---|---|---|---|---|
| slide | 3.67 / 14.3 | **4.00 / 8.7** | 4.00 / 9.0 | 4.00 / 6.4 | 4.00 / 9.9 | 3.87 / 12.7 |
| lift | 3.80 / 87.6 (0.80 risk ev.) | 3.96 / 60.2 (0.16) | 3.96 / 81.1 | 3.96 / 41.3 | 3.96 / 61.9 | 3.96 / 59.1 |
| insert | 3.80 / 0.19 | 4.00 / 0.13 | 4.00 / 0.11 | 4.00 / 0.12 | 4.00 / 0.06 | 4.00 / 0.14 |
| drawer | 3.02 / 1542 (succ 0.41) | **3.85 / 242 (succ 0.92 [0.86,0.97])** | 4.00 / 216 | 3.97 / 233 | 3.65 / 237 | 3.39 / 210 |
| sweep | 4.00 / 0.70 | 4.00 / 1.31 | 4.00 / 0.67 | 4.00 / 0.38 | 4.00 / 1.56 | 4.00 / 1.70 |
| drawer, held-out robot | 3.17 / 1983 | 3.30 / 349 | 3.94 / 590 | 3.92 / 646 | 3.08 / 305 | 2.49 / 276 |

- **vs the scripted heuristic: met** on the synthetic world for the physics-head planner: success >= heuristic
  in every family/split, large gains on drawer (+0.43 single, +0.51 lot) and slide (+0.09), with equal or lower
  servo pain on slide/lift/drawer (drawer 6-8x lower); higher effort than the heuristic on insert (VOI probe) and sweep.
- **vs the generic / no-memory planner: NOT met** — they reach the same success, often with less effort (both use
  the same physics-structured head at prior-mean latents), and typed is worse on held-out-robot drawer.
- **value-of-latent check (oracle - prior-mean success)**: slide +0.05, drawer +0.08 (+0.15 held-out robot), lot
  drawer +0.09 items-equivalent, lift / insert / sweep ~0. These synthetic families are NOT decision-relevant
  enough to show a latent-memory advantage (the head trained on prior-mean physics already picks robust actions);
  they count for general competence only. Note the oracle itself is below generic on drawer -> the remaining
  error is the head's use of explicit latents, not estimation.
- VOI probing still only adds effort (insert); it never changed success.

### 9c. On-policy model-based loop (synthetic, typed, seed 0; VAL seeds, 50 episodes per family per round)
| round | slide | lift | insert | drawer | sweep | (heuristic: 0.94 / 0.96 / 0.90 / 0.60 / 1.00) |
|---|---|---|---|---|---|---|
| 0 (r4 base) | 1.00 | 1.00 | 0.98 | 0.92 | 1.00 | |
| 1 | 1.00 | 1.00 | 0.96 | 0.92 | 1.00 | |
| 2 | 1.00 | 1.00 | 0.96 | **1.00** | 1.00 | |
| 3 | 1.00 | 1.00 | 0.96 | **1.00** | 1.00 | drawer effort 53 -> 39 over rounds |
Rollout success (train-range seeds, 20% exploration) 0.93-0.99 every round; 150 rollouts / family / round added
(failures kept). Generic got the same loop (`v2_onpolicy_r4_generic_s0`, round 0 val already 1.00 everywhere).

### 9d. Real stations with ROBOTS embodiments (tranche1: panda / ur5e_2f85 / probe, ~130 train episodes per
family; scripted-anchored ranked planner; 20 paired test episodes; Gen3 held out = zero-shot)
| family | robot | heuristic | typed | generic | prior-mean |
|---|---|---|---|---|---|
| peg_insert | ur5e / panda | 1.00 / 1.00 | 1.00 / 1.00 (more effort) | 1.00 / 0.95 | 1.00 / 1.00 |
| drawer_open | ur5e / panda | 0.95 / 0.70 | 0.65 / 0.60 | 0.90 / 0.55 | 0.95 / 0.60 |
| door_open | ur5e / panda | 0.45 / 0.50 | **0.05 / 0.15** | 0.05 / 0.10 | 0.35 / 0.20 |
| twist_lock | ur5e / panda | 0.65 / 0.60 | 0.55 / 0.50 | 0.40 / 0.40 | 0.60 / 0.40 |
| tool_return | ur5e / panda | 0.55 / 0.30 | 0.55 / 0.15 | 0.50 / 0.20 | 0.50 / 0.20 |
| machine_tending | ur5e / panda | 0.10 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.05 / 0.00 |
| peg_insert / drawer_open / twist_lock / fixture_place | **gen3 (held out)** | 0.80 / 0.35 / 0.60 / 0.80 | 0.90 / 0.25 / 0.55 / 0.55 | 0.90 / 0.35 / 0.45 / 0.75 | 0.85 / 0.30 / 0.55 / 0.75 |
(remaining families / robots: `results/v2/closed_loop/v2_g3_robots_t1a_*.json` when finished.) Earlier CEM
(unconstrained) runs on the same robots: singulation 0.00 vs heuristic 0.25-0.40, twist_lock 0.00-0.10 vs 0.60-0.65.
**G3 on real robots: NOT met** with tranche1-size data (~100 episodes per robot x family): the planner is at or
below the scripted policy, much worse on door_open; the anchor margin (0.15) does not stop harmful deviations
because the model is confidently wrong there. **G5 (Gen3 zero-shot, typed planner / same planner on UR5e)**:
peg_insert 0.90 / 1.00 (90%), twist_lock 0.55 / 0.55 (100%), drawer_open 0.25 / 0.65 (38%) — mixed; adapter data 0.
Multi-robot latent recovery (tranche1 test): bulk friction 0.90 @1, fixture_place mass 0.99 @1, peg mass 0.96 @1,
machine_tending mass 0.94 @2, tool_return mass 0.90 @3; door/drawer damping, clearance, singulation friction fail.
Data hygiene: 5-16% of real-robot episodes in bin_clearing / conveyor_sort / machine_tending / tool_return /
pouch_transfer contain simulator blow-ups and are dropped (counts in the training logs) — flagged to ENVS/ROBOTS.

### 9e. Real-robot failure diagnosis (lead request 3; `scripts/v2/diagnose_env.py`, t1 model, UR5e, 4 episodes each)
- **door_open**: at the FIRST decision the planner deviates from the scripted `open(grip 20 N, speed 0.15)` to a
  high grip force (54-94 N) and slower speed, because the grip-margin physics feature says more grip = less slip
  (predicted p_success 0.24 -> 0.36). Reality: `handle_slipped` anyway, or contact forces up to 429 N with 4
  collisions; the door is then in a state from which every subsequent (scripted) call fails. Predicted effort
  150-470 vs realised 20k-74k (100x under-predicted: probe-dominated effort scale). Wrong quantities: the sign of
  the grip-force effect on success for this handle (slip is not grip-limited here) and the effort scale.
- **singulation**: the planner picks faster and longer pushes (0.37-0.71 m/s, 6-12 cm vs scripted 0.15 m/s, 4 cm)
  predicting +0.07-0.10 success; realised progress lower and effort 50k-310k vs predicted 0.1-4k. Success
  predictions are nearly uniform (0.08-0.22) -> the model does not discriminate pushes on real arms (~250 UR5e rows).
- Root causes: (i) ~75% probe rows (probe effort ~100x smaller, different dynamics), ~100 episodes per arm x family;
  (ii) the planner's selection exploited confident-but-wrong predictions; (iii) one bad first action is
  unrecoverable in door/drawer.
- Fixes applied: robot-balanced sampling (`--robot-balance`, inverse robot frequency per family) on tranche1 +
  tranche2-partial UR5e (`data/v2_robots_t12`), and the **residual planner** (`harness.ResidualPlanner`): scripted
  call + 15 x U(0.85,1.15) perturbations + 16 env candidates, scored by a 3-seed ensemble; deviate only if
  LCB = mean_m(dC) - 1.0 * sqrt(std_ens^2 + latent spread^2) > 0.
- Calibration after the fix (t12b typed s0, test): effort log-bias (pred - true) panda -0.01, ur5e +0.06,
  probe 0.00 (per-row log-RMSE 1.7-2.1: real-arm effort is heavy-tailed); success mean predicted vs rate
  panda 0.18 / 0.17, ur5e 0.20 / 0.18, probe 0.11 / 0.09.

### 9f. HEADLINE — synthetic G3, 3 training seeds (r4 s0/s1/s2), 100 paired episodes per seed per cell (n=300),
Δ success vs scripted heuristic [95% bootstrap CI over pooled paired episodes] / effort ratio vs heuristic
(geometric mean of per-episode EffortMeter cost ratios) — `results/v2/closed_loop/agg_r4_lot{1,4}_3seeds.{json,txt}`
| split:family | typed | generic | no-memory | oracle (labelled) | prior-mean | value of latent (oracle - prior) |
|---|---|---|---|---|---|---|
| test:drawer | **+0.44 [+0.38,+0.50] / 0.11** | +0.44 / 0.11 | +0.45 / 0.11 | +0.41 / 0.11 | +0.38 / 0.13 | +0.03 |
| test:slide | **+0.09 [+0.06,+0.12] / 0.38** | +0.09 / 0.35 | +0.09 / 0.47 | +0.09 / 0.54 | +0.07 / 0.70 | +0.02 |
| test:lift | +0.04 [+0.02,+0.06] / 0.77 | +0.04 / 0.81 | +0.04 / 0.62 | +0.04 / 0.71 | +0.04 / 0.72 | 0.00 |
| test:insert | +0.05 [+0.03,+0.08] / 3.03 | +0.05 / 1.03 | +0.05 / 1.34 | +0.05 / 0.74 | +0.05 / 1.23 | 0.00 |
| test:sweep | +0.00 / 0.76 | -0.00 / 2.16 | -0.01 / 1.49 | +0.00 / 1.04 | -0.00 / 1.29 | 0.00 |
| held-out robot:drawer | +0.31 [+0.25,+0.37] / 0.16 | **+0.36** / 0.17 | +0.34 / 0.18 | +0.29 / 0.14 | +0.21 / 0.20 | +0.08 |
| held-out robot:slide | +0.05 / 0.35 | +0.05 / 0.34 | +0.05 / 0.53 | +0.05 / 0.60 | +0.03 / 0.94 | +0.02 |
| held-out robot:sweep | **+0.00 / 0.62** | -0.07 [-0.10,-0.04] / 3.35 | -0.02 / 1.73 | +0.00 / 0.94 | -0.01 / 1.11 | +0.01 |
| lot-4 test:drawer | +0.53 [+0.47,+0.59] / 0.13 | **+0.57** / 0.11 | +0.56 / 0.11 | +0.48 / 0.12 | +0.43 / 0.13 | +0.05 |
| lot-4 test:slide | +0.10 [+0.07,+0.14] / 0.52 | +0.10 / 0.52 | +0.10 / 0.53 | +0.10 / 0.61 | +0.08 / 0.79 | +0.02 |
| lot-4 held-out robot:drawer | +0.25 [+0.17,+0.33] / 0.16 | **+0.39** / 0.19 | +0.39 / 0.19 | +0.19 / 0.14 | +0.11 / 0.17 | +0.08 |
| lot-4 held-out robot:sweep | **-0.00 / 0.95** | -0.08 [-0.11,-0.05] / 1.71 | -0.05 / 1.81 | +0.00 / 1.03 | -0.03 / 1.20 | +0.03 |
(lift / insert lot-4 and held-out-robot lift / insert: all planners +0.02..+0.06, same pattern; full table in the txt.)
Per-seed success is stable (e.g. typed drawer 0.97 / 0.98 / 0.99). **Synthetic G3 vs heuristic: met** (success
+0.04..+0.57 with CIs excluding 0 in drawer, slide, lift, insert; effort 0.11-0.77x except insert, where the typed
planner's VOI probe costs 3x). **vs generic memory: not met** (equal success; typed is better only on held-out-robot
sweep, where generic/no-memory lose 0.05-0.08 success and use 1.7-3.4x the effort, and worse on held-out-robot drawer).
**Value of latent** (oracle - prior-mean) is <= +0.08 in every synthetic family -> these families cannot show a
latent-memory advantage (reported separately per the lead's rule; they count for general competence).

### 9g. HEADLINE — real robots (MuJoCo stations, ROBOTS embodiments), paired n=30 per cell
Model: robot-balanced, trained on tranche1 + tranche2-partial + **tranche5_r4** (real-arm data; `data/v2_robots_t125`,
probe / panda / ur5e_2f85 / ur5e_suction / fr3_suction / aloha / tidybot / so100 / z1 / g1; Gen3 held out). Policies:
- **typed**: ranked planner with scripted anchor (margin 0.15), seed 0;
- **residual**: ensemble of 3 typed training seeds (s0/s1/s2), deviate only if LCB = mean - 1 * sqrt(ens^2 + latent^2) > 0;
- **residual_gated**: residual + competence gate — deviations allowed only in families whose VALIDATION success AUC
  >= 0.8 (`results/v2/trust_t125.json`: untrusted = twist_lock 0.62, bulk_sweep 0.64, singulation 0.69,
  kitting_pocket 0.795). In untrusted families it plays the scripted action every step, i.e. exactly the heuristic
  policy on the same seeded tasks (its column there equals the heuristic by construction);
- **generic**: generic-memory model, same ranked planner; **oracle**: typed model + true latents (labelled).
Cells: Δ success vs the scripted heuristic [95% paired bootstrap CI] / effort ratio (geometric mean of per-episode
EffortMeter cost ratios); full table `results/v2/closed_loop/robot_table_t125.{md,json}`, rows
`experiment=v2_g3_env_closed_loop`, run ids `v2_tab_t125_<robot>_<family>`.

| robot | family | heur. | typed | residual (dev rate) | **residual_gated** | generic | oracle |
|---|---|---|---|---|---|---|---|
| ur5e | peg_insert | 0.63 | +0.00 / 1.01 | +0.10 [+0.00,+0.20] / 0.66 (0.29) | **+0.10 [+0.00,+0.20] / 0.66** | +0.00 / 1.03 | +0.00 / 1.02 |
| ur5e | fixture_place | 0.67 | +0.00 / 1.02 | +0.07 [+0.00,+0.17] / 1.16 | **+0.07 / 1.16** | +0.07 / 1.02 | +0.00 / 1.03 |
| ur5e | door_open | 0.53 | -0.17 [-0.30,-0.07] / 5.11 | +0.00 / 1.17 (0.07) | **+0.00 / 1.17** | -0.40 [-0.57,-0.23] / 4.31 | -0.07 / 4.37 |
| ur5e | drawer_open | 0.93 | +0.00 / 1.04 | +0.00 / 0.98 | **+0.00 / 0.98** | +0.00 / 0.75 | -0.03 / 1.16 |
| ur5e | tool_return | 0.63 | -0.03 / 1.04 | -0.03 [-0.17,+0.13] / 0.95 | -0.03 / 0.95 | -0.07 / 1.15 | -0.03 / 1.04 |
| ur5e | singulation* | 0.77 | -0.30 [-0.53,-0.07] / 4.22 | -0.40 [-0.63,-0.17] / 4.03 (0.94) | **+0.00 / 1.00** | -0.30 / 3.63 | -0.23 / 4.07 |
| ur5e | twist_lock* | 0.67 | -0.13 [-0.27,-0.03] / 5.02 | +0.00 / 1.30 | **+0.00 / 1.00** | -0.13 / 4.29 | -0.03 / 4.24 |
| ur5e | kitting_pocket* | 0.63 | -0.17 / 1.05 | +0.00 / 1.24 | **+0.00 / 1.00** | +0.07 / 1.26 | -0.30 [-0.53,-0.07] / 0.42 |
| panda | fixture_place | 0.63 | +0.07 / 1.32 | +0.13 [+0.03,+0.27] / 1.34 | **+0.13 [+0.03,+0.27] / 1.34** | +0.00 / 1.05 | +0.10 / 1.30 |
| panda | peg_insert | 0.73 | +0.00 / 1.01 | +0.07 [-0.07,+0.20] / 0.91 | **+0.07 / 0.91** | +0.00 / 1.05 | +0.00 / 1.01 |
| panda | tool_return | 0.33 | +0.00 / 1.02 | +0.10 [-0.03,+0.23] / 1.07 | **+0.10 / 1.07** | +0.03 / 1.12 | +0.00 / 1.07 |
| panda | door_open | 0.50 | -0.20 [-0.40,+0.00] / 5.40 | -0.13 [-0.30,+0.00] / 2.11 | -0.13 [-0.30,+0.03] / 2.11 | -0.37 [-0.57,-0.17] / 5.86 | +0.03 / 3.42 |
| panda | drawer_open | 0.47 | +0.00 / 0.90 | +0.00 / 1.00 | +0.00 / 1.00 | +0.00 / 0.99 | +0.00 / 0.99 |
| panda | singulation* | 0.73 | -0.30 [-0.53,-0.03] / 1.80 | -0.30 [-0.50,-0.10] / 1.96 | **+0.00 / 1.00** | -0.37 / 1.11 | -0.47 [-0.67,-0.27] / 1.77 |
| panda | twist_lock* | 0.57 | -0.03 / 2.60 | +0.00 / 1.00 | **+0.00 / 1.00** | -0.10 / 3.72 | -0.07 / 3.57 |
| panda | kitting_pocket* | 0.63 | -0.27 [-0.47,-0.07] / 0.46 | +0.03 / 1.35 | **+0.00 / 1.00** | -0.03 / 1.63 | -0.33 [-0.53,-0.13] / 0.62 |
| probe | bin_clearing | 0.23 | -0.03 / 0.95 | +0.07 [-0.07,+0.20] / 1.08 | **+0.07 / 1.08** | +0.03 / 1.17 | -0.07 / 1.42 |
| probe | conveyor_sort | 0.30 | +0.07 / 0.90 | +0.03 / 0.83 | **+0.03 / 0.83** | -0.03 / 0.97 | +0.03 / 0.88 |
| probe | bulk_sweep* | 0.70 | +0.03 / 0.89 | -0.03 / 1.09 | **+0.00 / 1.00** | -0.03 / 1.12 | -0.03 / 1.11 |
| fr3_suction | depalletize | 0.00 | +0.03 / ~0 | +0.03 / ~0 | +0.03 / ~0 | +0.03 / ~0 | +0.03 / ~0 |
| **gen3 (held out)** | door_open | 0.87 | -0.20 [-0.37,-0.03] / 7.25 | -0.03 [-0.10,+0.00] / 1.25 | -0.03 / 1.25 | -0.60 [-0.77,-0.40] / 4.17 | +0.03 / 5.05 |
| gen3 | drawer_open | 0.30 | +0.03 / 2.00 | +0.00 / 1.00 | +0.00 / 1.00 | +0.00 / 0.98 | +0.03 / 3.75 |
| gen3 | fixture_place | 0.80 | +0.00 / 1.07 | -0.07 [-0.17,+0.00] / 1.23 | -0.07 / 1.23 | +0.00 / 0.97 | +0.00 / 1.06 |
| gen3 | peg_insert | 0.13 | +0.00 / 1.00 | -0.07 [-0.17,+0.00] / 1.09 | -0.07 / 1.09 | +0.00 / 1.02 | +0.00 / 1.10 |
| gen3 | tool_return | 0.33 | -0.07 / 0.87 | -0.10 [-0.27,+0.07] / 0.78 | -0.10 / 0.78 | -0.20 [-0.37,-0.03] / 0.94 | -0.17 [-0.30,-0.03] / 0.86 |
| gen3 | singulation* | 0.53 | -0.10 / 3.42 | -0.17 / 3.08 | **+0.00 / 1.00** | -0.30 [-0.53,-0.07] / 3.82 | -0.20 / 3.53 |
| gen3 | twist_lock* | 0.77 | -0.13 [-0.27,+0.00] / 2.89 | +0.00 / 1.00 | **+0.00 / 1.00** | -0.17 / 1.48 | +0.00 / 3.33 |
| probe | depalletize | 0.60 | -0.60 [-0.77,-0.43] / ~0 | -0.23 [-0.43,-0.03] / 1.50 (0.50) | **-0.23 [-0.43,-0.03] / 1.50** | -0.47 [-0.67,-0.23] / 2.09 | -0.57 [-0.73,-0.37] / 0.18 |
| ur5e_suction | depalletize | 0.00 | +0.00 | +0.00 (0.96) | +0.00 | +0.00 | +0.00 |
| gen3 | kitting_pocket* | 0.20 | -0.03 / 1.60 | +0.10 [-0.03,+0.23] / 1.40 | +0.00 / 1.00 | +0.07 / 1.28 | -0.03 / 0.46 |
(* = family untrusted by the validation gate; 30 cells total.)

**Verdicts (real robots, single n=30 cells):**
- **G3 vs heuristic**: the competence-gated residual planner is significantly worse than the scripted policy in
  ONE of 30 cells — probe depalletize -0.23 [-0.43,-0.03] (a family that passed the validation gate, AUC 0.95: the
  exception-lane decision (flag vs lift an overweight carton) is where the model is confidently wrong; the gate is
  necessary but not sufficient) — otherwise within CI of the heuristic (next worst panda door_open -0.13
  [-0.30,+0.03]) and better in 9/30 cells, significantly in panda fixture_place (+0.13 [+0.03,+0.27]); ur5e
  peg_insert +0.10 [+0.00,+0.20] with 0.66x servo pain. Typed-alone,
  generic and even the ORACLE planner are significantly worse in door / singulation / kitting / twist_lock
  (3-7x effort) -> the failure mode is action selection under model error, not latent estimation. G3 overall:
  **partially met** (safe, small gains on fine-manipulation families; no family-wide significant win yet).
- **vs generic memory**: residual_gated >= generic in 28/30 cells; generic is significantly worse than the
  heuristic in 10 cells.
- **G5 (Gen3 held out, zero adapter data)**: residual_gated within -0.10..+0.00 of the heuristic in all 8 cells
  (none significant); ratio to in-distribution success (residual_gated, gen3 / ur5e): door 0.84/0.53, drawer
  0.30/0.93, fixture 0.73/0.73, peg 0.07/0.73, tool 0.23/0.60, twist 0.77/0.67 — the Gen3 gap is mostly the
  scripted policy's own gap on that robot (heuristic gen3 peg 0.13 vs ur5e 0.63), not the model's.
- **value of latent (oracle vs typed)**: oracle - typed is between -0.17 and +0.20 with overlapping CIs in every cell;
  together with the synthetic check (<= +0.08), none of the current families is decision-relevant enough to show a
  latent-memory gain — reported separately as the lead asked; ENVS is redesigning several families for this.


### 9h. Pending / next
- 3-seed real-robot evaluation of the typed / generic single models (the real-robot table uses seed 0 for typed /
  generic / oracle; the residual planner is the 3-seed ensemble). On-policy rounds on real stations not run (CPU
  budget); the synthetic loop is in 9c.
- `move_base` (api SKILLS_V2) is in the named encoding (`spec.SKILLS`, 11 skills; legacy 10-skill checkpoints load via
  `n_skills`), but no trained model has seen mobile-base data yet (tranche6_r5 / shelf_restock).
- Depalletize exception-lane error (planner lifts / flags wrongly) needs a dedicated diagnosis; ENVS' redesigned
  decision-relevant families (overweight -> exception lane, peeling cartons, twist-lock breakaway, tool_return grip,
  conveyor speed) are where a latent-memory advantage can actually show (current value-of-latent <= ~0.1).
- Effort on real arms is heavy-tailed (per-row log-RMSE ~2, unbiased per robot); servo-pain distributions per
  policy are in the closed-loop JSONs (`effort` per episode) and summarised as effort ratios / medians / p90.

## 10. Round 3 (2026-11-24): fragile parts, safe residual, envs-r6 (lead + ENVS r6)

### 10a. Changes (model schema `physmem-v2-model-3`; r4/r5 checkpoints are legacy and do not load)
- **Fragility in the typed memory**: new typed latent `crush` (N per pad; ENVS `crush_n`, and `dent_n` for pegs;
  parts ENVS marks unbreakable (`inf`) are labelled 500 N so the latent learns the fragile/robust mixture), new
  supervised event `damage` (increase of the cumulative `metrics.damaged`), counted in the planner's risk term.
- **Physics head**: +3 margins — crush margin log(crush) - log(grip_force) for pick/open/close, dent margin
  log(crush) - log(down_force) for insert, and the exception-lane margin log(m g) - log(payload g) for `handover`.
  Minimal safe grip = max(slip margin from mass / grip friction, crush margin) is now an explicit trade-off.
- **Actor-legal fragility evidence (ENVS r6)**: entity tokens carry the latest `vision_material` and
  `tactile_squeeze` STIFFNESS readings (log, plus a has-tactile flag); the typed-slot prior network reads the target
  token, so a tactile reading tightens the crush prior. Labels map `stiffness` -> STIFFNESS (N/m).
- **Identifiability finding (before r6)**: with r2-r5 data the crush limit is NOT recoverable (corr ~0 after 0-3
  interactions in bin_clearing / conveyor / tool_return / fixture / twist_lock; pouch 0.5-0.8 via compliance, because
  pouch crush = 4 + Young/2500). Requested a legal signal in docs/ENVS.md; ENVS r6 added it (vision + tactile).
- **Residual planner**: (i) may lower grip_force (x0.25-0.85) and speed caps (x0.5-0.7) of the scripted call;
  (ii) **parameter-only** — it never changes the scripted discrete decision (skill + target), see 10b;
  (iii) variant anchored on the ENVS probe-then-act scripted call (`residual_gated_probe`).
- **Baselines reported separately**: plain scripted heuristic and the ENVS probe-then-act heuristic
  (`env.probe_then_act`: 4 N first squeeze, then grip <= 0.7 x estimated crush limit).
- **Data**: zero-length skill windows (rejected calls, t_start == t_end, NaN EE velocity) are guarded in
  `build_segment` instead of dropping the episode: load-time drops on train went from 1,759 episodes / 27.6k rows to
  74 / 1.6k; episodes with ENVS `integrity.unstable` / `metrics.unstable` are skipped at conversion. G5: drawer_open
  train/val episodes excluded from every tranche (held-out family); Gen3 = held-out robot.
- **Evaluation**: envs-r6 site-dressed scenes (`params["dressing"] = "site"`), 3 training seeds for typed / generic /
  oracle, damage rate per policy.

### 10b. Depalletize diagnosis (probe, residual_gated -0.23 [-0.43,-0.03] in round 2)
The scripted baseline is an information-gathering protocol: heft every carton with a 4 cm lift, estimate its mass
from wrist F/T (or a lower bound from a failed full-suction lift), then deliver it, or put it back and flag it
(`handover`). The round-2 residual pool also contained the env's random candidates (20% `handover` on un-hefted
cartons, picks with lift_height 0.1 m and random suction): a deviation could flag a light carton or lift an overweight
one clear of the pallet, both of which fail the episode. Fix: the residual planner only modifies the continuous
parameters of the scripted call (same skill, same target), never the protocol's discrete decision.

### 10c. State at the shared-host hold (2026-11-24 07:12Z) and the resume plan
**Done (r6 schema):** r2-r6 tranches re-converted with the zero-length-window guard (`data/v2_r6_t{1,2,5,6,3}`,
merged `data/v2_r6_base`, drawer_open train/val excluded); interim typed models `v2_robots_r6i0_typed_s{0,1,2}`
(VAL gate: singulation 0.54, twist_lock 0.61, bulk_sweep 0.56, kitting 0.79 untrusted; the other 12 trusted);
**on-policy round 1** on dressed r6 scenes (gated residual, 20% exploration, 24 train-range episodes per cell, all
interactions incl. failures kept: `data/v2_r6_rollouts/round1`, merged `data/v2_r6_op1`); round-1 retrain
`v2_robots_r6i1_typed_s{0,1}` finished.

Round-1 rollout success / damage per cell (planner + 20% random exploration, so NOT an evaluation):
| robot | conveyor | bin_clear | singul. | door | tool_ret | pouch | twist | mach_tend | fixture |
|---|---|---|---|---|---|---|---|---|---|
| ur5e | 0.00 / 0.58 | 0.00 / 0.67 | 0.25 / 0.75 | 0.25 / 0.67 | 0.75 / 0.17 | 0.13 / 0.83 | 0.67 / 0.29 | 0.04 / 0.71 | 0.50 / 0.29 |
| panda | 0.08 / 0.58 | 0.00 / 0.38 | 0.17 / 0.83 | 0.42 / 0.46 | 0.38 / 0.33 | 0.33 / 0.63 | 0.58 / 0.33 | 0.04 / 0.46 | 0.71 / 0.17 |
probe: conveyor 0.17 / 0.63, singulation 0.21 / 0.75, depalletize 0.67 / 0.00; ur5e_suction depalletize 0.00 / 0.00.
These interim models were trained on r2-r5 data WITHOUT the r6 stiffness evidence (crush corr ~0 there), so damage
stays high; the round-1 rollouts are the first r6 data with vision/tactile stiffness readings in the training set.

**Stopped at the hold:** `v2_robots_r6i1_typed_s2` (GPU, ~60% of 8k steps, started before incremental checkpoints
existed -> re-run, 12 min), the r6 pipeline driver and its watcher. Nothing else of MODEL-V2 runs; `~/dynagraph/HOLD`
is set (every MODEL-V2 script waits while it exists).

**Resume** (one command, idempotent; skips finished outputs, waits while HOLD exists, CPU niced, GPU via gpu_heavy):
`rm ~/dynagraph/HOLD; nohup scripts/v2/r6_resume.sh > logs/v2_r6_resume.log 2>&1 &`
1. `v2_robots_r6i1_typed_s2` (12 min GPU) -> VAL gate `trust_r6i1.json` (2 min CPU).
2. On-policy round 2: 22 cells x 24 episodes with the round-1 ensemble (~35 min on 8 procs) ->
   `data/v2_r6_rollouts/round2`.
3. `r6_final.sh`: wait for tranche8_r6_site (ENVS; 9,875 dressed episodes with stiffness evidence, G1 cell, tidybot
   move_base), convert, merge base + tranche8 + rollouts rounds 1-2 -> `data/v2_r6_final`; FINAL models
   `v2_robots_r6f_{typed,generic}_s{0,1,2}` + `nomem_s0` (7 x ~12 min GPU, incremental checkpoints every 2k steps);
   VAL gate `trust_r6f.json`.
4. Final robot table (`run_robot_table_r6.sh`, n=30 paired episodes per cell, site-dressed, skips finished cells,
   partial JSON every 5 tasks): ur5e / panda x {conveyor_sort, bin_clearing, singulation, tool_return, pouch,
   twist_lock, machine_tending, tote_transfer, door_open, drawer_open (G5 family), fixture_place}; gen3 (G5 robot) x
   {conveyor, bin_clearing, door, tool_return, twist_lock, fixture}; probe / ur5e_suction / fr3_suction depalletize;
   probe conveyor + singulation. Policies: plain heuristic, probe-then-act heuristic, typed / generic / oracle x 3
   seeds, residual_gated (plain anchor), residual_gated_probe (probe-then-act anchor). ~33 cells x 13 policies x 30
   -> ~6-7 h on 8 procs. Table: `scripts/v2/robot_table5.py r6f` -> `results/v2/closed_loop/robot_table5_r6f.md`
   (Δ success vs plain AND vs probe-then-act heuristic [95% paired CI], damage rate, effort ratio, n per group).
5. Paired videos: `pick_video_cells.py` takes the cells where a residual variant is significantly better than the
   plain heuristic (CI > 0), best first, up to 4, with a PRE-REGISTERED seed rule (first paired task where the
   heuristic fails and the planner succeeds, same planner rng as in the table); `paired_videos.py` renders
   heuristic | residual side by side (same seed, dressed scene, label strip: policy, live SUCCESS / FAIL (damaged
   part), effort; ENVS Recorder) -> `results/v2/videos_paired/*.mp4` + `.json` + `stills/*.png`; the lead's rsync then
   copies 4 stills to `~/Downloads/dynagraph_gallery/paired/`. If no cell is significantly positive, no videos are
   rendered (reported as such).
Compute if run serially after the hold: ~1.5 h GPU + ~8 h CPU (8 procs), plus the tranche8 wait.
