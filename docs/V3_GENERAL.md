# V3: one general physical-memory model across domains (plan; started 2026-11-26)

Goal (PI): **a single model for all domains, which ideally generalizes beyond them.** Today we have separate model
lines: sprint compact, v2 synthetic, v2 stations and Inkling System I. V3 merges them into one checkpoint family. It
adds passive third-person observation, new physical domains and human bodies, and every claim is tested with
held-out domains.

## 0. Centerpiece (added 2026-11-26 after the PI's external audit; takes priority over breadth)

The PI's original thesis: the looped core of Inkling-2-Mini is a semi-editable, semantically structured variable
store (key = entity × property class, value = current belief, with recency-normalized writes). Recurrent attention
updates it according to rigid-body, Coulomb static/dynamic friction, Hookean elasticity and an indeterminacy-aware
model of other agents. Anchors are added, updated and removed, with availability masks. The anchors are:
- retrospective annotations;
- real-time linear-probe estimates from encoder features;
- simulator insider state (training only);
- digital-twin and golden sensors (IMU, joint encoders, camera SE(3), contact points, tracked centroids).

Property identity is reached by traversing the interaction graph from those anchors (SE(3) chains plus
property-class identifiers).

The audit and our own code agree on what is missing (`models/inkling.py:494`: readout reuses the pre-loop typed
estimates; losses constrain only final predictions):
1. **Explicit per-pass typed belief state.** Each pass k decodes, updates and writes back named physical beliefs
   (poses, velocities, contact modes, parameters, uncertainty, anchor provenance), and the next pass reads them.
   Refinement depth k is kept separate from physical time t. Candidate actions and horizons enter the loop.
2. **Law regularization on intermediate states**, not only on outputs: Newton–Euler/contact consistency,
   friction cone, Hooke, plus supervision of intermediate physical consistency and final prediction.
3. **The acceptance test is causal, not decodability:** change one named property, and the intermediate estimates,
   future motion, chosen action and realized outcome must change as physics says. Also test anchor removal,
   delayed evidence, graph-edge blocking and held-out chain lengths.
4. **Architectural separation under matched observations, labels, planner and budget:**
   - current v1;
   - explicit per-pass updates;
   - the same without law regularization;
   - v2/classical estimation plus analytic physics.

   Add relation and recurrence ablations, and the pretrained vs random-init core (already started: offline ≈ equal).
5. **Strongest practical comparator for lot memory:** a classical estimator that probes once, caches the lot estimate
   and uses the same planner and risk rule.
6. **One identifiable family first:** a small contact chain with hidden mass and friction, a fixed-force excitation, a
   post-release slide, an optional spring, and sparse pose anchors. Then transfer it to one competent G1 skill and
   show evidence → revised belief → different action → better outcome with the controller held fixed.
7. **Modalities:** force/touch is one channel among vision, third-person tracks, proprioception, digital-twin anchors
   and other agents. The PI's bar is general real-world performance, so every track must say how its claim reaches
   real data (EXT real human-object datasets; real video benchmarks; a real-robot pathway if hardware becomes available).

## Hypotheses (each will be pre-registered in its track section before any test run)
- **H1 (unification):** one model trained on all domains is within a small tolerance of each single-domain
  specialist on that domain's test split. The tolerance is set in the pre-registration.
- **H2 (transfer):** leave-one-domain-out. The general model beats a from-scratch specialist trained on that held-out
  domain's small adaptation set, and beats a model trained on the other domains without a shared typed schema.
- **H3 (passive identifiability):** from noisy third-person keypoint tracks alone, the typed slots recover what is
  identifiable:
  - for springs, k/m from the period and c/m from the decay;
  - for collisions, mass ratios;
  - for bounces, restitution;
  - for slides, Coulomb friction (absolute).

  A single interaction with a known force then fixes the absolute scale (k, m). Slots for non-identifiable
  quantities must report wide posteriors (calibration is checked).
- **H4 (new domains):** on deformables, liquids, granular material, stacking, legged contact and handover, typed memory
  improves outcome prediction and closed-loop decisions over generic memory and no memory. Paired tests, n ≥ 30.
- **H5 (external, third-person):** zero-shot and few-shot on Physion (and CLEVRER/PHYRE/IntPhys where feasible),
  against the published baselines in those papers.
- **H6 (humans):** treating humans as an embodiment without force sensing, the model infers object properties (mass,
  fragility) from watching human-object interaction tracks better than generic memory does.

## Tracks
| track | owner | scope | first deliverable |
|---|---|---|---|
| V3-MODEL | agent V3-MODEL | unified data contract (an observation-modality token: proprio/force, third-person tracks, human pose); merge sprint worlds, v2 synthetic and stations into one checkpoint; leave-one-domain-out; passive spring/rigid-body domain with simulated noisy keypoint tracks (H1–H3) | §1 contract + §2 pre-registration, then H1 |
| ENVS-V3 | agent ENVS-V3 | new MuJoCo families: cable/cloth (flex), pouring (particle liquid approximation, documented), granular scoop, stacking stability, G1 terrain friction, handover, all with hidden per-object/lot physics, oracle ceilings and scripted baselines; frozen tree `envs-v3` | ceiling table per family |
| EXT | agent EXT | external benchmarks (Physion first) and human-hand data: licensing audit, local download (no re-upload), a pose/keypoint front end into the V3 contract, and loaders | license table + Physion loader + baseline numbers to compare against |

## Rules (inherited)
- Pre-register each test, commit it on its own, then run.
- Paired evaluation, n ≥ 30, 95% bootstrap CIs. VAL-only tuning. The labelled oracle is a ceiling only.
- No privileged simulator state crosses the actor boundary. Third-person tracks are rendered or tracked with noise,
  never read from simulator state as is.
- Datasets: obey their licenses, never re-upload, record provenance and sha256 in `data/*/manifest.json`. Data that
  needs registration is fetched only after the PI registers.
- Compute: the shared GB10 tenancy (≤ 6 niced CPU processes across all our tracks, < 30 GB, small GPU jobs under
  `.gpu_small.lock` at < 6 GB each, GB10 disk 95% full so ≤ 15 GB new there in total). Large datasets stay on the
  local disk, and only preprocessed subsets go to GB10.

## Status
- 2026-11-26: plan written; tracks launched. Results are appended per track below.

---

# V3-MODEL track (owner: agent V3-MODEL; branch `v3-model`, worktree `dynagraph-v3`, gb10 dir `~/dynagraph_v3`)

Redirect (2026-11-26, lead, after the PI's audit): §0 outranks H1–H3. The order is: the looped typed belief store on
one identifiable family → its causal acceptance test → matched comparisons. Unification (H1–H3) follows on the
same architecture. The §1 contract below is written for that architecture.

## §0a. Implementation (what was built)

Code: `src/physmem/models/v3/` (`chain.py` family + analytic simulator, `tracks.py` camera tracker,
`belief.py` model/losses, `contract.py` §1 contract), `src/physmem/v3_chain.py` (data, training),
`src/physmem/v3_chain_eval.py` (prediction, classical baseline, causal tests, planner), tests
`tests/test_v3_belief.py`.

**Family: 1-D contact chain** (MuJoCo 4.0.1, dt 2 ms, elliptic cones). A force-controlled pusher (0.2 kg, known
joint damping 2 N s/m) drives L boxes (L ∈ {1, 2, 3} train/val/test; **L = 4 held out**). Hidden per box: mass
log-U[0.2, 1.0] kg and Coulomb friction U[0.15, 0.6]. With p = 0.4 the last box is tied to a wall mount by a spring
(hidden k log-U[15, 150] N/m, known damping 1 N s/m). Box widths 4–8 cm, initial gaps 0–1 cm. Each episode has 4
windows. In each window there is a **fixed-force excitation**: a known constant force F ~ U[1, 15] N for
T ~ U[0.1, 0.3] s, with an admittance speed limit (the force fades to 0 between 0.5 and 0.6 m/s). Then a −2 N
retract for 0.15 s, then coasting: the **post-release slide**. Then 0.8 s of settling, and the pusher is
re-placed 1–5 mm behind box 1. The excitation is drawn independently of the hidden properties, so the data policy
cannot leak them.

**Observations (three modalities; force is one of them):**
- `proprio_force`: pusher encoder (sd 0.05 mm, 100 Hz), velocity, wrist force along x (sd 0.2 N plus a
  per-episode bias with sd 0.3 N), contact flag, commanded force.
- `keypoint_track`: an RGB-D keypoint tracker of every box (`tracks.CameraTracker`):
  - random camera pose; 30 Hz with 3 ms timestamp jitter and 8% frame drops;
  - 1 px noise, 1% outliers (15 px), 5% keypoint dropout;
  - an occluder band in 50% of episodes;
  - depth noise of 0.3% plus 1 mm;
  - actor calibration error: focal length 0.5%, 0.2° rotation, 1 mm translation.

  Detections are back-projected with the actor's own calibration.
- `pose_anchor`: sparse golden fixes (sd 0.3 mm, Poisson(1.2) per box and window, plus a fix just before the
  window in 50% of cases).
- A map anchor gives the spring rest position (sd 1 mm).

The current pose estimate of each box is the fresh anchor if one exists, else the tracker median over 0.25 s, else
the last known detection. Its provenance (anchor/tracker/known/stale) and age are inputs. Validation: re-simulating
a window from its true start state reproduces the logged window to ≤ 4 mm. With true parameters and the actor's
start estimates, the analytic simulator predicts the final positions to 3.7 mm mean (the labelled oracle ceiling).
Truth law residuals are small (friction cone 4e-4, sliding 4e-4, complementarity 3e-3, normalised).

**Model `LBS` (looped belief store).** The store is keyed (entity × property class). Per entity it holds log
mass, friction, log k (parameters) and pose x(τ) and velocity v(τ) on the physical-time grid τ = 0, 0.1, …, 1.5 s
for the queried action, each as mean + log-variance. Per relation it holds a contact logit c(τ) and an
interval-average normal force f(τ) ≥ 0. Each pass k:
1. Re-embeds every named value with known inputs (type, size, provenance, age, current estimate, the queried
   action for the pusher) and an opaque random id per entity.
2. Runs 2 attention layers (weights tied across passes, pass embedding) over [entity tokens, relation tokens,
   evidence tokens]. A **graph mask** routes all entity–entity information through relation tokens: an entity
   sees itself, its incident relations and its own evidence; a relation sees its two endpoints and their evidence.
3. Decodes a proposal (mean, log-variance) and a write gate w for every named belief, and writes back with an
   uncertainty-aware, recency-normalised convex update in natural parameters (λ' = (1−w)λ + wλ_prop, η likewise).
4. Re-applies anchors (pusher pose and velocity at τ0 from encoders; wall pose from the map) and any interventions
   (clamped beliefs, tiny variance).

Pass k+1 reads only the written store, so there is no hidden channel (unit test). Refinement depth k (2–6 random in
training, 4 at evaluation) is separate from physical time τ. The candidate action enters the loop as the pusher's
known command, and the horizon is the τ grid. Evidence tokens are one per (entity, modality, past window): a
1-D conv over 32 bins with a modality embedding, the window age and that window's action.

**Losses.**
- Gaussian NLL on x(τ), v(τ) (pusher and boxes) and on the parameters, plus BCE on contacts, at the final pass
  and with weight 0.3 at earlier passes (deep supervision, identical for every LBS variant).
- **Law regularisation on every pass's decoded beliefs** (weight 1; `belief.law_residuals`):
  - kinematic consistency;
  - pusher Newton with the known speed-limited command;
  - box Newton with the Coulomb friction cone |f_fr| ≤ μmg and the sliding equality f_fr = −μmg·sgn(v);
  - Hooke spring force −k(x − x_rest) − c v on the spring box;
  - contact non-penetration and force–gap complementarity.
- **Counterfactual consistency**: in 25% of training rows one parameter of one box is clamped to a random prior
  value. These rows get only the law loss (no labels exist for that world), so the laws, not data, define how the
  rest of the store must respond to an edit.
- Modality dropout: 15% of rows have no tracks, 15% no proprio/force, 25% no anchors.

**Matched comparators** (same observations, labels, data, steps (15k × 256 rows), optimiser and planner):
- `V1`: the v1/Inkling design. Typed parameter estimates are decoded once after an encoder pass, before the loop.
  The loop refines hidden tokens without writeback, and the readout uses the final hidden tokens plus the pre-loop
  typed estimates. Law loss on the final outputs only, with the same counterfactual rows. Interventions edit the
  pre-loop typed estimates. A unit test checks that loop depth cannot change the typed estimates.
- `LBS-nolaw`: law weight 0 and no counterfactual rows.
- `classical`: CEM system identification (64 × 8 iterations, MAP under the training prior) of all masses,
  frictions and k. It uses the vectorised analytic chain simulator against the same binned tracker/anchor
  displacements, pusher force and encoder evidence, then predicts with the analytic simulator.
- `oracle` (labelled): true parameters + analytic simulator.
- Ablations: recurrence depth K ∈ {1, 2, 4, 8} at evaluation; relation ablation `dense` (dense attention, LBS loss).

## §1. Unified data contract (`src/physmem/models/v3/contract.py`; built for the §0 architecture)

One contract for every domain, in three parts.

**1. Evidence tokens.** There is one token per (entity, modality, time window). The modality is explicit:
- `proprio_force`;
- `keypoint_track`;
- `pose_anchor`;
- `vision_material`;
- `human_pose` (EXT front end);
- `map_anchor`.

A missing modality is a masked token, never a zero signal. Force is one modality among several, and in the chain
family every modality can be dropped independently, during training and at test.

**2. Entity and relation tokens.** Entity types are effector, object, fixture, agent, link, particle, goal_region
and human_part, and each entity has an opaque random id. Relations are contact, spring, joint, support, grasp,
carried_by and observes. Identity comes from the relation graph, never from list position. In LBS, all
entity–entity information flows through relation tokens, which makes "edge blocking" an exact intervention.

**3. Typed belief store.** The key is (entity, property class). The value is a belief: a mean and log-variance in a
normalised space, or a logit. Each value carries its provenance (anchor / tracker / known / stale / model / prior)
and age. The property classes are a superset of the v2 latents (`spec.LATENTS`, same order and transforms; unit
test):

| group | property classes |
|---|---|
| v2 latents | mass, friction, grip_friction, stiffness, clearance, offset_x/y, damping, breakaway, bulk_friction, crush |
| V3 additions | spring_k, k_over_m, c_over_m, mass_ratio, restitution, length_scale (metric scale of monocular tracks) |
| states on the physical-time grid | pose, velocity |
| relations | contact_mode, contact_force |

Every class has an identifiability tag:
- `passive`: recoverable from third-person tracks alone (friction, k/m, c/m, mass ratio, restitution, pose,
  velocity, contact mode);
- `force`: needs a known-force interaction (mass, spring_k, contact_force, …);
- `anchor`: needs an anchor (length_scale).

The H3 test checks these tags. A structural identity (log k = log k/m + log m) is exposed as a cross-domain
consistency check. Domains may use their own normalisation inside a store (the chain uses log-mass loc
log 0.447, scale 0.6); conversion always goes through SI.

**Domain mapping (status).**
- Chain (§0): done.
- Passive third-person domain (`models/v3/passive.py`): draft generator. It has spring, collide, bounce and slide
  scenes seen only through the tracker, an optional known-force poke, and realised-restitution labels. It is not
  yet trained or evaluated.
- Human modality (EXT H6 interface, `contract.human_evidence`, commit e610e78). Each hand is a `human_part`
  entity of an embodiment without force sensing (`human_pose` tokens only). The object's typed `mass` slot is
  reached through a `grasp` relation. The matched twin is LBS-generic (untyped store of equal width).
  Plan (lead, 2026-11-27), queued behind §0:
  - train or adapt the human pathway on REAL DexYCB TRAIN subjects 01–05, with VAL subject 06, using EXT's
    leave-objects-out folds (object_index % 4), so no test object's mass is ever seen;
  - add a sim human-like lift scene as extra pretraining;
  - report sim-only vs sim+real;
  - subjects 07–08 are never touched before EXT's H6 freeze commit.
- Sprint w1/w2, v2 synthetic and v2 stations: adapters are pending (H1). The v2 segment channels map to
  `proprio_force` tokens; the tracker channel of v2 (`obj_*`) maps to `keypoint_track`.

## §0b. Pre-registration: causal acceptance test and matched comparisons (frozen before any TEST run)

Status: written and committed on its own before any TEST-split evaluation. Training and the pilot used train/VAL
only. Every number below will come from `results/v3/eval/*` via `physmem.v3_chain_report`.

**Runs.**
- Training: 4000 steps × 512 rows each, AdamW 1e-3 one-cycle, bf16 autocast, passes ~ U{2..5}, d = 128, a hard
  5 GB GPU cap. Data `data/v3_chain` (manifest with sha256): train 10k, val 600, test 600 and test_L4 300 episodes.
- Runs:
  - `lbs_law_s{0,1,2}`, `lbs_nolaw_s{0,1,2}` and `v1_s{0,1,2}`;
  - `dense_law_s0` (relation ablation);
  - `classical` (deterministic CEM, seed 0) and `oracle` (labelled).
- The law weight is fixed at 1.0. No hyperparameter is tuned on TEST.
- The budget was lowered from 15k × 256: the first run at batch 1024 exceeded the 6 GB lane rule and was stopped.

**VAL-pilot decisions taken before this freeze** (classical/oracle only; no trained model had been evaluated):
1. Planner goals are displacements the true world can reach with one grid action (≥ 1 cm when possible; drawn
   model-independently). Uniform goals were often infeasible (labelled-oracle error 23 mm).
2. The A1 correlation is Spearman (Pearson was dominated by spring outliers).

VAL reference values:
- classical (CEM + analytic): own-box sign 0.96, action sign 0.79, counterfactual-outcome Δ −4.0 mm,
  closed loop 9.9 mm / 82% within 15 mm;
- oracle: 0.99, 0.82, −8.7 mm, 3.6 mm / 97%.

**Evaluation protocol.**
- TEST split: 600 episodes, L ∈ {1, 2, 3}. `test_L4`: 300 episodes, held-out chain length 4.
- Prediction is scored on all rows. Causal tests and the planner use window k = 3 (three windows of evidence) of the
  first 200 TEST episodes (and the first 200 `test_L4` episodes for the secondary L4 check), with K = 4 passes.
- Edit plan (`v3_chain_eval.edit_plan`, rng seed 12345, identical for every model): per episode, a random box, a
  random property (friction or mass, or k when the box carries the spring) and a random sign. The edit is ±0.67
  normalised units: friction ±0.10, mass ×/÷1.49, k ×/÷1.60. The model's belief is clamped at every pass to
  (posterior mean + edit); the reference is the same clamp without the edit. The physical counterfactual is a
  MuJoCo re-simulation of the same window from its true start state with the true property changed by the same
  physical amount (reference: the replayed factual window).
- Planner (identical for every model): slide-to-target for the last box, the goal is a world-coordinate position the true world reaches with one grid action
  (decision 1 above). F is chosen from 29 values in [1, 15] N with T = 0.2 s, minimising |predicted final x_L − goal|.
  Outcomes are realised in MuJoCo. Oracle ΔF = argmin over the same grid in the true and counterfactual worlds.

**Acceptance criteria for the causal test (LBS-law; must hold in each of the 3 seeds).**
- A1 future motion: sign agreement of Δ(final position) for the edited box ≥ 0.85, wherever |Δsim| > 2 mm;
  Spearman corr(Δmodel, Δsim) ≥ 0.6. For the other boxes, sign agreement ≥ 0.75.
- A2 intermediate physical states: sign agreement at τ = 0.3 s and 0.6 s for the edited box ≥ 0.80. Sign agreement of
  the change in contact impulse on the edited box's incoming relation ≥ 0.75 (|Δsim| > 0.005 N s).
- A3 chosen action: sign agreement of ΔF (edited − unedited belief) with the oracle ΔF ≥ 0.80, over episodes where
  the oracle ΔF ≠ 0.
- A4 realised outcome: in the counterfactual world, the action chosen with the edited belief has lower realised error
  than the action chosen without the edit: the paired mean difference has a 95% CI upper bound < 0.
- A5 anchor removal (`no_anchor`, all rows with k ≥ 1): the predicted final-position sd increases (paired CI > 0), and
  80% coverage of final positions lies in [0.70, 0.90] both with and without anchors.
- A6 delayed evidence: the last window's evidence becomes available only from pass 2 of 4. The parameter abs error
  is ≤ 1.10 × the on-time error, and lower than never-available (paired CI < 0).
- A7 edge blocking (L ≥ 2): blocking relation (B1, B2) suppresses B2's predicted displacement to ≤ 20% of the
  unblocked value in ≥ 80% of rows where the unblocked prediction is > 1 cm. The change in B1's displacement agrees
  in sign with the simulator (boxes 2.. removed) in ≥ 0.80 of rows.
- A8 held-out L = 4: LBS-law final-position error (rows k ≥ 1) on `test_L4` ≤ 1.5 × its TEST error.

The causal test is **passed** if A1–A4 hold. A5–A8 are reported as pass/fail individually.

**Matched comparisons (paired over episodes, seeds averaged per episode, 95% bootstrap CI).**
- C1 LBS-law vs V1 on (i) the own-box sign agreement (A1 indicator), (ii) action sign agreement (A3), (iii) the
  closed-loop realised error, and (iv) final-position error. Prediction: LBS-law better on (i) and (ii).
- C2 LBS-law vs LBS-nolaw on the same four. Prediction: law regularisation improves (i) and (ii) (counterfactual
  consistency); no directional prediction for (iii) and (iv).
- C3 LBS-law vs classical (CEM + analytic) on (iii) and (iv), and on A1/A3. No directional prediction: the
  classical estimator has the exact model class.
- C4 relation ablation: `dense_law_s0` on A7 and A8 (expected: no edge-blocking control, worse L = 4).
- C5 recurrence depth: LBS-law final error and parameter error at K = 1, 2, 4, 8 (trained with K ∈ 2..5), descriptive.
- C6 modalities (descriptive): final error and parameter recovery with tracks only (`no_proprio`), force/proprio only
  (`no_track`), and all.

Nulls and failures are reported as such. A V1 failure on A6 is expected by construction: its typed estimates are
fixed before the loop. It is reported, not counted as a win.

## §0c. Frozen matrix: first read (seed 0 of LBS-law; the full table goes in §0f when all seeds are evaluated)

LBS-law s0 on TEST (`results/v3/eval/lbs_law_s0_test_*`): **fails the causal test**, while its prediction is decent.

| what | LBS-law s0 | classical (CEM + analytic) | oracle (labelled) |
|---|---|---|---|
| final-position error (K = 4) | 16.0 mm (K=1 43.8, K=2 25.8, K=8 18.2) | ≈ 13 mm | 3.7 mm (VAL) |
| own-box sign agreement (A1) | 0.52 | 0.95 | 0.98 |
| Spearman (A1) | 0.10 | 0.92 | 0.98 |
| model Δ / sim Δ | 0.06 | 0.94 | 1.00 |
| τ = 0.3 / 0.6 s sign agreement (A2) | 0.54 / 0.51 | 0.99 / 0.96 | 0.99 / 0.98 |
| contact-impulse sign agreement (A2) | 0.88 | – | – |
| action sign agreement (A3) | 0.10 | 0.80 | 0.83 |
| counterfactual-outcome Δ (A4) | −0.29 mm [−0.85, +0.22] | −5.8 [−7.6, −3.9] | −8.2 |
| closed loop | 9.7 mm / 79% | 7.4 mm / 87% | 4.1 mm / 96% |
| delayed evidence: param error (on-time / late / never) (A6) | 0.511 / 0.534 / 0.534 | – | – |
| edge blocking: B2 suppressed (A7) | 0 / 82 rows | – | – |

Other measurements for LBS-law s0:
- Modalities, final error: tracks-only 19.4 mm, force-only 17.5 mm, all 16.0 mm. Anchor removal: 16.2 mm.
- `test_L4`: 25.0 mm, 1.56× TEST, so A8 narrowly fails.

The edit moves the predicted motion by only 6% of the physical effect. Writing back per pass, the law loss on
intermediate states and law-only counterfactual rows did not make the named slots causal. The future-motion
beliefs are carried by the evidence and state paths. This is the same failure as the v1 friction slot. My diagnosis:
the law gradient of a Δμ = 0.1 edit is ≈ 0.01 per sliding interval (normalised), so laws can be satisfied with
negligible motion change. Downstream boxes predict motion from their own evidence history, which explains A7.

## §0d. Redirect after the first read (lead, 2026-11-27): two EXPLORATORY arms, each pre-registered separately on fresh seeds

The frozen matrix (§0b) is finished as registered. The exploratory arms use fresh TEST seed sets: `test2` (seeds
4 000 000+, 600 episodes) and `test2_L4` (5 000 000+, 300 episodes). They share the matched comparators.
- **Identifiability check first (lead item iii)**, VAL, `scratchpad ident.py`: a well-converged CEM (256 × 20
  iterations) on the same observations, non-spring episodes.
  - L = 1: corr(mass) 0.95 and corr(friction) 0.68.
  - L = 2: individual masses 0.66 / 0.54, total mass 0.90, frictions 0.84 / 0.79.

  Mass is identifiable under the fixed-force excitation. Individual masses of boxes that move together are only
  partly identifiable, as physics predicts. **The excitation is adequate; the learned model's 0.27 is a model
  problem.** Note: the pre-registered classical comparator (64 × 8 CEM) is under-converged (VAL mass abs error 0.57
  normalised). It stays as registered, and a converged variant will be reported separately.
- **Arm I (tests the PI's thesis: a learned loop).** `belief.LBSI` (`arm_i`):
  - *Non-bypass bottleneck.* The learned loop over evidence writes only the parameter slots and the τ0 state. Its
    future fields are reset after every pass. A learned `Rollout` (graph-masked attention over entity and relation
    tokens, no evidence access) computes all future beliefs from the slots, the τ0 state and known inputs.
    Positions integrate by construction. A unit check confirms that with the slots fixed, future predictions are
    invariant to the evidence and respond to slot edits.
  - *Interventional supervision* (`v3_chain cfgen`). For every training-episode window: one random named edit
    (box; mass, friction or k; ±0.67 normalised) and the **MuJoCo re-simulation** from the true start state. In 30%
    of rows the rollout gets the edited true parameters and is trained on the counterfactual trajectory.
  - *v2 recipe for the rest.* The rollout gets the TRUE parameters with p = 0.5, otherwise the DETACHED loop
    estimate. The loop is trained only by parameter and τ0-state NLL. The law loss applies to rollout outputs.
- **Arm P (hybrid; reported as "the loop can host a physics engine", not "attention learned physics").**
  `kind=proj`: after every write-back, 4 learned-step gradient steps on the law residual move the state beliefs
  (x, v, contact force) given the current parameter beliefs.
- The same budget (4000 × 512), VAL-only pilot, then a separate pre-registration commit before any `test2` run.
  The Inkling port waits until an arm passes A1–A4 on VAL.

### §0d.1 VAL attempt log (every exploratory VAL evaluation; the count is cited in the arm pre-registration)
`test2` / `test2_L4` were generated (sha256 in the manifest) and have not been evaluated by anything: no model, no
classical, no oracle, no simcf.

| # | arm / run | config | VAL result | what changed next |
|---|---|---|---|---|
| 1 | arm_i_s0 | LBSI, cf-frac 0.3, tf 0.5, law 1, 4000 × 512; clamps applied inside the loop | A1 0.91 / Spearman 0.80 / other 0.88; A2 0.98 / 0.94 / impulse 0.88; A3 0.54 (37% ties); A4 −3.3 mm [−6.0, −0.7]; A6 fail; A7 B2 suppressed 4%; final 16.8 mm | the clamp leaked into the τ0 state (plan changed in 51% of episodes when clamping at the posterior mean) → do() acts only on the rollout's slots (#2) |
| 2 | arm_i_s0 (same weights) | eval change only: do() at the rollout input | pending | – |
| 3 | proj_s0 | LBS + 4 law-projection steps (η init 0.3) | **null**: final error 62.9 mm, parameter corr ≈ 0, own-box sign 0.59, edit effect 0.8% of physics, A3 0.19, A4 +0.7 mm [−5.3, +7.3], closed loop 52 mm | not pursued further now (hybrid; lead priority is arm I); a different projection design would need its own attempt |
| 4 | arm_i2_s0 | #2 + relation do()-pairs (rel-frac 0.15) + late-evidence augmentation (0.2) | pending (training queued) | – |

Classical (CEM + analytic) goes through the same do()/A3/A4 harness on `test2`, so the comparison is matched.
The converged CEM classical (`classical_conv`, 256 × 20) on VAL gives own-box sign 0.94, Spearman 0.87, action 0.73,
counterfactual outcome −3.6 mm [−6.2, −0.8] and closed loop 9.6 mm / 83%: no better than the registered 64 × 8 CEM.
So A3 ≥ 0.80 lies above both classical estimators and near the oracle (0.82) on VAL.

## §0e. Pre-registration: exploratory arm I on fresh seeds (`test2`, `test2_L4`), frozen before any test2 run

Status: committed on its own. `test2` (seeds 4 000 000–4 000 599) and `test2_L4` (5 000 000–5 000 299) have
been generated and never evaluated by anything (no model, classical, oracle or simcf).

**Forking paths, disclosed.**
- VAL attempts before this freeze: 4 (§0d.1).
  1. arm_i_s0 with in-loop clamps;
  2. arm_i_s0 with the clamp at the rollout input (eval change only);
  3. proj_s0 (arm P), a VAL null, not carried to test2;
  4. arm_i2_s0 (relation do()-pairs + late-evidence augmentation).
- One additional VAL reference: `classical_conv`.
- Candidates 2 and 4 were not yet scored on VAL when this was written.

**Selection rule (deterministic, VAL only; `scripts/v3/select_arm.py`).**
- Score each candidate {`arm_i` = attempt 2, `arm_i2` = attempt 4} by the number of A1–A4 criteria it passes on
  VAL, using the §0b thresholds. The higher score wins; a tie goes to `arm_i2`. The output goes to
  `results/v3/eval/arm_selection.json`.
- The chosen configuration is trained for seeds 1 and 2 at the same budget (4000 × 512). It is evaluated on test2
  **whether or not it passed on VAL**.

**Evaluation.** Identical to §0b (same harness, edit-plan seed 12345, planner, n = 200 causal episodes of `test2`
and 200 of `test2_L4`, K = 4), applied to:
- the chosen arm (3 seeds);
- the frozen-matrix checkpoints LBS-law, V1 and LBS-nolaw (3 seeds each), as matched comparators;
- classical (registered 64 × 8 CEM) and the labelled oracle.

**Criteria.** A1–A8 exactly as in §0b, required in each of the 3 seeds of the chosen arm.

**Comparisons (paired over test2 episodes, seeds averaged, 95% bootstrap CI).**
- D1: chosen arm vs LBS-law on own-box sign agreement, action sign agreement, closed-loop error and final error.
  Prediction: arm better on the first two.
- D2: chosen arm vs classical on the same four (no directional prediction).
- D3: chosen arm vs V1 (prediction: arm better on own-box sign agreement).

**Interpretation, fixed now.**
- A pass by arm I is evidence that a learned loop with named slots can be made causally correct when it is trained
  with real interventional data behind a non-bypass bottleneck.
- It is not evidence that law regularisation alone does it (§0c says it does not).
- The rollout is a learned network; MuJoCo only produces the training pairs. A failure is reported as such.
- The Inkling port waits for a VAL pass and the lead's approval.
