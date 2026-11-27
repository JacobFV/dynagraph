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
