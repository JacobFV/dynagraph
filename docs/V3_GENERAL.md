# V3: one general physical-memory model across domains (plan; started 2026-11-26)

Goal (PI): **a single model for all domains, which ideally generalizes beyond them.** Today we have separate model
lines: sprint compact, v2 synthetic, v2 stations and Inkling System I. V3 merges them into one checkpoint family. It
adds passive third-person observation, new physical domains and human bodies, and every claim is tested with
held-out domains.

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
