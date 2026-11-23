# v2: industrial multi-robot, multi-environment physical memory that WORKS

started 2026-11-23 ~13:30Z. all heavy execution on gb10-direct (see COORDINATION.md for machine rules; they still apply).

## why v1 was not enough
one box, one table, one robot, scripted position-servo controller with no actuator-effort cost. the controller jerked
(3 m/s wrist flip) and stalled (convergence waits) and its speed scatter swamped the physics the model was meant to use.
the model never learned latent properties because the benchmark barely needed them.

## definition of "works" (gates; all measured on HELD-OUT seeds, scene layouts, object sets and at least one held-out robot)
G1 control: every robot tracks min-jerk task-space trajectories under torque limits with an explicit effort/jerk cost:
   tracking error < 1 cm mid-stroke, release-speed ratio p10-p90 within ±7%, no convergence-wait stalls, peak joint
   jerk and torque-rate logged; zero integrity violations (no welds / pose writes / applied object forces).
G2 breadth: >= 12 industrial task families (>= 5 bulk, >= 5 fine) and >= 6 robots (>= 3 arms with different grippers,
   1 bimanual, 1 humanoid floating-base, 1 mobile manipulator), procedurally randomized (geometry, counts, mass,
   friction, restitution, clutter, sensor noise/occlusion/latency).
G3 model utility: the model-based planner (memory world model + MPC over skill parameters with task + effort + risk
   cost) beats (a) the scripted/heuristic skill baseline and (b) the same planner with a no-memory (generic) model on
   task success, with equal or lower actuator effort, on held-out conditions, with bootstrap CIs over episodes.
G4 latent physics: on tasks where hidden mass / friction / compliance / clearance matter, the model's typed estimate
   tracks the truth after interaction (corr >= 0.7) and interventions on it change predictions, actions and outcomes
   in the physically correct direction.
G5 transfer: the held-out robot and held-out task family reach >= 70% of in-distribution success with the frozen world
   model + its embodiment adapter (report adapter data used).
report all failures; gates are engineering targets, not claims.

## architecture
embodiments (src/physmem/embodiments/): one interface for every robot
  build(spec, mount) -> handles ; Controller.step(model, data) -> ctrl (torque) ; capability descriptor (reach, payload,
  max speed/force, gripper type) ; proprio sensors ; effort/jerk/torque-rate meters.
  control: operational-space / task-space inverse-dynamics with gravity+coriolis compensation, torque limits, min-jerk
  task trajectories with pos/vel/acc feedforward and orientation slerp, impedance + guarded/force modes for contact,
  null-space posture; humanoids reuse CommandAGI wbc.py (QP, balance) where possible.
skills (src/physmem/skills/): parameterized primitives shared by all robots: reach, push, sweep (bulk), pick, place,
  insert (fine, guarded + compliant search), open/close articulated, handover, regrasp. each returns logs incl. effort.
envs (src/physmem/envs/): procedural MjSpec scene generator: station types (bin/clutter cell, conveyor line, pallet/
  depalletize, kitting bench, assembly/insertion fixture, shelving/aisle, machine tending with articulated door/drawer,
  bulk-material bin with many small parts). task families with success predicates + evaluator-only labels.
  observation = ObservationEvent streams (schema.py) from synthetic sensors (pose tracker w/ noise/occlusion, wrist
  F/T, joint encoders, contact/tactile, optional depth). batch headless generation over CPU processes.
model (src/physmem/models/v2/): object-centric typed memory over arbitrary entity counts + embodiment tokens; explicit
  per-interaction system identification (mass, friction, compliance, clearance) supervised from evaluator labels, fused
  by precision; skill-conditioned outcome + effort + risk prediction; uncertainty; planner = CEM/MPC over skill params.
evaluation (src/physmem/v2_eval.py): paired held-out episodes per (robot, task family), baselines, effort, CIs.
