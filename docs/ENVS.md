# ENVS: v2 industrial stations and task families (agent ENVS)

Status 2026-11-23 ~21:30Z. All numbers below were measured on gb10-direct with the **test-only probe robot** unless
a row says otherwise (real-embodiment results: section "Robots"). Raw data: `results/v2/envs/sensitivity.json`,
`results/v2/envs/generation.jsonl`, `data/v2_envs/manifest.json` (remote), stills `results/v2/envs/*.png`.

Code `src/physmem/envs/` (owned by ENVS): `scene.py` (MjSpec builder), `stations.py` (bench + mounts),
`families_bulk.py`, `families_fine.py`, `families_logistics.py`, `base.py` (StationEnv = api.IndustrialEnv),
`sensors.py` (ObservationEvent streams, ActorView), `props.py` (extended Prop vocabulary), `probe.py` (TEST-ONLY
robot + reference skills), `robots.py` (adapter to ROBOTS' embodiments), `splits.py`. Tests `tests/test_envs.py`
(47 tests). Scripts: `envs_smoke.py`, `envs_trace.py`, `envs_sensitivity.py`, `envs_generate.py`, `envs_render.py`.

## API (implements `physmem.v2.api.IndustrialEnv`)

```python
from physmem.envs import make_env, FAMILIES
from physmem.envs.splits import make_tasks
from physmem.v2.api import TaskSpec
env = make_env("bin_clearing")                    # optional robot_factory=callable(env, name)->(robot, skills)
task = make_tasks("bin_clearing", "train", n=1, robot="probe")[0]      # or TaskSpec(...) directly
events = env.reset(task)                          # actor-legal initial ObservationEvents
env.goal()                                        # actor-legal: description, CAD regions (+ joint_goal / sort_rule)
cands = env.candidate_skills(belief, n=16, rng)   # feasible SkillCalls from actor-legal info only
out = env.execute(cands[0])                       # StepOut(events, result, done, success, metrics, labels)
env.scripted()                                    # scripted heuristic BASELINE's next SkillCall (None = it gives up)
img = env.render()                                # RGB overview (EGL on the remote)
env.capture_depth()                               # optional depth image from the tracker camera (not in actor loop)
env.idle(T)                                       # harness: let the world run T s (conveyor) without a skill call
```

* `reset`: builds a fresh MjSpec scene from `seed_of(task)` = sha256(family|seed|split) (params do not change the
  rng stream, so a params override only changes the overridden quantity); settles physics with sensors off; resets
  the clock to 0; returns SIZE / SHAPE / REGION + first tracker and proprio packets. Deterministic (tested).
* `execute`: one skill to completion. Physics advances only through robot actuator torques/forces (and the env-owned
  conveyor drive). `done` = success or fatal (object on the floor, part in the reject chute, ...) or skill budget.
* `metrics`: `progress` in [0,1] (family-specific), `progress_delta`, `collisions` (robot contacts whose normal
  force crossed 150 N, rising edges), `max_robot_contact_n`, `integrity_violations`, `drops` (objects that touched
  the floor), `sim_instabilities` (MuJoCo NaN auto-resets; fatal), `effort_cost` (api.EffortMeter.cost of this skill), `n_calls`, `t`, `fatal` + family extras.
* `labels` (EVALUATOR-ONLY): `hidden` per entity (mass, mu, mu_grip, restitution, timeconst, heavy/oily flags,
  clearance, chamfer, damping, frictionloss, content_mass, head_imbalance, young ...), `true_state` (poses, joint
  values), `contacts` (true contact graph), `sensor_cfg` (this episode's noise/latency/dropout), `held`,
  `task_hidden` / family extras (e.g. `belt_speed`, `depth_frac`, `err_xy`).
* Hidden-physics overrides for studies: `TaskSpec.params["hidden"] = {"mass": .., "mu": .., "mu_grip": ..,
  "restitution": .., "timeconst": ..}` (applied to every part/compound), plus family params listed below.
* **Integrity monitor** (every physics step): before the step, object qpos/qvel must equal their values after the
  previous step (no pose/velocity writes) and `xfrc_applied` / `qfrc_applied` on objects must be zero, no active
  weld/connect equality on an object; after the step, object qpos must equal `mj_integratePos(prev, qvel_new)`
  (catches teleports even when another module owns the stepping loop: `env.pre_physics()` / `env.post_physics()`).
  Counted in `metrics.integrity_violations` (0 in every run so far; the monitor is tested to fire on a teleport and
  on an applied force). Suction = MuJoCo `adhesion` actuator on the robot's cup (contact-mediated, capped force).

## Observations (record layer `schema.ObservationEvent`)

Standard props (ingestable by `memory.WorldBelief`): POSE2, SIZE, CONTACT. Extended props (`envs/props.py`,
`XProp` values >= 100; use `props.split_events` or `props.ingest(belief, events, now)` which ingests the standard part
and returns the rest): POSE3, WRENCH, JOINT, GRIPPER, SHAPE, TACTILE, EE_POSE, REGION (+ label-only ids
GRIP_FRICTION, RESTITUTION, COMPLIANCE, CLEARANCE, JOINT_FRICTION, never emitted).
**PROPOSAL to the lead:** promote these into `schema.Prop` with dims `XPROP_DIM` (POSE3 7, WRENCH 6, JOINT 2,
GRIPPER 3, SHAPE 8, TACTILE 4, EE_POSE 7, REGION 7) and add GRIP_FRICTION/RESTITUTION/COMPLIANCE/CLEARANCE/
JOINT_FRICTION to LATENT_PROPS.

| sensor | entity | props | rate | noise / artefacts (per-episode random; in labels.sensor_cfg) |
|---|---|---|---|---|
| pose tracker (fixed camera) | each movable object | POSE3 + POSE2 | 10 Hz (bulk_sweep: 2 Hz, change-triggered 3 mm / 2 s keyframe) | pos sigma 1-3 mm (fine families 0.5-1.5 mm), yaw 0.5-2 deg, latency 30-90 ms + 10 ms jitter, dropout 0-8 %, line-of-sight occlusion (mj_ray camera -> entity centre: robot, bin walls, other parts block it) |
| tracker on links | door / drawer | JOINT (q) | tracker rate | 0.01 rad / 1 mm |
| joint encoders | robot/<joint> | JOINT (q, dq) | 25 Hz | 1e-5 |
| FK of encoders | robot/hand | EE_POSE (DERIVED) | 25 Hz | - |
| wrist F/T | robot/hand | WRENCH (world axes, tared at t=0) | 25 Hz | 0.2-0.6 N |
| gripper | robot/hand | GRIPPER (width, commanded force, suction level) | 25 Hz | - |
| tactile pads | robot/hand | TACTILE (left, right, palm, sum normal force) + CONTACT | 25 Hz | 0.5 N |
| one-shot at t=0 | entities / CAD | SIZE (sigma 1 mm), SHAPE (one-hot), REGION (DERIVED, source "cad") | - | - |

Events are pushed into `env.view` (an `ActorView`, respects `available_at`) as they are emitted, so skills react
to fresh packets mid-skill (moving-part intercept, in-hand correction). Skills and `candidate_skills` read ONLY the
ActorView (+ belief POSE2 if a WorldBelief is passed + CAD regions/meta); tested by corrupting true poses.

## Robots

TEST-ONLY `probe` (`envs/probe.py`): Cartesian gantry x/y/z + yaw, PD + min-jerk / trapezoid profiles, torque/force
limits (150/150/250 N, 15 Nm), effort meter (api.EffortMeter); tools: parallel gripper (force-controlled, coupled,
gearbox damping), suction cup (adhesion actuator, 150-300 N), sweep blade. Reference skills: reach(+dwell), pick
(static or moving-target intercept), place (with tracker-based in-hand correction / yaw alignment), push, sweep,
insert (compliant force-controlled descent + spiral search + yaw wiggle), open/close (handle path from CAD + tracked
joint). Probe results are NOT robot results. Every station defines mounts for `arm`, `bimanual` (two),
`humanoid` (floor-standing), `mobile_manipulator` (floor base pose) and `gantry` (probe).
Real embodiments (ROBOTS: panda, ur5e_2f85, gen3_2f85, z1, so100, aloha, tidybot, g1_dex3, g1_dex3_fixed):
`envs/robots.py` adapter, see "Robots: real embodiments" below. The tracker camera sits on the FAR side of every
station (looking back at the robot) for all robots, so occlusion geometry is robot-independent.

### Robots: real embodiments (`envs/robots.py`, `scripts/envs_robot_smoke.py fams robots seeds`)
`make_robot_adapter(env, name)` -> `(EmbodimentRobot, EmbodimentSkills)`; `StationEnv.reset` calls it for any
robot name other than `probe*`. Build: `E.scene_defaults(spec, timestep=<family dt>)` (1 ms fine families run
fine), fixed-base arms on a static pedestal at the station's `arm` mount, ALOHA at the bench centre (arms at the
bench ends), G1 at the `humanoid` mount (0.15 m behind the bench edge), TidyBot at `mobile_manipulator`.
Object<->pad friction = explicit pairs with the object's hidden `mu_grip` on the collision geoms of
`GripperCfg.pad_bodies` (2F-85: right/left pad1/2 + followers), same convention as the probe. `grip_force` keeps the
probe's meaning (N per pad): fingers close at full drive effort, then the effort is regulated to
`cfg_force * clip(grip_force / GRIP_N_MAX, 0.05, 1)` (GRIP_N_MAX measured for panda 35 N and 2F-85 44 N per pad;
other robots: spec guesses). Skills: pick/place/push/sweep/insert/reach -> `physmem.skills.run_skill` with
`ObjectEstimate`s built only from `env.est_pose/est_pose_avg/est_size/region/handle_path`; open/close composed from
ROBOTS controller primitives (top-down pinch of the handle tab + impedance pull along `env.handle_path`); the skill
loop calls `env.pre_physics()` / `env.post_physics()` around every mj_step (integrity + sensors). Bimanual/humanoid:
the arm nearest the target picks and keeps the part.

Scripted-baseline successes / seeds tried (adapter agent run, before the last station fixes - door tab height,
humanoid/mobile mounts, far-side camera were applied after this table; re-run numbers below if present):

| robot | bin_clearing | singulation | peg_insert | drawer_open | fixture_place | door_open | kitting_pocket |
|---|---|---|---|---|---|---|---|
| panda | 0/3 (progress .2-.38) | 2/3 | 0/3 | 2/3 | 3/3 | 0/3 (tab low) | 0/3 |
| ur5e_2f85 | 0/3 (progress .14-.5) | 1/3 | 2/3 | 3/3 | 3/3 | 1/3 | 0/3 |
| aloha (bimanual) | - | 2/2 | 0/2 (dropped) | 1/2 | 0/2 | - | - |
| tidybot (mobile) | - | 0/2 | 0/2 unreachable | 0/2 | 0/2 unreachable | - | - |
| g1_dex3_fixed | - | 1/1 | - | - | 0/1 unreachable | - | - |
| g1_dex3 (floating) | - | not finished (minutes/episode) | - | killed (too slow) | 0/1 unreachable | - | - |

**Round 2 (final code: ROBOTS calibrated `emb.set_grip_force`, event-based close, door tab at the panel top,
humanoid/mobile mounts 0.15/0.35 m from the bench, far-side camera, drawer pull impedance 2500 N/m + 2 cm overshoot;
`logs/envs_robots_arms.log`, seeds 0-2):**

| robot | singulation | peg_insert | drawer_open | door_open | fixture_place | bin_clearing | kitting_pocket |
|---|---|---|---|---|---|---|---|
| panda | 1/3 (prog .6/1/.6) | 2/3 | 1/3 (stalls at 56-68 %) | 3/3 | 3/3 | 0/3 (prog .25/.2/0) | 0/3 |
| ur5e_2f85 | 0/3 (prog 1.0 but a drop / off-area -> fatal) | 0/3 (was 2/3 in round 1) | 3/3 | 0/3 (top tab, grasp fails) | 3/3 | 0/3 (prog .63/.3/.43) | 0/2 |

Suction embodiments (`ur5e_suction`, `fr3_suction`) on depalletize / tote_transfer / conveyor_sort: 0/2 each -
the skills return `unreachable` immediately (pallet/aisle/conveyor stations now have pedestal `arm` mounts at
0.7-0.8 m; still unreachable: the adapter's pick path for suction tools needs work). NOT WORKING YET.
Integrity violations: 0 in every real-robot episode. Known real-robot failure causes: guarded descent trips on bin
walls / neighbours for 12-30 mm parts with Panda/2F-85 hands (bin_clearing); a failed panda insert leaves the peg
tilted; TidyBot's home TCP (z 0.54) is below the 0.76 m bench (reset collisions up to 7.4 kN on wrist links,
unresolved, reported to ROBOTS); floating-base G1 episodes take minutes. Adapter workarounds ROBOTS should know:
2F-85 free close ~0.8 s vs skill wait 0.55 s (doubled per instance); 2F-85 linkage stalls below ~40 % effort.

## Stations (procedural MjSpec; primitive / procedural geometry only, no external assets)

(a) clutter bin -> tote cell: 6-60 parts (box, cylinder, capsule, sphere, ellipsoid, random convex-hull mesh), sizes
24-60 mm, density 1000-8000 kg/m3, heavy outliers x4-8, oily parts, restitution 0-0.4, 20 % compliant (solref
0.02-0.04). (b) conveyor: DRIVEN belt = long plate on a slide joint velocity-servoed by a force-limited actuator
(4000 N s/m, 600 N), parts ride it by friction; sort bins A/B on pedestals, reject chute at the belt end.
(c) EUR pallet 1.2 x 0.8 x 0.144 with 1-3 layers of 2x2 cartons (hidden heavy ones), outfeed table.
(d) kitting bench: source tray + kit tray with 2-4 tight pockets (1-3 mm clearance). (e) insertion fixture: square
peg / thin connector module / tool shank into sockets with 0.5-3 mm clearance and 0-3 mm 45-degree chamfers.
(f) staggered two-tier flow rack with KLT-style totes (0.30 x 0.20 x 0.12, hidden ballast) + cart.
(g) machine tending: tool-cabinet drawer (slide joint, damping, frictionloss breakaway, contents mass) and hinged
machine door (hinge damping, frictionloss, mass) with pinch-tab handles. (h) bulk material on a lipped bench: 40-100
nuts / washers / blocks / capsules. (i) deformable pouch: MuJoCo 3-D flexcomp grid (3x4x2, Young 2e3-3e4 Pa, 1 ms step).
(j) pushing / singulation table with 4-7 packed boxes/cylinders.
Dimensions from the CommandAGI factory library where it had them (bench top 0.76 m, fixture tolerance idea,
tote/pallet sizes); the success-predicate vocabulary mirrors `packages/domain/world/factory-tasks.js` (part in box,
joint_ge, upright, below_m failure, collisions_over_n). Stills: `results/v2/envs/<family>_probe_{reset,end}.png`,
`contact_sheet_probe.png`.

## Task families (14: 6 bulk, 7 fine, 1 deformable)

Success predicates are evaluated on TRUE state after every skill (`env.success()`), documented in each class's
`success_doc`; hidden variables in `hidden_doc`.

| family | cat. | stn | success predicate | hidden variables (labels) | which decision they change |
|---|---|---|---|---|---|
| bin_clearing | bulk | a | every part's centre inside the tote region | mass (heavy x4-8), mu_grip (oily 0.08-0.2), mu, restitution, compliance | grip force / lift speed per part (hold iff 2 mu_g F >= m(g+a)) |
| singulation | bulk | j | every object >= 3 cm from every other (mj_geomDistance), all on the work area | mu object<->bench 0.15-0.9, mass 0.1-3 kg | push speed / distance (post-release slide ~ v^2/2 mu g) |
| bulk_sweep | bulk | h | >= 85 % of parts inside the target region | episode mu mean 0.12-0.8 (+-15 % per part), restitution | sweep stop point / stop time (parts slide on after the blade stops) |
| depalletize | bulk | c | all cartons upright on the outfeed, no drops | carton mass 1-6 kg + hidden heavy 12-22 kg (p .25), carton friction | suction level / lift speed (hold iff F_suction >= m(g+a)) |
| conveyor_sort | bulk | b | every part in the bin of its class (cylinder->A, box->B); reject/floor = fatal | belt speed 0.03-0.12 m/s (estimable), mass, mu_grip (oily), mu | intercept lead (belt speed), grip force |
| tote_transfer | bulk | f | all totes upright on the cart | content mass 0.3-14 kg (invisible ballast), mu_grip 0.4-0.9, tote<->shelf mu | grip force / lift speed |
| peg_insert | fine | e | peg bottom >= 80 % of hole depth below the mouth, released | clearance 0.5-2 mm, chamfer 0-3 mm (0 on 35 %), mu 0.2-0.9 | straight vs spiral search, search radius, down force |
| connector_insert | fine | e | module seated >= 80 % of slot depth | clearance 0.5-1.5 mm (10 mm thin axis), chamfer 0-1 mm, mu 0.3-1.0, slot yaw | yaw wiggle / compliance (WEAK, see below) |
| tool_return | fine | e/g | tool shank >= 80 % into its holder | head mass 0.1-0.9 kg with hidden left/right imbalance (CoM offset), mu_grip, clearance 1-3 mm | grip force / grasp height (in-hand tilt) |
| fixture_place | fine | machine table | fixture centre within 2 mm and yaw within 3 deg of the mark, resting, released | mass 0.3-4 kg, mu_grip 0.2-1.0, mu 0.15-0.8 | grip force, carry speed (in-hand slip) |
| kitting_pocket | fine | d | every part fully seated in its pocket (inside, bottom <= floor + 3 mm) | pocket clearance 1-3 mm, mu, heavy parts, mu_grip; 1-3 mm tracker noise | plain place vs compliant insert, search radius |
| drawer_open | fine (artic.) | g | drawer >= 85 % of 0.20 m travel | frictionloss 2-40 N, damping 5-60 N s/m, contents 1-12 kg, handle mu_grip | grip force vs pull force (pinch tab: capacity 2 mu_g F), pull speed |
| door_open | fine (artic.) | g | door >= 1.2 rad | hinge frictionloss 0.2-4 N m, damping 0.1-3, mass 1-6 kg, handle mu_grip | grip force, swing speed |
| pouch_transfer | deformable | i | pouch centroid inside the tote region, released | Young 2e3-3e4 Pa, mass 0.1-0.6 kg, mu | grip force window |

### Hidden-variable sensitivity (fixed action, outcome across the hidden range; 6 seeds per point, probe)
`scripts/envs_sensitivity.py all 6 3` (final code, far-side camera; 87 s on 3 procs, 0 errors). Every protocol fixes the skill parameters (no retries,
no adaptation) and sweeps ONE hidden variable through `TaskSpec.params`.

| family | fixed action | hidden variable: outcome |
|---|---|---|
| bin_clearing | pick, grip 15 N, lift 0.3 m/s, mu_g 0.5 | mass 0.1/0.5/1.0/1.5/2.5 kg: P(hold) 1/1/1/0/0 |
| bin_clearing | same, mass 0.5 kg | mu_grip 0.08/0.15/0.3/0.6/1.0: P(hold) 0/0/1/1/1 |
| singulation | push 0.6 m/s x 5 cm | mu 0.15/0.3/0.5/0.7/0.9: displacement 0.203/0.120/0.086/0.068/0.052 m (3.9x) |
| bulk_sweep | one sweep 0.5 m/s, 0.1 s stop | mu 0.12/0.25/0.4/0.6/0.8: fraction in target 0.27/0.43/0.47/0.49/0.50 |
| depalletize | suction 150 N, lift 0.25 m/s | carton 2/6/10/13/16/22 kg: P(hold) .67/.67/.67/0/0/0 |
| conveyor_sort | intercept assuming 0.075 m/s | belt 0.03/0.05/0.075/0.10/0.12 m/s: P(pick) 1/1/1/0/0 |
| tote_transfer | grip 40 N, mu_g 0.6 | content 0.5/2/4/7/10/14 kg: P(hold) 1/1/0/0/0/0 |
| peg_insert | straight insert, no search, 6 N | no chamfer, clearance 0.5/1/1.5/2 mm: P .00/.17/.50/.50; 2 mm chamfer: .83 at every clearance |
| connector_insert | straight, no wiggle, 15 N | clearance 0.5/1/1.5 mm: mu 0.3: P .50/.83/.83; mu 1.0: .33/.67/.83 (moderate) |
| tool_return | grip 12 N, mu_g 0.4, carry | head mass 0.1/0.3/0.5/0.7 kg: in-hand tilt 1.4/2.7/4.5/33 deg (0.9 kg: dropped) |
| fixture_place | grip 20 N, fast carry 0.8 m/s | mu_grip 0.15/0.25/0.4/0.7/1.0: P(+-2 mm) 0/0/1/1/1; mass 0.3/1/2/3/4 kg: 1/1/.83/0/0 |
| kitting_pocket | plain place (no insert) | clearance 1/1.5/2/3/4 mm: P(seated) .33/.33/.67/.83/.83 |
| drawer_open | grip 15 N, pull 0.15 m/s | frictionloss 2/5/10/20/40 N: travel 0.167/0.164/0.145/0.002/0.001 m |
| door_open | grip 8 N | frictionloss 0.2/0.5/1/2/4 N m: angle 1.17/1.17/1.16/1.15/0.03 rad |
| pouch_transfer | grip 12 N | Young 2e3/6e3/2e4/6e4 Pa: P(success) 1/1/0/.17 (stiff pouches slip out of a light pinch; 6e4 is outside the final train range) |

Values of 0.83 / 0.67 instead of 1.0 come from individual layout seeds failing for unrelated reasons (kept, not
filtered). connector_insert is the weakest family (an earlier run with the near-side camera showed almost no
clearance effect; with the final code tight + high-friction slots drop to 0.33) - treat its G4 evidence as weak.

## Scripted heuristic baseline (required baseline; `env.scripted()`)
Per family: pick the most accessible target from the tracker (highest in bin / most crowded / most downstream),
default skill parameters (e.g. grip 15 N, lift 0.3 m/s; suction 0.5; straight-ish insertion with 1.5 mm spiral),
and a "retry harder" rule after a failure on the same target (firm grip / full suction / larger spiral). It never
reads labels. Success per family / split and episodes per hour: see "Generation" below.

## Splits (`envs/splits.py`)
* train / val / test by ROOT SEED: sha256 bucket 80/10/10 decided before generation (no root in two splits).
* heldout_layout: test-bucket seeds; bench stations rotated 180 deg about the station centre (source/target sides
  swap relative to the robot; regions/CAD meta rotated consistently); pallet/conveyor/aisle stations: count shift
  (3 carton layers, 7 items, 4 totes, heavier pouch).
* heldout_objects: test-bucket seeds with object sets/ranges never used in train: bin_clearing mesh+ellipsoid parts
  (train uses box/cylinder/capsule/sphere), all-cylinder singulation, capsule-heavy bulk, 60 % heavy cartons/items,
  16 kg totes, 16 mm pegs, 1.2 kg tool heads, 5 kg fixtures, 4 kitting parts, 16 kg drawers, 8 kg doors, 5e4 Pa pouches.
* heldout_family per category: bulk -> tote_transfer, fine -> connector_insert (`make_tasks` refuses train/val).
* held-out robot: lead (TaskSpec.robot).
Sensor-noise / latency / dropout are randomized per episode in every split (labels.sensor_cfg).

## Generation (`scripts/envs_generate.py`)
Multiprocessing (one episode per task, deterministic per seed), shards `data/v2_envs/<family>/<split>.pkl.gz`
(list of episode dicts: task, all ObservationEvents, steps with call/result/effort/metrics/labels), manifest with
sha256 per file, root seeds, code hash and per-family throughput; one row per episode in
`results/v2/envs/generation.jsonl` (status observed). Final run (code as committed: far-side camera, 1 ms pouch,
per-episode 300 s wall guard): `--splits train,val,test,heldout_layout,heldout_objects --n 8 --procs 6` = 528
episodes in 316 s = **6010 episodes/h on 6 processes**; 0 integrity violations; 1 error (a bin_clearing episode hit
the 300 s guard - a hang not yet diagnosed; kept as an error row). Earlier runs kept: `data/v2_envs_run1`,
`results/v2/envs/generation_run1.jsonl` (near-side camera). Scripted-baseline success (n episodes), probe robot:

| bin_clearing | 0.12 (8) | 0.25 (8) | 0.12 (8) | 0.12 (8) | 0.00 (8) | - | 13.4 | 268 | 319 | 24.6 | 105389 | 0 |
| bulk_sweep | 0.50 (8) | 0.62 (8) | 0.75 (8) | 0.88 (8) | 0.00 (8) | - | 8.8 | 408 | 42 | 8.0 | 13986 | 0 |
| connector_insert | - | - | - | 1.00 (8) | 1.00 (8) | 0.88 (8) | 0.9 | 3889 | 18 | 3.1 | 5059 | 0 |
| conveyor_sort | 0.38 (8) | 0.50 (8) | 0.12 (8) | 0.00 (8) | 0.38 (8) | - | 2.4 | 1505 | 65 | 11.2 | 23401 | 0 |
| depalletize | 0.25 (8) | 0.00 (8) | 0.12 (8) | 0.00 (8) | 0.38 (8) | - | 4.4 | 825 | 110 | 12.1 | 37458 | 0 |
| door_open | 1.00 (8) | 1.00 (8) | 1.00 (8) | 1.00 (8) | 1.00 (8) | - | 0.2 | 21698 | 8 | 1.0 | 2391 | 0 |
| drawer_open | 1.00 (8) | 1.00 (8) | 0.88 (8) | 0.88 (8) | 0.88 (8) | - | 0.2 | 16858 | 10 | 1.3 | 2959 | 0 |
| fixture_place | 1.00 (8) | 0.88 (8) | 0.88 (8) | 1.00 (8) | 0.00 (8) | - | 0.9 | 3863 | 18 | 3.5 | 5172 | 0 |
| kitting_pocket | 0.50 (8) | 0.12 (8) | 0.25 (8) | 0.25 (8) | 0.12 (8) | - | 4.5 | 798 | 75 | 11.6 | 24395 | 0 |
| peg_insert | 0.88 (8) | 0.75 (8) | 0.88 (8) | 1.00 (8) | 1.00 (8) | - | 0.9 | 3901 | 18 | 3.1 | 5158 | 0 |
| pouch_transfer | 1.00 (8) | 1.00 (8) | 1.00 (8) | 0.62 (8) | 0.50 (8) | - | 2.3 | 1569 | 16 | 2.8 | 4822 | 0 |
| singulation | 0.88 (8) | 0.75 (8) | 0.88 (8) | 0.75 (8) | 0.62 (8) | - | 0.8 | 4776 | 18 | 5.5 | 7124 | 0 |
| tool_return | 1.00 (8) | 0.75 (8) | 0.88 (8) | 0.88 (8) | 0.00 (8) | - | 1.9 | 1890 | 34 | 4.7 | 9431 | 0 |
| tote_transfer | - | - | - | 0.00 (8) | 0.00 (8) | 0.12 (8) | 4.5 | 801 | 96 | 7.8 | 31483 | 0 |

[exited with code 0]

(`scripts/envs_summarize.py` -> `results/v2/envs/generation_summary.json`.) Throughput is per process; the episode
length is set by the skill budget and the baseline's retries, so a planner that succeeds sooner is faster.
Held-out-object rows at 0.00 (bin_clearing meshes/ellipsoids, 5 kg fixtures, 1.2 kg tool heads, 1.2e5 Pa pouches,
capsule-heavy bulk) show these shifts are hard for the fixed-parameter baseline - intended to need adaptation.

## What failed / limitations (honest list)
* connector_insert hidden variables are only moderately decision-relevant (see above).
* One bin_clearing episode (of 40) hung until the generator's 300 s wall guard; root cause not found.
* Suction embodiments are not yet usable through the adapter (skills return unreachable); real-robot success is
  low on bin_clearing / kitting / UR5e door + peg (see the robots tables).
* Scripted baseline is weak on depalletize (drops when a heavy carton slips on the first try), conveyor_sort
  (late intercepts let parts reach the reject chute), tote_transfer (heavy totes) and singulation (pushes that
  re-cluster): this is by design a baseline, but it also means these families have low baseline success.
* The probe has no wrist tilt; tasks are designed for top-down manipulation. Real-arm skills may need different
  approach directions for the door/drawer tabs.
* Velocity writes are detected only by the pre-step check (the post-step qvel-vs-qacc identity is not exact under
  elliptic-cone contact solves), so a module that owns the stepping loop is checked for pose writes, applied forces
  and welds, but not velocity writes.
* Tracker pose for the deformable pouch = vertex centroid, quaternion identity; no occlusion test for it.
* The 3-D flex pouch can go numerically unstable (MuJoCo NaN auto-reset). It runs at 1 ms with Young <= 3e4 Pa
  (train) / 5e4 Pa (heldout_objects); a reset-time scan of 8 seeds per split found 0 instabilities in train/val/
  test/heldout_layout and 2/8 in heldout_objects. Every instability is counted in `metrics.sim_instabilities` and
  makes the episode fatal (never silently kept).
* MjSpec defaults to degrees for joint ranges: `Scene` sets `compiler.degree = False` (an early bug clamped the
  probe yaw to +-4 deg).
