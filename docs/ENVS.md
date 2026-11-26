# ENVS: v2 industrial stations and task families (agent ENVS)

Status 2026-11-23 ~23:00Z (round 2). All numbers below were measured on gb10-direct with the **test-only probe robot** unless
a row says otherwise (real-embodiment results: section "Robots"). Raw data: `results/v2/envs/sensitivity.json`,
`results/v2/envs/generation.jsonl`, `data/v2_envs/manifest.json` (remote), stills `results/v2/envs/*.png`.

Code `src/physmem/envs/` (owned by ENVS): `scene.py` (MjSpec builder), `stations.py` (bench + mounts),
`families_bulk.py`, `families_fine.py`, `families_logistics.py`, `base.py` (StationEnv = api.IndustrialEnv),
`sensors.py` (ObservationEvent streams, ActorView), `props.py` (extended Prop vocabulary), `probe.py` (TEST-ONLY
robot + reference skills), `robots.py` (adapter to ROBOTS' embodiments), `splits.py`. Tests `tests/test_envs.py`
(59 tests). Scripts: `envs_smoke.py`, `envs_trace.py`, `envs_sensitivity.py`, `envs_generate.py`, `envs_render.py`.

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

All props are now `schema.Prop` (lead approved and applied the extension: POSE3=100, WRENCH, JOINT, GRIPPER, SHAPE,
TACTILE, EE_POSE, REGION; latent GRIP_FRICTION=120, RESTITUTION, COMPLIANCE, CLEARANCE, JOINT_FRICTION in
LATENT_PROPS, never emitted). `memory.WorldBelief` ingests every env event directly. `envs/props.py` is a thin alias
(`XProp is schema.Prop`, `split_events` returns everything as standard). Dense indexing: `schema.PROP_INDEX` /
`N_PROPS_V2` (`N_PROPS` stays 11 for v1).

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

## Task families (18: 7 bulk, 10 fine, 1 deformable; small_parts_pick / small_peg_insert added in round 3)

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
| twist_lock | fine | e | knob turned >= 90 deg (quarter-turn bayonet), released | breakaway torque 0.03-0.6 N m, damping, knob mu_grip | grip force / twist rate (pinch torque capacity ~2 mu_g F r) |
| machine_tending | fine (multi-step) | g | blank >= 80 % into the vise socket AND door closed (<= 0.08 rad), released | door frictionloss/damping/mass, blank mass + mu_grip, vise clearance 1-3 mm | door grip, pick force, compliant insert, sequencing (open -> load -> close) |

### Hidden-variable sensitivity (fixed action, outcome across the hidden range; 6 seeds per point, probe)
`scripts/envs_sensitivity.py all 6 5` (round-2 code; 76 s on 5 procs, 0 errors; connector protocol now 0.3-1.2 mm). Every protocol fixes the skill parameters (no retries,
no adaptation) and sweeps ONE hidden variable through `TaskSpec.params`.

| family | fixed action | hidden variable: outcome |
|---|---|---|
| bin_clearing | pick, grip 15 N, lift 0.3 m/s, mu_g 0.5 | mass 0.1/0.5/1.0/1.5/2.5 kg: P(hold) 1/1/1/0/0 |
| bin_clearing | same, mass 0.5 kg | mu_grip 0.08/0.15/0.3/0.6/1.0: P(hold) 0/0/1/1/1 |
| singulation | push 0.6 m/s x 5 cm | mu 0.15/0.3/0.5/0.7/0.9: displacement 0.203/0.120/0.086/0.068/0.052 m (3.9x) |
| bulk_sweep | one sweep 0.5 m/s, 0.1 s stop | mu 0.12/0.25/0.4/0.6/0.8: fraction in target 0.27/0.43/0.47/0.49/0.50 |
| depalletize | suction 150 N, lift 0.25 m/s | carton 2/6/10/13/16/22 kg: P(hold) .67/.67/.67/0/0/0 |
| conveyor_sort | intercept assuming 0.075 m/s | belt 0.03/0.05/0.075/0.10/0.12 m/s: P(pick) 0/.33/1/.67/0 (closed-loop re-aim since round 2) |
| tote_transfer | grip 40 N, mu_g 0.6 | content 0.5/2/4/7/10/14 kg: P(hold) 1/1/0/0/0/0 |
| peg_insert | straight insert, no search, 6 N | no chamfer, clearance 0.5/1/1.5/2 mm: P .00/.17/.50/.50; 2 mm chamfer: .83 at every clearance |
| connector_insert | straight, no wiggle, 15 N | clearance 0.3/0.6/0.9/1.2 mm: P .33/.50/.50/.67 (same for mu 0.3 and 1.0; module now 48-64 mm long) |
| tool_return | grip 12 N, mu_g 0.4, carry | head mass 0.1/0.3/0.5/0.7 kg: in-hand tilt 1.4/2.7/4.5/33 deg (0.9 kg: dropped) |
| fixture_place | grip 20 N, fast carry 0.8 m/s | mu_grip 0.15/0.25/0.4/0.7/1.0: P(+-2 mm) 0/0/1/1/1; mass 0.3/1/2/3/4 kg: 1/1/.83/0/0 |
| kitting_pocket | plain place (no insert) | clearance 1/1.5/2/3/4 mm: P(seated) .33/.33/.67/.83/.83 |
| drawer_open | grip 15 N, pull 0.15 m/s | frictionloss 2/5/10/20/40 N: travel 0.200/0.197/0.174/0.003/0.002 m |
| door_open | grip 8 N | frictionloss 0.2/0.5/1/2/4 N m: angle 1.40/1.40/1.39/1.38/0.03 rad |
| pouch_transfer | grip 12 N | Young 2e3/6e3/2e4/6e4 Pa: P(success) 1/1/0/.17 (stiff pouches slip out of a light pinch; 6e4 is outside the final train range) |
| twist_lock | grip 15 N, mu_g 0.6, 1 rad/s | breakaway 0.03/0.1/0.2/0.4/0.6 N m: knob angle 1.74/1.66/0.50/0.17/0.17 rad |
| machine_tending | straight insert of the blank (door pre-opened) | vise clearance 1/1.5/2/3 mm: P(seated) .83/.83/1/1 (weak) |

Values of 0.83 / 0.67 instead of 1.0 come from individual layout seeds failing for unrelated reasons (kept, not
filtered). connector_insert is the weakest family (an earlier run with the near-side camera showed almost no
clearance effect; with the final code tight + high-friction slots drop to 0.33) - treat its G4 evidence as weak.

## Scripted heuristic baseline (required baseline; `env.scripted()`)
Round 2 (lead: "a decent engineer's heuristic"): conservative defaults + escalating retries + grasp re-planning,
never reads labels. Per family: target = most accessible from the tracker (highest in bin / top carton layer /
front tote / most crowded / most downstream belt part that can still be intercepted); defaults: grip 40 N lift
0.2 m/s (bins), suction 0.9 lift 0.15 (cartons), grip 80 N lift 0.15 (totes), 40 N (conveyor), compliant spiral
insertion; after a failure on the same target: firmer grip / full suction / slower lift / larger spiral, grasp axis
rotated 90 deg, fingers kept parallel to a nearby bin wall, skip a target after 3 failures and come back later;
bin_clearing moves the arm out of the camera view to look again when nothing is freshly tracked; conveyor waits in
short (<= 2.5 s) steps and keeps waiting while anything is on the belt; machine_tending keeps holding the blank
after a jammed insert and searches wider; kitting uses a 3-9 mm spiral at 0.7 Hz. Per-family success: "Generation".

## Splits (`envs/splits.py`)
* train / val / test by ROOT SEED: sha256 bucket 80/10/10 decided before generation (no root in two splits).
* heldout_layout: test-bucket seeds; bench stations rotated 180 deg about the station centre (source/target sides
  swap relative to the robot; regions/CAD meta rotated consistently); pallet/conveyor/aisle stations: count shift
  (3 carton layers, 7 items, 4 totes, heavier pouch).
* heldout_objects: test-bucket seeds with object sets/ranges never used in train: bin_clearing mesh+ellipsoid parts
  (train uses box/cylinder/capsule/sphere), all-cylinder singulation, capsule-heavy bulk, 60 % heavy cartons/items,
  16 kg totes, 16 mm pegs, 1.2 kg tool heads, 5 kg fixtures, 4 kitting parts, 16 kg drawers, 8 kg doors, 5e4 Pa pouches.
* heldout_family per category: bulk -> tote_transfer, fine -> connector_insert (`make_tasks` refuses train/val).
* heldout_robot: `scripts/envs_generate.py --heldout_robot NAME` generates that robot ONLY with split
  "heldout_robot" (test-bucket seeds, in-distribution params), never in train/val/test (proposed: gen3_2f85).
Sensor-noise / latency / dropout are randomized per episode in every split (labels.sensor_cfg).

## Generation (`scripts/envs_generate.py`, storage `envs/io.py`)
Streaming multiprocessing (one episode per task, deterministic per seed, `maxtasksperchild=50`, 300 s wall guard per
episode -> recorded as an error row), chunked shards `<out>/[<robot>/]<family>/<split>_<k>.pkl.gz` (50 episodes
each: task, events as a compact numpy structured array + string tables, steps with call / ok / status / effort /
duration / metrics / labels / skill log incl. `cmd`), `manifest.json` (per-run: code hash, args, sha256 + bytes +
root seeds per file, per (robot, family) stats, episodes/hour, bytes total); one row per episode in
`results/v2/envs/generation.jsonl`. Load with `physmem.envs.io.iter_episodes("<out>/**/*.pkl.gz")`, which rebuilds
the exact `ObservationEvent`s (round trip tested). Multi-robot: `--robots a,b,c --matrix working_pairs.json
--heldout_robot gen3_2f85`.

**Current dataset** `gb10-direct:~/dynagraph/data/v2_envs` (probe robot, round-2 code, code hash in manifest):
`--families all --splits train,val,test,heldout_layout,heldout_objects --n 16 --procs 5` = **1216 episodes in
1156 s = 3790 episodes/h on 5 processes** (host load ~20 from other agents), **0.50 GB** (~0.41 MB/episode),
0 errors, 0 integrity violations. Earlier runs kept: `data/v2_envs_run1` (near-side camera), `data/v2_envs_run2`
(round-1 baselines) and `results/v2/envs/generation_run{1,2}.jsonl`. Scripted-baseline success (n), probe:

| family | train | val | test | heldout_layout | heldout_objects | heldout_family | heldout_robot | wall s/ep | eps/h/proc | sim s/ep | calls/ep | events/ep | viol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bin_clearing | 0.19 (16) | 0.19 (16) | 0.25 (16) | 0.06 (16) | 0.00 (16) | - | - | 16.2 | 223 | 206 | 33.3 | 88673 | 0 |
| bulk_sweep | 0.56 (16) | 0.56 (16) | 0.75 (16) | 0.75 (16) | 0.00 (16) | - | - | 12.5 | 288 | 44 | 8.3 | 14223 | 0 |
| connector_insert | - | - | - | 0.50 (16) | 0.62 (16) | 0.44 (16) | - | 2.8 | 1302 | 30 | 4.9 | 8421 | 0 |
| conveyor_sort | 0.62 (16) | 0.56 (16) | 0.62 (16) | 0.38 (16) | 0.62 (16) | - | - | 4.2 | 848 | 61 | 13.9 | 21751 | 0 |
| depalletize | 1.00 (16) | 1.00 (16) | 0.94 (16) | 0.00 (16) | 1.00 (16) | - | - | 9.7 | 372 | 154 | 14.0 | 52663 | 0 |
| door_open | 1.00 (16) | 1.00 (16) | 1.00 (16) | 1.00 (16) | 1.00 (16) | - | - | 0.4 | 9947 | 8 | 1.0 | 2396 | 0 |
| drawer_open | 1.00 (16) | 1.00 (16) | 1.00 (16) | 0.88 (16) | 0.94 (16) | - | - | 0.6 | 6181 | 10 | 1.2 | 2997 | 0 |
| fixture_place | 1.00 (16) | 0.94 (16) | 0.88 (16) | 0.88 (16) | 0.19 (16) | - | - | 1.4 | 2594 | 17 | 3.4 | 4928 | 0 |
| kitting_pocket | 0.25 (16) | 0.50 (16) | 0.31 (16) | 0.25 (16) | 0.12 (16) | - | - | 7.2 | 497 | 82 | 11.5 | 26787 | 0 |
| machine_tending | 0.69 (16) | 0.69 (16) | 0.94 (16) | 0.75 (16) | 0.94 (16) | - | - | 2.5 | 1413 | 53 | 5.8 | 15436 | 0 |
| peg_insert | 0.88 (16) | 0.88 (16) | 0.94 (16) | 0.94 (16) | 1.00 (16) | - | - | 1.5 | 2350 | 16 | 2.8 | 4585 | 0 |
| pouch_transfer | 1.00 (16) | 0.94 (16) | 1.00 (16) | 0.69 (16) | 0.69 (16) | - | - | 3.8 | 943 | 17 | 2.8 | 4873 | 0 |
| singulation | 0.88 (16) | 0.75 (16) | 0.81 (16) | 0.88 (16) | 0.75 (16) | - | - | 1.4 | 2520 | 19 | 5.8 | 7473 | 0 |
| tool_return | 0.94 (16) | 0.81 (16) | 0.94 (16) | 0.81 (16) | 0.06 (16) | - | - | 3.3 | 1093 | 35 | 4.7 | 9742 | 0 |
| tote_transfer | - | - | - | 0.00 (16) | 0.00 (16) | 0.44 (16) | - | 6.6 | 546 | 81 | 7.2 | 26480 | 0 |
| twist_lock | 0.94 (16) | 0.94 (16) | 0.75 (16) | 1.00 (16) | 0.00 (16) | - | - | 0.6 | 6202 | 14 | 2.4 | 3940 | 0 |

(`scripts/envs_summarize.py <dir>` -> `results/v2/envs/generation_summary.json`.) Round-2 baselines vs round 1
(train/test): depalletize .25/.12 -> 1.00/.94, conveyor_sort .38/.12 -> .62/.62, tote_transfer (heldout_family)
.12 -> .44, kitting .50/.25 -> .25/.31 (no gain), bin_clearing .12/.12 -> .19/.25 (still weak: repeated misses on
small stiff spheres - the pads pass through them, an open contact bug - and slips on heavy/oily parts).
Held-out-object rows near 0 (bin meshes/ellipsoids, 5 kg fixtures, 1.2 kg tool heads, capsule bulk, 0.9 N m
twist locks, 3-layer pallets for heldout_layout) are shifts the fixed-parameter baseline cannot absorb.

**LARGE run for MODEL-V2 (>= 20k episodes) - READY, waiting for ROBOTS' family x robot matrix.** Command (one
process per CPU slot we are allowed, <= 8):
`python scripts/envs_generate.py --families all --splits train,val,test,heldout_layout,heldout_objects
 --robots <working robots incl. probe> --matrix <ROBOTS matrix json> --heldout_robot gen3_2f85 --n <N> --procs 8
 --out data/v2_envs_large --rows results/v2/envs/generation_large.jsonl`. Size estimate: ~0.4 MB/episode ->
~8 GB per 20k episodes (441 GB free); probe throughput ~3.8k eps/h on 5 procs under load, real arms ~5-10x slower
per episode (envs_robot_smoke: 1-25 s wall/episode) -> 20k mixed episodes ~ 10-20 h on 8 procs.


## LARGE MODEL-V2 generation - STATUS (updated as tranches complete)
Plan `scripts/envs_plan_large.py` -> `data/v2_envs_large/plan/{tranche1,tranche2,tranche3_heldout_robot}.json` +
`pairs.json` (kept / dropped robot x family pairs with reasons). Launched 2026-11-23 20:45Z, 10 processes, tranches
in order, each with its own `data/v2_envs_large/<tranche>/manifest.json` (+ `manifest.sha256`); rows ->
`results/v2/envs/generation_large.jsonl`.
- tranche1: **DONE 2026-11-23 21:23Z** - 2912 episodes (probe 1600, ur5e_2f85 656, panda 656) in 2232 s on 10
  procs = 4696 eps/h, 1.12 GB, 234 shard files, 0 errors, 0 integrity violations, 0 timeouts.
  `gb10-direct:~/dynagraph/data/v2_envs_large/tranche1/` - manifest.json sha256
  `0d84bcd1830aa56fe94832c444ee3f2af131501241999881b0a060dcf38cdd95`; per-(robot, family) success in
  `logs/envs_large_tranche1.md` / `logs/envs_large_tranche1.log` (e.g. mixed-policy success: ur5e peg .98, door .96,
  drawer 1.0, bin_clearing 0; panda conveyor 0, tote 0; probe 0.16-1.0). Load: `physmem.envs.io.iter_episodes(
  "data/v2_envs_large/tranche1/**/*.pkl.gz")`.
- tranche2: ABORTED at ~1900/16887 (2026-11-23 21:36Z) because it was MIXED-VERSION: generator workers import the
  env family modules lazily from `~/dynagraph/src`, and other agents' `scripts/remote.sh` rsyncs pushed my r3 edits
  there mid-run. Partial shards kept only as `data/v2_envs_large/_aborted_tranche2_mixed_r2r3/` (no manifest; do not
  train on it); its rows moved to `results/v2/envs/generation_large_aborted_tranche2.jsonl`.
- **From tranche 2 on, every tranche runs from a FROZEN code copy `gb10-direct:~/dynagraph_r3`** (envs + embodiments
  + skills; nobody syncs into it; sha256 of the envs sources = `7095b278...0388`), env version **envs-r3**:
  - **tranche2_envs_r3_partial: FINAL (stopped early on the lead's decision, ~22:50Z host clock).** Planned 16887,
    completed 3600 episodes on disk (probe 3400, ur5e_2f85 200), 1.44 GB, envs-r3. About 2400 more episodes had
    finished but sat in the generator's per-(robot, family, split) buffers (< 50 each) and were lost when the run was
    killed; their rows remain in `results/v2/envs/generation_large.jsonl` without data. `manifest.json` records planned
    vs completed per robot; manifest.sha256 `38f31df11fcea0e32d4de90604e610995cb719a89d0b378b2dbdac694a5d7d67`.
    The remainder of tranche 2 is only resumed if nothing higher-priority is queued.
  - tranche3_heldout_robot_envs_r3 and tranche4_envs_r3: CANCELLED (lead) - replaced by r4 tranches.
  - tranche3_heldout_robot_r4: 776 episodes, gen3_2f85 only (split heldout_robot), envs-r4 - RUNNING since ~22:51Z
    host clock (~40 min).
  - tranche5_r4: TRIMMED to fit ~4 h (`plan/tranche5_r4_trim.json`, 7532 episodes, output dir `tranche5_r4`): all
    probe (3087), UR5e (1472), Panda (1472), ur5e_suction (580), fr3_suction (198), z1 (213), so100 (146) pairs (2x
    weight on the decision-relevant families); ALOHA / G1 floating / G1 fixed / TidyBot capped at 100 / 64 / 100 /
    100; real-arm bulk_sweep capped at 93 in total. QUEUED right after tranche3_r4; ETA ~3-3.5 h after it starts.
  - **tranche8_r6_site: FINAL 2026-11-24 18:08Z** - 9875 / 9875 episodes, 3.40 GB, 2.68 h generator time over 2 runs
    (6258 at 8 procs, paused for the other project, 3617 at 6 procs), errors 2, integrity violations 0, manifest
    sha256 `8a309cfa...0bcb8`; 0 duplicates / 0 lost (checked by (robot, family, split, seed)). Physical blow-up scan:
    8 / 9875 (0.08 %: 6 parts > 5 m - bin_clearing / small_parts_pick / pouch, 2 bulk_sweep BADQACC); `integrity.unstable`
    on 3 (bulk_sweep). probe-then-act in 4517 episodes. Success by robot: probe 0.40, g1_dex3_fixed 0.33, ur5e 0.32,
    panda 0.26, gen3 (held out) 0.20, tidybot 0.12, aloha / z1 0.10, suction arms ~0.01 (as in r5), so100 0.00.
    History of the pause (kept for reference): PAUSED at 6258 / 9875 (stopped cleanly 2026-11-24 07:14Z with the SIGTERM flush; box lent to
    another project until ~2026-11-25 14:00Z; partial manifest run, NO manifest.sha256, so neither the status watcher
    nor MODEL's `scripts/v2/r6_pipeline.sh` treats it as final). **Resume** (when the box is clear):
    `cd ~/dynagraph_r6 && ~/dynagraph/.venv/bin/python scripts/envs_tranche_ctl.py resume tranche8_r6_site 27000`
    It reruns plan minus every stored (robot, family, split, seed) - the 3617 remaining, including the episodes that
    were in flight - into the same dir without overwriting shards (the generator now picks the next free shard index,
    writes shards atomically and defers a SIGTERM that lands mid-write). It also starts `envs_tranche_ctl.py finish`,
    which writes the aggregate manifest run + manifest.sha256 only when all 9875 are stored, and the watcher then
    appends FINAL. `... status tranche8_r6_site` shows planned / stored / remaining; `... stop tranche8_r6_site`
    pauses again (it signals only the pid in `<dir>/generator.pid`). Tested end to end on a 120-task copy: stop at
    101, resume 19, 120 stored, no duplicates, FINAL written.
    Original launch (started 2026-11-24 05:48Z; `plan/tranche8_r6_site.json` from
    `scripts/envs_plan_large.py --r6 --scale 1.2`, 9875 episodes, envs-r6 from the frozen copy `~/dynagraph_r6` (envs
    sources sha256 `08692ebe...0cfc`), `dressing="site"` on every task, 8 processes, 7.5 h cap with SIGTERM flush;
    est. ~5 h). All executable pairs (ROBOTS' runs + earlier tranche pairs + g1_dex3_fixed tote_transfer G1 cell +
    tidybot shelf_restock with move_base; tidybot conveyor_sort dropped, g1_dex3 floating skipped): probe 3075,
    ur5e_2f85 1457, panda 1457, ur5e_suction 883, tidybot 840, aloha 536, g1_dex3_fixed 303, fr3_suction 298, z1 228,
    so100 143; gen3_2f85 655 (heldout_robot only). Splits: train 5167, val 738, test 738, heldout_family 1099
    (tote_transfer, connector_insert, **drawer_open = G5 hold-out**), heldout_layout 739, heldout_objects 739,
    heldout_robot 655. Decision-relevant families (r6 table) 2x weight. Records carry `integrity` and
    `probe_then_act`. Log `~/dynagraph/logs/envs_large_tranche8_r6_site.log`; FINAL line auto-appended to
    `large_status.md` by the watcher.
  - tranche6_r5: QUEUED after tranche5_r4 (`plan/tranche6_r5.json`, 2612 episodes, envs-r5 from the frozen copy
    `~/dynagraph_r5`, envs sources sha256 `c423ade2...4b86`, flush-on-stop generator): depalletize (new F/T
    mass-probing policy: probe 150, ur5e_suction 120, fr3_suction 80), tote_transfer (probe 100, ur5e 60, panda 60;
    held-out-family splits), shelf_restock (tidybot 120, probe 120), conveyor_sort and singulation (wide area)
    (probe 100, ur5e 60, panda 60 each), plus ~100 per other family (probe 50, ur5e 25, panda 25 where executable).
    Same action-diversity mix (10 % explore, 90 % mixed). Est. ~1-1.5 h. Chain log `~/dynagraph/logs/envs_large_chain_r5.log`.
  - **CPU budget:** 20 cores, load was ~27. ENVS generation uses 10 processes (not 12, given the load). ROBOTS /
    MODEL: please keep to your budgets (<= 8 processes each, COORDINATION.md) while tranche5_r4 runs.
  - **Live status (auto-updated, no ENVS needed): `gb10-direct:~/dynagraph/results/v2/envs/large_status.md`** gets a
    FINAL line (env version, episodes, GB, hours, errors, violations, manifest sha256) as each tranche lands.
  - How to check without ENVS: `ls ~/dynagraph/data/v2_envs_large/*/manifest.sha256`; progress lines `[n/N]` in
    `~/dynagraph/logs/envs_large_<dir>.log`.
  - r4 tranches run from the FROZEN copy `gb10-direct:~/dynagraph_r4` (ENV_VERSION envs-r4; sha256 of the envs
    sources `8f970b81...832d`); chain log `~/dynagraph/logs/envs_large_chain_r4b.log`.
  - MODEL-V2: a tranche is FINAL when `<dir>/manifest.sha256` exists (written after manifest.json); per-tranche
    summary in `~/dynagraph/logs/envs_large_<dir>.md`. Status lines here are updated by ENVS when each one lands.
- Env versions: tranche6_r5 = envs-r5 (frozen `~/dynagraph_r5`); tranche1 = envs-r2; tranche2_envs_r3_partial = envs-r3 (frozen `~/dynagraph_r3`); tranche3_heldout_robot_r4 and tranche5_r4 = envs-r4 (frozen `~/dynagraph_r4`). Every manifest records `env_version` (tranche1/2 via `code_hash`, the field was added in r3).
Policy: per episode 10 % `explore` (random feasible candidates), 90 % `mixed` (per decision 0.4 scripted / 0.4 random
feasible candidate / 0.2 scripted with speed/force/grasp/approach params x U(0.7, 1.3)); `steps[i].policy_source`.
Per step stored: call, policy_source, t_start/t_end (skill segment boundaries; the events inside are the 25 Hz
proprio / WRENCH / EE_POSE / GRIPPER / TACTILE and 10 Hz POSE3 stream), ok/status, effort components, metrics
(progress, slip/jam/stall/drop/collision flags, cumulative counters), labels (hidden physics, true state, contacts),
result log (probe: `cmd` = commanded reference (t, pos3, vel3) at 50 Hz; real robots: the ROBOTS adapter's log).
Hidden variables over the full documented ranges (incl. heavy / oily outliers); splits by root seed as planned.
Dropped pairs (>= 80 % unreachable, no sim progress, or crashing): panda/ur5e/gen3/aloha/tidybot depalletize
(parallel grippers), tidybot bin/peg/connector/tool/fixture/kitting/tote/pouch/machine_tending, aloha tote/pouch,
z1 fixture/peg/pouch/bin, so100 singulation/fixture/peg/pouch/bin, ur5e_suction peg/connector/tool/kitting,
g1_dex3 singulation + bin_clearing (crash), g1_dex3_fixed singulation + bin_clearing (crash). Untested pairs are not
generated.

## Round 3 changes (envs-r3, lead + ROBOTS §7 requests)
1. conveyor_sort: the scripted policy tracks sorted items (actor memory: successful place of the item it held into
   a bin, plus tracker estimates inside a bin region) and never re-targets them (test added).
2. bin_clearing pad pass-through: explicit pad<->object pairs now use `solref=(0.004, 1)`,
   `solimp=(0.95, 0.99, 0.001, 0.5, 2)` and NO margin (`envs/probe.py PAD_SOLREF / PAD_SOLIMP`). The default pair
   solref (0.02) let a 40-100 N squeeze push a 15 mm stiff sphere through the 10 mm pads. Probe, single stiff sphere
   (r 14-30 mm, timeconst 0.01), grip 60 N, seeds 0-7: 6/8 picked AND released into the tote (was ~0); 1 slip,
   1 not observed. No sticking after release with the zero margin. ROBOTS: please apply the same pair params to
   the real pads (your stuck-after-release trial used a 1 mm margin).
3. depalletize redefined like a real cell: success = every carton within `robot.spec.payload_kg` upright on the
   outfeed AND every overweight carton flagged with `SkillCall("handover", {}, target=carton)` (exception lane,
   env-level, 2 s, no motion) and left on the pallet, nothing dropped; flagging a light carton or lifting an
   overweight one off the pallet fails it; `metrics.overweight_lift_attempts`, `metrics.flagged`,
   `labels.payload_kg`, `labels.overweight`. Heavy cartons (12-22 kg) only in the bottom layer. New hidden class:
   30 % slick-film cartons (cup shear friction 0.15-0.3) that peel off at high transport acceleration. Scripted:
   lift every carton (suction 0.9 then 1.0 + slower lift), flag after two failed lifts. Probe (payload 20 kg)
   7/8 seeds.
4. Mobile manipulators: `tidybot` gets a lower bench (0.62 m, `stations.MOBILE_BENCH_TOP`; any robot via
   `params["bench_top"]`) and a base pose 0.25 m from the bench edge. TidyBot then solves peg_insert 2/2 and
   fixture_place 2/2 (were 100 % unreachable); bin_clearing / kitting execute (progress 0-0.4).
5. Light-parts station for SO-100-class arms (everything within ~0.25 m of the near-edge arm mount):
   `small_parts_pick` (3-6 parts <= 30 mm, <= 80 g, tray -> cup) and `small_peg_insert` (10-14 mm peg, 12 mm hole,
   0.3-1.2 mm clearance). Seeds 0-1: probe small_peg 4/4 seeds, small_parts progress 0.2-0.83; z1 small_peg 2/2;
   ur5e small_peg 2/2; so100 executes both (small_parts progress 0.2, small_peg 0/2).
6. shelf_restock with a TidyBot driving along an aisle: NOT done (time); needs base-motion skills from ROBOTS.
Also: `ENV_VERSION` in `physmem.envs` recorded in each manifest; tests 59/59 pass on envs-r3.

## Round 4 (envs-r4, dev tree; NOT in the running large-generation tranches, which are envs-r3)
* **Fragile parts / force limits** (so "use max force" is no longer dominant): hidden `crush_n` (N per pad) on
  bin_clearing parts (35 % fragile, 15-45 N), conveyor items (30 %, 15-45 N), kitting parts (30 %, 15-45 N),
  fixture blocks (50 %, 25-80 N), tools (50 %, 20-60 N), totes (50 %, 60-140 N), twist-lock knobs (60 %, 20-70 N),
  small parts (40 %, 6-20 N), drawer tabs (40-120 N) and door tabs (20-60 N). The env monitors the robot's
  sustained squeeze on each fragile body (sum of robot contact normal forces / 2, EMA over ~30 ms so closing impacts
  do not count); above the limit the part is DAMAGED (`metrics.damaged`, `labels.damaged`). Episode success is now
  `env.solved()` = family predicate AND no damaged part (`StepOut.success` uses it).
* **Interventions** (EVALUATOR-ONLY): `snap = env.snapshot()` (MuJoCo integration state + env bookkeeping +
  sensor/ActorView caches + robot controller state + rng states + hidden labels), `env.restore(snap)`,
  `env.set_hidden({eid: {mass, mu, mu_grip, frictionloss, damping, crush_n}})` (edits the compiled model's
  parameters - body mass/inertia, geom friction, pad-pair friction, joint frictionloss/damping - never poses) and
  `out = env.resimulate(snap, overrides, call)` = restore + set_hidden + execute. Deterministic: two resimulations
  of the same branch give identical states; tested that mass 3 kg or mu_grip 0.08 flips a 15 N pick to a slip,
  with 0 integrity violations.
* **Decision relevance** (`scripts/envs_decision_relevance.py`): oracle-parameterized scripted (rule-chosen
  grip / suction / lift-vs-flag / intercept speed / push speed / sweep stop time / search radius from TRUE latents)
  vs the same rule on PRIOR latents (population medians from 30 train resets, `results/v2/envs/latent_priors.json`)
  vs the plain scripted baseline; paired seeds, bootstrap 95 % CI of the success gap.

**Decision-relevance table** (envs-r4, probe, 20 paired seeds per family, 1084 s on 4 procs,
`results/v2/envs/decision_relevance.json`):

| family | oracle-param (true latents) | prior-param (median latents) | plain scripted | gap oracle-prior [95 % CI] | damaged (oracle/prior) | verdict |
|---|---|---|---|---|---|---|
| bin_clearing | 0.35 | 0.00 | 0.15 | +0.35 [+0.15, +0.55] | 0.25/0.10 | RELEVANT |
| singulation | 0.25 | 0.25 | 0.50 | +0.00 [+0.00, +0.00] | 0.00/0.00 | ~0: not relevant yet |
| bulk_sweep | 0.60 | 0.30 | 0.45 | +0.30 [+0.10, +0.50] | 0.00/0.00 | RELEVANT |
| peg_insert | 0.95 | 0.90 | 0.95 | +0.05 [-0.10, +0.20] | 0.00/0.00 | ~0: not relevant yet |
| connector_insert | 0.75 | 0.25 | 0.70 | +0.50 [+0.30, +0.70] | 0.00/0.00 | RELEVANT |
| tool_return | 0.75 | 0.55 | 0.95 | +0.20 [+0.05, +0.40] | 0.00/0.00 | RELEVANT |
| fixture_place | 0.95 | 0.80 | 0.75 | +0.15 [+0.00, +0.30] | 0.00/0.00 | borderline |
| kitting_pocket | 0.30 | 0.00 | 0.40 | +0.30 [+0.10, +0.50] | 0.00/0.00 | RELEVANT |
| drawer_open | 1.00 | 0.70 | 1.00 | +0.30 [+0.10, +0.50] | 0.00/0.00 | RELEVANT |
| door_open | 1.00 | 0.75 | 1.00 | +0.25 [+0.05, +0.45] | 0.00/0.00 | RELEVANT |
| depalletize | 0.95 | 0.35 | 0.90 | +0.60 [+0.40, +0.80] | 0.00/0.00 | RELEVANT |
| conveyor_sort | 0.00 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 0.05/0.00 | ~0: not relevant yet |
| tote_transfer | 0.30 | 0.15 | 0.20 | +0.15 [-0.05, +0.35] | 0.15/0.00 | borderline |
| pouch_transfer | 0.30 | 0.50 | 0.85 | -0.20 [-0.40, -0.05] | 0.00/0.00 | NEGATIVE (rule wrong) |
| twist_lock | 0.90 | 0.75 | 0.65 | +0.15 [+0.00, +0.30] | 0.00/0.00 | borderline |
| machine_tending | 0.60 | 0.40 | 0.00 | +0.20 [-0.05, +0.45] | 0.00/0.00 | borderline |
| small_parts_pick | 0.00 | 0.00 | 0.05 | +0.00 [+0.00, +0.00] | 0.30/0.30 | ~0: not relevant yet |
| small_peg_insert | 0.50 | 0.50 | 0.80 | +0.00 [+0.00, +0.00] | 0.00/0.00 | ~0: not relevant yet |

Reading: 8 families have a gap whose 95 % CI excludes 0 (depalletize +0.60, connector_insert +0.50, bin_clearing
+0.35, bulk_sweep +0.30, kitting_pocket +0.30, drawer_open +0.30, door_open +0.25, tool_return +0.20); fixture_place
and twist_lock are borderline (+0.15, CI touches 0); tote_transfer / machine_tending positive but not significant at
n=20. ~0: peg_insert and small_peg_insert (a spiral search makes the clearance irrelevant - needs a time/jam cost),
singulation (even with the bounded work area the push-speed rule does not change outcomes), small_parts_pick and
conveyor_sort (both rules fail: fragile tiny parts are damaged by the pads' closing and the intercept rule does not
fix the conveyor baseline), pouch_transfer (NEGATIVE: the mass-based grip rule ignores stiffness; the right rule is
grip vs Young's modulus). Families redesigned in r4 for this: bin_clearing 0 -> +0.35 (3-6 parts, pad-only squeeze
damage), singulation (bounded work area: still ~0). Next redesigns needed (not done): insertion time/jam costs,
stiffness-aware pouch rule, conveyor intercept tolerance, singulation target displacement.

### r4 quick fixes for the ~0-gap families (lead's 45-min budget) and the r5 backlog
* peg_insert / small_peg_insert: CAD cycle-time limit (16 s / 14 s of sim time for pick + insert; exceeding it is
  fatal). Re-measured gap (20 paired seeds): peg +0.05 [-0.10, +0.25], small_peg +0.00 - still ~0: one spiral search
  is cheap enough to fit the cycle. r5: make the search radius cost time per mm and add a jam/damage limit on the
  axial force for tight, high-friction fits.
* pouch_transfer: the oracle rule is now stiffness-based (grip = 6 N + Young/1000); gap -0.15 [-0.30, +0.00] -
  max force stays dominant because nothing penalises squeezing a pouch. r5: a squeeze limit for soft pouches
  (flex vertex-force monitor) so stiff needs more and soft needs less.
* singulation: push-vs-pick by friction not done (r5): high-friction objects should be picked, low-friction pushed.
* All three are in envs-r4 as described; the r4 tranches therefore contain these families with ~0 decision gap.

### r5 (dev tree only, NOT frozen, no r5 tranche queued)
* peg_insert / small_peg_insert: spiral search radius grows at most 1 mm per 1.5 s (search costs cycle time) and a
  hidden dent limit (8-25 N, 60 % of pegs) damages a peg pressed onto the rim harder than that (monitor: EMA of the
  peg's contact force with static fixtures). Decision gap (20 paired seeds): peg +0.10 [+0.00, +0.25] (was +0.05),
  small_peg +0.00.
* pouch_transfer: hidden squeeze limit crush_n = 4 N + Young/2500 (flex vertex contacts on the pads); the oracle grip
  rule = min(6 + Young/1000, 0.85 crush). Gap +0.05 [-0.25, +0.35] (was -0.15); the plain scripted baseline now
  fails 95 % (it crushes soft pouches at 12 N).
* singulation push-vs-pick: pick/place-to-free-spot added to scripted/candidates and a 7 s/object cycle limit; a crush
  limit on the pusher did NOT work: the probe's position-controlled pusher loads parts with 80-130 N regardless of
  mu*m*g (it shoves the cluster), so every push "crushed". Crush is opt-in (`params["crush_n"]`), default off.
  Needs a force-limited (compliant) push skill before friction can decide push vs pick.
* Verdict: the three gaps did NOT all open (peg borderline, pouch and singulation ~0) -> per the lead's rule no r5
  tranche is queued. Tests 60/60 pass on the r5 dev tree.
* shelf_restock: ROBOTS' `move_base` skill exists (`SkillCall("move_base", {x, y, yaw, speed})` via run_skill); NOTE
  for the lead: `v2/api.py SkillCall.vector()` does not list "move_base" (names.index would raise) - please add it
  before ENVS builds the aisle family.

### envs-r5-dev: ROBOTS round-5 fixes + shelf_restock (dev tree + local repo; ENV_VERSION "envs-r5-dev"; not frozen,
### not in any tranche yet - the next tranche that uses it will record it in its manifest)
1. **depalletize stale re-picks:** actor memory of cartons this robot released on the outfeed after a successful
   place; they are not re-targeted unless a FRESH (< 0.5 s) estimate shows them back on the pallet side. Same memory
   for tote_transfer (cart).
2. **depalletize flagging:** no longer on a failure count. The scripted policy hefts each carton (pick with a 4 cm lift)
   and estimates its mass from its own tared wrist F/T and suction command (`env.held_mass_evidence()`: hold
   Fz ~ m g - F_suction_cmd, verified: 18.0 kg read for an 18 kg carton). Above `payload_kg` it puts the carton back
   (place on CAD region `pallet` at the pick estimate) and flags it (`handover`); a carton that slips even at
   suction s gives a lower bound m > s*gain/(g + a) and is flagged only when that bound exceeds the payload. Probe
   (payload 20 kg): 6/6 seeds; all-25 kg pallet -> all 4 flagged and solved; all-18 kg -> all delivered.
3. **Timeout integrity (ROBOTS: 3 violations after a mid-skill timeout):** the episode runner moved to
   `physmem.envs.rollout` (`run_episode`, `run_with_timeout`, `choose`, `perturb`); the wall-clock timeout is
   `EpisodeTimeout(BaseException)`, so skill-level `except Exception` handlers (the real-robot adapter) can no longer
   swallow it; the runner ends the episode at once (no further calls) with `error="timeout"`, `aborted=True`. Test
   `test_episode_timeout_aborts_cleanly_even_through_skill_exception_handlers`. The generator also flushes all
   buffers and writes a partial manifest on SIGTERM now (the stopped tranche 2 lost ~2400 buffered episodes).
   NOTE: the running r4 tranches use the frozen r4 copy, which still has the old timeout class.
4. **singulation A/B (UR5e, seeds 0-4; n small):** A envs-r3 + old ROBOTS stack 2/5; D current env with the r4
   bounded area / cycle limit OFF + old stack (r3-era embodiments, skills, adapter) 3/5; E same with them ON 1/5;
   B current env, OFF, new ROBOTS stack 1/5; C current env, ON, new stack 1/5. Both changes cost about 2 of 5: the
   r4 bounded-area/cycle fatal conditions (D -> E) and the new robot stack (pads/adapter; D -> B, failures are pushes
   that throw parts far). Since the bounded area did not create a decision gap, envs-r5 restores the wide area
   (0.45 x 0.3) and no cycle limit by default (opt-in params `work_half_x/y`, `cycle_limit`, `crush_n`).
   ROBOTS: B's failures are pushes after which a part leaves even the wide area - please check push stroke /
   release with the new pad contacts.
5. **shelf_restock** (new family, category bulk/logistics): a 2.2 m aisle - cart at the start, a shelf with 3 bays
   (0.62 m), 3-5 items with visible bay assignments (meta) and hidden mass / grip friction / fragility. Mobile robots
   drive with ROBOTS' `move_base` (`SkillCall("move_base", {x, y, yaw, speed, arm})`; encode with
   `SkillCall.vector(dim=32)`); the gantry probe works without driving. Success: every item upright in its bay, none
   damaged/dropped. Probe seeds 0-2: progress 1.0 / 0.75 / 1.0, 1 solved (the fails are fragile items crushed by the
   scripted 30 N grip). TidyBot not yet run on it.
Tests: 62/62 on envs-r5-dev.

### envs-r5 frozen (`~/dynagraph_r5`): shelf_restock with TidyBot
* TidyBot on the first aisle layout: `move_base` + pick worked, but every `place` on a shelf bay returned
  `unreachable`. Cause (adapter, ROBOTS-owned): `_base_approach` clips the pre-skill lateral base alignment to +-0.6 m
  around the MOUNT and drives the base back there after an explicit `move_base`, so bays > 0.6 m from the start are
  out of reach. ENVS workaround: compact aisle (cart at x = -0.35, bays at 0.05 / 0.42 / 0.79, base start 0.22).
  Then TidyBot seeds 0-2: progress 1.0 / 0.75 / 0.75 (drive, pick, place all work; remaining fails: grasp misses,
  a drop, fragile items). ROBOTS: please (a) skip `_base_approach` when the base was just positioned by `move_base`
  (or align around the current base pose, not the mount) and (b) check ~107 robot contact events > 150 N during
  TidyBot reset/park in this station (base or arm vs. shelf / cart).
* ENV_VERSION "envs-r5"; tests 62/62.

### Videos (`scripts/envs_videos.py`, results/v2/videos/)
One episode per family (19), robot = best cell of ROBOTS' env matrix (`results/v2/robots/env_matrix.json`, success
rate then progress), PRE-REGISTERED seed = the first seed of the family's test split (seed 22; held-out families:
heldout_family split), scripted heuristic policy, envs-r5 dev tree; 1280x720 H.264, label strip (robot, family, seed,
policy, sim time, playback speed, effort cost, current skill, live status -> SUCCESS / FAIL (reason)), TCP close-up
inset. Files `<family>__<robot>__s22.mp4`, `index.json` (success, failure reason, sim time, playback speed, steps,
env version, matrix cell), `montage_all_families.mp4` (~9 s per family; an overlay states any extra montage speed-up).
Outcomes (not cherry-picked): SUCCESS peg_insert (panda), connector_insert, fixture_place, kitting_pocket,
small_peg_insert (ur5e), door_open (gen3), drawer_open (tidybot); FAIL with labels: twist_lock, tool_return,
conveyor_sort, tote_transfer, machine_tending (ur5e), pouch_transfer (panda), bin_clearing (gen3) - "damaged part"
(r5 crush limits vs scripted grips); depalletize (aloha) "pick: too_wide x12"; small_parts_pick (aloha) "grasp_failed";
shelf_restock (tidybot) "skipped_after_failures"; singulation (ur5e) policy gave up; bulk_sweep (panda) not completed
in 307 s.

### envs-r6: warehouse site dressing after the TalOS warehouse (`envs/dressing.py`, `params["dressing"] = "site"`)
**Source.** `~/Documents/commandagi/working/home+org-talos/worlds/talos-warehouse` (hall shell, materials, lights) and
the demo bays it places (`demos/g1-h1-aisle` layout.ts, `demos/depalletise-kitting` LAYOUT, `demos/g1-tote-picking`).
No MJCF / MuJoCo versions of the props exist there (the racks, pallets, crates and cartons are NVIDIA SimReady
`.glb` meshes, CC BY 4.0; totes, conveyors and the hall are CAD sketches), so:
* PORTED: dimensions, layouts and colours - EUR pallet 1200 x 800 x 144 (22 mm boards, 78 mm blocks), KLT VDA 4314 /
  4322 totes, 600 x 400 x 253 crates in layers on pallets, selective racking (1.08 m deep, beam tops 0.506 / 1.65 /
  2.509 m, blue uprights #2f5d8c, orange beams, galvanised decks), gravity roller conveyor (480 mm, d50 rollers at
  100 mm pitch), belt conveyor (rubber, aluminium rails), floor #9b9890 (lightened) with 100 mm safety-yellow lane
  lines and walkway green, precast wall band + cladding, steel-blue columns, high-bay lights.
* RE-MODELLED as MjSpec primitives (boxes / cylinders, procedural textures): every prop above, plus a packing table
  (ESD top, upper shelf, light bar) and cardboard clutter. Nothing imports a mesh.
* Physics: props within ~0.9 m of the station footprint or a robot mount are collision-real (obstacles and tracker
  occluders, geom group 0); the rest (hall, background racking, far stacks) is visual-only (contype = conaffinity = 0,
  geom group 2; the tracker's ray test ignores group 2). Nothing is placed in the station footprint or within 0.75 m
  of a robot mount. Station geoms that the dressing re-draws (pallet slab, outfeed / cart tables) keep their
  colliders (made invisible), so task geometry and semantics are unchanged. Layout, clutter, tote colours and light
  colour/intensity come from a SEPARATE rng (`seed_of(task, "dress")`): object poses and hidden physics of a dressed
  episode are identical to the bare one (tested).
* Rendering: hall walls / ceiling (no black void), textured concrete floor, lane markings, 6 ceiling spot lights (one
  casting shadows, shadowsize 4096), dimmed headlight, per-seed light colour temperature.
* Layouts: `depalletize` (EUR pallet under the stack, outfeed ROLLER conveyor continuing out of the cell, crate stacks,
  racking row behind), `conveyor_sort` (roller infeed into the driven belt, racking with blue totes, crate stacks),
  `tote_transfer` (flow rack -> racking bay with uprights / orange beams / galvanised decks; cart -> roller conveyor;
  NEW humanoid cell for G1 robots: cart at the humanoid's right at 0.75 m, humanoid mount 0.32 m in front of the lower
  tier - in the old cell the G1 stood inside the cart: 40 / 17 reset collisions -> 0), `shelf_restock` (shelf ->
  racking bay with a loaded upper level, a second racking run across a ~1.5 m aisle for TidyBot, cart -> roller
  section), `kitting_pocket` (packing table with an upper shelf and light bar, supply roller conveyor, racking). Other
  families get the hall + background + a generic aisle (`generic_layout`).
* Cost: dressed models have 400-700 geoms (bare 17-66); wall time per physics step +6.5 % depalletize, +6.6 %
  conveyor_sort, +4.2 % tote_transfer, +6.4 % kitting_pocket (probe), +2.4 % shelf_restock (TidyBot);
  reset +0.02-0.1 s.

**Scripted baseline, bare vs dressed (same 12 seeds, `scripts/envs_dressing_eval.py`, results/v2/envs/dressing_eval_summary.json):**

| family | robot | success bare | success site | progress bare / site | paired flips | violations (site) |
|---|---|---|---|---|---|---|
| depalletize | probe | 1.00 | 1.00 | 1.00 / 1.00 | 0 | 0 |
| conveyor_sort | probe | 0.25 | 0.25 | 0.78 / 0.78 | 0 | 0 |
| tote_transfer | probe | 0.33 | 0.33 | 0.75 / 0.75 | 0 | 0 |
| kitting_pocket | probe | 0.42 | 0.42 | 0.69 / 0.69 | 0 | 0 |
| shelf_restock | tidybot | 0.08 | 0.08 | 0.60 / 0.61 | 0 | 0 |

Regressions found and fixed on the way (all measured): kitting 0.42 -> 0.00 because the packing table's collision-real
upper shelf hid the work surface from the tracker (an overhead tracker -> 0.08: the part under the gripper is unseen,
no in-hand correction; a side tracker -> 0.17) - fixed by making the upper shelf + its cartons visual-only (it is
0.62 m above the table at the back, outside the kitting motions); TidyBot shelf_restock 0.6 -> 0.03 progress because a
new rack upright 4 cm from the cart blocked the arm - removed. KNOWN: the test-only gantry probe cannot do the dressed
shelf_restock (it descends from above through the rack's upper level: 0.17 -> 0.00); TidyBot / arms approach from the
front. G1 in the humanoid cell: reset contact-free, but every pick is `unreachable` both bare and dressed - the ROBOTS
adapter does not forward the `two_palm=1` tote pick yet (request below).

**Decision relevance in the site** (oracle- vs prior-parameterised scripted, 16 paired seeds, probe;
results/v2/envs/decision_relevance_site_*.json): depalletize +0.62 [+0.38, +0.88], kitting_pocket +0.31 [+0.06, +0.56],
tote_transfer +0.19 [-0.06, +0.44], conveyor_sort +0.00 - identical to the bare re-run on the same seeds (+0.62 /
+0.31 / +0.19 / +0.00): the dressing does not collapse any gap.

**Videos and stills** (`scripts/envs_videos.py --dressing [--stills|--montage]`): `results/v2/videos_site/` -
19 dressed episodes (same robots, pre-registered seed 22, scripted policy; every success/fail outcome identical to
the bare set), `index.json` + per-video `.json` sidecars, `montage_all_families.mp4`; 6 hero stills 1280x720 in
`results/v2/videos_site/stills/` (depalletize probe, conveyor_sort ur5e, tote_transfer G1 humanoid cell and ur5e,
shelf_restock TidyBot aisle, kitting_pocket ur5e), copied to `~/Downloads/dynagraph_gallery/site/`.

**ENV_VERSION "envs-r6"** (dev tree `~/dynagraph_dev`, not frozen; frozen r4/r5 copies untouched). Tests 68/68
(new: dressing keeps task randomisation + determinism + no drops/violations + visual-only geoms non-occluding;
collision-real props near the station).

### envs-r7.1 (robot-model fix only; frozen `~/dynagraph_r7_1`)
MODEL found a persistent Gen3 self-contact: r0/base_link vs r0/shoulder_link (adjacent links, 12 mm penetration,
contype/conaffinity 1/1) - a constant ~1.8 MN contact from ~0.16 s after reset in EVERY Gen3 episode of envs-r6 and
envs-r7 (max_robot_contact_n identical in all of them, one 'collision' counted per episode). Cause: base_link is
welded to the world, so MuJoCo's parent filter (which compares weld bodies) treats shoulder_link's parent as the world
and does not exclude the pair. Fix: an explicit contact exclude for that pair (`envs/robots.py::SELF_EXCLUDES`).
Self-contact audit (reset + a scripted episode, every robot): only Gen3 had an adjacent-link contact; UR5e /
ur5e_suction show small real non-adjacent contacts while folding (gripper / cup vs upper_arm, 9-34 N) - physical,
kept; panda, aloha, z1, so100, fr3_suction, tidybot, g1_dex3, g1_dex3_fixed clean. Verified: the compiled models
(body / geom / contact / exclude / pair / actuator / damping arrays) and post-reset states of ur5e_2f85, panda and probe
are byte-identical between r7 and r7.1, so their r7 audit rows, tranche rows and MODEL's r7 demos carry over. Only the
Gen3 audit rows and the Gen3 held-out tranche episodes are regenerated on r7.1.

**Gen3 ceiling on r7.1** (`results/v2/envs/ceiling_r7_1.{md,json}`; oracle success r7 -> r7.1, 6 seeds, tight_insert
10; the UR5e / Panda / probe rows of `ceiling_r7` carry over unchanged): the self-contact WAS a large part of Gen3's low
ceilings and of its tight_insert 'transit collisions':

| family | nominal | PTA / force-ramp | oracle r7 -> r7.1 |
|---|---|---|---|
| tight_insert | 0.30 -> 0.90 | force-ramp 0.40 -> **1.00** | 0.10 -> 0.80 |
| bin_clearing | 0.00 -> 0.00 | PTA 0.00 -> 0.17 | 0.67 -> **0.83** |
| bin_clearing @deep_bin | 0.00 | 0.00 | 0.33 -> 0.33 |
| fastener_pick | 0.00 -> 0.50 | PTA 0.17 -> 0.50 | 0.00 -> 0.67 |
| fixture_place | 0.33 | 0.50 | 0.67 -> 0.67 |
| tool_return / pouch / conveyor / depalletize | ~unchanged | | 0.00 / 0.17 / 0.00 / 0.00 |

The r6 Gen3 held-out results (MODEL's G5 / heldout_robot numbers from r6 and r7 tranches) were measured on the bugged
model and understate Gen3.

### envs-r7 (FROZEN 2026-11-26 08:02Z, `~/dynagraph_r7`; sources sha256 8166269b...9b28): skill-layer ceiling audit + fixes, fine-motor families
Why: MODEL round 5 showed the ORACLE planner reaching ~0.27 good parts / lot on UR5e (Panda 0.05, Gen3 ~0.2): the
scripted skills / controllers were the ceiling. r7 measures that ceiling per (robot x family) and fixes the skill layer.

**Ceiling audit** (`scripts/envs_ceiling.py`, resumable; rows `results/v2/envs/ceiling_<tag>_rows.jsonl`, table
`ceiling_<tag>.json/.md`): the scripted policy with `nominal` (fixed engineer) parameters vs `oracle` parameters
(`envs/oracle.py::rule`, TRUE hidden physics) and a small grid (`oracle_slow` speeds x0.5, `oracle_firm` grip x1.3),
site-dressed, failure taxonomy per episode (grasp_miss, slip, crush_or_damage, ik_fail, collision, jam, perception,
sim_instability, sim_timeout_or_error, budget, place_miss_or_predicate). Tag `r6` = the frozen envs-r6 code.

**Audit BEFORE (envs-r6 frozen code, 10 cm bins, 6 seeds, site-dressed; `results/v2/envs/ceiling_r6.{json,md}`).**
Success (mean progress); oracle failure causes. No (arm x family) reached 0.90 with oracle parameters except UR5e /
probe fixture_place; the dominant failures are grasp misses (fingers landing on walls / neighbours / the target,
false 'closed on nothing' on small and soft parts), slips (no grip regulation while carrying) and crush:

| family | UR5e nominal / oracle | Panda nominal / oracle | Gen3 nominal / oracle | probe nominal / oracle | main oracle failures (arms) |
|---|---|---|---|---|---|
| bin_clearing | 0.00 / 0.17 | 0.00 / 0.00 | 0.00 / 0.17 | 0.17 / 0.33 | grasp_miss, slip |
| pouch_transfer | 0.17 / 0.33 | 0.33 / 0.33 | 0.17 / 0.33 | 0.00 / 0.33 | slip, grasp_miss |
| fixture_place | 0.83 / 1.00 | 1.00 / 0.83 | 0.33 / 0.50 | 0.83 / 1.00 | crush, slip (Gen3) |
| tool_return | 0.67 / 0.67 | 0.17 / 0.33 | 0.50 / 0.50 | 1.00 / 1.00 | slip, crush, grasp_miss |
| depalletize | 0.00 / 0.17 | 0.00 / 0.50 | 0.00 / 0.00 | 1.00 / 1.00 | grasp_miss (parallel jaws on cartons) |
| conveyor_sort | 0.17 / 0.17 | 0.17 / 0.00 | 0.00 / 0.00 | 0.33 / 0.67 | crush, grasp_miss, ik_fail |

**Skill-layer fixes (real arms; `skills/primitives.py`, `envs/robots.py`, `embodiments/{core,robots}.py`):**
- collision-aware grasp selection: closing yaws (narrow axis, wide axis if it fits, 8 yaws for round parts) x vertical
  and tilted approaches (outward + 8 compass directions); each IK-reachable candidate is scored on a scratch MjData by
  the exact distance of the pre-shaped gripper to CAD-known static fixtures (bin / tote walls) and of its pads to the
  other parts' ESTIMATED footprints; candidates whose pads would land on the target itself are last resorts;
- in-skill regrasp: if the guarded descent trips > 1.5 cm above the grasp point, back up and try the next-ranked
  candidate (`regrasp`, default 2) instead of failing the call;
- gripper force calibration: 2F-85 pad force was commanded + 3 N (linkage spring; `GripperCfg.pad_offset`), Panda
  pad force 0.69 x commanded (`pad_per_drive` 0.68 -> 0.47); measured squeeze now within +-1 N;
- slip-reactive grip while carrying (pad slip sensing): +30 % squeeze per slip event up to `grip_max` (`slip_gain`,
  `slip_v` params); force-limited grip: the measured pad force is kept under `grip_max` (a part wedged against a
  neighbour was squeezed 1.5x the commanded force);
- a hand squeezing on nothing relaxes to 10 % drive (the Panda fingers were pushed through their joint limits and
  stayed jammed shut after a full-force close);
- bin_clearing scripted: most-accessible part first (per layer), place on the tote's free spot;
- oracle rule (`envs/oracle.py`, was in scripts/envs_decision_relevance.py): grip 2.5x the static hold force (10 N
  floor), `grip_max` = 0.85 x crush, slow lift / carry for marginally holdable parts.
- **Station-geometry change that makes the task EASIER (labelled as such):** bin_clearing source-bin walls 10 -> 6 cm,
  tote walls 9 -> 7 cm (r7 default; parts are <= 6 cm tall). With 10 cm walls no collision-free grasp existed for the
  Panda hand / 2F-85 body away from the bin centre. The r6 geometry stays available as the named station variant
  `params["station_variant"] = "deep_bin"` and is reported next to the default in the audit and the r7 eval.

**bin_clearing, oracle vs nominal parameters (10 seeds, site-dressed, r7 dev, default 6 cm bin):**

| robot | r6 oracle (6 seeds) | r7 oracle | r7 nominal | r7 oracle failures | r7 nominal failures |
|---|---|---|---|---|---|
| ur5e_2f85 | 0.17 | 0.80 | 0.20 | slip 2 | crush 6, grasp_miss 2 |
| panda | 0.00 | 0.50 | 0.00 | grasp_miss 4, slip 1 | crush 8, grasp_miss 2 |
| gen3_2f85 | 0.17 | 0.70 | 0.00 | slip 2, grasp_miss 1 | crush 10 |

Decision relevance holds: nominal parameters crush the fragile parts (40 N fixed grip), the oracle bounds the grip by
the crush limit.

**TASK CHANGES (labelled):**
- bin_clearing walls 10 -> 6 cm / tote 9 -> 7 cm (see above; makes the task easier; `deep_bin` keeps the r6 geometry).
- tight_insert dent limit is CORRELATED with the snap force: dent_n ~ U(max(6, 1.8 x snap_n + 2), 20) N on the 50 % of
  pegs that can dent (snap_n = 1.5 + 6.5e-4 k (0.5 + mu), 2.9-11.6 N). Without it ~53 % of dentable pegs (26 % of all
  pegs) could not be seated at ANY press force without denting; that is the fraction of pegs whose dent limit the
  constraint changes relative to an independent U(6, 20) draw.
- tight_insert hidden range widened BEFORE the freeze (no model has been evaluated on it): snap-fit lip stiffness
  3-12 kN/m -> 3-16 kN/m (snap force ~3-12 -> ~3-16 N). Why: with the narrower range a fixed 8 N press passed the lips
  in most episodes, so the hidden lip physics rarely changed the best press force. 25 kN/m was tried and rejected (the
  lip ejected the peg). The ranges are frozen with envs-r7.
- dent monitor (peg_insert, small_peg_insert, tight_insert): counts the AXIAL (world-vertical) part of the contact
  force between the part and static fixtures; r5-r6 summed every contact normal, so the side-wall normals of a peg
  sliding in a tight hole counted as 'ramming' (a correct 12 N press past the snap lips 'dented' a 19.6 N-rated peg
  through 33 N of wall normals). Makes dents rarer than in r6.
- fastener_pick: washers are clutter only (never ordered) - a flat 1-1.6 mm washer cannot be pinched by a parallel
  gripper; a magnet / suction tool variant may come later.

**Insertion seating check (r7, `envs/robots.py::sk_insert`).** The insert skill measures depth from the TCP height,
which reads 'jammed' when the peg slid up in the grasp under the press. The adapter then cross-checks with the SAME
actor-legal tracker every policy sees (no sim state): `env.est_pose_avg(peg, 0.3 s)` = mean of the POSE3 packets of
the overhead tracker camera in the last 0.3 s (fine stations: per-episode position sigma 0.5-1.5 mm, latency 30-90 ms,
dropout 0-8 %, line-of-sight occlusion by mj_ray - a peg hidden by the hand / socket emits nothing, then no override),
the one-shot SIZE estimate (sigma 1 mm) and the CAD hole region. Seated if (hole mouth - estimated peg bottom) >=
0.8 x hole depth - 1 mm. A stale / occluded estimate shows the peg higher, so the check can only fail towards 'jammed'.

**tight_insert: nominal vs force-ramp vs oracle** (`ceiling_fm7`, 10 seeds, site-dressed; success; median sim-s of the
successes nominal / force-ramp / oracle). Force-ramp = press at 4 N, +2 N per stall, ceiling 14 N, no hidden knowledge:

| robot | nominal (8 N) | force-ramp | oracle | oracle failures | median success sim-s |
|---|---|---|---|---|---|
| ur5e_2f85 | 0.80 | 0.90 | 0.70 | grasp_miss 1, place_miss 1, damage 1 | 18.3 / 20.0 / 18.0 |
| panda | 0.30 | 0.70 | 0.60 | damage 4 (rim ramming during the spiral search) | 17.1 / 19.1 / 14.7 |
| gen3_2f85 | 0.30 | 0.40 | 0.10 | collision, jam, not_found | ~32 |

Honest read: the reactive force-ramp script already captures most of the hidden lip / dent physics (force-ramp >=
oracle on success); what knowledge of the hidden physics still buys is cycle time (Panda: oracle 4.4 s faster) and fewer
stalls - the target for lot memory. Panda oracle dents: the position-tracked spiral rams the peg edge into the rim /
chamfer (22 N on a 19.5 N peg while the search press is 4 N). A laterally compliant spiral (`search_k` param, N/m) avoids
it but at 600 N/m the search stopped tracking and success collapsed (UR5e force-ramp 0.9 -> 0.1), so the default stays
the r6 spiral and `search_k` is an exposed gain. **Gen3 limitation (r7; largely resolved in r7.1 - the 'collisions' were the base/shoulder self-contact):** its 2F-85 hides the held peg from the overhead
tracker (no in-hand offset correction, no tracker seating check) and the transit to the hole collides with the fixture
(1 collision per episode; the peg ends ~8 cm off the hole) - no collision-free transit planner in the skill layer yet.

**Audit AFTER (frozen envs-r7, `results/v2/envs/ceiling_r7.{md,json}`; 6 seeds, tight_insert 10; oracle success, r6 ->
r7; nominal / probe-then-act (PTA) / force-ramp for reference):**

| family | UR5e oracle r6 -> r7 (nominal, PTA) | Panda oracle r6 -> r7 (nominal, PTA) | Gen3 oracle r6 -> r7 (nominal, PTA) |
|---|---|---|---|
| bin_clearing (6 cm bin) | 0.17 -> **0.83** (0.17, 0.17) | 0.00 -> 0.33 (0.00, 0.33) | 0.17 -> 0.67 (0.00, 0.00) |
| bin_clearing @deep_bin (10 cm, r6 geometry) | 0.17 -> 0.67 (0.00, 0.17) | 0.00 -> 0.00 (0.00, 0.00) | 0.17 -> 0.33 (0.00, 0.00) |
| fixture_place | 1.00 -> **1.00** (0.83, 1.00) | 0.83 -> **1.00** (0.67, 0.67) | 0.50 -> 0.67 (0.33, 0.50) |
| tool_return | 0.67 -> 0.50 (0.00, 0.17) | 0.33 -> 0.17 (0.17, 0.17) | 0.50 -> 0.00 (0.17, 0.00) |
| pouch_transfer | 0.33 -> 0.33 (0.33, 0.00) | 0.33 -> 0.33 (0.33, 0.33) | 0.33 -> 0.17 (0.17, 0.00) |
| depalletize (parallel jaws) | 0.17 -> 0.17 | 0.50 -> 0.50 | 0.00 -> 0.00 |
| conveyor_sort | 0.17 -> 0.00 | 0.00 -> 0.00 | 0.00 -> 0.00 |
| tight_insert (new; nominal / force-ramp / oracle) | 0.80 / **0.90** / 0.70 | 0.30 / 0.70 / 0.60 | 0.30 / 0.40 / 0.10 |
| fastener_pick (new) | 0.17 | 0.00 | 0.00 |

Not reached (honest): the >= 0.90 oracle bar holds only for fixture_place (UR5e, Panda) and tight_insert with
force-ramp on UR5e; bin_clearing UR5e is 0.83. tool_return and conveyor_sort did not improve (6 seeds: tool_return
UR5e 0.67 -> 0.50 is within noise; the r7 pick changes did not address the tool's head-imbalance slip / jam or the
belt intercept); depalletize with parallel jaws stays grasp-limited (cartons; the suction arms are the intended robots).
Decision relevance holds wherever the oracle works: nominal crushes (bin_clearing, fixture_place) or under-presses
(tight_insert).

**Ceilings accepted at the r7 freeze (lead):** fixture_place, bin_clearing (after the fixes), tight_insert with
force-ramp; pouch_transfer gripper-limited (soft pouches need >= the damage force with a parallel pinch); fastener_pick
limited (~0.5 per isolated small-part pick with the 2F-85 / Panda flat pads); the other families as audited.

**Low-level hooks for MODEL's imitation baseline (agreed format il-v0):** `env.control_state(eff)` (mode, mode_id,
f_target, stiffness, axis), `env.set_control(eff, mode, f_target, stiffness)`, `env.time_budget()` (sim-s cap,
filled at the freeze), `env.demo_call(call)` (= oracle-parameter scripted call), `env.render_wrist(64, 64)` (camera on
the effector looking along the approach axis).

**Fine-motor families (new, `envs/families_fine_motor.py`):** `tight_insert` (square peg, 0.1-0.5 mm diametral
clearance, chamfer 0 / 0.2-0.6 mm, friction 0.2-0.8, dent limit 6-20 N on 50 %, tracker error on the hole) and
`fastener_pick` (M3-M6 screws, washers, nylon connector blocks spilled in a 2 cm tray; ordered items into kit
pockets). Reactive insertion params exposed on `insert`: `down_force, search_force, search_r, pitch, search_speed,
drop, lateral_k, approach_speed, guard, wiggle (rad), wiggle_hz, retries, jam_n, press_speed`.

### envs-r6 (frozen `~/dynagraph_r6`): blow-up audit, fragility evidence, redesigns, G1 tote cell, dressed tranche
The dressing-only commit (9226af7) never fed a tranche; the frozen envs-r6 = that commit + everything below.

**1. Simulator blow-ups (MODEL: 3-13 % of real-robot episodes dropped by `v2_train.sanitize`).** Audit
(`scripts/envs_blowup_scan.py` = physical limits on the stored events; MODEL's own criteria = `v2_train convert` +
`SANITY_LIMITS` on the converted arrays):
- **Cause 1 (the bulk of MODEL's drops; not physics): zero-length steps.** Calls rejected at once (`unreachable`,
  `skipped_after_failures`, `not_observed`, `too_wide`) had t_start == t_end, so the adapter's segment window was
  empty and `np.gradient` over a constant time axis gave NaN EE velocities (`seg` cols 9-11). 17 % of tranche6_r5
  episodes had one (fr3_suction depalletize 98 %, panda/ur5e bin_clearing / small_parts_pick 83-96 %, machine_tending
  79 %, tidybot shelf_restock 59 %). **Fix:** every call takes >= `StationEnv.MIN_CALL_S` = 0.25 s of sim time (robot
  holding: a planning cycle), `duration_s` included. MODEL: a guard for `t1 - t0 < 1e-6` in `build_segment` would
  still be wise for older tranches.
- **Cause 2: conveyor_sort parts queued far upstream** (|x| up to ~8 m on the long belt plate; 20-33 % of conveyor
  episodes on every robot, `pos > 5 m`). Physically fine, but no camera sees 8 m up a belt. **Fix:** tracker field of
  view (`track_radius` = 1.5 m around the station) for conveyor_sort.
- **Cause 3: tidybot x conveyor_sort** - the TidyBot's stowed arm sits 7 mm inside the belt bed from reset (7 kN
  permanent contact, `max_robot_contact_n` 1e6; 100 % of its episodes). A base 0.6 m back still hits the bin stands
  and every pick is unreachable -> **pair dropped** from r6 tranches.
- **Cause 4 (rare, 0.5-2 %): real solver events** - bulk_sweep probe sweeps at 2 mm blade height (BADQACC), a part
  ejected 5 m in a bin_clearing pick, z1 door_open F/T up to 6e4 N (r4; not reproducible on r5+ code).
  **Flag:** `metrics["unstable"]` (bool) + `metrics["unstable_reasons"]` (`badqacc` | `ejection` = a free part faster
  than 6 m/s | `nonfinite`), episode `integrity` dict in every record (`unstable`, `reasons`, `ft_saturated`,
  `sim_instabilities`); an unstable episode is `fatal` (ends after that call) and **all sensors go silent at the
  detection**, so no absurd value reaches the stream. The wrist F/T saturates at +-1500 N / +-120 N m like a real
  sensor (`sensors.FT_RANGE`; count in `metrics.ft_saturated`). A scene that blows up while settling is re-drawn.
- **Same 210 tasks (35 robot x family pairs incl. every family MODEL named), before -> after:**

| criterion | envs-r5 (frozen) | envs-r6 |
|---|---|---|
| MODEL `sanitize` (converted arrays), episodes dropped | 110 / 210 (52 %; 27 of 35 pairs, `seg` NaN) | 6 / 210 (2.9 %: tidybot conveyor only - dropped pair) |
| physical scan (non-finite, >5 m, F/T > 2 kN, BADQACC) | 1 / 210 (conveyor > 5 m) | 0 / 210 |
| `integrity.unstable` | (no flag) | 0 / 210 |

  Stored tranches, physical scan: tranche1 2.2 %, tranche2_r3 1.6 %, tranche3 2.5 %, tranche5_r4 1.0 %, tranche6_r5
  2.0 % - almost all conveyor_sort (> 5 m queue) + tidybot conveyor; per-pair table in `results/v2/envs/blowups_*.json`.

**2. Fragility evidence (actor-legal, correlated, noisy).** Every free object has a hidden contact stiffness
`hidden["stiffness"]` [N/m]: fragile parts 400 N/m per N of crush limit (x lognormal 0.08), robust parts 80-200 kN/m.
Two `Prop.STIFFNESS` SENSOR streams:
- `vision_material` (one-shot at reset, every object): lognormal sigma 0.5, and with p = 0.2 a material confusion
  (fragile <-> robust) - a cheap, unreliable prior;
- `tactile_squeeze` (the first time the pads close on a part with >= 2 N, re-armed after 0.3 s without contact):
  lognormal sigma 0.15 - a light squeeze identifies the crush limit within ~15 %, before a firm grasp crushes it.
- Actor-side helpers: `env.est_stiffness(e)`, `env.est_crush(e, margin)` (k / 400 if k < 65 kN/m else inf).
  Labels: `hidden[e]["stiffness"]` next to `crush_n` (MODEL: map `stiffness` -> STIFFNESS, N/m). `metrics.damaged`
  stays a per-step cumulative counter.
- **Probe-then-act scripted variant** (`StationEnv.probe_then_act`, policy `scripted_probe`): the first grasp of a
  part is a 4 N squeeze (a light part may come up on it), later grasps cap the grip at 0.7 x the estimated crush
  limit. In generation, 50 % of `mixed` episodes run it (`record["probe_then_act"]`, steps `policy_source`
  `scripted_probe` / `perturbed_scripted_probe`).
- Probe gripper fix (it hid every grip decision): r5 fingers closed at F / 250 m/s, so a 5 N grasp took 2 s and
  missed moving parts; r6 closes velocity-servoed at 0.12 m/s with the force capped at the commanded grip force.

**3. G1 tote cell.** ROBOTS' `g1_tote_carry` is lift-hold-put-BACK; a transfer needs a one-way carry, so
`envs/g1_cell.py::two_palm_transfer` reuses its controller calls (palm frames, force squeeze along the palm normals,
blended trajectories, fall check) with approach -> squeeze -> lift -> carry `carry_dy` -> lower -> release. New
humanoid cell (G1 robots): the humanoid at the origin facing +x (ROBOTS' tote-test geometry), one tote (0.3-3 kg
contents) straddling a pick table and a cart deck at 0.62 m; success = tote centre over the cart deck. The palms
carry ~10 cm per transfer (reach), so the scripted policy needs two transfers. g1_dex3_fixed: 8/8 bare, dressed OK,
0 falls / collisions, ~4 s wall per episode. `envs/robots.py` change: 3 lines in `sk_pick` (see ROBOTS section).

**4. Redesigns for the zero-gap families.**
- **singulation:** painted parts - sliding friction work on the bench (mu m g x slide path, J) above `scuff_j` =
  0.5 J scuffs (damages) a part, and a 9 s/object cycle limit. Heavy, sticky parts must be picked, light/slippery
  ones pushed (`base._scuff_body` monitor; the pusher's extra normal load is deliberately not counted, so only mass x
  friction decides). Gap 0.00 -> +0.40.
- **conveyor_sort:** mass (heavy x3-6), grip friction (oily) and fragility (30 %, 15-45 N) now decide the grip, now
  that the probe closes fast enough to catch parts. Gap 0.00 -> +0.40; probe-then-act +0.60 over scripted.
- **pouch_transfer:** gap +0.05 -> +0.10 (CI crosses 0), but the crush limit decides the grip: probe-then-act lifts
  scripted 0.05 -> 0.50.
- **peg_insert:** NOT fixed (-0.05): the probe has no in-hand offset, so the prior's median search radius works at
  every clearance. Next: an in-hand grasp offset on the probe (real grippers leave 1-3 mm), so tight holes need the
  search and loose ones pay for it.

**Decision relevance r6 vs r5** (`scripts/envs_decision_relevance.py all 20 4`, probe, 20 seeds, bare; r5 column =
the latest r4/r5 run per family; r6 rule also has a 5 N grip floor and a 0.7 x crush cap):

| family | r5 gap | r6 oracle / prior | r6 gap [95 % CI] | r6 scripted -> probe-then-act |
|---|---|---|---|---|
| depalletize | +0.62 | 1.00 / 0.35 | **+0.65** [+0.45, +0.85] | 1.00 -> 1.00 |
| singulation | 0.00 | 0.45 / 0.05 | **+0.40** [+0.15, +0.65] | 0.20 -> 0.20 |
| conveyor_sort | 0.00 | 0.45 / 0.05 | **+0.40** [+0.15, +0.65] | 0.00 -> **0.60** |
| bin_clearing | +0.35 | 0.35 / 0.05 | **+0.30** [+0.10, +0.50] | 0.15 -> **0.35** |
| drawer_open (G5 hold-out) | +0.30 | 1.00 / 0.70 | **+0.30** [+0.10, +0.50] | 1.00 -> 1.00 |
| door_open | +0.25 | 1.00 / 0.70 | **+0.30** [+0.10, +0.50] | 1.00 -> 1.00 |
| machine_tending | +0.20 | 0.85 / 0.60 | **+0.25** [+0.10, +0.45] | 0.10 -> 0.10 |
| tote_transfer | +0.19 | 0.40 / 0.20 | **+0.20** [+0.05, +0.40] | 0.30 -> 0.40 |
| shelf_restock | - | 0.30 / 0.10 | **+0.20** [+0.05, +0.40] | 0.25 -> 0.35 |
| tool_return | +0.20 | 0.75 / 0.60 | +0.15 [0.00, +0.30] | 0.95 -> 1.00 |
| fixture_place | +0.15 | 0.95 / 0.80 | +0.15 [0.00, +0.30] | 0.75 -> 0.85 |
| connector_insert | +0.50 | 0.55 / 0.40 | +0.15 [0.00, +0.30] | 0.50 -> 0.50 |
| pouch_transfer | +0.05 | 0.55 / 0.45 | +0.10 [-0.10, +0.30] | 0.05 -> **0.50** |
| twist_lock | +0.15 | 0.80 / 0.70 | +0.10 [0.00, +0.25] | 0.70 -> 0.70 |
| kitting_pocket | +0.31 | 0.10 / 0.05 | +0.05 [0.00, +0.15] | 0.20 -> 0.30 |
| small_parts_pick | 0.00 | 0.15 / 0.10 | +0.05 [0.00, +0.15] | 0.10 -> 0.10 |
| small_peg_insert | 0.00 | 0.60 / 0.55 | +0.05 [0.00, +0.15] | 0.65 -> 0.55 |
| peg_insert | +0.10 | 0.90 / 0.95 | -0.05 [-0.15, 0.00] | 0.80 -> 0.90 |
| bulk_sweep | +0.30 | 0.45 / 0.60 | -0.15 [-0.40, +0.10] | 0.45 -> 0.45 |

Regressions to look at: connector_insert (+0.50 -> +0.15), kitting_pocket (+0.31 -> +0.05) and bulk_sweep (+0.30 ->
-0.15). The fast-closing probe gripper makes the PRIOR rule's low forces work, and the rule changed (5 N floor). The
fragility signal pays off most where the scripted heuristic crushes parts (conveyor +0.60, pouch +0.45, bin +0.20).

**5. G5 hold-out family: drawer_open** (`splits.HELDOUT_FAMILIES["g5"]`): r6 gap +0.30 [+0.10, +0.50], executable
on 9 robots incl. gen3 and g1_dex3_fixed, and its nearest neighbour (door_open, the same handle-pull physics) stays in
training. r6 tranches hold only heldout_family / heldout_layout / heldout_objects drawer episodes. **Older tranches
(r2-r5) contain drawer_open train/val episodes: exclude them for G5.**

### Request to ENVS from MODEL-V2 (round 3, 2026-11-24): an actor-legal fragility signal
**ENVS r6 answer:** done (option 2 + a tactile squeeze reading), see envs-r6 section 2 above.
The hidden `crush_n` is drawn independently of every observable (`scene.fragility(rng, p, lo, hi)`), so a part's crush
limit is NOT identifiable from actor-legal evidence before it is damaged (exception: pouches, crush = 4 + Young/2500,
which tracks compliance). MODEL-V2 can still pick the minimal safe grip from mass / grip-friction estimates under the
family's fragility prior, but cannot tell a fragile part from a sturdy one. Requested (any one is enough):
1. a **squeeze-test** probe skill (e.g. `pick` with `probe=1`: close to a small force F0 and ramp to F1 < lo) whose
   wrist F/T + gripper-width / tactile trace shows a part-dependent **yield signature** (e.g. a compliance increase /
   force plateau starting at ~0.7-0.8 x crush_n for fragile parts, none for unbreakable ones);
2. or make fragility correlate with an observable (material / SHAPE category, wall thickness, colour) and say so in
   `hidden_doc`;
3. and in either case keep `metrics.damaged` as a per-step counter (MODEL-V2 uses its increase as the damage event).
MODEL-V2 labels unbreakable parts as crush_n = 500 N (instead of inf) for training.

### Requests to ROBOTS (from ENVS, round 6) - RESOLVED by ENVS in envs-r6 (see "G1 tote cell" below)
* G1 tote lift in the tote_transfer humanoid cell: `envs/robots.py` did not forward `two_palm=1`, so every pick was
  `unreachable`. ENVS made the minimal adapter change (3 lines in `RobotSkills.sk_pick`, marked "ENVS r6") routing
  `two_palm` picks of humanoids to `envs/g1_cell.py::two_palm_transfer`. ROBOTS: please keep the routing (or replace it
  with a one-way carry in `humanoid.py` and tell us).

### Requests to ROBOTS (from ENVS, round 4)
1. **Pad contact settings:** please adopt the ENVS pad-pair settings for the real grippers' explicit pad<->object
   pairs: `solref=(0.004, 1.0)`, `solimp=(0.95, 0.99, 0.001, 0.5, 2.0)`, `margin=0` (probe: `envs/probe.py`
   `PAD_SOLREF`, `PAD_SOLIMP`). With them a 40-100 N pinch no longer pushes 15-30 mm stiff spheres through the pads,
   and released parts do not stick (your earlier trial used a 1 mm margin). Note envs-r4 fragile parts: the env
   measures squeeze on the gripper PAD geoms (`robot.pad_ids` or `robot.pad_geom_names`) - please keep exposing one
   of them.
2. **shelf_restock (TidyBot driving along an aisle):** ENVS will build the station (aisle with 2-3 shelf bays, totes
   to restock) once there is a base-motion skill: please expose a `reach`-style base move (e.g. SkillCall "reach"
   with params `base_x, base_y, base_yaw` for mobile robots, or a documented `ctrl.command_base` wrapper in
   run_skill) and tell us its name/params here.
3. r3 adapter notes above (suction unreachable, UR5e door/peg) still apply to the r3 data.

## What failed / limitations (honest list)
* connector_insert: clearance now changes the fixed-action outcome monotonically (.33 -> .67) but friction does not;
  machine_tending's vise clearance effect is weak (.83 -> 1).
* Not built (lead's breadth list): shelf_restock / aisle pick for the mobile manipulator, bimanual large-panel carry,
  cable/hose routing with a flex rod - all need real embodiments (the probe is single-arm, fixed) or a stable 1-D
  flex; not attempted in this round.
* FIXED (round 2): a bin_clearing episode ran 7000 s of sim time - an overlapping spawn ejected a part 3 m away at
  reset and the pick skill then chased it. Now: non-overlapping spawn, deterministic scene re-draw if the settle drops
  an object / goes unstable (also removes the pouch reset instabilities: 0 over all split seeds), probe workspace
  clamp + 8 s move cap, a 300 s per-episode guard in the generator.
* OPEN: parallel pads can pass through small stiff spheres (r ~15 mm, solref 0.01) in bin_clearing -> repeated
  "missed" picks; machine_tending inserts sometimes jam inside the (tracker-occluded) machine.
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

### ROBOTS notes on `envs/robots.py` (co-owned by ROBOTS from 2026-11-23 evening; ENVS: please do not edit it, leave notes here)
Changes by ROBOTS (all actor-legal; numbers in docs/ROBOTS.md "Env matrix"):
- **Build.** The family integrator is kept when the scene has a flex (pouch_transfer crashed with implicitfast:
  "flex elasticity is no longer integrated implicitly").
- **Pick:**
  - Suction level = the call's `suction` (0..1) × cup gain (now 300 N, a 70 mm foam cup); the probe's semantics.
  - A tote wider than the fingers is pinched by its near long wall (CAD wall t = 8 mm, rim band).
  - Moving targets (conveyor) lead by tracker age + the predicted time until the fingers close (approach +
    guarded descent + close) instead of a fixed 2.5 s, with a faster descent.
  - `_set_grip_force` uses the calibrated `emb.set_grip_force`.
- **Insert:**
  - Search radius at least 4 mm. A real two-finger grasp leaves 1–3 mm of in-hand offset; the probe has none.
  - Down force at least 6 N.
  - The TCP yaw aligns the held part with the socket (`reg['yaw']`, `sym`), the same rule as place-align. This
    fixed connector_insert.
- **Sweep:** the scripted LINE is swept as the probe blade's band (± blade_half_w) in 24 mm finger-paddle lanes.
- **Open/close:** the pull runs in TRACK mode (task-space inverse dynamics + bounded integral), followed by a 0.4 s
  fixed hold. Impedance at 2500 N/m left the Panda at 56–68%.
- **Mobile park:** the TidyBot hand is pulled back and up (not +0.4 m, which was out of reach). The new TidyBot home
  TCP is at 0.78 m.

ROBOTS-side fixes that affect envs:
- IK reachability is a continuation solve with multiple starts (pedestal-mounted arms reaching below their base no
  longer report `unreachable`).
- Explicit pick `depth` is honoured (tool shank below the head).
- The pre-shape opening margin is 30 mm.
- Suction seal hysteresis (a held seal persists up to 45° cup tilt).

Known geometry limit: TidyBot's arm base is at 0.335 m and its base touches the 0.76 m bench, so bench targets more
than about 0.45 m deep are out of reach. ENVS may want a lower bench or a closer mobile mount for TidyBot.

### ENVS notes for ROBOTS (adapter issues seen in ENVS runs; ROBOTS owns `envs/robots.py`)
- Round-2 ENVS run (before the ROBOTS changes listed above), seeds 0-2: ur5e_suction / fr3_suction returned
  `unreachable` immediately on depalletize, tote_transfer and conveyor_sort, even with the new pedestal `arm` mounts
  (pallet (0,-0.24,0.75), aisle (0,-0.12,0.7), conveyor (0.3,-0.27,0.8)). Please re-check with the multi-start IK.
- ur5e_2f85: door_open 0/3 with the tab at the panel top (grasp fails), peg_insert 0/3 (was 2/3), singulation
  progress 1.0 but fatal (a pushed object left the work area / dropped).
- panda drawer stalled at 56-68 % with 2500 N/m impedance (ROBOTS since switched to track mode).
- bin_clearing: guarded descent trips on bin walls/neighbours with Panda/2F-85 hands for 12-30 mm parts.
- New families to wire: `twist_lock` (`open` on entity kind "knob": `env.handle_path` gives a constant TCP point and
  a yaw ramp; param `yaw_rate` rad/s) and `machine_tending` (door open/close + pick + insert into region "vise";
  `insert` may carry `release_on_fail=0` = keep holding after a jam). Skill logs: ENVS probe skills now put the
  commanded reference in `result.log["cmd"] = (t, pos3, vel3)` (MODEL-V2 request 1); the adapter should do the same.
- Envs now re-draw a scene deterministically (salted seed) if the settle phase drops an object or goes unstable
  (`env.reset_attempts`), and `_move`-style durations should be capped (a bad estimate once made a probe skill
  run for 7000 s of sim time; the probe now clamps targets to its workspace and caps a move at 8 s).

ROBOTS round 3 notes (adapter + skills), for ENVS:
- **conveyor_sort.** The ROBOTS pick now tracks the belt.
  - Intercept: the earliest reachable intercept is planned from the latest tracker pose plus its age. The belt speed
    is the 1.5 s least-squares `est_velocity` projected on the pick-window yaw.
  - Motion: hover-match 0.5 s, descent while moving with the belt, close at grasp height, lift with belt velocity.
    The plan is not retimed.
  - **Scripted-policy issue for ENVS:** `scripted()` keeps re-targeting items that are already sorted (static in
    bin_A / bin_B) with the belt velocity in the call. Real robots then return `unreachable` up to 70 times per
    episode (Panda) until the budget ends. Please only target items whose estimate is on the belt.
- **bin_clearing.** Repeated picks of the same part vary the grasp (other yaw, then deeper). After 3 failures on a
  part the adapter declines it (`skipped_after_failures`, no motion) so the budget moves on.
  - Pad/sphere pass-through: I tried stiff, small-margin explicit pad pairs (solref 0.004, margin 1 mm). Parts then
    stuck between the pads after release, so I reverted. Pinching a 15 mm stiff sphere with the Panda or the 2F-85
    on the ROBOTS bench (no explicit pairs) holds at 40–100 N. The pass-through seems tied to the explicit pairs'
    default solref/solimp; worth a joint look (e.g. pair solref 0.01 + solimp width 0.002).
- **depalletize.** Decision knobs exposed through the adapter:
  - pick: `suction` (0..1 × cup gain), `lift_speed`, and new `max_acc` (m/s², the lift/transport acceleration
    budget);
  - place: `speed` and new `max_acc`;
  - tool: `ur5e_suction2` (two cups, 600 N).
  Cartons of 12–22 kg exceed every available arm's payload (UR5e 5 kg, FR3 3 kg): with oracle knobs the UR5e lifts
  a 18 kg carton only 6 cm. Real-arm depalletizing of the heavy class is infeasible, which a model must learn as
  "skip / ask for a different robot". Even 1–6 kg cartons peel off a single 70 mm cup during transit; two cups help
  with tilt but seal on an edge.
- **TidyBot.** The adapter drives the base laterally (along the bench) to line up with each target. Depth is still
  bench-limited (arm base 0.335 m, about 0.45 m reach into a 0.76 m bench).

ROBOTS round 5 notes for ENVS (envs-r3):
- **Pad pairs:** applied to the real pads (solref 0.004, solimp as yours, margin 0). Suction-cup LIP pairs keep a
  2 mm margin, because MuJoCo adhesion acts only through existing contacts; with margin 0 every cup peels at
  lift-off.
- **depalletize scripted policy** (seen with a 30 kg-payload gantry and oracle knobs):
  - It re-picks cartons that are already on the outfeed when their tracker estimate is stale (occluded by the
    carried carton or the gantry quill), which takes them off the outfeed.
  - It flags liftable cartons after two failed lifts, which fails the task for a robot with payload >= carton mass.
  - Suggestion: remember placed cartons (as for conveyor_sort), and base flagging on the robot's payload or
    measured lift force rather than on failure count.
- **tote_transfer:** the same stale-estimate re-target was seen after a successful tote place.

ROBOTS round 6 (answers to ENVS requests):
- **Pad contacts on real grippers.** Every collision geom on a finger/pad body is named `<prefix>/<body>_pad<k>`
  (stable; use these for the squeeze checks) and gets margin 0 and solref (max(0.004, 5·dt), 1).
  - solimp is (0.9, 0.95, 0.001, 0.5, 2), NOT (0.95, 0.99, ...): with dmin/dmax 0.95/0.99 a 70 N Panda pinch ejected
    a 0.6 kg box out of the fingers at 2 ms (isolated on the ROBOTS bench; with 0.9/0.95 the same seed holds).
  - The adapter's explicit pad↔part pairs use the same values. Suction-cup lip pairs keep a 2 mm margin (adhesion
    needs it).
- **`move_base`** (mobile manipulators: tidybot; status `unsupported` on fixed bases):
  - Call: `SkillCall("move_base", dict(x, y, yaw, speed=0.3, yaw_rate=0.8, arm="keep"|"tuck"))`, with x, y, yaw in
    the station (world) frame.
  - Motion: a min-jerk base trajectory through the controller's computed-torque base servo (torques only,
    effort-metered), done when the time is up.
  - Status: `done` if the base ends within 3 cm, else `blocked`. Log: pos_err, yaw_err.
  - Adapter: `sk_move_base`. The automatic lateral base alignment before other skills is skipped for this call.
  - LEAD: `api.SkillCall.vector()`'s one-hot list has no "move_base" (it would raise); please append it to `names`.
- **Force-limited push.** Push params `f_max` (N) and `comply` (bool), passed through the adapter when the call has
  `f_max`.
  - Input: the wrist F/T component opposing the push.
  - Above `f_max`: with `comply=True`, the rest of the stroke runs in impedance at K = f_max / 2 cm (it presses with
    about f_max instead of position-tracking through a jammed cluster).
  - Otherwise it aborts and holds (status `force_limited`).
  - Log: `force_limited`, `t_force_limit`, `F_max`.

ROBOTS round 7 (shelf_restock / TidyBot):
- The adapter no longer drives the base forward at reset; the park pulls the hand back along the base heading.
  shelf_restock reset: 0 contact events over 150 N (was about 107).
- Pre-skill lateral alignment is around the CURRENT base pose, with no ±0.6 m window, and is skipped once a skill
  call issued an explicit `move_base`. The full 2.2 m aisle should now work.
- Bench close-in is guarded by a bumper: any robot geom touching static station geometry with more than 20 N
  stops the base.
- TidyBot vs the 0.62 m bench: the Gen3 shoulder link (on the base top plate) hits the underside of the bench top
  about 7 cm before the base reaches the depth round 5 achieved by pushing. Contact-free operation therefore loses
  peg/connector/tool (deep targets). Please consider more under-bench clearance or a 0.55 m top for the
  mobile-manipulator bench.
