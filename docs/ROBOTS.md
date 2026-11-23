# ROBOTS (v2 embodiments, controller, skills)

Owner: agent ROBOTS. The code lives in `src/physmem/embodiments/**` and `src/physmem/skills/**`. The gate scripts
are `scripts/robots_*.py`, and their results are in `results/v2/robots/` on gb10-direct. The G1 humanoid
(`humanoid.py`, `wbc_qp.py`, `scripts/robots_g1.py`) was built by a ROBOTS sub-agent.

## 1. Python API (exact)

```python
import mujoco, numpy as np
import physmem.embodiments as E
from physmem.v2.api import SkillCall

E.names()   # ['aloha','fr3_suction','gen3_2f85','panda','so100','tidybot','ur5e_2f85','ur5e_suction','z1','g1_dex3','g1_dex3_fixed']
emb = E.make("ur5e_2f85", prefix="r0", grip_force=150)   # optional per-pad squeeze in N (calibrated)
emb = E.make("ur5e_2f85", prefix="r0")       # api.Embodiment; every name in the robot gets the "r0/" prefix
emb.spec                                      # api.EmbodimentSpec (kind, effectors, gripper, reach, payload, max speed/force, dof, ...)

spec = mujoco.MjSpec()
E.scene_defaults(spec)                        # REQUIRED: 2 ms, implicitfast, elliptic cones, impratio 10 (attach drops robot options)
# ... the floor, tables, objects ...
emb.build(spec, mount_pose=((x, y, z), (qw, qx, qy, qz)))   # the robot faces +x of the mount frame (front_yaw normalised)
model = spec.compile(); data = mujoco.MjData(model)
emb.bind(model, data)                         # resolves ids; call after every compile
emb.reset(model, data)                        # home posture (TCP pointing down over cfg.home_target), zero velocity, mj_forward

ctrl = emb.controller()                       # api.Controller: TaskSpaceController (arms) / WholeBodyController (g1_dex3)
# EVERY physics step (the controller is also the effort meter, so it must see every step):
ctrl.apply(model, data)                       # = data.ctrl[emb.act_ids] = ctrl.step(model, data)
mujoco.mj_step(model, data)
```
For several robots in one scene, give each its own prefix (`"r0"`, `"r1"`) and call each robot's `apply` before
`mj_step`.

Task-space control (api.TaskTrajectory, or `E.make_traj` / `E.BlendedTrajectory`, which can also take via-point
velocities and a mode switch):
```python
p, q = emb.tcp_pose(data, "hand")             # encoders + FK (actor-legal)
pred = ctrl.command(E.BlendedTrajectory(effector="hand", waypoints=[(p1, q1), (p2, q2)], durations=[0.8, 0.6],
                                        via_vel=[v1, None], gripper=0.0))   # via_vel optional; None = planner blends
# pred: {'scale': retime factor >= 1, 'peak_v', 'peak_a', 'peak_j', 'cost', 'duration'}
ctrl.set_mode("guarded", "hand", threshold=8.0, axis=(0, 0, -1), bias=F0)  # trips + holds when the wrist force exceeds threshold
ctrl.set_mode("force", "hand", axis=(0, 0, -1), force=10.0, approach_speed=0.02, compliant=True)
ctrl.set_mode("impedance", "hand", stiffness=400.0, rot_stiffness=20.0)
ctrl.set_mode("track")                        # default
ctrl.set_gripper("hand", 1.0)                 # 0 open .. 1 closed (force-limited squeeze)
ctrl.done(); ctrl.tripped("hand")             # done = time is up or a guard tripped; NEVER waits for convergence
ctrl.effort                                    # api.EffortMeter, cumulative;  m = ctrl.acc.open(); ...; peaks = ctrl.acc.close(m)
ctrl.effort_weight = 0.5                       # 0..1: tighter speed/acc/jerk budget -> retimed slower/smoother
emb.proprio(data)  # per effector: q, dq, ee_pos, ee_quat, ee_vel, ee_angvel, wrist_force, wrist_torque, grip (0..1), grip_contacts
emb.wrist_ft(model, data, eff); emb.gripper_state(model, data, eff)
emb.set_grip_force(eff, newtons); emb.max_grip_force(eff)   # per-pad squeeze (N), calibrated; returns the force set
emb.cup_seal(model, data, eff)                               # suction robots: (sealed, n_contacts)
```
Mobile base (tidybot): `ctrl.command_base((x, y, yaw), duration)` sends a min-jerk move to the base joints, servoed
by computed torque.

IK for reachability (kinematic only; uses a scratch MjData, never the live state):
`physmem.embodiments.ik.solve_ik(emb, model, data, eff, pos, quat)`.

**TCP convention (all robots):** TCP z is the approach direction (out of the gripper), TCP y is the finger closing
axis, and the origin is between the pads. For single-jaw grippers (Z1, SO-100) the origin is at the fixed jaw's face,
and `GripperCfg.jaw_dir` gives the side of the moving jaw. Site name: `"<prefix>/tcp_<effector>"`.

### Skills
```python
from physmem.skills import SkillContext, ObjectEstimate, run_skill, SKILLS
ctx = SkillContext(model, data, emb, on_step=hook)   # hook(m, d) runs after EVERY mj_step (sensors, recording, other robots)
est = ObjectEstimate(pos=..., quat=..., half_size=(hx, hy, hz), shape="box")   # ACTOR-LEGAL estimate (tracker + geometry), never truth
res = run_skill(ctx, SkillCall("pick", dict(speed=0.4), target="obj3", effector="hand"), est)
res.ok, res.status, res.effort, res.duration_s, res.log     # api.SkillResult; log has per-skill measurements + window peaks
```
Reuse **one ctx per robot per episode**. It remembers grasps (`ctx.held[eff]`: TCP-to-object-bottom offset and the
object's offset in the TCP frame), which place, insert and handover need. Use `before_step=` to apply other robots'
controllers. Skills move the robot only through the controller. They never write poses, add welds, apply forces or
read object state; `tests/test_skills.py` checks this statically.

## 2. Robots, assets, torque limits

| name | kind | model (asset) | gripper | dof | torque limits (N m; ctrlrange = these) |
|---|---|---|---|---|---|
| panda | arm | Franka Emika Panda + Franka Hand | parallel | 7 | 87,87,87,87,12,12,12; hand drive limit 280 (tendon) -> default 70 N per pad (Franka spec: 70 N continuous) |
| ur5e_2f85 | arm | UR5e + Robotiq 2F-85 | parallel (linkage) | 6 | 150,150,150,28,28,28; 2F-85 drive limit 28 (raised from menagerie's 5) -> default 100 N per pad, max ~250 N (Robotiq spec 20-235 N) |
| ur5e_suction | arm | UR5e + vacuum cup (adhesion actuator) | suction | 6 | as ur5e; cup 120 N |
| fr3_suction | arm | Franka FR3 + vacuum cup (adhesion actuator) | suction | 7 | 87 x4, 12 x3; cup 120 N |
| gen3_2f85 | arm | Kinova Gen3 7-dof + 2F-85 | parallel (linkage) | 7 | 105 x4, 52 x3; 5 |
| z1 | arm | Unitree Z1 + its jaw | parallel, single moving jaw | 6 | 30,60,30,30,30,30; jaw 8 (limit 30) |
| so100 | arm (low-cost) | TRS SO-ARM100 | parallel, single moving jaw | 5 | 3.5 each; jaw 3.0 |
| aloha | bimanual | ALOHA, 2 x ViperX 300s | parallel x2 | 12 | per arm 35,144,59,22,35,35; gripper 20 N |
| tidybot | mobile manipulator | TidyBot holonomic base + Gen3 + 2F-85 | parallel | 3+7 | base 800 N, 800 N, 300 N m (override); arm as gen3 |
| g1_dex3 | humanoid, FLOATING base | Unitree G1 29-dof + Dex3 hands | dexterous | 6+29+14 | model actuatorfrcrange (legs 88/139, waist 88/50, arms 25/5, hand 1.4-2.45) |
| g1_dex3_fixed | humanoid, pelvis welded (labelled fallback) | same | dexterous | 29+14 | same |

**Actuators.** Every actuator is converted to a torque motor: fixed gain 1, no bias, `ctrlrange = forcerange =
torque limit`. The limit comes from the actuator `forcerange` if set, else the joint `actuatorfrcrange`, else an
override; the only override is TidyBot's base joints. Only torques move the robots.

**Changes to the models (documented):**
- **Grip force (calibrated, `scripts/robots_grip_calib.py`).** Each gripper closes on a fixed block, and the pad
  normal force per side is measured against the drive effort: `GripperCfg.pad_per_drive` is 0.68 for Panda, 9.07
  for the 2F-85, about 25 for Z1 and about 15 for SO-100 (that one is nonlinear). Defaults are 70 / 100 / 80 / 40 N per
  pad. Pad friction is 1.2 (silicone-like, condim 4). Menagerie's 2F-85 tendon limit of 5 gave only about 45 N; it is
  now 28. ENVS: please replace `envs/robots.py` GRIP_N_MAX with `emb.max_grip_force(eff)` and `emb.set_grip_force`.
- **Suction (`fr3_suction`, `ur5e_suction`).** A vacuum-cup tool body (stem plus a 40 mm compliant pad with
  margin/gap 4 mm) sits on the flange, with a MuJoCo **adhesion actuator** on the cup body (gain 120 N, roughly a
  40 mm cup at -0.95 bar). The controller sets the adhesion control to 1 only while the cup is commanded on AND
  sealed. Sealed means pad contacts with a non-robot surface whose normals are within 20° of the cup axis; this plays
  the role of a pressure switch. With no seal there is no holding force. No welds.
- **Gripper close is event-based.** It steps until pad contact plus the fingers stopping, or full closure, capped at
  1.5 s. It used to be a fixed 0.55 s, which was shorter than the 2F-85's ~1 s close, so the 2F-85 never grasped
  2 cm pegs.
- **Pick pre-shapes the opening** to the part width + 15 mm, and place releases the same way.
- **Single-jaw grippers (Z1, SO-100)** clear the fixed jaw by 8 mm.
- Finger drives get implicit joint damping sized so a free close takes about 0.4 s at full force (gearbox/back-EMF
  model). Without it, an explicit finger servo at 2 ms either chatters or stays weak.
- Z1 arm joints get armature 0.02 kg m², a reflected-rotor-inertia model. The menagerie file has 0, which made the
  torque loop ring.
- G1 hand joints get damping 0.05 (sub-agent).

**Assets:**
- `~/dynagraph/assets/menagerie` is mujoco_menagerie @f3b6013 (universal_robots_ur5e, stanford_tidybot, unitree_g1).
- `~/dynagraph/assets/menagerie_extra` is mujoco_menagerie @370c43965323c0944b433b202aac5058b5f907a9: a sparse
  clone of franka_emika_panda, robotiq_2f85, unitree_z1, trs_so_arm100, aloha, kinova_gen3 and trossen_vx300s,
  about 112 MB.

Licences, from each model's LICENSE file:

| model | licence |
|---|---|
| Panda | Apache-2.0 |
| Robotiq 2F-85 | BSD (ROS-Industrial) |
| Z1 | BSD-3-Clause (Unitree) |
| SO-ARM100 | Apache-2.0 |
| ALOHA | BSD (Trossen) |
| Gen3 | BSD (Kinova) |
| UR5e | BSD (ROS-Industrial) |
| TidyBot | MIT |
| G1 | BSD-3-Clause (Unitree) |

Code adapted from CommandAGI `scripts/hardware/wbc.py` and `physics_run.py` (same owner, internal) carries
attribution comments.

## 3. Controller (src/physmem/embodiments/osc.py, trajectory.py)

**Task-space inverse dynamics.** This is the operational-space form, run per effector on that effector's joints at
every 2 ms physics step:
- `qdd = Jbar (a_des - Jdot qd) + (I - Jbar J) qdd_posture`, with `Jbar = M^-1 J^T (J M^-1 J^T + lambda W^-1)^-1`,
  where W weights orientation (rot_weight < 1 for the 5-dof SO-100).
- `tau = M qdd + h + J^T f`, with `h = qfrc_bias - qfrc_passive`. This compensates gravity, Coriolis and passive forces.
- `a_des = a_ref + Kp e + Kd (v_ref - v) + Ki ∫e`.
  - Kp = 900, Kd = 60, Ki = 3000.
  - The integral is bounded to 2 mm·s, and the error is clipped to 5 cm (as in physics_run.py).
  - Orientation uses the same law (Kr = 600) on the quaternion error.
- **Torque limits:** the task part is scaled by the largest alpha in [0, 1] that keeps every joint inside its limit.
  This keeps the direction of motion. The result is then clipped.
- **Joint ranges:** each joint's acceleration is bounded so that it would stop 10 mm short of its range within 40 ms
  (from wbc.py).

**Trajectories.**
- Position: quintic Hermite segments with zero end accelerations. Rest-to-rest segments are exactly the min-jerk
  `smooth3`. Interior via-points carry a blend velocity, either chosen by the planner (the average of the adjacent
  segment velocities, scaled by the corner cosine) or given explicitly, so motion does not stop unless the index is
  in `stop_at`.
- Orientation: slerp with a min-jerk phase.
- Feedforward: position, velocity and acceleration are all fed forward.
- A plan is done when its time is up. There are no convergence waits.
- Queued plans start exactly when the previous one ends, or immediately if the robot is idle.

**Effort-aware retiming ("servo pain").** Each robot has a budget of peak TCP speed, acceleration, jerk and angular
rate (`cfg.limits`). Every command is time-scaled uniformly by k ≥ 1 so the predicted peaks stay inside the budget.
`effort_weight` w in [0, 1] tightens the budget to v·(1-0.5w), a·(1-0.75w) and j·(1-0.875w). The prediction
(`scale`, peaks, an acceleration²+jerk² `cost`) is returned by `command()`.

**Modes:**
- **track.**
- **impedance:** apparent stiffness K in N/m, damping 2√(K·m_eff), no integral.
- **guarded:** tracks the path. When the wrist force minus `bias`, projected on `axis`, exceeds the threshold, it
  freezes the reference at the measured pose, holds compliantly and marks `tripped`.
- **force:** along `axis`, approaches at `approach_speed` with a light force feedforward. After contact it commands
  `F + ki·∫(F - F_meas)` (integral bounded to ±1 N·s) with damping. The other axes are tracked; with `compliant`,
  translation is impedance-controlled and orientation stays stiff.

**Wrist F/T.** The sum of contact wrenches between the effector subtree and anything that is not the robot. This is
an ideal gravity-compensated wrist sensor; it is actor-legal proprioception.

**EffortMeter.** Accumulated every physics step (api.EffortMeter):
- Σ(τ/τmax)².
- Σ(dτ/dt/τmax)².
- Joint jerk Σ(dddq)², from second differences of the encoder velocities.
- Energy Σ|τ·dq|.
- Time with any actuator at ≥ 99% of its limit.
- Peak TCP speed.

Per-window meters (`acc.open()` / `acc.close()`) also return the peak normalised torque-rate, peak joint jerk and
peak torque. Each skill result carries its own window.

**Humanoid (g1_dex3).** A WholeBodyController using the ProxQP task-space inverse dynamics of `wbc_qp.py`, adapted
from wbc.py. It handles friction pyramids at the sole corners, torque limits and joint-range acceleration bounds.
Tasks: CoM over the feet, pelvis height and uprightness, torso upright, hand tasks with min-jerk feedforward plus a
bounded integral, and posture. The fingers have their own torque-limited PD. The API is the same, except that
`command_base` is not implemented (no walking).

## 4. Skills (src/physmem/skills/primitives.py); parameters in SI

| skill | params (defaults) | notes |
|---|---|---|
| reach | x, y, z, yaw?, speed 0.3, mode?, gripper? | min-jerk move |
| pick | grasp 0=top/1=side, yaw (offset), speed 0.3, depth (default clip(hz, 1.5–3.5 cm)), clear 3 mm, lift (reach-scaled 10 cm), squeeze 1.0, guard N, approach_speed 0.08 | Closing axis goes across the narrower horizontal extent. `too_wide` if the object is wider than 0.95·max_width. Kinematic IK reachability check, with the other yaw and then tilted approaches (low-dof arms) as fallbacks; `unreachable` if none works. Single-jaw grippers offset the TCP by half the width. Guarded descent, timed squeeze, closure/contact check (`grasp_failed`), lift, re-check (`dropped`). |
| place | x, y, z (surface) or est = target, yaw?, speed 0.3, guard 6 N, retreat | Puts the OBJECT (not the TCP) on the target using the grasp memory. Guarded descent against the wrist-force baseline, release, retreat. |
| push | theta (push direction in the object frame), offset (-1..1 along the face), speed, distance, height, tip_radius, closing_along | Run-up → constant-speed cruise from contact → min-jerk stop (explicit via-point velocities). Logs `speed_ratio` (mid-cruise), `release_ratio`, mid-stroke tracking error, F_max. Push height avoids catching the top edge with a linkage gripper. |
| sweep (bulk) | x0, y0, x1, y1 (region), direction, surface_z, height (≥ fingertip + 4 mm), speed 0.25, paddle 0.03, lane, overshoot, guard | Closed fingers as a paddle, overlapping lanes, lift-and-return, jam guard per lane. |
| insert (fine) | depth, force 10 N (press), search_force 4 N, search_r 4 mm, pitch 0.8 mm, speed 6 mm/s, guard 3 N, drop 3 mm, lateral_k 300 N/m | Guarded descent at the hole estimate, then an Archimedean spiral under constant force until the peg drops, then a compliant press (lateral impedance, stiff orientation). Status `done`, `not_found` or `jammed`. Needs the peg held by `pick` in the same ctx. |
| open / close | est.kind 'drawer' (axis = pull dir) or 'door'/'lid' (axis = hinge axis, hinge = a point on it), distance / angle, speed 0.15, stiffness 400 N/m, grasp 1 | Approach against the pull direction, guarded contact, grasp the handle, follow a straight line or arc in impedance mode, release, retreat. **Not yet gated (no articulated test bench).** |
| handover | x, y, z (meeting point), speed, grip_low | Giver (top grasp) and receiver (horizontal side approach) move together; receiver closes, giver opens and retreats. Bimanual only. |
| regrasp | - | returns `unsupported` |

## 5. Gates (measured; `scripts/robots_gate.py <robot> all`, `scripts/robots_g1.py`)

Measured 2026-11-23 on gb10-direct, single process per robot, seeds fixed. Raw JSON is in
`results/v2/robots/<robot>/*.json`; `scripts/robots_summary.py` rebuilds this table and `summary.json`.

What each column measures:
- **Strokes:** 50 free-space horizontal strokes per robot. Each is run-up → constant-speed cruise → min-jerk stop,
  using explicit via-point velocities. Cruise length is 0.2–0.3 m (SO-100: 0.08–0.12, Z1: 0.15–0.25). Speed is
  log-uniform in [0.1, budget v] (1.2 m/s for panda/UR5e, 1.0 for the others, 0.6 for SO-100). The line is sampled
  inside the bench workspace and checked for IK reachability.
- **Mid-stroke metrics:** the middle 50% of the cruise. "Release" is the last 3 ticks of the cruise.
- **Stall:** the stroke did not finish, or the mid-stroke speed was below 0.5x the command.
- **Pick+place:** box 2.4–6 cm (SO-100: 2.4–3.6 cm), mass 0.1–0.6 kg, friction 0.5–1.0, random yaw. The tracker
  estimate has 2 mm and 0.02 rad noise. Success means within 2 cm of the target and upright.
- **Push:** box, mass and friction random; speed log-U[0.1, 0.8], 12 cm push.
- **Sweep:** 30 parts of 1.6 cm (scaled down for the small arms), counted as moved when they cross the target line.
- **Insert:** a square peg in a square hole, clearance 1.0 / 0.5 mm, with 1 mm estimate noise on the peg and on the
  hole.
- **Integrity:** violations counted over strokes, pick/place, push and sweep (no applied forces, no welds on free
  objects).

| robot | mid-stroke err p50/p95/max (mm) | speed ratio p10/p50/p90 | release ratio p10/p90 | stalls /50 | time at torque limit (s) | peak torque-rate p50/p90 (1/s) | peak joint jerk p50/p90 (rad/s^3) | sim/wall | pick+place /20 | place err p50 (mm) | push ratio p10/p50/p90 | sweep 30 parts moved | insert 1.0 / 0.5 mm | integrity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| panda | 0.7/1.7/3.4 | 0.999/1.000/1.001 | 0.998/1.000 | 0 | 0.67 | 0.9/2.4 | 118/278 | 7.9x | 20 | 1.7 | 0.943/1.014/1.052 | 90% | 8/10, 6/10 | 0 |
| ur5e_2f85 | 0.6/1.3/1.7 | 1.000/1.000/1.000 | 0.999/1.000 | 0 | 0.11 | 0.6/1.0 | 87/133 | 7.0x | 20 | 1.6 | 0.773/1.034/1.076 | 90% | 10/10, 10/10 | 0 |
| gen3_2f85 | 2.4/6.4/11.4 | 0.994/1.003/1.014 | 0.954/1.007 | 0 | 1.81 | 1.0/3.0 | 122/233 | 13.3x | 20 | 3.5 | 0.521/1.034/1.073 | 90% | 7/10, 9/10 | 0 |
| z1 | 2.2/6.5/8.2 | 0.989/1.000/1.021 | 0.987/1.032 | 0 | 0.00 | 1.5/1.9 | 351/598 | 6.6x | 18 | 4.1 | 0.091/0.989/1.107 | 83% | - | 0 |
| so100 | 1.5/5.5/46.3 | 0.976/0.995/1.018 | 0.982/1.016 | 0 | 1.09 | 8.8/59.0 | 609/2188 | 9.2x | 3 | 78.9 | 0.980/1.007/1.030 | 20% | - | 0 |
| aloha | 3.2/34.0/65.2 | 0.977/0.997/1.027 | 0.961/1.031 | 0 | 3.02 | 1.7/45.7 | 163/86201 | 5.8x | 16 | 1.3 | 0.739/1.018/1.120 | 47% | - | 0 |
|   handover (aloha) | 8/10 successful | | | | | | | | | | | | | |
| tidybot | 3.1/11.3/42.5 | 0.975/0.987/1.002 | 0.923/1.004 | 0 | 0.00 | 0.4/4.1 | 120/1998 | 12.0x | 20 | 1.9 | 0.710/1.016/1.117 | 73% | - | 0 |
| ur5e_suction | 0.5/1.5/2.0 | 0.999/1.000/1.000 | 0.999/1.000 | 0 | 0.33 | 0.5/1.3 | 90/236 | 8.5x | 20 | 5.4 | - | - | - | 0 |
| fr3_suction | 0.9/1.5/3.0 | 0.997/0.999/1.002 | 0.997/1.001 | 0 | 1.05 | 1.4/5.4 | 271/1235 | 8.5x | 20 | 3.5 | - | - | - | 0 |

Round 2 (lead follow-up), measured 2026-11-23 ~23:00Z:
- **Grip force.** Calibrated forces and silicone pad friction are in, and the gripper close is event-based. Strokes
  are sampled IK-feasible along 7 points of the line; a stroke is shortened, then skipped, if no feasible line
  exists. Skipped: Gen3 8/50, Z1 9/50. Push now returns `unreachable` instead of executing out of reach.
- **Insertion (2 cm square peg, ±1 mm estimate noise).**

  | robot | 1.0 mm clearance | 0.5 mm clearance |
  |---|---|---|
  | UR5e + 2F-85 @100 N | 10/10 | 10/10 |
  | Gen3 + 2F-85 | 7/10 | 9/10 |
  | Panda @70 N | 8/10 | 6/10 |

  Panda failures are jams from the peg rotating in the two-finger pinch. → The **≥ 2 arms gate is met (UR5e, Gen3).**
- **Suction (vacuum cup, adhesion actuator, seal-gated).** Pick/place of 6–12 cm boxes:

  | robot | box mass | pick+place |
  |---|---|---|
  | ur5e_suction | 0.5–4.0 kg | 20/20 |
  | fr3_suction | 0.3–2.5 kg | 20/20 |

  Strokes: ratio 0.999–1.000, mid-stroke error max 2–3 mm.
- **Z1.** Strokes 0 stalls, ratio 0.989–1.021, mid-stroke error max 8.2 mm (was 330 mm: the stroke sampler used to
  command out-of-reach lines and wedge the arm into its joint limits). Pick+place 18/20; the 2 failures are reported
  `unreachable` before moving. Push speed p10 is still 0.09: some pushes stall on contact.
- **SO-100.** Strokes and push are fine (ratio 0.98–1.03). Pick 13/20 but place 3/20: the light part slips out of
  the single 3 Nm jaw at transit start (the jaw is levered open), and its weak release knocks parts away.
  Raising the squeeze to the full 3 Nm made picks worse (9/20). **Not fixed.**
- **ALOHA.** Pick+place 16/20 and handover 8/10, down from 20/20 and 10/10 in round 1 after the close/pre-shape
  changes (drops). Not yet diagnosed.
- **Articulated (item 6)** uses ENVS's machine-tending station (drawer_open / door_open) through `envs/robots.py`.
  That is ENVS's top-down tab pinch plus an impedance pull, with the max-travel open candidate (6 seeds each):

  | robot | drawer_open | door_open |
  |---|---|---|
  | UR5e | 6/6 | 2/6 |
  | Panda | 1/6 (stops at 77–82% of travel: impedance lag at 600 N/m against drawer friction) | 0/6 (tab grasp fails) |

  Results are in `results/v2/robots/articulated.json`. These need ENVS's adapter tuned (stiffer pull or track mode
  with force limits; door tab approach). The ROBOTS `articulate` skill (horizontal approach) collides with the bench
  for these low handles, per ENVS.

### G1 humanoid (`scripts/robots_g1.py`, results/v2/robots/g1/gate.json)

| | g1_dex3 (FLOATING base, balancing) | g1_dex3_fixed (fallback) |
|---|---|---|
| strokes completed | 50/50, 0 falls, 0 stalls | 50/50 |
| idle stand | 30 s; drift 0.04 mm over the last 10 s | - |
| mid-stroke err p50/p95/max | 2.8 / 5.4 / 6.6 mm | 3.7 / 26.5 / 46.7 mm |
| speed ratio p10/p50/p90 | 0.971 / 0.998 / 1.026 | 0.942 / 0.990 / 1.019 |
| CoM excursion | 54 mm during strokes | - |
| QP | 75,586 solves, 0 failures | - |
| time at torque limit | 0 s | - |
| peak torque-rate | 48.9 /s | - |
| sim/wall | 3.5x | 7.4x |

The floating base passes the G1 gate. The fixed fallback fails the 1 cm gate on strokes at the edge of reach: with the
pelvis welded it cannot lean. A fast impact (0.2 m/s into a rigid wall) made the QP infeasible and the robot fell. A
balance-only re-solve fallback was added afterwards, but that case was not re-tested.

### G1 round 2 (sub-agent; `scripts/robots_g1_tasks.py`, results/v2/robots/g1/{wall,pick,tote}.json)

**Wall impact (floating base).** No falls in any of the 8 cases (0.05/0.1/0.2/0.3 m/s × guarded/track):

| speed (m/s) | peak force guarded (N) | peak force track (N) |
|---|---|---|
| 0.05 | 24 | 28 |
| 0.1 | 49 | 49 |
| 0.2 | 96 | 110 |
| 0.3 | 133 | 154 |

The CoM moves at most 61 mm, and at most 43 of about 2,000 ticks per run needed the QP fallback. The fix has three
parts:
- **Hand contact-force cap** (`f_cap` 15 N by default; the hand yields along the contact normal above it).
- **Bounded external force** passed to the QP (`ext_total` 80 N).
- **Staged QP fallbacks:** first balance and posture only, then soft feet without range bounds, then the last solved
  torques.

ENVS: pass `f_cap=` to `set_mode` (or use force mode) where pressing harder is intended.

**Dex3 pinch pick: 10/10.** 4–6 cm boxes, 0.12–0.30 kg, lifted 11.7–13.3 cm and held 3 s. Dispatched from
`run_skill('pick')` for the humanoid; see `humanoid.g1_pinch_pick`.

**Two-palm tote carry: 8/10.** Tote 0.35 x 0.25 x 0.15 m, 1–3 kg. Palms squeeze at 12 N/kg (20–40 N) under force
control along the palm normals. Each tote is lifted, held 5 s and placed back; odd seeds are also carried 0.15 m.
There were no drops and no falls; the 2 failures lifted 7.5 and 7.9 cm, just under the 8 cm bar. CoM excursion is at
most 62 mm and peak torque 0.96 of the limit. Dispatched as `run_skill('pick', params two_palm=1)`.

Two test-scene choices: the tote has torsional friction 0.03 (standing in for the palm's contact patch), and the tote
sits at x = 0.30 m. One caveat: `g1_tote_carry` reads the tote's true pose to log max_lift. That value is not passed
to the actor (the skill dispatcher drops it).

Videos: `g1_pinch_pick.mp4`, `g1_tote_carry.mp4`. H1/Apollo are not done.

### Against v1's G1 controller (docs/G1.md)

| | v1 (position servos + IK, convergence waits) | v2 |
|---|---|---|
| achieved speed | 0.83–1.16x of command across 0.1–0.8 m/s (stiction at 0.05: 0.62–0.83) | panda/UR5e strokes 0.999–1.001 (p10–p90); worst fixed arm Z1 0.98–1.23 |
| release speed | ±5% target not met | panda 0.998–1.000 |
| stalls / wrist flips | convergence-wait stalls; 3 m/s wrist flip | no waits by construction; 0 stalls on panda/UR5e/ALOHA/SO-100/TidyBot; yaw chosen as the nearest equivalent (no flips) |
| max reliable speed | 0.8 m/s | ≥ 1.2 m/s on panda/UR5e (the budget limit, not a tracking limit) |

### Speed (data generation)
Single-process sim-time/wall-time is about 7–17x for the fixed arms with one object, about 5x for ALOHA (two arms),
about 6.5x with 30 free parts (sweep), and 3.5x for the floating G1 (one QP per 2 ms tick). Episodes of about 10 s
therefore cost 0.6–2 s of wall time per CPU process.

### Failures and limits (round-1 list; where it differs, Round 2 above supersedes it: insertion gate now met, Z1 strokes fixed, 2F-85 force fixed)
- **SO-100 pick: 0/20.** The 5-dof arm reaches the grasp poses only with tilted approaches. Its single jaw (3 Nm)
  stops part-closed without pad contact, or drops the part. Strokes, push and sweep work. Not fixed.
- **Z1 pick+place: 11/20.** Single-jaw grasp drops and two unreachable cases. Strokes p90 error is large (0.28 m)
  because some sampled lines leave its short reach even after the IK check (the check covers only the start, middle
  and end points). Speed ratio p50 is 1.00.
- **Gen3: 2/50 stalls.** Peak error 14 cm on two strokes near a joint limit. p50 tracking is 3.3 mm.
- **Push speed p10 0.62–0.92 on the 2F-85 robots and ALOHA.** The impact transient when the box is hit at speed sits
  inside the short mid-window. Free-space tracking is ≥ 0.99.
- **Insertion.** Panda: 6/10 at 1 mm, 5/10 at 0.5 mm. The failures are jams from the peg rotating inside the
  two-finger pinch. 2F-85 robots (UR5e, Gen3): the 2 cm peg could not be grasped (the linkage closes in an arc and
  tips the slim peg); a retry with a 3 cm peg also failed at the pick stage (0/10, `grasp_failed`: the pads stop part-closed without contact; not diagnosed). The "≥ 2 arms at 1 mm and 0.5 mm" target is therefore NOT
  met yet.
- **Sweep.** Panda 93%, Gen3 87%, UR5e 83%, Z1 77%, TidyBot 63%, ALOHA 40%, SO-100 17%. A ~3 cm finger paddle in
  lanes; small arms have small paddles.
- **open/close (drawer/door)** is implemented but not gated: there is no articulated bench here, so it waits for
  ENVS scenes.
- **Robot models.** The wrist F/T sensor is ideal (contact-resolved, no noise). Add noise in ENVS. The 2F-85 grip
  force is 5 (menagerie tendon limit), which is weak. H1/Apollo are not done.

### Videos
`results/v2/robots/<robot>/<robot>_smooth.mp4` for panda, ur5e_2f85, gen3_2f85, z1, so100, aloha and tidybot
(pick → place → push, 640x480, 30 fps). Also `results/v2/robots/g1/g1_dex3_strokes.mp4` and
`g1_dex3_fixed_strokes.mp4`.



## 6. Env matrix: scripted policy x real robots in the ENVS stations

`scripts/robots_env_matrix.py` (multi-process, 300 s wall guard per episode) runs `env.scripted()` until done.
Each cell is successes/5 seeds (mean final progress). Measured 2026-11-23 evening with the co-owned
`envs/robots.py` adapter.

Files:
- Raw: `results/v2/robots/env_runs_final.jsonl`.
- Matrix: `results/v2/robots/env_matrix.json` (`--summarize`).
- Launcher: `scripts/robots_env_final.sh`.

Run details:
- Suction arms ran on 5 relevant families, Z1 and SO-100 on 7.
- Families with no row for a robot were not run.
- 0 integrity violations and 0 errors in 520 episodes.
- sim/wall per process: 2.0–4.4x (Panda 3.4, UR5e 3.8, Gen3 4.3, ALOHA 3.5, TidyBot 2.8).

| family | panda | ur5e_2f85 | gen3_2f85 | aloha | tidybot | ur5e_suction | fr3_suction | z1 | so100 |
|---|---|---|---|---|---|---|---|---|---|
| bin_clearing | 0/5 (0.09) | 0/5 (0.57) | 0/5 (0.47) | 0/5 (0.00) | 0/5 (0.00) | - | - | 0/5 (0.00) | 0/5 (0.00) |
| bulk_sweep | 2/5 (0.83) | 1/5 (0.82) | 0/5 (0.63) | 0/5 (0.00) | 0/5 (0.01) | 0/5 (0.00) | 0/5 (0.00) | - | - |
| connector_insert | 1/5 (0.66) | 5/5 (1.00) | 4/5 (0.80) | 0/5 (0.00) | 0/5 (0.00) | - | - | - | - |
| conveyor_sort | 0/5 (0.00) | 0/5 (0.24) | 0/5 (0.16) | 0/5 (0.00) | 0/5 (0.00) | - | - | - | - |
| depalletize | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.05) | 0/5 (0.00) | - | - |
| door_open | 4/5 (0.85) | 5/5 (1.00) | 5/5 (1.00) | 0/5 (0.00) | 5/5 (1.00) | - | - | 0/5 (0.00) | 0/5 (0.00) |
| drawer_open | 3/5 (0.84) | 5/5 (1.00) | 3/5 (0.90) | 4/5 (0.80) | 5/5 (1.00) | - | - | 4/5 (0.80) | 0/5 (0.00) |
| fixture_place | 5/5 (1.00) | 5/5 (1.00) | 5/5 (1.00) | 0/5 (0.00) | 0/5 (0.00) | 1/5 (0.20) | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.00) |
| kitting_pocket | 3/5 (0.67) | 3/5 (0.83) | 1/5 (0.50) | 0/5 (0.00) | 0/5 (0.00) | - | - | - | - |
| machine_tending | 0/5 (0.47) | 4/5 (0.93) | 1/5 (0.40) | 0/5 (0.00) | 0/5 (0.33) | - | - | - | - |
| peg_insert | 5/5 (1.00) | 5/5 (1.00) | 5/5 (1.00) | 0/5 (0.00) | 0/5 (0.00) | - | - | 0/5 (0.00) | 0/5 (0.00) |
| pouch_transfer | 4/5 (0.80) | 3/5 (0.60) | 2/5 (0.40) | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.00) |
| singulation | 4/5 (0.92) | 2/5 (1.00) | 3/5 (0.80) | 3/5 (0.72) | 3/5 (0.80) | 3/5 (0.75) | 2/5 (0.92) | 3/5 (0.73) | 0/5 (0.07) |
| tool_return | 0/5 (0.00) | 4/5 (0.80) | 1/5 (0.20) | 2/5 (0.40) | 0/5 (0.00) | - | - | - | - |
| tote_transfer | 1/5 (0.53) | 2/5 (0.53) | 0/5 (0.33) | 0/5 (0.00) | 0/5 (0.00) | - | - | - | - |
| twist_lock | 4/5 (0.88) | 5/5 (1.00) | 5/5 (1.00) | 2/5 (0.50) | 1/5 (0.37) | - | - | - | - |
| **total** | 36/80 | 49/80 | 35/80 | 11/80 | 14/80 | 4/25 | 2/25 | 7/35 | 0/35 |

For reference, the probe-robot scripted baseline (docs/ENVS.md, train split) is roughly:

| family | probe success |
|---|---|
| bin_clearing | 0.12 |
| bulk_sweep | 0.50 |
| conveyor_sort | 0.38 |
| depalletize | 0.25 |
| kitting_pocket | 0.50 |
| peg_insert | 0.88 |
| singulation | 0.88 |
| door_open, drawer_open, fixture_place, pouch_transfer, tool_return | 1.0 |

**What was fixed to get here.** Before these fixes, Panda / UR5e / suction scored 6/28, 4/28 and 3/28 on 2 seeds.
- **Suction IK:** reachability was a single IK solve from home, which hit local minima. It is now a continuation
  plus multi-start solve, so pedestal arms and suction no longer report `unreachable`.
- **Pouch integrator:** the flex scene keeps its integrator (pouch_transfer had crashed).
- **Suction level:** the scripted `suction` level × a 300 N cup, with seal hysteresis.
- **Tote wall pinch:** totes are pinched by the near wall (tote_transfer went from too_wide to 2/2 at first).
- **Conveyor lead:** the conveyor lead uses the predicted time to grasp.
- **Insertion search:** search radius at least 4 mm. Peg_insert: UR5e 0/3 → 5/5, Panda 5/5.
- **Insert yaw alignment:** insertion aligns yaw to the socket, which solves connector_insert (UR5e 5/5).
- **Explicit grasp depth:** an explicit grasp depth is honoured (tool_return shank, UR5e 4/5).
- **Sweep band:** the probe blade's band is swept in finger lanes.
- **Drawer pull:** the drawer pull runs in track mode (Panda drawer 56–68% → 3/5 at 0.84).
- **Clutter yaw:** in clutter, the closing yaw is chosen against neighbouring estimates (bin_clearing UR5e progress
  .46 → .57; one seed reached .875).
- **Gripper close:** the gripper closes at full drive until pad contact and only then regulates to the requested
  squeeze. Low grip forces (20 N) used to make the 2F-85 take about 5 s to close, so UR5e/Gen3 door_open went
  0/5 → 5/5. The 2F-85 also grasps the door tab toward its tip.
- **ALOHA pre-shape:** ALOHA pick/place is back to 20/20 and handover 10/10 (the pre-shape opening margin is now
  30 mm).
- **TidyBot home:** home TCP 0.78 m (was 0.54, below the bench).

**Remaining failures (diagnosed, not fixed):**
- **depalletize:** 0 for every robot. Success needs all 8 cartons. Hidden heavy cartons (12–22 kg) exceed the
  scripted 50% suction level and drop off the pallet edge, which is fatal; the probe also only reaches 0.25.
  Parallel grippers cannot hold 0.29 m cartons.
- **conveyor_sort:** picks of moving parts. The descent does not track the moving item (no visual servoing in
  `pick`), so grasps land behind it.
- **bin_clearing:** 0 successes but progress up to 0.57–0.875. The call budget runs out after re-grasps on parts
  wedged against walls; the probe gets 0.12.
- **TidyBot:** the arm base is at 0.335 m and its base touches the 0.76 m bench, so targets more than about 0.45 m
  deep are unreachable (fixture/peg/kitting 0). Drawer and door 5/5.
- **ALOHA in envs:** 11/80. Its bench-centre layout puts most targets at the edge of the vx300s reach.
- **Z1 and SO-100:** weak grippers and short reach (Z1 drawer 4/5; SO-100 0/35).
- **tool_return with Panda:** 0/5. The small pads let the head-imbalanced tool rotate (the family's hidden
  variable); the TCP-depth check reports `done`.
- **Panda placement:** in `tests/test_skills.py` the box lands 12 mm from target (< 15 mm); round 1 was about 2 mm.
  It appeared after the grip-force/pad-friction changes; not investigated.
- **G1 (sub-agent, `scripts/robots_g1_envsweep.py`, `results/v2/robots/env_matrix_g1.json`).**
  - **Speed-up:** the whole-body QP runs at 250 Hz with torques held and bias-corrected between solves; contact
    lookups use boolean masks; ProxQP max_iter is 400, warm-started.
    - Stroke bench: 3.5x → 5.5x real time.
    - Env episodes: minutes → 1.4–3.1x (floating), 5.7x (fixed).
  - **Gates after the speed-up:** strokes still pass (error p95 5.4 mm, ratio 0.971–1.028, 0 falls); no wall-impact
    falls; pinch pick 10/10; tote carry 7/10 (misses lifted 7.8–7.98 cm); peak torque-rate 113 /s (was 49).
  - **Env (5 seeds × 7 families, before the mount change):** floating 0/35 (fixture_place progress 0.49); fixed base
    drawer_open 3/5 and the rest 0.
  - **Main causes:** the humanoid mount was about 4 cm out of reach, and Dex3 pinch grasps fail on 16 mm pegs and
    heavy fixtures. The adapter now steps the humanoid 0.12 m closer.
  - **Rerun with the closer mount** (5 seeds; `env_runs_g1_mount.jsonl`, `env_matrix_g1_mount.json`):

    | family | g1_dex3 (floating) | g1_dex3_fixed |
    |---|---|---|
    | door_open | 0/5 | 3/5 |
    | drawer_open | 0/5 | 4/5 |
    | fixture_place | 0/5 | 0/5 (0.54) |
    | peg_insert | 0/5 | 0/5 |
    | singulation | 0/5 | 0/5 (0.20) |

    Floating-base failure modes: push `unreachable` 84, pick `grasp_failed` 66, open `grasp_failed` 45, pick `fell`
    14. The floating G1 is not usable for data generation yet; the fixed-base fallback is, for articulated tasks.
  - **API fixes:** `set_grip_force` / `max_grip_force` added (the Dex3 is nominally 10 N); `g1_pinch_pick` fills
    `ctx.held`; the hand IK chain includes waist pitch (effector dofs/home are 8 long, waist first; arm-only versions
    as `arm_*`).
