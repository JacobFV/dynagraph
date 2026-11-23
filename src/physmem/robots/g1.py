"""Unitree G1 (with Dex3 hands) fixed-base tabletop endpoint (agent G1).

Layers (privilege boundary is explicit):

* ``G1Sim``     -- PRIVILEGED. Owns the MuJoCo MjModel/MjData of the full scene
                   (robot + table + dynamic box + fixtures). Only advances physics
                   with ``mj_step``. Emits *synthetic sensor* ObservationEvents
                   (noisy object pose, tactile contact indicator, object size).
                   Instruments the object: object qpos may only change inside
                   mj_step; no equality constraints, no applied forces.
* ``ActorPort`` -- the ONLY handle the actor gets: emulated encoder readings,
                   joint position targets, advance-time, drain sensor events.
                   It exposes no MjData / MjModel / true friction / mass.
* ``G1Robot``   -- ACTOR side. Robot-only kinematic model (no scene objects),
                   damped-least-squares IK on encoder joint state, joint servos
                   (position targets), hand open/close, push / grasp / reach /
                   lift / place primitives, proprioceptive POSE2 (forward
                   kinematics of the encoders) and own COMMAND events.

Model source: MuJoCo Menagerie ``unitree_g1/g1_with_hands.xml`` pinned at commit
prefix f3b6013 (copied to robots/assets/menagerie_g1, unmodified). All scene
modifications are applied programmatically in ``build_scene_spec`` (listed in
robots/assets/README.md and docs/G1.md).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, asdict, replace
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

from ..schema import (
    ActionCandidate,
    CapabilityDescriptor,
    ObservationEvent,
    Prop,
    Provenance,
)

ASSET_DIR = Path(__file__).resolve().parent / "assets"
MENAGERIE_DIR = ASSET_DIR / "menagerie_g1"
MENAGERIE_XML = MENAGERIE_DIR / "g1_with_hands.xml"
MENAGERIE_COMMIT = "f3b6013"

G = 9.81
PHYSICS_DT = 0.002
CONTROL_DT = 0.01  # 100 Hz IK / servo target update
SENSOR_DT = 0.05  # 20 Hz observation streams
PELVIS_POS = (0.0, 0.0, 0.793)

# table: top surface at TABLE_TOP, centred at TABLE_XY
TABLE_TOP = 0.74
TABLE_XY = (0.45, 0.0)
TABLE_HALF = (0.25, 0.42)

# hand-object friction, mirrors worlds.PUSHER_FRICTION (geom priority mixing like w2)
HAND_FRICTION = 0.3
HAND_TORSIONAL = 0.01  # m, torsional friction coefficient of hand geoms (condim 4)
HAND_TILT = 0.7  # horizontal/vertical ratio of the hand axis (legacy hand_axis helper)
PUSHER_SITE_X = 0.19  # fingertip region of the flat (open) hand
PUSH_CLOSURE = 0.0
PUSH_Z = 0.035
PUSH_LEAD = 0.3
PUSH_LEADS = (0.3, 0.0, -0.3)  # hand lean options tried in order (IK feasibility)
IK_FEASIBLE_TOL = 0.008
BIMANUAL_W_AXIS = 0.15
BIMANUAL_WAIST_TRACK = True
GRIP_BELOW_CENTER = 0.008
RELEASE_BACKOFF = 0.03  # m, hand withdraws along -push direction right after the push
RELEASE_SPEED = 0.6
VELOCITY_FEEDFORWARD = False  # ramp-lag compensation tried: overshoot worse (fast-push ratio 1.35, sd 4.0 cm); off
PUSH_SERVO = "v1"  # "v1": open-loop Cartesian line (default); "v2": closed-loop speed servo (bounded attempt)
SERVO_GAINS = dict(runup=0.04, ki=0.0, alpha=0.5, i_clip=0.05, stop_lead=0.0, timeout=0.3, max_lead=None)
TRANSIT_Z = 0.20  # m above the table: hands travel between pushes at this height
PUSH_GAP = 0.025  # pre-contact clearance outside the box boundary (m); w2 uses 3 mm (G1 tracking error ~1-2 cm)
CONTACT_FORCE_THRESHOLD = 0.5  # N, synthetic tactile indicator threshold

# entity / source ids at record level
OBJ = "obj_g1box"  # opaque-id prefix convention of data._ENT_PREFIX (obj_/pus_)
PUSHER = "pus_g1hand"
PUSHER_SIZE = (0.015, 0.015, 0.015)  # effective hand-tip half thickness reported to the actor (cf. w2 sphere15)
SRC_VEL = "synthetic_pose_sensor_vel"
CONTACT_NOISE, CONTACT_FLIP = 0.3, 0.02  # as data.py synthetic tactile
SRC_POSE = "synthetic_pose_sensor"
SRC_TACTILE = "synthetic_tactile"
SRC_SIZE = "synthetic_size_sensor"
SRC_PROPRIO = "g1_proprio_fk"
SRC_CMD = "g1_command"

ARM_JOINTS = {
    s: [f"{s}_shoulder_pitch_joint", f"{s}_shoulder_roll_joint", f"{s}_shoulder_yaw_joint", f"{s}_elbow_joint",
        f"{s}_wrist_roll_joint", f"{s}_wrist_pitch_joint", f"{s}_wrist_yaw_joint"]
    for s in ("left", "right")
}
WAIST_JOINTS = ["waist_yaw_joint", "waist_pitch_joint"]
HAND_JOINTS = {
    s: [f"{s}_hand_thumb_0_joint", f"{s}_hand_thumb_1_joint", f"{s}_hand_thumb_2_joint",
        f"{s}_hand_middle_0_joint", f"{s}_hand_middle_1_joint", f"{s}_hand_index_0_joint", f"{s}_hand_index_1_joint"]
    for s in ("left", "right")
}
# hand postures (fraction of joint range toward "closed"); filled from model ranges
HAND_LINKS = ("hand_palm_link", "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link",
              "hand_middle_0_link", "hand_middle_1_link", "hand_index_0_link", "hand_index_1_link", "wrist_yaw_link")

# position-actuator tuning (official model: kp=500, dampratio=1 everywhere; tuning left to user)
ACTUATOR_TUNING = {
    # group: (kp, dampratio)
    "leg": (500.0, 1.0),
    "waist": (800.0, 1.0),
    "shoulder": (600.0, 1.0),
    "elbow": (500.0, 1.0),
    "wrist": (150.0, 1.0),
    "hand": (15.0, 1.0),
}
HAND_JOINT_FRICTIONLOSS = 0.02  # official 0.3 Nm stalls the low-gain finger servos
HAND_ARMATURE = 0.002

G1_CAPABILITY = CapabilityDescriptor(
    name="unitree_g1_fixed_base_dex3",
    max_push_speed=0.80,  # measured: hand mid-push speed within ~0.8-1.15x of command up to 0.8 m/s (scripts/demo_g1_speed_tracking.py)
    max_push_duration=0.5,
    workspace_min=(0.24, -0.30),
    workspace_max=(0.53, 0.30),
    effector_radius=0.015,
    action_kinds=("push",),
)


def _actuator_group(name: str) -> str:
    if "hip" in name or "knee" in name or "ankle" in name:
        return "leg"
    if name.startswith("waist"):
        return "waist"
    if "shoulder" in name:
        return "shoulder"
    if "elbow" in name:
        return "elbow"
    if "wrist" in name:
        return "wrist"
    return "hand"


# --------------------------------------------------------------------------
# scene construction
# --------------------------------------------------------------------------


@dataclass
class BoxSpec:
    half: tuple = (0.04, 0.04, 0.04)
    mass: float = 0.5
    friction: float = 0.5  # hidden: box<->support sliding coefficient
    pos: tuple = (0.45, -0.10)
    yaw: float = 0.0
    z: Optional[float] = None  # None -> resting on table


@dataclass
class EpisodeSpec:
    variant: str = "push"  # "push" | "rail"
    box: BoxSpec = field(default_factory=BoxSpec)
    target: tuple = (0.45, 0.0)  # target region centre (xy, on the table / in the tray)
    target_radius: float = 0.03
    seed: int = 0
    # rail variant
    rail_xy: tuple = (0.40, -0.10)
    rail_gap: float = 0.05  # inner distance between rails (rails run along y; box spans them in x)
    rail_height: float = 0.04
    tray_xy: tuple = (0.40, 0.12)
    tray_half: tuple = (0.065, 0.075)

    def to_json(self):
        return asdict(self)


def _load_base_spec(fixed_base: bool = True):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(MENAGERIE_XML))
    if fixed_base:
        # MODIFICATION 1: pelvis welded to the world by removing the floating base joint
        spec.delete(spec.joint("floating_base_joint"))
        for k in list(spec.keys):
            spec.delete(k)  # 'stand' keyframe has the 7 freejoint qpos entries
    spec.option.timestep = PHYSICS_DT
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # MODIFICATION 2: actuator tuning per group
    for a in spec.actuators:
        kp, dr = ACTUATOR_TUNING[_actuator_group(a.name)]
        a.set_to_position(kp=kp, dampratio=dr)
        a.inheritrange = 1.0
    # MODIFICATION 3: hand joints lighter friction-loss / armature for low-gain finger servos
    for j in spec.joints:
        if "_hand_" in j.name:
            j.frictionloss = HAND_JOINT_FRICTIONLOSS
            j.armature = HAND_ARMATURE
    # MODIFICATION 4: gravity compensation on robot bodies (feed-forward gravity term)
    for b in spec.bodies:
        if b.name and b.name != "world":
            b.gravcomp = 1.0
    # MODIFICATION 5: collision filtering -- robot contype=1/conaffinity=2 => no robot
    # self-collision and no robot-floor contact; robot still collides with table/box/fixtures.
    for g in spec.geoms:
        if g.contype != 0 or g.conaffinity != 0:
            g.contype, g.conaffinity = 1, 2
            if g.parent is not None and any(g.parent.name.endswith(h) for h in HAND_LINKS):
                g.priority = 2
                g.friction = [HAND_FRICTION, HAND_TORSIONAL, 0.0001]
                g.condim = 4  # soft-finger torsional friction (patch contact) for grasp stability
    # MODIFICATION 6: pusher reference sites on both hands (fist knuckle region)
    for s in ("left", "right"):
        b = spec.body(f"{s}_wrist_yaw_link")
        b.add_site(name=f"{s}_pusher", pos=[PUSHER_SITE_X, 0.0, 0.0], size=[0.01, 0, 0], rgba=[1, 0.4, 0, 1], group=4)
        b.add_site(name=f"{s}_palm", pos=[0.09, 0.0, 0.0], size=[0.01, 0, 0], rgba=[0, 0.4, 1, 1], group=4)
    return spec


def _add_box_body(spec, name, half, pos, yaw, mass=None, rgba=(0.8, 0.8, 0.8, 1), contype=3, conaffinity=3,
                  friction=0.5, priority=0):
    import mujoco

    q = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    b = spec.worldbody.add_body(name=name, pos=list(pos), quat=q)
    g = b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), rgba=list(rgba))
    g.contype, g.conaffinity = contype, conaffinity
    g.friction = [friction, 0.005, 0.0001]
    g.priority = priority
    g.condim = 3
    if mass is not None:
        g.mass = mass
    return b, g


def build_scene_spec(ep: EpisodeSpec):
    """Full scene (privileged). Collision bits: robot 1/2, table+fixtures 3/3,
    object 7/7 (with priority 1 so its friction decides box<->support pairs),
    floor 4/4, markers 0/0 (visual only)."""
    import mujoco

    spec = _load_base_spec(True)
    wb = spec.worldbody
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.25, 0.27, 0.3], rgb2=[0.2, 0.22, 0.25], width=256, height=256)
    spec.add_material(name="floor", textures=["", "grid"], texrepeat=[6, 6])
    fl = wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], material="floor")
    fl.contype, fl.conaffinity = 4, 4
    wb.add_light(pos=[0.6, -1.0, 2.5], dir=[-0.2, 0.4, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.5, 0.5, 0.5])
    # fixed-base mount (visual only): makes the welded pelvis explicit in renders
    mnt = wb.add_geom(name="fixed_base_mount", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, PELVIS_POS[2] / 2 - 0.06, 0],
                      pos=[-0.12, 0, PELVIS_POS[2] / 2 - 0.06], rgba=[0.95, 0.55, 0.1, 1])
    mnt.contype = mnt.conaffinity = 0
    arm = wb.add_geom(name="fixed_base_clamp", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.025, 0.02],
                      pos=[-0.07, 0, PELVIS_POS[2] - 0.08], rgba=[0.95, 0.55, 0.1, 1])
    arm.contype = arm.conaffinity = 0
    # table (static, collides with robot + object)
    tz = TABLE_TOP - 0.02
    _add_box_body(spec, "table", (TABLE_HALF[0], TABLE_HALF[1], 0.02), (TABLE_XY[0], TABLE_XY[1], tz), 0.0,
                  rgba=(0.55, 0.42, 0.3, 1), contype=3, conaffinity=3, friction=0.5, priority=0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            lg = wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, tz / 2],
                             pos=[TABLE_XY[0] + sx * (TABLE_HALF[0] - 0.03), TABLE_XY[1] + sy * (TABLE_HALF[1] - 0.03), tz / 2],
                             rgba=[0.4, 0.3, 0.2, 1])
            lg.contype = lg.conaffinity = 0
    support_top = TABLE_TOP
    if ep.variant == "rail":
        rx, ry = ep.rail_xy
        rh = ep.rail_height
        for sgn in (-1, 1):
            # rails run along x; the box spans across them (gap below the box between rails)
            _add_box_body(spec, f"rail_{'a' if sgn < 0 else 'b'}", (0.008, 0.06, rh / 2),
                          (rx + sgn * (ep.rail_gap / 2 + 0.008), ry, TABLE_TOP + rh / 2), 0.0,
                          rgba=(0.3, 0.35, 0.45, 1), contype=3, conaffinity=3, friction=0.5)
        support_top = TABLE_TOP + rh
        # tray: thin floor + low walls (static)
        tx, ty = ep.tray_xy
        thx, thy = ep.tray_half
        _add_box_body(spec, "tray_floor", (thx, thy, 0.004), (tx, ty, TABLE_TOP + 0.004), 0.0,
                      rgba=(0.2, 0.5, 0.35, 1), contype=3, conaffinity=3, friction=0.6)
        wall_h = 0.012
        for name, half, pos in (
            ("tray_wx0", (0.004, thy, wall_h), (tx - thx, ty, TABLE_TOP + wall_h)),
            ("tray_wx1", (0.004, thy, wall_h), (tx + thx, ty, TABLE_TOP + wall_h)),
            ("tray_wy0", (thx, 0.004, wall_h), (tx, ty - thy, TABLE_TOP + wall_h)),
            ("tray_wy1", (thx, 0.004, wall_h), (tx, ty + thy, TABLE_TOP + wall_h)),
        ):
            _add_box_body(spec, name, half, pos, 0.0, rgba=(0.2, 0.5, 0.35, 1), contype=3, conaffinity=3, friction=0.6)
    # dynamic box: freejoint, gravity, collision geometry, hidden friction + mass
    bx = ep.box
    z = bx.z if bx.z is not None else support_top + bx.half[2] + 1e-4
    b, g = _add_box_body(spec, "box", bx.half, (bx.pos[0], bx.pos[1], z), bx.yaw, mass=bx.mass,
                         rgba=(0.85, 0.75, 0.2, 1), contype=7, conaffinity=7, friction=bx.friction, priority=1)
    b.add_freejoint(name="box_free")
    # target region marker: mocap body, visual only (no collision, massless)
    tb = wb.add_body(name="target_marker", mocap=True, pos=[ep.target[0], ep.target[1], support_top + 0.0006])
    tg = tb.add_geom(name="target_marker", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[ep.target_radius, 0.0005, 0],
                     rgba=[0.1, 0.9, 0.3, 0.45])
    tg.contype = tg.conaffinity = 0
    tg.density = 0
    tg.group = 1
    # cameras
    wb.add_camera(name="overview", pos=[1.45, -0.95, 1.45], xyaxes=[0.55, 0.835, 0, -0.42, 0.28, 0.86])
    wb.add_camera(name="top", pos=[0.48, 0.0, 1.75], xyaxes=[0, -1, 0, 1, 0, 0])
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    return spec


def build_robot_kinematic_model():
    """Robot-only model (no table/box/fixtures). Used by the actor for IK / FK."""
    spec = _load_base_spec(True)
    return spec.compile()


# --------------------------------------------------------------------------
# privileged simulator
# --------------------------------------------------------------------------


class PhysicsIntegrityError(RuntimeError):
    pass


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class G1Sim:
    """Privileged simulator. Never handed to the actor or predictor."""

    def __init__(self, ep: EpisodeSpec, sensor_noise=(0.002, math.radians(1.0)), latency=0.0, seed=0):
        import mujoco

        self._mj = mujoco
        self.ep = ep
        self.spec = build_scene_spec(ep)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.box_bid = m.body("box").id
        self.box_gid = m.geom("box").id
        j = m.joint("box_free")
        self.box_qadr, self.box_vadr = m.jnt_qposadr[j.id], m.jnt_dofadr[j.id]
        self.rng = np.random.default_rng(seed)
        self.sensor_sigma = sensor_noise
        self.latency = latency
        self.act_index = {m.actuator(i).name: i for i in range(m.nu)}
        self.joint_qadr = {m.joint(i).name: m.jnt_qposadr[i] for i in range(m.njnt)}
        self.hand_geoms = {s: set() for s in ("left", "right")}
        for gi in range(m.ngeom):
            bn = m.body(m.geom_bodyid[gi]).name
            for s in ("left", "right"):
                if bn.startswith(s) and any(bn.endswith(h) for h in HAND_LINKS) and m.geom_contype[gi]:
                    self.hand_geoms[s].add(gi)
        self.robot_geoms = {gi for gi in range(m.ngeom) if m.geom_contype[gi] == 1 and m.geom_conaffinity[gi] == 2}
        self.support_geoms = {m.geom(n).id for n in ("table",) + (("rail_a", "rail_b") if ep.variant == "rail" else ())}
        self.dest_geoms = {m.geom(n).id for n in (("tray_floor",) if ep.variant == "rail" else ("table",))}
        self.audit = dict(external_object_writes=0, eq_active=0, applied_force_steps=0, mj_steps=0,
                          max_obj_speed=0.0, nan_steps=0)
        self._evcount = 0
        self._events: list[ObservationEvent] = []
        self._next_sensor_t = 0.0
        self._last_meas = None
        self._tick_count = 0
        self._site_l = self.model.site("left_pusher").id
        self._site_r = self.model.site("right_pusher").id
        self._last_forces = dict(hand=0.0, robot=0.0, support=0.0, dest=0.0, other=0.0)
        self._trace = []  # privileged ground-truth trace (evaluator / video only)
        self.episode_id = f"g1-{ep.variant}-s{ep.seed}"
        self._reset_robot()
        self._last_obj_q = self.data.qpos[self.box_qadr:self.box_qadr + 7].copy()
        self._obj_hash0 = self.object_hash()
        if m.neq != 0:
            raise PhysicsIntegrityError("scene must contain no equality constraints")

    # --- reset (the ONLY place object state is written) ---------------------
    def _reset_robot(self):
        m, d = self.model, self.data
        self._mj.mj_resetData(m, d)
        home = home_posture()
        for jn, v in home.items():
            d.qpos[self.joint_qadr[jn]] = v
            if jn in self.act_index:
                d.ctrl[self.act_index[jn]] = v
        self._mj.mj_forward(m, d)
        # settle with servos holding home (object resting); 0.3 s, part of reset
        for _ in range(int(0.3 / PHYSICS_DT)):
            self._mj.mj_step(m, d)
        d.time = 0.0
        self._emit_size()

    # --- integrity instrumentation ------------------------------------------
    def object_hash(self) -> str:
        q = self.data.qpos[self.box_qadr:self.box_qadr + 7]
        return hashlib.sha1(np.round(q, 12).tobytes()).hexdigest()[:16]

    def _check_integrity(self):
        d = self.data
        q = d.qpos[self.box_qadr:self.box_qadr + 7]
        if not np.array_equal(q, self._last_obj_q):
            self.audit["external_object_writes"] += 1
        if self.model.neq and np.any(d.eq_active):
            self.audit["eq_active"] += 1
        if np.any(d.xfrc_applied[self.box_bid] != 0) or np.any(d.qfrc_applied[self.box_vadr:self.box_vadr + 6] != 0):
            self.audit["applied_force_steps"] += 1

    def step(self, n: int = 1):
        m, d = self.model, self.data
        for _ in range(n):
            self._check_integrity()
            self._mj.mj_step(m, d)
            self.audit["mj_steps"] += 1
            self._last_obj_q = d.qpos[self.box_qadr:self.box_qadr + 7].copy()
            if not np.all(np.isfinite(d.qpos)):
                self.audit["nan_steps"] += 1
                raise PhysicsIntegrityError("non-finite state")
            sp = float(np.linalg.norm(d.qvel[self.box_vadr:self.box_vadr + 3]))
            self.audit["max_obj_speed"] = max(self.audit["max_obj_speed"], sp)
            if d.time + 1e-9 >= self._next_sensor_t:
                self._emit_sensors()
                self._next_sensor_t += SENSOR_DT

    @property
    def time(self) -> float:
        return float(self.data.time)

    # --- servo interface (used through ActorPort only) ----------------------
    def set_targets(self, targets: dict):
        for jn, v in targets.items():
            i = self.act_index[jn]
            lo, hi = self.model.actuator_ctrlrange[i]
            self.data.ctrl[i] = float(np.clip(v, lo, hi))

    def read_encoders(self, names):
        # emulated encoders: 16-bit/turn quantisation + small noise
        q = np.array([self.data.qpos[self.joint_qadr[n]] for n in names])
        res = 2 * math.pi / 65536
        q = np.round(q / res) * res + self.rng.normal(0, 5e-5, size=q.shape)
        return q

    # --- privileged truth (evaluator / video only) --------------------------
    def true_object_state(self):
        d = self.data
        q = d.qpos[self.box_qadr:self.box_qadr + 7]
        v = d.qvel[self.box_vadr:self.box_vadr + 6]
        w_world = d.xmat[self.box_bid].reshape(3, 3) @ v[3:6]
        return dict(pos=q[:3].copy(), yaw=_yaw(q[3:7]), quat=q[3:7].copy(), lin_vel=v[:3].copy(), ang_vel=w_world.copy())

    def contact_forces(self):
        """Normal force between object and (hand_left, hand_right, other robot, supports, destination, other)."""
        m, d = self.model, self.data
        f = dict(hand=0.0, robot=0.0, support=0.0, dest=0.0, other=0.0)
        buf = np.zeros(6)
        for ci in range(d.ncon):
            c = d.contact[ci]
            if self.box_gid not in (c.geom1, c.geom2):
                continue
            o = c.geom2 if c.geom1 == self.box_gid else c.geom1
            self._mj.mj_contactForce(m, d, ci, buf)
            fn = abs(buf[0])
            if o in self.hand_geoms["left"] or o in self.hand_geoms["right"]:
                f["hand"] += fn
            elif o in self.robot_geoms:
                f["robot"] += fn
            if o in self.support_geoms and o not in self.dest_geoms:
                f["support"] += fn
            if o in self.dest_geoms:
                f["dest"] += fn
            if not (o in self.robot_geoms or o in self.support_geoms or o in self.dest_geoms):
                f["other"] += fn
        return f

    def object_aabb_zmin(self):
        d = self.data
        R = d.xmat[self.box_bid].reshape(3, 3)
        h = np.array(self.ep.box.half)
        ext = np.abs(R) @ h
        return float(d.xpos[self.box_bid][2] - ext[2])

    # --- synthetic sensors ---------------------------------------------------
    def _ev(self, entity, prop, value, var, source, prov=Provenance.SENSOR, related=None, frame="world"):
        self._evcount += 1
        t = self.time
        return ObservationEvent(
            event_id=f"{self.episode_id}-{source}-{self._evcount}", episode_id=self.episode_id, entity_id=entity,
            property_type=prop, value=tuple(float(x) for x in value), variance=tuple(float(x) for x in var),
            frame_id=frame, measured_at=t, received_at=t + self.latency, source_id=source, provenance=prov,
            related_entity_id=related)

    def _emit_size(self):
        h = np.array(self.ep.box.half) + self.rng.normal(0, 0.002, 3)
        self._events.append(self._ev(OBJ, Prop.SIZE, h, (0.002**2,) * 3, SRC_SIZE))
        self._events.append(self._ev(PUSHER, Prop.SIZE, PUSHER_SIZE, (1e-8,) * 3, "g1_hand_config"))

    def _emit_sensors(self):
        st = self.true_object_state()
        sp, sy = self.sensor_sigma
        t = self.time
        x = st["pos"][0] + self.rng.normal(0, sp)
        y = st["pos"][1] + self.rng.normal(0, sp)
        yaw = st["yaw"] + self.rng.normal(0, sy)
        self._events.append(self._ev(OBJ, Prop.POSE2, (x, y, math.cos(yaw), math.sin(yaw)),
                                     (sp**2, sp**2, sy**2, sy**2), SRC_POSE))
        # causal derived velocity from consecutive noisy pose measurements (as data.py)
        if self._last_meas is not None:
            t0, x0, y0, yaw0 = self._last_meas
            dt = t - t0
            if 1e-6 < dt <= 0.3:
                dyaw = (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
                vv = (2 * sp**2 / dt**2, 2 * sp**2 / dt**2, 2 * sy**2 / dt**2)
                self._events.append(self._ev(OBJ, Prop.VEL2, ((x - x0) / dt, (y - y0) / dt, dyaw / dt), vv, SRC_VEL,
                                             prov=Provenance.DERIVED))
        self._last_meas = (t, x, y, yaw)
        self._last_forces = self.contact_forces()
        f = self._last_forces["hand"]
        self._tick_count += 1
        if self._tick_count % 2 == 0:  # synthetic tactile at 10 Hz, thresholded noisy normal force
            ind = 1.0 if f + self.rng.normal(0, CONTACT_NOISE) > CONTACT_FORCE_THRESHOLD else 0.0
            if self.rng.random() < CONTACT_FLIP:
                ind = 1.0 - ind
            self._events.append(self._ev(PUSHER, Prop.CONTACT, (ind,), (0.05,), SRC_TACTILE))
        fc = self._last_forces
        self._trace.append(dict(t=t, x=float(st["pos"][0]), y=float(st["pos"][1]), z=float(st["pos"][2]),
                                yaw=float(st["yaw"]), v=float(np.linalg.norm(st["lin_vel"][:3])),
                                w=float(np.linalg.norm(st["ang_vel"])), f_hand=float(f), zmin=self.object_aabb_zmin(),
                                f_support=float(fc["support"]), f_dest=float(fc["dest"]),
                                hand_l=[float(v) for v in self.data.site_xpos[self._site_l]],
                                hand_r=[float(v) for v in self.data.site_xpos[self._site_r]]))

    def drain_events(self):
        ev, self._events = self._events, []
        return ev


class ActorPort:
    """The only interface the actor holds. No MjData/MjModel, no truth."""

    __slots__ = ("_sim",)

    def __init__(self, sim: G1Sim):
        self._sim = sim

    @property
    def time(self) -> float:
        return self._sim.time

    @property
    def episode_id(self) -> str:
        return self._sim.episode_id

    def encoders(self, names):
        return self._sim.read_encoders(names)

    def command(self, targets: dict):
        self._sim.set_targets(targets)

    def advance(self, dt: float):
        self._sim.step(max(1, int(round(dt / PHYSICS_DT))))

    def drain_events(self):
        return self._sim.drain_events()


# --------------------------------------------------------------------------
# actor side: kinematics, IK, primitives
# --------------------------------------------------------------------------


def home_posture():
    q = {}
    for s, sg in (("left", 1.0), ("right", -1.0)):
        q[f"{s}_shoulder_pitch_joint"] = 0.3
        q[f"{s}_shoulder_roll_joint"] = 0.25 * sg
        q[f"{s}_elbow_joint"] = -0.3
        q[f"{s}_wrist_pitch_joint"] = 0.0
    return q


def hand_posture(model, side: str, closure: float):
    """closure 0 = open (thumb/fingers extended), 1 = fist. Moves each joint
    from the 'open' end of its range toward the 'closed' end."""
    out = {}
    for jn in HAND_JOINTS[side]:
        lo, hi = model.jnt_range[model.joint(jn).id]
        # closing direction: fingers flex toward the non-zero end; thumb_0 stays centred
        if "thumb_0" in jn:
            out[jn] = 0.0
            continue
        closed = lo if abs(lo) > abs(hi) else hi
        opened = 0.0 if (lo <= 0.0 <= hi) else (hi if closed == lo else lo)
        out[jn] = opened + closure * (closed - opened) * 0.85
    return out


class IKFailure(RuntimeError):
    pass


class G1Robot:
    """Actor-side robot controller. Holds a robot-only kinematic model and an ActorPort."""

    def __init__(self, port: ActorPort, use_waist: bool = True):
        import mujoco

        self._mj = mujoco
        self.port = port
        self.km = build_robot_kinematic_model()
        self.kd = mujoco.MjData(self.km)
        self.use_waist = use_waist
        self.q_cmd = {}  # last commanded joint targets
        self.joint_order = [self.km.joint(i).name for i in range(self.km.njnt)]
        self.dof = {n: self.km.jnt_dofadr[self.km.joint(n).id] for n in self.joint_order}
        self.qadr = {n: self.km.jnt_qposadr[self.km.joint(n).id] for n in self.joint_order}
        self.events: list[ObservationEvent] = []
        self._evcount = 0
        self._next_obs_t = 0.0
        self.active_side = "right"
        self.active_arm_lowered = {"left": False, "right": False}
        self._last_cmd = None
        self._q_ref = {}
        self._servo_tau = {}
        for i in range(self.km.nu):
            kp, kv = self.km.actuator_gainprm[i, 0], -self.km.actuator_biasprm[i, 2]
            self._servo_tau[self.km.actuator(i).name] = float(kv / kp) if kp > 0 else 0.0
        self.rng = np.random.default_rng(zlib.crc32(port.episode_id.encode()))
        self.cmd_vel = np.zeros(2)
        self.log: list[dict] = []  # per control tick (for video / replay)
        self.q_cmd.update(self.encoder_state())
        home = home_posture()
        self.q_cmd.update(home)
        for s in ("left", "right"):
            self.q_cmd.update(hand_posture(self.km, s, 0.0))
        self.port.command(self.q_cmd)
        self.lam = 0.05  # DLS damping

    # --- kinematics ------------------------------------------------------------
    def _ik_joints(self, side):
        return (WAIST_JOINTS if self.use_waist else []) + ARM_JOINTS[side]

    def _sync_kin(self, q: dict):
        for n, v in q.items():
            if n in self.qadr:
                self.kd.qpos[self.qadr[n]] = v
        self._mj.mj_kinematics(self.km, self.kd)
        self._mj.mj_comPos(self.km, self.kd)

    def encoder_state(self, names=None):
        names = names or [n for n in self.joint_order]
        vals = self.port.encoders(names)
        return dict(zip(names, vals))

    def site_pose(self, q: dict, site: str):
        self._sync_kin(q)
        sid = self.km.site(site).id
        return self.kd.site_xpos[sid].copy(), self.kd.site_xmat[sid].reshape(3, 3).copy()

    def fk_pusher(self, side=None):
        side = side or self.active_side
        enc = self.encoder_state(WAIST_JOINTS + ARM_JOINTS["left"] + ARM_JOINTS["right"])
        return self.site_pose(enc, f"{side}_pusher")

    def solve_ik(self, side, p_des, axis_des=None, q0=None, iters=30, tol=1e-3, w_axis=0.3, posture=None,
                 site=None):
        """Damped least squares on the robot-only model. Task: pusher site position
        (3) + soft alignment of the hand axis (site local x) to axis_des. Returns
        joint dict. Joint limits enforced by clipping; null-space pull to posture."""
        names = self._ik_joints(side)
        q = dict(self.q_cmd if q0 is None else q0)
        lo = np.array([self.km.jnt_range[self.km.joint(n).id][0] for n in names])
        hi = np.array([self.km.jnt_range[self.km.joint(n).id][1] for n in names])
        cols = [self.dof[n] for n in names]
        sid = self.km.site(site or f"{side}_pusher").id
        jp = np.zeros((3, self.km.nv))
        jr = np.zeros((3, self.km.nv))
        post = posture or {}
        err = np.inf
        for _ in range(iters):
            self._sync_kin(q)
            p = self.kd.site_xpos[sid]
            R = self.kd.site_xmat[sid].reshape(3, 3)
            e_p = p_des - p
            self._mj.mj_jacSite(self.km, self.kd, jp, jr, sid)
            Jp = jp[:, cols]
            rows_J = [Jp]
            rows_e = [e_p]
            if axis_des is not None and np.ndim(axis_des) == 2:
                # full orientation target (columns = desired site axes in world)
                e_o = 0.5 * sum(np.cross(R[:, i], axis_des[:, i]) for i in range(3))
                rows_J.append(w_axis * jr[:, cols])
                rows_e.append(w_axis * e_o)
            elif axis_des is not None:
                a = R[:, 0]
                e_o = np.cross(a, axis_des)
                P = np.eye(3) - np.outer(axis_des, axis_des)
                rows_J.append(w_axis * (P @ jr[:, cols]))
                rows_e.append(w_axis * (P @ e_o))
            J = np.vstack(rows_J)
            e = np.concatenate(rows_e)
            err = float(np.linalg.norm(e_p))
            if err < tol and (axis_des is None or np.linalg.norm(rows_e[1]) < 0.02):
                break
            JJt = J @ J.T + (self.lam ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            # null-space posture pull
            if post:
                qv = np.array([q[n] for n in names])
                qp = np.array([post.get(n, q[n]) for n in names])
                N = np.eye(len(names)) - np.linalg.pinv(J) @ J
                dq = dq + 0.1 * N @ (qp - qv)
            dq = np.clip(dq, -0.2, 0.2)
            for i, n in enumerate(names):
                q[n] = float(np.clip(q[n] + dq[i], lo[i] + 1e-3, hi[i] - 1e-3))
        return q, err

    # --- events (proprioception / own command) ----------------------------------
    def _ev(self, entity, prop, value, var, source, prov):
        self._evcount += 1
        t = self.port.time
        return ObservationEvent(
            event_id=f"{self.port.episode_id}-{source}-{self._evcount}", episode_id=self.port.episode_id,
            entity_id=entity, property_type=prop, value=tuple(float(x) for x in value),
            variance=tuple(float(x) for x in var), frame_id="world", measured_at=t, received_at=t,
            source_id=source, provenance=prov)

    def _maybe_emit(self):
        t = self.port.time
        # own command: emitted at every change (push start / end), known immediately (as data.py)
        if self._last_cmd is None or np.any(np.abs(self.cmd_vel - self._last_cmd) > 1e-9):
            self.events.append(self._ev(PUSHER, Prop.COMMAND, tuple(self.cmd_vel), (1e-6, 1e-6), SRC_CMD,
                                        Provenance.SENSOR))
            self._last_cmd = self.cmd_vel.copy()
        if t + 1e-9 < self._next_obs_t:
            return
        self._next_obs_t += SENSOR_DT
        # proprioception: forward kinematics of the emulated encoder joint state (robot-only model).
        # Convention matches data.py for spherical pushers: yaw reported as 0 (+ N(0, 0.5 deg) noise),
        # position noise 0.5 mm, arrival latency U(0.2, 1.0) ms, provenance SENSOR.
        p, _ = self.fk_pusher()
        sp, sy = 0.0005, math.radians(0.5)
        yaw = self.rng.normal(0, sy)
        ev = self._ev(PUSHER, Prop.POSE2, (p[0] + self.rng.normal(0, sp), p[1] + self.rng.normal(0, sp),
                                           math.cos(yaw), math.sin(yaw)),
                      (sp**2, sp**2, sy**2, sy**2), SRC_PROPRIO, Provenance.SENSOR)
        self.events.append(replace(ev, received_at=ev.measured_at + float(self.rng.uniform(0.0002, 0.001))))

    def drain_events(self):
        ev = self.port.drain_events() + self.events
        self.events = []
        return ev

    # --- low-level motion --------------------------------------------------------
    def _tick(self, targets: dict, phase: str = ""):
        self.q_cmd.update(targets)
        self.port.command(self.q_cmd)
        self.port.advance(CONTROL_DT)
        self._maybe_emit()
        self.log.append(dict(t=self.port.time, phase=phase, cmd_vel=[float(self.cmd_vel[0]), float(self.cmd_vel[1])]))

    def hold(self, duration: float, phase="hold"):
        self.cmd_vel[:] = 0
        for _ in range(max(1, int(round(duration / CONTROL_DT)))):
            self._tick({}, phase)

    def push_stroke_servo(self, side, p0, p1, speed, R, phase="push"):
        """Controller v2 for the push stroke (measured-progress stroke). The reference moves at the
        commanded speed from a run-up point behind p0 (so the hand is at speed before contact) and is
        extrapolated past p1 until the MEASURED progress (encoder FK) reaches the commanded stroke end;
        then the stroke ends (release follows). Optional PI term on measured speed error (SERVO_GAINS)."""
        L = float(np.linalg.norm(p1 - p0))
        d = (p1 - p0) / max(L, 1e-9)
        names = self._ik_joints(side)
        enc_names = WAIST_JOINTS + ARM_JOINTS[side]
        g = SERVO_GAINS
        q_ref = dict(self.q_cmd)
        s_cmd = -self._runup  # reference position along the stroke (relative to p0)
        integ, v_f, s_prev, worst = 0.0, None, None, 0.0
        n_max = int((L + self._runup) / max(speed, 1e-3) / CONTROL_DT + g["timeout"] / CONTROL_DT)
        self.servo_log = []
        for i in range(n_max):
            enc = self.encoder_state(enc_names)
            pm, _ = self.site_pose({**self.q_cmd, **enc}, f"{side}_pusher")
            s_m = float((pm - p0) @ d)
            if s_prev is not None:
                v_raw = (s_m - s_prev) / CONTROL_DT
                v_f = v_raw if v_f is None else (1 - g["alpha"]) * v_f + g["alpha"] * v_raw
            s_prev = s_m
            if s_m >= L - speed * g["stop_lead"]:
                break
            if v_f is not None and s_m > 0.0:
                integ = float(np.clip(integ + (speed - v_f) * CONTROL_DT, -g["i_clip"], g["i_clip"]))
            s_cmd += speed * CONTROL_DT
            if g.get("max_lead") is not None and s_m > 0.0:  # anti-surge: bound reference lead over measured
                s_cmd = min(s_cmd, s_m + g["max_lead"])
            s_t = s_cmd + g["ki"] * integ
            q, err = self.solve_ik(side, p0 + d * s_t, R, q0=q_ref, posture=self._posture(side))
            worst = max(worst, err)
            q_ref = q
            self.cmd_vel[:] = d[:2] * speed
            self.servo_log.append((s_m, v_f if v_f is not None else 0.0, s_cmd))
            self._tick({k: q[k] for k in names}, phase)
        self.cmd_vel[:] = 0
        return worst

    def wait_converged(self, side, p_target, tol=0.006, tmax=0.6, phase="pre_contact"):
        """Hold until FK of the ENCODER state puts the pusher site within tol of p_target."""
        t0 = self.port.time
        while True:
            self.hold(0.02, phase)
            enc = self.encoder_state(WAIST_JOINTS + ARM_JOINTS[side])
            p, _ = self.site_pose({**self.q_cmd, **enc}, f"{side}_pusher")
            err = float(np.linalg.norm(p - p_target))
            if err < tol or self.port.time - t0 >= tmax:
                return err

    def set_hand(self, side, closure, duration=0.3):
        tgt = hand_posture(self.km, side, closure)
        self._tick(tgt, f"hand_{closure:.1f}")
        self.hold(duration, "hand")

    def move_linear(self, side, p_to, speed, axis_des=None, phase="move", record_cmd=False, p_from=None):
        """Cartesian straight line of the pusher site at constant speed (IK each tick)."""
        if p_from is None:
            p_from, _ = self.site_pose(self.q_cmd, f"{side}_pusher")
        p_from = np.asarray(p_from, float)
        p_to = np.asarray(p_to, float)
        L = float(np.linalg.norm(p_to - p_from))
        n = max(1, int(math.ceil(L / max(speed, 1e-3) / CONTROL_DT)))
        u = (p_to - p_from) / max(L, 1e-9)
        worst = 0.0
        names = self._ik_joints(side)
        ff = record_cmd and VELOCITY_FEEDFORWARD
        q_ref = dict(self.q_cmd) if not ff else dict(self._q_ref.get(side, self.q_cmd))
        for i in range(1, n + 1):
            p = p_from + (p_to - p_from) * (i / n)
            q, err = self.solve_ik(side, p, axis_des, q0=q_ref, posture=self._posture(side))
            worst = max(worst, err)
            self.cmd_vel[:] = (u[:2] * speed) if record_cmd else 0.0
            if ff:
                # PD ramp-lag compensation: target = q_ik + (kv/kp) * dq_ik/dt (robot-model servo constants)
                tgt = {k: q[k] + self._servo_tau.get(k, 0.0) * (q[k] - q_ref[k]) / CONTROL_DT for k in names}
            else:
                tgt = {k: q[k] for k in names}
            q_ref = q
            self._tick(tgt, phase)
        if ff:  # final target exactly on the path end (no offset) -> crisp stop
            self._tick({k: q_ref[k] for k in names}, phase)
        self._q_ref[side] = q_ref
        self.cmd_vel[:] = 0
        return worst

    def _posture(self, side):
        sg = 1.0 if side == "left" else -1.0
        return {"waist_yaw_joint": 0.0, "waist_pitch_joint": 0.25, f"{side}_shoulder_roll_joint": 0.15 * sg,
                f"{side}_shoulder_yaw_joint": 0.0, f"{side}_wrist_roll_joint": 0.0}

    def park(self, side, duration=0.4):
        h = home_posture()
        tgt = {k: v for k, v in h.items() if k.startswith(side)}
        for k in ARM_JOINTS[side]:
            tgt.setdefault(k, 0.0)
        self._interp_joints(tgt, duration, "park")

    def _interp_joints(self, tgt: dict, duration: float, phase: str):
        start = {k: self.q_cmd.get(k, 0.0) for k in tgt}
        n = max(1, int(round(duration / CONTROL_DT)))
        for i in range(1, n + 1):
            a = i / n
            a = 0.5 - 0.5 * math.cos(math.pi * a)
            self._tick({k: start[k] + a * (tgt[k] - start[k]) for k in tgt}, phase)

    # --- task-space push primitive ----------------------------------------------
    @staticmethod
    def push_geometry(half, theta, phi):
        """Same convention as worlds._push_geometry: boundary point (local),
        push direction (local), outward normal (local)."""
        ux, uy = math.cos(theta), math.sin(theta)
        hx, hy = half[0], half[1]
        sx = hx / abs(ux) if abs(ux) > 1e-9 else np.inf
        sy = hy / abs(uy) if abs(uy) > 1e-9 else np.inf
        if sx < sy:
            p = (sx * ux, sx * uy)
            n = (math.copysign(1.0, ux), 0.0)
        else:
            p = (sy * ux, sy * uy)
            n = (0.0, math.copysign(1.0, uy))
        cp, sp = math.cos(phi), math.sin(phi)
        d = (-(cp * n[0] - sp * n[1]), -(sp * n[0] + cp * n[1]))
        return np.array(p), np.array(d), np.array(n)

    def plan_push(self, cand: ActionCandidate, obj_pose, half, effector_radius=None):
        """World-frame plan for a push candidate given the actor's pose estimate
        (x, y, cos, sin) at decision time and estimated half extents."""
        th, ph, speed, dur = cand.params
        x, y, c, s = obj_pose[:4]
        yaw = math.atan2(s, c)
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        p_l, d_l, n_l = self.push_geometry(half, th, ph)
        r = G1_CAPABILITY.effector_radius if effector_radius is None else effector_radius
        start_l = p_l + n_l * (r + PUSH_GAP)
        start = np.array([x, y]) + R @ start_l
        dw = R @ d_l
        z_push = TABLE_TOP + PUSH_Z
        z_above = TABLE_TOP + 2 * half[2] + 0.07
        end = start + dw * speed * dur
        nin = -(R @ n_l)  # inward normal of the contacted face (world)
        plan = dict(start=start, dir=dw, speed=speed, duration=dur, z_push=z_push, z_above=z_above, end=end,
                    face=nin)
        # choose arm + hand lean by IK feasibility on the robot-only model (actor side)
        pref = "left" if start[1] > 0.0 else "right"
        best = None
        for side in (pref, "right" if pref == "left" else "left"):
            for lead in PUSH_LEADS:
                Rh = self.hand_frame(side, dw, lead, nin)
                worst = 0.0
                q = dict(self.q_cmd)
                sols = []
                for wp in (np.r_[start, TABLE_TOP + TRANSIT_Z], np.r_[start, z_push], np.r_[0.5 * (start + end), z_push],
                           np.r_[end, z_push]):
                    q, err = self.solve_ik(side, wp, Rh, q0=q, iters=60, posture=self._posture(side))
                    sols.append(q)
                    worst = max(worst, err)
                    if worst > IK_FEASIBLE_TOL:
                        break
                if best is None or worst < best[0]:
                    best = (worst, side, lead, sols[0])
                if worst <= IK_FEASIBLE_TOL:
                    break
            if best[0] <= IK_FEASIBLE_TOL:
                break
        plan.update(side=best[1], lead=best[2], ik_err=best[0], feasible=bool(best[0] <= IK_FEASIBLE_TOL),
                    q_above={k: best[3][k] for k in self._ik_joints(best[1])})
        return plan

    def hand_axis(self, side, xy):
        """Hand pointing down, tilted ~35 deg outward from the shoulder (reach)."""
        sh = np.array([0.0, 0.1 if side == "left" else -0.1])
        h = np.asarray(xy) - sh
        h = h / max(np.linalg.norm(h), 1e-6)
        a = np.array([HAND_TILT * h[0], HAND_TILT * h[1], -1.0])
        return a / np.linalg.norm(a)

    def hand_frame(self, side, d_xy, lead=None, face_xy=None):
        """Flat-hand paddle: fingers (site x) point down, leaning PUSH_LEAD toward the
        push direction so the fingertips lead (low contact); the back of the hand
        (thumb on the far side) faces the push direction."""
        d = np.array([d_xy[0], d_xy[1], 0.0])
        d /= max(np.linalg.norm(d), 1e-9)
        x = np.array([0.0, 0.0, -1.0]) + (PUSH_LEAD if lead is None else lead) * d
        x /= np.linalg.norm(x)
        f = d if face_xy is None else np.array([face_xy[0], face_xy[1], 0.0]) / max(np.linalg.norm(face_xy), 1e-9)
        y = (-f if side == "right" else f)  # back of the hand faces `f` (the inward face normal)
        y = y - x * (x @ y)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        return np.stack([x, y, z], 1)

    def execute_push(self, cand: ActionCandidate, obj_pose, half, plan=None):
        """approach above pre-contact point -> descend -> push along direction at
        commanded speed for duration -> retract. Returns a log dict."""
        plan = plan or self.plan_push(cand, obj_pose, half)
        side = plan["side"]
        other = "right" if side == "left" else "left"
        self.active_side = side
        t0 = self.port.time
        self.set_hand(side, PUSH_CLOSURE, 0.05)
        st, en = plan["start"], plan["end"]
        ax = self.hand_frame(side, plan["dir"], plan["lead"], plan.get("face"))
        above = np.array([st[0], st[1], plan["z_above"]])
        # approach: other arm parks, active arm moves in joint space to the IK solution above the
        # pre-contact point (avoids sweeping the hand through the object), then descends.
        z_tr = TABLE_TOP + TRANSIT_Z
        if self.active_arm_lowered.get(other):  # lift the other hand vertically, then park it
            po, _ = self.site_pose(self.q_cmd, f"{other}_pusher")
            if po[2] < z_tr:
                self.move_linear(other, np.r_[po[:2], z_tr], 0.6, None, "transit")
            self.park(other, 0.4)
            self.active_arm_lowered[other] = False
        pc, _ = self.site_pose(self.q_cmd, f"{side}_pusher")
        if self.active_arm_lowered.get(side) and pc[2] < z_tr:
            self.move_linear(side, np.r_[pc[:2], z_tr], 0.6, None, "transit")
        self._interp_joints(plan["q_above"], 0.7, "approach")  # transit-height IK solution above start
        e1 = e2 = 0.0
        self.active_arm_lowered[side] = True
        self._runup = SERVO_GAINS["runup"] if PUSH_SERVO == "v2" else 0.0
        sb = np.asarray(st) - np.asarray(plan["dir"]) * self._runup  # run-up start (controller v2)
        self.move_linear(side, np.array([sb[0], sb[1], plan["z_above"]]), 0.4, ax, "descend")
        self.wait_converged(side, np.array([sb[0], sb[1], plan["z_above"]]), tol=0.006, tmax=0.5, phase="align")
        e3 = self.move_linear(side, np.array([sb[0], sb[1], plan["z_push"]]), 0.15, ax, "descend")
        self.wait_converged(side, np.array([sb[0], sb[1], plan["z_push"]]), tol=0.006, tmax=0.6)
        t_push = self.port.time
        p_st, p_en = np.array([st[0], st[1], plan["z_push"]]), np.array([en[0], en[1], plan["z_push"]])
        if PUSH_SERVO == "v2":
            e4 = self.push_stroke_servo(side, p_st, p_en, plan["speed"], ax)
        else:
            e4 = self.move_linear(side, p_en, plan["speed"], ax, "push", record_cmd=True)
        t_end = self.port.time
        # hand stops and immediately withdraws along -d (release), so the object slides
        # freely under table friction (hidden, friction-dependent); then lifts.
        back = np.array([en[0], en[1], plan["z_push"]]) - np.r_[plan["dir"], 0.0] * RELEASE_BACKOFF
        self.move_linear(side, back, RELEASE_SPEED, ax, "release")
        self.move_linear(side, np.r_[back[:2], plan["z_above"]], 0.5, ax, "retract")
        return dict(side=side, lead=plan["lead"], t_start=t0, t_push=t_push, t_push_end=t_end,
                    ik_err=max(e1, e2, e3, e4), start=st.tolist(), end=en.tolist(), dir=plan["dir"].tolist())

    # --- reach / grasp / lift / place (bimanual flat-hand squeeze) ----------------
    def reach(self, side, p, R=None, speed=0.4, phase="reach"):
        return self.move_linear(side, p, speed, R, phase)

    def move_bimanual(self, pL, pR, RL, RR, speed, phase):
        """Both pusher sites along straight lines simultaneously (waist frozen)."""
        aL, _ = self.site_pose(self.q_cmd, "left_pusher")
        aR, _ = self.site_pose(self.q_cmd, "right_pusher")
        L = max(np.linalg.norm(pL - aL), np.linalg.norm(pR - aR))
        n = max(1, int(math.ceil(L / max(speed, 1e-3) / CONTROL_DT)))
        worst = 0.0
        saved = self.use_waist
        self.use_waist = False
        try:
            for i in range(1, n + 1):
                a = i / n
                tgt = {}
                mid = 0.5 * ((aL + (pL - aL) * a) + (aR + (pR - aR) * a))
                # waist yaw faces the object midpoint so both arms work symmetrically
                wy = float(np.clip(math.atan2(mid[1], mid[0]), -0.6, 0.6)) if BIMANUAL_WAIST_TRACK else \
                    self.q_cmd.get("waist_yaw_joint", 0.0)
                self.q_cmd["waist_yaw_joint"] = wy
                tgt["waist_yaw_joint"] = wy
                for side, p0, p1, R in (("left", aL, pL, RL), ("right", aR, pR, RR)):
                    q, err = self.solve_ik(side, p0 + (p1 - p0) * a, R, posture=self._posture(side), iters=30,
                                           w_axis=BIMANUAL_W_AXIS)
                    worst = max(worst, err)
                    tgt.update({k: q[k] for k in ARM_JOINTS[side]})
                self._tick(tgt, phase)
        finally:
            self.use_waist = saved
        return worst

    def lift_place(self, obj_pose, half, dest_xy, lift_clear=0.06, squeeze=0.012, speed=0.15, z_support=None):
        """reach -> bimanual grasp (flat hands squeeze the +/-y faces) -> lift ->
        carry -> place in the destination -> release -> withdraw. Uses only the
        actor's pose/size estimates. Returns phase log."""
        x, y, c, s_ = obj_pose[:4]
        yaw = math.atan2(s_, c)
        ny = np.array([-math.sin(yaw), math.cos(yaw)])  # object +y axis in world
        hy = half[1]
        zs = z_support if z_support is not None else TABLE_TOP
        z_grip = zs + half[2] - GRIP_BELOW_CENTER  # fingertip just below the box centre height
        z_above = z_grip + 2 * half[2] + 0.06
        center = np.array([x, y])
        out = {}
        self._interp_joints({"waist_yaw_joint": 0.0, "waist_pitch_joint": 0.25}, 0.3, "waist")
        self.set_hand("left", 0.0, 0.05)
        self.set_hand("right", 0.0, 0.05)
        # left hand at +y face (pushes -y inward), right at -y face
        RL = self.hand_frame("left", -ny, 0.0)
        RR = self.hand_frame("right", ny, 0.0)
        def pts(cxy, off, z):
            return (np.r_[cxy + ny * off, z], np.r_[cxy - ny * off, z])
        pre = hy + 0.045
        pl, pr = pts(center, pre + 0.03, z_above + 0.12)  # high, wide pre-pose (joint-space move)
        # joint-space approach to IK solutions above the faces
        saved = self.use_waist
        self.use_waist = False
        self.q_cmd["waist_yaw_joint"] = float(np.clip(math.atan2(center[1], center[0]), -0.6, 0.6))
        ql, el = self.solve_ik("left", pl, RL, iters=80, posture=self._posture("left"))
        qr, er = self.solve_ik("right", pr, RR, iters=80, posture=self._posture("right"))
        self.use_waist = saved
        wy = self.q_cmd["waist_yaw_joint"]
        self.q_cmd["waist_yaw_joint"] = 0.0
        self._interp_joints({**{k: ql[k] for k in ARM_JOINTS["left"]}, **{k: qr[k] for k in ARM_JOINTS["right"]},
                             "waist_yaw_joint": wy}, 0.8, "reach")
        out["reach_ik_err"] = max(el, er)
        pl, pr = pts(center, pre, z_above)
        out["reach"] = self.move_bimanual(pl, pr, RL, RR, 0.4, "reach")
        pl, pr = pts(center, pre, z_grip)
        out["descend"] = self.move_bimanual(pl, pr, RL, RR, 0.3, "descend")
        pl, pr = pts(center, hy + 0.015 - squeeze - PUSHER_SIZE[0], z_grip)
        out["grasp"] = self.move_bimanual(pl, pr, RL, RR, 0.08, "grasp")
        self.hold(0.3, "grasp_settle")
        pl, pr = pts(center, hy + 0.015 - squeeze - PUSHER_SIZE[0], z_grip + lift_clear)
        out["lift"] = self.move_bimanual(pl, pr, RL, RR, speed, "lift")
        dest = np.asarray(dest_xy, float)
        pl, pr = pts(dest, hy + 0.015 - squeeze - PUSHER_SIZE[0], z_grip + lift_clear)
        out["carry"] = self.move_bimanual(pl, pr, RL, RR, speed, "carry")
        z_place = TABLE_TOP + 0.008 + half[2] - GRIP_BELOW_CENTER + 0.004
        pl, pr = pts(dest, hy + 0.015 - squeeze - PUSHER_SIZE[0], z_place)
        out["place"] = self.move_bimanual(pl, pr, RL, RR, 0.08, "place")
        self.hold(0.2, "place_settle")
        pl, pr = pts(dest, pre, z_place)
        out["release"] = self.move_bimanual(pl, pr, RL, RR, 0.1, "release")
        pl, pr = pts(dest, pre, z_above)
        out["withdraw"] = self.move_bimanual(pl, pr, RL, RR, 0.3, "withdraw")
        return out

    def retreat(self, side=None, duration=0.5):
        side = side or self.active_side
        self.park(side, duration)


# --------------------------------------------------------------------------
# environment wrapper used by control.run_episode
# --------------------------------------------------------------------------


def sample_push_episode(seed: int, friction: Optional[float] = None, mass: Optional[float] = None,
                        fric_range=(0.3, 0.7), mass_range=(0.2, 2.0)) -> EpisodeSpec:
    rng = np.random.default_rng(seed)
    mu = float(rng.uniform(*fric_range)) if friction is None else friction
    m = float(np.exp(rng.uniform(np.log(mass_range[0]), np.log(mass_range[1])))) if mass is None else mass
    half = (float(rng.uniform(0.035, 0.05)), float(rng.uniform(0.035, 0.05)), float(rng.uniform(0.02, 0.025)))
    pos = (float(rng.uniform(0.34, 0.44)), float(rng.uniform(-0.15, 0.15)))
    yaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
    for _ in range(100):
        ang = rng.uniform(-math.pi, math.pi)
        dist = rng.uniform(0.08, 0.16)
        tgt = (pos[0] + dist * math.cos(ang), pos[1] + dist * math.sin(ang))
        if 0.32 <= tgt[0] <= 0.46 and -0.20 <= tgt[1] <= 0.20:
            break
    return EpisodeSpec(variant="push", box=BoxSpec(half=half, mass=m, friction=mu, pos=pos, yaw=yaw),
                       target=(float(tgt[0]), float(tgt[1])), seed=seed)


SLIDE_TARGET_DIST = (0.15, 0.30)
SLIDE_TARGET_REGION = ((0.30, 0.62), (-0.34, 0.34))  # target centre stays on the table with margin


def sample_slide_episode(seed: int, friction: Optional[float] = None, mass: Optional[float] = None,
                         fric_range=(0.3, 0.7), mass_range=(0.2, 2.0)) -> EpisodeSpec:
    """'slide-to-target': target 15-30 cm from the box along a direction the hand can push from
    (pre-contact point inside the capability workspace)."""
    rng = np.random.default_rng(10_000 + seed)
    mu = float(rng.uniform(*fric_range)) if friction is None else friction
    m = float(np.exp(rng.uniform(np.log(mass_range[0]), np.log(mass_range[1])))) if mass is None else mass
    half = (float(rng.uniform(0.035, 0.05)), float(rng.uniform(0.035, 0.05)), float(rng.uniform(0.02, 0.025)))
    for _ in range(1000):
        pos = (float(rng.uniform(0.32, 0.40)), float(rng.uniform(-0.10, 0.10)))
        ang = float(rng.uniform(-math.pi, math.pi))
        dist = float(rng.uniform(*SLIDE_TARGET_DIST))
        u = np.array([math.cos(ang), math.sin(ang)])
        tgt = np.array(pos) + dist * u
        pre = np.array(pos) - u * (max(half[:2]) * 1.42 + G1_CAPABILITY.effector_radius + PUSH_GAP)
        (xl, xh), (yl, yh) = SLIDE_TARGET_REGION
        lo, hi = np.array(G1_CAPABILITY.workspace_min), np.array(G1_CAPABILITY.workspace_max)
        if xl <= tgt[0] <= xh and yl <= tgt[1] <= yh and np.all(pre >= lo) and np.all(pre <= hi):
            # reachability of the direct push toward the target (IK on the robot-only model, home posture)
            yaw_try = float(rng.uniform(-math.pi / 4, math.pi / 4))
            th = ang - yaw_try + math.pi
            cand = ActionCandidate("chk", "push", (th, 0.0, 0.5, 0.12), OBJ)
            plan = _kin_checker().plan_push(cand, (pos[0], pos[1], math.cos(yaw_try), math.sin(yaw_try)), half)
            if plan["feasible"]:
                break
    return EpisodeSpec(variant="push", box=BoxSpec(half=half, mass=m, friction=mu, pos=pos, yaw=yaw_try),
                       target=(float(tgt[0]), float(tgt[1])), seed=seed)


class _NullPort:
    """Kinematics-only port (episode sampling); never connected to a simulator."""
    time = 0.0
    episode_id = "kin-check"

    def encoders(self, names):
        return np.zeros(len(names))

    def command(self, targets):
        pass

    def advance(self, dt):
        raise RuntimeError("kinematic checker cannot advance time")

    def drain_events(self):
        return []


_KIN = None


def _kin_checker():
    global _KIN
    if _KIN is None:
        _KIN = G1Robot(_NullPort())
    return _KIN


class G1PushWorld:
    """Bundles a privileged sim (evaluator side) and an actor (robot + port).
    control.run_episode talks to ``world.actor`` for observation/actuation and
    to ``world.evaluate_*`` (privileged) only for success/metrics."""

    def __init__(self, ep: EpisodeSpec, seed: int = 0):
        self.ep = ep
        self._sim = G1Sim(ep, seed=seed)
        self.port = ActorPort(self._sim)
        self.actor = G1Robot(self.port)
        self.capability = G1_CAPABILITY
        self.goal = dict(target_xy=tuple(ep.target), radius=ep.target_radius)
        self.target_entity = OBJ

    # evaluator-side (privileged) views
    def true_object(self):
        return self._sim.true_object_state()

    def audit(self):
        a = dict(self._sim.audit)
        a["object_hash_t0"] = self._sim._obj_hash0
        a["object_hash_now"] = self._sim.object_hash()
        a["n_equality"] = int(self._sim.model.neq)
        return a

    def trace(self):
        return list(self._sim._trace)

    def push_success(self, window=0.5, pos_tol=0.03, speed_tol=0.02):
        """Exact push criterion: centroid within 3 cm of target, speed < 2 cm/s for
        the last 0.5 s, and no reset / object-state write / constraint / applied force."""
        tr = self._sim._trace
        st = self._sim.true_object_state()
        d = float(np.hypot(st["pos"][0] - self.ep.target[0], st["pos"][1] - self.ep.target[1]))
        t = self._sim.time
        recent = [r for r in tr if r["t"] >= t - window - 1e-9]
        still = len(recent) >= int(window / SENSOR_DT) and all(r["v"] < speed_tol for r in recent)
        a = self._sim.audit
        clean = a["external_object_writes"] == 0 and a["eq_active"] == 0 and a["applied_force_steps"] == 0
        return dict(success=bool(d <= pos_tol and still and clean), dist=d, still=bool(still), clean=bool(clean))


# --------------------------------------------------------------------------
# rail lift/place variant
# --------------------------------------------------------------------------

RAIL_EPISODE_CAP = 15.0
RAIL_X, TRAY_X = 0.38, 0.51
RAIL_LAYOUT = "y"  # tested both; 'y' was better in the pilot (1/4 vs 0/4)  # carry direction (x) is perpendicular to the squeeze axis (y)


def sample_rail_episode(seed: int) -> EpisodeSpec:
    rng = np.random.default_rng(seed)
    half = (float(rng.uniform(0.048, 0.055)), float(rng.uniform(0.03, 0.04)), float(rng.uniform(0.028, 0.035)))
    mass = float(rng.uniform(0.2, 0.5))
    mu = float(rng.uniform(0.3, 0.7))
    ep = EpisodeSpec(variant="rail", seed=seed)
    if RAIL_LAYOUT == "y":  # rails on the right, tray on the left (carry along the squeeze axis)
        ep.rail_xy = (float(0.40 + rng.uniform(-0.02, 0.02)), float(-0.07 + rng.uniform(-0.015, 0.015)))
        ep.tray_xy = (float(0.40 + rng.uniform(-0.02, 0.02)), float(0.10 + rng.uniform(-0.01, 0.01)))
        ep.tray_half = (0.075, 0.065)
    else:  # rails near, tray far (carry perpendicular to the squeeze axis)
        ep.rail_xy = (float(RAIL_X + rng.uniform(-0.01, 0.01)), float(rng.uniform(-0.03, 0.03)))
        ep.tray_xy = (float(TRAY_X + rng.uniform(-0.01, 0.01)), float(rng.uniform(-0.03, 0.03)))
    ep.box = BoxSpec(half=half, mass=mass, friction=mu,
                     pos=(ep.rail_xy[0] + float(rng.uniform(-0.004, 0.004)), ep.rail_xy[1] + float(rng.uniform(-0.01, 0.01))),
                     yaw=float(rng.uniform(-0.08, 0.08)))
    ep.target = ep.tray_xy
    return ep


class G1RailWorld(G1PushWorld):
    def lift_place_success(self, window=0.5, pos_tol=0.03, lin_tol=0.02, ang_tol=0.1, clear=0.02):
        """Exact lift/place criterion. Evaluator-only (privileged truth)."""
        ep = self.ep
        tr = self._sim._trace
        rail_top = TABLE_TOP + ep.rail_height
        max_clear = max((r["zmin"] - rail_top) for r in tr) if tr else -1.0
        st = self._sim.true_object_state()
        d = float(np.hypot(st["pos"][0] - ep.tray_xy[0], st["pos"][1] - ep.tray_xy[1]))
        t = self._sim.time
        recent = [r for r in tr if r["t"] >= t - window - 1e-9]
        still = len(recent) >= int(window / SENSOR_DT) and all(r["v"] < lin_tol and r["w"] < ang_tol for r in recent)
        f = self._sim.contact_forces()
        supported = f["dest"] > 0.5 * ep.box.mass * G and f["support"] == 0.0
        released = f["hand"] == 0.0 and f["robot"] == 0.0
        a = self._sim.audit
        clean = a["external_object_writes"] == 0 and a["eq_active"] == 0 and a["applied_force_steps"] == 0 \
            and a["nan_steps"] == 0
        stable = a["max_obj_speed"] < 2.0
        ok = bool(max_clear >= clear and d <= pos_tol and still and supported and released and clean and stable
                  and t <= RAIL_EPISODE_CAP)
        return dict(success=ok, max_clearance=max_clear, dist=d, still=bool(still), supported=bool(supported),
                    released=bool(released), clean=bool(clean), stable=bool(stable), sim_time=t,
                    f_dest=f["dest"], f_hand=f["hand"])
