"""Adapter: real embodiments (agent ROBOTS: physmem.embodiments + physmem.skills) inside StationEnv.

    env = make_env("peg_insert")
    env.reset(TaskSpec("peg_insert", "fine", 0, robot="panda", split="train"))   # base calls make_robot_adapter

EmbodimentRobot exposes what StationEnv / SensorSuite need (build/bind/step/tcp/joint_state/grip_width/wrench/
pad_contacts/effort/robot_geoms/grip_cmd/suction_cmd). EmbodimentSkills.run(env, call) translates the env's
SkillCall params (the probe's conventions, see envs/probe.py) into physmem.skills.run_skill calls whose
ObjectEstimates are built ONLY from actor-legal env helpers (est_pose/est_size/est_velocity/region/handle_path +
CAD meta), never from simulator truth. The skill loop runs env.pre_physics() before and env.post_physics() after
every mj_step (integrity monitor, contact monitor, sensors, clock).

Object<->gripper friction: explicit contact pairs between each graspable object's geoms and every collision geom of
the gripper's pad bodies (GripperCfg.pad_bodies), with the object's hidden mu_grip, plus body excludes so the pair is
the only contact - same convention as the probe.

grip_force (N, per pad, like the probe) is set with ROBOTS' calibrated emb.set_grip_force(eff, N) (clipped to
emb.max_grip_force(eff)); suction embodiments (ur5e_suction, fr3_suction) ignore it.
"""
from __future__ import annotations

import math
import os

import mujoco
import numpy as np

from ..v2.api import EffortMeter, SkillCall, SkillResult

# nominal max grasp force (N) of each gripper at full drive effort (spec sheets; used only to scale grip_force)
# PER-PAD normal force (N) at full drive effort, measured on a 20 mm peg (scripts/envs_robot_smoke.py dev runs):
# panda 70 N total -> 35/pad; 2F-85 model 89 N total -> 44/pad. Others: spec-sheet guesses (not measured).
# (GRIP_N_MAX table removed: grip force now via emb.set_grip_force / emb.max_grip_force, ROBOTS round 2)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw_quat(y):
    return np.array([math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)])


def make_robot_adapter(env, name):
    r = EmbodimentRobot(name)
    return r, EmbodimentSkills(r)


class EmbodimentRobot:
    mount_key = None

    def __init__(self, name):
        import physmem.embodiments as E
        self.E = E
        self.name = name
        self.emb = E.make(name, prefix="r0")
        import copy
        self.emb.cfg = copy.deepcopy(self.emb.cfg)     # we adjust gripper wait times per instance (see bind)
        self.spec = self.emb.spec
        self.mount_key = {"arm": "arm", "bimanual": "bimanual", "humanoid": "humanoid",
                          "mobile_manipulator": "mobile_manipulator", "gantry": "gantry"}.get(self.spec.kind, "arm")
        self.grip_cmd = ("open", 0.0)
        self.suction_cmd = 0.0
        self.tare = np.zeros(6)
        self.t = 0.0
        self._eff0 = EffortMeter()
        self.robot_geoms = set()

    # ------------------------------------------------------------------ build
    def build(self, scene, mount):
        s = scene.spec
        ts = float(s.option.timestep)
        integ = s.option.integrator
        self.E.scene_defaults(s, timestep=ts)      # implicitfast, elliptic, impratio 10 (keep the family's dt)
        if len(s.flexes) if hasattr(s, 'flexes') else False:
            s.option.integrator = integ            # flex elasticity needs the family's (implicit) integrator
        pos, yaw = mount
        pos = np.asarray(pos, float)
        if self.spec.kind == "bimanual":
            # ALOHA layout: both arm bases on the work surface at the two ends of the bench, facing each other
            ms = scene.mounts.get("bimanual", [mount])
            cx = float(np.mean([m[0][0] for m in ms]))
            cy = float(scene.camera.get("view_lookat", scene.camera["lookat"])[1])
            pos, yaw = np.array([cx, cy, float(ms[0][0][2])]), 0.0
        elif self.spec.kind == "humanoid":
            # station humanoid mounts stand 0.15 m behind the bench edge; with the G1's short arms the work sat ~4 cm
            # out of reach even with waist pitch -> step 0.12 m closer (ROBOTS G1 sub-agent finding)
            # (G1 round 3: +0.12 put the pelvis 3 cm from the bench edge -> hips hit the bench top and the robot was
            # shoved 0.45 m back at reset; +0.03 = pelvis 0.12 m behind the edge, reach comes from waist lean)
            pos = pos + 0.03 * np.array([math.cos(yaw), math.sin(yaw), 0.0])
        elif self.spec.kind == "arm" and pos[2] > 0.05:
            # pedestal under a fixed-base arm mounted at work-surface height (static, visual + contact)
            scene.static_box((pos[0], pos[1], pos[2] / 2), (0.09, 0.09, pos[2] / 2 - 0.002), rgba=(0.3, 0.3, 0.32, 1))
        self.mount = (pos, yaw)
        # tracker camera: the stations put it behind the near (robot) edge, looking over the robot's shoulder - fine
        # for the gantry probe, but a real arm's elbow then hides the work area. Mirror it to the far side of the
        # station (same height / look-at); rendering views (view_pos) are unchanged.
        cam = dict(scene.camera)
        cp, lk = np.asarray(cam["pos"], float), np.asarray(cam["lookat"], float)
        if cp[1] < lk[1]:
            cam["pos"] = (float(cp[0]), float(2 * lk[1] - cp[1]), float(cp[2]))
            scene.camera = cam
        self.emb.build(s, mount_pose=(tuple(pos), tuple(_yaw_quat(yaw))))
        # explicit object<->pad friction pairs (hidden mu_grip), as in ProbeRobot.build
        P = self.emb.prefix
        pad_geoms, pad_bodies = [], []
        for e in self.emb.cfg.effectors:
            for bn in e.gripper.pad_bodies:
                b = s.body(P + bn)
                if b is None:
                    continue
                pad_bodies.append(b.name)
                for k, g in enumerate(b.geoms):
                    if g.contype == 0 and g.conaffinity == 0:
                        continue
                    if not g.name:
                        g.name = f"{b.name}_padg{k}"
                    pad_geoms.append(g.name)
        self.pad_geom_names = pad_geoms
        for eid in scene.grip_pairs_objects:
            info = scene.entities[eid]
            mu = info.hidden.get("mu_grip", 0.8)
            for gname in info.geoms:
                for pg in pad_geoms:
                    # stiff pad pairs with NO margin (ENVS envs-r3 PAD_SOLREF/PAD_SOLIMP: the default 0.02 s pair let a
                    # squeeze push small stiff spheres through the pads; a 1 mm margin made parts stick after release)
                    # suction lips need a contact MARGIN: MuJoCo adhesion acts only through existing contacts, so with
                    # margin 0 the lip contact vanishes at the first micron of separation and the vacuum cannot hold
                    if os.environ.get("PHYSMEM_PAD_PAIRS", "envs") == "legacy":     # A/B switch: default pair contact
                        s.add_pair(geomname1=pg, geomname2=gname, condim=4, friction=[mu, mu, 0.005, 0.0001, 0.0001])
                    else:
                        s.add_pair(geomname1=pg, geomname2=gname, condim=4, friction=[mu, mu, 0.005, 0.0001, 0.0001],
                                   solref=[max(0.004, 5.0 * ts), 1.0], solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
                                   margin=0.002 if "_lip" in pg else 0.0)
            for bn in pad_bodies:
                s.add_exclude(bodyname1=info.body, bodyname2=bn)

    def bind(self, m, d):
        self.m, self.d = m, d
        self.emb.bind(m, d)
        self.emb.reset(m, d)
        self.ctrl = self.emb.controller()
        self.ctrl.reset(m, d)
        self.robot_geoms = set(int(g) for g in self.emb.geoms)
        self.effs = list(self.emb.eff)
        self.eff = self.effs[0] if self.spec.kind != "bimanual" else "right"
        self.gforce0 = {k: np.array(v["gforce"], float).copy() for k, v in self.emb.eff.items()}
        # gripper drive dofs (their implicit damping sets the free closing speed ~ force / damping)
        self.gdofs = {}
        for k, e in self.emb.eff.items():
            dofs = []
            for a in self.emb.act_ids[e["gacts"]]:
                if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                    dofs.append(int(m.jnt_dofadr[m.actuator_trnid[a, 0]]))
                elif m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON:
                    t = m.actuator_trnid[a, 0]
                    dofs += [int(m.jnt_dofadr[m.wrap_objid[w]]) for w in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t])]
            self.gdofs[k] = np.array(dofs, int)
        self.gdamp0 = {k: m.dof_damping[v].copy() for k, v in self.gdofs.items()}
        self.grip_k = {k: 1.0 for k in self.emb.eff}
        self._gi = 0
        self.ctx = None
        self.t = 0.0
        # measured: a free close takes ~2x GripperCfg.close_time (2F-85: 0.8 s vs 0.4 s) because of the linkage;
        # the skills wait close_time + 0.15 s after commanding the close, so double the WAIT (damping already baked)
        # (ROBOTS round 2: the gripper close is event-based, no close-time workaround needed any more)
        # park the hand(s) low and close to the base: the home TCP hovers over the station and the elbow hides parts
        # from the fixed tracker camera (which looks over the robot's shoulder); executed during the env's settle
        from physmem.embodiments.trajectory import BlendedTrajectory
        mpos, yaw = np.asarray(self.mount[0], float), float(self.mount[1])
        f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        for i, k in enumerate(self.effs):
            p, q = self.emb.tcp_pose(d, k)
            if self.spec.kind in ("arm",):
                park = mpos + 0.15 * f + np.array([0, 0, 0.2])
            elif self.spec.kind == "mobile_manipulator":
                park = p - 0.12 * f + np.array([0, 0, 0.12])   # pull the hand back along the heading and up
            elif self.spec.kind == "humanoid":
                # hands up above the bench top first, then forward over it (the stand pose holds them below the
                # top, behind the edge: a straight move to a target would cut the bench edge)
                ztop = 0.76
                try:
                    g = m.geom("bench_top").id
                    ztop = float(d.geom_xpos[g][2] + m.geom_size[g][2])
                except Exception:
                    pass
                up = np.array([p[0], p[1], ztop + 0.12])
                fwd = up + 0.18 * f
                if i == 0 and hasattr(self.ctrl, "root_z0"):
                    # work crouch: pelvis 6 cm lower (shoulders closer to the waist-high bench; less waist lean)
                    self.ctrl.root_z0 = float(self.ctrl.root_z0) - float(getattr(self, "g1_crouch", 0.06))
                    # work lean: pelvis pitched forward (hip flexion); the CoM task keeps the CoM over the feet. The
                    # waist alone (0.52 rad limit) left the hands 4-5 cm short of bench-top targets 0.5 m out
                    lean = float(getattr(self, "g1_lean", 0.0))   # 0.2 rad tested: worse (more unreachable)
                    if lean > 0 and hasattr(self.ctrl, "root_q0"):
                        lat = np.cross([0.0, 0.0, 1.0], f)
                        ql = np.zeros(4)
                        mujoco.mju_axisAngle2Quat(ql, lat / max(np.linalg.norm(lat), 1e-9), lean)
                        q0 = np.zeros(4)
                        mujoco.mju_mulQuat(q0, ql, np.asarray(self.ctrl.root_q0, float))
                        self.ctrl.root_q0 = q0
                self.ctrl.command(BlendedTrajectory(effector=k, waypoints=[(up, q), (fwd, q)], durations=[0.8, 0.6]))
                continue
            else:
                park = p + np.array([0, 0, 0.05])
            self.ctrl.command(BlendedTrajectory(effector=k, waypoints=[(park, q)], durations=[0.6]))
        # (no automatic drive-in any more: ENVS mounts the mobile base 0.25 m from the bench / in the aisle lane; driving
        #  0.35 m 'forward' at reset pushed TidyBot into the shelf -> ~100 contact events > 150 N in shelf_restock)
        self.base_explicit = False

    # ------------------------------------------------------------------ effort (cumulative meter of the controller)
    @property
    def effort(self):
        tot = self.ctrl.effort
        return EffortMeter(**{k: getattr(tot, k) - getattr(self._eff0, k) for k in tot.__dict__ if k != "peak_ee_speed"},
                           peak_ee_speed=tot.peak_ee_speed)

    @effort.setter
    def effort(self, _v):
        tot = self.ctrl.effort if hasattr(self, "ctrl") else EffortMeter()
        self._eff0 = EffortMeter(**dict(tot.__dict__))

    # ------------------------------------------------------------------ per step
    def regulate_grip(self):
        """full effort while the fingers travel, regulated effort once the pads touch (every 5th step)."""
        self._gi += 1
        if self._gi % 5:
            return
        for k, e in self.emb.eff.items():
            if self.grip_k[k] >= 1.0:
                continue
            _, n = self.emb.gripper_state(self.m, self.d, k)
            e["gforce"] = self.gforce0[k] * (self.grip_k[k] if n > 0 else 1.0)

    def step(self, dt):
        self.t += dt
        self.regulate_grip()
        self.ctrl.apply(self.m, self.d)

    # ------------------------------------------------------------------ proprio (actor-legal)
    def tcp(self, eff=None):
        p, q = self.emb.tcp_pose(self.d, eff or self.eff)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        return p, math.atan2(-R[0, 1], R[1, 1])       # probe convention: hand y (closing) = (-sin yaw, cos yaw)

    def joint_names(self):
        out = []
        for name, e in self.emb.eff.items():
            out += [f"{name}/j{i}" for i in range(len(e["qadr"]))]
        if self.emb.base is not None:
            out += ["base/x", "base/y", "base/th"]
        return out

    def joint_state(self):
        q, dq = [], []
        for e in self.emb.eff.values():
            q += list(self.d.qpos[e["qadr"]])
            dq += list(self.d.qvel[e["dofs"]])
        if self.emb.base is not None:
            q += list(self.d.qpos[self.emb.base["qadr"]])
            dq += list(self.d.qvel[self.emb.base["dofs"]])
        return q, dq

    def grip_width(self, eff=None):
        eff = eff or self.eff
        c, _ = self.emb.gripper_state(self.m, self.d, eff)
        return float(max(0.0, 1.0 - c) * self.emb.eff[eff]["cfg"].gripper.max_width)

    def wrench(self, eff=None):
        F, T, _ = self.emb.wrist_ft(self.m, self.d, eff or self.eff)
        return np.concatenate([F, T]) - self.tare

    def pad_contacts(self, eff=None):
        eff = eff or self.eff
        _, n = self.emb.gripper_state(self.m, self.d, eff)
        F, _, _ = self.emb.wrist_ft(self.m, self.d, eff)
        return float(n > 0), float(n > 1), 0.0, float(np.linalg.norm(F))


# ====================================================================== skills
class EmbodimentSkills:
    def __init__(self, robot: EmbodimentRobot):
        self.r = robot

    def _ctx(self, env):
        r = self.r
        if r.ctx is None or r.ctx.model is not env.m:
            from physmem.skills import SkillContext
            r.ctx = SkillContext(env.m, env.d, r.emb, on_step=lambda m, d: self._post(env),
                                 before_step=lambda m, d: (r.regulate_grip(), env.pre_physics()))
        return r.ctx

    @staticmethod
    def _post(env):
        env.robot.t += env.m.opt.timestep
        env.post_physics()

    def _est(self, env, eid):
        from physmem.skills import ObjectEstimate
        est = env.est_pose_avg(eid, window=0.3) or env.est_pose(eid)
        if est is None:
            return None
        pos, quat, yaw = est
        h = env.est_size(eid)
        info = env.scene.entities.get(eid)
        shape = "cylinder" if info is not None and info.shape == "cylinder" else "box"
        # tracker reports full orientation; skills only need yaw for upright parts
        return ObjectEstimate(pos=tuple(pos), quat=tuple(_yaw_quat(yaw)), half_size=tuple(h), shape=shape)

    def _set_grip_force(self, eff, grip_force):
        r = self.r
        # ROBOTS round 2: calibrated per-pad squeeze (GripperCfg.pad_per_drive); suction effectors ignore it
        got = r.emb.set_grip_force(eff, float(grip_force))
        r.grip_k[eff] = 1.0
        r.grip_cmd = ("force", float(got if got > 0 else grip_force))

    def _clutter_yaw(self, env, target, est, gcfg):
        """0 or pi/2 (closing across the narrow or the wide side): the one whose finger landing spots are farther from
        the other parts' ESTIMATED footprints (tracker poses + size estimates; actor-legal). 0 if not cluttered."""
        R = est.R
        hx, hy = est.half_size[0], est.half_size[1]
        yaw_o = math.atan2(R[1, 0], R[0, 0])
        narrow = (yaw_o, 2 * hx) if hx <= hy else (yaw_o + math.pi / 2, 2 * hy)
        wide = (narrow[0] + math.pi / 2, 2 * max(hx, hy))
        others = []
        for eid in env.scene.entities:
            if eid == target or eid not in env.scene.grip_pairs_objects:
                continue
            e2 = env.est_pose(eid)
            if e2 is None:
                continue
            h2 = env.est_size(eid)
            try:
                r2 = float(np.hypot(h2[0], h2[1]))
            except Exception:
                continue
            if np.linalg.norm(np.asarray(e2[0][:2]) - np.asarray(est.pos[:2])) < 0.15:
                others.append((np.asarray(e2[0][:2], float), r2))
        if not others:
            return 0.0

        def clearance(yc, w):
            if w > 0.9 * gcfg.max_width:
                return -1.0
            u = np.array([math.cos(yc), math.sin(yc)])
            spots = [np.asarray(est.pos[:2]) + sgn * (w / 2 + 0.012) * u for sgn in (-1, 1)]
            return min(float(np.linalg.norm(sp - c)) - r for sp in spots for c, r in others)
        c0, c1 = clearance(*narrow), clearance(*wide)
        return math.pi / 2 if c1 > c0 + 0.004 else 0.0

    def _base_approach(self, env, call, ctx):
        """mobile manipulator: before a skill, drive the holonomic base sideways (along the bench) so the arm base is in
        line with the target estimate / region centre; the forward (depth) coordinate is kept - the base already
        stands at the bench. Min-jerk base move through the controller; actor-legal (estimates, regions)."""
        r = self.r
        if getattr(r, "base_explicit", False):
            return          # an explicit move_base placed the base: respect it (no automatic re-alignment)
        tgt = None
        if call.target in env.scene.entities:
            e = env.est_pose(call.target)
            tgt = None if e is None else np.asarray(e[0][:2], float)
        if tgt is None and call.target:
            try:
                reg = env.region(call.target)
                tgt = np.asarray(reg["center"][:2], float) if reg else None
            except Exception:
                tgt = None
        if tgt is None:
            return
        mpos, yaw = np.asarray(r.mount[0][:2], float), float(r.mount[1])
        Rm = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        qb = r.d.qpos[r.emb.base["qadr"]].copy()
        want = Rm.T @ (tgt - mpos)                 # target in the mount frame (x forward, y lateral)
        lat = float(qb[1] + np.clip(want[1] - qb[1], -2.0, 2.0))   # around the CURRENT base pose (full aisle)
        # forward: close in only as far as needed (target depth - ~0.42 m comfortable arm reach), never back off, at
        # most 0.3 m beyond the mount pose (ENVS mounts the base 0.25 m from the bench edge)
        fwd = float(np.clip(want[0] - 0.42, qb[0], 0.3))
        if abs(lat - qb[1]) < 0.03 and abs(fwd - qb[0]) < 0.03:
            return
        tgt_q = np.array([fwd, lat, qb[2]])
        dur = max(0.8, 1.875 * max(abs(lat - qb[1]), abs(fwd - qb[0])) / 0.2)
        ctx.ctrl.command_base(tgt_q, dur)
        # bumper: stop (hold the current base pose) at the first robot contact with static station geometry > 120 N
        m, d = env.m, env.d
        bgeoms = getattr(r, "_base_geoms", None)
        if bgeoms is None:      # ANY robot geom (base, shoulder, arm) against STATIC station geometry
            bgeoms = r._base_geoms = np.array(sorted(r.robot_geoms), int)
        f6 = np.zeros(6)

        def bumped(c):
            if not d.ncon:
                return False
            g1, g2 = d.contact.geom1, d.contact.geom2
            for i in np.where(np.isin(g1, bgeoms) ^ np.isin(g2, bgeoms))[0]:
                other = g2[i] if g1[i] in bgeoms else g1[i]
                if other in r.robot_geoms or m.body_weldid[m.geom_bodyid[other]] != 0:
                    continue    # only static (welded-to-world) station geometry bumps; parts / held objects do not
                mujoco.mj_contactForce(m, d, int(i), f6)
                if f6[0] > float(os.environ.get("PHYSMEM_BUMPER_N", 120.0)):   # below the env's 150 N collision count
                    return True
            return False
        st = ctx.run(until=bumped, max_t=dur + 0.5)
        if st == "until":
            qh = d.qpos[r.emb.base["qadr"]].copy()
            ctx.ctrl.command_base(qh - np.array([0.01, 0.0, 0.0]) * np.sign(fwd - qb[0] + 1e-9), 0.3)   # back off 1 cm
            ctx.run(max_t=0.6)

    def run(self, env, call):
        from physmem.skills import run_skill
        r = self.r
        ctx = self._ctx(env)
        if call.effector is None and len(r.effs) > 1 and env.held is None and call.target in env.scene.entities:
            # bimanual / humanoid: use the arm whose TCP is nearest the (estimated) target; it keeps the part
            est = env.est_pose(call.target)
            if est is not None:
                r.eff = min(r.effs, key=lambda k: float(np.linalg.norm(r.emb.tcp_pose(env.d, k)[0][:2] - est[0][:2])))
        eff = call.effector or r.eff
        if r.emb.base is not None and call.skill not in ("move_base", "insert"):   # no base motion before mm-precision inserts
            self._base_approach(env, call, ctx)
        fn = getattr(self, "sk_" + call.skill, None)
        if fn is None:
            return SkillResult(False, "unsupported_skill", EffortMeter(), 0.0)
        try:
            res = fn(env, call, ctx, eff, run_skill)
        except Exception as ex:      # keep episodes alive; report the failure honestly
            res = SkillResult(False, f"error:{type(ex).__name__}", EffortMeter(), 0.0, dict(msg=str(ex)[:200]))
        return res

    # -------------------------------------------------------------- skills
    def sk_reach(self, env, call, ctx, eff, run_skill):
        p = dict(call.params)
        dwell = float(np.clip(p.pop("dwell", 0.0), 0, 10))
        if "yaw" in p:
            p["yaw"] = p["yaw"] + math.pi / 2          # probe hand yaw -> closing-axis yaw
        res = run_skill(ctx, SkillCall("reach", p, effector=eff), None)
        if dwell > 0:
            env.run(dwell)
        return res

    def sk_pick(self, env, call, ctx, eff, run_skill):
        p = call.params
        est = self._est(env, call.target)
        if est is None:
            return SkillResult(False, "not_observed", EffortMeter(), 0.0)
        pos = np.array(est.pos)
        yaw = math.atan2(2 * est.quat[0] * est.quat[3], 1 - 2 * est.quat[3] ** 2)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = p.get("dx", 0.0), p.get("dy", 0.0)
        pos[:2] += [c * dx - s * dy, s * dx + c * dy]
        v = np.array([p.get("vx", 0.0), p.get("vy", 0.0)])
        moving = np.linalg.norm(v) > 1e-4         # the call says the part rides the belt
        if moving:   # latest tracker packet, not the 0.3 s average (which lags a moving part)
            e_last = env.est_pose(call.target)
            if e_last is not None:
                pos = np.array(e_last[0], float)
                pos[:2] += [c * dx - s * dy, s * dx + c * dy]
        ve = env.est_velocity(call.target, window=1.5) if (moving and hasattr(env, "est_velocity")) else None
        if ve is not None and np.linalg.norm(np.asarray(ve)[:2]) > 5e-3:
            # belt speed from the actor's own pose events (1.5 s least squares), along the belt direction given by the
            # call (a free lateral component is tracker noise: 1-3 mm sigma -> ~2 cm/s over a short window)
            u = v / np.linalg.norm(v)
            try:   # belt axis from the station CAD (pick window yaw) if available: removes the call's lateral noise too
                reg = env.region("pick_window")
                if reg is not None:
                    u0 = np.array([math.cos(reg.get("yaw", 0.0)), math.sin(reg.get("yaw", 0.0))])
                    u = u0 if float(u0 @ u) >= 0 else -u0
            except Exception:
                pass
            v = u * float(np.asarray(ve, float)[:2] @ u)
        # moving targets: the ROBOTS pick plans the intercept and tracks the belt (vel + tracker age passed below)
        est.pos = tuple(pos)
        gcfg = self.r.emb.eff[eff]["cfg"].gripper
        info = env.scene.entities.get(call.target)
        wall_pinch = False
        if info is not None and getattr(info, "shape", "") == "tote" and gcfg.kind != "suction" and \
                2 * min(est.half_size[:2]) > 0.95 * gcfg.max_width:
            # a tote wider than the fingers open: pinch its NEAR long wall from above (CAD meta: wall t = 8 mm).
            # The estimate handed to pick becomes the wall (centre, thin width); the tote's bottom stays the support.
            hx, hy, hzt = est.half_size
            t = 0.008
            R2 = np.array([[c, -s], [s, c]])
            long_x = hx >= hy
            wall_n = R2[:, 1] if long_x else R2[:, 0]                       # wall normal (unit, horizontal)
            base = np.asarray(self.r.mount[0][:2], float)
            sgn = 1.0 if float((base - pos[:2]) @ wall_n) > 0 else -1.0      # the wall facing the robot
            off = (hy if long_x else hx) - t / 2
            pos[:2] = pos[:2] + sgn * off * wall_n
            est.pos = tuple(pos)
            est.half_size = ((hx, t / 2, hzt) if long_x else (t / 2, hy, hzt))
            wall_pinch = True
        hz = est.half_size[2]
        dyaw_n = self._clutter_yaw(env, call.target, est, gcfg) if (gcfg.kind == "parallel" and not wall_pinch) else 0.0
        q = dict(speed=p.get("approach_speed", 0.3), lift=p.get("lift_height", 0.10),
                 lift_speed=p.get("lift_speed", 0.2), yaw=p.get("dyaw", 0.0) + dyaw_n,
                 **({"max_acc": float(p["max_acc"])} if "max_acc" in p else {}))
        if moving:
            q.update(vel=(float(v[0]), float(v[1]), 0.0), age=float(env.est_age(call.target)), pre=0.05,
                     speed=max(0.4, p.get("approach_speed", 0.4)))
        if "depth" in p:
            q["depth"] = float(np.clip(hz * (1 + p["depth"]), 0.005, 2 * hz))
        if wall_pinch:
            q["depth"] = min(0.035, hz)   # pinch the rim band of the wall
        # re-grasp loop breaker: vary the grasp on repeated failures on the SAME part (other yaw, then deeper), and
        # decline (no motion) after 3 failures so the episode budget moves on to other parts
        fails = getattr(env, "_rb_pick_fail", None)
        if fails is None:
            fails = env._rb_pick_fail = {}
        k = fails.get(call.target, 0)
        if k >= 3 and not moving:
            return SkillResult(False, "skipped_after_failures", EffortMeter(), 0.0, dict(fails=k))
        if k == 1 and not wall_pinch:
            q["yaw"] = q.get("yaw", 0.0) + math.pi / 2
        elif k == 2 and not wall_pinch:
            q["depth"] = float(min(2 * hz, q.get("depth", hz) + 0.01))
        self._set_grip_force(eff, p.get("grip_force", 40.0))
        if self.r.emb.eff[eff]["cfg"].gripper.kind == "suction":
            self.r.emb.eff[eff]["suction_level"] = float(np.clip(p.get("suction", 1.0), 0.0, 1.0))
            self.r.suction_cmd = self.r.emb.eff[eff]["suction_level"]
        res = run_skill(ctx, SkillCall("pick", q, target=call.target, effector=eff), est)
        held = ctx.held.get(eff) if hasattr(ctx, "held") else None
        if not res.ok and res.status not in ("unreachable",):
            fails[call.target] = k + 1
        if res.ok and held is not None:
            env.held = call.target
            _, tyaw = self.r.tcp(eff)
            env._held_off = (float(held["offset"]), float(_wrap(yaw - tyaw)))
        else:
            self.r.grip_cmd = ("open", 0.0)
        return res

    def sk_place(self, env, call, ctx, eff, run_skill):
        p = call.params
        reg = env.region(call.target)
        cx, cy, _ = reg["center"]
        ry = reg.get("yaw", 0.0)
        c, s = math.cos(ry), math.sin(ry)
        dx, dy = p.get("dx", 0.0), p.get("dy", 0.0)
        q = dict(x=cx + c * dx - s * dy, y=cy + s * dx + c * dy, z=reg.get("support_z", reg["center"][2]),
                 speed=p.get("speed", 0.3), approach_speed=min(0.1, p.get("lower_speed", 0.06)),
                 **({"max_acc": float(p["max_acc"])} if "max_acc" in p else {}))
        if p.get("align", 0):
            sym = reg.get("sym", math.pi)
            _, tyaw = self.r.tcp(eff)
            want = ry + p.get("dyaw", 0.0)
            dyw = _wrap(want - (tyaw + env.held_yaw_offset()))
            dyw = (dyw + sym / 2) % sym - sym / 2
            q["yaw"] = tyaw + dyw + math.pi / 2
        res = run_skill(ctx, SkillCall("place", q, target=call.target, effector=eff), None)
        env.held = None
        self.r.grip_cmd = ("open", 0.0)
        return res

    def sk_push(self, env, call, ctx, eff, run_skill):
        p = call.params
        est = self._est(env, call.target)
        if est is None:
            return SkillResult(False, "not_observed", EffortMeter(), 0.0)
        # probe: theta = contact-point angle, phi = rotation of the inward normal; ROBOTS: theta = push direction
        q = dict(theta=float(p["theta"] + math.pi + p.get("phi", 0.0)), speed=p.get("speed", 0.2),
                 distance=p.get("distance", 0.05), height=p.get("height", 0.012))
        if "f_max" in p:      # force-limited push (wrist F/T opposing the push); comply or abort
            q.update(f_max=float(p["f_max"]), comply=bool(p.get("comply", False)))
        if self.r.spec.kind == "humanoid":
            # G1: push with the hand's natural (reachable) orientation, either hand; see humanoid.g1_push
            from physmem.embodiments.humanoid import g1_push
            from physmem.skills.base import SkillRun
            with SkillRun(ctx, "push") as run:
                st, log = g1_push(ctx, est, q["theta"], speed=q["speed"], distance=q["distance"])
                return run.result(st == "done", st, **log)
        return run_skill(ctx, SkillCall("push", q, target=call.target, effector=eff), est)

    def sk_sweep(self, env, call, ctx, eff, run_skill):
        p = call.params
        gc = self.r.emb.eff[eff]["cfg"].gripper
        a = math.atan2(p["y1"] - p["y0"], p["x1"] - p["x0"])
        # the scripted sweep is a LINE for the probe's flat blade (half width blade_half_w); a real gripper sweeps that
        # band with overlapping finger-paddle lanes (bounding box of the band; the skill lanes it across `direction`)
        bh = float(getattr(env, "probe_kw", {}).get("blade_half_w", 0.08))
        n = np.array([-math.sin(a), math.cos(a)])
        pts = np.array([[p["x0"], p["y0"]], [p["x1"], p["y1"]]])
        band = np.vstack([pts + n * bh, pts - n * bh])
        q = dict(x0=float(band[:, 0].min()), y0=float(band[:, 1].min()), x1=float(band[:, 0].max()),
                 y1=float(band[:, 1].max()), direction=a, surface_z=env.surface_z,
                 height=gc.tip + max(0.002, p.get("height", 0.002)), speed=p.get("speed", 0.25), margin=0.0,
                 overshoot=0.0, lane=0.024)
        return run_skill(ctx, SkillCall("sweep", q, effector=eff), None)

    def sk_insert(self, env, call, ctx, eff, run_skill):
        from physmem.skills import ObjectEstimate
        p = call.params
        reg = env.region(call.target)
        cx, cy, _ = reg["center"]
        hx, hy = cx + p.get("dx", 0.0), cy + p.get("dy", 0.0)
        if p.get("correct", 1) and env.held is not None:
            # tracker view of the held part vs the TCP (actor-legal): aim the PART, not the TCP, at the hole
            o = env.est_pose_avg(env.held, window=0.3)
            if o is not None:
                tcp, _ = self.r.tcp(eff)
                off = np.asarray(o[0][:2]) - tcp[:2]
                if np.all(np.abs(off) < 0.02):
                    hx, hy = hx - off[0], hy - off[1]
        est = ObjectEstimate(pos=(hx, hy, reg["support_z"]))
        yaw_q = {}
        if env.held is not None and "yaw" in reg:
            # align the held part with the socket (sym = the socket's symmetry): closing-axis yaw for the TCP
            sym = reg.get("sym", math.pi)
            _, tyaw = self.r.tcp(eff)
            dyw = _wrap(reg["yaw"] - (tyaw + env.held_yaw_offset()))
            dyw = (dyw + sym / 2) % sym - sym / 2
            yaw_q = dict(yaw=tyaw + dyw + math.pi / 2)
        # a real two-finger grasp leaves 1-3 mm of in-hand offset along the finger axis (the probe's is ~0): the spiral
        # must cover it, so the search radius is at least 4 mm (ROBOTS insert gate: 4 mm)
        q = dict(depth=0.9 * reg.get("depth", 0.02), force=max(p.get("down_force", 6.0), 6.0),
                 search_r=max(p.get("search_r", 0.002), 0.004), release=p.get("release", True), **yaw_q)
        res = run_skill(ctx, SkillCall("insert", q, target=call.target, effector=eff), est)
        if p.get("release", True) and (res.ok or True):
            if not res.ok:        # ROBOTS insert keeps holding on failure: let go like the probe does
                ctx.ctrl.set_gripper(eff, 0.0)
                ctx.wait(0.5)
                if hasattr(ctx, "held"):
                    ctx.held.pop(eff, None)
            env.held = None
            self.r.grip_cmd = ("open", 0.0)
        return res

    def _articulate(self, env, call, ctx, eff, run_skill):
        """Top-down pinch of the handle tab, then a compliant pull along the CAD handle path (drawer: straight,
        door: arc about the hinge), release, retreat - the probe's semantics, composed from the ROBOTS controller
        (ROBOTS' articulate approaches horizontally, which collides with the bench for these low handles)."""
        from physmem.embodiments.ik import down_quat
        from physmem.embodiments.trajectory import BlendedTrajectory
        from physmem.skills.base import SkillRun
        import physmem.skills.primitives as P
        p = call.params
        path = env.handle_path(call.target, p.get("travel", 0.2) * (1 if call.skill == "open" else -1), n=10)
        with SkillRun(ctx, call.skill) as run:
            if path is None:
                return run.result(False, "not_observed")
            pts, yaws = path
            gc = self.r.emb.eff[eff]["cfg"].gripper
            dz = 0.03 - 0.008                  # probe TCP sits 3 cm below the tab centre; grip 8 mm below centre
            P_ = [np.asarray(x, float) + np.array([0, 0, dz]) for x in pts]
            # wide grippers (2F-85 base/linkage ~ +-3.5 cm across the fingers) hit the panel when centred on the 6 cm
            # tab: grasp it nearer its outer tip, i.e. shift along the initial pull direction (outward)
            if (gc.link_h < 0.045 or self.r.spec.kind == "humanoid") and len(P_) >= 2:
                u = P_[1] - P_[0]
                u[2] = 0.0
                u = u / max(float(np.linalg.norm(u)), 1e-9)
                P_ = [x + p.get("tab_out", 0.03 if self.r.spec.kind == "humanoid" else 0.018) * u for x in P_]
            Q = [down_quat(P._near_yaw(ctx, eff, y + math.pi / 2)) for y in yaws]
            if self.r.spec.kind == "humanoid":
                # Dex3: the pinch closes between thumb and index/middle, PINCH_CENTRE below/ahead of the palm TCP
                from physmem.embodiments.humanoid import PINCH_CENTRE
                from physmem.embodiments.trajectory import quat2mat
                P_ = [x - quat2mat(qq) @ PINCH_CENTRE for x, qq in zip(P_, Q)]
            ctx.ctrl.set_mode("track", eff)
            st = P._move(ctx, eff, [P_[0] + np.array([0, 0, 0.12]), P_[0] + np.array([0, 0, 0.04])], 0.3, Q[0],
                         gripper=0.0)
            if st != "done":
                return run.result(False, st, phase="approach")
            ctx.ctrl.set_mode("guarded", eff, threshold=P._guard_n(ctx, p, 8.0), axis=(0, 0, -1))
            P._move(ctx, eff, [P_[0]], 0.06, Q[0])
            ctx.ctrl.set_mode("track", eff)
            self._set_grip_force(eff, p.get("grip_force", 40.0))
            P._close(ctx, eff, 1.0)
            c, n = ctx.emb.gripper_state(ctx.model, ctx.data, eff)
            if n == 0:
                P._move(ctx, eff, [P_[0] + np.array([0, 0, 0.1])], 0.3, Q[0], gripper=0.0)
                self.r.grip_cmd = ("open", 0.0)
                return run.result(False, "grasp_failed", closure=c)
            # stiff impedance (600 N/m lagged drawer friction: Panda stalled at 77-82 %) + 2 cm overshoot of the last
            # waypoint so the pull continues until the joint stop / the target
            # ROBOTS: pull in TRACK mode (task-space ID + bounded integral) - impedance (even 2500 N/m) left the Panda
            # at 56-68 % against 2-40 N drawer friction; the path is the CAD handle path, so stiff tracking is safe
            pull_mode = p.get("pull_mode", "track")   # humanoid impedance pull (stiffness 400, soft wrist) tested: no better
            if self.r.spec.kind == "humanoid" and pull_mode == "impedance":
                # G1: compliant pull - a stiffly tracked hand fights the door arc / drawer rail through the hooked
                # Dex3 fingers (70-100 N on the tab, beyond ENVS r5's 20-60 N crush limit)
                ctx.ctrl.set_mode("impedance", eff, stiffness=p.get("stiffness", 400.0), rot_stiffness=p.get("rot_stiffness", 2.0),
                                  f_cap=float(p.get("pull_fcap", 25.0)))
            elif pull_mode == "track":
                ctx.ctrl.set_mode("track", eff)
                if self.r.spec.kind == "humanoid":
                    # G1: hand-force cap during the pull. 60 N let the pull load reach ~80-100 N on the thumb/index
                    # (ENVS r5 counts robot contact force/2 on the tab against its hidden 20-60 N crush limit) -> 25 N
                    ctx.ctrl.set_mode("track", eff, f_cap=float(p.get("pull_fcap", 25.0)))
            else:
                ctx.ctrl.set_mode("impedance", eff, stiffness=p.get("stiffness", 2500.0), rot_stiffness=30.0)
            sp = p.get("speed", 0.1)
            if self.r.spec.kind == "humanoid":
                sp = min(sp, 0.06)       # Dex3 pinch (~20 N/side measured): slower pull = less damping force -> less slip
            if len(P_) >= 2:
                dlast = P_[-1] - P_[-2]
                P_[-1] = P_[-1] + 0.02 * dlast / max(float(np.linalg.norm(dlast)), 1e-6)
            L = [float(np.linalg.norm(P_[i + 1] - P_[i])) for i in range(len(P_) - 1)]
            durs = [max(0.05, 1.2 * l / max(sp, 1e-3)) for l in L]
            ctx.ctrl.command(BlendedTrajectory(effector=eff, waypoints=list(zip(P_[1:], Q[1:])), durations=durs))
            ctx.run(max_t=sum(durs) * 2 + 1.0, effector=eff)
            ctx.wait(0.4)        # fixed hold at the path end (lets a lagging joint catch up; not a convergence wait)
            c2, n2 = ctx.emb.gripper_state(ctx.model, ctx.data, eff)
            ctx.ctrl.set_mode("track", eff)
            here, hq = P._tcp(ctx, eff)
            ctx.ctrl.command(BlendedTrajectory(effector=eff, waypoints=[(here, hq)], durations=[0.2]), preempt=True)
            ctx.run(max_t=0.4, effector=eff)
            P._close(ctx, eff, 0.0)
            P._move(ctx, eff, [here + np.array([0, 0, 0.1])], 0.2, hq, gripper=0.0)
            self.r.grip_cmd = ("open", 0.0)
            held = n2 > 0
            return run.result(held, "done" if held else "handle_slipped", closure=c2)

    def _articulate_two_hand(self, env, call, ctx, eff, run_skill):
        """humanoid (G1 + Dex3): hand-over-hand. A 6 cm tab cannot take two Dex3 pinches at once, so the second hand is
        used sequentially: if the first hand's pull stops short (slip at ~20 N/side, reach, or the hand-force cap),
        the OTHER hand re-grasps the tab where it now is (tracked joint, actor-legal) and pulls the remaining travel;
        up to 3 alternations. Balance is kept by the whole-body QP throughout."""
        travel = float(call.params.get("travel", 0.2))
        q_start = env.est_joint(call.target)
        res, hand, tot = None, eff, EffortMeter()
        for k in range(3):
            q_now = env.est_joint(call.target)
            done_tr = 0.0 if (q_start is None or q_now is None) else abs(q_now - q_start)
            remaining = travel - done_tr
            if k > 0 and remaining < 0.1 * travel:
                break
            c2 = SkillCall(call.skill, dict(call.params, travel=remaining), target=call.target, effector=hand,
                           candidate_id=call.candidate_id)
            res = self._articulate(env, c2, ctx, hand, run_skill)
            for f in tot.__dict__:
                if f == "peak_ee_speed":
                    tot.peak_ee_speed = max(tot.peak_ee_speed, res.effort.peak_ee_speed)
                else:
                    setattr(tot, f, getattr(tot, f) + getattr(res.effort, f))
            if res.status not in ("done", "handle_slipped", "grasp_failed"):
                break
            others = [h for h in self.r.effs if h != hand]
            hand = others[0] if others else hand
        q_end = env.est_joint(call.target)
        moved = 0.0 if (q_start is None or q_end is None) else abs(q_end - q_start)
        ok = moved >= 0.85 * travel
        log = dict(res.log if res is not None else {}, moved=moved, hands_used=k + 1)
        return SkillResult(ok, "done" if ok else (res.status if res is not None else "not_observed"), tot,
                           sum([0.0]) if res is None else res.duration_s, log)

    def sk_move_base(self, env, call, ctx, eff, run_skill):
        """mobile base motion: params x, y, yaw (station = world frame), speed, yaw_rate, arm ('keep'|'tuck')."""
        self.r.base_explicit = True
        return run_skill(ctx, SkillCall("move_base", dict(call.params), effector=eff), None)

    def sk_open(self, env, call, ctx, eff, run_skill):
        if self.r.spec.kind == "humanoid":
            return self._articulate_two_hand(env, call, ctx, eff, run_skill)
        return self._articulate(env, call, ctx, eff, run_skill)

    def sk_close(self, env, call, ctx, eff, run_skill):
        if self.r.spec.kind == "humanoid":
            return self._articulate_two_hand(env, call, ctx, eff, run_skill)
        return self._articulate(env, call, ctx, eff, run_skill)
