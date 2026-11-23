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
                          "mobile_manipulator": "mobile_manipulator"}.get(self.spec.kind, "arm")
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
            pass    # station humanoid mounts stand 0.15 m behind the bench edge (G1 arm reach 0.55 m)
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
                    s.add_pair(geomname1=pg, geomname2=gname, condim=4, friction=[mu, mu, 0.005, 0.0001, 0.0001])
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
                park = p + np.array([0, 0, 0.4])          # home TCP is below bench height: lift before driving in
            else:
                park = p + np.array([0, 0, 0.05])
            self.ctrl.command(BlendedTrajectory(effector=k, waypoints=[(park, q)], durations=[0.6]))
        if self.emb.base is not None:
            # drive the holonomic base 0.35 m toward the station (base joints are in the mount frame, +x = forward)
            qb = d.qpos[self.emb.base["qadr"]].copy()
            self.ctrl.command_base(qb + np.array([0.35, 0.0, 0.0]), 1.5)

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
        if np.linalg.norm(v) > 1e-4:     # moving target: lead by tracker age + a nominal pick approach time
            lead = env.est_age(call.target) + float(p.get("lead_s", 2.5))
            pos[:2] += v * lead
        est.pos = tuple(pos)
        hz = est.half_size[2]
        q = dict(speed=p.get("approach_speed", 0.3), lift=p.get("lift_height", 0.10),
                 lift_speed=p.get("lift_speed", 0.2), yaw=p.get("dyaw", 0.0))
        if "depth" in p:
            q["depth"] = float(np.clip(hz * (1 + p["depth"]), 0.005, 2 * hz))
        self._set_grip_force(eff, p.get("grip_force", 40.0))
        if self.r.emb.eff[eff]["cfg"].gripper.kind == "suction":
            self.r.emb.eff[eff]["suction_level"] = float(np.clip(p.get("suction", 1.0), 0.0, 1.0))
            self.r.suction_cmd = self.r.emb.eff[eff]["suction_level"]
        res = run_skill(ctx, SkillCall("pick", q, target=call.target, effector=eff), est)
        held = ctx.held.get(eff) if hasattr(ctx, "held") else None
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
                 speed=p.get("speed", 0.3), approach_speed=min(0.1, p.get("lower_speed", 0.06)))
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
        return run_skill(ctx, SkillCall("push", q, target=call.target, effector=eff), est)

    def sk_sweep(self, env, call, ctx, eff, run_skill):
        p = call.params
        gc = self.r.emb.eff[eff]["cfg"].gripper
        a = math.atan2(p["y1"] - p["y0"], p["x1"] - p["x0"])
        q = dict(x0=p["x0"], y0=p["y0"], x1=p["x1"], y1=p["y1"], direction=a, surface_z=env.surface_z,
                 height=gc.tip + max(0.002, p.get("height", 0.002)), speed=p.get("speed", 0.25), margin=0.0,
                 overshoot=0.0)
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
        q = dict(depth=0.9 * reg.get("depth", 0.02), force=p.get("down_force", 6.0),
                 search_r=max(p.get("search_r", 0.002), 0.001), release=p.get("release", True))
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
            Q = [down_quat(P._near_yaw(ctx, eff, y + math.pi / 2)) for y in yaws]
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
            ctx.ctrl.set_mode("impedance", eff, stiffness=p.get("stiffness", 2500.0), rot_stiffness=30.0)
            sp = p.get("speed", 0.1)
            if len(P_) >= 2:
                dlast = P_[-1] - P_[-2]
                P_[-1] = P_[-1] + 0.02 * dlast / max(float(np.linalg.norm(dlast)), 1e-6)
            L = [float(np.linalg.norm(P_[i + 1] - P_[i])) for i in range(len(P_) - 1)]
            durs = [max(0.05, 1.2 * l / max(sp, 1e-3)) for l in L]
            ctx.ctrl.command(BlendedTrajectory(effector=eff, waypoints=list(zip(P_[1:], Q[1:])), durations=durs))
            ctx.run(max_t=sum(durs) * 2 + 1.0, effector=eff)
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

    def sk_open(self, env, call, ctx, eff, run_skill):
        return self._articulate(env, call, ctx, eff, run_skill)

    def sk_close(self, env, call, ctx, eff, run_skill):
        return self._articulate(env, call, ctx, eff, run_skill)
