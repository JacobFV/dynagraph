"""TEST-ONLY stub robot: a Cartesian gantry 'probe' (x, y, z slides + yaw hinge) with a tool head
(parallel gripper | suction pad | sweep blade). Driven ONLY by joint torques/forces from a PD +
min-jerk controller (no mocap, no welds, no pose writes). Exists so stations and task families can be
developed and validated before the real embodiments (src/physmem/embodiments, agent ROBOTS) are bound.
Never report probe results as robot results.

The probe also carries a reference implementation of every skill used by the task families
(pick, place, push, sweep, insert, open, close, reach). Real robots use src/physmem/skills via
envs.robots.EmbodimentAdapter.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..v2.api import EffortMeter, EmbodimentSpec, SkillResult

J = mujoco.mjtJoint
G = mujoco.mjtGeom

PROBE_SPEC = EmbodimentSpec(name="probe", kind="arm", effectors=("hand",), gripper="parallel", reach_m=1.0,
                            payload_kg=20.0, max_ee_speed=1.0, max_ee_force=250.0, dof=6, floating_base=False,
                            source="test-only stub, src/physmem/envs/probe.py")


PAD_SOLREF = (0.004, 1.0)
PAD_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)


def minjerk(t, T):
    s = np.clip(t / max(T, 1e-6), 0.0, 1.0)
    p = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    v = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / max(T, 1e-6)
    a = (60 * s - 180 * s ** 2 + 120 * s ** 3) / max(T, 1e-6) ** 2
    return p, v, a


def trapezoid(t, T, f):
    """unit trapezoid: accel over f*T, cruise, decel over f*T (normalized displacement 0..1)."""
    f = float(np.clip(f, 0.02, 0.5))
    vmax = 1.0 / (T * (1 - f))
    ta = f * T
    acc = vmax / ta
    t = float(np.clip(t, 0, T))
    if t < ta:
        return 0.5 * acc * t * t, acc * t, acc
    if t < T - ta:
        return 0.5 * acc * ta * ta + vmax * (t - ta), vmax, 0.0
    tr = T - t
    return 1.0 - 0.5 * acc * tr * tr, acc * tr, -acc


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ProbeRobot:
    """tool: 'parallel' | 'suction' | 'blade'. max_open = max half opening of the parallel gripper (m)."""

    LIM = np.array([150.0, 150.0, 250.0, 15.0])   # x y z [N], yaw [Nm]
    KP = np.array([4000.0, 4000.0, 6000.0, 60.0])
    KD = np.array([250.0, 250.0, 300.0, 3.0])

    mount_key = "gantry"

    def __init__(self, tool="parallel", max_open=0.06, grip_max=100.0, suction_max=150.0, blade_half_w=0.08,
                 pad_mu=0.8, finger_half=0.04, pad_solref=0.005):
        self.spec = PROBE_SPEC
        self.tool = tool
        self.max_open = max_open
        self.grip_max = grip_max
        self.suction_max = suction_max
        self.blade_half_w = blade_half_w
        self.pad_mu = pad_mu
        self.finger_half = finger_half
        self.pad_solref = pad_solref
        self.effort = EffortMeter()
        self.mount = None

    # ------------------------------------------------------------------ build
    def build(self, scene, mount):
        pos, yaw = mount
        self.mount = (np.asarray(pos, float), float(yaw))
        s = scene.spec
        base = s.worldbody.add_body(name="probe_base", pos=list(pos))
        bx = base.add_body(name="probe_x")
        bx.add_joint(name="probe_jx", type=J.mjJNT_SLIDE, axis=[1, 0, 0], range=[-1.2, 1.2], limited=True,
                     damping=2.0, armature=0.5)
        bx.add_geom(type=G.mjGEOM_SPHERE, size=[0.01, 0, 0], contype=0, conaffinity=0, rgba=[0, 0, 0, 0]).mass = 3.0
        by = bx.add_body(name="probe_y")
        by.add_joint(name="probe_jy", type=J.mjJNT_SLIDE, axis=[0, 1, 0], range=[-1.2, 1.2], limited=True,
                     damping=2.0, armature=0.5)
        by.add_geom(type=G.mjGEOM_SPHERE, size=[0.01, 0, 0], contype=0, conaffinity=0, rgba=[0, 0, 0, 0]).mass = 2.0
        bz = by.add_body(name="probe_z")
        bz.add_joint(name="probe_jz", type=J.mjJNT_SLIDE, axis=[0, 0, 1], range=[-1.6, 0.3], limited=True,
                     damping=2.0, armature=0.3)
        # vertical quill (visual only)
        bz.add_geom(type=G.mjGEOM_CYLINDER, size=[0.015, 0.25, 0], pos=[0, 0, 0.35], contype=0, conaffinity=0,
                    rgba=[0.9, 0.55, 0.1, 1]).mass = 1.0
        byaw = bz.add_body(name="probe_yaw")
        byaw.add_joint(name="probe_jyaw", type=J.mjJNT_HINGE, axis=[0, 0, 1], range=[-4.0, 4.0], limited=True,
                       damping=0.1, armature=0.02)
        byaw.add_geom(type=G.mjGEOM_SPHERE, size=[0.01, 0, 0], contype=0, conaffinity=0, rgba=[0, 0, 0, 0]).mass = 0.2
        hand = byaw.add_body(name="probe_hand", pos=[0, 0, 0])
        hand.add_site(name="probe_ft", pos=[0, 0, 0.09])
        hand.add_site(name="probe_tcp", pos=[0, 0, 0])
        palm = hand.add_geom(type=G.mjGEOM_BOX, size=[0.025, 0.025, 0.01], pos=[0, 0, 2 * self.finger_half + 0.01],
                             rgba=[0.9, 0.55, 0.1, 1], name="probe_palm", friction=[0.3, 0.005, 0.0001])
        palm.mass = 0.4
        self.pad_geoms = []
        if self.tool == "parallel":
            for side, sgn in (("l", 1), ("r", -1)):
                f = hand.add_body(name=f"probe_f{side}", pos=[0, 0, self.finger_half])
                # gearbox-like damping: free closing speed ~ F / 250 m/s (<= 0.4 m/s), no tunnelling through walls
                f.add_joint(name=f"probe_jf{side}", type=J.mjJNT_SLIDE, axis=[0, sgn, 0], range=[0.0, self.max_open],
                            limited=True, damping=250.0, armature=0.05)
                g = f.add_geom(type=G.mjGEOM_BOX, size=[0.012, 0.005, self.finger_half], pos=[0, sgn * 0.005, 0],
                               rgba=[0.2, 0.2, 0.22, 1], name=f"probe_pad{side}",
                               friction=[self.pad_mu, 0.01, 0.0001], condim=4, solref=[self.pad_solref, 1])
                g.mass = 0.08
                self.pad_geoms.append(g.name)
                a = s.add_actuator(name=f"probe_af{side}", target=f"probe_jf{side}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
                a.set_to_motor()
                a.ctrlrange = [-self.grip_max, self.grip_max]
                a.forcerange = [-self.grip_max, self.grip_max]
                a.forcelimited = True
                a.ctrllimited = True
            s.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="probe_jfr", name2="probe_jfl",
                           data=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif self.tool == "suction":
            g = hand.add_geom(type=G.mjGEOM_CYLINDER, size=[0.045, 0.03, 0], pos=[0, 0, 0.03],
                              rgba=[0.15, 0.15, 0.15, 1], name="probe_cup", friction=[1.0, 0.01, 0.0001], condim=4,
                              solref=[0.01, 1])
            g.mass = 0.2
            self.pad_geoms.append(g.name)
            a = s.add_actuator(name="probe_suction", target="probe_hand", trntype=mujoco.mjtTrn.mjTRN_BODY)
            a.set_to_adhesion(gain=self.suction_max)
            a.ctrlrange = [0, 1]
        elif self.tool == "blade":
            g = hand.add_geom(type=G.mjGEOM_BOX, size=[0.005, self.blade_half_w, 0.03], pos=[0, 0, 0.03],
                              rgba=[0.2, 0.2, 0.22, 1], name="probe_blade", friction=[0.3, 0.005, 0.0001])
            g.mass = 0.2
            self.pad_geoms.append(g.name)
        for b in (bx, by, bz, byaw, hand):
            b.gravcomp = 1.0
        for nm in ("x", "y", "z", "yaw"):
            a = s.add_actuator(name=f"probe_a{nm}", target=f"probe_j{nm}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_motor()
            lim = float(self.LIM["x y z yaw".split().index(nm)])
            a.ctrlrange = [-lim, lim]
            a.forcerange = [-lim, lim]
            a.ctrllimited = True
            a.forcelimited = True
        s.add_sensor(name="probe_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                     objname="probe_ft")
        s.add_sensor(name="probe_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                     objname="probe_ft")
        # explicit object<->pad friction pairs (hidden mu_grip per object)
        if self.tool in ("parallel", "suction"):
            for eid in scene.grip_pairs_objects:
                info = scene.entities[eid]
                mu = info.hidden.get("mu_grip", 0.8)
                for gname in info.geoms:
                    for pg in self.pad_geoms:
                        # pad pairs: stiff + hard impedance so a squeezed small/stiff part cannot be pushed through
                        # the 10 mm pad (default pair solref 0.02 let 40-100 N squeezes pass a 15 mm sphere);
                        # no margin, so released parts do not stick between the pads
                        s.add_pair(geomname1=pg, geomname2=gname, condim=4, friction=[mu, mu, 0.005, 0.0001, 0.0001],
                                   solref=list(PAD_SOLREF), solimp=list(PAD_SOLIMP))
                s.add_exclude(bodyname1=info.body, bodyname2="probe_hand")
                if self.tool == "parallel":
                    for side in "lr":
                        s.add_exclude(bodyname1=info.body, bodyname2=f"probe_f{side}")
            # palm still needs to touch objects: pair palm with object geoms at low friction
            for eid in scene.grip_pairs_objects:
                for gname in scene.entities[eid].geoms:
                    s.add_pair(geomname1="probe_palm", geomname2=gname, condim=3, friction=[0.3, 0.3, 0.005, 0.0001, 0.0001])

    def bind(self, model, data):
        self.m, self.d = model, data
        jn = ["probe_jx", "probe_jy", "probe_jz", "probe_jyaw"]
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in jn])
        self.vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in jn])
        self.act = np.array([model.actuator(f"probe_a{n}").id for n in ("x", "y", "z", "yaw")])
        self.fing = None
        if self.tool == "parallel":
            self.fing = dict(qadr=np.array([model.jnt_qposadr[model.joint(f"probe_jf{s}").id] for s in "lr"]),
                             vadr=np.array([model.jnt_dofadr[model.joint(f"probe_jf{s}").id] for s in "lr"]),
                             act=np.array([model.actuator(f"probe_af{s}").id for s in "lr"]))
        self.suction_act = model.actuator("probe_suction").id if self.tool == "suction" else None
        self.pad_ids = {model.geom(n).id for n in self.pad_geoms}
        self.palm_id = model.geom("probe_palm").id
        self.hand_body = model.body("probe_hand").id
        self.robot_bodies = {model.body(n).id for n in ("probe_base", "probe_x", "probe_y", "probe_z", "probe_yaw",
                                                        "probe_hand")}
        if self.tool == "parallel":
            self.robot_bodies |= {model.body("probe_fl").id, model.body("probe_fr").id}
        self.robot_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in self.robot_bodies}
        self.f_sens = model.sensor("probe_force").adr[0]
        self.t_sens = model.sensor("probe_torque").adr[0]
        self.mount_pos = self.mount[0]
        # controller state
        self.ref = self.q().copy()
        self.vref = np.zeros(4)
        self.aref = np.zeros(4)
        self.segments = []
        self.t_seg = 0.0
        self.kscale = np.ones(4)
        self.fz_mode = None          # None or downward force target (N)
        self.xy_offset_fn = None     # callable(t)-> xy offset added to ref (search patterns)
        self.grip_cmd = ("open", self.max_open)   # ("open", width) | ("force", F)
        self.suction_cmd = 0.0
        self.prev_tau = None
        self.prev_dq = None
        self.prev_acc = None
        self.tare = np.zeros(6)
        self.t = 0.0
        self.effort = EffortMeter()
        self._cmdlog = None
        self._cmd_i = 0

    # ------------------------------------------------------------------ proprio (encoders = actor-legal)
    def q(self):
        return self.d.qpos[self.qadr].copy()

    def dq(self):
        return self.d.qvel[self.vadr].copy()

    def tcp(self):
        """TCP world position from encoders (FK of a Cartesian gantry: mount + q)."""
        q = self.q()
        return self.mount_pos + q[:3], q[3]

    def q_for(self, pos, yaw):
        return np.array([pos[0] - self.mount_pos[0], pos[1] - self.mount_pos[1], pos[2] - self.mount_pos[2], yaw])

    def grip_width(self):
        if self.fing is None:
            return 0.0
        return float(self.d.qpos[self.fing["qadr"]].mean() * 2)

    def wrench(self):
        f = self.d.sensordata[self.f_sens:self.f_sens + 3]
        t = self.d.sensordata[self.t_sens:self.t_sens + 3]
        # site axes are hand axes (yaw only): rotate to world
        R = self.d.site_xmat[self.m.site("probe_ft").id].reshape(3, 3)
        return np.concatenate([R @ f, R @ t]) - self.tare

    def joint_names(self):
        n = ["jx", "jy", "jz", "jyaw"]
        if self.fing is not None:
            n += ["jfl", "jfr"]
        return n

    def joint_state(self):
        qs, dqs = list(self.q()), list(self.dq())
        if self.fing is not None:
            qs += list(self.d.qpos[self.fing["qadr"]])
            dqs += list(self.d.qvel[self.fing["vadr"]])
        return qs, dqs

    def pad_contacts(self):
        """(left, right, palm, |normal force| sum) from the robot's own tactile pads (actor-legal)."""
        d, m = self.d, self.m
        flags = [0.0, 0.0, 0.0]
        fsum = 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            gs = (c.geom1, c.geom2)
            for k, gname in enumerate(self.pad_geoms[:2] if self.tool == "parallel" else self.pad_geoms[:1]):
                gid = m.geom(gname).id
                if gid in gs:
                    flags[k] = 1.0
                    mujoco.mj_contactForce(m, d, i, f6)
                    fsum += abs(f6[0])
            if self.palm_id in gs:
                flags[2] = 1.0
        return flags[0], flags[1], flags[2], fsum

    # ------------------------------------------------------------------ trajectory API
    def command(self, waypoints, durations, profile="minjerk", ramp=0.25):
        """waypoints: list of (pos3, yaw) for the TCP; min-jerk (or trapezoid with ramp fraction) per
        segment, stop at each waypoint."""
        start = self.ref.copy()
        segs = []
        for (p, yaw), T in zip(waypoints, durations):
            goal = self.q_for(p, yaw)
            goal[3] = start[3] + wrap(goal[3] - start[3])
            segs.append((start.copy(), goal, max(float(T), 0.05), profile, ramp))
            start = goal
        self.segments = segs
        self.t_seg = 0.0

    def hold(self):
        self.segments = []
        self.vref[:] = 0
        self.aref[:] = 0

    def done(self):
        return not self.segments

    def set_grip(self, mode, value):
        self.grip_cmd = (mode, float(value))

    def set_suction(self, level):
        self.suction_cmd = float(np.clip(level, 0, 1))

    # ------------------------------------------------------------------ control tick
    def step(self, dt):
        m, d = self.m, self.d
        self.t += dt
        if self._cmdlog is not None:
            self._cmd_i += 1
            if self._cmd_i % max(1, int(round(0.02 / dt))) == 0:       # 50 Hz commanded task-space reference
                self._cmdlog.append((self.t, *(self.ref[:3] + self.mount_pos), *self.vref[:3]))
        if self.segments:
            a, b, T, prof, ramp = self.segments[0]
            self.t_seg += dt
            p, v, acc = minjerk(self.t_seg, T) if prof == "minjerk" else trapezoid(self.t_seg, T, ramp)
            self.ref = a + (b - a) * p
            self.vref = (b - a) * v
            self.aref = (b - a) * acc
            if self.t_seg >= T:
                self.segments.pop(0)
                self.t_seg = 0.0
                self.ref = b.copy()
                self.vref[:] = 0
                self.aref[:] = 0
        q, dq = self.q(), self.dq()
        ref = self.ref.copy()
        if self.xy_offset_fn is not None:
            off = self.xy_offset_fn(self.t)
            ref[:len(off)] += off
        e = ref - q
        e[3] = wrap(e[3])
        mass_ff = np.array([5.6, 2.6, 1.0, 0.0])
        tau = self.KP * self.kscale * e + self.KD * np.sqrt(self.kscale) * (self.vref - dq) + mass_ff * self.aref
        if self.fz_mode is not None:
            tau[2] = -self.fz_mode - 80.0 * dq[2]
        tau = np.clip(tau, -self.LIM, self.LIM)
        d.ctrl[self.act] = tau
        taus = [tau / self.LIM]
        if self.fing is not None:
            mode, val = self.grip_cmd
            if mode == "force":
                gf = -np.clip(val, 0, self.grip_max)
                u = np.array([gf, gf])
            else:
                qf = d.qpos[self.fing["qadr"]]
                dqf = d.qvel[self.fing["vadr"]]
                u = np.clip(3000.0 * (val / 2 - qf), -self.grip_max, self.grip_max)
            d.ctrl[self.fing["act"]] = u
            taus.append(u / self.grip_max)
        if self.suction_act is not None:
            d.ctrl[self.suction_act] = self.suction_cmd
            taus.append(np.array([self.suction_cmd]))
        tn = np.concatenate(taus)
        ef = self.effort
        ef.torque_sq += float((tn ** 2).sum() * dt)
        if self.prev_tau is not None and len(self.prev_tau) == len(tn):
            ef.torque_rate_sq += float((((tn - self.prev_tau) / dt) ** 2).sum() * dt)
        self.prev_tau = tn
        if self.prev_dq is not None:
            acc = (dq - self.prev_dq) / dt
            if self.prev_acc is not None:
                jerk = (acc - self.prev_acc) / dt
                ef.joint_jerk_sq += float(np.minimum(jerk[:3] ** 2, 1e8).sum() * dt)
            self.prev_acc = acc
        self.prev_dq = dq
        ef.energy_j += float(np.abs(tau * dq).sum() * dt)
        if np.any(np.abs(tau) >= self.LIM * 0.999):
            ef.time_at_limit_s += dt
        ef.peak_ee_speed = max(ef.peak_ee_speed, float(np.linalg.norm(dq[:3])))


# ====================================================================== probe skills
class ProbeSkills:
    """Reference skill implementations on the probe. Every skill reads only actor-legal information
    (env.est_pose / env.est_size from emitted sensor events, CAD regions, own encoders)."""

    CARRY_CLEAR = 0.10

    def __init__(self, robot: ProbeRobot):
        self.r = robot

    def run(self, env, call):
        fn = getattr(self, "sk_" + call.skill, None)
        if fn is None:
            return SkillResult(False, "unsupported_skill", EffortMeter(), 0.0)
        r = self.r
        e0 = EffortMeter(**r.effort.__dict__)
        t0 = env.t
        r._cmdlog, r._cmd_i = [], 0
        status, log = fn(env, call)
        c = np.array(r._cmdlog, float).reshape(-1, 7)
        r._cmdlog = None
        log = dict(log, skill=call.skill, params=dict(call.params), target=call.target,
                   cmd=(c[:, 0], c[:, 1:4], c[:, 4:7]))
        eff = EffortMeter(**{k: getattr(r.effort, k) - getattr(e0, k) for k in e0.__dict__})
        eff.peak_ee_speed = r.effort.peak_ee_speed
        r.effort.peak_ee_speed = 0.0
        return SkillResult(status == "done", status, eff, env.t - t0, log)

    # -------------------------------------------------------------- helpers
    def _move(self, env, pos, yaw, speed=0.4, min_T=0.3, timeout=None):
        r = self.r
        # workspace clamp (joint ranges) + duration cap: a bad estimate can never make a skill run for minutes
        lo = r.mount_pos + np.array([-1.15, -1.15, -1.55])
        hi = r.mount_pos + np.array([1.15, 1.15, 0.25])
        pos = np.clip(np.nan_to_num(np.asarray(pos, float), nan=0.0), lo, hi)
        speed = max(float(speed), 0.02)
        p0, y0 = r.ref[:3] + r.mount_pos, r.ref[3]
        dist = np.linalg.norm(np.asarray(pos) - p0)
        T = min(8.0, max(min_T, 1.875 * dist / max(speed, 1e-3), abs(wrap(yaw - y0)) / 2.0))
        r.command([(pos, yaw)], [T])
        env.run(T + 0.05 if timeout is None else timeout, stop=r.done)
        return T

    def _via(self, env, pts, speed=0.4):
        for p, y in pts:
            self._move(env, p, y, speed)

    def wait_closed(self, env, t_min=0.2, t_max=1.5):
        """wait until the fingers stop moving (contact or fully closed), from the gripper encoder."""
        r = self.r
        env.run(t_min)
        w0 = r.grip_width()
        t = t_min
        while t < t_max:
            env.run(0.05)
            t += 0.05
            w1 = r.grip_width()
            if abs(w1 - w0) < 3e-4:
                break
            w0 = w1
        env.run(0.1)

    def _safe_z(self, env):
        return env.safe_z

    def _held_ok(self, env, min_width=0.004):
        r = self.r
        if r.tool == "parallel":
            lf, rf, _, _ = r.pad_contacts()
            return r.grip_width() > min_width and (lf + rf) >= 1
        if r.tool == "suction":
            lf, _, _, _ = r.pad_contacts()
            return lf > 0
        return False

    # -------------------------------------------------------------- reach
    def sk_reach(self, env, call):
        """params: x, y, z, yaw, speed, dwell (s to wait after arriving: lets the world evolve, e.g. a conveyor)."""
        p = call.params
        self._move(env, (p["x"], p["y"], p["z"]), p.get("yaw", 0.0), p.get("speed", 0.4))
        env.run(float(np.clip(p.get("dwell", 0.0), 0.0, 10.0)))
        return "done", {}

    @staticmethod
    def _T(dist, speed, dyaw=0.0, min_T=0.3):
        return max(min_T, 1.875 * dist / max(speed, 1e-3), abs(dyaw) / 2.0)

    # -------------------------------------------------------------- pick
    def sk_pick(self, env, call):
        """params: dx, dy (grasp offset in object frame, m), dyaw (rad), grip_force (N) | suction (0..1),
        lift_speed (m/s), depth (fraction of object half height below its centre for the TCP, parallel),
        lift_height (m), approach_speed (m/s), vx, vy (m/s: ESTIMATED target velocity -> intercept a moving
        part; the hand matches that velocity while closing), close_yaw (rad, world yaw of the hand: pads close
        along (-sin, cos); overrides the narrow-axis choice, e.g. to keep the fingers off a bin wall)."""
        r, p = self.r, call.params
        est = env.est_pose(call.target)
        if est is None:
            return "not_observed", {}
        pos, yaw = est[0].copy(), est[2]
        half = env.est_size(call.target)
        if pos[2] < 0.03 or not np.all(np.isfinite(pos)):
            return "unreachable", dict(reason="target on the floor")
        v = np.array([p.get("vx", 0.0), p.get("vy", 0.0)])
        moving = float(np.linalg.norm(v)) > 1e-4
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = p.get("dx", 0.0), p.get("dy", 0.0)
        gx, gy = pos[0] + c * dx - s * dy, pos[1] + s * dx + c * dy
        gyaw = yaw + p.get("dyaw", 0.0)
        if r.tool == "parallel":
            if moving or "close_yaw" in p:
                # moving: hand x along the motion (the part slides between the pads); close_yaw: explicit hand yaw
                gyaw = math.atan2(v[1], v[0]) if moving else float(p["close_yaw"])
                n = np.array([-math.sin(gyaw), math.cos(gyaw)])
                cc, ss = math.cos(-yaw), math.sin(-yaw)
                span = abs(cc * n[0] - ss * n[1]) * half[0] + abs(ss * n[0] + cc * n[1]) * half[1]
            else:
                # close across the object's narrower horizontal axis: pads move along hand y
                if half[0] < half[1] and abs(p.get("dyaw", 0.0)) < 1e-9:
                    gyaw = yaw + math.pi / 2
                span = None
            gyaw = wrap(gyaw)
            if abs(gyaw) > math.pi / 2:      # keep the yaw joint near zero (symmetric gripper)
                gyaw = wrap(gyaw + math.pi)
            if span is None:
                span = half[1] if abs(math.cos(gyaw - yaw)) > 0.7 else half[0]
            if span + 0.01 > r.max_open:
                return "too_wide", {}
            z_tcp = pos[2] - p.get("depth", 0.6) * half[2]
            z_tcp = max(z_tcp, env.support_z(call.target) + 0.004)
            r.set_grip("open", min(2 * span + 0.03, 2 * r.max_open))
        else:   # suction from the top
            gyaw = wrap(yaw)
            z_tcp = pos[2] + half[2] - 0.004
        sz = self._safe_z(env)
        top_z = max(sz, z_tcp + 0.12)
        sp = p.get("approach_speed", 0.5)
        if moving:
            tcp, yaw_now = r.tcp()
            T1 = self._T(np.linalg.norm(np.subtract((gx, gy, top_z), tcp)), sp, wrap(gyaw - yaw_now)) + 0.05
            T2 = self._T(top_z - z_tcp - 0.03, sp) + 0.05
            T3 = self._T(0.03, 0.1) + 0.05
            lead = env.est_age(call.target) + T1 + T2 + T3
            gx, gy = gx + v[0] * lead, gy + v[1] * lead
        self._move(env, (gx, gy, top_z), gyaw, sp)
        self._move(env, (gx, gy, z_tcp + 0.03), gyaw, sp)
        if moving:    # re-aim from a fresh tracker estimate before the last 3 cm (closed-loop intercept)
            est2 = env.est_pose(call.target)
            if est2 is not None:
                tcp_now = np.asarray(r.tcp()[0])
                bx, by = est2[0][0] + c * dx - s * dy, est2[0][1] + s * dx + c * dy
                lead2 = env.est_age(call.target) + 0.7
                for _ in range(4):     # fixed point: the descent duration depends on the re-aim distance
                    ax, ay = bx + v[0] * lead2, by + v[1] * lead2
                    T3 = self._T(float(np.linalg.norm([ax - tcp_now[0], ay - tcp_now[1], 0.03])), 0.15)
                    lead2 = env.est_age(call.target) + T3 + 0.1
                gx, gy = bx + v[0] * lead2, by + v[1] * lead2
        self._move(env, (gx, gy, z_tcp), gyaw, 0.15 if moving else 0.1)
        if r.tool == "parallel":
            r.set_grip("force", p.get("grip_force", 40.0))
            if moving:     # ride along with the part while the fingers close
                gx, gy = gx + v[0] * 0.4, gy + v[1] * 0.4
                r.command([((gx, gy, z_tcp), gyaw)], [0.4], profile="trap", ramp=0.05)
                env.run(0.4)
            else:
                self.wait_closed(env)
        else:
            r.set_suction(p.get("suction", 0.8))
            # press slightly
            self._move(env, (gx, gy, z_tcp - 0.006), gyaw, 0.05)
            env.run(0.2)
        held_before = self._held_ok(env)
        lift = p.get("lift_height", 0.15)
        zl = max(top_z, z_tcp + lift)
        if moving:
            Tl = self._T(zl - z_tcp, p.get("lift_speed", 0.3))
            r.command([((gx + v[0] * Tl * 0.3, gy + v[1] * Tl * 0.3, zl), gyaw)], [Tl])
            env.run(Tl + 0.05, stop=r.done)
        else:
            self._move(env, (gx, gy, zl), gyaw, p.get("lift_speed", 0.3))
        env.run(0.2)
        held = self._held_ok(env)
        if not held:
            r.set_grip("open", 2 * r.max_open) if r.tool == "parallel" else r.set_suction(0)
            return ("missed" if not held_before else "slipped"), dict(held_before=held_before)
        env.held = call.target
        env._held_off = (float(z_tcp - (pos[2] - half[2])), float(wrap(yaw - gyaw)))
        return "done", dict(width=r.grip_width())

    # -------------------------------------------------------------- place
    def sk_place(self, env, call):
        """target = region name. params: dx, dy (in region frame), dyaw, release_h (m above the support for
        the object's bottom), speed (m/s), lower_speed, align (0/1: put the OBJECT yaw at region yaw + dyaw,
        else keep the carried yaw + dyaw), correct (0/1: use the tracker's view of the held object during the
        carry to cancel the in-hand offset)."""
        r, p = self.r, call.params
        reg = env.region(call.target)
        cx, cy, cz = reg["center"]
        ryaw = reg.get("yaw", 0.0)
        c, s = math.cos(ryaw), math.sin(ryaw)
        dx, dy = p.get("dx", 0.0), p.get("dy", 0.0)
        x, y = cx + c * dx - s * dy, cy + s * dx + c * dy
        tcp, yaw_now = r.tcp()
        held = env.held
        off = env.held_offset_z()          # TCP z - object bottom z, actor estimate at pick time
        z_rel = reg.get("support_z", cz) + p.get("release_h", 0.01) + off
        sym = reg.get("sym", math.pi)
        if p.get("align", 0):
            obj_yaw = yaw_now + env.held_yaw_offset()
            want = ryaw + p.get("dyaw", 0.0)
            dyaw = wrap(want - obj_yaw)
            dyaw = (dyaw + sym / 2) % sym - sym / 2
            yaw = wrap(yaw_now + dyaw)
        else:
            yaw = wrap(yaw_now + p.get("dyaw", 0.0))
        sz = max(self._safe_z(env), z_rel + 0.08)
        sp = p.get("speed", 0.4)
        self._move(env, (tcp[0], tcp[1], sz), yaw_now, sp)
        self._move(env, (x, y, sz), yaw, sp)
        self._move(env, (x, y, z_rel + 0.04), yaw, sp)
        if p.get("correct", 0) and held is not None:
            env.run(0.35)
            est = env.est_pose_avg(held, window=0.3)
            if est is not None:
                tcp2, yaw2 = r.tcp()
                ox, oy = est[0][0] - tcp2[0], est[0][1] - tcp2[1]      # object centre relative to TCP (world)
                x2, y2 = x - ox, y - oy
                if p.get("align", 0):
                    want = ryaw + p.get("dyaw", 0.0)
                    dyw = wrap(want - est[2])
                    dyw = (dyw + sym / 2) % sym - sym / 2
                    # rotating the hand by dyw about the TCP also rotates the offset
                    cc, ss = math.cos(dyw), math.sin(dyw)
                    rx, ry = cc * ox - ss * oy, ss * ox + cc * oy
                    x2, y2 = x - rx, y - ry
                    yaw = wrap(yaw2 + dyw)
                if abs(x2 - x) < 0.03 and abs(y2 - y) < 0.03:
                    x, y = x2, y2
                self._move(env, (x, y, z_rel + 0.04), yaw, 0.1)
        self._move(env, (x, y, z_rel), yaw, p.get("lower_speed", 0.15))
        env.run(0.15)
        if r.tool == "parallel":
            r.set_grip("open", min(2 * r.max_open, r.grip_width() + 0.04))
        else:
            r.set_suction(0.0)
        env.run(0.3)
        env.held = None
        self._move(env, (x, y, sz), yaw, 0.5)
        return "done", dict(held=held)

    # -------------------------------------------------------------- push (object frame, like v1 w2)
    def sk_push(self, env, call):
        """params: theta (contact point angle on the object boundary, object frame), phi (push direction
        relative to the inward normal), speed (m/s), distance (m), height (TCP above support)."""
        r, p = self.r, call.params
        est = env.est_pose(call.target)
        if est is None:
            return "not_observed", {}
        pos, _, yaw = est
        half = env.est_size(call.target)
        th, ph = p["theta"], p.get("phi", 0.0)
        # ray from centre to boundary of the box footprint
        d_obj = np.array([math.cos(th), math.sin(th)])
        tb = min(half[0] / max(abs(d_obj[0]), 1e-6), half[1] / max(abs(d_obj[1]), 1e-6))
        c, s = math.cos(yaw), math.sin(yaw)
        Rm = np.array([[c, -s], [s, c]])
        bpt = pos[:2] + Rm @ (d_obj * tb)
        n_in = -(Rm @ d_obj)
        cph, sph = math.cos(ph), math.sin(ph)
        u = np.array([cph * n_in[0] - sph * n_in[1], sph * n_in[0] + cph * n_in[1]])
        tool_r = 0.015
        start = bpt - u * (tool_r + 0.02)
        z = env.support_z(call.target) + p.get("height", 0.012)
        sz = self._safe_z(env)
        pyaw = math.atan2(u[1], u[0])   # pads face the push direction (gripper closed = paddle)
        if r.tool == "parallel":
            r.set_grip("open", 0.004)
        self._move(env, (start[0], start[1], sz), 0.0, 0.5)
        self._move(env, (start[0], start[1], z), 0.0, 0.3)
        end = start + u * (p.get("distance", 0.05) + 0.02)
        T = max(0.1, p.get("distance", 0.05) / max(p.get("speed", 0.1), 1e-3))
        r.command([((end[0], end[1], z), 0.0)], [T * 1.875 / 1.0])
        env.run(T * 1.875 + 0.05, stop=r.done)
        back = end - u * 0.03
        self._move(env, (back[0], back[1], z), 0.0, 0.3)
        self._move(env, (back[0], back[1], sz), 0.0, 0.5)
        return "done", {}

    # -------------------------------------------------------------- sweep (bulk)
    def sk_sweep(self, env, call):
        """params: x0, y0, x1, y1 (world), speed (m/s), height (blade bottom above support), lift (bool)."""
        r, p = self.r, call.params
        x0, y0, x1, y1 = p["x0"], p["y0"], p["x1"], p["y1"]
        u = np.array([x1 - x0, y1 - y0])
        L = float(np.linalg.norm(u))
        yaw = math.atan2(u[1], u[0])
        z = env.surface_z + p.get("height", 0.003)
        sz = self._safe_z(env)
        self._move(env, (x0, y0, sz), yaw, 0.6)
        self._move(env, (x0, y0, z), yaw, 0.3)
        v = max(p.get("speed", 0.2), 1e-3)
        ts = float(np.clip(p.get("stop_time", 0.3), 0.03, 2.0))   # accel / decel ramp time
        T = max(L / v + ts, 2.05 * ts)
        r.command([((x1, y1, z), yaw)], [T], profile="trap", ramp=ts / T)
        env.run(T + 0.05, stop=r.done)
        env.run(p.get("dwell", 0.0))
        ub = u / max(L, 1e-6)
        self._move(env, (x1 - 0.012 * ub[0], y1 - 0.012 * ub[1], z), yaw, 0.1)
        self._move(env, (x1 - 0.012 * ub[0], y1 - 0.012 * ub[1], sz), yaw, 0.3)
        return "done", dict(length=L)

    # -------------------------------------------------------------- insert (fine)
    def sk_insert(self, env, call):
        """Held object into a CAD hole region. params: dx, dy (aim offset, m), dyaw, down_force (N),
        search_r (m, spiral radius; 0 = straight), search_hz (spiral revs/s), compliance (xy stiffness scale
        0.02..1), speed (approach m/s), timeout (s), release (bool)."""
        r, p = self.r, call.params
        reg = env.region(call.target)
        cx, cy, cz = reg["center"]
        x, y = cx + p.get("dx", 0.0), cy + p.get("dy", 0.0)
        sym = reg.get("sym", math.pi)
        _, yaw_now = r.tcp()
        dyw = wrap(reg.get("yaw", 0.0) + p.get("dyaw", 0.0) - (yaw_now + env.held_yaw_offset()))
        dyw = (dyw + sym / 2) % sym - sym / 2
        yaw = wrap(yaw_now + dyw)
        off = env.held_offset_z()
        top = reg["support_z"]                      # hole mouth height
        z_above = top + off + 0.015
        sz = max(self._safe_z(env), z_above + 0.06)
        tcp, _ = r.tcp()
        self._move(env, (tcp[0], tcp[1], sz), yaw, p.get("speed", 0.3))
        self._move(env, (x, y, sz), yaw, p.get("speed", 0.3))
        self._move(env, (x, y, z_above + 0.03), yaw, p.get("speed", 0.3))
        if p.get("correct", 1) and env.held is not None:
            env.run(0.35)
            est = env.est_pose_avg(env.held, window=0.3)
            if est is not None:
                tcp2, yaw2 = r.tcp()
                ox, oy = est[0][0] - tcp2[0], est[0][1] - tcp2[1]
                if abs(ox) < 0.02 and abs(oy) < 0.02:
                    x, y = x - ox, y - oy
                dyw = wrap(reg.get("yaw", 0.0) + p.get("dyaw", 0.0) - est[2])
                dyw = (dyw + sym / 2) % sym - sym / 2
                if abs(dyw) < 0.3:
                    yaw = wrap(yaw2 + dyw)
        self._move(env, (x, y, z_above), yaw, 0.1)
        # compliant force-controlled descent with spiral search
        r.kscale = np.array([p.get("compliance", 0.1), p.get("compliance", 0.1), 1.0, 1.0])
        r.fz_mode = p.get("down_force", 15.0)
        R, hz = p.get("search_r", 0.0), p.get("search_hz", 1.0)
        wig = p.get("wiggle", 0.0)          # yaw wiggle amplitude (rad) at 1.7x the spiral frequency
        t0 = r.t
        if R > 0 or wig > 0:
            def spiral(t, t0=t0):
                tt = t - t0
                rr = R * min(1.0, tt * hz / 3.0)
                return np.array([rr * math.cos(2 * math.pi * hz * tt), rr * math.sin(2 * math.pi * hz * tt), 0.0,
                                 wig * math.sin(2 * math.pi * 1.7 * hz * tt)])
            r.xy_offset_fn = spiral
        depth_goal = reg.get("depth", 0.02)
        target_tcp_z = top + off - depth_goal * 0.9

        def reached():
            return r.tcp()[0][2] <= target_tcp_z
        env.run(p.get("timeout", 4.0), stop=reached)
        ok = reached()
        r.xy_offset_fn = None
        r.ref = r.q().copy()
        r.fz_mode = None
        r.kscale = np.ones(4)
        r.ref[2] = r.q()[2]
        if p.get("release", True) and (ok or p.get("release_on_fail", 1)):
            if r.tool == "parallel":
                r.set_grip("open", min(2 * r.max_open, r.grip_width() + 0.03))
            else:
                r.set_suction(0)
            env.run(0.25)
            env.held = None
            tcp, yy = r.tcp()
            self._move(env, (tcp[0], tcp[1], sz), yy, 0.2)
        elif not ok:      # keep holding: back off 1 cm so the next insert can search again
            tcp, yy = r.tcp()
            self._move(env, (tcp[0], tcp[1], tcp[2] + 0.01), yy, 0.05)
        return ("done" if ok else "jammed"), dict(depth_reached=ok)

    # -------------------------------------------------------------- open / close articulated
    def _articulate(self, env, call, sign):
        """Grasp the handle (CAD pose of the link from its tracked joint state), move along the joint's
        path (slide: straight; hinge: arc about the CAD hinge axis), release. params: grip_force (N),
        travel (m or rad, magnitude), speed (m/s at handle), regrasp (bool)."""
        r, p = self.r, call.params
        h = env.handle_path(call.target, p.get("travel", 0.2) * sign, n=12)
        if h is None:
            return "not_observed", {}
        pts, yaws = h
        sz = self._safe_z(env)
        r.set_grip("open", 0.034)
        x0, y0, z0 = pts[0]
        self._move(env, (x0, y0, max(sz, z0 + 0.12)), yaws[0], 0.5)
        self._move(env, (x0, y0, z0), yaws[0], 0.2)
        r.set_grip("force", p.get("grip_force", 40.0))
        self.wait_closed(env)
        sp = p.get("speed", 0.1)
        seg_len = [np.linalg.norm(np.subtract(pts[i + 1], pts[i])) for i in range(len(pts) - 1)]
        yr = p.get("yaw_rate", 1.0)        # rad/s (twist-type joints rotate the hand in place)
        Ts = [max(0.05, L / max(sp, 1e-3) * 1.2, abs(wrap(yaws[i + 1] - yaws[i])) / max(yr, 1e-3))
              for i, L in enumerate(seg_len)]
        r.kscale = np.array([0.35, 0.35, 1.0, 0.5])   # compliant so the arc errors do not fight the joint
        r.command([(pts[i + 1], yaws[i + 1]) for i in range(len(pts) - 1)], Ts)
        env.run(sum(Ts) + 0.3, stop=r.done)
        held = self._held_ok(env, min_width=0.008)
        r.set_grip("open", 0.07)
        env.run(0.25)
        r.kscale = np.ones(4)
        r.ref = r.q().copy()
        tcp, yy = r.tcp()
        back = np.asarray(tcp)
        self._move(env, (back[0], back[1], max(sz, back[2] + 0.1)), yy, 0.3)
        return ("done" if held else "handle_slipped"), dict(held_to_end=held)

    def sk_open(self, env, call):
        return self._articulate(env, call, +1)

    def sk_close(self, env, call):
        return self._articulate(env, call, -1)
