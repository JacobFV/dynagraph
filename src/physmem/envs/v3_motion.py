"""envs-v3: robot-agnostic task-space motion for the new family skills (agent ENVS-V3).

The envs-v3 families (cable, cloth, pouring, granular scoop, stacking, handover) need a few motions the r7 skill set
does not have: tilting a held vessel about the hand's x axis (pour / scoop / dump), slow guided drags (cable routing),
holding still while watching a sensor. They are written ONCE against `Motion`, which drives either

* the TEST-ONLY probe gantry (`ProbeTiltRobot`: the r7 probe + one wrist TILT hinge about the hand x axis through the
  TCP; same PD + min-jerk control; used only by the envs-v3 families, so every older family's probe model is
  unchanged), or
* a real arm through the ROBOTS task-space controller (`physmem.skills.primitives._move` with a full orientation).

Conventions (probe convention everywhere): hand yaw `yaw` -> the pads close along (-sin yaw, cos yaw); the hand x axis
is (cos yaw, sin yaw, 0); `tilt` (rad) rotates the hand about its x axis through the TCP (positive = right-handed about
+x, so the hand's -z approach axis swings towards +y_hand). Physics advances only through robot actuators (no pose
writes); every step goes through env.pre_physics / env.post_physics (integrity, sensors, contact monitor).

Family skills are env methods `fsk_<name>(M, call) -> (status, log)`; `StationEnv.execute` routes a SkillCall whose
skill has such a method through `run_family_skill`, which meters effort like the probe / ROBOTS skill runners.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..v2.api import EffortMeter, SkillResult
from .probe import ProbeRobot, minjerk, wrap

J = mujoco.mjtJoint
G = mujoco.mjtGeom


class ProbeTiltRobot(ProbeRobot):
    """probe gantry + a wrist tilt hinge (about the hand x axis, through the TCP). TEST-ONLY like the probe."""

    TILT_LIM = 20.0          # N m
    TILT_KP, TILT_KD = 400.0, 12.0

    def build(self, scene, mount):
        super().build(scene, mount)
        hand = scene.spec.body("probe_hand")
        hand.add_joint(name="probe_jtilt", type=J.mjJNT_HINGE, axis=[1, 0, 0], range=[-2.4, 2.4], limited=True,
                       damping=0.3, armature=0.01)
        a = scene.spec.add_actuator(name="probe_atilt", target="probe_jtilt", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_motor()
        a.ctrlrange = [-self.TILT_LIM, self.TILT_LIM]
        a.forcerange = [-self.TILT_LIM, self.TILT_LIM]
        a.ctrllimited = True
        a.forcelimited = True

    def bind(self, model, data):
        super().bind(model, data)
        j = model.joint("probe_jtilt").id
        self.tq, self.tv = model.jnt_qposadr[j], model.jnt_dofadr[j]
        self.tact = model.actuator("probe_atilt").id
        self.tilt_ref, self.tilt_seg, self.tilt_t = 0.0, None, 0.0

    def tilt(self):
        return float(self.d.qpos[self.tq])

    def command_tilt(self, goal, T):
        self.tilt_seg = (self.tilt_ref, float(goal), max(float(T), 0.05))
        self.tilt_t = 0.0

    def done(self):
        return super().done() and self.tilt_seg is None

    def step(self, dt):
        super().step(dt)
        vref = 0.0
        if self.tilt_seg is not None:
            a, b, T = self.tilt_seg
            self.tilt_t += dt
            p, v, _ = minjerk(min(self.tilt_t, T), T)
            self.tilt_ref, vref = a + (b - a) * p, (b - a) * v
            if self.tilt_t >= T:
                self.tilt_ref, vref, self.tilt_seg = b, 0.0, None
        q, dq = self.d.qpos[self.tq], self.d.qvel[self.tv]
        tau = float(np.clip(self.TILT_KP * (self.tilt_ref - q) + self.TILT_KD * (vref - dq), -self.TILT_LIM,
                            self.TILT_LIM))
        self.d.ctrl[self.tact] = tau
        ef = self.effort
        ef.torque_sq += (tau / self.TILT_LIM) ** 2 * dt
        ef.energy_j += abs(tau * dq) * dt

    def joint_names(self):
        return super().joint_names() + ["jtilt"]

    def joint_state(self):
        q, dq = super().joint_state()
        return q + [float(self.d.qpos[self.tq])], dq + [float(self.d.qvel[self.tv])]


def _axis_quat(axis, ang):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float) / np.linalg.norm(axis), float(ang))
    return q


def hand_quat(yaw, tilt=0.0):
    """real-arm TCP quaternion for probe-convention hand yaw + tilt about the hand x axis."""
    from physmem.embodiments.ik import down_quat
    q0 = down_quat(yaw + math.pi / 2)                 # ROBOTS: TCP y = closing axis at angle yaw + pi/2
    if abs(tilt) < 1e-9:
        return q0
    x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    # probe: hinge axis +x_hand; R_tilt(x, tilt) applied in the world frame about the (horizontal) hand x axis
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, _axis_quat(x, tilt), q0)
    return q


class Motion:
    """task-space motion API over the probe (ProbeTiltRobot / ProbeRobot) or a ROBOTS embodiment."""

    def __init__(self, env):
        self.env = env
        self.r = env.robot
        self.probe = isinstance(self.r, ProbeRobot)
        if not self.probe:
            self.sk = env.skills
            self.ctx = env.skills._ctx(env)
            self.eff = self.r.eff
        self._tilt = 0.0

    # ------------------------------------------------------------------ state (actor-legal: encoders / FK)
    def tcp(self):
        p, yaw = self.r.tcp()
        return np.asarray(p, float), float(yaw)

    def tilt(self):
        if self.probe:
            return self.r.tilt() if hasattr(self.r, "tilt") else 0.0
        return self._tilt

    def can_tilt(self):
        return (not self.probe) or hasattr(self.r, "command_tilt")

    def wrench(self, window=0.25):
        """median tared wrist F/T over the last `window` s from the EMITTED event stream (actor-legal)."""
        from ..schema import Prop
        env = self.env
        evs = [e for e in env.view.buf.get(("robot/hand", int(Prop.WRENCH)), ())
               if e.available_at <= env.t + 1e-9 and e.measured_at >= env.t - window]
        if not evs:
            return np.zeros(6)
        return np.median(np.array([e.value for e in evs]), 0)

    def tip(self):
        """fingertip extent below the TCP along the approach (m): the lowest safe TCP height above a surface is
        surface + tip + margin (probe: its TCP is the fingertip)."""
        if self.probe:
            return 0.0
        return float(self.r.emb.eff[self.eff]["cfg"].gripper.tip)

    def reach_m(self):
        return 1.0 if self.probe else float(self.r.emb.spec.reach_m)

    # ------------------------------------------------------------------ motion
    def move(self, pos, yaw=None, tilt=None, speed=0.3, min_T=0.3, until=None):
        """go to pos (TCP) with hand yaw / tilt (None = keep); returns 'done' | 'until' | 'timeout' | 'unstable' |
        'unreachable'."""
        env = self.env
        pos = np.asarray(pos, float)
        p0, y0 = self.tcp()
        yaw = y0 if yaw is None else float(yaw)
        tilt = self.tilt() if tilt is None else float(tilt)
        if self.probe:
            r = self.r
            dist = float(np.linalg.norm(pos - (r.ref[:3] + r.mount_pos)))
            dt_ = abs(tilt - self.tilt())
            T = min(10.0, max(min_T, 1.875 * dist / max(speed, 1e-3), abs(wrap(yaw - y0)) / 2.0, 1.875 * dt_ / 1.5))
            lo = r.mount_pos + np.array([-1.15, -1.15, -1.55])
            hi = r.mount_pos + np.array([1.15, 1.15, 0.25])
            r.command([(np.clip(pos, lo, hi), yaw)], [T])
            if hasattr(r, "command_tilt"):
                r.command_tilt(tilt, T)
            stop = r.done if until is None else (lambda: r.done() or until())
            env.run(T + 0.1, stop=stop)
            if until is not None and until():
                return "until"
            return "done"
        from physmem.skills.primitives import _move, _reachable
        q = hand_quat(yaw, tilt)
        if not _reachable(self.ctx, self.eff, pos, q, tol=0.015):
            return "unreachable"
        self._tilt = tilt
        u = None if until is None else (lambda c: until())
        st = _move(self.ctx, self.eff, [pos], max(speed, 0.02), q, until=u)
        return st

    def path(self, pts, yaw=None, tilt=None, speed=0.3):
        st = "done"
        for p in pts:
            st = self.move(p, yaw, tilt, speed)
            if st not in ("done",):
                return st
        return st

    def wait(self, T, until=None):
        self.env.run(float(T), stop=until)

    # ------------------------------------------------------------------ gripper
    def open(self, width=None):
        env = self.env
        if self.probe:
            r = self.r
            r.set_grip("open", 2 * r.max_open if width is None else min(width, 2 * r.max_open))
            env.run(0.35)
            return
        from physmem.skills.primitives import _close
        # full drive effort to open: after a light squeeze the drive force is tiny (2F-85: (F - 3 N) / 9.07) and the
        # linkage did not reopen (the plug stayed pinched after a 4 N probe)
        e = self.r.emb.eff[self.eff]
        if e.get("gforce") is not None and self.eff in getattr(self.r, "gforce0", {}):
            e["gforce"] = np.array(self.r.gforce0[self.eff], float).copy()
        self.r._gmax, self.r._slip = None, None
        _close(self.ctx, self.eff, 0.0)
        self.r.grip_cmd = ("open", 0.0)

    def max_open(self):
        if self.probe:
            return 2 * self.r.max_open
        return float(self.r.emb.eff[self.eff]["cfg"].gripper.max_width)

    def grip(self, force):
        env = self.env
        if self.probe:
            self.r.set_grip("force", float(force))
            env.skills.wait_closed(env)
            return
        from physmem.skills.primitives import _close
        self.sk._set_grip_force(self.eff, float(force))
        _close(self.ctx, self.eff, 1.0)

    def set_grip(self, force):
        """change the squeeze without waiting (e.g. a hand-over release / regrip)."""
        if self.probe:
            self.r.set_grip("force", float(force))
        else:
            self.sk._set_grip_force(self.eff, float(force))

    def holding(self):
        """grasp check from proprioception: fingers stopped on something (not fully closed) with pad contact."""
        if self.probe:
            return self.env.skills._held_ok(self.env)
        c, n = self.r.emb.gripper_state(self.env.m, self.env.d, self.eff)
        return n > 0 and c < 0.97


def run_family_skill(env, call, fn):
    """run an env-defined skill fn(M, call) -> (status, log) and meter it like the robot skill runners."""
    r = env.robot
    M = Motion(env)
    t0 = env.t
    if M.probe:
        e0 = EffortMeter(**r.effort.__dict__)
        try:
            status, log = fn(M, call)
        except Exception as ex:          # keep the episode alive; report honestly
            status, log = f"error:{type(ex).__name__}", dict(msg=str(ex)[:200])
        eff = EffortMeter(**{k: getattr(r.effort, k) - getattr(e0, k) for k in e0.__dict__})
        eff.peak_ee_speed = r.effort.peak_ee_speed
        r.effort.peak_ee_speed = 0.0
        log = dict(log, skill=call.skill, params=dict(call.params), target=call.target)
        return SkillResult(status == "done", status, eff, env.t - t0, log)
    from physmem.skills.base import SkillRun
    with SkillRun(M.ctx, call.skill) as run:
        try:
            status, log = fn(M, call)
        except Exception as ex:
            status, log = f"error:{type(ex).__name__}", dict(msg=str(ex)[:200])
        return run.result(status == "done", status, **dict(log, params=dict(call.params), target=call.target))
