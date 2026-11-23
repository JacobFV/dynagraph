"""Task-space trajectories: blended quintic (min-jerk family) position segments + slerp orientation, effort retiming.

Position: each segment is a quintic Hermite (p, v, a=0 at both ends). With zero end velocities it is exactly the
min-jerk profile `smooth3` of CommandAGI physics_run.py (adapted, same author/org); interior via-points carry a
non-zero blend velocity, so the hand flows through them without stopping unless the index is in `stop_at`.
Orientation: slerp between consecutive waypoint quaternions with a min-jerk phase per segment (angular velocity and
acceleration fed forward).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

from physmem.v2.api import TaskTrajectory


def smooth3(u):
    """min-jerk value, rate and acceleration at u in [0,1] (per unit u). (from CommandAGI physics_run.py)"""
    if u <= 0 or u >= 1:
        return (0.0 if u <= 0 else 1.0), 0.0, 0.0
    return u ** 3 * (10 - 15 * u + 6 * u * u), 30 * u * u * (1 - u) ** 2, 60 * u * (1 - u) * (1 - 2 * u)


def quat_err(qa, qb):
    """rotation vector (world frame) taking qb to qa. (from CommandAGI wbc.py)"""
    qi, qe, v = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(qi, qb)
    mujoco.mju_mulQuat(qe, qa, qi)
    mujoco.mju_quat2Vel(v, qe, 1.0)
    return v


def quat_mul(a, b):
    o = np.zeros(4)
    mujoco.mju_mulQuat(o, np.asarray(a, float), np.asarray(b, float))
    return o


def quat_from_rotvec(w):
    q = np.zeros(4)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.array([1.0, 0, 0, 0])
    mujoco.mju_axisAngle2Quat(q, np.asarray(w, float) / th, th)
    return q


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def canon(q_prev, q):
    q = np.asarray(q, float) / np.linalg.norm(q)
    return -q if q_prev is not None and float(np.dot(q_prev, q)) < 0 else q


# quintic Hermite basis with zero end accelerations
def _hermite5(p0, v0, p1, v1, T, t):
    u = min(max(t / T, 0.0), 1.0)
    u2, u3, u4, u5 = u * u, u ** 3, u ** 4, u ** 5
    h00 = 1 - 10 * u3 + 15 * u4 - 6 * u5
    h10 = u - 6 * u3 + 8 * u4 - 3 * u5
    h01 = 10 * u3 - 15 * u4 + 6 * u5
    d00 = (-30 * u2 + 60 * u3 - 30 * u4)
    d10 = (1 - 18 * u2 + 32 * u3 - 15 * u4)
    d01 = (30 * u2 - 60 * u3 + 30 * u4)
    a00 = (-60 * u + 180 * u2 - 120 * u3)
    a10 = (-36 * u + 96 * u2 - 60 * u3)
    a01 = (60 * u - 180 * u2 + 120 * u3)
    j00 = (-60 + 360 * u - 360 * u2)
    j10 = (-36 + 192 * u - 180 * u2)
    j01 = (60 - 360 * u + 360 * u2)
    h11 = -4 * u3 + 7 * u4 - 3 * u5
    d11 = -12 * u2 + 28 * u3 - 15 * u4
    a11 = -24 * u + 84 * u2 - 60 * u3
    j11 = -24 + 168 * u - 180 * u2
    p = h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1
    v = (d00 * p0 + d10 * T * v0 + d01 * p1 + d11 * T * v1) / T
    a = (a00 * p0 + a10 * T * v0 + a01 * p1 + a11 * T * v1) / (T * T)
    j = (j00 * p0 + j10 * T * v0 + j01 * p1 + j11 * T * v1) / (T ** 3)
    if t >= T:
        p, v, a, j = p1, v1, a * 0.0, j * 0.0
    elif t <= 0:
        p, v, a, j = p0, v0, a * 0.0, j * 0.0
    return p, v, a, j


@dataclass
class BlendedTrajectory(TaskTrajectory):
    """TaskTrajectory + optional explicit via-point velocities (m/s, world frame), one per waypoint.
    `via_vel[i] = None` lets the planner choose the blend velocity."""
    via_vel: Optional[list] = None
    mode: Optional[str] = None          # optional mode switch applied when this trajectory starts
    mode_kw: dict = field(default_factory=dict)


class Plan:
    """An evaluated trajectory: absolute start time t0, segments, plus retiming info."""

    def __init__(self, traj: TaskTrajectory, p_start, q_start, t0, v_start=None, limits=None, effort_weight=0.0):
        wps = list(traj.waypoints)
        P, Q = [], []
        for i, w in enumerate(wps):
            if w is None:
                pos, quat = p_start, q_start
            else:
                pos, quat = w
                pos = p_start if pos is None else pos
                quat = (Q[-1] if Q else q_start) if quat is None else quat
            P.append(np.asarray(pos, float))
            Q.append(canon(Q[-1] if Q else None, quat))
        if wps and wps[0] is not None:  # first waypoint is a target: start from the current pose
            P.insert(0, np.asarray(p_start, float))
            Q.insert(0, canon(Q[0], q_start))
            durs = list(traj.durations)
            stop = tuple(i + 1 for i in traj.stop_at)
            vv = getattr(traj, 'via_vel', None)
            vv = None if vv is None else [None] + list(vv)
        else:
            durs = list(traj.durations)
            stop = tuple(traj.stop_at)
            vv = getattr(traj, 'via_vel', None)
        # make Q[0] consistent with q_start
        assert len(durs) == len(P) - 1, f'durations {len(durs)} vs segments {len(P) - 1}'
        self.P, self.Q, self.T = P, Q, [max(1e-3, float(d)) for d in durs]
        n = len(P)
        V = [np.zeros(3) for _ in range(n)]
        V[0] = np.zeros(3) if v_start is None else np.asarray(v_start, float)
        for i in range(1, n - 1):
            if vv is not None and i < len(vv) and vv[i] is not None:
                V[i] = np.asarray(vv[i], float)
                continue
            if i in stop:
                continue
            va = (P[i] - P[i - 1]) / self.T[i - 1]
            vb = (P[i + 1] - P[i]) / self.T[i]
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na < 1e-9 or nb < 1e-9:
                continue
            c = float(va @ vb) / (na * nb)
            if c <= 0.0:  # a reversal or sharp corner: stop there
                continue
            # blend: average of adjacent mean velocities, shrunk by the corner's cosine, no faster than the slower side
            v = 0.5 * (va + vb) * c
            sp = np.linalg.norm(v)
            lim = min(na, nb) * 1.5
            V[i] = v * (min(sp, lim) / max(sp, 1e-12))
        if vv is not None and len(vv) == n and vv[-1] is not None:
            V[-1] = np.asarray(vv[-1], float)
        self.V = V
        self.scale = 1.0
        self.effort_weight = effort_weight
        self.limits = limits
        if limits is not None:
            self._retime(limits, effort_weight)
        self.t0 = float(t0)
        self.starts = np.concatenate([[0.0], np.cumsum(self.T)])
        self.duration = float(self.starts[-1])
        self.stop_times = [self.starts[i] for i in stop if i < len(self.starts)]
        self.gripper = traj.gripper
        self.traj = traj

    # ------------------------------------------------------------------ retiming ("servo pain" budget)
    def peaks(self, n=40):
        pv = pa = pj = 0.0
        cost_a = cost_j = 0.0
        for k in range(len(self.T)):
            T = self.T[k]
            for t in np.linspace(0, T, n):
                _, v, a, j = _hermite5(self.P[k], self.V[k], self.P[k + 1], self.V[k + 1], T, t)
                pv, pa, pj = max(pv, np.linalg.norm(v)), max(pa, np.linalg.norm(a)), max(pj, np.linalg.norm(j))
                cost_a += float(a @ a) * T / n
                cost_j += float(j @ j) * T / n
            # orientation: min-jerk slerp, angular rate peaks 1.875 * angle / T
            ang = np.linalg.norm(quat_err(self.Q[k + 1], self.Q[k]))
            pv = max(pv, 0.0)
            self._wpk = max(getattr(self, '_wpk', 0.0), 1.875 * ang / T)
        return pv, pa, pj, cost_a, cost_j

    def _retime(self, lim, w):
        """uniform time scaling k >= 1 so peak speed/acc/jerk (and angular rate) stay within the robot's budget,
        tightened by the effort weight w in [0, 1] (w=1: half speed, quarter acceleration, eighth jerk budget)."""
        vmax = lim['v'] * (1 - 0.5 * w)
        amax = lim['a'] * (1 - 0.75 * w)
        jmax = lim['j'] * (1 - 0.875 * w)
        wmax = lim.get('w', 3.0) * (1 - 0.5 * w)
        self._wpk = 0.0
        pv, pa, pj, _, _ = self.peaks()
        k = max(1.0, pv / vmax, math.sqrt(pa / amax), (pj / jmax) ** (1 / 3), self._wpk / wmax)
        if k > 1.0 + 1e-6:
            self.T = [t * k for t in self.T]
            self.V = [v / k for v in self.V]
            self.scale = k
        self._wpk = 0.0
        pv, pa, pj, ca, cj = self.peaks()
        self.pred = dict(scale=round(self.scale, 4), peak_v=pv, peak_a=pa, peak_j=pj,
                         cost=ca / lim['a'] ** 2 + cj / lim['j'] ** 2, duration=float(sum(self.T)))

    # ------------------------------------------------------------------ evaluation
    def at(self, t):
        """reference at absolute time t -> (p, v, a, quat, omega, alpha)."""
        tau = t - self.t0
        if tau >= self.duration:
            return self.P[-1], np.zeros(3), np.zeros(3), self.Q[-1], np.zeros(3), np.zeros(3)
        if tau <= 0:
            k = 0
            tau = 0.0
        else:
            k = int(np.searchsorted(self.starts, tau, side='right') - 1)
            k = min(max(k, 0), len(self.T) - 1)
        ts = tau - self.starts[k]
        T = self.T[k]
        p, v, a, _ = _hermite5(self.P[k], self.V[k], self.P[k + 1], self.V[k + 1], T, ts)
        s, sd, sdd = smooth3(ts / T)
        rv = quat_err(self.Q[k + 1], self.Q[k])
        q = quat_mul(quat_from_rotvec(rv * s), self.Q[k])
        return p, v, a, q, rv * sd / T, rv * sdd / (T * T)

    def done(self, t):
        return t - self.t0 >= self.duration

    def end_pose(self):
        return self.P[-1].copy(), self.Q[-1].copy()


def make_traj(effector, waypoints, durations=None, speed=None, stop_at=(), gripper=None, via_vel=None,
              mode=None, mode_kw=None, start=None, min_T=0.15):
    """helper: build a BlendedTrajectory. If durations is None, each segment's duration is chosen from `speed`
    (m/s average) so a min-jerk segment's average speed is `speed` (peak = 1.875x for rest-to-rest)."""
    if durations is None:
        assert speed is not None, 'need durations or speed'
        pts = [start] + [w[0] for w in waypoints] if start is not None else [w[0] for w in waypoints]
        durations = []
        for a, b in zip(pts[:-1], pts[1:]):
            durations.append(max(min_T, float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / speed))
    return BlendedTrajectory(effector=effector, waypoints=list(waypoints), durations=list(durations),
                             stop_at=tuple(stop_at), gripper=gripper, via_vel=via_vel, mode=mode,
                             mode_kw=dict(mode_kw or {}))
