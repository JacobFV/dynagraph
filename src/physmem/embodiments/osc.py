"""Task-space inverse-dynamics controller (operational-space form) for fixed/mobile-base arms.

Per effector and control tick (called every physics step):
  qdd = Jbar (a_des - Jdot qd) + (I - Jbar J) qdd_posture,   Jbar = M^-1 J^T (J M^-1 J^T + lambda W^-1)^-1
  tau = M qdd + h  (+ J^T f for force/impedance terms),  h = qfrc_bias - qfrc_passive (gravity + Coriolis + passive)
a_des = a_ref + Kp e + Kd (v_ref - v) + Ki * int(e)  (min-jerk pos/vel/acc feedforward; integral bounded, as in
CommandAGI physics_run.py's hand task: error clipped to 5 cm, integral clipped). Torque limits: the task part is
scaled by the largest alpha in [0, 1] that keeps every joint within its limit (direction preserved), then clipped.
Joint ranges: each joint's acceleration is bounded so that, held 40 ms, it stops short of its range (wbc.py).
Modes: track | impedance (apparent stiffness K, no integral) | guarded (track; trips and holds when the wrist force
exceeds a threshold) | force (hold a normal force along an axis, track the rest). No convergence waits anywhere:
a trajectory is done when its time is up.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from physmem.v2.api import EffortMeter, TaskTrajectory
from physmem.embodiments.trajectory import Plan, quat_err, smooth3

DEFAULT_GAINS = dict(kp=900.0, kd=None, ki=3000.0, i_max=0.002, kr=600.0, kdr=None, kir=600.0, ir_max=0.01,
                     e_max=0.05, a_max=60.0, post_kp=60.0, post_kd=None, damp=1e-3,
                     imp_k=600.0, imp_kr=30.0, guard_n=15.0, force_ki=3.0, force_kd=40.0,
                     base_kp=150.0, base_kd=None)


def clip_norm(v, most):
    n = float(np.linalg.norm(v))
    return v if n <= most else v * (most / n)


class EffortAccumulator:
    """fills api.EffortMeter every physics step; also keeps peak statistics."""

    def __init__(self):
        self.meters = []
        self.total = EffortMeter()
        self.peaks = dict(torque_rate=0.0, joint_jerk=0.0, torque_norm=0.0)
        self.win_peaks = {}

    def open(self):
        m = EffortMeter()
        self.meters.append(m)
        self.win_peaks[id(m)] = dict(torque_rate=0.0, joint_jerk=0.0, torque_norm=0.0)
        return m

    def close(self, m):
        if m in self.meters:
            self.meters.remove(m)
        return self.win_peaks.pop(id(m), None)

    def add(self, dt, tau_n_sq, rate_sq, jerk_sq, energy, at_limit, ee_speed, rate_peak, jerk_peak, tau_peak):
        for m in [self.total] + self.meters:
            m.torque_sq += tau_n_sq * dt
            m.torque_rate_sq += rate_sq * dt
            m.joint_jerk_sq += jerk_sq * dt
            m.energy_j += energy * dt
            m.time_at_limit_s += dt if at_limit else 0.0
            m.peak_ee_speed = max(m.peak_ee_speed, ee_speed)
        for p in [self.peaks] + list(self.win_peaks.values()):
            p['torque_rate'] = max(p['torque_rate'], rate_peak)
            p['joint_jerk'] = max(p['joint_jerk'], jerk_peak)
            p['torque_norm'] = max(p['torque_norm'], tau_peak)


class TaskSpaceController:
    """api.Controller for MjEmbodiment."""

    def __init__(self, emb, effort_weight: float = 0.0):
        self.emb = emb
        g = dict(DEFAULT_GAINS)
        g.update(emb.cfg.gains)
        for k, kk in (('kd', 'kp'), ('kdr', 'kr'), ('post_kd', 'post_kp'), ('base_kd', 'base_kp')):
            if g[k] is None:
                g[k] = 2 * math.sqrt(g[kk])
        self.g = g
        self.limits = dict(emb.cfg.limits)
        self.effort_weight = effort_weight
        self.acc = EffortAccumulator()
        self.t = 0.0
        self.st = {}
        self._ready = False

    # api: `effort` = cumulative meter
    @property
    def effort(self):
        return self.acc.total

    # ------------------------------------------------------------------ lifecycle
    def reset(self, model, data):
        emb = self.emb
        self.m = model
        self.nv = model.nv
        self.dt = model.opt.timestep
        self.t = float(data.time)
        self.M = np.zeros((model.nv, model.nv))
        self.st = {}
        for name, e in emb.eff.items():
            p, q = emb.tcp_pose(data, name)
            self.st[name] = dict(plans=[], hold=(p, q), mode='track', kw={}, ei=np.zeros(3), eri=np.zeros(3),
                                 grip=0.0, tripped=False, fi=0.0, touched=False, ft=np.zeros(3), ref=None,
                                 err=0.0, home=e['home'] if len(e['home']) else data.qpos[e['qadr']].copy())
        if emb.base is not None:
            qb = data.qpos[emb.base['qadr']].copy()
            self.base = dict(q0=qb, q1=qb.copy(), t0=self.t, T=1e-3)
        self.tau_prev = np.zeros(len(emb.act_ids))
        self.qd1 = data.qvel.copy()
        self.qd2 = data.qvel.copy()
        self.tick = 0
        self._ready = True

    def _ensure(self, model, data):
        if not self._ready or self.m is not model:
            self.reset(model, data)

    # ------------------------------------------------------------------ api
    def command(self, traj: TaskTrajectory, preempt: bool = False) -> dict:
        """queue a task-space trajectory. Its first waypoint may be None (= where the previous one ends / the current
        reference). Returns the retiming prediction (scale >= 1, predicted peaks, duration)."""
        s = self.st[traj.effector]
        if preempt:
            s['plans'] = []
        if s['plans']:
            last = s['plans'][-1]
            p0, q0 = last.end_pose()
            t0 = max(last.t0 + last.duration, self.t)
        else:
            p0, q0 = s['hold']
            t0 = self.t
        plan = Plan(traj, p0, q0, t0, limits=self.limits, effort_weight=self.effort_weight)
        s['plans'].append(plan)
        s['tripped'] = False
        return plan.pred

    def set_mode(self, mode: str, effector: str = None, **kw) -> None:
        assert mode in ('track', 'impedance', 'guarded', 'force'), mode
        for name in ([effector] if effector else list(self.st)):
            s = self.st[name]
            s['mode'], s['kw'] = mode, dict(kw)
            s['tripped'], s['touched'], s['fi'] = False, False, 0.0
            if mode in ('impedance', 'force'):
                s['ei'][:] = 0

    def set_gripper(self, effector: str, g: float):
        self.st[effector]['grip'] = float(np.clip(g, 0.0, 1.0))

    def command_base(self, target, duration):
        """mobile base: min-jerk move of (x, y, yaw) joints to target over duration (s)."""
        b = self.base
        b['q0'] = self._base_ref()[0]
        b['q1'] = np.asarray(target, float)
        b['t0'], b['T'] = self.t, max(duration, 1e-3)

    def done(self, effector: str = None) -> bool:
        names = [effector] if effector else list(self.st)
        for n in names:
            s = self.st[n]
            if s['tripped']:
                continue
            if s['plans'] and not (len(s['plans']) == 1 and s['plans'][0].done(self.t)):
                return False
        if self.emb.base is not None and effector is None and self.t < self.base['t0'] + self.base['T']:
            return False
        return True

    def tripped(self, effector):
        return self.st[effector]['tripped']

    def time_left(self, effector=None):
        out = 0.0
        for n in ([effector] if effector else list(self.st)):
            for p in self.st[n]['plans']:
                out = max(out, p.t0 + p.duration - self.t)
        return max(out, 0.0)

    def reference(self, effector):
        return self.st[effector]['ref']

    # ------------------------------------------------------------------ the tick
    def _base_ref(self):
        b = self.base
        s, sd, sdd = smooth3((self.t - b['t0']) / b['T'])
        d = b['q1'] - b['q0']
        return b['q0'] + d * s, d * sd / b['T'], d * sdd / b['T'] ** 2

    def step(self, model, data) -> np.ndarray:
        self._ensure(model, data)
        emb, g = self.emb, self.g
        dt = self.dt
        self.t = float(data.time)
        mujoco.mj_fullM(model, data, self.M)
        h_all = data.qfrc_bias - data.qfrc_passive
        tau_act = np.zeros(len(emb.act_ids))
        ee_speed = 0.0
        for name, e in emb.eff.items():
            s = self.st[name]
            D = e['dofs']
            n = len(D)
            # ---------------------------------------------------- reference
            plans = s['plans']
            while len(plans) > 1 and plans[0].done(self.t):
                plans.pop(0)  # the next plan's t0 was fixed at command time (= previous end, or now if idle)
            if plans:
                pl = plans[0]
                if pl.gripper is not None and not getattr(pl, '_grip_applied', False):
                    s['grip'] = float(pl.gripper)
                    pl._grip_applied = True
                tr = pl.traj
                if getattr(tr, 'mode', None) and not getattr(pl, '_mode_applied', False):
                    self.set_mode(tr.mode, name, **tr.mode_kw)
                    pl._mode_applied = True
                pr, vr, ar, qr, wr, alr = pl.at(self.t)
                if pl.done(self.t) and len(plans) == 1:
                    s['hold'] = pl.end_pose()
            else:
                pr, qr = s['hold']
                vr = ar = wr = alr = np.zeros(3)
            # ---------------------------------------------------- measurement (encoders + FK)
            p = data.site_xpos[e['site']]
            Rm = data.site_xmat[e['site']].reshape(3, 3)
            qm = np.zeros(4)
            mujoco.mju_mat2Quat(qm, data.site_xmat[e['site']])
            jp, jr = np.zeros((3, self.nv)), np.zeros((3, self.nv))
            mujoco.mj_jacSite(model, data, jp, jr, e['site'])
            dp, dr = np.zeros((3, self.nv)), np.zeros((3, self.nv))
            mujoco.mj_jacDot(model, data, dp, dr, p, e['body'])
            qd = data.qvel
            J = np.vstack([jp[:, D], jr[:, D]])
            Jdqd = np.concatenate([dp @ qd, dr @ qd])
            vel = J @ qd[D]
            v, w = vel[:3], vel[3:]
            ee_speed = max(ee_speed, float(np.linalg.norm(v)))
            # ---------------------------------------------------- wrist F/T and mode logic
            mode, kw = s['mode'], s['kw']
            F = np.zeros(3)
            if mode in ('guarded', 'force') or kw.get('measure_ft'):
                F, _, _ = emb.wrist_ft(model, data, name)
            s['ft'] = F
            if 'bias' in kw:
                F = F - np.asarray(kw['bias'], float)
            if mode == 'guarded' and not s['tripped']:
                axis = kw.get('axis')
                fmag = float(-(F @ np.asarray(axis))) if axis is not None else float(np.linalg.norm(F))
                if fmag > kw.get('threshold', g['guard_n']):
                    s['tripped'] = True
                    s['trip_t'] = self.t
                    hold_p = p.copy() - (np.asarray(axis) * kw.get('backoff', 0.0) if axis is not None else 0)
                    s['hold'] = (hold_p, qm.copy())
                    s['plans'] = []
                    pr, qr = s['hold']
                    vr = ar = wr = alr = np.zeros(3)
            if mode == 'guarded' and s['tripped']:
                pr, qr = s['hold']
                vr = ar = wr = alr = np.zeros(3)
            e_p = clip_norm(pr - p, g['e_max'])
            e_r = quat_err(qr, qm)
            s['err'] = float(np.linalg.norm(pr - p))
            s['ref'] = (pr, vr, qr)
            # ---------------------------------------------------- dynamics on this effector's dofs
            M = self.M[np.ix_(D, D)]
            h = h_all[D]
            Minv = np.linalg.inv(M)
            W = np.array([1, 1, 1] + [e['cfg'].rot_weight] * 3, float)
            A = J @ Minv @ J.T
            lam = g['damp'] * np.trace(A) / 6.0
            Lw = np.linalg.inv(A + lam * np.diag(1.0 / W))
            Jbar = Minv @ J.T @ Lw
            meff = 1.0 / max(np.trace(A[:3, :3]) / 3.0, 1e-6)
            f_task = np.zeros(6)
            if mode == 'impedance' or (mode == 'guarded' and s['tripped'] and kw.get('hold_soft', True)) or \
                    (mode == 'force' and kw.get('compliant')):
                K = kw.get('stiffness', g['imp_k'])
                Kr = kw.get('rot_stiffness', g['imp_kr'])
                Dp = kw.get('damping', 2 * math.sqrt(K * meff))
                a_lin = ar + (K * e_p + Dp * (vr - v)) / meff
                ieff = 1.0 / max(np.trace(A[3:, 3:]) / 3.0, 1e-9)
                if mode == 'force':  # compliant force mode: soft translation, stiff orientation (keeps a held peg square)
                    s['eri'] = clip_norm(s['eri'] + e_r * dt, g['ir_max'])
                    a_rot = alr + g['kr'] * e_r + g['kdr'] * (wr - w) + g['kir'] * s['eri']
                else:
                    a_rot = alr + (Kr * e_r + 2 * math.sqrt(Kr * ieff) * (wr - w)) / ieff
            else:
                s['ei'] = clip_norm(s['ei'] + e_p * dt, g['i_max'])
                s['eri'] = clip_norm(s['eri'] + e_r * dt, g['ir_max'])
                a_lin = ar + g['kp'] * e_p + g['kd'] * (vr - v) + g['ki'] * s['ei']
                a_rot = alr + g['kr'] * e_r + g['kdr'] * (wr - w) + g['kir'] * s['eri']
            if mode == 'force':
                nax = np.asarray(kw['axis'], float)
                nax = nax / np.linalg.norm(nax)
                Fd = float(kw.get('force', 10.0))
                Fm = float(-(F @ nax))          # environment pushes back along -axis
                vn = float(v @ nax)
                a_lin = a_lin - nax * (a_lin @ nax)
                s['ei'] = s['ei'] - nax * (s['ei'] @ nax)
                if not s['touched'] and Fm < kw.get('touch_n', 1.0):
                    # approach: velocity servo along the axis + a light force feed-forward (overcomes model bias)
                    a_lin = a_lin + nax * (g['force_kd'] * (kw.get('approach_speed', 0.03) - vn))
                    f_task[:3] = nax * min(Fd, 2.0)
                else:
                    s['touched'] = True
                    s['fi'] = float(np.clip(s['fi'] + (Fd - Fm) * dt, -1.0, 1.0))   # N s, bounded
                    a_lin = a_lin - nax * (g['force_kd'] * vn)
                    f_task[:3] = nax * (Fd + g['force_ki'] * s['fi'])
            s['a_lin'], s['f_task'] = a_lin, f_task
            a = np.concatenate([clip_norm(a_lin, g['a_max']), clip_norm(a_rot, 4 * g['a_max'])])
            qdd_t = Jbar @ (a - Jdqd)
            N = np.eye(n) - Jbar @ J
            q = data.qpos[e['qadr']]
            qd_d = qd[D]
            qdd_0 = g['post_kp'] * (s['home'] - q) - g['post_kd'] * qd_d
            qdd_p = N @ qdd_0
            # joint ranges: an acceleration that, held 40 ms, stops each joint short of its ends (wbc.py)
            Hh = 0.04
            jl = e.get('_jlim')
            if jl is None:
                jl = []
                for k, dof in enumerate(D):
                    j = model.dof_jntid[dof]
                    if model.jnt_limited[j]:
                        jl.append((k, *model.jnt_range[j]))
                e['_jlim'] = jl
            qdd = qdd_t + qdd_p
            for k, lo, hi in jl:
                ql, vl = q[k], qd_d[k]
                amin = 2 * (lo + 0.01 - ql - vl * Hh) / (Hh * Hh)
                amax = 2 * (hi - 0.01 - ql - vl * Hh) / (Hh * Hh)
                if qdd[k] < amin or qdd[k] > amax:
                    qdd[k] = np.clip(qdd[k], amin, min(amax, max(amin, amax)))
            tau_t = M @ qdd_t
            tau_b = M @ (qdd - qdd_t) + h + J.T @ f_task
            lim = e['tau_lim']
            # torque limits: largest alpha in [0,1] with tau_b + alpha tau_t inside limits
            alpha = 1.0
            for k in range(n):
                if abs(tau_t[k]) < 1e-12:
                    continue
                lo_, hi_ = (lim[k, 0] - tau_b[k]) / tau_t[k], (lim[k, 1] - tau_b[k]) / tau_t[k]
                if tau_t[k] < 0:
                    lo_, hi_ = hi_, lo_
                alpha = min(alpha, max(hi_, 0.0))
            tau = np.clip(tau_b + alpha * tau_t, lim[:, 0], lim[:, 1])
            s['alpha'] = alpha
            tau_act[e['acts']] = tau / emb.act_gear[e['acts']]
            # ---------------------------------------------------- gripper (force-limited servo on transmission length)
            if len(e['gacts']) and e['cfg'].gripper.kind == 'suction':
                # vacuum: adhesion on only while commanded AND sealed (no seal -> no holding force)
                # seal with hysteresis: making a seal needs an aligned cup (20 deg); a held seal persists while the pad
                # still touches (tilt up to 45 deg) - a hanging load tilts the cup without breaking the vacuum
                held_seal = s.get('sealed', False) and s['grip'] >= 0.5
                sealed, _ = emb.cup_seal(model, data, name, max_tilt_deg=45.0 if held_seal else 20.0)
                s['sealed'] = sealed
                tau_act[e['gacts']] = float(np.clip(e.get('suction_level', 1.0), 0, 1)) if (s['grip'] >= 0.5 and sealed) else 0.0
            elif len(e['gacts']):
                ga = emb.act_ids[e['gacts']]
                L = data.actuator_length[ga]
                Ld = data.actuator_velocity[ga]
                gg = s['grip']
                stroke = e['gclosed'] - e['gopen']
                tgt = e['gopen'] + gg * stroke + (0.3 * stroke if gg >= 0.99 else 0.0)
                Fg = e['gforce']
                gc = e['cfg'].gripper
                kp = gc.kp if gc.kp > 0 else Fg / (0.1 * np.abs(stroke) + 1e-9)
                u = np.clip(kp * (tgt - L) - gc.kd_ratio * kp * Ld, -Fg, Fg)
                lo_g, hi_g = emb.act_lo[e['gacts']] * emb.act_gear[e['gacts']], emb.act_hi[e['gacts']] * emb.act_gear[e['gacts']]
                tau_act[e['gacts']] = np.clip(u, lo_g, hi_g) / emb.act_gear[e['gacts']]
        # ---------------------------------------------------- mobile base servo (computed-torque PD)
        if emb.base is not None:
            B = emb.base['dofs']
            qr, vr, ar = self._base_ref()
            qb, vb = data.qpos[emb.base['qadr']], data.qvel[B]
            eb = qr - qb
            eb[2] = math.remainder(eb[2], 2 * math.pi)
            Mb = self.M[np.ix_(B, B)]
            tb = Mb @ (ar + g['base_kp'] * eb + g['base_kd'] * (vr - vb)) + h_all[B]
            lim = emb.base['tau_lim']
            tau_act[emb.base['acts']] = np.clip(tb, lim[:, 0], lim[:, 1])
        tau_act = np.clip(tau_act, emb.act_lo, emb.act_hi)
        self._meter(model, data, tau_act, ee_speed)
        return tau_act

    def apply(self, model, data):
        """compute and write this robot's torques into data.ctrl (call before each mj_step)."""
        u = self.step(model, data)
        data.ctrl[self.emb.act_ids] = u
        return u

    # ------------------------------------------------------------------ effort ("servo pain")
    def _meter(self, model, data, tau_act, ee_speed):
        emb, dt = self.emb, self.dt
        span = np.maximum(np.abs(emb.act_lo), np.abs(emb.act_hi))
        tn = tau_act / span
        rate = (tau_act - self.tau_prev) / dt / span if self.tick > 0 else np.zeros_like(tn)
        D = emb.dofs
        qd = data.qvel[D]
        if self.tick >= 2:
            jerk = (qd - 2 * self.qd1[D] + self.qd2[D]) / (dt * dt)
        else:
            jerk = np.zeros(len(D))
        av = data.actuator_velocity[emb.act_ids]
        energy = float(np.sum(np.abs(tau_act * emb.act_gear * av)))
        at_lim = bool(np.any(np.abs(tn) >= 0.99))
        self.acc.add(dt, float(tn @ tn), float(rate @ rate), float(jerk @ jerk), energy, at_lim, ee_speed,
                     float(np.max(np.abs(rate))) if len(rate) else 0.0, float(np.max(np.abs(jerk))) if len(jerk) else 0.0,
                     float(np.max(np.abs(tn))))
        self.tau_prev = tau_act.copy()
        self.qd2 = self.qd1
        self.qd1 = data.qvel.copy()
        self.tick += 1
