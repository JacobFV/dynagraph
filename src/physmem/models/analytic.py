"""analytic baselines (agent MODEL). Same call signature/ModelOutput as compact models.

- ConstantVelocity: persistence of the target's latest velocity estimate (from VEL2 events,
  else finite difference of the two latest POSE2 events), pose integrated at constant velocity.
- CoulombAnalytic: kinematic pusher-contact approximation during the push (object moves with the
  pusher along the push direction after the approach gap closes) followed by Coulomb sliding
  deceleration mu*g. mu is estimated from the episode HISTORY by a grid posterior over
  post-push deceleration fits of the target's observed pose track (prior when no informative
  history). Uses only actor-available evidence.
- CoulombOracle: identical kinematics but is GIVEN the true friction from LabelTensors.
  Labelled `oracle` everywhere and excluded from normal comparisons.

Predictive variances: per-horizon/per-dim residual variance calibrated on TRAIN batches
(`calibrate`), plus (Coulomb) propagated friction uncertainty.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from physmem.schema import HORIZONS, STATE_DIM, EntityType, EventTensors, ModelOutput, Prop, QueryTensors

G = 9.81
PUSH_GAP = 0.003
MU_PRIOR = (0.3, 0.7)
MU_GRID = np.linspace(0.05, 1.3, 126)


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _target_track(ev, b, tgt):
    """sorted (t, x, y, cos, sin, var_xy) of target POSE2 events, t = -age (s, now = 0)."""
    m = ev["mask"][b] & (ev["entity"][b] == tgt) & (ev["prop"][b] == int(Prop.POSE2))
    if not m.any():
        return None
    t = -ev["age"][b][m]
    v = ev["value"][b][m]
    lv = ev["logvar"][b][m]
    o = np.argsort(t, kind="stable")
    return t[o], v[o, :4], np.exp(lv[o, 0])


def _latest_vel(ev, b, tgt):
    m = ev["mask"][b] & (ev["entity"][b] == tgt) & (ev["prop"][b] == int(Prop.VEL2))
    if m.any():
        i = np.argmin(np.where(m, ev["age"][b], np.inf))
        return ev["value"][b][i, :3].astype(np.float64)
    tr = _target_track(ev, b, tgt)
    if tr is None or len(tr[0]) < 2:
        return np.zeros(3)
    t, v, _ = tr
    dt = max(t[-1] - t[-2], 1e-3)
    return np.array([(v[-1, 0] - v[-2, 0]) / dt, (v[-1, 1] - v[-2, 1]) / dt, 0.0])


def _size(ev, b, tgt):
    m = ev["mask"][b] & (ev["entity"][b] == tgt) & (ev["prop"][b] == int(Prop.SIZE))
    if not m.any():
        return np.array([0.035, 0.035, 0.035])
    i = np.argmin(np.where(m, ev["age"][b], np.inf))
    return np.abs(ev["value"][b][i, :3].astype(np.float64))


def _pusher_radius(ev, b):
    et = ev["ent_type"][b]
    em = ev["ent_mask"][b]
    ps = np.where(em & (et == int(EntityType.PUSHER)))[0]
    for p in ps:
        m = ev["mask"][b] & (ev["entity"][b] == p) & (ev["prop"][b] == int(Prop.SIZE))
        if m.any():
            i = np.argmin(np.where(m, ev["age"][b], np.inf))
            return float(abs(ev["value"][b][i, 0]))
    return 0.015


def push_geometry(half, theta, phi):
    """same convention as worlds._push_geometry: boundary point p, push direction d, outward normal n."""
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


def push_slide_kinematics(action, half, mu, horizons, gap=PUSH_GAP):
    """(H, STATE_DIM) predicted state for a push action vector (canonical object frame)."""
    a = np.asarray(action, np.float64)
    H = len(horizons)
    out = np.zeros((H, STATE_DIM))
    out[:, 2] = 1.0
    if a[7] > 0.5:  # impulse kind: (jx, jy) in a[4:6]; mass unknown -> unit mass approx
        v0 = a[4:6]
        sp0 = np.linalg.norm(v0)
        dirv = v0 / max(sp0, 1e-9)
        for i, h in enumerate(horizons):
            ts = sp0 / max(mu * G, 1e-6)
            tt = min(h, ts)
            s = sp0 * tt - 0.5 * mu * G * tt * tt
            v = max(0.0, sp0 - mu * G * h)
            out[i, 0:2] = dirv * s
            out[i, 4:6] = dirv * v
        return out
    if a[6] < 0.5:  # noop
        return out
    theta = math.atan2(a[1], a[0])
    phi = math.atan2(a[3], a[2])
    speed, dur = float(a[4]), float(a[5])
    _, d, n = push_geometry(half, theta, phi)
    cos_in = max(abs(float(np.dot(d, -n))), 0.2)
    t_c = min(gap / cos_in / max(speed, 1e-3), dur)  # contact time
    for i, h in enumerate(horizons):
        if h <= t_c:
            s, v = 0.0, 0.0
        elif h <= dur:
            s, v = speed * (h - t_c), speed
        else:
            s_push = speed * (dur - t_c)
            v0 = speed if dur > t_c else 0.0
            ts = v0 / max(mu * G, 1e-6)
            tt = min(h - dur, ts)
            s = s_push + v0 * tt - 0.5 * mu * G * tt * tt
            v = max(0.0, v0 - mu * G * (h - dur))
        out[i, 0:2] = d * s
        out[i, 4:6] = d * v
    return out


V0_GRID = np.linspace(0.0, 0.6, 61)  # NOTE: v3 speeds reach 1.0 m/s (grid kept as used for reported rows)
MODEL_ERR = 0.004  # m, unmodelled contact transients/rotation (set on train data)


def _push_end_times(ev, b, mode="zero"):
    """(t_on, t_end) pairs (t=-age) from the actor's own COMMAND events.
    mode "zero" (used for ALL reported analytic results): t_end = command goes nonzero -> zero, t_on = previous
    command change (in v2/v3, where the pusher retracts after the push, this is the END OF THE RETRACT, not the
    release; kept for reproducibility of the reported rows). mode "release": t_on = push start, t_end = the next
    command change (true release). Post-hoc check on v2/v3 train: "release" mode with the grid estimator is
    strongly biased (abs err 0.13-0.17 vs prior 0.094-0.097), so it was not adopted."""
    m = ev["mask"][b] & (ev["prop"][b] == int(Prop.COMMAND))
    if not m.any():
        return []
    t = -ev["age"][b][m]
    v = np.linalg.norm(ev["value"][b][m][:, :2], axis=1)
    o = np.argsort(t, kind="stable")
    t, v = t[o], v[o]
    if mode == "zero":
        return [(float(t[i - 1]), float(t[i])) for i in range(1, len(t)) if v[i] < 1e-6 and v[i - 1] > 1e-6]
    out = []
    for i in range(len(t) - 1):
        if v[i] > 1e-6 and (i == 0 or v[i - 1] < 1e-6):
            out.append((float(t[i]), float(t[i + 1])))
    return out


def estimate_friction(ev, b, tgt, prior=MU_PRIOR, window=1.0, pre=0.25):
    """grid posterior over mu from the free-slide phase after each past push (push ends are known from the
    actor's own COMMAND events). Within [t_end+skip, t_end+window] the target displacement along its motion
    direction is fit by s0 + v0*tau - mu*g*tau^2/2 (tau < v0/(mu g), constant afterwards) on a (mu, v0) grid
    with s0 profiled out. returns (mean, var, n_informative_points)."""
    pm = 0.5 * (prior[0] + prior[1])
    pv = (prior[1] - prior[0]) ** 2 / 12.0
    tr = _target_track(ev, b, tgt)
    ends = _push_end_times(ev, b)
    if tr is None or not ends:
        return pm, pv, 0
    t, v, var = tr
    xy = v[:, :2].astype(np.float64)
    sig2 = float(np.median(var)) + 1e-8
    ll = np.zeros((len(MU_GRID), len(V0_GRID)))
    n_inf = 0
    for t_on, te in ends:
        m = (t >= max(te - pre, t_on + 0.05)) & (t <= te + window)
        if m.sum() < 5:
            continue
        ts, p = t[m] - te, xy[m]
        dvec = p[-1] - p[0]
        nd = np.linalg.norm(dvec)
        if nd < 4 * math.sqrt(sig2):  # no visible slide: weakly informative (object stopped quickly / not moved)
            continue
        u = dvec / nd
        sproj = (p - p[0]) @ u
        a = MU_GRID[:, None, None] * G
        v0 = V0_GRID[None, :, None]
        tau = np.minimum(np.maximum(ts, 0.0)[None, None, :], v0 / a)
        model = v0 * tau - 0.5 * a * tau * tau + v0 * np.minimum(ts, 0.0)[None, None, :]  # pushed phase: constant v0
        res = sproj[None, None, :] - model
        res = res - res.mean(-1, keepdims=True)
        ll += -0.5 * (res**2).sum(-1) / (sig2 + MODEL_ERR**2)
        n_inf += int(m.sum())
    if n_inf == 0:
        return pm, pv, 0
    lp = -0.5 * (MU_GRID - pm) ** 2 / pv
    lw = ll + lp[:, None]
    w = np.exp(lw - lw.max())
    wm = w.sum(1)
    wm /= wm.sum()
    mean = float((wm * MU_GRID).sum())
    var = float((wm * (MU_GRID - mean) ** 2).sum()) + 1e-4
    return mean, var, n_inf


def _ev_np(events: EventTensors):
    return {k: _np(getattr(events, k)) for k in ("entity", "prop", "value", "logvar", "age", "mask", "ent_type", "ent_mask")}


class _Analytic:
    name = "analytic"
    oracle = False

    def __init__(self):
        self.res_logvar = np.full((len(HORIZONS), STATE_DIM), -6.0)

    def eval(self):
        return self

    def to(self, device):
        return self

    def predict_np(self, events, query, labels=None):
        raise NotImplementedError

    def __call__(self, events: EventTensors, query: QueryTensors, passes=None, labels=None) -> ModelOutput:
        mean, fr_m, fr_v = self.predict_np(events, query, labels)
        dev = query.action.device if torch.is_tensor(query.action) else "cpu"
        B, H = mean.shape[:2]
        lv = np.broadcast_to(self.res_logvar[None, :H], mean.shape).copy()
        if fr_v is not None:
            # propagate friction uncertainty through finite difference of kinematics
            lv = np.logaddexp(lv, np.log(self._dmu2 + 1e-12) + np.log(fr_v)[:, None, None])
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
        return ModelOutput(
            mean=t(mean),
            logvar=t(lv),
            friction_mean=t(fr_m) if fr_m is not None else None,
            friction_logvar=t(np.log(fr_v)) if fr_v is not None else None,
            contact_logits=None,
            cost={"passes": 0},
        )

    _dmu2 = 0.0

    def calibrate(self, batches, max_batches=40):
        """per-horizon/dim residual variance on TRAIN batches (no friction-var propagation)."""
        r2, n = 0.0, 0.0
        for i, (ev, q, lab) in enumerate(batches):
            if i >= max_batches:
                break
            mean, _, _ = self.predict_np(ev, q, lab if self.oracle else None)
            y = _np(lab.future).astype(np.float64)
            fm = _np(lab.future_mask).astype(np.float64)[..., None]
            r2 = r2 + ((mean - y) ** 2 * fm).sum(0)
            n = n + fm.sum(0)
        self.res_logvar = np.log(np.asarray(r2) / np.maximum(n, 1) + 1e-10)
        return self


class ConstantVelocity(_Analytic):
    name = "constvel"

    def predict_np(self, events, query, labels=None):
        ev = _ev_np(events)
        tg = _np(query.target)
        hz = _np(query.horizons).astype(np.float64)
        B, H = hz.shape
        out = np.zeros((B, H, STATE_DIM))
        out[..., 2] = 1.0
        for b in range(B):
            v = _latest_vel(ev, b, int(tg[b]))
            out[b, :, 0] = v[0] * hz[b]
            out[b, :, 1] = v[1] * hz[b]
            out[b, :, 2] = np.cos(v[2] * hz[b])
            out[b, :, 3] = np.sin(v[2] * hz[b])
            out[b, :, 4:7] = v
        return out, None, None


class CoulombAnalytic(_Analytic):
    name = "coulomb"

    def __init__(self, prior=MU_PRIOR):
        super().__init__()
        self.prior = prior
        self._dmu2 = 0.0

    def frictions(self, ev, tg, labels):
        B = len(tg)
        res = [estimate_friction(ev, b, int(tg[b]), self.prior) for b in range(B)]
        return np.array([r[0] for r in res]), np.array([r[1] for r in res])

    def predict_np(self, events, query, labels=None):
        ev = _ev_np(events)
        tg = _np(query.target)
        hz = _np(query.horizons).astype(np.float64)
        act = _np(query.action).astype(np.float64)
        B, H = hz.shape
        fm, fv = self.frictions(ev, tg, labels)
        out = np.zeros((B, H, STATE_DIM))
        d2 = np.zeros((B, H, STATE_DIM))
        for b in range(B):
            half = _size(ev, b, int(tg[b]))
            gap = PUSH_GAP
            out[b] = push_slide_kinematics(act[b], half, fm[b], hz[b], gap)
            hi = push_slide_kinematics(act[b], half, fm[b] + 0.02, hz[b], gap)
            d2[b] = ((hi - out[b]) / 0.02) ** 2
        self._dmu2 = d2
        return out, fm, fv


class CoulombOracle(CoulombAnalytic):
    """ORACLE: given the TRUE friction from labels. Upper bound only; never a comparison row."""

    name = "coulomb_oracle"
    oracle = True

    def frictions(self, ev, tg, labels):
        if labels is None:
            raise ValueError("oracle requires labels")
        f = _np(labels.friction).astype(np.float64)
        return f, np.full_like(f, 1e-6)


def estimate_friction_closed_form(ev, b, tgt, prior=MU_PRIOR, pre=0.2, rest_from=0.3, rest_to=0.9, model_err=0.003):
    """closed-form Coulomb estimate per past push: release speed v0 = slope of a line fit to the target track over
    the last `pre` s of the push; slide distance d = (median rest position after release) - (line extrapolated to
    release); mu = v0^2 / (2 g d). Segments combined with the prior by precision weighting in log-mu space
    (delta-method variances with `model_err` m added to position noise). returns (mean, var, n_segments)."""
    pm = 0.5 * (prior[0] + prior[1])
    pv = (prior[1] - prior[0]) ** 2 / 12.0
    tr = _target_track(ev, b, tgt)
    ends = _push_end_times(ev, b, mode="release")
    if tr is None or not ends:
        return pm, pv, 0
    t, v, var = tr
    xy = v[:, :2].astype(np.float64)
    sig = math.sqrt(float(np.median(var)) + model_err**2)
    lm0, lv0 = math.log(pm), pv / pm**2
    num, den = lm0 / lv0, 1.0 / lv0
    nseg = 0
    for t_on, te in ends:
        m1 = (t >= max(te - pre, t_on + 0.05)) & (t <= te + 1e-6)
        m2 = (t >= te + rest_from) & (t <= te + rest_to)
        if m1.sum() < 3 or m2.sum() < 3:
            continue
        T1 = t[m1] - te
        A_ = np.stack([np.ones_like(T1), T1], 1)
        coef, *_ = np.linalg.lstsq(A_, xy[m1], rcond=None)  # (2,2): intercept, velocity
        vel = coef[1]
        v0 = float(np.linalg.norm(vel))
        if v0 < 0.05:
            continue
        u = vel / v0
        s_te = 0.0
        s_rest = float(np.median((xy[m2] - coef[0]) @ u))
        d = s_rest - s_te
        if d < 0.002:
            continue
        mu = v0**2 / (2 * G * d)
        if not (0.02 < mu < 3.0):
            continue
        sv = sig * math.sqrt(12.0 / max(m1.sum(), 3)) / max(T1.max() - T1.min(), 0.05)
        sd_ = sig * math.sqrt(1.0 / m2.sum() + 1.0 / m1.sum())
        lv = (2 * sv / v0) ** 2 + (sd_ / d) ** 2
        num += math.log(mu) / lv
        den += 1.0 / lv
        nseg += 1
    lm, lvar = num / den, 1.0 / den
    mean = math.exp(lm + 0.5 * lvar)
    return mean, mean**2 * lvar + 1e-5, nseg
