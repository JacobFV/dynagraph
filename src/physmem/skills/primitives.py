"""Robot-agnostic parameterized skills. Each takes (ctx, call, est) and returns api.SkillResult.

Conventions: TCP z = approach direction, TCP y = finger closing axis (see embodiments/robots.py). Every motion is a
min-jerk / blended task-space trajectory sent to the controller; contact phases use guarded / force / impedance modes.
Inputs are actor-legal: ObjectEstimate (pose tracker + geometry estimate) and SkillCall params. Grasp state is read
from proprioception (gripper closure, pad contacts, wrist F/T).
"""
from __future__ import annotations

import math

import numpy as np

from physmem.embodiments.ik import down_quat
from physmem.embodiments.trajectory import BlendedTrajectory, make_traj, quat_err, mat2quat, quat2mat
from physmem.skills.base import ObjectEstimate, SkillContext, SkillRun

W_MAX = 1.5  # rad/s nominal peak angular rate for reorientation


def _eff(ctx, call):
    return call.effector or next(iter(ctx.emb.eff))


def _tcp(ctx, eff):
    return ctx.emb.tcp_pose(ctx.data, eff)


def _T(dist, v, ang=0.0, tmin=0.25):
    """rest-to-rest min-jerk duration with peak speed v (peak = 1.875 * mean)."""
    return max(tmin, 1.875 * dist / max(v, 1e-3), 1.875 * ang / W_MAX)


def _grip_cfg(ctx, eff):
    return ctx.emb.eff[eff]['cfg'].gripper


def _wrap(a):
    return math.remainder(a, 2 * math.pi)


def _yaw_of_closing(q):
    R = quat2mat(q)
    return math.atan2(R[1, 1], R[0, 1])


def _near_yaw(ctx, eff, yaw):
    """pick yaw or yaw+pi (same grasp for a symmetric gripper), whichever needs less wrist turn (no wrist flips)."""
    _, q = _tcp(ctx, eff)
    cur = _yaw_of_closing(q)
    return yaw if abs(_wrap(yaw - cur)) <= abs(_wrap(yaw + math.pi - cur)) else yaw + math.pi


def _quat_axes(z, y):
    z = np.asarray(z, float) / np.linalg.norm(z)
    y = np.asarray(y, float)
    y = y - z * (y @ z)
    y /= np.linalg.norm(y)
    return mat2quat(np.column_stack([np.cross(y, z), y, z]))


def _move(ctx, eff, pts, v, q=None, gripper=None, stop_last=True, until=None, max_t=None, via_vel=None, durs=None):
    """go through points (blended), orientation q (or current) at each; returns run status."""
    p0, q0 = _tcp(ctx, eff)
    ref = ctx.ctrl.reference(eff)
    if ref is not None:
        p0 = ref[0]
    q = q0 if q is None else q
    wps, durations, prev, qprev = [], [], np.asarray(p0, float), q0
    for i, p in enumerate(pts):
        qq = q if not isinstance(q, list) else q[i]
        ang = float(np.linalg.norm(quat_err(qq, qprev)))
        durations.append(_T(float(np.linalg.norm(np.asarray(p) - prev)), v, ang))
        wps.append((np.asarray(p, float), qq))
        prev, qprev = np.asarray(p, float), qq
    if durs is not None:
        durations = list(durs)
    tr = BlendedTrajectory(effector=eff, waypoints=wps, durations=durations, gripper=gripper, via_vel=via_vel)
    ctx.ctrl.command(tr)
    return ctx.run(until=until, max_t=max_t if max_t is not None else sum(durations) / 1.0 * 3 + 2.0, effector=eff)


def _close(ctx, eff, g=1.0, t_max=1.5):
    """command the gripper and step until an EVENT: (closing) pad contact + the fingers have stopped, or fully
    closed; (opening) open. Bounded by t_max, then a 0.1 s settle. Event-based, not a convergence wait on a pose."""
    ctx.ctrl.set_gripper(eff, g)
    gc = _grip_cfg(ctx, eff)
    if gc.kind == 'suction':
        ctx.wait(0.15)
        return
    t_end = ctx.data.time + t_max
    prev, still = None, 0
    while ctx.data.time < t_end:
        ctx.step(10)
        c, n = ctx.emb.gripper_state(ctx.model, ctx.data, eff)
        still = still + 1 if (n > 0 and prev is not None and abs(c - prev) < 2e-3) else 0
        t_min = ctx.data.time >= t_end - t_max + gc.close_time   # at least the nominal close time (force build-up)
        if g >= 0.5 and (c >= 0.98 or (still >= 3 and t_min)):     # touching and stalled for 60 ms
            break
        if g < 0.5 and (c <= 0.05 or (prev is not None and abs(c - prev) < 1e-4 and n == 0)):
            break
        prev = c
    ctx.wait(0.15)


def _held(ctx):
    if not hasattr(ctx, 'held'):
        ctx.held = {}
    return ctx.held


def _box_support(h, d):
    """distance from the centre to the box boundary along unit 2D direction d (object frame)."""
    tx = h[0] / abs(d[0]) if abs(d[0]) > 1e-9 else np.inf
    ty = h[1] / abs(d[1]) if abs(d[1]) > 1e-9 else np.inf
    return min(tx, ty)


def _clear(ctx, k=0.08):
    """approach / transit clearance scaled to the robot's reach (short arms cannot hover 8 cm above low targets)."""
    return float(min(k, k * ctx.emb.spec.reach_m / 0.6))


def _guard_n(ctx, p, default=10.0):
    return float(p.get('guard', min(default, 0.5 * ctx.emb.spec.max_ee_force)))


def _reachable(ctx, eff, pos, q, tol=0.01):
    """kinematic IK check from the current posture (scratch data; the live sim is untouched)."""
    from physmem.embodiments.ik import solve_ik
    if not hasattr(ctx, '_ikd'):
        import mujoco
        ctx._ikd = mujoco.MjData(ctx.model)
    e = ctx.emb.eff[eff]
    rw = e['cfg'].rot_weight
    good = lambda ep, er: ep < tol and (er < 0.1 or rw < 0.5)
    _, ok, ep, er = solve_ik(ctx.emb, ctx.model, ctx.data, eff, pos, q, iters=300, scratch=ctx._ikd)
    ctx._ik_last = (ep, er)
    if good(ep, er):
        return True
    # continuation: solve from above and step down (the way the arm actually approaches); escapes local minima of a
    # single solve started at home (e.g. reaching down below a pedestal-mounted base)
    starts = [None]
    if len(e['home']):
        starts.append(np.asarray(e['home'], float))
    for q_start in starts:
        if q_start is not None:
            _, ok, ep, er = solve_ik(ctx.emb, ctx.model, ctx.data, eff, pos, q, q0=q_start, iters=300, scratch=ctx._ikd)
            if good(ep, er):
                return True
        q0 = q_start
        for dz in (0.3, 0.2, 0.12, 0.06, 0.0):
            q0, ok, ep, er = solve_ik(ctx.emb, ctx.model, ctx.data, eff, np.asarray(pos) + [0, 0, dz], q, q0=q0,
                                      iters=300, scratch=ctx._ikd)
        ctx._ik_last = (ep, er)
        if good(ep, er):
            return True
    return False


# ============================================================================ reach
def reach(ctx: SkillContext, call, est: ObjectEstimate = None):
    eff, p = _eff(ctx, call), call.params
    with SkillRun(ctx, 'reach') as r:
        tgt = np.array([p['x'], p['y'], p['z']], float)
        if est is not None and p.get('relative', False):
            tgt = est.p + est.R @ tgt
        q = down_quat(_near_yaw(ctx, eff, p['yaw'])) if 'yaw' in p else None
        if 'mode' in p:
            ctx.ctrl.set_mode(p['mode'], eff)
        st = _move(ctx, eff, [tgt], p.get('speed', 0.3), q, gripper=p.get('gripper'))
        err = float(np.linalg.norm(_tcp(ctx, eff)[0] - tgt))
        return r.result(st == 'done', st, final_err=err)


# ============================================================================ pick
def pick(ctx: SkillContext, call, est: ObjectEstimate):
    eff, p = _eff(ctx, call), call.params
    gc = _grip_cfg(ctx, eff)
    with SkillRun(ctx, 'pick') as r:
        v = p.get('speed', 0.3)
        R = est.R
        yaw_o = math.atan2(R[1, 0], R[0, 0])
        hx, hy, hz = est.h
        side = int(p.get('grasp', 0)) == 1
        if hx <= hy:
            cy, width = yaw_o, 2 * hx
        else:
            cy, width = yaw_o + math.pi / 2, 2 * hy
        cy += p.get('yaw', 0.0)
        vel = np.asarray(p.get('vel', (0.0, 0.0, 0.0)), float)
        if np.linalg.norm(vel[:2]) > 1e-3 and int(p.get('grasp', 0)) == 0:
            return _moving_pick(ctx, eff, p, est, r, width, cy, vel)
        if gc.kind == 'suction':
            return _suction_pick(ctx, eff, p, est, r, width, cy)
        if width > 0.95 * gc.max_width:
            return r.result(False, 'too_wide', width=width, max_width=gc.max_width)
        top, bottom = est.top_z(), est.bottom_z()
        ctx.ctrl.set_mode('track', eff)
        ctx.ctrl.set_gripper(eff, 0.0)
        if not side:
            # default: keep the linkage/palm above a flat top; an explicit depth (e.g. a shank below a head) is honoured
            depth = p['depth'] if 'depth' in p else min(float(np.clip(hz, 0.015, 0.035)), gc.link_h - 0.004)
            gz = max(bottom + gc.tip + p.get('clear', 0.003), top - depth)
            y0 = _near_yaw(ctx, eff, cy)
            base_xy = np.asarray(getattr(ctx.emb, 'mount_pose', (np.zeros(3),))[0], float)[:2]
            rad = est.p[:2] - base_xy
            rad = np.array([*(rad / (np.linalg.norm(rad) + 1e-9)), 0.0])
            tilts = (0.0, 0.35, 0.7)     # vertical first; tilted (outward) approaches only if out of reach / limits
            cands = [(yq, tl) for tl in tilts for yq in (y0, y0 + math.pi)]
            for yq, tl in cands:     # nearer yaw first; the other yaw / a tilted approach only if out of reach
                q = down_quat(yq)
                if tl:  # tilt the approach outward (TCP z toward -z cos + radial sin), for low-dof arms
                    from physmem.embodiments.trajectory import quat_from_rotvec, quat_mul
                    q = quat_mul(quat_from_rotvec(np.cross([0, 0, 1.0], rad) * tl), q)
                g = np.array([est.p[0], est.p[1], gz])
                if gc.jaw_dir:  # single moving jaw: put the fixed jaw's face at the object's side
                    g[:2] -= gc.jaw_dir * (width / 2 + p.get('jaw_margin', 0.008)) * np.array([math.cos(yq), math.sin(yq)])
                pre = g + np.array([0, 0, top - gz + p.get('pre', _clear(ctx))])
                if _reachable(ctx, eff, g, q) and _reachable(ctx, eff, pre, q):
                    break
            approach = np.array([0, 0, -1.0])
        else:  # side grasp: approach horizontally along the closing-orthogonal axis, fingers across the narrow side
            app = np.array([math.cos(cy + math.pi / 2), math.sin(cy + math.pi / 2), 0.0])
            base = _tcp(ctx, eff)[0]
            if (est.p - base)[:2] @ app[:2] < 0:
                app = -app
            q = _quat_axes(app, (math.cos(cy), math.sin(cy), 0))
            gz = bottom + min(hz, 0.04)
            g = np.array([est.p[0], est.p[1], gz])
            pre = g - app * (np.abs(R[:2, :2].T @ app[:2]) @ est.h[:2] + 0.08)
            approach = app
        if not _reachable(ctx, eff, g, q) or not _reachable(ctx, eff, pre, q):
            return r.result(False, 'unreachable', target=g.tolist(), ik=getattr(ctx, '_ik_last', None))
        hover = pre + np.array([0, 0, 0.02])
        # pre-shape: open only as far as needed (a wide-swinging jaw would hit the part or the table)
        g_pre = float(np.clip(1.0 - (width + p.get('open_margin', 0.015 if gc.jaw_dir else 0.03)) / gc.max_width, 0.0, 0.6))
        st = _move(ctx, eff, [hover, pre], v, q, gripper=g_pre)
        if st not in ('done',):
            return r.result(False, st, phase='approach')
        ctx.ctrl.set_mode('guarded', eff, threshold=_guard_n(ctx, p), axis=tuple(approach))
        st = _move(ctx, eff, [g], p.get('approach_speed', 0.08), q)
        tripped = ctx.ctrl.tripped(eff)
        tcp_now = _tcp(ctx, eff)[0]
        miss = float(np.linalg.norm(tcp_now - g))
        ctx.ctrl.set_mode('track', eff)
        _close(ctx, eff, p.get('squeeze', 1.0))
        c, n = ctx.emb.gripper_state(ctx.model, ctx.data, eff)
        if c > 0.93 or n == 0:
            _move(ctx, eff, [pre], v, q, gripper=0.0)
            return r.result(False, 'grasp_failed', closure=c, contacts=n, tripped=tripped, miss=miss)
        lift = p.get('lift', _clear(ctx, 0.10))
        top_pt = _tcp(ctx, eff)[0] + np.array([0, 0, lift])
        st = _move(ctx, eff, [top_pt], min(v, p.get('lift_speed', 0.2)), q)
        c2, n2 = ctx.emb.gripper_state(ctx.model, ctx.data, eff)
        held = n2 > 0 and c2 < 0.95
        if held:
            tz = _tcp(ctx, eff)[0][2]
            Rg = quat2mat(q)
            _held(ctx)[eff] = dict(offset=float(gz - bottom), half_size=tuple(est.h), side=side, width=width,
                                   obj_in_tcp=tuple(Rg.T @ (np.array([est.p[0], est.p[1], gz]) - g)))
        return r.result(held, 'done' if held else 'dropped', closure=c2, contacts=n2, tripped=tripped, miss=miss,
                        grasp_width_est=width)


def _moving_pick(ctx, eff, p, est, r, width, cy, vel):
    """pick a part riding a belt (estimate `est` taken `age` s ago, xy velocity `vel` estimated from pose events):
    intercept planning + velocity feed-forward. One blended trajectory: hover point at the intercept, reached WITH the
    belt velocity; descent while tracking; the fingers close while the hand keeps pace (closing time from the
    gripper), then a lift that keeps the belt velocity and decelerates. No guarded mode (the hand moves with the
    part); the grasp check is proprioceptive (closure + pad contacts)."""
    gc = _grip_cfg(ctx, eff)
    v = np.array([vel[0], vel[1], 0.0])
    sp = min(float(p.get('speed', 0.3)), 0.3)   # the belt plan is not retimed: keep the approach trackable
    top, bottom = est.top_z(), est.bottom_z()
    suction = gc.kind == 'suction'
    if suction:
        gz = top - p.get('press', 0.004)               # press the cup lip onto the moving part's top face
    else:
        depth = p['depth'] if 'depth' in p else min(float(np.clip(est.h[2], 0.015, 0.035)), gc.link_h - 0.004)
        gz = max(bottom + gc.tip + p.get('clear', 0.003), top - depth)
    hov = p.get('pre', 0.05)
    q = down_quat(_near_yaw(ctx, eff, cy))
    if suction:
        mask = _suction_zones(ctx, est, q)
        if not mask.any():
            return r.result(False, 'too_small', width=width)
        ctx.emb.eff[eff]['zone_mask'] = mask
    tcp, _ = _tcp(ctx, eff)
    age = float(p.get('age', 0.0))
    T_desc = max(0.25, float(p.get('t_desc', 0.6)))
    T_match = float(p.get('t_match', 0.5))      # hover over the part at belt speed before descending (lets lag settle)
    T_close = gc.close_time + 0.35
    # intercept: the hand reaches the hover point at time T_app, when the part is there (fixed point iteration)
    T_app = 1.0
    for _ in range(4):
        hp = est.p + v * (age + T_app)
        T_app = max(0.4, 1.875 * float(np.linalg.norm(hp[:2] - tcp[:2]) + abs(top + hov - tcp[2])) / sp)
    lift = p.get('lift', _clear(ctx, 0.10))
    T_lift = max(0.5, 2 * lift / max(p.get('lift_speed', 0.2), 1e-3))
    # earliest REACHABLE intercept: a part still upstream is met later (the hand waits on its way), one that has
    # passed the workspace is reported unreachable
    ok = False
    for extra in np.arange(0.0, float(p.get('max_wait', 6.0)) + 1e-9, 0.5):
        T_i = T_app + extra
        p_hover = np.array([*(est.p[:2] + v[:2] * (age + T_i)), gz + hov])
        p_match = p_hover + v * T_match
        p_grasp = np.array([*(p_match[:2] + v[:2] * T_desc), gz])
        p_hold = p_grasp + v * (T_close + 0.4)
        p_lift = p_hold + v * T_lift * 0.5 + np.array([0, 0, lift])
        if all(_reachable(ctx, eff, x, q) for x in (p_hover, p_grasp, p_lift)):
            ok, T_app = True, T_i
            break
    if not ok:
        return r.result(False, 'unreachable', intercept=p_hover.tolist())
    g_pre = 0.0 if suction else \
        float(np.clip(1.0 - (width + p.get('open_margin', 0.015 if gc.jaw_dir else 0.03)) / gc.max_width, 0.0, 0.6))
    ctx.ctrl.set_mode('track', eff)
    ctx.ctrl.set_gripper(eff, g_pre)
    tr = BlendedTrajectory(effector=eff, waypoints=[(p_hover, q), (p_match, q), (p_grasp, q), (p_hold, q), (p_lift, q)],
                           durations=[T_app, T_match, T_desc, T_close + 0.4, T_lift], via_vel=[v, v, v, v, np.zeros(3)])
    tr.no_retime = True       # the belt sets the timing
    pred = ctx.ctrl.command(tr, preempt=True)
    k = pred.get('scale', 1.0)
    T_app = T_app + T_match
    t_close = ctx.data.time + (T_app + T_desc) * k
    ctx.run(until=lambda c: c.data.time >= t_close, max_t=(T_app + T_desc) * k + 1.0, effector=eff)
    # close when the fingers are actually at grasp height (the hand keeps pace with the belt meanwhile), <= 0.4 s
    ctx.run(until=lambda c: _tcp(c, eff)[0][2] <= gz + 0.006, max_t=0.4, effector=eff)
    ctx.ctrl.set_gripper(eff, p.get('squeeze', 1.0))
    tcp_close = _tcp(ctx, eff)[0].copy()
    t_cl = float(ctx.data.time)
    pred_close = est.p + v * (age + (t_cl - (t_close - (T_app + T_desc) * k)))
    st = ctx.run(max_t=(T_close + T_lift) * k + 1.5, effector=eff)
    c, n = ctx.emb.gripper_state(ctx.model, ctx.data, eff)
    held = (c > 0.0) if suction else (n > 0 and c < 0.95)
    if held:
        here = _tcp(ctx, eff)[0]
        _held(ctx)[eff] = dict(offset=float(gz - bottom), half_size=tuple(est.h), side=False, width=width,
                               obj_in_tcp=(0.0, 0.0, 0.0))
    else:
        ctx.ctrl.set_gripper(eff, 0.0)
    return r.result(held, 'done' if held else 'grasp_failed', closure=c, contacts=n, intercept_t=T_app, retime=k,
                    t_close=t_cl, tcp_close=tcp_close.round(4).tolist(), pred_item_close=np.round(pred_close, 4).tolist(),
                    grasp_z=float(gz), vel=v[:2].round(4).tolist())


class _AccBudget:
    """decision knob `max_acc` (m/s^2): the controller's retiming acceleration budget for this skill (a suction /
    grip load needs F >= m (g + a)); restored afterwards."""

    def __init__(self, ctx, p):
        self.ctx, self.a = ctx, p.get('max_acc')

    def __enter__(self):
        if self.a is not None:
            self.old = dict(self.ctx.ctrl.limits)
            lim = self.ctx.ctrl.limits
            lim['a'] = float(self.a)
            lim['j'] = min(lim['j'], 20.0 * float(self.a))

    def __exit__(self, *exc):
        if self.a is not None:
            self.ctx.ctrl.limits.clear()
            self.ctx.ctrl.limits.update(self.old)
        return False


def _suction_zones(ctx, est, q):
    """zone control from the actor's geometry estimate: enable only the cups whose whole lip lies on the top face
    (TCP over the face centre with orientation q)."""
    from physmem.embodiments.core import suction_cups
    tool = ctx.emb.cfg.tool
    cups = suction_cups(tool)
    radii = list(tool.get('cup_radii', [float(tool.get('radius', 0.035))] * len(cups)))
    Rq, Ro = quat2mat(q), est.R
    out = []
    ctx._zone_dbg = dict(h=np.round(est.h, 3).tolist(), us=[])
    for (ox, oy), rad in zip(cups, radii):
        u = Ro.T @ (Rq[:, 0] * ox + Rq[:, 1] * oy)
        ctx._zone_dbg['us'].append(np.round(u[:2], 3).tolist())
        # the whole lip must sit >= 8 mm inside the estimated face (tracker pose/size noise of a few mm)
        out.append(abs(u[0]) <= est.h[0] - rad - 0.008 and abs(u[1]) <= est.h[1] - rad - 0.008)
    return np.array(out, bool)


def _suction_pick(ctx, eff, p, est, r, width, cy):
    with _AccBudget(ctx, p):
        return _suction_pick_(ctx, eff, p, est, r, width, cy)


def _suction_pick_(ctx, eff, p, est, r, width, cy):
    """vacuum pick: cup onto the top face centre (guarded), vacuum on, seal check, lift, seal check."""
    from physmem.embodiments.core import suction_cups
    tool = ctx.emb.cfg.tool
    rad = float(tool.get('radius', 0.035))
    cups = suction_cups(tool)
    v = p.get('speed', 0.3)
    top, bottom = est.top_z(), est.bottom_z()
    q = down_quat(_near_yaw(ctx, eff, cy))
    mask = _suction_zones(ctx, est, q)
    if not mask.any():
        return r.result(False, 'too_small', width=width, cup=2 * min(ctx.emb.cfg.tool.get('cup_radii', [rad])))
    ctx.emb.eff[eff]['zone_mask'] = mask
    need = float(p.get('min_zones', 1))
    if mask.sum() < need:
        return r.result(False, 'too_small', zones=int(mask.sum()))
    g = np.array([est.p[0], est.p[1], top - p.get('press', 0.003)])
    pre = g + np.array([0, 0, p.get('pre', _clear(ctx))])
    if not _reachable(ctx, eff, g, q) or not _reachable(ctx, eff, pre, q):
        return r.result(False, 'unreachable')
    ctx.ctrl.set_mode('track', eff)
    ctx.ctrl.set_gripper(eff, 0.0)
    st = _move(ctx, eff, [pre + np.array([0, 0, 0.02]), pre], v, q)
    if st != 'done':
        return r.result(False, st, phase='approach')
    ctx.ctrl.set_mode('guarded', eff, threshold=_guard_n(ctx, p, 8.0), axis=(0, 0, -1))
    _move(ctx, eff, [g - np.array([0, 0, 0.01])], p.get('approach_speed', 0.06), q)
    tripped = ctx.ctrl.tripped(eff)
    ctx.ctrl.set_mode('track', eff)
    touch = _tcp(ctx, eff)[0]
    ctx.ctrl.set_gripper(eff, 1.0)
    _move(ctx, eff, [touch - np.array([0, 0, p.get('press', 0.003)])], 0.03, q)   # compress the cup lip
    ctx.wait(0.1)
    sealed, n = ctx.emb.cup_seal(ctx.model, ctx.data, eff)
    if not sealed:
        ctx.ctrl.set_gripper(eff, 0.0)
        _move(ctx, eff, [pre], v, q)
        return r.result(False, 'no_seal', contacts=n, tripped=tripped)
    here = _tcp(ctx, eff)[0]
    _move(ctx, eff, [here + np.array([0, 0, p.get('lift', _clear(ctx, 0.10))])], min(v, p.get('lift_speed', 0.2)), q)
    c, n = ctx.emb.gripper_state(ctx.model, ctx.data, eff)
    held = c > 0.0          # at least one enabled cup still sealed and pulling
    r.log['zones'] = getattr(ctx, '_zone_dbg', None)
    if held:
        _held(ctx)[eff] = dict(offset=float(here[2] - bottom), half_size=tuple(est.h), side=False, width=width,
                               obj_in_tcp=tuple(quat2mat(q).T @ (np.array([est.p[0], est.p[1], here[2]]) - here)))
    return r.result(held, 'done' if held else 'dropped', tripped=tripped, contacts=n)


# ============================================================================ place
def place(ctx: SkillContext, call, est: ObjectEstimate = None):
    with _AccBudget(ctx, call.params):
        return _place(ctx, call, est)


def _place(ctx: SkillContext, call, est: ObjectEstimate = None):
    """params: x, y, z (surface height at the target; or est = target region/fixture estimate), yaw (TCP closing yaw,
    optional), speed, guard (N), release (m of drop allowed), retreat (m)."""
    eff, p = _eff(ctx, call), call.params
    with SkillRun(ctx, 'place') as r:
        v = p.get('speed', 0.3)
        held = _held(ctx).get(eff)
        off = p.get('hold_offset', held['offset'] if held else 0.02)
        if est is not None and 'x' not in p:
            tx, ty, sz = est.p[0], est.p[1], est.top_z()
        else:
            tx, ty, sz = p['x'], p['y'], p['z']
        cur_p, cur_q = _tcp(ctx, eff)
        q = down_quat(_near_yaw(ctx, eff, p['yaw'])) if 'yaw' in p else cur_q
        if held and 'obj_in_tcp' in held:  # put the OBJECT (not the TCP) on the target
            o = quat2mat(q) @ np.asarray(held['obj_in_tcp'])
            tx, ty = tx - o[0], ty - o[1]
        if not _reachable(ctx, eff, np.array([tx, ty, sz + off]), q):
            return r.result(False, 'unreachable')
        goal_z = sz + off
        # carry high enough that the part's underside clears neighbours as tall as itself (placed cartons/parts)
        h_obj = 2 * float(held['half_size'][2]) if held else 0.0
        transit = max(cur_p[2], goal_z + max(_clear(ctx), h_obj + 0.05))
        ctx.ctrl.set_mode('track', eff)
        st = _move(ctx, eff, [np.array([cur_p[0], cur_p[1], transit]), np.array([tx, ty, transit]),
                              np.array([tx, ty, goal_z + 0.03])], v, q)
        if st != 'done':
            return r.result(False, st, phase='transit')
        F0, _, _ = ctx.emb.wrist_ft(ctx.model, ctx.data, eff)
        # guarded descent: trips when the object meets the surface (wrist force rises by `guard` over its baseline)
        ctx.ctrl.set_mode('guarded', eff, threshold=_guard_n(ctx, p, 6.0), axis=(0, 0, -1), bias=tuple(F0))
        st = _move(ctx, eff, [np.array([tx, ty, goal_z - 0.015])], p.get('approach_speed', 0.06), q)
        tripped = ctx.ctrl.tripped(eff)
        ctx.ctrl.set_mode('track', eff)
        gc = _grip_cfg(ctx, eff)
        # release by opening just past the part (a wide, fast jaw swing knocks light parts away)
        g_rel = 0.0 if gc.kind == 'suction' or not held or not gc.jaw_dir else \
            float(np.clip(1.0 - (held['width'] + p.get('open_margin', 0.015)) / gc.max_width, 0.0, 0.6))
        ctx.ctrl.set_gripper(eff, g_rel)
        ctx.wait(gc.close_time + 0.1)
        up = _tcp(ctx, eff)[0] + np.array([0, 0, p.get('retreat', _clear(ctx))])
        st = _move(ctx, eff, [up], min(v, 0.2), q, gripper=g_rel)
        _held(ctx).pop(eff, None)
        c, n = ctx.emb.gripper_state(ctx.model, ctx.data, eff)
        return r.result(st == 'done' and n == 0, 'done' if n == 0 else 'stuck', tripped=tripped)


# ============================================================================ push
def push(ctx: SkillContext, call, est: ObjectEstimate):
    """params (object frame): theta (push direction), offset (-1..1 along the contact face), speed (m/s cruise),
    distance (m of pushing after contact), height (TCP height above the surface, m), approach (m run-up)."""
    eff, p = _eff(ctx, call), call.params
    lim = ctx.ctrl.limits
    with SkillRun(ctx, 'push') as r:
        v = float(p.get('speed', 0.3))
        dist = float(p.get('distance', 0.1))
        th = float(p.get('theta', 0.0))
        R = est.R
        d_o = np.array([math.cos(th), math.sin(th)])
        s = _box_support(est.h, -d_o)
        perp_o = np.array([-d_o[1], d_o[0]])
        face_half = abs(perp_o[0]) * est.h[0] + abs(perp_o[1]) * est.h[1]
        c_o = -s * d_o + perp_o * float(np.clip(p.get('offset', 0.0), -1, 1)) * 0.8 * face_half
        Rz = R[:2, :2]
        d = np.array([*(Rz @ d_o), 0.0])
        d /= np.linalg.norm(d)
        sz = est.bottom_z()
        H = est.top_z() - sz
        gc = _grip_cfg(ctx, eff)
        hz = float(min(0.03, 0.6 * H))
        if 'height' in p:
            hz = float(p['height'])
        elif gc.link_h < 0.045:   # short fingers under a linkage (2F-85): push high enough that the linkage clears the top
            hz = float(min(0.75 * H, max(H - gc.link_h + 0.012, gc.tip + 0.004)))
        elif -0.006 <= hz + gc.link_h - H <= 0.01:   # the gripper's linkage/palm would catch the top edge
            lo = H - gc.link_h - 0.008
            hz = lo if lo >= gc.tip + 0.004 else H - gc.link_h + 0.012
        contact = np.array([*(est.p[:2] + Rz @ c_o), sz + hz])
        r_tip = float(p.get('tip_radius', 0.012))
        a_lim = lim['a'] * 0.6
        T_r = max(0.2, 3 * v / a_lim)
        L_r = max(float(p.get('approach', 0.0)), 0.5 * v * T_r)
        T_r = 2 * L_r / v
        T_d = max(0.15, 3 * v / a_lim)
        L_d = 0.5 * v * T_d
        cruise = max(dist - L_d, 0.01)
        touch = contact - d * r_tip
        start = touch - d * L_r
        end = touch + d * cruise
        stop = end + d * L_d
        along = p.get('closing_along', False)
        cy = _near_yaw(ctx, eff, math.atan2(d[1], d[0]) + (0.0 if along else math.pi / 2))
        q = down_quat(cy)
        if not all(_reachable(ctx, eff, x, q) for x in (start, touch, end, stop)):
            return r.result(False, 'unreachable', cmd_speed=v)
        ctx.ctrl.set_mode('track', eff)
        st = _move(ctx, eff, [start + np.array([0, 0, 0.08]), start], min(0.4, lim['v']), q, gripper=1.0)
        if st != 'done':
            return r.result(False, st, phase='approach')
        ctx.wait(0.05)
        t_c0 = ctx.data.time + T_r
        T_c = cruise / v
        tr = BlendedTrajectory(effector=eff, waypoints=[(touch, q), (end, q), (stop, q)], durations=[T_r, T_c, T_d],
                               via_vel=[d * v, d * v, np.zeros(3)])
        pred = ctx.ctrl.command(tr)
        samples, F_max = [], 0.0
        k = pred.get('scale', 1.0)

        def mon(c):
            nonlocal F_max
            t = c.data.time
            F_max = max(F_max, float(np.linalg.norm(c.ctrl.st[eff]['ft'])) if 'ft' in c.ctrl.st[eff] else 0.0)
            if t_c0 <= t <= t_c0 + T_c * k:
                vv, _ = c.emb.tcp_vel(c.model, c.data, eff)
                pp, _ = c.emb.tcp_pose(c.data, eff)
                ref = c.ctrl.reference(eff)
                samples.append((t, float(vv @ d), float(np.linalg.norm(pp - ref[0])), float(np.linalg.norm(vv - d * (vv @ d)))))
        ctx.ctrl.set_mode('track', eff, measure_ft=True)
        # force limit (params f_max [N], comply [bool]): the wrist F/T component opposing the push. Exceeded ->
        # comply: finish the stroke in low-stiffness impedance (presses with <= ~f_max instead of position-tracking
        # through a jammed cluster); else: abort and hold where the hand is (status 'force_limited').
        f_max = p.get('f_max')
        limited = {'hit': False, 't': None}

        def over(c):
            if f_max is None or limited['hit']:
                return False
            Fp = -float(c.ctrl.st[eff]['ft'] @ d)
            if Fp > float(f_max):
                limited['hit'], limited['t'] = True, float(c.data.time)
                if p.get('comply', False):
                    K = float(p.get('comply_k', max(50.0, float(f_max) / 0.02)))   # f_max over ~2 cm of lag
                    c.ctrl.set_mode('impedance', eff, stiffness=K, rot_stiffness=20.0, measure_ft=True)
                    return False
                return True
            return False
        st = ctx.run(monitor=mon, until=over, max_t=(T_r + T_c + T_d) * k + 2.0, effector=eff)
        if st == 'until':        # abort: hold where we are
            here, hq = _tcp(ctx, eff)
            ctx.ctrl.command(BlendedTrajectory(effector=eff, waypoints=[(here, hq)], durations=[0.2]), preempt=True)
            ctx.run(max_t=0.4, effector=eff)
        ctx.ctrl.set_mode('track', eff)
        st2 = _move(ctx, eff, [_tcp(ctx, eff)[0] + np.array([0, 0, 0.06])], 0.3, q)
        S = np.array(samples) if samples else np.zeros((0, 4))
        log = dict(cmd_speed=v, retime=k, F_max=F_max, force_limited=limited['hit'], t_force_limit=limited['t'])
        if len(S) > 6:
            n = len(S)
            mid = S[n // 4: 3 * n // 4]
            log.update(speed_ratio=float(np.mean(mid[:, 1]) / (v / k)), release_ratio=float(np.mean(S[-3:, 1]) / (v / k)),
                       track_err_mid_max=float(np.max(mid[:, 2])), track_err_mid_mean=float(np.mean(mid[:, 2])),
                       lateral_speed_mid=float(np.mean(mid[:, 3])))
        if st == 'until':
            return r.result(False, 'force_limited', **log)
        return r.result(st == 'done', st, **log)


# ============================================================================ sweep (bulk)
def sweep(ctx: SkillContext, call, est: ObjectEstimate = None):
    """drag the open gripper (fingers across the motion) through a region in lanes.
    params: x0, y0, x1, y1 (region, world), direction (rad, world), surface_z, height (TCP above surface), speed,
    lane (lane spacing m, default 0.8 * finger span), overshoot (m past the region), guard (N, jam stop)."""
    eff, p = _eff(ctx, call), call.params
    with SkillRun(ctx, 'sweep') as r:
        v = p.get('speed', 0.25)
        a = p.get('direction', 0.0)
        d = np.array([math.cos(a), math.sin(a), 0.0])
        n = np.array([-d[1], d[0], 0.0])
        corners = np.array([[p['x0'], p['y0']], [p['x1'], p['y0']], [p['x0'], p['y1']], [p['x1'], p['y1']]])
        along, across = corners @ d[:2], corners @ n[:2]
        span = p.get('paddle', 0.03)            # frontal width of the closed fingers (the 'flat effector')
        lane = p.get('lane', 0.8 * span)
        nl = max(1, int(math.ceil((across.max() - across.min()) / lane)))
        offs = across.min() + (np.arange(nl) + 0.5) * (across.max() - across.min()) / nl
        z = p.get('surface_z', 0.0) + max(p.get('height', 0.0), _grip_cfg(ctx, eff).tip + 0.004)
        q = down_quat(_near_yaw(ctx, eff, a + math.pi / 2))
        ctx.ctrl.set_gripper(eff, 1.0)
        jams, lanes_done = 0, 0
        for o in offs:
            s = d * (along.min() - p.get('margin', 0.05)) + n * o
            e = d * (along.max() + p.get('overshoot', 0.05)) + n * o
            s[2] = e[2] = z
            ctx.ctrl.set_mode('track', eff)
            st = _move(ctx, eff, [s + np.array([0, 0, 0.04]), s], min(0.5, ctx.ctrl.limits['v']), q, gripper=1.0)
            if st != 'done':
                return r.result(False, st, lanes=lanes_done, jams=jams)
            ctx.ctrl.set_mode('guarded', eff, threshold=p.get('guard', 0.6 * ctx.emb.spec.max_ee_force), axis=tuple(d))
            L = float(np.linalg.norm(e - s))
            _move(ctx, eff, [e], v, q)
            jams += int(ctx.ctrl.tripped(eff))
            ctx.ctrl.set_mode('track', eff)
            _move(ctx, eff, [_tcp(ctx, eff)[0] + np.array([0, 0, 0.04])], 0.5, q)
            lanes_done += 1
        return r.result(True, 'done', lanes=lanes_done, jams=jams)


# ============================================================================ insert (fine)
def insert(ctx: SkillContext, call, est: ObjectEstimate):
    """peg (held, from pick) into a hole. est: hole estimate (pos = entry centre on the top surface).
    params: depth (m), force (N, search/press force), search_r (m), pitch (m per spiral turn), speed (search m/s),
    guard (N), hold_offset (TCP height above the peg bottom; default from the pick)."""
    eff, p = _eff(ctx, call), call.params
    with SkillRun(ctx, 'insert') as r:
        held = _held(ctx).get(eff)
        off = p.get('hold_offset', held['offset'] if held else 0.02)
        depth = p.get('depth', 0.02)
        Fn = p.get('force', min(10.0, 0.3 * ctx.emb.spec.max_ee_force))
        entry = est.p.copy()
        _, q = _tcp(ctx, eff)
        q = down_quat(p['yaw'] if 'yaw' in p else _near_yaw(ctx, eff, _yaw_of_closing(q)))
        ctx.ctrl.set_mode('track', eff)
        above = entry + np.array([0, 0, off + 0.02])
        cur = _tcp(ctx, eff)[0]
        st = _move(ctx, eff, [np.array([cur[0], cur[1], max(cur[2], above[2] + 0.03)]), above + [0, 0, 0.03], above],
                   p.get('transit_speed', 0.25), q)
        if st != 'done':
            return r.result(False, st, phase='transit')
        ctx.wait(0.1)
        F0, _, _ = ctx.emb.wrist_ft(ctx.model, ctx.data, eff)
        goal_z = entry[2] + off - depth
        ctx.ctrl.set_mode('guarded', eff, threshold=p.get('guard', 3.0), axis=(0, 0, -1), bias=tuple(F0))
        st = _move(ctx, eff, [np.array([entry[0], entry[1], goal_z])], p.get('approach_speed', 0.03), q)
        z_contact = _tcp(ctx, eff)[0][2]
        inserted = z_contact <= goal_z + 0.002
        searched, found = False, inserted
        partial = (not inserted) and z_contact <= entry[2] + off - 0.003   # already part-way in: just press home
        if partial:
            found = True
        if not inserted and not partial:
            # spiral search under a constant downward force; stop as soon as the peg drops into the hole
            searched = True
            ctx.ctrl.set_mode('force', eff, axis=(0, 0, -1), force=p.get('search_force', min(4.0, Fn)), approach_speed=0.01,
                              bias=tuple(F0), compliant=False)
            pitch, rmax = p.get('pitch', 0.0008), p.get('search_r', 0.004)
            turns = rmax / pitch
            th = np.linspace(0, 2 * math.pi * turns, int(turns * 16) + 2)[1:]
            rr = pitch * th / (2 * math.pi)
            c0 = _tcp(ctx, eff)[0]
            pts = [np.array([c0[0] + rr[i] * math.cos(th[i]), c0[1] + rr[i] * math.sin(th[i]), c0[2]]) for i in range(len(th))]
            vs = p.get('speed', 0.006)
            durs = [max(0.02, float(np.linalg.norm(b - a)) / vs) for a, b in zip([c0] + pts[:-1], pts)]
            vv = [None] * len(pts)
            tr = BlendedTrajectory(effector=eff, waypoints=[(x, q) for x in pts], durations=durs, via_vel=vv)
            ctx.ctrl.limits_backup = ctx.ctrl.limits
            ctx.ctrl.command(tr, preempt=True)
            st = ctx.run(until=lambda c: _tcp(c, eff)[0][2] < z_contact - p.get('drop', 0.003), max_t=sum(durs) + 1.0,
                         effector=eff)
            found = st == 'until'
        if found and not inserted:
            searched = searched or partial
            # press home compliantly: lateral impedance, constant downward force
            here = _tcp(ctx, eff)[0]
            ctx.ctrl.set_mode('force', eff, axis=(0, 0, -1), force=Fn, approach_speed=0.02, bias=tuple(F0), compliant=True,
                              stiffness=p.get('lateral_k', 300.0))
            ctx.ctrl.command(BlendedTrajectory(effector=eff, waypoints=[(np.array([here[0], here[1], goal_z]), q)],
                                               durations=[max(0.5, depth / 0.02)]), preempt=True)
            ctx.run(until=lambda c: _tcp(c, eff)[0][2] <= goal_z + 0.001, max_t=depth / 0.01 + 2.0, effector=eff)
            ctx.wait(0.2)
        z_end = _tcp(ctx, eff)[0][2]
        ins_depth = float(entry[2] + off - z_end)
        ok = ins_depth >= depth - 0.003
        ctx.ctrl.set_mode('track', eff)
        here = _tcp(ctx, eff)[0]
        ctx.ctrl.command(BlendedTrajectory(effector=eff, waypoints=[(here, q)], durations=[0.2]), preempt=True)
        ctx.run(max_t=0.5, effector=eff)
        if ok and p.get('release', True):
            _close(ctx, eff, 0.0)
            _held(ctx).pop(eff, None)
            _move(ctx, eff, [_tcp(ctx, eff)[0] + np.array([0, 0, 0.06])], 0.2, q, gripper=0.0)
        return r.result(ok, 'done' if ok else ('not_found' if searched and not found else 'jammed'),
                        depth=ins_depth, searched=searched, found=found, z_contact=float(z_contact))


# ============================================================================ open / close (articulated)
def articulate(ctx: SkillContext, call, est: ObjectEstimate):
    """open/close a drawer (straight pull along est.axis) or door/lid (arc about est.hinge/est.axis) by its handle
    (est.pos). params: distance (m, drawer) or angle (rad, door), speed, stiffness (N/m, impedance during the motion),
    grasp (1: grasp the handle; 0: push with closed fingers), approach (unit vector into the handle; default -pull dir)."""
    eff, p = _eff(ctx, call), call.params
    sign = 1.0 if call.skill == 'open' else -1.0
    with SkillRun(ctx, call.skill) as r:
        v = p.get('speed', 0.15)
        h = est.p.copy()
        grasp = bool(p.get('grasp', 1))
        if est.kind == 'drawer':
            pull = np.asarray(est.axis, float)
            pull /= np.linalg.norm(pull)
        else:
            ax = np.asarray(est.axis, float)
            ax /= np.linalg.norm(ax)
            rad = h - np.asarray(est.hinge, float)
            rad -= ax * (rad @ ax)
            pull = np.cross(ax, rad)
            pull /= np.linalg.norm(pull)
        app = np.asarray(p['approach'], float) if 'approach' in p else -pull * sign if grasp else -pull * sign
        app /= np.linalg.norm(app)
        if grasp:
            app = -pull  # reach into the handle against the pull direction
        up = np.array([0, 0, 1.0]) if abs(app[2]) < 0.9 else np.array([1.0, 0, 0])
        q = _quat_axes(app, up if p.get('vertical_bar', 0) else np.cross(app, up))
        ctx.ctrl.set_mode('track', eff)
        pre = h - app * 0.10
        st = _move(ctx, eff, [pre], 0.3, q, gripper=0.0 if grasp else 1.0)
        if st != 'done':
            return r.result(False, st, phase='approach')
        ctx.ctrl.set_mode('guarded', eff, threshold=_guard_n(ctx, p, 8.0), axis=tuple(app))
        _move(ctx, eff, [h + app * p.get('insert', 0.0)], 0.08, q)
        ctx.ctrl.set_mode('track', eff)
        if grasp:
            _close(ctx, eff, 1.0)
            c, n = ctx.emb.gripper_state(ctx.model, ctx.data, eff)
            if n == 0:
                _move(ctx, eff, [pre], 0.3, q, gripper=0.0)
                return r.result(False, 'grasp_failed', closure=c)
        start = _tcp(ctx, eff)[0]
        K = p.get('stiffness', 400.0)
        ctx.ctrl.set_mode('impedance', eff, stiffness=K, rot_stiffness=p.get('rot_stiffness', 20.0))
        if est.kind == 'drawer':
            dist = sign * p.get('distance', 0.15)
            pts, qs = [start + pull * dist], [q]
        else:
            ang = sign * p.get('angle', 0.6)
            hinge = np.asarray(est.hinge, float)
            n = max(3, int(abs(ang) / 0.15))
            pts, qs = [], []
            for k in range(1, n + 1):
                a = ang * k / n
                Rk = quat2mat(np.array([math.cos(a / 2), *(ax * math.sin(a / 2))]))
                pts.append(hinge + Rk @ (start - hinge))
                qs.append(mat2quat(Rk @ quat2mat(q)))
        L = sum(float(np.linalg.norm(b - a)) for a, b in zip([start] + pts[:-1], pts))
        T = _T(L, v)
        durs = [T / len(pts)] * len(pts)
        tr = BlendedTrajectory(effector=eff, waypoints=list(zip(pts, qs)), durations=durs)
        ctx.ctrl.command(tr)
        Fm = []
        st = ctx.run(monitor=lambda c: Fm.append(float(np.linalg.norm(c.emb.wrist_ft(c.model, c.data, eff)[0]))),
                     max_t=T * 2 + 2, effector=eff)
        end = _tcp(ctx, eff)[0]
        moved = float((end - start) @ (pts[-1] - start) / max(np.linalg.norm(pts[-1] - start), 1e-9))
        ctx.ctrl.set_mode('track', eff)
        here_q = _tcp(ctx, eff)[1]
        ctx.ctrl.command(BlendedTrajectory(effector=eff, waypoints=[(end, here_q)], durations=[0.2]), preempt=True)
        ctx.run(max_t=0.5, effector=eff)
        if grasp:
            _close(ctx, eff, 0.0)
        Rq = quat2mat(here_q)
        _move(ctx, eff, [end - Rq[:, 2] * 0.08], 0.2, here_q, gripper=0.0)
        frac = moved / max(np.linalg.norm(pts[-1] - start), 1e-9)
        return r.result(frac > 0.8, 'done' if frac > 0.8 else 'blocked', moved=moved, fraction=float(frac),
                        F_peak=float(max(Fm) if Fm else 0.0))


# ============================================================================ handover (bimanual)
def handover(ctx: SkillContext, call, est: ObjectEstimate):
    """giver = call.effector (holding the object from a top grasp), receiver = the other effector. params: x, y, z
    (meeting point of the object's centre), speed, grip_low (m below the centre where the receiver grasps)."""
    p = call.params
    giver = _eff(ctx, call)
    others = [e for e in ctx.emb.eff if e != giver]
    if not others:
        with SkillRun(ctx, 'handover') as r:
            return r.result(False, 'unsupported')
    recv = p.get('receiver', others[0])
    with SkillRun(ctx, 'handover') as r:
        v = p.get('speed', 0.25)
        held = _held(ctx).get(giver)
        if held is None:
            return r.result(False, 'not_holding')
        hz = held['half_size'][2]
        meet = np.array([p['x'], p['y'], p['z']], float)
        gp, gq = _tcp(ctx, giver)
        top_off = held['offset'] - hz  # TCP height above the object centre
        g_goal = meet + np.array([0, 0, top_off])
        rp, _ = _tcp(ctx, recv)
        app = meet - rp
        app[2] = 0
        app /= np.linalg.norm(app)
        qr = _quat_axes(app, np.cross(app, [0, 0, 1.0]))
        r_grasp = meet - np.array([0, 0, p.get('grip_low', 0.5 * hz)])
        r_pre = r_grasp - app * 0.10
        ctx.ctrl.set_mode('track')
        T1 = _T(float(np.linalg.norm(g_goal - gp)), v)
        T2 = _T(float(np.linalg.norm(r_pre - rp)), v, float(np.linalg.norm(quat_err(qr, _tcp(ctx, recv)[1]))))
        T = max(T1, T2)
        ctx.ctrl.command(BlendedTrajectory(effector=giver, waypoints=[(g_goal, gq)], durations=[T]))
        ctx.ctrl.command(BlendedTrajectory(effector=recv, waypoints=[(r_pre, qr)], durations=[T], gripper=0.0))
        st = ctx.run(max_t=T + 2)
        ctx.ctrl.command(BlendedTrajectory(effector=recv, waypoints=[(r_grasp, qr)], durations=[_T(0.1, 0.1)]))
        st = ctx.run(max_t=3.0)
        _close(ctx, recv, 1.0)
        c, n = ctx.emb.gripper_state(ctx.model, ctx.data, recv)
        if n == 0:
            return r.result(False, 'receiver_missed', closure=c)
        _close(ctx, giver, 0.0)
        _move(ctx, giver, [_tcp(ctx, giver)[0] + np.array([0, 0, 0.08])], 0.2, gq, gripper=0.0)
        c, n = ctx.emb.gripper_state(ctx.model, ctx.data, recv)
        ok = n > 0 and c < 0.95
        _held(ctx).pop(giver, None)
        if ok:
            _held(ctx)[recv] = dict(offset=float(hz - p.get('grip_low', 0.5 * hz)), half_size=held['half_size'], side=True,
                                    width=held['width'])
        return r.result(ok, 'done' if ok else 'dropped', closure=c, contacts=n)


# ============================================================================ move_base (mobile manipulators)
def move_base(ctx: SkillContext, call, est: ObjectEstimate = None):
    """drive a mobile base to a pose. params: x, y, yaw (world / station frame, m, rad), speed (m/s, default 0.3),
    yaw_rate (rad/s, default 0.8), arm ('keep' | 'tuck': lift the hand 0.1 m first). Min-jerk base trajectory through
    the controller's computed-torque base servo (torques only, effort-metered); done when the time is up (no
    convergence wait); status 'unsupported' for fixed-base robots."""
    p = call.params
    eff = _eff(ctx, call)
    with SkillRun(ctx, 'move_base') as r:
        emb = ctx.emb
        if getattr(emb, 'base', None) is None:
            return r.result(False, 'unsupported')
        mpos, mquat = emb.mount_pose
        Rm = quat2mat(mquat)
        myaw = math.atan2(Rm[1, 0], Rm[0, 0])
        # base joints are expressed in the mount frame (x forward, y left, yaw about z)
        dxy = np.array([p['x'], p['y']], float) - np.asarray(mpos[:2], float)
        c, s_ = math.cos(myaw), math.sin(myaw)
        tq = np.array([c * dxy[0] + s_ * dxy[1], -s_ * dxy[0] + c * dxy[1], _wrap(float(p.get('yaw', myaw)) - myaw)])
        qb = np.asarray(emb.proprio(ctx.data)['base']['q'], float).copy()     # base encoders (actor-legal)
        tq[2] = qb[2] + _wrap(tq[2] - qb[2])
        if p.get('arm', 'keep') == 'tuck':
            here, hq = _tcp(ctx, eff)
            _move(ctx, eff, [here + np.array([0, 0, 0.1])], 0.3, hq)
        dist = float(np.linalg.norm(tq[:2] - qb[:2]))
        dur = max(0.5, 1.875 * dist / max(float(p.get('speed', 0.3)), 1e-3),
                  1.875 * abs(tq[2] - qb[2]) / max(float(p.get('yaw_rate', 0.8)), 1e-3))
        ctx.ctrl.command_base(tq, dur)
        st = ctx.run(max_t=dur + 1.0)
        qe = np.asarray(emb.proprio(ctx.data)['base']['q'], float)
        err = float(np.linalg.norm(qe[:2] - tq[:2]))
        return r.result(st == 'done' and err < 0.03, 'done' if err < 0.03 else 'blocked', pos_err=err,
                        yaw_err=float(abs(_wrap(qe[2] - tq[2]))), duration_plan=dur)
