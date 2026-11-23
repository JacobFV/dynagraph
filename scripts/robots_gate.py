"""ROBOTS gates (docs/V2_PLAN.md G1 + skills) per robot. Writes results/v2/robots/<robot>/<test>.json.

python scripts/robots_gate.py <robot> [strokes|pickplace|push|insert|sweep|handover|video|all] [--n N]
"""
import argparse, json, math, os, subprocess, sys, time

import mujoco
import numpy as np

from physmem.v2.api import SkillCall, EffortMeter
from physmem.embodiments import testbench as TB
from physmem.embodiments.trajectory import BlendedTrajectory
from physmem.embodiments.ik import down_quat
from physmem.skills import SkillContext, run_skill, ObjectEstimate
from physmem.skills import primitives as P

OUT = 'results/v2/robots'


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return {}
    return dict(n=int(len(x)), mean=float(x.mean()), p10=float(np.percentile(x, 10)), p50=float(np.median(x)),
                p90=float(np.percentile(x, 90)), p95=float(np.percentile(x, 95)), max=float(x.max()), min=float(x.min()))


def save(robot, test, obj):
    os.makedirs(f'{OUT}/{robot}', exist_ok=True)
    with open(f'{OUT}/{robot}/{test}.json', 'w') as f:
        json.dump(obj, f, indent=1, default=float)
    print(robot, test, brief(obj.get('summary', {})))


def brief(x):
    if isinstance(x, dict):
        if 'p50' in x and 'p10' in x:
            return {k: round(x[k], 4) for k in ('p10', 'p50', 'p90', 'max')}
        return {k: brief(v) for k, v in x.items()}
    return round(x, 4) if isinstance(x, float) else x


def effectors(emb):
    return list(emb.eff)


def ws_for(robot, eff):
    (x0, x1), (y0, y1) = TB.BENCH[robot]['ws']
    if robot == 'aloha' and eff == 'right':   # mirror the left arm's workspace
        x0, x1 = -x1, -x0
    return (x0, x1), (y0, y1)


# --------------------------------------------------------------------------- G1: free-space strokes
def strokes(robot, n=50, seed=0):
    rng = np.random.default_rng(seed)
    spec, emb, B = TB.make_scene(robot)
    m, d = TB.compile_scene(spec, emb)
    ctx = SkillContext(m, d, emb)
    ctrl = ctx.ctrl
    vmax = emb.cfg.limits['v']
    rows = []
    t_wall0, t_sim0 = time.perf_counter(), d.time
    for i in range(n):
        eff = effectors(emb)[i % len(emb.eff)]
        (x0, x1), (y0, y1) = ws_for(robot, eff)
        v = float(np.exp(rng.uniform(np.log(0.1), np.log(vmax))))
        L = rng.uniform(*B['stroke'])
        a_lim = ctrl.limits['a'] * 0.6
        T_r = max(0.2, 3 * v / a_lim); L_r = 0.5 * v * T_r
        T_d = max(0.15, 3 * v / a_lim); L_d = 0.5 * v * T_d
        tot = L_r + L + L_d
        # a straight horizontal line through the workspace centre region (IK-feasible along its length); if none is
        # found the stroke is shortened, and finally skipped (never commanded out of reach)
        feasible = False
        for attempt in range(150):
            if attempt and attempt % 50 == 0:
                L *= 0.75
                tot = L_r + L + L_d
            th = rng.uniform(-math.pi, math.pi)
            dvec = np.array([math.cos(th), math.sin(th), 0.0])
            c = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), B['table_z'] + rng.uniform(0.06, 0.15)])
            s = c - dvec * tot / 2
            e = c + dvec * tot / 2
            q = down_quat(P._near_yaw(ctx, eff, th + math.pi / 2))
            if x0 - 0.05 <= s[0] <= x1 + 0.05 and x0 - 0.05 <= e[0] <= x1 + 0.05 and y0 - 0.1 <= s[1] <= y1 + 0.1 and \
                    y0 - 0.1 <= e[1] <= y1 + 0.1 and all(P._reachable(ctx, eff, s + (e - s) * u, q, tol=0.004) for u in np.linspace(0, 1, 7)):
                feasible = True
                break
        if not feasible:
            rows.append(dict(i=i, eff=eff, v_cmd=v, skipped=True, ratio=None, release_ratio=None, err_mid_max=None,
                             err_mid_mean=None, stall=False, retime=1.0, effort=EffortMeter().as_dict(),
                             peaks=dict(torque_rate=0.0, joint_jerk=0.0, torque_norm=0.0)))
            continue
        ctrl.set_mode('track')
        st0 = P._move(ctx, eff, [s + [0, 0, 0.04], s], min(0.4, vmax), q)
        ctx.wait(0.05)
        a = s + dvec * L_r
        b = a + dvec * L
        f = b + dvec * L_d
        meter = ctrl.acc.open()
        tr = BlendedTrajectory(effector=eff, waypoints=[(a, q), (b, q), (f, q)], durations=[T_r, L / v, T_d],
                               via_vel=[dvec * v, dvec * v, np.zeros(3)])
        t_plan = d.time
        pred = ctrl.command(tr)
        k = pred['scale']
        tc0, tc1 = t_plan + T_r * k, t_plan + (T_r + L / v) * k
        S = []

        def mon(c_):
            t = c_.data.time
            if tc0 <= t <= tc1:
                vv, _ = emb.tcp_vel(m, d, eff)
                pp, _ = emb.tcp_pose(d, eff)
                ref = ctrl.reference(eff)
                S.append((float(vv @ dvec), float(np.linalg.norm(pp - ref[0])), float(np.linalg.norm(vv))))
        st = ctx.run(monitor=mon, max_t=(T_r + L / v + T_d) * k * 1.5 + 0.5, effector=eff)
        pk = ctrl.acc.close(meter)
        S = np.array(S)
        nS = len(S)
        mid = S[nS // 4: 3 * nS // 4] if nS > 8 else S
        vc = v / k
        rows.append(dict(i=i, eff=eff, v_cmd=v, v_cmd_retimed=vc, retime=k, length=L, status=st,
                         ratio=float(np.mean(mid[:, 0]) / vc) if len(mid) else None,
                         release_ratio=float(np.mean(S[-3:, 0]) / vc) if nS >= 3 else None,
                         err_mid_max=float(mid[:, 1].max()) if len(mid) else None,
                         err_mid_mean=float(mid[:, 1].mean()) if len(mid) else None,
                         stall=bool(st != 'done' or (len(mid) and np.mean(mid[:, 0]) < 0.5 * vc)),
                         effort=meter.as_dict(), peaks=pk))
    wall = time.perf_counter() - t_wall0
    sim = d.time - t_sim0
    r = [x['ratio'] for x in rows]
    rr = [x['release_ratio'] for x in rows]
    summ = dict(robot=robot, n=n, vmax=vmax, ratio=stats(r), release_ratio=stats(rr),
                err_mid_max=stats([x['err_mid_max'] for x in rows]), err_mid_mean=stats([x['err_mid_mean'] for x in rows]),
                stalls=int(sum(x['stall'] for x in rows)), retimed=int(sum(x['retime'] > 1.001 for x in rows)),
                skipped=int(sum(x.get('skipped', False) for x in rows)),
                peak_torque_rate=stats([x['peaks']['torque_rate'] for x in rows if not x.get('skipped')]),
                peak_joint_jerk=stats([x['peaks']['joint_jerk'] for x in rows if not x.get('skipped')]),
                time_at_limit_s=float(sum(x['effort']['time_at_limit_s'] for x in rows)),
                frac_ratio_within_7pct=float(np.mean([abs(x - 1) <= 0.07 for x in r if x is not None])),
                integrity=TB.integrity(m, d), sim_s=sim, wall_s=wall, rt_factor=sim / wall)
    # per speed band
    bands = {}
    for lo, hi in ((0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.81), (0.81, 2.0)):
        sel = [x['ratio'] for x in rows if lo <= x['v_cmd'] < hi and x['ratio'] is not None]
        if sel:
            bands[f'{lo}-{hi}'] = stats(sel)
    summ['ratio_by_speed'] = bands
    save(robot, 'strokes', dict(summary=summ, rows=rows))
    return summ


# --------------------------------------------------------------------------- pick & place
def pickplace(robot, n=20, seed=100):
    rows = []
    wall = sim = 0.0
    for s in range(n):
        rng = np.random.default_rng(seed + s)
        B = TB.BENCH[robot]
        hs = rng.uniform(*B['obj'])
        half = (hs, hs * rng.uniform(1, 1.3), hs * rng.uniform(1, 1.6))
        eff_name = ['left', 'right'][s % 2] if robot == 'aloha' else None
        (x0, x1), (y0, y1) = ws_for(robot, eff_name)
        pos = (rng.uniform(x0, x1), rng.uniform(y0, y1), B['table_z'] + half[2])
        while True:
            tgt = (rng.uniform(x0, x1), rng.uniform(y0, y1))
            if np.hypot(tgt[0] - pos[0], tgt[1] - pos[1]) > 2.5 * max(half):
                break
        mass = rng.uniform(*B['mass'])
        spec, emb, _ = TB.make_scene(robot, extra=lambda sp, b: TB.add_box(sp, 'obj', pos, half, mass, yaw=rng.uniform(-1, 1),
                                                                         friction=rng.uniform(0.5, 1.0)))
        m, d = TB.compile_scene(spec, emb)
        eff = effectors(emb)[s % len(emb.eff)]
        assert eff_name is None or eff == eff_name
        ctx = SkillContext(m, d, emb)
        ctx.wait(0.2)
        est = TB.estimate(m, d, 'obj', half, rng)
        t0 = time.perf_counter()
        r1 = run_skill(ctx, SkillCall('pick', dict(speed=0.4), target='obj', effector=eff), est)
        r2 = run_skill(ctx, SkillCall('place', dict(x=tgt[0], y=tgt[1], z=B['table_z'], speed=0.4), target='obj', effector=eff)) if r1.ok else None
        ctx.wait(0.3)
        wall += time.perf_counter() - t0
        sim += d.time
        p, qo = TB.body_pose(m, d, 'obj')
        err = float(np.hypot(p[0] - tgt[0], p[1] - tgt[1]))
        upright = abs(p[2] - B['table_z'] - half[2]) < 0.005
        ok = bool(r1.ok and r2 is not None and r2.ok and err < 0.02 and upright)
        rows.append(dict(seed=seed + s, eff=eff, mass=mass, half=half, pick=r1.status, pick_log={k: v for k, v in r1.log.items() if k != 'peaks'},
                         place=None if r2 is None else r2.status,
                         err_xy=err, upright=upright, success=ok, effort_pick=r1.effort.as_dict(),
                         effort_place=None if r2 is None else r2.effort.as_dict(), dur=r1.duration_s + (r2.duration_s if r2 else 0),
                         integrity=TB.integrity(m, d)))
    summ = dict(robot=robot, n=n, success=float(np.mean([r['success'] for r in rows])),
                pick_ok=float(np.mean([r['pick'] == 'done' for r in rows])), err_xy=stats([r['err_xy'] for r in rows if r['place']]),
                failures={k: int(sum(1 for r in rows if (r['pick'] if r['pick'] != 'done' else (r['place'] or '')) == k and not r['success']))
                          for k in set(r['pick'] if r['pick'] != 'done' else (r['place'] or 'none') for r in rows)},
                integrity_violations=int(sum(len(r['integrity']) for r in rows)), rt_factor=sim / max(wall, 1e-9),
                mean_duration_s=float(np.mean([r['dur'] for r in rows])))
    save(robot, 'pickplace', dict(summary=summ, rows=rows))
    return summ


# --------------------------------------------------------------------------- push speed tracking (contact)
def push(robot, n=20, seed=200):
    rows = []
    vmax = None
    for s in range(n):
        rng = np.random.default_rng(seed + s)
        B = TB.BENCH[robot]
        hs = B['obj'][1]
        half = (hs, hs * 1.2, hs)
        (x0, x1), (y0, y1) = ws_for(robot, ['left', 'right'][s % 2] if robot == 'aloha' else None)
        pos = ((x0 + x1) / 2 + rng.uniform(-0.03, 0.03), (y0 + y1) / 2 + rng.uniform(-0.03, 0.03), B['table_z'] + half[2])
        spec, emb, _ = TB.make_scene(robot, extra=lambda sp, b: TB.add_box(sp, 'obj', pos, half, rng.uniform(*B['mass']),
                                                                         friction=rng.uniform(0.3, 0.7)))
        m, d = TB.compile_scene(spec, emb)
        eff = effectors(emb)[s % len(emb.eff)]
        vmax = emb.cfg.limits['v']
        v = float(np.exp(rng.uniform(np.log(0.1), np.log(min(0.8, vmax)))))
        ctx = SkillContext(m, d, emb)
        ctx.wait(0.2)
        est = TB.estimate(m, d, 'obj', half, rng)
        # push direction: away from the robot base +- 60 deg (object frame theta)
        base_dir = math.atan2(pos[1] - emb.tcp_pose(d, eff)[0][1] * 0, pos[0] - (0 if robot != 'aloha' else (-0.469 if eff == 'left' else 0.469)))
        th_w = base_dir + rng.uniform(-1.0, 1.0)
        yaw_o = math.atan2(est.R[1, 0], est.R[0, 0])
        dist = min(0.12, 0.5 * (B['ws'][0][1] - B['ws'][0][0]) + 0.05)
        p0, _ = TB.body_pose(m, d, 'obj')
        r = run_skill(ctx, SkillCall('push', dict(theta=th_w - yaw_o, speed=v, distance=dist), target='obj', effector=eff), est)
        ctx.wait(1.0)
        p1, _ = TB.body_pose(m, d, 'obj')
        rows.append(dict(seed=seed + s, v=v, status=r.status, moved=float(np.linalg.norm(p1[:2] - p0[:2])),
                         **{k: r.log.get(k) for k in ('speed_ratio', 'release_ratio', 'track_err_mid_max', 'track_err_mid_mean', 'retime', 'F_max')},
                         effort=r.effort.as_dict(), integrity=TB.integrity(m, d)))
    summ = dict(robot=robot, n=n, speed_ratio=stats([r['speed_ratio'] for r in rows]),
                release_ratio=stats([r['release_ratio'] for r in rows]), err_mid_max=stats([r['track_err_mid_max'] for r in rows]),
                moved=stats([r['moved'] for r in rows]), ok=float(np.mean([r['status'] == 'done' for r in rows])),
                unreachable=int(sum(r['status'] == 'unreachable' for r in rows)),
                frac_within_7pct=float(np.mean([r['speed_ratio'] is not None and abs(r['speed_ratio'] - 1) <= 0.07 for r in rows])),
                integrity_violations=int(sum(len(r['integrity']) for r in rows)))
    save(robot, 'push', dict(summary=summ, rows=rows))
    return summ


# --------------------------------------------------------------------------- insertion (fine)
def insert(robot, n=10, seed=300, clearances=(0.001, 0.0005)):
    out = {}
    allrows = []
    for cl in clearances:
        rows = []
        for s in range(n):
            rng = np.random.default_rng(seed + s + int(cl * 1e5))
            B = TB.BENCH[robot]
            (x0, x1), (y0, y1) = B['ws']
            pw = 0.01
            ph = (pw, pw, 0.03)
            peg = (x0 + 0.3 * (x1 - x0), y0 + 0.3 * (y1 - y0) + rng.uniform(-0.02, 0.02), B['table_z'] + ph[2])
            hole = np.array([x0 + 0.7 * (x1 - x0) + rng.uniform(-0.02, 0.02), y0 + 0.7 * (y1 - y0), B['table_z'] + 0.03])

            def extra(sp, b):
                TB.add_box(sp, 'peg', peg, ph, 0.05, friction=0.4)
                TB.add_hole_fixture(sp, 'hole', hole, ph[0] + cl / 2)
            spec, emb, _ = TB.make_scene(robot, extra=extra)
            m, d = TB.compile_scene(spec, emb)
            eff = effectors(emb)[0]
            ctx = SkillContext(m, d, emb)
            ctx.wait(0.2)
            est = TB.estimate(m, d, 'peg', ph, rng, pos_sd=0.001, yaw_sd=0.01)
            r1 = run_skill(ctx, SkillCall('pick', dict(speed=0.3, depth=0.035, lift=0.08), target='peg', effector=eff), est)
            hole_est = ObjectEstimate(pos=hole + np.r_[rng.normal(0, 0.001, 2), 0], half_size=(ph[0] + cl / 2,) * 2 + (0.015,))
            r2 = run_skill(ctx, SkillCall('insert', dict(depth=0.012, search_r=0.004), target='hole', effector=eff), hole_est) if r1.ok else None
            ctx.wait(0.3)
            pp, _ = TB.body_pose(m, d, 'peg')
            depth_true = float(hole[2] - (pp[2] - ph[2]))
            lat = float(np.hypot(pp[0] - hole[0], pp[1] - hole[1]))
            ok = bool(depth_true > 0.010 and lat < 0.003)
            rows.append(dict(seed=s, clearance=cl, pick=r1.status, insert=None if r2 is None else r2.status, depth_true=depth_true,
                             lateral=lat, success=ok, searched=None if r2 is None else r2.log.get('searched'),
                             dur=None if r2 is None else r2.duration_s, effort=None if r2 is None else r2.effort.as_dict(),
                             integrity=TB.integrity(m, d)))
        out[f'{cl * 1000:.1f}mm'] = dict(success=float(np.mean([r['success'] for r in rows])),
                                         pick_ok=float(np.mean([r['pick'] == 'done' for r in rows])),
                                         insert_given_pick=float(np.mean([r['success'] for r in rows if r['pick'] == 'done'])) if any(r['pick'] == 'done' for r in rows) else None,
                                         statuses={k: sum(1 for r in rows if r['insert'] == k) for k in set(r['insert'] for r in rows)},
                                         dur=stats([r['dur'] for r in rows]))
        allrows += rows
    save(robot, 'insert', dict(summary=dict(robot=robot, n=n, **out), rows=allrows))
    return out


# --------------------------------------------------------------------------- sweep 30 small parts (bulk)
def sweep(robot, n=3, seed=400, parts=30):
    rows = []
    for s in range(n):
        rng = np.random.default_rng(seed + s)
        B = TB.BENCH[robot]
        (x0, x1), (y0, y1) = B['ws']
        sc = (x1 - x0) / 0.3
        hs = 0.008 * min(1.0, max(0.6, sc))
        rx0, rx1 = x0 + 0.1 * (x1 - x0), x1 - 0.1 * (x1 - x0)
        ry0, ry1 = y0 + 0.15 * (y1 - y0), (y0 + y1) / 2 + 0.1 * (y1 - y0)
        pts = []
        while len(pts) < parts:
            c = (rng.uniform(rx0, rx1), rng.uniform(ry0, ry1))
            if all(np.hypot(c[0] - a, c[1] - b) > 2.6 * hs for a, b in pts):
                pts.append(c)

        def extra(sp, b):
            for k, (a, bb) in enumerate(pts):
                TB.add_box(sp, f'part{k}', (a, bb, B['table_z'] + hs), (hs, hs, hs), 0.01, rgba=(0.9, 0.6, 0.1, 1), yaw=rng.uniform(-1, 1))
        spec, emb, _ = TB.make_scene(robot, extra=extra)
        m, d = TB.compile_scene(spec, emb)
        eff = effectors(emb)[0]
        ctx = SkillContext(m, d, emb)
        ctx.wait(0.2)
        t0 = time.perf_counter()
        line = ry1 + 0.01   # parts must end beyond this line (+y)
        r = run_skill(ctx, SkillCall('sweep', dict(x0=rx0, y0=ry0 - 0.02, x1=rx1, y1=ry1, direction=math.pi / 2,
                                                   surface_z=B['table_z'], height=0.012 if hs > 0.006 else 0.008,
                                                   speed=0.25, overshoot=0.06), effector=eff), None)
        ctx.wait(0.5)
        wall = time.perf_counter() - t0
        ys = np.array([TB.body_pose(m, d, f'part{k}')[0][1] for k in range(parts)])
        zs = np.array([TB.body_pose(m, d, f'part{k}')[0][2] for k in range(parts)])
        moved = float(np.mean(ys > line))
        rows.append(dict(seed=s, status=r.status, lanes=r.log.get('lanes'), jams=r.log.get('jams'), frac_moved=moved,
                         fell=int(np.sum(zs < B['table_z'] - 0.05)), dur=r.duration_s, wall=wall, effort=r.effort.as_dict(),
                         integrity=TB.integrity(m, d), nv=int(m.nv)))
    summ = dict(robot=robot, n=n, parts=parts, frac_moved=stats([r['frac_moved'] for r in rows]),
                dur=stats([r['dur'] for r in rows]), rt_factor=float(sum(r['dur'] for r in rows) / sum(r['wall'] for r in rows)),
                integrity_violations=int(sum(len(r['integrity']) for r in rows)))
    save(robot, 'sweep', dict(summary=summ, rows=rows))
    return summ


# --------------------------------------------------------------------------- bimanual handover
def handover(robot='aloha', n=10, seed=500):
    rows = []
    for s in range(n):
        rng = np.random.default_rng(seed + s)
        half = (0.015, 0.015, 0.04)
        pos = (-0.2 + rng.uniform(-0.03, 0.03), rng.uniform(-0.05, 0.05), half[2])
        spec, emb, _ = TB.make_scene(robot, extra=lambda sp, b: TB.add_box(sp, 'obj', pos, half, 0.08))
        m, d = TB.compile_scene(spec, emb)
        ctx = SkillContext(m, d, emb)
        ctx.wait(0.2)
        est = TB.estimate(m, d, 'obj', half, rng)
        r1 = run_skill(ctx, SkillCall('pick', dict(speed=0.3, depth=0.02, lift=0.12), target='obj', effector='left'), est)
        r2 = run_skill(ctx, SkillCall('handover', dict(x=0.0, y=0.0, z=0.18, speed=0.25), target='obj', effector='left'), est) if r1.ok else None
        ctx.wait(0.5)
        p, _ = TB.body_pose(m, d, 'obj')
        c, nc = emb.gripper_state(m, d, 'right')
        ok = bool(r2 is not None and r2.ok and p[2] > 0.08 and nc > 0)
        rows.append(dict(seed=s, pick=r1.status, handover=None if r2 is None else r2.status, success=ok, z=float(p[2]),
                         effort=None if r2 is None else r2.effort.as_dict(), integrity=TB.integrity(m, d)))
    summ = dict(robot=robot, n=n, success=float(np.mean([r['success'] for r in rows])),
                statuses={k: sum(1 for r in rows if (r['handover'] or r['pick']) == k) for k in set((r['handover'] or r['pick']) for r in rows)})
    save(robot, 'handover', dict(summary=summ, rows=rows))
    return summ


# --------------------------------------------------------------------------- smoothness video
def video(robot, seed=7, fps=30):
    rng = np.random.default_rng(seed)
    B = TB.BENCH[robot]
    hs = B['obj'][1]
    half = (hs, hs * 1.2, hs * 1.3)
    (x0, x1), (y0, y1) = B['ws']
    pos = (x0 + 0.35 * (x1 - x0), y0 + 0.3 * (y1 - y0), B['table_z'] + half[2])
    spec, emb, _ = TB.make_scene(robot, extra=lambda sp, b: TB.add_box(sp, 'obj', pos, half, 0.2))
    m, d = TB.compile_scene(spec, emb)
    eff = effectors(emb)[0]
    rend = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [(x0 + x1) / 2, (y0 + y1) / 2, B['table_z'] + 0.1]
    cam.distance = 2.2 * max(0.35, x1 - x0 + 0.3) if robot != 'so100' else 0.7
    cam.azimuth, cam.elevation = 150, -25
    frames, nxt = [], [0.0]

    def grab(m_, d_):
        if d_.time >= nxt[0]:
            rend.update_scene(d_, cam)
            frames.append(rend.render().copy())
            nxt[0] += 1.0 / fps
    ctx = SkillContext(m, d, emb, on_step=grab)
    ctx.wait(0.2)
    est = TB.estimate(m, d, 'obj', half, rng)
    run_skill(ctx, SkillCall('pick', dict(speed=0.4), target='obj', effector=eff), est)
    run_skill(ctx, SkillCall('place', dict(x=x0 + 0.7 * (x1 - x0), y=y0 + 0.7 * (y1 - y0), z=B['table_z'], speed=0.5), effector=eff))
    ctx.wait(0.3)
    est = TB.estimate(m, d, 'obj', half, rng)
    run_skill(ctx, SkillCall('push', dict(theta=math.pi, speed=min(0.5, emb.cfg.limits['v']), distance=0.08), target='obj', effector=eff), est)
    ctx.wait(0.3)
    os.makedirs(f'{OUT}/{robot}', exist_ok=True)
    path = f'{OUT}/{robot}/{robot}_smooth.mp4'
    ff = os.path.expanduser('~/.local/bin/ffmpeg')
    p = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x480', '-r', str(fps),
                          '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', path], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    print(robot, 'video', path, len(frames), 'frames', round(d.time, 1), 's sim')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('robot')
    ap.add_argument('tests', nargs='*', default=['all'])
    ap.add_argument('--n', type=int, default=None)
    a = ap.parse_args()
    tests = a.tests if a.tests != ['all'] else ['strokes', 'pickplace', 'push', 'insert', 'sweep', 'video'] + (['handover'] if a.robot == 'aloha' else [])
    for t in tests:
        fn = globals()[t]
        if t == 'video':
            fn(a.robot)
        elif a.n:
            fn(a.robot, n=a.n)
        else:
            fn(a.robot)
