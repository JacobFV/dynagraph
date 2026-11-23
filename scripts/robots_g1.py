"""G1 + Dex3 gate: stand idle, then N hand strokes (0.2-0.3 m, commanded mid-stroke speed 0.1-0.6 m/s, blended
via-points with constant-speed cruise), measuring mid-stroke tracking error, commanded-vs-achieved speed ratio, CoM
excursion, falls, QP status, effort peaks, sim/wall speed. Writes results/v2/robots/g1/gate.json (+ video).

    python scripts/robots_g1.py --robot g1_dex3 --strokes 50 --idle 30 --video
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

import physmem.embodiments as E
from physmem.embodiments.trajectory import BlendedTrajectory
from physmem.embodiments import trajectory as _traj


def _hermite5_fixed(p0, v0, p1, v1, T, t):
    """quintic Hermite (zero end accelerations) INCLUDING the end-velocity basis h11 (missing in trajectory.py at
    the time of writing; reported to the module owner). Installed only if the module's version is wrong."""
    u = min(max(t / T, 0.0), 1.0)
    u2, u3, u4, u5 = u * u, u ** 3, u ** 4, u ** 5
    H = [(1 - 10 * u3 + 15 * u4 - 6 * u5, -30 * u2 + 60 * u3 - 30 * u4, -60 * u + 180 * u2 - 120 * u3, -60 + 360 * u - 360 * u2),
         (u - 6 * u3 + 8 * u4 - 3 * u5, 1 - 18 * u2 + 32 * u3 - 15 * u4, -36 * u + 96 * u2 - 60 * u3, -36 + 192 * u - 180 * u2),
         (10 * u3 - 15 * u4 + 6 * u5, 30 * u2 - 60 * u3 + 30 * u4, 60 * u - 180 * u2 + 120 * u3, 60 - 360 * u + 360 * u2),
         (-4 * u3 + 7 * u4 - 3 * u5, -12 * u2 + 28 * u3 - 15 * u4, -24 * u + 84 * u2 - 60 * u3, -24 + 168 * u - 180 * u2)]
    out = []
    for i in range(4):
        out.append((H[0][i] * p0 + H[1][i] * T * v0 + H[2][i] * p1 + H[3][i] * T * v1) / T ** i)
    p, v, a, j = out
    if t >= T:
        p, v, a, j = p1, v1, a * 0.0, j * 0.0
    elif t <= 0:
        p, v, a, j = p0, v0, a * 0.0, j * 0.0
    return p, v, a, j


_z, _o = np.zeros(3), np.ones(3)
if np.linalg.norm(_traj._hermite5(_z, _o, _o, _o, 1.0, 0.5)[1] - _o) > 1e-6:  # constant-velocity segment must stay at v
    print('NOTE: patching trajectory._hermite5 (end-velocity basis missing)')
    _traj._hermite5 = _hermite5_fixed

ap = argparse.ArgumentParser()
ap.add_argument('--robot', default='g1_dex3')
ap.add_argument('--strokes', type=int, default=50)
ap.add_argument('--idle', type=float, default=30.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--vmin', type=float, default=0.1)
ap.add_argument('--vmax', type=float, default=0.6)
ap.add_argument('--video', action='store_true')
ap.add_argument('--video_s', type=float, default=8.0)
ap.add_argument('--out', default='results/v2/robots/g1')
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

emb = E.make(args.robot, prefix='r0')
spec = mujoco.MjSpec()
E.scene_defaults(spec)
spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
spec.add_texture(name='grid', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=[0.6, 0.62, 0.64], rgb2=[0.52, 0.54, 0.56], width=256, height=256)
spec.add_material(name='grid', textures=['', 'grid'], texrepeat=[8, 8])
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], material='grid')
spec.worldbody.add_light(pos=[1, -1, 3], dir=[-0.3, 0.3, -1], diffuse=[0.8, 0.8, 0.8])
emb.build(spec, ((0, 0, 0), (1, 0, 0, 0)))
m = spec.compile()
d = mujoco.MjData(m)
emb.bind(m, d)
emb.reset(m, d)
c = emb.controller()
root = emb.root_body

renderer = frames = None
if args.video:
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.15, 0, 0.85]
    cam.distance, cam.azimuth, cam.elevation = 1.9, 150, -15
    frames = []
next_frame = [0.0]
video_t0 = [None]

stats = dict(com=[], pelvis_min=np.inf, fell=False)


def tick():
    c.apply(m, d)
    mujoco.mj_step(m, d)
    stats['com'].append(d.subtree_com[root][:2].copy())
    stats['pelvis_min'] = min(stats['pelvis_min'], float(d.xpos[root][2]))
    if c.fallen(d) if hasattr(c, 'fallen') else False:
        stats['fell'] = True
    if renderer is not None and video_t0[0] is not None and d.time - video_t0[0] <= args.video_s and d.time >= next_frame[0]:
        renderer.update_scene(d, cam)
        frames.append(renderer.render().copy())
        next_frame[0] = d.time + 1 / 30


wall0 = time.time()
# ------------------------------------------------------------------ idle stand
foot_mid = None
n_idle = int(args.idle / m.opt.timestep)
for _ in range(n_idle):
    tick()
    if stats['fell']:
        break
idle_com = np.array(stats['com'])
idle = dict(seconds=round(d.time, 2), fell=stats['fell'], pelvis_min=round(stats['pelvis_min'], 4),
            com_excursion_mm=round(1e3 * float(np.max(np.linalg.norm(idle_com - idle_com[len(idle_com) // 2:].mean(0), axis=1))), 2)
            if len(idle_com) else None,
            com_drift_last10s_mm=round(1e3 * float(np.ptp(idle_com[-5000:], axis=0).max()), 2))
print('idle', idle, flush=True)

# ------------------------------------------------------------------ strokes
home = {e: emb.tcp_pose(d, e) for e in emb.eff}
video_t0[0] = d.time
next_frame[0] = d.time
rows = []
com0 = np.array(stats['com'][-1])
com_stroke = []
for k in range(args.strokes):
    if stats['fell']:
        break
    eff = ('left', 'right')[k % 2]
    sg = 1 if eff == 'left' else -1
    p0, q0 = home[eff]
    L = rng.uniform(0.2, 0.3)
    v = float(np.exp(rng.uniform(np.log(args.vmin), np.log(args.vmax))))
    a = rng.uniform(-0.9, 0.9)
    u = np.array([np.cos(a), np.sin(a), 0.0]) * rng.choice([-1, 1])
    M = p0 + np.array([rng.uniform(0.0, 0.06), sg * rng.uniform(-0.02, 0.06), rng.uniform(-0.05, 0.05)])
    S, Eend = M - u * L / 2, M + u * L / 2
    la = max(0.03, 0.12 * v)
    A, B = S + u * la, Eend - u * la
    # reposition (rest-to-rest), then the stroke: accelerate to v, cruise at v, decelerate
    T0 = max(0.5, float(np.linalg.norm(S - emb.tcp_pose(d, eff)[0])) / 0.2)
    c.command(BlendedTrajectory(effector=eff, waypoints=[(S, q0)], durations=[T0]))
    tr = BlendedTrajectory(effector=eff, waypoints=[(A, q0), (B, q0), (Eend, q0)],
                           durations=[2 * la / v, (L - 2 * la) / v, 2 * la / v], via_vel=[u * v, u * v, None])
    pred = c.command(tr)
    plan = c.st[eff]['plans'][-1]
    v_cmd = v / plan.scale
    t_cruise0 = plan.t0 + plan.starts[1]
    t_cruise1 = plan.t0 + plan.starts[2]
    w0 = t_cruise0 + 0.2 * (t_cruise1 - t_cruise0)
    w1 = t_cruise1 - 0.2 * (t_cruise1 - t_cruise0)
    errs, sp, stalls = [], [], 0
    sat = set()
    last_prog, t_end = None, plan.t0 + plan.duration
    while not c.done():
        tick()
        if w0 <= d.time <= w1:
            errs.append(c.st[eff]['err'])
            uu = d.ctrl[emb.act_ids]
            for i in np.where(np.abs(uu) >= 0.99 * emb.act_hi)[0]:
                sat.add(m.actuator(emb.act_ids[i]).name.split('/')[-1].replace('_joint', ''))
            vv, _ = emb.tcp_vel(m, d, eff)
            sp.append(float(vv @ u))
        if d.time > t_end + 5.0:
            stalls += 1
            break
    for _ in range(50):  # 0.1 s settle (not a convergence wait: fixed)
        tick()
    com_stroke.append(np.array(stats['com'][-1]) - com0)
    fin_err = c.st[eff]['err']
    rows.append(dict(k=k, eff=eff, L=round(L, 3), v_cmd=round(v_cmd, 4), scale=round(plan.scale, 4),
                     mid_err_mm_max=round(1e3 * max(errs), 3) if errs else None,
                     mid_err_mm_mean=round(1e3 * float(np.mean(errs)), 3) if errs else None,
                     v_ach=round(float(np.mean(sp)), 4) if sp else None,
                     ratio=round(float(np.mean(sp)) / v_cmd, 4) if sp else None,
                     end_err_mm=round(1e3 * fin_err, 2), stalled=bool(stalls), n_mid=len(errs), saturated=sorted(sat)))
    print(rows[-1], flush=True)
wall = time.time() - wall0

# ------------------------------------------------------------------ summary
ok = [r for r in rows if r['ratio'] is not None]
ratios = np.array([r['ratio'] for r in ok])
merr = np.array([r['mid_err_mm_max'] for r in ok])
com = np.array(stats['com'])
eff_d = c.effort.as_dict()
summary = dict(
    robot=args.robot, floating_base=emb.spec.floating_base, seed=args.seed, idle=idle,
    strokes=len(rows), strokes_measured=len(ok), fell=stats['fell'], pelvis_min=round(stats['pelvis_min'], 4),
    mid_err_mm=dict(p50=round(float(np.percentile(merr, 50)), 3), p95=round(float(np.percentile(merr, 95)), 3),
                    max=round(float(merr.max()), 3)) if len(ok) else None,
    speed_ratio=dict(p10=round(float(np.percentile(ratios, 10)), 4), p50=round(float(np.percentile(ratios, 50)), 4),
                     p90=round(float(np.percentile(ratios, 90)), 4)) if len(ok) else None,
    stalls=int(sum(r['stalled'] for r in rows)),
    com_excursion_during_strokes_mm=round(1e3 * float(np.max(np.linalg.norm(np.array(com_stroke), axis=1))), 2) if com_stroke else None,
    com_range_total_mm=[round(1e3 * float(x), 2) for x in np.ptp(com, axis=0)],
    qp_status=c.qp_status if hasattr(c, 'qp_status') else None,
    effort=dict({k: round(float(v), 4) for k, v in eff_d.items()}),
    peaks=dict({k: round(float(v), 3) for k, v in c.acc.peaks.items()}),
    sim_s=round(d.time, 2), wall_s=round(wall, 2), sim_per_wall=round(d.time / wall, 2),
    rows=rows,
    note='peaks.joint_jerk is the raw 2 ms encoder finite difference over ALL actuated joints incl. foot touchdown at t=0',
)
gates = dict(mid_err_lt_1cm=bool(len(ok) and merr.max() < 10.0),
             ratio_p10_p90_within_7pct=bool(len(ok) and ratios.min() >= 0 and np.percentile(ratios, 10) >= 0.93 and np.percentile(ratios, 90) <= 1.07),
             no_stalls=summary['stalls'] == 0, no_fall=not stats['fell'])
summary['gates'] = gates
tag = args.robot
(out / f'gate_{tag}.json').write_text(json.dumps(summary, indent=1))
if args.robot == 'g1_dex3':
    (out / 'gate.json').write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))

if frames:
    ff = os.path.expanduser('~/.local/bin/ffmpeg')
    vid = out / f'{tag}_strokes.mp4'
    p = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x480', '-r', '30',
                          '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(vid)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    print('video', vid, len(frames), 'frames')
