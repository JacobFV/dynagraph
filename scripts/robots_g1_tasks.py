"""G1 + Dex3 contact tasks (ROBOTS item 5): wall impact, Dex3 pinch pick, two-palm tote carry.

python scripts/robots_g1_tasks.py wall|pick|tote|all [--n N] [--video]
Writes results/v2/robots/g1/{wall,pick,tote}.json (+ mp4).
"""
import argparse, json, math, os, subprocess, time
from pathlib import Path

import mujoco
import numpy as np

import physmem.embodiments as E
from physmem.embodiments.trajectory import BlendedTrajectory, quat2mat
from physmem.embodiments import humanoid as H
from physmem.skills import SkillContext


def make_ctx(R):
    """a SkillContext whose on_step does the Runner's bookkeeping (CoM, falls, video frames)"""
    m, d = R.m, R.d

    def hook(m_, d_):
        if int(round(d_.time / m_.opt.timestep)) % 10 == 0:
            R.com.append(d_.subtree_com[R.emb.root_body][:2].copy())
        if R.c.fallen(d_):
            R.fell = True
        if R.rend is not None and d_.time >= R.nxt:
            R.rend.update_scene(d_, R.cam)
            R.frames.append(R.rend.render().copy())
            R.nxt += 1 / 30
    return SkillContext(m, d, R.emb, on_step=hook, max_time=20.0)

OUT = Path('results/v2/robots/g1')


def scene(extra=None, prefix='h0'):
    spec = mujoco.MjSpec()
    E.scene_defaults(spec)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
    spec.worldbody.add_light(pos=[0.5, -1.5, 2.5], dir=[-0.2, 0.6, -1], diffuse=[0.8, 0.8, 0.8])
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], rgba=[0.8, 0.8, 0.8, 1])
    emb = E.make('g1_dex3', prefix=prefix)
    emb.build(spec, ((0, 0, 0), (1, 0, 0, 0)))
    if extra:
        extra(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    emb.bind(m, d)
    emb.reset(m, d)
    return m, d, emb


class Runner:
    def __init__(self, m, d, emb, video=False, cam=None):
        self.m, self.d, self.emb = m, d, emb
        self.c = emb.controller()
        self.frames, self.nxt = [], 0.0
        self.rend = mujoco.Renderer(m, 480, 640) if video else None
        self.cam = cam
        self.com = []
        self.fell = False

    def step(self, n=1, mon=None):
        m, d = self.m, self.d
        for _ in range(n):
            self.c.apply(m, d)
            mujoco.mj_step(m, d)
            if mon:
                mon()
            if int(round(d.time / m.opt.timestep)) % 10 == 0:
                self.com.append(d.subtree_com[self.emb.root_body][:2].copy())
            if self.c.fallen(d):
                self.fell = True
            if self.rend is not None and d.time >= self.nxt:
                self.rend.update_scene(d, self.cam)
                self.frames.append(self.rend.render().copy())
                self.nxt += 1 / 30
        return self

    def wait(self, s, mon=None):
        return self.step(int(round(s / self.m.opt.timestep)), mon)

    def run(self, max_t=10.0, mon=None, eff=None):
        t1 = self.d.time + max_t
        while not self.c.done(eff) and self.d.time < t1 and not self.fell:
            self.step(1, mon)

    def move(self, eff, pts, T, q=None, mon=None, max_t=None):
        p0, q0 = self.emb.tcp_pose(self.d, eff)
        q = q0 if q is None else q
        wps = [(np.asarray(p, float), q) for p in pts]
        self.c.command(BlendedTrajectory(effector=eff, waypoints=wps, durations=list(T)))
        self.run(max_t if max_t else sum(T) * 2 + 1, mon, eff)

    def save_video(self, path):
        if not self.frames:
            return
        ff = os.path.expanduser('~/.local/bin/ffmpeg')
        p = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x480', '-r', '30',
                              '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(path)], stdin=subprocess.PIPE)
        for f in self.frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()

    def com_exc(self):
        C = np.array(self.com)
        return float(np.max(np.linalg.norm(C - C[0], axis=1))) * 1e3 if len(C) else 0.0


def cam(lookat, dist=1.6, az=150, el=-20):
    c = mujoco.MjvCamera()
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


# ----------------------------------------------------------------------------- (a) wall impact
def wall_test(speeds=(0.05, 0.1, 0.2, 0.3), modes=('guarded', 'track')):
    rows = []
    for mode in modes:
        for v in speeds:
            def extra(spec):
                b = spec.worldbody.add_body(name='wall', pos=[0.32, -0.2, 0.8])
                b.add_geom(name='wall', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.25, 0.4], rgba=[0.6, 0.6, 0.7, 1])
            m, d, emb = scene(extra)
            R = Runner(m, d, emb)
            R.wait(0.5)
            c = R.c
            p, q = emb.tcp_pose(d, 'right')
            tgt = np.array([0.45, p[1], p[2]])
            if mode == 'guarded':
                c.set_mode('guarded', 'right', threshold=15.0)
            T = float(np.linalg.norm(tgt - p) / v)
            Fpk = [0.0]

            def mon():
                Fpk[0] = max(Fpk[0], float(np.linalg.norm(c.st['right']['ft'])))
            c.command(BlendedTrajectory(effector='right', waypoints=[(tgt, q)], durations=[T]))
            R.run(T + 0.5, mon)
            c.set_mode('track', 'right')
            R.wait(2.0, mon)
            rows.append(dict(mode=mode, v=v, fell=R.fell, tripped=bool(c.st['right'].get('trip_t')), peak_force_n=round(Fpk[0], 1),
                             qp=dict(c.qp_status), com_exc_mm=round(R.com_exc(), 1), pelvis_z=round(float(d.xpos[emb.root_body][2]), 3)))
            print(rows[-1], flush=True)
    json.dump(rows, open(OUT / 'wall.json', 'w'), indent=1)
    return rows


# ----------------------------------------------------------------------------- (b) Dex3 pinch pick
def pick_test(n=10, video=False):
    rows = []
    for s in range(n):
        rng = np.random.default_rng(700 + s)
        hs = rng.uniform(0.02, 0.03)
        half = (hs, hs, hs)
        mass = rng.uniform(0.1, 0.3)
        TZ = 0.70
        side = 'right' if s % 2 == 0 else 'left'
        sg = -1 if side == 'right' else 1
        pos = np.array([0.30 + rng.uniform(-0.02, 0.02), sg * (0.16 + rng.uniform(-0.02, 0.02)), TZ + hs])

        def extra(spec):
            t = spec.worldbody.add_body(name='table', pos=[0.45, 0, TZ / 2])
            t.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.4, TZ / 2], rgba=[0.55, 0.5, 0.45, 1], friction=[0.8, 0.005, 1e-4])
            b = spec.worldbody.add_body(name='obj', pos=list(pos))
            b.add_freejoint()
            b.add_geom(name='obj', type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), mass=mass, rgba=[0.2, 0.5, 0.9, 1],
                       friction=[1.0, 0.005, 1e-4], condim=4)
        m, d, emb = scene(extra)
        R = Runner(m, d, emb, video=video and s == 0, cam=cam([0.25, 0, 0.75], 1.3, 140, -25))
        R.wait(0.5)
        est = pos + rng.normal(0, 0.003, 3) * np.array([1, 1, 0])   # tracker estimate
        ctx = make_ctx(R)
        st = H.g1_pinch_pick(ctx, side, est, half)
        ctx.wait(3.0)
        ob = d.xpos[m.body('obj').id]
        lifted = float(ob[2] - pos[2])
        ok = bool(lifted > 0.08 and not R.fell)
        rows.append(dict(seed=s, side=side, mass=round(mass, 3), size=round(2 * hs, 3), status=st, lifted_m=round(lifted, 3),
                         fell=R.fell, success=ok, qp=dict(R.c.qp_status)))
        print(rows[-1], flush=True)
        if R.rend is not None:
            R.save_video(OUT / 'g1_pinch_pick.mp4')
    summ = dict(n=n, success=float(np.mean([r['success'] for r in rows])), falls=int(sum(r['fell'] for r in rows)))
    json.dump(dict(summary=summ, rows=rows), open(OUT / 'pick.json', 'w'), indent=1)
    print('PICK', summ)
    return summ


# ----------------------------------------------------------------------------- (c) two-palm tote carry
def tote_test(n=10, video=False):
    rows = []
    for s in range(n):
        rng = np.random.default_rng(800 + s)
        mass = 1.0 + 2.0 * s / max(n - 1, 1)
        TZ = 0.62
        half = (0.175, 0.125, 0.075)       # 0.35 (x, depth) x 0.25 (y, between the palms) x 0.15 m
        pos = np.array([0.30 + rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), TZ + half[2]])

        def extra(spec):
            t = spec.worldbody.add_body(name='table', pos=[0.47, 0, TZ / 2])
            t.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.4, TZ / 2], rgba=[0.55, 0.5, 0.45, 1], friction=[0.8, 0.005, 1e-4])
            b = spec.worldbody.add_body(name='tote', pos=list(pos))
            b.add_freejoint()
            # torsional friction 0.03 m: a ~3 cm palm contact patch (MuJoCo point contacts otherwise pivot freely)
            b.add_geom(name='tote', type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), mass=mass, rgba=[0.9, 0.5, 0.1, 1],
                       friction=[0.9, 0.03, 1e-4], condim=4)
        m, d, emb = scene(extra)
        R = Runner(m, d, emb, video=video and s == n - 1, cam=cam([0.3, 0, 0.8], 1.8, 135, -20))
        R.wait(0.5)
        est = pos + rng.normal(0, 0.003, 3) * np.array([1, 1, 0])
        tq = [0.0]

        def mon():
            tq[0] = max(tq[0], float(np.max(np.abs(d.ctrl[emb.act_ids] / emb.act_hi))))
        ctx = make_ctx(R)
        ctx.tote_body = m.body('tote').id
        info = H.g1_tote_carry(ctx, est, half, squeeze_n=float(np.clip(12 * mass, 20, 40)), mon=mon, carry_dy=0.15 if s % 2 else 0.0, hold_s=5.0)
        ob = d.xpos[m.body('tote').id]
        ok = bool(info.get('max_lift', 0) > 0.08 and not info.get('dropped') and not R.fell)
        info.pop('fell', None)
        rows.append(dict(seed=s, mass=round(mass, 2), success=ok, fell=R.fell, com_exc_mm=round(R.com_exc(), 1),
                         peak_torque_frac=round(tq[0], 3), final_tote_z=round(float(ob[2] - pos[2]), 3),
                         qp=dict(R.c.qp_status), **info))
        print(rows[-1], flush=True)
        if R.rend is not None:
            R.save_video(OUT / 'g1_tote_carry.mp4')
    summ = dict(n=n, success=float(np.mean([r['success'] for r in rows])), falls=int(sum(r['fell'] for r in rows)),
                com_exc_mm_max=max(r['com_exc_mm'] for r in rows), peak_torque_frac_max=max(r['peak_torque_frac'] for r in rows))
    json.dump(dict(summary=summ, rows=rows), open(OUT / 'tote.json', 'w'), indent=1)
    print('TOTE', summ)
    return summ


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('test')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--video', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.test in ('wall', 'all'):
        wall_test()
    if a.test in ('pick', 'all'):
        pick_test(a.n, a.video)
    if a.test in ('tote', 'all'):
        tote_test(a.n, a.video)
    print('wall time', round(time.time() - t0, 1))
