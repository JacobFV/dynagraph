import sys, math
sys.path.insert(0, 'scripts')
import numpy as np, mujoco
import robots_g1_tasks as T
from physmem.embodiments import humanoid as H
mass = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
TZ = 0.62; half = (0.175, 0.125, 0.075); pos = np.array([0.30, 0, TZ + half[2]])
def extra(spec):
    t = spec.worldbody.add_body(name='table', pos=[0.47, 0, TZ / 2]); t.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.4, TZ / 2])
    b = spec.worldbody.add_body(name='tote', pos=list(pos)); b.add_freejoint()
    b.add_geom(name='tote', type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), mass=mass, friction=[0.9, 0.03, 1e-4], condim=4)
m, d, emb = T.scene(extra); R = T.Runner(m, d, emb); R.wait(0.5)
ctx = T.make_ctx(R); ctx.tote_body = m.body('tote').id
names = [m.actuator(i).name.split('/')[-1] for i in emb.act_ids]
nxt = [0.0]
rend = mujoco.Renderer(m, 480, 640)
def mon():
    if abs(d.time - 7.0) < 1e-6 or (6.999 < d.time < 7.0015 and not getattr(mon, 'done', False)):
        mon.done = True
        imgs = []
        for az, el in ((180, -10), (90, -10), (150, -45)):
            cm = mujoco.MjvCamera(); cm.lookat[:] = [0.35, 0, 0.72]; cm.distance = 0.8; cm.azimuth = az; cm.elevation = el
            rend.update_scene(d, cm); imgs.append(rend.render())
        import imageio.v2 as iio; iio.imwrite('results/v2/robots/g1/tote_dbg.png', np.concatenate(imgs, 1))
        print('tote xyz', d.xpos[ctx.tote_body].round(3))
    if d.time >= nxt[0]:
        nxt[0] += 0.25
        tb = ctx.tote_body; Rt = d.xmat[tb].reshape(3, 3)
        sat = [names[k] for k in range(len(names)) if abs(d.ctrl[emb.act_ids[k]]) >= 0.98 * emb.act_hi[k]]
        c = R.c
        print(round(d.time, 2), 'tote dz', round(d.xpos[tb][2] - pos[2], 3), 'tilt', round(math.degrees(math.acos(min(1, Rt[2, 2]))), 1), 'xax_z', round(Rt[2, 0], 2), 'yax_z', round(Rt[2, 1], 2),
              '| L', c.st['left']['mode'][:5], np.round(emb.tcp_pose(d, 'left')[0], 3), np.round(c.st['left']['ft'], 0), round(c.st['left']['err'], 3),
              '| R', np.round(emb.tcp_pose(d, 'right')[0], 3), np.round(c.st['right']['ft'], 0), round(c.st['right']['err'], 3),
              '| pel', round(d.xpos[emb.root_body][2], 3), 'sat', sat, flush=True)
ctx.on_step_orig = ctx.on_step
def hook(m_, d_):
    ctx.on_step_orig(m_, d_); mon()
ctx.on_step = hook
info = H.g1_tote_carry(ctx, pos, half, squeeze_n=float(np.clip(12 * mass, 20, 40)), hold_s=2.0, palm_dx=float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
print(info)
