import sys
import numpy as np, mujoco
import physmem.embodiments as E
name, eff = sys.argv[1], sys.argv[2]
dv = np.array([float(x) for x in sys.argv[3].split(',')])
emb = E.make(name, prefix='r0')
spec = mujoco.MjSpec(); E.scene_defaults(spec)
emb.build(spec, ((0, 0, 0), (1, 0, 0, 0)))
m = spec.compile(); d = mujoco.MjData(m)
emb.bind(m, d); emb.reset(m, d)
c = emb.controller()
names = [m.actuator(i).name for i in emb.act_ids]
p, q = emb.tcp_pose(d, eff)
c.command(E.make_traj(eff, [(p + dv, q), (p, q)], durations=[1.0, 1.0]))
k = 0
while not c.done():
    u = c.apply(m, d); mujoco.mj_step(m, d); k += 1
    if k % 50 == 0:
        e = emb.eff[eff]
        sat = [names[i] for i in range(len(u)) if abs(u[i]) >= 0.99 * emb.act_hi[i]]
        print(round(d.time, 2), 'err', round(c.st[eff]['err'] * 1e3, 1), 'alpha', round(c.st[eff]['alpha'], 2), 'q', d.qpos[e['qadr']].round(2), 'sat', sat)
