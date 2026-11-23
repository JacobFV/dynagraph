import sys
import numpy as np, mujoco
import physmem.embodiments as E
name = sys.argv[1]
emb = E.make(name, prefix='r0')
spec = mujoco.MjSpec(); E.scene_defaults(spec)
emb.build(spec, ((0, 0, 0), (1, 0, 0, 0)))
m = spec.compile(); d = mujoco.MjData(m)
emb.bind(m, d); emb.reset(m, d)
c = emb.controller()
names = [m.actuator(i).name for i in emb.act_ids]
T = []
for k in range(int(sys.argv[2]) if len(sys.argv) > 2 else 300):
    u = c.apply(m, d); mujoco.mj_step(m, d); T.append(u.copy())
T = np.array(T)
rate = np.abs(np.diff(T, axis=0)).max(0) / 0.002
for n, lo, hi, r, last in zip(names, emb.act_lo, emb.act_hi, rate, T[-1]):
    print(f'{n:30s} lim {hi:7.2f} maxrate {r:10.1f} last {last:8.3f} min {T[:, names.index(n)].min():8.3f} max {T[:, names.index(n)].max():8.3f}')
for en, e in emb.eff.items():
    print(en, 'err', c.st[en]['err'], 'alpha', c.st[en].get('alpha'), 'q', d.qpos[e['qadr']].round(3), 'home', e['home'])
