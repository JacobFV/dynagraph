"""G1 in an ENVS station: stance, hand TCPs, targets, IK reachability (palm-down) per hand."""
import sys, math
import numpy as np
from physmem.envs import make_env
from physmem.v2.api import TaskSpec
from physmem.embodiments.ik import solve_ik, down_quat
fam, rob, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
env = make_env(fam)
env.reset(TaskSpec(fam, env.category, seed, rob, 'train'))
r = env.robot; emb = r.emb; m, d = env.m, env.d
print('mount', np.round(r.mount[0], 3), round(r.mount[1], 2), 'pelvis', d.xpos[emb.root_body].round(3), 'surface', getattr(env, 'surface_z', None), 'fallen', r.ctrl.fallen(d))
for k in emb.eff:
    print('tcp', k, emb.tcp_pose(d, k)[0].round(3))
rg = set(int(g) for g in emb.geoms)
print('robot contacts', [(m.geom(d.contact.geom1[i]).name or m.body(m.geom_bodyid[d.contact.geom1[i]]).name, m.geom(d.contact.geom2[i]).name or m.body(m.geom_bodyid[d.contact.geom2[i]]).name) for i in range(d.ncon) if (d.contact.geom1[i] in rg) != (d.contact.geom2[i] in rg)][:8])
c = env.scripted(); print('call', c)
for eid, info in list(env.scene.entities.items())[:6]:
    e = env.est_pose(eid)
    if e is None:
        continue
    p = np.array(e[0])
    out = []
    for k in emb.eff:
        for dz in (0.03, 0.08):
            q, ok, ep, er = solve_ik(emb, m, d, k, p + [0, 0, dz], down_quat(0.0), iters=400)
            q2, ok2, ep2, er2 = solve_ik(emb, m, d, k, p + [0, 0, dz], None, iters=400)
            out.append((k, dz, round(ep, 3), round(er, 2), 'posonly', round(ep2, 3), 'waist', round(float(q2[0]), 2)))
    print(eid, p.round(3), 'dist_pelvis_xy', round(float(np.linalg.norm(p[:2] - d.xpos[emb.root_body][:2])), 3), out)
