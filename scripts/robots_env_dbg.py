"""debug one family x robot: mount, target estimates, IK reachability of the first scripted call."""
import sys, math
import numpy as np
from physmem.envs import make_env
from physmem.v2.api import TaskSpec
from physmem.embodiments.ik import solve_ik, down_quat
fam, rob, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
env = make_env(fam)
env.reset(TaskSpec(fam, env.category, seed, rob, 'train'))
r = env.robot
emb = r.emb
print('mount', np.round(r.mount[0], 3), round(r.mount[1], 2), 'kind', emb.spec.kind, 'surface_z', getattr(env, 'surface_z', None))
for k in emb.eff:
    p, q = emb.tcp_pose(env.d, k)
    print('tcp', k, p.round(3))
c = env.scripted()
print('first call', c)
if c is not None and c.target in env.scene.entities and env.est_pose(c.target) is not None and False:
    est = env.est_pose(c.target)
    print('est', np.round(est[0], 3), 'size', np.round(env.est_size(c.target), 3))
    eff = r.eff
    for dz in (0.0, 0.05, 0.1, 0.2):
        tgt = np.array(est[0]) + [0, 0, env.est_size(c.target)[2] + dz]
        qq, ok, ep, er = solve_ik(emb, env.m, env.d, eff, tgt, down_quat(0.0), iters=400)
        print(' IK top+%.2f' % dz, tgt.round(3), ok, round(ep, 4), round(er, 3), 'dist from mount', round(float(np.linalg.norm(tgt - r.mount[0])), 3))

# --- trace: run the scripted policy, print the TRUE pose of each call's target after the call (debug only)
import mujoco
calls = 0
while calls < int(sys.argv[4]) if len(sys.argv) > 4 else 6:
    c = env.scripted() if calls else c
    if c is None:
        break
    out = env.execute(c)
    calls += 1
    tgt = c.target if c.target in env.scene.entities else env.held
    info = env.scene.entities.get(c.target) or env.scene.entities.get(getattr(env, '_last_pick', '') or '')
    line = [c.skill, c.target, out.result.status, round(out.metrics.get('progress', 0), 3), 'drops', out.metrics.get('drops'),
            'coll', out.metrics.get('collisions'), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.result.log.items() if k not in ('peaks', 'skill')}]
    if c.target in env.scene.entities:
        b = env.m.body(env.scene.entities[c.target].body).id
        line += ['true', env.d.xpos[b].round(3).tolist()]
        env._last_pick = c.target
    elif getattr(env, '_last_pick', None):
        b = env.m.body(env.scene.entities[env._last_pick].body).id
        line += ['last-picked true', env.d.xpos[b].round(3).tolist(), 'region', {k: v for k, v in env.region(c.target).items()} if c.target and env.region(c.target) else None]
    line += ['tcp', r.emb.tcp_pose(env.d, r.eff)[0].round(3).tolist()]
    print(*line)
    if out.done:
        print('DONE success', out.success)
        break
