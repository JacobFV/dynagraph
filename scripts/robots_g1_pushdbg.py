"""trace G1 pushes in singulation: hand tracking error, object displacement (true pose; debug only)."""
import sys, math
import numpy as np
from physmem.envs import make_env
from physmem.v2.api import TaskSpec
env = make_env('singulation'); env.reset(TaskSpec('singulation', env.category, int(sys.argv[1]) if len(sys.argv) > 1 else 0, 'g1_dex3', 'train'))
r = env.robot; m, d = env.m, env.d
import physmem.embodiments.humanoid as H
import mujoco
REC = {}
CON = {}
_st = H.WholeBodyController.step
def step(self, model, data):
    out = _st(self, model, data)
    rg = self.emb.own_mask
    for i in range(data.ncon):
        g1, g2 = data.contact.geom1[i], data.contact.geom2[i]
        if rg[g1] != rg[g2]:
            gr, go = (g1, g2) if rg[g1] else (g2, g1)
            bn = model.body(model.geom_bodyid[gr]).name
            if 'ankle' not in bn:
                CON[(bn, model.geom(go).name or model.body(model.geom_bodyid[go]).name)] = CON.get((bn, model.geom(go).name or model.body(model.geom_bodyid[go]).name), 0) + 1
    for e in self.st:
        REC.setdefault(e, []).append((self.st[e]['err'], *self.emb.tcp_pose(data, e)[0], *(self.st[e]['ref'][0] if self.st[e].get('ref') is not None else (0, 0, 0))))
    return out
H.WholeBodyController.step = step
for k in range(4):
    REC.clear(); CON.clear()
    c = env.scripted()
    if c is None:
        break
    b = m.body(env.scene.entities[c.target].body).id
    p0 = d.xpos[b].copy()
    errs = []
    ctx = r.skills._ctx(env) if hasattr(r, 'skills') else None
    out = env.execute(c)
    est = env.est_pose(c.target)
    eff = out.result.log.get('eff')
    if eff in REC:
        A = np.array(REC[eff]); i = int(np.argmax(A[:, 0]))
        dist = np.min(np.linalg.norm(A[:, 1:4] - p0, axis=1))
        print('   max err', round(A[i, 0], 3), 'at tcp', A[i, 1:4].round(3), 'ref', A[i, 4:7].round(3), 'min tcp-obj dist', round(float(dist), 3), 'obj0', p0.round(3))
    print('   contacts', sorted(CON.items(), key=lambda x: -x[1])[:6])
    print('   log', {k: v for k, v in out.result.log.items() if k in ('touch', 'stop', 'tcp_end', 'run', 'ik', 'drops', 'H')})
    print(c.skill, c.target, out.result.status, out.result.log.get('eff'), 'moved', np.round(d.xpos[b] - p0, 3), 'prog', round(out.metrics['progress'], 2),
          'pelvis', d.xpos[r.emb.root_body].round(3), 'fallen', r.ctrl.fallen(d), 'err', {e: round(r.ctrl.st[e]['err'], 3) for e in r.emb.eff})
