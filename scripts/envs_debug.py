"""ad-hoc debugging of single skills (dev tool)."""
import sys, json, math
import numpy as np
from physmem.envs import make_env
from physmem.v2.api import TaskSpec, SkillCall

fam = sys.argv[1]; seed = int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
env = make_env(fam)
env.reset(TaskSpec(fam, env.category, seed, "probe", "train"))
for k in range(n):
    c = env.scripted()
    if c is None: break
    tgt = c.target
    info = env.scene.entities.get(tgt)
    pre = None
    if info is not None:
        p, q = env.true_pose(tgt); est = env.est_pose(tgt)
        pre = dict(true=np.round(p, 4).tolist(), est=np.round(est[0], 4).tolist() if est else None, half=np.round(info.half, 3).tolist(),
                   shape=info.shape, hidden={k2: round(v, 3) if isinstance(v, float) else v for k2, v in info.hidden.items()})
    out = env.execute(c)
    r = env.robot
    print(c.skill, tgt, {k2: round(v, 3) for k2, v in c.params.items()}, "->", out.result.status, out.result.log,
          "width", round(r.grip_width(), 4), "tcp", np.round(r.tcp()[0], 3), "prog", round(out.metrics["progress"], 3),
          "dur", round(out.result.duration_s, 2), "effort", round(out.result.effort.cost(), 2))
    if pre: print("    ", pre)
