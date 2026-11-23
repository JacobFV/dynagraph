"""find a down-pointing home posture per robot by IK toward cfg.home_target (mount at origin)."""
import sys
import numpy as np, mujoco
import physmem.embodiments as E
from physmem.embodiments.ik import solve_ik, down_quat
targets = {'aloha': {'left': (-0.22, 0.0, 0.15), 'right': (0.22, 0.0, 0.15)}}
for name in (sys.argv[1:] or list(E.ROBOTS)):
    emb = E.make(name)
    spec = mujoco.MjSpec(); E.scene_defaults(spec)
    emb.build(spec, ((0, 0, 0), (1, 0, 0, 0)))
    m = spec.compile(); d = mujoco.MjData(m)
    emb.bind(m, d); emb.reset(m, d)
    for en in emb.eff:
        p, q = emb.tcp_pose(d, en)
        tgt = targets.get(name, {}).get(en, emb.cfg.home_target)
        best = None
        for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 2):
            qq, ok, ep, er = solve_ik(emb, m, d, en, tgt, down_quat(yaw), iters=500, rot_weight=1.0)
            if best is None or ep + 0.1 * er < best[2] + 0.1 * best[3]:
                best = (qq, ok, ep, er, yaw)
        print(name, en, 'tcp_now', p.round(3), 'tgt', tgt, 'ok', best[1], 'err', round(best[2], 4), round(best[3], 3), 'yaw', round(best[4], 2), 'q', np.round(best[0], 4).tolist())
