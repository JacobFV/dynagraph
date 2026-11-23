"""item 6: ENVS articulated station (drawer_open / door_open) with real ROBOTS embodiments via envs/robots.py."""
import json, sys, time
import numpy as np
from physmem.envs import make_env
from physmem.v2.api import TaskSpec

robots = sys.argv[1].split(',') if len(sys.argv) > 1 else ['panda', 'ur5e_2f85']
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
out = {}
for fam in ('drawer_open', 'door_open'):
    for robot in robots:
        rows = []
        for seed in range(n):
            env = make_env(fam)
            t0 = time.time()
            try:
                env.reset(TaskSpec(fam, 'articulated', seed, robot=robot, split='val'))
                rng = np.random.default_rng(seed)
                cands = [c for c in env.candidate_skills(None, 16, rng) if c.skill == 'open']
                cands.sort(key=lambda c: -float(c.params.get('travel', 0)))
                so = env.execute(cands[0]) if cands else None
                rows.append(dict(seed=seed, status=None if so is None else so.result.status, success=bool(so and so.success),
                                 metrics={k: v for k, v in (so.metrics if so else {}).items() if isinstance(v, (int, float, bool, str))},
                                 params=None if not cands else cands[0].params, wall=time.time() - t0))
            except Exception as ex:
                rows.append(dict(seed=seed, status='error', error=repr(ex)[:200], success=False))
            print(fam, robot, rows[-1], flush=True)
        out[f'{fam}/{robot}'] = dict(success=float(np.mean([r['success'] for r in rows])), rows=rows)
json.dump(out, open('results/v2/robots/articulated.json', 'w'), indent=1, default=str)
print({k: v['success'] for k, v in out.items()})
