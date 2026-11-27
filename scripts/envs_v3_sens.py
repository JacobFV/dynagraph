"""envs-v3 hidden-variable sensitivity: a FIXED policy (nominal by default) over a grid of hidden-physics overrides.

    python scripts/envs_v3_sens.py <family> <robot> <n_seeds> <grid-json> [policy] [procs] [out.jsonl]
    grid-json e.g. '{"thickness": [4e-4, 1e-3, 3e-3], "mu": [0.15, 1.0]}'
Prints one row per grid point: success rate and the family's metrics_extra means.
"""
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "scripts")


def run(job):
    fam, robot, seed, over, pol = job
    from physmem.envs import family_class, make_env
    from physmem.v2.api import TaskSpec
    from envs_ceiling import apply_policy
    env = make_env(fam)
    t0 = time.time()
    try:
        env.reset(TaskSpec(fam, family_class(fam).category, seed, robot, "test", dict(over)))
        if pol.startswith("set:"):                # fixed engineer parameters other than NOMINAL (calibration grids)
            env.NOMINAL = dict(env.NOMINAL, **json.loads(pol[4:]))
            pol = "nominal"
        last = None
        for _ in range(4 * env.budget + 10):
            c = env.scripted()
            if c is None:
                break
            last = env.execute(apply_policy(env, c, pol))
            if last.done:
                break
        ex = {k: float(v) for k, v in (env.metrics_extra() or {}).items() if isinstance(v, (int, float, bool))}
        return dict(over=over, seed=seed, success=bool(env.solved()), progress=float(env.progress()),
                    unstable=bool(env.unstable), wall=time.time() - t0, sim=env.t, **ex)
    except Exception as e:
        return dict(over=over, seed=seed, error=repr(e)[:200])


if __name__ == "__main__":
    fam, robot, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    grid = json.loads(sys.argv[4])
    pol = sys.argv[5] if len(sys.argv) > 5 else "nominal"
    procs = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    out = open(sys.argv[7], "a") if len(sys.argv) > 7 else None
    keys = list(grid)
    jobs = [(fam, robot, s, dict(zip(keys, vals)), pol) for vals in itertools.product(*[grid[k] for k in keys])
            for s in range(1000, 1000 + n)]
    rows = []
    with Pool(procs) as p:
        for r in p.imap(run, jobs):
            rows.append(r)
            if out:
                out.write(json.dumps(dict(r, family=fam, robot=robot, policy=pol)) + "\n")
                out.flush()
    G = {}
    for r in rows:
        G.setdefault(json.dumps(r["over"]), []).append(r)
    for k, rs in G.items():
        ok = [r for r in rs if "error" not in r]
        mets = sorted({m for r in ok for m in r if m not in ("over", "seed", "success", "wall", "sim")})
        s = " ".join(f"{m}={np.mean([r[m] for r in ok]):.3f}" for m in mets)
        print(f"{k}: n={len(ok)} err={len(rs) - len(ok)} success={np.mean([r['success'] for r in ok]) if ok else 0:.2f} "
              f"{s} wall={np.mean([r['wall'] for r in ok]) if ok else 0:.0f}s", flush=True)
