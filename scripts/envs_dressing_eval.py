"""Dressing (envs-r6) checks: scripted-baseline success with vs without the site layer (same seeds), and wall-time
per physics step overhead.  python scripts/envs_dressing_eval.py [families] [n] [procs] [robot]"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from physmem.envs import family_class, make_env
from physmem.envs.rollout import run_episode
from physmem.v2.api import TaskSpec

FAMS = ["depalletize", "conveyor_sort", "tote_transfer", "shelf_restock", "kitting_pocket"]


def job(a):
    fam, seed, dressed, robot = a
    p = dict(policy="scripted")
    if dressed:
        p["dressing"] = "site"
    t = TaskSpec(fam, family_class(fam).category, seed, robot, "train", p)
    t0 = time.time()
    r = run_episode(t, keep_events=False)
    return dict(family=fam, seed=seed, dressed=dressed, success=r["success"], err=r["error"],
                progress=(r["final_metrics"] or {}).get("progress"), sim_t=r["sim_t"], wall=time.time() - t0,
                violations=(r["final_metrics"] or {}).get("integrity_violations", 0),
                collisions=(r["final_metrics"] or {}).get("collisions", 0))


def step_cost(fam, dressed, robot="probe", T=4.0):
    p = {"dressing": "site"} if dressed else {}
    env = make_env(fam)
    env.reset(TaskSpec(fam, family_class(fam).category, 3, robot, "train", p))
    n = int(T / env.m.opt.timestep)
    t0 = time.perf_counter()
    env.run(T)
    return (time.perf_counter() - t0) / n * 1e6, env.m.ngeom


def main():
    fams = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else FAMS
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    robot = sys.argv[4] if len(sys.argv) > 4 else "probe"
    jobs = [(f, s, d, robot) for f in fams for s in range(n) for d in (False, True)]
    with Pool(procs) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    out = {}
    for f in fams:
        a = [r for r in rows if r["family"] == f and not r["dressed"]]
        b = [r for r in rows if r["family"] == f and r["dressed"]]
        us0, ng0 = step_cost(f, False, robot)
        us1, ng1 = step_cost(f, True, robot)
        out[f] = dict(n=n, robot=robot, success_bare=float(np.mean([r["success"] for r in a])),
                      success_site=float(np.mean([r["success"] for r in b])),
                      progress_bare=float(np.mean([r["progress"] or 0 for r in a])),
                      progress_site=float(np.mean([r["progress"] or 0 for r in b])),
                      paired_flips=sum(int(x["success"] != y["success"]) for x, y in zip(a, b)),
                      violations_site=int(sum(r["violations"] or 0 for r in b)),
                      collisions_bare=float(np.mean([r["collisions"] or 0 for r in a])),
                      collisions_site=float(np.mean([r["collisions"] or 0 for r in b])),
                      us_per_step_bare=round(us0, 1), us_per_step_site=round(us1, 1),
                      overhead_pct=round(100 * (us1 / us0 - 1), 1), ngeom_bare=ng0, ngeom_site=ng1,
                      errors=[r["err"] for r in rows if r["family"] == f and r["err"]][:3])
        print(f, json.dumps(out[f]))
    json.dump(out, open(f"results/v2/envs/dressing_eval_{robot}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
