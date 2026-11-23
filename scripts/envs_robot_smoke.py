"""Scripted episodes with REAL embodiments: python scripts/envs_robot_smoke.py fams robots seeds [-v]
prints one JSON line per episode."""
import json, sys, time, traceback
from physmem.envs import make_env
from physmem.v2.api import TaskSpec

fams = sys.argv[1].split(",")
robots = sys.argv[2].split(",")
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
verbose = "-v" in sys.argv
for rob in robots:
    for fam in fams:
        for seed in seeds:
            t0 = time.time()
            row = dict(robot=rob, fam=fam, seed=seed)
            try:
                env = make_env(fam)
                env.reset(TaskSpec(fam, env.category, seed, rob, "train"))
                row["reset_wall"] = round(time.time() - t0, 2)
                out, steps, eff = None, [], 0.0
                while True:
                    c = env.scripted()
                    if c is None:
                        break
                    out = env.execute(c)
                    eff += out.result.effort.cost()
                    steps.append((c.skill, c.target, out.result.status, round(out.metrics["progress"], 3),
                                  round(out.result.duration_s, 2)))
                    if out.done:
                        break
                m = out.metrics if out else {}
                row.update(success=bool(out and out.success), calls=len(steps), sim_t=round(env.t, 1),
                           wall=round(time.time() - t0, 1), progress=m.get("progress"),
                           violations=m.get("integrity_violations"), collisions=m.get("collisions"),
                           drops=m.get("drops"), effort_cost=round(eff, 2))
                if verbose:
                    row["steps"] = steps
            except Exception as ex:
                row.update(error=f"{type(ex).__name__}: {ex}"[:300], wall=round(time.time() - t0, 1))
                if verbose:
                    traceback.print_exc()
            print(json.dumps(row, default=str), flush=True)
