"""Smoke-run scripted episodes per family: python scripts/envs_smoke.py fam [seeds] [robot]"""
import sys, time, json
import numpy as np
from physmem.envs import make_env, family_class, FAMILIES
from physmem.v2.api import TaskSpec

fams = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else list(FAMILIES)
seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
robot = sys.argv[3] if len(sys.argv) > 3 else "probe"
render = "--render" in sys.argv
for fam in fams:
    for seed in seeds:
        env = make_env(fam)
        t0 = time.time()
        ev = env.reset(TaskSpec(fam, env.category, seed, robot, "train"))
        tr = time.time() - t0
        if render:
            import imageio.v2 as iio, os
            os.makedirs("results/v2/envs", exist_ok=True)
            iio.imwrite(f"results/v2/envs/{fam}_s{seed}_reset.png", env.render(width=960, height=720))
        steps = []
        out = None
        while True:
            c = env.scripted()
            if c is None:
                break
            out = env.execute(c)
            steps.append((c.skill, c.target, out.result.status, round(out.metrics["progress"], 3), round(out.result.duration_s, 2)))
            if out.done:
                break
        wall = time.time() - t0
        m = out.metrics if out else {}
        print(json.dumps(dict(fam=fam, seed=seed, success=bool(out and out.success), calls=len(steps), sim_t=round(env.t, 1),
                              wall=round(wall, 1), reset_wall=round(tr, 2), n_ev=len(env.events_all),
                              progress=m.get("progress"), collisions=m.get("collisions"), viol=m.get("integrity_violations"),
                              drops=m.get("drops"), maxF=round(m.get("max_robot_contact_n", 0), 1))))
        if "-v" in sys.argv:
            for s in steps:
                print("   ", s)
        if render:
            iio.imwrite(f"results/v2/envs/{fam}_s{seed}_end.png", env.render(width=960, height=720))
