"""r7 dev: pouch_transfer grip grid on one robot (which pinch holds a soft pouch without bursting it?).
    python scripts/envs_pouch_grid.py <robot> <n_seeds> [procs]"""
import itertools, json, sys, time, collections
from multiprocessing import Pool
import numpy as np


def job(a):
    rob, seed, fac, depth, lift = a
    import warnings; warnings.filterwarnings("ignore")
    from physmem.envs import make_env
    from physmem.envs.splits import split_params
    from physmem.v2.api import TaskSpec, SkillCall
    env = make_env("pouch_transfer")
    env.reset(TaskSpec("pouch_transfer", "deformable", seed, rob, "test", dict(split_params("pouch_transfer", "test"), dressing="site")))
    h = env.hidden_task
    st = []
    for _ in range(6):
        if env.held is None:
            if env.success():
                break
            c = SkillCall("pick", dict(grip_force=float(fac * h["crush_n"]), grip_max=float(0.9 * h["crush_n"]), lift_speed=lift,
                                       depth=depth, slip_gain=1.2), target="pouch")
        else:
            c = SkillCall("place", dict(release_h=0.02, speed=0.2 if lift < 0.1 else 0.35), target="tote")
        o = env.execute(c)
        st.append(o.result.status)
        if o.done:
            break
    return dict(rob=rob, seed=seed, fac=fac, depth=depth, lift=lift, success=bool(env.solved()), damaged=len(env.damaged),
                status=st, young=h["young"], crush=h["crush_n"], mass=h["mass"])


if __name__ == "__main__":
    rob, n = sys.argv[1], int(sys.argv[2])
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    from physmem.envs.splits import make_tasks
    seeds = [t.seed for t in make_tasks("pouch_transfer", "test", n, rob, 1_700_000)]
    jobs = [(rob, s, f, d, l) for s in seeds for f, d, l in itertools.product((0.5, 0.8), (0.0, 0.9), (0.05, 0.2))]
    with Pool(procs) as p:
        rows = p.map(job, jobs, chunksize=1)
    json.dump(rows, open(f"results/v2/envs/pouch_grid_{rob}.json", "w"), indent=1)
    G = collections.defaultdict(list)
    for r in rows:
        G[(r["fac"], r["depth"], r["lift"])].append(r)
    for k, v in sorted(G.items()):
        print(k, "succ", np.mean([r["success"] for r in v]), "dmg", np.mean([r["damaged"] > 0 for r in v]),
              collections.Counter(s for r in v for s in r["status"]).most_common(3))
