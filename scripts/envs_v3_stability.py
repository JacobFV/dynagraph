"""envs-v3 SIM-STABILITY SCAN: per family x robot, reset N seeds over the full hidden ranges (site-dressed), idle 2 s,
then run the scripted policy's first 2 calls; record MuJoCo warnings (badqacc), the blow-up monitor (non-finite /
ejection), reset re-draws, integrity violations, max free-body speed and wall time.

    python scripts/envs_v3_stability.py <families> <robots> <n_seeds> [out_tag]
Rows -> results/v2/envs/v3_stability_<tag>.jsonl, summary printed (and .json).
"""
import json
import sys
import time

import numpy as np


def scan(fam, robot, seed):
    import mujoco
    from physmem.envs import family_class, make_env
    from physmem.v2.api import TaskSpec
    t0 = time.time()
    env = make_env(fam)
    rec = dict(family=fam, robot=robot, seed=seed)
    try:
        env.reset(TaskSpec(fam, family_class(fam).category, seed, robot, "test", dict(dressing="site")))
        rec["reset_attempts"] = env.reset_attempts
        env.idle(2.0)
        vmax = 0.0
        for _ in range(2):
            c = env.scripted()
            if c is None:
                break
            out = env.execute(c)
            if out.done:
                break
        free = [env.m.jnt_dofadr[j] for j in range(env.m.njnt) if env.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if free:
            v = np.array([env.d.qvel[a:a + 3] for a in free])
            vmax = float(np.max(np.linalg.norm(v, axis=1)))
        rec.update(unstable=bool(env.unstable), reasons=list(env.unstable_reasons),
                   badqacc=int(env.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number),
                   integrity_violations=int(env.metrics_acc["integrity_violations"]),
                   sim_instabilities=int(env.metrics_acc["sim_instabilities"]), vmax_end=vmax, sim_t=float(env.t))
    except Exception as e:
        rec["error"] = repr(e)[:300]
    rec["wall_s"] = time.time() - t0
    return rec


if __name__ == "__main__":
    fams, robots, n = sys.argv[1].split(","), sys.argv[2].split(","), int(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else "v3"
    path = f"results/v2/envs/v3_stability_{tag}.jsonl"
    rows = []
    with open(path, "a") as f:
        for fam in fams:
            for rob in robots:
                for s in range(3_000_000, 3_000_000 + n):
                    r = scan(fam, rob, s)
                    rows.append(r)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
    summ = {}
    for fam in fams:
        for rob in robots:
            R = [r for r in rows if r["family"] == fam and r["robot"] == rob]
            ok = [r for r in R if "error" not in r]
            summ[f"{fam}/{rob}"] = dict(n=len(R), errors=len(R) - len(ok),
                                        unstable=sum(r["unstable"] for r in ok), badqacc=sum(r["badqacc"] > 0 for r in ok),
                                        redraws=sum(r["reset_attempts"] > 1 for r in ok),
                                        integrity=sum(r["integrity_violations"] for r in ok),
                                        vmax_end=max([r["vmax_end"] for r in ok] or [0.0]),
                                        wall_s=float(np.mean([r["wall_s"] for r in ok])) if ok else None)
    json.dump(summ, open(f"results/v2/envs/v3_stability_{tag}.json", "w"), indent=1)
    for k, v in summ.items():
        print(k, v)
