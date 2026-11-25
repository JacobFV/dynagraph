import sys, time, numpy as np
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2.lot import LotEnvAdapter
fam, robot, npart = sys.argv[1], sys.argv[2], int(sys.argv[3]); SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 2_700_000
for pol in ("heuristic", "heuristic_probe"):
    t0 = time.time(); ad = LotEnvAdapter(make_env(fam))
    ad.reset(api.TaskSpec(fam, "", SEED, robot, "test", params={"dressing": "site", "n_parts": npart}))
    print(pol, "reset", round(time.time() - t0, 1), "s lot", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in ad.lot.items()}, "budget", ad.env.budget, flush=True)
    for k in range(40):
        c = ad.env.scripted()
        if c is not None and pol == "heuristic_probe": c = ad.env.probe_then_act(c)
        if c is None: print("  scripted None"); break
        t1 = time.time(); out = ad.execute(c, relative=False)
        print(f"  k={k} {c.skill:6s} {str(c.target):8s} grip={c.params.get('grip_force','-')} lift={c.params.get('lift_speed','-')} status={out.result.status} wall={time.time()-t1:.1f}s sim_t={out.metrics.get('t',0):.1f} dmg={out.metrics.get('damaged')} prog={out.metrics.get('progress',0):.2f}", flush=True)
        if out.done: break
    print(pol, "success", out.success, "total wall", round(time.time() - t0, 1), flush=True)
