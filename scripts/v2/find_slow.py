import time, sys
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2.lot import LotEnvAdapter
for i in range(int(sys.argv[1]), int(sys.argv[2])):
    for pol in ("heuristic", "heuristic_probe"):
        task = api.TaskSpec("bin_clearing", "", 2_700_000 + i, "ur5e_2f85", "test", params={"dressing": "site", "n_parts": 3})
        t0 = time.time(); ad = LotEnvAdapter(make_env("bin_clearing")); ad.reset(task); calls = []
        for k in range(ad.env.budget):
            c = ad.env.scripted()
            if c is not None and pol == "heuristic_probe": c = ad.env.probe_then_act(c)
            if c is None: break
            t1 = time.time(); out = ad.execute(c, relative=False); w = time.time() - t1
            calls.append((c.skill, out.result.status, round(w, 1), round(out.result.duration_s, 1)))
            if out.done: break
        print(f"task {i} {pol:15s} wall {time.time()-t0:6.1f}s calls {len(calls)} slow={[c for c in calls if c[2] > 20]}", flush=True)
