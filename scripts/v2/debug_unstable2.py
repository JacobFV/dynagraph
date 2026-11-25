import time
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2.env_adapter import EnvAdapter
from physmem.models.v2.lot import lot_part_ids
bad = {}
for f in (0.5, 2.0, 2.5):
    t0 = time.time()
    for seed in range(2_700_000, 2_700_004):
        env = make_env("bin_clearing"); ad = EnvAdapter(env)
        ad.reset(api.TaskSpec("bin_clearing", "", seed, "ur5e_2f85", "test", params={"dressing": "site", "n_parts": 3}))
        ids = lot_part_ids(env)
        env.set_hidden({e: {"mass": env.scene.entities[e].hidden["mass"] * f} for e in ids})
        st = [ad.execute(env.scripted(), relative=False).result.status for _ in range(2)]
        bad[f] = bad.get(f, 0) + int("unstable" in st)
    print(f"factor {f}: unstable {bad[f]}/4, wall {time.time() - t0:.0f}s for 8 picks", flush=True)
