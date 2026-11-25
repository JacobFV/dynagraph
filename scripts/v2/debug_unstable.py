import numpy as np
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2.env_adapter import EnvAdapter
from physmem.models.v2.lot import lot_part_ids
def trial(name, ov_fn, reemit):
    env = make_env("bin_clearing"); ad = EnvAdapter(env)
    ad.reset(api.TaskSpec("bin_clearing", "", 2_700_000, "ur5e_2f85", "test", params={"dressing": "site", "n_parts": 3}))
    ids = lot_part_ids(env)
    orig = {e: dict(env.scene.entities[e].hidden) for e in ids}
    env.set_hidden({e: ov_fn(orig[e]) for e in ids})
    if reemit:
        env.sensors.initial(env.t); ev = env.sensors.flush(); env.view.add(ev); ad.idx.add(ev)
    out = ad.execute(env.scripted(), relative=False)
    print(f"{name:28s} status={out.result.status} reasons={getattr(env, 'unstable_reasons', None)} orig_mass={[round(orig[e]['mass'],3) for e in ids]}")
trial("no override", lambda h: {}, False)
trial("reemit only", lambda h: {}, True)
trial("mass x2", lambda h: {"mass": h["mass"] * 2}, False)
trial("mass 0.24", lambda h: {"mass": 0.24}, False)
trial("mu_grip 0.54", lambda h: {"mu_grip": 0.54}, False)
trial("crush 27", lambda h: {"crush_n": 27.1}, False)
