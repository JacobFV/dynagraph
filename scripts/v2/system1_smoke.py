"""smoke test of the System I head on a live r6 episode (bin_clearing, ur5e): propose params for the scripted pick."""
import numpy as np
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2.env_adapter import EnvAdapter
from physmem.models.v2.system1 import System1Head
head = System1Head.load("results/v2/checkpoints/v2_robots_r7_typed_s{}.pt")
env = make_env("bin_clearing"); ad = EnvAdapter(env)
ad.reset(api.TaskSpec("bin_clearing", "", 2_950_001, "ur5e_2f85", "val", params={"dressing": "site", "n_parts": 3}))
hist = []
for k in range(4):
    req = env.scripted()
    if req is None: break
    if req.skill == "pick":
        out = head.propose(req, hist, ad.frame(req.target), ad.embodiment_spec(), ad.target_pos(req.target),
                           grids=[{"grip_force": [5, 10, 20, 40], "lift_speed": [0.05, 0.1, 0.2]}])
        c = out["call"]
        print(f"step {k}: request grip {req.params.get('grip_force')} -> {c.params.get('grip_force'):.1f} lift "
              f"{c.params.get('lift_speed')} p_success {out['predicted']['p_success']:.2f} sd {out['uncertainty']['score_sd']:.2f} "
              f"latency {out['latency_ms']:.0f} ms; crush {out['memory']['crush']['mean']:.1f}N [{out['memory']['crush']['lo']:.1f},{out['memory']['crush']['hi']:.1f}]"
              f" mass {out['memory']['mass']['mean']:.2f}kg")
        req = c
    o = ad.execute(req, relative=False); hist.append(o.result.log["interaction"])
    print(f"   executed {req.skill}: {o.result.status}")
