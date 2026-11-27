"""executor fidelity check: replay a recorded demo's actions open-loop on the same seed (il-v0).
usage: il_replay.py NPZ [k demos]"""
import json, sys
import numpy as np
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2.env_adapter import EnvAdapter
from physmem.models.v2.imitation import data as D
D.env_version()
z = np.load(sys.argv[1], allow_pickle=True)["eps"]
meta = json.load(open(sys.argv[1][:-4] + ".json"))
fam, robot = sys.argv[1].split("/")[-1][:-4].split("__")
for e, m in list(zip(z, meta))[: int(sys.argv[2]) if len(sys.argv) > 2 else 3]:
    if not m["success"]:
        continue
    env = make_env(fam); ad = EnvAdapter(env)
    ad.reset(api.TaskSpec(fam, "", m["seed"], robot, "train", params=m.get("params", {"dressing": "site"})))
    ex = D.Executor(env)
    for a in e["act"]:
        ex.apply(a)
    env.run(0.5)
    print(f"seed {m['seed']} demo ok, replay solved={env.solved()} sim_t={env.t:.1f} (demo {m['sim_t']:.1f})", flush=True)
