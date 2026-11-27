"""record/eval observation consistency: replay a demo's actions through the Executor on the same seed and compare
the live il observation (data.observe) with the recorded one at every tick. usage: il_obs_check.py NPZ [k]"""
import copy, json, sys
import numpy as np
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2.env_adapter import EnvAdapter
from physmem.models.v2.imitation import data as D
D.env_version()
npz = sys.argv[1]; j = int(sys.argv[2]) if len(sys.argv) > 2 else 0
z = np.load(npz, allow_pickle=True)["eps"]; mm = json.load(open(npz[:-4] + ".json"))
fam, robot = npz.split("/")[-1][:-4].split("__")
i = [k for k, m in enumerate(mm) if m["success"]][j]; e, m = z[i], mm[i]
env = make_env(fam); ad = EnvAdapter(env)
ad.reset(api.TaskSpec(fam, "", m["seed"], robot, "train", params=m.get("params", {})))
peek = D.StreamPeek(env); peek.idx = copy.deepcopy(ad.idx); orig = env.post_physics
env.post_physics = lambda stop=None: (lambda r: (peek.poll(), r)[1])(orig(stop))
ex, t0 = D.Executor(env), float(env.t)
names = ["p", "obj", "reg"]
for k in range(len(e["act"])):
    o = D.observe(env, peek, t0)
    if k % 20 == 0 or k < 3:
        d = {n: float(np.abs(o[n] - e[n][k]).max()) for n in names}
        dp = np.abs(o["p"] - e["p"][k]); worst = int(np.argmax(dp))
        print(f"t={k/10:5.1f} maxdiff {d}  worst p dim {worst} live {o['p'][worst]:.3f} rec {e['p'][k][worst]:.3f}")
    ex.apply(e["act"][k])
print("solved", env.solved())
