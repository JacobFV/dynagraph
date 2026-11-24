import numpy as np
from physmem.envs import make_env
from physmem.v2.api import TaskSpec
from physmem.models.v2 import env_adapter as A, spec as S
for fam in ("conveyor_sort", "bin_clearing"):
    env = make_env(fam); f, r, steps = A.collect_episode(env, TaskSpec(fam, "x", 3, "ur5e_2f85", "train", params={"dressing": "site"}), max_steps=4)
    for st in steps:
        fr, it = st["frame"], st["inter"]
        print(fam, it.skill, "target stiff [vision,tactile]", np.round(fr.stiff[0], 0), "labels", {k: round(v, 2) for k, v in it.labels.items()},
              "dmg_event", it.outcome_events[6], "tok", np.round(fr.tokens()[0, -3:], 2))
