"""MODEL-V2: collect one ENVS episode per family through the adapter, check segments/labels/timing."""
import sys, time, numpy as np
from physmem.envs import make_env, FAMILIES
from physmem.v2.api import TaskSpec
from physmem.models.v2 import env_adapter as A, spec as S
from physmem.models.v2.tensorize import build_family_arrays
for fam in (sys.argv[1].split(",") if len(sys.argv) > 1 else FAMILIES):
    env = make_env(fam)
    t = time.time()
    f, r, steps = A.collect_episode(env, TaskSpec(fam, "x", seed=2, robot="probe", split="train"), max_steps=3)
    dt = time.time() - t
    for st in steps:
        it = st["inter"]
        x = it.segment
        print(fam, it.skill, "seg", x.shape, "dur", round(float(x[-1, -1]), 2), "|F|max", round(float(np.abs(x[:, 12:15]).max()), 2),
              "obj_valid", int(x[:, 23].sum()), "contact", int(x[:, 24].sum()), "dobj", np.round(x[-1, 20:23] - x[0, 20:23], 3),
              "labels", {k: round(v, 3) for k, v in it.labels.items()}, "cont", np.round(it.outcome_cont, 3), "effort", round(it.effort_cost, 3),
              "nent", len(st["frame"].types))
    arr = build_family_arrays([(f, r, steps)], lambda rr: S.embodiment_vector(env.robot.spec))
    print(fam, "arrays ok", arr["seg"].shape, arr["ent"].shape, f"{dt:.1f}s wall for {len(steps)} steps")
