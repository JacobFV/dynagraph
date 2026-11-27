"""regenerate the memory-model view of a demo cell (all N make_tasks episodes, demo policy, no images); used where a
resume lost part of it. Deterministic: same tasks / seeds as il_record. usage: il_mem_fill.py FAM ROBOT SPLIT N OUT_MEM_DIR"""
import os, sys
import numpy as np
from physmem.envs import make_env
from physmem.envs.splits import make_tasks
from physmem.models.v2 import harness as H, spec as S
from physmem.models.v2.env_adapter import EnvAdapter, robot_spec
from physmem.models.v2.imitation.data import env_version
from physmem.models.v2.tensorize import build_family_arrays
env_version()
fam, robot, split, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
recs = []
for i, task in enumerate(make_tasks(fam, split, n, robot, base_seed=3_000_000)):
    recd = []
    try:
        H.run_episode(EnvAdapter(make_env(fam)), task, "demo", None, rng_seed=int(task.seed), record=recd)
    except Exception as e:
        print("fail", task.seed, repr(e)[:120], flush=True)
    if recd:
        recs.append((fam, robot, recd))
os.makedirs(out, exist_ok=True)
arr = build_family_arrays(recs, lambda r: S.embodiment_vector(robot_spec(robot)), np.random.default_rng(0))
np.savez(f"{out}/{robot}__{fam}.npz", **arr)
print("MEM_FILL_DONE", fam, robot, split, len(recs), "episodes", len(arr["k"]), "rows", flush=True)
