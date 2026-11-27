"""il-v0 demo recording (scripted policy, train split). usage: il_record.py FAMILY ROBOT N OUT_DIR [SEED_BASE]
Writes OUT_DIR/FAMILY__ROBOT.npz (padded episodes) + .json meta; resumable via a partial npz every 10 demos.
Only successful demos are kept for training (all are logged in meta)."""
import json, os, sys, time
import numpy as np
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2 import spec as S
from physmem.models.v2.env_adapter import EnvAdapter
from physmem.models.v2.imitation import data as D
ENV_V = D.env_version()
from physmem.models.v2 import harness as H
from physmem.models.v2.env_adapter import robot_spec
from physmem.models.v2.tensorize import build_family_arrays

fam, robot, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
base = int(sys.argv[5]) if len(sys.argv) > 5 else 3_000_000
os.makedirs(out, exist_ok=True)
stem = f"{out}/{fam}__{robot}"
eps, meta, mem_recs = [], [], []
POLICY = os.environ.get("DEMO_POLICY", "heuristic")      # envs-r7: "demo" (env.demo_call = oracle-param scripted)
IMG = int(os.environ.get("IL_IMG", "0"))                  # wrist camera side (envs-r7: 64), 0 = off
TASKS = os.environ.get("TASKS", "")                       # "make_tasks": seeds from physmem.envs.splits.make_tasks


class RecAdapter(EnvAdapter):
    """EnvAdapter that attaches the il-v0 stream peek + recorder right after every reset (the harness resets)."""

    def reset(self, task):
        ev = super().reset(task)
        import copy
        self.peek = D.StreamPeek(self.env)
        self.peek.idx = copy.deepcopy(self.idx)       # reset-time events (sizes, vision material, first track)
        self.rec = D.Recorder(self.env, self.peek, img=IMG)
        return ev
if os.path.exists(stem + ".partial.json"):
    meta = json.load(open(stem + ".partial.json"))
    z = np.load(stem + ".partial.npz", allow_pickle=True)
    eps = list(z["eps"])
    if os.path.exists(stem + ".partial.mem.pkl"):          # memory-model view of the already-recorded episodes
        import pickle
        mem_recs = pickle.load(open(stem + ".partial.mem.pkl", "rb"))
t0 = time.time()
if TASKS == "make_tasks":
    from physmem.envs.splits import make_tasks
    TL = make_tasks(fam, os.environ.get("SPLIT", "train"), n, robot, base_seed=base)
else:
    TL = [api.TaskSpec(fam, "", base + i, robot, os.environ.get("SPLIT", "train"), params={"dressing": "site"})
          for i in range(n)]
for i in range(len(meta), n):
    task = TL[i]
    seed = int(task.seed)
    env = make_env(fam)
    ad = RecAdapter(env)
    ok, err, recd = False, None, []
    try:       # scripted policy through the harness, so the SAME episodes are also logged as memory-model
        H.run_episode(ad, task, POLICY, None, rng_seed=seed, record=recd)    # interactions (data-budget match)
        ok = bool(env.solved()) and not env.unstable_reasons
    except Exception as e:      # recorded as failure
        err = repr(e)[:200]
    if recd:
        mem_recs.append((fam, robot, recd))
    arr = ad.rec.arrays() if getattr(ad, "rec", None) else None
    emb = S.embodiment_vector(ad.embodiment_spec()).astype(np.float32)
    meta.append(dict(env_version=ENV_V, policy=POLICY, params=dict(task.params), seed=seed, success=ok, T=0 if arr is None else len(arr["act"]), sim_t=float(env.t),
                     damage=len(getattr(env, "damaged", ())), err=err))
    eps.append(dict(arr or {}, emb=emb, success=ok))
    if (i + 1) % 10 == 0 or i + 1 == n:
        np.savez_compressed(stem + ".partial.npz", eps=np.array(eps, dtype=object))
        json.dump(meta, open(stem + ".partial.json", "w"))
        import pickle
        pickle.dump(mem_recs, open(stem + ".partial.mem.pkl", "wb"))
        print(f"{fam} {robot} {i + 1}/{n} ok={sum(m['success'] for m in meta)} {time.time() - t0:.0f}s", flush=True)
if mem_recs:     # memory-model view of the same episodes (only those recorded in this process; resume-safe enough
    os.makedirs(f"{out}/mem", exist_ok=True)       # for the budget control: the trainer reads both views by seed)
    arr = build_family_arrays(mem_recs, lambda r: S.embodiment_vector(robot_spec(robot)), np.random.default_rng(0))
    np.savez(f"{out}/mem/{robot}__{fam}.npz", **arr)
os.replace(stem + ".partial.npz", stem + ".npz")
os.replace(stem + ".partial.json", stem + ".json")
if os.path.exists(stem + ".partial.mem.pkl"):
    os.remove(stem + ".partial.mem.pkl")
print("RECORD_DONE", fam, robot)
