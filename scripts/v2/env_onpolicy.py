"""MODEL-V2 r5: on-policy rollouts on REAL stations (one robot x family per process).
The gated residual planner (3-seed ensemble) acts on TRAIN-range seeds with exploration (random feasible candidate
with prob EXPLORE); every executed interaction (failures included) is recorded with evaluator labels attached
for TRAINING ONLY. Writes <out>/<robot>__<family>.npz (build_family_arrays) + a json with rollout success / damage.
usage: env_onpolicy.py <round> <robot> <family> <n_episodes> <out_dir> <tag> <trust.json>"""
import json, os, sys, time
import numpy as np
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2 import harness as H, spec as S
from physmem.models.v2.env_adapter import EnvAdapter, robot_spec
from physmem.models.v2.tensorize import build_family_arrays
from physmem.planner import CEMPlanner, PlannerConfig
from physmem.v2_train import load_model
rnd, robot, fam, n, out, tag, trust = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
os.makedirs(out, exist_ok=True)
cfg = PlannerConfig(pop=96, iters=3, n_z=16, voi=False, c_prog=2.0, lam_effort=0.03)
C = "results/v2/checkpoints"
pls = [CEMPlanner(load_model(f"{C}/v2_robots_{tag}_typed_s{s}.pt")[0], "typed", cfg) for s in (0, 1, 2)]
tr = {f: v["trusted"] for f, v in json.load(open(trust))["families"].items()}
planner = H.ResidualPlanner(pls, trust=tr)
explore = float(os.environ.get("EXPLORE", "0.2"))
eps, recs, t0 = [], [], time.time()
for i in range(n):
    seed = 8_000_000 + rnd * 100_000 + i           # train-range seeds, disjoint from val/test/heldout and across rounds
    task = api.TaskSpec(fam, "", seed, robot, "train", params={"dressing": "site"})
    recd = []
    try:
        e = H.run_episode(EnvAdapter(make_env(fam)), task, "planner", planner, rng_seed=seed, record=recd, explore=explore)
    except Exception as ex:
        print(f"rollout fail {robot} {fam} {seed}: {type(ex).__name__}: {ex}", flush=True)
        continue
    eps.append(dict(success=bool(e["success"]), damage=int(e.get("damage", 0)), steps=e["steps"], effort=e["effort"]))
    if recd:
        recs.append((fam, robot, recd))
emb = S.embodiment_vector(robot_spec(robot))
if recs:
    arr = build_family_arrays(recs, lambda r: emb, np.random.default_rng(rnd))
    np.savez(os.path.join(out, f"{robot}__{fam}.npz"), **arr)
summ = dict(round=rnd, robot=robot, family=fam, n=len(eps), success=float(np.mean([e["success"] for e in eps])) if eps else None,
            damage=float(np.mean([e["damage"] for e in eps])) if eps else None, interactions=int(sum(len(r[2]) for r in recs)),
            seconds=time.time() - t0)
json.dump(dict(summary=summ, episodes=eps), open(os.path.join(out, f"{robot}__{fam}.json"), "w"))
print(json.dumps(summ), flush=True)
