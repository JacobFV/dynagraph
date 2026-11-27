"""MODEL-V2 round 6 (r9) closed-loop evaluation of SKILL-LEVEL policies on envs-r7 (asserted).
usage: r9_eval.py RUN FAMILY ROBOT SPLIT N [BASE_SEED]
env: POLICIES (comma list of: heuristic, heuristic_probe, force_ramp, oracle(labelled), typed, generic, nomem,
     typed_full, generic_full ...), TAG (checkpoint tag, e.g. r9 -> v2_robots_r9_<variant>_s{0,1,2}), TRUST (json),
     OUT_DIR. Tasks = physmem.envs.splits.make_tasks(family, split, N, robot, base_seed) (paired across policies).
Per episode: success, cycle_t (sim time from reset to success), success within env.time_budget() (primary for every
policy, the same cap the imitation policy gets), damage, effort, steps. Resumable (partial json every 5 tasks)."""
import json, os, sys, time
import numpy as np
from physmem.envs import make_env
from physmem.envs.splits import make_tasks
from physmem.models.v2 import harness as H
from physmem.models.v2.env_adapter import EnvAdapter
from physmem.models.v2.imitation.data import env_version
from physmem.planner import CEMPlanner, PlannerConfig
from physmem.v2_train import load_model

ENV_V = env_version()
run, fam, robot, split, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
base = int(sys.argv[6]) if len(sys.argv) > 6 else 3_000_000
pols_req = os.environ.get("POLICIES", "heuristic,heuristic_probe,typed,generic,oracle(labelled)").split(",")
tag = os.environ.get("TAG", "r9")
out_dir = os.environ.get("OUT_DIR", "results/v2/closed_loop")
os.makedirs(out_dir, exist_ok=True)
C = "results/v2/checkpoints"
cfg = PlannerConfig(pop=96, iters=3, n_z=16, voi=False, c_prog=2.0, lam_effort=0.03)
trust = json.load(open(os.environ["TRUST"]))["families"] if os.environ.get("TRUST") else None
tr = {f: v["trusted"] for f, v in trust.items()} if trust else None
ANCHOR = {"tight_insert": "force_ramp", "fixture_place": None, "*": "probe"}   # best scripted per family on VAL (13a)


def residual(variant):
    v, suffix = (variant.split("_", 1) + [""])[:2]
    ck = f"{C}/v2_robots_{tag}{('_' + suffix) if suffix else ''}_{v}_s{{}}.pt"
    pls = [CEMPlanner(load_model(ck.format(s))[0], v if v != "oracle" else "oracle", cfg) for s in (0, 1, 2)]
    tf = f"results/v2/trust_r9_{suffix}_{v}.json"          # per (regime, variant) VAL trust gate (13a)
    trv = {f: x["trusted"] for f, x in json.load(open(tf))["families"].items()} if os.path.exists(tf) else tr
    p = H.ResidualPlanner(pls, trust=trv)
    p.anchor = ANCHOR
    return p


pols = {}
for k in pols_req:
    if k in ("heuristic", "heuristic_probe", "force_ramp"):
        pols[k] = (k, None)
    elif k == "oracle(labelled)":
        pols[k] = ("demo", None)           # oracle-parameter scripted policy (= the demonstrator; labelled)
    else:
        pols[k] = ("planner", residual(k))
tasks = make_tasks(fam, split, n, robot, base_seed=base)
pp = f"{out_dir}/{run}.partial.json"
res, start = {k: [] for k in pols}, 0
if os.path.exists(pp):
    old = json.load(open(pp))
    if set(old["episodes"]) == set(pols):
        res, start = old["episodes"], int(old["done_tasks"])
t0 = time.time()
for i in range(start, n):
    task = tasks[i]
    for k, (pol, pl) in pols.items():
        ad = EnvAdapter(make_env(fam))
        e = H.run_episode(ad, task, pol, pl, rng_seed=i)
        env = ad.env
        cap = float(env.time_budget()) if hasattr(env, "time_budget") else float("inf")
        t_end = float(e.get("sim_t", env.t))
        ok = bool(e["success"])
        res[k].append(dict(seed=int(task.seed), success=ok, cycle_t=t_end if ok else None, time_budget=cap,
                           success_in_budget=bool(ok and t_end <= cap), damage=int(e.get("damage", 0)),
                           effort=float(e["effort"]), steps=int(e["steps"]), probes=int(e.get("probes", 0)),
                           timeout=int(e.get("timeout", 0)), unstable=bool(getattr(env, "unstable_reasons", []))))
    if (i + 1) % 5 == 0 or i + 1 == n:
        json.dump(dict(episodes=res, done_tasks=i + 1, env_version=ENV_V, family=fam, robot=robot, split=split,
                       base_seed=base), open(pp, "w"))
        print(f"{run} {i + 1}/{n} " + " ".join(f"{k}={np.mean([r['success_in_budget'] for r in v]):.2f}"
                                               for k, v in res.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
json.dump(dict(episodes=res, env_version=ENV_V, family=fam, robot=robot, split=split, base_seed=base, n=n),
          open(f"{out_dir}/{run}.json", "w"))
os.remove(pp)
print("R9_EVAL_DONE", run, flush=True)
