"""MODEL-V2: paired closed loop on ENVS stations (G3 on real MuJoCo stations).
usage: ROBOT=<name> SPLIT=<split> python scripts/v2/env_closed_loop.py <run> <families> <episodes> <typed.pt> [generic.pt] [nomem.pt]"""
import json, os, sys, time, numpy as np, torch
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2 import harness as H
from physmem.models.v2.env_adapter import EnvAdapter
from physmem.planner import CEMPlanner, PlannerConfig
from physmem.v2_train import load_model, write_rows, _utc, SEED_BASE
run, fams, n = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
cks = dict(zip(["typed", "generic", "nomem"], sys.argv[4:]))
cfg = PlannerConfig(pop=96, iters=3, n_z=16, voi=False, c_prog=2.0, lam_effort=0.03)
pols = {"heuristic": ("heuristic", None)}
for name, ck in cks.items():
    m, _ = load_model(ck)
    pols[f"planner_{name}"] = ("planner", CEMPlanner(m, name, cfg))
    if name == "typed":
        pols["planner_oracle(labelled)"] = ("planner", CEMPlanner(m, "oracle", cfg))
        if os.environ.get("NO_PRIOR", "0") != "1":
            pols["planner_prior_mean"] = ("planner", CEMPlanner(m, "prior", cfg))
RES = [c for c in os.environ.get("RESIDUAL", "").split(",") if c]     # ensemble checkpoints (training seeds)
if RES:
    pols["planner_residual"] = ("planner", H.ResidualPlanner([CEMPlanner(load_model(c)[0], "typed", cfg) for c in RES]))
for _, pl_ in pols.values():
    if pl_ is not None and not isinstance(pl_, H.ResidualPlanner):
        pl_.search = os.environ.get("SEARCH", "ranked")      # real stations: support-constrained ranking by default
        am = os.environ.get("ANCHOR", "0.15")                 # scripted-anchored margin ("none" = pure ranking)
        pl_.anchor_margin = None if am == "none" else float(am)
out, rows = {}, []
for fam in fams:
    t0 = time.time()
    robot, split = os.environ.get("ROBOT", "probe"), os.environ.get("SPLIT", "test")
    tasks = [api.TaskSpec(fam, "", SEED_BASE.get(split, 2_000_000) + 900_000 + i, robot, split) for i in range(n)]
    res = H.paired_eval(lambda: EnvAdapter(make_env(fam)), tasks, pols)
    summ = H.summarize(res, ref="heuristic")
    out[fam] = dict(summary=summ, episodes=res)
    print(f"== {fam} ({time.time() - t0:.0f}s)", flush=True)
    for name, sm in summ.items():
        dev = np.mean([e.get("deviations", 0) / max(e["steps"], 1) for e in res[name]])
        print(f"  {name:26s} succ={sm['success'][0]:.2f} [{sm['success'][1]:.2f},{sm['success'][2]:.2f}] effort={sm['effort'][0]:.2f} dev={dev:.2f} "
              f"risk={sm['risk_events'][0]:.2f} steps={sm['steps']:.1f}", flush=True)
        for k, v in sm.items():
            if isinstance(v, tuple):
                rows.append(dict(run_id=f"{run}-{fam}", experiment="v2_g3_env_closed_loop", model=name, stratum=f"{split}:{fam}:{robot}",
                                 metric=k, value=v[0], ci_lo=v[1], ci_hi=v[2], n_episodes=sm["n"], status="observed", time_utc=_utc()))
write_rows(rows)
os.makedirs("results/v2/closed_loop", exist_ok=True)
json.dump(out, open(f"results/v2/closed_loop/{run}.json", "w"), default=float)
