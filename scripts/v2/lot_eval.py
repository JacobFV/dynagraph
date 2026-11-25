"""MODEL-V2 round 4: paired LOT-protocol closed loop on real-robot stations (site-dressed, envs-r6).
usage: TAG=r7 TAU=0.15 ROBOT=ur5e_2f85 SPLIT=test python scripts/v2/lot_eval.py <run> <family> <n_tasks>
policies (same seeded tasks / lots): heuristic (never-probe plain scripted), heuristic_probe (always-probe PTA),
typed (memory-conditioned probing, 3-seed typed ensemble x lot posterior), typed_nolot (lot fusion off), generic
(3-seed generic ensemble), oracle (typed s0 head + TRUE lot latents, labelled). Partial json every 5 tasks; resumes."""
import json, os, sys, time
import numpy as np
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2 import harness as H
from physmem.models.v2.lot import LotEnvAdapter, LotPlanner
from physmem.planner import CEMPlanner, PlannerConfig
from physmem.v2_train import load_model, write_rows, _utc, SEED_BASE

run, fam, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
tag, tau = os.environ.get("TAG", "r7"), float(os.environ.get("TAU", "0.15"))
robot, split = os.environ.get("ROBOT", "ur5e_2f85"), os.environ.get("SPLIT", "test")
C = "results/v2/checkpoints"
cfg = PlannerConfig(n_z=16, voi=False)
only = os.environ.get("POLICIES", "heuristic,heuristic_probe,typed,typed_nolot,generic,oracle(labelled)").split(",")
pols = {"heuristic": ("heuristic", None), "heuristic_probe": ("heuristic_probe", None)}
if any(k in only for k in ("typed", "typed_nolot", "oracle(labelled)")):
    T = [CEMPlanner(load_model(f"{C}/v2_robots_{tag}_typed_s{s}.pt")[0], "typed", cfg) for s in (0, 1, 2)]
    pols["typed"] = ("planner", LotPlanner(T, tau=tau, name="typed"))
    pols["typed_nolot"] = ("planner", LotPlanner(T, tau=tau, nolot=True, name="typed_nolot"))
    pols["oracle(labelled)"] = ("planner", LotPlanner([CEMPlanner(T[0].model, "oracle", cfg)], tau=tau, name="oracle"))
if any(k.endswith("_v2") for k in only):      # round-4 EXPLORATORY v2 (crush-gated probing, 11e)
    T = T if "T" in dir() else [CEMPlanner(load_model(f"{C}/v2_robots_{tag}_typed_s{s}.pt")[0], "typed", cfg) for s in (0, 1, 2)]
    pols["typed_v2"] = ("planner", LotPlanner(T, tau=tau, name="typed_v2", crush_gate=True))
    pols["typed_nolot_v2"] = ("planner", LotPlanner(T, tau=tau, nolot=True, name="typed_nolot_v2", crush_gate=True))
    pols["oracle_v2(labelled)"] = ("planner", LotPlanner([CEMPlanner(T[0].model, "oracle", cfg)], tau=tau,
                                                          name="oracle_v2", crush_gate=True))
if "generic" in only:
    Gm = [CEMPlanner(load_model(f"{C}/v2_robots_{tag}_generic_s{s}.pt")[0], "generic", cfg) for s in (0, 1, 2)]
    pols["generic"] = ("planner", LotPlanner(Gm, tau=tau, name="generic"))
pols = {k: v for k, v in pols.items() if k in only}
NP = int(os.environ.get("NPARTS", "3"))     # parts per lot (bin_clearing n_parts); budget = 4 x parts (ENVS)
SEED_OFF = int(os.environ.get("SEED_OFF", "700000"))   # 700000 = pre-registered set; 800000 = fresh set for v2 (11e)
tasks = [api.TaskSpec(fam, "", SEED_BASE.get(split, 2_000_000) + SEED_OFF + i, robot, split,
                      params={"dressing": "site", "n_parts": NP}) for i in range(n)]
out_dir = os.environ.get("OUT_DIR", "results/v2/closed_loop")
os.makedirs(out_dir, exist_ok=True)
pp = f"{out_dir}/{run}.partial.json"
res, start = {k: [] for k in pols}, 0
if os.path.exists(pp):
    old = json.load(open(pp))[fam]
    if set(old["episodes"]) == set(pols):
        res, start = old["episodes"], int(old["done_tasks"])
t0 = time.time()
for c0 in range(start, n, 5):
    part = H.paired_eval(lambda: LotEnvAdapter(make_env(fam)), tasks[c0:c0 + 5], pols, offset=c0)
    for k in res:
        res[k].extend(part[k])
    json.dump({fam: dict(episodes=res, done_tasks=len(res["heuristic"]) if "heuristic" in res else c0 + 5, tau=tau,
                         robot=robot, split=split)}, open(pp, "w"), default=float)
summ = H.summarize(res, ref="heuristic")
json.dump({fam: dict(summary=summ, episodes=res, tau=tau, robot=robot, split=split)}, open(f"{out_dir}/{run}.json", "w"),
          default=float)
print(f"== {fam} {robot} {split} ({time.time() - t0:.0f}s)")
for k, eps in res.items():
    s = np.mean([e["success"] for e in eps]); dm = np.mean([e.get("damage", 0) > 0 for e in eps])
    pr = np.mean([e.get("probes", 0) for e in eps]); st = np.mean([e.get("sim_t", 0) for e in eps])
    gf = np.nanmean([e.get("good_frac", np.nan) for e in eps])
    print(f"  {k:18s} good_parts={gf:.2f} succ={s:.2f} damage={dm:.2f} probes/ep={pr:.2f} sim_t={st:.0f}s effort={np.mean([e['effort'] for e in eps]):.0f}")
write_rows([dict(run_id=f"{run}", experiment="v2_r7_lot_closed_loop", model=k, stratum=f"{split}:{fam}:{robot}",
                 metric="success", value=float(np.mean([e["success"] for e in eps])), n_episodes=len(eps),
                 status="observed", time_utc=_utc(), tau=tau) for k, eps in res.items()])
