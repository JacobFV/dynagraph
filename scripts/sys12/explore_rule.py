"""EXPLORATORY (not pre-registered; lead request after the pre-registered closed loop, run on FRESH seeds +960000).
Question: is the pre-registered System I loss due to the Inkling model or to the System I decision rule?
Same MODEL-V2 decision rules, different models (scripted skill choice everywhere, no System II):
  bin lots : crush-gated LotPlanner (MODEL-V2's 'typed_v2' rule, tau 0.15)
  pouch    : ResidualPlanner anchored on probe-then-act, r6f trust gate (MODEL-V2's pouch rule)
policies:  pta | v2_ens3 (MODEL-V2 r7 typed 3-seed ensemble = the MODEL-V2-alone baseline) |
           v2_s0 (single r7 typed s0, same memory front end as System I) | native_v2rule (Inkling-2-Mini core
           (pretrained) gated System I, checkpoint s1r1_inkling_gated_s0, single model)
usage (frozen tree, after cl_guard GUARD_ONLY=1): CELL=bin_ur5e|pouch_ur5e N=30 python scripts/sys12/explore_rule.py"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
import physmem.envs as E  # noqa: E402

assert E.ENV_VERSION == "envs-r6", E.ENV_VERSION
from physmem.envs import make_env  # noqa: E402
from physmem.models.v2 import harness as H  # noqa: E402
from physmem.models.v2.env_adapter import EnvAdapter  # noqa: E402
from physmem.models.v2.lot import LotEnvAdapter, LotPlanner  # noqa: E402
from physmem.planner import CEMPlanner, PlannerConfig  # noqa: E402
from physmem.v2 import api  # noqa: E402
from physmem.v2_train import SEED_BASE, load_model  # noqa: E402

CELL = os.environ.get("CELL", "bin_ur5e")
N = int(os.environ.get("N", "30"))
SEED_OFF = int(os.environ.get("SEED_OFF", "960000"))
S1 = os.environ.get("S1_CKPT", "results/sys12/checkpoints/s1r1_inkling_gated_s0.pt")
fam, robot = {"bin_ur5e": ("bin_clearing", "ur5e_2f85"), "pouch_ur5e": ("pouch_transfer", "ur5e_2f85")}[CELL]
C = "results/v2/checkpoints"
torch.set_num_threads(2)
lot_cfg = PlannerConfig(n_z=16, voi=False)
res_cfg = PlannerConfig(pop=96, iters=3, n_z=16, voi=False, c_prog=2.0, lam_effort=0.03)
cfg = lot_cfg if fam == "bin_clearing" else res_cfg
T = [CEMPlanner(load_model(f"{C}/v2_robots_r7_typed_s{s}.pt", torch.device("cpu"))[0], "typed", cfg) for s in (0, 1, 2)]

from physmem.models.sys12 import sys1_planner  # noqa: E402
from physmem.sys12_train import build, load_state  # noqa: E402

ck = torch.load(S1, map_location="cpu", weights_only=False)
m, _ = build(ck["info"]["arm"], ck["info"]["passes"], ck["info"]["seed"])
load_state(m, ck["state"])
m.eval()
m.family = fam
nat = sys1_planner(m, cfg)
nat.skill = "pick"


def rule(planners, name):
    if fam == "bin_clearing":
        return LotPlanner(planners, tau=0.15, name=name, crush_gate=True)
    tr = {f: v["trusted"] for f, v in json.load(open("results/v2/trust_r6f.json"))["families"].items()}
    rp = H.ResidualPlanner(planners, trust=tr)
    rp.anchor_probe = True
    return rp


pols = {"pta": ("heuristic_probe", None), "v2_ens3": ("planner", rule(T, "v2_ens3")),
        "v2_s0": ("planner", rule([T[0]], "v2_s0")), "native_v2rule": ("planner", rule([nat], "native_v2rule"))}
params = {"dressing": "site"}
if fam == "bin_clearing":
    params["n_parts"] = 3
factory = (lambda: LotEnvAdapter(make_env(fam))) if fam == "bin_clearing" else (lambda: EnvAdapter(make_env(fam)))
tasks = [api.TaskSpec(fam, "", SEED_BASE["test"] + SEED_OFF + i, robot, "test", params=dict(params)) for i in range(N)]
out_dir = "results/sys12/closed_loop"
pp = f"{out_dir}/explore1_{CELL}.partial.json"
res, start = {k: [] for k in pols}, 0
if os.path.exists(pp):
    old = json.load(open(pp))
    res, start = old["episodes"], int(old["done_tasks"])
t0 = time.time()
for c0 in range(start, N, 5):
    part = H.paired_eval(factory, tasks[c0:c0 + 5], pols, offset=c0)
    for k in res:
        res[k].extend(part[k])
    json.dump(dict(episodes=res, done_tasks=c0 + 5, cell=CELL, family=fam, robot=robot, seed_off=SEED_OFF,
                   label="EXPLORATORY"), open(pp, "w"), default=float)
    key = "good_frac" if fam == "bin_clearing" else "success"
    print(f"[{CELL}] {c0 + 5}/{N} ({time.time() - t0:.0f}s) " + " ".join(
        f"{k}={np.nanmean([float(e.get(key, np.nan)) for e in v]):.2f}" for k, v in res.items()), flush=True)
json.dump(dict(episodes=res, cell=CELL, family=fam, robot=robot, seed_off=SEED_OFF, label="EXPLORATORY",
               s1_ckpt=S1, seconds=time.time() - t0), open(f"{out_dir}/explore1_{CELL}.json", "w"), default=float)
print("DONE", flush=True)
