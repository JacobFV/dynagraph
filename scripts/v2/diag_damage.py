"""where does damage happen? per step: skill, grip_force, damage increment, crush of target (label, evaluator-only print),
for the plain heuristic and the gated residual (with its deviation decision + predicted damage of scripted vs chosen)."""
import json, sys, numpy as np, torch
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2 import harness as H, spec as S
from physmem.models.v2.env_adapter import EnvAdapter, rel_params
from physmem.planner import CEMPlanner, PlannerConfig
from physmem.v2_train import load_model
fam, robot, tag, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
C = "results/v2/checkpoints"; cfg = PlannerConfig(pop=96, iters=3, n_z=16, voi=False, c_prog=2.0, lam_effort=0.03)
pls = [CEMPlanner(load_model(f"{C}/v2_robots_{tag}_typed_s{s}.pt", torch.device("cpu"))[0], "typed", cfg) for s in (0, 1, 2)]
tr = {f: v["trusted"] for f, v in json.load(open(f"results/v2/trust_{tag}.json"))["families"].items()}
rp = H.ResidualPlanner(pls, trust=tr)
for i in range(n):
    for pol in ("heuristic", "residual"):
        env = make_env(fam); ad = EnvAdapter(env)
        ad.reset(api.TaskSpec(fam, "", 2_900_000 + i, robot, "test", params={"dressing": "site"}))
        emb = S.embodiment_vector(ad.embodiment_spec()); rng = np.random.default_rng(i); hist = []; prev = 0
        print(f"=== task {i} {pol}")
        for k in range(12):
            if pol == "heuristic":
                c, info = env.scripted(), {}
            else:
                c, info = rp.decide(ad, hist, emb, rng)
            if c is None: break
            hid = env.labels()["hidden"].get(c.target, {})
            out = ad.execute(c, relative=False); hist.append(out.result.log["interaction"])
            dm = float(out.metrics.get("damaged", 0) or 0)
            print(f"  k={k} {c.skill:6s} {str(c.target):10s} grip={c.params.get('grip_force', '-')!s:6.6} crush={hid.get('crush_n', '-')!s:6.6} "
                  f"dev={info.get('deviated', '-')} lcb={info.get('lcb', 0):+.3f} dmg_inc={dm - prev:.0f} status={out.result.status} succ={out.success}")
            prev = dm
            if out.done: break
