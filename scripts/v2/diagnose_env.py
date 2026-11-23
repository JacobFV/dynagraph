"""MODEL-V2 diagnosis: why does the planner lose to the scripted policy? Per decision: scripted vs chosen params,
predicted (p_success, progress, effort) for both, realised outcome of the chosen action.
usage: python scripts/v2/diagnose_env.py <family> <robot> <ckpt> <n_episodes>"""
import sys, numpy as np, torch
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2 import harness as H, spec as S
from physmem.models.v2.env_adapter import EnvAdapter, rel_params
from physmem.planner import CEMPlanner, PlannerConfig
from physmem.v2_train import load_model, SEED_BASE
fam, robot, ck, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
m, _ = load_model(ck, torch.device("cpu"))
pl = CEMPlanner(m, "typed", PlannerConfig(n_z=8, c_prog=2.0, lam_effort=0.03)); pl.anchor_margin = 0.15

def predict(ad, hist, emb, c):
    p0 = ad.target_pos(c.target); qt = (lambda t: t[0] if len(t) == 1 else f"pile:{c.target}")(ad.target_ids(c.target))
    ctx, mem = pl._prepare(hist, ad.frame(c.target), emb, qt); zin, _ = pl._latent_inputs(mem)
    pl.eff_ref = 1.0
    cost, parts = pl._costs(ctx, zin, torch.as_tensor(S.named_call_vector(c.skill, rel_params(c.params, p0))[None]))
    return float(cost.mean()), {k: round(float(v[0]), 3) for k, v in parts.items()}

for i in range(n):
    ad = EnvAdapter(make_env(fam)); ad.reset(api.TaskSpec(fam, "", SEED_BASE["test"] + 900_000 + i, robot, "test"))
    emb = S.embodiment_vector(ad.embodiment_spec()); rng = np.random.default_rng(i); hist = []
    print(f"=== episode {i}")
    for k in range(6):
        sc = ad.env.scripted()
        if sc is None: break
        best = H.decide_env_ranked(pl, ad, hist, emb, rng)
        c = best[1]
        cs, ps = predict(ad, hist, emb, sc)
        cc, pc = predict(ad, hist, emb, c)
        out = ad.execute(c, relative=False); it = out.result.log["interaction"]; hist.append(it)
        fmt = lambda p: {kk: (round(v, 3) if np.isscalar(v) and not isinstance(v, bool) else v) for kk, v in p.items()}
        print(f" k={k} SCRIPTED {sc.skill} {fmt(sc.params)} pred={ps}\n      CHOSEN   {c.skill} {'(=scripted)' if c is sc else fmt(c.params)} pred={pc}\n"
              f"      REAL success={out.success} progress={out.metrics.get('progress'):.2f} effort={out.result.effort.cost():.0f} "
              f"status={out.result.status} collisions={out.metrics.get('collisions')} max_contact={out.metrics.get('max_robot_contact_n', 0):.0f}")
        if out.done: break
