import sys, numpy as np, torch
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2.env_adapter import EnvAdapter, rel_params
from physmem.models.v2 import spec as S
from physmem.planner import CEMPlanner, PlannerConfig
from physmem.v2_train import load_model
fam, robot = sys.argv[1], sys.argv[2]
m, _ = load_model("results/v2/checkpoints/v2_robots_t1_typed_s0.pt", torch.device("cpu"))
pl = CEMPlanner(m, "typed", PlannerConfig(n_z=8, c_prog=2.0, lam_effort=0.03))
ad = EnvAdapter(make_env(fam)); ad.reset(api.TaskSpec(fam, "", 2_900_001, robot, "test"))
emb = S.embodiment_vector(ad.embodiment_spec()); rng = np.random.default_rng(0)
sc = ad.env.scripted(); cs = [sc] + list(ad.candidate_skills(None, 8, rng))
for c in cs:
    p0 = ad.target_pos(c.target)
    ctx, mem = pl._prepare([], ad.frame(c.target), emb, c.target)
    zin, Z = pl._latent_inputs(mem)
    pl.eff_ref = 1.0
    cost, parts = pl._costs(ctx, zin, torch.as_tensor(S.named_call_vector(c.skill, rel_params(c.params, p0))[None]))
    print("SCRIPTED" if c is sc else "cand    ", c.skill, c.target, {k: round(float(v), 3) if np.isscalar(v) else v for k, v in c.params.items()},
          "| cost", round(float(cost.mean()), 3), {k: round(float(v[0]), 3) for k, v in parts.items()})
