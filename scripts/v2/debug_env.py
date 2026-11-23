import numpy as np, torch
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2.env_adapter import EnvAdapter, rel_params
from physmem.models.v2 import spec as S
from physmem.models.v2.tensorize import single_batch
from physmem.v2_train import load_model
m, _ = load_model("results/v2/checkpoints/v2_env_r1_typed_s0.pt", torch.device("cpu"))
ad = EnvAdapter(make_env("singulation")); ad.reset(api.TaskSpec("singulation", "", 2000000, "probe", "test"))
cs = ad.candidate_skills(None, 8, np.random.default_rng(0))
c = cs[0]; fr = ad.frame(c.target)
print("frame finite", np.isfinite(fr.tokens()).all(), fr.tokens().shape, "target", c.target, "params", c.params)
p0 = ad.target_pos(c.target); print("p0", p0)
calls = np.stack([api.SkillCall(x.skill, rel_params(x.params, ad.target_pos(x.target))).vector() for x in cs])
print("calls finite", np.isfinite(calls).all(), calls[:2])
b = single_batch([], fr, S.embodiment_vector(ad.embodiment_spec()), calls)
with torch.no_grad():
    o = m(b); d = m.decode(o)
print({k: v[:3] for k, v in d.items() if torch.is_tensor(v)})
from physmem.planner import CEMPlanner, PlannerConfig
pl = CEMPlanner(m, "typed", PlannerConfig(pop=32, iters=2, n_z=8, voi=False))
ctx, mem = pl._prepare([], fr, S.embodiment_vector(ad.embodiment_spec()))
print("z_mu", mem["z_mu"], "z_lv", mem["z_logvar"], "scale", m.z_var_scale)
zin, Z = pl._latent_inputs(mem)
keys = sorted(k for k, v in c.params.items() if np.isscalar(v))
rel = rel_params(c.params, p0)
P = np.array([[rel[k] for k in keys]])
cost, parts = pl._costs(ctx, zin, pl._vec(c.skill, keys, P))
print("keys", keys, "cost", cost, parts)
