import copy, numpy as np, torch
from physmem.v2_train import load_model
from physmem.models.v2 import synth as Y, spec as S
from physmem.models.v2.tensorize import single_batch
from physmem.models.v2.model import TF_LOGVAR
from physmem.v2 import api
m, _ = load_model("results/v2/checkpoints/v2_synth_r1_typed_s0.pt", torch.device("cpu"))
env = Y.SynthEnv(); env.reset(api.TaskSpec("drawer", "x", 6000000, "synth_arm_a", "val"))
tz, tm = S.labels_to_z(env.lat); emb = S.embodiment_vector(Y.ROBOTS["synth_arm_a"].spec)
F = np.linspace(2, 60, 12); D = np.linspace(0.1, 2.0, 10)
P = np.array([[f, d] for f in F for d in D])
calls = np.stack([api.SkillCall("open", {"duration": d, "force": f}).vector() for f, d in P])
b = single_batch([], env.frame(), emb, calls)
Z = torch.as_tensor(np.tile(tz, (len(P), 1))); M = torch.as_tensor(np.tile(tm, (len(P), 1)))
with torch.no_grad():
    o = m(b, z_override=(Z, torch.full_like(Z, TF_LOGVAR)), z_mask_override=M); dd = m.decode(o)
ps = dd["success"].numpy(); pr = dd["cont"][:, 3].numpy(); op = dd["cont"][:, 0].numpy()
for i, (f, d) in enumerate(P):
    st = copy.deepcopy(env.state); sim = Y.step_drawer(st, env.lat, {"duration": d, "force": f}, Y.ROBOTS["synth_arm_a"], np.random.default_rng(0), np.zeros(6))
    if sim.success or ps[i] > 0.2 or i % 9 == 0:
        print(f"F={f:5.1f} D={d:4.2f} sim succ={sim.success} open={sim.cont[0]:.3f} prog={sim.cont[3]:.2f} | model p={ps[i]:.2f} open={op[i]:.3f} prog={pr[i]:.2f}")
# training data stats: how often is drawer success at k=0
z = np.load("data/v2_synth/drawer_train.npz"); k = z["k"]; ev = z["events"]
print("train drawer success rate by k:", [round(float(ev[k == j, 0].mean()), 3) for j in range(5)])
from physmem.planner import CEMPlanner, PlannerConfig
pl = CEMPlanner(m, "oracle", PlannerConfig(pop=128, iters=3, n_z=16, voi=False))
ctx, mem = pl._prepare([], env.frame(), emb)
zin, Z = pl._latent_inputs(mem, tz, tm)
pl.eff_ref = 0.2
cost, parts = pl._costs(ctx, zin, torch.as_tensor(calls))
order = np.argsort(cost.mean(1).numpy())
for i in list(order[:5]) + list(order[-3:]):
    print("PLAN", P[i], "cost", round(float(cost[i].mean()), 3), {k: round(float(v[i]), 3) for k, v in parts.items()})
print("zin oracle", zin[0, :10].numpy().round(2), "tz", tz.round(2))
for i, (f, d) in enumerate(P):
    if abs(d - 0.311) < 0.01:
        print("GOOD", (round(f, 1), round(d, 2)), "cost", round(float(cost[i].mean()), 3), {k: round(float(v[i]), 3) for k, v in parts.items()},
              "events p", dd["p"][i].numpy().round(2))
