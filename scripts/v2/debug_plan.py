"""MODEL-V2 debug: one closed-loop episode with per-step predicted vs realised outcome."""
import sys, numpy as np, torch
from physmem.v2_train import load_model
from physmem.planner import CEMPlanner, PlannerConfig
from physmem.models.v2 import synth as Y, spec as S, harness as H
from physmem.v2 import api
fam = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "oracle"
m, _ = load_model("results/v2/checkpoints/v2_synth_r1_typed_s0.pt", torch.device("cpu"))
pl = CEMPlanner(m, mode, PlannerConfig(pop=128, iters=3, n_z=16, voi=False, lam_effort=float(sys.argv[3]) if len(sys.argv) > 3 else 0.1))
for seed in range(1_000_000 + 5_000_000, 1_000_000 + 5_000_000 + 3):
    env = Y.SynthEnv(); env.reset(api.TaskSpec(fam, "x", seed, "synth_arm_a", "val"))
    tz, tm = S.labels_to_z(env.lat); hist = []; emb = S.embodiment_vector(Y.ROBOTS["synth_arm_a"].spec)
    print("== seed", seed, {k: round(v, 3) for k, v in env.lat.items()}, {k: (round(v, 3) if np.isscalar(v) else None) for k, v in env.state.items()})
    for k in range(5):
        d = pl.decide(hist, env.frame(), emb, Y.FAMILY_SKILL[fam], env.param_bounds(), np.random.default_rng(k), true_z=tz, true_mask=tm)
        out = env.execute(d.call("target")); it = out.result.log["interaction"]
        print(f" k={k} params={ {kk: round(v, 3) for kk, v in d.params.items()} } pred p_succ={d.info['p_success']:.2f} prog={d.info['progress']:.2f} "
              f"eff={d.info['effort']:.3f} ref={pl.eff_ref:.3f} | real succ={out.success} prog={out.metrics['progress']:.2f} eff={out.result.effort.cost():.3f} cont={np.round(it.outcome_cont[:4],3)}")
        hist.append(it)
        if out.done: break
