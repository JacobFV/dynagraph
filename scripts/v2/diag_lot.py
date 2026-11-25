"""why does the typed lot planner probe every part? per decision: risk split (damage vs slip+drop), UCB, kind,
crush slot posterior (lot) mean/sd, true lot crush (evaluator print)."""
import sys, numpy as np, torch
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2 import spec as S
from physmem.models.v2.lot import LotEnvAdapter, LotPlanner, _probed
from physmem.models.v2.env_adapter import rel_params
from physmem.planner import CEMPlanner, PlannerConfig
from physmem.v2_train import load_model
robot, n = sys.argv[1], int(sys.argv[2])
cfg = PlannerConfig(n_z=16, voi=False)
T = [CEMPlanner(load_model(f"results/v2/checkpoints/v2_robots_r7_typed_s{s}.pt", torch.device("cpu"))[0], "typed", cfg) for s in (0, 1, 2)]
lp = LotPlanner(T, tau=0.15)
ci = S.LATENT_INDEX["crush"]
for i in range(n):
    ad = LotEnvAdapter(make_env("bin_clearing"))
    ad.reset(api.TaskSpec("bin_clearing", "", 2_700_000 + i, robot, "test", params={"dressing": "site", "n_parts": 3}))
    emb = S.embodiment_vector(ad.embodiment_spec()); rng = np.random.default_rng(i); hist = []
    print(f"=== lot {i}: {ad.lot}")
    for k in range(12):
        env = ad.env; sc = env.scripted()
        if sc is None: break
        if sc.skill == "pick" and sc.target in ad.lot_ids:
            pl = T[0]; ctx, mem = pl._prepare(hist, ad.frame(sc.target), emb, "lot")
            mu, sd = float(mem["z_mu"][0, ci]), float(torch.exp(0.5 * mem["z_logvar"][0, ci]))
            crush_hat = float(S.z_to_latent("crush", mu))
            # risk split for the chosen skip
            p0 = ad.target_pos(sc.target)
            c, info = lp.decide(ad, hist, emb, rng)
            call = api.SkillCall(sc.skill, dict(sc.params, grip_force=info.get("grip", sc.params["grip_force"]), lift_speed=info.get("lift", 0.2)), target=sc.target)
            o = pl.model(*[]) if False else None
            print(f"  k={k} {sc.target} probed={_probed(env, sc.target)} crush_post={crush_hat:.1f}N (z sd {sd:.2f}) -> {info.get('kind')} risk={info.get('risk', 0):.3f} ucb={info.get('ucb', 0):.3f} grip={info.get('grip')} lift={info.get('lift')}")
        else:
            c, info = sc, {}
        out = ad.execute(c, relative=False); hist.append(out.result.log["interaction"])
        print(f"      -> {c.skill} grip={c.params.get('grip_force', '-')} status={out.result.status} dmg={out.metrics.get('damaged')} prog={out.metrics.get('progress', 0):.2f} tactile={hist[-1].tactile_k:.0f}")
        if out.done: break
