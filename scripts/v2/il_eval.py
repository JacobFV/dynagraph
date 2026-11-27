"""closed-loop evaluation of the ACT imitation baseline (il-v0).
usage: il_eval.py RUN FAMILY ROBOT SPLIT N [SEED_BASE] [OUT_JSON]
Sim-time cap: env.time_budget() if ENVS provides it (r7), else TIME_BUDGET env, else 1.5 x the median successful
demo duration of that cell (DEMO_DIR/<fam>__<robot>.json). Records success, sim time to success (cycle time),
damage, instability; resumable (partial json every 5 episodes)."""
import copy, json, os, sys, time
import numpy as np
import torch
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2 import spec as S
from physmem.models.v2.env_adapter import EnvAdapter
from physmem.models.v2.imitation import data as D
ENV_V = D.env_version()
from physmem.models.v2.imitation.act import ACTPolicy, TemporalEnsembler

run, fam, robot, split, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
base = int(sys.argv[6]) if len(sys.argv) > 6 else 2_950_000
out = sys.argv[7] if len(sys.argv) > 7 else f"results/v2_il/il_{run}__{fam}__{robot}__{split}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.set_num_threads(1)
ck = torch.load(f"results/v2/checkpoints/il_{run}.pt", map_location="cpu", weights_only=False)
meta = ck["meta"]
cfg = dict(meta["cfg"]); cfg.setdefault("a_dim", D.A_DIM_V0)       # il-v0 checkpoints (dry run) have 17 dims
model = ACTPolicy(meta["ctx_dim"], **{k: v for k, v in cfg.items() if k != "ctx_dim"})
model.load_state_dict(ck["model"]); model.eval()
fams, H = meta["fams"], meta["cfg"]["H"]
m_ens = os.environ.get("ENS_M", "0.01")
m_ens = None if m_ens == "latest" else ("open" if m_ens == "open" else float(m_ens))
demo_dir = os.environ.get("DEMO_DIR", meta["args"]["data"])
cap = float(os.environ.get("TIME_BUDGET", "0") or 0)
if not cap and os.path.exists(f"{demo_dir}/{fam}__{robot}.json"):
    dm = json.load(open(f"{demo_dir}/{fam}__{robot}.json"))
    cap = 1.5 * float(np.median([m["sim_t"] for m in dm if m["success"]] or [60.0]))
cap = cap or 60.0

res = json.load(open(out + ".partial")) if os.path.exists(out + ".partial") else []
t_wall = time.time()
if os.environ.get("TASKS") == "make_tasks":       # r9: the same task list as every other policy (r9_eval)
    from physmem.envs.splits import make_tasks
    TL = make_tasks(fam, split, n, robot, base_seed=base)
else:
    TL = [api.TaskSpec(fam, "", base + i, robot, split, params={"dressing": "site"}) for i in range(n)]
for i in range(len(res), n):
    seed = int(TL[i].seed)
    env = make_env(fam); ad = EnvAdapter(env)
    ad.reset(TL[i])
    peek = D.StreamPeek(env); peek.idx = copy.deepcopy(ad.idx)
    orig = env.post_physics

    def post(stop=None, _o=orig, _p=peek):
        r = _o(stop); _p.poll(); return r
    env.post_physics = post
    budget = float(env.time_budget()) if hasattr(env, "time_budget") else cap
    emb = S.embodiment_vector(ad.embodiment_spec()).astype(np.float32)
    ctx = np.zeros(meta["ctx_dim"], np.float32); ctx[: len(emb)] = emb
    if fam in fams:
        ctx[len(emb) + fams.index(fam)] = 1
    ex, ens = D.Executor(env), TemporalEnsembler(H, m_ens)
    ABS = meta.get("pos") == "abs"
    T = lambda x: torch.as_tensor(np.asarray(x)[None])
    t0, prev, k, solved_t, err = float(env.t), None, 0, None, None
    try:
        while env.t - t0 < budget and not env.unstable_reasons:
            o = D.observe(env, peek, t0)
            img = T(env.render_wrist(model.img, model.img)) if model.img else None
            p2 = np.concatenate([o["p"], (prev if prev is not None else o["p"])]); prev = o["p"]
            p2 = D.mask_obs_p(p2.reshape(2, -1), meta.get("drop_cmd", 0)).reshape(-1)
            ch = model.act_chunk(T(ctx), T(p2), T(o["obj"]), T(o["reg"]), img)[0].numpy()
            if ABS:
                ch = ch.copy(); ch[:, 0:3] = o["p"][:3] + ch[:, 0:3] * D.ABS_S     # world targets
            ens.push(k, ch)
            a_ = ens.get(k)
            ex.apply(a_, p_target=a_[:3] if ABS else None); k += 1
            if env.solved():
                solved_t = float(env.t - t0)
                break
    except Exception as e:
        err = repr(e)[:200]
    res.append(dict(seed=seed, success=solved_t is not None, cycle_t=solved_t, sim_t=float(env.t - t0), budget=budget,
                    damage=len(getattr(env, "damaged", ())), unstable=bool(env.unstable_reasons), err=err))
    if (i + 1) % 5 == 0 or i + 1 == n:
        json.dump(res, open(out + ".partial", "w"))
        print(f"{run} {fam} {robot} {split} {i + 1}/{n} succ={np.mean([r['success'] for r in res]):.2f} "
              f"({time.time() - t_wall:.0f}s)", flush=True)
json.dump(dict(env_version=ENV_V, run=run, family=fam, robot=robot, split=split, episodes=res, cap=cap,
               success=float(np.mean([r["success"] for r in res]))), open(out, "w"), indent=1)
os.remove(out + ".partial")
print("IL_EVAL_DONE", out, flush=True)
