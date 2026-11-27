"""closed-loop diagnosis of an ACT run on a TRAIN demo seed: side-by-side every second: demo (recorded) vs policy
tcp position, gripper command, mode. usage: il_diag.py RUN NPZ [demo index]"""
import copy, json, os, sys
import numpy as np, torch
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2 import spec as S
from physmem.models.v2.env_adapter import EnvAdapter
from physmem.models.v2.imitation import data as D
from physmem.models.v2.imitation.act import ACTPolicy, TemporalEnsembler
D.env_version()
run, npz = sys.argv[1], sys.argv[2]; j = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ck = torch.load(f"results/v2/checkpoints/il_{run}.pt", map_location="cpu", weights_only=False); meta = ck["meta"]
cfg = dict(meta["cfg"]); cfg.setdefault("a_dim", D.A_DIM_V0)
model = ACTPolicy(meta["ctx_dim"], **{k: v for k, v in cfg.items() if k != "ctx_dim"}); model.load_state_dict(ck["model"]); model.eval()
z = np.load(npz, allow_pickle=True)["eps"]; mm = json.load(open(npz[:-4] + ".json"))
fam, robot = npz.split("/")[-1][:-4].split("__")
good = [i for i, m in enumerate(mm) if m["success"]]; i = good[j]; e, m = z[i], mm[i]
env = make_env(fam); ad = EnvAdapter(env)
ad.reset(api.TaskSpec(fam, "", m["seed"], robot, "train", params=m.get("params", {"dressing": "site"})))
peek = D.StreamPeek(env); peek.idx = copy.deepcopy(ad.idx); orig = env.post_physics
env.post_physics = lambda stop=None: (lambda r: (peek.poll(), r)[1])(orig(stop))
emb = S.embodiment_vector(ad.embodiment_spec()).astype(np.float32)
ctx = np.zeros(meta["ctx_dim"], np.float32); ctx[:len(emb)] = emb; ctx[len(emb) + meta["fams"].index(fam)] = 1
T = lambda x: torch.as_tensor(np.asarray(x)[None])
ex, ens, prev, t0 = D.Executor(env), TemporalEnsembler(cfg["H"], {"latest": None, "open": "open"}.get(os.environ.get("ENS_M"), None) if os.environ.get("ENS_M") in ("latest", "open") else float(os.environ.get("ENS_M", "0.01"))), None, float(env.t)
ABS = meta.get("pos") == "abs"
# teacher-forced check on the demo's own observations
p2 = np.concatenate([e["p"][0], e["p"][0]])
ch = model.act_chunk(T(ctx), T(p2), T(e["obj"][0]), T(e["reg"][0]))[0].numpy()
print("tf t=0 pred a0[:8]", np.round(ch[0, :8], 2), "demo", np.round(e["act"][0, :8], 2))
for k in range(len(e["act"]) + 20):
    o = D.observe(env, peek, t0)
    p2 = np.concatenate([o["p"], prev if prev is not None else o["p"]]); prev = o["p"]
    p2 = D.mask_obs_p(p2.reshape(2, -1), meta.get("drop_cmd", 0)).reshape(-1)
    ch = model.act_chunk(T(ctx), T(p2), T(o["obj"]), T(o["reg"]))[0].numpy()
    if ABS:
        ch = ch.copy(); ch[:, 0:3] = o["p"][:3] + ch[:, 0:3] * D.ABS_S
    ens.push(k, ch); a = ens.get(k)
    if k % 10 == 0 and os.environ.get("SHOW_CHUNK"):
        print("   chunk targets (world) i=0,4,9,end:", [np.round(ch[i, :3], 3).tolist() for i in (0, 4, min(9, len(ch) - 1), len(ch) - 1)],
              "g", np.round(ch[::5, 6], 2).tolist())
    if k % 10 == 0:
        kk = min(k, len(e["p"]) - 1)
        print(f"t={k/10:4.1f} pol tcp {np.round(o['p'][:3],3)} g {a[6]:.2f} mode {int(np.argmax(a[8:12]))} dp {np.round(a[:3],2)} | "
              f"demo tcp {np.round(e['p'][kk][:3],3)} g {e['act'][kk][6]:.2f} mode {int(np.argmax(e['act'][kk][8:12]))} dp {np.round(e['act'][kk][:3],2)}"
              f" | obj0 pol {np.round(o['obj'][0][:3],2)} demo {np.round(e['obj'][kk][0][:3],2)}")
    ex.apply(a, p_target=a[:3] if ABS else None)
    if env.solved():
        print("SOLVED at", env.t - t0); break
print("final solved", env.solved())
