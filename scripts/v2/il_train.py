"""train the ACT-style imitation baseline on il-v0 demos (successful demos only).
usage: il_train.py DATA_DIR RUN [--steps N] [--H 20] [--beta 10] [--d 192] [--bs 256] [--lr 1e-4] [--seed 0]
       [--max-demos N (per cell; data-budget control)]
Checkpoints results/v2/checkpoints/il_<RUN>.pt (+ .partial.pt every 1000 steps; resumes from it)."""
import argparse, glob, json, os, time
import numpy as np
import torch
from physmem.models.v2.imitation import data as D
from physmem.models.v2.imitation.act import ACTPolicy, action_stats

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("run")
ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--H", type=int, default=20)
ap.add_argument("--beta", type=float, default=10.0); ap.add_argument("--d", type=int, default=192)
ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-demos", type=int, default=0)
ap.add_argument("--cells", default=""); ap.add_argument("--img", type=int, default=0)
ap.add_argument("--pos", default="delta", choices=["delta", "abs"])
ap.add_argument("--drop-cmd", type=int, default=1)
ap.add_argument("--trim-idle", type=int, default=1)   # drop leading idle ticks of each demo    # zero controller-command proprio dims (D.CMD_DIMS)   # abs: ACT-style absolute targets per chunk
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _Img5:
    """10 Hz tick t -> 5 Hz stored frame t // 2 (split format)."""
    def __init__(self, arr, off, T, s0=0):
        self.arr, self.off, self.T, self.s0 = arr, off, T, s0

    def __getitem__(self, t):
        t = int(t) + self.s0
        return self.arr[self.off + min(t // 2, (self.T - 1) // 2)]


files = sorted(f for f in glob.glob(f"{a.data}/*__*.npz") if not f.endswith(".partial.npz"))
if a.cells:
    files = [f for f in files if os.path.basename(f)[:-4] in a.cells.split(",")]
fams = sorted({os.path.basename(f).split("__")[0] for f in files})
eps = []
for f in files:
    fam = os.path.basename(f).split("__")[0]
    z = np.load(f, allow_pickle=True)["eps"]
    imgf = f[:-4] + ".img5.npy"
    IMG5 = np.load(imgf) if (a.img and os.path.exists(imgf)) else None     # split format (il_split_img.py)
    good = [e for e in z if e.get("success") and "act" in e and len(e["act"]) > 2]
    if a.max_demos:
        good = good[: a.max_demos]
    for e in good:
        e = dict(e); e["fam"] = fam
        if a.trim_idle and "ref_p" in e:    # drop the leading idle ticks (reference not moving, gripper unchanged)
            mv = np.linalg.norm(np.diff(e["ref_p"], axis=0), axis=1) > 1e-4
            mv |= np.abs(np.diff(e["act"][:, 6], prepend=e["act"][0, 6]))[: len(mv)] > 1e-3
            s0 = max(int(np.argmax(mv)) - 1, 0) if mv.any() else 0
            if s0 > 0:
                for k_ in ("p", "obj", "reg", "act", "t"):
                    if k_ in e:
                        e[k_] = e[k_][s0:]
                e["ref_p"] = e["ref_p"][s0:]
                e["img_s0"] = s0
                e["p"] = e["p"].copy()
                e["p"][:, 27] -= e["p"][0, 27]       # episode-time feature restarts at 0 (as at test time)
        if not a.img:
            e.pop("img", None)          # keep RAM under the small-job cap (lead: < 6 GB)
        elif IMG5 is not None and "img_off" in e:
            e["img"] = _Img5(IMG5, int(e["img_off"]), len(e["act"]) + int(e.get("img_s0", 0)), int(e.get("img_s0", 0)))
        eps.append(e)
    del z
print(f"{len(eps)} demos from {len(files)} cells, families {fams}, device {dev}", flush=True)
nval = max(1, len(eps) // 10)
perm = rng.permutation(len(eps))
val = [eps[i] for i in perm[:nval]]; tr = [eps[i] for i in perm[nval:]]
if a.pos == "abs":     # stats of the absolute-position chunk targets (sampled)
    _s = []
    for e in tr[:400]:
        refp = e["p"][:, :3] + e["p"][:, 3:6] if "ref_p" not in e else e["ref_p"]
        T_ = len(e["act"])
        for t in range(0, T_, 5):
            idx = np.minimum(np.arange(t + 1, t + a.H + 1), len(refp) - 1)
            x = e["act"][np.minimum(np.arange(t, t + a.H), T_ - 1)].copy()
            x[:, 0:3] = (refp[idx] - e["p"][t, :3]) / D.ABS_S
            _s.append(x)
    mu, sd = action_stats(_s)
else:
    mu, sd = action_stats([e["act"] for e in tr])
ctx_dim = len(tr[0]["emb"]) + len(fams)


def ctx_of(e):
    c = np.zeros(ctx_dim, np.float32); c[: len(e["emb"])] = e["emb"]; c[len(e["emb"]) + fams.index(e["fam"])] = 1
    return c


def batch(pool, bs):
    H = a.H
    out = dict(ctx=[], p2=[], obj=[], reg=[], act=[], apad=[])
    for _ in range(bs):
        e = pool[rng.integers(len(pool))]
        T = len(e["act"]); t = int(rng.integers(T))
        out["ctx"].append(ctx_of(e))
        out["p2"].append(D.mask_obs_p(np.concatenate([e["p"][t], e["p"][max(t - 1, 0)]]).reshape(2, -1),
                                      a.drop_cmd).reshape(-1))
        out["obj"].append(e["obj"][t]); out["reg"].append(e["reg"][t])
        ch = e["act"][t:t + H]; pad = np.zeros(H, bool); pad[len(ch):] = True
        if len(ch) < H:
            ch = np.concatenate([ch, np.repeat(ch[-1:], H - len(ch), 0)])
        if a.pos == "abs":       # position dims = (reference at t+i+1) - tcp(t), world frame, / ABS_S
            ch = ch.copy()
            refp = e["p"][:, :3] + e["p"][:, 3:6] if "ref_p" not in e else e["ref_p"]
            idx = np.minimum(np.arange(t + 1, t + H + 1), len(refp) - 1)
            ch[:, 0:3] = (refp[idx] - e["p"][t, :3]) / D.ABS_S
        out["act"].append(ch); out["apad"].append(pad)
        if IMG:
            out.setdefault("img", []).append(e["img"][t])
    return {k: torch.as_tensor(np.stack(v)).to(dev) for k, v in out.items()}


A_DIM = tr[0]["act"].shape[1]
IMG = a.img if all("img" in e for e in tr) else 0
model = ACTPolicy(ctx_dim, d=a.d, H=a.H, a_mean=mu, a_std=sd, a_dim=A_DIM, img=IMG).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
os.makedirs("results/v2/checkpoints", exist_ok=True)
ck = f"results/v2/checkpoints/il_{a.run}"
if os.path.exists(ck + ".pt"):
    print("IL_TRAIN_DONE (exists)", a.run, flush=True)
    raise SystemExit(0)
step0 = 0
if os.path.exists(ck + ".partial.pt"):
    s = torch.load(ck + ".partial.pt", map_location=dev, weights_only=False)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step0 = s["step"]; meta = s["meta"]
    print("resumed at", step0, flush=True)
t0 = time.time()
for step in range(step0, a.steps):
    model.train()
    lr = a.lr * min(1.0, (step + 1) / 500)
    for g in opt.param_groups:
        g["lr"] = lr
    loss, info = model.loss(batch(tr, a.bs), beta=a.beta)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if (step + 1) % 1000 == 0 or step + 1 == a.steps:
        model.eval()
        with torch.no_grad():
            vl = np.mean([model.loss(batch(val, a.bs), beta=a.beta)[1]["l1"] for _ in range(4)])
        print(f"step {step + 1} loss {float(loss):.4f} l1 {info['l1']:.4f} ce {info['ce']:.4f} kl {info['kl']:.3f} "
              f"val_l1 {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        meta = dict(trim_idle=a.trim_idle, drop_cmd=a.drop_cmd, pos=a.pos, cfg=model.cfg, fams=fams, ctx_dim=ctx_dim, args=vars(a), step=step + 1, val_l1=float(vl),
                    schema=D.SCHEMA, env_versions=sorted({json.load(open(f[:-4] + '.json'))[0].get('env_version', 'unlabelled(pre-assert)') for f in files}), n_demos=len(tr))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1, meta=meta), ck + ".partial.pt")
torch.save(dict(model=model.state_dict(), meta=meta), ck + ".pt")
print("IL_TRAIN_DONE", a.run, flush=True)
