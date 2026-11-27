"""V3 section 0: contact-chain family, data generation (owner: V3-MODEL).

  python -m physmem.v3_chain gen --out data/v3_chain --split train --n 10000
Splits (seed ranges never overlap): train 0+, val 1_000_000+, test 2_000_000+, test_L4 3_000_000+ (held-out chain
length 4). Evaluator-only arrays are prefixed `lab_` / `eval_` and never enter a forward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from physmem.models.v3 import chain as C

SEED_BASE = {"train": 0, "val": 1_000_000, "test": 2_000_000, "test_L4": 3_000_000,
             # fresh TEST seed sets for the exploratory arms (V3 §0d): never used before their pre-registration
             "test2": 4_000_000, "test2_L4": 5_000_000}


def gen(out, split, n, start=0):
    os.makedirs(out, exist_ok=True)
    Ls = (4,) if split.endswith("_L4") else (1, 2, 3)
    acc = {}
    t0 = time.time()
    for i in range(start, start + n):
        seed = SEED_BASE[split] + i
        p, tr, obs, a, lab = C.generate_episode(seed, Ls=Ls)
        row = dict(a)
        row.update({f"lab_{k}": v for k, v in lab.items()})
        row["eval_seed"] = np.int64(seed)
        row["eval_mass"] = np.pad(p.mass, (0, C.L_MAX - p.L)).astype(np.float32)
        row["eval_mu"] = np.pad(p.mu, (0, C.L_MAX - p.L)).astype(np.float32)
        row["eval_half"] = np.pad(p.half, (0, C.L_MAX - p.L)).astype(np.float32)
        row["eval_gaps"] = np.pad(p.gaps, (0, C.L_MAX - 1 - len(p.gaps))).astype(np.float32)
        row["eval_k"] = np.float32(p.k)
        row["eval_pgap"] = np.array(p.pusher_gap, np.float32)
        sx = np.zeros((C.N_WIN, C.L_MAX + 1), np.float32); sv = np.zeros_like(sx)
        sx[:, :p.L + 1] = tr["start_x"]; sv[:, :p.L + 1] = tr["start_v"]
        row["eval_start_x"], row["eval_start_v"] = sx, sv
        row["eval_occ"] = np.bool_(obs["cam_occ"])
        for k, v in row.items():
            acc.setdefault(k, []).append(np.asarray(v))
        if (i - start + 1) % 500 == 0:
            print(f"{split} {i - start + 1}/{n} {time.time() - t0:.0f}s", flush=True)
    arr = {k: np.stack(v) for k, v in acc.items()}
    fn = os.path.join(out, f"{split}.npz" if start == 0 else f"{split}_{start}.npz")
    np.savez(fn, **arr)
    h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
    man_fn = os.path.join(out, "manifest.json")
    man = json.load(open(man_fn)) if os.path.exists(man_fn) else {}
    man[os.path.basename(fn)] = dict(n=n, seeds=[SEED_BASE[split] + start, SEED_BASE[split] + start + n - 1],
                                     Ls=list(Ls), sha256=h, seconds=round(time.time() - t0, 1),
                                     time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(man, open(man_fn, "w"), indent=1)
    print(f"wrote {fn} sha256={h[:16]} in {time.time() - t0:.0f}s", flush=True)


def cfgen(data, split, n=None):
    """real do()-data (Arm I): for every episode window, one random named-property edit (box, mass/friction/k,
    sign, 0.67 normalised units) and the MuJoCo re-simulation of that window from its TRUE start state with the
    edited property. Evaluator/training-side only (labels)."""
    from physmem.v3_chain_eval import true_params, perturb, sim_window
    from physmem.models.v3.belief import si_to_p
    z = dict(np.load(os.path.join(data, f"{split}.npz")))
    N = len(z["L"]) if n is None else min(n, len(z["L"]))
    W, E, NT = C.N_WIN, C.N_ENT, C.N_TAU
    X = np.zeros((N, W, E, NT), np.float32); V = np.zeros_like(X)
    ent = np.zeros((N, W), np.int8); prop = np.zeros((N, W), np.int8); val = np.zeros((N, W), np.float32)
    t0 = time.time()
    for e in range(N):
        L = int(z["L"][e]); p = true_params(z, e)
        for k in range(W):
            rng = np.random.default_rng([7, SEED_BASE[split] + e, k])
            box = int(rng.integers(L))
            props = [0, 1] + ([2] if (p.spring and box == L - 1) else [])
            pr = int(props[rng.integers(len(props))]); sg = float(rng.choice([-1.0, 1.0]))
            q = perturb(p, box, pr, sg)
            F, T = z["action"][e, k]
            Xg, _ = sim_window(q, z, e, k, F, T)
            _, Vg = None, None
            # velocities: finite difference of the grid (sim_window returns positions); adequate on a 0.1 s grid
            Vg = np.gradient(Xg, C.TAU, axis=0)
            X[e, k, :L + 1] = Xg.T; V[e, k, :L + 1] = Vg.T
            lm, mu, lk = si_to_p(q.mass, q.mu, q.k if q.spring else None)
            ent[e, k] = 1 + box; prop[e, k] = pr
            val[e, k] = (lm[box], mu[box], lk if lk is not None else 0.0)[pr]
        if (e + 1) % 500 == 0:
            print(f"cfgen {split} {e + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    fn = os.path.join(data, f"{split}_cf.npz")
    np.savez(fn, cf_X=X, cf_V=V, cf_ent=ent, cf_prop=prop, cf_val=val)
    print(f"wrote {fn} {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- training
RESULTS = os.environ.get("V3_RESULTS", "results/v3")


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@__import__("torch").no_grad()
def quick_eval(model, store, passes=4, bs=1024, drop=None, seed=0):
    """VAL/TEST summary: final-position error of boxes (mm), trajectory RMSE (mm), param corr/abs err per k."""
    import torch
    from physmem.models.v3 import belief as B
    model.eval()
    rows = store.rows()
    g = torch.Generator(device=store.device).manual_seed(seed)
    fe, tr, P, T, M, K = [], [], [], [], [], []
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        b, lab = store.gather(r, drop)
        ids = torch.randn(len(r), C.N_ENT, B.ID_D, device=store.device, generator=g)
        st = model(b, passes=passes, ids=ids)[-1]
        box = b.ent_type == C.ENT_BOX
        err = (st.xm() * B.X_SC - lab.X).abs()
        fe.append(torch.where(box, err[..., -1], torch.nan))
        tr.append(torch.where(box[..., None], err, torch.nan))
        pm_t, pmask, *_ = B.label_targets(b, lab)
        P.append(st.pm()); T.append(pm_t); M.append(pmask); K.append(r[:, 1])
    fe, tr = torch.cat(fe), torch.cat(tr)
    out = dict(final_err_mm=float(torch.nanmean(fe) * 1e3), traj_err_mm=float(torch.nanmean(tr) * 1e3))
    P, T, M, K = map(torch.cat, (P, T, M, K))
    for j, name in enumerate(B.P_FIELDS):
        for k in (1, 3):
            m = M[..., j] & (K[:, None] == k)
            if m.sum() > 10:
                p_, t_ = P[..., j][m], T[..., j][m]
                out[f"{name}:corr@k{k}"] = float(torch.corrcoef(torch.stack([p_, t_]))[0, 1])
    model.train()
    return out


def train(args):
    import torch
    from physmem.models.v3 import belief as B
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":     # hard cap: shared host rule (< 6 GB per small GPU job)
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_cap_gb * 2 ** 30 / tot))
    tr = B.ChainStore(os.path.join(args.data, "train.npz"), dev, args.max_episodes)
    va = B.ChainStore(os.path.join(args.data, "val.npz"), dev)
    model = B.build(args.kind, args.d).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    law_w = 0.0 if args.nolaw else args.law_w
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps + 1, pct_start=0.05)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    rows = tr.rows()
    cfg = {k: v for k, v in vars(args).items() if k != "func"}
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    os.makedirs(os.path.join(RESULTS, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "curves"), exist_ok=True)
    print(f"[{args.run}] kind={args.kind} law_w={law_w} params={nparam} rows={len(rows)} dev={dev} cfg={cfg_hash}",
          flush=True)
    curve, t0 = [], time.time()
    for step in range(args.steps + 1):
        idx = torch.randint(0, len(rows), (args.bs,), device=dev, generator=gen)
        drop = {}
        u = float(torch.rand((), generator=gen, device=dev))
        if u < 0.15:
            drop = {"no_track": True}
        elif u < 0.30:
            drop = {"no_proprio": True}
        elif u < 0.55:
            drop = {"no_anchor": True}
        b, lab = tr.gather(rows[idx], drop)
        passes = int(torch.randint(args.pmin, args.pmax + 1, (), generator=gen, device=dev))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda" and not args.no_amp)):
          if args.kind == "arm_i":
            cf = tr.gather_cf(rows[idx], b, lab, args.cf_frac, gen) if tr.has_cf else None
            loss, logs = B.arm_i_loss(model, b, lab, passes, cf=cf, law_w=law_w, gen=gen)
          else:
            loss, logs = B.total_loss(model, b, lab, passes, law_w=law_w,
                                      cf_frac=0.0 if args.nolaw else args.cf_frac, ds_w=args.ds_w, gen=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0:
            logs.update(step=step, passes=passes, gnorm=float(gn), elapsed=round(time.time() - t0, 1),
                        peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == "cuda" else 0)
            curve.append(logs)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in logs.items()}), flush=True)
        if step > 0 and (step % args.eval_every == 0 or step == args.steps):
            ev = quick_eval(model, va)
            curve.append(dict(step=step, val=ev))
            print(json.dumps(dict(step=step, val={k: round(v, 3) for k, v in ev.items()})), flush=True)
            torch.save(dict(state=model.state_dict(), kind=args.kind, d=args.d, cfg=cfg, cfg_hash=cfg_hash,
                            params=nparam, partial_step=step),
                       os.path.join(RESULTS, "checkpoints", f"{args.run}.partial.pt"))
    ck = os.path.join(RESULTS, "checkpoints", f"{args.run}.pt")
    torch.save(dict(state=model.state_dict(), kind=args.kind, d=args.d, cfg=cfg, cfg_hash=cfg_hash, params=nparam), ck)
    json.dump(curve, open(os.path.join(RESULTS, "curves", f"{args.run}.json"), "w"))
    with open(os.path.join(RESULTS, "train_runs.jsonl"), "a") as f:
        f.write(json.dumps(dict(run=args.run, kind=args.kind, law_w=law_w, seed=args.seed, params=nparam,
                                steps=args.steps, seconds=time.time() - t0, cfg_hash=cfg_hash, time_utc=_utc(),
                                val=curve[-1].get("val"))) + "\n")
    print(f"[{args.run}] done {time.time() - t0:.0f}s -> {ck}", flush=True)


def load(ck, device=None):
    import torch
    from physmem.models.v3 import belief as B
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = torch.load(ck, map_location=dev, weights_only=False)
    m = B.build(c["kind"], c["d"]).to(dev)
    m.load_state_dict(c["state"])
    m.eval()
    return m, c


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", default="data/v3_chain")
    g.add_argument("--split", required=True, choices=list(SEED_BASE))
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--start", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/v3_chain")
    t.add_argument("--run", required=True)
    t.add_argument("--kind", default="lbs", choices=["lbs", "v1", "dense", "generic", "proj", "arm_i"])
    t.add_argument("--nolaw", action="store_true")
    t.add_argument("--law-w", type=float, default=1.0)
    t.add_argument("--cf-frac", type=float, default=0.25)
    t.add_argument("--ds-w", type=float, default=0.3)
    t.add_argument("--d", type=int, default=128)
    t.add_argument("--steps", type=int, default=15000)
    t.add_argument("--bs", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--pmin", type=int, default=2)
    t.add_argument("--pmax", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-episodes", type=int, default=None)
    t.add_argument("--gpu-cap-gb", type=float, default=5.0)
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--log-every", type=int, default=250)
    t.add_argument("--eval-every", type=int, default=2500)
    c = sub.add_parser("cfgen")
    c.add_argument("--data", default="data/v3_chain"); c.add_argument("--split", default="train")
    c.add_argument("--n", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "cfgen":
        cfgen(a.data, a.split, a.n)
    elif a.cmd == "gen":
        gen(a.out, a.split, a.n, a.start)
    elif a.cmd == "train":
        train(a)


if __name__ == "__main__":
    main()
