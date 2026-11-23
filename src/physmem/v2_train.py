"""v2 world-model training (owner: MODEL-V2).

  python -m physmem.v2_train gen   --data data/v2_synth --n-train 3000          # synthetic sanity world
  python -m physmem.v2_train train --data data/v2_synth --variant typed --steps 20000 --run v2_synth_typed_s0
  python -m physmem.v2_train eval  --ckpt results/v2/checkpoints/<run>.pt --data data/v2_synth

Splits (decided BEFORE any windowing; a root episode never crosses splits):
  train / val / test       : disjoint episode seed ranges, training robots
  test_robot               : held-out robot(s) (synth_mobile), test seeds
  test_stress              : out-of-range latents (slide friction 0.07 / 1.0), test seeds
  heldout family (G5)      : --holdout-family drops that family from train/val (evaluated on its test split)
Scalers are fit on TRAIN only. Rows -> results/v2/metrics.jsonl (run ids prefixed v2_).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from physmem.models.v2 import spec as S
from physmem.models.v2 import synth as Y
from physmem.models.v2.model import PhysMemV2, count_params, v2_loss
from physmem.models.v2.tensorize import FamilyStore, build_family_arrays, arrays_hash

RESULTS = os.environ.get("V2_RESULTS", "results/v2")
SEED_BASE = {"train": 0, "val": 1_000_000, "test": 2_000_000, "test_robot": 3_000_000, "test_stress": 4_000_000}


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_rows(rows, path=None):
    path = path or os.path.join(RESULTS, "metrics.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")


# ----------------------------------------------------------------- data generation
def _gen_job(args):
    fam, robots, split, n, stress = args
    base = SEED_BASE[split] + Y.FAMILIES.index(fam) * 100_000
    eps = []
    for i in range(n):
        robot = robots[i % len(robots)]
        ep = Y.generate_episode(fam, robot, base + i, stress=stress)
        eps.append((fam, robot, ep.steps))
    emb = {r: S.embodiment_vector(Y.ROBOTS[r].spec) for r in Y.ROBOTS}
    arr = build_family_arrays(eps, lambda r: emb[r], np.random.default_rng(base))
    return fam, split, arr


def generate(out, n_train=3000, n_eval=400, workers=8):
    os.makedirs(out, exist_ok=True)
    jobs = []
    for fam in Y.FAMILIES:
        jobs += [(fam, Y.TRAIN_ROBOTS, "train", n_train, False), (fam, Y.TRAIN_ROBOTS, "val", n_eval, False),
                 (fam, Y.TRAIN_ROBOTS, "test", n_eval, False), (fam, Y.HELDOUT_ROBOTS, "test_robot", n_eval, False)]
        if fam == "slide":
            jobs.append((fam, Y.TRAIN_ROBOTS, "test_stress", n_eval, True))
    t0 = time.time()
    manifest = {}
    with Pool(workers) as p:
        for fam, split, arr in p.imap_unordered(_gen_job, jobs):
            fn = os.path.join(out, f"{fam}_{split}.npz")
            np.savez(fn, **arr)
            manifest[f"{fam}_{split}"] = dict(n=int(len(arr["k"])), hash=arrays_hash(arr), n_ent_max=int(arr["ent"].shape[1]))
            print(f"{fam:7s} {split:11s} rows={len(arr['k'])} ent_max={arr['ent'].shape[1]} ({time.time() - t0:.0f}s)",
                  flush=True)
    manifest["_meta"] = dict(n_train=n_train, n_eval=n_eval, time_utc=_utc(), schema=S.V2_MODEL_SCHEMA,
                             gen_seconds=time.time() - t0)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)


def _collect_job(args):
    fam, robot, split, seeds = args
    import warnings
    warnings.filterwarnings("ignore")
    from physmem.envs import make_env
    from physmem.models.v2 import env_adapter as A
    from physmem.v2 import api
    eps, fails = [], 0
    for sd in seeds:
        try:
            env = make_env(fam)
            eps.append(A.collect_episode(env, api.TaskSpec(fam, "", int(sd), robot, split), rng=np.random.default_rng(sd)))
        except Exception as e:  # keep going; failures are counted and reported
            fails += 1
            print(f"collect fail {fam} {robot} {split} {sd}: {type(e).__name__}: {e}", flush=True)
    emb = S.embodiment_vector(env.robot.spec) if eps else np.zeros(S.EMB_DIM, np.float32)
    return fam, split, robot, eps, emb, fails


def collect(out, families, robots, n_train, n_eval, workers=8, chunk=10):
    """ENVS stations -> v2 arrays (data/v2_env/{family}_{split}.npz). seeds: train [0,n), val/test offset ranges."""
    os.makedirs(out, exist_ok=True)
    jobs = []
    for fam in families:
        for robot in robots:
            for split, n in (("train", n_train), ("val", n_eval), ("test", n_eval)):
                base = SEED_BASE[split]
                seeds = list(range(base, base + n))
                jobs += [(fam, robot, split, seeds[i:i + chunk]) for i in range(0, n, chunk)]
    t0 = time.time()
    acc, embs, nfail = {}, {}, 0
    with Pool(workers) as p:
        for fam, split, robot, eps, emb, fails in p.imap_unordered(_collect_job, jobs):
            acc.setdefault((fam, split), []).extend(eps)
            embs[robot] = emb
            nfail += fails
    manifest = dict(_meta=dict(families=families, robots=robots, n_train=n_train, n_eval=n_eval, failures=nfail,
                               seconds=time.time() - t0, time_utc=_utc()))
    for (fam, split), eps in acc.items():
        if not eps:
            continue
        arr = build_family_arrays(eps, lambda r: embs[r], np.random.default_rng(0))
        np.savez(os.path.join(out, f"{fam}_{split}.npz"), **arr)
        manifest[f"{fam}_{split}"] = dict(n=int(len(arr["k"])), episodes=len(eps), hash=arrays_hash(arr),
                                          success_rate=float(arr["events"][:, 0].mean()))
        print(f"{fam:14s} {split:6s} episodes={len(eps)} rows={len(arr['k'])}", flush=True)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1, default=float)
    print(f"collect done in {time.time() - t0:.0f}s, failures={nfail}")


SANITY_LIMITS = dict(seg=200.0, an=1e3, ent=100.0, cont=1e3, call=1e4)   # normalized-unit magnitude limits


def sanitize(arr, name=""):
    """drop WHOLE episodes containing non-finite or physically absurd values (simulator blow-ups); re-index."""
    n = len(arr["k"])
    bad_row = np.zeros(n, bool)
    for k, lim in SANITY_LIMITS.items():
        a = arr[k].reshape(n, -1).astype(np.float32)
        bad_row |= ~np.isfinite(a).all(1) | (np.abs(np.nan_to_num(a)) > lim).any(1)
    if not bad_row.any():
        return arr
    bad_eps = np.unique(arr["ep_index"][bad_row])
    keep = ~np.isin(arr["ep_index"], bad_eps)
    out = {k: v[keep] for k, v in arr.items()}
    # recompute ep_start: first kept row of each episode
    epi = out["ep_index"]
    first = np.r_[True, epi[1:] != epi[:-1]]
    starts = np.maximum.accumulate(np.where(first, np.arange(len(epi)), 0))
    out["ep_start"] = starts
    print(f"[sanitize] {name}: dropped {len(bad_eps)} episodes ({int((~keep).sum())} rows) with non-finite/absurd values",
          flush=True)
    return out


def _convert_shard(path):
    import warnings
    warnings.filterwarnings("ignore")
    from physmem.envs.io import load_shard
    from physmem.models.v2 import env_adapter as A
    out, fails = [], 0
    for ep in load_shard(path):
        if ep.get("error"):
            fails += 1
            continue
        try:
            out.append((ep["task"]["split"], A.convert_episode(ep)))
        except Exception as e:
            fails += 1
            print(f"convert fail {path} {ep.get('episode_id')}: {type(e).__name__}: {e}", flush=True)
    return out, fails


def convert(pattern, out, workers=8):
    """stored ENVS/ROBOTS tranches (physmem.envs.io shards) -> v2 arrays per (family, split), robots mixed."""
    import glob as _g
    from physmem.models.v2 import env_adapter as A
    os.makedirs(out, exist_ok=True)
    paths = sorted(_g.glob(pattern, recursive=True))
    acc, nf, t0 = {}, 0, time.time()
    with Pool(workers) as p:
        for res, fails in p.imap_unordered(_convert_shard, paths):
            nf += fails
            for split, (fam, robot, steps) in res:
                if steps:
                    acc.setdefault((fam, split), []).append((fam, robot, steps))
    manifest = dict(_meta=dict(pattern=pattern, shards=len(paths), failures=nf, seconds=time.time() - t0, time_utc=_utc()))
    embs = {}
    for (fam, split), eps in sorted(acc.items()):
        for _, r, _s in eps:
            if r not in embs:
                embs[r] = S.embodiment_vector(A.robot_spec(r))
        arr = build_family_arrays(eps, lambda r: embs[r], np.random.default_rng(0))
        np.savez(os.path.join(out, f"{fam}_{split}.npz"), **arr)
        rob = {r: int((arr["robot"] == r).sum()) for r in np.unique(arr["robot"])}
        manifest[f"{fam}_{split}"] = dict(n=int(len(arr["k"])), episodes=len(eps), robots=rob, hash=arrays_hash(arr))
        print(f"{fam:16s} {split:16s} episodes={len(eps):5d} rows={len(arr['k']):6d} robots={rob}", flush=True)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1, default=float)
    print(f"convert done in {time.time() - t0:.0f}s, failures={nf}", flush=True)


def load_stores(data, split, device, families=None, max_episodes=None):
    st = {}
    if families is None:
        suf = f"_{split}.npz"
        known = set(Y.FAMILIES)
        families = sorted({f[:-len(suf)] for f in os.listdir(data) if f.endswith(suf)
                           and not any(f[:-len(suf)].endswith("_" + s_) for s_ in ("test", "val", "train"))})
    for fam in families:
        fn = os.path.join(data, f"{fam}_{split}.npz")
        if os.path.exists(fn):
            z = np.load(fn, allow_pickle=False)
            arr = {k: z[k] for k in z.files}
            arr = sanitize(arr, f"{fam}_{split}")
            if max_episodes is not None:     # contiguous prefix of whole episodes (adapter-data budget)
                n = int((arr["ep_index"] < max_episodes).sum())
                arr = {k: v[:n] for k, v in arr.items()}
            st[fam] = FamilyStore(arr, fam, device)
    return st


def fit_scalers(model, stores):
    c = torch.cat([s.cont for s in stores.values()])
    e = torch.cat([s.effort for s in stores.values()])
    model.cont_mean.copy_(c.mean(0))
    model.cont_std.copy_(c.std(0).clamp(min=1e-3))
    model.eff_mean.copy_(e.mean())
    model.eff_std.copy_(e.std().clamp(min=1e-3))
    model.ev_supervised.copy_(torch.cat([s.events_mask for s in stores.values()]).any(0))
    calls = torch.cat([s.call for s in stores.values()])
    nk, P = len(S.SKILLS), S.N_PARAM
    vals, msk = calls[:, nk:nk + P], calls[:, nk + P:] > 0.5
    for i in range(P):
        m = msk[:, i]
        if m.sum() > 1:
            model.call_mean[i] = vals[m, i].mean()
            model.call_std[i] = vals[m, i].std().clamp(min=1e-6)


@torch.no_grad()
def evaluate(model, stores, max_hist=(0, 1, 2, 3), bs=1024, oracle=False):
    """per family: latent recovery after j interactions (corr, abs err in normalized units, calibration),
    outcome metrics (success Brier/AUC, progress RMSE, cont NLL), effort log-RMSE."""
    model.eval()
    out = {}
    for fam, st in stores.items():
        dv = st.k.device
        res = {}
        idx_all = torch.arange(st.n, device=dv)
        # outcome prediction at native history
        P, Y_, E, PE, Pr, Yp, NLL = [], [], [], [], [], [], []
        for i in range(0, st.n, bs):
            idx = idx_all[i:i + bs]
            b, lab = st.gather(idx)
            zo = (lab.z, torch.full_like(lab.z, math.log(0.05 ** 2))) if oracle else None
            o = model(b, z_override=zo, z_mask_override=lab.z_mask if oracle else None)
            d = model.decode(o)
            P.append(d["success"]); Y_.append(lab.events[:, 0])
            E.append(d["log_effort"]); PE.append(lab.effort)
            Pr.append(d["cont"][:, 3]); Yp.append(lab.cont[:, 3])
            yc = (lab.cont - model.cont_mean) / model.cont_std
            NLL.append((0.5 * (o["cont_logvar"] + (yc - o["cont_mu"]) ** 2 * torch.exp(-o["cont_logvar"]))).mean(1))
        P, Y_ = torch.cat(P).float(), torch.cat(Y_).float()
        res["success_brier"] = float(((P - Y_) ** 2).mean())
        res["success_auc"] = _auc(P.cpu().numpy(), Y_.cpu().numpy())
        res["success_rate"] = float(Y_.mean())
        res["effort_log_rmse"] = float(((torch.cat(E) - torch.cat(PE)) ** 2).mean().sqrt())
        res["progress_rmse"] = float(((torch.cat(Pr) - torch.cat(Yp)) ** 2).mean().sqrt())
        res["cont_nll"] = float(torch.cat(NLL).mean())
        # event metrics per applicable event
        if model.variant == "typed" and not oracle:
            for j in max_hist:
                sel = idx_all[st.k >= j]
                if len(sel) < 20:
                    continue
                MU, LV, Z, ZM = [], [], [], []
                for i in range(0, len(sel), bs):
                    idx = sel[i:i + bs]
                    b, lab = st.gather(idx, k_override=torch.full_like(idx, j))
                    o = model(b)
                    MU.append(o["mem"]["z_mu"]); LV.append(o["mem"]["z_logvar"]); Z.append(lab.z); ZM.append(lab.z_mask)
                MU, LV, Z, ZM = map(torch.cat, (MU, LV, Z, ZM))
                for li, name in enumerate(S.LATENT_NAMES):
                    m = ZM[:, li]
                    if m.sum() < 20:
                        continue
                    mu, z, lv = MU[m, li], Z[m, li], LV[m, li]
                    c = float(np.corrcoef(mu.cpu().numpy(), z.cpu().numpy())[0, 1]) if mu.std() > 1e-6 else 0.0
                    z2 = float(((mu - z) ** 2 * torch.exp(-lv)).mean())
                    cov80 = float(((mu - z).abs() <= 1.2816 * torch.exp(0.5 * lv)).float().mean())
                    res[f"latent:{name}:corr@k{j}"] = c
                    res[f"latent:{name}:abs_err@k{j}"] = float((mu - z).abs().mean())
                    res[f"latent:{name}:std@k{j}"] = float(torch.exp(0.5 * lv).mean())
                    res[f"latent:{name}:z2@k{j}"] = z2
                    res[f"latent:{name}:cov80@k{j}"] = cov80
        out[fam] = res
    model.train()
    return out


def _auc(p, y):
    pos, neg = p[y > 0.5], p[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.r_[pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


# ----------------------------------------------------------------- training
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keep = lambda f: f != args.holdout_family and (not args.families or f in args.families.split(","))
    tr = {f: s for f, s in load_stores(args.data, "train", dev, max_episodes=args.max_episodes).items() if keep(f)}
    va = {f: s for f, s in load_stores(args.data, "val", dev).items() if keep(f)}
    model = PhysMemV2(args.variant, d=args.d, phys=not args.no_phys).to(dev)
    fit_scalers(model, tr)
    if args.init:
        # adapter / fine-tune: start from a trained model; keep its output scalers, add call scalers for new skills
        init, _ = load_model(args.init, dev)
        new_call_mean, new_call_std = model.call_mean.clone(), model.call_std.clone()
        model.load_state_dict(init.state_dict())
        unseen = (init.call_std == 1) & (init.call_mean == 0)       # param names the init model never saw
        model.call_mean[unseen], model.call_std[unseen] = new_call_mean[unseen], new_call_std[unseen]
        if args.freeze:
            for name, p_ in model.named_parameters():
                p_.requires_grad = not any(name.startswith(x) for x in args.freeze.split(","))
    cfg = {k: v for k, v in vars(args).items() if k not in ("func",)}
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    nparam = count_params(model)
    print(f"[{args.run}] variant={args.variant} params={nparam} device={dev} cfg={cfg_hash}", flush=True)
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps + 1, pct_start=0.05)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    os.makedirs(os.path.join(RESULTS, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "curves"), exist_ok=True)
    curve = []
    t0 = time.time()
    seen = 0
    fam_list = list(tr)
    amp = dev.type == "cuda" and not args.no_amp
    for step in range(args.steps + 1):
        # curriculum: early steps oversample short histories (k<=1) and high teacher forcing; anneal to uniform
        frac = min(step / max(args.steps * 0.3, 1), 1.0)
        fam = fam_list[step % len(fam_list)]
        st = tr[fam]
        idx = torch.randint(0, st.n, (args.bs,), device=dev, generator=gen)
        b, lab = st.gather(idx, drop_hist=args.drop_hist, gen=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, logs = v2_loss(model, b, lab, tf_prob=args.tf_prob * (1.5 - 0.5 * frac))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        seen += args.bs
        if step % args.log_every == 0:
            el = time.time() - t0
            logs.update(step=step, fam=fam, samples_per_s=seen / max(el, 1e-6), elapsed=el)
            curve.append(logs)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in logs.items()}), flush=True)
        if step > 0 and (step % args.eval_every == 0 or step == args.steps):
            ev = evaluate(model, va)
            summ = {f"{f}:{k}": v for f, r in ev.items() for k, v in r.items()
                    if ("corr@k1" in k or k in ("success_brier", "success_auc", "progress_rmse", "effort_log_rmse"))}
            curve.append(dict(step=step, val=summ))
            print(json.dumps(dict(step=step, val={k: round(v, 3) for k, v in summ.items()})), flush=True)
    elapsed = time.time() - t0
    if model.variant == "typed":
        print("val variance temperature:", np.round(calibrate(model, va), 3).tolist(), flush=True)
    ck = os.path.join(RESULTS, "checkpoints", f"{args.run}.pt")
    torch.save(dict(state=model.state_dict(), variant=args.variant, d=args.d, phys=model.phys, cfg=cfg, cfg_hash=cfg_hash,
                    params=nparam), ck)
    json.dump(curve, open(os.path.join(RESULTS, "curves", f"{args.run}.json"), "w"))
    rows = [dict(run_id=args.run, experiment="v2_train", model="physmem_v2", variant=args.variant, seed=args.seed,
                 metric="throughput_samples_per_s", value=args.steps * args.bs / elapsed, status="observed",
                 config_hash=cfg_hash, time_utc=_utc(), params=nparam, train_seconds=elapsed, amp=amp)]
    write_rows(rows)
    print(f"[{args.run}] done in {elapsed:.0f}s ({args.steps * args.bs / elapsed:.0f} samples/s) -> {ck}", flush=True)
    if not args.no_test:
        run_eval(ck, args.data, splits=("test", "test_robot", "test_stress"),
                 families=args.families.split(",") if args.families else None)
    return ck


@torch.no_grad()
def calibrate(model, stores, bs=1024):
    """per-latent variance temperature on VAL: scale so the 80% interval covers 80% (k>=1 interactions)."""
    model.z_var_scale.fill_(1.0)
    R = [[] for _ in range(S.N_LATENT)]
    for st in stores.values():
        idx_all = torch.nonzero(st.k >= 1).flatten()
        for i in range(0, len(idx_all), bs):
            idx = idx_all[i:i + bs]
            b, lab = st.gather(idx)
            o = model(b)
            r = (o["mem"]["z_mu"] - lab.z).abs() / torch.exp(0.5 * o["mem"]["z_logvar"])
            for l in range(S.N_LATENT):
                R[l].append(r[lab.z_mask[:, l], l])
    for l in range(S.N_LATENT):
        r = torch.cat(R[l]) if R[l] else torch.zeros(0)
        if len(r) > 50:
            model.z_var_scale[l] = float(np.clip(float(torch.quantile(r.float(), 0.8) / 1.2816) ** 2, 0.25, 10.0))
    return model.z_var_scale.cpu().numpy()


def load_model(ck, device=None):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = torch.load(ck, map_location=dev, weights_only=False)
    m = PhysMemV2(c["variant"], d=c["d"], phys=c.get("phys", False)).to(dev)
    sd = c["state"]
    if "ev_supervised" not in sd:
        sd["ev_supervised"] = torch.ones(S.N_EVENTS, dtype=torch.bool, device=dev)
    if "z_var_scale" not in sd:
        sd["z_var_scale"] = torch.ones(S.N_LATENT, device=dev)
    m.load_state_dict(sd)
    m.eval()
    return m, c


def run_eval(ck, data, splits=("test", "test_robot", "test_stress"), oracle_too=True, families=None):
    model, c = load_model(ck)
    dev = next(model.parameters()).device
    run = os.path.basename(ck)[:-3]
    rows = []
    allres = {}
    for split in splits:
        st = load_stores(data, split, dev, families)
        if not st:
            continue
        variants = [("model", False)] + ([("oracle", True)] if oracle_too and model.variant == "typed" else [])
        for tag, orc in variants:
            ev = evaluate(model, st, oracle=orc)
            allres[f"{split}:{tag}"] = ev
            for fam, r in ev.items():
                for k, v in r.items():
                    rows.append(dict(run_id=f"{run}-{split}", experiment="v2_eval", model="physmem_v2" if not orc
                                     else "physmem_v2_oracle(labelled)", variant=c["variant"], seed=c["cfg"]["seed"],
                                     stratum=f"{split}:{fam}", metric=k, value=v, status="observed",
                                     config_hash=c["cfg_hash"], time_utc=_utc(), n_rows=st[fam].n))
    write_rows(rows)
    os.makedirs(os.path.join(RESULTS, "eval"), exist_ok=True)
    json.dump(allres, open(os.path.join(RESULTS, "eval", f"{run}.json"), "w"), indent=1, default=float)
    for key, ev in allres.items():
        for fam, r in ev.items():
            lat = {k.split(":")[1] + k.split(":")[2][-3:]: round(v, 3) for k, v in r.items() if ":corr@" in k}
            print(key, fam, {k: round(r[k], 3) for k in ("success_brier", "success_auc", "progress_rmse",
                                                          "effort_log_rmse")}, lat, flush=True)
    return allres


def closed_loop(args):
    """G3: paired closed-loop episodes on the synthetic world, all policies on identical task seeds."""
    from physmem.models.v2 import harness as H
    from physmem.planner import CEMPlanner, PlannerConfig
    from physmem.v2 import api
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typed, _ = load_model(args.typed, dev)
    pols = {} if getattr(args, "typed_only", False) else {"heuristic": ("heuristic", None)}
    cfg = PlannerConfig(pop=args.pop, iters=args.iters, n_z=args.n_z)
    if args.lam_effort is not None:
        cfg.lam_effort = args.lam_effort
    if args.lam_unc is not None:
        cfg.lam_unc = args.lam_unc
    if args.c_prog is not None:
        cfg.c_prog = args.c_prog
    pols["planner_typed"] = ("planner", CEMPlanner(typed, "typed", cfg))
    if not args.no_voi_variant:
        pols["planner_typed_novoi"] = ("planner", CEMPlanner(typed, "typed", PlannerConfig(**{**cfg.__dict__, "voi": False})))
    if args.generic:
        g, _ = load_model(args.generic, dev)
        pols["planner_generic"] = ("planner", CEMPlanner(g, "generic", cfg))
    if args.nomem:
        g2, _ = load_model(args.nomem, dev)
        pols["planner_nomem"] = ("planner", CEMPlanner(g2, "nomem", cfg))
    for extra in (args.extra or "").split(","):
        if extra:       # name=path (typed checkpoints, e.g. G5 adapter models)
            nm, pth = extra.split("=")
            pols[f"planner_{nm}"] = ("planner", CEMPlanner(load_model(pth, dev)[0], "typed", cfg))
    if not getattr(args, "typed_only", False):
        pols["planner_oracle(labelled)"] = ("planner", CEMPlanner(typed, "oracle", cfg))
        pols["planner_prior_mean"] = ("planner", CEMPlanner(typed, "prior", cfg))   # value-of-latent baseline
    out = {}
    rows = []
    fams = args.families.split(",") if args.families else Y.FAMILIES
    robots_of = {"val": Y.TRAIN_ROBOTS, "test": Y.TRAIN_ROBOTS, "test_robot": Y.HELDOUT_ROBOTS}
    for split in args.splits.split(","):
        robots = robots_of[split]
        for fam in fams:
            t0 = time.time()
            tasks = [api.TaskSpec(fam, Y.FAMILY_CATEGORY[fam], 5_000_000 + SEED_BASE[split] + i,
                                  robots[i % len(robots)], split, params={"lot": args.lot} if args.lot > 1 else {})
                     for i in range(args.episodes)]
            res = H.paired_eval(Y.SynthEnv, tasks, pols, seed=args.seed)
            summ = H.summarize(res)
            out[f"{split}:{fam}"] = dict(summary=summ, episodes=res)
            for name, sm in summ.items():
                for k, v in sm.items():
                    if isinstance(v, tuple):
                        rows.append(dict(run_id=f"{args.run}-{split}-{fam}", experiment="v2_g3_closed_loop", model=name,
                                         stratum=f"{split}:{fam}", metric=k, value=v[0], ci_lo=v[1], ci_hi=v[2],
                                         n_episodes=sm["n"], status="observed", time_utc=_utc(), seed=args.seed))
                    elif isinstance(v, float):
                        rows.append(dict(run_id=f"{args.run}-{split}-{fam}", experiment="v2_g3_closed_loop", model=name,
                                         stratum=f"{split}:{fam}", metric=k, value=v, n_episodes=sm["n"],
                                         status="observed", time_utc=_utc(), seed=args.seed))
            print(f"== {split} {fam} ({time.time() - t0:.0f}s)", flush=True)
            for name, sm in summ.items():
                print(f"  {name:26s} succ={sm['success'][0]:.2f} [{sm['success'][1]:.2f},{sm['success'][2]:.2f}] "
                      f"items={sm['items_done'][0]:.2f} eff/item={sm['effort_per_item']:.3f} "
                      f"effort={sm['effort'][0]:.3f} (med {sm['effort_median']:.3f}) risk={sm['risk_events'][0]:.2f} "
                      f"steps={sm['steps']:.2f} probes={sm['probes']:.2f}", flush=True)
    write_rows(rows)
    os.makedirs(os.path.join(RESULTS, "closed_loop"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "closed_loop", f"{args.run}.json"), "w"), default=float)


def interventions(args):
    """G4: model-vs-simulator finite differences + closed-loop slot interventions, per family."""
    from physmem.models.v2 import harness as H
    from physmem.planner import CEMPlanner, PlannerConfig
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typed, c = load_model(args.typed, dev)
    out, rows = {}, []
    pl = CEMPlanner(typed, "typed", PlannerConfig(pop=args.pop, iters=args.iters, n_z=8))
    for fam in (args.families.split(",") if args.families else Y.FAMILIES):
        fd = H.intervention_fd(typed, fam, n_states=args.n, seed=args.seed)
        cl = H.closed_loop_intervention(pl, fam, n=max(args.n // 3, 20), seed=args.seed)
        out[fam] = dict(fd=fd, closed_loop=cl)
        print(fam, json.dumps(fd, default=float), json.dumps(cl, default=float), flush=True)
        for ln, r in fd.items():
            for k in ("sign_agree", "corr", "ratio_median", "slope", "mean_abs_sim", "mean_abs_model"):
                rows.append(dict(run_id=f"{args.run}-{fam}", experiment="v2_g4_intervention_fd", model="physmem_v2",
                                 variant="typed", stratum=f"{fam}:{ln}:{r['outcome']}", metric=k, value=r[k],
                                 n=r["n"], n_sig=r["n_sig"], status="observed", time_utc=_utc(),
                                 config_hash=c["cfg_hash"]))
        for k, v in cl.items():
            if isinstance(v, float):
                rows.append(dict(run_id=f"{args.run}-{fam}", experiment="v2_g4_intervention_closed_loop",
                                 model="physmem_v2", variant="typed", stratum=f"{fam}:{cl['latent']}", metric=k,
                                 value=v, n=cl["n"], status="observed", time_utc=_utc(), config_hash=c["cfg_hash"]))
    write_rows(rows)
    os.makedirs(os.path.join(RESULTS, "interventions"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "interventions", f"{args.run}.json"), "w"), default=float, indent=1)


def onpolicy(args):
    """model-based on-policy loop: (train) -> plan in env with the planner (+ exploration) -> add the planner's own
    rollouts incl. failures -> fine-tune -> ... ; per-round rollout success + VAL closed-loop success reported."""
    import shutil
    from physmem.models.v2 import harness as H
    from physmem.models.v2.tensorize import concat_arrays
    from physmem.planner import CEMPlanner, PlannerConfig
    from physmem.v2 import api
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fams = args.families.split(",") if args.families else Y.FAMILIES
    emb = {r: S.embodiment_vector(Y.ROBOTS[r].spec) for r in Y.ROBOTS}
    root = os.path.join(args.work, args.run)
    os.makedirs(root, exist_ok=True)
    ck = args.init
    rollouts = {f: [] for f in fams}
    rows = []
    cfg = PlannerConfig(pop=args.pop, iters=args.iters, n_z=args.n_z, c_prog=2.0, lam_effort=0.03, voi=False)
    for r in range(args.rounds + 1):
        model, c = load_model(ck, dev)
        pl = CEMPlanner(model, c["variant"], cfg)       # typed / generic / nomem: same loop, same budget
        # --- val closed-loop success of the current model (fixed val seeds, same every round)
        for fam in fams:
            tasks = [api.TaskSpec(fam, "", 5_000_000 + SEED_BASE["val"] + i, Y.TRAIN_ROBOTS[i % 4], "val",
                                  params={"lot": args.lot} if args.lot > 1 else {}) for i in range(args.val_episodes)]
            res = H.paired_eval(Y.SynthEnv, tasks, {"planner_typed": ("planner", pl), "heuristic": ("heuristic", None)})
            # (key "planner_typed" = the planner of THIS run's variant; variant recorded in the row)
            sm = H.summarize(res)
            for name in sm:
                for key in ("success", "items_done", "effort"):
                    v = sm[name][key]
                    rows.append(dict(run_id=f"{args.run}-round{r}", experiment="v2_onpolicy", model=name, variant=c["variant"],
                                     stratum=f"val:{fam}", metric=key, value=v[0], ci_lo=v[1], ci_hi=v[2], round=r,
                                     n_episodes=sm[name]["n"], status="observed", time_utc=_utc(), lot=args.lot))
            print(f"[round {r}] val {fam}: typed succ={sm['planner_typed']['success'][0]:.2f} "
                  f"items={sm['planner_typed']['items_done'][0]:.2f} eff={sm['planner_typed']['effort'][0]:.2f} | "
                  f"heuristic succ={sm['heuristic']['success'][0]:.2f} items={sm['heuristic']['items_done'][0]:.2f}",
                  flush=True)
        if r == args.rounds:
            break
        # --- on-policy rollouts on TRAIN-range seeds (disjoint from val/test), with exploration
        for fam in fams:
            succ = []
            for i in range(args.episodes):
                seed = 7_000_000 + r * 100_000 + Y.FAMILIES.index(fam) * 10_000 + i
                task = api.TaskSpec(fam, "", seed, Y.TRAIN_ROBOTS[i % 4], "train",
                                    params={"lot": args.lot} if args.lot > 1 else {})
                recd = []
                e = H.run_episode(Y.SynthEnv(), task, "planner", pl, rng_seed=seed, record=recd, explore=args.explore)
                succ.append(e["success"])
                rollouts[fam].append((fam, task.robot, recd))
            rows.append(dict(run_id=f"{args.run}-round{r}", experiment="v2_onpolicy", model="planner_typed+explore",
                             stratum=f"rollout:{fam}", metric="success", value=float(np.mean(succ)), round=r,
                             n_episodes=len(succ), status="observed", time_utc=_utc()))
            print(f"[round {r}] rollouts {fam}: success {np.mean(succ):.2f} ({len(succ)} eps, "
                  f"{sum(len(x[2]) for x in rollouts[fam])} interactions total)", flush=True)
        # --- data = base train + all rollouts so far ; val/test unchanged
        dr = os.path.join(root, f"data_round{r + 1}")
        os.makedirs(dr, exist_ok=True)
        for fam in fams:
            base = dict(np.load(os.path.join(args.data, f"{fam}_train.npz")))
            ro = build_family_arrays(rollouts[fam], lambda rb: emb[rb], np.random.default_rng(r))
            np.savez(os.path.join(dr, f"{fam}_train.npz"), **concat_arrays([base, ro]))
            for split in ("val", "test", "test_robot"):
                src = os.path.join(args.data, f"{fam}_{split}.npz")
                if os.path.exists(src) and not os.path.exists(os.path.join(dr, f"{fam}_{split}.npz")):
                    os.symlink(os.path.abspath(src), os.path.join(dr, f"{fam}_{split}.npz"))
        run_r = f"{args.run}_round{r + 1}"
        targs = argparse.Namespace(data=dr, variant=c["variant"], run=run_r, steps=args.steps, bs=512, lr=3e-4, d=c["d"],
                                   seed=args.seed, tf_prob=0.5, drop_hist=0.15, holdout_family=None, families=None,
                                   max_episodes=None, init=ck, freeze=None, log_every=500, eval_every=10 ** 9,
                                   no_amp=False, no_test=True, no_phys=not c.get("phys", False))
        ck = train(targs)
    write_rows(rows)
    print("onpolicy done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--data", default="data/v2_synth")
    g.add_argument("--n-train", type=int, default=3000)
    g.add_argument("--n-eval", type=int, default=400)
    g.add_argument("--workers", type=int, default=8)
    t = sp.add_parser("train")
    t.add_argument("--data", default="data/v2_synth")
    t.add_argument("--variant", default="typed")
    t.add_argument("--run", required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--bs", type=int, default=512)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--d", type=int, default=128)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--tf-prob", type=float, default=0.5)
    t.add_argument("--drop-hist", type=float, default=0.15)
    t.add_argument("--holdout-family", default=None)
    t.add_argument("--families", default=None)
    t.add_argument("--max-episodes", type=int, default=None)
    t.add_argument("--init", default=None)
    t.add_argument("--freeze", default=None, help="comma list of param-name prefixes to freeze, e.g. scene,head")
    t.add_argument("--log-every", type=int, default=200)
    t.add_argument("--eval-every", type=int, default=5000)
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--no-phys", action="store_true", help="r1-style free head (no physics margins)")
    t.add_argument("--no-test", action="store_true")
    cb = sp.add_parser("calibrate")
    cb.add_argument("--ckpt", required=True)
    cb.add_argument("--data", default="data/v2_synth")
    e = sp.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--data", default="data/v2_synth")
    e.add_argument("--splits", default="test,test_robot,test_stress")
    op = sp.add_parser("onpolicy")
    op.add_argument("--init", required=True)
    op.add_argument("--data", default="data/v2_synth_r4")
    op.add_argument("--run", required=True)
    op.add_argument("--work", default="data/v2_onpolicy")
    op.add_argument("--rounds", type=int, default=3)
    op.add_argument("--episodes", type=int, default=200)
    op.add_argument("--val-episodes", type=int, default=50)
    op.add_argument("--steps", type=int, default=3000)
    op.add_argument("--explore", type=float, default=0.2)
    op.add_argument("--lot", type=int, default=1)
    op.add_argument("--families", default=None)
    op.add_argument("--pop", type=int, default=192)
    op.add_argument("--iters", type=int, default=3)
    op.add_argument("--n-z", type=int, default=16)
    op.add_argument("--seed", type=int, default=0)
    cv = sp.add_parser("convert")
    cv.add_argument("--pattern", required=True)
    cv.add_argument("--out", required=True)
    cv.add_argument("--workers", type=int, default=8)
    co = sp.add_parser("collect")
    co.add_argument("--data", default="data/v2_env")
    co.add_argument("--families", default=None)
    co.add_argument("--robots", default="probe")
    co.add_argument("--n-train", type=int, default=400)
    co.add_argument("--n-eval", type=int, default=60)
    co.add_argument("--workers", type=int, default=8)
    c = sp.add_parser("closedloop")
    c.add_argument("--typed", required=True)
    c.add_argument("--generic", default=None)
    c.add_argument("--nomem", default=None)
    c.add_argument("--run", required=True)
    c.add_argument("--episodes", type=int, default=100)
    c.add_argument("--families", default=None)
    c.add_argument("--pop", type=int, default=256)
    c.add_argument("--iters", type=int, default=4)
    c.add_argument("--n-z", type=int, default=32)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--splits", default="test,test_robot")
    c.add_argument("--extra", default=None)
    c.add_argument("--lot", type=int, default=1)
    c.add_argument("--no-voi-variant", action="store_true")
    c.add_argument("--lam-effort", type=float, default=None)
    c.add_argument("--lam-unc", type=float, default=None)
    c.add_argument("--c-prog", type=float, default=None)
    c.add_argument("--typed-only", action="store_true")
    iv = sp.add_parser("interv")
    iv.add_argument("--typed", required=True)
    iv.add_argument("--run", required=True)
    iv.add_argument("--n", type=int, default=150)
    iv.add_argument("--families", default=None)
    iv.add_argument("--pop", type=int, default=128)
    iv.add_argument("--iters", type=int, default=3)
    iv.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        generate(a.data, a.n_train, a.n_eval, a.workers)
    elif a.cmd == "train":
        train(a)
    elif a.cmd == "onpolicy":
        onpolicy(a)
    elif a.cmd == "convert":
        convert(a.pattern, a.out, a.workers)
    elif a.cmd == "collect":
        from physmem.envs import FAMILIES as EF
        collect(a.data, a.families.split(",") if a.families else list(EF), a.robots.split(","), a.n_train, a.n_eval,
                a.workers)
    elif a.cmd == "closedloop":
        closed_loop(a)
    elif a.cmd == "interv":
        interventions(a)
    elif a.cmd == "calibrate":
        m, c = load_model(a.ckpt)
        sc = calibrate(m, load_stores(a.data, "val", next(m.parameters()).device))
        print("val variance temperature:", np.round(sc, 3).tolist())
        c["state"] = m.state_dict()
        out = a.ckpt.replace(".pt", "_cal.pt")
        torch.save(c, out)
        run_eval(out, a.data, splits=("test", "test_robot", "test_stress"), oracle_too=False)
    else:
        run_eval(a.ckpt, a.data, splits=tuple(a.splits.split(",")))


if __name__ == "__main__":
    main()
