"""evaluation of compact models + analytic baselines (agent MODEL).

  evaluate_suite("primary", config)
  python -m physmem.evaluate primary [--device cuda] [--strata test,occ_0.5,...] [--diagnostics e05,e07,...]

All statistics are computed per ROOT episode first (mean over its samples), then bootstrapped over
roots (paired across models by root). Rows -> results/metrics.jsonl, summaries -> results/summary.csv.
Models: compact b00/b10/b01/b11 (best-validation checkpoints, 3 seeds), analytic `constvel`, `coulomb`,
and `coulomb_oracle` (ORACLE: given true friction; status/labels mark it, excluded from comparisons).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from physmem.models.analytic import ConstantVelocity, CoulombAnalytic, CoulombOracle
from physmem.models.compact import build_model
from physmem.models.losses import fit_loss_scales, physics_residuals
from physmem.schema import HORIZONS, PRIMARY_HORIZONS, STATE_DIM, EventTensors, Prop, QueryTensors
from physmem.train import RES, ROOT, SplitView, append_jsonl, config_hash, fit_target_scaler, load_dataset, utc

Z = {50: 0.6744897501960817, 80: 1.2815515655446004, 95: 1.959963984540054}
H_IDX = {h: i for i, h in enumerate(HORIZONS)}
PRIMARY_IDX = [H_IDX[h] for h in PRIMARY_HORIZONS]
VARIANTS = ("b00", "b10", "b01", "b11")
EVAL_VERSION = "eval-1"


# ----------------------------------------------------------------------------
# model loading / prediction
# ----------------------------------------------------------------------------


def load_compact(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    hp = ck["hp"]
    m = build_model(ck["variant"], d=hp.get("width", 128), passes_train=hp.get("train_passes", (1, 2, 4)),
                    n_blocks=hp.get("shared_blocks", 2)).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    m.ckpt_meta = dict(path=str(path), step=ck.get("step"), config_hash=ck.get("config_hash"), seed=ck.get("seed"),
                       val=ck.get("val"))
    return m


def find_checkpoints(tag="main"):
    out = {}
    for p in sorted(glob.glob(str(RES / "checkpoints" / f"compact-{tag}-*-s*.pt"))):
        name = Path(p).stem
        if "_" in name:  # _final / _atXX
            continue
        parts = name.split("-")
        variant, seed = parts[-2], int(parts[-1][1:])
        out[(variant, seed)] = p
    return out


@torch.no_grad()
def predict(model, ds: SplitView, passes=4, bs=256, idx=None, edit=None, lesion=None, bf16=False):
    """returns dict of numpy arrays: mean (n,H,7), logvar, fr_mean, fr_logvar, contact (n,H,4) or None."""
    outs = defaultdict(list)
    is_compact = hasattr(model, "encode")
    for ev, q, lab in ds.iterate(bs, idx=idx):
        if edit is not None:
            ev, q = edit(ev, q)
        if is_compact:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and ds.device.type == "cuda"):
                if lesion is None:
                    o = model(ev, q, passes=passes)
                else:
                    o = lesion(model, ev, q, passes)
        else:
            o = model(ev, q, labels=lab)
        outs["mean"].append(o.mean.float().cpu())
        outs["logvar"].append(o.logvar.float().cpu())
        if o.friction_mean is not None:
            outs["fr_mean"].append(o.friction_mean.float().cpu().reshape(-1))
            outs["fr_logvar"].append(o.friction_logvar.float().cpu().reshape(-1))
        if o.contact_logits is not None:
            outs["contact"].append(o.contact_logits.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in outs.items()}


def labels_np(ds: SplitView, idx=None):
    t = ds.ps.t
    sel = (lambda x: x) if idx is None else (lambda x: x[torch.as_tensor(idx, device=x.device)])
    g = lambda k: sel(t[k]).cpu().numpy()
    return dict(future=g("future").astype(np.float64), future_mask=g("future_mask"), friction=g("friction"),
                mass=g("mass"), contact_mode=g("contact_mode"), free_slide=g("free_slide_mask"), law=g("law"),
                root=g("root"), k=g("k"))


# ----------------------------------------------------------------------------
# per-sample metrics
# ----------------------------------------------------------------------------


def sample_metrics(pred, lab, y_mean, y_std, loss_scales):
    """dict name -> (n,H) or (n,) arrays (nan where undefined)."""
    mu, lv = pred["mean"].astype(np.float64), pred["logvar"].astype(np.float64)
    y = lab["future"]
    fm = lab["future_mask"].astype(bool)
    ys, ym = y_std[None], y_mean[None]
    zn = (y - mu) / np.sqrt(np.exp(lv))
    yn, mun = (y - ym) / ys, (mu - ym) / ys
    lvn = lv - 2 * np.log(ys)
    out = {}
    nanm = lambda a: np.where(fm, a, np.nan)
    out["pos_err_m"] = nanm(np.linalg.norm(mu[..., 0:2] - y[..., 0:2], axis=-1))
    ya = np.arctan2(y[..., 3], y[..., 2])
    pa = np.arctan2(mu[..., 3], mu[..., 2])
    out["yaw_err_rad"] = nanm(np.abs((pa - ya + np.pi) % (2 * np.pi) - np.pi))
    out["vel_err_mps"] = nanm(np.linalg.norm(mu[..., 4:6] - y[..., 4:6], axis=-1))
    out["wz_err_radps"] = nanm(np.abs(mu[..., 6] - y[..., 6]))
    out["nmse"] = nanm(((mun - yn) ** 2).mean(-1))
    out["nll"] = nanm((0.5 * (lvn + (yn - mun) ** 2 * np.exp(-lvn) + math.log(2 * math.pi))).mean(-1))
    for c, z in Z.items():
        out[f"cover{c}"] = nanm((np.abs(zn) <= z).mean(-1))
    out["sharp_nstd"] = nanm(np.exp(0.5 * lvn).mean(-1))
    out["pos_std_m"] = nanm(np.sqrt(np.exp(lv[..., 0]) + np.exp(lv[..., 1])))
    if "fr_mean" in pred:
        out["friction_abs_err"] = np.abs(pred["fr_mean"] - lab["friction"])
        out["friction_std"] = np.exp(0.5 * pred["fr_logvar"])
        out["friction_z2"] = (pred["fr_mean"] - lab["friction"]) ** 2 / np.exp(pred["fr_logvar"])
    if "contact" in pred:
        out["contact_acc"] = nanm((pred["contact"].argmax(-1) == lab["contact_mode"]).astype(float))
    if "fr_mean" in pred:
        tmu = torch.as_tensor(mu)
        hz = torch.as_tensor(np.tile(np.array(HORIZONS), (len(mu), 1)))
        coul, cm, kin, km = physics_residuals(tmu, torch.as_tensor(pred["fr_mean"].astype(np.float64)), hz,
                                              torch.as_tensor(lab["free_slide"]), torch.as_tensor(fm))
        c = (coul.abs() / loss_scales["coulomb"]).numpy()
        out["coulomb_res"] = np.where(cm.numpy(), c, np.nan)
        k = (kin.norm(dim=-1) / loss_scales["kinematic"]).numpy()
        out["kinematic_res"] = np.where(km.numpy(), k, np.nan)
    return out


def root_means(arr, roots):
    """arr (n,) -> (unique_roots, per-root nanmean)."""
    ur, inv = np.unique(roots, return_inverse=True)
    s = np.zeros(len(ur))
    c = np.zeros(len(ur))
    ok = ~np.isnan(arr)
    np.add.at(s, inv[ok], arr[ok])
    np.add.at(c, inv[ok], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ur, np.where(c > 0, s / np.maximum(c, 1), np.nan)


def bootstrap_ci(per_root, n_boot=2000, seed=0, ci=0.95):
    x = per_root[~np.isnan(per_root)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bs = x[idx].mean(1)
    a = (1 - ci) / 2
    return float(x.mean()), float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a)), len(x)


def reduce_metric(name, arr):
    """(n,H) -> dict horizon_label -> (n,) ; (n,) -> {'-': arr}"""
    if arr.ndim == 1:
        return {"-": arr}
    H = arr.shape[1]
    if H == len(HORIZONS):
        d = {f"h{HORIZONS[i]}": arr[:, i] for i in PRIMARY_IDX}
    else:  # residual pairs
        d = {}
    with np.errstate(invalid="ignore"):
        d["all"] = np.nanmean(arr, axis=1) if np.isfinite(arr).any() else np.full(arr.shape[0], np.nan)
    return d


# ----------------------------------------------------------------------------
# rows
# ----------------------------------------------------------------------------


class Writer:
    def __init__(self, n_boot=2000, chash="", dry=False, tag="main", exp_override=None):
        self.tag = tag
        self.exp_override = exp_override
        self.n_boot = n_boot
        self.chash = chash
        self.rows = []
        self.dry = dry

    def row(self, experiment, model, variant, seed, stratum, metric, value, n_roots, status="observed", ci=None, **extra):
        if self.exp_override == "fp":  # post-hoc b11p push-summary test: primary -> e02fp, eNN -> eNNfp
            experiment = "e02fp" if experiment == "primary" else experiment + "fp"
            extra = {**extra, "post_hoc": True}
        elif self.exp_override == "x":  # post-hoc exploratory (v3x): primary -> e02x, eNN -> eNNx
            experiment = "e02x" if experiment == "primary" else experiment + "x"
            extra = {**extra, "post_hoc": True}
        elif self.exp_override:
            experiment = self.exp_override
        r = dict(run_id=f"eval-{self.tag}-{experiment}-{model}-{variant}-s{seed}-{stratum}", tag=self.tag, experiment=experiment, model=model, variant=variant, seed=seed,
                 stratum=stratum, metric=metric, value=None if value is None or (isinstance(value, float) and math.isnan(value)) else value,
                 n_roots=int(n_roots), status=status, config_hash=self.chash, time_utc=utc(), eval_version=EVAL_VERSION)
        if ci is not None:
            r["ci_lo"], r["ci_hi"] = ci
        r.update(extra)
        self.rows.append(r)
        if not self.dry:
            append_jsonl(RES / "metrics.jsonl", r)
        return r

    def metric_rows(self, experiment, model, variant, seed, stratum, met, roots, status="observed", prefix="", **extra):
        per_root = {}
        for name, arr in met.items():
            for hl, a in reduce_metric(name, arr).items():
                ur, pr = root_means(a, roots)
                mean, lo, hi, n = bootstrap_ci(pr, self.n_boot)
                key = f"{prefix}{name}@{hl}" if hl != "-" else f"{prefix}{name}"
                self.row(experiment, model, variant, seed, stratum, key, mean, n, status, ci=(lo, hi), **extra)
                per_root[key] = (ur, pr)
        return per_root


def write_summary(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["experiment", "stratum", "model", "variant", "seed", "metric", "value", "ci_lo", "ci_hi", "n_roots", "status", "config_hash"]
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------------
# paired comparisons / factorial
# ----------------------------------------------------------------------------


def paired_diff(per_root_a, per_root_b, n_boot=2000):
    ua, va = per_root_a
    ub, vb = per_root_b
    common, ia, ib = np.intersect1d(ua, ub, return_indices=True)
    return bootstrap_ci(va[ia] - vb[ib], n_boot)


def factorial(per_root_by_variant_seed, metric, n_boot=2000):
    """per_root_by_variant_seed[(variant, seed)][metric] = (roots, values). seed-average per root, then
    main effects of structure (first digit) and physics (second) and interaction u11-u10-u01+u00."""
    avg = {}
    for v in VARIANTS:
        seeds = [s for (vv, s) in per_root_by_variant_seed if vv == v and metric in per_root_by_variant_seed[(vv, s)]]
        if not seeds:
            return None
        rs = [per_root_by_variant_seed[(v, s)][metric] for s in seeds]
        common = rs[0][0]
        for r in rs[1:]:
            common = np.intersect1d(common, r[0])
        vals = np.stack([r[1][np.searchsorted(r[0], common)] for r in rs])
        avg[v] = (common, np.nanmean(vals, 0), len(seeds))
    common = avg["b00"][0]
    for v in VARIANTS[1:]:
        common = np.intersect1d(common, avg[v][0])
    u = {v: avg[v][1][np.searchsorted(avg[v][0], common)] for v in VARIANTS}
    eff = {
        "structure_main": 0.5 * ((u["b11"] - u["b01"]) + (u["b10"] - u["b00"])),
        "physics_main": 0.5 * ((u["b11"] - u["b10"]) + (u["b01"] - u["b00"])),
        "interaction": u["b11"] - u["b10"] - u["b01"] + u["b00"],
    }
    return {k: bootstrap_ci(v, n_boot) for k, v in eff.items()}, {v: avg[v][2] for v in VARIANTS}


# ----------------------------------------------------------------------------
# event edits (e05 / e14) operating on EventTensors batches (actor-side evidence only)
# ----------------------------------------------------------------------------


def _append_events(ev: EventTensors, new: dict):
    """append per-sample new events (dict of (B,K,...) tensors with key 'mask'); returns new EventTensors."""
    cat = lambda a, b: torch.cat([a, b.to(a.dtype)], 1)
    return EventTensors(
        entity=cat(ev.entity, new["entity"]), related=cat(ev.related, new["related"]), prop=cat(ev.prop, new["prop"]),
        value=cat(ev.value, new["value"]), logvar=cat(ev.logvar, new["logvar"]), age=cat(ev.age, new["age"]),
        latency=cat(ev.latency, new["latency"]), source=cat(ev.source, new["source"]),
        provenance=cat(ev.provenance, new["provenance"]), mask=cat(ev.mask, new["mask"]),
        ent_type=ev.ent_type, ent_mask=ev.ent_mask, dt_obs=ev.dt_obs)


def _select_events(ev, sel):
    """sel (B,E) bool -> dict of the selected events packed into (B,K) (K = max count)."""
    B, E = sel.shape
    K = int(sel.sum(1).max().item()) if sel.any() else 1
    order = torch.argsort((~sel).to(torch.int8), dim=1, stable=True)[:, :K]
    g = lambda x: torch.gather(x, 1, order) if x.dim() == 2 else torch.gather(x, 1, order[..., None].expand(-1, -1, x.shape[-1]))
    d = {k: g(getattr(ev, k)) for k in ("entity", "related", "prop", "value", "logvar", "age", "latency", "source", "provenance")}
    d["mask"] = torch.gather(sel, 1, order)
    return d


def _free_source(ev):
    used = torch.where(ev.mask, ev.source.long(), torch.zeros_like(ev.source.long()))
    return used.max(1).values + 1


def edit_duplicate_flood(ev, q, reps=4):
    """exact duplicates of the target's 10 newest pose events, x reps (no new information)."""
    tgt = q.target[:, None]
    tp = ev.mask & (ev.entity == tgt) & (ev.prop == int(Prop.POSE2))
    age = torch.where(tp, ev.age, torch.full_like(ev.age, 1e9))
    thr = torch.sort(age, 1).values[:, 9:10]
    sel = tp & (ev.age <= thr)
    d = _select_events(ev, sel)
    for _ in range(reps):
        ev = _append_events(ev, d)
    return ev, q


def edit_stale_flood(ev, q, reps=3, shift=1.5):
    """stale writes: copies of all target pose events re-timestamped `shift` s OLDER (saturation)."""
    tgt = q.target[:, None]
    sel = ev.mask & (ev.entity == tgt) & (ev.prop == int(Prop.POSE2))
    d = _select_events(ev, sel)
    for r in range(reps):
        dd = dict(d)
        dd["age"] = d["age"] + shift * (r + 1)
        ev = _append_events(ev, dd)
    return ev, q


def edit_conflicting_source(ev, q, offset=0.02, var_mult=25.0):
    """a second, 5x noisier source reports the target pose offset by `offset` m."""
    tgt = q.target[:, None]
    sel = ev.mask & (ev.entity == tgt) & (ev.prop == int(Prop.POSE2))
    d = _select_events(ev, sel)
    d["source"] = _free_source(ev)[:, None].expand_as(d["source"]).to(d["source"].dtype)
    v = d["value"].clone()
    v[..., 0] += offset
    v[..., 1] += offset
    d["value"] = v
    lv = d["logvar"].clone()
    lv[..., :4] += math.log(var_mult)
    d["logvar"] = lv
    return _append_events(ev, d), q


def edit_authoritative_correction(ev, q, offset=0.03):
    """old target pose evidence (age > 0.5 s) is corrupted by an offset; a fresh low-variance ESTIMATED
    correction at age 0 restates the latest pose. A memory honoring the newest authoritative write
    should be ~unaffected for the future (which depends on the current pose)."""
    tgt = q.target[:, None]
    tp = ev.mask & (ev.entity == tgt) & (ev.prop == int(Prop.POSE2))
    old = tp & (ev.age > 0.5)
    v = ev.value.clone()
    v[..., 0] = torch.where(old, v[..., 0] + offset, v[..., 0])
    ev2 = EventTensors(**{**ev.__dict__, "value": v})
    age = torch.where(tp, ev.age, torch.full_like(ev.age, 1e9))
    newest = tp & (ev.age <= age.min(1, keepdim=True).values)
    d = _select_events(ev2, newest)
    d = {k: v_[:, :1] for k, v_ in d.items()}
    d["age"] = torch.zeros_like(d["age"])
    d["provenance"] = torch.full_like(d["provenance"], 2)  # ESTIMATED
    d["logvar"] = torch.where(d["logvar"] != 0, d["logvar"] - 2.0, d["logvar"])
    d["source"] = _free_source(ev)[:, None].to(d["source"].dtype)
    return _append_events(ev2, d), q


def edit_delete_velocity(ev, q):
    """deletion: all derived VEL2 events of the target are removed (masked)."""
    tgt = q.target[:, None]
    dm = (ev.entity == tgt) & (ev.prop == int(Prop.VEL2))
    return EventTensors(**{**ev.__dict__, "mask": ev.mask & ~dm}), q


def edit_add_distractor(ev, q, n_events=40, dist=0.4):
    """adds a static distractor OBJECT entity (needs a free entity slot) with pose + size events."""
    B, N = ev.ent_type.shape
    free = ~ev.ent_mask
    has = free.any(1)
    slot = torch.argmax(free.to(torch.int8), 1)
    et = ev.ent_type.clone()
    em = ev.ent_mask.clone()
    bi = torch.arange(B, device=et.device)
    et[bi[has], slot[has]] = 0
    em[bi[has], slot[has]] = True
    K = n_events + 1
    dev = ev.value.device
    ang = torch.rand(B, generator=torch.Generator(device="cpu").manual_seed(7)).to(dev) * 2 * math.pi
    val = torch.zeros(B, K, ev.value.shape[-1], device=dev, dtype=ev.value.dtype)
    val[:, :n_events, 0] = (dist * torch.cos(ang))[:, None]
    val[:, :n_events, 1] = (dist * torch.sin(ang))[:, None]
    val[:, :n_events, 2] = 1.0
    val[:, n_events, :3] = 0.03
    lv = torch.zeros_like(val)
    lv[:, :n_events, :4] = math.log(0.002**2)
    lv[:, n_events, :3] = math.log(0.002**2)
    prop = torch.full((B, K), int(Prop.POSE2), device=dev, dtype=ev.prop.dtype)
    prop[:, n_events] = int(Prop.SIZE)
    age = torch.linspace(0, 2.0, n_events, device=dev)[None].expand(B, -1)
    age = torch.cat([age, torch.full((B, 1), 0.3, device=dev)], 1)
    src = _free_source(ev)[:, None].expand(B, K)
    d = dict(entity=slot[:, None].expand(B, K), related=torch.full((B, K), -1, device=dev), prop=prop, value=val, logvar=lv,
             age=age, latency=torch.full((B, K), 0.002, device=dev), source=src, provenance=torch.zeros(B, K, dtype=torch.long, device=dev),
             mask=has[:, None].expand(B, K))
    ev2 = _append_events(ev, d)
    ev2 = EventTensors(**{**ev2.__dict__, "ent_type": et, "ent_mask": em})
    return ev2, q


E05_EDITS = {
    "duplicate_flood": edit_duplicate_flood,
    "stale_flood": edit_stale_flood,
    "conflicting_source": edit_conflicting_source,
    "authoritative_correction": edit_authoritative_correction,
    "delete_velocity": edit_delete_velocity,
}


# ----------------------------------------------------------------------------
# lesions (e11)
# ----------------------------------------------------------------------------


def lesion_typed_token(prop, random=False, seed=0):
    """mask the typed slot token for `prop` from core attention + readout (targeted), or a random valid
    non-typed token per sample (matched count, random=True)."""

    def f(model, ev, q, passes):
        t, mk, s = model.encode(ev, q)
        mk = mk.clone()
        if not random:
            mk[:, s["typed_index"][prop]] = False
        else:
            g = torch.Generator(device="cpu").manual_seed(seed)
            cand = mk & (s["token_kind"] <= 2)
            r = torch.rand(cand.shape, generator=g).to(cand.device)
            r = torch.where(cand, r, torch.full_like(r, -1))
            j = r.argmax(1)
            mk[torch.arange(len(j), device=mk.device), j] = False
        t = model.refine(t, mk, passes)
        return model.readout(t, mk, q, s)

    return f


def lesion_edge(kind="target_pusher", seed=0):
    """cut relational edges (structured encoder): target<->pusher (targeted) or one random other valid pair."""

    def f(model, ev, q, passes):
        B, N = ev.ent_type.shape
        les = torch.zeros(B, N, N, dtype=torch.bool, device=ev.ent_type.device)
        bi = torch.arange(B, device=les.device)
        if kind == "target_pusher":
            pus = torch.argmax(((ev.ent_type == 1) & ev.ent_mask).to(torch.int8), 1)
            les[bi, q.target, pus] = True
            les[bi, pus, q.target] = True
        else:
            g = torch.Generator(device="cpu").manual_seed(seed)
            valid = ev.ent_mask[:, :, None] & ev.ent_mask[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=les.device)[None]
            pus = torch.argmax(((ev.ent_type == 1) & ev.ent_mask).to(torch.int8), 1)
            tp = torch.zeros_like(valid)
            tp[bi, q.target, pus] = True
            tp[bi, pus, q.target] = True
            cand = valid & ~tp
            r = torch.rand(cand.shape, generator=g).to(cand.device)
            r = torch.where(cand, r, torch.full_like(r, -1)).view(B, -1)
            j = r.argmax(1)
            ok = cand.view(B, -1)[bi, j]
            i1, i2 = j // N, j % N
            les[bi[ok], i1[ok], i2[ok]] = True
            les[bi[ok], i2[ok], i1[ok]] = True
        t, mk, s = model.encode(ev, q, edge_lesion=les)
        t = model.refine(t, mk, passes)
        return model.readout(t, mk, q, s)

    return f


# ----------------------------------------------------------------------------
# e07: interventions vs simulator finite differences
# ----------------------------------------------------------------------------


def sim_counterfactual(stratum, root_ids, dataset, prop, values, procs=8):
    """re-simulate k=0 of each root with prop (friction|mass) set to each of `values`; returns
    future labels (n_roots, len(values), H, 7) in the same canonical frame as the stored sample."""
    import multiprocessing as mp

    from physmem import data as D

    path = os.path.join(os.path.expanduser(D.DATA_DIR), dataset, f"{stratum}_params.jsonl")
    params = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            params[r["root"]] = r["params"]
    jobs = [(params[int(r)], prop, float(v)) for r in root_ids for v in values]
    ctx = mp.get_context("fork")
    with ctx.Pool(procs) as pool:
        res = pool.map(_cf_worker, jobs, chunksize=4)
    return np.array(res).reshape(len(root_ids), len(values), len(HORIZONS), STATE_DIM)


def _ep_from_json(pj):
    from physmem import worlds as W

    pj = dict(pj)
    acts = pj.pop("actions")
    for k in ("half", "pos", "drift", "pusher_park", "anchor"):
        if k in pj and isinstance(pj[k], list):
            pj[k] = tuple(pj[k])
    ep = W.EpisodeParams(**pj)
    ep.actions = [W.PushAction(*a) for a in acts] if ep.world != "w1" else [tuple(a) for a in acts]
    return ep


def _cf_worker(job):
    from physmem import data as D
    from physmem import worlds as W

    pj, prop, val = job
    ep = _ep_from_json(pj)
    setattr(ep, prop, val)
    traj = W.simulate(ep)
    # k=0: object at rest at its initial pose before the first push; canonical frame = TRUE initial pose
    # (the stored sample used the noisy estimate; we compare DIFFERENCES between counterfactuals, in which the
    # frame offset cancels to first order)
    est = (ep.pos[0], ep.pos[1], ep.yaw)
    lab = D.make_labels_np(traj, ep, 0, est)
    return lab["future"]


# ----------------------------------------------------------------------------
# suite
# ----------------------------------------------------------------------------

DEFAULT_STRATA = ["test", "occ_0.0", "occ_0.2", "occ_0.5", "occ_1.0", "delay_0", "delay_1", "delay_3", "drop_0.0", "drop_0.25",
                  "drop_0.5", "rate_10", "rate_20", "rate_40", "heldout_cells", "ood_fric_0.15", "ood_fric_0.9", "paddle",
                  "misspec_velfric", "misspec_drift", "w3", "w1", "agent_k0", "agent_klate", "chain_L1", "chain_L2", "chain_L4",
                  "chain_L8"]
CHAIN_STRATA = ["chain_L1", "chain_L2", "chain_L4", "chain_L8"]
STRATUM_EXPERIMENT = {
    "test": "primary", **{s: "e03" for s in ("occ_0.0", "occ_0.2", "occ_0.5", "occ_1.0", "delay_0", "delay_1", "delay_3", "drop_0.0",
                                             "drop_0.25", "drop_0.5", "rate_10", "rate_20", "rate_40")},
    "heldout_cells": "e04", "ood_fric_0.15": "e04", "ood_fric_0.9": "e04", "paddle": "e04",
    "misspec_velfric": "e06", "misspec_drift": "e06", "w3": "e14", "w1": "e04",
    "agent_k0": "e16", "agent_klate": "e16", **{c: "e10" for c in ("chain_L1", "chain_L2", "chain_L4", "chain_L8")},
}


def _available(stratum, dataset):
    from physmem import data as D

    base = getattr(D, "VIEWS", {}).get(stratum, (stratum, None))[0]
    return os.path.exists(os.path.join(os.path.expanduser(D.DATA_DIR), dataset, f"{base}_tensors.pt"))


def measure_latency(model, ds, passes=4, n=50, device="cuda"):
    ts = []
    is_compact = hasattr(model, "encode")
    for i in range(n + 5):
        ev, q, lab = ds.batch(torch.tensor([i % ds.n]))
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            o = model(ev, q, passes=passes) if is_compact else model(ev, q, labels=lab)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= 5:
            ts.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))


def evaluate_suite(suite="primary", config=None, device=None, strata=None, diagnostics=None, tag="main", n_boot=None,
                   max_samples=None, dry=False, parts="all"):
    """parts: 'analytic' (CPU baselines only; caches per-root values), 'compact' (compact models only; loads the
    analytic cache for paired comparisons) or 'all'."""
    import pickle

    do_a, do_c = parts in ("all", "analytic"), parts in ("all", "compact")
    config = config or {}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = config.get("compact", {}).get("dataset", "v1")
    n_boot = n_boot or config.get("stats", {}).get("bootstrap_resamples", 2000)
    ev_cfg = dict(suite=suite, dataset=dataset, tag=tag, n_boot=n_boot, version=EVAL_VERSION)
    chash = config_hash(ev_cfg)
    W = Writer(n_boot, chash, dry=dry, tag=tag, exp_override="e17" if tag.startswith("e17") else ("fp" if tag.startswith("fpx") else ("x" if tag.endswith("x") else None)))
    strata = [s for s in (strata or DEFAULT_STRATA) if _available(s, dataset)]
    diagnostics = diagnostics if diagnostics is not None else ["e05", "e07", "e10", "e10chain", "e11", "e14", "e16", "latency"]
    t0 = time.time()
    print(f"[eval] strata={strata} diagnostics={diagnostics}", flush=True)
    # TRAIN scaler + loss scales (same for every model)
    train = load_dataset("train", config, device)
    y_mean, y_std = [x.numpy().astype(np.float64) for x in fit_target_scaler(train)]
    scales = fit_loss_scales(train.iterate(1024))
    analytic = {"constvel": ConstantVelocity(), "coulomb": CoulombAnalytic(), "coulomb_oracle": CoulombOracle()}
    cal_idx = torch.randperm(train.n, generator=torch.Generator().manual_seed(0))[:2048]
    cal = list(train.iterate(256, idx=cal_idx.to(train.device)))
    for m in analytic.values():
        m.calibrate(cal)
    del train, cal
    ckpts = find_checkpoints(tag)
    print(f"[eval] checkpoints: {sorted(ckpts)}", flush=True)
    models = {k: load_compact(p, device) for k, p in ckpts.items()} if do_c else {}
    cache_path = RES / "eval_cache" / f"analytic_{dataset}_{suite}.pkl"
    a_cache = {}
    if not do_a and cache_path.exists():
        a_cache = pickle.load(open(cache_path, "rb"))
    per_root = {}  # (stratum, model_key) -> {metric: (roots, vals)}
    summary_rows = []
    for st in strata:
        ds = load_dataset(st, config, device)
        idx = None
        if max_samples and ds.n > max_samples:
            idx = torch.arange(max_samples, device=ds.device)
        lab = labels_np(ds, None if idx is None else idx.cpu())
        exp = STRATUM_EXPERIMENT.get(st, "e03")
        for name, am in (analytic.items() if do_a else []):
            pred = predict(am, ds, idx=idx, bs=512)
            met = sample_metrics(pred, lab, y_mean, y_std, scales)
            status = "observed"
            pr = W.metric_rows(exp, name, "oracle" if am.oracle else "analytic", -1, st, met, lab["root"], status,
                               oracle=am.oracle)
            per_root[(st, name, -1)] = pr
            a_cache[(st, name, -1)] = pr
        if not do_a:
            for key, val in a_cache.items():
                if key[0] == st:
                    per_root[key] = val
        for (v, s), m in sorted(models.items()):
            pred = predict(m, ds, passes=4, idx=idx)
            met = sample_metrics(pred, lab, y_mean, y_std, scales)
            per_root[(st, v, s)] = W.metric_rows(exp, "compact", v, s, st, met, lab["root"], passes=4,
                                                 ckpt_step=m.ckpt_meta["step"])
            if st == "test" and "e16" in diagnostics:
                for kk in range(4):
                    sel = lab["k"] == kk
                    sub = {n_: a[sel] for n_, a in met.items() if n_ in ("friction_abs_err", "friction_std", "sharp_nstd", "pos_err_m", "nll", "cover80")}
                    W.metric_rows("e16", "compact", v, s, f"test_k{kk}", sub, lab["root"][sel])
            if st in ("heldout_cells", "ood_fric_0.15", "ood_fric_0.9"):
                for kk in (0, 1, 3):
                    sel = lab["k"] == kk
                    sub = {n_: a[sel] for n_, a in met.items() if n_ in ("friction_abs_err", "friction_std", "sharp_nstd", "nll")}
                    W.metric_rows("e04", "compact", v, s, f"{st}_k{kk}", sub, lab["root"][sel])
        if st == "test" and "e16" in diagnostics and do_a:
            for name in ("coulomb", "coulomb_oracle"):
                pred = predict(analytic[name], ds, idx=idx, bs=512)
                met = sample_metrics(pred, lab, y_mean, y_std, scales)
                for kk in range(4):
                    sel = lab["k"] == kk
                    sub = {n_: a[sel] for n_, a in met.items() if n_ in ("friction_abs_err", "friction_std", "pos_err_m", "nll")}
                    W.metric_rows("e16", name, "oracle" if analytic[name].oracle else "analytic", -1, f"test_k{kk}", sub,
                                  lab["root"][sel], oracle=analytic[name].oracle)
        # factorial + paired comparisons on this stratum
        keys = [(v, s) for (stt, v, s) in per_root if stt == st and v in VARIANTS]
        by_vs = {(v, s): per_root[(st, v, s)] for (v, s) in keys}
        for metric in ([] if not do_c else  ("nll@all", "nmse@all", "pos_err_m@h0.5", "pos_err_m@h1.0", "pos_err_m@all", "yaw_err_rad@all", "vel_err_mps@all",
                       "cover80@all", "friction_abs_err", "coulomb_res@all")):
            if not by_vs:
                continue
            fx = factorial(by_vs, metric, n_boot)
            if fx is None:
                W.row(exp, "compact", "factorial", -1, st, f"factorial:{metric}", None, 0, "untested")
                continue
            eff, nseeds = fx
            for en, (mean, lo, hi, n) in eff.items():
                W.row(exp, "compact", "factorial", -1, st, f"{en}:{metric}", mean, n, ci=(lo, hi), n_seeds=nseeds)
            # each variant (seed-averaged) vs analytic coulomb, paired by root
            for v in VARIANTS:
                seeds = [s for (vv, s) in by_vs if vv == v]
                if not seeds or metric not in per_root.get((st, "coulomb", -1), {}):
                    continue
                rs = [by_vs[(v, s)][metric] for s in seeds]
                ur = rs[0][0]
                vals = np.nanmean(np.stack([r[1] for r in rs]), 0)
                mean, lo, hi, n = paired_diff((ur, vals), per_root[(st, "coulomb", -1)][metric], n_boot)
                W.row(exp, "compact", v, -1, st, f"diff_vs_coulomb:{metric}", mean, n, ci=(lo, hi), n_seeds=len(seeds))
        print(f"[eval] {st} done t={time.time()-t0:.0f}s", flush=True)
        del ds
    # ---------------- diagnostics on test
    test = load_dataset("test", config, device) if _available("test", dataset) else None
    if test is not None:
        lab = labels_np(test)
        nr = len(np.unique(lab["root"]))
        for (v, s), m in (sorted(models.items()) if "latency" in diagnostics else []):
            for k in (1, 4, 8):
                p50, p95 = measure_latency(m, test, passes=k, device=str(device))
                W.row("primary", "compact", v, s, "test", f"latency_ms_p50@passes{k}", p50, nr, batch=1, device=str(device))
                W.row("primary", "compact", v, s, "test", f"latency_ms_p95@passes{k}", p95, nr, batch=1, device=str(device))
        for name, am in (analytic.items() if do_a and "latency" in diagnostics else []):
            p50, p95 = measure_latency(am, test, device=str(device))
            W.row("primary", name, "oracle" if am.oracle else "analytic", -1, "test", "latency_ms_p50", p50, nr, batch=1, device="cpu-numpy")
            W.row("primary", name, "oracle" if am.oracle else "analytic", -1, "test", "latency_ms_p95", p95, nr, batch=1, device="cpu-numpy")
        if "e10" in diagnostics:
            for (v, s), m in sorted(models.items()):
                for k in (1, 2, 4, 8):
                    pred = predict(m, test, passes=k)
                    met = sample_metrics(pred, lab, y_mean, y_std, scales)
                    met = {n_: a for n_, a in met.items() if n_ in ("nll", "nmse", "pos_err_m", "cover80", "friction_abs_err")}
                    W.metric_rows("e10", "compact", v, s, f"test_passes{k}", met, lab["root"], passes=k,
                                  status="observed", extrapolation=(k == 8))
                    # dependency length proxy: history length (push index k)
                    for kk in range(4):
                        sel = lab["k"] == kk
                        W.metric_rows("e10", "compact", v, s, f"test_passes{k}_k{kk}", {"nll": met["nll"][sel], "pos_err_m": met["pos_err_m"][sel]},
                                      lab["root"][sel], passes=k)
        if "e10chain" in diagnostics:
            # passes x dependency length (chain strata: target = last box of an L-chain)
            for cs in (CHAIN_STRATA if models else []):
                if not _available(cs, dataset):
                    W.row("e10", "compact", "all", -1, cs, "passes_x_length", None, 0, "untested", note="stratum not generated")
                    continue
                cds = load_dataset(cs, config, device)
                clab = labels_np(cds)
                for (v, s), m in sorted(models.items()):
                    for k in (1, 2, 4, 8):
                        pred = predict(m, cds, passes=k, bs=128)
                        met = sample_metrics(pred, clab, y_mean, y_std, scales)
                        met = {n_: a for n_, a in met.items() if n_ in ("nll", "nmse", "pos_err_m", "cover80")}
                        W.metric_rows("e10", "compact", v, s, f"{cs}_passes{k}", met, clab["root"], passes=k, extrapolation=(k == 8),
                                      chain_L=int(cs.split("L")[1]))
                del cds
        if "e05" in diagnostics:
            cands = [(("compact", v, s), m) for (v, s), m in sorted(models.items())] + \
                    ([((name, "analytic", -1), analytic[name]) for name in ("constvel", "coulomb")] if do_a else [])
            for (mname, v, s), m in cands:
                base_pred = predict(m, test, passes=4, bs=256)
                base = sample_metrics(base_pred, lab, y_mean, y_std, scales)
                for en, fn in E05_EDITS.items():
                    pred = predict(m, test, passes=4, edit=fn, bs=128)
                    met = sample_metrics(pred, lab, y_mean, y_std, scales)
                    d = {f"delta_{n_}": met[n_] - base[n_] for n_ in ("nll", "pos_err_m", "friction_abs_err") if n_ in met}
                    d["pred_change_pos_m"] = np.linalg.norm(pred["mean"][..., :2] - base_pred["mean"][..., :2], axis=-1)
                    W.metric_rows("e05", mname, v, s, f"test_{en}", d, lab["root"], edit=en)
        if "e14" in diagnostics:
            for (v, s), m in sorted(models.items()):
                base = sample_metrics(predict(m, test, passes=4), lab, y_mean, y_std, scales)
                pred = predict(m, test, passes=4, edit=edit_add_distractor, bs=128)
                met = sample_metrics(pred, lab, y_mean, y_std, scales)
                d = {f"delta_{n_}": met[n_] - base[n_] for n_ in ("nll", "pos_err_m") if n_ in met}
                W.metric_rows("e14", "compact", v, s, "test_add_distractor", d, lab["root"], edit="add_distractor")
        if "e11" in diagnostics:
            for (v, s), m in sorted(models.items()):
                if v not in ("b10", "b11", "b11p"):
                    W.row("e11", "compact", v, s, "test", "lesion", None, 0, "skipped", note="generic model has no typed slots/edges")
                    continue
                base_pred = predict(m, test, passes=4)
                base = sample_metrics(base_pred, lab, y_mean, y_std, scales)
                for ln, lf in (("typed_friction_token", lesion_typed_token(Prop.FRICTION)),
                               ("random_token", lesion_typed_token(Prop.FRICTION, random=True)),
                               ("edge_target_pusher", lesion_edge("target_pusher")),
                               ("edge_random_other", lesion_edge("random"))):
                    pred = predict(m, test, passes=4, lesion=lf, bs=128)
                    met = sample_metrics(pred, lab, y_mean, y_std, scales)
                    d = {f"delta_{n_}": met[n_] - base[n_] for n_ in ("nll", "pos_err_m")}
                    d["pred_change_pos_m"] = np.nanmean(np.linalg.norm(pred["mean"][..., :2] - base_pred["mean"][..., :2], axis=-1), 1)
                    W.metric_rows("e11", "compact", v, s, f"test_lesion_{ln}", d, lab["root"], lesion=ln)
                # restoration: unlesioned forward reproduces baseline exactly
                again = predict(m, test, passes=4)
                W.row("e11", "compact", v, s, "test_restoration", "restoration_max_abs_diff", float(np.abs(again["mean"] - base_pred["mean"]).max()),
                      len(np.unique(lab["root"])))
        if "e07" in diagnostics and models:
            run_e07(W, models, test, lab, dataset, n_roots=config.get("compact", {}).get("e07_roots", 96))
    if do_a and not dry:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(a_cache, open(cache_path, "wb"))
    rows = W.rows
    write_summary(rows, RES / "summary.csv")
    print(f"[eval] done rows={len(rows)} t={time.time()-t0:.0f}s", flush=True)
    return rows


def run_e07(W, models, test, lab, dataset, n_roots=96, fr_vals=(0.35, 0.65), ms_vals=(0.4, 1.2)):
    sel = np.nonzero(lab["k"] == 0)[0]
    roots = lab["root"][sel]
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(sel), size=min(n_roots, len(sel)), replace=False))
    sel, roots = sel[pick], roots[pick]
    t0 = time.time()
    cf_f = sim_counterfactual("test", roots, dataset, "friction", fr_vals)
    cf_m = sim_counterfactual("test", roots, dataset, "mass", ms_vals)
    print(f"[e07] counterfactual sims {time.time()-t0:.0f}s", flush=True)
    d_sim_f = cf_f[:, 1] - cf_f[:, 0]  # (n,H,7)
    d_sim_m = cf_m[:, 1] - cf_m[:, 0]
    idx = torch.as_tensor(sel, device=test.device)
    for (v, s), m in sorted(models.items()):
        if v not in ("b10", "b11", "b11p"):
            W.row("e07", "compact", v, s, "test_k0", "intervention", None, 0, "skipped", note="generic model has no typed slots")
            continue
        preds = {}
        for prop, vals in ((Prop.FRICTION, fr_vals), (Prop.MASS, ms_vals)):
            for val in vals:
                outs = []
                for ev, q, _ in test.iterate(128, idx=idx):
                    with torch.no_grad():
                        t, mk, sl = m.encode(ev, q)
                        B = t.shape[0]
                        lv = torch.full((B,), math.log(0.01**2), device=t.device)
                        t2, s2 = m.intervene(t, mk, sl, prop, torch.full((B,), float(val), device=t.device), lv)
                        outs.append(m.readout(m.refine(t2, mk, 4), mk, q, s2).mean.float().cpu())
                preds[(prop, val)] = torch.cat(outs).numpy()
        d_pred_f = preds[(Prop.FRICTION, fr_vals[1])] - preds[(Prop.FRICTION, fr_vals[0])]
        d_pred_m = preds[(Prop.MASS, ms_vals[1])] - preds[(Prop.MASS, ms_vals[0])]
        for nm, dp, ds_ in (("friction", d_pred_f, d_sim_f), ("mass_nuisance", d_pred_m, d_sim_m)):
            for hi in PRIMARY_IDX + [len(HORIZONS) - 1]:
                pp = dp[:, hi, :2]
                ss = ds_[:, hi, :2]
                npred = np.linalg.norm(pp, axis=-1)
                nsim = np.linalg.norm(ss, axis=-1)
                cos = (pp * ss).sum(-1) / np.maximum(npred * nsim, 1e-9)
                err = np.linalg.norm(pp - ss, axis=-1)
                h = HORIZONS[hi]
                for metric, arr in (("pred_change_pos_m", npred), ("sim_change_pos_m", nsim), ("abs_err_change_pos_m", err),
                                    ("cosine_pred_vs_sim", np.where(nsim > 1e-4, cos, np.nan))):
                    ur, pr = root_means(arr, roots)
                    mean, lo, hi_, n = bootstrap_ci(pr, W.n_boot)
                    W.row("e07", "compact", v, s, "test_k0", f"{nm}:{metric}@h{h}", mean, n, ci=(lo, hi_))
                # correlation across roots of predicted vs simulated displacement change (x and y pooled)
                a = pp.reshape(-1)
                b = ss.reshape(-1)
                corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
                W.row("e07", "compact", v, s, "test_k0", f"{nm}:corr_pred_vs_sim@h{h}", corr, len(roots))


def evaluate_e15(config=None, device="cpu", dataset="g1cal", pairs=(("b00", 1), ("b11", 1)), n_boot=2000, dry=False):
    """e15 conditional embodiment transfer: g1cal test metrics for zero-shot main checkpoints vs g1cal-finetuned
    copies, plus analytic baselines, the G1 prior-Coulomb stub and the (excluded) oracle."""
    from physmem.control import PriorCoulombStub

    config = {**(config or {}), "compact": {**(config or {}).get("compact", {}), "dataset": dataset}}
    chash = config_hash(dict(suite="e15", dataset=dataset, version=EVAL_VERSION))
    W = Writer(n_boot, chash, dry=dry, tag="e15", exp_override="e15")
    v2cfg = {"compact": {"dataset": "v2"}}
    train = load_dataset("train", v2cfg, device)
    y_mean, y_std = [x.numpy().astype(np.float64) for x in fit_target_scaler(train)]
    scales = fit_loss_scales(train.iterate(1024))
    cal = list(train.iterate(256, idx=torch.randperm(train.n, generator=torch.Generator().manual_seed(0))[:2048].to(train.device)))
    analytic = {"constvel": ConstantVelocity(), "coulomb": CoulombAnalytic(), "coulomb_oracle": CoulombOracle()}
    for m in analytic.values():
        m.calibrate(cal)  # calibrated on v2 TRAIN (no g1cal labels)
    del train, cal
    test = load_dataset("test", config, device)
    gtr = load_dataset("train", config, device)
    n_adapt = dict(samples=int(gtr.n), roots=int(len(np.unique(gtr.roots))))
    del gtr
    lab = labels_np(test)
    rows = []
    for name, am in list(analytic.items()) + [("stub_prior_coulomb_mu0.5", PriorCoulombStub(0.5))]:
        pred = predict(am, test, bs=512)
        met = sample_metrics(pred, lab, y_mean, y_std, scales)
        orc = getattr(am, "oracle", False)
        W.metric_rows("e15", name, "oracle" if orc else "analytic", -1, "g1cal_test", met, lab["root"], oracle=orc, adaptation="none")
    for v, s in pairs:
        for adapt, path in (("zero_shot", RES / "checkpoints" / f"compact-main-{v}-s{s}.pt"),
                            ("finetuned_g1cal", RES / "checkpoints" / f"compact-g1cal-{v}-s{s}.pt")):
            if not Path(path).exists():
                W.row("e15", "compact", v, s, "g1cal_test", "all", None, 0, "untested", adaptation=adapt, note=f"missing {path}")
                continue
            m = load_compact(path, device)
            pred = predict(m, test, passes=4)
            met = sample_metrics(pred, lab, y_mean, y_std, scales)
            W.metric_rows("e15", "compact", v, s, "g1cal_test", met, lab["root"], adaptation=adapt, ckpt=str(path),
                          adapter_data=n_adapt if adapt != "zero_shot" else None)
    write_summary(W.rows, RES / "summary.csv")
    print(f"[e15] rows={len(W.rows)}", flush=True)
    return W.rows


def _drop(pred_fn):
    def f(ev, q):
        return EventTensors(**{**ev.__dict__, "mask": ev.mask & ~pred_fn(ev, q)}), q
    return f


E12_REMOVALS = {
    "drop_contact": _drop(lambda ev, q: ev.prop == int(Prop.CONTACT)),
    "drop_command": _drop(lambda ev, q: ev.prop == int(Prop.COMMAND)),
    "drop_pusher_pose": _drop(lambda ev, q: (ev.prop == int(Prop.POSE2))
                              & (torch.gather(ev.ent_type, 1, ev.entity.long().clamp(min=0)) == 1)),
    "drop_size": _drop(lambda ev, q: ev.prop == int(Prop.SIZE)),
}


def e12_representation_roundtrip(models, n_roots=32, dataset="v2", seed=0):
    """equal-information representation change on the RECORD layer: every POSE2/VEL2/COMMAND/SIZE event is
    re-expressed in a random rotated+translated sensor frame and in millimetres (frames.transform_value /
    convert_length_units), then converted back to the SI world frame, re-tensorized via data.events_to_tensors
    (permute=False) and compared with the unmodified record path. returns {model: max|d mean|, max|d logvar|}."""
    import dataclasses

    from physmem import data as D
    from physmem.frames import Frame2, convert_length_units, transform_value
    from physmem.schema import PROP_DIM

    rng = np.random.default_rng(seed)
    t = torch.load(os.path.join(os.path.expanduser(D.DATA_DIR), dataset, "test_tensors.pt"), weights_only=True)
    lab = np.load(os.path.join(os.path.expanduser(D.DATA_DIR), dataset, "test_labels.npz"))
    evs_a, evs_b, nows, acts = [], [], [], []
    for i in range(n_roots):
        evs, root, _ = D.root_events("test", i, dataset)
        S = Frame2(float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)), float(rng.uniform(-math.pi, math.pi)))
        conv = []
        for e in evs:
            p = e.property_type
            if p in (Prop.POSE2, Prop.VEL2, Prop.COMMAND, Prop.SIZE):
                v, var = transform_value(p, e.value, e.variance, S.inverse())
                v, var = convert_length_units(p, v, var, 1.0, 0.001)  # -> mm in sensor frame
                v, var = convert_length_units(p, v, var, 0.001, 1.0)  # back to m
                v, var = transform_value(p, v, var, S)  # back to world
                e = dataclasses.replace(e, value=tuple(float(x) for x in v[: PROP_DIM[p]]),
                                        variance=tuple(float(x) for x in var[: PROP_DIM[p]]))
            conv.append(e)
        ri = int(np.nonzero(lab["root"] == root)[0][0])
        for k in range(4):
            sel = (t["root"] == root) & (t["k"] == k)
            j = int(torch.nonzero(sel)[0, 0])
            evs_a.append(evs)
            evs_b.append(conv)
            nows.append(float(lab["push_start"][ri][k]) - D.DECISION_EPS)
            acts.append(t["action"][j].numpy())
    ta, info = D.events_to_tensors(evs_a, nows, np.random.default_rng(0), permute=False, return_info=True)
    tb = D.events_to_tensors(evs_b, nows, np.random.default_rng(0), permute=False)
    q = D.make_query(info["target_index"], np.stack(acts))
    out = {"input_max_abs_value_diff": float((ta.value - tb.value).abs().max()),
           "input_max_abs_logvar_diff": float((ta.logvar - tb.logvar).abs().max())}
    for key, m in models.items():
        with torch.no_grad():
            dev = next(m.parameters()).device
            oa = m(ta.to(dev), q.to(dev), passes=4)
            ob = m(tb.to(dev), q.to(dev), passes=4)
        out[key] = (float((oa.mean - ob.mean).abs().max()), float((oa.logvar - ob.logvar).abs().max()))
    return out, len(evs_a)


def evaluate_e12(config=None, device="cpu", tag="main", dataset="v2", n_boot=2000, dry=False):
    config = {**(config or {}), "compact": {**(config or {}).get("compact", {}), "dataset": dataset}}
    chash = config_hash(dict(suite="e12", dataset=dataset, version=EVAL_VERSION))
    W = Writer(n_boot, chash, dry=dry, tag="e12", exp_override="e12")
    train = load_dataset("train", config, device)
    y_mean, y_std = [x.numpy().astype(np.float64) for x in fit_target_scaler(train)]
    scales = fit_loss_scales(train.iterate(1024))
    cal = list(train.iterate(256, idx=torch.randperm(train.n, generator=torch.Generator().manual_seed(0))[:2048].to(train.device)))
    analytic = {"constvel": ConstantVelocity(), "coulomb": CoulombAnalytic()}
    for m in analytic.values():
        m.calibrate(cal)
    del train, cal
    models = {k: load_compact(p, device) for k, p in find_checkpoints(tag).items() if k[0] in VARIANTS}
    # (1) equal-information representation change
    rt, n = e12_representation_roundtrip(models, dataset=dataset)
    W.row("e12", "pipeline", "input", -1, "test_roundtrip", "max_abs_value_diff", rt["input_max_abs_value_diff"], n // 4)
    W.row("e12", "pipeline", "input", -1, "test_roundtrip", "max_abs_logvar_diff", rt["input_max_abs_logvar_diff"], n // 4)
    for (v, s_), m in models.items():
        dm, dl = rt[(v, s_)]
        W.row("e12", "compact", v, s_, "test_roundtrip", "max_abs_delta_mean", dm, n // 4, n_samples=n)
        W.row("e12", "compact", v, s_, "test_roundtrip", "max_abs_delta_logvar", dl, n // 4, n_samples=n)
    print("[e12] roundtrip", rt, flush=True)
    # (2) sensor removal
    test = load_dataset("test", config, device)
    lab = labels_np(test)
    keep = ("pos_err_m", "nll", "cover50", "cover80", "cover95", "sharp_nstd")
    cands = [(("compact", v, s_), m) for (v, s_), m in sorted(models.items())] + [((n_, "analytic", -1), a) for n_, a in analytic.items()]
    for (mn, v, s_), m in cands:
        base = sample_metrics(predict(m, test, passes=4, bs=256), lab, y_mean, y_std, scales)
        for rn, fn in E12_REMOVALS.items():
            met = sample_metrics(predict(m, test, passes=4, edit=fn, bs=256), lab, y_mean, y_std, scales)
            d = {f"delta_{k}": met[k] - base[k] for k in keep}
            d.update({k: met[k] for k in keep})
            W.metric_rows("e12", mn, v, s_, f"test_{rn}", d, lab["root"], removal=rn)
        print(f"[e12] {mn} {v} {s_} done", flush=True)
    write_summary(W.rows, RES / "summary.csv")
    print(f"[e12] rows={len(W.rows)}", flush=True)
    return W.rows


def main(argv=None):
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("suite", nargs="?", default="primary")  # primary | e15
    p.add_argument("--config", default="configs/sprint.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--strata", default=None)
    p.add_argument("--diagnostics", default=None)
    p.add_argument("--tag", default="main")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--parts", default="all", choices=["all", "analytic", "compact"])
    p.add_argument("--dataset", default=None)
    a = p.parse_args(argv)
    cfg = yaml.safe_load(open(ROOT / a.config))
    if a.suite == "e12":
        evaluate_e12(cfg, device=a.device or "cpu", dry=a.dry)
        return
    if a.suite == "e15":
        evaluate_e15(cfg, device=a.device or "cpu", dry=a.dry)
        return
    if a.dataset:
        cfg.setdefault("compact", {})["dataset"] = a.dataset
    evaluate_suite(a.suite, cfg, device=a.device, strata=a.strata.split(",") if a.strata else None,
                   diagnostics=a.diagnostics.split(",") if a.diagnostics is not None else None, tag=a.tag,
                   max_samples=a.max_samples, dry=a.dry, parts=a.parts)


if __name__ == "__main__":
    main()
