"""training for the compact recurrent physical-memory model factorial (agent MODEL).

  train_suite("compact-factorial", config)      # orchestrates pilot + 4 variants x 3 seeds
  python -m physmem.train run --variant b11 --seed 0 --steps 3000 --tag main
  python -m physmem.train group --seed 0 --steps 3000 --variants b00,b10,b01,b11   (parallel, one lock hold)
  python -m physmem.train pilot

Policy (identical for every variant): same data order per seed (loader generator seeded by `seed`),
same passes schedule per seed (k in {1,2,4} sampled from a seeded generator), AdamW, cosine LR with
warmup, grad-clip 1.0, same step budget. Checkpoints at fixed fractions of the step budget (which
correspond to elapsed-time landmarks measured in the pilot) + final; the BEST-VALIDATION checkpoint
is saved as results/checkpoints/<run>.pt (selection by validation only), final as <run>_final.pt.
Failed runs are kept (ledger row with status=failed and the traceback).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from physmem.models.compact import build_model, count_params
from physmem.models.losses import LossConfig, compute_loss, fit_loss_scales
from physmem.schema import HORIZONS, STATE_DIM

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def config_hash(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


# ----------------------------------------------------------------------------
# data access (agent DATA's physmem.data; fake tensors when cfg fake_data)
# ----------------------------------------------------------------------------


class SplitView:
    """thin wrapper over physmem.data.PhysSplit (whole split resident on `device`).
    batches = (EventTensors, QueryTensors, LabelTensors); entity/source local indices are re-permuted
    per batch from a seeded generator during training (permute=True), fixed for evaluation."""

    def __init__(self, ps, name):
        self.ps = ps
        self.name = name
        self.n = ps.n
        self.device = ps.t["action"].device
        self.roots = ps.t["root"].cpu().numpy()
        self.k = ps.t["k"].cpu().numpy()

    def batch(self, idx, generator=None, permute=False):
        idx = torch.as_tensor(idx, dtype=torch.long).to(self.device)
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(0)
        return self.ps.batch(idx, generator, permute)

    def iterate(self, batch_size, shuffle=False, generator=None, permute=None, idx=None):
        permute = shuffle if permute is None else permute
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(0)
        base = torch.arange(self.n, device=self.device) if idx is None else torch.as_tensor(idx, device=self.device)
        order = base[torch.randperm(len(base), generator=generator, device=self.device)] if shuffle else base
        for i in range(0, len(order), batch_size):
            yield self.ps.batch(order[i : i + batch_size], generator, permute)

    def subset(self, idx):
        from physmem.data import PhysSplit

        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        return SplitView(PhysSplit({k: v[idx] for k, v in self.ps.t.items()}, self.name), self.name)


def _fake_split(n, seed, device):
    from physmem.data import PhysSplit

    sys.path.insert(0, str(ROOT / "tests"))
    from test_models import fake_batch

    ev, q, lab = fake_batch(B=n, E=64, N=4, seed=seed)
    t = dict(entity=ev.entity, related=ev.related, prop=ev.prop, value=ev.value, logvar=ev.logvar, age=ev.age,
             latency=ev.latency, source=ev.source, provenance=ev.provenance, mask=ev.mask,
             ent_type=torch.where(ev.ent_mask, ev.ent_type, torch.full_like(ev.ent_type, -1)), dt_obs=ev.dt_obs,
             action=q.action, future=lab.future, future_mask=lab.future_mask, friction=lab.friction, mass=lab.mass,
             contact_mode=lab.contact_mode, free_slide_mask=lab.free_slide_mask, law=lab.law,
             root=torch.arange(n) // 4, k=torch.arange(n) % 4, est=torch.zeros(n, 3, dtype=torch.float64))
    return PhysSplit(t, "fake", device)


def dataset_name(cfg):
    return cfg.get("compact", {}).get("dataset", "v1")


def load_dataset(split, cfg, device="cpu"):
    """load a split (physmem.data.load_split) as a SplitView on `device`."""
    mc = cfg.get("compact", {})
    if mc.get("fake_data"):
        n = {"train": 2048, "val": 256}.get(split, 256)
        return SplitView(_fake_split(n, abs(hash(split)) % 1000, device), split)
    from physmem import data as D

    sub = mc.get("train_subset") if split == "train" else None
    if sub:
        return SplitView(D.load_split(split, dataset_name(cfg), device, subset=sub), split)
    return SplitView(D.load_split(split, dataset_name(cfg), device), split)


def fit_target_scaler(ds: SplitView):
    y = ds.ps.t["future"].float().cpu()
    m = ds.ps.t["future_mask"].bool().cpu().float()[..., None]
    n = m.sum(0).clamp(min=1)
    mean = (y * m).sum(0) / n
    var = (((y - mean) ** 2) * m).sum(0) / n
    std = var.sqrt()
    # floor: late-horizon velocities are ~0 (object at rest); avoid exploding normalized errors
    std = torch.maximum(std, 0.1 * std.max(0, keepdim=True).values).clamp(min=1e-3)
    return mean, std


# ----------------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------------


def default_train_cfg(cfg):
    c = cfg.get("compact", {})
    return dict(
        width=c.get("width", 128),
        shared_blocks=c.get("shared_blocks", 2),
        train_passes=list(c.get("train_passes", [1, 2, 4])),
        eval_passes=list(c.get("eval_passes", [1, 2, 4, 8])),
        batch_size=c.get("batch_size", 128),
        lr=c.get("lr", 1e-3),
        weight_decay=c.get("weight_decay", 0.01),
        warmup=c.get("warmup", 200),
        grad_clip=1.0,
        bf16=c.get("bf16", True),
        val_every_frac=0.05,
        val_max=c.get("val_max", 2048),
        ckpt_fracs=[0.25, 0.5, 1.0],
        loss=dict(c.get("loss", {})),
    )


def evaluate_val(model, val: SplitView, passes, bs=512, bf16=True, scales=None, physics=True):
    model.eval()
    agg = {}
    n = 0
    dev = val.device
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and dev.type == "cuda"):
        for ev, q, lab in val.iterate(bs):
            out = model(ev, q, passes=passes)
            _, parts = compute_loss(out, lab, q, LossConfig(physics=physics), scales, model.head.y_mean, model.head.y_std)
            b = int(q.target.shape[0])
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v) * b
            n += b
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def train_run(variant, seed, steps, cfg, tag="main", device=None, train_ds=None, val_ds=None, log=print):
    tc = default_train_cfg(cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"compact-{tag}-{variant}-s{seed}"
    hp = dict(variant=variant, seed=seed, steps=steps, tag=tag, **tc)
    chash = config_hash(hp)
    curves = RES / "curves" / f"{run_id}.jsonl"
    ckdir = RES / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    if curves.exists():
        curves.rename(curves.with_suffix(f".prev{int(time.time())}.jsonl"))
    t_start = time.time()
    row = dict(time_utc=utc(), run_id=run_id, kind="compact_train", variant=variant, seed=seed, tag=tag, config_hash=chash,
               host=os.uname().nodename, device=str(device))
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if train_ds is None:
            train_ds = load_dataset("train", cfg, device)
        if val_ds is None:
            val_ds = load_dataset("val", cfg, device)
        if val_ds.n > tc["val_max"]:
            g = torch.Generator().manual_seed(999)
            val_ds = val_ds.subset(torch.randperm(val_ds.n, generator=g)[: tc["val_max"]].to(val_ds.device))
        model = build_model(variant, d=tc["width"], passes_train=tc["train_passes"], n_blocks=tc["shared_blocks"]).to(device)
        ym, ys = fit_target_scaler(train_ds)
        model.set_target_scaler(ym, ys)
        scales = fit_loss_scales(train_ds.iterate(1024))
        lcfg = LossConfig(physics=model.physics, **tc["loss"])
        n_params = count_params(model)
        row.update(n_params=n_params, loss_scales=scales, loss_cfg=lcfg.to_dict(), steps_planned=steps)
        opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
        warm = min(tc["warmup"], max(1, steps // 10))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))
        )
        gdata = torch.Generator(device=train_ds.device).manual_seed(seed)  # identical data order/perms per seed
        gpass = np.random.default_rng(10_000 + seed)  # identical passes schedule per seed
        val_every = max(1, int(steps * tc["val_every_frac"]))
        ck_steps = {max(1, int(round(f * steps))): f for f in tc["ckpt_fracs"]}
        best = (float("inf"), -1)
        step = 0
        examples = 0
        if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
            torch.cuda.reset_peak_memory_stats()
        it = None
        t_loop = time.time()
        model.train()
        while step < steps:
            if it is None:
                it = train_ds.iterate(tc["batch_size"], shuffle=True, generator=gdata)
            try:
                ev, q, lab = next(it)
            except StopIteration:
                it = None
                continue
            if int(q.target.shape[0]) < tc["batch_size"] // 2:
                continue
            k = int(gpass.choice(tc["train_passes"]))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tc["bf16"] and str(device).startswith("cuda")):
                out = model(ev, q, passes=k)
                loss, parts = compute_loss(out, lab, q, lcfg, scales, model.head.y_mean, model.head.y_std)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            examples += int(q.target.shape[0])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if step % 50 == 0 or step == 1:
                append_jsonl(curves, dict(run_id=run_id, split="train", step=step, examples=examples, elapsed_s=time.time() - t_start,
                                          passes=k, lr=sched.get_last_lr()[0], grad_norm=float(gn), **{kk: float(v) for kk, v in parts.items()}))
            if step % val_every == 0 or step == steps:
                vm = evaluate_val(model, val_ds, passes=4, bf16=tc["bf16"], scales=scales)
                append_jsonl(curves, dict(run_id=run_id, split="val", step=step, examples=examples, elapsed_s=time.time() - t_start, passes=4, **vm))
                if vm["nll"] < best[0]:
                    best = (vm["nll"], step)
                    torch.save(dict(state=model.state_dict(), variant=variant, seed=seed, step=step, hp=hp, scales=scales,
                                    config_hash=chash, val=vm), ckdir / f"{run_id}.pt")
                log(f"[{run_id}] step {step}/{steps} val_nll {vm['nll']:.4f} mse {vm['mse_norm']:.4f} fr_err {vm['friction_abs_err']:.3f} "
                    f"t={time.time()-t_start:.0f}s")
            if step in ck_steps:
                torch.save(dict(state=model.state_dict(), variant=variant, seed=seed, step=step, hp=hp, scales=scales, config_hash=chash,
                                elapsed_s=time.time() - t_start), ckdir / f"{run_id}_at{int(ck_steps[step]*100)}.pt")
        torch.save(dict(state=model.state_dict(), variant=variant, seed=seed, step=step, hp=hp, scales=scales, config_hash=chash),
                   ckdir / f"{run_id}_final.pt")
        wall = time.time() - t_start
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        row.update(status="ok", wall_s=wall, loop_s=time.time() - t_loop, steps=step, examples=examples,
                   step_s=(time.time() - t_loop) / max(step, 1), peak_mem_mib=peak, best_val_nll=best[0], best_step=best[1])
    except Exception as e:  # keep failed runs
        row.update(status="failed", error=repr(e), traceback=traceback.format_exc(), wall_s=time.time() - t_start)
        log(row["traceback"])
    row["end_utc"] = utc()
    append_jsonl(RES / "ledger.jsonl", row)
    return row


# ----------------------------------------------------------------------------
# pilot + suite
# ----------------------------------------------------------------------------


def pilot(cfg, steps=150, variants=("b00", "b10", "b01", "b11"), parallel=(1, 4)):
    """measure step time per variant (alone and with `p` concurrent processes) on the real data."""
    device = "cuda"
    tr = load_dataset("train", cfg, device)
    va = load_dataset("val", cfg, device)
    va = va.subset(torch.arange(min(512, va.n)))
    res = {}
    for v in variants:
        r = train_run(v, 0, steps, cfg, tag="pilot", device=device, train_ds=tr, val_ds=va)
        res[v] = r.get("step_s")
        print(v, r.get("status"), r.get("step_s"), r.get("n_params"), flush=True)
    out = dict(time_utc=utc(), kind="pilot", step_s_alone=res, n_train=tr.n, n_val=va.n,
               events_max=int(tr.ps.t["entity"].shape[1]), mean_events=float(tr.ps.t["mask"].float().sum(1).mean()))
    append_jsonl(RES / "ledger.jsonl", out)
    return out


def finetune(ckpt_path, out_name, dataset="g1cal", steps=1500, bs=64, lr=3e-4, seed=0, device="cuda", val_every=75):
    """e15: fine-tune a main checkpoint on `dataset` train. Same loss config/weights and loss scales as the source
    run (stored in its checkpoint), target scaler KEPT (not refit), AdamW lr const after 50-step warmup, best by
    `dataset` val NLL (passes=4). Saves results/checkpoints/<out_name>.pt."""
    from physmem.evaluate import load_compact

    cfg = {"compact": {"dataset": dataset}}
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = load_compact(ckpt_path, device)
    model.train()
    scales = ck["scales"]
    hp = ck["hp"]
    lcfg = LossConfig(physics=model.physics, **hp.get("loss", {}))
    tr = load_dataset("train", cfg, device)
    va = load_dataset("val", cfg, device)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=hp.get("weight_decay", 0.01))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / 50))
    g = torch.Generator(device=tr.device).manual_seed(seed)
    gp = np.random.default_rng(10_000 + seed)
    run_id = out_name
    curves = RES / "curves" / f"{run_id}.jsonl"
    t0 = time.time()
    v0 = evaluate_val(model, va, passes=4, bf16=True, scales=scales)
    append_jsonl(curves, dict(run_id=run_id, split="val", step=0, elapsed_s=0.0, **v0))
    best = (v0["nll"], 0)
    torch.save(dict(state=model.state_dict(), variant=ck["variant"], seed=ck["seed"], step=0, hp={**hp, "finetune": dict(
        source=str(ckpt_path), dataset=dataset, steps=steps, bs=bs, lr=lr)}, scales=scales, config_hash=ck.get("config_hash"), val=v0),
        RES / "checkpoints" / f"{out_name}.pt")
    step, ex = 0, 0
    it = None
    while step < steps:
        if it is None:
            it = tr.iterate(bs, shuffle=True, generator=g)
        try:
            ev, q, lab = next(it)
        except StopIteration:
            it = None
            continue
        if int(q.target.shape[0]) < bs // 2:
            continue
        k = int(gp.choice(hp.get("train_passes", [1, 2, 4])))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            out = model(ev, q, passes=k)
            loss, parts = compute_loss(out, lab, q, lcfg, scales, model.head.y_mean, model.head.y_std)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        ex += int(q.target.shape[0])
        if step % 25 == 0:
            append_jsonl(curves, dict(run_id=run_id, split="train", step=step, examples=ex, elapsed_s=time.time() - t0,
                                      **{kk: float(vv) for kk, vv in parts.items()}))
        if step % val_every == 0 or step == steps:
            vm = evaluate_val(model, va, passes=4, bf16=True, scales=scales)
            append_jsonl(curves, dict(run_id=run_id, split="val", step=step, examples=ex, elapsed_s=time.time() - t0, **vm))
            if vm["nll"] < best[0]:
                best = (vm["nll"], step)
                torch.save(dict(state=model.state_dict(), variant=ck["variant"], seed=ck["seed"], step=step, hp={**hp, "finetune": dict(
                    source=str(ckpt_path), dataset=dataset, steps=steps, bs=bs, lr=lr)}, scales=scales,
                    config_hash=ck.get("config_hash"), val=vm), RES / "checkpoints" / f"{out_name}.pt")
    row = dict(time_utc=utc(), run_id=run_id, kind="compact_finetune", source=str(ckpt_path), dataset=dataset, steps=step,
               examples=ex, n_train_samples=tr.n, n_train_roots=int(len(np.unique(tr.roots))), bs=bs, lr=lr, wall_s=time.time() - t0,
               val_nll_before=v0["nll"], best_val_nll=best[0], best_step=best[1], status="ok", variant=ck["variant"], seed=ck["seed"])
    append_jsonl(RES / "ledger.jsonl", row)
    print(json.dumps(row), flush=True)
    return row


def weight_pilot(cfg, steps=1500, mults=(0.1, 0.3, 1.0), variant="b11", seed=100):
    """physics-loss weight pilot (train/val only): b11 with all physics weights scaled by each multiplier;
    the multiplier with the lowest validation NLL (passes=4) is selected and logged. seed 100 is not a main seed."""
    device = "cuda"
    tr = load_dataset("train", cfg, device)
    va = load_dataset("val", cfg, device)
    base = LossConfig(physics=True)
    res = {}
    for m in mults:
        c = {**cfg, "compact": {**cfg.get("compact", {}), "loss": dict(w_coulomb=base.w_coulomb * m, w_kinematic=base.w_kinematic * m,
                                                                     w_friction=base.w_friction * m, w_mass=base.w_mass * m,
                                                                     w_contact=base.w_contact * m)}}
        r = train_run(variant, seed, steps, c, tag=f"wpilot{m}", device=device, train_ds=tr, val_ds=va, log=lambda *a: None)
        res[m] = r.get("best_val_nll")
        print("weight pilot", m, r.get("status"), r.get("best_val_nll"), flush=True)
    best = min(res, key=lambda k: res[k] if res[k] is not None else 1e9)
    out = dict(time_utc=utc(), kind="weight_pilot", dataset=dataset_name(cfg), steps=steps, val_nll=res, selected_mult=best,
               selected=dict(w_coulomb=base.w_coulomb * best, w_kinematic=base.w_kinematic * best, w_friction=base.w_friction * best,
                             w_mass=base.w_mass * best, w_contact=base.w_contact * best))
    append_jsonl(RES / "ledger.jsonl", out)
    print(json.dumps(out), flush=True)
    return out


def run_group(seed, steps, variants, cfg_path, tag="main"):
    """run variants for one seed concurrently (separate processes) inside ONE gpu lock hold."""
    procs = []
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    for v in variants:
        lf = open(logs / f"compact-{tag}-{v}-s{seed}.log", "w")
        p = subprocess.Popen([sys.executable, "-m", "physmem.train", "run", "--variant", v, "--seed", str(seed), "--steps", str(steps),
                              "--tag", tag, "--config", cfg_path], stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
        procs.append(p)
    rc = [p.wait() for p in procs]
    return rc


def _mult_loss(m):
    b = LossConfig(physics=True)
    return dict(w_coulomb=b.w_coulomb * m, w_kinematic=b.w_kinematic * m, w_friction=b.w_friction * m, w_mass=b.w_mass * m,
                w_contact=b.w_contact * m)


def train_suite(suite, config, cfg_path="configs/sprint.yaml", overrides=None):
    """compact-factorial: step budget from pilot timing; runs are executed SEQUENTIALLY (the pilot showed 4
    concurrent runs give no throughput gain on the GB10: 0.20 s/step each vs 0.047-0.050 s alone), each run in
    its OWN gpu_heavy.sh lock hold so other heavy jobs can interleave. Order: seed-major (s0: b00,b10,b01,b11, s1: ...)
    so a partially completed suite is still a balanced factorial."""
    if suite != "compact-factorial":
        raise SystemExit(f"unknown suite {suite}")
    c = config.get("compact", {})
    variants = c.get("variants", ["b00", "b10", "b01", "b11"])
    seeds = c.get("seeds", [0, 1, 2])
    extra = c.get("extra_runs", [])  # e.g. [["b11u", 0]] untied control
    runs = [(v, s) for s in seeds for v in variants] + [tuple(x) for x in extra]
    steps = c.get("steps")
    usable = float(c.get("usable_seconds", 60 * c.get("target_minutes_per_run", 8) * len(runs)))
    step_s = float(c.get("conservative_step_seconds", 0.06))
    if steps is None:
        steps = int(math.floor(usable / (len(runs) * step_s)))
    tag = c.get("tag", "main")
    append_jsonl(RES / "ledger.jsonl", dict(time_utc=utc(), kind="suite_plan", suite=suite, runs=runs, steps=steps, usable_seconds=usable,
                                            conservative_step_seconds=step_s, config=c))
    print(f"[suite] {len(runs)} runs x {steps} steps", flush=True)
    for v, s in runs:
        lm = (overrides or {}).get("loss_mult")
        cmd = f"cd {ROOT} && source .venv/bin/activate && export PYTHONPATH=src && python -W ignore -m physmem.train run " \
              f"--variant {v} --seed {s} --steps {steps} --tag {tag} --config {cfg_path}" + \
              (f" --loss-mult {lm}" if lm is not None else "") + \
              (f" --train-subset {c['train_subset']}" if c.get("train_subset") else "") + \
              (f" --dataset {c['dataset']}" if c.get("dataset") else "") + f" > logs/compact-{tag}-{v}-s{s}.log 2>&1"
        t0 = time.time()
        r = subprocess.run(["bash", str(ROOT / "scripts" / "gpu_heavy.sh"), f"compact-{tag}-{v}-s{s}", cmd], check=False)
        print(f"[suite] {v} s{s} rc={r.returncode} {time.time()-t0:.0f}s (incl. lock wait) {utc()}", flush=True)
    return steps


def main(argv=None):
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "group", "pilot", "suite", "wpilot", "finetune"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--variant", default="b11")
    p.add_argument("--variants", default="b00,b10,b01,b11")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--loss-mult", type=float, default=None, help="physics weight multiplier (from wpilot)")
    p.add_argument("--extra", default="", help="extra runs e.g. b11u:0")
    p.add_argument("--train-subset", default=None, help="e17: 1pct|10pct nested train subset (matched steps)")
    p.add_argument("--tag", default="main")
    p.add_argument("--config", default="configs/sprint.yaml")
    p.add_argument("--pilot-steps", type=int, default=150)
    a = p.parse_args(argv)
    cfg = yaml.safe_load(open(ROOT / a.config))
    if a.dataset:
        cfg.setdefault("compact", {})["dataset"] = a.dataset
    if a.train_subset:
        cfg.setdefault("compact", {})["train_subset"] = a.train_subset
    if a.mode == "finetune":
        finetune(a.ckpt, a.out, dataset=a.dataset or "g1cal", steps=a.steps or 1500)
        return
    if a.mode == "wpilot":
        weight_pilot(cfg, steps=a.steps or 1500)
    elif a.mode == "run":
        if a.loss_mult is not None:
            cfg.setdefault("compact", {})["loss"] = _mult_loss(a.loss_mult)
        r = train_run(a.variant, a.seed, a.steps or 1000, cfg, tag=a.tag)
        print(json.dumps({k: v for k, v in r.items() if k != "traceback"}, default=str))
        sys.exit(0 if r["status"] == "ok" else 1)
    elif a.mode == "group":
        rc = run_group(a.seed, a.steps or 1000, a.variants.split(","), a.config, a.tag)
        print("rc", rc)
    elif a.mode == "pilot":
        print(json.dumps(pilot(cfg, steps=a.pilot_steps, variants=a.variants.split(",")), default=str))
    elif a.mode == "suite":
        c = cfg.setdefault("compact", {})
        c["tag"] = a.tag
        if a.variants:
            c["variants"] = a.variants.split(",")
        if a.seeds:
            c["seeds"] = [int(x) for x in a.seeds.split(",")]
        if a.steps:
            c["steps"] = a.steps
        if a.extra:
            c["extra_runs"] = [[x.split(":")[0], int(x.split(":")[1])] for x in a.extra.split(",")]
        if a.loss_mult is not None:
            c["loss"] = _mult_loss(a.loss_mult)
        train_suite("compact-factorial", cfg, a.config, overrides=dict(steps=c.get("steps"), loss_mult=a.loss_mult))


if __name__ == "__main__":
    main()
