"""System I training / evaluation (owner: INKLING agent). See docs/INKLING_SYS12.md.

  python -m physmem.sys12_train train --arm inkling_gated  --run s1_gated_s0  --steps 4000 --bs 32
  python -m physmem.sys12_train train --arm inkling_ungated --run s1_ungated_s0 ...    (control i: ungated LoRA)
  python -m physmem.sys12_train train --arm inkling_random  --run s1_random_s0 ...     (control ii: random-init core)
  python -m physmem.sys12_train train --arm compact         --run s1_compact_s0 ...    (control iii: compact, same samples)
  python -m physmem.sys12_train eval  --run s1_gated_s0 [--max-rows 512]

Data: data/v2_r7_final (MODEL-V2 r7 data: drawer_open train/val excluded = held-out FAMILY, Gen3 = held-out ROBOT);
family-homogeneous batches (round robin), robot-balanced sampling, history dropout 0.15 - the r7 recipe. Memory front
end = frozen r7 typed s0. Every arm sees the same sample sequence (same seed -> same family / index draws).
Resumable: a partial checkpoint (adapters + optimizer + step + RNG states) is written every --save-every steps.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from physmem.models.v2 import spec as S
from physmem.v2_train import _auc, load_model, load_stores

RES = os.environ.get("SYS12_RESULTS", "results/sys12")
CKD = os.path.join(RES, "checkpoints")
BASE_CKPT = "results/v2/checkpoints/v2_robots_r7_typed_s0.pt"
DATA = "data/v2_r7_final"
HOLDOUT_FAMILY = "drawer_open"
_LM = {}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get_lm(random_init=False, seed=0):
    """load Inkling-2-Mini once per process (packed experts)."""
    key = ("lm", random_init)
    if key not in _LM:
        from physmem.models.inkling_quant import load_inkling_quant, randomize_text_model_
        model, tm, tok, rep = load_inkling_quant("ckpt/Inkling-2-Mini", keep_lm_head=not random_init)
        if random_init:
            randomize_text_model_(tm, range(0, tm.config.loop_layer_index_end), seed=1000 + seed)
            rep["random_init"] = dict(layers=f"0..{tm.config.loop_layer_index_end - 1}", embed="row-permuted",
                                      method="per-tensor element shuffle (dense) / per-expert code+scale shuffle")
        _LM.clear()
        _LM[key] = (model, tm, tok, rep)
    return _LM[key]


def build(arm, passes=1, seed=0, lora_rank=8, base_ckpt=BASE_CKPT, n_readout=16):
    from physmem.models.sys12 import CompactSys1Core, InklingSys1V2, Sys1Core
    dev = torch.device("cuda")
    base, _ = load_model(base_ckpt, dev)
    torch.manual_seed(seed)
    if arm == "compact":
        core = CompactSys1Core(d_mem=base.scene.ent[0].out_features, act_dim=base.n_skills).to(dev)
        rep = {}
    else:
        _, tm, tok, rep = get_lm(random_init=arm == "inkling_random", seed=seed)
        core = Sys1Core(tm, tok, d_mem=128, act_dim=base.n_skills, n_readout=n_readout, passes=passes,
                        lora_rank=lora_rank, lora_mode="always" if arm == "inkling_ungated" else "gated", seed=seed)
    m = InklingSys1V2(base, core)
    return m, rep


def trainable(m):
    own = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("core.")]
    return own + list(m.core.trainable_parameters())


def model_state(m):
    sd = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad and not n.startswith("core.")}
    return dict(model=sd, core=m.core.adapter_state())


def load_state(m, st):
    own = dict(m.named_parameters())
    with torch.no_grad():
        for n, v in st["model"].items():
            own[n].copy_(v)
    m.core.load_adapter_state(st["core"])


def count(ps):
    return int(sum(p.numel() for p in ps))


def train(a):
    os.makedirs(CKD, exist_ok=True)
    os.makedirs(os.path.join(RES, "curves"), exist_ok=True)
    dev = torch.device("cuda")
    tr = {f: s for f, s in load_stores(DATA, "train", dev).items() if f != HOLDOUT_FAMILY}
    m, rep = build(a.arm, a.passes, a.seed, a.lora_rank)
    params = trainable(m)
    # param groups: outcome head + typed-memory tokens (+ the whole compact core) at lr_head; Inkling adapters at lr;
    # zero-init read/write gates at 10 x lr
    gate_ids = {id(p) for n, p in m.core.named_parameters() if n in ("read_gate", "write_gate")}
    core_ids = {id(p) for p in m.core.trainable_parameters()} if a.arm != "compact" else set()
    g_head = [p for p in params if id(p) not in core_ids]
    g_core = [p for p in params if id(p) in core_ids and id(p) not in gate_ids]
    g_gate = [p for p in params if id(p) in gate_ids]
    groups = [dict(params=g_head, lr=a.lr_head), dict(params=g_core, lr=a.lr), dict(params=g_gate, lr=10 * a.lr)]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=a.steps + 1,
                                                pct_start=0.05)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    step0, curve, elapsed0 = 0, [], 0.0
    pp = os.path.join(CKD, f"{a.run}.partial.pt")
    if os.path.exists(pp) and not a.fresh:
        c = torch.load(pp, map_location="cpu", weights_only=False)
        load_state(m, c["state"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        gen.set_state(c["gen"].to(gen.device) if hasattr(c["gen"], "to") else c["gen"])
        torch.set_rng_state(c["cpu_rng"]); torch.cuda.set_rng_state(c["cuda_rng"])
        step0, curve, elapsed0 = c["step"] + 1, c["curve"], c["elapsed"]
        print(f"[{a.run}] resumed at step {step0}", flush=True)
    fam_list = list(tr)
    info = dict(run=a.run, arm=a.arm, passes=a.passes, seed=a.seed, steps=a.steps, bs=a.bs, lr=a.lr, lr_head=a.lr_head,
                trainable_params=count(params), lm=rep, families=fam_list, base_ckpt=BASE_CKPT, data=DATA,
                holdout_family=HOLDOUT_FAMILY, heldout_robot="gen3_2f85")
    print(json.dumps(info, default=str), flush=True)
    t0 = time.time() - elapsed0
    for step in range(step0, a.steps + 1):
        fam = fam_list[step % len(fam_list)]
        st = tr[fam]
        if not hasattr(st, "_w"):
            _, inv, cnt = np.unique(st.robot, return_inverse=True, return_counts=True)
            st._w = torch.as_tensor(1.0 / cnt[inv], device=dev, dtype=torch.float32)
        idx = torch.multinomial(st._w, a.bs, replacement=True, generator=gen)
        b, lab = st.gather(idx, drop_hist=0.15, gen=gen)
        frac = min(step / max(a.steps * 0.3, 1), 1.0)
        from physmem.models.sys12 import sys1_loss
        loss, logs = sys1_loss(m, b, lab, fam, tf_prob=0.5 * (1.5 - 0.5 * frac))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if hasattr(m.core, "disarm"):
            m.core.disarm()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % a.log_every == 0:
            logs.update(step=step, fam=fam, gn=float(gn), elapsed=round(time.time() - t0, 1),
                        s_per_step=round((time.time() - t0) / max(step + 1, 1), 3),
                        gates=[round(float(x), 4) for x in getattr(m.core, "read_gate", torch.zeros(0)).tolist()])
            curve.append(logs)
            print(json.dumps(logs), flush=True)
        stop = a.time_budget_s and (time.time() - t0) > a.time_budget_s
        if (step > 0 and step % a.save_every == 0) or step == a.steps or stop:
            torch.save(dict(state=model_state(m), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), step=step,
                            curve=curve, elapsed=time.time() - t0, info=info), pp + ".tmp")
            os.replace(pp + ".tmp", pp)
        if stop:
            print(f"[{a.run}] time budget reached at step {step}", flush=True)
            break
    info.update(final_step=step, train_seconds=round(time.time() - t0, 1), samples_seen=(step + 1) * a.bs,
                time_utc=utc())
    torch.save(dict(state=model_state(m), info=info), os.path.join(CKD, f"{a.run}.pt"))
    json.dump(dict(info=info, curve=curve), open(os.path.join(RES, "curves", f"{a.run}.json"), "w"), default=str)
    print(f"[{a.run}] done: {info['samples_seen']} samples in {info['train_seconds']}s", flush=True)
    if not a.no_eval:
        evaluate_run(a.run, m, max_rows=a.max_rows)
    return m


# ------------------------------------------------------------------------------------------- evaluation
EVAL_SPLITS = ("test", "heldout_robot", "heldout_family")


def eval_subset(st, max_rows, seed=0):
    g = np.random.default_rng(seed + st.fid)
    n = st.n
    idx = np.sort(g.choice(n, size=min(n, max_rows), replace=False))
    return torch.as_tensor(idx, device=st.k.device)


@torch.no_grad()
def evaluate_model(m, stores, max_rows=512, bs=64, sys1=True):
    """outcome metrics per family on a FIXED row subset (same for every arm). sys1=False -> plain MODEL-V2."""
    ev = S.OUT_EVENTS
    out = {}
    m.eval()
    for fam, st in stores.items():
        idx_all = eval_subset(st, max_rows)
        P, Y, M, NLL, E, PE = [], [], [], [], [], []
        t0 = time.time()
        for i in range(0, len(idx_all), bs):
            idx = idx_all[i:i + bs]
            b, lab = st.gather(idx)
            o = m(b, family=fam) if sys1 else m(b)
            d = m.decode(o)
            P.append(d["p"].float().cpu()); Y.append(lab.events.float().cpu()); M.append(lab.events_mask.cpu())
            yc = (lab.cont - m.cont_mean) / m.cont_std
            nll = 0.5 * (o["cont_logvar"] + (yc - o["cont_mu"]) ** 2 * torch.exp(-o["cont_logvar"]))
            NLL.append(((nll * lab.cont_mask).sum(1) / lab.cont_mask.sum(1).clamp(min=1)).cpu())
            E.append(d["log_effort"].cpu()); PE.append(lab.effort.cpu())
        P, Y, M = torch.cat(P), torch.cat(Y), torch.cat(M)
        r = dict(n=int(len(idx_all)), s_per_row=(time.time() - t0) / max(len(idx_all), 1))
        for e in ("success", "damage", "slip", "drop"):
            j = ev.index(e)
            mk = M[:, j]
            if mk.sum() > 10 and 0 < Y[mk, j].mean() < 1:
                r[f"{e}_auc"] = _auc(P[mk, j].numpy(), Y[mk, j].numpy())
                r[f"{e}_brier"] = float(((P[mk, j] - Y[mk, j]) ** 2).mean())
                r[f"{e}_rate"] = float(Y[mk, j].mean())
        r["cont_nll"] = float(torch.cat(NLL).mean())
        r["effort_log_rmse"] = float(((torch.cat(E) - torch.cat(PE)) ** 2).mean().sqrt())
        out[fam] = r
    return out


def evaluate_run(run, m=None, max_rows=512, splits=EVAL_SPLITS):
    dev = torch.device("cuda")
    if m is None:
        c = torch.load(os.path.join(CKD, f"{run}.pt"), map_location="cpu", weights_only=False)
        inf = c["info"]
        m, _ = build(inf["arm"], inf["passes"], inf["seed"])
        load_state(m, c["state"])
    res = {}
    for sp in splits:
        st = load_stores(DATA, sp, dev)
        if sp != "heldout_family":
            st = {f: s for f, s in st.items() if f != HOLDOUT_FAMILY}
        res[sp] = evaluate_model(m, st, max_rows=max_rows)
        print(sp, json.dumps(pool(res[sp])), flush=True)
    os.makedirs(os.path.join(RES, "eval"), exist_ok=True)
    json.dump(res, open(os.path.join(RES, "eval", f"{run}.json"), "w"), indent=1)
    return res


def pool(r):
    """row-weighted mean over families of each metric."""
    keys = sorted({k for v in r.values() for k in v if k not in ("n",)})
    out = {}
    for k in keys:
        num = sum(v[k] * v["n"] for v in r.values() if k in v and np.isfinite(v[k]))
        den = sum(v["n"] for v in r.values() if k in v and np.isfinite(v[k]))
        if den:
            out[k] = round(num / den, 4)
    return out


def eval_v2(ckpt, tag, max_rows=512):
    """reference: plain MODEL-V2 checkpoint on the same row subsets."""
    dev = torch.device("cuda")
    m, _ = load_model(ckpt, dev)
    res = {}
    for sp in EVAL_SPLITS:
        st = load_stores(DATA, sp, dev)
        if sp != "heldout_family":
            st = {f: s for f, s in st.items() if f != HOLDOUT_FAMILY}
        res[sp] = evaluate_model(m, st, max_rows=max_rows, sys1=False)
        print(tag, sp, json.dumps(pool(res[sp])), flush=True)
    os.makedirs(os.path.join(RES, "eval"), exist_ok=True)
    json.dump(res, open(os.path.join(RES, "eval", f"{tag}.json"), "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=["inkling_gated", "inkling_ungated", "inkling_random", "compact"])
    t.add_argument("--run", required=True)
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--lr-head", type=float, default=1e-3)
    t.add_argument("--passes", type=int, default=1)
    t.add_argument("--lora-rank", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--save-every", type=int, default=200)
    t.add_argument("--time-budget-s", type=float, default=0)
    t.add_argument("--max-rows", type=int, default=512)
    t.add_argument("--no-eval", action="store_true")
    t.add_argument("--fresh", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--run", required=True)
    e.add_argument("--max-rows", type=int, default=512)
    v = sub.add_parser("evalv2")
    v.add_argument("--ckpt", required=True)
    v.add_argument("--tag", required=True)
    v.add_argument("--max-rows", type=int, default=512)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        train(a)
    elif a.cmd == "eval":
        evaluate_run(a.run, max_rows=a.max_rows)
    else:
        eval_v2(a.ckpt, a.tag, a.max_rows)


if __name__ == "__main__":
    main()
