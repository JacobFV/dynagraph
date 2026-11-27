"""V3 section 0 evaluation (owner: V3-MODEL): prediction, causal acceptance tests, planner, classical baseline.

  python -m physmem.v3_chain_eval predict --ckpt <run.pt> --split test [--cond full|no_track|no_proprio|no_anchor]
  python -m physmem.v3_chain_eval classical --split test
  python -m physmem.v3_chain_eval causal  --ckpt <run.pt> --split test --n 200
  python -m physmem.v3_chain_eval classical_causal --split test --n 200

Outputs: results/v3/eval/<run>_<split>_<what>.npz/.json (per-row arrays so every paired comparison is recomputable).
Evaluator-only data (labels, true parameters, true start states for counterfactual re-simulation) are used ONLY to
score and to re-simulate the physical counterfactual, never as model input.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from physmem.models.v3 import chain as C

RESULTS = os.environ.get("V3_RESULTS", "results/v3")
F_GRID = np.round(np.linspace(1.0, 15.0, 29), 3)
PLAN_T = 0.2
DELTA = 0.67                        # normalised edit size (friction +-0.10, mass x/÷ 1.49, k x/÷ 1.6)


def _out(name):
    d = os.path.join(RESULTS, "eval")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


# ----------------------------------------------------------------------------- model helpers
def _ids(n, device, seed):
    import torch
    from physmem.models.v3 import belief as B
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(n, C.N_ENT, B.ID_D, device=device, generator=g)


def model_predict(model, b, passes=4, ids=None, clamps=(), block_edges=None, late_ev=None, late_from=None,
                  all_passes=False):
    import torch
    with torch.no_grad():
        outs = model(b, passes=passes, ids=ids, clamps=clamps, block_edges=block_edges, late_ev=late_ev,
                     late_from=late_from)
    return outs if all_passes else outs[-1]


def with_action(b, F, T):
    import torch
    from physmem.models.v3 import belief as B
    a = torch.stack([torch.as_tensor(F, dtype=torch.float32, device=b.xhat.device).expand(len(b.xhat)),
                     torch.as_tensor(T, dtype=torch.float32, device=b.xhat.device).expand(len(b.xhat))], 1)
    return B.Batch(**{**b.__dict__, "action": a})


def repeat_batch(b, n):
    from physmem.models.v3 import belief as B
    return B.Batch(**{k: v.repeat_interleave(n, 0) for k, v in b.__dict__.items()})


# ----------------------------------------------------------------------------- prediction metrics
def predict(args):
    import torch
    from physmem.models.v3 import belief as B
    from physmem.v3_chain import load
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load(args.ckpt, dev)
    run = os.path.basename(args.ckpt)[:-3]
    st = B.ChainStore(os.path.join(args.data, f"{args.split}.npz"), dev)
    drop = {} if args.cond == "full" else {args.cond: True}
    rows = st.rows()
    res = {}
    for K in [int(x) for x in args.passes.split(",")]:
        FE, PE, PZ, PS, PM, XS, XZ, LAW = [], [], [], [], [], [], [], []
        for i in range(0, len(rows), 1024):
            r = rows[i:i + 1024]
            b, lab = st.gather(r, drop)
            s = model_predict(model, b, K, _ids(len(r), dev, 1000 + i))
            box = b.ent_type == C.ENT_BOX
            e = (s.xm()[..., -1] * B.X_SC - lab.X[..., -1])
            sd = torch.exp(0.5 * s.xlv()[..., -1]) * B.X_SC
            FE.append(torch.where(box, e.abs(), torch.nan)); XS.append(torch.where(box, sd, torch.nan))
            XZ.append(torch.where(box, e / sd, torch.nan))
            pm_t, pmask, *_ = B.label_targets(b, lab)
            psd = torch.exp(0.5 * s.plv())
            PE.append(torch.where(pmask, (s.pm() - pm_t).abs(), torch.nan))
            PZ.append(torch.where(pmask, (s.pm() - pm_t) / psd, torch.nan))
            PS.append(torch.where(pmask, psd, torch.nan)); PM.append(torch.where(pmask, s.pm(), torch.nan))
            LAW.append(sum(B.store_law(s, b, per_row=True).values()))
        cat = lambda x: torch.cat(x).cpu().numpy()
        out = dict(rows=rows.cpu().numpy(), final_err=cat(FE), final_sd=cat(XS), final_z=cat(XZ), p_err=cat(PE),
                   p_z=cat(PZ), p_sd=cat(PS), p_mean=cat(PM), law=cat(LAW))
        np.savez(_out(f"{run}_{args.split}_{args.cond}_K{K}.npz"), **out)
        fe = out["final_err"]
        summ = dict(final_err_mm=float(np.nanmean(fe) * 1e3), final_cov80=float(np.nanmean(np.abs(out["final_z"]) < 1.2816)),
                    law=float(np.mean(out["law"])))
        for j, nm in enumerate(B.P_FIELDS):
            for k in (0, 1, 3):
                sel = out["rows"][:, 1] == k
                pe = out["p_err"][sel][..., j]
                if np.isfinite(pe).sum() > 10:
                    summ[f"{nm}:abs_err@k{k}"] = float(np.nanmean(pe))
                    pm_ = out["p_mean"][sel][..., j]; ok = np.isfinite(pm_)
                    tgt = (pm_ - (out["p_z"][sel][..., j] * out["p_sd"][sel][..., j]))[ok]
                    summ[f"{nm}:corr@k{k}"] = float(np.corrcoef(pm_[ok], tgt)[0, 1])
                    summ[f"{nm}:cov80@k{k}"] = float(np.nanmean(np.abs(out["p_z"][sel][..., j]) < 1.2816))
        res[f"K{K}"] = summ
        print(run, args.split, args.cond, f"K={K}", json.dumps({k: round(v, 3) for k, v in summ.items()}), flush=True)
    json.dump(res, open(_out(f"{run}_{args.split}_{args.cond}.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- classical baseline
class Classical:
    """CEM system identification on the SAME actor-legal evidence (binned tracker / anchor displacements, pusher
    contact-force bins) with the analytic chain simulator, then analytic prediction. MAP under the training prior."""
    POP, ITERS, ELITE = 64, 8, 8
    SIG_TRK, SIG_ANC, SIG_F = 0.003, 0.001, 0.5

    def __init__(self, z, rng_seed=0, converged=False):
        self.z = z
        if converged:                       # stronger, converged CEM (reported separately; VAL ident check)
            self.POP, self.ITERS, self.ELITE = 256, 20, 16
        self.rng = np.random.default_rng(rng_seed)

    def _row_inputs(self, e, w):
        z = self.z
        L = int(z["L"][e])
        xh = z["xhat"][e, w].astype(float)
        half = z["size"][e, 1:L + 1].astype(float)
        x0 = xh[:L + 1].copy()
        for i in range(1, L + 1):                       # unknown estimate: assume touching the previous entity
            if not np.isfinite(x0[i]):
                prev = x0[i - 1] + (C.PUSHER_R if i == 1 else half[i - 2])
                x0[i] = prev + half[i - 1]
        # project the noisy estimates onto a non-penetrating configuration (overlaps would launch the penalty
        # contacts of the analytic simulator)
        rad = np.r_[C.PUSHER_R, half]
        for i in range(1, L + 1):
            x0[i] = max(x0[i], x0[i - 1] + rad[i - 1] + rad[i])
        return L, x0, half

    @staticmethod
    def theta_to_params(th, L, spring):
        from physmem.models.v3.belief import LM0, LM_SC, MU0, MU_SC, LK0, LK_SC
        m = np.exp(th[:, :L] * LM_SC + LM0)
        mu = np.clip(th[:, L:2 * L] * MU_SC + MU0, 0.01, 2.0)
        k = np.exp(th[:, 2 * L] * LK_SC + LK0) if spring else np.zeros(len(th))
        return m, mu, k

    def fit(self, e, k_row):
        z = self.z
        L = int(z["L"][e]); spring = bool(z["lab_spring"][e])
        D = 2 * L + (1 if spring else 0)
        mean, sd = np.zeros(D), np.ones(D)
        if k_row == 0:
            return mean, sd
        x_rest = float(z["x_rest"][e])
        for it in range(self.ITERS):
            th = mean + sd * self.rng.standard_normal((self.POP, D))
            th[0] = mean
            m, mu, kk = self.theta_to_params(np.c_[th, np.zeros((self.POP, 1))] if not spring else th, L, spring)
            cost = 0.5 * (th ** 2).sum(1)
            for w in range(k_row):
                _, x0, half = self._row_inputs(e, w)
                F, T = z["action"][e, w]
                run = C.AnalyticChain.run(x0, half, m, mu, kk, spring, F, T, x_rest=x_rest)
                # bin the 100 Hz simulation like the evidence (32 bins over 1.6 s; first record is t=0)
                xs = run["x"][:, 1:161]
                fp = run["fp"][:, 1:161]
                xb = xs.reshape(self.POP, C.T_EV, 5, L + 1).mean(2) - x0[None, None]
                fb = fp.reshape(self.POP, C.T_EV, 5).mean(2)
                ev = z["ev"][e, w].astype(float)
                evm = z["ev_mask"][e, w]
                for i in range(L):
                    for mod, sig in ((C.MOD_TRACK, self.SIG_TRK), (C.MOD_ANCHOR, self.SIG_ANC)):
                        if evm[1 + i, mod]:
                            ok = ev[1 + i, mod, :, 5] > 0.5
                            cost += 0.5 * (((xb[:, ok, 1 + i] - ev[1 + i, mod, ok, 0]) / sig) ** 2).sum(1)
                if evm[0, C.MOD_PROPRIO]:
                    cost += 0.5 * (((fb - ev[0, C.MOD_PROPRIO, :, 2]) / self.SIG_F) ** 2).sum(1)
                    cost += 0.5 * (((xb[:, :, 0] - ev[0, C.MOD_PROPRIO, :, 0]) / 0.002) ** 2).sum(1)
            el = th[np.argsort(cost)[:self.ELITE]]
            mean = el.mean(0)
            sd = np.maximum(el.std(0), 0.02)
        return mean, sd

    def predict(self, e, k_row, th, F=None, T=None, init=None):
        """analytic prediction of window k_row from the actor's window-start estimates. th (P,D) -> x (P,NT,L+1)."""
        z = self.z
        L, x0, half = self._row_inputs(e, k_row)
        spring = bool(z["lab_spring"][e])
        th = np.atleast_2d(th)
        m, mu, kk = self.theta_to_params(np.c_[th, np.zeros((len(th), 1))] if not spring else th, L, spring)
        if F is None:
            F, T = z["action"][e, k_row]
        run = C.AnalyticChain.run(x0 if init is None else init, half, m, mu, kk, spring, F, T,
                                  x_rest=float(z["x_rest"][e]))
        return C.analytic_grid(run)


def classical(args):
    z = dict(np.load(os.path.join(args.data, f"{args.split}.npz")))
    cl = Classical(z, args.seed)
    n = len(z["L"]) if args.n is None else min(args.n, len(z["L"]))
    rows, fe, perr, pz, psd = [], [], [], [], []
    t0 = time.time()
    for e in range(n):
        L = int(z["L"][e])
        for k in range(C.N_WIN):
            mean, sd = cl.fit(e, k)
            x = cl.predict(e, k, mean)[0]                                       # (NT, L+1) absolute
            org = z["xhat"][e, k, 0]
            Xt = z["lab_X"][e, k]                                               # world
            err = np.full(C.N_ENT, np.nan)
            err[1:L + 1] = np.abs(x[-1, 1:] - Xt[1:L + 1, -1])
            # params (normalised)
            from physmem.models.v3.belief import si_to_p
            lm, mu, lk = si_to_p(z["lab_mass"][e, 1:L + 1], z["lab_mu"][e, 1:L + 1],
                                 z["lab_k"][e, L] if z["lab_spring"][e] else None)
            pe = np.full((C.N_ENT, 3), np.nan); pzz = np.full((C.N_ENT, 3), np.nan); ps = np.full((C.N_ENT, 3), np.nan)
            pe[1:L + 1, 0] = np.abs(mean[:L] - lm); pe[1:L + 1, 1] = np.abs(mean[L:2 * L] - mu)
            pzz[1:L + 1, 0] = (mean[:L] - lm) / sd[:L]; pzz[1:L + 1, 1] = (mean[L:2 * L] - mu) / sd[L:2 * L]
            ps[1:L + 1, 0] = sd[:L]; ps[1:L + 1, 1] = sd[L:2 * L]
            if z["lab_spring"][e]:
                pe[L, 2] = abs(mean[2 * L] - lk); pzz[L, 2] = (mean[2 * L] - lk) / sd[2 * L]; ps[L, 2] = sd[2 * L]
            rows.append((e, k)); fe.append(err); perr.append(pe); pz.append(pzz); psd.append(ps)
        if (e + 1) % 50 == 0:
            print(f"classical {e + 1}/{n} {time.time() - t0:.0f}s mean final err "
                  f"{np.nanmean(np.array(fe)) * 1e3:.1f} mm", flush=True)
    out = dict(rows=np.array(rows), final_err=np.array(fe), p_err=np.array(perr), p_z=np.array(pz), p_sd=np.array(psd))
    np.savez(_out(f"classical_{args.split}.npz"), **out)
    print("classical done", np.nanmean(out["final_err"]) * 1e3, "mm", flush=True)


# ----------------------------------------------------------------------------- counterfactual simulation
def true_params(z, e):
    L = int(z["L"][e])
    return C.ChainParams(L=L, mass=z["eval_mass"][e, :L].astype(float).copy(), mu=z["eval_mu"][e, :L].astype(float).copy(),
                         half=z["eval_half"][e, :L].astype(float).copy(),
                         gaps=z["eval_gaps"][e, :max(L - 1, 0)].astype(float).copy(), spring=bool(z["lab_spring"][e]),
                         k=float(z["eval_k"][e]), actions=[tuple(a) for a in z["action"][e]],
                         pusher_gap=[float(g) for g in z["eval_pgap"][e]])


def sim_window(p: C.ChainParams, z, e, k, F, T, far_from=None):
    """MuJoCo re-simulation of window k from its TRUE start state with action (F, T). far_from: boxes with index >=
    far_from (1-based box index) are moved 2 m away (edge-blocking counterfactual). -> (x grid (NT, L+1), Fint)."""
    L = p.L
    q = p.copy(actions=[(float(F), float(T))], pusher_gap=[p.pusher_gap[k]])
    x0 = z["eval_start_x"][e, k, :L + 1].astype(float).copy()
    v0 = z["eval_start_v"][e, k, :L + 1].astype(float).copy()
    if far_from is not None:
        x0[far_from:] += 2.0 + 0.2 * np.arange(L + 1 - far_from)
        v0[far_from:] = 0.0
    tr = C.simulate(q, windows=1, init_state=(x0, v0))
    X, _, _ = C.grid_truth(tr, 0)
    return X, tr["fint"][0]


def perturb(p: C.ChainParams, box, prop, sign):
    """physical counterfactual matching a normalised belief edit of +-DELTA."""
    from physmem.models.v3.belief import LM_SC, MU_SC, LK_SC
    q = p.copy()
    if prop == 0:
        q.mass[box] *= math.exp(sign * DELTA * LM_SC)
    elif prop == 1:
        q.mu[box] = max(q.mu[box] + sign * DELTA * MU_SC, 0.02)
    else:
        q.k *= math.exp(sign * DELTA * LK_SC)
    return q


def edit_plan(z, n, seed):
    """pre-registered edit draws: per episode a box, a property (friction / mass / spring k if present) and a sign;
    a goal displacement for the last box."""
    rng = np.random.default_rng(seed)
    plan = []
    for e in range(n):
        L = int(z["L"][e])
        box = int(rng.integers(L))
        props = [0, 1] + ([2] if (z["lab_spring"][e] and box == L - 1) else [])
        prop = int(props[rng.integers(len(props))])
        sign = float(rng.choice([-1.0, 1.0]))
        goal = float(rng.uniform(0.03, 0.15))
        plan.append(dict(e=e, box=box, prop=prop, sign=sign, goal=goal))
    return plan


def oracle_best_F(p, z, e, k, goal_abs, L):
    errs = []
    for F in F_GRID:
        X, _ = sim_window(p, z, e, k, F, PLAN_T)
        errs.append(abs(X[-1, L] - goal_abs))
    errs = np.array(errs)
    return float(F_GRID[int(np.argmin(errs))]), errs


def simcf(args):
    """model-independent simulator side of the causal tests (cached): factual / counterfactual re-simulation of the
    logged action, oracle action grids in both worlds, edge-blocking counterfactual. Evaluator only."""
    z = dict(np.load(os.path.join(args.data, f"{args.split}.npz")))
    n = min(args.n, len(z["L"]))
    plan = edit_plan(z, n, 12345)
    k = C.N_WIN - 1
    recs, t0 = [], time.time()
    for pl in plan:
        e = pl["e"]; L = int(z["L"][e])
        p = true_params(z, e)
        F0, T0 = z["action"][e, k]
        q = perturb(p, pl["box"], pl["prop"], pl["sign"])
        Xs0, Fs0 = sim_window(p, z, e, k, F0, T0)
        Xs1, Fs1 = sim_window(q, z, e, k, F0, T0)
        # goal: a displacement the TRUE world can reach with one grid action (>= 1 cm when possible), drawn
        # model-independently (VAL pilot: uniform goals were often infeasible, oracle error 23 mm)
        XL0 = np.array([sim_window(p, z, e, k, F, PLAN_T)[0][-1, L] for F in F_GRID])
        XL1 = np.array([sim_window(q, z, e, k, F, PLAN_T)[0][-1, L] for F in F_GRID])
        x0L = float(z["eval_start_x"][e, k, L])
        cand = np.nonzero(XL0 - x0L >= 0.01)[0]
        cand = cand if len(cand) else np.arange(len(F_GRID))
        gj = int(np.random.default_rng(54321 + e).choice(cand))
        goal_abs = float(XL0[gj])
        errs0, errs1 = np.abs(XL0 - goal_abs), np.abs(XL1 - goal_abs)
        r = dict(pl, L=L, goal_abs=goal_abs, goal_F=float(F_GRID[gj]), errs0=errs0.tolist(), errs1=errs1.tolist(),
                 sim_dx_final=(Xs1[-1, 1:] - Xs0[-1, 1:]).tolist(), sim_dx_t06=(Xs1[6, 1:] - Xs0[6, 1:]).tolist(),
                 sim_dx_t03=(Xs1[3, 1:] - Xs0[3, 1:]).tolist(), sim_dimp=((Fs1 - Fs0).sum(1) * 0.1).tolist(),
                 sim_d=(Xs0[-1, 1:] - Xs0[0, 1:]).tolist())
        if L >= 2:
            Xb, _ = sim_window(p, z, e, k, F0, T0, far_from=2)
            r["sim_blk_d"] = (Xb[-1, 1:] - Xb[0, 1:]).tolist()
        recs.append(r)
        if len(recs) % 20 == 0:
            print(f"simcf {args.split} {len(recs)}/{n} {time.time() - t0:.0f}s", flush=True)
    json.dump(recs, open(_out(f"simcf_{args.split}.json"), "w"))


def causal(args):
    """T1-T3 (edit -> intermediate / future / action / outcome), T5 (delayed evidence), T6 (edge blocking) and the
    closed-loop slide-to-target task, for one model checkpoint (or 'classical' / 'oracle')."""
    import torch
    from physmem.models.v3 import belief as B
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = dict(np.load(os.path.join(args.data, f"{args.split}.npz")))
    kind = args.ckpt
    if kind not in ("classical", "oracle", "classical_conv"):
        from physmem.v3_chain import load
        model, _ = load(args.ckpt, dev)
        run = os.path.basename(args.ckpt)[:-3]
        st = B.ChainStore(os.path.join(args.data, f"{args.split}.npz"), dev)
    else:
        model, run, st = None, kind, None
        cl = Classical(z, args.seed, converged=(kind == "classical_conv"))
    sims = json.load(open(_out(f"simcf_{args.split}.json")))
    n = min(args.n, len(sims))
    k = C.N_WIN - 1
    recs = []
    t0 = time.time()
    for sm in sims[:n]:
        e = sm["e"]
        L = int(z["L"][e])
        p = true_params(z, e)
        box, prop, sign = sm["box"], sm["prop"], sm["sign"]
        rec = {kk: v for kk, v in sm.items() if kk not in ("errs0", "errs1")}
        goal_abs = sm["goal_abs"]
        errs0, errs1 = np.array(sm["errs0"]), np.array(sm["errs1"])
        # ---- model / classical predictions: belief clamped at (posterior mean) and (posterior mean + sign*DELTA)
        org = float(z["xhat"][e, k, 0])
        if model is not None:
            r = torch.tensor([[e, k]], device=dev)
            b, lab = st.gather(r)
            ids = _ids(1, dev, 777 + e)
            base = model_predict(model, b, 4, ids)
            pmv = base.pm()[0, 1 + box, prop].item()
            def clamp(v, ent=1 + box, pr=prop):
                mm = torch.zeros(1, C.N_ENT, B.NP + 2 * B.NT, dtype=torch.bool, device=dev)
                val = torch.zeros(1, C.N_ENT, B.NP + 2 * B.NT, device=dev)
                mm[0, ent, pr] = True; val[0, ent, pr] = v
                return (mm, val)
            outs0 = model_predict(model, b, 4, ids, clamps=(clamp(pmv),), all_passes=True)
            outs1 = model_predict(model, b, 4, ids, clamps=(clamp(pmv + sign * DELTA),), all_passes=True)
            x0m = outs0[-1].xm()[0] * B.X_SC; x1m = outs1[-1].xm()[0] * B.X_SC
            rec["mod_dx_final"] = (x1m[1:L + 1, -1] - x0m[1:L + 1, -1]).tolist()
            rec["mod_dx_t06"] = (x1m[1:L + 1, 6] - x0m[1:L + 1, 6]).tolist()
            rec["mod_dx_t03"] = (x1m[1:L + 1, 3] - x0m[1:L + 1, 3]).tolist()
            rec["mod_dimp"] = (((outs1[-1].f() - outs0[-1].f())[0, :L] * B.F_SC).sum(-1) * 0.1).tolist()
            if len(outs0) > 1:
                h = len(outs0) // 2 - 1
                rec["mod_dx_final_pass_mid"] = ((outs1[h].xm() - outs0[h].xm())[0, 1:L + 1, -1] * B.X_SC).tolist()
            # planner: batch over the F grid
            bq = with_action(repeat_batch(b, len(F_GRID)), torch.as_tensor(F_GRID, dtype=torch.float32, device=dev),
                             PLAN_T)
            idq = ids.repeat_interleave(len(F_GRID), 0)
            def plan_F(cl_):
                cls = tuple((mm.repeat_interleave(len(F_GRID), 0), val.repeat_interleave(len(F_GRID), 0)) for mm, val in cl_)
                s = model_predict(model, bq, 4, idq, clamps=cls)
                pred = s.xm()[:, L, -1].cpu().numpy() * B.X_SC + org
                return float(F_GRID[int(np.argmin(np.abs(pred - goal_abs)))])
            F_plain = plan_F(())
            F_m0 = plan_F((clamp(pmv),))
            F_m1 = plan_F((clamp(pmv + sign * DELTA),))
            # T5 delayed evidence (last window's evidence available only from pass 2 of 4) vs never
            late = torch.zeros_like(b.ev_mask)
            nper = C.N_ENT * C.N_MOD
            late[:, :nper] = b.ev_mask[:, :nper]
            s_late = model_predict(model, b, 4, ids, late_ev=late, late_from=2)
            bn, _ = st.gather(r, {"drop_last_window": True})
            s_never = model_predict(model, bn, 4, ids)
            pm_t, pmask, *_ = B.label_targets(b, lab)
            pe = lambda s_: float(((s_.pm() - pm_t).abs()[pmask]).mean())
            fe = lambda s_: float(((s_.xm()[..., -1] * B.X_SC - lab.X[..., -1]).abs()[b.ent_type == C.ENT_BOX]).mean())
            rec.update(p_err_ontime=pe(base), p_err_late=pe(s_late), p_err_never=pe(s_never),
                       f_err_ontime=fe(base), f_err_late=fe(s_late), f_err_never=fe(s_never))
            # T6 edge blocking: block edge (B1, B2) (edge index 1)
            if L >= 2:
                blk = torch.zeros(1, C.N_EDGE, dtype=torch.bool, device=dev); blk[0, 1] = True
                s_blk = model_predict(model, b, 4, ids, block_edges=blk)
                xb_ = s_blk.xm()[0] * B.X_SC; xu_ = base.xm()[0] * B.X_SC
                rec["mod_blk_d"] = (xb_[1:L + 1, -1] - xb_[1:L + 1, 0]).tolist()
                rec["mod_unblk_d"] = (xu_[1:L + 1, -1] - xu_[1:L + 1, 0]).tolist()
        else:
            mean, sd = cl.fit(e, k)
            D = len(mean)
            if kind == "oracle":
                from physmem.models.v3.belief import si_to_p
                lm, mu, lk = si_to_p(p.mass, p.mu, p.k if p.spring else None)
                mean = np.r_[lm, mu, [lk] if p.spring else []]
            j = box if prop == 0 else (L + box if prop == 1 else 2 * L)
            th0 = mean.copy(); th1 = mean.copy(); th1[j] += sign * DELTA
            x0a = cl.predict(e, k, th0)[0]; x1a = cl.predict(e, k, th1)[0]
            rec["mod_dx_final"] = (x1a[-1, 1:] - x0a[-1, 1:]).tolist()
            rec["mod_dx_t06"] = (x1a[6, 1:] - x0a[6, 1:]).tolist()
            rec["mod_dx_t03"] = (x1a[3, 1:] - x0a[3, 1:]).tolist()
            def plan_F_an(th):
                xs = np.stack([cl.predict(e, k, th, F=F, T=PLAN_T)[0][-1, L] for F in F_GRID])
                return float(F_GRID[int(np.argmin(np.abs(xs - goal_abs)))])
            F_plain = plan_F_an(mean); F_m0 = F_plain; F_m1 = plan_F_an(th1)
        # ---- oracle best actions in the factual and counterfactual worlds, realized outcomes
        Fo0, Fo1 = float(F_GRID[int(np.argmin(errs0))]), float(F_GRID[int(np.argmin(errs1))])
        gi = lambda F: int(np.argmin(np.abs(F_GRID - F)))
        rec.update(F_plain=F_plain, F_m0=F_m0, F_m1=F_m1, F_or0=Fo0, F_or1=Fo1,
                   real_err_plain=float(errs0[gi(F_plain)]),             # closed loop, factual world
                   real_cf_edit=float(errs1[gi(F_m1)]), real_cf_noedit=float(errs1[gi(F_m0)]),
                   real_or_fact=float(errs0.min()), real_or_cf=float(errs1.min()))
        recs.append(rec)
        if len(recs) % 20 == 0:
            print(f"[{run}] causal {len(recs)}/{n} {time.time() - t0:.0f}s", flush=True)
    json.dump(recs, open(_out(f"{run}_{args.split}_causal.json"), "w"))
    print(json.dumps(summarize_causal(recs), indent=1), flush=True)


def _sign_agree(m, s, thr=0.002):
    m, s = np.asarray(m, float), np.asarray(s, float)
    ok = np.abs(s) > thr
    return float(np.mean(np.sign(m[ok]) == np.sign(s[ok]))) if ok.any() else float("nan"), int(ok.sum())


def summarize_causal(recs):
    out = {}
    for key in ("final", "t06", "t03"):
        own_m, own_s, oth_m, oth_s = [], [], [], []
        for r in recs:
            for i, (a, b) in enumerate(zip(r[f"mod_dx_{key}"], r[f"sim_dx_{key}"])):
                (own_m if i == r["box"] else oth_m).append(a)
                (own_s if i == r["box"] else oth_s).append(b)
        out[f"sign_own_{key}"], out[f"n_own_{key}"] = _sign_agree(own_m, own_s)
        out[f"sign_other_{key}"], out[f"n_other_{key}"] = _sign_agree(oth_m, oth_s)
        ok = np.abs(np.array(own_s)) > 0.002
        if ok.sum() > 3:
            rk = lambda a: np.argsort(np.argsort(a))
            out[f"corr_own_{key}"] = float(np.corrcoef(rk(np.array(own_m)[ok]), rk(np.array(own_s)[ok]))[0, 1])
            out[f"ratio_own_{key}"] = float(np.median(np.abs(np.array(own_m)[ok]) / np.abs(np.array(own_s)[ok])))
    if "mod_dimp" in recs[0]:
        m = [r["mod_dimp"][r["box"]] for r in recs]; s = [r["sim_dimp"][r["box"]] for r in recs]
        out["sign_imp_in"], out["n_imp_in"] = _sign_agree(m, s, 0.005)
    dF_m = np.array([r["F_m1"] - r["F_m0"] for r in recs]); dF_o = np.array([r["F_or1"] - r["F_or0"] for r in recs])
    ok = np.abs(dF_o) > 0.25
    out["sign_action"] = float(np.mean(np.sign(dF_m[ok]) == np.sign(dF_o[ok]))) if ok.any() else float("nan")
    out["n_action"] = int(ok.sum())
    d = np.array([r["real_cf_edit"] - r["real_cf_noedit"] for r in recs])
    out["real_cf_edit_minus_noedit_mm"] = float(d.mean() * 1e3)
    out["real_cf_ci_mm"] = [float(x * 1e3) for x in boot_ci(d)]
    out["closed_loop_err_mm"] = float(np.mean([r["real_err_plain"] for r in recs]) * 1e3)
    out["closed_loop_succ15"] = float(np.mean([r["real_err_plain"] < 0.015 for r in recs]))
    out["oracle_closed_loop_err_mm"] = float(np.mean([r["real_or_fact"] for r in recs]) * 1e3)
    if "p_err_late" in recs[0]:
        for kk in ("ontime", "late", "never"):
            out[f"p_err_{kk}"] = float(np.mean([r[f"p_err_{kk}"] for r in recs]))
            out[f"f_err_{kk}_mm"] = float(np.mean([r[f"f_err_{kk}"] for r in recs]) * 1e3)
    blk = [r for r in recs if "mod_blk_d" in r]
    if blk:
        q = [abs(r["mod_blk_d"][1]) <= 0.2 * abs(r["mod_unblk_d"][1]) for r in blk if abs(r["mod_unblk_d"][1]) > 0.01]
        out["edge_block_frac_B2_suppressed"] = float(np.mean(q)) if q else float("nan")
        out["n_edge_block"] = len(q)
        ag = [np.sign(r["mod_blk_d"][0] - r["mod_unblk_d"][0]) == np.sign(r["sim_blk_d"][0] - r["sim_d"][0])
              for r in blk if abs(r["sim_blk_d"][0] - r["sim_d"][0]) > 0.002]
        out["edge_block_B1_sign"] = float(np.mean(ag)) if ag else float("nan")
        out["n_edge_block_B1"] = len(ag)
    return out


def boot_ci(d, n=4000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    bs = rng.integers(0, len(d), (n, len(d)))
    m = d[bs].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--ckpt", required=True); p.add_argument("--data", default="data/v3_chain")
    p.add_argument("--split", default="test"); p.add_argument("--cond", default="full")
    p.add_argument("--passes", default="4")
    c = sub.add_parser("classical")
    c.add_argument("--data", default="data/v3_chain"); c.add_argument("--split", default="test")
    c.add_argument("--n", type=int, default=None); c.add_argument("--seed", type=int, default=0)
    sc = sub.add_parser("simcf")
    sc.add_argument("--data", default="data/v3_chain"); sc.add_argument("--split", default="test")
    sc.add_argument("--n", type=int, default=200)
    q = sub.add_parser("causal")
    q.add_argument("--ckpt", required=True); q.add_argument("--data", default="data/v3_chain")
    q.add_argument("--split", default="test"); q.add_argument("--n", type=int, default=200)
    q.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    {"predict": predict, "classical": classical, "causal": causal, "simcf": simcf}[a.cmd](a)


if __name__ == "__main__":
    main()
