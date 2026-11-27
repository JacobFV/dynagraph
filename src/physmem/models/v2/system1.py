"""MODEL-V2 as a callable System I head (round 6, for the INKLING System I/II agent). DRAFT interface v0.

System II (the LLM planner) decides WHAT to do: a SkillCall request (skill, target, rough params). System I
(this head) decides HOW: it reads the typed memory of everything the robot has touched in this episode and returns
concrete parameters, plus calibrated uncertainty and predicted outcome. It is env-free: the caller supplies the
interaction history (list of `spec.Interaction`, e.g. what `EnvAdapter.execute` logs), an actor-legal scene frame
(`spec.EntityFrame`), the embodiment spec, and the target's estimated position.

    head = System1Head.load("results/v2/checkpoints/v2_robots_r7_typed_s{}.pt", seeds=(0, 1, 2))
    out = head.propose(request, history, frame, embodiment_spec, target_pos)
    out["call"]          # SkillCall with chosen params (same skill / target as the request)
    out["uncertainty"]   # dict: score_sd (ensemble + posterior), p_success_sd, epistemic flag
    out["predicted"]     # p_success, p_slip, p_drop, p_damage, ..., duration_s (ensemble mean)
    out["memory"]        # typed latents of the target: {name: (mean, sd, unit)}  (typed models only)
    out["candidates"]    # top-k alternatives with the same fields (for System II to inspect / override)
    head.readout(history, frame, embodiment_spec, query)   # memory only, no proposal

Candidates = the request's params, multiplicative jitter of its scalar params (positional / angle keys never
scaled, as in the round-3 residual planner), and optional explicit grids (e.g. grip_force x lift_speed). Score =
P(slip) + P(drop) + P(damage) + lam_t * duration (p_success = task predicate, reported only), ranked by the upper bound mean + kappa * sd
(pessimistic, as in the LCB-gated planners of rounds 2-5). Latency: one batched forward per ensemble member.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch

from physmem.v2 import api
from . import spec as S
from .env_adapter import rel_params
from .harness import NO_SCALE_KEYS

INTERFACE = "physmem-v2-system1-v0"


class System1Head:
    def __init__(self, planners, lam_t=0.01, kappa=1.0):
        self.planners, self.lam_t, self.kappa = planners, lam_t, kappa
        self.mode = planners[0].mode

    @classmethod
    def load(cls, pattern, seeds=(0, 1, 2), mode=None, device="cpu", **kw):
        from physmem.planner import CEMPlanner, PlannerConfig
        from physmem.v2_train import load_model
        pls = []
        for s in seeds:
            m = load_model(pattern.format(s), torch.device(device))[0]
            pls.append(CEMPlanner(m, mode or m.variant, PlannerConfig(n_z=16, voi=False)))
        return cls(pls, **kw)

    # ------------------------------------------------------------------ memory readout
    @torch.no_grad()
    def readout(self, history, frame, emb_spec, query=None):
        """typed latents of the query target (ensemble-averaged posterior), physical units."""
        if self.mode != "typed":
            return {}
        emb = S.embodiment_vector(emb_spec)
        mus, lvs = [], []
        for pl in self.planners:
            _, mem = pl._prepare(history, frame, emb, query)
            mus.append(mem["z_mu"][0].cpu().numpy()); lvs.append(mem["z_logvar"][0].cpu().numpy())
        mu = np.mean(mus, 0); var = np.mean(np.exp(lvs), 0) + np.var(mus, 0)
        out = {}
        for i, (name, kind, loc, sc, unit) in enumerate(S.LATENTS):
            m_, s_ = float(mu[i]), float(np.sqrt(var[i]))
            lo, hi = S.z_to_latent(name, m_ - s_), S.z_to_latent(name, m_ + s_)
            out[name] = dict(mean=float(S.z_to_latent(name, m_)), lo=float(lo), hi=float(hi), z_sd=s_, unit=unit)
        return out

    # ------------------------------------------------------------------ proposal
    def candidates(self, request, n_jitter=48, grids=None, rng=None):
        rng = rng or np.random.default_rng(0)
        out = [request]
        keys = [k for k, v in request.params.items() if isinstance(v, (int, float)) and k not in NO_SCALE_KEYS
                and not k.startswith(("x", "y", "z", "p"))]
        for _ in range(n_jitter if keys else 0):
            p = dict(request.params)
            for k in keys:
                p[k] = float(p[k]) * float(np.exp(rng.normal(0, 0.35)))
            out.append(replace(request, params=p))
        for g in (grids or []):           # e.g. {"grip_force": [5, 10, 20], "lift_speed": [0.05, 0.1, 0.2]}
            import itertools
            ks = list(g)
            for vals in itertools.product(*[g[k] for k in ks]):
                out.append(replace(request, params=dict(request.params, **dict(zip(ks, vals)))))
        return out

    @torch.no_grad()
    def propose(self, request: api.SkillCall, history, frame, emb_spec, target_pos, query=None, grids=None,
                top_k=5, rng=None):
        t0 = time.perf_counter()
        emb = S.embodiment_vector(emb_spec)
        cands = self.candidates(request, grids=grids, rng=rng)
        calls = torch.as_tensor(np.stack([S.named_call_vector(c.skill, rel_params(c.params, np.asarray(target_pos)))
                                          for c in cands]))
        ev, PS, VS, DUR = S.OUT_EVENTS, [], [], []
        for pl in self.planners:
            ctx, mem = pl._prepare(history, frame, emb, query or request.target)
            zin, _ = pl._latent_inputs(mem)
            m = pl.model
            P, Sn = calls.shape[0], zin.shape[0]
            emb_, ent, entm = pl._scene
            n = P * Sn
            o = m.head_eval(ctx.expand(n, -1), calls.to(pl.dev).repeat_interleave(Sn, 0), zin.repeat(P, 1),
                            emb_.expand(n, -1), ent.expand(n, -1, -1), entm.expand(n, -1))
            d = m.decode(o)
            p = d["p"].reshape(P, Sn, -1)
            PS.append(p.mean(1).cpu().numpy()); VS.append(p.var(1).cpu().numpy())
            DUR.append(d["cont"][:, S.OUT_CONT.index("duration")].clamp(min=0).reshape(P, Sn).mean(1).cpu().numpy())
        Pm = np.mean(PS, 0); Pv = np.mean(VS, 0) + np.var(PS, 0); dur = np.mean(DUR, 0)
        bad = [ev.index(k) for k in ("slip", "drop", "damage")]
        # 'success' in OUT_EVENTS is the TASK predicate after the call (not the skill's own success), so it is
        # reported but not scored; the score is the call's failure-event risk + time
        score = Pm[:, bad].sum(1) + self.lam_t * dur
        sd = np.sqrt(Pv[:, bad].sum(1))
        ucb = score + self.kappa * sd
        order = np.argsort(ucb)

        def pack(i):
            return dict(call=cands[i], score=float(score[i]), score_sd=float(sd[i]), ucb=float(ucb[i]),
                        predicted={f"p_{k}": float(Pm[i, j]) for j, k in enumerate(ev)} | dict(duration_s=float(dur[i])),
                        p_sd={f"p_{k}": float(np.sqrt(Pv[i, j])) for j, k in enumerate(ev)})
        best = pack(order[0])
        return dict(interface=INTERFACE, mode=self.mode, call=best["call"], predicted=best["predicted"],
                    uncertainty=dict(score_sd=best["score_sd"], p_success_sd=best["p_sd"]["p_success"],
                                     epistemic=bool(best["score_sd"] > 0.15)),
                    memory=self.readout(history, frame, emb_spec, query or request.target),
                    candidates=[pack(i) for i in order[:top_k]], request=request,
                    latency_ms=1000 * (time.perf_counter() - t0))
