"""Model-predictive skill-parameter planner for the v2 world model (owner: MODEL-V2).

decide(): CEM over the scalar parameters of one skill with
    cost(a) = C_fail (1 - p_success) + C_prog (1 - progress) + lam_effort * log(1 + effort/effort_ref) + lam_risk * p_risk
              + lam_time * duration                                  (expected over latent samples)
              + lam_unc * std over latent samples                   (risk-averse under latent uncertainty)
Latent uncertainty: typed slots are sampled from the fused posterior and fed to the outcome head as known values;
the generic / no-memory models get one deterministic pass (their uncertainty lives only in predicted variances).

Active information gathering (typed only): expected value of perfect information
    EVPI = E_z[c(a*, z)] - E_z[min_a c(a, z)]   (a* = plan under uncertainty; min over the CEM population)
the learned information head predicts the per-latent log-variance a probe would yield; the fraction of
decision-relevant variance it removes (weights = corr^2 between each latent and the regret of a*) scales EVPI into a
VOI estimate. The probe is executed iff VOI > its predicted cost (effort + risk + time + per-step opportunity cost).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from physmem.models.v2 import spec as S
from physmem.models.v2.model import TF_LOGVAR, PhysMemV2
from physmem.models.v2.tensorize import single_batch
from physmem.v2 import api


@dataclass
class PlannerConfig:
    c_fail: float = 1.0
    c_prog: float = 0.5
    lam_effort: float = 0.1
    lam_risk: float = 1.0
    lam_time: float = 0.02
    lam_unc: float = 0.25
    pop: int = 256
    iters: int = 4
    elite_frac: float = 0.1
    n_z: int = 32
    voi: bool = True
    max_probes: int = 1
    step_cost: float = 0.05


@dataclass
class Decision:
    skill: str
    params: dict
    is_probe: bool
    info: dict = field(default_factory=dict)

    def call(self, target=None):
        return api.SkillCall(self.skill, dict(self.params), target=target, candidate_id="probe" if self.is_probe else "plan")


class CEMPlanner:
    """mode: 'typed' | 'generic' | 'nomem' | 'oracle' (typed model + true latents in the slots; labelled oracle)."""

    def __init__(self, model: PhysMemV2, mode: str = "typed", cfg: PlannerConfig = PlannerConfig(), device=None):
        self.model, self.mode, self.cfg = model, mode, cfg
        self.dev = device or next(model.parameters()).device
        self.eff_ref = float(torch.exp(model.eff_mean))
        self.probes_used = 0

    def reset(self):
        self.probes_used = 0

    # --------------------------------------------------------- model plumbing
    @torch.no_grad()
    def _prepare(self, history, frame, emb):
        m = self.model
        b = single_batch(history, frame, emb, np.zeros((1, S.CALL_DIM), np.float32), device=self.dev)
        b = m.norm_batch(b)
        ctx, tgt = m.scene(b.ent, b.ent_mask, b.target, b.emb)
        mem = m.memory(b, tgt)
        return ctx, mem

    def _latent_inputs(self, mem, true_z=None, true_mask=None, override=None):
        """-> zin (S, 2L) and the latent samples Z (S, L) (None for generic/nomem)."""
        L = S.N_LATENT
        if self.mode in ("generic", "nomem"):
            return mem["zin"], None
        mu, lv = mem["z_mu"][0], mem["z_logvar"][0]
        if override is not None:  # intervention on the typed slots: dict {latent index: (mean, logvar)}
            mu, lv = mu.clone(), lv.clone()
            for i, (a, b) in override.items():
                mu[i], lv[i] = a, b
        g = torch.Generator(device=self.dev).manual_seed(1234)
        n = self.cfg.n_z
        eps = torch.randn(n, L, device=self.dev, generator=g)
        Z = mu[None] + torch.exp(0.5 * lv)[None] * eps
        if self.mode == "oracle" and true_z is not None:
            tz = torch.as_tensor(true_z, device=self.dev, dtype=torch.float32)
            tm = torch.as_tensor(true_mask, device=self.dev, dtype=torch.bool)
            Z = torch.where(tm[None], tz[None], Z)
        zin = torch.cat([Z, torch.full_like(Z, 0.5 * TF_LOGVAR)], -1)
        return zin, Z

    @torch.no_grad()
    def _costs(self, ctx, zin, calls):
        """calls (P,24) raw -> cost (P,S) plus decoded parts averaged over S."""
        m, c = self.model, self.cfg
        P, Sn = calls.shape[0], zin.shape[0]
        cn = m.norm_call(calls)
        cr = cn.repeat_interleave(Sn, 0)
        zr = zin.repeat(P, 1)
        o = m.head(ctx.expand(P * Sn, -1), cr, zr)
        d = m.decode(o)
        prog = d["cont"][:, S.OUT_CONT.index("progress")].clamp(0, 1)
        dur = d["cont"][:, S.OUT_CONT.index("duration")].clamp(min=0)
        cost = (c.c_fail * (1 - d["success"]) + c.c_prog * (1 - prog) + c.lam_effort * torch.log1p(d["effort"] / self.eff_ref)
                + c.lam_risk * d["risk"] + c.lam_time * dur)
        parts = dict(p_success=d["success"], progress=prog, effort=d["effort"], risk=d["risk"], duration=dur)
        return cost.reshape(P, Sn), {k: v.reshape(P, Sn).mean(1) for k, v in parts.items()}

    # --------------------------------------------------------- CEM
    def _cem(self, ctx, zin, skill, bounds, rng, init_params=None):
        keys = sorted(bounds)
        lo = np.array([bounds[k][0] for k in keys], np.float64)
        hi = np.array([bounds[k][1] for k in keys], np.float64)
        c = self.cfg
        pop = rng.uniform(lo, hi, (c.pop, len(keys)))
        if init_params:
            ip = np.array([[p[k] for k in keys] for p in init_params], np.float64)
            pop[:len(ip)] = np.clip(ip, lo, hi)
        n_el = max(4, int(c.pop * c.elite_frac))
        # scale-free effort: relative to the median predicted effort of random feasible actions in this state
        self.eff_ref = 1.0
        _, p0 = self._costs(ctx, zin, self._vec(skill, keys, rng.uniform(lo, hi, (c.pop, len(keys)))))
        self.eff_ref = float(p0["effort"].median().clamp(min=1e-6))
        best, best_score, allpop, allcost = None, np.inf, [], []
        for it in range(c.iters):
            calls = self._vec(skill, keys, pop)
            cost, _ = self._costs(ctx, zin, calls)
            score = cost.mean(1) + c.lam_unc * (cost.std(1) if cost.shape[1] > 1 else 0)
            sc = score.cpu().numpy()
            allpop.append(pop.copy()); allcost.append(cost)
            order = np.argsort(sc)
            if sc[order[0]] < best_score:
                best_score, best = float(sc[order[0]]), pop[order[0]].copy()
            el = pop[order[:n_el]]
            mu, sd = el.mean(0), el.std(0) + 0.02 * (hi - lo)
            pop = np.clip(rng.normal(mu, sd, (c.pop, len(keys))), lo, hi)
            pop[0] = best
        return keys, best, best_score, np.concatenate(allpop), torch.cat(allcost)

    def _vec(self, skill, keys, P):
        P = np.atleast_2d(P)
        v = np.zeros((len(P), S.CALL_DIM), np.float32)
        v[:, S.SKILLS.index(skill)] = 1
        v[:, 10:10 + len(keys)] = P
        return torch.as_tensor(v, device=self.dev)

    # --------------------------------------------------------- decision
    @torch.no_grad()
    def decide(self, history, frame, emb, skill: str, bounds: dict, rng, probe_params: Optional[list] = None,
               true_z=None, true_mask=None, slot_override=None, init_params=None) -> Decision:
        ctx, mem = self._prepare(history, frame, emb)
        zin, Z = self._latent_inputs(mem, true_z, true_mask, slot_override)
        keys, best, best_score, allpop, allcost = self._cem(ctx, zin, skill, bounds, rng, init_params)
        params = {k: float(v) for k, v in zip(keys, best)}
        bc, parts = self._costs(ctx, zin, self._vec(skill, keys, best))
        info = dict(score=best_score, **{k: float(v[0]) for k, v in parts.items()})
        if Z is not None:
            info["z_mu"] = mem["z_mu"][0].cpu().numpy() if slot_override is None else Z.mean(0).cpu().numpy()
            info["z_std"] = torch.exp(0.5 * mem["z_logvar"][0]).cpu().numpy()
        # ---- value of information
        if (self.cfg.voi and self.mode == "typed" and Z is not None and probe_params
                and self.probes_used < self.cfg.max_probes):
            c_star = bc[0]                                                   # (S,)
            c_min = allcost.min(0).values                                    # (S,) best action per latent sample
            evpi = float((c_star - c_min).mean())
            regret = (c_star - c_min)
            zc = Z - Z.mean(0)
            rc = regret - regret.mean()
            corr2 = ((zc * rc[:, None]).mean(0) ** 2 / (zc.var(0) * rc.var() + 1e-9)).clamp(0, 1)
            w = corr2 + 1e-3
            var = torch.exp(mem["z_logvar"][0])
            best_probe, best_voi = None, -np.inf
            for pp in probe_params:
                pv = self._vec(skill, keys, np.array([[pp[k] for k in keys]]))
                info_lv = self.model.info(torch.cat([ctx, self.model.norm_call(pv)], -1))[0]
                var_new = 1.0 / (1.0 / var + torch.exp(-info_lv))
                frac = float((w * (1 - var_new / var)).sum() / w.sum())
                pc, pparts = self._costs(ctx, zin, pv)
                cpr = self.cfg
                probe_cost = float(cpr.lam_effort * math.log1p(float(pparts["effort"][0]) / self.eff_ref) + cpr.lam_risk * pparts["risk"][0]
                                   + cpr.lam_time * pparts["duration"][0] + cpr.step_cost)
                voi = evpi * frac - probe_cost
                if voi > best_voi:
                    best_voi, best_probe = voi, (pp, frac, probe_cost)
            info.update(evpi=evpi, voi=best_voi, probe_frac=best_probe[1], probe_cost=best_probe[2])
            if best_voi > 0:
                self.probes_used += 1
                return Decision(skill, {k: float(best_probe[0][k]) for k in keys}, True, info)
        return Decision(skill, params, False, info)
