"""MODEL-V2 round 4: LOT protocol + memory-conditioned probing (owner: MODEL-V2).

Lot protocol (evaluation protocol, evaluator-side scene setup through the ENVS interventions API `env.set_hidden`):
every free part of an episode comes from ONE lot and shares its density factor, grip friction, crush limit (and the
matching contact stiffness -> vision / tactile readings, re-emitted after the override). The actor only knows "these
parts are one lot" (tote label): its interactions with one part are evidence about the next, fused by the typed memory
under the opaque target id 'lot'. Lots are drawn so that crush x grip friction x mass interact: the grip window
[m (g + a) / (2 mu_g), 0.7 crush] can be narrow and needs a slower lift (smaller a) - no single scripted rule covers
every lot (fixed 40 N crushes fragile lots; probe-then-act's 0.7 x crush cap slips heavy, slippery ones at the
default lift speed).

Memory-conditioned probing (`LotPlanner`): for every pick of the scripted policy, score SKIP candidates (the scripted
call with grip x lift-speed from a grid; all other params unchanged) with a model ensemble on predicted damage + slip +
drop risk + a time cost; PROBE (the ENVS light 4 N squeeze, i.e. probe-then-act's first grasp: an extra call against
the episode budget) only if the part is not probed yet and the upper risk bound of the best skip exceeds tau.
Variants: typed (3-seed ensemble x lot-posterior samples), typed_nolot (typed, lot fusion disabled: memory ablation),
generic (3-seed generic-memory ensemble, point predictions), oracle (typed head with the TRUE lot latents, labelled).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from physmem.v2 import api
from . import spec as S
from .env_adapter import EnvAdapter, map_labels, rel_params
from .model import TF_LOGVAR
from .env_adapter import NOT_FRAGILE_N as S_NOT_FRAGILE

G = 9.81
GRIPS = (3.0, 5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0)
LIFTS = (0.2, 0.1, 0.05)
PROBE_F = 4.0


def lot_part_ids(env):
    from physmem.schema import EntityType
    return [e for e, i in env.scene.entities.items()
            if i.etype == EntityType.OBJECT and getattr(i, "movable", True) and "crush_n" in i.hidden]


MASS_FACTOR = (0.5, 2.0)      # lot density factor on each part's own mass (absolute masses eject parts: see 11a)


def sample_lot(rng, part_masses, n_try=80):
    """lot latents shared by every part: density factor f (log-U 0.5-2.0 on each part's own mass), grip friction
    0.2-0.9, fragile w.p. 0.6 (crush 8-30 N per pad) else unbreakable (stiffness 80-200 kN/m). Fragile lots are kept
    FEASIBLE for the heaviest part at the slowest lift: 2 mu 0.7 crush >= 1.15 m_max f (g + 0.7)."""
    m_max = float(max(part_masses)) if len(part_masses) else 0.2
    for _ in range(n_try):
        f = float(np.exp(rng.uniform(np.log(MASS_FACTOR[0]), np.log(MASS_FACTOR[1]))))
        mu = float(rng.uniform(0.2, 0.9))
        if rng.random() < 0.6:
            crush = float(rng.uniform(8.0, 30.0))
            if 2 * mu * 0.7 * crush < 1.15 * m_max * f * (G + 0.7):
                continue
            return dict(mass_factor=f, mu_grip=mu, crush_n=crush, fragile=True)
        return dict(mass_factor=f, mu_grip=mu, crush_n=float("inf"), stiffness=float(rng.uniform(8e4, 2e5)),
                    fragile=False)
    return dict(mass_factor=1.0, mu_grip=0.6, crush_n=float("inf"), stiffness=1.2e5, fragile=False)


class LotEnvAdapter(EnvAdapter):
    """EnvAdapter + lot protocol (lot seed = task seed; same lot for every policy of a paired task)."""

    def reset(self, task):
        """lot override + a 0.3 s settle check; a lot that destabilises the scene is redrawn (attempt seed + 1),
        deterministically, so every policy of a paired task sees the same lot (attempts recorded)."""
        for attempt in range(5):
            ev = super().reset(task)
            env = self.env
            ids = lot_part_ids(env)
            masses = [float(env.scene.entities[e].hidden["mass"]) for e in ids]
            self.lot = sample_lot(np.random.default_rng(int(task.seed) * 7 + 13 + 1000 * attempt), masses)
            if attempt == 4:
                self.lot = dict(mass_factor=1.0, mu_grip=0.6, crush_n=float("inf"), stiffness=1.2e5, fragile=False)
            ov = {k: v for k, v in self.lot.items() if k in ("mu_grip", "crush_n", "stiffness")}
            env.set_hidden({e: dict(ov, mass=m * self.lot["mass_factor"]) for e, m in zip(ids, masses)})
            env.run(0.3)
            if not getattr(env, "unstable_reasons", None):
                break
        self.lot["attempts"] = attempt + 1
        # re-emit the one-shot vision material readings so they reflect the lot's stiffness (latest reading wins)
        env.sensors.initial(env.t)
        ev2 = env.sensors.flush()
        env.view.add(ev2)
        self.idx.add(ev2)
        self.lot_ids = set(ids)
        return ev + ev2


def scripted_pick_probe(env, call):
    """ENVS probe-then-act on a scripted pick (light squeeze first, then <= 0.7 x estimated crush)."""
    return env.probe_then_act(call) if call is not None else None


def _probed(env, eid):
    k, src = env.est_stiffness(eid)
    return src == "tactile_squeeze"


class LotPlanner:
    """memory-conditioned probing + grip / lift choice. planners: list of CEMPlanner (ensemble, same mode)."""

    def __init__(self, planners, tau=0.15, kappa=1.0, lam_t=0.01, nolot=False, name="typed", crush_gate=False,
                 sigma_probe=1.0):
        """crush_gate (round-4 EXPLORATORY v2, 11e): use the typed crush posterior explicitly - probe only while the
        lot's crush is uncertain (z-sd > sigma_probe) and restrict grips to <= 0.7 x the posterior 10th percentile.
        Needs typed slots (typed / oracle); the generic model has no posterior and keeps the risk rule."""
        self.planners, self.tau, self.kappa, self.lam_t, self.nolot, self.name = planners, tau, kappa, lam_t, nolot, name
        self.mode = planners[0].mode
        self.search = "lot"
        self.crush_gate = crush_gate and self.mode in ("typed", "oracle")
        self.sigma_probe = sigma_probe
        self.cached = False         # set by cached_estimator(): classical analytic lot estimate (audit baseline, 13a)

    @classmethod
    def cached_estimator(cls, planners, tau=0.15, name="cached_estimator"):
        """CLASSICAL CACHED ESTIMATOR (13a secondary, external-audit request). Probe once per lot; the crush limit is
        fitted ANALYTICALLY from the actor-legal tactile squeeze reading (env.est_crush = k / STIFF_PER_N, no learned
        model) and cached for the whole lot. Then the same planner as typed v2: the same crush gate (probe while
        uncertain, grips <= 0.7 x the 10th percentile), the same candidate grid, risk head, tau and time cost. The risk
        head gets the cached crush in the crush slot and PRIOR means in every other slot, so there is no learned
        memory."""
        p = cls(planners, tau=tau, name=name, crush_gate=True)
        p.crush_gate, p.cached = True, True
        return p

    def _cached_z(self, ad):
        """(z mean, z sd) of the lot's crush limit from the cached analytic estimate; no reading -> prior, very wide.
        Tactile reading noise is lognormal sigma 0.15 (ENVS r6), i.e. z sd = 0.15 / 0.8 on the crush latent scale."""
        env = ad.env
        for e in sorted(getattr(ad, "lot_ids", ())):
            if _probed(env, e):
                c = env.est_crush(e, margin=1.0)
                if c is None:
                    continue
                c = min(float(c), S_NOT_FRAGILE)
                return float(S.latent_to_z("crush", c)), 0.15 / 0.8
        return 0.0, 9.0

    def _crush_posterior(self, ad, hist, emb, target, query):
        """ensemble-averaged typed crush posterior (z mean, z sd) for the query (lot or part); oracle: truth, sd 0."""
        ci = S.LATENT_INDEX["crush"]
        if self.cached:
            return self._cached_z(ad)
        if self.mode == "oracle":
            lab = ad.oracle_latents(target)
            c = lab.get("crush")
            return (float(S.latent_to_z("crush", c)), 0.0) if c is not None else (0.0, 9.0)
        mus, lvs = [], []
        for pl in self.planners:
            ctx, mem = pl._prepare(hist, ad.frame(target), emb, query)
            mus.append(float(mem["z_mu"][0, ci]))
            lvs.append(float(mem["z_logvar"][0, ci]))
        mu = float(np.mean(mus))
        var = float(np.mean(np.exp(lvs)) + np.var(mus))
        return mu, math.sqrt(var)

    def reset(self):
        pass

    @torch.no_grad()
    def _risk(self, pl, ad, hist, emb, call_target, calls_np, query):
        """per candidate: risk = P(damage) + P(slip) + P(drop), time; samples over typed-slot posterior / truth."""
        m = pl.model
        frame = ad.frame(call_target)
        ctx, mem = pl._prepare(hist, frame, emb, query)
        if pl.mode == "oracle":
            tz, tm = S.labels_to_z(ad.oracle_latents(call_target))
            mu = mem["z_mu"][0].clone()
            tzt = torch.as_tensor(tz, device=pl.dev)
            mu = torch.where(torch.as_tensor(tm, device=pl.dev), tzt, mu)
            zin = torch.cat([mu, torch.full_like(mu, 0.5 * TF_LOGVAR)])[None]
        elif self.cached:          # prior means everywhere, the cached analytic crush estimate in the crush slot
            mu = mem["prior_mu"][0].clone()
            zc, _ = self._cached_z(ad)
            if zc != 0.0:
                mu[S.LATENT_INDEX["crush"]] = zc
            zin = torch.cat([mu, torch.full_like(mu, 0.5 * TF_LOGVAR)])[None]
        else:
            zin, _ = pl._latent_inputs(mem)
        calls = torch.as_tensor(calls_np, device=pl.dev)
        P, Sn = calls.shape[0], zin.shape[0]
        emb_, ent, entm = pl._scene
        n = P * Sn
        o = m.head_eval(ctx.expand(n, -1), calls.repeat_interleave(Sn, 0), zin.repeat(P, 1), emb_.expand(n, -1),
                        ent.expand(n, -1, -1), entm.expand(n, -1))
        d = m.decode(o)
        ev = S.OUT_EVENTS
        p = d["p"]
        risk = (p[:, ev.index("damage")] + p[:, ev.index("slip")] + p[:, ev.index("drop")]).clamp(0, 1)
        dur = d["cont"][:, S.OUT_CONT.index("duration")].clamp(min=0)
        risk = risk.reshape(P, Sn)
        return risk.mean(1).cpu().numpy(), risk.var(1).cpu().numpy(), dur.reshape(P, Sn).mean(1).cpu().numpy()

    def decide(self, ad, hist, emb, rng):
        env = ad.env
        sc = env.scripted()
        if sc is None:
            return None, {}
        if sc.skill != "pick" or "grip_force" not in sc.params or sc.target not in getattr(ad, "lot_ids", set()):
            return sc, dict(kind="scripted")
        p0 = ad.target_pos(sc.target)
        query = sc.target if self.nolot else ad._tid(ad.target_ids(sc.target), sc.target)
        grips = GRIPS
        if self.crush_gate:
            zmu, zsd = self._crush_posterior(ad, hist, emb, sc.target, query)
            if zsd > self.sigma_probe and not _probed(env, sc.target):
                return scripted_pick_probe(env, sc), dict(kind="probe", crush_z_sd=zsd)
            c_lo = float(S.z_to_latent("crush", zmu - 1.2816 * zsd))
            if c_lo < 300.0:                       # believed fragile: never squeeze above 0.7 x the lower bound
                allowed = tuple(g for g in GRIPS if g <= 0.7 * c_lo)
                grips = allowed or (GRIPS[0],)
        cands = []
        for g in grips:
            for ls in LIFTS:
                c = api.SkillCall(sc.skill, dict(sc.params, grip_force=g, lift_speed=ls), target=sc.target,
                                  effector=sc.effector)
                cands.append(c)
        calls_np = np.stack([S.named_call_vector(c.skill, rel_params(c.params, p0)) for c in cands])
        R, V, D = [], [], []
        for pl in self.planners:
            r, v, dur = self._risk(pl, ad, hist, emb, sc.target, calls_np, query)
            R.append(r); V.append(v); D.append(dur)
        R, V, D = np.array(R), np.array(V), np.array(D)
        mean = R.mean(0)
        sd = np.sqrt(R.var(0) + V.mean(0))                     # ensemble disagreement + lot-posterior spread
        J = mean + self.lam_t * D.mean(0)
        j = int(np.argmin(J))
        ucb = float(mean[j] + self.kappa * sd[j])
        info = dict(kind="skip", risk=float(mean[j]), ucb=ucb, grip=cands[j].params["grip_force"],
                    lift=cands[j].params["lift_speed"])
        if not self.crush_gate and ucb > self.tau and not _probed(env, sc.target):
            info["kind"] = "probe"
            return scripted_pick_probe(env, sc), info
        return cands[j], info
