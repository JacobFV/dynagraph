"""G3/G4/G5 evaluation harness (owner: MODEL-V2).

* closed-loop paired episodes through api.IndustrialEnv (SynthEnv now; ENVS stations later): policies
  heuristic | planner(typed) | planner(generic) | planner(nomem) | planner(oracle, labelled); same task seeds for all.
* per-episode: success, steps, total EffortMeter cost, risk events, duration, probes, latent-tracking error.
* bootstrap CIs over episodes for rates/means and PAIRED differences vs each baseline.
* G4 intervention tests: (a) model slot finite difference vs simulator finite difference (oracle-context, one latent
  perturbed at a time); (b) closed-loop: plan with an overwritten slot, execute in the simulator, check the action
  and the realised outcome move in the physically expected direction.
"""
from __future__ import annotations

import copy
import math
from typing import Callable, Optional

import numpy as np
import torch

from physmem.planner import CEMPlanner, PlannerConfig
from physmem.v2 import api
from . import spec as S
from . import synth as Y
from .model import TF_LOGVAR


def interaction_from_step(call: api.SkillCall, out: api.StepOut, k: int, family: str, robot: str, t_end: float):
    """actor-side Interaction from a StepOut (no labels)."""
    it = out.result.log.get("interaction")
    if it is not None:
        it = copy.copy(it)
        it.labels = {}
        return it
    seg = out.result.log["segment"]
    return S.Interaction(episode_id="", step=k, family=family, robot=robot, call_vec=call.vector(), skill=call.skill,
                         params=dict(call.params), target_is_query=True, segment=np.asarray(seg["x"]),
                         effort_cost=out.result.effort.cost(), t_end=t_end)


def bootstrap_ci(x, n=2000, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    r = np.random.default_rng(seed)
    bs = [stat(x[r.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


RISK_KEYS = ("drop", "jam", "collision", "stall", "violation")


def decide_env_ranked(planner: CEMPlanner, ad, hist, emb, rng, n_cand=16, n_pert=15):
    """support-constrained decision on real stations: score ONLY actions inside the data-collection support —
    the env's scripted call, n_pert perturbations of it (scalar params x U(0.7, 1.3), as in the ENVS mixed policy)
    and n_cand random feasible candidates — with the same model cost; no CEM extrapolation."""
    from .env_adapter import rel_params
    pool = list(ad.candidate_skills(None, n_cand, rng))
    sc = ad.env.scripted() if hasattr(ad.env, "scripted") else None
    if sc is not None:
        pool.append(sc)
        for _ in range(n_pert):
            pool.append(api.SkillCall(sc.skill, {k: (v * float(rng.uniform(0.7, 1.3)) if isinstance(v, (int, float))
                                                      and not isinstance(v, bool) else v) for k, v in sc.params.items()},
                                      target=sc.target, effector=sc.effector))
    if not pool:
        return None
    best, sc_score = None, None
    groups = {}
    for c in pool:
        groups.setdefault((c.skill, c.target), []).append(c)
    for (skill, target), cs in groups.items():
        p0 = ad.target_pos(target)
        qt = (lambda t: t[0] if len(t) == 1 else f"pile:{target}")(ad.target_ids(target))
        true_z, true_m = S.labels_to_z(ad.oracle_latents(target)) if planner.mode == "oracle" else (None, None)
        ctx, mem = planner._prepare(hist, ad.frame(target), emb, qt)
        zin, Z = planner._latent_inputs(mem, true_z, true_m)
        calls = torch.as_tensor(np.stack([S.named_call_vector(skill, rel_params(c.params, p0)) for c in cs]),
                                device=planner.dev)
        planner.eff_ref = 1.0
        _, p_ = planner._costs(ctx, zin, calls)
        planner.eff_ref = float(p_["effort"].median().clamp(min=1e-6))
        cost, parts = planner._costs(ctx, zin, calls)
        score = cost.mean(1) + planner.cfg.lam_unc * (cost.std(1) if cost.shape[1] > 1 else 0)
        j = int(torch.nan_to_num(score, nan=1e9).argmin())
        if sc is not None:
            for i_, c_ in enumerate(cs):
                if c_ is sc:
                    sc_score = (float(score[i_]), sc, {k: float(v[i_]) for k, v in parts.items()})
        if best is None or float(score[j]) < best[0]:
            best = (float(score[j]), cs[j], {k: float(v[j]) for k, v in parts.items()})
    # scripted-anchored: deviate from the scripted action only for a clear predicted improvement
    anchor = getattr(planner, "anchor_margin", None)
    if anchor is not None and sc_score is not None and best[0] > sc_score[0] - anchor:
        return sc_score
    return best


class ResidualPlanner:
    """safe residual planner on real stations (lead request): the scripted action is the default; candidates are
    the scripted call, small perturbations of it and the env's feasible candidates; each is scored by an ENSEMBLE of
    models (independent training seeds). Deviate from scripted only when the lower confidence bound of the predicted
    improvement, mean_m(dC) - k * std_m(dC), is > 0 (epistemic uncertainty = ensemble disagreement)."""

    def __init__(self, planners, k=1.0, n_pert=15, n_cand=16, pert=0.15, trust=None):
        """trust: optional {family: bool} competence gate from VALIDATION calibration (results/v2/trust_*.json):
        in untrusted families the planner never deviates from the scripted action."""
        self.planners, self.k, self.n_pert, self.n_cand, self.pert = planners, k, n_pert, n_cand, pert
        self.trust = trust
        self.mode = planners[0].mode
        self.search = "residual"
        self.log = []

    def reset(self):
        for p in self.planners:
            p.reset()

    def decide(self, ad, hist, emb, rng):
        from .env_adapter import rel_params
        sc = ad.env.scripted() if hasattr(ad.env, "scripted") else None
        if sc is not None and self.trust is not None and not self.trust.get(ad.task.family, False):
            return sc, dict(deviated=False, gated=True)
        pool = list(ad.candidate_skills(None, self.n_cand, rng))
        if sc is None:
            return pool[0] if pool else None, {}
        pool = [sc] + [api.SkillCall(sc.skill, {k: (v * float(rng.uniform(1 - self.pert, 1 + self.pert)) if isinstance(v, (int, float))
                                                    and not isinstance(v, bool) else v) for k, v in sc.params.items()},
                                     target=sc.target, effector=sc.effector) for _ in range(self.n_pert)] + pool
        scores = np.full((len(self.planners), len(pool)), np.nan)
        lat_sd = np.zeros((len(self.planners), len(pool)))
        for (skill, target) in {(c.skill, c.target) for c in pool}:
            idx = [i for i, c in enumerate(pool) if c.skill == skill and c.target == target]
            p0 = ad.target_pos(target)
            qt = (lambda t: t[0] if len(t) == 1 else f"pile:{target}")(ad.target_ids(target))
            frame = ad.frame(target)
            calls = np.stack([S.named_call_vector(skill, rel_params(pool[i].params, p0)) for i in idx])
            for m, pl in enumerate(self.planners):
                true_z, true_m = S.labels_to_z(ad.oracle_latents(target)) if pl.mode == "oracle" else (None, None)
                ctx, mem = pl._prepare(hist, frame, emb, qt)
                zin, Z = pl._latent_inputs(mem, true_z, true_m)
                ct = torch.as_tensor(calls, device=pl.dev)
                pl.eff_ref = 1.0
                _, p_ = pl._costs(ctx, zin, ct)
                pl.eff_ref = float(p_["effort"].median().clamp(min=1e-6))
                cost, _ = pl._costs(ctx, zin, ct)
                sc_ = cost.mean(1) + pl.cfg.lam_unc * (cost.std(1) if cost.shape[1] > 1 else 0)
                scores[m, idx] = torch.nan_to_num(sc_, nan=1e9).cpu().numpy()
                if cost.shape[1] > 1:
                    lat_sd[m, idx] = cost.std(1).cpu().numpy()
        d = scores[:, :1] - scores                       # improvement over scripted, per model
        ens_sd = d.std(0) if len(self.planners) > 1 else np.zeros(d.shape[1])
        sd = np.sqrt(ens_sd ** 2 + (lat_sd.mean(0) ** 2 + lat_sd.mean(0)[0] ** 2))   # epistemic + latent spread
        lcb = d.mean(0) - self.k * sd
        lcb[0] = 0.0
        j = int(np.nanargmax(lcb))
        if not lcb[j] > 0:
            j = 0
        info = dict(chosen=j, deviated=j != 0, lcb=float(lcb[j]), mean_impr=float(d.mean(0)[j]),
                    scripted=dict(sc.params), chosen_params=dict(pool[j].params), chosen_skill=pool[j].skill)
        self.log.append(info)
        return pool[j], info


def decide_env(planner: CEMPlanner, ad, hist, emb, rng, n_cand=32):
    """real-station decision: group the env's candidates by (skill, target); CEM per group over target-relative scalar
    params (bounds = group candidate range, population seeded with them); return the best (call, is_probe, info)."""
    from .env_adapter import rel_params
    cands = ad.candidate_skills(None, n_cand, rng)
    groups = {}
    for c in cands:
        groups.setdefault((c.skill, c.target), []).append(c)
    best = None
    for (skill, target), cs in groups.items():
        p0 = ad.target_pos(target)
        keys = sorted(k for k, v in cs[0].params.items() if np.isscalar(v))
        rel = [rel_params(c.params, p0) for c in cs]
        vals = np.array([[r[k] for k in keys] for r in rel])
        bounds = {k: (float(lo), float(hi) if hi > lo else float(lo) + 1e-6) for k, lo, hi in zip(keys, vals.min(0), vals.max(0))}
        true_z, true_m = S.labels_to_z(ad.oracle_latents(target)) if planner.mode == "oracle" else (None, None)
        d = planner.decide(hist, ad.frame(target), emb, skill, bounds, rng, probe_params=None, true_z=true_z,
                           true_mask=true_m, init_params=rel, query_target=(lambda t: t[0] if len(t) == 1 else f"pile:{target}")(ad.target_ids(target)))
        if best is None or d.info["score"] < best[0]:
            best = (d.info["score"], api.SkillCall(skill, d.params, target=target), cs[0], d)
    return best


def run_episode(env, task: api.TaskSpec, policy: str, planner: Optional[CEMPlanner] = None, rng_seed=0,
                slot_override=None, max_steps=None, record: Optional[list] = None, explore: float = 0.0):
    """record: if a list is given, append dict(frame, inter) per step WITH evaluator labels (training data only).
    explore: per-step probability of replacing the planned action by a random feasible candidate (on-policy data)."""
    from .env_adapter import EnvAdapter
    if isinstance(env, EnvAdapter):
        return run_env_episode(env, task, policy, planner, rng_seed, max_steps, record=record, explore=explore)
    env.reset(task)
    fam = task.family
    emb = S.embodiment_vector(Y.ROBOTS[task.robot].spec) if task.robot in Y.ROBOTS else None
    if hasattr(env, "embodiment_spec"):
        emb = S.embodiment_vector(env.embodiment_spec())
    rng = np.random.default_rng(rng_seed)
    hist, rec = [], dict(success=False, steps=0, effort=0.0, risk_events=0, duration=0.0, probes=0, latent_err=[])
    if planner is not None:
        planner.reset()
    true_z, true_m = S.labels_to_z(getattr(env, "lat", {}))
    t = 0.0
    lot = int(task.params.get("lot", 1))
    for k in range(max_steps or Y.MAX_STEPS.get(fam, 6) * lot):
        frame = env.frame()
        skill = Y.FAMILY_SKILL.get(fam)
        if policy == "heuristic":
            params = (env.heuristic_params() if hasattr(env, "heuristic_params")
                      else Y.heuristic_params(fam, env.state, frame, None, rng))
            call, probe = api.SkillCall(skill, params, target="target"), False
        elif policy == "random":
            call, probe = env.candidate_skills(None, 1, rng)[0], False
        else:
            probes = [c.params for c in env.probe_skills()] if hasattr(env, "probe_skills") else None
            d = planner.decide(hist, frame, emb, skill, env.param_bounds(), rng, probe_params=probes,
                               true_z=true_z, true_mask=true_m, slot_override=slot_override)
            call, probe = d.call("target"), d.is_probe
            if "z_mu" in d.info and true_m.any():
                rec["latent_err"].append(float(np.abs(d.info["z_mu"][true_m] - true_z[true_m]).mean()))
        if explore > 0 and rng.random() < explore:
            call, probe = env.candidate_skills(None, 1, rng)[0], False
        out = env.execute(call)
        if record is not None:
            it = copy.copy(out.result.log["interaction"])
            it.labels = dict(getattr(env, "lat", {}))
            record.append(dict(frame=frame, inter=it))
        t += out.result.duration_s + 0.5
        hist.append(interaction_from_step(call, out, k, fam, task.robot, t))
        rec["steps"] += 1
        rec["probes"] += int(probe)
        rec["effort"] += out.result.effort.cost()
        rec["duration"] += out.result.duration_s
        rec["risk_events"] += int(sum(int(out.metrics.get(kk, 0) or 0) > 0 for kk in RISK_KEYS))
        rec["items_done"] = int(out.metrics.get("items_done", int(out.success)))
        if out.success:
            rec["success"] = True
        if out.done:
            break
    return rec


def run_env_episode(ad, task, policy, planner=None, rng_seed=0, max_steps=None, record=None, explore=0.0):
    """closed loop on an ENVS station through the adapter (policies: heuristic = env.scripted(), random, planner)."""
    ad.reset(task)
    rng = np.random.default_rng(rng_seed)
    spec = ad.embodiment_spec()
    emb = S.embodiment_vector(spec) if spec is not None else np.zeros(S.EMB_DIM, np.float32)
    hist, rec = [], dict(success=False, steps=0, effort=0.0, risk_events=0, duration=0.0, probes=0, latent_err=[])
    if planner is not None:
        planner.reset()
    for k in range(max_steps or getattr(ad.env, "budget", 20)):
        if policy == "heuristic":
            call = ad.env.scripted()
            if call is None:
                break
            out = ad.execute(call, relative=False)
        elif policy == "random":
            cs = ad.candidate_skills(None, 8, rng)
            if not cs:
                break
            out = ad.execute(cs[int(rng.integers(len(cs)))], relative=False)
        elif explore > 0 and rng.random() < explore:
            cs = ad.candidate_skills(None, 8, rng)
            if not cs:
                break
            c0 = cs[int(rng.integers(len(cs)))]
            frame0 = ad.frame(c0.target)
            out = ad.execute(c0, relative=False)
        elif getattr(planner, "search", "cem") == "residual":
            call, info = planner.decide(ad, hist, emb, rng)
            if call is None:
                break
            frame0 = ad.frame(call.target)
            out = ad.execute(call, relative=False)
            rec["deviations"] = rec.get("deviations", 0) + int(info.get("deviated", False))
        elif getattr(planner, "search", "cem") == "ranked":
            best = decide_env_ranked(planner, ad, hist, emb, rng)
            if best is None:
                break
            frame0 = ad.frame(best[1].target)
            out = ad.execute(best[1], relative=False)
        else:
            score, call, base, d = decide_env(planner, ad, hist, emb, rng)
            frame0 = ad.frame(call.target)
            out = ad.execute(call, relative=True, base=base)
            lat = out.result.log.get("latents", {})
            if "z_mu" in d.info and lat:
                tz, tm = S.labels_to_z(lat)
                if tm.any():
                    rec["latent_err"].append(float(np.abs(d.info["z_mu"][tm] - tz[tm]).mean()))
        hist.append(out.result.log["interaction"])
        if record is not None and policy != "heuristic":
            it = copy.copy(hist[-1])
            it.labels = dict(out.result.log.get("latents", {}))
            record.append(dict(frame=frame0 if policy != "random" else ad.frame(None), inter=it))
        rec["steps"] += 1
        rec["effort"] += out.result.effort.cost()
        rec["duration"] += out.result.duration_s
        ev = hist[-1].outcome_events
        rec["risk_events"] += int(ev[2] + ev[4] + ev[5] > 0)
        if out.success:
            rec["success"] = True
        if out.done:
            break
    return rec


def paired_eval(env_factory: Callable, tasks, policies: dict, seed=0):
    """policies: {name: (policy_str, planner_or_None)} -> per-policy episode records (same tasks, same order)."""
    res = {name: [] for name in policies}
    for i, task in enumerate(tasks):
        for name, (pol, pl) in policies.items():
            res[name].append(run_episode(env_factory(), task, pol, pl, rng_seed=seed * 100003 + i))
    return res


def summarize(res: dict, ref: str = "planner_typed"):
    out = {}
    for name, eps in res.items():
        s = np.array([e["success"] for e in eps], float)
        eff = np.array([e["effort"] for e in eps])
        risk = np.array([e["risk_events"] for e in eps], float)
        out[name] = dict(n=len(eps), success=bootstrap_ci(s), effort=bootstrap_ci(eff), effort_median=float(np.median(eff)),
                         risk_events=bootstrap_ci(risk), steps=float(np.mean([e["steps"] for e in eps])),
                         duration=float(np.mean([e["duration"] for e in eps])),
                         probes=float(np.mean([e["probes"] for e in eps])),
                         items_done=bootstrap_ci([e.get("items_done", int(e["success"])) for e in eps]),
                         effort_per_item=float(np.sum(eff) / max(1, sum(e.get("items_done", int(e["success"])) for e in eps))))
        if ref in res and name != ref:
            r = res[ref]
            ds = np.array([a["success"] for a in r], float) - s
            de = np.array([a["effort"] for a in r]) - eff
            out[name][f"paired_success_{ref}_minus_this"] = bootstrap_ci(ds)
            out[name][f"paired_effort_{ref}_minus_this"] = bootstrap_ci(de)
            out[name][f"paired_log_effort_ratio_{ref}_over_this"] = bootstrap_ci(np.log(np.array([a["effort"] for a in r]) + 1e-3)
                                                                            - np.log(eff + 1e-3))
    return out


# ----------------------------------------------------------------- G4 (a): model vs simulator finite differences
FD_OUTCOME = {  # family -> (outcome key, extractor from StepSim)
    "slide": ("travel", lambda st: float(st.cont[0])),
    "lift": ("success", lambda st: float(st.events[0])),
    "insert": ("success", lambda st: float(st.events[0])),
    "drawer": ("opening", lambda st: float(st.cont[0])),
    "sweep": ("progress", lambda st: float(st.cont[3])),
}


def _model_outcome(fam, d):
    if fam in ("slide", "drawer"):
        return d["cont"][:, 0]
    if fam == "sweep":
        return d["cont"][:, 3]
    return d["success"]


PHYS_MIN = {"friction": 0.03, "grip_friction": 0.05, "bulk_friction": 0.05}


@torch.no_grad()
def intervention_fd(model, family, n_states=200, delta=0.5, R=16, seed=0, robot="synth_arm_a", device=None):
    """for random (state, history, action): perturb ONE typed slot by +-delta (normalized units) and compare the
    model's predicted outcome change against the simulator's expected outcome change (mean over R noise seeds),
    other latents at their true values in both. returns per-latent sign agreement / corr / magnitude ratio."""
    from .tensorize import single_batch
    dev = device or next(model.parameters()).device
    rng = np.random.default_rng(seed)
    key, ext = FD_OUTCOME[family]
    out = {}
    for lname in Y.FAMILY_LATENTS[family]:
        li = S.LATENT_INDEX[lname]
        dm, ds = [], []
        for i in range(n_states):
            s0 = int(rng.integers(0, 2**31))
            ep = Y.generate_episode(family, robot, s0, n_steps=int(rng.integers(1, 4)))
            k = len(ep.steps) - 1
            hist = [st["inter"] for st in ep.steps[:k]]
            # re-simulate the state at step k
            r2 = np.random.default_rng(s0)
            lat = Y.sample_latents(family, r2)
            state = Y.init_state(family, r2, lat)
            bias = r2.normal(0, Y.ROBOTS[robot].ft_bias, 6)
            for st in ep.steps[:k]:
                Y.STEP[family](state, lat, st["inter"].params, Y.ROBOTS[robot], np.random.default_rng(1), bias)
            params = Y.sample_params(family, rng) if rng.random() < 0.5 else Y.heuristic_params(family, state, None, None, rng)
            if family == "slide":
                params["dir"] = float(np.sign(state["remaining"]) or 1)
            if family == "drawer":
                params["force"] = abs(params["force"]) * float(np.sign(state["goal"] - state["x"]) or 1)
            z0, zm = S.labels_to_z(lat)
            ys, zpert = [], []
            for sgn in (-1, 1):
                zz = z0.copy()
                zz[li] += sgn * delta
                lat2 = dict(lat)
                v = float(S.z_to_latent(lname, zz[li]))
                if lname in PHYS_MIN:                     # keep perturbed latents physical (e.g. friction > 0)
                    v = max(v, PHYS_MIN[lname])
                lat2[lname] = v
                zpert.append(float(S.latent_to_z(lname, v)))
                vals = []
                for r in range(R):
                    stc = copy.deepcopy(state)
                    sim = Y.STEP[family](stc, lat2, params, Y.ROBOTS[robot], np.random.default_rng(1000 + r), bias)
                    vals.append(ext(sim))
                ys.append(np.mean(vals))
            ds.append(ys[1] - ys[0])
            call = S.named_call_vector(Y.FAMILY_SKILL[family], params)[None]
            frame = Y.entity_frame(family, state, robot)
            emb = S.embodiment_vector(Y.ROBOTS[robot].spec)
            b = single_batch(hist, frame, emb, np.repeat(call, 2, 0), device=dev)
            Z = torch.as_tensor(np.stack([z0, z0]), device=dev)
            Z[0, li] = zpert[0]
            Z[1, li] = zpert[1]
            M = torch.as_tensor(np.stack([zm, zm]), device=dev)
            M[:, li] = True
            o = model(b, z_override=(Z, torch.full_like(Z, TF_LOGVAR)), z_mask_override=M)
            pm = _model_outcome(family, model.decode(o)).cpu().numpy()
            dm.append(pm[1] - pm[0])
        dm, ds = np.array(dm), np.array(ds)
        sig = np.abs(ds) > (0.05 if key in ("success", "progress") else 0.005)
        out[lname] = dict(outcome=key, n=len(ds), n_sig=int(sig.sum()),
                          sign_agree=float(np.mean(np.sign(dm[sig]) == np.sign(ds[sig]))) if sig.any() else float("nan"),
                          corr=float(np.corrcoef(dm, ds)[0, 1]) if ds.std() > 0 and dm.std() > 0 else float("nan"),
                          mean_abs_sim=float(np.abs(ds).mean()), mean_abs_model=float(np.abs(dm).mean()),
                          ratio_median=float(np.median(dm[sig] / ds[sig])) if sig.any() else float("nan"),
                          slope=float((dm * ds).sum() / max((ds * ds).sum(), 1e-12)))
    return out


# ----------------------------------------------------------------- G4 (b): closed-loop slot intervention
CL_TESTS = {  # family -> (latent, action param to watch, expected sign of d(param)/d(latent), realised outcome)
    "lift": ("mass", "grip_force", +1, "success"),
    "slide": ("friction", None, +1, "travel"),
    "drawer": ("damping", None, +1, "opening"),
    "insert": ("offset_x", "aim_x", +1, "success"),
    "sweep": ("bulk_friction", None, +1, "progress"),
}


def closed_loop_intervention(planner: CEMPlanner, family, n=60, delta=1.0, robot="synth_arm_a", seed=0):
    """plan from the same state with the typed slot overwritten to (posterior mean -/+ delta), execute both plans in the
    simulator (same noise) and report action / realised-outcome differences with expected signs."""
    lname, pkey, sgn, okey = CL_TESTS[family]
    li = S.LATENT_INDEX[lname]
    rng = np.random.default_rng(seed)
    ext = FD_OUTCOME[family][1] if FD_OUTCOME[family][0] == okey else (lambda st: float(st.cont[0]))
    da, do = [], []
    emb = S.embodiment_vector(Y.ROBOTS[robot].spec)
    for i in range(n):
        s0 = int(rng.integers(0, 2**31))
        r2 = np.random.default_rng(s0)
        lat = Y.sample_latents(family, r2)
        state = Y.init_state(family, r2, lat)
        bias = r2.normal(0, Y.ROBOTS[robot].ft_bias, 6)
        frame = Y.entity_frame(family, state, robot)
        planner.reset()
        planner.cfg.voi = False
        ctx, mem = planner._prepare([], frame, emb)
        mu = float(mem["z_mu"][0, li])
        res = []
        for s in (-1, 1):
            d = planner.decide([], frame, emb, Y.FAMILY_SKILL[family], Y.PARAM_BOUNDS[family], np.random.default_rng(5),
                               slot_override={li: (mu + s * delta, TF_LOGVAR)})
            stc = copy.deepcopy(state)
            sim = Y.STEP[family](stc, lat, d.params, Y.ROBOTS[robot], np.random.default_rng(77), bias)
            res.append((d.params, ext(sim)))
        if pkey:
            da.append(res[1][0][pkey] - res[0][0][pkey])
        do.append(res[1][1] - res[0][1])
    planner.cfg.voi = True
    da, do = np.array(da), np.array(do)
    r = dict(latent=lname, delta=delta, n=n, outcome=okey, outcome_diff_mean=float(do.mean()),
             outcome_sign_frac=float(np.mean(np.sign(do[do != 0]) == sgn)) if (do != 0).any() else float("nan"))
    if pkey:
        r.update(param=pkey, param_diff_mean=float(da.mean()),
                 param_sign_frac=float(np.mean(np.sign(da[da != 0]) == sgn)) if (da != 0).any() else float("nan"))
    return r
