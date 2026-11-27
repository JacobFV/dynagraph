"""Oracle / prior / probe parameter RULES for the scripted policies (moved here from
scripts/envs_decision_relevance.py in envs-r7 so the env can expose them: StationEnv.demo_call = rule(src='true')).

rule(env, call, src) returns the scripted call with skill parameters chosen from the hidden physics:
src = 'true' (oracle: TRUE latents), 'prior' (population medians), 'probe' / 'crush_probe' (crush limit from the
actor-legal stiffness evidence), 'crush_prior' (all true but the crush limit). Physics-based engineering rules: grip
force from mass / grip friction / crush limit, suction from mass, intercept from belt speed, push speed from friction..
"""
import json
import math
import os
from dataclasses import replace

import numpy as np

from ..v2.api import SkillCall

G = 9.81
PROBE_F = 4.0          # N: light-squeeze probe grasp (below every family's minimum crush limit, 6 N)
PRIOR_PATH = "results/v2/envs/latent_priors.json"
PRIORS = json.load(open(PRIOR_PATH)) if os.path.exists(PRIOR_PATH) else {}


def _h(env, e):
    return env.scene.entities[e].hidden if e in env.scene.entities else {}


def lat(env, e, key, src, default=None):
    """latent value for entity e: TRUE hidden value or the family's population median (prior)."""
    if src == "true" or (src.startswith("crush_") and key != "crush_n"):
        return _h(env, e).get(key, default)
    if src == "crush_prior":
        src = "prior"
    if src in ("probe", "crush_probe") and key == "crush_n" and e in env.scene.entities:
        # r6 probe-then-act: crush limit from ACTOR-LEGAL stiffness evidence (tactile squeeze, else vision prior);
        # 1.2 margin for the reading noise. Every other latent stays at the prior.
        est = env.est_crush(e, margin=1.2)
        if est is not None:
            return est
    pri = PRIORS.get(env.family, {})
    kind = env.scene.entities[e].kind if e in env.scene.entities else "_"
    return pri.get(f"{kind}.{key}", pri.get(key, default))


def grip_for(env, e, src, a=2.0, F_max=100.0, extra_mass=0.0):
    m = (lat(env, e, "mass", src, 0.3) or 0.3) + extra_mass
    mug = lat(env, e, "mu_grip", src, 0.6) or 0.6
    crush = lat(env, e, "crush_n", src, float("inf"))
    # r7: 2.5x the static hold force (point contacts, transport accelerations, wrench about the pads), 10 N floor
    # (5 N lost light round parts on the 2F-85); 0.7 crush cap (dynamic squeeze overshoot ~20%)
    F = 2.5 * m * (G + a) / (2 * mug)
    F = min(max(F, 10.0), F_max)
    if crush is not None and np.isfinite(crush):
        F = min(F, 0.7 * crush)
    return float(F)


def rule(env, call, src):
    """return the call with rule-chosen parameters (src = 'true' | 'prior')."""
    if hasattr(env, "v3_rule"):          # envs-v3 families carry their own physics rules (families_v3_*.py)
        return env.v3_rule(call, src)
    f, p, t = env.family, dict(call.params), call.target
    if (src in ("probe", "crush_probe") and call.skill == "pick" and t in env.scene.entities and "grip_force" in p
            and f not in ("pouch_transfer",) and env.est_stiffness(t)[1] != "tactile_squeeze"
            and np.isfinite(_h(env, t).get("stiffness", np.nan))):
        # probe-then-act: the first grasp of a part is a LIGHT squeeze (PROBE_F); its tactile stiffness reading
        # then sets the crush bound of every later grasp (a light part may simply come up on the probe grasp)
        k_v, _ = env.est_stiffness(t)
        if k_v is None or k_v < 6.5e4 or env.fail_count.get(t, 0) == 0:
            p["grip_force"] = PROBE_F
            return replace(call, params=p)
    if call.skill == "pick" and f == "pouch_transfer":
        # stiffness-based pinch: a stiff pouch needs more squeeze to hold by friction (soft ones wrap the pads)
        y = lat(env, t, "young", src, None) if t in env.scene.entities else None
        if y is None:
            y = env.hidden_task["young"] if src == "true" else PRIORS.get(f, {}).get("young", 8e3)
        crush = (env.hidden_task.get("crush_n") if src == "true" else PRIORS.get(f, {}).get("crush_n")) or 1e9
        p["grip_force"] = float(min(np.clip(6.0 + y / 1000.0, 6.0, 45.0), 0.7 * crush))
        return replace(call, params=p)
    if call.skill == "pick" and t in env.scene.entities:
        if f == "depalletize":
            m = lat(env, t, "mass", src, 3.0)
            if m > env.payload():
                return SkillCall("handover", dict(reason="rule: overweight"), target=t)
            mug = lat(env, t, "mu_grip", src, 0.8)
            p.update(suction=float(np.clip(1.4 * m * (G + 3) / 300.0, 0.3, 1.0)),
                     lift_speed=0.1 if mug < 0.4 else 0.2)
        elif f == "tool_return":
            mh = lat(env, t, "head_mass", src, 0.5)
            imb = lat(env, t, "head_imbalance", src, 0.25)
            mug = lat(env, t, "mu_grip", src, 0.6)
            F_t = 1.5 * mh * imb * 0.035 * G / (2 * mug * 0.006)       # torque capacity ~ 2 mu F r_eff
            F = max(grip_for(env, t, src), F_t)
            crush = lat(env, t, "crush_n", src, float("inf"))
            p["grip_force"] = float(min(F, 100.0, 0.7 * crush if np.isfinite(crush) else 100.0))
        elif f == "tote_transfer":
            p["grip_force"] = grip_for(env, t, src, a=1.5, F_max=150.0)
            p["lift_speed"] = 0.12
        elif "grip_force" in p or f in ("bin_clearing", "small_parts_pick", "conveyor_sort", "fixture_place",
                                        "kitting_pocket", "peg_insert", "connector_insert", "small_peg_insert",
                                        "machine_tending"):
            p["grip_force"] = grip_for(env, t, src)
            if f == "tight_insert":
                p["grip_force"] = max(p["grip_force"], 30.0)    # a rigid peg (no crush): keep the engineer's firm grip
        # r7: crush-safe ceiling for the slip-reactive, force-limited grip (real arms: measured pad force is kept under
        # grip_max); a part that needs nearly all of it is lifted / carried slowly (lower acceleration)
        crush_c = lat(env, t, "crush_n", src, float("inf"))
        if "grip_force" in p and crush_c is not None and np.isfinite(crush_c):
            p["grip_max"] = float(0.85 * crush_c)
            m_ = lat(env, t, "mass", src, 0.3) or 0.3
            mug_ = lat(env, t, "mu_grip", src, 0.6) or 0.6
            if 1.3 * m_ * G / (2 * mug_) > 0.6 * crush_c:
                p["lift_speed"] = min(float(p.get("lift_speed", 0.2)), 0.06)
                env._oracle_slow_carry = True
        if f == "conveyor_sort" and "vx" in p:
            # intercept speed: true belt speed vs prior median
            v = env.belt_speed if src == "true" else PRIORS.get(f, {}).get("belt_speed", 0.075)
            p["vx"], p["vy"] = float(v), 0.0
    elif call.skill in ("open", "close") and t in env.scene.entities:
        h = _h(env, t)
        fl = lat(env, t, "frictionloss", src, 1.0)
        dmp = lat(env, t, "damping", src, 1.0)
        mug = lat(env, t, "mu_grip", src, 0.6)
        crush = lat(env, t, "crush_n", src, float("inf"))
        if env.scene.entities[t].kind == "door":
            pull = fl / 0.31 + dmp * 0.5 / 0.31 + 2.0
        elif env.scene.entities[t].kind == "knob":
            pull = 1.5 * fl / 0.012 + 2.0                               # torque -> pad friction force at r = 12 mm
        else:
            pull = fl + dmp * p.get("speed", 0.12) + lat(env, t, "mass", src, 5.0) * 0.5
        F = 1.3 * pull / (2 * mug)
        if crush is not None and np.isfinite(crush):
            F = min(F, 0.7 * crush)
        p["grip_force"] = float(np.clip(F, 5.0, 100.0))
    elif call.skill == "push" and f == "singulation":
        mu = lat(env, t, "mu", src, 0.5)
        m = lat(env, t, "mass", src, 0.5)
        crush = lat(env, t, "crush_n", src, 1e9)
        scuff = lat(env, t, "scuff_j", src, 1e9) or 1e9
        if mu * m * G * 0.15 > 0.7 * scuff:   # r6: dragging it ~15 cm (push + slide + shoves) would scuff it: pick
            return env.pick_call(t, grip_force=float(np.clip(1.3 * m * (G + 2.0) / (2 * 0.8), 5.0, 60.0)))
        if 1.5 * mu * m * G / 2.0 > 0.8 * crush:            # a push would crush it: pick it instead
            F = min(1.3 * m * (G + 2.0) / (2 * 0.8), 0.7 * crush)
            return env.pick_call(t, grip_force=F)
        D = p.get("distance", 0.06) + 0.0
        d_push = 0.02
        p["distance"] = d_push
        p["speed"] = float(np.clip(math.sqrt(max(2 * mu * G * max(D - d_push, 0.0), 1e-4)), 0.05, 1.0))
    elif call.skill == "sweep" and f == "bulk_sweep":
        mu = env.mu_mean if src == "true" else PRIORS.get(f, {}).get("mu_mean", 0.45)
        v = p.get("speed", 0.4)
        p["stop_time"] = float(np.clip(1.2 * v / (mu * G), 0.05, 1.5))  # decel <= mu g: parts stay with the blade
    elif call.skill == "insert":
        reg = env.region(t) if t in env.scene.regions else {}
        clr = None
        if src == "true":
            clr = getattr(env, "hidden_task", {}).get("clearance")
            if clr is None and env.held in env.scene.entities:
                clr = _h(env, env.held).get("clearance")
        else:
            clr = PRIORS.get(f, {}).get("clearance")
        held = env.held if env.held in env.scene.entities else None
        dent = (lat(env, held, "dent_n", src, 1e9) if held else 1e9) or 1e9
        if f == "tight_insert":
            # r7 snap-fit: press past the lips (hidden snap force from lip stiffness x friction) but below the dent
            # limit; gentle when the margin is small
            snap = (getattr(env, "hidden_task", {}).get("snap_n") if src == "true" else PRIORS.get(f, {}).get("snap_n"))
            snap = 8.0 if snap is None else float(snap)
            F = 1.35 * snap + 1.0
            if np.isfinite(dent):
                F = min(F, 0.85 * dent)
            p["down_force"], p["min_force"] = float(F), 0.0
            p["wiggle"] = 0.0             # a yaw wiggle in a 0.1-0.5 mm square hole wedges the corners (side load -> dent)
            p["press_speed"] = 0.01 if (np.isfinite(dent) and 0.85 * dent < 1.6 * snap + 1.0) else 0.02
        elif np.isfinite(dent):
            p["down_force"] = float(min(p.get("down_force", 6.0), 0.6 * dent))
        if clr is not None and f != "tight_insert":     # tight_insert keeps the engineer's widening spiral
            p["search_r"] = 0.0 if clr >= 0.0015 else float(np.clip(0.003 - clr, 0.001, 0.004))
            if f != "tight_insert":
                p["wiggle"] = 0.02 if clr < 0.0008 else 0.0
    if call.skill == "place" and src == "true" and getattr(env, "_oracle_slow_carry", False):
        p["speed"] = min(float(p.get("speed", 0.4)), 0.2)      # r7: gentle transport of a marginally-held part
        env._oracle_slow_carry = False
    return replace(call, params=p) if call.skill != "handover" else call


