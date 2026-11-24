"""Decision relevance of hidden physics: success of an 'oracle-parameterized scripted' policy whose skill parameters
are computed from the TRUE latents vs the same rule fed PRIOR latents (population medians). A gap ~0 means the
hidden variable does not change the best action for this family.

    python scripts/envs_decision_relevance.py [families|all] [n_seeds] [procs]  -> results/v2/envs/decision_relevance.json

The rule per family is a simple physics-based engineering rule (grip force from mass / grip friction / crush limit,
suction from mass, intercept from belt speed, push speed from friction, ...). The scripted policy's structure is kept
(targets, order, retries) - only its parameters (and lift-vs-flag) come from the rule. Robot = probe.
"""
import json
import math
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from physmem.envs import FAMILIES, family_class, make_env
from physmem.envs.splits import split_params
from physmem.v2.api import SkillCall, TaskSpec

G = 9.81
PRIOR_PATH = "results/v2/envs/latent_priors.json"


def _h(env, e):
    return env.scene.entities[e].hidden if e in env.scene.entities else {}


def lat(env, e, key, src, default=None):
    """latent value for entity e: TRUE hidden value or the family's population median (prior)."""
    if src == "true":
        return _h(env, e).get(key, default)
    pri = PRIORS.get(env.family, {})
    kind = env.scene.entities[e].kind if e in env.scene.entities else "_"
    return pri.get(f"{kind}.{key}", pri.get(key, default))


def grip_for(env, e, src, a=2.0, F_max=100.0, extra_mass=0.0):
    m = (lat(env, e, "mass", src, 0.3) or 0.3) + extra_mass
    mug = lat(env, e, "mu_grip", src, 0.6) or 0.6
    crush = lat(env, e, "crush_n", src, float("inf"))
    F = 1.3 * m * (G + a) / (2 * mug)
    F = min(max(F, 2.0), F_max)
    if crush is not None and np.isfinite(crush):
        F = min(F, 0.85 * crush)
    return float(F)


def rule(env, call, src):
    """return the call with rule-chosen parameters (src = 'true' | 'prior')."""
    f, p, t = env.family, dict(call.params), call.target
    if call.skill == "pick" and f == "pouch_transfer":
        # stiffness-based pinch: a stiff pouch needs more squeeze to hold by friction (soft ones wrap the pads)
        y = lat(env, t, "young", src, None) if t in env.scene.entities else None
        if y is None:
            y = env.hidden_task["young"] if src == "true" else PRIORS.get(f, {}).get("young", 8e3)
        crush = (env.hidden_task.get("crush_n") if src == "true" else PRIORS.get(f, {}).get("crush_n")) or 1e9
        p["grip_force"] = float(min(np.clip(6.0 + y / 1000.0, 6.0, 45.0), 0.85 * crush))
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
            p["grip_force"] = float(min(F, 100.0, 0.85 * crush if np.isfinite(crush) else 100.0))
        elif f == "tote_transfer":
            p["grip_force"] = grip_for(env, t, src, a=1.5, F_max=150.0)
            p["lift_speed"] = 0.12
        elif "grip_force" in p or f in ("bin_clearing", "small_parts_pick", "conveyor_sort", "fixture_place",
                                        "kitting_pocket", "peg_insert", "connector_insert", "small_peg_insert",
                                        "machine_tending"):
            p["grip_force"] = grip_for(env, t, src)
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
            F = min(F, 0.85 * crush)
        p["grip_force"] = float(np.clip(F, 5.0, 100.0))
    elif call.skill == "push" and f == "singulation":
        mu = lat(env, t, "mu", src, 0.5)
        m = lat(env, t, "mass", src, 0.5)
        crush = lat(env, t, "crush_n", src, 1e9)
        if 1.5 * mu * m * G / 2.0 > 0.8 * crush:            # a push would crush it: pick it instead
            F = min(1.3 * m * (G + 2.0) / (2 * 0.8), 0.85 * crush)
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
        if np.isfinite(dent):
            p["down_force"] = float(min(p.get("down_force", 6.0), 0.6 * dent))
        if clr is not None:
            p["search_r"] = 0.0 if clr >= 0.0015 else float(np.clip(0.003 - clr, 0.001, 0.004))
            p["wiggle"] = 0.02 if clr < 0.0008 else 0.0
    return replace(call, params=p) if call.skill != "handover" else call


DRESS = os.environ.get("DR_DRESS", "") == "1"       # envs-r6: run the check in the dressed site


def _params(fam):
    p = split_params(fam, "train")
    if DRESS:
        p["dressing"] = "site"
    return p


def episode(fam, seed, src, budget_scale=1.0):
    env = make_env(fam)
    env.reset(TaskSpec(fam, family_class(fam).category, seed, "probe", "dr", _params(fam)))
    out = None
    for _ in range(200):
        c = env.scripted()
        if c is None:
            break
        if src != "scripted":
            c = rule(env, c, src)
        out = env.execute(c)
        if out.done:
            break
    return dict(success=float(env.solved()), progress=float(env.progress()), damaged=len(env.damaged),
                calls=env.n_calls, effort=float(sum(0 for _ in [0])))


def _job(a):
    fam, seed, src = a
    t0 = time.time()
    try:
        r = episode(fam, seed, src)
        r["error"] = None
    except Exception as e:
        r = dict(success=float("nan"), progress=float("nan"), damaged=0, calls=0, error=repr(e)[:200])
    r.update(family=fam, seed=seed, src=src, wall=time.time() - t0)
    return r


def compute_priors(fams, n=30):
    """population medians of hidden latents per family (train split, seeds 900000+): the 'prior-mean' engineer."""
    pri = {}
    for fam in fams:
        vals = {}
        for s in range(900000, 900000 + n):
            env = make_env(fam)
            env.reset(TaskSpec(fam, family_class(fam).category, s, "probe", "train", _params(fam)))
            for e, info in env.scene.entities.items():
                for k, v in info.hidden.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vals.setdefault(f"{info.kind}.{k}", []).append(float(v))
                        vals.setdefault(k, []).append(float(v))
            for k, v in getattr(env, "hidden_task", {}).items():
                vals.setdefault(k, []).append(float(v))
            for k in ("belt_speed", "mu_mean"):
                if hasattr(env, k):
                    vals.setdefault(k, []).append(float(getattr(env, k)))
        pri[fam] = {k: float(np.median(v)) for k, v in vals.items()}
    return pri


PRIORS = json.load(open(PRIOR_PATH)) if os.path.exists(PRIOR_PATH) else {}


def boot_ci(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    global PRIORS
    fams = list(FAMILIES) if len(sys.argv) < 2 or sys.argv[1] == "all" else sys.argv[1].split(",")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs("results/v2/envs", exist_ok=True)
    t0 = time.time()
    with Pool(procs) as pool:
        parts = pool.starmap(compute_priors, [([f],) for f in fams])
    for pp in parts:
        PRIORS.update(pp)
    if not DRESS:
        json.dump(PRIORS, open(PRIOR_PATH, "w"), indent=1)
    jobs = [(f, 500000 + s, src) for f in fams for s in range(n) for src in ("true", "prior", "scripted")]
    with Pool(procs) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    table = []
    for f in fams:
        R = {src: sorted([r for r in rows if r["family"] == f and r["src"] == src], key=lambda r: r["seed"])
             for src in ("true", "prior", "scripted")}
        st = [r["success"] for r in R["true"]]
        sp = [r["success"] for r in R["prior"]]
        ss = [r["success"] for r in R["scripted"]]
        gap, lo, hi = boot_ci(st, sp)
        table.append(dict(family=f, n=n, oracle=float(np.nanmean(st)), prior=float(np.nanmean(sp)),
                          scripted=float(np.nanmean(ss)), gap=gap, ci95=(lo, hi),
                          damaged_oracle=float(np.mean([r["damaged"] > 0 for r in R["true"]])),
                          damaged_prior=float(np.mean([r["damaged"] > 0 for r in R["prior"]])),
                          errors=sum(r["error"] is not None for r in rows if r["family"] == f)))
        t = table[-1]
        print(f"{f:18s} oracle {t['oracle']:.2f} prior {t['prior']:.2f} scripted {t['scripted']:.2f} "
              f"gap {gap:+.2f} [{lo:+.2f}, {hi:+.2f}] dmg {t['damaged_oracle']:.2f}/{t['damaged_prior']:.2f} err {t['errors']}")
    tag = "_site" if DRESS else ""
    json.dump(dict(table=table, rows=rows, priors=PRIORS, wall_s=time.time() - t0, robot="probe", dressed=DRESS),
              open(f"results/v2/envs/decision_relevance{tag}.json" if len(fams) == len(FAMILIES)
                   else f"results/v2/envs/decision_relevance{tag}_{'_'.join(fams)[:60]}.json", "w"), indent=1, default=str)
    print("wall", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
