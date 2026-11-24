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
PROBE_F = 4.0          # N: light-squeeze probe grasp (below every family's minimum crush limit, 6 N)
SRCS = tuple(os.environ.get("DR_SRCS", "true,prior,scripted").split(","))
PRIOR_PATH = "results/v2/envs/latent_priors.json"


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
    F = 1.3 * m * (G + a) / (2 * mug)
    F = min(max(F, 5.0), F_max)      # r6: 5 N floor (2 N missed light parts); 0.7 crush cap (dynamic squeeze overshoot ~20%)
    if crush is not None and np.isfinite(crush):
        F = min(F, 0.7 * crush)
    return float(F)


def rule(env, call, src):
    """return the call with rule-chosen parameters (src = 'true' | 'prior')."""
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
        if src == "scripted_probe":
            c = env.probe_then_act(c)          # r6: fragility-aware probe-then-act scripted variant (actor-legal)
        elif src != "scripted":
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
    jobs = [(f, 500000 + s, src) for f in fams for s in range(n) for src in SRCS]
    with Pool(procs) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    table = []
    for f in fams:
        R = {src: sorted([r for r in rows if r["family"] == f and r["src"] == src], key=lambda r: r["seed"])
             for src in SRCS}
        S_ = {src: [r["success"] for r in R[src]] for src in SRCS}
        st, sp = S_["true"], S_["prior"]
        gap, lo, hi = boot_ci(st, sp)
        row = dict(family=f, n=n, oracle=float(np.nanmean(st)), prior=float(np.nanmean(sp)), gap=gap, ci95=(lo, hi),
                   damaged_oracle=float(np.mean([r["damaged"] > 0 for r in R["true"]])),
                   damaged_prior=float(np.mean([r["damaged"] > 0 for r in R["prior"]])),
                   errors=sum(r["error"] is not None for r in rows if r["family"] == f))
        if "scripted" in SRCS:
            row["scripted"] = float(np.nanmean(S_["scripted"]))
        if "scripted_probe" in SRCS and "scripted" in SRCS:
            g5, l5, h5 = boot_ci(S_["scripted_probe"], S_["scripted"])
            row.update(scripted_probe=float(np.nanmean(S_["scripted_probe"])), scripted_probe_gain=g5,
                       scripted_probe_ci95=(l5, h5),
                       damaged_scripted=float(np.mean([r["damaged"] > 0 for r in R["scripted"]])),
                       damaged_scripted_probe=float(np.mean([r["damaged"] > 0 for r in R["scripted_probe"]])))
            print(f"{f:18s} [scripted] {row['scripted']:.2f} (dmg {row['damaged_scripted']:.2f}) -> probe-then-act "
                  f"{row['scripted_probe']:.2f} (dmg {row['damaged_scripted_probe']:.2f}) gain {g5:+.2f} [{l5:+.2f}, {h5:+.2f}]")
        if "probe" in SRCS:
            g2, l2, h2 = boot_ci(S_["probe"], sp)
            row.update(probe=float(np.nanmean(S_["probe"])), probe_gain=g2, probe_ci95=(l2, h2),
                       damaged_probe=float(np.mean([r["damaged"] > 0 for r in R["probe"]])))
        if "crush_prior" in SRCS and "crush_probe" in SRCS:
            # fragility isolated: every other latent TRUE; crush limit from the prior vs from actor-legal evidence
            g3, l3, h3 = boot_ci(st, S_["crush_prior"])
            g4, l4, h4 = boot_ci(S_["crush_probe"], S_["crush_prior"])
            row.update(crush_prior=float(np.nanmean(S_["crush_prior"])), crush_probe=float(np.nanmean(S_["crush_probe"])),
                       crush_gap=g3, crush_gap_ci95=(l3, h3), crush_probe_gain=g4, crush_probe_ci95=(l4, h4),
                       damaged_crush_prior=float(np.mean([r["damaged"] > 0 for r in R["crush_prior"]])),
                       damaged_crush_probe=float(np.mean([r["damaged"] > 0 for r in R["crush_probe"]])))
            print(f"{f:18s} [fragility] oracle {np.nanmean(st):.2f} crush-prior {row['crush_prior']:.2f} crush-probe "
                  f"{row['crush_probe']:.2f}  gap {g3:+.2f} [{l3:+.2f}, {h3:+.2f}]  probe gain {g4:+.2f} [{l4:+.2f}, {h4:+.2f}]"
                  f"  dmg {row['damaged_oracle']:.2f}/{row['damaged_crush_prior']:.2f}/{row['damaged_crush_probe']:.2f}")
        table.append(row)
        t = table[-1]
        extra = (f" probe {t['probe']:.2f} (vs prior {t['probe_gain']:+.2f} [{t['probe_ci95'][0]:+.2f}, "
                 f"{t['probe_ci95'][1]:+.2f}], dmg {t['damaged_probe']:.2f})") if "probe" in t else ""
        print(f"{f:18s} oracle {t['oracle']:.2f} prior {t['prior']:.2f} scripted {t.get('scripted', float('nan')):.2f} "
              f"gap {gap:+.2f} [{lo:+.2f}, {hi:+.2f}] dmg {t['damaged_oracle']:.2f}/{t['damaged_prior']:.2f}{extra} err {t['errors']}")
    tag = "_site" if DRESS else ""
    json.dump(dict(table=table, rows=rows, priors=PRIORS, wall_s=time.time() - t0, robot="probe", dressed=DRESS),
              open(f"results/v2/envs/decision_relevance{tag}.json" if len(fams) == len(FAMILIES)
                   else f"results/v2/envs/decision_relevance{tag}_{'_'.join(fams)[:60]}.json", "w"), indent=1, default=str)
    print("wall", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
