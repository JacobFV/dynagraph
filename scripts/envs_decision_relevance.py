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

from physmem.envs.oracle import G, PRIOR_PATH, PRIORS, PROBE_F, _h, grip_for, lat, rule  # noqa: E402,F401
SRCS = tuple(os.environ.get("DR_SRCS", "true,prior,scripted").split(","))


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
