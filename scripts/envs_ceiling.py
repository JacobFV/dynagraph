"""envs-r7 CEILING AUDIT: how well can the scripted skills do on each (robot x family) when the parameters are right?

    python scripts/envs_ceiling.py <tag> <robots> <families> <n_seeds> <policies> [procs]
      e.g. python scripts/envs_ceiling.py r6 ur5e_2f85,panda,gen3_2f85,probe bin_clearing,pouch_transfer 6 \
               nominal,oracle,oracle_slow,oracle_firm 2

policies: nominal = env.scripted() (fixed engineer parameters); oracle = the same scripted structure with every skill
parameter from the TRUE hidden physics (scripts/envs_decision_relevance.py::rule, src='true'); oracle_slow /
oracle_firm / oracle_soft = a small grid around the oracle (all speeds x0.5 / grip force x1.3 / grip force x0.75).
Per episode: success, progress, damage, calls, step-status counts, a FAILURE TAXONOMY (primary cause) and the
integrity flags. Rows append to results/v2/envs/ceiling_<tag>_rows.jsonl (resumable: done keys are skipped);
`summarize <tag>` writes results/v2/envs/ceiling_<tag>.json + a markdown table. Site dressing on (as the r6 tranche).
"""
import collections
import json
import math
import os
import signal
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

CAT = {  # skill status -> failure category
    "missed": "grasp_miss", "no_contact": "grasp_miss", "too_wide": "grasp_miss", "empty": "grasp_miss",
    "grasp_failed": "grasp_miss",
    "slipped": "slip", "contact_lost": "slip", "dropped": "slip", "handle_slipped": "slip", "lost": "slip",
    "unreachable": "ik_fail", "ik_failed": "ik_fail",
    "guard_tripped": "collision", "collided": "collision",
    "jammed": "jam", "not_inserted": "jam",
    "not_observed": "perception", "fell": "collision", "timeout": "timeout",
    # envs-v3 family-skill statuses
    "toppled": "topple", "tugged": "tug_or_snatch", "tug_hold": "tug_or_snatch", "foot_slip": "foot_slip",
    "stalled": "jam", "unsupported": "unsupported", "empty": "grasp_miss",
}
LIMIT_S = int(os.environ.get("CEIL_LIMIT_S", "300"))


class _TO(BaseException):
    pass


def _alarm(signum, frame):
    raise _TO()


def split_for(fam, robot):
    from physmem.envs.splits import HELDOUT_FAMILIES
    if robot == "gen3_2f85":
        return "heldout_robot"
    return "heldout_family" if fam in HELDOUT_FAMILIES.values() else "test"


def apply_policy(env, call, pol):
    from physmem.envs.oracle import rule
    if pol == "nominal" or call is None:
        return call
    if pol == "force_ramp":                 # r7 baseline (tight_insert): press-until-click
        return env.force_ramp(call) if hasattr(env, "force_ramp") else call
    if pol == "probe_then_act":
        return env.probe_then_act(call)
    c = rule(env, call, "true")
    p = dict(c.params)
    if pol == "oracle_slow":
        for k in list(p):
            if "speed" in k and isinstance(p[k], (int, float)):
                p[k] = float(p[k]) * 0.5
    elif pol in ("oracle_firm", "oracle_soft"):
        f = 1.3 if pol == "oracle_firm" else 0.75
        for k in ("grip_force", "squeeze_n", "suction"):
            if k in p and isinstance(p[k], (int, float)):
                p[k] = float(p[k]) * f if k != "suction" else float(min(1.0, p[k] * f))
    return replace(c, params=p)


def classify(rec):
    if rec["success"]:
        return "success"
    if rec["timeout"] or rec["error"]:
        return "sim_timeout_or_error"
    if rec["unstable"]:
        return "sim_instability"
    if rec["damaged"]:
        return "crush_or_damage"
    cats = collections.Counter()
    for st, n in rec["status"].items():
        c = CAT.get(st) or ("error" if st.startswith("error") else None)
        if c:
            cats[c] += n
    if rec["collisions"] and not cats:
        return "collision"
    if cats:
        return cats.most_common(1)[0][0]
    if rec["calls"] >= rec["budget"]:
        return "budget"
    return "place_miss_or_predicate"     # every call 'done' but the task predicate not met (placement / pose / time)


def episode(job):
    fam_v, robot, seed, pol, tag = job
    fam, _, variant = fam_v.partition("@")       # 'bin_clearing@deep_bin' = a named station variant
    from physmem.envs import family_class, make_env
    from physmem.envs.splits import split_params
    from physmem.v2.api import TaskSpec
    t0 = time.time()
    split = split_for(fam, robot)
    params = split_params(fam, split)
    params["dressing"] = "site"
    if variant:
        params["station_variant"] = variant
    env = make_env(fam)
    rec = dict(tag=tag, family=fam_v, robot=robot, seed=seed, policy=pol, split=split, success=False, progress=0.0,
               damaged=0, calls=0, budget=0, status={}, unstable=False, timeout=False, error=None, collisions=0,
               sim_t=0.0, tactile_readings=0, picks=0)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(LIMIT_S)
    try:
        env.reset(TaskSpec(fam, family_class(fam).category, seed, robot, split, params))
        rec["budget"] = int(env.budget)
        st = collections.Counter()
        for _ in range(4 * env.budget + 10):
            c = env.scripted()
            if c is None:
                break
            c = apply_policy(env, c, pol)
            out = env.execute(c)
            st[out.result.status] += 1
            rec["picks"] += c.skill == "pick"
            if out.done:
                break
        rec["status"] = dict(st)
    except _TO:
        rec["timeout"] = True
    except Exception as e:  # keep the row
        rec["error"] = repr(e)[:300]
    finally:
        signal.alarm(0)
    try:
        rec.update(success=bool(env.solved()) and not rec["timeout"] and rec["error"] is None,
                   progress=float(env.progress()), damaged=len(getattr(env, "damaged", ())), calls=int(env.n_calls),
                   unstable=bool(getattr(env, "unstable", False)), collisions=int(env.metrics_acc.get("collisions", 0)),
                   sim_t=float(env.t), tactile_readings=sum(1 for e in env.events_all if int(e.property_type) == 7
                                                            and e.source_id == "tactile_squeeze"))
    except Exception:
        pass
    rec["wall_s"] = time.time() - t0
    rec["cause"] = classify(rec)
    return rec


def rows_path(tag):
    return f"results/v2/envs/ceiling_{tag}_rows.jsonl"


def run(tag, robots, fams, n, pols, procs):
    from physmem.envs.splits import make_tasks
    os.makedirs("results/v2/envs", exist_ok=True)
    done = set()
    if os.path.exists(rows_path(tag)):
        for line in open(rows_path(tag)):
            r = json.loads(line)
            done.add((r["family"], r["robot"], r["seed"], r["policy"]))
    jobs = []
    for fam in fams:
        for rob in robots:
            f0 = fam.partition("@")[0]
            seeds = [t.seed for t in make_tasks(f0, split_for(f0, rob), n, rob, base_seed=1_700_000)]
            for s in seeds:
                for p in pols:
                    if (fam, rob, s, p) not in done:
                        jobs.append((fam, rob, s, p, tag))
    print(f"{len(jobs)} episodes to run ({len(done)} done)", flush=True)
    out = open(rows_path(tag), "a")
    with Pool(procs, maxtasksperchild=20) as pool:
        for i, r in enumerate(pool.imap_unordered(episode, jobs, chunksize=1)):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 20 == 0:
                print(f"[{i + 1}/{len(jobs)}] {r['robot']} {r['family']} {r['policy']} {r['cause']} "
                      f"{r['wall_s']:.0f}s", flush=True)


def summarize(tag):
    rows = [json.loads(l) for l in open(rows_path(tag))]
    G = collections.defaultdict(list)
    for r in rows:
        r["cause"] = classify(r)          # re-derive with the current taxonomy
        G[(r["robot"], r["family"], r["policy"])].append(r)
    table = []
    for (rob, fam, pol), rs in sorted(G.items()):
        causes = collections.Counter(r["cause"] for r in rs)
        table.append(dict(robot=rob, family=fam, policy=pol, n=len(rs),
                          success=float(np.mean([r["success"] for r in rs])),
                          progress=float(np.mean([r["progress"] for r in rs])),
                          damaged=float(np.mean([r["damaged"] > 0 for r in rs])),
                          calls=float(np.mean([r["calls"] for r in rs])),
                          wall_s=float(np.mean([r["wall_s"] for r in rs])),
                          tactile_per_pick=float(np.sum([r["tactile_readings"] for r in rs]) /
                                                 max(1, np.sum([r["picks"] for r in rs]))),
                          causes=dict(causes)))
    best = {}
    for t in table:
        if t["policy"] != "nominal":
            k = (t["robot"], t["family"])
            if k not in best or t["success"] > best[k]["success"]:
                best[k] = t
    json.dump(dict(tag=tag, table=table, best_oracle={f"{k[0]}/{k[1]}": v["policy"] for k, v in best.items()}),
              open(f"results/v2/envs/ceiling_{tag}.json", "w"), indent=1)
    pols = [p_ for p_ in ("nominal", "probe_then_act", "force_ramp", "oracle", "oracle_slow", "oracle_firm")
            if any(t["policy"] == p_ for t in table)]
    by = {(t["robot"], t["family"], t["policy"]): t for t in table}
    lines = ["| robot | family | " + " | ".join(pols) + " | oracle failure causes | nominal failure causes | "
             "median success sim-s (" + " / ".join(pols) + ") |",
             "|---|---|" + "---|" * (len(pols) + 3)]
    f = lambda t: f"{t['success']:.2f} ({t['progress']:.2f})" if t else "-"
    cs = lambda t: ", ".join(f"{k} {v}" for k, v in sorted(t["causes"].items(), key=lambda x: -x[1]) if k != "success") \
        if t else ""
    for (rob, fam) in sorted({(t["robot"], t["family"]) for t in table}):
        med = []
        for p_ in pols:
            ts = [r["sim_t"] for r in G.get((rob, fam, p_), []) if r["success"]]
            med.append(f"{np.median(ts):.1f}" if ts else "-")
        lines.append(f"| {rob} | {fam} | " + " | ".join(f(by.get((rob, fam, p_))) for p_ in pols) +
                     f" | {cs(by.get((rob, fam, 'oracle')))} | {cs(by.get((rob, fam, 'nominal')))} | {' / '.join(med)} |")
    md = "\n".join(lines)
    open(f"results/v2/envs/ceiling_{tag}.md", "w").write(
        f"success (mean progress) over seeds; site-dressed; tag {tag}\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    if sys.argv[1] == "summarize":
        summarize(sys.argv[2])
    else:
        tag, robots, fams, n, pols = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), int(sys.argv[4]), \
            sys.argv[5].split(",")
        procs = int(sys.argv[6]) if len(sys.argv) > 6 else 2
        run(tag, robots, fams, n, pols, procs)
