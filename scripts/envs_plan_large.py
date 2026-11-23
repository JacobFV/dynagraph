"""Plan the LARGE MODEL-V2 generation (tranches of task JSONs) from ROBOTS' measured env runs.

    python scripts/envs_plan_large.py --out data/v2_envs_large/plan

Pair rule (lead): include every robot x family pair ROBOTS actually ran that is physically executable - failures are
training signal. Exclude a pair only if (a) >= 80 % of its skill calls end 'unreachable' / 'not_observed' /
'too_wide' or it never advances sim time (mean sim < 1 s), or (b) it crashes (> 1 skill error per episode).
gen3_2f85 is held out entirely (split heldout_robot only). Untested pairs are not generated.
Policy per episode: 10 % 'explore' (random feasible candidates), 90 % 'mixed' (0.4 scripted / 0.4 random candidate /
0.2 perturbed scripted per decision; scripts/envs_generate.py::choose).
"""
import argparse
import collections
import glob
import json
import os
from dataclasses import asdict

from physmem.envs import FAMILIES
from physmem.envs.splits import HELDOUT_FAMILIES, make_tasks

SPLIT_SHARE = (("train", 0.70), ("val", 0.10), ("test", 0.10), ("heldout_layout", 0.05), ("heldout_objects", 0.05))
EXPENSIVE = {"bulk_sweep": 0.25}          # real-arm bulk_sweep episodes take ~2 min wall: fewer of them
HELDOUT_ROBOT = "gen3_2f85"


def executable_pairs(pattern="results/v2/robots/env_runs*.jsonl"):
    st = collections.defaultdict(lambda: dict(n=0, steps=0, unr=0, err=0, sim=0.0, wall=0.0))
    for f in glob.glob(pattern):
        if "stale" in f:
            continue
        for line in open(f):
            r = json.loads(line)
            s = st[(r["robot"], r["fam"])]
            s["n"] += 1
            s["sim"] += r.get("sim_t") or 0
            s["wall"] += r.get("wall") or 0
            for x in r.get("steps", []):
                s["steps"] += 1
                s["unr"] += x[2] in ("unreachable", "not_observed", "too_wide")
                s["err"] += str(x[2]).startswith("error")
            s["err"] += bool(r.get("error"))
    keep, drop = {}, {}
    for (rob, fam), s in st.items():
        if fam not in FAMILIES:
            continue
        unr = s["unr"] / max(s["steps"], 1)
        why = None
        if unr >= 0.8:
            why = f"{unr:.0%} unreachable"
        elif s["sim"] / s["n"] < 1.0:
            why = "no sim progress"
        elif s["err"] / s["n"] > 1.0:
            why = f"crash ({s['err']} errors / {s['n']} eps)"
        stats = dict(n=s["n"], unreachable=round(unr, 2), wall_s=round(s["wall"] / s["n"], 1), errors=s["err"])
        (drop if why else keep).setdefault(rob, {})[fam] = dict(stats, reason=why) if why else stats
    return keep, drop


def tasks_for(rob, fam, n_total, base_seed, explore_frac=0.1):
    out = []
    heldout_fam = fam in HELDOUT_FAMILIES.values()
    if rob == HELDOUT_ROBOT:
        shares = (("heldout_robot", 1.0),)
    elif heldout_fam:
        shares = (("heldout_family", 0.6), ("heldout_layout", 0.2), ("heldout_objects", 0.2))
    else:
        shares = SPLIT_SHARE
    for split, sh in shares:
        n = max(1, int(round(n_total * sh)))
        for i, t in enumerate(make_tasks(fam, split, n, rob, base_seed)):
            params = dict(t.params)
            params["policy"] = "explore" if (i % 10) == 9 and explore_frac > 0 else "mixed"
            out.append(asdict(t) | dict(params=params))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v2_envs_large/plan")
    ap.add_argument("--tranche4", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.tranche4:
        return plan_tranche4(a.out)
    keep, drop = executable_pairs()
    keep["probe"] = {f: dict(n=0) for f in FAMILIES}
    # tranche 1 (first usable data, ~3k): probe + ur5e + panda across all their families
    t1 = []
    per = {"probe": 100, "ur5e_2f85": 46, "panda": 46}
    for rob, n in per.items():
        for fam in keep.get(rob, {}):
            t1 += tasks_for(rob, fam, int(n * EXPENSIVE.get(fam, 1.0)) if rob != "probe" else n, base_seed=0)
    # tranche 2: more probe / ur5e / panda + every other executable robot; tranche 3: held-out robot
    t2 = []
    per2 = {"probe": 350, "ur5e_2f85": 200, "panda": 200, "tidybot": 150, "aloha": 120, "ur5e_suction": 150,
            "fr3_suction": 150, "z1": 150, "so100": 100, "g1_dex3_fixed": 100, "g1_dex3": 25}
    for rob, n in per2.items():
        for fam in keep.get(rob, {}):
            t2 += tasks_for(rob, fam, int(n * EXPENSIVE.get(fam, 1.0)) if rob != "probe" else n, base_seed=100000)
    t3 = []
    for fam in keep.get(HELDOUT_ROBOT, {}):
        t3 += tasks_for(HELDOUT_ROBOT, fam, int(50 * EXPENSIVE.get(fam, 1.0)), base_seed=200000)
    for name, tl in (("tranche1", t1), ("tranche2", t2), ("tranche3_heldout_robot", t3)):
        json.dump(tl, open(os.path.join(a.out, f"{name}.json"), "w"))
        c = collections.Counter((t["robot"]) for t in tl)
        print(name, len(tl), dict(c))
    json.dump(dict(keep=keep, drop=drop), open(os.path.join(a.out, "pairs.json"), "w"), indent=1)
    print("dropped pairs:", {r: {f: v["reason"] for f, v in d.items()} for r, d in drop.items()})





def plan_tranche4(out):
    """envs-r3 tranche: redesigned / fixed families and newly executable pairs (run after tranches 1-3)."""
    pairs = {
        "probe": ["depalletize", "conveyor_sort", "bin_clearing", "small_parts_pick", "small_peg_insert"],
        "ur5e_2f85": ["conveyor_sort", "small_parts_pick", "small_peg_insert"],
        "panda": ["conveyor_sort", "small_parts_pick", "small_peg_insert"],
        "so100": ["small_parts_pick", "small_peg_insert"],
        "z1": ["small_parts_pick", "small_peg_insert"],
        "ur5e_suction": ["depalletize"], "fr3_suction": ["depalletize"],
        "tidybot": ["peg_insert", "fixture_place", "bin_clearing", "kitting_pocket", "connector_insert", "tool_return",
                    "machine_tending", "conveyor_sort"],
    }
    n = {"probe": 200, "ur5e_2f85": 150, "panda": 150, "so100": 150, "z1": 150, "ur5e_suction": 200,
         "fr3_suction": 200, "tidybot": 100}
    t4 = []
    for rob, fams in pairs.items():
        for fam in fams:
            t4 += tasks_for(rob, fam, n[rob], base_seed=300000)
    t4h = []
    for fam in ("small_parts_pick", "small_peg_insert", "conveyor_sort"):
        t4h += tasks_for(HELDOUT_ROBOT, fam, 40, base_seed=300000)
    json.dump(t4 + t4h, open(os.path.join(out, "tranche4_envs_r3.json"), "w"))
    print("tranche4_envs_r3", len(t4 + t4h), dict(collections.Counter(t["robot"] for t in t4 + t4h)))


if __name__ == "__main__":
    main()
