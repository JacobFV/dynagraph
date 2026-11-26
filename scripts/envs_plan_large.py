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


def executable_pairs(pattern=None):
    if pattern is None:      # ROBOTS' measured runs live in the main tree (frozen copies have no results/)
        main = os.path.expanduser("~/dynagraph/results/v2/robots")
        pattern = os.path.join(main if os.path.isdir(main) else "results/v2/robots", "env_runs*.jsonl")
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
    ap.add_argument("--r4", action="store_true")
    ap.add_argument("--r6", action="store_true")
    ap.add_argument("--r7", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.r6:
        return plan_r6(a.out, a.scale)
    if a.r7:
        return plan_r7(a.out, a.scale)
    if a.r4:
        return plan_r4(a.out)
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


RELEVANT = {"depalletize", "connector_insert", "bin_clearing", "bulk_sweep", "kitting_pocket", "drawer_open",
            "door_open", "tool_return", "fixture_place", "twist_lock"}
T4_PAIRS = {
    "probe": ["depalletize", "conveyor_sort", "bin_clearing", "small_parts_pick", "small_peg_insert"],
    "ur5e_2f85": ["conveyor_sort", "small_parts_pick", "small_peg_insert"],
    "panda": ["conveyor_sort", "small_parts_pick", "small_peg_insert"],
    "so100": ["small_parts_pick", "small_peg_insert"], "z1": ["small_parts_pick", "small_peg_insert"],
    "ur5e_suction": ["depalletize"], "fr3_suction": ["depalletize"],
    "tidybot": ["peg_insert", "fixture_place", "bin_clearing", "kitting_pocket", "connector_insert", "tool_return",
                "machine_tending", "conveyor_sort"],
    "gen3_2f85": ["small_parts_pick", "small_peg_insert", "conveyor_sort"],
}


def plan_r4(out):
    """envs-r4: tranche3_heldout_robot_r4 (gen3 only) + tranche5_r4 (all executable pairs, 2x weight on the
    decision-relevant families). Replaces the r3 tranches 3 and 4."""
    keep, _ = executable_pairs()
    pairs = {r: set(f) for r, f in keep.items()}
    pairs["probe"] = set(FAMILIES)
    for r, fams in T4_PAIRS.items():
        pairs.setdefault(r, set()).update(fams)
    base = {"probe": 110, "ur5e_2f85": 55, "panda": 55, "tidybot": 30, "aloha": 25, "ur5e_suction": 40,
            "fr3_suction": 30, "z1": 30, "so100": 25, "g1_dex3_fixed": 20, "g1_dex3": 6}
    t5, t3 = [], []
    for rob, fams in sorted(pairs.items()):
        for fam in sorted(fams):
            if fam not in FAMILIES:
                continue
            if rob == HELDOUT_ROBOT:
                t3 += tasks_for(rob, fam, int(45 * EXPENSIVE.get(fam, 1.0)), base_seed=400000)
                continue
            if rob not in base:
                continue
            n = base[rob] * (2 if fam in RELEVANT else 1)
            if rob != "probe":
                n = int(n * EXPENSIVE.get(fam, 1.0))
            t5 += tasks_for(rob, fam, max(n, 4), base_seed=500000)
    for name, tl in (("tranche3_heldout_robot_r4", t3), ("tranche5_r4", t5)):
        json.dump(tl, open(os.path.join(out, f"{name}.json"), "w"))
        print(name, len(tl), dict(collections.Counter(t["robot"] for t in tl)))


R6_EXTRA = {"tidybot": ["shelf_restock"], "g1_dex3_fixed": ["tote_transfer"], "probe": ["shelf_restock"],
            "ur5e_2f85": ["tote_transfer", "conveyor_sort", "singulation"], "panda": ["tote_transfer", "conveyor_sort",
                                                                                     "singulation"]}
# r6 decision-relevant families (envs_decision_relevance r6 table, gap CI above 0): 2x weight
R6_RELEVANT = {"depalletize", "bin_clearing", "singulation", "conveyor_sort", "drawer_open", "door_open",
               "machine_tending", "tote_transfer", "shelf_restock", "tool_return", "fixture_place", "connector_insert",
               "pouch_transfer"}     # pouch: probe-then-act gain +0.45 (fragility evidence decides the grip)
R6_BASE = {"probe": 80, "ur5e_2f85": 42, "panda": 42, "tidybot": 26, "aloha": 18, "ur5e_suction": 40,
           "fr3_suction": 30, "z1": 24, "so100": 20, "g1_dex3_fixed": 22}
R6_SKIP = {"g1_dex3"}       # floating-base G1: minutes per episode (ENVS.md), not in the r6 tranche
# r6 blow-up audit: tidybot conveyor_sort = the stowed arm sits inside the belt bed from reset (7 kN permanent
# contact, 100 % of episodes blown; a mount 0.6 m back still collides with the bin stands, every pick unreachable)
R6_SKIP_PAIRS = {("tidybot", "conveyor_sort")}


def plan_r6(out, scale=1.0):
    """envs-r6 dressed tranche (dressing='site' on every task): all executable pairs (ROBOTS' runs + earlier
    tranche pairs + r6 additions g1_dex3_fixed tote_transfer cell / tidybot shelf_restock move_base), gen3 held out
    (heldout_robot split only), the G5 hold-out family (splits.HELDOUT_FAMILIES['g5']) only in held-out splits."""
    keep, _ = executable_pairs()
    pairs = {r: set(f) for r, f in keep.items() if r not in R6_SKIP}
    pairs["probe"] = set(FAMILIES)
    for src in (T4_PAIRS, R6_EXTRA):
        for r, fams in src.items():
            pairs.setdefault(r, set()).update(fams)
    t8, t8h = [], []
    for rob, fams in sorted(pairs.items()):
        for fam in sorted(fams):
            if fam not in FAMILIES or (rob, fam) in R6_SKIP_PAIRS:
                continue
            if rob == HELDOUT_ROBOT:
                t8h += tasks_for(rob, fam, max(4, int(32 * scale * EXPENSIVE.get(fam, 1.0))), base_seed=800000)
                continue
            if rob not in R6_BASE:
                continue
            n = R6_BASE[rob] * (2 if fam in R6_RELEVANT else 1)
            if rob == "g1_dex3_fixed" and fam == "tote_transfer":
                n = 80
            if rob != "probe":
                n = int(n * EXPENSIVE.get(fam, 1.0))
            t8 += tasks_for(rob, fam, max(int(n * scale), 4), base_seed=800000)
    allt = t8 + t8h
    for t in allt:
        t["params"]["dressing"] = "site"
    # deterministic shuffle: a partial run (time limit) is still balanced over robots x families x splits
    import hashlib
    allt.sort(key=lambda t: hashlib.sha256(f"{t['robot']}|{t['family']}|{t['split']}|{t['seed']}".encode()).hexdigest())
    json.dump(allt, open(os.path.join(out, "tranche8_r6_site.json"), "w"))
    print("tranche8_r6_site", len(allt), dict(collections.Counter(t["robot"] for t in allt)))
    print("splits", dict(collections.Counter(t["split"] for t in allt)))
    print("families", dict(collections.Counter(t["family"] for t in allt)))


R7_FAMS = ["bin_clearing", "tight_insert", "fastener_pick", "fixture_place", "tool_return", "pouch_transfer",
           "depalletize", "conveyor_sort", "peg_insert", "small_parts_pick", "kitting_pocket"]
R7_BASE = {"probe": 110, "ur5e_2f85": 150, "panda": 150}
R7_HELDOUT_N = 40


def plan_r7(out, scale=1.0):
    """envs-r7 tranche for the new / fixed skills: probe + UR5e + Panda on the r7 skill-layer families (site-dressed;
    bin_clearing 1/3 in the `deep_bin` station variant), Gen3 held out (heldout_robot split only)."""
    import hashlib
    t = []
    for rob, n in R7_BASE.items():
        for fam in R7_FAMS:
            tl = tasks_for(rob, fam, max(4, int(n * scale)), base_seed=3_500_000)
            for i, x in enumerate(tl):
                if fam == "bin_clearing" and i % 3 == 2:
                    x["params"]["station_variant"] = "deep_bin"
            t += tl
    for fam in R7_FAMS:
        t += tasks_for(HELDOUT_ROBOT, fam, max(4, int(R7_HELDOUT_N * scale)), base_seed=3_500_000)
    for x in t:
        x["params"]["dressing"] = "site"
    t.sort(key=lambda x: hashlib.sha256(f"{x['robot']}|{x['family']}|{x['split']}|{x['seed']}".encode()).hexdigest())
    json.dump(t, open(os.path.join(out, "tranche9_r7.json"), "w"))
    print("tranche9_r7", len(t), dict(collections.Counter(x["robot"] for x in t)))
    print("splits", dict(collections.Counter(x["split"] for x in t)))


if __name__ == "__main__":
    main()
