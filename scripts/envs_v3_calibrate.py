"""envs-v3 CALIBRATED ORACLE: run every CAL_GRID candidate of a family on CALIBRATION seeds and store the best candidate
per hidden-physics bin (TRUE hidden values, evaluator-only), per probe-then-act bin and overall ('prior').

    python scripts/envs_v3_calibrate.py <family> <robot> <n_seeds> [procs]       # appends rows, then writes the table
    python scripts/envs_v3_calibrate.py table <family>                          # rebuild the table from the rows

Seeds: train bucket from 2_000_000 (envs/splits.py seed_bucket) - disjoint from the ceiling seeds (test bucket from
1_700_000). Rows: results/v2/envs/v3_cal_<family>_rows.jsonl; table: results/v2/envs/v3_oracle_<family>.json
({robot: {bins: {bin: params}, pta_bins: {...}, prior: params, n: ...}}). Ties -> higher mean progress -> lower index.
"""
import collections
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "scripts")


def seeds(n):
    from physmem.envs.splits import seed_bucket
    out, s = [], 2_000_000
    while len(out) < n:
        if seed_bucket(s) == "train":
            out.append(s)
        s += 1
    return out


def job(a):
    fam, robot, seed, k, prm = a
    from envs_v3_sens import run
    from physmem.envs import family_class
    import physmem.envs.families_v3_common as C
    r = run((fam, robot, seed, {}, "set:" + json.dumps(prm)))
    r.update(family=fam, robot=robot, cand=k, params=prm)
    return r


def table(fam):
    from physmem.envs import family_class
    grid = family_class(fam).CAL_GRID
    rows = [json.loads(l) for l in open(f"results/v2/envs/v3_cal_{fam}_rows.jsonl")]
    out = {}
    for robot in sorted({r["robot"] for r in rows}):
        R = [r for r in rows if r["robot"] == robot and "error" not in r]

        def best(sel):
            G = collections.defaultdict(list)
            for r in R:
                if sel(r):
                    G[r["cand"]].append(r)
            if not G:
                return None, 0
            k = max(G, key=lambda c: (np.mean([x["success"] for x in G[c]]), np.mean([x["progress"] for x in G[c]]), -c))
            return k, len(G[k])
        bins = sorted({r["bin"] for r in R if r.get("bin")})
        pbins = sorted({r["pta_bin"] for r in R if r.get("pta_bin")})
        ent = dict(bins={}, pta_bins={}, n_rows=len(R), detail={})
        for b in bins:
            k, n = best(lambda r, b=b: r["bin"] == b)
            ent["bins"][b] = grid[k]
            ent["detail"][b] = dict(cand=k, n=n, success={c: float(np.mean([x["success"] for x in R if x["cand"] == c
                                                                             and x["bin"] == b])) for c in range(len(grid))})
        for b in pbins:
            k, n = best(lambda r, b=b: r.get("pta_bin") == b)
            ent["pta_bins"][b] = grid[k]
        k, n = best(lambda r: True)
        ent["prior"] = grid[k]
        ent["prior_cand"] = k
        out[robot] = ent
    json.dump(out, open(f"results/v2/envs/v3_oracle_{fam}.json", "w"), indent=1)
    print(json.dumps({r: dict(bins=e["bins"], prior=e["prior"]) for r, e in out.items()}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table(sys.argv[2])
        sys.exit()
    fam, robot, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    from physmem.envs import family_class
    grid = family_class(fam).CAL_GRID
    path = f"results/v2/envs/v3_cal_{fam}_rows.jsonl"
    os.makedirs("results/v2/envs", exist_ok=True)
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            done.add((r["robot"], r["seed"], r["cand"]))
    jobs = [(fam, robot, s, k, prm) for s in seeds(n) for k, prm in enumerate(grid) if (robot, s, k) not in done]
    print(f"{len(jobs)} calibration episodes", flush=True)
    with open(path, "a") as f, Pool(procs, maxtasksperchild=20) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"[{i + 1}/{len(jobs)}]", flush=True)
    table(fam)
