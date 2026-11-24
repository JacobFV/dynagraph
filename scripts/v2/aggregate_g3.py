"""MODEL-V2: aggregate paired closed-loop JSONs over training seeds -> headline table.
Per (split, family, policy): success, Δ success vs heuristic (paired, bootstrap CI over episodes pooled across seeds,
resampling seeds x episodes), effort ratio (geometric mean of per-episode effort / heuristic effort), per-seed success.
usage: aggregate_g3.py <out.json> <closed_loop_json> [more jsons from other seeds ...]"""
import json, sys
import numpy as np
out, files = sys.argv[1], sys.argv[2:]
runs = [json.load(open(f)) for f in files]
keys = sorted(set().union(*[set(r) for r in runs]))
rng = np.random.default_rng(0)
table = {}
for key in keys:
    per = [r[key]["episodes"] for r in runs if key in r]
    pols = sorted(set().union(*[set(p) for p in per]))
    H = np.concatenate([[e["success"] for e in p["heuristic"]] for p in per]).astype(float)
    He = np.concatenate([[e["effort"] for e in p["heuristic"]] for p in per])
    for pol in pols:
        if not all(pol in p for p in per):
            continue
        S = np.concatenate([[e["success"] for e in p[pol]] for p in per]).astype(float)
        E = np.concatenate([[e["effort"] for e in p[pol]] for p in per])
        d = S - H
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lr = np.log(E + 1e-3) - np.log(He + 1e-3)
        bl = [lr[rng.integers(0, len(lr), len(lr))].mean() for _ in range(2000)]
        table[f"{key}|{pol}"] = dict(n=len(S), seeds=len(per), success=float(S.mean()),
                                     per_seed=[float(np.mean([e["success"] for e in p[pol]])) for p in per],
                                     d_success=float(d.mean()), d_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                     effort_ratio=float(np.exp(lr.mean())),
                                     effort_ratio_ci=[float(np.exp(np.percentile(bl, 2.5))), float(np.exp(np.percentile(bl, 97.5)))],
                                     effort_median=float(np.median(E)), effort_p90=float(np.percentile(E, 90)))
json.dump(table, open(out, "w"), indent=1)
last = None
for k, v in table.items():
    key, pol = k.split("|")
    if key != last:
        print(f"== {key}"); last = key
    print(f"  {pol:26s} succ={v['success']:.2f} (seeds {[round(x, 2) for x in v['per_seed']]})  Δ vs heur={v['d_success']:+.2f} "
          f"[{v['d_ci'][0]:+.2f},{v['d_ci'][1]:+.2f}]  effort ratio={v['effort_ratio']:.2f} [{v['effort_ratio_ci'][0]:.2f},{v['effort_ratio_ci'][1]:.2f}]  n={v['n']}")
