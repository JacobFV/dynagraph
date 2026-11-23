"""Summarize data/v2_envs shards: scripted success per (family, split), episodes/hour, effort -> markdown + json."""
import glob
import gzip
import json
import os
import pickle

import numpy as np

out = {}
for path in sorted(glob.glob("data/v2_envs/*/*.pkl.gz")):
    fam, split = path.split("/")[-2], os.path.basename(path)[:-7]
    recs = pickle.load(gzip.open(path))
    out.setdefault(fam, {})[split] = dict(
        n=len(recs), success=float(np.mean([r["success"] for r in recs])),
        wall_s=float(np.mean([r["wall_s"] for r in recs])), sim_s=float(np.mean([r["sim_t"] for r in recs])),
        calls=float(np.mean([len(r["steps"]) for r in recs])),
        effort=float(np.mean([sum(s["effort"]["cost"] for s in r["steps"]) for r in recs])),
        violations=int(sum(r["final_metrics"].get("integrity_violations", 0) for r in recs)),
        events=float(np.mean([r["n_events"] for r in recs])))
json.dump(out, open("results/v2/envs/generation_summary.json", "w"), indent=1)
splits = ["train", "val", "test", "heldout_layout", "heldout_objects", "heldout_family"]
print("| family | " + " | ".join(splits) + " | wall s/ep | eps/h/proc | sim s/ep | calls/ep | events/ep | viol |")
print("|---" * (len(splits) + 6) + "|")
for fam, d in out.items():
    allr = list(d.values())
    n = sum(v["n"] for v in allr)
    w = sum(v["wall_s"] * v["n"] for v in allr) / n
    cells = [f"{d[s]['success']:.2f} ({d[s]['n']})" if s in d else "-" for s in splits]
    print(f"| {fam} | " + " | ".join(cells) + f" | {w:.1f} | {3600 / w:.0f} | "
          f"{sum(v['sim_s'] * v['n'] for v in allr) / n:.0f} | {sum(v['calls'] * v['n'] for v in allr) / n:.1f} | "
          f"{sum(v['events'] * v['n'] for v in allr) / n:.0f} | {sum(v['violations'] for v in allr)} |")
