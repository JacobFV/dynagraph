"""Summarise results/g1/<tag>/metrics.jsonl per model (paired seeds). Usage: demo_g1_summarize.py tag [tag...]"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

for tag in sys.argv[1:]:
    rows = [json.loads(l) for l in open(Path("results/g1") / tag / "metrics.jsonl")]
    last = {}
    for r in rows:  # keep the latest row per (pair, model)
        last[(r["pair_id"], r["model"])] = r
    by = defaultdict(list)
    for r in last.values():
        by[r["model"]].append(r)
    summ = {}
    print(f"== {tag}")
    for k, v in sorted(by.items()):
        sp = [s for x in v for s in x.get("chosen_speeds", [])]
        summ[k] = dict(n=len(v), success=int(sum(x["success"] for x in v)),
                       mean_final_dist_cm=100 * float(np.mean([x["final_dist"] for x in v])),
                       median_final_dist_cm=100 * float(np.median([x["final_dist"] for x in v])),
                       mean_pushes=float(np.mean([x["n_pushes"] for x in v])),
                       mean_objective=float(np.mean([x.get("objective", np.nan) for x in v])),
                       dist_after_first_push_cm=100 * float(np.nanmean([x["dist_after_first_push"] if x["dist_after_first_push"] is not None else np.nan for x in v])),
                       frac_fast_chosen=float(np.mean([s >= 0.45 for s in sp])) if sp else 0.0,
                       mean_latency_ms=1000 * float(np.mean([x["mean_inference_latency"] for x in v])),
                       rtf=float(np.mean([x["real_time_factor"] for x in v])),
                       audit_clean=all(x["audit_clean"] for x in v))
        s = summ[k]
        print(f"  {k:36s} n {s['n']:2d} success {s['success']:2d} final {s['mean_final_dist_cm']:5.1f} (med {s['median_final_dist_cm']:4.1f}) cm "
              f"pushes {s['mean_pushes']:.2f} J {s['mean_objective']:.3f} first-push {s['dist_after_first_push_cm']:5.1f} cm fast {s['frac_fast_chosen']:.2f} "
              f"lat {s['mean_latency_ms']:.1f} ms clean {s['audit_clean']}")
    (Path("results/g1") / tag / "summary.json").write_text(json.dumps(summ, indent=1))
