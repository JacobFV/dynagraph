"""select (lr, d) per variant from the seed-0 grid: lowest mean VAL (success_brier + progress_rmse) over families,
from the LAST {"step":..., "val": {...}} line of each run log. usage: r9_select.py REGIME"""
import glob, json, re, sys
import numpy as np
reg = sys.argv[1]
best = {}
for f in sorted(glob.glob(f"logs/v2_robots_r9_{reg}_*_s0_lr*_d*.log")):
    m = re.search(rf"v2_robots_r9_{reg}_(\w+?)_s0_lr([0-9e.-]+)_d(\d+)\.log", f)
    v, lr, d = m.group(1), m.group(2), m.group(3)
    last = None
    for l in open(f):
        if l.startswith('{"step"') and '"val"' in l:
            last = json.loads(l)
    if last is None:
        continue
    val = last["val"]
    fams = sorted({k.split(":")[0] for k in val})
    sc = [val.get(f"{fm}:success_brier", np.nan) + val.get(f"{fm}:progress_rmse", np.nan) for fm in fams]
    score = float(np.nanmean(sc))
    print(f"{v} lr {lr} d {d} step {last['step']} score {score:.4f} " + " ".join(f"{fm}={s:.3f}" for fm, s in zip(fams, sc)))
    if v not in best or score < best[v][2]:
        best[v] = (lr, d, score)
for v, (lr, d, sc) in sorted(best.items()):
    print("SELECT", v, lr, d, f"{sc:.4f}")
