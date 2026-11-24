"""MODEL-V2: per-robot calibration of effort (log bias, log RMSE) and success (Brier, mean pred vs rate) on a split."""
import sys, numpy as np, torch
from physmem.v2_train import load_model, load_stores
ck, data, split = sys.argv[1], sys.argv[2], sys.argv[3]
m, _ = load_model(ck, torch.device("cpu"))
st = load_stores(data, split, torch.device("cpu"))
acc = {}
with torch.no_grad():
    for fam, s in st.items():
        for i in range(0, s.n, 512):
            idx = torch.arange(i, min(i + 512, s.n))
            b, lab = s.gather(idx)
            d = m.decode(m(b))
            for j, r in enumerate(s.robot[i:i + 512]):
                a = acc.setdefault(r, dict(le=[], lp=[], ps=[], ys=[]))
                a["le"].append(float(lab.effort[j])); a["lp"].append(float(d["log_effort"][j]))
                a["ps"].append(float(d["success"][j])); a["ys"].append(float(lab.events[j, 0]))
for r, a in sorted(acc.items()):
    le, lp, ps, ys = map(np.array, (a["le"], a["lp"], a["ps"], a["ys"]))
    print(f"{r:12s} n={len(le):6d} effort log-bias(pred-true)={np.mean(lp - le):+.2f} log-RMSE={np.sqrt(np.mean((lp - le) ** 2)):.2f} "
          f"| success mean pred={ps.mean():.3f} rate={ys.mean():.3f} Brier={np.mean((ps - ys) ** 2):.3f}")
