"""r9 trust gate (13a): a family is trusted for a (regime, variant) if the mean over seeds 0-2 of the model's VAL
success AUC (last val line of each training log) is >= 0.8. Writes results/v2/trust_r9_<regime>_<variant>.json."""
import glob, json, re
import numpy as np
for f0 in sorted(glob.glob("results/v2/r9_select_*.txt")):
    reg = f0.split("_")[-1][:-4]
    for line in open(f0):
        if not line.startswith("SELECT"):
            continue
        _, v, lr, d, _ = line.split()
        aucs = {}
        for s in (0, 1, 2):
            last = None
            for l in open(f"logs/v2_robots_r9_{reg}_{v}_s{s}_lr{lr}_d{d}.log"):
                if l.startswith('{"step"') and '"val"' in l:
                    last = json.loads(l)["val"]
            for k, x in (last or {}).items():
                if k.endswith(":success_auc"):
                    aucs.setdefault(k.split(":")[0], []).append(x)
        fams = {fm: dict(auc=float(np.nanmean(a)), trusted=bool(np.nanmean(a) >= 0.8)) for fm, a in aucs.items()}
        json.dump(dict(regime=reg, variant=v, lr=lr, d=d, families=fams), open(f"results/v2/trust_r9_{reg}_{v}.json", "w"), indent=1)
        print(reg, v, {fm: round(x["auc"], 3) for fm, x in fams.items()})
