"""Task 6 reproduction: re-evaluate a saved checkpoint from scratch (fresh process, CPU) and
compare to the logged rows in results/metrics.jsonl. Writes results/reproduction.json."""
import json, sys
import numpy as np, torch, yaml
sys.path.insert(0, "src")
from physmem import evaluate as E
from physmem.models.losses import fit_loss_scales
from physmem.train import RES, fit_target_scaler, load_dataset

cfg = yaml.safe_load(open("configs/sprint.yaml"))
variant, seed = "b11", 1
ck = RES / "checkpoints" / f"compact-main-{variant}-s{seed}.pt"
model = E.load_compact(ck, "cpu")
tr = load_dataset("train", cfg, "cpu")
ym, ys = [x.numpy().astype(np.float64) for x in fit_target_scaler(tr)]
scales = fit_loss_scales(tr.iterate(1024))
te = load_dataset("test", cfg, "cpu")
pred = E.predict(model, te, passes=4, bs=256)
lab = E.labels_np(te)
met = E.sample_metrics(pred, lab, ym, ys, scales)
logged = {}
for l in open(RES / "metrics.jsonl"):
    r = json.loads(l)
    if r.get("experiment") == "primary" and r.get("model") == "compact" and r.get("variant") == variant and r.get("seed") == seed \
            and r.get("stratum") == "test" and r.get("passes", 4) == 4 and "tag" not in r:
        logged[r["metric"]] = r["value"]
out = {"checkpoint": str(ck), "variant": variant, "seed": seed, "tolerance_rel": 0.01, "metrics": {}}
for name, key, sl in (("pos_err_m", "pos_err_m@h0.5", 2), ("nll", "nll@all", None), ("yaw_err_rad", "yaw_err_rad@h0.5", 2)):
    a = met[name][:, sl] if sl is not None else np.nanmean(met[name], 1)
    _, pr = E.root_means(a, lab["root"])
    v = float(np.nanmean(pr))
    lv = logged.get(key)
    out["metrics"][key] = {"reproduced": v, "logged": lv,
                           "rel_diff": None if lv is None else abs(v - lv) / max(abs(lv), 1e-9)}
out["pass"] = all(m["rel_diff"] is not None and m["rel_diff"] <= 0.01 for m in out["metrics"].values())
(RES / "reproduction.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
