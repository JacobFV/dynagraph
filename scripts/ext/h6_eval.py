"""H6 analysis (owner: agent EXT). Implements docs/EXT.md §6 (DRAFT until the freeze commit).

Inputs: per-sequence feature JSONs from h6_features.py (+ optional per-sequence model outputs for TYPED/GENERIC,
a JSON {"<subject>__<seq>": {"typed": [z, log_sd], "generic": [f1..fk]}} supplied by V3-MODEL).
Readouts (identical for every arm): ridge regression to log-mass, alpha chosen by object-grouped CV on the fit set,
fit with leave-objects-out (4 object folds: object_index % 4 == k) so every prediction comes from a readout that never
saw that object. Arms: C0 train mean, C1 grasp aperture/extent, C2 lift kinematics, C3 identity lookup (privileged
diagnostic), TYPED, GENERIC. Primary: mean of d_i = e_GENERIC - e_TYPED with an object-cluster bootstrap
(10,000 resamples, seed 0).

--split val  : fit on TRAIN subjects, evaluate on VAL subject (allowed any time).
--split test : fit on TRAIN subjects, evaluate on TEST subjects; refuses to run before the freeze commit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physmem.ext import dexycb as D  # noqa: E402
from h6_features import TEST_SUBJECTS, _frozen  # noqa: E402

TRAIN = ["20200709-subject-01", "20200813-subject-02", "20200820-subject-03", "20200903-subject-04",
         "20200908-subject-05"]
VAL = ["20200918-subject-06"]
OBJ_INDEX = {v: k for k, v in D.YCB_CLASSES.items()}
ALPHAS = np.logspace(-3, 3, 13)


def _load(feat_dir: Path, subjects, model_out: dict):
    rows = []
    seqs = {(q.subject, q.name): q for q in D.sequences(subjects=subjects)}
    for (sub, name), q in seqs.items():
        p = feat_dir / f"{sub}__{name}.json"
        if not p.exists():
            continue
        f = json.loads(p.read_text())
        if f.get("det_frac", 0) < 0.5 or not f.get("lift_detected"):
            continue                                   # pre-registered exclusion (actor-legal signals only)
        key = f"{sub}__{name}"
        rows.append(dict(
            key=key, obj=q.grasped, obj_idx=OBJ_INDEX[q.grasped],
            y=np.log(q.mass_kg),
            C1=[f["c1_aperture"], *f["c1_extent"], f["c1_wrist_h"]],
            C2=[f["c2_grasp_to_lift_s"], f["c2_peak_vz"], f["c2_peak_az"], f["c2_rise_15"]],
            TYPED=model_out.get(key, {}).get("typed"), GENERIC=model_out.get(key, {}).get("generic")))
    return rows


def _ridge_fit_predict(Xtr, ytr, gtr, Xte):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold, cross_val_score
    Xtr, Xte = np.nan_to_num(np.asarray(Xtr, float)), np.nan_to_num(np.asarray(Xte, float))
    ng = len(set(gtr))
    best = max(ALPHAS, key=lambda a: cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=a)), Xtr, ytr,
                                                       groups=gtr, cv=GroupKFold(min(5, ng)),
                                                       scoring="neg_mean_absolute_error").mean())
    m = make_pipeline(StandardScaler(), Ridge(alpha=best)).fit(Xtr, ytr)
    return m.predict(Xte), float(best)


def predict_arm(arm, tr, te):
    """leave-objects-out: for each fold k, fit on TRAIN rows whose object index % 4 != k, predict TEST rows with
    object index % 4 == k."""
    pred = np.full(len(te), np.nan)
    for k in range(4):
        tri = [r for r in tr if r["obj_idx"] % 4 != k]
        tei = [i for i, r in enumerate(te) if r["obj_idx"] % 4 == k]
        if not tei:
            continue
        ytr = np.array([r["y"] for r in tri])
        if arm == "C0":
            pred[tei] = ytr.mean()
        elif arm == "C3":
            lut = {}
            for r in tri:
                lut.setdefault(r["obj"], []).append(r["y"])
            pred[tei] = [np.mean(lut[te[i]["obj"]]) if te[i]["obj"] in lut else ytr.mean() for i in tei]
        else:
            if any(r[arm] is None for r in tri) or any(te[i][arm] is None for i in tei):
                return None
            p, _ = _ridge_fit_predict([r[arm] for r in tri], ytr, [r["obj"] for r in tri], [te[i][arm] for i in tei])
            pred[tei] = p
    return pred


def cluster_bootstrap(d, groups, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    g = np.array(groups)
    ug = np.unique(g)
    by = {u: d[g == u] for u in ug}
    means = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug))
        means.append(np.mean(np.concatenate([rng.choice(by[u], len(by[u])) for u in pick])))
    return float(np.mean(d)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(D.ROOT.parent / "h6" / "h6-feat-v1"))
    ap.add_argument("--model-out", default="")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()
    if a.split == "test" and not _frozen():
        sys.exit("refusing --split test before the H6 freeze commit (docs/EXT.md §6)")
    mo = json.loads(Path(a.model_out).read_text()) if a.model_out else {}
    tr = _load(Path(a.feat), TRAIN, mo)
    te = _load(Path(a.feat), sorted(TEST_SUBJECTS) if a.split == "test" else VAL, mo)
    if not tr or not te:
        sys.exit(f"not enough sequences: train={len(tr)} eval={len(te)}")
    y = np.array([r["y"] for r in te])
    res = dict(split=a.split, n_train=len(tr), n_eval=len(te), arms={})
    errs = {}
    for arm in ("C0", "C1", "C2", "C3", "TYPED", "GENERIC"):
        p = predict_arm(arm, tr, te)
        if p is None:
            continue
        errs[arm] = np.abs(p - y)
        res["arms"][arm] = dict(mae_log=float(np.mean(errs[arm])),
                                spearman_obj=_obj_spearman(te, p))
    if "TYPED" in errs and "GENERIC" in errs:
        res["primary_generic_minus_typed"] = cluster_bootstrap(errs["GENERIC"] - errs["TYPED"], [r["obj"] for r in te])
    print(json.dumps(res, indent=1))
    out = Path(__file__).resolve().parents[2] / "results" / "ext" / f"h6_{a.split}.json"
    out.write_text(json.dumps(res, indent=1))


def _obj_spearman(te, p):
    from scipy.stats import spearmanr
    objs = sorted(set(r["obj"] for r in te))
    if len(objs) < 3:
        return None
    mp = [np.mean([p[i] for i, r in enumerate(te) if r["obj"] == o]) for o in objs]
    my = [np.mean([r["y"] for r in te if r["obj"] == o]) for o in objs]
    return float(spearmanr(mp, my).statistic)


if __name__ == "__main__":
    main()
