"""Scan generated shards for simulator blow-ups (MODEL v2_train.sanitize criteria, in physical units).

An episode is 'blown' if any event value is non-finite, any WRENCH force > F_LIM N (or torque > T_LIM Nm), any
POSE3/EE_POSE position > P_LIM m from the origin, or final_metrics.sim_instabilities > 0.
Writes a per (robot, family) table. Usage: python scripts/envs_blowup_scan.py 'data/.../tranche6_r5/**/*.pkl.gz' [--procs 4]
"""
import argparse, glob, gzip, json, pickle, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

F_LIM, T_LIM, P_LIM, V_LIM = 2000.0, 300.0, 5.0, 50.0


def scan_file(path):
    from physmem.schema import Prop
    out = []
    for r in pickle.load(gzip.open(path)):
        ev = r.get("events")
        t = r["task"]
        why = []
        m = r.get("final_metrics") or {}
        if m.get("sim_instabilities", 0):
            why.append("badqacc")
        if isinstance(ev, dict):
            a = ev["arr"]
            v = a["val"]
            if not np.isfinite(v).all() or not np.isfinite(a["var"]).all():
                why.append("nonfinite")
            vv = np.nan_to_num(v, nan=0, posinf=1e30, neginf=-1e30)
            pr = a["prop"]
            w = pr == int(Prop.WRENCH)
            if w.any():
                if np.abs(vv[w, :3]).max() > F_LIM: why.append("force")
                if np.abs(vv[w, 3:6]).max() > T_LIM: why.append("torque")
            for p in ("POSE3", "EE_POSE"):
                s = pr == int(getattr(Prop, p))
                if s.any() and np.abs(vv[s, :3]).max() > P_LIM: why.append("pos_" + p)
            if hasattr(Prop, "VELOCITY"):
                s = pr == int(Prop.VELOCITY)
                if s.any() and np.abs(vv[s, :3]).max() > V_LIM: why.append("vel")
            fmax = float(np.abs(vv[w, :3]).max()) if w.any() else 0.0
        else:
            fmax = float("nan")
        out.append(dict(robot=t["robot"], family=t["family"], split=t["split"], seed=t["seed"], why=why, fmax=fmax,
                        policy=r.get("policy"), err=bool(r.get("error")), path=path))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    files = sorted(set(f for p in a.patterns for f in glob.glob(p, recursive=True)))
    with Pool(a.procs) as pool:
        rows = [x for part in pool.imap_unordered(scan_file, files) for x in part]
    agg = defaultdict(lambda: [0, 0, defaultdict(int)])
    for r in rows:
        k = (r["robot"], r["family"])
        agg[k][0] += 1
        if r["why"]:
            agg[k][1] += 1
            for w in r["why"]: agg[k][2][w] += 1
    tot = sum(v[0] for v in agg.values()); bad = sum(v[1] for v in agg.values())
    print(f"files={len(files)} episodes={tot} blown={bad} ({100*bad/max(tot,1):.2f}%)")
    print("| robot | family | n | blown | % | reasons |\n|---|---|---|---|---|---|")
    for k in sorted(agg):
        n, b, why = agg[k]
        if b:
            print(f"| {k[0]} | {k[1]} | {n} | {b} | {100*b/n:.1f} | {dict(why)} |")
    if a.out:
        json.dump([r for r in rows if r["why"]], open(a.out, "w"), default=str)
