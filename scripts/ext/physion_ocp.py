"""Physion v1 OCP with the tier-A colour track front end (owner: agent EXT). docs/EXT.md §3-4.

  track    : run frontend_color on every local test movie and readout stimulus, cache TrackSets (.npz)
  validate : front-end accuracy vs the simulator segmentation on readout stimuli (evaluator-side check only)
  readout  : official readout protocol (per scenario, 3 seeds, StandardScaler + LogisticRegression, C grid
             logspace(-8, 8, 17), 5-fold stratified CV) on track features; test = the official 150 test movies
             per scenario. The readout-FITTING set is the local prefix subset -> results are labelled
             "reduced readout set (n per scenario), not the official protocol".

usage: python -I scripts/ext/physion_ocp.py track|validate|readout [--scen Collide,...] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from physmem.ext import frontend_color as FC  # noqa: E402
from physmem.ext import ocp  # noqa: E402
from physmem.ext import physion as P  # noqa: E402
from physmem.ext.tracks import TrackSet  # noqa: E402

TRK = P.ROOT / "tracks" / FC.VERSION
OUT = Path(__file__).resolve().parents[2] / "results" / "ext"


def _track_one(stim):
    dst = TRK / stim.split / stim.scenario / f"{stim.name}.npz"
    if dst.exists():
        return dst, None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if stim.split == "test":
        rgb, label = P.read_mp4(stim.path), stim.label
    else:
        d = P.read_hdf5(stim.path)
        rgb, label = d["rgb"], d["label"]
    ts = FC.track_frames(rgb, P.FPS, source=f"physion-v1/{stim.split}/{stim.scenario}/{stim.path.name}")
    ts.save(dst)
    lab = dst.with_suffix(".label.json")
    lab.write_text(json.dumps(dict(label=bool(label), n_frames=int(len(rgb)))))   # evaluator-only
    return dst, label


def cmd_track(a):
    stims = [s for s in P.test_set(scenarios=a.scen)] + P.readout_set(scenarios=a.scen)
    t0 = time.time()
    for i, s in enumerate(stims[: a.limit]):
        _track_one(s)
        if i % 100 == 0:
            print(f"[track] {i}/{len(stims)} {time.time() - t0:.0f}s", flush=True)


def _load(split, scen):
    X_obs, X_sim, X_full, y, names = [], [], [], [], []
    for p in sorted((TRK / split / scen).glob("*.npz")):
        ts = TrackSet.load(p)
        lab = json.loads(p.with_suffix(".label.json").read_text())["label"]
        o, s = FC.ocp_track_features(ts, P.T_VIS)
        of, _ = FC.ocp_track_features(ts, len(ts.t))          # "full": same features over the whole movie
        X_obs.append(o); X_sim.append(np.concatenate([o, s])); X_full.append(of); y.append(lab)
        names.append(p.stem)
    return dict(observed=np.array(X_obs), **{"observed+simulated": np.array(X_sim)}, full=np.array(X_full),
                y=np.array(y, bool), names=names)


def cmd_readout(a):
    human, parts = P.human_tables()
    res = dict(frontend=FC.VERSION, protocol_note="official readout model/CV/seeds; readout-fitting set is a "
               "local PREFIX subset of the official 1000/scenario (not the official protocol); test = official "
               "150/scenario", scenarios={})
    for scen in a.scen:
        tr, te = _load("readout", scen), _load("test", scen)
        if len(tr["y"]) < 20 or len(set(tr["y"])) < 2:
            print(f"[readout] {scen}: only {len(tr['y'])} readout stimuli, skipped", flush=True)
            continue
        r = dict(n_readout=int(len(tr["y"])), readout_pos_frac=float(tr["y"].mean()), n_test=int(len(te["y"])))
        for prot in ocp.PROTOCOLS:
            r[prot] = ocp.fit_eval(tr[prot], tr["y"], te[prot], te["y"], human=human, names_te=te["names"],
                                   participants=parts)
        print(scen, {p: round(r[p]["mean"]["accuracy"], 3) for p in ocp.PROTOCOLS}, f"n_readout={r['n_readout']}",
              flush=True)
        res["scenarios"][scen] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"physion_ocp_{FC.VERSION}.json").write_text(json.dumps(res, indent=1))


def cmd_validate(a):
    """centroid error of the colour tracker vs simulator segmentation masks of agent (target) and patient (zone)."""
    rows = {}
    for scen in a.scen:
        errs, det, vis = {0: [], 1: []}, {0: 0, 1: 0}, {0: 0, 1: 0}
        for st in P.readout_set(scenarios=[scen])[: a.limit or 20]:
            d = P.read_hdf5(st.path, with_seg=True)
            m = d["meta"]
            ids = list(m["object_ids"])
            ts = FC.track_frames(d["rgb"])
            for e, oid in enumerate([m["target_id"], m["zone_id"]]):
                if oid not in ids:
                    continue
                c = m["seg_colors"][ids.index(oid)]
                for t in range(len(d["rgb"])):
                    mk = (d["seg"][t] == c).all(-1)
                    if mk.sum() < FC.MIN_AREA:
                        continue
                    vis[e] += 1
                    if np.isfinite(ts.uv[t, e, 0]).all():
                        det[e] += 1
                        errs[e].append(float(np.hypot(*(ts.uv[t, e, 0] - np.argwhere(mk).mean(0)[::-1]))))
        rows[scen] = {n: dict(frames_visible=vis[e], recall=det[e] / max(vis[e], 1),
                              median_px=float(np.median(errs[e])) if errs[e] else None,
                              p90_px=float(np.percentile(errs[e], 90)) if errs[e] else None)
                      for e, n in enumerate(["agent", "patient"])}
        print(scen, rows[scen], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"physion_frontend_validation_{FC.VERSION}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["track", "validate", "readout"])
    ap.add_argument("--scen", default=",".join(P.SCENARIOS))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    a.scen = a.scen.split(",")
    dict(track=cmd_track, validate=cmd_validate, readout=cmd_readout)[a.cmd](a)
