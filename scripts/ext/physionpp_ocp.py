"""Physion++ OCP with SAM2 (tier-B) tracks (owner: agent EXT). docs/EXT.md §3b.

  list    : write the gb10 job list "<mp4> <map.png> <cue frame>" for readout + test trials
            (cue frame = static/start_frame_for_prediction, the frame at which humans see the red/yellow cue; the
            prediction-onset frame is part of the task definition, not a hidden property)
  readout : official-style readout (ocp.fit_eval: per property, 3 seeds, logistic regression with C grid + 5-fold
            CV) on track features. Readout fit on ALL local readout trials of that property; test = the official
            192 stimuli per property reconstructed from the human data (physionpp.official_test_ids).
            Protocols: observed = frames [0, cue); observed+simulated = + constant-velocity extrapolation;
            full = all frames (detection).

usage: python -I scripts/ext/physionpp_ocp.py list --root-remote ~/dynagraph_ext/data/physionpp > jobs.txt
       python -I scripts/ext/physionpp_ocp.py readout --tracks DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from physmem.ext import frontend_color as FC  # noqa: E402  (feature function is front-end agnostic)
from physmem.ext import ocp  # noqa: E402
from physmem.ext import physionpp as PP  # noqa: E402
from physmem.ext.tracks import TrackSet  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "ext"


def cmd_list(a):
    test_ids = set().union(*PP.official_test_ids().values())
    for split in ("readout", "test"):
        for t in PP.trials(split):
            if split == "test" and PP.stim_id(t) not in test_ids:
                continue
            m = t.meta()
            rel_mp4 = t.mp4.relative_to(PP.ROOT)
            rel_map = (t.mp4.parent / f"{t.idx}_map.png").relative_to(PP.ROOT)
            print(f"{a.root_remote}/{rel_mp4} {a.root_remote}/{rel_map} {int(m['start_frame_for_prediction'])}")


def _key(t):
    return f"{t.scenario}__{t.config}__{t.idx}_img.npz"


def _two(ts: TrackSet) -> TrackSet:
    """keep agent + patient entities (cue-prompted) in that order, for the shared feature function."""
    idx = [ts.entities.index(e) if e in ts.entities else None for e in ("agent", "patient")]
    uv = np.full((len(ts.t), 2) + ts.uv.shape[2:], np.nan, np.float32)
    area = np.zeros((len(ts.t), 2), np.float32)
    for j, i in enumerate(idx):
        if i is not None:
            uv[:, j] = ts.uv[:, i]
            area[:, j] = ts.extra["area"][:, i]
    return TrackSet(ts.kind, ts.t, uv, ["agent", "patient"], ts.keypoints, ts.image_hw, ts.source,
                    extra=dict(area=area))


def cmd_readout(a):
    tdir = Path(a.tracks)
    test_ids = PP.official_test_ids()
    res = dict(frontend="sam2-v1", protocol_note="official readout model/CV/seeds; readout fit on all local "
               "readout trials per property; test = 192/property reconstructed from official human data",
               properties={})
    for prop in ("mass", "friction", "elasticity", "deformability"):
        sets = {}
        for split in ("readout", "test"):
            X = {p: [] for p in ocp.PROTOCOLS}
            y, names = [], []
            for t in PP.trials(split):
                if t.prop != prop or (split == "test" and PP.stim_id(t) not in test_ids.get(prop, ())):
                    continue
                p = tdir / _key(t)
                if not p.exists():
                    continue
                m = t.meta()
                ts = _two(TrackSet.load(p))
                cue = int(m["start_frame_for_prediction"])          # in video frames (30 fps)
                n_obs = int(np.sum(ts.t < cue / 30.0))                  # tracks may be frame-subsampled
                o, s = FC.ocp_track_features(ts, n_obs)
                f, _ = FC.ocp_track_features(ts, len(ts.t))
                X["observed"].append(o); X["observed+simulated"].append(np.concatenate([o, s])); X["full"].append(f)
                y.append(m["label"]); names.append(PP.stim_id(t))
            sets[split] = dict(X={k: np.array(v) for k, v in X.items()}, y=np.array(y, bool), names=names)
        tr, te = sets["readout"], sets["test"]
        if len(tr["y"]) < 20 or len(te["y"]) < 20:
            print(f"[readout] {prop}: n_readout={len(tr['y'])} n_test={len(te['y'])}, skipped", flush=True)
            continue
        r = dict(n_readout=int(len(tr["y"])), n_test=int(len(te["y"])))
        for prot in ocp.PROTOCOLS:
            r[prot] = ocp.fit_eval(tr["X"][prot], tr["y"], te["X"][prot], te["y"])
        print(prop, {p: round(r[p]["mean"]["accuracy"], 3) for p in ocp.PROTOCOLS}, r["n_readout"], r["n_test"],
              flush=True)
        res["properties"][prop] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "physionpp_ocp_sam2-v1.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "readout"])
    ap.add_argument("--root-remote", default="~/dynagraph_ext/data/physionpp")
    ap.add_argument("--tracks", default=str(PP.ROOT.parent / "tracks" / "sam2-v1"))
    a = ap.parse_args()
    dict(list=cmd_list, readout=cmd_readout)[a.cmd](a)
