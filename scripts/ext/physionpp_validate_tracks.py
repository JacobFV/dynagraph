"""Tier-B (SAM2) agent/patient track check vs Physion++ simulator segmentation (_id.mp4; evaluator only).
The object's segmentation colour is the median _id colour inside the official cue mask at the cue frame.
usage: python -I scripts/ext/physionpp_validate_tracks.py [--split readout] [--prop mass]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physionpp_ocp as O  # noqa: E402
from physmem.ext import physionpp as PP  # noqa: E402
from physmem.ext.tracks import TrackSet  # noqa: E402


def main():
    import imageio.v2 as iio
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="readout")
    ap.add_argument("--prop", default="mass")
    ap.add_argument("--tracks", default=str(PP.ROOT.parent / "tracks" / "sam2-v1"))
    a = ap.parse_args()
    agg = {e: dict(vis=0, det=0, err=[], vis_obs=0, det_obs=0) for e in ("agent", "patient")}
    n = 0
    for t in PP.trials(a.split):
        p = Path(a.tracks) / O._key(t)
        if t.prop != a.prop or not p.exists():
            continue
        ts = TrackSet.load(p)
        m = t.meta()
        f0 = int(m["start_frame_for_prediction"])
        mp = iio.imread(str(t.mp4.parent / f"{t.idx}_map.png"))
        ids = [x for x in iio.get_reader(str(t.mp4.parent / f"{t.idx}_id.mp4"))]
        cues = {"agent": (mp[..., 0] > 200) & (mp[..., 1] < 60), "patient": (mp[..., 0] > 200) & (mp[..., 1] > 200)}
        for e, cue in cues.items():
            if e not in ts.entities or cue.sum() < 8:
                continue
            j = ts.entities.index(e)
            col = np.median(ids[f0][cue].astype(int), 0)
            for k, fr in enumerate(np.round(ts.t * 30).astype(int)):
                seg = np.abs(ids[min(fr, len(ids) - 1)].astype(int) - col).sum(-1) < 30
                if seg.sum() < 8:
                    continue
                g = agg[e]
                g["vis"] += 1
                obs = fr < f0
                g["vis_obs"] += obs
                if np.isfinite(ts.uv[k, j, 0]).all():
                    g["det"] += 1
                    g["det_obs"] += obs
                    g["err"].append(float(np.hypot(*(ts.uv[k, j, 0] - np.argwhere(seg).mean(0)[::-1]))))
        n += 1
    out = dict(split=a.split, prop=a.prop, n_trials=n)
    for e, g in agg.items():
        er = np.array(g["err"])
        out[e] = dict(frames_visible=g["vis"], recall=g["det"] / max(g["vis"], 1),
                      recall_observed_window=g["det_obs"] / max(g["vis_obs"], 1),
                      median_px=float(np.median(er)) if er.size else None,
                      p90_px=float(np.percentile(er, 90)) if er.size else None,
                      frac_within_5px=float(np.mean(er < 5)) if er.size else None)
    print(json.dumps(out, indent=1))
    dst = Path(__file__).resolve().parents[2] / "results" / "ext" / f"physionpp_tracks_validation_{a.split}_{a.prop}.json"
    dst.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
