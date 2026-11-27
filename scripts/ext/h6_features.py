"""H6 actor-legal hand features from DexYCB RGB-D (owner: agent EXT). docs/EXT.md §6 (draft pre-registration).

For each sequence and the fixed view:
  1. MediaPipe hand TrackSet, keypoints lifted to camera-frame xyz with the aligned depth (human_pose.lift_depth).
  2. "up" = normal of the dominant plane (the table) fitted by RANSAC to the frame-0 depth point cloud, oriented
     toward the camera. Uses only the depth image and intrinsics (no calibration files, no labels).
  3. Wrist height h(t) above that plane (median-filtered over 5 frames).
  4. t_min = frame of minimum (smoothed) wrist height among detected frames (the hand is lowest at the grasp);
     lift-off  t_L = first frame after t_min with h(t) - h(t_min) > 0.02 m sustained for 5 frames;
     grasp onset t_G = frame of minimum wrist speed in [t_min - 15, t_L].
  5. C1 static features at t_G : thumb-index tip distance, hand 3-D bbox extent (3), wrist height.
     C2 kinematic features      : t_L - t_G (s), peak vertical wrist velocity / acceleration in [t_L, t_L + 30],
                                  rise at t_L + 15 frames, detection fraction.
Saves per-sequence TrackSet (.npz) + features JSON. The mass label is NOT read here.

usage: python -I scripts/ext/h6_features.py --subjects 20200709-subject-01 [--serial 0] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from physmem.ext import dexycb as D  # noqa: E402
from physmem.ext.human_pose import HandTracker, lift_depth  # noqa: E402

FEATURE_VERSION = "h6-feat-v1"
TEST_SUBJECTS = {"20200928-subject-07", "20201002-subject-08"}
FREEZE_MSG = "H6 pre-registration, frozen before any test run"


def _frozen() -> bool:
    import subprocess
    r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "log", "--all", "--fixed-strings",
                        f"--grep={FREEZE_MSG}", "--format=%H"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def table_normal(depth_m, K, rng, iters=300, tol=0.008):
    H, W = depth_m.shape
    v, u = np.mgrid[0:H:4, 0:W:4]
    z = depth_m[::4, ::4]
    ok = (z > 0.2) & (z < 2.5)
    P = np.stack([(u[ok] - K[0, 2]) / K[0, 0] * z[ok], (v[ok] - K[1, 2]) / K[1, 1] * z[ok], z[ok]], -1)
    best, bn, bp = 0, None, None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        c = np.sum(np.abs((P - s[0]) @ n) < tol)
        if c > best:
            best, bn, bp = c, n, s[0]
    if bn @ (-bp) < 0:          # orient toward the camera (camera at origin looks down at the table)
        bn = -bn
    return bn, bp, best / len(P)


def features(xyz, fps, n, p0):
    T = len(xyz)
    w = xyz[:, 0]
    ok = np.isfinite(w).all(-1)
    out = dict(det_frac=float(ok.mean()))
    if ok.sum() < 10:
        return out
    h = np.full(T, np.nan)
    h[ok] = (w[ok] - p0) @ n
    hi = np.interp(np.arange(T), np.nonzero(ok)[0], h[ok])
    hs = np.array([np.median(hi[max(0, t - 2): t + 3]) for t in range(T)])
    first, last = np.nonzero(ok)[0][[0, -1]]
    tmin = first + int(np.argmin(hs[first: last + 1]))
    above = hs - hs[tmin] > 0.02
    tL = next((t for t in range(tmin, T - 5) if above[t: t + 5].all()), None)
    out["lift_detected"] = tL is not None
    if tL is None:
        return out
    wi = np.stack([np.interp(np.arange(T), np.nonzero(ok)[0], w[ok, d]) for d in range(3)], -1)
    sp = np.linalg.norm(np.gradient(wi, 1 / fps, axis=0), axis=-1)
    a0 = max(0, tmin - 15)
    tG = a0 + int(np.argmin(sp[a0: tL + 1]))
    vz = np.gradient(hs, 1 / fps)
    az = np.gradient(vz, 1 / fps)
    seg = slice(tL, min(T, tL + 30))
    k = xyz[tG]
    ap = float(np.linalg.norm(k[4] - k[8])) if np.isfinite(k[[4, 8]]).all() else np.nan
    kk = k[np.isfinite(k).all(-1)]
    ext = (kk.max(0) - kk.min(0)).tolist() if len(kk) > 3 else [np.nan] * 3
    out.update(t_grasp=int(tG), t_lift=int(tL), c1_aperture=ap, c1_extent=ext, c1_wrist_h=float(hs[tG]),
               c2_grasp_to_lift_s=float((tL - tG) / fps), c2_peak_vz=float(np.max(vz[seg])),
               c2_peak_az=float(np.max(az[seg])), c2_rise_15=float(hs[min(T - 1, tL + 15)] - hs[tL]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="")
    ap.add_argument("--serial", type=int, default=0)
    ap.add_argument("--out", default=str(D.ROOT.parent / "h6" / FEATURE_VERSION))
    ap.add_argument("--allow-test", action="store_true", help="only valid after the H6 freeze commit")
    a = ap.parse_args()
    if a.allow_test and not _frozen():
        sys.exit(f"refusing --allow-test: no commit '{FREEZE_MSG}' found (docs/EXT.md §6 freeze rule)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    subs = [s for s in a.subjects.split(",") if s]
    for q in D.sequences(subjects=subs or None):
        if q.subject in TEST_SUBJECTS and not a.allow_test:
            continue                       # TEST subjects are untouched until the pre-registration is frozen
        serial = q.serials[a.serial]
        if not D.view_complete(q, serial):
            continue
        dst = out / f"{q.subject}__{q.name}.json"
        if dst.exists():
            continue
        rgb, dep, K = D.load_view(q, serial)
        ht = HandTracker(num_hands=1)
        ts = ht(rgb, fps=D.FPS, source=f"dexycb/{q.subject}/{q.name}/{serial}", depth=dep, K=K, depth_scale=1e-3)
        ht.close()
        xyz = np.where(np.isfinite(ts.xyz[:, 0]), ts.xyz[:, 0], ts.xyz[:, 1])
        n, p0, inl = table_normal(dep[0] * 1e-3, K, np.random.default_rng(0))
        f = features(xyz, D.FPS, n, p0)
        f.update(subject=q.subject, seq=q.name, serial=serial, table_inlier_frac=float(inl), version=FEATURE_VERSION,
                 source=ts.source)
        ts.save(out / f"{q.subject}__{q.name}.npz")
        dst.write_text(json.dumps(f, indent=1))
        print(q.name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()
                       if k not in ("source",)}, flush=True)


if __name__ == "__main__":
    main()
