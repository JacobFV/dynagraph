"""Tier-A object-track front end for Physion red/yellow stimuli (owner: agent EXT).

Classical colour segmentation, no learned weights (provenance: this file + OpenCV). Physion's model-facing stimuli
paint the agent red and the patient yellow for the whole movie (paper §2.4), so these two entities can be tracked
from pixels alone. Per frame and entity we report the centroid, bbox corners and pixel area of the largest connected
component in the colour band; frames where it is smaller than MIN_AREA are NaN (occluded / off-screen).

Thresholds were set by inspection of 3 TEST-free readout/train frames (docs/EXT.md §4); they are fixed constants.
"""
from __future__ import annotations

import numpy as np

from .tracks import KIND_OBJECT, TrackSet

VERSION = "color-v2"   # v2: patient band widened for washed-out zones (tuned on Dominoes readout only)
# OpenCV HSV: H in [0, 180), S, V in [0, 255]
BANDS = {
    "agent": [((0, 110, 35), (3, 255, 255)), ((168, 110, 35), (180, 255, 255))],      # red (H~177; floors are H 5-8)
    "patient": [((22, 25, 200), (30, 255, 255))],                                     # pale/washed-out yellow zone
}
MIN_AREA = 12
KEYPOINTS = ["centroid", "bbox_min", "bbox_max"]


def _mask(hsv, bands):
    import cv2
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in bands:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return m


def track_frames(rgb: np.ndarray, fps: float = 30.0, source: str = "") -> TrackSet:
    """rgb (T, H, W, 3) uint8 -> TrackSet(kind=object, entities=[agent, patient], keypoints=KEYPOINTS)."""
    import cv2
    T, H, W, _ = rgb.shape
    ents = list(BANDS)
    uv = np.full((T, len(ents), len(KEYPOINTS), 2), np.nan, np.float32)
    area = np.zeros((T, len(ents)), np.float32)
    kern = np.ones((3, 3), np.uint8)
    for t in range(T):
        hsv = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2HSV)
        for e, name in enumerate(ents):
            m = cv2.morphologyEx(_mask(hsv, BANDS[name]), cv2.MORPH_OPEN, kern)
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
            if n <= 1:
                continue
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            a = stats[i, cv2.CC_STAT_AREA]
            if a < MIN_AREA:
                continue
            x, y, w, h = stats[i, :4]
            uv[t, e, 0] = cent[i]
            uv[t, e, 1] = (x, y)
            uv[t, e, 2] = (x + w, y + h)
            area[t, e] = a
    return TrackSet(kind=KIND_OBJECT, t=np.arange(T) / fps, uv=uv, entities=ents, keypoints=KEYPOINTS,
                    image_hw=(H, W), source=f"{source} | frontend={VERSION}", extra=dict(area=area))


def ocp_track_features(ts: TrackSet, t_obs: int = 45, horizon_s=(0.5, 1.0, 2.0, 4.0)) -> np.ndarray:
    """fixed-length features from the OBSERVED part of a red/yellow TrackSet (used by our reference readout).
    observed: agent/patient centroid and bbox at the last observed frame, velocity over the last 10 frames,
    current centroid distance, min distance so far, bbox gap, visibility fractions, mean areas.
    simulated (constant-velocity extrapolation of both centroids): distance and bbox gap at each horizon.
    Returns (obs_feats, sim_feats)."""
    uv = ts.uv[:t_obs]
    T = uv.shape[0]
    fps = 1.0 / np.median(np.diff(ts.t)) if len(ts.t) > 1 else 30.0
    H, W = ts.image_hw
    sc = float(max(H, W))

    def last_valid(e, k=0):
        ok = np.isfinite(uv[:, e, k]).all(-1)
        idx = np.nonzero(ok)[0]
        return (uv[idx[-1], e, k] / sc, idx) if len(idx) else (np.full(2, np.nan), idx)

    def vel(e):
        ok = np.isfinite(uv[:, e, 0]).all(-1)
        idx = np.nonzero(ok)[0]
        idx = idx[idx >= T - 10]
        if len(idx) < 2:
            return np.zeros(2)
        p = np.polyfit(idx / fps, uv[idx, e, 0] / sc, 1)
        return p[0]

    pa, ia = last_valid(0)
    pp, ip = last_valid(1)
    va, vp = vel(0), vel(1)
    ba = np.concatenate([uv[ia[-1], 0, 1], uv[ia[-1], 0, 2]]) / sc if len(ia) else np.full(4, np.nan)
    bp = np.concatenate([uv[ip[-1], 1, 1], uv[ip[-1], 1, 2]]) / sc if len(ip) else np.full(4, np.nan)

    def gap(b1, b2):
        dx = max(b2[0] - b1[2], b1[0] - b2[2], 0)
        dy = max(b2[1] - b1[3], b1[1] - b2[3], 0)
        return float(np.hypot(dx, dy))

    d_all = np.linalg.norm(uv[:, 0, 0] - uv[:, 1, 0], axis=-1) / sc
    vis = np.isfinite(uv[:, :, 0]).all(-1).mean(0)
    area = ts.extra.get("area", np.zeros((T, 2)))[:t_obs]
    obs = np.concatenate([pa, pp, va, vp, ba, bp, [np.linalg.norm(pa - pp), np.nanmin(d_all) if np.isfinite(d_all).any()
                          else np.nan, gap(ba, bp)], vis, area.mean(0) / sc ** 2])
    sim = []
    for h in horizon_s:
        qa, qp = pa + va * h, pp + vp * h
        sh_a, sh_p = np.tile(va * h, 2), np.tile(vp * h, 2)
        sim += [np.linalg.norm(qa - qp), gap(ba + sh_a, bp + sh_p)]
    obs = np.nan_to_num(obs, nan=-1.0)
    sim = np.nan_to_num(np.array(sim), nan=-1.0)
    return obs, sim
