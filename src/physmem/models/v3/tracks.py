"""Third-person keypoint tracker model (owner: V3-MODEL). Pure numpy, actor/evaluator split explicit.

The simulator gives TRUE 3-D keypoint positions (evaluator side). `CameraTracker.observe` turns them into what a real
RGB-D keypoint tracker would deliver, and the actor-side `CameraTracker.backproject` lifts the detections into the
world frame with the actor's (slightly wrong) calibration. Nothing else about the simulator crosses to the actor.

Error model (all per-episode randomised; constants documented in docs/V3_GENERAL.md section 0):
  * pinhole camera, f = 600 px, 640 x 480, random pose (distance 0.8-1.4 m, azimuth +-40 deg, elevation 15-45 deg)
  * frame times: nominal 30 Hz, Gaussian timestamp jitter (sd 3 ms), frame drops (p = 0.08)
  * pixel noise sd 1.0 px, outliers (p = 0.01, sd 15 px), per-keypoint dropout (p = 0.05)
  * occluder: with p = 0.5 a vertical pillar hides an image band 40-80 px wide (keypoints inside are not detected)
  * depth: z (1 + N(0, 0.003)) + N(0, 1 mm)  (RGB-D style; noise grows with range)
  * actor calibration error: focal x (1 + N(0, 0.005)), rotation N(0, 0.2 deg) per axis, translation N(0, 1 mm)
"""
from __future__ import annotations

import math

import numpy as np

F_PX, W_PX, H_PX = 600.0, 640.0, 480.0


def _rot(ax, ay, az):
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def look_at(eye, target):
    """world->camera rotation (rows = camera axes x right, y down, z forward) for a camera at eye looking at target."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])


class CameraTracker:
    def __init__(self, rng: np.random.Generator, center=(0.0, 0.0, 0.02), cfg: dict | None = None):
        c = dict(rate=30.0, jitter=0.003, frame_drop=0.08, px=1.0, outlier_p=0.01, outlier_px=15.0, kp_drop=0.05,
                 occ_p=0.5, depth_rel=0.003, depth_abs=0.001, calib_f=0.005, calib_rot=math.radians(0.2),
                 calib_t=0.001)
        c.update(cfg or {})
        self.cfg = c
        dist = rng.uniform(0.8, 1.4)
        az = math.radians(rng.uniform(-40, 40))
        el = math.radians(rng.uniform(15, 45))
        ctr = np.asarray(center, float)
        # camera in front of the chain (looking along +y), rotated by azimuth about z, raised by elevation
        self.eye = ctr + dist * np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
        self.R = look_at(self.eye, ctr)                    # TRUE extrinsics
        self.f = F_PX
        # actor calibration (what the actor believes)
        self.f_hat = F_PX * (1 + rng.normal(0, c["calib_f"]))
        dR = _rot(*rng.normal(0, c["calib_rot"], 3))
        self.R_hat = dR @ self.R
        self.eye_hat = self.eye + rng.normal(0, c["calib_t"], 3)
        self.occ = None
        if rng.random() < c["occ_p"]:
            w = rng.uniform(40, 80)
            u0 = rng.uniform(W_PX * 0.3, W_PX * 0.7 - w)
            self.occ = (u0, u0 + w)

    def frame_times(self, t0, t1, rng):
        c = self.cfg
        tn = np.arange(t0, t1, 1.0 / c["rate"])
        t = tn + rng.normal(0, c["jitter"], len(tn))
        keep = rng.random(len(t)) >= c["frame_drop"]
        return np.sort(t[keep])

    def project(self, P):
        """TRUE projection (evaluator side). P (...,3) world -> (u, v, depth)."""
        pc = (np.asarray(P, float) - self.eye) @ self.R.T
        z = pc[..., 2]
        return self.f * pc[..., 0] / z + W_PX / 2, self.f * pc[..., 1] / z + H_PX / 2, z

    def observe(self, P, rng):
        """P (T, N, 3) true keypoints at the frame times -> detections (u, v, d) with NaN where not detected."""
        c = self.cfg
        u, v, z = self.project(P)
        T, N = u.shape
        u = u + rng.normal(0, c["px"], (T, N))
        v = v + rng.normal(0, c["px"], (T, N))
        out = rng.random((T, N)) < c["outlier_p"]
        u = np.where(out, u + rng.normal(0, c["outlier_px"], (T, N)), u)
        v = np.where(out, v + rng.normal(0, c["outlier_px"], (T, N)), v)
        d = z * (1 + rng.normal(0, c["depth_rel"], (T, N))) + rng.normal(0, c["depth_abs"], (T, N))
        miss = rng.random((T, N)) < c["kp_drop"]
        miss |= (u < 0) | (u >= W_PX) | (v < 0) | (v >= H_PX)
        if self.occ is not None:
            miss |= (u >= self.occ[0]) & (u <= self.occ[1])
        u, v, d = (np.where(miss, np.nan, a) for a in (u, v, d))
        return u, v, d

    def backproject(self, u, v, d):
        """ACTOR side: detections -> world xyz with the actor's calibration (NaN stays NaN)."""
        x = (u - W_PX / 2) / self.f_hat * d
        y = (v - H_PX / 2) / self.f_hat * d
        pc = np.stack([x, y, d], -1)
        return pc @ self.R_hat + self.eye_hat
