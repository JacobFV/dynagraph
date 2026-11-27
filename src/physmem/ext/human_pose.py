"""Human hand/body pose front end -> TrackSet(kind="human") (owner: agent EXT). docs/EXT.md §5.

Estimator: Google MediaPipe Hand Landmarker (21 keypoints per hand, image + metric "world" coordinates) and Pose
Landmarker (33 body keypoints). Code Apache-2.0 (github.com/google-ai-edge/mediapipe); model files Apache-2.0 per
their model cards; no MANO/SMPL dependency (MANO-based estimators such as HaMeR/WiLoR inherit MANO's
non-commercial licence and need a PI registration -- see docs/EXT.md §1).

Model files are fetched by scripts/ext/fetch.py (group mediapipe_models) into $EXT_DATA_ROOT/mediapipe_models/raw and
their sha256 is part of every TrackSet.source string.

Optional depth lifting: given an aligned metric depth image and intrinsics (RGB-D source), each 2-D keypoint is
lifted to camera-frame xyz using the median depth in a small window (robust to edge pixels). This is what a real
RGB-D rig would do; nothing from mocap/annotation is used.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .tracks import KIND_HUMAN, MP_HAND, TrackSet

MODELS = Path(os.environ.get("EXT_DATA_ROOT", "/home/brandonin/datasets")) / "mediapipe_models" / "raw"
HAND_TASK = MODELS / "hand_landmarker.task"
POSE_TASK = MODELS / "pose_landmarker_full.task"
VERSION = "mediapipe-hand-v1"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


class HandTracker:
    """VIDEO-mode MediaPipe hand landmarker. Entities: "left_hand", "right_hand" (MediaPipe handedness, which
    assumes a mirrored selfie image; we report it as given and also keep the per-frame handedness score)."""

    def __init__(self, num_hands=2, min_det=0.3, min_track=0.3, mode="video"):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as V
        rm = V.RunningMode.VIDEO if mode == "video" else V.RunningMode.IMAGE
        opts = V.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(HAND_TASK)), running_mode=rm,
                                       num_hands=num_hands, min_hand_detection_confidence=min_det,
                                       min_hand_presence_confidence=min_det, min_tracking_confidence=min_track)
        self.lm = V.HandLandmarker.create_from_options(opts)
        self.mp = mp
        self.mode = mode
        self.tag = f"{VERSION} model_sha256={_sha(HAND_TASK)}"

    def close(self):
        self.lm.close()

    def __call__(self, rgb: np.ndarray, fps=30.0, source="", depth=None, K=None, depth_scale=1.0) -> TrackSet:
        """rgb (T,H,W,3) uint8; depth (T,H,W) raw depth (metres = raw * depth_scale) aligned to rgb, optional."""
        T, H, W, _ = rgb.shape
        ents = ["left_hand", "right_hand"]
        uv = np.full((T, 2, 21, 2), np.nan, np.float32)
        score = np.full((T, 2, 21), np.nan, np.float32)
        world = np.full((T, 2, 21, 3), np.nan, np.float32)
        for t in range(T):
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb[t]))
            r = (self.lm.detect_for_video(img, int(round(t * 1000 / fps))) if self.mode == "video"
                 else self.lm.detect(img))
            for lms, hd, wl in zip(r.hand_landmarks, r.handedness, r.hand_world_landmarks):
                e = 0 if hd[0].category_name.lower() == "left" else 1
                if np.isfinite(score[t, e, 0]) and score[t, e, 0] >= hd[0].score:
                    continue
                uv[t, e] = [(p.x * W, p.y * H) for p in lms]
                world[t, e] = [(p.x, p.y, p.z) for p in wl]
                score[t, e] = hd[0].score
        xyz = lift_depth(uv, depth, K, depth_scale) if depth is not None and K is not None else None
        return TrackSet(kind=KIND_HUMAN, t=np.arange(T) / fps, uv=uv, entities=ents, keypoints=list(MP_HAND),
                        image_hw=(H, W), source=f"{source} | {self.tag}", xyz=xyz, score=score, K_cam=K,
                        extra=dict(mp_world=world))


def lift_depth(uv, depth, K, depth_scale=1.0, win=2):
    """uv (T,E,J,2) px, depth (T,H,W) -> camera-frame xyz (T,E,J,3) m; NaN where depth is missing."""
    T, E, J, _ = uv.shape
    H, W = depth.shape[1:]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out = np.full((T, E, J, 3), np.nan, np.float32)
    for t in range(T):
        for e in range(E):
            for j in range(J):
                u, v = uv[t, e, j]
                if not np.isfinite(u):
                    continue
                iu, iv = int(round(u)), int(round(v))
                if not (0 <= iu < W and 0 <= iv < H):
                    continue
                patch = depth[t, max(iv - win, 0): iv + win + 1, max(iu - win, 0): iu + win + 1].astype(np.float64)
                patch = patch[patch > 0]
                if patch.size == 0:
                    continue
                z = float(np.median(patch)) * depth_scale
                out[t, e, j] = ((u - cx) / fx * z, (v - cy) / fy * z, z)
    return out
