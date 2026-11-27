"""EXT observation records for external third-person data (owner: agent EXT). Pure numpy.

Two modalities, both ACTOR-LEGAL (they come from an off-the-shelf detector/tracker or pose estimator run on pixels,
never from simulator/mocap state):

  * TrackSet with kind="object": per-entity 2-D image tracks (+ optional depth / 3-D) of objects;
  * TrackSet with kind="human" : per-keypoint tracks of a human hand/body (each hand or body is one entity whose
    keypoints follow a fixed name list, e.g. the 21 MediaPipe hand landmarks).

Layout mirrors the V3 tracker contract on branch v3-model (src/physmem/models/v3/tracks.py: detections (u, v, d)
with NaN where not detected, frame times with drops). `to_v3_evidence` produces the V3 MOD_TRACK evidence channels
(x, v, f, contact, f_cmd, valid) per entity; it will be re-pointed at the final V3 contract when it is committed.

Evaluator-only labels (ground-truth masses, OCP outcome, mocap poses) live in a separate `labels` dict that the
loaders return next to the TrackSet and that must never be fed to a forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KIND_OBJECT, KIND_HUMAN = "object", "human"

MP_HAND = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip",
           "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip", "ring_dip",
           "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]          # MediaPipe / DexYCB order


@dataclass
class TrackSet:
    """One clip of third-person observations.

    t      : (T,) frame times in seconds (monotone; dropped frames are simply absent)
    uv     : (T, E, K, 2) image coordinates in pixels (NaN = not detected)
    depth  : (T, E, K) metric depth along the optical axis or None (RGB-only sources)
    xyz    : (T, E, K, 3) actor-estimated 3-D position in the camera (or world) frame, or None
    score  : (T, E, K) detector confidence in [0, 1] (NaN where undetected)
    entities: names of the E entities (e.g. "agent", "patient", "obj3", "right_hand")
    keypoints: names of the K keypoints per entity (objects: ["centroid", "bbox_min", "bbox_max"] etc.)
    image_hw: (H, W) pixel size of the source frames
    K_cam  : (3, 3) intrinsics if known (actor-side estimate), else None
    source : provenance string: dataset/file + front end name/version + weights sha256
    """
    kind: str
    t: np.ndarray
    uv: np.ndarray
    entities: list
    keypoints: list
    image_hw: tuple
    source: str
    depth: np.ndarray | None = None
    xyz: np.ndarray | None = None
    score: np.ndarray | None = None
    K_cam: np.ndarray | None = None
    extra: dict = field(default_factory=dict)      # actor-legal per-entity extras (area, bbox) -- never labels

    def __post_init__(self):
        T, E, K, _ = self.uv.shape
        assert self.t.shape == (T,) and len(self.entities) == E and len(self.keypoints) == K, "TrackSet shape"
        assert self.kind in (KIND_OBJECT, KIND_HUMAN)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.uv).all(-1)

    def entity(self, name):
        return self.entities.index(name)

    def save(self, path):
        d = dict(kind=self.kind, t=self.t, uv=self.uv, entities=np.array(self.entities), keypoints=np.array(self.keypoints),
                 image_hw=np.array(self.image_hw), source=np.array(self.source))
        for k in ("depth", "xyz", "score", "K_cam"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        for k, v in self.extra.items():
            d["extra__" + k] = v
        np.savez_compressed(path, **d)

    @staticmethod
    def load(path) -> "TrackSet":
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in ("depth", "xyz", "score", "K_cam") if k in z}
        extra = {k[7:]: z[k] for k in z.files if k.startswith("extra__")}
        return TrackSet(kind=str(z["kind"]), t=z["t"], uv=z["uv"], entities=[str(x) for x in z["entities"]],
                        keypoints=[str(x) for x in z["keypoints"]], image_hw=tuple(int(x) for x in z["image_hw"]),
                        source=str(z["source"]), extra=extra, **kw)


def resample(ts: TrackSet, t_new: np.ndarray, max_gap: float = 0.15) -> np.ndarray:
    """linear interpolation of uv (or xyz if present) onto t_new; NaN where the nearest valid samples are further
    than max_gap seconds apart. Returns (len(t_new), E, K, D)."""
    X = ts.xyz if ts.xyz is not None else ts.uv
    T, E, K, D = X.shape
    out = np.full((len(t_new), E, K, D), np.nan, np.float32)
    for e in range(E):
        for k in range(K):
            ok = np.isfinite(X[:, e, k]).all(-1)
            if ok.sum() < 2:
                continue
            tv = ts.t[ok]
            idx = np.clip(np.searchsorted(tv, t_new), 1, len(tv) - 1)
            gap = tv[idx] - tv[idx - 1]
            inside = (t_new >= tv[0]) & (t_new <= tv[-1]) & (gap <= max_gap)
            for d in range(D):
                out[inside, e, k, d] = np.interp(t_new[inside], tv, X[ok, e, k, d])
    return out


V3_EV_C = 6   # x, v, f, contact, f_cmd, valid  (v3-model chain.py MOD_TRACK layout, 1-D)


def to_v3_evidence(ts: TrackSet, t_grid: np.ndarray, keypoint: int = 0, scale: float = 1.0) -> np.ndarray:
    """per-entity evidence series (E, len(t_grid), D, V3_EV_C) for the V3 MOD_TRACK modality, one block per spatial
    axis D (V3 chain is 1-D; multi-axis sources give D blocks). Channels: position (scaled), finite-difference
    velocity, force = 0 (third-person data has no force), contact = 0 (unknown), commanded force = 0, valid flag."""
    X = resample(ts, t_grid)[:, :, keypoint] * scale         # (Tg, E, D)
    Tg, E, D = X.shape
    ev = np.zeros((E, Tg, D, V3_EV_C), np.float32)
    ok = np.isfinite(X)
    Xf = np.nan_to_num(X)
    V = np.gradient(Xf, t_grid, axis=0) if Tg > 1 else np.zeros_like(Xf)
    okv = ok & np.roll(ok, 1, 0) & np.roll(ok, -1, 0)
    ev[..., 0] = Xf.transpose(1, 0, 2)
    ev[..., 1] = np.where(okv, V, 0).transpose(1, 0, 2)
    ev[..., 5] = ok.transpose(1, 0, 2)
    return ev
