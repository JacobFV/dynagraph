"""V3 unified data contract (docs/V3_GENERAL.md section 1; owner: V3-MODEL). Pure python/numpy.

One contract for every domain (sprint worlds, v2 synthetic, v2 stations, the passive third-person domain, the
contact chain, later human pose). It has three parts.

1. EVIDENCE TOKENS: one token per (entity, modality, time window). The modality is explicit (`MODALITIES`), so
   force is one channel among several, and a missing modality is a masked token, never a zero signal.
2. ENTITY / RELATION TOKENS: domain-agnostic entity types plus opaque ids. Identity comes from the relation graph
   (contact, spring, joint, support, grasp, carried-by), never from list position.
3. TYPED BELIEF STORE: key = (entity, property class), value = belief (mean + log-variance in a normalised space,
   or a logit), with provenance. It is a superset of the v2 latents (`spec.LATENTS`), adds elastic k/m, c/m, mass
   ratio, restitution and absolute scale, and adds state classes (pose, velocity, contact mode/force on the physical
   time grid). Each class carries an identifiability tag: what passive observation can recover, and what needs a
   known force or an anchor.
"""
from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------------------- 1. modalities
MODALITIES = {
    "proprio_force": "own effector: encoders, commanded reference, wrist F/T, gripper, tactile/contact flags",
    "keypoint_track": "third-person keypoint/centroid tracks from a camera tracker (noisy, dropout, occlusion, jitter)",
    "pose_anchor": "sparse golden pose fixes (motion capture / digital twin / fiducials)",
    "vision_material": "scalar visual material / stiffness readings",
    "human_pose": "body / hand keypoints of other agents (pose front end; EXT track)",
    "map_anchor": "static known fixtures (walls, spring mounts, goal regions)",
}
MOD_NAMES = list(MODALITIES)

# ----------------------------------------------------------------------------- 2. entities and relations
ENTITY_TYPES = ["effector", "object", "fixture", "agent", "link", "particle", "goal_region", "human_part"]
RELATIONS = ["contact", "spring", "joint", "support", "grasp", "carried_by", "observes"]

# ----------------------------------------------------------------------------- 3. property classes
# (name, kind, transform, loc, scale, unit, identifiability)
#   kind: param (time-invariant) | state (on the physical-time grid) | relation (per edge)
#   identifiability: 'passive' = recoverable from third-person tracks alone; 'ratio' = only as a ratio passively;
#                    'force' = needs a known-force interaction (own F/T or a known poke); 'anchor' = needs an anchor
PROPERTIES = [
    # v2 latents (spec.LATENTS), unchanged transforms
    ("mass", "param", "log", math.log(0.8), 1.0, "kg", "force"),
    ("friction", "param", "lin", 0.5, 0.25, "1", "passive"),
    ("grip_friction", "param", "lin", 0.6, 0.25, "1", "force"),
    ("stiffness", "param", "log", math.log(3000.0), 1.2, "N/m", "force"),
    ("clearance", "param", "log", math.log(6e-4), 0.8, "m", "force"),
    ("offset_x", "param", "lin", 0.0, 0.003, "m", "force"),
    ("offset_y", "param", "lin", 0.0, 0.003, "m", "force"),
    ("damping", "param", "log", math.log(15.0), 1.0, "N*s/m", "force"),
    ("breakaway", "param", "log", math.log(8.0), 0.7, "N", "force"),
    ("bulk_friction", "param", "lin", 0.5, 0.2, "1", "passive"),
    ("crush", "param", "log", math.log(40.0), 0.8, "N", "force"),
    # V3 additions
    ("spring_k", "param", "log", math.log(47.4), 0.7, "N/m", "force"),        # elastic stiffness (absolute)
    ("k_over_m", "param", "log", math.log(100.0), 1.0, "1/s^2", "passive"),   # from oscillation period
    ("c_over_m", "param", "log", math.log(1.0), 1.0, "1/s", "passive"),       # from decay
    ("mass_ratio", "param", "log", 0.0, 0.7, "1", "passive"),                 # m_entity / m_partner (collisions)
    ("restitution", "param", "lin", 0.5, 0.25, "1", "passive"),               # bounce / collision
    ("length_scale", "param", "log", 0.0, 0.1, "1", "anchor"),                # metric scale of a monocular track
    # states on the physical-time grid (beliefs about the queried future and the current state)
    ("pose", "state", "lin", 0.0, 0.05, "m", "passive"),
    ("velocity", "state", "lin", 0.0, 0.5, "m/s", "passive"),
    # relations
    ("contact_mode", "relation", "logit", 0.0, 1.0, "1", "passive"),
    ("contact_force", "relation", "lin", 0.0, 5.0, "N", "force"),
]
PROP_NAMES = [p[0] for p in PROPERTIES]
PROP_INDEX = {n: i for i, n in enumerate(PROP_NAMES)}
V2_LATENTS = PROP_NAMES[:11]          # identical order to spec.LATENTS (checked in tests)

# provenance of a belief value (known input of every entity token)
PROVENANCE = ["anchor", "tracker", "known", "stale", "model", "prior"]


def to_norm(name, x):
    _, _, tr, loc, sc, _, _ = PROPERTIES[PROP_INDEX[name]]
    x = np.asarray(x, np.float64)
    return ((np.log(np.maximum(x, 1e-12)) if tr == "log" else x) - loc) / sc


def from_norm(name, z):
    _, _, tr, loc, sc, _, _ = PROPERTIES[PROP_INDEX[name]]
    v = np.asarray(z, np.float64) * sc + loc
    return np.exp(v) if tr == "log" else v


def derived_consistency(beliefs: dict) -> dict:
    """the typed schema's structural identities (used as law checks across domains): log k = log(k/m) + log m."""
    out = {}
    if all(k in beliefs for k in ("spring_k", "k_over_m", "mass")):
        out["k_identity"] = float(np.log(beliefs["spring_k"]) - np.log(beliefs["k_over_m"]) - np.log(beliefs["mass"]))
    return out


# chain family (section 0) -> contract mapping (the chain store uses this subset)
CHAIN_PROPERTIES = {"log_mass": "mass", "friction": "friction", "log_k": "spring_k", "x": "pose", "v": "velocity",
                    "c": "contact_mode", "f": "contact_force"}
CHAIN_MODALITIES = {0: "keypoint_track", 1: "proprio_force", 2: "pose_anchor"}


# ----------------------------------------------------------------------------- human modality (EXT H6 interface)
# EXT (branch `ext`, src/physmem/ext/tracks.py) emits TrackSet(kind="human"): t (T,), uv (T,E,K,2), xyz (T,E,K,3)
# depth-lifted camera-frame keypoints (NaN = missed), entities e.g. ["right_hand"], keypoints = 21 MediaPipe hand
# joints (MANO/DexYCB order). We read it by duck typing (no import of the ext package, no edit of that branch).
# Mapping into this contract:
#   * each hand  -> entity type "human_part" of an embodiment WITHOUT force sensing (EMBODIMENT_HUMAN below): no
#                   proprio_force token can exist for it; its evidence is `human_pose` tokens only;
#   * the grasped object -> entity type "object" with NO direct track in the confirmatory H6 test; it is linked to
#                   the hand by a "grasp" relation, and its typed `mass` slot is what H6 reads out;
#   * evidence channels per hand and window (HUMAN_CH): wrist xyz, wrist speed, thumb-index aperture, palm-normal
#                   vertical component, valid. Gravity direction comes from a table-plane fit (EXT h6-feat-v1),
#                   supplied as a known input, never from labels.
# Matched twin for H6: LBS-generic = the same backbone with the named property fields replaced by an untyped
# vector of equal width and no property supervision (planned; not yet trained).
EMBODIMENT_HUMAN = dict(kind="human", force_sensing=False, proprio=False, modalities=("human_pose",))
HAND_JOINTS = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip",
               "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip",
               "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
HUMAN_CH = ["wrist_x", "wrist_y", "wrist_z", "wrist_speed", "aperture", "palm_up", "valid"]


def human_evidence(ts, up=(0.0, -1.0, 0.0), n_bins=32, t0=None, span=None):
    """TrackSet(kind="human") -> (E, n_bins, len(HUMAN_CH)) binned evidence + (E,) availability. `up` is the
    actor-estimated gravity-up direction in the TrackSet frame (table-plane normal). Positions relative to the
    first valid wrist sample of each entity."""
    assert ts.kind == "human" and ts.xyz is not None
    t = np.asarray(ts.t, float)
    t0 = t[0] if t0 is None else t0
    span = (t[-1] - t0 + 1e-6) if span is None else span
    kp = {n: i for i, n in enumerate(ts.keypoints)}
    up = np.asarray(up, float) / np.linalg.norm(up)
    E = ts.xyz.shape[1]
    out = np.zeros((E, n_bins, len(HUMAN_CH)), np.float32)
    avail = np.zeros(E, bool)
    b = np.clip(((t - t0) / span * n_bins).astype(int), 0, n_bins - 1)
    for e in range(E):
        X = ts.xyz[:, e]
        w = X[:, kp["wrist"]]
        ok = np.isfinite(w).all(1)
        if ok.sum() < 2:
            continue
        avail[e] = True
        w_rel = w - w[np.nonzero(ok)[0][0]]
        sp = np.r_[0.0, np.linalg.norm(np.diff(w, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)]
        ap = np.linalg.norm(X[:, kp["thumb_tip"]] - X[:, kp["index_tip"]], axis=1)
        n = np.cross(X[:, kp["index_mcp"]] - w, X[:, kp["pinky_mcp"]] - w)
        pu = (n @ up) / np.maximum(np.linalg.norm(n, axis=1), 1e-9)
        feats = np.c_[w_rel, sp, ap, pu]
        good = ok & np.isfinite(feats).all(1)
        for c in range(feats.shape[1]):
            s_ = np.bincount(b[good], feats[good, c], n_bins); n_ = np.bincount(b[good], None, n_bins)
            out[e, :, c] = np.where(n_ > 0, s_ / np.maximum(n_, 1), 0.0)
        out[e, :, -1] = np.bincount(b[good], None, n_bins) > 0
    return out, avail
