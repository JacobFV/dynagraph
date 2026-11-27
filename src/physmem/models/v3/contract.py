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
