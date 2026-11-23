"""v2 model data contract (owner: agent MODEL-V2). See docs/MODEL_V2.md.

Everything an environment must log for the v2 world model, and the fixed tensor layouts the model consumes.
Pure numpy (no torch) so environments can import it cheaply.

Three record types per episode:
  * Interaction  : one per executed SkillCall -> dense time series segment (actor-legal), skill call, effort, outcome
  * entity state : ObservationEvents (schema.py) or an EntityFrame snapshot at each decision time
  * labels       : evaluator-only latent truths (LATENTS vocabulary) + outcome targets; NEVER reach a forward pass
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

V2_MODEL_SCHEMA = "physmem-v2-model-1"
G = 9.81

# ----------------------------------------------------------------- latent vocabulary (typed slots)
# each latent lives in a NORMALIZED space z = (f(x) - loc) / scale ; f = log for positive scale-like quantities.
# (name, transform, loc, scale, unit)
LATENTS = [
    ("mass", "log", np.log(0.8), 1.0, "kg"),              # object mass
    ("friction", "lin", 0.5, 0.25, "1"),                  # object<->support sliding friction
    ("grip_friction", "lin", 0.6, 0.25, "1"),             # object<->finger friction
    ("stiffness", "log", np.log(3000.0), 1.2, "N/m"),     # contact / object compliance (inverse)
    ("clearance", "log", np.log(6e-4), 0.8, "m"),         # peg-hole radial clearance
    ("offset_x", "lin", 0.0, 0.003, "m"),                 # hole offset from perceived pose (x)
    ("offset_y", "lin", 0.0, 0.003, "m"),                 # hole offset from perceived pose (y)
    ("damping", "log", np.log(15.0), 1.0, "N*s/m"),       # articulation (drawer/door) viscous damping
    ("breakaway", "log", np.log(8.0), 0.7, "N"),          # articulation static breakaway force
    ("bulk_friction", "lin", 0.5, 0.2, "1"),              # granular / bulk-pile effective friction
]
LATENT_NAMES = [l[0] for l in LATENTS]
N_LATENT = len(LATENTS)
LATENT_INDEX = {n: i for i, n in enumerate(LATENT_NAMES)}


def latent_to_z(name: str, x):
    _, tr, loc, sc, _ = LATENTS[LATENT_INDEX[name]]
    x = np.asarray(x, np.float64)
    return ((np.log(np.maximum(x, 1e-9)) if tr == "log" else x) - loc) / sc


def z_to_latent(name: str, z):
    _, tr, loc, sc, _ = LATENTS[LATENT_INDEX[name]]
    v = np.asarray(z, np.float64) * sc + loc
    return np.exp(v) if tr == "log" else v


def labels_to_z(labels: dict):
    """evaluator label dict {latent_name: SI value} -> (z (L,), mask (L,))."""
    z = np.zeros(N_LATENT, np.float32)
    m = np.zeros(N_LATENT, bool)
    for n, v in labels.items():
        if n in LATENT_INDEX and v is not None and np.isfinite(v):
            z[LATENT_INDEX[n]] = latent_to_z(n, v)
            m[LATENT_INDEX[n]] = True
    return z, m


# ----------------------------------------------------------------- segment channels (dense time series)
# all positions are relative to the TARGET entity's actor-estimated position at segment start (translation only,
# world-aligned axes, z up); forces in world axes, gravity-compensated for the hand/tool (i.e. ~0 in free motion).
CHANNELS = [
    "cmd_x", "cmd_y", "cmd_z",            # 0-2  commanded ee position (rel)
    "cmd_vx", "cmd_vy", "cmd_vz",         # 3-5  commanded ee velocity
    "ee_x", "ee_y", "ee_z",               # 6-8  measured ee position (FK of encoders, rel)
    "ee_vx", "ee_vy", "ee_vz",            # 9-11 measured ee velocity
    "fx", "fy", "fz",                     # 12-14 wrist F/T force (N, world axes, tool-gravity compensated)
    "tx", "ty", "tz",                     # 15-17 wrist torque (N m)
    "grip_cmd",                           # 18 gripper command 0 open..1 closed (suction: on/off)
    "grip_pos",                           # 19 measured gripper opening (m) or suction pressure (normalized)
    "obj_x", "obj_y", "obj_z",            # 20-22 target pose observation (rel to its start; noisy; held when invalid)
    "obj_valid",                          # 23 1 if a fresh target observation arrived at this sample
    "contact",                            # 24 actor-legal contact/tactile flag (finger contact sensor / F/T threshold)
    "t",                                  # 25 time since segment start (s)
]
N_CH = len(CHANNELS)
CH = {n: i for i, n in enumerate(CHANNELS)}
# fixed per-channel normalisation scales (divide); documented constants, not fit -> identical across envs
CH_SCALE = np.array([0.1] * 3 + [0.5] * 3 + [0.1] * 3 + [0.5] * 3 + [20.0] * 3 + [2.0] * 3 + [1.0, 0.05]
                    + [0.1] * 3 + [1.0, 1.0, 2.0], np.float32)
SEG_T = 64          # resampled samples per segment
SEG_RATE_HINT = 50  # Hz suggested raw logging rate (>= 50 Hz, any rate accepted; we resample)

# ----------------------------------------------------------------- skills
SKILLS = ["reach", "push", "sweep", "pick", "place", "insert", "open", "close", "handover", "regrasp"]
CALL_DIM = 24       # api.SkillCall.vector()

# ----------------------------------------------------------------- outcome layout
# continuous outcome targets (masked per family; heteroscedastic)
OUT_CONT = [
    "dx", "dy", "dz",          # target displacement over the skill (m, world axes)
    "progress",                # family progress scalar after the step in [0,1]-ish (fraction moved, opening/goal, depth)
    "peak_force",              # peak |F| at wrist during the skill (N/50)
    "duration",                # skill duration (s)
    "aux0", "aux1",            # family-specific (e.g. insertion lateral error, drawer final speed)
]
N_CONT = len(OUT_CONT)
# binary events (logits)
OUT_EVENTS = ["success", "slip", "drop", "jam", "collision", "stall"]
N_EVENTS = len(OUT_EVENTS)
RISK_EVENTS = ("drop", "jam", "collision", "stall")   # summed into "risk" probability for the planner

# ----------------------------------------------------------------- embodiment token
KINDS = ["arm", "bimanual", "humanoid", "mobile_manipulator"]
GRIPPERS = ["parallel", "suction", "dexterous", "flat_palm"]
EMB_DIM = 4 + 4 + 7


def embodiment_vector(spec) -> np.ndarray:
    """api.EmbodimentSpec (or any object/dict with the same fields) -> (EMB_DIM,) float32."""
    g = (lambda k: spec[k]) if isinstance(spec, dict) else (lambda k: getattr(spec, k))
    v = np.zeros(EMB_DIM, np.float32)
    v[KINDS.index(g("kind")) if g("kind") in KINDS else 0] = 1
    v[4 + (GRIPPERS.index(g("gripper")) if g("gripper") in GRIPPERS else 0)] = 1
    v[8:] = [g("reach_m"), np.log(max(g("payload_kg"), 1e-3)), g("max_ee_speed"), np.log(max(g("max_ee_force"), 1e-3)),
             g("dof") / 10.0, float(g("floating_base")), len(g("effectors"))]
    return v


# ----------------------------------------------------------------- entity tokens
ENT_TYPES = ["object", "effector", "fixture", "agent", "link", "particle", "goal_region"]
ENT_DIM = len(ENT_TYPES) + 3 + 4 + 3 + 3 + 3  # type one-hot, pos(rel target), quat, size(half), vel, (is_target, age, valid)


@dataclass
class EntityFrame:
    """Snapshot of actor-legal entity state at decision time (alternative to raw ObservationEvents).
    pos/vel relative to the target's estimated position. Unknown -> zeros + valid=0."""
    types: np.ndarray           # (N,) int index into ENT_TYPES
    pos: np.ndarray             # (N,3)
    quat: np.ndarray            # (N,4) wxyz
    size: np.ndarray            # (N,3) half extents
    vel: np.ndarray             # (N,3)
    age: np.ndarray             # (N,) seconds since last observation
    valid: np.ndarray           # (N,) bool
    target: int = 0             # index of the query target
    edges: Optional[np.ndarray] = None   # (Ed,3) int: src, dst, rel (schema.Rel)

    def tokens(self) -> np.ndarray:
        N = len(self.types)
        t = np.zeros((N, ENT_DIM), np.float32)
        t[np.arange(N), self.types] = 1
        o = len(ENT_TYPES)
        t[:, o:o + 3] = self.pos / 0.1
        t[:, o + 3:o + 7] = self.quat
        t[:, o + 7:o + 10] = self.size / 0.05
        t[:, o + 10:o + 13] = self.vel / 0.5
        t[self.target, o + 13] = 1
        t[:, o + 14] = np.log1p(self.age)
        t[:, o + 15] = self.valid
        return t


# ----------------------------------------------------------------- recorder (envs call this every control tick)
class SegmentRecorder:
    """Accumulate one skill execution's actor-legal signals; .finish() -> segment dict for SkillResult.log['segment'].

    rec = SegmentRecorder(target_pos0)       # target's ACTOR-ESTIMATED position at skill start (world, m)
    each control tick (>= 50 Hz ideally):
      rec.tick(t, cmd_pos, cmd_vel, ee_pos, ee_vel, wrench6, grip_cmd, grip_pos, obj_pos_or_None, contact)
    seg = rec.finish()                        # {'x': (T_raw, N_CH) float32, 't0', 't1', 'schema'}
    """

    def __init__(self, target_pos0):
        self.p0 = np.asarray(target_pos0, np.float64)[:3]
        self.rows = []
        self.t0 = None
        self.last_obj = np.zeros(3)

    def tick(self, t, cmd_pos, cmd_vel, ee_pos, ee_vel, wrench, grip_cmd=0.0, grip_pos=0.0, obj_pos=None, contact=0.0):
        if self.t0 is None:
            self.t0 = float(t)
        valid = obj_pos is not None
        if valid:
            self.last_obj = np.asarray(obj_pos, np.float64)[:3] - self.p0
        r = np.concatenate([np.asarray(cmd_pos)[:3] - self.p0, np.asarray(cmd_vel)[:3], np.asarray(ee_pos)[:3] - self.p0,
                            np.asarray(ee_vel)[:3], np.asarray(wrench)[:6], [grip_cmd, grip_pos], self.last_obj,
                            [float(valid), float(contact), float(t) - self.t0]])
        self.rows.append(r.astype(np.float32))

    def finish(self) -> dict:
        x = np.stack(self.rows) if self.rows else np.zeros((1, N_CH), np.float32)
        return dict(x=x, t0=self.t0 or 0.0, t1=(self.t0 or 0.0) + float(x[-1, CH["t"]]), schema=V2_MODEL_SCHEMA)


def resample_segment(x: np.ndarray, T: int = SEG_T) -> np.ndarray:
    """(T_raw, N_CH) irregular-length segment -> (T, N_CH) by linear interpolation on the 't' channel.
    obj_valid / contact are max-pooled per bin (so sparse observations are not lost)."""
    x = np.asarray(x, np.float32)
    if len(x) == 1:
        return np.repeat(x, T, 0)
    t = x[:, CH["t"]]
    tq = np.linspace(t[0], t[-1], T)
    out = np.empty((T, x.shape[1]), np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(tq, t, x[:, c])
    for c in (CH["obj_valid"], CH["contact"]):
        idx = np.clip(np.searchsorted(tq, t), 0, T - 1)
        b = np.zeros(T, np.float32)
        np.maximum.at(b, idx, x[:, c])
        out[:, c] = b
    return out


# ----------------------------------------------------------------- analytic (actor-legal) per-segment features
AN_NAMES = ["log_mean_fpar", "log_fz_static", "ls_mass", "ls_bias", "slide_mu", "log_f_over_v", "flat_x", "flat_y",
            "log_peak_f", "grip_slip", "obj_moved", "log_duration", "release_v", "log_slide"]
N_AN = len(AN_NAMES)


def analytic_features(x: np.ndarray) -> np.ndarray:
    """Physically-motivated summary features from ONE resampled segment (T, N_CH) of actor-legal channels.
      - mean force along the motion direction while in contact (~ mu m g + m a for pushes)
      - static vertical load during hold (~ m g for lifts)
      - least-squares fit  F_motion = m_hat * a_ee + c_hat  over contact samples (inertia vs. friction)
      - post-release free-slide friction: mu ~= v_rel^2 / (2 g d_slide)  from target observations
      - force/velocity ratio (viscous damping, articulated)
      - mean lateral force (insertion misalignment direction), peak force, gripper slip, moved distance
    """
    f = np.zeros(N_AN, np.float32)
    t = x[:, CH["t"]]
    dur = max(float(t[-1] - t[0]), 1e-3)
    dt = dur / max(len(t) - 1, 1)
    F = x[:, CH["fx"]:CH["fx"] + 3].astype(np.float64)
    V = x[:, CH["ee_vx"]:CH["ee_vx"] + 3].astype(np.float64)
    con = x[:, CH["contact"]] > 0.5
    vh = V[:, :2]
    sp = np.linalg.norm(vh, axis=1)
    d = vh[np.argmax(sp)] / (sp.max() + 1e-9) if sp.max() > 1e-4 else np.array([1.0, 0.0])
    fpar = -(F[:, :2] @ d)      # reaction force opposing motion (positive when pushing against a load)
    if con.sum() >= 3:
        f[0] = np.log1p(max(np.mean(fpar[con]), 0.0))
        a = np.gradient(V[:, :2] @ d, dt)
        A = np.stack([a[con], np.ones(con.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, fpar[con], rcond=None)
        f[2], f[3] = np.clip(sol[0], -50, 50) / 5.0, np.clip(sol[1], -200, 200) / 20.0
        vm = np.abs(V[con] @ np.r_[d, 0.0]) + np.abs(V[con, 2])
        f[5] = np.log1p(np.mean(np.linalg.norm(F[con], axis=1)) / (np.mean(vm) + 0.02))
        lat = F[con, :2].mean(0)
        f[6], f[7] = lat[0] / 10.0, lat[1] / 10.0
    hold = np.abs(V[:, 2]) < 0.02
    if hold.sum() >= 3:
        f[1] = np.log1p(max(-np.median(F[hold, 2]), 0.0))
    f[8] = np.log1p(np.linalg.norm(F, axis=1).max())
    gp = x[:, CH["grip_pos"]]
    f[9] = float(np.clip(np.abs(np.diff(gp)).sum() * 20.0, 0, 5))
    obj = x[:, CH["obj_x"]:CH["obj_x"] + 3]
    f[10] = np.linalg.norm(obj[-1] - obj[0]) / 0.1
    f[11] = np.log(dur)
    # free slide after contact ends: release speed from ee speed at last contact; slide = obj travel after release
    if con.any() and (~con[np.argmax(con):]).any():
        last = len(con) - 1 - np.argmax(con[::-1])
        vrel = float(sp[last])
        dsl = float(np.linalg.norm(obj[-1, :2] - obj[last, :2]))
        f[12] = vrel / 0.5
        f[13] = np.log(dsl + 1e-3)
        if dsl > 2e-3 and vrel > 0.03:
            f[4] = np.clip(vrel ** 2 / (2 * G * dsl), 0, 3)
    return f


@dataclass
class Interaction:
    """One executed skill (what ENVS logs per SkillCall; see docs/MODEL_V2.md)."""
    episode_id: str
    step: int
    family: str
    robot: str
    call_vec: np.ndarray            # (CALL_DIM,) api.SkillCall.vector()
    skill: str
    params: dict
    target_is_query: bool           # whether this interaction acted on the entity later queried (equality only)
    segment: np.ndarray             # (T_raw, N_CH) raw segment from SegmentRecorder
    effort_cost: float              # api.EffortMeter.cost()
    effort: dict = field(default_factory=dict)
    t_end: float = 0.0              # absolute end time (for ages)
    outcome_cont: Optional[np.ndarray] = None   # (N_CONT,) actor-observable outcomes (training target)
    outcome_cont_mask: Optional[np.ndarray] = None
    outcome_events: Optional[np.ndarray] = None  # (N_EVENTS,) {0,1}
    outcome_events_mask: Optional[np.ndarray] = None
    labels: dict = field(default_factory=dict)  # EVALUATOR-ONLY latent truths {LATENT name: SI}
