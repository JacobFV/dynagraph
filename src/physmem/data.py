"""Observation adapter, tensorizer, labels, dataset generation and loaders (agent DATA).

Pipeline per ROOT episode:
  worlds.sample_episode -> worlds.simulate (ground truth, simulator-only)
  -> observe(): record-layer ObservationEvent stream (stored columnar)   [actor stream]
  -> labels from ground truth                                           [label store]
  -> K=4 samples: history up to just before push k (now_k = t_k - 1ms), action k,
     future target state at HORIZONS after t_k in the canonical frame of the actor's
     latest pose ESTIMATE of the target.

Canonical local indices (before the per-sample random permutation):
  entities: 0 target object, 1 pusher, 2 second object (w3) / anchor (w1)
  sources : see SRC_* constants.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from . import schema as S
from . import worlds as W
from .schema import Prop, Provenance, EntityType, HORIZONS, STATE_DIM, VALUE_DIM, ACTION_DIM

N_MAX = 4  # max entities per sample
S_MAX = 12  # max sources per sample
MAX_EVENTS = 512
DECISION_EPS = 1e-3  # now_k = push_start_k - DECISION_EPS
DATA_DIR = os.path.expanduser(os.environ.get("PHYSMEM_DATA", "~/dynagraph/data"))

SRC_OBJ_POSE, SRC_PUSHER, SRC_CMD, SRC_CONTACT, SRC_SIZE, SRC_VEL, SRC_OBJ2_POSE, SRC_OBJ2_VEL, SRC_PUSHER_CFG, \
    SRC_MAP, SRC_REL = range(11)

# sensor noise
SIG_POS, SIG_YAW = 0.002, math.radians(1.0)
SIG_PPOS, SIG_PYAW = 0.0005, math.radians(0.5)
SIG_SIZE = 0.002
CONTACT_THRESH, CONTACT_NOISE, CONTACT_FLIP = 0.5, 0.3, 0.02

# --------------------------------------------------------------------------
# frame helpers (lead may later swap these for frames.py)
# --------------------------------------------------------------------------


def rot2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def world_to_canonical_xy(xy, origin_xy, yaw0):
    """positions (...,2) world -> frame at origin_xy with heading yaw0."""
    c, s = np.cos(yaw0), np.sin(yaw0)
    d = np.asarray(xy) - np.asarray(origin_xy)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def world_to_canonical_vec(v, yaw0):
    """free vectors (...,2) world -> canonical (rotation only)."""
    c, s = np.cos(yaw0), np.sin(yaw0)
    v = np.asarray(v)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def canonical_values(prop, val, var, x0, y0, yaw0):
    """Convert (E,VALUE_DIM) world-frame values/variances to the canonical frame.
    prop (E,) ints. Returns new (val, var)."""
    val = val.copy()
    var = var.copy()
    c, s = math.cos(yaw0), math.sin(yaw0)

    def rot_var(a, b):  # diag of R^T diag(a,b) R for the inverse rotation
        return c * c * a + s * s * b, s * s * a + c * c * b

    m = prop == Prop.POSE2
    if m.any():
        dx, dy = val[m, 0] - x0, val[m, 1] - y0
        val[m, 0], val[m, 1] = c * dx + s * dy, -s * dx + c * dy
        pc, ps = val[m, 2].copy(), val[m, 3].copy()
        val[m, 2], val[m, 3] = c * pc + s * ps, -s * pc + c * ps  # cos/sin(yaw - yaw0)
        var[m, 0], var[m, 1] = rot_var(var[m, 0], var[m, 1])
        var[m, 2], var[m, 3] = rot_var(var[m, 2], var[m, 3])
    for p in (Prop.VEL2, Prop.COMMAND):
        m = prop == p
        if m.any():
            vx, vy = val[m, 0].copy(), val[m, 1].copy()
            val[m, 0], val[m, 1] = c * vx + s * vy, -s * vx + c * vy
            var[m, 0], var[m, 1] = rot_var(var[m, 0], var[m, 1])
    return val, var


# --------------------------------------------------------------------------
# sensor / observation adapter (record layer, stored columnar)
# --------------------------------------------------------------------------

EV_FIELDS = ("eid", "ent", "rel", "prop", "val", "var", "meas", "recv", "avail", "src", "prov")


class _EvBuf:
    def __init__(self, root_int):
        self.cols = {k: [] for k in EV_FIELDS}
        self.per_src = {}
        self.root = root_int

    def add(self, ent, prop, val, var, meas, recv, src, prov=Provenance.SENSOR, rel=-1, eid=None, avail=None):
        """vectorized add: val/var (n,d) arrays, meas/recv (n,)"""
        val = np.atleast_2d(np.asarray(val, np.float64))
        var = np.atleast_2d(np.asarray(var, np.float64))
        n = val.shape[0]
        meas = np.broadcast_to(np.asarray(meas, np.float64), (n,))
        recv = np.broadcast_to(np.asarray(recv, np.float64), (n,))
        V = np.zeros((n, VALUE_DIM), np.float32)
        Vv = np.zeros((n, VALUE_DIM), np.float32)
        V[:, : val.shape[1]] = val
        Vv[:, : var.shape[1]] = np.broadcast_to(var, (n, var.shape[1]))
        if eid is None:  # ids depend only on (root, source, per-source index) -> independent of the future
            base = self.per_src.get(src, 0)
            eid = self.root * 1_000_000 + int(src) * 50_000 + base + np.arange(n)
            self.per_src[src] = base + n
        c = self.cols
        c["eid"].append(np.asarray(eid, np.int64))
        c["ent"].append(np.full(n, ent, np.int8))
        c["rel"].append(np.full(n, rel, np.int8))
        c["prop"].append(np.full(n, int(prop), np.int8))
        c["val"].append(V)
        c["var"].append(Vv)
        c["meas"].append(meas.copy())
        c["recv"].append(recv.copy())
        c["avail"].append(recv.copy() if avail is None else np.broadcast_to(np.asarray(avail, np.float64), (n,)).copy())
        c["src"].append(np.full(n, src, np.int8))
        c["prov"].append(np.full(n, int(prov), np.int8))
        return eid

    def finish(self):
        out = {k: np.concatenate(v) for k, v in self.cols.items()}
        o = np.lexsort((out["eid"], out["avail"]))  # arrival order
        return {k: v[o] for k, v in out.items()}


def _pose_sensor(buf, rng, pose_arr, ent, src, vel_src, scfg, t_end, push_start):
    """Noisy object pose sensor with occlusion/drop/delay/duplicates + causal derived VEL2.
    pose_arr (T,3) ground truth x,y,yaw at REC_DT."""
    dt = 1.0 / scfg["rate"]
    tm = np.arange(0.0, t_end + 1e-9, dt)
    r = np.clip(np.round(tm / W.REC_DT).astype(int), 0, len(pose_arr) - 1)
    pose = pose_arr[r]
    keep = np.ones(len(tm), bool)
    occ = scfg["occ"]
    if occ > 0:
        for ts in push_start:  # one occlusion window per push period, never before the first push
            a = ts + rng.uniform(-0.5 * occ, W.PUSH_PERIOD - occ)
            a = max(a, W.T0)
            keep &= ~((tm >= a) & (tm < a + occ))
    if scfg["drop"] > 0:
        dropm = rng.random(len(tm)) < scfg["drop"]
        dropm[tm < W.T0 - 0.1] = False  # initial view always delivered
        keep &= ~dropm
    tm, pose = tm[keep], pose[keep]
    n = len(tm)
    x = pose[:, 0] + rng.normal(0, SIG_POS, n)
    y = pose[:, 1] + rng.normal(0, SIG_POS, n)
    yaw = pose[:, 2] + rng.normal(0, SIG_YAW, n)
    val = np.stack([x, y, np.cos(yaw), np.sin(yaw)], 1)
    var = np.array([[SIG_POS**2, SIG_POS**2, SIG_YAW**2, SIG_YAW**2]])
    recv = tm + scfg["delay"] * dt + rng.uniform(0.0005, 0.003, n)
    eid = buf.add(ent, Prop.POSE2, val, var, tm, recv, src)
    if scfg["dup"] > 0:
        dm = rng.random(n) < scfg["dup"]
        if dm.any():
            buf.add(ent, Prop.POSE2, val[dm], var, tm[dm], recv[dm] + dt * rng.uniform(0.3, 2.0, dm.sum()), src,
                    eid=eid[dm])
    # derived velocity: causal finite difference of consecutive RECEIVED pose events (arrival order)
    o = np.argsort(recv, kind="stable")
    tm_o, x_o, y_o, yaw_o, rc_o = tm[o], x[o], y[o], yaw[o], recv[o]
    dtm = np.diff(tm_o)
    ok = (dtm > 1e-6) & (dtm <= 0.3)
    if ok.any():
        i1 = np.nonzero(ok)[0] + 1
        i0 = i1 - 1
        d = dtm[ok]
        vx = (x_o[i1] - x_o[i0]) / d
        vy = (y_o[i1] - y_o[i0]) / d
        wz = W.wrap(yaw_o[i1] - yaw_o[i0]) / d
        vv = np.stack([2 * SIG_POS**2 / d**2, 2 * SIG_POS**2 / d**2, 2 * SIG_YAW**2 / d**2], 1)
        buf.add(ent, Prop.VEL2, np.stack([vx, vy, wz], 1), vv, tm_o[i1], rc_o[i1], vel_src, prov=Provenance.DERIVED)


def observe(ep: W.EpisodeParams, traj: dict, scfg: dict, rng: np.random.Generator, root_int: int) -> dict:
    """Produce the actor-side event stream (columnar dict of arrays, arrival-sorted).

    scfg: rate (Hz), occ (s), delay (intervals), drop (prob), dup (prob)."""
    buf = _EvBuf(root_int)
    starts, t_end = W.episode_times()
    dt = 1.0 / scfg["rate"]
    # object size estimate (once)
    buf.add(0, Prop.SIZE, np.array(ep.half) + rng.normal(0, SIG_SIZE, 3), [[SIG_SIZE**2] * 3], 0.0, 0.05, SRC_SIZE)
    _pose_sensor(buf, rng, traj["obj_pose"][:, 0], 0, SRC_OBJ_POSE, SRC_VEL, scfg, t_end, starts)
    if ep.world in ("w2", "w3", "chain"):
        tm = np.arange(0.0, t_end + 1e-9, dt)
        r = np.clip(np.round(tm / W.REC_DT).astype(int), 0, len(traj["t"]) - 1)
        pp = traj["pusher_pose"][r]
        n = len(tm)
        pyaw = pp[:, 2] + rng.normal(0, SIG_PYAW, n)
        val = np.stack([pp[:, 0] + rng.normal(0, SIG_PPOS, n), pp[:, 1] + rng.normal(0, SIG_PPOS, n), np.cos(pyaw),
                        np.sin(pyaw)], 1)
        buf.add(1, Prop.POSE2, val, [[SIG_PPOS**2, SIG_PPOS**2, SIG_PYAW**2, SIG_PYAW**2]], tm,
                tm + rng.uniform(0.0002, 0.001, n), SRC_PUSHER)
        buf.add(1, Prop.SIZE, [W.PUSHER_SIZE3[ep.pusher]], [[1e-8] * 3], 0.0, 0.001, SRC_PUSHER_CFG)
        # own command: emitted at every change (push start / push end), known immediately
        cm = traj["cmd"]
        ch = np.nonzero(np.any(np.abs(np.diff(cm, axis=0)) > 1e-9, axis=1))[0] + 1
        ch = np.concatenate([[0], ch])
        buf.add(1, Prop.COMMAND, cm[ch], [[1e-6, 1e-6]], traj["t"][ch], traj["t"][ch], SRC_CMD)
        # synthetic tactile contact indicator at half the observation rate (thresholded normal force)
        tc = np.arange(0.0, t_end + 1e-9, 2 * dt)
        rc = np.clip(np.round(tc / W.REC_DT).astype(int), 0, len(traj["t"]) - 1)
        f = traj["f_push"][rc] + rng.normal(0, CONTACT_NOISE, len(tc))
        ind = (f > CONTACT_THRESH).astype(float)
        flip = rng.random(len(tc)) < CONTACT_FLIP
        ind[flip] = 1 - ind[flip]
        buf.add(1, Prop.CONTACT, ind[:, None], [[0.05]], tc, tc + rng.uniform(0.001, 0.004, len(tc)), SRC_CONTACT)
    # extra boxes (w3 second box, chain boxes): entity j+1 for object j>=1, observed by the scene camera
    for j, ex in enumerate(W._extras(ep), start=1):
        buf.add(j + 1, Prop.SIZE, np.array(ex["half"]) + rng.normal(0, SIG_SIZE, 3), [[SIG_SIZE**2] * 3], 0.0,
                0.05, SRC_SIZE)
        _pose_sensor(buf, rng, traj["obj_pose"][:, j], j + 1, SRC_OBJ2_POSE, SRC_OBJ2_VEL, scfg, t_end, starts)
    if ep.agent is not None:  # moving agent: AGENT entity index 2 (agent worlds have no extra boxes)
        buf.add(2, Prop.SIZE, np.array([W.AGENT_R, W.AGENT_R, W.AGENT_HZ]) + rng.normal(0, SIG_SIZE, 3),
                [[SIG_SIZE**2] * 3], 0.0, 0.05, SRC_SIZE)
        _pose_sensor(buf, rng, traj["agent_pose"], 2, SRC_OBJ2_POSE, SRC_OBJ2_VEL, scfg, t_end, starts)
    if ep.world == "w1" and ep.law == W.LAW_SPRING:
        buf.add(2, Prop.POSE2, [[ep.anchor[0], ep.anchor[1], 1.0, 0.0]], [[1e-6, 1e-6, 1e-4, 1e-4]], 0.0, 0.01, SRC_MAP)
        buf.add(0, Prop.RELATION, [[1.0, 0, 0, 0]], [[0.01] * 4], 0.0, 0.01, SRC_REL, rel=2)
    return buf.finish()


def entity_types_for(ep: W.EpisodeParams) -> np.ndarray:
    n_ex = len(W._extras(ep))
    et = np.full(max(N_MAX, 2 + n_ex), -1, np.int64)
    et[0] = EntityType.OBJECT
    if ep.world in ("w2", "w3", "chain"):
        et[1] = EntityType.PUSHER
    for j in range(n_ex):
        et[2 + j] = EntityType.OBJECT
    if ep.agent is not None:
        et[2] = EntityType.AGENT
    if ep.world == "w1" and ep.law == W.LAW_SPRING:
        et[2] = EntityType.FIXTURE
    return et


# --------------------------------------------------------------------------
# record <-> columnar
# --------------------------------------------------------------------------

_ENT_PREFIX = {EntityType.OBJECT: "obj", EntityType.PUSHER: "pus", EntityType.FIXTURE: "fix", EntityType.AGENT: "agt",
               EntityType.LINK: "lnk"}


def opaque_ids(root_int: int, ent_types: np.ndarray):
    """Opaque string ids for entities/sources of a root (deterministic per root)."""
    h = lambda s: hashlib.sha1(f"{root_int}:{s}".encode()).hexdigest()[:10]
    ents = [f"{_ENT_PREFIX[EntityType(int(t))]}_{h('e%d' % i)}" if t >= 0 else None for i, t in enumerate(ent_types)]
    srcs = [f"src_{h('s%d' % i)}" for i in range(S_MAX)]
    return ents, srcs


def columnar_to_events(cols: dict, root_int: int, ent_types: np.ndarray, episode_id: str | None = None):
    ents, srcs = opaque_ids(root_int, ent_types)
    ep_id = episode_id or f"root-{root_int}"
    out = []
    for i in range(len(cols["eid"])):
        p = Prop(int(cols["prop"][i]))
        dmn = S.PROP_DIM[p]
        rel = int(cols["rel"][i])
        out.append(S.ObservationEvent(
            event_id=f"ev-{int(cols['eid'][i])}", episode_id=ep_id, entity_id=ents[int(cols["ent"][i])],
            property_type=p, value=tuple(float(v) for v in cols["val"][i, :dmn]),
            variance=tuple(float(v) for v in cols["var"][i, :dmn]), frame_id="world",
            measured_at=float(cols["meas"][i]), received_at=float(cols["recv"][i]), source_id=srcs[int(cols["src"][i])],
            provenance=Provenance(int(cols["prov"][i])),
            available_from=None if cols["avail"][i] == cols["recv"][i] else float(cols["avail"][i]),
            related_entity_id=ents[rel] if rel >= 0 else None))
    return out


def events_to_columnar(events, entity_order=None):
    """Record-layer list -> columnar dict + entity/source id tables (first-seen order unless given)."""
    ent_ids = list(entity_order) if entity_order is not None else []
    src_ids = []
    for e in events:
        if e.entity_id not in ent_ids:
            ent_ids.append(e.entity_id)
        if e.related_entity_id is not None and e.related_entity_id not in ent_ids:
            ent_ids.append(e.related_entity_id)
        if e.source_id not in src_ids:
            src_ids.append(e.source_id)
    n = len(events)
    cols = {k: None for k in EV_FIELDS}
    cols["eid"] = np.array([e.event_id for e in events], dtype=object)
    cols["ent"] = np.array([ent_ids.index(e.entity_id) for e in events], np.int64)
    cols["rel"] = np.array([ent_ids.index(e.related_entity_id) if e.related_entity_id is not None else -1
                            for e in events], np.int64)
    cols["prop"] = np.array([int(e.property_type) for e in events], np.int64)
    val = np.zeros((n, VALUE_DIM), np.float64)
    var = np.zeros((n, VALUE_DIM), np.float64)
    for i, e in enumerate(events):
        val[i, : len(e.value)] = e.value
        var[i, : len(e.variance)] = e.variance
    cols["val"], cols["var"] = val, var
    cols["meas"] = np.array([e.measured_at for e in events], np.float64)
    cols["recv"] = np.array([e.received_at for e in events], np.float64)
    cols["avail"] = np.array([e.available_at for e in events], np.float64)
    cols["src"] = np.array([src_ids.index(e.source_id) for e in events], np.int64)
    cols["prov"] = np.array([int(e.provenance) for e in events], np.int64)
    return cols, ent_ids, src_ids


# --------------------------------------------------------------------------
# tensorizer (numpy core, shared by the record path and the pre-tensorized cache)
# --------------------------------------------------------------------------


class PrivilegedEventError(ValueError):
    pass


def check_actor_boundary(prov):
    if np.any(np.asarray(prov) == int(Provenance.PRIVILEGED_LABEL)):
        raise PrivilegedEventError("Provenance.PRIVILEGED_LABEL event reached the actor boundary")


def latest_pose_estimate(cols, now, target=0):
    """Actor's latest available pose estimate of the target: (x, y, yaw) or None."""
    m = (cols["avail"] <= now) & (cols["ent"] == target) & (cols["prop"] == Prop.POSE2)
    if not m.any():
        return None
    idx = np.nonzero(m)[0]
    i = idx[np.lexsort((cols["avail"][idx], cols["meas"][idx]))[-1]]
    v = cols["val"][i]
    return float(v[0]), float(v[1]), math.atan2(float(v[3]), float(v[2]))


def tensorize_sample(cols, now, target=0, max_events=MAX_EVENTS):
    """Causal, deduped, canonical-frame, capped event arrays for ONE sample (no permutation).

    Returns dict with arrays of length max_events (ent, rel, prop, val, logvar, age, latency, src, prov, mask)
    plus est (x0,y0,yaw0). Entity/source indices are the canonical (unpermuted) ones of `cols`."""
    check_actor_boundary(cols["prov"])
    avail = cols["avail"]
    m = np.nonzero(avail <= now)[0]
    # dedupe by event id: keep the earliest-available copy (cols may be in any order)
    if len(m):
        o = m[np.lexsort((avail[m],))]
        _, first = np.unique(cols["eid"][o], return_index=True)
        m = o[np.sort(first)]
    est = latest_pose_estimate({k: v[m] for k, v in cols.items()}, now, target) if len(m) else None
    if est is None:
        est = (0.0, 0.0, 0.0)
    # cap: newest event of every (entity, prop, source) group always kept, then most recent others
    meas = cols["meas"][m]
    o = np.lexsort((cols["eid"][m].astype(str) if cols["eid"].dtype == object else cols["eid"][m], meas))
    m = m[o]  # chronological
    if len(m) > max_events:
        key = (cols["ent"][m].astype(np.int64) * 64 + cols["prop"][m]) * 64 + cols["src"][m]
        rev = key[::-1]
        _, last_rev = np.unique(rev, return_index=True)
        newest = len(m) - 1 - last_rev
        pri = np.zeros(len(m), bool)
        pri[newest] = True
        n_rest = max_events - pri.sum()
        rest = np.nonzero(~pri)[0]
        keep = np.concatenate([np.nonzero(pri)[0], rest[len(rest) - n_rest:] if n_rest > 0 else rest[:0]])
        m = m[np.sort(keep)]
    n = len(m)
    prop = cols["prop"][m].astype(np.int64)
    val, var = canonical_values(prop, cols["val"][m].astype(np.float64), cols["var"][m].astype(np.float64), *est)
    out = dict(
        ent=np.zeros(max_events, np.int64), rel=np.full(max_events, -1, np.int64), prop=np.zeros(max_events, np.int64),
        val=np.zeros((max_events, VALUE_DIM), np.float32), logvar=np.zeros((max_events, VALUE_DIM), np.float32),
        age=np.zeros(max_events, np.float32), latency=np.zeros(max_events, np.float32),
        src=np.zeros(max_events, np.int64), prov=np.zeros(max_events, np.int64), mask=np.zeros(max_events, bool))
    out["ent"][:n] = cols["ent"][m]
    out["rel"][:n] = cols["rel"][m]
    out["prop"][:n] = prop
    out["val"][:n] = val
    dims = np.array([S.PROP_DIM[Prop(int(p))] for p in range(S.N_PROPS)])[prop]
    lv = np.log(np.maximum(var, 1e-12))
    lv[np.arange(VALUE_DIM)[None, :] >= dims[:, None]] = 0.0
    out["logvar"][:n] = lv
    out["age"][:n] = now - cols["meas"][m]
    out["latency"][:n] = cols["recv"][m] - cols["meas"][m]
    out["src"][:n] = cols["src"][m]
    out["prov"][:n] = cols["prov"][m]
    out["mask"][:n] = True
    out["est"] = np.array(est, np.float64)
    return out


def _stack_and_permute(samples, ent_types, dt_obs, rng, permute):
    """samples: list of tensorize_sample dicts; ent_types (B,N). Returns torch EventTensors and target idx (B,)."""
    import torch

    B = len(samples)
    st = {k: np.stack([s[k] for s in samples]) for k in ("ent", "rel", "prop", "val", "logvar", "age", "latency",
                                                         "src", "prov", "mask")}
    ent_types = np.asarray(ent_types, np.int64)
    ent_mask = ent_types >= 0
    NN = ent_types.shape[1]
    if permute:
        eperm = np.stack([rng.permutation(NN) for _ in range(B)])
        sperm = np.stack([rng.permutation(S_MAX) for _ in range(B)])
    else:
        eperm = np.tile(np.arange(NN), (B, 1))
        sperm = np.tile(np.arange(S_MAX), (B, 1))
    bi = np.arange(B)[:, None]
    ent = eperm[bi, st["ent"]]
    rel = np.where(st["rel"] >= 0, eperm[bi, np.maximum(st["rel"], 0)], -1)
    src = sperm[bi, st["src"]]
    et_new = np.full_like(ent_types, -1)
    em_new = np.zeros_like(ent_mask)
    et_new[bi, eperm] = ent_types
    em_new[bi, eperm] = ent_mask
    pad = ~st["mask"]
    ent[pad] = 0
    src[pad] = 0
    et = S.EventTensors(
        entity=torch.from_numpy(ent), related=torch.from_numpy(rel), prop=torch.from_numpy(st["prop"]),
        value=torch.from_numpy(st["val"]), logvar=torch.from_numpy(st["logvar"]), age=torch.from_numpy(st["age"]),
        latency=torch.from_numpy(st["latency"]), source=torch.from_numpy(src), provenance=torch.from_numpy(st["prov"]),
        mask=torch.from_numpy(st["mask"]), ent_type=torch.from_numpy(np.maximum(et_new, 0)),
        ent_mask=torch.from_numpy(em_new), dt_obs=torch.as_tensor(np.asarray(dt_obs, np.float32)))
    return et, eperm[:, 0]


def _infer_type(eid: str):
    for t, p in _ENT_PREFIX.items():
        if eid.startswith(p + "_"):
            return int(t)
    return int(EntityType.OBJECT)


def events_to_tensors(list_of_event_lists, now_list, rng, max_events=MAX_EVENTS, permute=True, targets=None,
                      entity_types=None, dt_obs=None, return_info=False):
    """Record path: list (B) of ObservationEvent lists -> schema.EventTensors.

    targets: list of target entity ids (default: first entity whose id/type is OBJECT and has a POSE2 event).
    entity_types: optional list of {entity_id: EntityType}; default inferred from opaque id prefix.
    Raises PrivilegedEventError if any PRIVILEGED_LABEL event is present (even if not yet available).
    return_info=True also returns dict(target_index (B,), est_pose (B,3), entity_ids list)."""
    samples, ets, infos_ids = [], [], []
    for b, (evs, now) in enumerate(zip(list_of_event_lists, now_list)):
        for e in evs:
            if e.provenance == Provenance.PRIVILEGED_LABEL:
                raise PrivilegedEventError(f"privileged event {e.event_id} at actor boundary")
        tgt = targets[b] if targets is not None else None
        if tgt is None:
            tgt = next(e.entity_id for e in evs if e.property_type == Prop.POSE2 and _infer_type(e.entity_id) == 0)
        cols, ent_ids, _ = events_to_columnar(evs, entity_order=[tgt])
        s = tensorize_sample(cols, now, 0, max_events)
        samples.append(s)
        tmap = entity_types[b] if entity_types is not None else {}
        et = np.full(max(N_MAX, len(ent_ids)), -1, np.int64)
        for i, eid in enumerate(ent_ids):
            et[i] = int(tmap.get(eid, _infer_type(eid)))
        ets.append(et)
        infos_ids.append(ent_ids)
    if dt_obs is None:
        dt_obs = [0.05] * len(samples)
    NN = max(len(e) for e in ets)
    ets = [np.concatenate([e, np.full(NN - len(e), -1, np.int64)]) for e in ets]
    et, tidx = _stack_and_permute(samples, np.stack(ets), dt_obs, rng, permute)
    if return_info:
        return et, dict(target_index=tidx, est_pose=np.stack([s["est"] for s in samples]), entity_ids=infos_ids)
    return et


def make_query(target_index, actions, horizons=HORIZONS):
    """target_index (B,) local entity index; actions: list of ActionCandidate or (B,ACTION_DIM) array."""
    import torch

    if len(actions) and isinstance(actions[0], S.ActionCandidate):
        a = np.stack([c.vector() for c in actions])
    else:
        a = np.asarray(actions, np.float32)
    B = len(a)
    return S.QueryTensors(target=torch.as_tensor(np.asarray(target_index, np.int64)),
                          action=torch.as_tensor(a, dtype=torch.float32),
                          horizons=torch.as_tensor(np.tile(np.asarray(horizons, np.float32), (B, 1))))


# --------------------------------------------------------------------------
# labels (simulator-only; label store)
# --------------------------------------------------------------------------

SPEED_EPS, WZ_EPS, FORCE_EPS, TILT_MAX = 0.005, 0.05, 0.05, 0.1


def make_labels_np(traj, ep: W.EpisodeParams, k: int, est, horizons=HORIZONS):
    """Future target state at t_k + h in the canonical frame of `est` = (x0,y0,yaw0) (actor estimate).
    Returns dict future (H,7), future_mask (H,), contact_mode (H,), free_slide_mask (H,), friction, mass, law."""
    x0, y0, yaw0 = est
    tk = float(traj["push_start"][k])
    tend = float(traj["push_end"][k])
    T = len(traj["t"])
    H = len(horizons)
    idx = np.array([int(round((tk + h) / W.REC_DT)) for h in horizons])
    fmask = idx < T
    idx = np.minimum(idx, T - 1)
    pose = traj["obj_pose"][idx, 0]
    vel = traj["obj_vel"][idx, 0]
    dxy = world_to_canonical_xy(pose[:, :2], (x0, y0), yaw0)
    dyaw = pose[:, 2] - yaw0
    v = world_to_canonical_vec(vel[:, :2], yaw0)
    fut = np.concatenate([dxy, np.cos(dyaw)[:, None], np.sin(dyaw)[:, None], v, vel[:, 2:3]], 1).astype(np.float32)
    sp = np.hypot(vel[:, 0], vel[:, 1])
    tilt = traj["obj_tilt"][idx, 0]
    fpush = traj["f_push"][idx]
    cm = np.where(fpush > FORCE_EPS, 1, np.where(tilt > 0.2, 0, np.where((sp > SPEED_EPS) | (np.abs(vel[:, 2]) > WZ_EPS),
                                                                         2, 3)))
    # free slide: after push end + 0.05s, no non-table contact since then, flat, moving at free-phase start,
    # Coulomb law declared (law 0 or 2 [drift only affects the pusher]).
    fs = np.zeros(H, bool)
    if ep.law in (W.LAW_COULOMB, W.LAW_DRIFT):
        tf = tend + 0.05
        i_f = int(round(tf / W.REC_DT))
        if i_f < T:
            sp_f = math.hypot(*traj["obj_vel"][i_f, 0, :2])
            if sp_f > SPEED_EPS:
                for hi, (h, ii) in enumerate(zip(horizons, idx)):
                    if tk + h < tf - 1e-9 or not fmask[hi]:
                        continue
                    seg = slice(i_f, ii + 1)
                    if (traj["f_push"][seg].max() <= FORCE_EPS and traj["f_other"][seg].max() <= FORCE_EPS
                            and traj["obj_tilt"][seg, 0].max() < TILT_MAX):
                        fs[hi] = True
    return dict(future=fut, future_mask=fmask, contact_mode=cm.astype(np.int64), free_slide_mask=fs,
                friction=np.float32(ep.friction), mass=np.float32(ep.mass), law=np.int64(ep.law))


def labels_to_tensors(label_dicts):
    import torch

    st = lambda k, dt=None: torch.as_tensor(np.stack([np.asarray(d[k]) for d in label_dicts]), dtype=dt)
    return S.LabelTensors(future=st("future", torch.float32), future_mask=st("future_mask", torch.bool),
                          friction=st("friction", torch.float32), mass=st("mass", torch.float32),
                          contact_mode=st("contact_mode", torch.long), free_slide_mask=st("free_slide_mask", torch.bool),
                          law=st("law", torch.long))


def action_for(ep: W.EpisodeParams, k: int) -> S.ActionCandidate:
    if ep.world == "w1":
        return S.ActionCandidate(f"a{k}", "impulse", tuple(ep.actions[k]), "target")
    return S.ActionCandidate(f"a{k}", "push", ep.actions[k].params(), "target")


# --------------------------------------------------------------------------
# strata & generation
# --------------------------------------------------------------------------

SENSOR_TRAIN = dict(rate=20, occ=(0.0, 0.0, 0.2, 0.5), delay=(0, 0, 1), drop=(0.0, 0.0, 0.25), dup=0.05)
SENSOR_CLEAN = dict(rate=20, occ=(0.0,), delay=(0,), drop=(0.0,), dup=0.05)
WORLD_TRAIN = dict(world="w2", fric=("uniform", 0.3, 0.7), mass=("loguniform", 0.2, 2.0), cells="exclude",
                   pushers=("sphere15", "sphere30"), law=0)


def _build_strata(base_world):
    def _st(world=None, sensor=None):
        return dict(world={**base_world, **(world or {})}, sensor={**SENSOR_TRAIN, **(sensor or {})})

    return {
    # primary (iid) splits
    "train": _st(), "val": _st(), "test": _st(),
    # held-out physical parameters
    "ood_fric_0.15": _st(world=dict(fric=("const", 0.15), cells="any")),
    "ood_fric_0.9": _st(world=dict(fric=("const", 0.9), cells="any")),
    "heldout_cells": _st(world=dict(cells="only")),
    "paddle": _st(world=dict(pushers=("paddle",))),
    # observation-rate strata (clean otherwise)
    "rate_10": _st(sensor=dict(SENSOR_CLEAN, rate=10)),
    "rate_20": _st(sensor=dict(SENSOR_CLEAN)),
    "rate_40": _st(sensor=dict(SENSOR_CLEAN, rate=40)),
    # corruption sweeps (one factor at a time, others clean)
    **{f"occ_{o}": _st(sensor=dict(SENSOR_CLEAN, occ=(o,))) for o in (0.0, 0.2, 0.5, 1.0)},
    **{f"delay_{d}": _st(sensor=dict(SENSOR_CLEAN, delay=(d,))) for d in (0, 1, 3)},
    **{f"drop_{p}": _st(sensor=dict(SENSOR_CLEAN, drop=(p,))) for p in (0.0, 0.25, 0.5)},
    # misspecified laws
    "misspec_velfric": _st(world=dict(law=W.LAW_VELFRIC)),
    "misspec_drift": _st(world=dict(law=W.LAW_DRIFT)),
    # other worlds
    "w1": _st(world=dict(world="w1", cells="any")),
    "w3": _st(world=dict(world="w3")),
    # e10 contact chains (target = LAST box; pushes applied to the first box, aimed through the chain)
    **{f"chain_L{L}": _st(world=dict(world="chain", chain_L=L)) for L in (1, 2, 4, 8)},
    # e16/h7 moving agent with latent behaviour mode (views: agent_k0 = k==0 samples, agent_klate = k>=1)
    "agent": _st(world=dict(agent=True)),
    }


# v2 (primary): faster pushes so friction changes the consequence of an action (long free slides), clean
# release (pusher retracts), stronger pusher, boxes low enough not to tip while sliding.
WORLD_TRAIN_V2 = dict(WORLD_TRAIN, speed=("loguniform", 0.05, 1.0), dur=(0.1, 0.5), retract=True, fmax=100.0,
                      hz="stable")
# v3 (EXPLORATORY / post-hoc): v2 with fast pushes only, so every push ends in a substantial free slide.
WORLD_TRAIN_V3 = dict(WORLD_TRAIN_V2, speed=("loguniform", 0.4, 1.0))
STRATA_BY_VERSION = {"v1": _build_strata(WORLD_TRAIN), "v2": _build_strata(WORLD_TRAIN_V2),
                     "v3": _build_strata(WORLD_TRAIN_V3)}
STRATA = STRATA_BY_VERSION["v1"]
SPLIT_CODE = {name: i + 1 for i, name in enumerate(STRATA)}


def root_int_for(split: str, idx: int) -> int:
    return SPLIT_CODE[split] * 100_000 + idx


def _sample_sensor(rng, sc):
    pick = lambda v: v[int(rng.integers(len(v)))] if isinstance(v, (tuple, list)) else v
    return dict(rate=sc["rate"], occ=float(pick(sc["occ"])), delay=int(pick(sc["delay"])), drop=float(pick(sc["drop"])),
                dup=float(sc["dup"]))


def generate_root(split: str, idx: int, base_seed: int = 0, max_events: int = MAX_EVENTS, keep_traj=True,
                  version: str = "v1"):
    """Simulate + observe + label one root. Returns dict with events (columnar), samples (tensorized arrays),
    labels, params. Deterministic given (base_seed, split, idx)."""
    root = root_int_for(split, idx)
    ss = np.random.SeedSequence([base_seed, root])
    r_world, r_sensor, r_obs = [np.random.default_rng(s) for s in ss.spawn(3)]
    cfg = STRATA_BY_VERSION[version][split]
    ep = W.sample_episode(r_world, cfg["world"])
    traj = W.simulate(ep)
    scfg = _sample_sensor(r_sensor, cfg["sensor"])
    cols = observe(ep, traj, scfg, r_obs, root)
    ent_types = entity_types_for(ep)
    samples, labels, actions = [], [], []
    for k in range(W.K_PUSHES):
        now = float(traj["push_start"][k]) - DECISION_EPS
        s = tensorize_sample(cols, now, 0, max_events)
        samples.append(s)
        labels.append(make_labels_np(traj, ep, k, s["est"]))
        if "actions_realized" in traj:  # chain: theta/phi realized at re-position time (before now)
            actions.append(S.ActionCandidate(f"a{k}", "push", tuple(traj["actions_realized"][k]), "t").vector())
        else:
            actions.append(action_for(ep, k).vector())
    out = dict(root=root, split=split, idx=idx, params=ep.to_json(), sensor=scfg, events=cols, ent_types=ent_types,
               samples=samples, labels=labels, actions=np.stack(actions), dt_obs=1.0 / scfg["rate"],
               push_start=traj["push_start"], push_end=traj["push_end"],
               agent_mode=int(ep.agent["mode"]) if ep.agent is not None else -1,
               chain_L=(len(ep.others) + 1) if ep.others else (0 if ep.world != "chain" else 1))
    if keep_traj:
        sub = slice(None, None, int(round(0.05 / W.REC_DT)))  # ground truth stored at 20 Hz
        out["traj20"] = {k: traj[k][sub].astype(np.float32) for k in
                         ("t", "obj_pose", "obj_vel", "obj_tilt", "pusher_pose", "f_push", "f_other", "agent_pose",
                          "f_agent") if k in traj}
    return out


def _gen_worker(args):
    split, idx, base_seed, max_events, version = args
    return generate_root(split, idx, base_seed, max_events, version=version)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_split(out_dir, split, roots, max_events):
    import torch

    NN = max(len(r["ent_types"]) for r in roots)
    for r in roots:
        r["ent_types"] = np.concatenate([r["ent_types"], np.full(NN - len(r["ent_types"]), -1, np.int64)])
    # --- actor stream (raw columnar events)
    offs = np.cumsum([0] + [len(r["events"]["eid"]) for r in roots])
    ev = {k: np.concatenate([r["events"][k] for r in roots]) for k in EV_FIELDS}
    np.savez(os.path.join(out_dir, f"{split}_events.npz"), offsets=offs, root=np.array([r["root"] for r in roots]),
             ent_types=np.stack([r["ent_types"] for r in roots]), dt_obs=np.array([r["dt_obs"] for r in roots]), **ev)
    # --- label store (ground truth 20Hz + params) -- simulator-only
    lab = {f"traj_{k}": np.stack([r["traj20"][k] for r in roots]) for k in roots[0]["traj20"]}
    # evaluator-only extra labels, only written for strata that have them (keeps earlier files byte-reproducible)
    extra = {k: np.array([r[k] for r in roots], np.int64) for k in ("agent_mode", "chain_L")
             if any(r[k] > (-1 if k == "agent_mode" else 0) for r in roots)}
    np.savez(os.path.join(out_dir, f"{split}_labels.npz"), root=np.array([r["root"] for r in roots]), **extra,
             push_start=np.stack([r["push_start"] for r in roots]), push_end=np.stack([r["push_end"] for r in roots]),
             **lab)
    with open(os.path.join(out_dir, f"{split}_params.jsonl"), "w") as f:
        for r in roots:
            f.write(json.dumps(dict(root=r["root"], params=r["params"], sensor=r["sensor"])) + "\n")
    # --- pre-tensorized samples (canonical, unpermuted)
    smp = [s for r in roots for s in r["samples"]]
    labs = [l for r in roots for l in r["labels"]]
    t = dict(
        entity=torch.from_numpy(np.stack([s["ent"] for s in smp]).astype(np.int8)),
        related=torch.from_numpy(np.stack([s["rel"] for s in smp]).astype(np.int8)),
        prop=torch.from_numpy(np.stack([s["prop"] for s in smp]).astype(np.int8)),
        value=torch.from_numpy(np.stack([s["val"] for s in smp])),
        logvar=torch.from_numpy(np.stack([s["logvar"] for s in smp])).half(),
        age=torch.from_numpy(np.stack([s["age"] for s in smp])),
        latency=torch.from_numpy(np.stack([s["latency"] for s in smp])),
        source=torch.from_numpy(np.stack([s["src"] for s in smp]).astype(np.int8)),
        provenance=torch.from_numpy(np.stack([s["prov"] for s in smp]).astype(np.int8)),
        mask=torch.from_numpy(np.stack([s["mask"] for s in smp])),
        ent_type=torch.from_numpy(np.repeat(np.stack([r["ent_types"] for r in roots]), W.K_PUSHES, 0)),
        dt_obs=torch.from_numpy(np.repeat(np.array([r["dt_obs"] for r in roots], np.float32), W.K_PUSHES)),
        action=torch.from_numpy(np.concatenate([r["actions"] for r in roots]).astype(np.float32)),
        future=torch.from_numpy(np.stack([l["future"] for l in labs])),
        future_mask=torch.from_numpy(np.stack([l["future_mask"] for l in labs])),
        friction=torch.tensor([float(l["friction"]) for l in labs]),
        mass=torch.tensor([float(l["mass"]) for l in labs]),
        contact_mode=torch.from_numpy(np.stack([l["contact_mode"] for l in labs])),
        free_slide_mask=torch.from_numpy(np.stack([l["free_slide_mask"] for l in labs])),
        law=torch.tensor([int(l["law"]) for l in labs]),
        root=torch.from_numpy(np.repeat(np.array([r["root"] for r in roots], np.int64), W.K_PUSHES)),
        k=torch.from_numpy(np.tile(np.arange(W.K_PUSHES), len(roots))),
        est=torch.from_numpy(np.stack([s["est"] for s in smp])),
    )
    for k, v in extra.items():  # evaluator-only labels (never actor input)
        t[k] = torch.from_numpy(np.repeat(v, W.K_PUSHES))
    torch.save(t, os.path.join(out_dir, f"{split}_tensors.pt"))
    files = [f"{split}_events.npz", f"{split}_labels.npz", f"{split}_params.jsonl", f"{split}_tensors.pt"]
    return {f: _sha256_file(os.path.join(out_dir, f)) for f in files}, t


def fit_scales(t: dict) -> dict:
    """Normalization scales from a tensor dict (call on TRAIN only)."""
    mask = t["mask"].numpy()
    prop = t["prop"].numpy().astype(np.int64)
    val = t["value"].numpy()
    out = {"value_mean": {}, "value_std": {}}
    for p in Prop:
        m = mask & (prop == int(p))
        if m.sum() < 2:
            continue
        v = val[m][:, : S.PROP_DIM[p]].astype(np.float64)
        out["value_mean"][p.name] = v.mean(0).tolist()
        out["value_std"][p.name] = np.maximum(v.std(0), 1e-6).tolist()
    age = t["age"].numpy()[mask]
    out["age_mean"], out["age_std"] = float(age.mean()), float(age.std())
    lat = t["latency"].numpy()[mask]
    out["latency_mean"], out["latency_std"] = float(lat.mean()), float(max(lat.std(), 1e-6))
    a = t["action"].numpy().astype(np.float64)
    out["action_mean"], out["action_std"] = a.mean(0).tolist(), np.maximum(a.std(0), 1e-6).tolist()
    f = t["future"].numpy().astype(np.float64)
    out["future_mean"], out["future_std"] = f.mean(0).tolist(), np.maximum(f.std(0), 1e-6).tolist()
    out["friction_mean"], out["friction_std"] = float(t["friction"].mean()), float(t["friction"].std())
    lm = np.log(t["mass"].numpy())
    out["logmass_mean"], out["logmass_std"] = float(lm.mean()), float(lm.std())
    out["fit_split"] = "train"
    out["n_samples"] = int(len(a))
    return out


DEFAULT_GEN = dict(name="v1", version="v1", n_train=4096, n_val=512, n_test=512, n_strata=512, procs=8, base_seed=20261123,
                   max_events=MAX_EVENTS, strata=None)


def generate(config: dict | None = None):
    """Generate a dataset. config keys: name, n_train, n_val, n_test, n_strata, procs, base_seed, max_events,
    strata (list of stratum names besides train/val/test; None = all), out_root."""
    import multiprocessing as mp

    cfg = {**DEFAULT_GEN, **(config or {})}
    out_dir = os.path.join(os.path.expanduser(cfg.get("out_root", DATA_DIR)), cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    strata = STRATA_BY_VERSION[cfg["version"]]
    others = cfg["strata"] if cfg["strata"] is not None else [s for s in strata if s not in ("train", "val", "test")]
    base = [] if cfg.get("skip_base") else [("train", cfg["n_train"]), ("val", cfg["n_val"]), ("test", cfg["n_test"])]
    plan = base + [(s, cfg["n_strata"]) for s in others]
    manifest = dict(schema=S.SCHEMA_REVISION, config=cfg, base_seed=cfg["base_seed"], created_utc=time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), splits={}, strata_defs={k: strata[k] for k, _ in plan},
        heldout_cells=dict(mass_edges=W.MASS_EDGES.tolist(), fric_edges=W.FRIC_EDGES.tolist(), cells=W.HELDOUT_CELLS),
        horizons=HORIZONS, decision_eps=DECISION_EPS, n_max=N_MAX, s_max=S_MAX,
        notes="split by ROOT id before windowing; root ids = split_code*100000+idx; seeds = SeedSequence([base_seed, root])")
    t_all = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(cfg["procs"]) as pool:
        for split, n in plan:
            t0 = time.time()
            roots = pool.map(_gen_worker, [(split, i, cfg["base_seed"], cfg["max_events"], cfg["version"])
                                           for i in range(n)],
                             chunksize=8)
            tsim = time.time() - t0
            hashes, t = _write_split(out_dir, split, roots, cfg["max_events"])
            manifest["splits"][split] = dict(n_roots=n, n_samples=n * W.K_PUSHES, gen_seconds=round(tsim, 2),
                                             write_seconds=round(time.time() - t0 - tsim, 2), sha256=hashes,
                                             root_range=[root_int_for(split, 0), root_int_for(split, n - 1)],
                                             max_events_hit=float(t["mask"].all(1).float().mean()))
            if split == "train":
                scales = fit_scales(t)
                with open(os.path.join(out_dir, "scales.json"), "w") as f:
                    json.dump(scales, f, indent=1)
                manifest["scales_sha256"] = _sha256_file(os.path.join(out_dir, "scales.json"))
            print(f"[data] {split}: {n} roots in {tsim:.1f}s (+write {manifest['splits'][split]['write_seconds']}s)",
                  flush=True)
            del roots, t
    manifest["total_seconds"] = round(time.time() - t_all, 1)
    mpath = os.path.join(out_dir, "manifest.json")
    if cfg.get("skip_base") and os.path.exists(mpath):  # adding strata to an existing dataset: merge
        with open(mpath) as f:
            old = json.load(f)
        old["splits"].update(manifest["splits"])
        old["strata_defs"].update(manifest["strata_defs"])
        old.setdefault("additions", []).append(dict(created_utc=manifest["created_utc"], splits=[k for k, _ in plan],
                                                    total_seconds=manifest["total_seconds"]))
        manifest = old
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    return manifest


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


class PhysSplit:
    """In-memory pre-tensorized split. Index with a LongTensor of sample ids via `batch(idx, rng)`."""

    def __init__(self, t: dict, name: str, device=None):
        self.name = name
        self.t = {k: (v.to(device) if device is not None else v) for k, v in t.items()}
        self.n = len(self.t["action"])
        self.device = device

    def __len__(self):
        return self.n

    def batch(self, idx, generator=None, permute=True):
        import torch

        t = self.t
        idx = idx.to(t["action"].device)
        B = len(idx)
        dev = t["action"].device
        ent = t["entity"][idx].long()
        rel = t["related"][idx].long()
        src = t["source"][idx].long()
        mask = t["mask"][idx]
        etype = t["ent_type"][idx]
        NN = etype.shape[1]
        if permute:
            eperm = torch.argsort(torch.rand(B, NN, generator=generator, device=dev), 1)
            sperm = torch.argsort(torch.rand(B, S_MAX, generator=generator, device=dev), 1)
        else:
            eperm = torch.arange(NN, device=dev).expand(B, NN)
            sperm = torch.arange(S_MAX, device=dev).expand(B, S_MAX)
        ent_n = torch.gather(eperm, 1, ent)
        rel_n = torch.where(rel >= 0, torch.gather(eperm, 1, rel.clamp(min=0)), torch.full_like(rel, -1))
        src_n = torch.gather(sperm, 1, src)
        ent_n = torch.where(mask, ent_n, torch.zeros_like(ent_n))
        src_n = torch.where(mask, src_n, torch.zeros_like(src_n))
        et_new = torch.full_like(etype, -1).scatter(1, eperm, etype)
        ev = S.EventTensors(entity=ent_n, related=rel_n, prop=t["prop"][idx].long(), value=t["value"][idx],
                            logvar=t["logvar"][idx].float(), age=t["age"][idx], latency=t["latency"][idx],
                            source=src_n, provenance=t["provenance"][idx].long(), mask=mask,
                            ent_type=et_new.clamp(min=0), ent_mask=et_new >= 0, dt_obs=t["dt_obs"][idx])
        H = len(HORIZONS)
        q = S.QueryTensors(target=eperm[:, 0].contiguous(), action=t["action"][idx],
                           horizons=torch.tensor(HORIZONS, dtype=torch.float32, device=dev).expand(B, H))
        lab = S.LabelTensors(future=t["future"][idx], future_mask=t["future_mask"][idx], friction=t["friction"][idx],
                             mass=t["mass"][idx], contact_mode=t["contact_mode"][idx],
                             free_slide_mask=t["free_slide_mask"][idx], law=t["law"][idx])
        return ev, q, lab

    def meta(self, idx=None):
        """root id, push index k and evaluator-only labels (agent_mode, chain_L; -1/0 if n/a)."""
        idx = slice(None) if idx is None else idx
        out = dict(root=self.t["root"][idx], k=self.t["k"][idx])
        for key in ("agent_mode", "chain_L"):
            if key in self.t:
                out[key] = self.t[key][idx]
        return out


# sample-level views of stored splits: name -> (file split, predicate on k)
VIEWS = {"agent_k0": ("agent", lambda k: k == 0), "agent_klate": ("agent", lambda k: k >= 1)}


def load_split(name: str, dataset: str = "v1", device=None, data_root: str | None = None, subset: str | None = None
               ) -> PhysSplit:
    """Load a pre-tensorized split (or a VIEWS entry such as 'agent_k0'). subset in {'1pct','10pct','100pct'}
    restricts TRAIN to the nested root subsets of subsets.json."""
    import torch

    base, pred = VIEWS.get(name, (name, None))
    droot = os.path.join(os.path.expanduser(data_root or DATA_DIR), dataset)
    t = torch.load(os.path.join(droot, f"{base}_tensors.pt"), weights_only=True)
    sel = None
    if pred is not None:
        sel = pred(t["k"])
    if subset is not None:
        with open(os.path.join(droot, "subsets.json")) as f:
            roots = torch.tensor(json.load(f)["subsets"][subset]["roots"])
        m = torch.isin(t["root"], roots)
        sel = m if sel is None else sel & m
    if sel is not None:
        t = {k: v[sel] for k, v in t.items()}
    return PhysSplit(t, name if subset is None else f"{name}[{subset}]", device)


SUBSET_FRACS = {"1pct": 0.01, "10pct": 0.10, "100pct": 1.0}


def make_subsets(dataset: str = "v2", seed: int = 1234, data_root: str | None = None) -> dict:
    """Nested train subsets by ROOT id (1% c 10% c 100%), fixed seed; writes <dataset>/subsets.json."""
    import torch

    droot = os.path.join(os.path.expanduser(data_root or DATA_DIR), dataset)
    t = torch.load(os.path.join(droot, "train_tensors.pt"), weights_only=True)
    roots = np.unique(t["root"].numpy())
    order = np.random.default_rng(seed).permutation(roots)
    r_all = t["root"].numpy()
    out = dict(dataset=dataset, seed=seed, n_roots_total=int(len(roots)), method="nested prefixes of a seeded "
               "permutation of train root ids", subsets={})
    for name, fr in SUBSET_FRACS.items():
        n = max(1, int(round(fr * len(roots))))
        rs = sorted(int(x) for x in order[:n])
        idx = np.nonzero(np.isin(r_all, rs))[0].tolist()
        blob = json.dumps(rs).encode()
        out["subsets"][name] = dict(n_roots=n, n_samples=len(idx), roots=rs, sample_indices=idx,
                                    sha256_roots=hashlib.sha256(blob).hexdigest())
    with open(os.path.join(droot, "subsets.json"), "w") as f:
        json.dump(out, f)
    out["file_sha256"] = _sha256_file(os.path.join(droot, "subsets.json"))
    return out


def load_scales(dataset: str = "v1", data_root: str | None = None) -> dict:
    with open(os.path.join(os.path.expanduser(data_root or DATA_DIR), dataset, "scales.json")) as f:
        return json.load(f)


def load_manifest(dataset: str = "v1", data_root: str | None = None) -> dict:
    with open(os.path.join(os.path.expanduser(data_root or DATA_DIR), dataset, "manifest.json")) as f:
        return json.load(f)


def make_loader(split, batch_size=256, shuffle=True, seed=0, device=None, permute=True, drop_last=False,
                dataset="v1", data_root=None):
    """Iterable over (EventTensors, QueryTensors, LabelTensors) batches. `split` is a PhysSplit or a split name.
    With device="cuda" the whole split lives on the GPU (fastest). Each iteration (epoch) reshuffles and re-draws
    the per-sample entity/source permutations from a generator seeded by (seed, epoch)."""
    import torch

    ds = split if isinstance(split, PhysSplit) else load_split(split, dataset, device, data_root)

    class _Loader:
        def __init__(self):
            self.epoch = 0
            self.ds = ds

        def __len__(self):
            return ds.n // batch_size if drop_last else -(-ds.n // batch_size)

        def __iter__(self):
            dev = ds.t["action"].device
            g = torch.Generator(device=dev)
            g.manual_seed(seed * 100003 + self.epoch)
            self.epoch += 1
            order = torch.randperm(ds.n, generator=g, device=dev) if shuffle else torch.arange(ds.n, device=dev)
            for i in range(0, ds.n, batch_size):
                idx = order[i:i + batch_size]
                if drop_last and len(idx) < batch_size:
                    break
                yield ds.batch(idx, g, permute)

    return _Loader()


def root_events(split: str, i: int, dataset="v1", data_root=None):
    """Record-layer ObservationEvent list for root i of a stored split (+ root id, entity types)."""
    z = np.load(os.path.join(os.path.expanduser(data_root or DATA_DIR), dataset, f"{split}_events.npz"))
    a, b = z["offsets"][i], z["offsets"][i + 1]
    cols = {k: z[k][a:b] for k in EV_FIELDS}
    root = int(z["root"][i])
    return columnar_to_events(cols, root, z["ent_types"][i]), root, z["ent_types"][i]


def main_generate(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--version", default=None, help="world/strata version (default: same as name if v1/v2)")
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--n_val", type=int, default=512)
    ap.add_argument("--n_test", type=int, default=512)
    ap.add_argument("--n_strata", type=int, default=512)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--base_seed", type=int, default=DEFAULT_GEN["base_seed"])
    ap.add_argument("--strata", default=None, help="comma list; 'none' for only train/val/test")
    a = ap.parse_args(argv)
    strata = None if a.strata is None else ([] if a.strata == "none" else a.strata.split(","))
    version = a.version or (a.name if a.name in STRATA_BY_VERSION else "v1")
    m = generate(dict(name=a.name, version=version, n_train=a.n_train, n_val=a.n_val, n_test=a.n_test, n_strata=a.n_strata,
                      procs=a.procs, strata=strata, base_seed=a.base_seed))
    print(json.dumps({k: v["gen_seconds"] for k, v in m["splits"].items()}), m["total_seconds"])


if __name__ == "__main__":
    main_generate()
