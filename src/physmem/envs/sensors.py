"""Synthetic sensors -> schema.ObservationEvent streams (record layer), and the ActorView cache.

Every event here is provenance SENSOR or DERIVED (CAD layout / encoder FK). Nothing PRIVILEGED is ever
emitted by an environment; evaluator truths go to StepOut.labels only.

Sensors (per-episode noise / latency / dropout drawn from the task rng; their values are labels):
  * pose tracker (fixed camera): POSE3 + POSE2 per movable entity, Gaussian position / yaw noise,
    latency (mean + jitter), random dropout, line-of-sight occlusion by mj_ray from the camera to the
    entity centre (any other geom first => no measurement). Change-triggered for bulk stations:
    an entity is re-reported when it moved > trigger_m since its last report or every keyframe_s.
  * joint encoders (JOINT per robot joint), effector pose by FK of encoders (EE_POSE, DERIVED),
    wrist F/T (WRENCH), gripper width/command (GRIPPER), pad tactile (TACTILE + schema CONTACT flags).
  * articulated links: JOINT of each LINK entity (door angle / drawer travel) through the tracker.
  * one-shot at reset: SIZE (noisy half extents), SHAPE (visual category), REGION (CAD fixtures / goals).
  * optional depth image (not in the actor loop by default): StationEnv.capture_depth().
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import mujoco
import numpy as np

from ..schema import EntityType, ObservationEvent, Prop, Provenance
from .props import SHAPES, XProp


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class SensorSuite:
    def __init__(self, env, rng, bulk=False, fine=False):
        self.env = env
        self.rng = rng
        self.ep = env.episode_id
        self.n = 0
        self.out: list = []
        fine_k = 0.5 if fine else 1.0
        self.cfg = dict(
            track_hz=2.0 if bulk else 10.0,
            pos_sigma=float(rng.uniform(0.001, 0.003) * fine_k),
            yaw_sigma=float(math.radians(rng.uniform(0.5, 2.0))),
            z_sigma=float(rng.uniform(0.001, 0.003) * fine_k),
            latency=float(rng.uniform(0.03, 0.09)),
            jitter=0.01,
            dropout=float(rng.uniform(0.0, 0.08)),
            trigger_m=0.003 if bulk else 0.0,
            keyframe_s=2.0 if bulk else 0.0,
            proprio_hz=25.0,
            enc_sigma=1e-5,
            ft_sigma=float(rng.uniform(0.2, 0.6)),
            size_sigma=0.001,
        )
        self.cam = np.asarray(env.scene.camera["pos"], float)
        self._next_track = 0.0
        self._next_prop = 0.0
        self._last_rep = {}

    # ------------------------------------------------------------------ event factory
    def ev(self, eid, prop, value, var, t_meas, latency=0.0, source="tracker", prov=Provenance.SENSOR,
           frame="world", related=None):
        self.n += 1
        e = ObservationEvent(event_id=f"{self.ep}:{self.n}", episode_id=self.ep, entity_id=eid,
                             property_type=prop, value=tuple(float(v) for v in value),
                             variance=tuple(float(v) for v in var), frame_id=frame, measured_at=float(t_meas),
                             received_at=float(t_meas + latency), source_id=source, provenance=prov,
                             related_entity_id=related)
        self.out.append(e)
        view = getattr(self.env, "view", None)
        if view is not None:          # online: skills see packets as soon as they are available
            view.add((e,))
        return e

    def flush(self):
        o, self.out = self.out, []
        return o

    # ------------------------------------------------------------------ one-shot
    def initial(self, t):
        env, c = self.env, self.cfg
        for eid, info in env.scene.entities.items():
            h = np.asarray(info.half) + self.rng.normal(0, c["size_sigma"], 3)
            self.ev(eid, Prop.SIZE, np.maximum(h, 0.002), [c["size_sigma"] ** 2] * 3, t, 0.0, "tracker")
            oh = np.zeros(len(SHAPES))
            oh[SHAPES.index(info.shape) if info.shape in SHAPES else 0] = 1
            self.ev(eid, XProp.SHAPE, oh, [0.0] * len(SHAPES), t, 0.0, "tracker")
        for eid, info in env.scene.entities.items():      # articulated links: first joint reading always arrives
            if info.etype == EntityType.LINK:
                q = float(env.d.qpos[env.m.jnt_qposadr[env.m.joint(info.meta["joint"]).id]])
                sig = 0.01 if info.kind in ("door", "knob") else 0.001
                self.ev(eid, XProp.JOINT, (q + self.rng.normal(0, sig), 0.0), (sig ** 2, 1.0), t, c["latency"])
        for name, reg in env.scene.regions.items():
            cx, cy, cz = reg["center"]
            hx, hy, hz = reg["half"]
            self.ev(name, XProp.REGION, (cx, cy, cz, hx, hy, hz, reg.get("yaw", 0.0)), [1e-8] * 7, t, 0.0, "cad",
                    Provenance.DERIVED)

    # ------------------------------------------------------------------ periodic
    def tick(self, t):
        if t + 1e-9 >= self._next_track:
            self._next_track = t + 1.0 / self.cfg["track_hz"]
            self.track(t)
        if t + 1e-9 >= self._next_prop:
            self._next_prop = t + 1.0 / self.cfg["proprio_hz"]
            self.proprio(t)

    def visible(self, body_id, p):
        m, d = self.env.m, self.env.d
        vec = p - self.cam
        dist = float(np.linalg.norm(vec))
        gid = np.array([-1], np.int32)
        hit = mujoco.mj_ray(m, d, self.cam, vec / dist, None, 1, -1, gid)
        if gid[0] < 0:
            return True
        return m.geom_bodyid[gid[0]] == body_id or hit >= dist - 0.002

    def track(self, t):
        env, c, rng = self.env, self.cfg, self.rng
        d = env.d
        for eid, info in env.scene.entities.items():
            if not info.movable:
                continue
            if info.etype == EntityType.LINK:
                q = float(d.qpos[env.m.jnt_qposadr[env.m.joint(info.meta["joint"]).id]])
                sig = 0.01 if info.kind == "door" else 0.001
                if rng.random() > c["dropout"]:
                    self.ev(eid, XProp.JOINT, (q + rng.normal(0, sig), 0.0), (sig ** 2, 1.0), t,
                            c["latency"] + abs(rng.normal(0, c["jitter"])))
                continue
            p, quat = env.true_pose(eid)
            if c["trigger_m"] > 0:
                last = self._last_rep.get(eid)
                if last is not None and np.linalg.norm(p - last[0]) < c["trigger_m"] and t - last[1] < c["keyframe_s"]:
                    continue
            if rng.random() < c["dropout"]:
                continue
            bid = env.body_id[eid]
            if bid >= 0 and not self.visible(bid, p):
                continue
            self._last_rep[eid] = (p.copy(), t)
            lat = c["latency"] + abs(rng.normal(0, c["jitter"]))
            pn = p + rng.normal(0, [c["pos_sigma"], c["pos_sigma"], c["z_sigma"]])
            yaw = quat_yaw(quat) + rng.normal(0, c["yaw_sigma"])
            # noisy quaternion: perturb yaw only (tilt reported exactly from the measured quaternion's tilt)
            w, x, y, z = quat
            tilt_q = quat_mul(quat_from_yaw(-quat_yaw(quat)), quat)
            qn = quat_mul(quat_from_yaw(yaw), tilt_q)
            ps2 = c["pos_sigma"] ** 2
            ys2 = c["yaw_sigma"] ** 2
            self.ev(eid, XProp.POSE3, (*pn, *qn), (ps2, ps2, c["z_sigma"] ** 2, ys2, ys2, ys2, ys2), t, lat)
            self.ev(eid, Prop.POSE2, (pn[0], pn[1], math.cos(yaw), math.sin(yaw)), (ps2, ps2, ys2, ys2), t, lat)

    def proprio(self, t):
        env, c, rng = self.env, self.cfg, self.rng
        r = env.robot
        if r is None:
            return
        qs, dqs = r.joint_state()
        for nm, q, dq in zip(r.joint_names(), qs, dqs):
            self.ev("robot/" + nm, XProp.JOINT, (q + rng.normal(0, c["enc_sigma"]), dq), (c["enc_sigma"] ** 2, 1e-6),
                    t, 0.002, "encoders")
        tcp, yaw = r.tcp()
        self.ev("robot/hand", XProp.EE_POSE, (*tcp, *quat_from_yaw(yaw)), [1e-8] * 7, t, 0.002, "encoders",
                Provenance.DERIVED)
        w = r.wrench() + rng.normal(0, c["ft_sigma"], 6) * np.array([1, 1, 1, 0.05, 0.05, 0.05])
        self.ev("robot/hand", XProp.WRENCH, w, [c["ft_sigma"] ** 2] * 3 + [(0.05 * c["ft_sigma"]) ** 2] * 3, t, 0.004,
                "wrist_ft")
        gw = r.grip_width()
        mode, val = getattr(r, "grip_cmd", ("open", 0.0))
        self.ev("robot/hand", XProp.GRIPPER, (gw, val if mode == "force" else 0.0, getattr(r, "suction_cmd", 0.0)),
                (1e-8, 0.0, 0.0), t, 0.002, "gripper")
        lf, rf, pf, fn = r.pad_contacts()
        self.ev("robot/hand", XProp.TACTILE, (lf, rf, pf, fn + abs(rng.normal(0, 0.5))), (0, 0, 0, 0.25), t, 0.004,
                "tactile")
        self.ev("robot/hand", Prop.CONTACT, (float(max(lf, rf, pf)),), (0.0,), t, 0.004, "tactile")


def quat_from_yaw(y):
    return np.array([math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class ActorView:
    """Actor-legal cache over the events an env has EMITTED (respects available_at). Used by skills and
    candidate generation; equivalent to what an actor could rebuild from the event stream."""

    def __init__(self, keep=6):
        self.keep = keep
        self.buf = defaultdict(lambda: deque(maxlen=keep))

    def add(self, events):
        for e in events:
            if e.provenance == Provenance.PRIVILEGED_LABEL:
                raise ValueError("privileged event reached ActorView")
            self.buf[(e.entity_id, int(e.property_type))].append(e)

    def latest(self, eid, prop, now):
        best = None
        for e in self.buf.get((eid, int(prop)), ()):
            if e.available_at <= now + 1e-9 and (best is None or e.measured_at > best.measured_at):
                best = e
        return best

    def entities(self, prop):
        return sorted({k[0] for k in self.buf if k[1] == int(prop)})
