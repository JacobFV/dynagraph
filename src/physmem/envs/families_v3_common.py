"""envs-v3 shared pieces (agent ENVS-V3): base class, load-cell scale sensor, particle media, vessel geometry.

Contract (unchanged from r7): actor-legal observation events only (tracker / proprio / F/T / tactile / CAD regions, plus
the new SCALE packets below), hidden per-object / per-lot physics in `labels` only, evaluator-only oracle
(`v3_rule(call, 'true')`), failure-cause tags through the skill status vocabulary of scripts/envs_ceiling.py.

New sensor (envs-v3): a LOAD-CELL SCALE under a receiving container. Physics: the container is a body on a stiff
vertical slide joint (spring k, damping c); the reading is the spring force from the joint deflection, tared after the
reset settle, as `Prop.MASS` events on entity id ``scale/<name>`` (value = tared load in kg, source_id ``scale``),
25 Hz, 40-70 ms latency, sigma 1-2 g. It is a real measurement of the container's contents (not a sim-state read of
the particles), so it is actor-legal.

Particle media (pouring, granular): free spheres that are NOT tracked entities (no per-particle events; the actor sees
the scale, its own F/T and the tracked vessel / tool). Their state is evaluator-only (labels / success predicate).
A blow-up monitor covers them (non-finite state or a particle faster than EJECT_V marks the episode unstable).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..schema import Prop
from .base import StationEnv
from .probe import ProbeSkills
from .scene import quat_z

G = mujoco.mjtGeom
J = mujoco.mjtJoint


class V3Base(StationEnv):
    category = "v3"
    tilt_probe = False             # True: the probe gets the envs-v3 wrist tilt hinge (ProbeTiltRobot)
    tool = "parallel"
    PARTICLE_EJECT_V = 6.0

    # ------------------------------------------------------------------ robot
    def _make_robot(self, name):
        if self.robot_factory is None and name == "probe" and self.tilt_probe:
            from .v3_motion import ProbeTiltRobot
            r = ProbeTiltRobot(tool="parallel", **self.probe_kw)
            return r, ProbeSkills(r)
        return super()._make_robot(name)

    # ------------------------------------------------------------------ oracle / PTA hooks (scripts/envs_ceiling.py)
    def v3_rule(self, call, src):
        return call

    def probe_then_act(self, call):
        return call

    # ------------------------------------------------------------------ generic light-squeeze probe (actor-legal)
    PROBE_F = 4.0

    def fsk_squeeze_probe(self, M, call):
        """pinch the target at a light PROBE_F (below every crush limit) and let go: the pads' squeeze yields one
        tactile stiffness reading (StationEnv._tactile_squeeze) -> est_crush(). params: yaw (hand yaw; default: across
        the tracked yaw), dz (TCP height offset from the tracked centre)."""
        p = call.params
        est = self.est_pose(call.target)
        if est is None:
            return "not_observed", {}
        pos, _, yaw_t = est
        yaw = float(p.get("yaw", yaw_t))
        tcp, y_now = M.tcp()
        if abs(math.remainder(yaw - y_now, 2 * math.pi)) > math.pi / 2:
            yaw = math.remainder(yaw + math.pi, 2 * math.pi)
        z = max(pos[2] + float(p.get("dz", 0.0)), self.surface_z + M.tip() + 0.003)
        top = max(tcp[2], z + 0.08)
        M.open()
        for q, v in (((tcp[0], tcp[1], top), 0.4), ((pos[0], pos[1], top), 0.4), ((pos[0], pos[1], z), 0.08)):
            if M.move(q, yaw, 0.0, v) == "unreachable":
                return "unreachable", {}
        M.grip(self.PROBE_F)
        M.wait(0.3)
        M.open()
        M.move((pos[0], pos[1], top), yaw, 0.0, 0.2)
        k, src = self.est_stiffness(call.target)
        return "done", dict(stiffness=k, source=src)

    def _probed(self, eid):
        return self.est_stiffness(eid)[1] == "tactile_squeeze"

    # ------------------------------------------------------------------ scale
    def add_scale(self, scene, name, center, inner_half, wall_h, k=8000.0, mass=0.4, rgba=(0.85, 0.85, 0.88, 1),
                  t=0.006, mu=0.3):
        """open-top receiving container on a load cell (vertical slide joint with spring k). Returns its floor z."""
        cx, cy, z0 = center
        b = scene.spec.worldbody.add_body(name=f"{name}_body", pos=[cx, cy, z0])
        b.add_joint(name=f"{name}_cell", type=J.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.01, 0.01], limited=True,
                    stiffness=k, springref=0.0, damping=2.0 * math.sqrt(k * mass), armature=0.0)
        hx, hy = inner_half
        fr = [mu, 0.005, 0.0001]
        g = b.add_geom(type=G.mjGEOM_BOX, size=[hx + t, hy + t, t / 2], pos=[0, 0, t / 2], rgba=list(rgba),
                       friction=fr, name=f"{name}_floor")
        g.mass = mass * 0.4
        for sx, sy, hh in ((1, 0, (t / 2, hy + t, wall_h / 2)), (-1, 0, (t / 2, hy + t, wall_h / 2)),
                           (0, 1, (hx, t / 2, wall_h / 2)), (0, -1, (hx, t / 2, wall_h / 2))):
            w = b.add_geom(type=G.mjGEOM_BOX, size=list(hh), pos=[sx * (hx + t / 2), sy * (hy + t / 2), t + wall_h / 2],
                           rgba=list(rgba), friction=fr)
            w.mass = mass * 0.15
        # the load cell's housing under it (static)
        scene.static_box((cx, cy, z0 - 0.012), (hx * 0.6, hy * 0.6, 0.01), rgba=(0.2, 0.2, 0.22, 1))
        fz = z0 + t
        scene.region(name, (cx, cy, fz + wall_h / 2), (hx, hy, wall_h / 2), support_z=fz, scale=f"scale/{name}")
        self._scales = getattr(self, "_scales", {})
        self._scales[name] = dict(joint=f"{name}_cell", k=k, half=(hx, hy), wall_h=wall_h, z0=z0, t=t)
        return fz

    def _scale_bind(self):
        m = self.m
        self._scale_rt = {}
        for name, s in getattr(self, "_scales", {}).items():
            j = m.joint(s["joint"]).id
            self._scale_rt[name] = dict(qadr=m.jnt_qposadr[j], k=s["k"], tare=0.0, next=0.0)
        rng = np.random.default_rng(int(self.rng.integers(1 << 31)))
        self._scale_cfg = dict(sigma=float(rng.uniform(0.001, 0.002)), latency=float(rng.uniform(0.04, 0.07)),
                               hz=25.0)
        self._scale_rng = rng
        self._scale_tared = False

    def scale_true_kg(self, name):
        s = self._scale_rt[name]
        return float(-s["k"] * self.d.qpos[s["qadr"]] / 9.81 - s["tare"])

    def scale_reading(self, name, window=0.0):
        """actor-legal: latest AVAILABLE scale packet (kg), or the median over the last `window` s; None if none."""
        evs = [e for e in self.view.buf.get((f"scale/{name}", int(Prop.MASS)), ()) if e.available_at <= self.t + 1e-9]
        if not evs:
            return None
        if window <= 0:
            return float(max(evs, key=lambda e: e.measured_at).value[0])
        last = max(e.measured_at for e in evs)
        return float(np.median([e.value[0] for e in evs if e.measured_at >= last - window]))

    # ------------------------------------------------------------------ particles
    def add_particles(self, scene, prefix, centers, radius, mass, mu, solref=(0.004, 1.0), rgba=(0.2, 0.45, 0.9, 1),
                      condim=3, solimp=None, priority=1, damping=0.0):
        """free spheres (not entities). Their contacts use THEIR friction / solref (priority over fixtures)."""
        names = []
        for i, c in enumerate(centers):
            b = scene.spec.worldbody.add_body(name=f"{prefix}{i:03d}", pos=[float(v) for v in c])
            j = b.add_freejoint(name=f"{prefix}{i:03d}_free")
            if damping > 0:
                j.damping = [float(damping), 0.0, 0.0]   # linear drag vs the world frame (pour: viscosity proxy)
            g = b.add_geom(type=G.mjGEOM_SPHERE, size=[radius, 0, 0], rgba=list(rgba), friction=[mu, 0.002, 0.0001],
                           solref=list(solref), condim=condim, priority=priority, name=f"{prefix}{i:03d}_g")
            if solimp is not None:
                g.solimp = list(solimp)
            g.mass = float(mass)
            names.append(b.name)
        self._particle_names = getattr(self, "_particle_names", []) + names
        return names

    def _particle_bind(self):
        m = self.m
        names = getattr(self, "_particle_names", [])
        self._pbody = np.array([m.body(n).id for n in names], int)
        self._pq = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in self._pbody], int)
        self._pd = np.array([m.jnt_dofadr[m.body_jntadr[b]] for b in self._pbody], int)

    def particle_pos(self):
        if not len(getattr(self, "_pbody", ())):
            return np.zeros((0, 3))
        return self.d.xpos[self._pbody].copy()

    # ------------------------------------------------------------------ sim loop hooks
    def _index(self):
        super()._index()
        self._scale_bind()
        self._particle_bind()

    def post_physics(self, stop=None):
        out = super().post_physics(stop)
        if self._step_i % 5 == 0 and len(getattr(self, "_pd", ())):
            v = self.d.qvel[self._pd[:, None] + np.arange(3)[None, :]]
            if not np.isfinite(v).all():
                self._mark_unstable("nonfinite")
            elif float(np.max(np.einsum("ij,ij->i", v, v))) > self.PARTICLE_EJECT_V ** 2:
                self.metrics_acc["ejections"] += 1
                self._mark_unstable("ejection")
        if getattr(self, "_scale_rt", None) and not self._settling and not self.unstable_reasons \
                and getattr(self, "sensors", None) is not None:
            c = self._scale_cfg
            if not self._scale_tared:            # tare at the first sensing step after the reset settle
                self._tare_scales()
                self._scale_tared = True
            for name, s in self._scale_rt.items():
                if self.t + 1e-9 >= s["next"]:
                    s["next"] = self.t + 1.0 / c["hz"]
                    val = self.scale_true_kg(name) + self._scale_rng.normal(0.0, c["sigma"])
                    self.sensors.ev(f"scale/{name}", Prop.MASS, (val,), (c["sigma"] ** 2,), self.t, c["latency"],
                                    "scale")
        return out

    def _tare_scales(self):
        for name, s in getattr(self, "_scale_rt", {}).items():
            s["tare"] = 0.0
            s["tare"] = self.scale_true_kg(name)

    def reset(self, task):
        ev = super().reset(task)
        return ev

    def _reset_once(self, task, attempt=0):
        self._scales, self._particle_names = {}, []
        return super()._reset_once(task, attempt)


def vessel(scene, eid, pos, yaw, r_in, h, wall=0.004, n_seg=14, post=(0.012, 0.010, 0.05), mass=0.12, mu=0.4,
           mu_grip=0.9, rgba=(0.9, 0.9, 0.92, 1), kind="vessel", extra_hidden=None, post_rgba=(0.2, 0.2, 0.22, 1),
           spout=None):
    """free open cylinder (n_seg wall boxes + floor) with a vertical HANDLE POST rising from the rim at +x (vessel frame):
    pinched across x (pads on the post's inner / outer faces) the vessel is rigid in the hand and tilts about the
    hand x axis (world tangential direction) -> it pours over the far (-x) lip. Body origin = floor centre."""
    from .scene import EntityInfo
    from ..schema import EntityType
    b = scene.spec.worldbody.add_body(name=eid, pos=list(pos), quat=list(quat_z(yaw)))
    b.add_freejoint(name=eid + "_free")
    geoms = []
    fr = [mu, 0.005, 0.0001]
    seg_w = 2 * (r_in + wall / 2) * math.tan(math.pi / n_seg) + 0.001
    m_wall = mass * 0.6 / n_seg
    for i in range(n_seg):
        a = 2 * math.pi * i / n_seg
        rr = r_in + wall / 2
        g = b.add_geom(type=G.mjGEOM_BOX, size=[wall / 2, seg_w / 2, h / 2], pos=[rr * math.cos(a), rr * math.sin(a), h / 2],
                       quat=list(quat_z(a)), rgba=list(rgba), friction=fr, name=f"{eid}_w{i}")
        g.mass = m_wall
        geoms.append(g.name)
    g = b.add_geom(type=G.mjGEOM_CYLINDER, size=[r_in + wall, wall / 2, 0], pos=[0, 0, -wall / 2], rgba=list(rgba),
                   friction=fr, name=f"{eid}_floor")
    g.mass = mass * 0.25
    geoms.append(g.name)
    if spout is not None:
        # lid over the mouth with a SPOUT slot at the far (-x) lip: width spout[0] (y), depth spout[1] (x); throttles
        # the flow (an open cylinder of spheres pours as one avalanche)
        sw, sd = spout
        lt = 0.003
        x_cut = -r_in + sd
        R = r_in + wall
        g = b.add_geom(type=G.mjGEOM_BOX, size=[(R - x_cut) / 2, R, lt / 2], pos=[(R + x_cut) / 2, 0, h + lt / 2],
                       rgba=list(rgba), friction=fr, name=f"{eid}_lid")
        g.mass = mass * 0.05
        geoms.append(g.name)
        for sy in (-1, 1):
            g = b.add_geom(type=G.mjGEOM_BOX, size=[(x_cut + R) / 2, (R - sw / 2) / 2, lt / 2],
                           pos=[(x_cut - R) / 2, sy * (sw / 2 + (R - sw / 2) / 2), h + lt / 2], rgba=list(rgba),
                           friction=fr, name=f"{eid}_lid{sy > 0:d}")
            g.mass = mass * 0.02
            geoms.append(g.name)
    px, py, ph = post
    # post: outside the rim at +x, from 1 cm below the rim to ph above it (gripped above the rim)
    g = b.add_geom(type=G.mjGEOM_BOX, size=[px / 2, py, (ph + 0.01) / 2],
                   pos=[r_in + wall + px / 2, 0, h + (ph - 0.01) / 2], rgba=list(post_rgba), friction=fr,
                   name=f"{eid}_post")
    g.mass = mass * 0.15
    geoms.append(g.name)
    half = ((2 * (r_in + wall) + px) / 2, r_in + wall, (h + ph) / 2)
    hidden = dict(mass=float(mass), mu=float(mu), mu_grip=float(mu_grip))
    hidden.update(extra_hidden or {})
    info = EntityInfo(eid, kind, EntityType.OBJECT, eid, "cylinder", tuple(float(v) for v in half), hidden,
                      graspable=True, geoms=geoms,
                      meta=dict(r_in=r_in, h=h, post_r=r_in + wall + px / 2, post_top=h + ph, post_bot=h - 0.01,
                                origin="floor_centre"))
    scene.entities[eid] = info
    scene.grip_pairs_objects.append(eid)
    return b, info


def hex_fill(rng, center, r_in, z0, radius, n, jitter=0.15):
    """n sphere centres stacked in layers inside a cylinder of inner radius r_in from floor z0 (vessel frame offset
    applied by the caller)."""
    pts = []
    d = 2.05 * radius
    z = z0 + radius * 1.05
    while len(pts) < n:
        layer = []
        k = 0
        for ix in range(-20, 21):
            for iy in range(-20, 21):
                x = ix * d + (0.5 * d if (len(pts) // 50) % 2 else 0.0)
                y = iy * d * 0.866
                if ix % 2:
                    y += d * 0.433
                if math.hypot(x, y) <= r_in - radius * 1.1:
                    layer.append((x, y))
        rng.shuffle(layer)
        for x, y in layer:
            if len(pts) >= n:
                break
            j = rng.uniform(-jitter, jitter, 2) * radius
            pts.append((center[0] + x + j[0], center[1] + y + j[1], z))
            k += 1
        z += d * 0.95
    return pts
