"""Procedural MjSpec scene builder for v2 industrial stations (agent ENVS).

All geometry is primitive / procedural (boxes, cylinders, capsules, spheres, ellipsoids, random convex
hull meshes). Dimensions follow the CommandAGI factory library where it has them (bench top 0.76 m,
KLT-style totes 0.30 x 0.20 x 0.12 m, EUR pallet 1.2 x 0.8 x 0.144 m, cartons 0.2-0.4 m).

Friction convention (MuJoCo combines two geoms' friction by max at equal priority):
  * fixture surfaces (bench, bin, pallet, shelf, hole walls) get friction FIX_MU = 0.01, so the
    object's own geom friction IS the object<->surface coefficient (hidden variable `mu`);
  * object<->gripper friction (hidden `mu_grip`) is set with explicit contact pairs between the
    gripper pad geoms and each graspable object's geoms (scene.grip_pairs, added by the robot build);
  * restitution / compliance through per-geom solref (timeconst, dampratio).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

from ..schema import EntityType

FIX_MU = 0.01
BENCH_H = 0.76
G = mujoco.mjtGeom

PALETTE = [(0.85, 0.33, 0.1, 1), (0.13, 0.47, 0.71, 1), (0.17, 0.63, 0.17, 1), (0.84, 0.15, 0.16, 1),
           (0.58, 0.4, 0.74, 1), (0.55, 0.34, 0.29, 1), (0.89, 0.47, 0.76, 1), (0.74, 0.74, 0.13, 1),
           (0.09, 0.75, 0.81, 1), (0.5, 0.5, 0.5, 1)]
GREY = (0.6, 0.6, 0.62, 1)
DARK = (0.25, 0.25, 0.28, 1)
CARDBOARD = (0.72, 0.56, 0.38, 1)
BLUE_TOTE = (0.15, 0.3, 0.6, 1)


def quat_z(yaw):
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def restitution_to_dampratio(e):
    e = float(np.clip(e, 1e-3, 0.95))
    l = math.log(e)
    return -l / math.sqrt(math.pi ** 2 + l * l)


@dataclass
class EntityInfo:
    eid: str
    kind: str                      # part | carton | tote | peg | module | tool | fixture | link | pouch | bin | marker
    etype: EntityType
    body: str
    shape: str = "box"
    half: tuple = (0.02, 0.02, 0.02)  # bounding half extents in body frame (observable SIZE)
    hidden: dict = field(default_factory=dict)   # evaluator-only physics (mass, mu, mu_grip, ...)
    movable: bool = True
    graspable: bool = True
    geoms: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)     # actor-legal static info (e.g. class, sort label)


class Scene:
    def __init__(self, rng: np.random.Generator, timestep=0.002, name="station"):
        self.rng = rng
        self.spec = mujoco.MjSpec()
        self.spec.modelname = name
        self.spec.compiler.degree = False      # joint ranges in radians
        o = self.spec.option
        o.timestep = timestep
        o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        o.impratio = 5.0
        o.noslip_iterations = 0
        o.iterations = 50
        self.spec.visual.global_.offwidth = 960
        self.spec.visual.global_.offheight = 720
        self.entities: dict[str, EntityInfo] = {}
        self.regions: dict[str, dict] = {}       # actor-legal CAD regions (goal areas, pockets, holes)
        self.mounts: dict[str, list] = {}       # robot kind -> list of (pos, yaw)
        self.camera = dict(pos=(0.0, -1.2, 2.0), lookat=(0.0, 0.0, BENCH_H))
        self.grip_pairs_objects: list[str] = []  # entity ids whose geoms get explicit gripper pairs
        self._n = 0
        self.actuated: dict = {}                # env-owned actuators (conveyor) name -> info
        self.overrides: dict = {}               # evaluator-set hidden physics (sensitivity studies): mass, mu, ...

    # ------------------------------------------------------------------ helpers
    def _name(self, base):
        self._n += 1
        return f"{base}_{self._n}"

    def static_box(self, pos, half, rgba=GREY, yaw=0.0, mu=FIX_MU, body=None, name=None, contact=True, quat=None,
                   solref=None):
        parent = body if body is not None else self.spec.worldbody
        g = parent.add_geom(type=G.mjGEOM_BOX, size=list(half), pos=list(pos),
                            quat=list(quat if quat is not None else quat_z(yaw)),
                            rgba=list(rgba), friction=[mu, 0.005, 0.0001], name=name or self._name("fx"))
        if solref is not None:
            g.solref = list(solref)
        if not contact:
            g.contype = 0
            g.conaffinity = 0
        return g

    def floor(self):
        s = self.spec
        s.worldbody.add_light(pos=[0, -1, 3], dir=[0, 0.3, -1], diffuse=[0.8, 0.8, 0.8], castshadow=False)
        s.worldbody.add_light(pos=[1, 1, 3], dir=[-0.2, -0.2, -1], diffuse=[0.4, 0.4, 0.4], castshadow=False)
        g = s.worldbody.add_geom(type=G.mjGEOM_PLANE, size=[4, 4, 0.1], rgba=[0.35, 0.36, 0.38, 1],
                                 friction=[0.6, 0.005, 0.0001], name="floor")
        return g

    def bench(self, center, half_xy, top=BENCH_H, mu=FIX_MU, rgba=(0.55, 0.57, 0.6, 1), name=None):
        cx, cy = center
        hz = 0.02
        self.static_box((cx, cy, top - hz), (half_xy[0], half_xy[1], hz), rgba=rgba, mu=mu, name=name)
        for sx in (-1, 1):  # legs (visual + contact so things falling hit them)
            for sy in (-1, 1):
                self.static_box((cx + sx * (half_xy[0] - 0.03), cy + sy * (half_xy[1] - 0.03), (top - 2 * hz) / 2),
                                (0.025, 0.025, (top - 2 * hz) / 2), rgba=DARK)
        return top

    def open_box(self, center, inner_half, wall_h, t=0.008, base_z=None, rgba=BLUE_TOTE, mu=FIX_MU,
                 body=None, yaw=0.0, floor=True, prefix="bin"):
        """Static open-top box (bin / tote / pocket). Returns its inner floor height (world z)."""
        cx, cy, z0 = center[0], center[1], center[2] if base_z is None else base_z
        hx, hy = inner_half
        c, s = math.cos(yaw), math.sin(yaw)

        def P(x, y, z):
            return (cx + c * x - s * y, cy + s * x + c * y, z)
        if floor:
            self.static_box(P(0, 0, z0 + t / 2), (hx + t, hy + t, t / 2), rgba=rgba, mu=mu, yaw=yaw, body=body,
                            name=self._name(prefix + "_floor"))
        zc = z0 + t + wall_h / 2
        self.static_box(P(hx + t / 2, 0, zc), (t / 2, hy + t, wall_h / 2), rgba=rgba, mu=mu, yaw=yaw, body=body)
        self.static_box(P(-hx - t / 2, 0, zc), (t / 2, hy + t, wall_h / 2), rgba=rgba, mu=mu, yaw=yaw, body=body)
        self.static_box(P(0, hy + t / 2, zc), (hx, t / 2, wall_h / 2), rgba=rgba, mu=mu, yaw=yaw, body=body)
        self.static_box(P(0, -hy - t / 2, zc), (hx, t / 2, wall_h / 2), rgba=rgba, mu=mu, yaw=yaw, body=body)
        return z0 + (t if floor else 0.0)

    def marker(self, center, half_xy, z, rgba=(0.1, 0.8, 0.2, 0.35)):
        """Visual-only floor/bench marking (goal region)."""
        return self.static_box((center[0], center[1], z + 0.0005), (half_xy[0], half_xy[1], 0.0005), rgba=rgba,
                               contact=False)

    def region(self, name, center, half, yaw=0.0, **meta):
        self.regions[name] = dict(center=tuple(map(float, center)), half=tuple(map(float, half)), yaw=float(yaw),
                                  **meta)

    # ------------------------------------------------------------------ objects
    def random_mesh_vertices(self, half):
        n = int(self.rng.integers(10, 18))
        v = self.rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v *= self.rng.uniform(0.75, 1.0, size=(n, 1))
        v *= np.asarray(half)[None, :]
        return v

    def part(self, eid, shape, half, pos, yaw=0.0, quat=None, mass=0.1, mu=0.5, mu_grip=0.8, restitution=0.1,
             timeconst=0.02, rgba=None, kind="part", graspable=True, meta=None, extra_hidden=None, condim=4):
        """A free rigid part. `half` = bounding half extents (box semantics) for every shape."""
        ov = self.overrides
        mass, mu, mu_grip = ov.get("mass", mass), ov.get("mu", mu), ov.get("mu_grip", mu_grip)
        restitution, timeconst = ov.get("restitution", restitution), ov.get("timeconst", timeconst)
        b = self.spec.worldbody.add_body(name=eid, pos=list(pos),
                                         quat=list(quat if quat is not None else quat_z(yaw)))
        b.add_freejoint(name=eid + "_free")
        hx, hy, hz = half
        dr = restitution_to_dampratio(restitution)
        common = dict(friction=[mu, 0.005, 0.0001], solref=[timeconst, dr], condim=condim,
                      rgba=list(rgba if rgba is not None else PALETTE[len(self.entities) % len(PALETTE)]))
        geoms = []
        if shape == "box":
            g = b.add_geom(type=G.mjGEOM_BOX, size=[hx, hy, hz], **common)
        elif shape == "cylinder":
            g = b.add_geom(type=G.mjGEOM_CYLINDER, size=[hx, hz, 0], **common)
        elif shape == "capsule":   # axis along x, lying
            r = min(hy, hz)
            g = b.add_geom(type=G.mjGEOM_CAPSULE, size=[r, max(hx - r, 1e-3), 0], quat=[0.7071068, 0, 0.7071068, 0],
                           **common)
        elif shape == "sphere":
            g = b.add_geom(type=G.mjGEOM_SPHERE, size=[hx, 0, 0], **common)
        elif shape == "ellipsoid":
            g = b.add_geom(type=G.mjGEOM_ELLIPSOID, size=[hx, hy, hz], **common)
        elif shape == "mesh":
            mname = eid + "_mesh"
            self.spec.add_mesh(name=mname, uservert=self.random_mesh_vertices(half).ravel().tolist())
            g = b.add_geom(type=G.mjGEOM_MESH, meshname=mname, **common)
        else:
            raise ValueError(shape)
        g.mass = float(mass)
        g.name = eid + "_g0"
        geoms.append(g.name)
        hidden = dict(mass=float(mass), mu=float(mu), mu_grip=float(mu_grip), restitution=float(restitution),
                      timeconst=float(timeconst))
        if extra_hidden:
            hidden.update(extra_hidden)
        info = EntityInfo(eid, kind, EntityType.OBJECT, eid, shape, tuple(map(float, half)), hidden,
                          graspable=graspable, geoms=geoms, meta=dict(meta or {}))
        self.entities[eid] = info
        if graspable:
            self.grip_pairs_objects.append(eid)
        return b, info

    def compound(self, eid, boxes, pos, yaw=0.0, mass=1.0, mu=0.5, mu_grip=0.8, restitution=0.05, timeconst=0.02,
                 rgba=CARDBOARD, kind="part", half=None, graspable=True, meta=None, extra_hidden=None,
                 mass_split=None, quat=None):
        """Free body made of several boxes [(pos, half)] (tote, peg+flange, tool...). Mass spread by volume
        unless mass_split (list of masses) is given (lets us hide a heavy ballast in one piece)."""
        ov = self.overrides
        mu, mu_grip = ov.get("mu", mu), ov.get("mu_grip", mu_grip)
        b = self.spec.worldbody.add_body(name=eid, pos=list(pos), quat=list(quat if quat is not None else quat_z(yaw)))
        b.add_freejoint(name=eid + "_free")
        dr = restitution_to_dampratio(restitution)
        # recentre so the body origin is the bounding-box centre (tracker + SIZE semantics)
        lo = np.min([np.subtract(p, h) for p, h in boxes], 0)
        hi = np.max([np.add(p, h) for p, h in boxes], 0)
        shift = (lo + hi) / 2
        boxes = [(tuple(np.subtract(p, shift)), h) for p, h in boxes]
        vols = [8 * h[0] * h[1] * h[2] for _, h in boxes]
        geoms = []
        for i, (p, h) in enumerate(boxes):
            g = b.add_geom(type=G.mjGEOM_BOX, size=list(h), pos=list(p), friction=[mu, 0.005, 0.0001],
                           solref=[timeconst, dr], condim=4, rgba=list(rgba), name=f"{eid}_g{i}")
            g.mass = float(mass_split[i]) if mass_split is not None else float(mass * vols[i] / sum(vols))
            geoms.append(g.name)
        if half is None:
            lo = np.min([np.subtract(p, h) for p, h in boxes], 0)
            hi = np.max([np.add(p, h) for p, h in boxes], 0)
            half = tuple(((hi - lo) / 2).tolist())
        hidden = dict(mass=float(mass if mass_split is None else sum(mass_split)), mu=float(mu), mu_grip=float(mu_grip),
                      restitution=float(restitution), timeconst=float(timeconst))
        if extra_hidden:
            hidden.update(extra_hidden)
        info = EntityInfo(eid, kind, EntityType.OBJECT, eid, kind, tuple(map(float, half)), hidden,
                          graspable=graspable, geoms=geoms, meta=dict(meta or {}))
        self.entities[eid] = info
        if graspable:
            self.grip_pairs_objects.append(eid)
        return b, info

    # ------------------------------------------------------------------ stations
    def conveyor(self, start_xy, length, width, top_z, speed, travel=6.0, name="belt"):
        """DRIVEN belt: a long plate on a slide joint (axis +x), velocity-servoed by an actuator with
        a force limit. Visible bed spans [start, start+length]; the plate is long enough to keep moving
        for `travel` metres (one episode). Parts ride on it by friction."""
        x0, y0 = start_xy
        hz = 0.01
        plate_len = length + travel
        b = self.spec.worldbody.add_body(name=name, pos=[x0 + length / 2 - travel / 2, y0, top_z - hz])
        j = b.add_joint(name=name + "_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], damping=5.0,
                        range=[-0.1, travel + 0.5], limited=True, armature=1.0)
        g = b.add_geom(type=G.mjGEOM_BOX, size=[plate_len / 2, width / 2, hz], rgba=[0.12, 0.12, 0.12, 1],
                       friction=[FIX_MU, 0.005, 0.0001], name=name + "_plate")
        g.mass = 20.0
        a = self.spec.add_actuator(name=name + "_drive", target=name + "_slide",
                                   trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_velocity(kv=4000.0)
        a.ctrlrange = [-1, 1]
        a.forcerange = [-600, 600]
        a.forcelimited = True
        # side guards + support bed (bed friction tiny: plate slides on it)
        self.static_box((x0 + length / 2, y0, top_z - 2 * hz - 0.02), (length / 2, width / 2, 0.02), rgba=DARK)
        for s in (-1, 1):
            self.static_box((x0 + length / 2, y0 + s * (width / 2 + 0.01), top_z + 0.02), (length / 2, 0.01, 0.03),
                            rgba=(0.8, 0.7, 0.1, 1))
        self.actuated[name] = dict(kind="conveyor", speed=float(speed), joint=name + "_slide", actuator=name + "_drive")
        return b

    def hinge_door(self, eid, hinge_pos, width, height, thick, axis=(0, 0, 1), closed_yaw=0.0, damping=0.5,
                   frictionloss=1.0, stiffness=0.0, springref=0.0, mass=2.0, rng_open=(0.0, 1.9), handle_off=None,
                   handle_mu_grip=0.8):
        """Hinged machine door (hinge at one vertical edge). Door panel extends along +x of its body."""
        b = self.spec.worldbody.add_body(name=eid, pos=list(hinge_pos), quat=list(quat_z(closed_yaw)))
        b.add_joint(name=eid + "_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(axis), damping=damping,
                    frictionloss=frictionloss, stiffness=stiffness, springref=springref, range=list(rng_open),
                    limited=True, armature=0.02)
        g = b.add_geom(type=G.mjGEOM_BOX, size=[width / 2, thick / 2, height / 2], pos=[width / 2, 0, 0],
                       rgba=[0.75, 0.78, 0.8, 1], friction=[FIX_MU, 0.005, 0.0001], name=eid + "_panel")
        g.mass = mass
        hp = handle_off or (width - 0.05, -thick / 2 - 0.03, 0.0)
        # pinch-tab handle: a fin normal to the panel; the gripper pinches it across body x, so the pull
        # (along the panel normal) is carried by pad friction (capacity 2 * mu_grip * F_grip)
        h = b.add_geom(type=G.mjGEOM_BOX, size=[0.004, 0.03, 0.025], pos=list(hp), rgba=list(DARK),
                       friction=[FIX_MU, 0.005, 0.0001], name=eid + "_handle")
        h.mass = 0.1
        info = EntityInfo(eid, "door", EntityType.LINK, eid, "box", (width / 2, thick / 2, height / 2),
                          dict(mass=mass, damping=damping, frictionloss=frictionloss, stiffness=stiffness,
                               mu_grip=handle_mu_grip),
                          movable=True, graspable=True, geoms=[eid + "_handle"],
                          meta=dict(joint=eid + "_hinge", handle_local=tuple(hp), hinge_pos=tuple(hinge_pos),
                                    closed_yaw=closed_yaw, width=width))
        self.entities[eid] = info
        self.grip_pairs_objects.append(eid)
        return b, info

    def drawer(self, eid, pos, inner_half, wall_h, axis_yaw, travel, damping=5.0, frictionloss=5.0, mass=2.0,
               detent=None, handle_mu_grip=0.8):
        """Drawer on a slide joint along local -y (pulls toward the robot). Housing is static."""
        c, s = math.cos(axis_yaw), math.sin(axis_yaw)
        b = self.spec.worldbody.add_body(name=eid, pos=list(pos), quat=list(quat_z(axis_yaw)))
        b.add_joint(name=eid + "_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, -1, 0], damping=damping,
                    frictionloss=frictionloss, range=[0.0, travel], limited=True, armature=0.05)
        hx, hy = inner_half
        t = 0.008
        self.open_box((0, 0, 0), inner_half, wall_h, t=t, base_z=0.0, rgba=(0.7, 0.72, 0.75, 1), body=b,
                      prefix=eid)
        # front plate + bar handle (horizontal along local x)
        fp = b.add_geom(type=G.mjGEOM_BOX, size=[hx + 0.02, 0.008, wall_h / 2 + 0.01], pos=[0, -hy - t - 0.008, wall_h / 2 + 0.01],
                        rgba=[0.6, 0.62, 0.65, 1], friction=[FIX_MU, 0.005, 0.0001], name=eid + "_front")
        hp = (0.0, -hy - t - 0.016 - 0.03, wall_h / 2)
        hg = b.add_geom(type=G.mjGEOM_BOX, size=[0.004, 0.03, 0.025], pos=list(hp), rgba=list(DARK),
                        friction=[FIX_MU, 0.005, 0.0001], name=eid + "_handle")
        # total drawer mass: put it on the floor plate via a dummy inertial geom-free approach: scale geoms
        for g in b.geoms:
            g.mass = 0.0
        fp.mass = mass * 0.6
        hg.mass = mass * 0.05
        b.add_geom(type=G.mjGEOM_BOX, size=[hx * 0.9, hy * 0.9, 0.002], pos=[0, 0, 0.01], contype=0, conaffinity=0,
                   rgba=[0, 0, 0, 0], name=eid + "_ballast").mass = mass * 0.35
        hidden = dict(mass=mass, damping=damping, frictionloss=frictionloss, mu_grip=handle_mu_grip)
        if detent:
            hidden.update(detent=detent)
        info = EntityInfo(eid, "drawer", EntityType.LINK, eid, "box", (hx + t, hy + t, wall_h / 2), hidden,
                          movable=True, graspable=True, geoms=[eid + "_handle"],
                          meta=dict(joint=eid + "_slide", handle_local=hp, axis_yaw=axis_yaw, travel=travel, pos=tuple(pos),
                                    pull_dir=(s, -c)))
        self.entities[eid] = info
        self.grip_pairs_objects.append(eid)
        # housing (static): side rails + back stop
        X = lambda x, y: (pos[0] + c * x - s * y, pos[1] + s * x + c * y)
        for sx in (-1, 1):
            p = X(sx * (hx + t + 0.006), 0.025)
            self.static_box((p[0], p[1], pos[2] + wall_h / 2), (0.004, hy + 0.025, wall_h / 2 + 0.03), yaw=axis_yaw,
                            rgba=DARK)
        pbk = X(0, hy + t + 0.012)
        self.static_box((pbk[0], pbk[1], pos[2] + wall_h / 2), (hx + 0.02, 0.004, wall_h / 2 + 0.03), yaw=axis_yaw,
                        rgba=DARK)
        return b, info

    def knob(self, eid, pos, radius=0.012, height=0.025, frictionloss=0.2, damping=0.02, mass=0.1, rng_rot=(0.0, 3.2),
             mu_grip=0.8):
        """Twist-lock knob: a cylinder on a vertical hinge (quarter-turn bayonet). Breakaway torque = frictionloss."""
        b = self.spec.worldbody.add_body(name=eid, pos=list(pos))
        b.add_joint(name=eid + "_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=damping,
                    frictionloss=frictionloss, range=list(rng_rot), limited=True, armature=0.002)
        g = b.add_geom(type=G.mjGEOM_CYLINDER, size=[radius, height / 2, 0], pos=[0, 0, height / 2],
                       rgba=[0.85, 0.2, 0.2, 1], friction=[FIX_MU, 0.005, 0.0001], name=eid + "_handle")
        g.mass = mass
        b.add_geom(type=G.mjGEOM_BOX, size=[radius * 0.9, 0.002, 0.001], pos=[0, 0, height + 0.001], rgba=[1, 1, 1, 1],
                   contype=0, conaffinity=0, name=eid + "_mark")
        info = EntityInfo(eid, "knob", EntityType.LINK, eid, "cylinder", (radius, radius, height / 2),
                          dict(frictionloss=frictionloss, damping=damping, mass=mass, mu_grip=mu_grip), movable=True,
                          graspable=True, geoms=[eid + "_handle"],
                          meta=dict(joint=eid + "_hinge", pos=tuple(pos), height=height, radius=radius))
        self.entities[eid] = info
        self.grip_pairs_objects.append(eid)
        return b, info

    def flex_pouch(self, eid, center, size, count=(4, 6, 3), mass=0.2, young=3e4, poisson=0.3, damping=0.002,
                   mu=0.6, radius=0.004):
        """Soft pouch: MuJoCo 3-D flexcomp grid (finite elements). Attached into the world."""
        sp = [size[k] / (count[k] - 1) for k in range(3)]
        xml = (f'<mujoco><worldbody><flexcomp name="{eid}" type="grid" count="{count[0]} {count[1]} {count[2]}" '
               f'spacing="{sp[0]} {sp[1]} {sp[2]}" pos="{center[0]} {center[1]} {center[2]}" dim="3" '
               f'radius="{radius}" mass="{mass}" rgba="0.9 0.92 0.95 1">'
               f'<elasticity young="{young}" poisson="{poisson}" damping="{damping}"/>'
               f'<contact condim="3" selfcollide="none" friction="{mu} 0.005 0.0001" solref="0.01 1"/>'
               f'<edge equality="false"/></flexcomp></worldbody></mujoco>')
        self.spec.attach(mujoco.MjSpec.from_string(xml), prefix="", frame=self.spec.worldbody.add_frame())
        info = EntityInfo(eid, "pouch", EntityType.OBJECT, eid, "box", tuple(s / 2 for s in size),
                          dict(mass=mass, young=young, poisson=poisson, mu=mu), graspable=True,
                          meta=dict(flex=True))
        self.entities[eid] = info
        return info


    def socket(self, center, hole_half, depth, outer=0.03, chamfer=0.0, yaw=0.0, rgba=(0.45, 0.47, 0.5, 1),
               solref=(0.004, 1.0), name="socket"):
        """Static block with a rectangular through-hole (hole_half = (hx, hy)) of `depth`, top face at center z.
        Chamfer: 45-degree bevel of width `chamfer` on the four mouth edges. Returns mouth (top) height."""
        cx, cy, ztop = center
        hx, hy = hole_half
        c, s = math.cos(yaw), math.sin(yaw)

        def P(x, y, z):
            return (cx + c * x - s * y, cy + s * x + c * y, z)
        zc = ztop - depth / 2
        ch = max(chamfer, 0.0)
        wall_top = ztop - ch
        hd = (depth - ch) / 2
        zc = ztop - ch - hd
        # four walls (full height below the chamfer)
        self.static_box(P(hx + outer / 2, 0, zc), (outer / 2, hy + outer, hd), yaw=yaw, rgba=rgba, solref=solref,
                        name=self._name(name + "_w"))
        self.static_box(P(-hx - outer / 2, 0, zc), (outer / 2, hy + outer, hd), yaw=yaw, rgba=rgba, solref=solref,
                        name=self._name(name + "_w"))
        self.static_box(P(0, hy + outer / 2, zc), (hx, outer / 2, hd), yaw=yaw, rgba=rgba, solref=solref,
                        name=self._name(name + "_w"))
        self.static_box(P(0, -hy - outer / 2, zc), (hx, outer / 2, hd), yaw=yaw, rgba=rgba, solref=solref,
                        name=self._name(name + "_w"))
        if ch > 1e-4:
            # bevel strips: boxes rotated 45 deg about the edge axis, their inner face forming the chamfer
            t = ch * 0.7071068 * 2   # thickness so that the rotated box covers the bevel
            for sx, axis in ((1, "x"), (-1, "x"), (1, "y"), (-1, "y")):
                if axis == "x":    # walls at +-hx: edge along y, rotate about y
                    ang = sx * math.pi / 4
                    q = np.array([math.cos(ang / 2), 0, math.sin(ang / 2), 0])
                    q = _qmul(quat_z(yaw), q)
                    off = hx + ch / 2 + t / 2 * 0.7071068
                    self.static_box(P(sx * off, 0, ztop - ch / 2 - t / 2 * 0.7071068), (t / 2, hy + ch, t / 2),
                                    quat=q, rgba=rgba, solref=solref, name=self._name(name + "_ch"))
                else:
                    ang = -sx * math.pi / 4
                    q = np.array([math.cos(ang / 2), math.sin(ang / 2), 0, 0])
                    q = _qmul(quat_z(yaw), q)
                    off = hy + ch / 2 + t / 2 * 0.7071068
                    self.static_box(P(0, sx * off, ztop - ch / 2 - t / 2 * 0.7071068), (hx + ch, t / 2, t / 2),
                                    quat=q, rgba=rgba, solref=solref, name=self._name(name + "_ch"))
            # outer top ring around the bevel
            for sx in (-1, 1):
                self.static_box(P(sx * (hx + ch + (outer - ch) / 2), 0, ztop - ch / 2), ((outer - ch) / 2, hy + outer, ch / 2),
                                yaw=yaw, rgba=rgba, solref=solref, name=self._name(name + "_r"))
                self.static_box(P(0, sx * (hy + ch + (outer - ch) / 2), ztop - ch / 2), (hx + ch, (outer - ch) / 2, ch / 2),
                                yaw=yaw, rgba=rgba, solref=solref, name=self._name(name + "_r"))
        return ztop


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rotate_station(scene: "Scene", alpha: float, center=(0.0, 0.0)):
    """Held-out layout transform: rotate every top-level world body / static geom, the CAD regions and the
    CAD meta of articulated entities by `alpha` about the vertical axis through `center`. A proper rotation,
    so joint axes (in body frames) and handedness are preserved. Robot mounts and the tracker camera stay."""
    import math as _m
    c, s_ = _m.cos(alpha), _m.sin(alpha)
    cx, cy = center
    qa = quat_z(alpha)

    def R(x, y):
        dx, dy = x - cx, y - cy
        return cx + c * dx - s_ * dy, cy + s_ * dx + c * dy
    sp = scene.spec
    for b in list(sp.worldbody.bodies):
        x, y = R(b.pos[0], b.pos[1])
        b.pos = [x, y, b.pos[2]]
        b.quat = list(_qmul(qa, np.asarray(b.quat)))
    for g in list(sp.worldbody.geoms):
        if g.type == G.mjGEOM_PLANE:
            continue
        x, y = R(g.pos[0], g.pos[1])
        g.pos = [x, y, g.pos[2]]
        g.quat = list(_qmul(qa, np.asarray(g.quat)))
    for name, reg in scene.regions.items():
        x, y = R(reg["center"][0], reg["center"][1])
        reg["center"] = (x, y, reg["center"][2])
        reg["yaw"] = reg.get("yaw", 0.0) + alpha
    for info in scene.entities.values():
        m = info.meta
        if "hinge_pos" in m:
            x, y = R(m["hinge_pos"][0], m["hinge_pos"][1])
            m["hinge_pos"] = (x, y, m["hinge_pos"][2])
            m["closed_yaw"] = m["closed_yaw"] + alpha
        if "axis_yaw" in m or info.kind == "knob":
            x, y = R(m["pos"][0], m["pos"][1])
            m["pos"] = (x, y, m["pos"][2])
            if "axis_yaw" in m:
                m["axis_yaw"] = m["axis_yaw"] + alpha


# ---------------------------------------------------------------------- randomization helpers
def sample_part_params(rng, half_range=(0.012, 0.035), shapes=("box", "cylinder", "capsule", "sphere", "mesh"),
                       density_range=(400, 2500), heavy_p=0.1, heavy_mult=(4, 8), mu_range=(0.2, 0.9),
                       mu_grip_range=(0.3, 1.0), oily_p=0.1, restitution_range=(0.0, 0.4), soft_p=0.2):
    shape = shapes[int(rng.integers(len(shapes)))]
    h = rng.uniform(*half_range, size=3)
    if shape in ("cylinder",):
        h[1] = h[0]
    if shape == "sphere":
        h[:] = h[0]
    if shape == "capsule":
        h[0] = max(h[0], h[1] * 1.6)
        h[2] = h[1]
    vol = 8 * h.prod() * (0.6 if shape != "box" else 1.0)
    mass = vol * rng.uniform(*density_range)
    heavy = rng.random() < heavy_p
    if heavy:
        mass *= rng.uniform(*heavy_mult)
    mu_grip = rng.uniform(*mu_grip_range)
    oily = rng.random() < oily_p
    if oily:
        mu_grip = rng.uniform(0.08, 0.2)
    soft = rng.random() < soft_p
    return dict(shape=shape, half=tuple(h.tolist()), mass=float(max(mass, 0.005)), mu=float(rng.uniform(*mu_range)),
                mu_grip=float(mu_grip), restitution=float(rng.uniform(*restitution_range)),
                timeconst=float(rng.uniform(0.02, 0.04) if soft else 0.01), heavy=bool(heavy), oily=bool(oily))
