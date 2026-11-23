"""Additional families (round 2): twist_lock (fine, torque-limited) and machine_tending (multi-step CNC-style cycle)."""
from __future__ import annotations

import math

import numpy as np

from ..v2.api import SkillCall
from .families_fine import DoorOpen, _FineBase
from .stations import bench_station


# ====================================================================== twist-lock (quarter-turn bayonet knob)
class TwistLock(_FineBase):
    family = "twist_lock"
    station = "assembly_fixture"
    timestep = 0.002
    success_doc = "the twist-lock knob is turned by >= 90 deg (quarter-turn bayonet locked) and released"
    hidden_doc = ("breakaway torque (hinge frictionloss) 0.03-0.6 N m, damping, knob<->gripper friction 0.3-1.0: the pinch "
                  "transmits at most ~2 mu_g F r (r = 12 mm) of torque -> grip force / twist rate; too little slips")

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.2
        fl = float(params.get("frictionloss", np.exp(rng.uniform(np.log(0.03), np.log(0.6)))))
        mug = float(params.get("mu_grip", rng.uniform(0.3, 1.0)))
        dmp = float(rng.uniform(0.005, 0.05))
        x, y = rng.uniform(-0.15, 0.15), rng.uniform(-0.05, 0.12)
        base_h = 0.03
        scene.static_box((x, y, top + base_h / 2), (0.04, 0.04, base_h / 2), rgba=(0.35, 0.37, 0.4, 1))
        _, kinfo = scene.knob("knob", (x, y, top + base_h), frictionloss=fl, damping=dmp, mu_grip=mug)
        from .scene import fragility
        kinfo.hidden["crush_n"] = fragility(rng, 0.6, 20, 70, params)      # plastic knob: too hard a pinch cracks it
        self.hidden_task = dict(frictionloss=fl, mu_grip=mug, damping=dmp)
        self.target = math.pi / 2
        self.budget = int(params.get("budget", 5))

    def q(self):
        return float(self.d.qpos[self.m.jnt_qposadr[self.m.joint("knob_hinge").id]])

    def success(self):
        return self.q() >= self.target and self.held is None

    def progress(self):
        return float(np.clip(self.q() / self.target, 0, 1))

    def goal_extra(self):
        return dict(joint_goal={"knob": dict(min=self.target, unit="rad")},
                    cad={k: v for k, v in self.scene.entities["knob"].meta.items()})

    def _twist(self, rng=None, **kw):
        q = self.est_joint("knob") or 0.0
        p = dict(grip_force=40.0, travel=float(max(0.05, self.target + 0.15 - q)), speed=0.1, yaw_rate=1.0)
        if rng is not None:
            p.update(grip_force=float(rng.uniform(5, 100)), yaw_rate=float(rng.uniform(0.3, 3.0)),
                     travel=float(rng.uniform(0.3, 1.9)))
        p.update(kw)
        return SkillCall("open", p, target="knob")

    def candidates(self, n, rng):
        return [self._twist(rng) for _ in range(n)]

    def scripted(self):
        k = self.fail_count.get("knob", 0)
        return self._twist(grip_force=(40.0, 80.0, 100.0)[min(k, 2)], yaw_rate=(1.0, 0.5, 0.3)[min(k, 2)])

    def labels_extra(self):
        return dict(task_hidden=self.hidden_task, joint=self.q())


# ====================================================================== machine tending: door + part-in-vise + door
class MachineTending(DoorOpen):
    family = "machine_tending"
    station = "machine_tending"
    timestep = 0.002
    success_doc = ("CNC-style cycle: the blank sits >= 80 % deep in the machine's vise socket AND the machine door is "
                   "closed (<= 0.08 rad) AND released")
    hidden_doc = ("door hinge frictionloss / damping / mass, blank mass + blank<->gripper friction, vise clearance 1-3 mm: "
                  "door grip force and swing, pick force, compliant insert search")

    def build(self, scene, rng, params):
        super().build(scene, rng, params)
        door = self.scene.entities["door"]
        hx, hy, hz = door.meta["hinge_pos"]
        width = door.meta["width"]
        top = self.surface_z
        # vise socket inside the machine (behind the closed door plane)
        clr = float(params.get("clearance", rng.uniform(0.001, 0.003)))
        s = float(rng.uniform(0.012, 0.016))
        vx, vy = hx - width / 2 + rng.uniform(-0.06, 0.06), hy + 0.11
        depth = 0.025
        scene.socket((vx, vy, top + depth), (s + clr / 2, s + clr / 2), depth, outer=0.02, name="vise")
        scene.region("vise", (vx, vy, top + depth / 2), (s + clr / 2, s + clr / 2, depth / 2), support_z=top + depth,
                     depth=depth, sym=math.pi / 2)
        # raw blank on the bench in front of the machine
        bx, by = rng.uniform(-0.3, -0.05), rng.uniform(-0.25, -0.1)
        mass = float(rng.uniform(0.1, 0.8))
        scene.part("blank", "box", (s, s, 0.035), (bx, by, top + 0.036), yaw=rng.uniform(-0.5, 0.5), mass=mass,
                   mu=float(rng.uniform(0.3, 0.8)), mu_grip=float(rng.uniform(0.3, 1.0)), timeconst=0.01, kind="blank",
                   rgba=(0.75, 0.75, 0.8, 1))
        self.vise_top, self.vise_depth = top + depth, depth
        self.hidden_task.update(clearance=clr, blank_mass=mass)
        self.budget = int(params.get("budget", 12))

    def depth_frac(self):
        if not self.in_region_true("blank", self.scene.regions["vise"], margin=0.002, zcheck=False):
            return 0.0
        p, _ = self.true_pose("blank")
        bottom = p[2] - self.scene.entities["blank"].half[2]
        return float(np.clip((self.vise_top - bottom) / self.vise_depth, 0, 1.5))

    def success(self):
        return self.depth_frac() >= 0.8 and self.q() <= 0.08 and self.held is None

    def progress(self):
        # 3 stages: door open (or part already in) / part seated / door closed
        seated = self.depth_frac() >= 0.8
        opened = self.q() >= 1.0
        return float((1 / 3) * (opened or seated) + (1 / 3) * seated + (1 / 3) * (seated and self.q() <= 0.08))

    def goal_extra(self):
        g = super().goal_extra()
        g["sequence"] = ["open door", "load blank into vise", "close door"]
        return g

    def _est_seated(self):
        est = self.est_pose("blank")
        if est is None:
            return False
        reg = self.scene.regions["vise"]
        h = self.est_size("blank")
        return (abs(est[0][0] - reg["center"][0]) < 0.004 and abs(est[0][1] - reg["center"][1]) < 0.004
                and est[0][2] - h[2] <= self.vise_top - 0.8 * self.vise_depth + 0.003)

    def candidates(self, n, rng):
        q = self.est_joint("door") or 0.0
        if self.held is not None:
            return [self._insert_call(rng) for _ in range(n)]
        if not self._est_seated() and q < 1.0:
            return [self._open(rng) for _ in range(n)]
        if not self._est_seated():
            return [self._pick("blank", rng, depth=float(rng.uniform(-0.4, 0.0))) for _ in range(n)]
        return [SkillCall("close", dict(grip_force=float(rng.uniform(5, 100)), travel=float(q + rng.uniform(0, 0.2)),
                                        speed=float(rng.uniform(0.05, 0.4))), target="door") for _ in range(n)]

    def _insert_call(self, rng=None, **kw):
        p = dict(search_r=0.002, search_hz=1.0, down_force=6.0, compliance=0.4, speed=0.3, timeout=5.0, correct=1)
        if rng is not None:
            p.update(search_r=float(rng.uniform(0.0, 0.004)), down_force=float(rng.uniform(2, 20)))
        p.update(kw)
        return SkillCall("insert", p, target="vise")

    def scripted(self):
        q = self.est_joint("door") or 0.0
        # the tracker often cannot see into the machine: also trust the robot's own successful insert (actor memory)
        lc, lr = getattr(self, "last_call", None), getattr(self, "last_result", None)
        if lc is not None and lc.target == "vise" and lc.skill == "insert":
            self._seated_belief = bool(lr.ok)
        if lc is not None and lc.skill == "pick" and lc.target == "blank" and lr.ok:
            self._seated_belief = False
        seated = self._est_seated() or getattr(self, "_seated_belief", False)
        if self.held is not None:
            k = self.fail_count.get("vise", 0)
            return self._insert_call(search_r=0.002 * (1 + k), release_on_fail=0 if k < 3 else 1)
        if not seated and q < 1.0:
            k = self.fail_count.get("door", 0)
            return self._open(travel=float(np.clip(1.45 - q, 0.1, 1.6)), grip_force=(30.0, 60.0, 90.0)[min(k, 2)])
        if not seated:
            k = self.fail_count.get("blank", 0)
            return self._pick("blank", grip_force=(30.0, 60.0, 90.0)[min(k, 2)], depth=-0.3)
        if q > 0.05:
            k = self.fail_count.get("door", 0)
            return SkillCall("close", dict(grip_force=(30.0, 60.0, 90.0)[min(k, 2)], travel=float(q + 0.05), speed=0.15),
                             target="door")
        return None


# ====================================================================== light-parts station (SO-100-class arms)
# Everything within ~0.25 m of the `arm` mount at the near bench edge (0, -0.45, bench top); parts <= 30 mm, <= 80 g.
from .families_bulk import BinClearing                      # noqa: E402
from .families_fine import PegInsert                        # noqa: E402
from .scene import fragility, sample_part_params            # noqa: E402

LIGHT_ZONE_Y = -0.24


class SmallPartsPick(BinClearing):
    family = "small_parts_pick"
    category = "bulk"
    station = "light_parts"
    probe_kw = dict(max_open=0.03, grip_max=30.0)
    success_doc = "every tiny part (<= 30 mm, <= 80 g) from the small tray ends inside the target cup"
    hidden_doc = "part mass (<= 80 g, dense metal outliers), part<->gripper friction (oily), compliance: grip force / lift speed"

    def build(self, scene, rng, params):
        from .stations import bench_station
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.12
        n = int(params.get("n_parts", rng.integers(2, 5)))
        sx, sy = -0.08 + rng.uniform(-0.01, 0.01), LIGHT_ZONE_Y + rng.uniform(-0.01, 0.01)
        tray = (0.055, 0.045)
        fz = scene.open_box((sx, sy, top), tray, 0.03, prefix="srcbin", t=0.005)
        tx, ty = 0.08 + rng.uniform(-0.01, 0.01), LIGHT_ZONE_Y + rng.uniform(-0.01, 0.01)
        cup = (0.045, 0.04)
        tfz = scene.open_box((tx, ty, top), cup, 0.03, rgba=(0.6, 0.2, 0.15, 1), prefix="tote", t=0.005)
        scene.region("source_bin", (sx, sy, fz + 0.02), (tray[0], tray[1], 0.03), support_z=fz)
        scene.region("tote", (tx, ty, tfz + 0.02), (cup[0], cup[1], 0.03), support_z=tfz)
        placed = []
        for i in range(n):
            pp = sample_part_params(rng, half_range=(0.006, 0.015),
                                    shapes=tuple(params.get("shapes", ("box", "cylinder", "capsule", "sphere"))),
                                    density_range=(1000, 8000), heavy_p=0.1, heavy_mult=(1.5, 2.5), oily_p=0.15)
            mass = float(min(pp["mass"], 0.08))
            rad = float(np.linalg.norm(pp["half"][:2])) + 0.003
            for _ in range(60):
                x = sx + rng.uniform(-tray[0] + rad, tray[0] - rad)
                y = sy + rng.uniform(-tray[1] + rad, tray[1] - rad)
                if all(np.hypot(x - a, y - b) > rad + r2 for a, b, r2 in placed):
                    break
            placed.append((x, y, rad))
            z = fz + 0.02 + 0.03 * (i // 4)
            scene.part(f"part{i:02d}", pp["shape"], pp["half"], (x, y, z), yaw=rng.uniform(-math.pi, math.pi), mass=mass,
                       mu=pp["mu"], mu_grip=pp["mu_grip"], restitution=pp["restitution"], timeconst=pp["timeconst"],
                       extra_hidden=dict(heavy=pp["heavy"], oily=pp["oily"], crush_n=fragility(rng, 0.4, 6, 20, params)))
        self.budget = int(params.get("budget", 4 * n))
        # SO-100-class robots mount at the near edge; the light zone is within ~0.25 m of that mount
        scene.camera = dict(pos=(0.0, LIGHT_ZONE_Y + 0.4, top + 0.6), lookat=(0.0, LIGHT_ZONE_Y, top),
                            view_pos=(0.35, LIGHT_ZONE_Y - 0.45, top + 0.45), view_lookat=(0.0, LIGHT_ZONE_Y, top))


def _small_place(self, rng=None):
    p = dict(dx=0.0, dy=0.0, release_h=0.02, speed=0.3)
    r = rng if rng is not None else self.rng
    p.update(dx=float(r.uniform(-0.02, 0.02)), dy=float(r.uniform(-0.015, 0.015)))
    if rng is not None:
        p.update(release_h=float(rng.uniform(0.005, 0.04)), speed=float(rng.uniform(0.1, 0.6)))
    return SkillCall("place", p, target="tote")


SmallPartsPick._place_call = _small_place


class SmallPegInsert(PegInsert):
    family = "small_peg_insert"
    station = "light_parts"
    probe_kw = dict(max_open=0.03, grip_max=30.0, pad_solref=0.003)
    hole_center = (0.07, LIGHT_ZONE_Y)
    peg_center = (-0.07, LIGHT_ZONE_Y)
    layout_jitter = 0.015
    mass_range = (0.01, 0.06)
    default_cycle_limit = 14.0
    success_doc = "a 10-14 mm square peg (<= 60 g) seated >= 80 % into its 12 mm deep hole, released"
    hidden_doc = "clearance 0.3-1.2 mm, chamfer 0-1 mm, friction 0.2-0.9: straight vs search insertion"

    def geometry(self, rng, params):
        sh = float(params.get("peg_half", rng.uniform(0.005, 0.007)))
        return dict(half=(sh, sh, 0.02), sym=math.pi / 2, depth=0.012)

    def build(self, scene, rng, params):
        params = dict(params)
        params.setdefault("clearance", float(rng.uniform(0.0003, 0.0012)))
        params.setdefault("chamfer", 0.0 if rng.random() < 0.4 else float(rng.uniform(0.0003, 0.001)))
        super().build(scene, rng, params)
        self.safe_z = self.surface_z + 0.12
