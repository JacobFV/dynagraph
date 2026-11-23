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
        scene.knob("knob", (x, y, top + base_h), frictionloss=fl, damping=dmp, mu_grip=mug)
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
