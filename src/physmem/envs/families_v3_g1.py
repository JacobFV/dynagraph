"""envs-v3 legged contact (agent ENVS-V3): Unitree G1 + Dex3 pushing a loaded box while STANDING on a floor patch with
hidden friction and compliance.

g1_push_terrain  The floating-base G1 (ROBOTS' whole-body QP: balance + arm task, no walking) stands at the bench and
                 pushes a box 8 cm away from itself. The reaction of the push goes through the feet: the floor under the
                 feet has a hidden Coulomb friction 0.08-0.9 (oily steel plate ... rubber mat) and a hidden contact
                 compliance (solref time constant 5-30 ms: hard floor ... soft mat; 2 ms was tried and made the QP-balanced G1 chatter and fall even on high friction - excluded). The box needs a hidden push force
                 (mass 1-4 kg x box<->bench friction 0.3-0.5 -> 3-20 N; heavier boxes topple the 35 kg robot). Decisions: the hand force cap (below it the
                 hand yields, so the box does not move if the cap is under the needed force; above the feet's friction
                 capacity the feet slide), the push speed, and the friction coefficient the balance QP assumes for its
                 foot-force cone.
                 FIXED-BASE G1 (g1_dex3_fixed, pelvis welded) is run as a CONTROL: the floor is irrelevant to it by
                 construction (it cannot slip or fall); it isolates the arm/push part of the task.
Honest scope: standing balance only (ROBOTS' G1 has no walking controller in v2); the feet's contact is MuJoCo's soft
point contact on the ankle collision spheres, not a foot sole model; slip is measured from TRUE foot positions
(evaluator-only).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..v2.api import SkillCall
from .families_v3_common import V3Base
from .stations import bench_station

G9 = 9.81


class G1PushTerrain(V3Base):
    family = "g1_push_terrain"
    station = "g1_terrain_bench"
    timestep = 0.002
    settle_s = 1.0
    budget = 4
    SLIP_LIMIT = 0.02
    success_doc = ("push the box >= 6 cm away (+y) along the bench; the robot stays up (pelvis no more than 25 cm below "
                   "its stand height) and neither foot slides more than 2 cm")
    hidden_doc = ("floor friction 0.08-0.9 and compliance (solref 5-30 ms) under the feet; box mass 1-4 kg and box<->bench "
                  "friction 0.3-0.5 (push force 3-20 N): the hand force cap must exceed the push force but the feet's "
                  "friction capacity bounds it")

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.30
        mu_f = float(params.get("mu_floor", math.exp(rng.uniform(math.log(0.08), math.log(0.9)))))
        tc = float(params.get("floor_tc", math.exp(rng.uniform(math.log(0.005), math.log(0.03)))))
        m_b = float(params.get("box_mass", rng.uniform(1.0, 4.0)))
        mu_b = float(params.get("box_mu", rng.uniform(0.3, 0.5)))
        # the box: in front of the robot, near the bench edge (G1's short arms)
        edge = -0.4
        bx, by = 0.0 + rng.uniform(-0.03, 0.03), edge + 0.16 + rng.uniform(-0.02, 0.02)
        half = (0.06, 0.05, 0.05)
        _, info = scene.part("box", "box", half, (bx, by, top + half[2] + 0.001), yaw=0.0, mass=m_b, mu=mu_b, mu_grip=0.8,
                             restitution=0.05, timeconst=0.01, kind="box", rgba=(0.75, 0.55, 0.3, 1))
        scene.region("push_goal", (bx, by + 0.08, top + half[2]), (half[0], 0.04, half[2]), support_z=top)
        # floor patch under the robot (visual; the physics are the explicit foot<->floor pairs added with the robot)
        scene.static_box((0.0, edge - 0.3, 0.0005), (0.4, 0.3, 0.0005), rgba=(0.25, 0.3, 0.35, 1), contact=False)
        self.hidden_task = dict(mu_floor=mu_f, floor_tc=tc, box_mass=m_b, box_mu=mu_b, push_n=mu_b * m_b * G9)
        self.box0 = None
        self.budget = int(params.get("budget", 4))

    def _make_robot(self, name):
        r, sk = super()._make_robot(name)
        orig = r.build
        env = self

        def build(scene, mount):
            orig(scene, mount)
            env._feet_pairs(scene)
        r.build = build
        return r, sk

    def _feet_pairs(self, scene):
        """explicit foot<->floor contact pairs with the hidden friction / compliance; the default (contype) contacts of
        the feet with the world are excluded so the pair is the only foot<->floor contact."""
        h = self.hidden_task
        s = scene.spec
        for side in ("left", "right"):
            b = None
            for bb in s.bodies:
                if bb.name.endswith(f"{side}_ankle_roll_link"):
                    b = bb
            if b is None:
                continue
            for k, g in enumerate(b.geoms):
                if g.type != mujoco.mjtGeom.mjGEOM_SPHERE or (g.contype == 0 and g.conaffinity == 0):
                    continue
                if not g.name:
                    g.name = f"{b.name}_foot{k}"
                s.add_pair(geomname1=g.name, geomname2="floor", condim=3,
                           friction=[h["mu_floor"], h["mu_floor"], 0.005, 0.0001, 0.0001], solref=[h["floor_tc"], 1.0])
            s.add_exclude(bodyname1="world", bodyname2=b.name)
        self._has_feet = True

    def _index(self):
        super()._index()
        m = self.m
        self._feet_b = [i for i in range(m.nbody) if m.body(i).name.endswith("ankle_roll_link")]
        self._feet0 = None
        self._pelvis = next((i for i in range(m.nbody) if m.body(i).name.endswith("pelvis")), None)

    def post_physics(self, stop=None):
        out = super().post_physics(stop)
        if not self._settling and self._feet0 is None:
            self._feet0 = self.d.xpos[self._feet_b][:, :2].copy()
            self.box0 = self.true_pose("box")[0].copy()
            self._pel0 = float(self.d.xpos[self._pelvis][2]) if self._pelvis is not None else 0.0
        return out

    # ------------------------------------------------------------------ predicates
    def foot_slip(self):
        if self._feet0 is None:
            return 0.0
        return float(np.max(np.linalg.norm(self.d.xpos[self._feet_b][:, :2] - self._feet0, axis=1)))

    def fell(self):
        if self._pelvis is None or self._feet0 is None:
            return False
        return float(self.d.xpos[self._pelvis][2]) < self._pel0 - 0.25

    def box_moved(self):
        if self.box0 is None:
            return 0.0
        return float(self.true_pose("box")[0][1] - self.box0[1])

    def success(self):
        return self.box_moved() >= 0.06 and not self.fell() and self.foot_slip() <= self.SLIP_LIMIT

    def progress(self):
        return float(np.clip(self.box_moved() / 0.06, 0, 1) * (not self.fell()) * (self.foot_slip() <= self.SLIP_LIMIT))

    def fatal(self):
        return self.fell()

    def metrics_extra(self):
        return dict(x_box_moved=self.box_moved(), x_foot_slip=self.foot_slip(), x_fell=float(self.fell()))

    def labels_extra(self):
        return dict(task_hidden=dict(self.hidden_task))

    # ------------------------------------------------------------------ skill
    def fsk_g1_push(self, M, call):
        """push the box along +y with the G1's hand. params: f_cap (N: the hand yields above it), speed (m/s),
        distance (m), qp_mu (friction the balance QP assumes for the foot-force cone)."""
        p = call.params
        if M.probe or getattr(self.robot.spec, "kind", "") != "humanoid":
            return "unsupported", {}
        from physmem.embodiments.humanoid import g1_push
        est = self.skills._est(self, "box")
        if est is None:
            return "not_observed", {}
        ctrl = M.ctx.ctrl
        if hasattr(ctrl, "wb"):
            ctrl.wb.mu = float(np.clip(p.get("qp_mu", 0.6), 0.05, 1.2))
        # g1_push theta = push direction in the object frame (contact on the opposite face): +y, away from the robot
        st, log = g1_push(M.ctx, est, math.pi / 2, speed=float(np.clip(p.get("speed", 0.2), 0.02, 0.5)),
                          distance=float(np.clip(p.get("distance", 0.09), 0.02, 0.15)),
                          kw_fcap=float(np.clip(p.get("f_cap", 35.0), 5.0, 120.0)))
        if self.fell():
            return "fell", log
        if self.foot_slip() > self.SLIP_LIMIT:
            return "foot_slip", log
        if self.box_moved() < 0.06:
            return "stalled", log
        return "done", log

    NOMINAL = dict(f_cap=35.0, speed=0.2, distance=0.09, qp_mu=0.6)

    def scripted(self):
        """engineer's heuristic: ROBOTS' default push cap (35 N), 0.2 m/s, QP friction 0.6; one retry at +50 % cap."""
        if self.success() or self.n_calls >= 2 or (self.n_calls and self.fell()):
            return None
        p = dict(self.NOMINAL)
        if self.n_calls == 1:
            p["f_cap"] *= 1.5
        return SkillCall("g1_push", p, target="box")

    def v3_rule(self, call, src):
        """oracle: cap = 1.4 x the needed push force (box friction x weight) - if the feet's friction capacity
        (0.6 x floor mu x robot weight) is lower, push at that capacity, slowly; the QP assumes the true floor mu;
        slower pushes on soft floors."""
        if call is None or call.skill != "g1_push":
            return call
        from dataclasses import replace as _rep
        h = self.hidden_task
        if src != "true":
            return call
        W = 35.0 * G9
        need = h["push_n"]
        cap_feet = 0.6 * h["mu_floor"] * W
        p = dict(call.params, f_cap=float(min(1.4 * need + 5.0, max(cap_feet, need * 1.05))),
                 qp_mu=float(np.clip(0.8 * h["mu_floor"], 0.05, 1.0)),
                 speed=0.1 if h["floor_tc"] > 0.01 or need > 0.5 * cap_feet else 0.2)
        return _rep(call, params=p)

    def probe_then_act(self, call):
        """force ramp (no hidden knowledge): first push at a 15 N cap, then +15 N per retry (the box moves once the cap
        exceeds its friction); QP friction 0.4 (conservative)."""
        if call is None or call.skill != "g1_push":
            return call
        from dataclasses import replace as _rep
        return _rep(call, params=dict(call.params, f_cap=15.0 + 15.0 * self.n_calls, qp_mu=0.4, speed=0.15))

    def candidates(self, n, rng):
        return [SkillCall("g1_push", dict(f_cap=float(rng.uniform(10, 90)), speed=float(rng.uniform(0.05, 0.3)),
                                          distance=0.09, qp_mu=float(rng.uniform(0.1, 0.9))), target="box")
                for _ in range(n)]
