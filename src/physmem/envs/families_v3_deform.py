"""envs-v3 deformable families (agent ENVS-V3): cloth fold and cable routing / insertion.

cloth_fold   a square cloth (MuJoCo 2-D flexcomp shell: inextensible edges via edge equality + elastic2d="bend") lies
             on the bench. A small rigid PULL TAB is fixed to one corner vertex (labelled simplification: picking a
             thin cloth edge off a table with a parallel gripper is not reliable in MuJoCo or in practice; the tab plays
             the role of a pre-grasped / hemmed corner). Fold the tab corner onto the opposite corner (diagonal fold).
             Corner markers (tiny visual fiducials on two corner vertices) are tracked like any part.
             Hidden: bending stiffness (shell thickness x Young), cloth<->bench friction, areal mass.
cable_route  a cable (chain of capsules joined by ball joints with hidden bending stiffness and damping = a discrete
             elastic rod; NOT the MuJoCo cable plugin) is anchored at one end in a wall connector; its free end carries
             a rigid plug. Route the cable around a post (robot side) and insert the plug in a panel socket (1-1.8 mm clearance
             per side, 20 mm deep, horizontal).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..schema import EntityType
from ..v2.api import SkillCall
from .families_v3_common import V3Base
from .scene import EntityInfo, quat_z
from .stations import bench_station

G = mujoco.mjtGeom
J = mujoco.mjtJoint


class ClothFold(V3Base):
    family = "cloth_fold"
    station = "cloth_bench"
    probe_kw = dict(max_open=0.03, grip_max=60.0)
    timestep = 0.002
    settle_s = 1.0
    budget = 6
    success_doc = ("diagonal fold: 2 s after release the tab corner lies within 3 cm (xy) of the opposite corner, that "
                   "corner moved < 4 cm from where it started (cloth not dragged), the cloth is on the bench, released")
    hidden_doc = ("bending stiffness (shell thickness 0.3-1.5 mm at Young 2e5-1e6 Pa): a stiff cloth springs back after "
                  "release -> the corner must be placed beyond the target / pressed; cloth<->bench friction 0.25-1.0: a "
                  "slippery cloth is dragged along when the corner is pulled low and fast -> lift higher / slower; "
                  "areal mass 0.1-0.4 kg/m2")
    N = 9
    SPACING = 0.03

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.30
        scene.spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER      # MuJoCo 4: flex elasticity needs Euler
        thick = float(params.get("thickness", math.exp(rng.uniform(math.log(3e-4), math.log(1.5e-3)))))
        young = float(params.get("young", math.exp(rng.uniform(math.log(2e5), math.log(1e6)))))
        mu = float(params.get("mu", rng.uniform(0.25, 1.0)))
        areal = float(params.get("areal_kg", rng.uniform(0.1, 0.4)))
        n, sp = self.N, self.SPACING
        L = (n - 1) * sp
        cx, cy = -0.02 + rng.uniform(-0.03, 0.03), 0.04 + rng.uniform(-0.03, 0.03)
        yaw = float(rng.uniform(-0.3, 0.3))
        mass = areal * L * L
        xml = (f'<mujoco><worldbody><flexcomp name="cloth" type="grid" count="{n} {n} 1" spacing="{sp} {sp} {sp}" '
               f'pos="0 0 0" dim="2" radius="0.0015" mass="{mass}" rgba="0.75 0.25 0.3 1">'
               f'<elasticity young="{young}" poisson="0.3" thickness="{thick}" damping="0.002" elastic2d="bend"/>'
               f'<contact condim="3" selfcollide="none" friction="{mu} 0.005 0.0001" solref="0.004 1"/>'
               f'<edge equality="true" damping="0.05"/></flexcomp></worldbody></mujoco>')
        frame = scene.spec.worldbody.add_frame(pos=[cx, cy, top + 0.0016], quat=list(quat_z(yaw)))
        scene.spec.attach(mujoco.MjSpec.from_string(xml), prefix="", frame=frame)
        # vertex bodies cloth_<k>, k = i * n + j (grid index): corners 0 (i=0,j=0) and n*n-1 opposite
        c, s = math.cos(yaw), math.sin(yaw)

        def corner_xy(k):
            i, j = divmod(k, n)
            x, y = (i - (n - 1) / 2) * sp, (j - (n - 1) / 2) * sp
            return cx + c * x - s * y, cy + s * x + c * y
        ka, kc = 0, n * n - 1
        self.k_tab, self.k_far = ka, kc
        tab_b = scene.spec.body(f"cloth_{ka}")
        th = (0.004, 0.012, 0.0125)                 # pull tab (corner clip): fin 8 mm thick, 24 mm long, 25 mm tall, 12 g
        g = tab_b.add_geom(type=G.mjGEOM_BOX, size=list(th), pos=[0, 0, th[2] + 0.0005], rgba=[0.15, 0.15, 0.17, 1],
                           friction=[0.6, 0.005, 0.0001], name="tab_g", quat=list(quat_z(math.pi / 4)))   # body already has the cloth yaw
        g.mass = 0.012                              # a 4 g tab was pushed aside by the first finger to touch it
        info = EntityInfo("tab", "cloth_tab", EntityType.OBJECT, f"cloth_{ka}", "box", (th[0], th[1], th[2]),
                          dict(mass=0.012, mu_grip=0.9), graspable=True, geoms=["tab_g"], meta=dict(on_flex=True))
        scene.entities["tab"] = info
        scene.grip_pairs_objects.append("tab")
        for name, k in (("corner_far", kc), ("corner_b", n - 1), ("corner_d", n * (n - 1))):
            b = scene.spec.body(f"cloth_{k}")
            b.add_geom(type=G.mjGEOM_SPHERE, size=[0.004, 0, 0], pos=[0, 0, 0.003], rgba=[1, 0.9, 0.1, 1],
                       contype=0, conaffinity=0, name=f"{name}_mark").mass = 0.0
            scene.entities[name] = EntityInfo(name, "cloth_marker", EntityType.OBJECT, f"cloth_{k}", "sphere",
                                              (0.004, 0.004, 0.004), {}, graspable=False, geoms=[f"{name}_mark"])
        scene.entities["cloth"] = EntityInfo("cloth", "cloth", EntityType.OBJECT, "cloth", "box", (L / 2, L / 2, 0.002),
                                             dict(mass=mass, mu=mu, thickness=thick, young=young, areal_kg=areal,
                                                  bend_d=young * thick ** 3 / 12.0),
                                             graspable=False, meta=dict(flex=True))
        fx, fy = corner_xy(kc)
        dg = math.atan2(fy - corner_xy(ka)[1], fx - corner_xy(ka)[0])     # fold direction (tab -> far corner)
        scene.region("fold_target", (fx, fy, top + 0.01), (0.03, 0.03, 0.02), yaw=dg, support_z=top + 0.004)
        scene.region("bench_area", (0.0, 0.0, top + 0.05), (0.55, 0.38, 0.06), support_z=top)
        self.hidden_task = dict(thickness=thick, young=young, mu=mu, areal_kg=areal, bend_d=young * thick ** 3 / 12.0)
        self.budget = int(params.get("budget", 6))

    def _index(self):
        super()._index()
        m = self.m
        fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, "cloth")
        self._fv0 = int(m.flex_vertadr[fid])
        self._far0 = None

    def _vert(self, k):
        return self.d.flexvert_xpos[self._fv0 + k].copy()

    def fold_err(self):
        return float(np.linalg.norm((self._vert(self.k_tab) - self._vert(self.k_far))[:2]))

    def far_moved(self):
        if self._far0 is None:
            return 0.0
        return float(np.linalg.norm((self._vert(self.k_far) - self._far0)[:2]))

    def post_physics(self, stop=None):
        if self._far0 is None and not self._settling:
            self._far0 = self._vert(self.k_far)
        return super().post_physics(stop)

    def success(self):
        P = self.d.flexvert_xpos[self._fv0:self._fv0 + self.N * self.N]
        on = bool(np.all(P[:, 2] > self.surface_z - 0.01))
        return self.held is None and self.fold_err() <= 0.03 and self.far_moved() < 0.04 and on

    def progress(self):
        L = (self.N - 1) * self.SPACING * math.sqrt(2)
        return float(np.clip(1.0 - (self.fold_err() - 0.03) / L, 0.0, 1.0) * (self.far_moved() < 0.04))

    def fatal(self):
        P = self.d.flexvert_xpos[self._fv0:self._fv0 + self.N * self.N]
        return bool(np.any(P[:, 2] < self.surface_z - 0.05))           # cloth pulled off the bench

    def metrics_extra(self):
        return dict(fold_err=self.fold_err(), far_moved=self.far_moved())

    def labels_extra(self):
        return dict(task_hidden=dict(self.hidden_task))

    def goal_extra(self):
        return dict(fold_from="tab", fold_to="corner_far")

    # ------------------------------------------------------------------ family skill
    def fsk_fold(self, M, call):
        """pinch the tab, carry the corner along the fold diagonal at a low height and lay it down past the far corner.
        params: grip_force (N), lift (m: corner height during the carry), speed (m/s), overshoot (m beyond the far
        corner along the fold direction), release_h (m above the bench at release), dwell (s held at release height)."""
        p = call.params
        est = self.est_pose("tab")
        far = self.est_pose("corner_far")
        if est is None or far is None:
            return "not_observed", {}
        tpos, _, tyaw = est
        # fold direction from the TRACKED tab and far-corner marker (the fin's long axis lies along this diagonal; the
        # tab rides on a flex vertex whose tracked orientation carries no yaw information)
        dvec = np.asarray(far[0][:2]) - np.asarray(tpos[:2])
        dg = math.atan2(dvec[1], dvec[0])
        u = np.array([math.cos(dg), math.sin(dg)])
        yaw = dg                                    # pads close across the fin (perpendicular to the diagonal)
        tcp, y_now = M.tcp()
        if abs(math.remainder(yaw - y_now, 2 * math.pi)) > math.pi / 2:
            yaw = math.remainder(yaw + math.pi, 2 * math.pi)
        z_g = max(self.surface_z + 0.016, self.surface_z + M.tip() + 0.004)
        top = self.surface_z + 0.12
        M.open()
        for q in ((tcp[0], tcp[1], max(top, tcp[2])), (tpos[0], tpos[1], top), (tpos[0], tpos[1], z_g + 0.02),
                  (tpos[0], tpos[1], z_g)):
            if M.move(q, yaw, 0.0, 0.3 if q[2] > z_g + 0.01 else 0.06) == "unreachable":
                return "unreachable", {}
        M.grip(float(p.get("grip_force", 15.0)))
        if not M.holding():
            M.open()
            return "missed", {}
        self.held = "tab"
        h = float(np.clip(p.get("lift", 0.06), 0.01, 0.3))
        sp = float(np.clip(p.get("speed", 0.3), 0.02, 1.0))
        goal = np.asarray(far[0][:2]) + u * float(p.get("overshoot", 0.0))
        start = np.asarray(tpos[:2])
        mid = 0.5 * (start + goal)
        # arc: up, over the middle (a fold lifts the corner to about half the carry distance), down at the goal
        L = float(np.linalg.norm(goal - start))
        z_mid = self.surface_z + max(h, 0.5 * L * float(p.get("arc", 0.0)))
        h = max(h, z_g - self.surface_z)
        for q, v in (((start[0], start[1], self.surface_z + h), 0.1), ((mid[0], mid[1], z_mid), sp),
                     ((goal[0], goal[1], self.surface_z + h), sp),
                     ((goal[0], goal[1], z_g + float(p.get("release_h", 0.01))), 0.05)):
            st = M.move(q, yaw, 0.0, v)
            if st == "unreachable":
                M.open()
                self.held = None
                return "unreachable", {}
        M.wait(float(np.clip(p.get("dwell", 0.2), 0.0, 3.0)))
        slipped = not M.holding()
        M.open()
        self.held = None
        tcp, _ = M.tcp()
        M.move((tcp[0], tcp[1], tcp[2] + 0.08), yaw, 0.0, 0.1)
        M.wait(2.0)                                        # the fold relaxes (spring-back) before it is judged
        st = "done" if (self.success() or not slipped) else "slipped"
        return st, dict(fold_err=self.fold_err(), far_moved=self.far_moved(), lost_before_release=slipped)

    # ------------------------------------------------------------------ policies
    NOMINAL = dict(grip_force=15.0, lift=0.06, speed=0.3, overshoot=0.0, release_h=0.01, arc=0.0, dwell=0.2)
    CAL_GRID = [{}, dict(arc=1.0, lift=0.06, speed=0.15, overshoot=0.03), dict(lift=0.12, arc=0.5),
                dict(arc=1.0, speed=0.1, overshoot=0.05, release_h=0.0, dwell=1.0),
                dict(overshoot=0.04, release_h=0.0, dwell=1.0), dict(lift=0.03, arc=0.8, speed=0.1),
                dict(arc=0.5, overshoot=0.02, speed=0.15), dict(arc=1.0, lift=0.1, speed=0.1, overshoot=0.05, dwell=1.0)]

    @staticmethod
    def bin_of(h):
        return f"thick{'H' if h['thickness'] >= 7e-4 else 'L'}_mu{'H' if h['mu'] >= 0.55 else 'L'}"

    def calls_from(self, prm, call):
        return self._fold_calls(prm) if call.skill == "fold" else call

    def probe_then_act(self, call):
        return call            # no actor-legal probe of bending / bench friction is implemented for the fold

    def _fold_calls(self, prm):
        return SkillCall("fold", dict(prm), target="tab")

    def scripted(self):
        if self.n_calls > 0 and (self.last_result.ok or self.fail_count.get("tab", 0) >= 2):
            return None
        return self._fold_calls(self.NOMINAL)

    def candidates(self, n, rng):
        out = []
        for _ in range(n):
            prm = dict(grip_force=float(rng.uniform(5, 40)), lift=float(rng.uniform(0.02, 0.2)),
                       speed=float(rng.uniform(0.05, 0.5)), overshoot=float(rng.uniform(-0.02, 0.08)),
                       release_h=float(rng.uniform(0.0, 0.04)), arc=float(rng.uniform(0.0, 1.0)),
                       dwell=float(rng.uniform(0.0, 1.5)))
            out.append(self._fold_calls(prm))
        return out


class CableRoute(V3Base):
    family = "cable_route"
    station = "cable_bench"
    probe_kw = dict(max_open=0.03, grip_max=80.0)
    timestep = 0.001                     # 2 ms: solver blow-ups (badqacc) with the 2F-85 pinching the chain end
    settle_s = 1.0
    budget = 4
    success_doc = ("route the cable around the post on the robot side (cable winding about the post > 1 rad, "
                   "2 s after release) and plug the connector into the panel socket (front >= 70 % of the 20 mm socket "
                   "depth), released")
    hidden_doc = ("cable bending stiffness 0.003-0.2 N m/rad per 3 cm joint (EI ~1e-4-6e-3 N m2: thin data cable to power cord) (a discrete elastic rod; rest shape = its "
                  "coiled set), joint damping 2e-4-3e-3 N m s/rad, mass 0.06-0.16 kg/m. A stiff cable wrapped round "
                  "the post pulls back and twists the plug in a light pinch (the plug yaws off the socket axis and jams) -> grip "
                  "force and the wrap radius depend on the stiffness; 50 % of the nylon connector housings crack above a "
                  "hidden 10-22 N pinch (crush limit) -> the grip window is [cable pull-back need, crush limit]")
    N_LINK, SEG, R_C = 20, 0.03, 0.0035
    PLUG = (0.022, 0.008, 0.011)          # plug half extents (x along the cable): 44 mm body incl. strain relief
    GRASP_BACK = 0.017                    # pinch 17 mm behind the plug centre: 39 mm of plug ahead of the TCP

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.25
        scene.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        kb = float(params.get("bend_k", math.exp(rng.uniform(math.log(0.003), math.log(0.2)))))
        damp = float(params.get("cable_damp", math.exp(rng.uniform(math.log(2e-4), math.log(3e-3)))))
        lin = float(params.get("lin_kg", rng.uniform(0.06, 0.16)))
        mu = float(params.get("mu", rng.uniform(0.3, 0.8)))
        mug = float(params.get("mu_grip", rng.uniform(0.5, 0.9)))
        # routing POST (strain-relief peg, 4 cm tall, r 12 mm): the cable must pass it on the ROBOT side (-y); its rest
        # coil lies on the far side, so the plug has to be carried around below the post
        pcx, pcy = 0.0 + rng.uniform(-0.02, 0.02), -0.12 + rng.uniform(-0.02, 0.02)
        r_post = 0.012
        g = scene.spec.worldbody.add_geom(type=G.mjGEOM_CYLINDER, size=[r_post, 0.02, 0], pos=[pcx, pcy, top + 0.02],
                                          rgba=[0.3, 0.55, 0.3, 1], friction=[0.3, 0.005, 0.0001], name="route_post")
        scene.region("post", (pcx, pcy, top + 0.02), (r_post, r_post, 0.02), support_z=top, side="robot")
        zc = top + self.R_C + 0.0005
        ax, ay = pcx - 0.28, pcy + 0.06 + rng.uniform(-0.02, 0.02)
        # anchor connector block (static) on the bench
        scene.static_box((ax - 0.012, ay, top + 0.012), (0.012, 0.015, 0.012), rgba=(0.2, 0.2, 0.22, 1))
        # rest shape: a coiled set (constant curvature, clockwise from heading +y): R = 0.18 m
        Rr = float(params.get("coil_r", rng.uniform(0.2, 0.24)))
        dth = -self.SEG / Rr
        parent = scene.spec.worldbody
        pos, quat = [ax, ay, zc], list(quat_z(math.pi / 2))
        geoms, bodies = [], []
        m_link = lin * self.SEG
        for i in range(self.N_LINK):
            b = parent.add_body(name=f"cab{i:02d}", pos=pos, quat=quat)
            b.add_joint(name=f"cab{i:02d}_j", type=J.mjJNT_BALL, stiffness=kb, damping=damp, armature=1e-5)
            g = b.add_geom(type=G.mjGEOM_CAPSULE, fromto=[0.0, 0, 0, self.SEG, 0, 0], size=[self.R_C, 0, 0],
                           rgba=[0.1, 0.1, 0.12, 1], friction=[mu, 0.005, 0.0001], condim=3, name=f"cab{i:02d}_g")
            g.mass = m_link
            geoms.append(g.name)
            bodies.append(b)
            parent = b
            pos, quat = [self.SEG, 0, 0], list(quat_z(dth if i < self.N_LINK - 1 else 0.0))
        px, py, pz = self.PLUG
        # plug body origin = plug centre (what the tracker reports); its ball joint sits at the cable end (-px)
        plug = parent.add_body(name="plug", pos=[self.SEG + px, 0, 0], quat=[1, 0, 0, 0])
        plug.add_joint(name="plug_j", type=J.mjJNT_BALL, stiffness=kb * 3, damping=damp, armature=1e-4, pos=[-px, 0, 0])
        g = plug.add_geom(type=G.mjGEOM_BOX, size=[px, py, pz], pos=[0, 0, 0], rgba=[0.9, 0.75, 0.1, 1],
                          friction=[0.4, 0.005, 0.0001], condim=4, name="plug_g")
        g.mass = 0.03
        self._cable_geoms = geoms
        from .scene import fragility
        crush = fragility(rng, 0.5, 10.0, 22.0, params)         # nylon connector housing (50 % fragile)
        scene.entities["plug"] = EntityInfo("plug", "cable_plug", EntityType.OBJECT, "plug", "box", (px, py, pz),
                                            dict(mass=0.03, mu_grip=mug, bend_k=kb, cable_damp=damp, lin_kg=lin, mu=mu,
                                                 crush_n=crush),
                                            graspable=True, geoms=["plug_g"], meta=dict(on_chain=True, front=px))
        scene.grip_pairs_objects.append("plug")
        # panel socket: horizontal tunnel through a static panel, axis +x (plug pushed along +x), 1 mm clearance / side
        sx, sy = 0.22 + rng.uniform(-0.02, 0.02), -0.06 + rng.uniform(-0.03, 0.03)
        clr = float(params.get("clearance", rng.uniform(0.001, 0.0018)))
        depth = 0.02
        hy_, hz_ = py + clr, pz + clr
        hz = top + hz_ + 0.003               # hole axis height (3 mm floor under the hole)
        X = sx + depth / 2
        scene.static_box((X, sy, hz + hz_ + 0.01), (depth / 2, hy_ + 0.02, 0.01), rgba=(0.5, 0.52, 0.55, 1), mu=0.2)
        scene.static_box((X, sy, (top + hz - hz_) / 2), (depth / 2, hy_ + 0.02, (hz - hz_ - top) / 2),
                         rgba=(0.5, 0.52, 0.55, 1), mu=0.2)
        for sgn in (-1, 1):
            scene.static_box((X, sy + sgn * (hy_ + 0.01), hz), (depth / 2, 0.01, hz_), rgba=(0.5, 0.52, 0.55, 1), mu=0.2)
        scene.static_box((sx + depth + 0.004, sy, hz), (0.004, hy_ + 0.02, hz_ + 0.02), rgba=(0.4, 0.4, 0.42, 1))
        scene.region("socket", (sx + depth / 2, sy, hz), (depth / 2, hy_, hz_), yaw=0.0, support_z=hz - hz_, mouth_x=sx,
                     depth=depth)
        # latch: two spring lips in the side walls at 55 % depth protrude 0.8 mm into the plug's path (a connector
        # latch); pushed past, they retain the plug against the cable's pull-back. Lip<->plug contacts are explicit pairs
        k_l = 6000.0
        prot = clr + 0.0008
        for sgn in (-1, 1):
            b = scene.spec.worldbody.add_body(name=f"latch{sgn > 0:d}", pos=[sx + 0.55 * depth, sy + sgn * (hy_ + 0.0015 - prot), hz])
            b.add_joint(name=f"latch{sgn > 0:d}_j", type=J.mjJNT_SLIDE, axis=[0, sgn, 0], range=[0.0, prot + 0.001],
                        limited=True, stiffness=k_l, springref=0.0, damping=2.0, armature=0.001)
            g = b.add_geom(type=G.mjGEOM_CAPSULE, size=[0.0015, hz_ * 0.8, 0], rgba=[0.8, 0.3, 0.2, 1],
                           name=f"latch{sgn > 0:d}_g", contype=0, conaffinity=0)
            g.mass = 0.002
            scene.spec.add_pair(geomname1="plug_g", geomname2=f"latch{sgn > 0:d}_g", condim=3,
                                friction=[0.3, 0.3, 0.005, 0.0001, 0.0001], solref=[0.004, 1.0])
        self.sock = dict(x=sx, y=sy, z=hz, depth=depth, clr=clr)
        self.hidden_task = dict(bend_k=kb, cable_damp=damp, lin_kg=lin, mu=mu, mu_grip=mug, clearance=clr, coil_r=Rr,
                                crush_n=crush)
        self.budget = int(params.get("budget", 4))

    def _index(self):
        super()._index()
        m = self.m
        self._cab_b = np.array([m.body(f"cab{i:02d}").id for i in range(self.N_LINK)], int)
        self._plug_b = m.body("plug").id

    # ------------------------------------------------------------------ evaluator predicates
    def _plug_frame(self):
        R = self.d.xmat[self._plug_b].reshape(3, 3)
        return self.d.xpos[self._plug_b].copy(), R

    def seated_frac(self):
        p, R = self._plug_frame()
        front = p + R[:, 0] * self.PLUG[0]
        s = self.sock
        lat = abs(front[1] - s["y"]) < 0.004 and abs(front[2] - s["z"]) < 0.004 and R[0, 0] > 0.97
        return float(np.clip((front[0] - s["x"]) / s["depth"], 0.0, 1.0)) if lat else 0.0

    def winding(self):
        """signed angle (rad) swept by the cable polyline (anchor -> plug) around the post axis (evaluator only):
        about +4 rad when it wraps the post on the robot side, negative when it passes on the far side."""
        reg = self.scene.regions["post"]
        c = np.asarray(reg["center"][:2])
        P = [self.d.xpos[b][:2] for b in self._cab_b] + [self.d.xpos[self._plug_b][:2]]
        V = np.array(P) - c
        ang = np.arctan2(V[:, 1], V[:, 0])
        return float(np.sum(np.remainder(np.diff(ang) + math.pi, 2 * math.pi) - math.pi))

    def in_channel(self):
        """route predicate (name kept from the channel prototype): the cable wraps the post on the robot side."""
        return self.winding() > 1.0

    def success(self):
        return self.held is None and self.seated_frac() >= 0.7 and self.in_channel()

    def progress(self):
        return float(0.5 * self.in_channel() + 0.5 * min(1.0, self.seated_frac() / 0.7))

    def metrics_extra(self):
        return dict(seated=self.seated_frac(), route_ok=float(self.in_channel()), wind=self.winding())

    def labels_extra(self):
        return dict(task_hidden=dict(self.hidden_task))

    def goal_extra(self):
        return dict(route=["post", "socket"], post_side="robot")

    # ------------------------------------------------------------------ family skill
    def fsk_route_insert(self, M, call):
        """pinch the plug, carry it low around the post on the robot side (the cable wraps the post), carry it to the socket mouth, push it
        in along +x, release. params: grip_force, speed,
        drag_h (plug height above the bench while threading), push (m beyond the seated depth), push_speed, dwell."""
        p = call.params
        est = self.est_pose("plug")
        if est is None:
            return "not_observed", {}
        ppos, _, pyaw = est
        ax = np.array([math.cos(pyaw), math.sin(pyaw)])
        # grasp point: plug centre (body origin is the cable end of the plug; tracker reports the body origin)
        gp = ppos[:2] - ax * self.GRASP_BACK
        yaw = pyaw                                    # pads close across the plug (hand x along the plug axis)
        tcp, y_now = M.tcp()
        if abs(math.remainder(yaw - y_now, 2 * math.pi)) > math.pi / 2:
            yaw = math.remainder(yaw + math.pi, 2 * math.pi)
        zc_plug = self.surface_z + self.PLUG[2] + 0.0015
        zg = max(zc_plug, self.surface_z + M.tip() + 0.003)        # fingertips clear the bench
        dz = zg - zc_plug                                            # grasp height above the plug centre
        M.open()
        for q, v in (((tcp[0], tcp[1], max(tcp[2], self.surface_z + 0.12)), 0.4), ((gp[0], gp[1], self.surface_z + 0.12), 0.4),
                     ((gp[0], gp[1], zg + 0.02), 0.2), ((gp[0], gp[1], zg), 0.05)):
            if M.move(q, yaw, 0.0, v) == "unreachable":
                return "unreachable", {}
        M.grip(float(p.get("grip_force", 15.0)))
        if not M.holding():
            M.open()
            return "missed", {}
        self.held = "plug"
        sp = float(np.clip(p.get("speed", 0.2), 0.03, 0.6))
        dh = float(np.clip(p.get("drag_h", 0.012), 0.0, 0.05))
        reg = self.scene.regions["post"]
        pc = np.asarray(reg["center"][:2])
        rw = float(np.clip(p.get("wrap_r", 0.05), 0.025, 0.12))        # path clearance radius around the post
        z_up = self.surface_z + 0.10
        z_low = zg + float(np.clip(p.get("drag_h", 0.01), 0.0, 0.03))  # carried low: the cable cannot hop the post
        s = self.sock
        mouth = np.array([s["x"] - 0.04, s["y"]])
        zs = s["z"] + dz
        # around the post on the robot side: lower-left -> below -> lower-right, then to the socket mouth (+x)
        w1 = pc + np.array([-rw, -0.2 * rw])
        w2 = pc + np.array([0.0, -rw])
        w3 = pc + np.array([rw, -0.2 * rw])
        lead = float(np.clip(p.get("lead", 0.0), 0.0, 0.1))
        w4 = w3 + np.array([lead, 0.0])
        steps = (((gp[0], gp[1], z_up), yaw, 0.15), ((w1[0], w1[1], z_up), 0.0, sp),
                 ((w1[0], w1[1], z_low), 0.0, 0.08), ((w2[0], w2[1], z_low), 0.0, sp * 0.5),
                 ((w3[0], w3[1], z_low), 0.0, sp * 0.5), ((w4[0], w4[1], z_low), 0.0, sp * 0.5),
                 ((mouth[0], mouth[1], zs), 0.0, sp * 0.5))
        for q, yw, v in steps:
            # plug axis = hand x: the plug's front points along the hand x axis (yaw) or its opposite
            yw_h = yw if abs(math.remainder(yw - yaw, 2 * math.pi)) <= math.pi / 2 else math.remainder(yw + math.pi, 2 * math.pi)
            if M.move(q, yw_h, 0.0, v) == "unreachable":
                M.open()
                self.held = None
                return "unreachable", {}
            yaw = yw_h
        # correct the in-hand yaw / offset from the tracker before the push (actor-legal)
        M.wait(0.3)
        est = self.est_pose_avg("plug", 0.3)
        tcp, _ = M.tcp()
        log = {}
        if est is not None:
            pyaw2 = est[2]
            log["plug_yaw_err"] = float(math.remainder(pyaw2, 2 * math.pi))
            front = est[0][:2] + np.array([math.cos(pyaw2), math.sin(pyaw2)]) * self.PLUG[0]
            off = front - tcp[:2]
            if p.get("correct", 1):
                # rotate the hand by the plug's in-hand yaw error about the plug front, then line the front up
                e = math.remainder(pyaw2, 2 * math.pi)
                if abs(e) < 0.6:
                    yaw = yaw - e
                    cc, ss = math.cos(-e), math.sin(-e)
                    off = np.array([cc * off[0] - ss * off[1], ss * off[0] + cc * off[1]])
                tgt = np.array([s["x"] - 0.01, s["y"]]) - off
                M.move((tgt[0], tgt[1], zs), yaw, 0.0, 0.05)
        tcp, _ = M.tcp()
        push = float(np.clip(p.get("push", 0.004), 0.0, 0.02))
        M.move((tcp[0] + 0.01 + s["depth"] + push, tcp[1], zs), yaw, 0.0, float(np.clip(p.get("push_speed", 0.03), 0.005, 0.2)))
        M.wait(float(np.clip(p.get("dwell", 0.3), 0.0, 3.0)))
        slipped = not M.holding()
        log["seated_held"] = self.seated_frac()
        tcp_end, _ = M.tcp()
        log["push_short"] = float(tcp[0] + 0.01 + s["depth"] + push - tcp_end[0])
        M.open()
        self.held = None
        tcp, _ = M.tcp()
        M.move((tcp[0] - 0.01, tcp[1], tcp[2] + 0.08), yaw, 0.0, 0.1)
        M.wait(2.0)
        log.update(seated=self.seated_frac(), in_channel=self.in_channel())
        if self.seated_frac() < 0.7:
            return ("slipped" if slipped else "jammed"), log
        return "done", log

    NOMINAL = dict(grip_force=15.0, speed=0.2, wrap_r=0.08, lead=0.0, drag_h=0.01, push=0.004, push_speed=0.03, dwell=0.3,
                   correct=1)

    def scripted(self):
        if self.n_calls > 0 and self.last_call.skill == "route_insert" and (self.last_result.ok or
                                                                             self.fail_count.get("plug", 0) >= 2):
            return None
        if self.n_calls >= self.budget:
            return None
        return SkillCall("route_insert", dict(self.NOMINAL), target="plug")

    @staticmethod
    def grip_rule(bend_k, crush):
        """cable pull-back torque on the pinch grows with the bending stiffness (calibrated on the probe / UR5e
        sweeps: 6 N failed at both ends, 15 N held at every stiffness): 12 N + 20 N per N m/rad, capped at 0.7 x crush."""
        F = float(np.clip(12.0 + 20.0 * bend_k, 12.0, 18.0))
        if crush is not None and np.isfinite(crush):
            F = min(F, 0.7 * crush)
        return F

    def v3_rule(self, call, src):
        if call is None or call.skill != "route_insert":
            return call
        from dataclasses import replace as _rep
        h = self.hidden_task
        if src == "true":
            kb, crush = h["bend_k"], h["crush_n"]
        else:
            kb, crush = 0.025, None                 # population median stiffness; no crush knowledge
        p = dict(call.params, grip_force=self.grip_rule(kb, crush), wrap_r=0.08 if kb < 0.1 else 0.1)
        return _rep(call, params=p)

    def probe_then_act(self, call):
        """light-squeeze the plug first (tactile stiffness -> crush estimate), then pinch below 0.7 x the estimate."""
        if call is None or call.skill != "route_insert":
            return call
        if not self._probed("plug"):
            est = self.est_pose("plug")
            return SkillCall("squeeze_probe", dict(yaw=est[2] if est is not None else 0.0), target="plug")
        from dataclasses import replace as _rep
        crush = self.est_crush("plug", margin=1.2)
        F = call.params.get("grip_force", 15.0)
        if crush is not None and np.isfinite(crush):
            F = min(F, 0.7 * crush)
        return _rep(call, params=dict(call.params, grip_force=float(F)))

    def candidates(self, n, rng):
        return [SkillCall("route_insert", dict(grip_force=float(rng.uniform(5, 50)), speed=float(rng.uniform(0.05, 0.4)),
                                               wrap_r=float(rng.uniform(0.03, 0.1)), lead=float(rng.uniform(0.0, 0.08)),
                                               drag_h=float(rng.uniform(0.0, 0.03)),
                                               push=float(rng.uniform(0.0, 0.012)),
                                               push_speed=float(rng.uniform(0.01, 0.08)),
                                               dwell=float(rng.uniform(0.0, 1.5)), correct=1), target="plug")
                for _ in range(n)]
