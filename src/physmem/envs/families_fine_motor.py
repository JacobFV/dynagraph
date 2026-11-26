"""envs-r7 FINE-MOTOR families (reactive closed-loop skills; the lead's round-7 item 3).

tight_insert   square peg into a hole with 0.1-0.5 mm diametral clearance (hidden), chamfer 0 (50 %) / 0.2-0.6 mm,
               peg<->hole friction 0.2-0.8, a soft-alloy peg that dents above a hidden press force (50 %), tracker
               noise on the hole pose -> force-controlled spiral search (F/T drop detection) + compliant press with
               yaw wiggle and F/T jam recovery. Every gain / threshold is a SkillCall param of `insert`.
fastener_pick  M3-M6 screws (shank + head), washers and small connector blocks spilled in a shallow tray; pick the
               ordered items (class shown by SHAPE / SIZE, actor-legal) into the kit tray pockets. Hidden: per-item
               mass, grip friction (oily screws), and the washer's lying orientation makes a top pinch slip unless the
               pinch is on the rim (depth / grip force).

Both run on the real arms' reactive skills (envs/robots.py -> skills/primitives.py) and on the probe gantry.
"""
from __future__ import annotations

import math

import numpy as np

from ..v2.api import SkillCall
from .families_fine import PegInsert, _FineBase
from .scene import fragility
from .stations import bench_station


def _quat_z(a):
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


class TightInsert(PegInsert):
    family = "tight_insert"
    station = "insertion_fixture"
    success_doc = ("the 12-18 mm square peg is seated >= 80 % of the 22 mm hole depth, released, undented, within the "
                   "35 s cycle")
    hidden_doc = ("diametral clearance 0.1-0.5 mm, chamfer 0 (50 %) or 0.2-0.6 mm, peg<->hole friction 0.2-0.8, dent "
                  "limit 6-20 N (50 % of pegs, >= 1.8 x the snap force), snap-fit lips at 55 % depth needing 3-16 N "
                  "(lip stiffness 3-16 kN/m x friction): a straight press jams on the tight holes; the spiral must cover the "
                  "tracker error (0.5-1.5 mm) but costs time; pressing above the dent limit damages the peg")
    default_cycle_limit = 35.0

    def geometry(self, rng, params):
        s = float(params.get("peg_half", rng.uniform(0.006, 0.009)))
        return dict(half=(s, s, 0.03), sym=math.pi / 2, depth=0.022)

    def build(self, scene, rng, params):
        params = dict(params)
        params.setdefault("clearance", float(rng.uniform(0.0001, 0.0005)))
        params.setdefault("chamfer", 0.0 if rng.random() < 0.5 else float(rng.uniform(0.0002, 0.0006)))
        params.setdefault("mu", float(rng.uniform(0.2, 0.8)))
        # snap-fit lips first (their stiffness sets the press force), then the dent limit: a soft-alloy peg is rated for
        # its own latch force (dent >= 1.8 x snap + 2 N, still within 6-20 N) - else no press force could seat it undented
        k = float(params.get("snap_k", rng.uniform(3000.0, 16000.0)))
        params["snap_k"] = k
        snap = 1.5 + 6.5e-4 * k * (0.5 + float(params["mu"]))
        if "dent_n" not in params:
            lo = min(max(6.0, 1.8 * snap + 2.0), 19.0)
            params["dent_n"] = float(rng.uniform(lo, 20)) if rng.random() < 0.5 else float("inf")
        super().build(scene, rng, params)
        self._add_snap_lips(scene, rng, params)

    def _add_snap_lips(self, scene, rng, params):
        """r7 snap-fit retention: two spring-loaded round lips protrude 0.8 mm into the hole at 55 % depth (a latching
        connector / circlip). The peg must be pressed past them: below the hidden snap force it sticks at the lips
        (times out), far above it the peg dents (dent_n) - so the right press force depends on the hidden lip stiffness
        and the peg<->lip friction. Lip<->peg contacts are explicit pairs (the lips sit in recesses of the CAD walls)."""
        import mujoco
        reg = scene.regions[self.region_name]
        cx, cy, _ = reg["center"]
        yaw = reg.get("yaw", 0.0)
        hxh, hyh = reg["half"][0], reg["half"][1]
        c, s_ = math.cos(yaw), math.sin(yaw)
        k = float(params.get("snap_k", rng.uniform(3000.0, 16000.0)))
        r, prot = 0.0015, 0.0008
        z = self.ztop - 0.55 * self.depth
        mu_l = float(self.hidden_task["mu"])
        self.hidden_task["snap_k"] = k
        # measured (scripts/envs_snap_calib.py, UR5e): press force needed to pass the lips ~ 1.5 + 6.5e-4 k (0.5 + mu)
        self.hidden_task["snap_n"] = float(1.5 + 6.5e-4 * k * (0.5 + mu_l))
        for sgn in (-1, 1):
            off = sgn * (hxh + r - prot)
            pos = (cx + c * off, cy + s_ * off, z)
            b = scene.spec.worldbody.add_body(name=f"lip{sgn > 0:d}", pos=list(pos), quat=list(_quat_z(yaw)))
            b.add_joint(name=f"lip{sgn > 0:d}_j", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[sgn, 0, 0], range=[0.0, prot + 0.001],
                        limited=True, stiffness=k, springref=0.0, damping=2.0, armature=0.001)
            g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r, hyh * 0.8, 0], quat=[0.7071068, 0.7071068, 0, 0],
                           rgba=[0.8, 0.3, 0.2, 1], name=f"lip{sgn > 0:d}_g", contype=0, conaffinity=0)
            g.mass = 0.002
            for pg in self.scene.entities[self.obj].geoms:
                scene.spec.add_pair(geomname1=pg, geomname2=f"lip{sgn > 0:d}_g", condim=3,
                                    friction=[mu_l, mu_l, 0.005, 0.0001, 0.0001], solref=[0.004, 1.0])

    def scripted(self):
        """engineer's default: pick, then insert with a 2 mm spiral at 8 N; after a failure a wider spiral + wiggle."""
        if self.held is None:
            if self.success():
                return None
            return self._pick(self.obj)
        k = self.n_calls // 2
        return self._insert(self.region_name, search_r=0.002 * (1 + k), down_force=8.0, wiggle=0.01 * k,
                            pitch=0.0006, search_speed=0.006, retries=1 + k)


    RAMP = dict(down_force=4.0, ramp_step=2.0, ramp_max=14.0)

    def force_ramp(self, call):
        """r7 FORCE-RAMP scripted baseline (the insertion analogue of probe-then-act; no hidden knowledge): press at
        4 N, step +2 N after each stall, never above a conservative 14 N ceiling; no wiggle."""
        if call is None or call.skill != "insert":
            return call
        from dataclasses import replace as _rep
        return _rep(call, params=dict(call.params, min_force=0.0, wiggle=0.0, **self.RAMP))


class FastenerPick(_FineBase):
    family = "fastener_pick"
    station = "fastener_tray"
    probe_kw = dict(max_open=0.03, grip_max=40.0, pad_solref=0.003)
    settle_s = 1.0
    success_doc = ("every ordered item (2-3 of the spilled screws / connector blocks; washers are clutter only) sits in its kit "
                   "pocket; nothing dropped off the tray / bench")
    hidden_doc = ("item mass (steel 7.8 g/cm3 vs brass / nylon), item<->pad friction 0.15-0.9 (oily screws), washer "
                  "lying flat: a pinch across a 1-1.6 mm thick washer slips unless pinched at the rim with enough force; "
                  "nylon connector blocks crush above 5-15 N (50 %)")
    SIZES = {"M3": (0.0015, 0.008, 0.0028, 0.0012), "M4": (0.002, 0.010, 0.0036, 0.0015),
             "M5": (0.0025, 0.012, 0.0045, 0.0018), "M6": (0.003, 0.014, 0.0055, 0.002)}

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.15
        tray_c = (-0.12, 0.04)
        tz = scene.open_box((tray_c[0], tray_c[1], top), (0.09, 0.07), float(params.get("tray_wall_h", 0.02)),
                            prefix="tray", rgba=(0.25, 0.3, 0.35, 1))
        scene.region("tray", (tray_c[0], tray_c[1], tz + 0.02), (0.09, 0.07, 0.03), support_z=tz)
        n = int(params.get("n_items", rng.integers(6, 11)))
        self.items_order = []
        placed = []
        for i in range(n):
            kind = ("screw", "washer", "block")[int(rng.choice(3, p=(0.5, 0.3, 0.2)))]
            size = list(self.SIZES)[int(rng.integers(4))]
            r_sh, L, r_head, h_head = self.SIZES[size]
            dens = float(rng.choice([7800.0, 8500.0, 1150.0], p=(0.6, 0.25, 0.15)))
            for _ in range(60):
                x = tray_c[0] + rng.uniform(-0.075, 0.075)
                y = tray_c[1] + rng.uniform(-0.055, 0.055)
                if all(math.hypot(x - a, y - b) > 0.022 for a, b in placed):
                    break
            placed.append((x, y))
            yaw = float(rng.uniform(-math.pi, math.pi))
            oily = rng.random() < 0.25
            mug = float(rng.uniform(0.15, 0.3) if oily else rng.uniform(0.4, 0.9))
            eid = f"item{i:02d}"
            if kind == "screw":
                # shank (lying along x) + head disc at one end: compound of a box shank and a box head (screws lie
                # flat; the hexagonal head keeps them from rolling)
                boxes = [((0.0, 0.0, 0.0), (L / 2, r_sh, r_sh)), ((L / 2 + h_head / 2, 0.0, 0.0),
                                                                  (h_head / 2, r_head, r_head))]
                mass = dens * (L * (2 * r_sh) ** 2 + h_head * (2 * r_head) ** 2)
                _, info = scene.compound(eid, boxes, (x, y, tz + r_head + 0.001), yaw=yaw, mass=float(mass),
                                         mu=float(rng.uniform(0.3, 0.6)),
                                         mu_grip=mug, timeconst=0.004, kind="screw", rgba=(0.7, 0.7, 0.72, 1),
                                         mass_split=None, extra_hidden=dict(size=size, oily=bool(oily)))
                info.shape = "mesh"
                half = (L / 2 + h_head, r_head, r_head)
            elif kind == "washer":
                ro, t = r_head * 1.6, h_head * 0.8
                mass = dens * math.pi * ro ** 2 * t * 0.75
                _, info = scene.part(eid, "cylinder", (ro, ro, t / 2), (x, y, tz + t / 2 + 0.0005), yaw=yaw, mass=mass,
                                     mu=float(rng.uniform(0.3, 0.6)), mu_grip=mug, restitution=0.05, timeconst=0.004,
                                     kind="washer", extra_hidden=dict(size=size, oily=bool(oily)))
                half = (ro, ro, t / 2)
            else:
                hb = (float(rng.uniform(0.004, 0.007)), float(rng.uniform(0.003, 0.005)), float(rng.uniform(0.003, 0.005)))
                mass = 1150.0 * 8 * hb[0] * hb[1] * hb[2] * float(rng.uniform(0.8, 2.5))
                _, info = scene.part(eid, "box", hb, (x, y, tz + hb[2] + 0.0005), yaw=yaw, mass=mass,
                                     mu=float(rng.uniform(0.3, 0.7)), mu_grip=mug, restitution=0.05, timeconst=0.004,
                                     kind="connector_block", rgba=(0.15, 0.15, 0.15, 1),
                                     extra_hidden=dict(size="block", oily=False,
                                                       crush_n=fragility(rng, 0.5, 5, 15, params)))
                half = hb
            info.meta["item_class"] = kind
        ids = [f"item{i:02d}" for i in range(n)]
        # r7: washers stay in the tray as clutter but are never ordered: a 1-1.6 mm washer lying flat cannot be
        # pinched by a parallel gripper (real cells use a magnet or a vacuum nozzle)
        pickable = [e for e in ids if scene.entities[e].kind != "washer"]
        k = min(int(params.get("n_order", rng.integers(2, 4))), len(pickable))
        self.items_order = [pickable[j] for j in rng.permutation(len(pickable))[:k]]
        # kit tray with one pocket per ordered item (CAD region per pocket)
        kc = (0.14, 0.04)
        for j, eid in enumerate(self.items_order):
            px = kc[0] + (j - (k - 1) / 2) * 0.045
            fz = scene.open_box((px, kc[1], top), (0.017, 0.017), 0.012, prefix=f"pocket{j}", rgba=(0.8, 0.6, 0.1, 1))
            scene.region(f"pocket{j}", (px, kc[1], fz + 0.01), (0.017, 0.017, 0.015), support_z=fz, item=eid)
        self.budget = int(params.get("budget", 4 * k + 2))

    def items(self):
        return [e for e, i in self.scene.entities.items() if i.kind in ("screw", "washer", "connector_block")]

    def _pocket_of(self, eid):
        for name, reg in self.scene.regions.items():
            if name.startswith("pocket") and reg.get("item") == eid:
                return name
        return None

    def success(self):
        return all(self.in_region_true(e, self.scene.regions[self._pocket_of(e)], margin=0.004)
                   for e in self.items_order) and self.held is None

    def progress(self):
        return sum(self.in_region_true(e, self.scene.regions[self._pocket_of(e)], margin=0.004)
                   for e in self.items_order) / len(self.items_order)

    def fatal(self):
        return self.metrics_acc["drops"] > 0

    def goal_extra(self):
        return dict(order={e: self._pocket_of(e) for e in self.items_order})

    def _todo(self):
        out = []
        for e in self.items_order:
            if self.in_region_est(e, self.scene.regions[self._pocket_of(e)], margin=0.004):
                continue
            if self.est_pose(e) is not None:
                out.append(e)
        return out

    def _pick_item(self, e, rng=None, **kw):
        kind = self.scene.entities[e].kind
        p = dict(grip_force=12.0, lift_speed=0.15, depth=0.6 if kind != "washer" else 0.9, approach_speed=0.05,
                 open_margin=0.012, regrasp=2)
        if rng is not None:
            p.update(grip_force=float(rng.uniform(3, 40)), lift_speed=float(rng.uniform(0.05, 0.4)),
                     depth=float(rng.uniform(0.2, 1.0)))
        p.update(kw)
        return SkillCall("pick", p, target=e)

    def candidates(self, n, rng):
        if self.held is not None:
            pk = self._pocket_of(self.held)
            return [SkillCall("place", dict(release_h=float(rng.uniform(0.002, 0.02)), speed=float(rng.uniform(0.1, 0.5)),
                                            lower_speed=0.05), target=pk or "tray") for _ in range(n)]
        td = self._todo()
        return [self._pick_item(td[i % len(td)], rng) for i in range(n)] if td else []

    def scripted(self):
        if self.held is not None:
            pk = self._pocket_of(self.held)
            return SkillCall("place", dict(release_h=0.005, speed=0.3, lower_speed=0.05), target=pk or "tray")
        td = self._todo()
        if not td:
            return None
        cand = [e for e in td if self.fail_count.get(e, 0) < 3] or td
        e = cand[0]
        k = self.fail_count.get(e, 0)
        return self._pick_item(e, grip_force=(12.0, 20.0, 30.0)[min(k, 2)])

    def labels_extra(self):
        return dict(order=list(self.items_order))
