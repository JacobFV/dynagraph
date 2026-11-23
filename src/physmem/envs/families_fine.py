"""Fine task families: peg_insert, connector_insert, fixture_place, kitting_pocket, tool_return,
and articulated drawer_open, door_open (category "fine", params subtype "articulated")."""
from __future__ import annotations

import math

import numpy as np

from ..v2.api import SkillCall
from .base import StationEnv
from .probe import wrap
from .sensors import quat_yaw
from .scene import fragility
from .stations import bench_station


class _FineBase(StationEnv):
    category = "fine"
    timestep = 0.001
    fine_sensors = True
    tool = "parallel"
    probe_kw = dict(max_open=0.05, grip_max=100.0, pad_solref=0.003)

    def _pick(self, eid, rng=None, **kw):
        p = dict(grip_force=30.0, lift_speed=0.2, depth=-0.3, approach_speed=0.4)
        if rng is not None:
            p.update(grip_force=float(rng.uniform(8, 90)), lift_speed=float(rng.uniform(0.05, 0.5)),
                     depth=float(rng.uniform(-0.5, 0.0)), dx=float(rng.normal(0, 0.002)))
        p.update(kw)
        return SkillCall("pick", p, target=eid)

    def _insert(self, region, rng=None, **kw):
        p = dict(search_r=0.0015, search_hz=1.0, down_force=6.0, compliance=0.5, speed=0.3, timeout=4.0, wiggle=0.0,
                 correct=1)
        if rng is not None:
            p.update(search_r=float(rng.choice([0.0, rng.uniform(0.0005, 0.004)])), search_hz=float(rng.uniform(0.5, 2.0)),
                     down_force=float(rng.uniform(2, 30)), compliance=float(rng.uniform(0.05, 1.0)),
                     wiggle=float(rng.choice([0.0, rng.uniform(0.005, 0.05)])), timeout=float(rng.uniform(1.5, 5.0)))
        p.update(kw)
        return SkillCall("insert", p, target=region)


# ====================================================================== (e) peg in hole
class PegInsert(_FineBase):
    family = "peg_insert"
    station = "insertion_fixture"
    success_doc = "the square peg's bottom face is >= 80% of the hole depth below the hole mouth"
    hidden_doc = ("diametral clearance 0.5-2 mm, chamfer 0-3 mm (0 on 35% of episodes), peg<->hole friction "
                  "0.2-0.9: straight insertion jams on tight/no-chamfer holes -> needs spiral search, lower force")
    obj = "peg"
    region_name = "hole"

    def geometry(self, rng, params):
        s = float(params.get("peg_half", rng.uniform(0.009, 0.014)))
        return dict(half=(s, s, 0.04), sym=math.pi / 2, depth=0.03)

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.22
        g = self.geometry(rng, params)
        hx, hy, hz = g["half"]
        clr = float(params.get("clearance", rng.uniform(0.0005, 0.002)))
        ch = float(params.get("chamfer", 0.0 if rng.random() < 0.35 else rng.uniform(0.0005, 0.003)))
        mu = float(params.get("mu", rng.uniform(0.2, 0.9)))
        yaw_h = float(rng.uniform(-math.pi / 2, math.pi / 2)) if g.get("random_yaw") else float(rng.uniform(-0.3, 0.3))
        hc, pc, jit = getattr(self, "hole_center", (0.15, 0.05)), getattr(self, "peg_center", (-0.18, 0.0)), \
            getattr(self, "layout_jitter", 0.04)
        hpos = (hc[0] + rng.uniform(-jit, jit), hc[1] + rng.uniform(-jit, jit))
        ztop = top + g["depth"]
        scene.socket((hpos[0], hpos[1], ztop), (hx + clr / 2, hy + clr / 2), g["depth"], outer=0.03, chamfer=ch,
                     yaw=yaw_h, name="sock")
        scene.region(self.region_name, (hpos[0], hpos[1], ztop - g["depth"] / 2),
                     (hx + clr / 2, hy + clr / 2, g["depth"] / 2), yaw=yaw_h, support_z=ztop, depth=g["depth"],
                     sym=g["sym"])
        ppos = (pc[0] + rng.uniform(-jit, jit), pc[1] + rng.uniform(-jit, jit))
        mass = float(params.get("mass", rng.uniform(*getattr(self, "mass_range", (0.05, 0.4)))))
        self.make_object(scene, rng, g, ppos, top, mass, mu)
        self.hidden_task = dict(clearance=clr, chamfer=ch, mu=mu, mass=mass, hole_yaw=yaw_h)
        self.depth = g["depth"]
        self.ztop = ztop
        self.budget = int(params.get("budget", 8))

    def make_object(self, scene, rng, g, ppos, top, mass, mu):
        hx, hy, hz = g["half"]
        scene.part(self.obj, "box", (hx, hy, hz), (ppos[0], ppos[1], top + hz + 0.001), yaw=rng.uniform(-0.6, 0.6),
                   mass=mass, mu=mu, mu_grip=float(rng.uniform(0.5, 1.0)), restitution=0.05, timeconst=0.004,
                   kind="peg", rgba=(0.85, 0.6, 0.15, 1))

    def depth_frac(self):
        p, q = self.true_pose(self.obj)
        if not self.in_region_true(self.obj, self.scene.regions[self.region_name], margin=0.002, zcheck=False):
            return 0.0
        hz = self.scene.entities[self.obj].half[2]
        bottom = p[2] - hz
        return float(np.clip((self.ztop - bottom) / self.depth, 0, 1.5))

    def success(self):
        return self.depth_frac() >= 0.8 and self.held is None

    def progress(self):
        return float(min(1.0, self.depth_frac() / 0.8))

    def fatal(self):
        return self.metrics_acc["drops"] > 0

    def candidates(self, n, rng):
        if self.held is None:
            return [self._pick(self.obj, rng) for _ in range(n)]
        return [self._insert(self.region_name, rng) for _ in range(n)]

    def scripted(self):
        if self.held is None:
            if self.success():
                return None
            return self._pick(self.obj)
        k = self.n_calls // 2
        return self._insert(self.region_name, search_r=0.0015 * (1 + k))

    def labels_extra(self):
        return dict(task_hidden=self.hidden_task, depth_frac=self.depth_frac())


# ====================================================================== (e) connector / module into slot
class ConnectorInsert(PegInsert):
    family = "connector_insert"
    success_doc = "the connector module is seated >= 80% of the slot depth (yaw-critical thin slot)"
    hidden_doc = ("clearance 0.3-1.2 mm on a 10 mm thin axis, chamfer 0 (60 %) or 0.3-1 mm, friction 0.3-1.0, random slot "
                  "yaw, 48-64 mm long module: 1 deg of yaw error = 0.5 mm at its ends -> needs yaw wiggle + compliance")
    obj = "module"
    region_name = "slot"

    def geometry(self, rng, params):
        # long thin module: a yaw error e displaces its ends by half_x * e (1 deg -> 0.5 mm at 30 mm half length)
        return dict(half=(float(rng.uniform(0.024, 0.032)), 0.005, 0.035), sym=math.pi, depth=0.025, random_yaw=True)

    def build(self, scene, rng, params):
        params = dict(params)
        params.setdefault("clearance", float(rng.uniform(0.0003, 0.0012)))
        params.setdefault("chamfer", 0.0 if rng.random() < 0.6 else float(rng.uniform(0.0003, 0.001)))
        params.setdefault("mu", float(rng.uniform(0.3, 1.0)))
        super().build(scene, rng, params)

    def make_object(self, scene, rng, g, ppos, top, mass, mu):
        hx, hy, hz = g["half"]
        scene.part(self.obj, "box", (hx, hy, hz), (ppos[0], ppos[1], top + hz + 0.001), yaw=rng.uniform(-1.2, 1.2),
                   mass=mass, mu=mu, mu_grip=float(rng.uniform(0.5, 1.0)), restitution=0.05, timeconst=0.004,
                   kind="module", rgba=(0.2, 0.5, 0.25, 1))

    def scripted(self):
        if self.held is None:
            if self.success():
                return None
            return self._pick(self.obj)
        k = self.n_calls // 2
        return self._insert(self.region_name, search_r=0.001 * (1 + k), wiggle=0.01 * (1 + k))


# ====================================================================== (e/g) tool return into a holder
class ToolReturn(PegInsert):
    family = "tool_return"
    success_doc = "the tool's shank sits >= 80% of the holder depth inside its holder socket"
    hidden_doc = ("tool head mass 0.1-0.9 kg with a hidden left/right mass imbalance (CoM torque in the grasp), part<->gripper "
                  "friction 0.3-1.0, holder clearance 1-3 mm: low grip force lets the tool rotate in the grasp -> tilt jam")
    obj = "tool"
    region_name = "holder"
    probe_kw = dict(max_open=0.05, grip_max=100.0, pad_solref=0.003, finger_half=0.025)

    def geometry(self, rng, params):
        return dict(half=None, sym=math.pi / 2, depth=0.03)

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.3
        s = 0.008
        clr = float(params.get("clearance", rng.uniform(0.001, 0.003)))
        ch = float(params.get("chamfer", rng.uniform(0.0, 0.0015)))
        mu = float(rng.uniform(0.2, 0.8))
        depth = 0.03
        hpos = (0.18 + rng.uniform(-0.03, 0.03), 0.06 + rng.uniform(-0.03, 0.03))
        ztop = top + depth
        scene.socket((hpos[0], hpos[1], ztop), (s + clr / 2, s + clr / 2), depth, outer=0.025, chamfer=ch, name="holder")
        scene.region("holder", (hpos[0], hpos[1], ztop - depth / 2), (s + clr / 2, s + clr / 2, depth / 2),
                     support_z=ztop, depth=depth, sym=math.pi / 2)
        # cart cup (1.6 mm clearance, 40 mm deep) where the tool starts standing
        cpos = (-0.2 + rng.uniform(-0.03, 0.03), 0.0 + rng.uniform(-0.04, 0.04))
        scene.socket((cpos[0], cpos[1], top + 0.04), (s + 0.0008, s + 0.0008), 0.04, outer=0.02, name="cup")
        # head: two visually identical halves; the mass split between them is HIDDEN (CoM offset up to 17.5 mm)
        imb = float(params.get("head_imbalance", rng.uniform(0.0, 0.5)))
        m_head = float(params.get("head_mass", rng.uniform(0.1, 0.9)))
        m_shank = float(rng.uniform(0.05, 0.12))
        mu_grip = float(params.get("mu_grip", rng.uniform(0.3, 1.0)))
        boxes = [((0, 0, 0), (s, s, 0.06)), ((0.0175, 0, 0.07), (0.0175, 0.012, 0.01)),
                 ((-0.0175, 0, 0.07), (0.0175, 0.012, 0.01))]
        lo_z, hi_z = -0.06, 0.08
        zc = top + 0.001 + (hi_z - lo_z) / 2
        scene.compound(self.obj, boxes, (cpos[0], cpos[1], zc), yaw=float(rng.uniform(-0.3, 0.3)), mass=m_head + m_shank,
                       mu=mu, mu_grip=mu_grip, timeconst=0.004, kind="tool", rgba=(0.8, 0.15, 0.1, 1),
                       mass_split=[m_shank, m_head * (0.5 + imb), m_head * (0.5 - imb)],
                       extra_hidden=dict(head_mass=m_head, head_imbalance=imb, com_offset=m_head * imb * 0.035 / (m_head + m_shank),
                                         crush_n=fragility(rng, 0.5, 20, 60, params)))
        e = imb * 0.035
        self.scene.entities[self.obj].shape = "box"
        self.hidden_task = dict(clearance=clr, chamfer=ch, mu=mu, head_mass=m_head, head_imbalance=imb, mu_grip=mu_grip)
        self.depth = depth
        self.ztop = ztop
        self.budget = int(params.get("budget", 8))
        self.shank_from_centre = (hi_z - lo_z) / 2     # shank bottom = centre z - this (CAD)

    def depth_frac(self):
        p, q = self.true_pose(self.obj)
        if not self.in_region_true(self.obj, self.scene.regions[self.region_name], margin=0.02, zcheck=False):
            return 0.0
        bottom = p[2] - self.shank_from_centre
        return float(np.clip((self.ztop - bottom) / self.depth, 0, 1.5))

    def _pick(self, eid, rng=None, **kw):
        # grasp the shank below the head, above the cup rim: TCP at shank bottom + 5 cm (pads 5 cm tall)
        kw.setdefault("depth", 0.29)
        return super()._pick(eid, rng, **kw)

    def candidates(self, n, rng):
        if self.held is None:
            return [self._pick(self.obj, rng, depth=float(rng.uniform(0.15, 0.29))) for _ in range(n)]
        return [self._insert(self.region_name, rng) for _ in range(n)]

    def scripted(self):
        if self.held is None:
            if self.success():
                return None
            return self._pick(self.obj, grip_force=25.0)
        k = self.n_calls // 2
        return self._insert(self.region_name, search_r=0.0015 * (1 + k))


# ====================================================================== fixture placement (+-2 mm)
class FixturePlace(_FineBase):
    family = "fixture_place"
    station = "machine_table"
    tol_xy = 0.002
    tol_yaw = math.radians(3.0)
    success_doc = "the fixture rests on the machine table with its centre within 2 mm of the mark and yaw within 3 deg"
    hidden_doc = ("fixture mass 0.3-4 kg, fixture<->gripper friction 0.2-1.0 (in-hand slip/rotation during the carry), "
                  "fixture<->table friction 0.15-0.8 (slide after release): sets carry speed, grip force and release height")

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.2
        hx, hy, hz = rng.uniform(0.03, 0.045), rng.uniform(0.022, 0.035), rng.uniform(0.02, 0.03)
        mass = float(params.get("mass", np.exp(rng.uniform(np.log(0.3), np.log(4.0)))))
        mu = float(params.get("mu", rng.uniform(0.15, 0.8)))
        mu_grip = float(params.get("mu_grip", rng.uniform(0.2, 1.0)))
        p0 = (-0.2 + rng.uniform(-0.05, 0.05), rng.uniform(-0.08, 0.08))
        scene.part("fixture", "box", (hx, hy, hz), (p0[0], p0[1], top + hz + 0.001), yaw=rng.uniform(-0.5, 0.5),
                   mass=mass, mu=mu, mu_grip=mu_grip, restitution=0.05, timeconst=0.004, kind="fixture_block",
                   extra_hidden=dict(crush_n=fragility(rng, 0.5, 25, 80, params)),
                   rgba=(0.3, 0.35, 0.75, 1))
        mx, my = 0.18 + rng.uniform(-0.04, 0.04), 0.05 + rng.uniform(-0.05, 0.05)
        myaw = float(rng.uniform(-0.4, 0.4))
        scene.region("mark", (mx, my, top + hz), (self.tol_xy, self.tol_xy, hz), yaw=myaw, support_z=top, sym=math.pi)
        # visual mark: outline corners
        for sx in (-1, 1):
            for sy in (-1, 1):
                c, s = math.cos(myaw), math.sin(myaw)
                lx, ly = sx * hx, sy * hy
                scene.marker((mx + c * lx - s * ly, my + s * lx + c * ly), (0.006, 0.006), top, rgba=(0.9, 0.9, 0.1, 0.8))
        self.hidden_task = dict(mass=mass, mu=mu, mu_grip=mu_grip)
        self.budget = int(params.get("budget", 8))

    def errors(self):
        p, q = self.true_pose("fixture")
        reg = self.scene.regions["mark"]
        e_xy = float(np.hypot(p[0] - reg["center"][0], p[1] - reg["center"][1]))
        dy = wrap(quat_yaw(q) - reg["yaw"])
        dy = (dy + math.pi / 2) % math.pi - math.pi / 2
        up = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        return e_xy, abs(dy), up, p[2]

    def success(self):
        e, y, up, z = self.errors()
        hz = self.scene.entities["fixture"].half[2]
        return (e <= self.tol_xy and y <= self.tol_yaw and up > 0.99 and abs(z - (self.surface_z + hz)) < 0.003
                and self.held is None)

    def progress(self):
        e, y, up, z = self.errors()
        return float(np.clip(1 - max(e - self.tol_xy, 0) / 0.2, 0, 1) * (1.0 if y <= self.tol_yaw else 0.8))

    def fatal(self):
        return self.metrics_acc["drops"] > 0

    def _place(self, rng=None, **kw):
        p = dict(dx=0.0, dy=0.0, dyaw=0.0, release_h=0.002, speed=0.3, lower_speed=0.05, align=1, correct=1)
        if rng is not None:
            p.update(release_h=float(rng.uniform(0.0, 0.02)), speed=float(rng.uniform(0.1, 0.6)),
                     lower_speed=float(rng.uniform(0.02, 0.2)), dx=float(rng.normal(0, 0.001)),
                     dy=float(rng.normal(0, 0.001)))
        p.update(kw)
        return SkillCall("place", p, target="mark")

    def _est_err(self):
        est = self.est_pose_avg("fixture", window=0.5)
        if est is None:
            return None
        reg = self.scene.regions["mark"]
        return float(np.hypot(est[0][0] - reg["center"][0], est[0][1] - reg["center"][1]))

    def candidates(self, n, rng):
        if self.held is None:
            return [self._pick("fixture", rng, depth=float(rng.uniform(-0.2, 0.4))) for _ in range(n)]
        return [self._place(rng) for _ in range(n)]

    def scripted(self):
        if self.held is not None:
            return self._place()
        e = self._est_err()
        if self.n_calls > 0 and e is not None and e < 0.0012:
            return None        # believes it is done
        return self._pick("fixture", depth=0.0, grip_force=30.0)

    def labels_extra(self):
        e, y, up, z = self.errors()
        return dict(task_hidden=self.hidden_task, err_xy=e, err_yaw=y)


# ====================================================================== (d) kitting into tight pockets
class KittingPocket(_FineBase):
    family = "kitting_pocket"
    station = "kitting_bench"
    fine_sensors = False          # kitting cells use a wide-FOV tracker: 1-3 mm noise vs 1-3 mm clearance
    success_doc = "every part sits fully seated in its own kit-tray pocket (centre inside the pocket, bottom within 3 mm of the pocket floor)"
    hidden_doc = ("pocket clearance 1-3 mm (per pocket), part<->pocket friction 0.2-0.9, part mass incl. heavy outliers, "
                  "part<->gripper friction: a plain place lands on the rim when clearance < placement error -> compliant insert")

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.22
        k = int(params.get("n_parts", rng.integers(2, 4)))
        # source tray (open box) on the left
        sx, sy = -0.22 + rng.uniform(-0.02, 0.02), 0.0 + rng.uniform(-0.03, 0.03)
        sfloor = scene.open_box((sx, sy, top), (0.11, 0.1), 0.04, rgba=(0.2, 0.3, 0.55, 1), prefix="src")
        # kit tray: base slab + pockets (sockets) in a row
        tx, ty = 0.16 + rng.uniform(-0.02, 0.02), 0.05 + rng.uniform(-0.03, 0.03)
        slab_t = 0.01
        scene.static_box((tx, ty, top + slab_t / 2), (0.14, 0.07, slab_t / 2), rgba=(0.15, 0.15, 0.17, 1))
        depth = 0.015
        self.pockets = {}
        xs = np.linspace(-0.08, 0.08, k) if k > 1 else [0.0]
        for i in range(k):
            shape = "box" if rng.random() < 0.6 else "cylinder"
            hx = float(rng.uniform(0.012, 0.02))
            hy = hx if shape == "cylinder" else float(rng.uniform(0.012, 0.02))
            hz = float(rng.uniform(0.018, 0.026))
            clr = float(params.get("clearance", rng.uniform(0.001, 0.003)))
            mu = float(rng.uniform(0.2, 0.9))
            heavy = rng.random() < 0.2
            mass = float(8 * hx * hy * hz * rng.uniform(1500, 7800) * (rng.uniform(3, 6) if heavy else 1.0))
            pid = f"part{i}"
            px, py = tx + xs[i], ty
            ztop = top + slab_t + depth
            scene.socket((px, py, ztop), (hx + clr / 2, hy + clr / 2), depth, outer=0.012, chamfer=0.0,
                         rgba=(0.3, 0.3, 0.33, 1), name=f"pocket{i}")
            sym = math.pi if shape == "box" else 2 * math.pi
            scene.region(f"pocket_{pid}", (px, py, ztop - depth / 2), (hx + clr / 2, hy + clr / 2, depth / 2),
                         support_z=ztop, depth=depth, sym=sym, part=pid)
            ox, oy = sx + rng.uniform(-0.06, 0.06), sy + (i - (k - 1) / 2) * 0.06
            scene.part(pid, shape, (hx, hy, hz), (ox, oy, sfloor + hz + 0.002), yaw=rng.uniform(-0.8, 0.8), mass=mass,
                       mu=mu, mu_grip=float(rng.uniform(0.3, 1.0)), restitution=0.05, timeconst=0.004, kind="kit_part",
                       extra_hidden=dict(clearance=clr, heavy=bool(heavy), crush_n=fragility(rng, 0.3, 15, 45, params)))
            self.pockets[pid] = dict(region=f"pocket_{pid}", floor=top + slab_t, clearance=clr)
        self.budget = int(params.get("budget", 5 * k))

    def seated(self, pid):
        p, q = self.true_pose(pid)
        reg = self.scene.regions[self.pockets[pid]["region"]]
        hz = self.scene.entities[pid].half[2]
        inside = self.in_region_true(pid, reg, margin=0.001, zcheck=False)
        return bool(inside and p[2] - hz <= self.pockets[pid]["floor"] + 0.003)

    def success(self):
        return self.held is None and all(self.seated(p) for p in self.pockets)

    def progress(self):
        return sum(self.seated(p) for p in self.pockets) / len(self.pockets)

    def fatal(self):
        return self.metrics_acc["drops"] > 0

    def _est_seated(self, pid):
        est = self.est_pose(pid)
        if est is None:
            return False
        reg = self.scene.regions[self.pockets[pid]["region"]]
        h = self.est_size(pid)
        inside = abs(est[0][0] - reg["center"][0]) < 0.004 + reg["half"][0] and abs(est[0][1] - reg["center"][1]) < 0.004 + reg["half"][1]
        return bool(inside and est[0][2] - h[2] <= self.pockets[pid]["floor"] + 0.004)

    def _todo(self):
        return [p for p in self.pockets if not self._est_seated(p)]

    def candidates(self, n, rng):
        if self.held is not None:
            return [self._insert(self.pockets[self.held]["region"], rng) for _ in range(n)]
        todo = self._todo() or list(self.pockets)
        return [self._pick(todo[i % len(todo)], rng) for i in range(n)]

    def scripted(self):
        if self.held is not None:
            k = self.fail_count.get(self.pockets[self.held]["region"], 0)
            return self._insert(self.pockets[self.held]["region"], search_r=0.003 * (1 + k), search_hz=0.7,
                                down_force=4.0, compliance=0.3, timeout=8.0)
        todo = self._todo()
        if not todo:
            return None
        return self._pick(todo[0], grip_force=20.0)


# ====================================================================== (g) machine tending: drawer
class DrawerOpen(_FineBase):
    family = "drawer_open"
    station = "machine_tending"
    timestep = 0.002
    success_doc = "the tool-cabinet drawer is pulled open to >= 85% of its 0.20 m travel"
    hidden_doc = ("drawer slide frictionloss 2-40 N (breakaway), damping 5-60 N s/m, contents mass 1-12 kg, handle<->"
                  "gripper friction 0.3-1.0: pull force needed vs grip capacity 2*mu_g*F_grip -> grip force / pull speed")

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.3
        self.travel = 0.2
        fl = float(params.get("frictionloss", np.exp(rng.uniform(np.log(2), np.log(40)))))
        dmp = float(params.get("damping", rng.uniform(5, 60)))
        mass = float(params.get("mass", rng.uniform(1, 12)))
        mug = float(params.get("mu_grip", rng.uniform(0.3, 1.0)))
        pos = (rng.uniform(-0.1, 0.1), 0.12 + rng.uniform(-0.03, 0.03), top + 0.004)
        ay = float(rng.uniform(-0.25, 0.25))
        scene.drawer("drawer", pos, (0.14, 0.1), 0.07, ay, self.travel, damping=dmp, frictionloss=fl, mass=mass,
                     handle_mu_grip=mug)
        self.scene.entities["drawer"].hidden["crush_n"] = fragility(rng, 1.0, 40, 120, params)   # handle tab limit
        self.hidden_task = dict(frictionloss=fl, damping=dmp, mass=mass, mu_grip=mug)
        self.budget = int(params.get("budget", 5))

    def q(self):
        return float(self.d.qpos[self.m.jnt_qposadr[self.m.joint("drawer_slide").id]])

    def goal_extra(self):
        ent = "door" if "door" in self.scene.entities else "drawer"
        thr = 1.2 if ent == "door" else 0.85 * self.travel
        info = self.scene.entities[ent]
        return dict(joint_goal={ent: dict(min=thr, unit="rad" if ent == "door" else "m")},
                    cad={k: v for k, v in info.meta.items()})

    def success(self):
        return self.q() >= 0.85 * self.travel

    def progress(self):
        return float(np.clip(self.q() / (0.85 * self.travel), 0, 1))

    def _open(self, rng=None, **kw):
        p = dict(grip_force=40.0, travel=0.2, speed=0.12)
        if rng is not None:
            p.update(grip_force=float(rng.uniform(5, 100)), travel=float(rng.uniform(0.05, 0.22)),
                     speed=float(rng.uniform(0.03, 0.4)))
        p.update(kw)
        return SkillCall("open", p, target="drawer")

    def candidates(self, n, rng):
        return [self._open(rng) for _ in range(n)]

    def scripted(self):
        q = self.est_joint("drawer") or 0.0
        return self._open(travel=float(max(0.0, 0.2 - q) + 0.01))

    def labels_extra(self):
        return dict(task_hidden=self.hidden_task, joint=self.q())


# ====================================================================== (g) machine tending: hinged door
class DoorOpen(DrawerOpen):
    family = "door_open"
    success_doc = "the machine door is swung open to >= 1.2 rad (of 1.9) and stays there after release"
    hidden_doc = ("hinge frictionloss 0.2-4 N m, damping 0.1-3 N m s, door mass 1-6 kg, handle<->gripper friction: "
                  "the handle torque arm is 0.35 m so required pull force varies 1-20 N; low grip slips, fast swings "
                  "overshoot / rebound off the stop")

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.42
        fl = float(params.get("frictionloss", np.exp(rng.uniform(np.log(0.2), np.log(4.0)))))
        dmp = float(params.get("damping", rng.uniform(0.1, 3.0)))
        mass = float(params.get("mass", rng.uniform(1.0, 6.0)))
        mug = float(params.get("mu_grip", rng.uniform(0.3, 1.0)))
        width, height = 0.36, 0.3
        hx = 0.2 + rng.uniform(-0.03, 0.03)
        hy = 0.2 + rng.uniform(-0.03, 0.03)
        hz = top + 0.02 + height / 2
        # machine housing behind the door (static), opening framed
        scene.static_box((hx - width / 2, hy + 0.2, top + 0.2), (width / 2 + 0.05, 0.01, 0.2), rgba=(0.85, 0.85, 0.8, 1))
        scene.static_box((hx - width - 0.04, hy + 0.1, top + 0.2), (0.01, 0.1, 0.2), rgba=(0.85, 0.85, 0.8, 1))
        scene.static_box((hx + 0.03, hy + 0.1, top + 0.2), (0.01, 0.1, 0.2), rgba=(0.85, 0.85, 0.8, 1))
        scene.hinge_door("door", (hx, hy, hz), width, height, 0.02, axis=(0, 0, 1), closed_yaw=math.pi, damping=dmp,
                         frictionloss=fl, mass=mass, rng_open=(0.0, 1.9),
                         handle_off=(width - 0.05, 0.01 + 0.03, height / 2 - 0.04), handle_mu_grip=mug)
        self.scene.entities["door"].hidden["crush_n"] = fragility(rng, 1.0, 20, 60, params)     # handle tab limit
        self.hidden_task = dict(frictionloss=fl, damping=dmp, mass=mass, mu_grip=mug)
        self.budget = int(params.get("budget", 5))
        self.travel = 1.9

    def q(self):
        return float(self.d.qpos[self.m.jnt_qposadr[self.m.joint("door_hinge").id]])

    def success(self):
        return self.q() >= 1.2

    def progress(self):
        return float(np.clip(self.q() / 1.2, 0, 1))

    def _open(self, rng=None, **kw):
        p = dict(grip_force=20.0, travel=1.4, speed=0.15)
        if rng is not None:
            p.update(grip_force=float(rng.uniform(5, 100)), travel=float(rng.uniform(0.3, 1.6)),
                     speed=float(rng.uniform(0.05, 0.5)))
        p.update(kw)
        return SkillCall("open", p, target="door")

    def scripted(self):
        q = self.est_joint("door") or 0.0
        return self._open(travel=float(np.clip(1.45 - q, 0.1, 1.6)))
