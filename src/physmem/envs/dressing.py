"""Site dressing (envs-r6): a procedural warehouse "site" layer around any station, after the CommandAGI TalOS
warehouse (working/home+org-talos/worlds/talos-warehouse and its demo bays g1-h1-aisle, depalletise-kitting,
g1-tote-picking).

PORTED from TalOS (numbers, layouts, colours; their props are NVIDIA SimReady .glb meshes or CAD sketches, no MJCF
exists, so every prop here is RE-MODELLED from MjSpec primitives):
  * EUR pallet 1200 x 800 x 144 (EN 13698-1: 22 mm deck boards, 78 mm blocks, bottom boards) - depalletise-kitting
  * KLT totes VDA 4314 300 x 400 x 147 and VDA 4322 396 x 297 x 214, blue #1f5fa8 / grey #7d8288 - all three bays
  * KLT crates 600 x 400 x 253 in layers on pallets - g1-tote-picking loaded pallets
  * selective pallet racking: bays 2.7-4.0 m, 1.08 m deep, beam tops 0.506 / 1.65 / 2.509 m, blue uprights
    #2f5d8c, orange beams, galvanised deck panels - g1-h1-aisle layout.ts / depalletise-kitting racking
  * gravity roller conveyor 480 mm wide, d50 rollers at 100 mm pitch, tops 0.80-1.04 m; belt conveyor top 0.80 m,
    rubber #1b1c1e, aluminium rails - depalletise-kitting / g1-tote-picking
  * floor: power-floated concrete #9b9890 (lighter here), 100 mm safety-yellow #f2c200 lane lines, walkway green
    #3f7d4e; walls in a precast band + cladding, HEA columns steel-blue, high-bay LED lights every ~4 m
NEW here: packing table (1.6 x 0.8 m, ESD top, upper shelf + light bar), cardboard clutter, lighting rig.

Physics: props a robot could touch (within `touch_m` of the station footprint or a robot mount) are collision-real
static geoms (obstacles and tracker occluders, group 0); the background is visual-only (contype = conaffinity = 0,
geom group 2, which the tracker's line-of-sight test ignores). Nothing is placed inside the station footprint or
within `keepout_m` of any robot mount. Layout, clutter, colours and lighting are randomised from a SEPARATE rng
(seed_of(task, "dress")), so a dressed episode has exactly the same task randomisation as the undressed one.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .scene import G, Scene, quat_z

VIS_GROUP = 2          # visual-only geoms (ignored by the tracker's ray test)
WOOD = (0.79, 0.64, 0.42, 1)
WOOD_D = (0.66, 0.52, 0.33, 1)
KLT_BLUE = (0.12, 0.37, 0.66, 1)
KLT_GREY = (0.49, 0.51, 0.53, 1)
CRATE_BLUE = (0.16, 0.40, 0.70, 1)
UPRIGHT = (0.18, 0.36, 0.55, 1)
BEAM = (0.93, 0.45, 0.10, 1)
GALV = (0.72, 0.74, 0.76, 1)
ALU = (0.79, 0.80, 0.82, 1)
RUBBER = (0.11, 0.11, 0.12, 1)
YELLOW = (0.95, 0.76, 0.0, 1)
GREEN = (0.25, 0.49, 0.31, 1)
CARDBOARD = (0.72, 0.56, 0.38, 1)
WALL = (0.83, 0.82, 0.80, 1)
BAND = (0.24, 0.29, 0.34, 1)
CEIL = (0.77, 0.78, 0.77, 1)


class Dresser:
    def __init__(self, scene: Scene, rng, station_box, mounts, touch_m=0.9, keepout_m=0.75):
        self.s = scene
        self.rng = rng
        self.box = station_box            # (xmin, ymin, xmax, ymax) of the existing station (world)
        self.mounts = [np.asarray(p[:2], float) for p in mounts]
        self.touch_m = touch_m
        self.keepout_m = keepout_m
        self.placed = []                  # footprints of dressing props: (xmin, ymin, xmax, ymax)
        self.n_coll = 0
        self.n_vis = 0
        self._mat_cache = {}

    # ------------------------------------------------------------------ materials
    def mat(self, name, rgba, tex=None, rough=0.6, spec=0.2, refl=0.0, repeat=(1, 1)):
        if name in self._mat_cache:
            return self._mat_cache[name]
        sp = self.s.spec
        m = sp.add_material(name="dr_" + name)
        m.rgba = list(rgba)
        m.specular = spec
        m.shininess = 1.0 - rough
        m.reflectance = refl
        if tex is not None:
            t = sp.add_texture(name="drt_" + name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                               builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=list(tex[0])[:3], rgb2=list(tex[1])[:3],
                               mark=mujoco.mjtMark.mjMARK_RANDOM, random=tex[2], markrgb=list(tex[1])[:3],
                               width=512, height=512)
            tx = list(m.textures)
            tx[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = t.name
            m.textures = tx
            m.texrepeat = list(repeat)
            m.texuniform = True
        self._mat_cache[name] = m.name
        return m.name

    # ------------------------------------------------------------------ geometry
    def near(self, x0, y0, x1, y1):
        """True if a footprint is within touch distance of the station or a robot mount (-> collision-real)."""
        bx0, by0, bx1, by1 = self.box
        dx = max(bx0 - x1, x0 - bx1, 0.0)
        dy = max(by0 - y1, y0 - by1, 0.0)
        if math.hypot(dx, dy) < self.touch_m:
            return True
        for m in self.mounts:
            ddx = max(x0 - m[0], m[0] - x1, 0.0)
            ddy = max(y0 - m[1], m[1] - y1, 0.0)
            if math.hypot(ddx, ddy) < self.touch_m + 0.3:
                return True
        return False

    def free(self, x0, y0, x1, y1, margin=0.05):
        """footprint outside the station box (+margin), outside every mount keep-out, not overlapping other props."""
        bx0, by0, bx1, by1 = self.box
        if not (x1 < bx0 - margin or x0 > bx1 + margin or y1 < by0 - margin or y0 > by1 + margin):
            return False
        for m in self.mounts:
            ddx = max(x0 - m[0], m[0] - x1, 0.0)
            ddy = max(y0 - m[1], m[1] - y1, 0.0)
            if math.hypot(ddx, ddy) < self.keepout_m:
                return False
        for (a0, b0, a1, b1) in self.placed:
            if not (x1 < a0 - margin or x0 > a1 + margin or y1 < b0 - margin or y0 > b1 + margin):
                return False
        return True

    def geom(self, pos, half, rgba=None, mat=None, yaw=0.0, solid=True, gtype=G.mjGEOM_BOX, quat=None, mu=0.6):
        g = self.s.spec.worldbody.add_geom(type=gtype, size=list(half), pos=list(pos),
                                           quat=list(quat if quat is not None else quat_z(yaw)),
                                           friction=[mu, 0.005, 0.0001], name=self.s._name("dress"))
        if mat is not None:
            g.material = mat
            g.rgba = [0.5, 0.5, 0.5, 1]       # MuJoCo default: a geom rgba other than this overrides the material
        elif rgba is not None:
            g.rgba = list(rgba)
        if not solid:
            g.contype = 0
            g.conaffinity = 0
            g.group = VIS_GROUP
            self.n_vis += 1
        else:
            self.n_coll += 1
        return g

    def fp(self, cx, cy, hx, hy, yaw):
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        ex, ey = c * hx + s * hy, s * hx + c * hy
        return (cx - ex, cy - ey, cx + ex, cy + ey)

    # ------------------------------------------------------------------ props
    def pallet(self, cx, cy, z0=0.0, yaw=0.0, solid=None):
        """EUR pallet 1.2 x 0.8 x 0.144 (long side along local x). Returns its top height."""
        f = self.fp(cx, cy, 0.6, 0.4, yaw)
        solid = self.near(*f) if solid is None else solid
        c, s = math.cos(yaw), math.sin(yaw)
        P = lambda x, y, z: (cx + c * x - s * y, cy + s * x + c * y, z)
        wood = self.mat("wood", WOOD, tex=(WOOD, WOOD_D, 0.15), repeat=(4, 4))
        for y in (-0.4 + 0.05, -0.4 + 0.226, 0.0, 0.4 - 0.226, 0.4 - 0.05):        # 5 top boards (100/145 mm)
            self.geom(P(0, y, z0 + 0.144 - 0.011), (0.6, 0.05 if abs(y) > 0.3 else 0.0725, 0.011), mat=wood, yaw=yaw,
                      solid=solid)
        for x in (-0.6 + 0.05, 0.0, 0.6 - 0.05):                                    # 9 blocks
            for y in (-0.35, 0.0, 0.35):
                self.geom(P(x, y, z0 + 0.022 + 0.039), (0.05, 0.05 if abs(y) > 0.1 else 0.0725, 0.039), mat=wood,
                          yaw=yaw, solid=solid)
        for y in (-0.35, 0.0, 0.35):                                                # 3 bottom boards
            self.geom(P(0, y, z0 + 0.011), (0.6, 0.05, 0.011), mat=wood, yaw=yaw, solid=solid)
        self.placed.append(f)
        return z0 + 0.144

    def klt(self, cx, cy, z0, yaw=0.0, size=(0.396, 0.297, 0.214), color=None, solid=None, hollow=True):
        lx, ly, h = size
        f = self.fp(cx, cy, lx / 2, ly / 2, yaw)
        solid = self.near(*f) if solid is None else solid
        col = color or (KLT_BLUE if self.rng.random() < 0.8 else KLT_GREY)
        m = self.mat("klt_" + ("b" if col == KLT_BLUE else "g"), col, rough=0.5, spec=0.3)
        c, s = math.cos(yaw), math.sin(yaw)
        P = lambda x, y, z: (cx + c * x - s * y, cy + s * x + c * y, z)
        if not hollow:
            self.geom(P(0, 0, z0 + h / 2), (lx / 2, ly / 2, h / 2), mat=m, yaw=yaw, solid=solid)
            return z0 + h
        t = 0.004
        self.geom(P(0, 0, z0 + t), (lx / 2, ly / 2, t), mat=m, yaw=yaw, solid=solid)
        for sx in (-1, 1):
            self.geom(P(sx * (lx / 2 - t), 0, z0 + h / 2), (t, ly / 2, h / 2), mat=m, yaw=yaw, solid=solid)
            self.geom(P(0, sx * (ly / 2 - t), z0 + h / 2), (lx / 2 - 2 * t, t, h / 2), mat=m, yaw=yaw, solid=solid)
        return z0 + h

    def crate_stack(self, cx, cy, yaw=0.0, layers=None):
        """EUR pallet with KLT crates 600 x 400 x 253 in layers (TalOS loaded pallet)."""
        top = self.pallet(cx, cy, 0.0, yaw)
        f = self.fp(cx, cy, 0.6, 0.4, yaw)
        solid = self.near(*f)
        layers = int(self.rng.integers(1, 4)) if layers is None else layers
        c, s = math.cos(yaw), math.sin(yaw)
        m = self.mat("crate", CRATE_BLUE, rough=0.55)
        md = self.mat("crate_d", (0.10, 0.28, 0.52, 1))
        for l in range(layers):
            for a in (-0.3, 0.3):
                for b in (-0.2, 0.2):
                    if l == layers - 1 and self.rng.random() < 0.25:
                        continue
                    x, y = cx + c * a - s * b, cy + s * a + c * b
                    z = top + l * 0.254 + 0.1265
                    self.geom((x, y, z), (0.298, 0.198, 0.1265), mat=m, yaw=yaw, solid=solid)
                    self.geom((x, y, z + 0.1266), (0.27, 0.17, 0.0005), mat=md, yaw=yaw, solid=False)   # dark rim/top

    def carton(self, cx, cy, z0, yaw=0.0, half=None, solid=None):
        h = half or tuple(self.rng.uniform([0.12, 0.09, 0.08], [0.25, 0.18, 0.18]))
        f = self.fp(cx, cy, h[0], h[1], yaw)
        solid = self.near(*f) if solid is None else solid
        self.geom((cx, cy, z0 + h[2]), h, mat=self.mat("carton", CARDBOARD, rough=0.85, spec=0.05), yaw=yaw, solid=solid)
        self.geom((cx, cy, z0 + 2 * h[2] + 0.0005), (h[0] * 0.98, 0.025, 0.0005), rgba=(0.83, 0.75, 0.6, 1), yaw=yaw,
                  solid=False)
        return z0 + 2 * h[2]

    def rack_bay(self, cx, cy, yaw, bay=2.7, depth=1.08, levels=(0.506, 1.65, 2.509), height=3.0, load=0.75,
                 solid=None, skip_level0=False):
        """One selective-racking bay; local x along the beams, local y = depth (front at -y). Loaded with KLTs,
        crates on pallets and cartons."""
        f = self.fp(cx, cy, bay / 2 + 0.05, depth / 2, yaw)
        solid = self.near(*f) if solid is None else solid
        c, s = math.cos(yaw), math.sin(yaw)
        P = lambda x, y, z: (cx + c * x - s * y, cy + s * x + c * y, z)
        up, bm, dk = self.mat("upright", UPRIGHT, spec=0.4), self.mat("beam", BEAM, spec=0.3), self.mat("deck", GALV, spec=0.5)
        for sx in (-1, 1):
            for sy in (-1, 1):
                self.geom(P(sx * bay / 2, sy * (depth / 2 - 0.05), height / 2), (0.05, 0.05, height / 2), mat=up,
                          yaw=yaw, solid=solid)
            self.geom(P(sx * bay / 2, 0, 0.25), (0.04, depth / 2 - 0.05, 0.01), mat=up, yaw=yaw, solid=False)    # bracing
            self.geom(P(sx * bay / 2, 0, 1.4), (0.04, depth / 2 - 0.05, 0.01), mat=up, yaw=yaw, solid=False)
        for z in levels:
            for sy in (-1, 1):
                self.geom(P(0, sy * (depth / 2 - 0.05), z - 0.05), (bay / 2 - 0.05, 0.025, 0.05), mat=bm, yaw=yaw,
                          solid=solid)
            self.geom(P(0, 0, z + 0.0025), (bay / 2 - 0.05, depth / 2 - 0.03, 0.0025), mat=dk, yaw=yaw, solid=solid)
        for li, z in enumerate(levels):
            if li == 0 and skip_level0:
                continue
            kind = self.rng.choice(["klt", "klt", "crates", "cartons"])
            if kind == "klt":
                n = int((bay - 0.2) // 0.43)
                for k in range(n):
                    if self.rng.random() < load:
                        x = -bay / 2 + 0.3 + k * 0.43
                        self.klt(*P(x, -depth / 2 + 0.2, 0)[:2], z + 0.005, yaw=yaw + math.pi / 2, solid=solid,
                                 hollow=solid)
                        if self.rng.random() < 0.5:
                            self.klt(*P(x, depth / 2 - 0.25, 0)[:2], z + 0.005, yaw=yaw + math.pi / 2, solid=solid,
                                     hollow=False)
            elif kind == "crates":
                for x in (-bay / 4, bay / 4) if bay > 2.5 else (0.0,):
                    if self.rng.random() < load:
                        px, py, _ = P(x, 0, 0)
                        top = z + 0.005
                        ww = self.mat("wood", WOOD, tex=(WOOD, WOOD_D, 0.15), repeat=(4, 4))
                        self.geom((px, py, top + 0.072), (0.4, 0.6, 0.072), mat=ww, yaw=yaw, solid=solid)
                        for l in range(int(self.rng.integers(1, 3))):
                            self.geom((px, py, top + 0.144 + 0.1265 + l * 0.254), (0.398, 0.298, 0.1265),
                                      mat=self.mat("crate", CRATE_BLUE, rough=0.55), yaw=yaw, solid=solid)
            else:
                x = -bay / 2 + 0.35
                while x < bay / 2 - 0.3:
                    if self.rng.random() < load:
                        hh = tuple(self.rng.uniform([0.12, 0.12, 0.08], [0.2, 0.3, 0.2]))
                        px, py, _ = P(x, 0, 0)
                        self.carton(px, py, z + 0.005, yaw=yaw, half=hh, solid=solid)
                    x += 0.45
        self.placed.append(f)

    def roller_conveyor(self, x0, y0, x1, y1, top=0.8, width=0.48, solid=None, flat_top=True):
        """Gravity roller conveyor between two points (TalOS: 480 mm, d50 rollers at 100 mm pitch). With flat_top
        the rollers are visual and one flat collider sits at their top (parts rest on the roller tops)."""
        L = math.hypot(x1 - x0, y1 - y0)
        yaw = math.atan2(y1 - y0, x1 - x0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        f = self.fp(cx, cy, L / 2, width / 2 + 0.05, yaw)
        solid = self.near(*f) if solid is None else solid
        c, s = math.cos(yaw), math.sin(yaw)
        P = lambda x, y, z: (cx + c * x - s * y, cy + s * x + c * y, z)
        rl, fr = self.mat("roller", (0.82, 0.83, 0.85, 1), spec=0.8, rough=0.2), self.mat("alu", ALU, spec=0.6)
        q = tuple(np.asarray(_qmul(quat_z(yaw), (math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0))))
        n = int(L / 0.1)
        for k in range(n):
            x = -L / 2 + 0.05 + k * 0.1
            self.geom(P(x, 0, top - 0.025), (0.025, width / 2, 0), mat=rl, gtype=G.mjGEOM_CYLINDER, quat=q,
                      solid=False)
        if flat_top and solid:
            g = self.geom(P(0, 0, top - 0.004), (L / 2, width / 2, 0.004), solid=True, mu=0.4)
            g.rgba = [0, 0, 0, 0]
        for sy in (-1, 1):
            self.geom(P(0, sy * (width / 2 + 0.025), top - 0.03), (L / 2, 0.02, 0.045), mat=fr, yaw=yaw, solid=solid)
        k = 0
        while k * 1.2 <= L:
            for sy in (-1, 1):
                self.geom(P(-L / 2 + 0.05 + k * 1.2, sy * (width / 2), (top - 0.07) / 2), (0.025, 0.025, (top - 0.07) / 2),
                          mat=fr, yaw=yaw, solid=solid)
            k += 1
        self.placed.append(f)

    def belt_conveyor(self, x0, y0, x1, y1, top=0.8, width=0.36, solid=None):
        L = math.hypot(x1 - x0, y1 - y0)
        yaw = math.atan2(y1 - y0, x1 - x0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        f = self.fp(cx, cy, L / 2, width / 2 + 0.06, yaw)
        solid = self.near(*f) if solid is None else solid
        c, s = math.cos(yaw), math.sin(yaw)
        P = lambda x, y, z: (cx + c * x - s * y, cy + s * x + c * y, z)
        self.geom(P(0, 0, top - 0.02), (L / 2, width / 2, 0.02), mat=self.mat("belt", RUBBER, rough=0.9, spec=0.05),
                  yaw=yaw, solid=solid)
        for sy in (-1, 1):
            self.geom(P(0, sy * (width / 2 + 0.03), top + 0.015), (L / 2, 0.03, 0.05),
                      mat=self.mat("rail", (0.91, 0.89, 0.83, 1)), yaw=yaw, solid=solid)
        k = 0
        while k * 1.5 <= L:
            self.geom(P(-L / 2 + 0.1 + k * 1.5, 0, (top - 0.04) / 2), (0.03, width / 2, (top - 0.04) / 2),
                      mat=self.mat("alu", ALU, spec=0.6), yaw=yaw, solid=solid)
            k += 1
        self.placed.append(f)

    def packing_table(self, cx, cy, yaw=0.0, top=0.9, solid=None, upper=True):
        f = self.fp(cx, cy, 0.8, 0.4, yaw)
        solid = self.near(*f) if solid is None else solid
        c, s = math.cos(yaw), math.sin(yaw)
        P = lambda x, y, z: (cx + c * x - s * y, cy + s * x + c * y, z)
        tp = self.mat("esd", (0.38, 0.47, 0.44, 1), tex=((0.38, 0.47, 0.44), (0.33, 0.42, 0.39), 0.08), repeat=(3, 2))
        fr = self.mat("alu", ALU, spec=0.6)
        self.geom(P(0, 0, top - 0.015), (0.8, 0.4, 0.015), mat=tp, yaw=yaw, solid=solid)
        for sx in (-1, 1):
            for sy in (-1, 1):
                self.geom(P(sx * 0.76, sy * 0.36, (top - 0.03) / 2), (0.02, 0.02, (top - 0.03) / 2), mat=fr, yaw=yaw,
                          solid=solid)
        if upper:
            for sx in (-1, 1):
                self.geom(P(sx * 0.76, 0.36, top + 0.45), (0.02, 0.02, 0.45), mat=fr, yaw=yaw, solid=solid)
            self.geom(P(0, 0.25, top + 0.55), (0.78, 0.13, 0.01), mat=self.mat("deck", GALV, spec=0.5), yaw=yaw,
                      solid=solid)
            self.geom(P(0, 0.3, top + 0.88), (0.7, 0.04, 0.012), rgba=(1, 1, 0.95, 1), yaw=yaw, solid=False)  # light bar
            for k in range(int(self.rng.integers(1, 4))):
                self.carton(*P(-0.6 + 0.4 * k, 0.25, 0)[:2], top + 0.56, yaw=yaw,
                            half=(0.11, 0.09, float(self.rng.uniform(0.05, 0.1))), solid=False)
        self.placed.append(f)

    def floor_and_hall(self, hall_half=(7.0, 5.5), height=8.0):
        sp = self.s.spec
        fl = sp.geom("floor") if hasattr(sp, "geom") else None
        floor_mat = self.mat("floor", (0.78, 0.77, 0.74, 1), tex=((0.80, 0.79, 0.76), (0.70, 0.69, 0.66), 0.05),
                             repeat=(6, 6), rough=0.55, spec=0.15, refl=0.03)
        if fl is not None:
            fl.material = floor_mat
            fl.rgba = [0.5, 0.5, 0.5, 1]
            fl.size = [hall_half[0], hall_half[1], 0.1]
        hx, hy = hall_half
        wall = self.mat("wall", WALL, rough=0.85, spec=0.05)
        band = self.mat("band", BAND, rough=0.7)
        for sx in (-1, 1):
            self.geom((sx * hx, 0, height / 2), (0.1, hy, height / 2), mat=wall, solid=False)
            self.geom((sx * (hx - 0.11), 0, 0.6), (0.01, hy, 0.6), mat=band, solid=False)
        for sy in (-1, 1):
            self.geom((0, sy * hy, height / 2), (hx, 0.1, height / 2), mat=wall, solid=False)
            self.geom((0, sy * (hy - 0.11), 0.6), (hx, 0.01, 0.6), mat=band, solid=False)
        self.geom((0, 0, height + 0.1), (hx, hy, 0.1), mat=self.mat("ceil", CEIL, rough=0.9, spec=0.02), solid=False)
        col = self.mat("column", UPRIGHT, spec=0.4)
        for x in np.arange(-hx + 2.0, hx - 1.0, 6.0):
            for y in (-hy + 0.4, hy - 0.4):
                self.geom((x, y, height / 2), (0.15, 0.15, height / 2), mat=col, solid=False)
        lamp = self.mat("lamp", (1, 1, 0.96, 1), spec=0.0)
        for x in np.arange(-hx + 1.5, hx, 3.0):
            for y in np.arange(-hy + 1.5, hy, 3.0):
                self.geom((x, y, height - 0.6), (0.23, 0.06, 0), mat=lamp, gtype=G.mjGEOM_CYLINDER, solid=False)

    def markings(self, aisles):
        """yellow lane lines (100 mm) along each aisle [(x0, y0, x1, y1, half_width)] + green walkways."""
        y_m = self.mat("yellow", YELLOW, rough=0.45, spec=0.1)
        for (x0, y0, x1, y1, hw) in aisles:
            L = math.hypot(x1 - x0, y1 - y0)
            yaw = math.atan2(y1 - y0, x1 - x0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            c, s = math.cos(yaw), math.sin(yaw)
            for sy in (-1, 1):
                self.geom((cx - s * sy * hw, cy + c * sy * hw, 0.0006), (L / 2, 0.05, 0.0006), mat=y_m, yaw=yaw,
                          solid=False)
        g_m = self.mat("green", GREEN, rough=0.5, spec=0.05)
        bx0, by0, bx1, by1 = self.box
        self.geom(((bx0 + bx1) / 2, by0 - 1.1, 0.0005), ((bx1 - bx0) / 2 + 1.0, 0.35, 0.0005), mat=g_m, solid=False)

    def lights(self, n_shadow=1):
        sp = self.s.spec
        # drop the scene's two default lights' influence: softer headlight, ceiling high-bay rig, warm/cool per seed
        sp.visual.headlight.ambient = [0.18, 0.18, 0.18]
        sp.visual.headlight.diffuse = [0.12, 0.12, 0.12]
        sp.visual.headlight.specular = [0.05, 0.05, 0.05]
        sp.visual.quality.shadowsize = 4096
        temp = self.rng.uniform(-0.06, 0.06)
        base = float(self.rng.uniform(0.22, 0.32))
        col = [base * (1 + temp), base, base * (1 - temp)]
        bx0, by0, bx1, by1 = self.box
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        k = 0
        for dx in (-2.0, 0.0, 2.0):
            for dy in (-1.5, 1.5):
                l = sp.worldbody.add_light(pos=[cx + dx, cy + dy, 6.5], dir=[0, 0, -1], diffuse=col,
                                           specular=[0.1, 0.1, 0.1], castshadow=(k < n_shadow and dx == 0.0),
                                           cutoff=60.0, exponent=2.0)
                k += 1
        for l in list(sp.worldbody.lights)[:2]:          # the scene's own two lights (added by Scene.floor)
            l.diffuse = [0.18, 0.18, 0.18]
            l.castshadow = False


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def station_box(scene):
    """footprint of everything the station built (static geoms + bodies), world xy."""
    pts = []
    for g in scene.spec.worldbody.geoms:
        if g.type == G.mjGEOM_PLANE:
            continue
        r = float(max(g.size[0], g.size[1] if len(g.size) > 1 else 0))
        pts += [(g.pos[0] - r, g.pos[1] - r), (g.pos[0] + r, g.pos[1] + r)]
    for b in scene.spec.worldbody.bodies:
        pts += [(b.pos[0] - 0.15, b.pos[1] - 0.15), (b.pos[0] + 0.15, b.pos[1] + 0.15)]
    for r in scene.regions.values():
        cx, cy, _ = r["center"]
        hx, hy, _ = r["half"]
        pts += [(cx - hx, cy - hy), (cx + hx, cy + hy)]
    P = np.array(pts) if pts else np.zeros((1, 2))
    # long conveyor plates extend far upstream: clip the box to +-3 m
    lo, hi = np.maximum(P.min(0), -3.0), np.minimum(P.max(0), 3.0)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def dress(env, scene, rng, robot_kind_mounts=None):
    """Add the site around the station of `env` (a StationEnv mid-reset, scene built). Returns a stats dict."""
    box = station_box(scene)
    mounts = [p for lst in scene.mounts.values() for (p, _) in lst]
    D = Dresser(scene, rng, box, mounts)
    hall = (float(rng.uniform(6.5, 8.5)), float(rng.uniform(5.0, 6.5)))
    D.floor_and_hall(hall)
    D.lights()
    layout = LAYOUTS.get(env.family, generic_layout)
    aisles = layout(env, D, rng) or []
    D.markings(aisles)
    background(D, rng, hall)
    return dict(collision_geoms=D.n_coll, visual_geoms=D.n_vis, station_box=box, hall=hall, layout=layout.__name__)


def background(D, rng, hall):
    """racking rows along the far walls, crate stacks and cartons scattered on free floor."""
    hx, hy = hall
    for y, yaw in ((hy - 0.9, 0.0), (-hy + 0.9, math.pi)):
        x = -hx + 1.8
        while x < hx - 1.5:
            bay = float(rng.choice([2.7, 3.6]))
            f = D.fp(x + bay / 2, y, bay / 2 + 0.05, 0.54, yaw)
            if D.free(*f):
                D.rack_bay(x + bay / 2, y, yaw, bay=bay)
            x += bay + 0.1
    for _ in range(int(rng.integers(4, 9))):
        for _try in range(20):
            x, y = rng.uniform(-hx + 1.0, hx - 1.0), rng.uniform(-hy + 2.0, hy - 2.0)
            yaw = float(rng.choice([0.0, math.pi / 2]))
            if D.free(*D.fp(x, y, 0.62, 0.42, yaw), margin=0.3):
                if rng.random() < 0.7:
                    D.crate_stack(x, y, yaw)
                else:
                    D.pallet(x, y, 0.0, yaw)
                    z = 0.144
                    for _k in range(int(rng.integers(1, 4))):
                        z = D.carton(x + rng.uniform(-0.2, 0.2), y + rng.uniform(-0.1, 0.1), z, yaw)
                break


def generic_layout(env, D, rng):
    bx0, by0, bx1, by1 = D.box
    return [(bx0 - 1.5, by0 - 0.8, bx1 + 1.5, by0 - 0.8, 1.2)]


# ====================================================================== per-family site layouts
def _hide(scene, name):
    """keep a station geom's collision (task semantics intact) but make it invisible: the dressing draws the
    real-looking prop in its place."""
    try:
        g = scene.spec.geom(name)
    except Exception:
        g = None
    if g is not None:
        g.rgba = [0, 0, 0, 0]
    return g


def _restyle(scene, name, mat):
    try:
        g = scene.spec.geom(name)
    except Exception:
        g = None
    if g is not None:
        g.material = mat
        g.rgba = [0.5, 0.5, 0.5, 1]


def site_depalletize(env, D, rng):
    """EUR pallet with the station's carton stack, an outfeed ROLLER conveyor, crate stacks and racking behind."""
    s = env.scene
    _hide(s, "pallet")                                     # collider (0.144 m deck) kept, real pallet drawn on it
    D.pallet(0.0, 0.25, 0.0, 0.0, solid=False)
    _hide(s, "outfeed_top")
    D.roller_conveyor(-0.6, -0.55, 0.6, -0.55, top=0.55, width=0.44, solid=False, flat_top=False)
    D.roller_conveyor(0.62, -0.55, 2.4, -0.55, top=0.55, width=0.44)      # continues out of the cell
    for x in (-1.6, 1.6):
        if D.free(*D.fp(x, 0.3, 0.62, 0.42, math.pi / 2), margin=0.15):
            D.crate_stack(x, 0.3, math.pi / 2)
    x = -2.7
    while x < 2.7:
        if D.free(*D.fp(x + 1.35, 1.75, 1.4, 0.54, 0.0)):
            D.rack_bay(x + 1.35, 1.75, 0.0, bay=2.7)
        x += 2.8
    return [(-3.5, -1.15, 3.5, -1.15, 1.0)]


def site_conveyor_sort(env, D, rng):
    """the driven belt fed by a roller conveyor, KLT-bin sort targets, racking with blue totes behind bin A."""
    D.roller_conveyor(-2.6, 0.0, -0.95, 0.0, top=0.8, width=0.3, solid=False, flat_top=False)
    x = -2.4
    while x < 2.4:
        if D.free(*D.fp(x + 1.35, 1.25, 1.4, 0.54, 0.0)):
            D.rack_bay(x + 1.35, 1.25, 0.0, bay=2.7)
        x += 2.8
    for x in (1.4, 2.3):
        if D.free(*D.fp(x, -1.1, 0.62, 0.42, 0.0), margin=0.15):
            D.crate_stack(x, -1.1, 0.0)
    return [(-3.5, -0.75, 3.5, -0.75, 0.6)]


def site_tote_transfer(env, D, rng):
    """the two-tier flow rack becomes a racking bay (uprights, orange beams, galvanised decks) loaded around the task
    totes; the cart becomes a roller-conveyor section; an aisle of racking behind. Humanoid-scale: the floor in front
    of the rack (G1 / TidyBot mounts) stays clear."""
    s = env.scene
    up, bm = D.mat("upright", UPRIGHT, spec=0.4), D.mat("beam", BEAM, spec=0.3)
    dk = D.mat("deck", GALV, spec=0.5)
    for nm in ("tier0", "tier1"):
        _restyle(s, nm, dk)
    for (y, z) in ((0.15, 0.45), (0.45, 0.85)):
        for sy in (-1, 1):
            D.geom((0.0, y + sy * 0.14, z - 0.045), (0.55, 0.02, 0.035), mat=bm, solid=True)   # 1 cm below the deck
    for x in (-0.58, 0.58):
        for y in (0.02, 0.58):
            D.geom((x, y, 1.0), (0.04, 0.04, 1.0), mat=up, solid=True)
    D.geom((0.0, 0.58, 1.95), (0.6, 0.02, 0.04), mat=bm, solid=False)
    _hide(s, "tote_cart")
    reg = s.regions["cart"]
    (ccx, ccy, _), (chx, chy, _), cz = reg["center"], reg["half"], reg["support_z"]
    if getattr(env, "humanoid_cell", False):       # roller section along y at the humanoid's right, leading away
        D.roller_conveyor(ccx, ccy - chy, ccx, ccy + chy, top=cz, width=2 * chx, solid=False, flat_top=False)
        D.roller_conveyor(ccx, ccy - chy - 0.02, ccx, ccy - chy - 1.6, top=cz, width=2 * chx)
    else:
        D.roller_conveyor(ccx - chx, ccy, ccx + chx, ccy, top=cz, width=2 * chy, solid=False, flat_top=False)
        D.roller_conveyor(ccx + chx + 0.02, ccy, ccx + chx + 1.75, ccy, top=cz, width=2 * chy)
    for x0 in (-3.2, 1.0):
        if D.free(*D.fp(x0 + 1.1, 1.35, 1.15, 0.54, 0.0)):
            D.rack_bay(x0 + 1.1, 1.35, 0.0, bay=2.2)
    return [(-3.5, -0.95, 3.5, -0.95, 0.9)]


def site_shelf_restock(env, D, rng):
    """the restock shelf becomes a racking bay (beams, deck, loaded upper level); a second racking run across the
    aisle, so a TidyBot drives a real 1.5 m aisle between two racks; the cart becomes a roller conveyor section."""
    s = env.scene
    sy, top = env.SHELF_Y, env.TOP
    up, bm = D.mat("upright", UPRIGHT, spec=0.4), D.mat("beam", BEAM, spec=0.3)
    _restyle(s, "shelf", D.mat("deck", GALV, spec=0.5))
    for sgn in (-1, 1):
        D.geom((0.42, sy + sgn * 0.13, top - 0.05), (0.55, 0.025, 0.04), mat=bm, solid=True)
    # uprights: none at the front-left corner next to the cart (an arm reaches past it to the cart's items)
    for x, yy in ((-0.13, sy + 0.13), (0.97, sy - 0.13), (0.97, sy + 0.13)):
        D.geom((x, yy, 1.0), (0.04, 0.04, 1.0), mat=up, solid=True)
    D.geom((0.42, sy, 1.42), (0.55, 0.15, 0.005), mat=D.mat("deck", GALV, spec=0.5), solid=True)
    for sgn in (-1, 1):
        D.geom((0.42, sy + sgn * 0.13, 1.37), (0.55, 0.025, 0.04), mat=bm, solid=True)
    for k in range(4):
        if rng.random() < 0.75:
            D.klt(-0.02 + 0.3 * k, sy, 1.425, yaw=math.pi / 2, size=(0.297, 0.2, 0.147), solid=True, hollow=False)
    _hide(s, "cart")
    D.roller_conveyor(env.CART_X - 0.18, sy, env.CART_X + 0.18, sy, top=top, width=0.3, solid=False, flat_top=False)
    lane = getattr(env, "lane_y", sy - 0.45)
    across = lane - 0.85 - 0.54
    x = -2.6
    while x < 2.6:
        if D.free(*D.fp(x + 1.25, across, 1.3, 0.54, math.pi)):
            D.rack_bay(x + 1.25, across, math.pi, bay=2.5)
        x += 2.6
    return [(-3.0, lane - 0.6, 3.0, lane - 0.6, 0.05), (-3.0, lane + 0.45, 3.0, lane + 0.45, 0.05)]


def site_kitting_pocket(env, D, rng):
    """the bench becomes a packing table (ESD top, upper shelf with a light bar at the back, cartons), with a supply
    roller conveyor and racking nearby."""
    s = env.scene
    _restyle(s, "bench_top", D.mat("esd", (0.38, 0.47, 0.44, 1), tex=((0.38, 0.47, 0.44), (0.33, 0.42, 0.39), 0.08),
                                   repeat=(3, 2)))
    fr = D.mat("alu", ALU, spec=0.6)
    for x in (-0.57, 0.57):
        D.geom((x, 0.38, 0.76 + 0.5), (0.02, 0.02, 0.5), mat=fr, solid=True)
    # the upper shelf (0.62 m above the table, at the back, outside the kitting motions) and its cartons are visual-only:
    # as colliders they hid the work surface from the station tracker (tried: an overhead tracker loses the part under
    # the gripper, a side tracker degraded the in-hand correction - both cut scripted success 0.42 -> 0.08-0.17)
    D.geom((0.0, 0.31, 0.76 + 0.62), (0.58, 0.09, 0.01), mat=D.mat("deck", GALV, spec=0.5), solid=False)
    D.geom((0.0, 0.33, 0.76 + 0.98), (0.5, 0.04, 0.012), rgba=(1, 1, 0.95, 1), solid=False)
    for k in range(int(rng.integers(2, 4))):
        D.carton(-0.45 + 0.35 * k, 0.31, 0.76 + 0.63, half=(0.1, 0.07, float(rng.uniform(0.04, 0.08))), solid=False)
    D.roller_conveyor(-2.6, 0.9, 2.6, 0.9, top=0.85, width=0.48)
    for x0 in (-2.4, 0.4):
        if D.free(*D.fp(x0 + 1.0, 2.1, 1.05, 0.54, 0.0)):
            D.rack_bay(x0 + 1.0, 2.1, 0.0, bay=2.0)
    if D.free(*D.fp(1.9, -0.2, 0.62, 0.42, math.pi / 2), margin=0.2):
        D.crate_stack(1.9, -0.2, math.pi / 2)
    return [(-3.0, -1.0, 3.0, -1.0, 0.7)]


LAYOUTS = {"depalletize": site_depalletize, "conveyor_sort": site_conveyor_sort, "tote_transfer": site_tote_transfer,
           "shelf_restock": site_shelf_restock, "kitting_pocket": site_kitting_pocket}
