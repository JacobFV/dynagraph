"""Bulk task families: bin_clearing, singulation, bulk_sweep (+ logistics ones in families_logistics)."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..v2.api import SkillCall
from .base import StationEnv
from .scene import BENCH_H, PALETTE, sample_part_params
from .stations import bench_station

PICK_DEFAULT = dict(grip_force=15.0, lift_speed=0.3, depth=0.5)


def _pick_call(eid, rng=None, **kw):
    p = dict(PICK_DEFAULT)
    if rng is not None:
        p.update(grip_force=float(rng.uniform(10, 100)), lift_speed=float(rng.uniform(0.1, 0.6)),
                 depth=float(rng.uniform(0.2, 0.8)), dx=float(rng.normal(0, 0.003)), dy=float(rng.normal(0, 0.003)))
    p.update(kw)
    return SkillCall("pick", p, target=eid)


# ====================================================================== (a) clutter bin -> tote
class BinClearing(StationEnv):
    family = "bin_clearing"
    category = "bulk"
    station = "clutter_bin"
    tool = "parallel"
    probe_kw = dict(max_open=0.06, grip_max=100.0)
    settle_s = 1.2
    success_doc = ("every part from the source bin ends inside the target tote region "
                   "(centre within the tote's inner footprint, below its rim)")
    hidden_doc = ("per-part mass (heavy outliers x4-8 at same size), part<->gripper friction (oily parts 0.08-0.2), "
                  "part<->surface friction, restitution, contact compliance")

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.24
        n = int(params.get("n_parts", rng.integers(6, 13)))
        shapes = tuple(params.get("shapes", ("box", "cylinder", "capsule", "sphere", "mesh")))
        sx, sy = -0.2 + rng.uniform(-0.03, 0.03), 0.05 + rng.uniform(-0.03, 0.03)
        bin_half = (0.17, 0.12)
        floor_z = scene.open_box((sx, sy, top), bin_half, 0.10, prefix="srcbin")
        tx, ty = 0.24 + rng.uniform(-0.02, 0.02), 0.05 + rng.uniform(-0.03, 0.03)
        tote_half = (0.14, 0.11)
        tfloor = scene.open_box((tx, ty, top), tote_half, 0.09, rgba=(0.6, 0.2, 0.15, 1), prefix="tote")
        scene.region("source_bin", (sx, sy, floor_z + 0.06), (bin_half[0], bin_half[1], 0.08), support_z=floor_z)
        scene.region("tote", (tx, ty, tfloor + 0.05), (tote_half[0], tote_half[1], 0.07), support_z=tfloor)
        k = 0
        for i in range(n):
            pp = sample_part_params(rng, half_range=(0.012, 0.03), shapes=shapes, density_range=(1000, 8000),
                                    heavy_p=params.get("heavy_p", 0.15), oily_p=params.get("oily_p", 0.15))
            # non-overlapping spawn: rejection-sample xy within the layer (bounding-circle test)
            rad = float(np.linalg.norm(pp["half"][:2])) + 0.004
            layer = i // 6
            if layer != getattr(self, "_layer", -1):
                self._layer, self._placed = layer, []
            for _ in range(60):
                x = sx + rng.uniform(-bin_half[0] + rad, bin_half[0] - rad)
                y = sy + rng.uniform(-bin_half[1] + rad, bin_half[1] - rad)
                if all(np.hypot(x - a, y - b) > rad + r2 for a, b, r2 in self._placed):
                    break
            self._placed.append((x, y, rad))
            z = floor_z + 0.045 + 0.075 * layer + rng.uniform(0, 0.01)
            scene.part(f"part{i:02d}", pp["shape"], pp["half"], (x, y, z), yaw=rng.uniform(-math.pi, math.pi),
                       mass=pp["mass"], mu=pp["mu"], mu_grip=pp["mu_grip"], restitution=pp["restitution"],
                       timeconst=pp["timeconst"], extra_hidden=dict(heavy=pp["heavy"], oily=pp["oily"]))
            k += 1
        self.budget = int(params.get("budget", 4 * n))

    def parts(self):
        return self.objects("part")

    def success(self):
        reg = self.scene.regions["tote"]
        return all(self.in_region_true(e, reg, margin=0.01) for e in self.parts())

    def progress(self):
        reg = self.scene.regions["tote"]
        p = self.parts()
        return sum(self.in_region_true(e, reg, margin=0.01) for e in p) / len(p)

    def _pickable(self, stale_ok=False):
        reg = self.scene.regions["source_bin"]
        out = []
        for e in self.parts():
            est = self.est_pose(e)
            if est is None:
                continue
            if self.est_age(e) > 1.0 and not (stale_ok and self.in_region_est(e, reg, margin=0.03)):
                continue          # stale (occluded since): may already be in the tote
            if self.in_region_est(e, reg, margin=0.03) or (not self.in_region_est(e, self.scene.regions["tote"], 0.01)
                                                           and est[0][2] < self.surface_z + 0.3):
                out.append((est[0][2], e))
        return [e for _, e in sorted(out, reverse=True)]

    def _place_call(self, rng=None):
        p = dict(dx=0.0, dy=0.0, release_h=0.03, speed=0.4)
        if rng is not None:
            p.update(dx=float(rng.uniform(-0.08, 0.08)), dy=float(rng.uniform(-0.06, 0.06)),
                     release_h=float(rng.uniform(0.01, 0.08)), speed=float(rng.uniform(0.2, 0.8)))
        else:
            p.update(dx=float(self.rng.uniform(-0.07, 0.07)), dy=float(self.rng.uniform(-0.05, 0.05)))
        return SkillCall("place", p, target="tote")

    def candidates(self, n, rng):
        if self.held is not None:
            return [self._place_call(rng) for _ in range(n)]
        pk = self._pickable()[:4]
        if not pk:
            return []
        return [_pick_call(pk[i % len(pk)], rng) for i in range(n)]

    def scripted(self):
        if self.held is not None:
            return self._place_call()
        pk = self._pickable()
        if not pk:
            # nothing freshly seen: move the arm out of the camera's view and look again (twice), then also accept
            # stale estimates of parts last seen in the source bin
            self._looks = getattr(self, "_looks", 0) + 1
            if self._looks <= 2:
                reg = self.scene.regions["tote"]
                return SkillCall("reach", dict(x=float(reg["center"][0]), y=float(reg["center"][1]) - 0.3,
                                               z=float(self.safe_z), speed=0.6, dwell=0.6), target="tote")
            pk = self._pickable(stale_ok=True)
            if not pk:
                return None
        else:
            self._looks = 0
        # engineer's heuristic: conservative grip / lift, then escalate; re-plan the grasp axis after a miss;
        # skip a part after 3 failures (come back to it once the others are done)
        cand = [e for e in pk if self.fail_count.get(e, 0) < 3] or pk
        e = cand[0]
        k = self.fail_count.get(e, 0)
        kw = dict(grip_force=(40.0, 90.0, 100.0, 100.0)[min(k, 3)], lift_speed=(0.2, 0.1, 0.08, 0.08)[min(k, 3)],
                  depth=(0.5, 0.5, 0.3, 0.7)[min(k, 3)])
        # grasp re-planning: near a bin wall close the pads PARALLEL to that wall; after a miss rotate 90 deg
        est = self.est_pose(e)
        reg = self.scene.regions["source_bin"]
        if est is not None and self.in_region_est(e, reg, margin=0.03):
            gx = reg["half"][0] - abs(est[0][0] - reg["center"][0])
            gy = reg["half"][1] - abs(est[0][1] - reg["center"][1])
            if min(gx, gy) < 0.045:
                base = 0.0 if gx < gy else math.pi / 2
                kw["close_yaw"] = base + (math.pi / 2 if k % 2 else 0.0)
        if "close_yaw" not in kw and k % 2 == 1:
            kw["dyaw"] = math.pi / 2
        return _pick_call(e, **kw)

    def metrics_extra(self):
        return dict(n_parts=len(self.parts()))


# ====================================================================== (j) singulation by pushing
class Singulation(StationEnv):
    family = "singulation"
    category = "bulk"
    station = "push_table"
    tool = "parallel"
    gap_req = 0.03
    success_doc = "every object has >= 3 cm clearance to every other object and stays on the bench work area"
    hidden_doc = "object<->bench friction (0.15-0.9, per object) and mass (log-U 0.1-3 kg): push travel after release"

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.2
        n = int(params.get("n_objects", rng.integers(4, 8)))
        mu_mean = float(params.get("mu_mean", rng.uniform(0.15, 0.9)))
        # pack objects on a jittered grid so they touch
        cols = int(math.ceil(math.sqrt(n)))
        pitch = 0.07
        k = 0
        for i in range(n):
            r_, c_ = divmod(i, cols)
            hx, hy = rng.uniform(0.022, 0.033, 2)
            shape = "box" if rng.random() < params.get("p_box", 0.7) else "cylinder"
            if shape == "cylinder":
                hy = hx
            hz = rng.uniform(0.02, 0.035)
            mass = float(np.exp(rng.uniform(np.log(0.1), np.log(3.0))))
            x = (c_ - (cols - 1) / 2) * pitch + rng.uniform(-0.004, 0.004)
            y = (r_ - (cols - 1) / 2) * pitch + 0.02 + rng.uniform(-0.004, 0.004)
            mu = float(np.clip(mu_mean * rng.uniform(0.85, 1.15), 0.1, 1.0))
            scene.part(f"obj{i}", shape, (hx, hy, hz), (x, y, top + hz + 0.002), yaw=rng.uniform(-0.2, 0.2),
                       mass=mass, mu=mu, mu_grip=0.8, restitution=0.05)
            k += 1
        scene.region("work", (0.0, 0.02, top + 0.03), (0.45, 0.3, 0.05), support_z=top)
        self.budget = int(params.get("budget", 3 * n))

    def _gaps(self):
        m, d = self.m, self.d
        ob = self.objects()
        gids = {e: [m.geom(g).id for g in self.scene.entities[e].geoms] for e in ob}
        gap = {e: 1.0 for e in ob}
        ft = np.zeros(6)
        for i, a in enumerate(ob):
            for b in ob[i + 1:]:
                dist = mujoco.mj_geomDistance(m, d, gids[a][0], gids[b][0], 0.2, ft)
                gap[a] = min(gap[a], dist)
                gap[b] = min(gap[b], dist)
        return gap

    def success(self):
        reg = self.scene.regions["work"]
        g = self._gaps()
        return all(v >= self.gap_req for v in g.values()) and all(self.in_region_true(e, reg) for e in g)

    def progress(self):
        g = self._gaps()
        return sum(v >= self.gap_req for v in g.values()) / len(g)

    def fatal(self):
        return self.metrics_acc["drops"] > 0

    def _est_gaps(self):
        """actor-legal: gap estimate from tracked centres and SIZE (circumscribed-free box approx)."""
        ob = self.objects()
        P = {}
        for e in ob:
            est = self.est_pose(e)
            if est is not None:
                P[e] = (est[0][:2], est[2], self.est_size(e))
        gap = {}
        for a in P:
            best, nb = 1.0, None
            for b in P:
                if a == b:
                    continue
                dvec = P[b][0] - P[a][0]
                dist = np.linalg.norm(dvec)
                ra = _support_radius(P[a][2], P[a][1], dvec / max(dist, 1e-6))
                rb = _support_radius(P[b][2], P[b][1], -dvec / max(dist, 1e-6))
                g = dist - ra - rb
                if g < best:
                    best, nb = g, b
            gap[a] = (best, nb)
        return P, gap

    def _push_away(self, e, P, gap, speed=0.25, dist=0.06):
        """push e directly away from the centroid of its crowded neighbours; returns object-frame params."""
        pa, yaw, half = P[e]
        cen = np.mean([P[b][0] for b in P if b != e and gap[e][1] is not None and
                       np.linalg.norm(P[b][0] - pa) < 0.15] or [np.mean([v[0] for v in P.values()], 0)], 0)
        u = pa - cen
        if np.linalg.norm(u) < 1e-4:
            u = pa - np.array([0.0, 0.02])
        u /= np.linalg.norm(u)
        # contact on the side opposite the motion, in the object frame
        c, s = math.cos(-yaw), math.sin(-yaw)
        uo = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])
        theta = math.atan2(-uo[1], -uo[0])
        return dict(theta=theta, phi=0.0, speed=speed, distance=dist, height=0.012)

    def candidates(self, n, rng):
        P, gap = self._est_gaps()
        crowded = sorted([e for e in P if gap[e][0] < self.gap_req + 0.01], key=lambda e: gap[e][0])
        if not crowded:
            crowded = list(P)
        out = []
        for i in range(n):
            e = crowded[i % len(crowded)]
            p = self._push_away(e, P, gap, speed=float(rng.uniform(0.05, 0.8)), dist=float(rng.uniform(0.02, 0.12)))
            p["theta"] = float(p["theta"] + rng.normal(0, 0.4))
            p["phi"] = float(rng.uniform(-0.3, 0.3))
            out.append(SkillCall("push", p, target=e))
        return out

    def scripted(self):
        P, gap = self._est_gaps()
        crowded = sorted([e for e in P if gap[e][0] < self.gap_req + 0.008], key=lambda e: gap[e][0])
        if not crowded:
            return None
        e = crowded[self.n_calls % len(crowded)] if self.n_calls % 2 else crowded[0]
        return SkillCall("push", self._push_away(e, P, gap), target=e)


def _support_radius(half, yaw, u):
    c, s = math.cos(-yaw), math.sin(-yaw)
    lx, ly = c * u[0] - s * u[1], s * u[0] + c * u[1]
    return abs(lx) * half[0] + abs(ly) * half[1]


# ====================================================================== (h) bulk material sweep
class BulkSweep(StationEnv):
    family = "bulk_sweep"
    category = "bulk"
    station = "bulk_material"
    tool = "blade"
    probe_kw = dict(blade_half_w=0.08)
    bulk_sensors = True
    settle_s = 0.8
    frac_req = 0.85
    success_doc = ">= 85% of the small parts have their centre inside the marked target region on the bench"
    hidden_doc = ("parts<->bench friction (episode mean 0.12-0.8, per part +-15%): after the blade's stop the parts "
                  "slide on by ~v^2/(2 mu g) - so the right stop point / stop time depends on mu")

    def build(self, scene, rng, params):
        top = bench_station(scene, half=(0.6, 0.4))
        self.surface_z = top
        self.safe_z = top + 0.15
        n = int(params.get("n_parts", rng.integers(40, 101)))
        mu_mean = float(params.get("mu_mean", rng.uniform(0.12, 0.8)))
        self.mu_mean = mu_mean
        # source patch on the left, target region on the right (sweeps go +x)
        sx0, sx1 = -0.38, -0.18
        sy0, sy1 = -0.10, 0.10
        tx = float(params.get("target_x", rng.uniform(0.12, 0.22)))
        scene.region("source", ((sx0 + sx1) / 2, (sy0 + sy1) / 2, top + 0.01), ((sx1 - sx0) / 2, (sy1 - sy0) / 2, 0.03),
                     support_z=top)
        scene.region("target", (tx, 0.0, top + 0.01), (0.09, 0.16, 0.03), support_z=top)
        scene.marker((tx, 0.0), (0.09, 0.16), top, rgba=(0.1, 0.8, 0.2, 0.5))
        # 2.5 cm perimeter lip (bulk trays have one): low-friction parts squirt sideways instead of off the bench
        for sgn in (-1, 1):
            scene.static_box((0.0, sgn * 0.39, top + 0.0125), (0.6, 0.01, 0.0125), rgba=(0.8, 0.7, 0.1, 1))
            scene.static_box((sgn * 0.59, 0.0, top + 0.0125), (0.01, 0.38, 0.0125), rgba=(0.8, 0.7, 0.1, 1))
        p_nut, p_blk = params.get("mix", (0.55, 0.3))
        for i in range(n):
            u = rng.random()
            mu = float(np.clip(mu_mean * rng.uniform(0.85, 1.15), 0.05, 1.0))
            x, y = rng.uniform(sx0, sx1), rng.uniform(sy0, sy1)
            if u < p_nut:   # nuts / washers lying flat
                r = rng.uniform(0.006, 0.01)
                h = (r, r, rng.uniform(0.003, 0.005))
                shape = "cylinder"
            elif u < p_nut + p_blk:
                h = tuple(rng.uniform(0.004, 0.008, 3))
                shape = "box"
            else:
                r = rng.uniform(0.004, 0.006)
                h = (r * rng.uniform(1.6, 2.4), r, r)
                shape = "capsule"
            z = top + h[2] + 0.002 + 0.012 * (i % 3)
            mass = float(8 * np.prod(h) * rng.uniform(2500, 7800))
            scene.part(f"p{i:03d}", shape, h, (x, y, z), yaw=rng.uniform(-math.pi, math.pi), mass=mass, mu=mu,
                       mu_grip=0.5, restitution=rng.uniform(0.0, 0.2), graspable=False, condim=3,
                       rgba=(0.75, 0.7, 0.55, 1) if shape == "cylinder" else (0.6, 0.6, 0.65, 1))
        self.budget = int(params.get("budget", 12))

    def _in_target(self):
        reg = self.scene.regions["target"]
        return [self.in_region_true(e, reg, zcheck=False) for e in self.objects()]

    def success(self):
        v = self._in_target()
        return sum(v) / len(v) >= self.frac_req

    def progress(self):
        v = self._in_target()
        return sum(v) / len(v)

    def fatal(self):
        return self.metrics_acc["drops"] > 0.1 * len(self.objects())

    def _outside_est(self):
        reg = self.scene.regions["target"]
        pts = []
        for e in self.objects():
            est = self.est_pose(e)
            if est is None:
                continue
            if not self.in_region_est(e, dict(reg, half=(reg["half"][0], reg["half"][1], 1.0)), 0.0):
                pts.append(est[0][:2])
        return np.array(pts).reshape(-1, 2)

    def _lane_call(self, lane, speed=0.4, stop_time=0.15, end_off=-0.02, height=0.002, lat_noise=0.0):
        """lane = (count, axis (0:x, 1:y), sign (+1 sweeps toward +axis), lateral coord, start coord along axis)."""
        _, ax, sg, lat, start = lane
        reg = self.scene.regions["target"]
        c = np.array(reg["center"][:2])
        end = c[ax] + sg * end_off
        p0, p1 = np.zeros(2), np.zeros(2)
        p0[ax], p1[ax] = start, end
        p0[1 - ax] = p1[1 - ax] = lat + lat_noise
        return SkillCall("sweep", dict(x0=float(p0[0]), y0=float(p0[1]), x1=float(p1[0]), y1=float(p1[1]),
                                       speed=float(speed), stop_time=float(stop_time), height=float(height)),
                         target="target")

    def _lanes(self):
        """Candidate blade lanes from ESTIMATED outside parts: each part is assigned the axis direction that
        brings it toward the target box; lanes on a 1 cm lateral grid (clipped so the blade stays over the
        target's span), scored by how many such parts the blade (half width 7.5 cm) would carry."""
        pts = self._outside_est()
        if len(pts) == 0:
            return []
        reg = self.scene.regions["target"]
        c = np.array(reg["center"][:2])
        h = np.array(reg["half"][:2])
        o = pts - c
        over = np.abs(o) - h
        axis = (over[:, 1] > over[:, 0]).astype(int)
        sign = -np.sign(o[np.arange(len(o)), axis])
        bw = 0.075
        lanes = []
        for ax in (0, 1):
            for sg in (-1, 1):
                sel0 = (axis == ax) & (sign == sg)
                if not sel0.any():
                    continue
                q = pts[sel0]
                lo, hi = c[1 - ax] - h[1 - ax] - 0.1, c[1 - ax] + h[1 - ax] + 0.1
                lim = max(h[1 - ax] - bw, 0.0)
                for lat in np.arange(lo, hi + 1e-9, 0.01):
                    sel = np.abs(q[:, 1 - ax] - lat) <= bw
                    if sel.sum():
                        latc = float(np.clip(lat, c[1 - ax] - lim, c[1 - ax] + lim))
                        start = float(q[sel, ax].min() - 0.03) if sg > 0 else float(q[sel, ax].max() + 0.03)
                        lim_xy = 0.55 if ax == 0 else 0.35
                        start = float(np.clip(start, -lim_xy, lim_xy))
                        lanes.append((int(sel.sum()), ax, int(sg), latc, start))
        return sorted(lanes, key=lambda t: (-t[0], abs(t[3])))

    def candidates(self, n, rng):
        lanes = self._lanes()
        if not lanes:
            return []
        out = []
        for i in range(n):
            out.append(self._lane_call(lanes[i % min(len(lanes), 6)], speed=float(rng.uniform(0.1, 0.8)),
                                       stop_time=float(rng.uniform(0.05, 0.8)), end_off=float(rng.uniform(-0.1, 0.06)),
                                       lat_noise=float(rng.normal(0, 0.01))))
        return out

    def scripted(self):
        lanes = self._lanes()
        if not lanes:
            return None
        return self._lane_call(lanes[0])

    def labels_extra(self):
        return dict(mu_mean=self.mu_mean)
