"""Logistics/bulk task families: depalletize (c), conveyor_sort (b), tote_transfer (f), pouch_transfer (i)."""
from __future__ import annotations

import math

import numpy as np

from ..v2.api import SkillCall
from .base import StationEnv
from .scene import BENCH_H, BLUE_TOTE, CARDBOARD, DARK
from .stations import standard_mounts


# ====================================================================== (c) depalletize
class Depalletize(StationEnv):
    family = "depalletize"
    category = "bulk"
    station = "pallet"
    tool = "suction"
    probe_kw = dict(suction_max=300.0)
    settle_s = 1.0
    success_doc = ("like a real cell (round 3): every carton within the robot's declared payload "
                   "(EmbodimentSpec.payload_kg) ends upright on the outfeed AND every overweight carton is flagged to the "
                   "exception lane (SkillCall 'handover' on it) and left on the pallet; nothing dropped. Light cartons "
                   "flagged needlessly, overweight cartons lifted off the pallet, or any drop -> failure")
    hidden_doc = ("carton mass 1-6 kg + hidden heavy cartons 12-22 kg (p=0.25, bottom layer, same look) -> lift vs flag "
                  "depends on the hidden mass vs payload (estimable from a probing partial lift + wrist F/T); 30 % of "
                  "cartons have slick film (cup shear friction 0.15-0.3) and peel off a cup at high transport "
                  "acceleration -> suction level / place speed; carton<->carton friction 0.3-0.8")

    def build(self, scene, rng, params):
        # EUR pallet 1.2 x 0.8 x 0.144 on the floor; outfeed table on the robot side
        px, py = 0.0, 0.25
        scene.static_box((px, py, 0.072), (0.6, 0.4, 0.072), rgba=(0.62, 0.48, 0.3, 1), name="pallet")
        tz = 0.55
        ox, oy = 0.0, -0.55
        scene.static_box((ox, oy, tz - 0.02), (0.6, 0.22, 0.02), rgba=(0.5, 0.52, 0.55, 1), name="outfeed_top")
        for sx in (-1, 1):
            for sy in (-1, 1):
                scene.static_box((ox + sx * 0.55, oy + sy * 0.18, (tz - 0.04) / 2), (0.025, 0.025, (tz - 0.04) / 2),
                                 rgba=DARK)
        scene.region("outfeed", (ox, oy, tz + 0.1), (0.6, 0.22, 0.2), support_z=tz)
        self.surface_z = tz
        hx, hy, hz = rng.uniform(0.12, 0.15), rng.uniform(0.09, 0.11), rng.uniform(0.07, 0.11)
        layers = int(params.get("layers", rng.integers(1, 3)))
        nx, ny = 2, 2
        k = 0
        heavy_p = params.get("heavy_p", 0.25)
        for L in range(layers):
            for i in range(nx):
                for j in range(ny):
                    x = px + (i - 0.5) * (2 * hx + 0.006) + rng.uniform(-0.004, 0.004)
                    y = py + (j - 0.5) * (2 * hy + 0.006) + rng.uniform(-0.004, 0.004)
                    z = 0.144 + hz + L * (2 * hz + 0.001) + 0.001
                    # heavy cartons only in the bottom layer (standard stacking practice): the light ones above stay
                    # reachable whatever the robot decides about the heavy ones
                    heavy = (L == 0) and rng.random() < heavy_p
                    mass = float(rng.uniform(12, 22) if heavy else rng.uniform(1, 6))
                    slick = rng.random() < params.get("slick_p", 0.3)
                    mug = float(rng.uniform(0.15, 0.3) if slick else rng.uniform(0.7, 1.0))
                    _, info = scene.part(f"carton{k}", "box", (hx, hy, hz), (x, y, z), yaw=rng.uniform(-0.02, 0.02),
                                         mass=mass, mu=float(rng.uniform(0.3, 0.8)), mu_grip=mug,
                                         restitution=0.05, timeconst=0.02, kind="carton", rgba=CARDBOARD,
                                         extra_hidden=dict(heavy=bool(heavy), slick=bool(slick)))
                    info.shape = "carton"
                    k += 1
        self.n = k
        self.half = (hx, hy, hz)
        top = 0.144 + layers * 2 * hz
        # TCP transit height: a carried carton (2hz below the cup) must clear the stack AND cartons already placed
        self.safe_z = max(top, tz + 2 * hz) + 2 * hz + 0.08
        standard_mounts(scene, (0.0, -0.2), 0.0, (0.6, 0.35), gantry_h=1.75)
        scene.mounts["humanoid"] = [((0.0, -0.15, 0.0), math.pi / 2)]
        scene.mounts["arm"] = [((0.0, -0.24, 0.75), math.pi / 2)]    # pedestal between outfeed and pallet
        scene.camera = dict(pos=(0.0, -0.5, 2.4), lookat=(0.0, 0.0, 0.3), view_pos=(1.6, -1.9, 1.9),
                            view_lookat=(0.0, -0.1, 0.35))
        # outfeed slots: 4 x 2 grid
        self.slots = [(sx * (2 * hx + 0.03), sy * (hy + 0.03)) for sy in (-1, 1) for sx in (-1.5, -0.5, 0.5, 1.5)]
        self.budget = int(params.get("budget", 3 * k))
        self._next_slot = 0
        self.flagged = set()

    def cartons(self):
        return self.objects("carton")

    def _on_outfeed(self, e):
        p, q = self.true_pose(e)
        reg = self.scene.regions["outfeed"]
        up = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        hz = self.scene.entities[e].half[2]
        return (self.in_region_true(e, reg, zcheck=False) and abs(p[2] - (self.surface_z + hz)) < 0.02 and up > 0.95)

    def payload(self):
        return float(getattr(self.robot.spec, "payload_kg", 1e9))

    def overweight(self, e):
        return self.scene.entities[e].hidden["mass"] > self.payload()

    def _on_pallet(self, e):
        p, _ = self.true_pose(e)
        return abs(p[0]) < 0.62 and 0.25 - 0.42 < p[1] < 0.25 + 0.42 and p[2] > 0.144

    def _carton_ok(self, e):
        if self.overweight(e):
            return e in self.flagged and self._on_pallet(e)
        return self._on_outfeed(e) and e not in self.flagged

    def success(self):
        return all(self._carton_ok(e) for e in self.cartons()) and self.metrics_acc["drops"] == 0

    def progress(self):
        c = self.cartons()
        return sum(self._carton_ok(e) for e in c) / len(c)

    def fatal(self):
        return self.metrics_acc["drops"] > 0

    def execute(self, call):
        """'handover' on a carton = flag it to the exception lane (operator / forklift): env-level, 2 s cycle, no
        motion; picks of overweight cartons are counted (an attempted overweight lift costs)."""
        if call.skill == "pick" and call.target in self.scene.entities and self.overweight(call.target):
            self._ow_attempts = getattr(self, "_ow_attempts", 0) + 1
        if call.skill != "handover":
            return super().execute(call)
        from ..v2.api import EffortMeter, SkillResult, StepOut
        self.n_calls += 1
        before = self.progress()
        if call.target in self.scene.entities:
            self.flagged.add(call.target)
        self.run(2.0)
        ev = self.sensors.flush()
        self.events_all.extend(ev)
        res = SkillResult(call.target in self.scene.entities, "flagged", EffortMeter(), 2.0,
                          dict(skill="handover", params=dict(call.params), target=call.target))
        self.last_held_before, self.last_call, self.last_result = self.held, call, res
        succ = bool(self.success())
        fatal = self.fatal() or self.metrics_acc["sim_instabilities"] > 0
        prog = self.progress()
        metrics = dict(progress=prog, progress_delta=prog - before, n_calls=self.n_calls, t=self.t, slip=False,
                       jam=False, stall=False, drop=False, collision=False, success=succ,
                       **{k: v for k, v in self.metrics_acc.items()}, effort_cost=0.0, fatal=fatal,
                       **self.metrics_extra())
        return StepOut(events=ev, result=res, done=succ or fatal or self.n_calls >= self.budget, success=succ,
                       metrics=metrics, labels=self.labels())

    def metrics_extra(self):
        return dict(flagged=len(self.flagged), overweight_lift_attempts=getattr(self, "_ow_attempts", 0))

    def labels_extra(self):
        return dict(payload_kg=self.payload(), overweight={e: self.overweight(e) for e in self.cartons()},
                    flagged=sorted(self.flagged))

    def _remaining(self):
        out = []
        for e in self.cartons():
            est = self.est_pose(e)
            if est is None:
                continue
            if est[0][1] > -0.2 and e not in self.flagged:     # still on the pallet side, not flagged
                out.append((est[0][2], e))
        return [e for _, e in sorted(out, reverse=True)]

    def _pick_call(self, e, rng=None, **kw):
        p = dict(suction=0.5, lift_speed=0.25, approach_speed=0.5, lift_height=0.1)
        if rng is not None:
            p.update(suction=float(rng.uniform(0.2, 1.0)), lift_speed=float(rng.uniform(0.05, 0.6)),
                     dx=float(rng.normal(0, 0.01)), dy=float(rng.normal(0, 0.01)))
        p.update(kw)
        return SkillCall("pick", p, target=e)

    def _place_call(self, rng=None):
        n_on = sum(1 for e in self.cartons() if (self.est_pose(e) is not None and self.est_pose(e)[0][1] < -0.2))
        sx, sy = self.slots[min(n_on, len(self.slots) - 1)]
        p = dict(dx=sx, dy=sy, release_h=0.01, speed=0.35, lower_speed=0.1)
        if rng is not None:
            p.update(release_h=float(rng.uniform(0.0, 0.05)), speed=float(rng.uniform(0.1, 0.7)))
        return SkillCall("place", p, target="outfeed")

    def candidates(self, n, rng):
        if self.held is not None:
            return [self._place_call(rng) for _ in range(n)]
        rem = self._remaining()[:4]
        if not rem:
            return []
        out = []
        for i in range(n):
            e = rem[i % len(rem)]
            out.append(SkillCall("handover", dict(), target=e) if rng.random() < 0.2 else self._pick_call(e, rng))
        return out

    def scripted(self):
        """baseline: lift every carton; after two failed lifts of the same carton, flag it (handover)."""
        if self.held is not None:
            return self._place_call()
        rem = self._remaining()
        if not rem:
            return None
        e = rem[0]                                           # top layer first
        k = self.fail_count.get(e, 0)
        if k >= 2:
            return SkillCall("handover", dict(reason="lift failed twice"), target=e)
        return self._pick_call(e, suction=(0.9, 1.0)[min(k, 1)], lift_speed=(0.15, 0.06)[min(k, 1)])


# ====================================================================== (b) conveyor sort
class ConveyorSort(StationEnv):
    family = "conveyor_sort"
    category = "bulk"
    station = "conveyor"
    tool = "parallel"
    probe_kw = dict(max_open=0.06, grip_max=100.0)
    settle_s = 0.5
    success_doc = ("every part is picked off the DRIVEN belt and dropped in the bin of its class (cylinders -> bin_A, "
                   "boxes -> bin_B); a part reaching the belt end (reject chute) or the floor fails the episode")
    hidden_doc = ("belt speed 0.03-0.12 m/s (estimable from tracking), part mass incl. heavy x3-6, part<->gripper "
                  "friction (oily parts), part<->belt friction: intercept timing / lead, grip force")

    def build(self, scene, rng, params):
        top = 0.80
        self.surface_z = top
        self.safe_z = top + 0.2
        v = float(params.get("belt_speed", rng.uniform(0.03, 0.12)))
        self.belt_speed = v
        n = int(params.get("n_parts", rng.integers(3, 6)))
        x0, L, W = -0.9, 1.5, 0.3
        gap = v * float(params.get("cycle_s", 13.0))
        travel = v * (n * 16.0 + 30.0) + 1.0
        scene.conveyor((x0, 0.0), L, W, top, v, travel=travel)
        # sort bins either side, reject chute at the belt end
        for name, y, col in (("bin_A", 0.42, (0.8, 0.5, 0.1, 1)), ("bin_B", -0.42, (0.2, 0.4, 0.75, 1))):
            scene.static_box((0.0, y, 0.33), (0.14, 0.11, 0.33), rgba=DARK)
            fz = scene.open_box((0.0, y, 0.66), (0.14, 0.11), 0.10, rgba=col, prefix=name)
            scene.region(name, (0.0, y, fz + 0.06), (0.14, 0.11, 0.09), support_z=fz)
        scene.open_box((x0 + L + 0.2, 0.0, 0.0), (0.15, 0.2), 0.3, rgba=(0.6, 0.1, 0.1, 1), prefix="reject")
        scene.region("reject", (x0 + L + 0.2, 0.0, 0.15), (0.2, 0.25, 0.3), support_z=0.0)
        scene.region("belt", (x0 + L / 2, 0.0, top + 0.05), (L / 2, W / 2, 0.05), support_z=top)
        scene.region("pick_window", (-0.1, 0.0, top + 0.05), (0.3, W / 2, 0.05), support_z=top)
        for i in range(n):
            cls = "A" if rng.random() < 0.5 else "B"
            if cls == "A":
                r = rng.uniform(0.018, 0.03)
                half, shape = (r, r, rng.uniform(0.02, 0.035)), "cylinder"
            else:
                half, shape = (rng.uniform(0.02, 0.04), rng.uniform(0.018, 0.03), rng.uniform(0.015, 0.03)), "box"
            heavy = rng.random() < params.get("heavy_p", 0.15)
            mass = float(8 * np.prod(half) * rng.uniform(800, 3000) * (rng.uniform(3, 6) if heavy else 1))
            oily = rng.random() < 0.15
            x = x0 + 0.15 - i * gap
            scene.part(f"item{i}", shape, half, (x, rng.uniform(-0.06, 0.06), top + half[2] + 0.002),
                       yaw=rng.uniform(-math.pi, math.pi), mass=mass, mu=float(rng.uniform(0.4, 0.9)),
                       mu_grip=float(rng.uniform(0.08, 0.2) if oily else rng.uniform(0.4, 1.0)), restitution=0.1,
                       timeconst=0.01, meta=dict(sort_class=cls), extra_hidden=dict(heavy=bool(heavy), oily=bool(oily)))
        standard_mounts(scene, (0.0, 0.0), top, (0.6, 0.15), gantry_h=1.5)
        scene.mounts["arm"] = [((0.3, -0.27, 0.8), math.pi / 2)]     # pedestal beside the belt, between bins
        scene.camera = dict(pos=(-0.1, -0.6, 2.0), lookat=(-0.1, 0.0, top), view_pos=(0.9, -1.6, 1.8),
                            view_lookat=(-0.2, 0.0, top))
        self.budget = int(params.get("budget", 4 * n + 6))

    def items(self):
        return self.objects("part")

    def _where(self, e):
        for name in ("bin_A", "bin_B", "reject"):
            if self.in_region_true(e, self.scene.regions[name], margin=0.01):
                return name
        p, _ = self.true_pose(e)
        return "floor" if p[2] < 0.3 else "belt"

    def success(self):
        return all(self._where(e) == "bin_" + self.scene.entities[e].meta["sort_class"] for e in self.items())

    def progress(self):
        it = self.items()
        return sum(self._where(e) == "bin_" + self.scene.entities[e].meta["sort_class"] for e in it) / len(it)

    def fatal(self):
        return any(self._where(e) in ("reject", "floor") for e in self.items()) or self.metrics_acc["drops"] > 0

    def goal_extra(self):
        return dict(sort_rule={"cylinder": "bin_A", "box": "bin_B"})

    def _belt_v(self):
        """actor-legal belt velocity estimate: median of tracked velocities of parts lying on the belt."""
        vs = []
        for e in self.items():
            est = self.est_pose(e)
            if est is None or not (-0.95 < est[0][0] < 0.65 and abs(est[0][1]) < 0.16):
                continue
            v = self.est_velocity(e, window=1.0)
            if v is not None:
                vs.append(v)
        return np.median(np.array(vs), 0) if vs else np.zeros(2)

    def _class_of(self, e):
        from .props import SHAPES, XProp
        ev = self.view.latest(e, XProp.SHAPE, self.t + 1e9)
        return "A" if ev is not None and SHAPES[int(np.argmax(ev.value))] == "cylinder" else "B"

    def _sorted_ids(self):
        """actor memory + tracker: items this robot already placed into a bin (successful place of a held item), or
        whose latest estimate lies inside a bin region. Never re-targeted."""
        lc, lr = getattr(self, "last_call", None), getattr(self, "last_result", None)
        done = getattr(self, "_sorted", set())
        held = getattr(self, "last_held_before", None)
        if lc is not None and lc.skill == "place" and lr is not None and lr.ok and held is not None \
                and lc.target in ("bin_A", "bin_B"):
            done.add(held)
        for e in self.items():
            est = self.est_pose(e)
            if est is not None and any(self.in_region_est(e, self.scene.regions[b], margin=0.02)
                                       for b in ("bin_A", "bin_B")):
                done.add(e)
        self._sorted = done
        return done

    def _in_window(self):
        out = []
        skip = self._sorted_ids()
        reg = self.scene.regions["pick_window"]
        for e in self.items():
            est = self.est_pose(e)
            if est is None or self.held == e or e in skip:
                continue
            # reachable stretch of the belt: the pick window plus the last 0.25 m before the reject chute
            if abs(est[0][1]) < 0.16 and reg["center"][0] - reg["half"][0] - 0.05 < est[0][0] < 0.45 \
                    and est[0][2] > self.surface_z:
                out.append((est[0][0], e))
        # most downstream first, but skip parts too close to the belt end to be intercepted
        ok = [(x, e) for x, e in out if x < 0.33]
        return [e for _, e in sorted(ok or out, reverse=True)]

    def _upstream(self):
        out = []
        skip = self._sorted_ids()
        for e in self.items():
            if e in skip:
                continue
            est = self.est_pose(e)
            if est is not None and est[0][0] < -0.4 and abs(est[0][1]) < 0.16:
                out.append((est[0][0], e))
        return sorted(out, reverse=True)

    def _pick_call(self, e, rng=None, **kw):
        v = self._belt_v()
        p = dict(grip_force=25.0, lift_speed=0.3, depth=0.4, vx=float(v[0]), vy=float(v[1]), approach_speed=0.6,
                 lift_height=0.12)
        if rng is not None:
            p.update(grip_force=float(rng.uniform(8, 90)), vx=float(v[0] * rng.uniform(0.6, 1.4)),
                     lift_speed=float(rng.uniform(0.1, 0.6)), depth=float(rng.uniform(0.1, 0.7)))
        p.update(kw)
        return SkillCall("pick", p, target=e)

    def _place_call(self, rng=None):
        cls = self._class_of(self.held)
        p = dict(dx=0.0, dy=0.0, release_h=0.04, speed=0.5)
        if rng is not None:
            p.update(dx=float(rng.uniform(-0.06, 0.06)), dy=float(rng.uniform(-0.04, 0.04)),
                     release_h=float(rng.uniform(0.01, 0.1)), speed=float(rng.uniform(0.2, 0.8)))
        return SkillCall("place", p, target="bin_" + cls)

    def _wait_call(self, dwell):
        return SkillCall("reach", dict(x=-0.1, y=0.0, z=self.safe_z, yaw=0.0, speed=0.5, dwell=float(dwell)),
                         target="pick_window")

    def candidates(self, n, rng):
        if self.held is not None:
            return [self._place_call(rng) for _ in range(n)]
        win = self._in_window()
        if not win:
            return [self._wait_call(float(rng.uniform(0.5, 4.0))) for _ in range(n)]
        return [self._pick_call(win[i % len(win)], rng) for i in range(n)]

    def scripted(self):
        if self.held is not None:
            return self._place_call()
        win = self._in_window()
        if win:
            k = self.fail_count.get(win[0], 0)
            return self._pick_call(win[0], grip_force=(40.0, 80.0, 100.0)[min(k, 2)], lift_speed=(0.25, 0.15, 0.1)[min(k, 2)])
        up = self._upstream()
        if not up:
            # anything still on the belt (e.g. just past the window)? wait briefly and re-check rather than give up
            skip = self._sorted_ids()
            on = [e for e in self.items() if e not in skip and (lambda est: est is not None and abs(est[0][1]) < 0.16
                                               and est[0][2] > self.surface_z and est[0][0] < 0.6)(self.est_pose(e))]
            return self._wait_call(0.5) if on and self.n_calls < self.budget - 1 else None
        v = max(self._belt_v()[0], 0.02)
        x = up[0][0]
        reg = self.scene.regions["pick_window"]
        dwell = (reg["center"][0] - reg["half"][0] + 0.25 - x) / v
        return self._wait_call(float(np.clip(dwell, 0.3, 2.5)))     # short waits: re-check the window often

    def labels_extra(self):
        return dict(belt_speed=self.belt_speed, where={e: self._where(e) for e in self.items()})


# ====================================================================== (f) shelving / aisle: tote transfer
class ToteTransfer(StationEnv):
    family = "tote_transfer"
    category = "bulk"
    station = "shelving_aisle"
    tool = "parallel"
    probe_kw = dict(max_open=0.125, grip_max=150.0)
    settle_s = 0.8
    success_doc = "every tote is moved from the flow rack to the cart and rests upright inside the cart footprint"
    hidden_doc = ("tote content mass 0.3-9 kg (ballast, invisible), tote<->gripper friction 0.4-0.9, tote<->shelf "
                  "friction 0.2-0.6: grip capacity 2*mu_g*F vs m(g+a) -> grip force / lift speed")

    def build(self, scene, rng, params):
        # staggered two-tier flow rack (lower tier in front), cart on the robot side
        tiers = [(0.15, 0.45), (0.45, 0.85)]       # (y, z of shelf top)
        for y, z in tiers:
            scene.static_box((0.0, y, z - 0.015), (0.5, 0.15, 0.015), rgba=(0.7, 0.55, 0.2, 1))
            for sx in (-1, 1):
                scene.static_box((sx * 0.5, y, z / 2), (0.02, 0.02, z / 2), rgba=DARK)
        cz = 0.6
        scene.static_box((0.0, -0.45, cz - 0.02), (0.45, 0.2, 0.02), rgba=(0.5, 0.52, 0.55, 1))
        for sx in (-1, 1):
            for sy in (-1, 1):
                scene.static_box((sx * 0.4, -0.45 + sy * 0.15, (cz - 0.04) / 2), (0.02, 0.02, (cz - 0.04) / 2), rgba=DARK)
        scene.region("cart", (0.0, -0.45, cz + 0.07), (0.45, 0.2, 0.08), support_z=cz)
        self.surface_z = cz
        n = int(params.get("n_totes", rng.integers(2, 4)))
        hx, hy, hz, t = 0.15, 0.1, 0.06, 0.008
        slots = [(tiers[0], -0.2), (tiers[0], 0.2), (tiers[1], -0.2), (tiers[1], 0.2)]
        order = rng.permutation(len(slots))[:n]
        for k, si in enumerate(order):
            (y, z), x = slots[si]
            content = float(params.get("content_mass", np.exp(rng.uniform(np.log(0.3), np.log(9.0)))))
            boxes = [((0, 0, -hz + t / 2), (hx, hy, t / 2)),
                     ((hx - t / 2, 0, 0), (t / 2, hy, hz)), ((-hx + t / 2, 0, 0), (t / 2, hy, hz)),
                     ((0, hy - t / 2, 0), (hx - t, t / 2, hz)), ((0, -hy + t / 2, 0), (hx - t, t / 2, hz))]
            ms = [0.4 + content, 0.15, 0.15, 0.15, 0.15]
            _, info = scene.compound(f"tote{k}", boxes, (x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02),
                                                         z + hz + 0.002),
                                     yaw=rng.uniform(-0.05, 0.05), mu=float(rng.uniform(0.2, 0.6)),
                                     mu_grip=float(rng.uniform(0.4, 0.9)), timeconst=0.01, kind="tote", rgba=BLUE_TOTE,
                                     mass_split=ms, extra_hidden=dict(content_mass=content))
            info.shape = "tote"
        self.cart_slots = [(-0.22, 0.0), (0.22, 0.0), (0.0, 0.0)]
        self.safe_z = 0.85 + 2 * hz + 0.12 + 0.05
        standard_mounts(scene, (0.0, -0.1), 0.0, (0.5, 0.35), gantry_h=1.75)
        scene.mounts["arm"] = [((0.0, -0.12, 0.7), math.pi / 2)]     # pedestal between the cart and the rack
        scene.camera = dict(pos=(0.0, -0.6, 2.2), lookat=(0.0, 0.0, 0.6), view_pos=(1.5, -1.6, 1.7),
                            view_lookat=(0.0, 0.0, 0.6))
        self.budget = int(params.get("budget", 3 * n))

    def totes(self):
        return self.objects("tote")

    def _on_cart(self, e):
        p, q = self.true_pose(e)
        up = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        return self.in_region_true(e, self.scene.regions["cart"], zcheck=True) and up > 0.95

    def success(self):
        return all(self._on_cart(e) for e in self.totes())

    def progress(self):
        t = self.totes()
        return sum(self._on_cart(e) for e in t) / len(t)

    def fatal(self):
        return self.metrics_acc["drops"] > 0

    def _remaining(self):
        out = []
        for e in self.totes():
            est = self.est_pose(e)
            if est is not None and est[0][1] > -0.1:
                out.append((est[0][1], e))        # front tier first
        return [e for _, e in sorted(out)]

    def _pick_call(self, e, rng=None, **kw):
        p = dict(grip_force=40.0, lift_speed=0.25, depth=0.5, approach_speed=0.5, lift_height=0.15)
        if rng is not None:
            p.update(grip_force=float(rng.uniform(10, 120)), lift_speed=float(rng.uniform(0.05, 0.6)),
                     depth=float(rng.uniform(0.2, 0.7)))
        p.update(kw)
        return SkillCall("pick", p, target=e)

    def _place_call(self, rng=None):
        n_on = sum(1 for e in self.totes() if self.est_pose(e) is not None and self.est_pose(e)[0][1] < -0.2)
        sx, sy = self.cart_slots[min(n_on, len(self.cart_slots) - 1)]
        p = dict(dx=sx, dy=sy, release_h=0.01, speed=0.35, lower_speed=0.1)
        if rng is not None:
            p.update(release_h=float(rng.uniform(0.0, 0.05)), speed=float(rng.uniform(0.1, 0.7)))
        return SkillCall("place", p, target="cart")

    def candidates(self, n, rng):
        if self.held is not None:
            return [self._place_call(rng) for _ in range(n)]
        rem = self._remaining()
        return [self._pick_call(rem[i % len(rem)], rng) for i in range(n)] if rem else []

    def scripted(self):
        if self.held is not None:
            return self._place_call()
        rem = self._remaining()
        if not rem:
            return None
        cand = [e for e in rem if self.fail_count.get(e, 0) < 3] or rem
        e = cand[0]
        k = self.fail_count.get(e, 0)
        return self._pick_call(e, grip_force=(80.0, 140.0, 150.0)[min(k, 2)], lift_speed=(0.15, 0.06, 0.04)[min(k, 2)],
                               depth=(0.5, 0.5, 0.3)[min(k, 2)])


# ====================================================================== (i) deformable pouch
class PouchTransfer(StationEnv):
    family = "pouch_transfer"
    category = "deformable"
    station = "pouch_bench"
    tool = "parallel"
    probe_kw = dict(max_open=0.06, grip_max=60.0)
    settle_s = 0.8
    timestep = 0.001      # the 3-D flex goes unstable at 2 ms for the stiffer pouches
    success_doc = "the soft pouch (3-D flexcomp) ends with its centroid inside the tote region"
    hidden_doc = ("pouch stiffness (Young 2e3-3e4 Pa), mass 0.1-0.6 kg, friction 0.3-0.9: a light pinch holds a soft "
                  "pouch (it wraps the pads) but a stiff one slips out -> grip force depends on stiffness")

    def build(self, scene, rng, params):
        from .stations import bench_station
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.25
        young = float(params.get("young", np.exp(rng.uniform(np.log(2e3), np.log(3e4)))))
        mass = float(params.get("mass", rng.uniform(0.1, 0.6)))
        mu = float(rng.uniform(0.3, 0.9))
        size = (0.07, 0.12, 0.04)
        cx, cy = -0.18 + rng.uniform(-0.03, 0.03), rng.uniform(-0.04, 0.04)
        scene.flex_pouch("pouch", (cx - size[0] / 2, cy - size[1] / 2, top + 0.006), size, count=(3, 4, 2), mass=mass,
                         young=young, mu=mu)
        tx, ty = 0.2, 0.05
        tfloor = scene.open_box((tx, ty, top), (0.14, 0.11), 0.08, rgba=(0.6, 0.2, 0.15, 1), prefix="tote")
        scene.region("tote", (tx, ty, tfloor + 0.05), (0.14, 0.11, 0.08), support_z=tfloor)
        self.hidden_task = dict(young=young, mass=mass, mu=mu)
        self.budget = int(params.get("budget", 6))

    def success(self):
        return self.in_region_true("pouch", self.scene.regions["tote"], margin=0.0) and self.held is None

    def progress(self):
        return float(self.success())

    def candidates(self, n, rng):
        if self.held is not None:
            return [SkillCall("place", dict(release_h=float(rng.uniform(0.01, 0.08)), speed=float(rng.uniform(0.1, 0.6))),
                              target="tote") for _ in range(n)]
        return [SkillCall("pick", dict(grip_force=float(rng.uniform(2, 50)), lift_speed=float(rng.uniform(0.05, 0.4)),
                                       depth=float(rng.uniform(-0.5, 0.5))), target="pouch") for _ in range(n)]

    def scripted(self):
        if self.held is not None:
            return SkillCall("place", dict(release_h=0.03, speed=0.3), target="tote")
        if self.n_calls > 0 and self.success():
            return None
        return SkillCall("pick", dict(grip_force=12.0, lift_speed=0.2, depth=0.0), target="pouch")

    def labels_extra(self):
        return dict(task_hidden=self.hidden_task)
