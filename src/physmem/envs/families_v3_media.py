"""envs-v3 particle-media families (agent ENVS-V3): pouring and granular scooping.

LIQUID APPROXIMATION (pour) - read before using the data. MuJoCo has no fluid solver. The "liquid" is 45-100 small
rigid spheres (r = 6.5 mm) in an opaque vessel. Hidden per-lot "viscosity proxy" = sphere<->sphere / sphere<->wall
friction (0.0-0.5) plus contact damping; hidden fill mass = sphere count x per-sphere mass (density class). What it
reproduces: a fill-dependent onset angle (a fuller vessel starts pouring at a lower tilt), a flow rate that drops with
the friction proxy, a lag between tilting back and the end of the flow (material in flight / on the lip), and spill
when the stream misses. What it does NOT reproduce: surface tension, a continuous stream, splashing, wetting /
dripping, rate-dependent (true viscous) shear - 'viscous' particles here are frictional, i.e. closer to a dense
slurry or a granular medium than to oil; pour-rate quantisation is one sphere (1.6-3 g). Treat the family as
"pouring a flowable medium", not as fluid dynamics.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..v2.api import SkillCall
from .families_v3_common import V3Base, hex_fill, vessel
from .probe import wrap
from .stations import bench_station

G = 9.81


class Pour(V3Base):
    family = "pour"
    station = "pour_bench"
    tilt_probe = True
    probe_kw = dict(max_open=0.03, grip_max=80.0)
    timestep = 0.002
    settle_s = 1.2
    budget = 6
    success_doc = ("pour the stated target mass (40-90 g, goal 'target_kg') from the opaque vessel into the receiving "
                   "container on the scale: |poured - target| <= 15 g, spilled <= 10 g, vessel set back upright in its "
                   "home area and released")
    hidden_doc = ("fill: 45-100 spheres x 1.6-2.8 g (fill mass 0.08-0.27 kg, opaque vessel: not visible; weighable "
                  "with the wrist F/T once lifted); viscosity proxy: sphere friction 0.0-0.5 + contact damping "
                  "(flow onset angle / rate / lag after tilt-back). A fuller vessel starts pouring at a lower tilt; a "
                  "'thicker' medium starts late and then avalanches -> the tilt rate and the stop lead depend on both")
    R_P = 0.0065
    R_IN, H = 0.032, 0.075
    TOL, SPILL_TOL = 0.015, 0.010
    SPOUT_W = 0.040

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.30
        n = int(params.get("fill_n", rng.integers(45, 101)))
        dens = float(params.get("density", rng.uniform(0.8, 1.4)))
        mu_p = float(params.get("mu_p", rng.uniform(0.0, 0.05)))
        visc = float(params.get("visc", math.exp(rng.uniform(math.log(0.001), math.log(0.05)))))
        damp = float(params.get("p_damp", rng.uniform(0.6, 1.0)))
        m_p = 2.0e-3 * dens
        vx, vy = -0.14 + rng.uniform(-0.02, 0.02), 0.02 + rng.uniform(-0.02, 0.02)
        vyaw = -math.pi / 2 + float(rng.uniform(-0.35, 0.35))        # handle post towards the robot side (-y)
        _, info = vessel(scene, "vessel", (vx, vy, top + 0.004), vyaw, self.R_IN, self.H, mass=0.12, mu=0.5,
                         mu_grip=0.9, spout=(self.SPOUT_W, 0.016),
                         extra_hidden=dict(fill_n=n, fill_kg=n * m_p, density=dens, mu_p=mu_p, visc=visc,
                                                        p_damp=damp))
        c, s = math.cos(vyaw), math.sin(vyaw)
        pts = hex_fill(rng, (0.0, 0.0), self.R_IN, 0.0, self.R_P, n)
        world = [(vx + c * x - s * y, vy + s * x + c * y, top + 0.004 + z) for x, y, z in pts]
        self.add_particles(scene, "liq", world, self.R_P, m_p, mu_p, solref=(0.004, damp), damping=visc)
        rx, ry = 0.13, 0.10
        self.recv_floor = self.add_scale(scene, "receiver", (rx, ry, top + 0.02), (0.06, 0.06), 0.06, k=8000.0)
        scene.region("home", (vx, vy, top + 0.04), (0.05, 0.05, 0.05), support_z=top)
        self.target_kg = float(params.get("target_kg", min(rng.uniform(0.04, 0.09), 0.6 * n * m_p)))
        self.hidden_task = dict(fill_n=n, fill_kg=n * m_p, density=dens, mu_p=mu_p, visc=visc, p_damp=damp, m_p=m_p)
        self.budget = int(params.get("budget", 6))
        self._pour_sign = 1.0
        self.pour_log = []

    # ------------------------------------------------------------------ evaluator-side accounting
    def _vessel_frame(self, P):
        p, q = self.true_pose("vessel")
        import mujoco
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        return (P - p) @ R

    def media_split(self):
        """(in receiver, in vessel, spilled) particle counts from TRUE state (evaluator only)."""
        P = self.particle_pos()
        if not len(P):
            return 0, 0, 0
        s = self._scales["receiver"]
        bp = self.d.xpos[self.m.body("receiver_body").id]
        hx, hy = s["half"]
        rel = P - bp
        in_r = (np.abs(rel[:, 0]) <= hx + 0.002) & (np.abs(rel[:, 1]) <= hy + 0.002) & (rel[:, 2] >= -0.005) & \
               (rel[:, 2] <= s["t"] + s["wall_h"] + 0.02)
        L = self._vessel_frame(P)
        in_v = (np.hypot(L[:, 0], L[:, 1]) <= self.R_IN + 0.003) & (L[:, 2] >= -0.006) & (L[:, 2] <= self.H + 0.01)
        in_v &= ~in_r
        n_r, n_v = int(in_r.sum()), int(in_v.sum())
        return n_r, n_v, int(len(P) - n_r - n_v)

    def poured_kg(self):
        return self.media_split()[0] * self.hidden_task["m_p"]

    def spilled_kg(self):
        return self.media_split()[2] * self.hidden_task["m_p"]

    def vessel_upright_home(self):
        p, q = self.true_pose("vessel")
        up = 1 - 2 * (q[1] ** 2 + q[2] ** 2)          # z-axis z component
        return up > math.cos(math.radians(10)) and self.in_region_true("vessel", self.scene.regions["home"], 0.03,
                                                                       zcheck=False) and p[2] < self.surface_z + 0.02

    def success(self):
        nr, nv, ns = self.media_split()
        m = self.hidden_task["m_p"]
        return (abs(nr * m - self.target_kg) <= self.TOL and ns * m <= self.SPILL_TOL and self.held is None
                and self.vessel_upright_home())

    def progress(self):
        nr, nv, ns = self.media_split()
        m = self.hidden_task["m_p"]
        acc = max(0.0, 1.0 - abs(nr * m - self.target_kg) / max(self.target_kg, 1e-6))
        return float(0.6 * acc + 0.2 * (ns * m <= self.SPILL_TOL) + 0.2 * (self.held is None and self.vessel_upright_home()))

    def fatal(self):
        return self.metrics_acc["drops"] > 0

    def goal_extra(self):
        return dict(target_kg=self.target_kg, tol_kg=self.TOL, spill_tol_kg=self.SPILL_TOL, receiver="receiver",
                    vessel_geometry=dict(r_in=self.R_IN, h=self.H))

    def metrics_extra(self):
        nr, nv, ns = self.media_split()
        m = self.hidden_task["m_p"]
        return dict(poured_kg=nr * m, spilled_kg=ns * m, left_kg=nv * m, target_kg=self.target_kg)

    def labels_extra(self):
        return dict(task_hidden=dict(self.hidden_task), pour_log=list(self.pour_log))

    # ------------------------------------------------------------------ family skills
    def fsk_grasp_vessel(self, M, call):
        """pinch the vessel's handle post across the radial axis (pads on its inner / outer faces) and lift.
        params: grip_force (N), lift (m), lift_speed (m/s)."""
        p = call.params
        est = self.est_pose("vessel")
        if est is None:
            return "not_observed", {}
        pos, _, yv = est
        meta = self.scene.entities["vessel"].meta
        u = np.array([math.cos(yv), math.sin(yv)])
        post = pos[:2] + u * meta["post_r"]
        z_bottom = pos[2] - self.scene.entities["vessel"].half[2]           # tracked centre of the bounding box
        z_grip = z_bottom + meta["post_top"] - 0.018
        yaw = wrap(yv - math.pi / 2)                     # pads close along u
        tcp, y_now = M.tcp()
        if abs(wrap(yaw - y_now)) > math.pi / 2:          # symmetric pinch: keep the wrist near its current yaw
            yaw = wrap(yaw + math.pi)
        hy = np.array([-math.sin(yaw), math.cos(yaw)])
        self._pour_sign = 1.0 if float(hy @ u) > 0 else -1.0     # tilt sign that pours AWAY from the post
        M.open()
        top = max(self.safe_z, z_grip + 0.12)
        for st in (M.move((tcp[0], tcp[1], top), yaw, 0.0, 0.5), M.move((post[0], post[1], top), yaw, 0.0, 0.5),
                   M.move((post[0], post[1], z_grip + 0.02), yaw, 0.0, 0.3), M.move((post[0], post[1], z_grip), yaw, 0.0, 0.08)):
            if st == "unreachable":
                return "unreachable", {}
        M.grip(float(p.get("grip_force", 30.0)))
        held0 = M.holding()
        M.move((post[0], post[1], z_grip + float(p.get("lift", 0.10))), yaw, 0.0, float(p.get("lift_speed", 0.15)))
        M.wait(0.3)
        if not M.holding():
            M.open()
            return ("missed" if not held0 else "slipped"), {}
        self.held = "vessel"
        self._grip_z_off = float(z_grip - z_bottom)
        return "done", dict(weigh_n=float(-M.wrench()[2]))

    def _pour_pose(self, M, tilt_hint=1.5):
        reg = self.scene.regions["receiver"]
        cx, cy, _ = reg["center"]
        tcp, yaw = M.tcp()
        u = self._pour_sign * np.array([-math.sin(yaw), math.cos(yaw)])     # hand y = towards the post
        meta = self.scene.entities["vessel"].meta
        rim = reg["support_z"] + reg["half"][2] * 2
        z = rim + 0.03 + meta["post_r"] + self.R_IN + 0.02
        xy = np.array([cx, cy]) + u * 0.035
        return np.array([xy[0], xy[1], z]), yaw

    def fsk_pour(self, M, call):
        """tilt the held vessel over the receiver. params: tilt_max (rad), tilt_rate (rad/s), hold_s (max time at
        tilt_max), stop_kg (tilt back once the scale reads >= stop_kg; None/-1 = open loop), pre_tilt (rad: fast tilt
        to this angle first), back_rate (rad/s), speed (transport m/s)."""
        p = call.params
        if self.held != "vessel":
            return "empty", {}
        if not M.can_tilt():
            return "unsupported", {}
        pos, yaw = self._pour_pose(M)
        if M.move(pos, yaw, 0.0, float(p.get("speed", 0.25))) == "unreachable":
            return "unreachable", {}
        M.wait(0.3)
        sgn = self._pour_sign
        tmax = float(np.clip(p.get("tilt_max", 1.6), 0.2, 2.2))
        rate = float(np.clip(p.get("tilt_rate", 0.4), 0.05, 3.0))
        pre = float(np.clip(p.get("pre_tilt", 0.0), 0.0, tmax))
        stop = p.get("stop_kg", None)
        stop = None if stop is None or stop < 0 else float(stop)
        t0 = self.t
        log = dict(stop_kg=stop)
        if pre > 0:
            M.move(pos, yaw, sgn * pre, 0.25, min_T=pre / 1.2)
        # tilt in small increments so the scale can stop it
        hit = lambda: stop is not None and (self.scale_reading("receiver") or 0.0) >= stop
        cur = pre
        stopped = False
        trace = []
        while cur < tmax - 1e-6 and not stopped:
            nxt = min(tmax, cur + rate * 0.1)
            st = M.move(pos, yaw, sgn * nxt, 0.25, min_T=0.1, until=hit)
            cur = nxt
            trace.append((round(self.t - t0, 2), round(cur, 3), round(self.scale_reading("receiver") or 0.0, 4)))
            if st == "unreachable":
                log["unreach_tilt"] = cur
                break
            stopped = hit()
        if not stopped:
            hold = float(np.clip(p.get("hold_s", 2.0), 0.0, 8.0))
            M.wait(hold, until=hit if stop is not None else None)
        log["trace"] = trace
        log["t_stop"] = self.t - t0
        log["tilt_at_stop"] = cur
        log["scale_at_stop"] = self.scale_reading("receiver")
        back = float(np.clip(p.get("back_rate", 2.0), 0.3, 4.0))
        M.move(pos, yaw, 0.0, 0.25, min_T=cur / back)
        M.wait(1.0)
        log["scale_end"] = self.scale_reading("receiver", window=0.3)
        self.pour_log.append(log)
        if not M.holding():
            self.held = None
            return "dropped", log
        return "done", log

    def fsk_set_down(self, M, call):
        """put the vessel back upright in its home area and release. params: speed, lower_speed."""
        p = call.params
        if self.held != "vessel":
            return "empty", {}
        reg = self.scene.regions["home"]
        cx, cy, _ = reg["center"]
        tcp, yaw = M.tcp()
        u = self._pour_sign * np.array([-math.sin(yaw), math.cos(yaw)])
        meta = self.scene.entities["vessel"].meta
        xy = np.array([cx, cy]) + u * meta["post_r"]
        z = self.surface_z + self._grip_z_off + 0.004
        M.move((tcp[0], tcp[1], max(tcp[2], z + 0.1)), yaw, 0.0, 0.3)
        M.move((xy[0], xy[1], z + 0.1), yaw, 0.0, float(p.get("speed", 0.25)))
        M.move((xy[0], xy[1], z + 0.005), yaw, 0.0, float(p.get("lower_speed", 0.05)))
        M.wait(0.2)
        M.open()
        self.held = None
        tcp, _ = M.tcp()
        M.move((tcp[0], tcp[1], tcp[2] + 0.12), yaw, 0.0, 0.3)
        M.wait(0.5)
        return "done", {}

    # ------------------------------------------------------------------ policies
    NOMINAL = dict(tilt_max=2.0, tilt_rate=0.35, hold_s=4.0, lead_kg=0.010)

    def _stage(self):
        if self.held is None and not self.pour_log:
            return "grasp"
        if self.held == "vessel" and not self.pour_log:
            return "pour"
        if self.held == "vessel":
            return "set_down"
        return "done"

    def scripted(self):
        """engineer's heuristic (no hidden knowledge): pinch the post at 30 N, tilt at 0.35 rad/s while watching the
        scale, tilt back at target - 10 g (fixed lead), set the vessel down."""
        st = self._stage()
        if st == "grasp":
            if self.fail_count.get("vessel", 0) >= 3:
                return None
            return SkillCall("grasp_vessel", dict(grip_force=30.0, lift=0.10, lift_speed=0.15), target="vessel")
        if st == "pour":
            n = self.NOMINAL
            return SkillCall("pour", dict(tilt_max=n["tilt_max"], tilt_rate=n["tilt_rate"], hold_s=n["hold_s"],
                                          stop_kg=self.target_kg - n["lead_kg"], back_rate=2.0), target="receiver")
        if st == "set_down":
            return SkillCall("set_down", dict(speed=0.25, lower_speed=0.05), target="home")
        return None

    def candidates(self, n, rng):
        st = self._stage()
        if st == "grasp":
            return [SkillCall("grasp_vessel", dict(grip_force=float(rng.uniform(10, 60)), lift=0.10,
                                                   lift_speed=float(rng.uniform(0.05, 0.3))), target="vessel")
                    for _ in range(n)]
        if st == "pour":
            return [SkillCall("pour", dict(tilt_max=float(rng.uniform(1.2, 2.2)), tilt_rate=float(rng.uniform(0.1, 1.0)),
                                           hold_s=float(rng.uniform(1, 5)), pre_tilt=float(rng.uniform(0.0, 1.0)),
                                           stop_kg=float(self.target_kg - rng.uniform(0.0, 0.04)), back_rate=2.0),
                              target="receiver") for _ in range(n)]
        if st == "set_down":
            return [SkillCall("set_down", dict(speed=0.25, lower_speed=0.05), target="home")]
        return []
