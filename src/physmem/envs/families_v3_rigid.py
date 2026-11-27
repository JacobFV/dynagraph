"""envs-v3 rigid-body families (agent ENVS-V3): stacking stability and human-robot handover.

stack_overhang  three 10 x 5 x 4 cm blocks with a hidden internal ballast (centre-of-mass offset along the block's
                long axis, up to +-30 % of the half length) and hidden masses / friction are stacked on a pedestal so
                that the top block reaches a stated overhang beyond the pedestal edge (a cantilever / staircase stack).
                A stack stands iff, for every level, the centre of mass of everything above lies over the support
                below - so the safe offsets depend on the hidden COMs and masses. Probe-then-act: lift each block
                once and read its mass and COM offset from the wrist F/T (torque / force).
handover        see class Handover.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..v2.api import SkillCall
from .families_v3_common import V3Base
from .scene import quat_z
from .stations import bench_station

G9 = 9.81


def _greedy(L, masses, offs, margin):
    n = len(masses)
    rel = [0.0] * n                 # rel[k] = x[k] - x[k-1]
    for k in range(n - 1, 0, -1):
        M = sum(masses[k:])
        s, cum = 0.0, 0.0
        for j in range(k, n):
            if j > k:
                cum += rel[j]
            s += masses[j] * (cum + offs[j])
        rel[k] = L / 2 - margin - s / M          # COM(k..top) sits `margin` inside the outer edge of block k-1
    M = sum(masses)
    s, cum = 0.0, 0.0
    for j in range(n):
        if j > 0:
            cum += rel[j]
        s += masses[j] * (cum + offs[j])
    x = [-margin - s / M]                        # COM(all) sits `margin` inside the pedestal edge
    for k in range(1, n):
        x.append(x[-1] + rel[k])
    return x, x[-1] + L / 2


def stack_plan(L, masses, offs, D, margin=0.008, maximal=False):
    """block-centre x positions (relative to the pedestal edge, +x outward) for a bottom-to-top stack from per-block
    masses and COM offsets (along +x). Greedy from the top: every sub-stack's COM sits `margin` inside its supporting
    edge (the maximal-overhang stack). If not `maximal`, the margin is increased (bisection) until the top reaches only
    D + 1 cm: every stability margin is then >= `margin`. Returns (x list, top overhang)."""
    x, top = _greedy(L, masses, offs, margin)
    if maximal or top <= D + 0.01:
        return x, top
    lo, hi = margin, L / 2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _greedy(L, masses, offs, mid)[1] > D + 0.01:
            lo = mid
        else:
            hi = mid
    return _greedy(L, masses, offs, lo)


class StackOverhang(V3Base):
    family = "stack_overhang"
    station = "stack_bench"
    probe_kw = dict(max_open=0.05, grip_max=80.0)
    timestep = 0.002
    settle_s = 0.8
    budget = 12
    HALF = (0.05, 0.025, 0.02)
    success_doc = ("stack the three blocks in order (block0 at the bottom) on the pedestal so that the top block's outer "
                   "face reaches the stated overhang D beyond the pedestal edge (goal 'overhang_m'); 3 s after the last "
                   "release every block rests on the stack (none on the bench / floor), released")
    hidden_doc = ("per block: mass 0.15-0.8 kg, internal ballast -> centre-of-mass offset along the long axis up to "
                  "+-15 mm (+-30 % of the half length), block<->block friction 0.3-0.9. The offsets / masses decide the "
                  "largest stable overhang per level: a geometric (uniform-block) plan topples when the ballast sits "
                  "outward")

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.30
        hx, hy, hz = self.HALF
        L = 2 * hx
        # pedestal: static block, top 6 cm above the bench, outer edge at x = ex (blocks overhang towards +x)
        ex, ey = 0.10 + rng.uniform(-0.02, 0.02), 0.08 + rng.uniform(-0.02, 0.02)
        ph = 0.06
        scene.static_box((ex - 0.06, ey, top + ph / 2), (0.06, 0.05, ph / 2), rgba=(0.35, 0.35, 0.4, 1), mu=0.5,
                         name="pedestal")
        self.edge = (ex, ey, top + ph)
        self.blocks = []
        masses, offs = [], []
        for i in range(3):
            m = float(params.get(f"mass{i}", rng.uniform(0.15, 0.8)))
            e = float(params.get(f"com{i}", rng.uniform(-0.3, 0.3) * hx))
            mu = float(rng.uniform(0.3, 0.9))
            # two halves of the block: the ballast imbalance sets the COM offset e (mass split between x<0 / x>0)
            # COM of halves at -hx/2, +hx/2: e = (m_r - m_l) / m * hx / 2 -> m_r = m (1/2 + e / hx)
            mr = m * (0.5 + e / hx)
            mr = float(np.clip(mr, 0.02 * m, 0.98 * m))
            e = (2 * mr - m) / m * hx / 2
            bx, by = -0.18 + 0.12 * i + rng.uniform(-0.01, 0.01), -0.12 + rng.uniform(-0.02, 0.02)
            yaw = float(rng.uniform(-0.25, 0.25))
            eid = f"block{i}"
            _, info = scene.compound(eid, [((-hx / 2, 0, 0), (hx / 2, hy, hz)), ((hx / 2, 0, 0), (hx / 2, hy, hz))],
                                     (bx, by, top + hz + 0.001), yaw=yaw, mass=m, mu=mu, mu_grip=0.9, restitution=0.05,
                                     timeconst=0.01, kind="block", rgba=(0.8, 0.55 - 0.15 * i, 0.2 + 0.2 * i, 1),
                                     mass_split=[m - mr, mr], extra_hidden=dict(com_off=e))
            info.shape = "box"
            self.blocks.append(eid)
            masses.append(m)
            offs.append(e)
            scene.region(f"level{i}", (ex, ey, top + ph + (2 * i + 1) * hz), (hx, hy, hz), yaw=0.0,
                         support_z=top + ph + 2 * i * hz, sym=2 * math.pi)
        # target overhang: a fraction of the block length (uniform blocks reach 11/12 L)
        self.D = float(params.get("overhang_m", rng.uniform(0.55, 0.72) * L))
        _, best = stack_plan(L, masses, offs, self.D, margin=0.006, maximal=True)
        self.hidden_task = dict(masses=masses, com_offs=offs, oracle_max_overhang=best, feasible=bool(best >= self.D))
        self.budget = int(params.get("budget", 12))
        self.n_placed = 0
        self._est = {}
        self._grasp_off = {}

    # ------------------------------------------------------------------ predicates (true state)
    def _block_state(self, eid):
        p, q = self.true_pose(eid)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return p, R.reshape(3, 3)

    def on_stack(self, i):
        p, R = self._block_state(self.blocks[i])
        lvl = self.scene.regions[f"level{i}"]
        return abs(p[2] - lvl["center"][2]) < 0.008 and R[2, 2] > 0.98 and abs(p[1] - self.edge[1]) < 0.06

    def overhang(self):
        p, R = self._block_state(self.blocks[-1])
        hx = self.HALF[0]
        return float(max(p[0] + abs(R[0, 0]) * hx, p[0] - abs(R[0, 0]) * hx) - self.edge[0])

    def success(self):
        return (self.held is None and self.n_placed >= 3 and all(self.on_stack(i) for i in range(3))
                and self.overhang() >= self.D)

    def progress(self):
        st = sum(self.on_stack(i) for i in range(3)) / 3
        oh = float(np.clip(self.overhang() / self.D, 0, 1)) if self.on_stack(2) else 0.0
        return 0.7 * st + 0.3 * oh

    def fatal(self):
        return self.metrics_acc["drops"] > 0

    def goal_extra(self):
        return dict(overhang_m=self.D, edge=self.edge, order=list(self.blocks), block_half=self.HALF)

    def metrics_extra(self):
        return dict(x_overhang=self.overhang(), x_on_stack=float(sum(self.on_stack(i) for i in range(3))))

    def labels_extra(self):
        return dict(task_hidden=dict(self.hidden_task), est=dict(self._est))

    # ------------------------------------------------------------------ skills
    def fsk_grasp_block(self, M, call):
        """pinch the block across its width at the tracked centre and lift. params: grip_force, lift (m), dx (grasp
        offset along the block's long axis, m). Records the F/T mass / COM-offset reading of the held block."""
        p = call.params
        est = self.est_pose(call.target)
        if est is None:
            return "not_observed", {}
        pos, _, yb = est
        hx, hy, hz = self.HALF
        u = np.array([math.cos(yb), math.sin(yb)])
        g = pos[:2] + u * float(p.get("dx", 0.0))
        yaw = yb                                # pads close along the block's y (hand y = block y)
        tcp, y_now = M.tcp()
        if abs(math.remainder(yaw - y_now, 2 * math.pi)) > math.pi / 2:
            yaw = math.remainder(yaw + math.pi, 2 * math.pi)
        z = max(pos[2], pos[2] - hz + M.tip() + 0.004)
        top = max(self.safe_z, z + 0.12)
        M.open()
        for q, v in (((tcp[0], tcp[1], top), 0.5), ((g[0], g[1], top), 0.5), ((g[0], g[1], z + 0.03), 0.3),
                     ((g[0], g[1], z), 0.08)):
            if M.move(q, yaw, 0.0, v) == "unreachable":
                return "unreachable", {}
        w0 = M.wrench()
        M.grip(float(p.get("grip_force", 25.0)))
        held0 = M.holding()
        M.move((g[0], g[1], z + float(p.get("lift", 0.06))), yaw, 0.0, 0.1)
        M.wait(0.6)
        if not M.holding():
            M.open()
            return ("missed" if not held0 else "slipped"), {}
        w = M.wrench() - w0 * 0.0
        self.held = call.target
        # actor-legal F/T reading (world-aligned, tared at t=0 with the hand empty): load F_z = m g, torque about the
        # world y axis = (COM_x - TCP_x) m g (sign convention measured, see docs) -> COM offset along the block axis
        m_est = float(abs(w[2]) / G9)
        r_x = float(-w[4] / max(abs(w[2]), 1e-3))
        r_y = float(w[3] / max(abs(w[2]), 1e-3))
        e_axis = r_x * u[0] + r_y * u[1] + float(p.get("dx", 0.0))       # COM offset from the block centre, along +u
        self._est[call.target] = dict(mass=m_est, com=e_axis, yaw=yb)
        self._grasp_off[call.target] = (float(p.get("dx", 0.0)), float(math.remainder(yb - yaw, 2 * math.pi)),
                                        float(z - (pos[2] - hz)))
        return "done", dict(m_est=m_est, com_est=e_axis)

    def fsk_put_back(self, M, call):
        """set the held block down where it was picked (after a weighing lift) and release."""
        if self.held is None:
            return "empty", {}
        tcp, yaw = M.tcp()
        dzg = self._grasp_off[self.held][2]
        M.move((tcp[0], tcp[1], self.surface_z + dzg + 0.002), yaw, 0.0, 0.05)
        M.open()
        self.held = None
        M.move((tcp[0], tcp[1], tcp[2] + 0.05), yaw, 0.0, 0.2)
        return "done", {}

    def fsk_stack_place(self, M, call):
        """put the held block on level `level` with its centre at x_off beyond the pedestal edge (block long axis
        along +x, COM-offset sign per `flip`: 0 keep the block's tracked orientation class, the skill picks the yaw
        nearest to 0 / pi that the caller asked for via `yaw_goal`). params: level, x_off, yaw_goal (0 | pi),
        release_h, lower_speed, dwell, settle (s to wait after release)."""
        p = call.params
        if self.held is None:
            return "empty", {}
        eid = self.held
        k = int(p.get("level", self.n_placed))
        reg = self.scene.regions[f"level{k}"]
        dxg, yoff, dzg = self._grasp_off[eid]
        ygoal = float(p.get("yaw_goal", 0.0))
        hand_yaw = math.remainder(ygoal - yoff, 2 * math.pi)        # block yaw = hand yaw + yoff
        ub = np.array([math.cos(ygoal), math.sin(ygoal)])
        cx = self.edge[0] + float(p.get("x_off", 0.0))
        tgt = np.array([cx, self.edge[1]]) + ub * dxg
        z = reg["support_z"] + dzg + float(np.clip(p.get("release_h", 0.002), 0.0, 0.03))
        tcp, y_now = M.tcp()
        top = max(self.safe_z, z + 0.08)
        for q, v in (((tcp[0], tcp[1], top), 0.4), ((tgt[0], tgt[1], top), 0.4), ((tgt[0], tgt[1], z + 0.03), 0.2)):
            if M.move(q, hand_yaw, 0.0, v) == "unreachable":
                return "unreachable", {}
        # in-hand correction from the tracker (actor-legal): measured block centre vs TCP
        M.wait(0.3)
        est = self.est_pose_avg(eid, 0.3)
        if est is not None and p.get("correct", 1):
            tcp2, _ = M.tcp()
            want = np.array([cx, self.edge[1]])
            err = est[0][:2] - want
            if np.all(np.abs(err) < 0.03):
                tgt = tgt - err
                M.move((tgt[0], tgt[1], z + 0.03), hand_yaw, 0.0, 0.05)
        M.move((tgt[0], tgt[1], z), hand_yaw, 0.0, float(np.clip(p.get("lower_speed", 0.03), 0.005, 0.2)))
        M.wait(float(np.clip(p.get("dwell", 0.2), 0.0, 2.0)))
        M.open()
        self.held = None
        self.n_placed = max(self.n_placed, k + 1)
        tcp, _ = M.tcp()
        M.move((tcp[0], tcp[1], tcp[2] + 0.06), hand_yaw, 0.0, 0.1)
        M.wait(float(np.clip(p.get("settle", 0.5), 0.0, 5.0)))
        ok = all(self.on_stack(i) for i in range(k + 1))       # evaluator check only for the status tag
        return ("done" if ok else "toppled"), {}

    # ------------------------------------------------------------------ policies
    NOMINAL = dict(grip_force=25.0, margin=0.005)

    def _yaw_goal(self, eid):
        """a block is placed keeping its yaw class (0 if it lies near yaw 0, else pi): no wrist flip. Its block-frame
        COM offset then points along world +x (yaw 0) or -x (yaw pi)."""
        est = self.est_pose(eid)
        yb = est[2] if est is not None else 0.0
        return 0.0 if abs(math.remainder(yb, 2 * math.pi)) <= math.pi / 2 else math.pi

    def _plan_world(self, src):
        """x offsets of the three block centres beyond the pedestal edge from per-block (mass, COM offset):
        src 'true' (hidden values), 'est' (wrist F/T readings), 'uniform' (equal masses, centred COMs)."""
        L = 2 * self.HALF[0]
        ms, es = [], []
        for i, b in enumerate(self.blocks):
            sgn = 1.0 if abs(self._yaw_goal(b)) < 1 else -1.0
            if src == "true":
                ms.append(self.hidden_task["masses"][i])
                es.append(sgn * self.hidden_task["com_offs"][i])
            elif src == "est" and b in self._est:
                ms.append(self._est[b]["mass"])
                es.append(sgn * self._est[b]["com"])
            else:
                ms.append(0.45)
                es.append(0.0)
        x, _ = stack_plan(L, ms, es, self.D, margin=self.NOMINAL["margin"])
        return x

    def _next(self, src):
        k = self.n_placed
        if k >= 3:
            return None
        eid = self.blocks[k]
        if self.held is None:
            if self.fail_count.get(eid, 0) >= 3:
                return None
            return SkillCall("grasp_block", dict(grip_force=self.NOMINAL["grip_force"], lift=0.06), target=eid)
        x = self._plan_world(src)
        return SkillCall("stack_place", dict(level=k, x_off=float(x[k]), yaw_goal=self._yaw_goal(eid), release_h=0.002,
                                             lower_speed=0.03, dwell=0.2, settle=3.0 if k == 2 else 0.5),
                         target=f"level{k}")

    def scripted(self):
        """engineer's heuristic: uniform-block greedy plan (equal masses, centred COMs), margin >= 5 mm, aimed to
        reach D + 1 cm (5 mm minimum margin)."""
        return self._next("uniform")

    def v3_rule(self, call, src):
        """oracle: the same greedy plan with the TRUE masses / COM offsets ('prior' = uniform)."""
        if call is None or call.skill != "stack_place":
            return call
        from dataclasses import replace as _rep
        x = self._plan_world("true" if src == "true" else "uniform")
        return _rep(call, params=dict(call.params, x_off=float(x[int(call.params["level"])])))

    def probe_then_act(self, call):
        """weigh pass first: lift every unplaced block once (wrist F/T -> mass, COM offset) and put it back; then the
        greedy plan with the estimates."""
        if call is None:
            return call
        todo = [b for b in self.blocks[self.n_placed:] if b not in self._est]
        if self.held is not None and (todo or getattr(self, "_weighing", False)) and call.skill == "stack_place":
            if todo or self._weighing:
                self._weighing = bool(todo)
                return SkillCall("put_back", {}, target=self.held)
        if self.held is None and todo:
            self._weighing = True
            return SkillCall("grasp_block", dict(grip_force=self.NOMINAL["grip_force"], lift=0.06), target=todo[0])
        self._weighing = False
        if call.skill == "stack_place":
            from dataclasses import replace as _rep
            x = self._plan_world("est")
            return _rep(call, params=dict(call.params, x_off=float(x[int(call.params["level"])])))
        return call

    def candidates(self, n, rng):
        c = self.scripted()
        if c is None:
            return []
        if c.skill != "stack_place":
            return [c]
        from dataclasses import replace as _rep
        return [_rep(c, params=dict(c.params, x_off=float(c.params["x_off"] + rng.uniform(-0.03, 0.03))))
                for _ in range(n)]


class Handover(V3Base):
    """HUMAN-TO-ROBOT HANDOVER with a scripted kinematic human PROXY (labelled: not a human model).

    The 'human' is a mocap-driven hand (a kinematic driver welded to a hand body) with two force-controlled finger pads
    that hold a tool horizontally by its far end, plus a visual-only forearm. Nothing about it is a biomechanical or
    behavioural human model: its release behaviour is a scripted rule with hidden parameters.
    Hidden per episode: release mode ('timed': lets go tau after it feels the robot's grasp, ignoring pulls;
    'pull': lets go once the robot pulls with >= F_rel), tau 0.3-2.0 s, F_rel 2-9 N, the proxy's grip force 15-30 N,
    tool mass 0.2-0.9 kg. Pulling against a hand that has not let go loads it: the horizontal force on the proxy's
    pads ('tug') must stay below TUG_LIMIT (15 N, a comfort / safety bound) and the tool must not be pulled through
    the proxy's still-closed fingers (> 1 cm slide = 'snatched'). Release is detected by the robot only
    through its own wrist F/T and the tracker (actor-legal).
    """
    family = "handover"
    station = "handover_bench"
    probe_kw = dict(max_open=0.04, grip_max=80.0)
    timestep = 0.002
    settle_s = 0.8
    budget = 6
    TUG_LIMIT = 15.0
    TOOL = (0.07, 0.015, 0.015)
    success_doc = ("take the tool the human proxy presents and put it in the tote: tool in the tote region, released, the "
                   "horizontal force on the proxy's hand never above 15 N, nothing dropped")
    hidden_doc = ("proxy release mode (timed / pull-triggered, 50/50), release delay 0.3-2.0 s after it feels the robot's "
                  "grasp, pull threshold 2-9 N, proxy grip 15-30 N, tool mass 0.2-0.9 kg: pulling before a timed proxy "
                  "lets go -> tug above the limit; waiting for a pull-triggered proxy -> it never lets go")

    def build(self, scene, rng, params):
        top = bench_station(scene)
        self.surface_z = top
        self.safe_z = top + 0.30
        hx, hy, hz = self.TOOL
        mode = params.get("mode", "timed" if rng.random() < 0.5 else "pull")
        tau = float(params.get("tau", rng.uniform(0.3, 2.0)))
        f_rel = float(params.get("f_rel", rng.uniform(2.0, 9.0)))
        g_h = float(params.get("human_grip", rng.uniform(15.0, 30.0)))
        mass = float(params.get("mass", rng.uniform(0.2, 0.9)))
        # handover pose: the tool's near end over the bench centre, far end in the proxy's hand at +x
        cx, cy = 0.02 + rng.uniform(-0.03, 0.03), 0.06 + rng.uniform(-0.03, 0.03)
        cz = top + 0.16 + rng.uniform(-0.02, 0.02)
        _, info = scene.part("tool", "box", (hx, hy, hz), (cx, cy, cz), yaw=0.0, mass=mass, mu=0.5, mu_grip=0.8,
                             restitution=0.05, timeconst=0.01, kind="tool", rgba=(0.9, 0.45, 0.1, 1))
        # proxy hand: mocap driver + welded hand body with two pads pinching the tool's far end (across y)
        hpos = np.array([cx + hx - 0.03, cy, cz])
        mb = scene.spec.worldbody.add_body(name="human_mocap", pos=list(hpos), mocap=True)
        mb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004, 0, 0], contype=0, conaffinity=0, rgba=[0, 0, 0, 0],
                    group=2)
        hb = scene.spec.worldbody.add_body(name="human_hand", pos=list(hpos))
        hb.add_freejoint(name="human_hand_free")
        palm = hb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.04, 0.012], pos=[0.035, 0, 0.03],
                           rgba=[0.85, 0.7, 0.6, 1], name="human_palm", contype=0, conaffinity=0, group=2)
        palm.mass = 1.5                      # hand + forearm share: with the stiff weld the proxy hand holds still
        hb.gravcomp = 1.0                    # the proxy 'arm' carries its own hand's weight
        pads = []
        for sgn in (-1, 1):
            f = hb.add_body(name=f"human_f{sgn > 0:d}", pos=[0, sgn * (hy + 0.012), 0])
            f.add_joint(name=f"human_jf{sgn > 0:d}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, -sgn, 0],
                        range=[-0.03, 0.02], limited=True, damping=20.0, armature=0.01)
            g = f.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.006, 0.016], rgba=[0.85, 0.7, 0.6, 1],
                           friction=[1.0, 0.005, 0.0001], condim=4, name=f"human_pad{sgn > 0:d}")
            g.mass = 0.05
            pads.append(g.name)
            a = scene.spec.add_actuator(name=f"human_af{sgn > 0:d}", target=f"human_jf{sgn > 0:d}",
                                        trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_motor()
            a.ctrlrange = [-60, 60]
            a.forcerange = [-60, 60]
            a.ctrllimited = True
            a.forcelimited = True
        scene.spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1="human_mocap", name2="human_hand",
                                objtype=mujoco.mjtObj.mjOBJ_BODY, solref=[-20000.0, -300.0],
                                solimp=[0.95, 0.99, 0.001, 0.5, 2])     # 20 kN/m 'arm' stiffness (direct form)
        scene.spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="human_jf0", name2="human_jf1",
                                data=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        for pg in pads:
            scene.spec.add_pair(geomname1=pg, geomname2="tool_g0", condim=4, friction=[1.0, 1.0, 0.005, 0.0001, 0.0001])
        # visual-only forearm / upper arm (proxy), from the hand to a 'shoulder' beside the bench
        sh = hpos + np.array([0.35, 0.12, 0.25])
        el = hpos + np.array([0.22, 0.05, -0.05])
        for a_, b_ in ((hpos + np.array([0.05, 0, 0.03]), el), (el, sh)):
            scene.spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=list(a_) + list(b_), size=[0.035, 0, 0],
                                          rgba=[0.55, 0.6, 0.75, 0.6], contype=0, conaffinity=0, group=2)
        tx, ty = -0.12, -0.24                 # tote near the robot, clear of the pull path
        tf = scene.open_box((tx, ty, top), (0.1, 0.07), 0.06, rgba=(0.2, 0.45, 0.25, 1), prefix="tote")
        scene.region("tote", (tx, ty, tf + 0.04), (0.1, 0.07, 0.05), support_z=tf)
        scene.region("handover", (cx, cy, cz), (hx, hy, hz), support_z=cz - hz, proxy="scripted_kinematic")
        self.hidden_task = dict(mode=mode, tau=tau, f_rel=f_rel, human_grip=g_h, mass=mass)
        self.hpos0 = hpos
        self.budget = int(params.get("budget", 6))
        self.human = dict(state="holding", t_touch=None, t_release=None, pull_t=0.0, snatched=False)
        self.max_tug = 0.0

    def _index(self):
        super()._index()
        m = self.m
        self._h_act = [m.actuator(f"human_af{k}").id for k in (0, 1)]
        self._h_pads = {m.geom(f"human_pad{k}").id for k in (0, 1)}
        self._mocap = m.body_mocapid[m.body("human_mocap").id]
        self._tool_b = m.body("tool").id
        self._tool_g = m.geom("tool_g0").id
        self._weld_id = [i for i in range(m.neq) if m.eq_type[i] == mujoco.mjtEq.mjEQ_WELD
                         and m.body(m.eq_obj2id[i]).name == "human_hand"][0]

    def pre_physics(self):
        d = self.d
        h = self.human
        g = self.hidden_task["human_grip"]
        # proxy hand: still at the handover pose with a slight tremor (2 mm, 1.3 Hz); retracts after releasing
        t = self.t
        trem = 0.002 * np.array([math.sin(8.2 * t), math.sin(6.1 * t + 1.0), math.sin(7.3 * t + 2.0)])
        pos = self.hpos0 + trem
        if h["state"] == "released" and h["t_release"] is not None:
            k = min(1.0, (t - h["t_release"]) / 1.0)
            pos = pos + k * np.array([0.12, 0.08, 0.05])
        d.mocap_pos[self._mocap] = pos
        u = g if h["state"] == "holding" else -15.0       # closing force (N) / open
        for a in self._h_act:
            d.ctrl[a] = u
        super().pre_physics()

    def _tug(self):
        """horizontal force the proxy's arm exerts on its hand beyond its pre-grasp holding load = the weld (mocap ->
        hand) constraint force, horizontal part, minus its average while the proxy held the tool alone (evaluator;
        also drives the pull-triggered release)."""
        rows = self._weld_rows()
        if not len(rows):
            return 0.0
        f = self.d.efc_force[rows[:3]][:2]
        h = self.human
        if h["t_touch"] is None:
            r0 = getattr(self, "_f_rest", None)
            self._f_rest = f.copy() if r0 is None else 0.9 * r0 + 0.1 * f
            return 0.0
        return float(np.linalg.norm(f - getattr(self, "_f_rest", np.zeros(2))))

    def _weld_rows(self):
        d = self.d
        return np.nonzero((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) &
                          (d.efc_id[:d.nefc] == self._weld_id))[0]

    def _robot_touch(self):
        """the proxy 'feels' the robot's grasp: robot PAD contact on the tool with >= 2 N normal force."""
        m, d = self.m, self.d
        f6 = np.zeros(6)
        tot = 0.0
        for i in range(d.ncon):
            g1, g2 = d.contact.geom1[i], d.contact.geom2[i]
            if (g1 == self._tool_g and self._pad_arr[g2]) or (g2 == self._tool_g and self._pad_arr[g1]):
                mujoco.mj_contactForce(m, d, i, f6)
                tot += abs(f6[0])
        return tot >= 2.0

    def post_physics(self, stop=None):
        out = super().post_physics(stop)
        if self._settling or self._step_i % 5:
            return out
        h, ht = self.human, self.hidden_task
        if h["state"] == "holding":
            tug = self._tug()
            if h["t_touch"] is not None:          # the hand-over starts when the proxy feels the robot's grasp
                self.max_tug = max(self.max_tug, tug)
            # the tool sliding out of the proxy's still-closed hand (> 1 cm) = snatched: a failure like the tug limit
            hb = self.m.body("human_hand").id
            rel = self.d.xpos[self._tool_b] - self.d.xpos[hb]
            if h["t_touch"] is None or getattr(self, "_rel0", None) is None:
                self._rel0 = rel.copy()          # reference taken while the robot grasps (snatch = slide after that)
            ax = self.d.xmat[self._tool_b].reshape(3, 3)[:, 0]          # slide ALONG the tool axis (pivoting is fine)
            if abs(float((rel - self._rel0) @ ax)) > 0.01 and not h["snatched"]:
                h["snatched"] = True
            if h["t_touch"] is None and self._robot_touch():
                h["t_touch"] = self.t
            if ht["mode"] == "timed":
                if h["t_touch"] is not None and self.t - h["t_touch"] >= ht["tau"]:
                    h["state"], h["t_release"] = "released", self.t
            else:
                h["pull_t"] = h["pull_t"] + 5 * self.m.opt.timestep if tug >= ht["f_rel"] else 0.0
                if h["pull_t"] >= 0.1:
                    h["state"], h["t_release"] = "released", self.t
        return out

    # ------------------------------------------------------------------ predicates
    def success(self):
        return (self.held is None and self.in_region_true("tool", self.scene.regions["tote"], 0.01)
                and self.max_tug <= self.TUG_LIMIT and self.human["state"] == "released" and not self.human["snatched"])

    def progress(self):
        return float(0.4 * (self.human["state"] == "released") + 0.3 * (self.max_tug <= self.TUG_LIMIT)
                     + 0.3 * self.in_region_true("tool", self.scene.regions["tote"], 0.01))

    def fatal(self):
        return self.metrics_acc["drops"] > 0 or self.max_tug > self.TUG_LIMIT or self.human["snatched"]

    def metrics_extra(self):
        return dict(tug_n=self.max_tug, x_released=float(self.human["state"] == "released"),
                    x_snatched=float(self.human["snatched"]))

    def labels_extra(self):
        return dict(task_hidden=dict(self.hidden_task), human=dict(self.human))

    # ------------------------------------------------------------------ skills
    def fsk_take(self, M, call):
        """grasp the tool's free (near, -x) end top-down and take it. params: grip_force, wait_s (hold still after the
        grasp), mode ('position': pull away at pull_speed | 'compliant': pull while the wrist F/T horizontal force <
        f_max, else hold and retry), pull_speed, f_max, pull_dist, timeout_s."""
        p = call.params
        est = self.est_pose("tool")
        if est is None:
            return "not_observed", {}
        pos, quat, yt = est
        hx = self.TOOL[0]
        u = np.array([math.cos(yt), math.sin(yt)])
        Rt = np.zeros(9)
        mujoco.mju_quat2Mat(Rt, np.asarray(quat, float))
        ax3 = Rt.reshape(3, 3)[:, 0]                 # tool axis incl. its droop in the proxy's pinch (tracker tilt)
        g3 = np.asarray(pos, float) - ax3 * (hx - 0.025)
        g = g3[:2]
        yaw = yt
        tcp, y_now = M.tcp()
        if abs(math.remainder(yaw - y_now, 2 * math.pi)) > math.pi / 2:
            yaw = math.remainder(yaw + math.pi, 2 * math.pi)
        z = float(g3[2])
        M.open()
        for q, v in (((tcp[0], tcp[1], z + 0.12), 0.4), ((g[0], g[1], z + 0.12), 0.4), ((g[0], g[1], z + 0.02), 0.2),
                     ((g[0], g[1], z), 0.05)):
            if M.move(q, yaw, 0.0, v) == "unreachable":
                return "unreachable", {}
        w0 = M.wrench()
        M.grip(float(p.get("grip_force", 25.0)))
        if not M.holding():
            M.open()
            return "missed", {}
        M.wait(float(np.clip(p.get("wait_s", 0.5), 0.0, 5.0)))
        away = -u                                    # pull along the tool axis, away from the proxy
        dist = float(np.clip(p.get("pull_dist", 0.12), 0.03, 0.3))
        log = {}
        if p.get("mode", "position") == "position":
            here, _ = M.tcp()
            M.move((here[0] + away[0] * dist, here[1] + away[1] * dist, here[2] + 0.04), yaw, 0.0,
                   float(np.clip(p.get("pull_speed", 0.15), 0.02, 0.6)))
        else:
            fmax = float(np.clip(p.get("f_max", 8.0), 1.0, 30.0))
            t_end = self.t + float(np.clip(p.get("timeout_s", 6.0), 0.5, 15.0))
            step = 0.01
            moved = 0.0
            while moved < dist and self.t < t_end:
                here, _ = M.tcp()
                M.move((here[0] + away[0] * step, here[1] + away[1] * step, here[2] + 0.002), yaw, 0.0,
                       float(np.clip(p.get("pull_speed", 0.05), 0.01, 0.3)), min_T=0.1)
                w = M.wrench(window=0.1) - w0
                fh = float(np.linalg.norm(w[:2]))
                if fh > fmax:                          # the hand still holds: back off 5 mm and wait
                    here, _ = M.tcp()
                    M.move((here[0] - away[0] * 0.005, here[1] - away[1] * 0.005, here[2]), yaw, 0.0, 0.05, min_T=0.1)
                    M.wait(0.25)
                else:
                    moved += step
            log["moved"] = moved
        M.wait(0.3)
        if not M.holding():
            M.open()
            return "dropped", log
        if self.human["snatched"] or self.max_tug > self.TUG_LIMIT:
            self.held = "tool" if M.holding() else None
            return "tugged", log
        if self.human["state"] != "released":
            # still held by both (evaluator-side status tag only; the actor sees the F/T)
            self.held = "tool"
            return "tug_hold", log
        self.held = "tool"
        return "done", log

    def fsk_stow(self, M, call):
        """carry the held tool to the tote and release it."""
        if self.held != "tool":
            return "empty", {}
        reg = self.scene.regions["tote"]
        cx, cy, _ = reg["center"]
        tcp, yaw = M.tcp()
        top = max(self.safe_z, tcp[2])
        z = reg["support_z"] + 0.06
        for q in ((tcp[0], tcp[1], top), (cx, cy, top), (cx, cy, z)):
            if M.move(q, yaw, 0.0, float(call.params.get("speed", 0.3))) == "unreachable":
                return "unreachable", {}
        M.open()
        self.held = None
        M.move((cx, cy, top), yaw, 0.0, 0.3)
        M.wait(0.5)
        return "done", {}

    # ------------------------------------------------------------------ policies
    NOMINAL = dict(grip_force=25.0, wait_s=0.5, mode="position", pull_speed=0.15, pull_dist=0.12)

    def scripted(self):
        """engineer's heuristic: grasp, wait 0.5 s, pull away at 0.15 m/s; stow."""
        if self.held == "tool":
            return SkillCall("stow", dict(speed=0.3), target="tote")
        if self.human["state"] == "released" and self.held is None:
            return None
        if self.fail_count.get("tool", 0) >= 2 or self.n_calls >= 3:
            return None
        return SkillCall("take", dict(self.NOMINAL), target="tool")

    @staticmethod
    def _grip_for(mass):
        return float(np.clip(2.5 * mass * (G9 + 2.0) / (2 * 0.8), 10.0, 40.0))

    def v3_rule(self, call, src):
        """oracle: timed proxy -> wait tau + 0.25 s then pull; pull-triggered -> no wait, pull (the position pull's
        F/T rises past F_rel well before 15 N); grip from the tool mass."""
        if call is None or call.skill != "take":
            return call
        from dataclasses import replace as _rep
        h = self.hidden_task
        if src != "true":
            return _rep(call, params=dict(call.params, mode="compliant", f_max=8.0, pull_speed=0.05))
        p = dict(call.params, grip_force=self._grip_for(h["mass"]), mode="position")
        p["wait_s"] = h["tau"] + 0.25 if h["mode"] == "timed" else 0.1
        p["pull_speed"] = 0.1
        return _rep(call, params=p)

    def probe_then_act(self, call):
        """reactive take (no hidden knowledge): compliant pull - advance while the wrist F/T horizontal force stays
        under 8 N (feels the proxy still holding), else back off and wait."""
        if call is None or call.skill != "take":
            return call
        from dataclasses import replace as _rep
        return _rep(call, params=dict(call.params, mode="compliant", f_max=8.0, pull_speed=0.05, timeout_s=6.0,
                                      wait_s=0.2))

    def candidates(self, n, rng):
        if self.held == "tool":
            return [SkillCall("stow", dict(speed=0.3), target="tote")]
        return [SkillCall("take", dict(grip_force=float(rng.uniform(10, 40)), wait_s=float(rng.uniform(0, 2.5)),
                                       mode=str(rng.choice(["position", "compliant"])),
                                       pull_speed=float(rng.uniform(0.03, 0.3)), f_max=float(rng.uniform(3, 14)),
                                       pull_dist=0.12), target="tool") for _ in range(n)]
