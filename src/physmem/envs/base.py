"""StationEnv: the shared implementation of api.IndustrialEnv for every v2 task family (agent ENVS).

Episode = reset(TaskSpec) then a sequence of execute(SkillCall) until success or the skill budget.
Physics only advances through robot actuator torques (and the env-owned conveyor drive). An integrity
monitor checks before every mj_step that no object pose was written, no external force/applied
generalized force is on an object, and no weld/connect equality involves an object.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import mujoco
import numpy as np

from ..schema import EntityType
from ..v2.api import SkillCall, SkillResult, StepOut, TaskSpec, EffortMeter
from .probe import ProbeRobot, ProbeSkills, wrap
from .scene import Scene
from .sensors import ActorView, SensorSuite, quat_yaw
from .props import XProp


MOBILE_ROBOTS = {"tidybot"}


def seed_of(task: TaskSpec, salt="") -> int:
    h = hashlib.sha256(f"{task.family}|{task.seed}|{task.split}|{salt}".encode()).digest()
    return int.from_bytes(h[:8], "little")


class StationEnv:
    family = "base"
    category = "bulk"
    station = "generic"
    tool = "parallel"              # tool the scripted probe uses
    probe_kw: dict = {}
    timestep = 0.002
    settle_s = 0.8
    budget = 20                    # max skill calls per episode
    bulk_sensors = False
    fine_sensors = False
    collision_n = 150.0            # contact normal force over which a robot contact counts as a collision
    success_doc = ""
    hidden_doc = ""
    station_center = (0.0, 0.0)

    def __init__(self, robot_factory=None):
        self.robot_factory = robot_factory
        self.renderer = None

    # ================================================================== API
    def reset(self, task: TaskSpec):
        for attempt in range(4):
            ev = self._reset_once(task, attempt)
            # a scene whose settle dropped an object or went numerically unstable is re-drawn deterministically
            if self.metrics_acc["drops"] == 0 and self.metrics_acc["sim_instabilities"] == 0:
                break
        self.reset_attempts = attempt + 1
        return ev

    def _reset_once(self, task, attempt=0):
        self.task = task
        self.episode_id = f"{task.family}-{task.split}-{task.seed}"
        rng = np.random.default_rng(seed_of(task, salt="" if attempt == 0 else f"redraw{attempt}"))
        self.rng = rng
        self.scene = Scene(rng, timestep=self.timestep, name=self.family)
        self.scene.floor()
        self.scene.overrides = dict(task.params.get("hidden", {}))   # evaluator-only (sensitivity studies)
        # station variant per robot class: mobile manipulators get the lower bench (bench_top param overrides)
        if "bench_top" in task.params:
            self.scene.bench_top_override = float(task.params["bench_top"])
        elif task.robot in MOBILE_ROBOTS:
            from .stations import MOBILE_BENCH_TOP
            self.scene.bench_top_override = MOBILE_BENCH_TOP
        self.held = None
        self._held_off = (0.05, 0.0)
        self.safe_z = 1.05
        self.surface_z = 0.76
        self.build(self.scene, rng, dict(task.params))
        # tracker camera on the FAR side of the station (looking back at the robot) for every robot, so an arm's
        # elbow does not hide the work area and all robots see the same occlusion geometry
        cam = dict(self.scene.camera)
        cp, lk = np.asarray(cam["pos"], float), np.asarray(cam["lookat"], float)
        if cp[1] < lk[1]:
            cam["pos"] = (float(cp[0]), float(2 * lk[1] - cp[1]), float(cp[2]))
            self.scene.camera = cam
        if task.params.get("layout") == "rot180":
            from .scene import rotate_station
            rotate_station(self.scene, math.pi, center=self.station_center)
        self.robot, self.skills = self._make_robot(task.robot)
        kind = getattr(self.robot, "mount_key", None) or getattr(self.robot.spec, "kind", "arm")
        mounts = self.scene.mounts.get(kind) or self.scene.mounts.get("arm")
        self.robot.build(self.scene, mounts[0])
        self.m = self.scene.spec.compile()
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.robot.bind(self.m, self.d)
        self._index()
        self.t = 0.0
        self.n_calls = 0
        self.fail_count = {}
        self.metrics_acc = dict(collisions=0, integrity_violations=0, drops=0, max_robot_contact_n=0.0,
                                sim_instabilities=0, damaged=0)
        self._coll_pairs = set()
        self._drops_before, self._coll_before = 0, 0
        self.view = None
        self.sensors = SensorSuite(self, rng, bulk=self.bulk_sensors, fine=self.fine_sensors)
        # settle (sensors off, robot holds its start pose), then reset the clock
        self._settling = True
        self.run(self.settle_s)
        self._settling = False
        self.t = 0.0
        self.robot.t = 0.0
        if hasattr(self.robot, "tare"):
            self.robot.tare = np.zeros(6)
            self.robot.tare = self.robot.wrench()
        self.robot.effort = EffortMeter()
        self.initial_state = self.true_state()
        self.view = ActorView()
        self.sensors.initial(0.0)
        self.sensors.tick(0.0)
        self.run(0.15)          # let first tracker packets arrive (latency)
        ev = self.sensors.flush()
        self.events_all = list(ev)
        return ev

    def idle(self, T):
        """harness helper: let the world evolve for T s with the robot holding still (no skill call);
        returns the emitted events."""
        self.run(T)
        ev = self.sensors.flush()
        self.events_all.extend(ev)
        return ev

    def goal(self) -> dict:
        return dict(family=self.family, description=self.success_doc, regions=dict(self.scene.regions),
                    **self.goal_extra())

    def execute(self, call: SkillCall) -> StepOut:
        self.n_calls += 1
        before = self.progress()
        self.last_held_before = self.held
        self.last_t0 = self.t
        res: SkillResult = self.skills.run(self, call)
        self.last_call, self.last_result = call, res
        if not res.ok and call.target is not None:
            self.fail_count[call.target] = self.fail_count.get(call.target, 0) + 1
        ev = self.sensors.flush()
        self.events_all.extend(ev)
        succ = self.solved()
        prog = self.progress()
        fatal = self.fatal() or self.metrics_acc["sim_instabilities"] > 0
        done = succ or fatal or self.n_calls >= self.budget
        st = res.status
        flags = dict(slip=st in ("slipped", "handle_slipped", "contact_lost"), jam=st == "jammed",
                     stall=st in ("unreachable", "guard_tripped", "timeout") or res.effort.time_at_limit_s > 0.2,
                     drop=self.metrics_acc["drops"] > getattr(self, "_drops_before", 0),
                     collision=self.metrics_acc["collisions"] > getattr(self, "_coll_before", 0), success=succ)
        self._drops_before, self._coll_before = self.metrics_acc["drops"], self.metrics_acc["collisions"]
        metrics = dict(progress=prog, progress_delta=prog - before, n_calls=self.n_calls, t=self.t, **flags,
                       **{k: v for k, v in self.metrics_acc.items()}, effort_cost=res.effort.cost(),
                       fatal=fatal, **self.metrics_extra())
        return StepOut(events=ev, result=res, done=done, success=succ, metrics=metrics, labels=self.labels())

    def candidate_skills(self, belief, n: int, rng) -> list:
        """Feasible SkillCalls from actor-legal info only (ActorView over emitted events + belief POSE2
        when present + CAD regions)."""
        self._belief = belief
        cands = self.candidates(n, rng)
        return [replace(c, candidate_id=c.candidate_id or f"c{i}") for i, c in enumerate(cands[:n])]

    def render(self, camera: str = "overview", width=640, height=480) -> np.ndarray:
        if self.renderer is None or self.renderer.width != width:
            self.renderer = mujoco.Renderer(self.m, height, width)
        cam = mujoco.MjvCamera()
        c = self.scene.camera
        look = np.asarray(c.get("view_lookat", c["lookat"]), float)
        vp = np.asarray(c.get("view_pos", c["pos"]), float)
        dv = vp - look
        cam.lookat[:] = look
        cam.distance = float(np.linalg.norm(dv))
        cam.azimuth = math.degrees(math.atan2(dv[1], dv[0])) + 180.0
        cam.elevation = -math.degrees(math.asin(dv[2] / cam.distance))
        self.renderer.update_scene(self.d, cam)
        return self.renderer.render()

    def capture_depth(self, width=320, height=240):
        """Optional depth image from the tracker camera (not part of the actor loop by default)."""
        r = mujoco.Renderer(self.m, height, width)
        r.enable_depth_rendering()
        cam = mujoco.MjvCamera()
        c = self.scene.camera
        dv = np.asarray(c["pos"]) - np.asarray(c["lookat"])
        cam.lookat[:] = c["lookat"]
        cam.distance = float(np.linalg.norm(dv))
        cam.azimuth = math.degrees(math.atan2(dv[1], dv[0])) + 180.0
        cam.elevation = -math.degrees(math.asin(dv[2] / cam.distance))
        r.update_scene(self.d, cam)
        img = r.render()
        r.close()
        return img

    # ================================================================== family hooks
    def build(self, scene: Scene, rng, params): raise NotImplementedError
    def success(self) -> bool: raise NotImplementedError

    def solved(self) -> bool:
        """episode success = the family predicate AND no fragile part damaged (crush limit exceeded)."""
        return bool(self.success()) and not getattr(self, "damaged", set())
    def progress(self) -> float: return float(self.success())
    def fatal(self) -> bool: return False
    def candidates(self, n, rng) -> list: raise NotImplementedError
    def scripted(self) -> SkillCall: raise NotImplementedError
    def goal_extra(self) -> dict: return {}
    def metrics_extra(self) -> dict: return {}
    def labels_extra(self) -> dict: return {}

    # ================================================================== robot
    def _make_robot(self, name):
        if self.robot_factory is not None:
            return self.robot_factory(self, name)
        if name in ("probe", "probe_parallel", "probe_suction", "probe_blade"):
            tool = self.tool if name == "probe" else name.split("_")[1]
            r = ProbeRobot(tool=tool, **self.probe_kw)
            return r, ProbeSkills(r)
        from .robots import make_robot_adapter      # real embodiments (agent ROBOTS)
        return make_robot_adapter(self, name)

    # ================================================================== sim loop
    def _index(self):
        m = self.m
        self.body_id, self.qadr, self.dadr = {}, {}, {}
        obj_q, obj_d, obj_b = [], [], []
        for eid, info in self.scene.entities.items():
            try:
                bid = m.body(info.body).id
            except KeyError:
                bid = -1
            self.body_id[eid] = bid
            if info.etype == EntityType.OBJECT and bid >= 0 and m.body_jntnum[bid] > 0 \
                    and m.jnt_type[m.body_jntadr[bid]] == mujoco.mjtJoint.mjJNT_FREE:
                j = m.body_jntadr[bid]
                self.qadr[eid] = m.jnt_qposadr[j]
                self.dadr[eid] = m.jnt_dofadr[j]
                obj_q.extend(range(m.jnt_qposadr[j], m.jnt_qposadr[j] + 7))
                obj_d.extend(range(m.jnt_dofadr[j], m.jnt_dofadr[j] + 6))
                obj_b.append(bid)
        self._obj_q = np.array(obj_q, int)
        self._obj_d = np.array(obj_d, int)
        self._obj_b = np.array(obj_b, int)
        self._obj_bset = set(obj_b)
        self._last_q = None
        self._n_badqacc = 0
        self._prev_full_q = None
        self._prev_v = None
        # equality constraints touching objects would be integrity violations
        self._bad_eq = []
        for i in range(m.neq):
            if m.eq_type[i] in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
                if m.eq_obj1id[i] in self._obj_bset or m.eq_obj2id[i] in self._obj_bset:
                    self._bad_eq.append(i)
        self.floor_gid = m.geom("floor").id
        self.robot_geoms = getattr(self.robot, "robot_geoms", set())
        self._robot_geom_arr = np.zeros(m.ngeom, bool)
        self._robot_geom_arr[list(self.robot_geoms)] = True
        self._conveyors = []
        for name, a in self.scene.actuated.items():
            if a["kind"] == "conveyor":
                self._conveyors.append((m.actuator(a["actuator"]).id, a["speed"]))
        self._fragile_body = {self.body_id[eid]: (eid, float(info.hidden["crush_n"]))
                              for eid, info in self.scene.entities.items()
                              if np.isfinite(info.hidden.get("crush_n", np.inf)) and self.body_id.get(eid, -1) >= 0}
        self.damaged = set()
        self._sq_ema = {}
        self._dent_body = {self.body_id[eid]: (eid, float(info.hidden["dent_n"]))
                           for eid, info in self.scene.entities.items()
                           if np.isfinite(info.hidden.get("dent_n", np.inf)) and self.body_id.get(eid, -1) >= 0}
        self._dent_ema = {}
        self._flex_ema = {}
        # squeeze is measured on the gripper PADS only (palm bumps / pushes do not crush a part)
        pads = set(getattr(self.robot, "pad_ids", set()))
        if not pads and hasattr(self.robot, "pad_geom_names"):
            pads = {m.geom(n).id for n in self.robot.pad_geom_names}
        self._pad_arr = np.zeros(m.ngeom, bool)
        self._pad_arr[list(pads or self.robot_geoms)] = True
        self._flex = {eid: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, info.body)
                      for eid, info in self.scene.entities.items() if info.meta.get("flex")}
        self._flex_crush = {fid: (eid, float(self.scene.entities[eid].hidden["crush_n"]))
                            for eid, fid in self._flex.items()
                            if np.isfinite(self.scene.entities[eid].hidden.get("crush_n", np.inf))}
        self._dropped = set()
        self._step_i = 0

    def _integrity_check(self):
        """Before a step: no applied external / generalized forces on objects, no weld/connect on objects,
        object qpos unchanged since the end of the previous step (no pose writes)."""
        d = self.d
        v = 0
        if self._last_q is not None and len(self._obj_q):
            if not np.array_equal(d.qpos[self._obj_q], self._last_q):
                v += 1
            if self._prev_v is not None and not np.array_equal(d.qvel[self._obj_d], self._prev_v):
                v += 1
        if len(self._obj_b) and np.any(d.xfrc_applied[self._obj_b] != 0):
            v += 1
        if len(self._obj_d) and np.any(d.qfrc_applied[self._obj_d] != 0):
            v += 1
        for i in self._bad_eq:
            if d.eq_active[i]:
                v += 1
        if v:
            self.metrics_acc["integrity_violations"] += v

    def _integrity_post(self):
        """After a step (also usable when another module owns the stepping loop): object qpos must be exactly
        integratePos(qpos_old, qvel_new, dt), so a pose write / teleport made between steps shows up as a
        mismatch. (A qvel check against dt*qacc is not exact under elliptic-cone contact solves, so velocity
        writes are caught by the pre-step equality check in run() instead.)"""
        m, d = self.m, self.d
        if not len(self._obj_q):
            return
        if self._prev_full_q is not None:
            qf = self._prev_full_q.copy()
            mujoco.mj_integratePos(m, qf, d.qvel, m.opt.timestep)
            if np.max(np.abs(qf[self._obj_q] - d.qpos[self._obj_q])) > 1e-7:
                self.metrics_acc["integrity_violations"] += 1
        self._prev_full_q = d.qpos.copy()
        self._prev_v = d.qvel[self._obj_d].copy()

    def pre_physics(self):
        """env bookkeeping BEFORE each mj_step (conveyor drive, integrity)."""
        d = self.d
        for aid, sp in self._conveyors:
            d.ctrl[aid] = sp
        self._integrity_check()

    def post_physics(self, stop=None):
        """env bookkeeping AFTER each mj_step: time, integrity, contact monitor, sensors. Returns stop()."""
        d = self.d
        dt = self.m.opt.timestep
        nw = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        if nw != self._n_badqacc:      # MuJoCo auto-reset the data after a NaN: the episode is invalid
            self._n_badqacc = nw
            self.metrics_acc["sim_instabilities"] += 1
            self._prev_full_q = None
        self._integrity_post()
        if len(self._obj_q):
            self._last_q = d.qpos[self._obj_q].copy()
        self.t += dt
        self._step_i += 1
        if self._step_i % 5 == 0:
            self._contact_monitor()
        if not self._settling:
            self.sensors.tick(self.t)
        return stop is not None and self._step_i % 5 == 0 and stop()

    def _contact_monitor(self):
        m, d = self.m, self.d
        if d.ncon == 0:
            return
        g = d.contact.geom
        rob = (self._robot_geom_arr[g[:, 0]] & (g[:, 0] >= 0)) | (self._robot_geom_arr[g[:, 1]] & (g[:, 1] >= 0))
        f6 = np.zeros(6)
        squeeze = {}
        for i in np.nonzero(rob)[0]:
            mujoco.mj_contactForce(m, d, int(i), f6)
            fn = abs(f6[0])
            if self._fragile_body and (self._pad_arr[g[i, 0]] or self._pad_arr[g[i, 1]]):
                ob = int(m.geom_bodyid[g[i, 1] if self._robot_geom_arr[g[i, 0]] else g[i, 0]])
                if ob in self._fragile_body:
                    squeeze[ob] = squeeze.get(ob, 0.0) + fn
            if fn > self.metrics_acc["max_robot_contact_n"]:
                self.metrics_acc["max_robot_contact_n"] = float(fn)
            key = (int(g[i, 0]), int(g[i, 1]))
            if fn > self.collision_n:
                if key not in self._coll_pairs:
                    self._coll_pairs.add(key)
                    self.metrics_acc["collisions"] += 1
            else:
                self._coll_pairs.discard(key)
        # fragile parts: total robot contact normal force / 2 (= per-pad squeeze for a two-pad pinch) above the
        # part's hidden crush limit damages it (counted once; success requires no damaged part)
        # sustained squeeze (EMA over ~30 ms of 10 ms samples): closing impacts do not count, holding force does
        ema = self._sq_ema
        for ob in list(ema):
            if ob not in squeeze:
                ema[ob] *= 0.7
        for ob, f in squeeze.items():
            ema[ob] = 0.7 * ema.get(ob, 0.0) + 0.3 * f
        for ob, f in list(ema.items()):
            eid, lim = self._fragile_body[ob]
            if f / 2.0 > lim and eid not in self.damaged:
                self.damaged.add(eid)
                self.metrics_acc["damaged"] += 1
        # r5: dent limit - contact force between a part and STATIC fixtures (socket rim, bench) above the part's
        # hidden dent_n damages it (axial jamming / ramming a peg into a rim); EMA like the squeeze monitor
        if self._dent_body:
            gb = m.geom_bodyid
            fixed = ((gb[g[:, 0]] == 0) & (g[:, 0] >= 0) & (g[:, 1] >= 0)) | ((gb[g[:, 1]] == 0) & (g[:, 1] >= 0)
                                                                            & (g[:, 0] >= 0))
            push = {}
            for i in np.nonzero(fixed)[0]:
                ob = int(gb[g[i, 1]] if gb[g[i, 0]] == 0 else gb[g[i, 0]])
                if ob in self._dent_body:
                    mujoco.mj_contactForce(m, d, int(i), f6)
                    push[ob] = push.get(ob, 0.0) + abs(f6[0])
            for ob in list(self._dent_ema):
                if ob not in push:
                    self._dent_ema[ob] *= 0.7
            for ob, f in push.items():
                self._dent_ema[ob] = 0.7 * self._dent_ema.get(ob, 0.0) + 0.3 * f
                eid, lim = self._dent_body[ob]
                if self._dent_ema[ob] > lim and eid not in self.damaged:
                    self.damaged.add(eid)
                    self.metrics_acc["damaged"] += 1
        # r5: soft flex parts (pouch): pad squeeze on flex vertices above the hidden crush limit damages it
        if self._flex_crush and d.ncon:
            fx = d.contact.flex
            sq = {}
            for i in np.nonzero(rob)[0]:
                for k in (0, 1):
                    if fx[i, k] >= 0 and int(fx[i, k]) in self._flex_crush and self._pad_arr[g[i, 1 - k]]:
                        mujoco.mj_contactForce(m, d, int(i), f6)
                        sq[int(fx[i, k])] = sq.get(int(fx[i, k]), 0.0) + abs(f6[0])
            for fid in list(self._flex_ema):
                if fid not in sq:
                    self._flex_ema[fid] *= 0.7
            for fid, f in sq.items():
                self._flex_ema[fid] = 0.7 * self._flex_ema.get(fid, 0.0) + 0.3 * f
                eid, lim = self._flex_crush[fid]
                if self._flex_ema[fid] / 2.0 > lim and eid not in self.damaged:
                    self.damaged.add(eid)
                    self.metrics_acc["damaged"] += 1
        fl = np.nonzero((g[:, 0] == self.floor_gid) | (g[:, 1] == self.floor_gid))[0]
        for i in fl:
            other = int(g[i, 1] if g[i, 0] == self.floor_gid else g[i, 0])
            b = int(m.geom_bodyid[other])
            if b in self._obj_bset and b not in self._dropped:
                self._dropped.add(b)
                self.metrics_acc["drops"] += 1

    def run(self, T, stop=None):
        m, d, r = self.m, self.d, self.robot
        dt = m.opt.timestep
        n = int(round(T / dt))
        for _ in range(n):
            r.step(dt)
            self.pre_physics()
            mujoco.mj_step(m, d)
            if self.post_physics(stop):
                break

    # ================================================================== interventions (EVALUATOR-ONLY)
    _SNAP_FIELDS = ("t", "held", "_held_off", "n_calls", "fail_count", "metrics_acc", "damaged", "_dropped",
                    "_coll_pairs", "_last_q", "_prev_full_q", "_prev_v", "_step_i", "_n_badqacc", "_drops_before",
                    "_coll_before", "last_call", "last_result", "last_held_before", "_believe", "_sorted",
                    "_seated_belief", "flagged", "_ow_attempts", "_looks", "_sq_ema", "_dent_ema", "_flex_ema", "last_t0",
                    "_placed_mem", "_mass_ev", "_pick_xy")

    def snapshot(self):
        """Full evaluator-side state: MuJoCo integration state + env bookkeeping + sensor/actor caches + robot
        controller state + rng states. Use with resimulate(); never give it to an actor."""
        import copy
        m, d = self.m, self.d
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        st = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, st, spec)
        memo = {id(m): m, id(d): d}
        if self.renderer is not None:
            memo[id(self.renderer)] = self.renderer
        env_f = {k: copy.deepcopy(getattr(self, k), memo) for k in self._SNAP_FIELDS if hasattr(self, k)}
        sens = {k: copy.deepcopy(v, memo) for k, v in self.sensors.__dict__.items() if k not in ("env", "out")}
        robot = copy.deepcopy({k: v for k, v in self.robot.__dict__.items()}, memo)
        return dict(state=st, env=env_f, sensors=sens, robot=robot, view=copy.deepcopy(self.view),
                    n_events=len(self.events_all), rng=copy.deepcopy(self.rng.bit_generator.state),
                    hidden={e: dict(i.hidden) for e, i in self.scene.entities.items()})

    def restore(self, snap):
        import copy
        m, d = self.m, self.d
        mujoco.mj_setState(m, d, snap["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(m, d)
        memo = {id(m): m, id(d): d}
        for k, v in snap["env"].items():
            setattr(self, k, copy.deepcopy(v, memo))
        for k, v in snap["sensors"].items():
            setattr(self.sensors, k, copy.deepcopy(v, memo))
        self.sensors.out = []
        self.robot.__dict__.update(copy.deepcopy(snap["robot"], memo))
        self.view = copy.deepcopy(snap["view"])
        del self.events_all[snap["n_events"]:]
        self.rng.bit_generator.state = copy.deepcopy(snap["rng"])
        for e, h in snap["hidden"].items():
            self.scene.entities[e].hidden.clear()
            self.scene.entities[e].hidden.update(h)

    def _param_baseline(self):
        if not hasattr(self, "_param0"):
            m = self.m
            self._param0 = dict(body_mass=m.body_mass.copy(), body_inertia=m.body_inertia.copy(),
                                geom_friction=m.geom_friction.copy(), pair_friction=m.pair_friction.copy(),
                                dof_frictionloss=m.dof_frictionloss.copy(), dof_damping=m.dof_damping.copy())
        return self._param0

    def set_hidden(self, overrides):
        """EVALUATOR-ONLY: change hidden physics in the compiled model (no pose writes): {eid: {mass, mu, mu_grip,
        frictionloss, damping, crush_n}}. Resets every hidden param to the episode's original value first."""
        m, p0 = self.m, self._param_baseline()
        for k, v in p0.items():
            getattr(m, k)[:] = v
        for eid, ov in (overrides or {}).items():
            info = self.scene.entities[eid]
            bid = self.body_id.get(eid, -1)
            gids = [m.geom(g).id for g in info.geoms]
            if "mass" in ov and bid >= 0:
                k = float(ov["mass"]) / max(float(p0["body_mass"][bid]), 1e-9)
                m.body_mass[bid] = p0["body_mass"][bid] * k
                m.body_inertia[bid] = p0["body_inertia"][bid] * k
            if "mu" in ov:
                for g in gids:
                    m.geom_friction[g, 0] = float(ov["mu"])
            if "mu_grip" in ov:
                for i in range(m.npair):
                    if m.pair_geom1[i] in gids or m.pair_geom2[i] in gids:
                        if not (m.pair_geom1[i] in gids and m.pair_geom2[i] in gids):
                            m.pair_friction[i, :2] = float(ov["mu_grip"])
            if ("frictionloss" in ov or "damping" in ov) and "joint" in info.meta:
                dof = m.jnt_dofadr[m.joint(info.meta["joint"]).id]
                if "frictionloss" in ov:
                    m.dof_frictionloss[dof] = float(ov["frictionloss"])
                if "damping" in ov:
                    m.dof_damping[dof] = float(ov["damping"])
            if "crush_n" in ov and bid >= 0:
                self._fragile_body[bid] = (eid, float(ov["crush_n"]))
            info.hidden.update({k: float(v) for k, v in ov.items()})

    def resimulate(self, snap, overrides, call):
        """Re-run `call` from a snapshot with modified hidden properties (station intervention tests). Returns the
        StepOut of that branch; the env is left in the branch's post-call state (restore(snap) to branch again)."""
        self.restore(snap)
        self.set_hidden(overrides)
        return self.execute(call)

    # ================================================================== truth (evaluator / sensors only)
    def true_pose(self, eid):
        if eid in self._flex:
            fid = self._flex[eid]
            a, n = self.m.flex_vertadr[fid], self.m.flex_vertnum[fid]
            p = self.d.flexvert_xpos[a:a + n].mean(0)
            return p, np.array([1.0, 0, 0, 0])
        bid = self.body_id[eid]
        return self.d.xpos[bid].copy(), self.d.xquat[bid].copy()

    def true_state(self):
        out = {}
        for eid, info in self.scene.entities.items():
            p, q = self.true_pose(eid)
            out[eid] = dict(pos=p.tolist(), quat=q.tolist())
            if info.etype == EntityType.LINK:
                out[eid]["joint"] = float(self.d.qpos[self.m.jnt_qposadr[self.m.joint(info.meta["joint"]).id]])
        return out

    def contact_graph(self):
        m, d = self.m, self.d
        b2e = {bid: eid for eid, bid in self.body_id.items() if bid >= 0}
        edges = set()
        for i in range(d.ncon):
            g1, g2 = d.contact.geom[i]
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            a = b2e.get(b1, "robot" if g1 in self.robot_geoms else ("world" if b1 == 0 else m.body(b1).name))
            b = b2e.get(b2, "robot" if g2 in self.robot_geoms else ("world" if b2 == 0 else m.body(b2).name))
            if a != b:
                edges.add(tuple(sorted((a, b))))
        return sorted(edges)

    def labels(self):
        return dict(hidden={eid: dict(info.hidden) for eid, info in self.scene.entities.items()},
                    true_state=self.true_state(), contacts=self.contact_graph(), sensor_cfg=dict(self.sensors.cfg),
                    held=self.held, damaged=sorted(getattr(self, "damaged", ())), **self.labels_extra())

    # ================================================================== actor-legal helpers
    def est_pose(self, eid):
        """(pos3, quat, yaw) from the latest AVAILABLE tracker POSE3; belief POSE2 overrides xy/yaw if set."""
        e = self.view.latest(eid, XProp.POSE3, self.t)
        if e is None:
            return None
        pos = np.array(e.value[:3])
        quat = np.array(e.value[3:])
        yaw = quat_yaw(quat)
        b = getattr(self, "_belief", None)
        if b is not None:
            try:
                from ..schema import Prop
                mu, _, _ = b.resolve_query(eid, Prop.POSE2)
                pos[:2] = mu[:2]
                yaw = math.atan2(mu[3], mu[2])
            except Exception:
                pass
        return pos, quat, yaw

    def est_age(self, eid):
        e = self.view.latest(eid, XProp.POSE3, self.t)
        return 0.0 if e is None else float(self.t - e.measured_at)

    def est_velocity(self, eid, window=1.0):
        """least-squares xy velocity from the recent tracker packets (actor-legal); None if < 2 packets."""
        evs = [e for e in self.view.buf.get((eid, int(XProp.POSE3)), ())
               if e.available_at <= self.t + 1e-9 and e.measured_at >= self.t - window]
        if len(evs) < 2:
            return None
        t = np.array([e.measured_at for e in evs])
        P = np.array([e.value[:2] for e in evs])
        if t.max() - t.min() < 0.05:
            return None
        A = np.stack([t - t.mean(), np.ones_like(t)], 1)
        sol, *_ = np.linalg.lstsq(A, P, rcond=None)
        return sol[0]

    def held_mass_evidence(self, window=0.6):
        """ACTOR-LEGAL mass estimate of the held object from the robot's own tared wrist F/T in the last `window` s
        and its own suction command: hold Fz ~ m g - F_suction_cmd (probe convention, world z). None if no data."""
        from ..schema import Prop
        evs = [e for e in self.view.buf.get(("robot/hand", int(Prop.WRENCH)), ())
               if e.available_at <= self.t + 1e-9 and e.measured_at >= self.t - window]
        if not evs:
            return None
        fz = float(np.median([e.value[2] for e in evs]))
        f_cmd = float(getattr(self.robot, "suction_cmd", 0.0)) * float(getattr(self.robot, "suction_max", 0.0))
        return max(0.0, (fz + f_cmd) / 9.81)

    def est_pose_avg(self, eid, window=0.3):
        """mean of the tracker POSE3 packets measured in the last `window` s (actor-legal); None if none."""
        evs = [e for e in self.view.buf.get((eid, int(XProp.POSE3)), ())
               if e.available_at <= self.t + 1e-9 and e.measured_at >= self.t - window - 0.1]
        if not evs:
            return None
        P = np.array([e.value[:3] for e in evs]).mean(0)
        ys = np.array([quat_yaw(np.array(e.value[3:])) for e in evs])
        yaw = math.atan2(np.sin(ys).mean(), np.cos(ys).mean())
        return P, np.array(evs[-1].value[3:]), yaw

    def est_size(self, eid):
        from ..schema import Prop
        e = self.view.latest(eid, Prop.SIZE, self.t + 1e9)
        return np.array(e.value) if e is not None else np.array([0.02, 0.02, 0.02])

    def est_joint(self, eid):
        e = self.view.latest(eid, XProp.JOINT, self.t)
        return None if e is None else float(e.value[0])

    def region(self, name):
        return self.scene.regions[name]

    def support_z(self, eid):
        est = self.est_pose(eid)
        h = self.est_size(eid)
        return float(est[0][2] - h[2]) if est is not None else self.surface_z

    def held_offset_z(self):
        return self._held_off[0]

    def held_yaw_offset(self):
        return self._held_off[1]

    def handle_path(self, eid, travel, n=10):
        info = self.scene.entities[eid]
        q0 = self.est_joint(eid)
        if q0 is None:
            return None
        qs = np.linspace(q0, q0 + travel, n + 1)
        pts, yaws = [], []
        for q in qs:
            p, yaw = self._handle_at(info, q)
            pts.append(p)
            yaws.append(yaw)
        return pts, yaws

    def _handle_at(self, info, q):
        """CAD kinematics of a door/drawer handle grasp point (TCP target) at joint value q."""
        if info.kind == "drawer":
            ay = info.meta["axis_yaw"]
            c, s = math.cos(ay), math.sin(ay)
            hx, hy, hz = info.meta["handle_local"]
            base = np.asarray(info.meta["pos"], float)
            px = base[0] + c * hx - s * (hy - q)
            py = base[1] + s * hx + c * (hy - q)
            # pads pinch the tab across drawer-x: hand y along drawer x
            return np.array([px, py, base[2] + hz - 0.03]), wrap(ay + math.pi / 2)
        if info.kind == "door":
            hpos = np.asarray(info.meta["hinge_pos"], float)
            a = info.meta["closed_yaw"] + q
            c, s = math.cos(a), math.sin(a)
            hx, hy, hz = info.meta["handle_local"]
            px = hpos[0] + c * hx - s * hy
            py = hpos[1] + s * hx + c * hy
            return np.array([px, py, hpos[2] + hz - 0.03]), wrap(a + math.pi / 2)
        if info.kind == "knob":
            # TCP 3 mm above the knob base so the pads cover the cylinder; the hand yaw follows the knob angle
            p = np.asarray(info.meta["pos"], float) + np.array([0.0, 0.0, 0.003])
            return p, wrap(q)
        raise ValueError(info.kind)

    def objects(self, kind=None):
        return [eid for eid, i in self.scene.entities.items() if i.etype == EntityType.OBJECT
                and (kind is None or i.kind == kind)]

    def in_region_true(self, eid, reg, margin=0.0, zcheck=True):
        p, _ = self.true_pose(eid)
        cx, cy, cz = reg["center"]
        hx, hy, hz = reg["half"]
        yaw = reg.get("yaw", 0.0)
        dx, dy = p[0] - cx, p[1] - cy
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        ok = abs(lx) <= hx + margin and abs(ly) <= hy + margin
        if zcheck:
            ok = ok and abs(p[2] - cz) <= hz + margin
        return ok

    def in_region_est(self, eid, reg, margin=0.0):
        est = self.est_pose(eid)
        if est is None:
            return None
        p = est[0]
        cx, cy, cz = reg["center"]
        hx, hy, hz = reg["half"]
        yaw = reg.get("yaw", 0.0)
        dx, dy = p[0] - cx, p[1] - cy
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return abs(lx) <= hx + margin and abs(ly) <= hy + margin and abs(p[2] - cz) <= hz + margin
