"""il-v0: demo recording and low-level execution for the end-to-end imitation baseline (round 6, agreed with ENVS).

Control tick = 0.1 s of sim time. Observation (actor-legal only):
  proprio (P_DIM): tcp pos, controller-reference minus tcp, tcp rotation (6D), gripper closure, pad contacts (2),
                   tared wrist wrench (6), controller mode one-hot (4), gripper command, grip-force setpoint, time
  objects (K_OBJ x O_DIM): tracker estimates from the sensor stream (pos rel. tcp, size, vision / tactile stiffness)
  regions (K_REG x R_DIM): actor-legal region estimates (pos rel. tcp, half extents)
  context: embodiment vector (spec.embodiment_vector) + family one-hot (meta)
Action (A_DIM) for tick t -> t+0.1: controller-reference delta pos (/0.05 m), rotation delta (rotvec /0.2 rad),
  gripper command, grip-force setpoint (/40 N), mode one-hot (4), guard threshold (/20 N), mode axis (3),
  log impedance stiffness. (envs-r7 adds [mode_id, f_target(3)] via env.control_state; see A_DIM_R7.)

The recorder never touches the env: it wraps env.post_physics and reads the controller reference / gripper command
and the NEW sensor events (a cursor over env.sensors.out, so the env's own flush per skill call is unaffected). The
executor commands the same controller API the skills use (ctrl.command / set_gripper / set_mode, emb.set_grip_force)
and steps env.run(0.1). Object estimates in both come from the same sensor stream -> no train/test observation shift.
"""
from __future__ import annotations

import math

import numpy as np

from .. import spec as S
from ..env_adapter import EventIndex, STIFF_PROP, S_PROP_SIZE

try:
    from physmem.envs.props import XProp
except Exception:       # pragma: no cover
    XProp = None

DT = 0.1
MODES = ("track", "impedance", "guarded", "force")
K_OBJ, O_DIM = 8, 10
K_REG, R_DIM = 4, 7
P_DIM = 28
A_DIM_V0 = 17
A_DIM = 20          # il-v1 (envs-r7): + force-mode target f_target (3, N world, /20)
POS_S, ROT_S, F_S = 0.05, 0.2, 40.0
ABS_S = 0.1          # absolute-position action mode: target - tcp(issue time), in units of 10 cm
SCHEMA = "il-v1"
# proprio dims that are CONTROLLER COMMANDS (reference - tcp, controller mode, gripper command, grip-force setpoint),
# i.e. the previous action leaking into the observation. ACT-style policies see measured state only: with these
# inputs the policy learned to copy the scripted controller's own lead ("move when the reference is ahead") and
# hesitated in closed loop (dry run 13.0). Zeroed when meta["drop_cmd"].
CMD_DIMS = np.r_[3:6, 21:27]


def mask_obs_p(p, drop_cmd):
    if not drop_cmd:
        return p
    p = np.array(p, copy=True)
    p[..., CMD_DIMS] = 0.0
    return p


def env_version(require=True):
    """the ENVS version string; with EXPECT_ENV set (all round-6 runs), assert it matches (the shared tree on gb10
    carried uncommitted envs-r7-dev code once - every run must state which envs it ran on)."""
    import os
    from physmem import envs as _E
    v = getattr(_E, "ENV_VERSION", "unknown")
    want = os.environ.get("EXPECT_ENV")
    if require and not want:
        raise RuntimeError(f"set EXPECT_ENV (this tree has ENV_VERSION={v})")
    if want and want != v:
        raise RuntimeError(f"ENV_VERSION mismatch: tree has {v}, run expects {want}")
    return v


# ------------------------------------------------------------------ small rotation helpers (wxyz)
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]]) / max(float(np.dot(q, q)), 1e-12)


def q2rotvec(q):
    q = np.asarray(q, float)
    q = q / max(np.linalg.norm(q), 1e-12)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-9:
        return 2 * q[1:]
    return q[1:] / s * 2 * math.atan2(s, q[0])


def rotvec2q(r):
    a = float(np.linalg.norm(r))
    if a < 1e-9:
        return np.array([1.0, *(np.asarray(r) / 2)])
    ax = np.asarray(r) / a
    return np.array([math.cos(a / 2), *(ax * math.sin(a / 2))])


def q2rot6(q):
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return np.concatenate([R[:, 0], R[:, 1]])


# ------------------------------------------------------------------ live sensor stream (non-destructive)
class StreamPeek:
    """EventIndex fed from env.sensors.out without flushing it (the env flushes per skill call)."""

    def __init__(self, env):
        self.env, self.idx = env, EventIndex()
        self._buf, self._n = None, 0

    def add(self, events):
        self.idx.add(events)

    def poll(self):
        out = self.env.sensors.out
        if out is not self._buf:
            self._buf, self._n = out, 0
        if len(out) > self._n:
            self.idx.add(out[self._n:])
            self._n = len(out)


def _robot(env):
    r = env.robot
    if not hasattr(r, "ctrl"):
        raise RuntimeError("il-v0 needs a controller-based arm (env.robot.ctrl); the probe gantry has none")
    return r, r.eff


def controller_state(env):
    """(ref_pos, ref_quat, grip_cmd, grip_force, mode, kw) of the active effector."""
    r, eff = _robot(env)
    st = r.ctrl.st[eff]
    ref = st.get("ref")
    if ref is None:
        p, q = r.emb.tcp_pose(env.d, eff)
        ref = (np.asarray(p, float), None, np.asarray(q, float))
    gc = getattr(r, "grip_cmd", None)
    gf = float(gc[1]) if isinstance(gc, tuple) and len(gc) > 1 else float(np.mean(r.gforce0[eff]))
    return (np.asarray(ref[0], float).copy(), np.asarray(ref[2], float).copy(), float(st.get("grip", 0.0)), gf,
            st.get("mode", "track"), dict(st.get("kw", {})))


def observe(env, peek, t0=0.0):
    """-> dict(p (P_DIM), obj (K_OBJ,O_DIM), reg (K_REG,R_DIM))  (actor-legal)."""
    r, eff = _robot(env)
    peek.poll()
    now = float(env.t)
    p, q = r.emb.tcp_pose(env.d, eff)
    p, q = np.asarray(p, float), np.asarray(q, float)
    rp, rq, g, gf, mode, _ = controller_state(env)
    c, n = r.emb.gripper_state(env.m, env.d, eff)
    pc = r.pad_contacts(eff)
    W = np.asarray(r.wrench(eff), float)
    P = np.concatenate([p, rp - p, q2rot6(q), [float(c)], [pc[0], pc[1]], np.clip(W / 20.0, -5, 5),
                        np.eye(4)[MODES.index(mode) if mode in MODES else 0], [g], [gf / F_S],
                        [min((now - t0) / 30.0, 3.0)]]).astype(np.float32)
    idx = peek.idx
    O = np.zeros((K_OBJ, O_DIM), np.float32)
    ids = sorted(e for e in idx.entities(XProp.POSE3) if not str(e).startswith("robot"))
    k = 0
    for e in ids:
        v = idx.latest(e, XProp.POSE3, now)
        if v is None or k >= K_OBJ:
            continue
        sz = idx.latest(e, S_PROP_SIZE, now)
        sz = np.asarray(sz[:3]) if sz is not None else np.full(3, 0.02)
        vk = idx.latest_src(e, STIFF_PROP, "vision_material", now)
        tk = idx.latest_src(e, STIFF_PROP, "tactile_squeeze", now)
        O[k] = np.concatenate([(np.asarray(v[:3]) - p) / 0.1, sz / 0.05,
                               [(math.log(max(vk[0], 1.0)) - S.STIFF_REF) / 2 if vk is not None else 0.0,
                                (math.log(max(tk[0], 1.0)) - S.STIFF_REF) / 2 if tk is not None else 0.0,
                                float(tk is not None), 1.0]])
        k += 1
    Rg = np.zeros((K_REG, R_DIM), np.float32)
    k = 0
    for e in sorted(idx.entities(XProp.REGION)):
        v = idx.latest(e, XProp.REGION, now)
        if v is None or k >= K_REG:
            continue
        Rg[k] = np.concatenate([(np.asarray(v[:3]) - p) / 0.1, np.asarray(v[3:6]) / 0.05, [1.0]])
        k += 1
    return dict(p=P, obj=O, reg=Rg)


def action_between(s0, s1):
    """action that takes controller state s0 (tick t) to s1 (tick t+1); states from controller_state()."""
    p0, q0, _, _, _, _ = s0
    p1, q1, g1, gf1, mode1, kw1 = s1
    dq = _qmul(q1, _qinv(q0))
    ax = np.asarray(kw1.get("axis", (0, 0, 0)), float)[:3] if kw1.get("axis") is not None else np.zeros(3)
    stiff = kw1.get("stiffness")
    ft = ax * float(kw1.get("force", 0.0)) if mode1 == "force" else np.zeros(3)
    return np.concatenate([(p1 - p0) / POS_S, q2rotvec(dq) / ROT_S, [g1], [gf1 / F_S],
                           np.eye(4)[MODES.index(mode1) if mode1 in MODES else 0],
                           [float(kw1.get("threshold", 0.0)) / 20.0], ax,
                           [math.log(float(stiff)) / 10.0 if stiff else 0.0], ft / 20.0]).astype(np.float32)


# ------------------------------------------------------------------ recorder (scripted demos)
class Recorder:
    """attach before running the scripted episode; samples obs + controller state every DT of sim time."""

    def __init__(self, env, peek, img=0):
        self.env, self.peek, self.img = env, peek, int(img)
        self.obs, self.states, self.times, self.tcpq = [], [], [], []
        self.next_t = float(env.t)
        self.t0 = float(env.t)
        orig = env.post_physics

        def post(stop=None):
            out = orig(stop)
            peek.poll()
            if env.t + 1e-9 >= self.next_t:
                self.next_t += DT
                try:
                    o = observe(env, peek, self.t0)
                    if self.img and hasattr(env, "render_wrist"):
                        o["img"] = env.render_wrist(self.img, self.img)
                    self.obs.append(o)
                    self.states.append(controller_state(env))
                    self.times.append(float(env.t))
                    r_, e_ = _robot(env)
                    self.tcpq.append(np.asarray(r_.emb.tcp_pose(env.d, e_)[1], float))
                except Exception:
                    pass
            return out
        env.post_physics = post

    def arrays(self):
        T = len(self.states)
        if T < 2:
            return None
        acts = np.stack([action_between(self.states[i], self.states[i + 1]) for i in range(T - 1)])
        out = dict(p=np.stack([o["p"] for o in self.obs[:-1]]), obj=np.stack([o["obj"] for o in self.obs[:-1]]),
                   reg=np.stack([o["reg"] for o in self.obs[:-1]]), act=acts,
                   t=np.asarray(self.times[:-1], np.float32),
                   # absolute-target view (ACT-style): controller reference (world) and tcp orientation per tick
                   ref_p=np.stack([s_[0] for s_ in self.states]).astype(np.float32),
                   ref_q=np.stack([s_[1] for s_ in self.states]).astype(np.float32),
                   tcp_q=np.stack(self.tcpq).astype(np.float32))
        if self.obs and "img" in self.obs[0]:
            out["img"] = np.stack([o["img"] for o in self.obs[:-1]]).astype(np.uint8)
        return out


# ------------------------------------------------------------------ executor
class Executor:
    """applies one il-v0 action for DT of sim time through the controller API the skills use."""

    def __init__(self, env, binary_grip=True):
        self.env = env
        self._mode = None
        self.binary_grip = binary_grip

    def apply(self, a, p_target=None):
        """a: il action; p_target: optional WORLD position target (ACT absolute-position mode) replacing ref + dp."""
        env = self.env
        r, eff = _robot(env)
        from physmem.embodiments.trajectory import BlendedTrajectory
        rp, rq, _, _, mode, _ = controller_state(env)
        a = np.asarray(a, float)
        dp = np.clip(a[0:3], -3, 3) * POS_S
        dr = np.clip(a[3:6], -3, 3) * ROT_S
        g = float(np.clip(a[6], 0, 1))
        if self.binary_grip:        # demos command the gripper open / closed (0 / 1): threshold the regressed value
            g = 1.0 if g >= 0.5 else 0.0
        gf = float(np.clip(a[7], 0.02, 2.5) * F_S)
        m = MODES[int(np.argmax(a[8:12]))]
        ft = a[17:20] * 20.0 if len(a) >= A_DIM else np.zeros(3)
        if m in ("force", "impedance") and hasattr(env, "set_control"):      # envs-r7 hook (il-v1)
            stiff = float(math.exp(a[16] * 10.0)) if a[16] > 0.1 else None
            if m != self._mode or m == "force":
                env.set_control(eff, m, f_target=tuple(ft) if m == "force" else (0.0, 0.0, 0.0),
                                stiffness=stiff if (m == "impedance" or stiff) else None)
                self._mode = m
        elif m != self._mode:
            kw = {}
            if m in ("guarded", "force"):
                ax = a[13:16]
                kw = dict(threshold=max(float(a[12]) * 20.0, 1.0),
                          axis=tuple(ax / np.linalg.norm(ax)) if np.linalg.norm(ax) > 0.3 else (0, 0, -1),
                          bias=tuple(np.asarray(r.wrench(eff), float)[:3]))
            elif m == "impedance":
                kw = dict(stiffness=float(math.exp(a[16] * 10.0)) if a[16] > 0.1 else 500.0, rot_stiffness=20.0)
            r.ctrl.set_mode(m, eff, **kw)
            self._mode = m
        q1 = _qmul(rotvec2q(dr), rq)
        pt = np.asarray(p_target, float) if p_target is not None else rp + dp
        tr = BlendedTrajectory(effector=eff, waypoints=[(pt, q1 / np.linalg.norm(q1))], durations=[DT])
        tr.no_retime = True
        r.ctrl.command(tr)          # each plan lasts exactly DT -> starts from the previous target (= reference)
        r.ctrl.set_gripper(eff, g)
        try:
            r.emb.set_grip_force(eff, gf)
            r.grip_k[eff] = 1.0
            r.grip_cmd = ("force", gf)
        except Exception:
            pass
        env.run(DT)
