"""Procedural, seed-deterministic simulators for the physical-memory sprint (agent DATA).

Worlds
------
w2  box on a table plane pushed by a primitive pusher (MuJoCo). Hidden per-episode
    sliding friction (box<->table) and mass. Pusher geometries: sphere r=1.5cm,
    sphere r=3cm, flat paddle (held out). K=4 push primitives per root episode.
    Optional misspecified laws: velocity-dependent friction, actuation drift.
w3  (minimal) w2 + a second box placed in the path of the first push (contact rewiring).
w1  puck with known impulses (+ optional damped spring to an anchor), numpy integrator.

MuJoCo contact model used for w2/w3 (documented in docs/DATA.md):
  * option: timestep=2.5ms, cone="elliptic", impratio=1, integrator="implicitfast".
  * condim=3 everywhere (tangential Coulomb friction, no torsional/rolling).
  * friction mixing via geom `priority`: box (priority 1) friction = mu decides the
    box<->table contact (table priority 0); pusher (priority 2, friction 0.3) decides
    the pusher<->box contact. Box<->box (w3) uses mu of the boxes (same priority -> max).
  * solref/solimp: MuJoCo defaults (0.02, 1) / (0.9, 0.95, 0.001).
  * box<->table is a box-plane contact (up to 4 corner points) so frictional torque
    about z emerges from the distributed contacts.
  * pusher: a body with slide-x, slide-y and hinge-z joints at fixed height z=2cm, driven by
    velocity actuators (kv=300, force limit 30N) with an outer position loop
    (ctrl = v_cmd + 20*(x_ref-x)); i.e. a strongly position-controlled pusher whose tracking
    lags slightly under heavy load (so mass is weakly observable through proprioception).
    It collides with boxes only (not the table). Between pushes it is re-positioned
    (teleported) 0.25s before the next push start.
  * the push primitive is executed relative to the TRUE object pose at re-position time
    (well-calibrated robot); actors/labels use their own noisy estimate frame.
  * velocity-dependent friction (law 1): geom_friction[box] is overwritten every step to
    mu0*(1+k*|v_xy|), k~U[1,3] s/m (contact friction is re-read from the model each step).
  * actuation drift (law 2): realized pusher reference velocity = commanded + constant
    per-episode bias b (|b|~U[0.02,0.05] m/s, random world direction) while pushing.
    COMMAND observations report the commanded value only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

G = 9.81
DT = 0.0025  # sim step (s)
REC_EVERY = 2  # record ground truth every 2 steps -> 200 Hz
REC_DT = DT * REC_EVERY
K_PUSHES = 4
T0 = 0.3  # first push start (s)
PUSH_PERIOD = 2.3  # spacing between push starts (>= max horizon 2.0 + margin)
REPOSITION_LEAD = 0.25  # pusher re-positioned this long before each push
PUSHER_Z = 0.02
PUSH_GAP = 0.003  # pusher starts this far outside the boundary point
PUSHER_FRICTION = 0.3

PUSHERS = {
    # name: (geom type, size tuple, effective half-thickness along push direction)
    "sphere15": ("sphere", (0.015,), 0.015),
    "sphere30": ("sphere", (0.030,), 0.030),
    "paddle": ("box", (0.004, 0.035, 0.015), 0.004),
}
PUSHER_SIZE3 = {"sphere15": (0.015, 0.015, 0.015), "sphere30": (0.03, 0.03, 0.03), "paddle": (0.004, 0.035, 0.015)}

LAW_COULOMB, LAW_VELFRIC, LAW_DRIFT, LAW_SPRING = 0, 1, 2, 3

# held-out (mass, friction) cells: grid of 4 log-mass bins x 4 friction bins over the train ranges.
MASS_RANGE = (0.2, 2.0)
FRIC_RANGE = (0.3, 0.7)
MASS_EDGES = np.exp(np.linspace(np.log(MASS_RANGE[0]), np.log(MASS_RANGE[1]), 5))
FRIC_EDGES = np.linspace(FRIC_RANGE[0], FRIC_RANGE[1], 5)
HELDOUT_CELLS = ((3, 0), (0, 3))  # (mass bin, friction bin): heavy+slippery, light+grippy


def cell_of(mass: float, fric: float) -> tuple:
    mb = int(np.clip(np.searchsorted(MASS_EDGES, mass, side="right") - 1, 0, 3))
    fb = int(np.clip(np.searchsorted(FRIC_EDGES, fric, side="right") - 1, 0, 3))
    return mb, fb


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# episode parameters
# ---------------------------------------------------------------------------


@dataclass
class PushAction:
    theta: float
    phi: float
    speed: float
    duration: float

    def params(self):
        return (self.theta, self.phi, self.speed, self.duration)


@dataclass
class EpisodeParams:
    world: str = "w2"
    friction: float = 0.5
    mass: float = 1.0
    half: tuple = (0.03, 0.03, 0.03)  # box half extents (m)
    pos: tuple = (0.0, 0.0)
    yaw: float = 0.0
    pusher: str = "sphere15"
    law: int = LAW_COULOMB
    vel_k: float = 0.0  # law 1
    drift: tuple = (0.0, 0.0)  # law 2 (world m/s)
    actions: list = field(default_factory=list)  # list[PushAction] (w2/w3) or impulses (w1)
    pusher_park: tuple = (0.3, 0.0)
    retract: bool = False  # v2: pusher retracts (0.2 m/s for 0.1 s) at push end -> clean release
    pusher_fmax: float = 30.0  # pusher actuator force limit (N); v2 uses 100
    # w3 second box
    second: dict | None = None
    # chain strata (e10): extra boxes in chain order (first = pushed box); target = last box = `half/pos/...`
    others: list | None = None
    push_body: int = 0  # object index (0 target, 1.. extras) the push primitive is applied to
    aim: bool = False  # chain: theta/phi chosen at re-position time to push through the chain toward the target
    chain_axis: float = 0.0
    # moving agent (e16/h7): dict(mode, r, start, ...); mode is LATENT (label store only)
    agent: dict | None = None
    # w1
    stiffness: float = 0.0
    damping: float = 0.0
    anchor: tuple = (0.0, 0.0)

    def to_json(self):
        d = asdict(self)
        d["actions"] = [list(a.params()) if isinstance(a, PushAction) else list(a) for a in self.actions]
        return d


def sample_push_action(rng: np.random.Generator, cfg: dict | None = None) -> PushAction:
    """v1 defaults: duration U[0.1,0.6], speed U[0.05,0.4]. cfg may set dur=(lo,hi), speed=(kind,lo,hi)."""
    cfg = cfg or {}
    dlo, dhi = cfg.get("dur", (0.1, 0.6))
    dur = round(float(rng.uniform(dlo, dhi)) / REC_DT) * REC_DT
    theta = float(rng.uniform(-np.pi, np.pi))
    phi = float(rng.uniform(-0.4, 0.4))
    speed = _sample_value(rng, cfg.get("speed", ("uniform", 0.05, 0.4)))
    return PushAction(theta=theta, phi=phi, speed=float(speed), duration=float(dur))


def _sample_value(rng, spec):
    kind = spec[0]
    if kind == "uniform":
        return float(rng.uniform(spec[1], spec[2]))
    if kind == "loguniform":
        return float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
    if kind == "const":
        return float(spec[1])
    if kind == "choice":
        return float(spec[1][rng.integers(len(spec[1]))])
    raise ValueError(spec)


def sample_episode(rng: np.random.Generator, cfg: dict) -> EpisodeParams:
    """cfg keys: world, fric, mass, cells ('exclude'|'only'|'any'), pushers, law, second_box."""
    world = cfg.get("world", "w2")
    for _ in range(1000):
        fric = _sample_value(rng, cfg.get("fric", ("uniform", *FRIC_RANGE)))
        mass = _sample_value(rng, cfg.get("mass", ("loguniform", *MASS_RANGE)))
        c = cell_of(mass, fric)
        mode = cfg.get("cells", "exclude")
        if mode == "exclude" and c in HELDOUT_CELLS and FRIC_RANGE[0] <= fric <= FRIC_RANGE[1]:
            continue
        if mode == "only" and c not in HELDOUT_CELLS:
            continue
        break
    half = tuple(float(x) for x in rng.uniform(0.02, 0.05, size=3))
    if cfg.get("hz") == "stable":  # v2: hz in [1.5cm, 0.9*min(hx,hy)] so fast free slides do not tip boxes
        hi = 0.9 * min(half[0], half[1])
        half = (half[0], half[1], 0.015 + (half[2] - 0.02) / 0.03 * (hi - 0.015))
    pos = tuple(float(x) for x in rng.uniform(-0.1, 0.1, size=2))
    yaw = float(rng.uniform(-np.pi, np.pi))
    pushers = cfg.get("pushers", ("sphere15", "sphere30"))
    pusher = pushers[int(rng.integers(len(pushers)))]
    law = int(cfg.get("law", LAW_COULOMB))
    ep = EpisodeParams(world=world, friction=fric, mass=mass, half=half, pos=pos, yaw=yaw, pusher=pusher, law=law,
                       retract=bool(cfg.get("retract", False)), pusher_fmax=float(cfg.get("fmax", 30.0)))
    park_ang = rng.uniform(-np.pi, np.pi)
    ep.pusher_park = (pos[0] + 0.3 * math.cos(park_ang), pos[1] + 0.3 * math.sin(park_ang))
    if law == LAW_VELFRIC:
        ep.vel_k = float(rng.uniform(1.0, 3.0))
    if law == LAW_DRIFT:
        a = rng.uniform(-np.pi, np.pi)
        m = rng.uniform(0.02, 0.05)
        ep.drift = (float(m * math.cos(a)), float(m * math.sin(a)))
    if world == "w1":
        # w1 defaults (overridable): low friction so impulses produce informative slides
        ep.friction = _sample_value(rng, cfg.get("w1_fric", ("uniform", 0.03, 0.15)))
        ep.mass = _sample_value(rng, cfg.get("w1_mass", ("uniform", 0.5, 2.0)))
        r = float(rng.uniform(0.02, 0.04))
        ep.half = (r, r, 0.01)
        ep.actions = []
        for _ in range(K_PUSHES):
            jm = rng.uniform(0.03, 0.3)
            ja = rng.uniform(-np.pi, np.pi)
            ep.actions.append((float(jm * math.cos(ja)), float(jm * math.sin(ja)), 0.0, 0.0))
        if rng.random() < cfg.get("spring_prob", 0.5):
            ep.law = LAW_SPRING
            ep.stiffness = float(rng.uniform(2.0, 20.0))
            ep.damping = float(rng.uniform(0.1, 1.0))
            off = rng.uniform(-0.05, 0.05, size=2)
            ep.anchor = (pos[0] + float(off[0]), pos[1] + float(off[1]))
        return ep
    ep.actions = [sample_push_action(rng, cfg) for _ in range(K_PUSHES)]
    if world == "chain":
        _sample_chain(rng, ep, int(cfg["chain_L"]), cfg)
    if cfg.get("agent"):
        _sample_agent(rng, ep, cfg)
    if world == "w3":
        # second box in the path of the first push
        a0 = ep.actions[0]
        ex = max(half[0], half[1])
        h2 = tuple(float(x) for x in rng.uniform(0.02, 0.05, size=3))
        d_local = _push_geometry(half, a0.theta, a0.phi)[1]
        c, s = math.cos(yaw), math.sin(yaw)
        d = (c * d_local[0] - s * d_local[1], s * d_local[0] + c * d_local[1])
        dist = ex * 1.42 + max(h2[0], h2[1]) * 1.42 + float(rng.uniform(0.005, 0.05))
        ep.second = dict(
            half=h2,
            pos=(pos[0] + d[0] * dist, pos[1] + d[1] * dist),
            yaw=float(rng.uniform(-np.pi, np.pi)),
            mass=_sample_value(rng, ("loguniform", *MASS_RANGE)),
            friction=_sample_value(rng, ("uniform", *FRIC_RANGE)),
        )
    return ep


def _stable_half(rng, hx_rng=(0.02, 0.04), hy_rng=(0.025, 0.05)):
    hx, hy = float(rng.uniform(*hx_rng)), float(rng.uniform(*hy_rng))
    hz = float(rng.uniform(0.015, max(0.0151, 0.9 * min(hx, hy))))
    return (hx, hy, hz)


def _sample_chain(rng, ep: EpisodeParams, L: int, cfg: dict):
    """A line of L boxes along the chain axis (= target yaw); the target is the LAST box; pushes are applied to the
    FIRST box, aimed through the chain at the target. Intermediate boxes: mass log-U[0.2,1.0], friction U[0.3,0.7],
    gaps U[0,4] mm, lateral jitter +-4 mm, yaw jitter +-0.05 rad."""
    ep.aim = True
    ep.chain_axis = ep.yaw
    u = np.array([math.cos(ep.yaw), math.sin(ep.yaw)])
    perp = np.array([-u[1], u[0]])
    cur = np.array(ep.pos, float)
    prev_hx = ep.half[0]
    near_to_far = []
    for _ in range(L - 1):
        h = _stable_half(rng)
        off = prev_hx + h[0] + float(rng.uniform(0.0, 0.004))
        cur = cur - u * off
        c = cur + perp * float(rng.uniform(-0.004, 0.004))
        near_to_far.append(dict(half=h, pos=(float(c[0]), float(c[1])), yaw=float(ep.yaw + rng.uniform(-0.05, 0.05)),
                                mass=_sample_value(rng, ("loguniform", 0.2, 1.0)),
                                friction=_sample_value(rng, ("uniform", *FRIC_RANGE))))
        prev_hx = h[0]
    ep.others = near_to_far[::-1]
    ep.push_body = 1 if L > 1 else 0
    far = np.array(ep.others[0]["pos"]) if L > 1 else np.array(ep.pos)
    ep.pusher_park = (float(far[0] - u[0] * 0.25 + perp[0] * 0.1), float(far[1] - u[1] * 0.25 + perp[1] * 0.1))


AGENT_MODES = ("stationary", "crossing", "approach", "oscillating")
AGENT_R, AGENT_HZ = 0.025, 0.02


def _sample_agent(rng, ep: EpisodeParams, cfg: dict):
    """Kinematic disc agent with a LATENT behaviour mode; motion starts at t=T0 (stationary before, so the k=0
    history is uninformative about the mode)."""
    mode = int(rng.integers(4))
    a = float(rng.uniform(-np.pi, np.pi))
    dvec = np.array([math.cos(a), math.sin(a)])
    p = np.array(ep.pos)
    ag = dict(mode=mode, mode_name=AGENT_MODES[mode], r=AGENT_R)
    if mode == 0:
        st = p + dvec * rng.uniform(0.12, 0.2)
    elif mode == 1:
        e = dvec
        perp = np.array([-e[1], e[0]])
        c = p + perp * rng.uniform(-0.05, 0.05)
        ag.update(center=c.tolist(), axis=e.tolist(), amp=0.25, speed=float(rng.uniform(0.04, 0.1)))
        st = c - e * 0.25
    elif mode == 2:
        st = p + dvec * rng.uniform(0.2, 0.3)
        ag.update(speed=float(rng.uniform(0.03, 0.06)))
    else:
        c = p + dvec * rng.uniform(0.12, 0.18)
        b = float(rng.uniform(-np.pi, np.pi))
        ag.update(center=c.tolist(), axis=[math.cos(b), math.sin(b)], amp=float(rng.uniform(0.04, 0.08)),
                  period=float(rng.uniform(1.5, 3.0)))
        st = c
    ag["start"] = [float(st[0]), float(st[1])]
    ep.agent = ag


def _agent_step(ag, t, prev_xy, target_xy, target_half):
    if t < T0 or ag["mode"] == 0:
        return np.array(ag["start"] if t < T0 or ag["mode"] == 0 else prev_xy)
    tt = t - T0
    if ag["mode"] == 1:
        A = ag["amp"]
        ph = (ag["speed"] * tt) % (4 * A)
        s = -A + (ph if ph < 2 * A else 4 * A - ph)
        return np.array(ag["center"]) + np.array(ag["axis"]) * s
    if ag["mode"] == 2:
        d = np.asarray(target_xy) - prev_xy
        n = float(np.hypot(*d))
        stop = ag["r"] + max(target_half[0], target_half[1]) + 0.005
        if n <= stop:
            return prev_xy
        return prev_xy + d / n * min(ag["speed"] * DT, n - stop)
    return np.array(ag["center"]) + np.array(ag["axis"]) * ag["amp"] * math.sin(2 * np.pi * tt / ag["period"])


def _extras(ep):
    if ep.others:
        return list(ep.others)
    return [ep.second] if ep.second is not None else []


def _push_geometry(half, theta, phi):
    """Boundary point (local), outward normal (local), push direction (local)."""
    ux, uy = math.cos(theta), math.sin(theta)
    hx, hy = half[0], half[1]
    sx = hx / abs(ux) if abs(ux) > 1e-9 else np.inf
    sy = hy / abs(uy) if abs(uy) > 1e-9 else np.inf
    if sx < sy:
        p = (sx * ux, sx * uy)
        n = (math.copysign(1.0, ux), 0.0)
    else:
        p = (sy * ux, sy * uy)
        n = (0.0, math.copysign(1.0, uy))
    cp, sp = math.cos(phi), math.sin(phi)
    d = (-(cp * n[0] - sp * n[1]), -(sp * n[0] + cp * n[1]))
    return p, d, n


# ---------------------------------------------------------------------------
# MuJoCo w2/w3
# ---------------------------------------------------------------------------


def _xml(ep: EpisodeParams) -> str:
    hx, hy, hz = ep.half
    gtype, gsize, _ = PUSHERS[ep.pusher]
    gs = " ".join(f"{v:.6f}" for v in gsize)
    cb = 7 if ep.agent is not None else 3  # collision bits of boxes (bit 4 = agent)
    second = ""
    for j, s in enumerate(_extras(ep)):
        q = (math.cos(s["yaw"] / 2), 0, 0, math.sin(s["yaw"] / 2))
        nm = f"box{j + 2}"
        second += f"""
    <body name="{nm}" pos="{s['pos'][0]:.6f} {s['pos'][1]:.6f} {s['half'][2]:.6f}" quat="{q[0]:.8f} 0 0 {q[3]:.8f}">
      <freejoint name="{nm}_free"/>
      <geom name="{nm}" type="box" size="{s['half'][0]:.6f} {s['half'][1]:.6f} {s['half'][2]:.6f}" mass="{s['mass']:.6f}"
            friction="{s['friction']:.6f} 0.005 0.0001" priority="1" condim="3" contype="{cb}" conaffinity="{cb}"/>
    </body>"""
    if ep.agent is not None:
        a = ep.agent
        second += f"""
    <body name="agent" mocap="true" pos="{a['start'][0]:.6f} {a['start'][1]:.6f} {AGENT_HZ + 0.001}">
      <geom name="agent" type="cylinder" size="{a['r']} {AGENT_HZ}" friction="0.3 0.005 0.0001" priority="2"
            condim="3" contype="4" conaffinity="4"/>
    </body>"""
    q = (math.cos(ep.yaw / 2), math.sin(ep.yaw / 2))
    return f"""<mujoco model="w2">
  <option timestep="{DT}" cone="elliptic" impratio="1" integrator="implicitfast" gravity="0 0 {-G}"/>
  <worldbody>
    <geom name="table" type="plane" size="0 0 0.05" friction="0.5 0.005 0.0001" priority="0" condim="3" contype="1" conaffinity="1"/>
    <body name="box" pos="{ep.pos[0]:.6f} {ep.pos[1]:.6f} {hz:.6f}" quat="{q[0]:.8f} 0 0 {q[1]:.8f}">
      <freejoint name="box_free"/>
      <geom name="box" type="box" size="{hx:.6f} {hy:.6f} {hz:.6f}" mass="{ep.mass:.6f}"
            friction="{ep.friction:.6f} 0.005 0.0001" priority="1" condim="3" contype="{cb}" conaffinity="{cb}"/>
    </body>{second}
    <body name="pusher" pos="0 0 {PUSHER_Z}">
      <joint name="px" type="slide" axis="1 0 0"/>
      <joint name="py" type="slide" axis="0 1 0"/>
      <joint name="pyaw" type="hinge" axis="0 0 1" armature="0.1"/>
      <geom name="pusher" type="{gtype}" size="{gs}" mass="1.0" friction="{PUSHER_FRICTION} 0.005 0.0001"
            priority="2" condim="3" contype="2" conaffinity="2"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="vx" joint="px" kv="300" forcelimited="true" forcerange="-{ep.pusher_fmax:g} {ep.pusher_fmax:g}"/>
    <velocity name="vy" joint="py" kv="300" forcelimited="true" forcerange="-{ep.pusher_fmax:g} {ep.pusher_fmax:g}"/>
  </actuator>
</mujoco>"""


def _yaw_from_quat(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _tilt_from_quat(q):
    w, x, y, z = q
    # angle between body z axis and world z
    zz = 1 - 2 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, zz)))


def episode_times():
    starts = np.array([T0 + PUSH_PERIOD * k for k in range(K_PUSHES)])
    t_end = starts[-1] + PUSH_PERIOD
    return starts, t_end


def simulate_w2(ep: EpisodeParams) -> dict:
    """Simulate a w2/w3 root episode. Returns ground-truth trajectory dict at 200 Hz.

    obj_pose (T, n_obj, 3) x,y,yaw (yaw unwrapped); obj_vel (T, n_obj, 3) world vx,vy,wz;
    obj_tilt (T, n_obj); pusher_pose (T,3); pusher_vel (T,3); cmd (T,2) commanded world vel;
    f_push (T,) normal force pusher<->target; f_other (T,) normal force target<->other objects;
    push_start (K,), push_end (K,).
    """
    import mujoco

    m = mujoco.MjModel.from_xml_string(_xml(ep))
    d = mujoco.MjData(m)
    gid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("box", "pusher", "table")}
    extras = _extras(ep)
    objs = ["box"] + [f"box{j + 2}" for j in range(len(extras))]
    halfs = [ep.half] + [e["half"] for e in extras]
    has_agent = ep.agent is not None
    if has_agent:
        agent_mid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "agent")]
        agent_xy = np.array(ep.agent["start"], float)
    gid_objs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in objs]
    bid_objs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in objs]
    jadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_free")] for n in objs]
    jvadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_free")] for n in objs]
    pq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("px", "py", "pyaw")]
    pv = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("px", "py", "pyaw")]
    mu0 = ep.friction
    gid_agent = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "agent") if ep.agent is not None else -2

    starts, t_end = episode_times()
    n_steps = int(round(t_end / DT))
    T = n_steps // REC_EVERY + 1
    n_obj = len(objs)
    rec = dict(
        t=np.zeros(T),
        obj_pose=np.zeros((T, n_obj, 3)),
        obj_vel=np.zeros((T, n_obj, 3)),
        obj_tilt=np.zeros((T, n_obj)),
        pusher_pose=np.zeros((T, 3)),
        pusher_vel=np.zeros((T, 3)),
        cmd=np.zeros((T, 2)),
        f_push=np.zeros(T),
        f_other=np.zeros(T),
    )
    if has_agent:
        rec["agent_pose"] = np.zeros((T, 3))
        rec["f_agent"] = np.zeros(T)
    realized = [list(a.params()) for a in ep.actions]
    start_steps = [int(round(s / DT)) for s in starts]
    repos_steps = [int(round((s - REPOSITION_LEAD) / DT)) for s in starts]
    end_steps = [st + int(round(a.duration / DT)) for st, a in zip(start_steps, ep.actions)]
    retract_end = [e + int(round(0.1 / DT)) for e in end_steps] if ep.retract else []
    # pusher initial (parked)
    d.qpos[pq[0]], d.qpos[pq[1]], d.qpos[pq[2]] = ep.pusher_park[0], ep.pusher_park[1], 0.0
    mujoco.mj_forward(m, d)
    x_ref = np.array(ep.pusher_park, dtype=float)
    v_cmd = np.zeros(2)
    v_real = np.zeros(2)
    cur = -1
    push_dir_world = []
    fbuf = np.zeros(6)
    prev_yaw = [None] * n_obj
    yaw_off = [0.0] * n_obj

    for i in range(n_steps + 1):
        # --- control schedule
        if i in repos_steps:
            k = repos_steps.index(i)
            a = ep.actions[k]
            pb = ep.push_body
            bx, by = d.qpos[jadr[pb]], d.qpos[jadr[pb] + 1]
            byaw = _yaw_from_quat(d.qpos[jadr[pb] + 3 : jadr[pb] + 7])
            th, ph = a.theta, a.phi
            if ep.aim:  # push through the chain toward the target (realized theta/phi recorded)
                dv = np.array([math.cos(ep.chain_axis), math.sin(ep.chain_axis)])
                if pb != 0:
                    dd = np.array([d.qpos[jadr[0]] - bx, d.qpos[jadr[0] + 1] - by])
                    if np.hypot(*dd) > 1e-6:
                        dv = dd / np.hypot(*dd)
                th = float(wrap(math.atan2(-dv[1], -dv[0]) - byaw))
                _, _, n0 = _push_geometry(halfs[pb], th, 0.0)
                cl, sl_ = math.cos(-byaw), math.sin(-byaw)
                dloc = (cl * dv[0] - sl_ * dv[1], sl_ * dv[0] + cl * dv[1])
                ph = float(np.clip(wrap(math.atan2(dloc[1], dloc[0]) - math.atan2(-n0[1], -n0[0])), -0.6, 0.6))
                realized[k][0], realized[k][1] = th, ph
            p, dl, n = _push_geometry(halfs[pb], th, ph)
            reff = PUSHERS[ep.pusher][2]
            sl = (p[0] + n[0] * (reff + PUSH_GAP), p[1] + n[1] * (reff + PUSH_GAP))
            c, s = math.cos(byaw), math.sin(byaw)
            start = np.array([bx + c * sl[0] - s * sl[1], by + s * sl[0] + c * sl[1]])
            dw = np.array([c * dl[0] - s * dl[1], s * dl[0] + c * dl[1]])
            push_dir_world.append(dw)
            d.qpos[pq[0]], d.qpos[pq[1]] = start
            d.qpos[pq[2]] = math.atan2(dw[1], dw[0]) if ep.pusher == "paddle" else 0.0
            d.qvel[pv[0]] = d.qvel[pv[1]] = d.qvel[pv[2]] = 0.0
            x_ref = start.copy()
            cur = k
        if i in start_steps:
            k = start_steps.index(i)
            v_cmd = push_dir_world[k] * ep.actions[k].speed
            v_real = v_cmd + (np.array(ep.drift) if ep.law == LAW_DRIFT else 0.0)
        if i in end_steps:
            if ep.retract:
                v_cmd = -push_dir_world[end_steps.index(i)] * 0.2
                v_real = v_cmd.copy()
            else:
                v_cmd = np.zeros(2)
                v_real = np.zeros(2)
        if i in retract_end:
            v_cmd = np.zeros(2)
            v_real = np.zeros(2)
        # --- record (state at time i*DT, before stepping)
        if i % REC_EVERY == 0:
            r = i // REC_EVERY
            rec["t"][r] = i * DT
            for j in range(n_obj):
                qa, va = jadr[j], jvadr[j]
                q = d.qpos[qa + 3 : qa + 7]
                yaw = _yaw_from_quat(q)
                if prev_yaw[j] is not None:
                    yaw_off[j] += round((prev_yaw[j] - yaw) / (2 * np.pi)) * 2 * np.pi
                prev_yaw[j] = yaw
                rec["obj_pose"][r, j] = (d.qpos[qa], d.qpos[qa + 1], yaw + yaw_off[j])
                w_world = d.xmat[bid_objs[j]].reshape(3, 3) @ d.qvel[va + 3 : va + 6]
                rec["obj_vel"][r, j] = (d.qvel[va], d.qvel[va + 1], w_world[2])
                rec["obj_tilt"][r, j] = _tilt_from_quat(q)
            rec["pusher_pose"][r] = (d.qpos[pq[0]], d.qpos[pq[1]], d.qpos[pq[2]])
            rec["pusher_vel"][r] = (d.qvel[pv[0]], d.qvel[pv[1]], d.qvel[pv[2]])
            rec["cmd"][r] = v_cmd
            fp = fo = fa = 0.0
            for ci in range(d.ncon):
                con = d.contact[ci]
                g1, g2 = con.geom1, con.geom2
                if gid_objs[0] not in (g1, g2):
                    continue
                other = g2 if g1 == gid_objs[0] else g1
                if other == gid["table"]:
                    continue
                mujoco.mj_contactForce(m, d, ci, fbuf)
                if other == gid["pusher"]:
                    fp += abs(fbuf[0])
                else:
                    fo += abs(fbuf[0])
                    if has_agent and other == gid_agent:
                        fa += abs(fbuf[0])
            rec["f_push"][r] = fp
            rec["f_other"][r] = fo
            if has_agent:
                rec["agent_pose"][r] = (agent_xy[0], agent_xy[1], 0.0)
                rec["f_agent"][r] = fa
        if i == n_steps:
            break
        # --- actuation
        x_ref = x_ref + v_real * DT
        xy = np.array([d.qpos[pq[0]], d.qpos[pq[1]]])
        d.ctrl[0:2] = v_real + 20.0 * (x_ref - xy)
        d.qvel[pv[2]] = 0.0  # hinge held kinematically
        if ep.law == LAW_VELFRIC:
            va = jvadr[0]
            sp = math.hypot(d.qvel[va], d.qvel[va + 1])
            m.geom_friction[gid_objs[0], 0] = mu0 * (1.0 + ep.vel_k * sp)
        if has_agent:
            agent_xy = _agent_step(ep.agent, (i + 1) * DT, agent_xy, d.qpos[jadr[0]:jadr[0] + 2], ep.half)
            d.mocap_pos[agent_mid, :2] = agent_xy
        mujoco.mj_step(m, d)
    rec["actions_realized"] = np.array(realized)
    rec["push_start"] = starts
    rec["push_end"] = np.array([s + a.duration for s, a in zip(starts, ep.actions)])
    return rec


# ---------------------------------------------------------------------------
# w1 numpy puck
# ---------------------------------------------------------------------------


def simulate_w1(ep: EpisodeParams) -> dict:
    """Point-mass puck on a plane with Coulomb friction (hidden mu), known impulses in the
    object frame at push starts, optional damped spring to an anchor (hidden k, c).
    Semi-implicit Euler at DT; returns the same trajectory dict layout as simulate_w2
    (pusher fields zero; no pusher exists)."""
    starts, t_end = episode_times()
    n_steps = int(round(t_end / DT))
    T = n_steps // REC_EVERY + 1
    start_steps = [int(round(s / DT)) for s in starts]
    x = np.array(ep.pos, dtype=float)
    v = np.zeros(2)
    m, mu = ep.mass, ep.friction
    anchor = np.array(ep.anchor)
    c, s = math.cos(ep.yaw), math.sin(ep.yaw)
    rec = dict(
        t=np.zeros(T), obj_pose=np.zeros((T, 1, 3)), obj_vel=np.zeros((T, 1, 3)), obj_tilt=np.zeros((T, 1)),
        pusher_pose=np.zeros((T, 3)), pusher_vel=np.zeros((T, 3)), cmd=np.zeros((T, 2)),
        f_push=np.zeros(T), f_other=np.zeros(T),
    )
    for i in range(n_steps + 1):
        if i in start_steps:
            jx, jy = ep.actions[start_steps.index(i)][:2]
            v = v + np.array([c * jx - s * jy, s * jx + c * jy]) / m
        if i % REC_EVERY == 0:
            r = i // REC_EVERY
            rec["t"][r] = i * DT
            rec["obj_pose"][r, 0] = (x[0], x[1], ep.yaw)
            rec["obj_vel"][r, 0] = (v[0], v[1], 0.0)
        if i == n_steps:
            break
        f = np.zeros(2)
        if ep.law == LAW_SPRING:
            f = -ep.stiffness * (x - anchor) - ep.damping * v
        sp = float(np.hypot(*v))
        fric = mu * m * G
        if sp > 1e-6:
            dv = (f - fric * v / sp) / m * DT
            vn = v + dv
            # Coulomb stop: friction cannot reverse velocity
            if np.dot(vn, v) <= 0 and np.hypot(*f) <= fric:
                vn = np.zeros(2)
            v = vn
        else:
            if np.hypot(*f) > fric:
                v = v + (f - fric * f / np.hypot(*f)) / m * DT
            else:
                v = np.zeros(2)
        x = x + v * DT
    rec["push_start"] = starts
    rec["push_end"] = starts + 0.0
    return rec


def simulate(ep: EpisodeParams) -> dict:
    if ep.world == "w1":
        return simulate_w1(ep)
    return simulate_w2(ep)
