"""Fast synthetic sanity world for the v2 model (owner: MODEL-V2). NOT the real benchmark (that is ENVS' MuJoCo
stations); this exists to prove the model can learn latent physics before real data exists.

Five families where hidden properties strongly change skill outcomes, all emitted in the exact v2 data contract
(spec.CHANNELS segments, EffortMeter, outcome vectors, evaluator-only labels):

  slide  (push)  : box with hidden mass m, support friction mu. push at speed v for distance s; after release the box
                   free-slides v^2/(2 mu g). F/T reads m*a + mu*m*g. goal: stop within 1.5 cm of a target line.
  lift   (pick)  : hidden mass, grip friction, part stiffness. grip force Fg, lift accel a, transport speed vt, height.
                   drop if m*(g+a_max) > 2*mu_g*Fg. grip force is painful (effort) -> want the minimal safe force.
  insert (insert): hidden hole offset (x,y), clearance, stiffness. aim correction, speed, spiral search radius.
                   chamfer contact reveals the misalignment direction through lateral F/T. jam at high speed.
  drawer (open)  : hidden damping + breakaway force. pull force F for Tp seconds -> opening (F-0.3Fb)/b*Tp.
                   goal: opening within 2 cm; slamming into the end stop is a collision.
  sweep  (sweep) : bulk pile of 20..200 particles (one entity each), hidden bulk friction + pile mass. blade depth,
                   speed, stroke. goal: >= 80% of particles inside the goal band; spill beyond it is a 'drop'.

Every quantity the actor sees goes through noisy synthetic sensors (F/T noise + per-episode bias, tracking lag,
20 Hz pose tracker with noise and dropout). Robots differ in force/speed/payload limits and sensor quality; one is
held out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from physmem.v2 import api
from . import spec as S

G = S.G
T = S.SEG_T
FAMILIES = ["slide", "lift", "insert", "drawer", "sweep"]
FAMILY_SKILL = {"slide": "push", "lift": "pick", "insert": "insert", "drawer": "open", "sweep": "sweep"}
FAMILY_CATEGORY = {"slide": "fine", "lift": "logistics", "insert": "fine", "drawer": "articulated", "sweep": "bulk"}
FAMILY_LATENTS = {
    "slide": ["mass", "friction"],
    "lift": ["mass", "grip_friction", "stiffness"],
    "insert": ["offset_x", "offset_y", "clearance", "stiffness"],
    "drawer": ["damping", "breakaway"],
    "sweep": ["bulk_friction", "mass"],
}
# skill parameter bounds (SI) per family; keys sorted == SkillCall.vector order
PARAM_BOUNDS = {
    "slide": {"dir": (-1.0, 1.0), "dist": (0.01, 0.30), "speed": (0.05, 0.80)},
    "lift": {"accel": (0.5, 10.0), "grip_force": (3.0, 100.0), "height": (0.01, 0.15), "speed": (0.1, 1.0)},
    "insert": {"aim_x": (-0.004, 0.004), "aim_y": (-0.004, 0.004), "search_r": (0.0, 0.003), "speed": (0.004, 0.05)},
    "drawer": {"duration": (0.1, 2.0), "force": (-60.0, 60.0)},
    "sweep": {"depth": (0.1, 1.0), "speed": (0.1, 1.0), "stroke": (0.03, 0.40)},
}
# event supervision mask (spec.OUT_EVENTS: success, slip, drop, jam, collision, stall). Events that cannot occur in a
# family are SUPERVISED AS 0 (a drawer cannot jam), so the risk head is meaningful everywhere.
EVENT_MASK = {f: [1, 1, 1, 1, 1, 1] for f in ("slide", "lift", "insert", "drawer", "sweep")}
MAX_STEPS = {"slide": 5, "lift": 4, "insert": 6, "drawer": 5, "sweep": 6}


# ----------------------------------------------------------------- synthetic robots
@dataclass
class SynthRobot:
    spec: api.EmbodimentSpec
    ft_noise: float       # N
    ft_bias: float        # N, per-episode bias scale
    lag: int              # tracking lag in samples
    obs_noise: float      # m
    obs_drop: float       # dropout prob of a pose sample
    arm_mass: float       # effective moving mass (effort)


def _spec(name, kind, grip, reach, payload, vmax, fmax, dof, floating, eff):
    return api.EmbodimentSpec(name=name, kind=kind, effectors=eff, gripper=grip, reach_m=reach, payload_kg=payload,
                              max_ee_speed=vmax, max_ee_force=fmax, dof=dof, floating_base=floating, source="synth")


ROBOTS = {
    "synth_arm_a": SynthRobot(_spec("synth_arm_a", "arm", "parallel", 0.85, 3.0, 1.0, 60.0, 7, False, ("hand",)),
                              0.3, 0.5, 1, 0.002, 0.1, 2.0),
    "synth_arm_b": SynthRobot(_spec("synth_arm_b", "arm", "parallel", 1.3, 5.0, 0.7, 100.0, 6, False, ("hand",)),
                              0.6, 1.0, 2, 0.002, 0.15, 4.0),
    "synth_bimanual": SynthRobot(_spec("synth_bimanual", "bimanual", "parallel", 0.7, 8.0, 0.8, 150.0, 14, False,
                                       ("left", "right")), 0.8, 1.0, 2, 0.003, 0.1, 5.0),
    "synth_humanoid": SynthRobot(_spec("synth_humanoid", "humanoid", "dexterous", 0.6, 2.5, 0.6, 50.0, 29, True,
                                       ("left", "right")), 1.0, 1.5, 3, 0.004, 0.2, 3.0),
    # held out
    "synth_mobile": SynthRobot(_spec("synth_mobile", "mobile_manipulator", "parallel", 0.9, 4.0, 0.9, 80.0, 9, False,
                                     ("hand",)), 0.5, 1.0, 2, 0.003, 0.15, 3.5),
}
TRAIN_ROBOTS = ["synth_arm_a", "synth_arm_b", "synth_bimanual", "synth_humanoid"]
HELDOUT_ROBOTS = ["synth_mobile"]


# ----------------------------------------------------------------- latent sampling
def sample_latents(family: str, rng, stress: bool = False) -> dict:
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    if family == "slide":
        return dict(mass=lu(0.2, 5.0), friction=float(rng.uniform(0.1, 0.9) if not stress else rng.choice([0.07, 1.0])))
    if family == "lift":
        return dict(mass=lu(0.1, 4.0), grip_friction=float(rng.uniform(0.25, 1.0)), stiffness=lu(2e3, 4e4))
    if family == "insert":
        return dict(offset_x=float(rng.uniform(-0.003, 0.003)), offset_y=float(rng.uniform(-0.003, 0.003)),
                    clearance=lu(1e-4, 1.5e-3), stiffness=lu(1e3, 2e4))
    if family == "drawer":
        return dict(damping=lu(3.0, 80.0), breakaway=lu(2.0, 25.0))
    if family == "sweep":
        return dict(bulk_friction=float(rng.uniform(0.2, 0.9)), mass=lu(0.2, 3.0))
    raise KeyError(family)


def sample_params(family: str, rng) -> dict:
    return {k: float(rng.uniform(*b)) for k, b in PARAM_BOUNDS[family].items()}


# ----------------------------------------------------------------- shared sensor model
class _Seg:
    """builds a (T, N_CH) segment from true signals with robot-specific sensor corruption."""

    def __init__(self, dur, robot: SynthRobot, rng, bias):
        self.t = np.linspace(0.0, dur, T)
        self.dt = dur / (T - 1)
        self.x = np.zeros((T, S.N_CH), np.float32)
        self.robot, self.rng, self.bias = robot, rng, bias
        self.x[:, S.CH["t"]] = self.t

    def set(self, cmd_pos, ee_pos=None, force=None, torque=None, grip_cmd=None, grip_pos=None, obj=None, contact=None):
        r, rng = self.robot, self.rng
        cmd_vel = np.gradient(cmd_pos, self.dt, axis=0)
        if ee_pos is None:
            ee_pos = np.concatenate([np.repeat(cmd_pos[:1], r.lag, 0), cmd_pos[:T - r.lag]], 0) if r.lag else cmd_pos
        ee_pos = ee_pos + rng.normal(0, 2e-4, ee_pos.shape)
        x = self.x
        x[:, 0:3], x[:, 3:6] = cmd_pos, cmd_vel
        x[:, 6:9], x[:, 9:12] = ee_pos, np.gradient(ee_pos, self.dt, axis=0)
        f = np.zeros((T, 3)) if force is None else force
        x[:, 12:15] = f + self.bias[:3] + rng.normal(0, r.ft_noise, (T, 3))
        tq = np.zeros((T, 3)) if torque is None else torque
        x[:, 15:18] = tq + self.bias[3:] * 0.05 + rng.normal(0, r.ft_noise * 0.02, (T, 3))
        if grip_cmd is not None:
            x[:, 18] = grip_cmd
        if grip_pos is not None:
            x[:, 19] = grip_pos + rng.normal(0, 5e-5, T)
        if obj is not None:
            # 20 Hz pose tracker with noise + dropout, sample-and-hold
            tick = np.floor(self.t * 20.0 + 1e-9)
            fresh = np.r_[True, tick[1:] != tick[:-1]] & (rng.random(T) > r.obs_drop)
            fresh[0] = True
            held = np.zeros(3)
            for i in range(T):
                if fresh[i]:
                    held = obj[i] + rng.normal(0, r.obs_noise, 3)
                x[i, 20:23] = held
                x[i, 23] = float(fresh[i])
        if contact is not None:
            x[:, 24] = contact
        return x


def _ramp(t, t0, t1, a, b):
    """min-jerk blend from a to b between t0 and t1."""
    s = np.clip((t - t0) / max(t1 - t0, 1e-6), 0, 1)
    s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    return a + (b - a) * s


def _effort(force_mag, accel_mag, dt, robot: SynthRobot, extra_sq=0.0, vmax_used=0.0):
    fm = robot.spec.max_ee_force
    tau = (force_mag + robot.arm_mass * accel_mag) / fm
    e = api.EffortMeter()
    e.torque_sq = float(np.sum(tau ** 2) * dt + extra_sq)
    e.torque_rate_sq = float(np.sum(np.gradient(tau, dt) ** 2) * dt)
    jerk = np.gradient(accel_mag, dt)
    e.joint_jerk_sq = float(np.sum(jerk ** 2) * dt)
    e.energy_j = float(np.sum(np.abs(force_mag) * vmax_used) * dt)
    e.time_at_limit_s = float(np.sum(tau >= 0.98) * dt)
    e.peak_ee_speed = float(vmax_used)
    return e


@dataclass
class StepSim:
    seg: np.ndarray
    effort: api.EffortMeter
    duration: float
    cont: np.ndarray
    events: np.ndarray
    status: str
    success: bool
    metrics: dict


# ----------------------------------------------------------------- family dynamics
def _ev(**kw):
    e = np.zeros(S.N_EVENTS, np.float32)
    for k, v in kw.items():
        e[S.OUT_EVENTS.index(k)] = float(v)
    return e


def step_slide(state, lat, p, robot: SynthRobot, rng, bias):
    m, mu = lat["mass"], lat["friction"]
    sgn = 1.0 if p["dir"] >= 0 else -1.0
    v = min(p["speed"], robot.spec.max_ee_speed)
    s = p["dist"]
    fmax = robot.spec.max_ee_force
    a_nom = 4.0
    stall = mu * m * G > fmax
    a = max(min(a_nom, fmax / m - mu * G), 0.3) if not stall else 0.0
    t_acc = v / max(a, 1e-6) if not stall else 0.2
    s_acc = 0.5 * a * t_acc ** 2
    if not stall and s_acc > s:  # never reaches v
        t_acc = math.sqrt(2 * s / a)
        v = a * t_acc
        s_acc = s
    t_cruise = 0.0 if stall else (s - s_acc) / v
    t_app = 0.12
    t_rel = t_app + t_acc + t_cruise
    slide = 0.0 if stall else v ** 2 / (2 * mu * G) * float(np.exp(rng.normal(0, 0.06)))
    t_slide = 0.0 if stall else v / (mu * G)
    dur = t_rel + max(t_slide, 0.15) + 0.05
    sg = _Seg(dur, robot, rng, bias)
    t = sg.t
    # 1-D kinematics along x (sgn), pusher starts 1 cm behind the contact face
    ee = np.zeros(T)
    obj = np.zeros(T)
    F = np.zeros(T)
    ee[t < t_app] = _ramp(t[t < t_app], 0, t_app, -0.01, 0.0)
    if stall:
        tt = t >= t_app
        ee[tt] = 0.002 * np.minimum((t[tt] - t_app) / 0.05, 1)
        F[tt] = fmax
        obj[:] = 0
    else:
        ph = t - t_app
        acc = (ph >= 0) & (ph < t_acc)
        cru = (ph >= t_acc) & (t < t_rel)
        aft = t >= t_rel
        ee[acc] = 0.5 * a * ph[acc] ** 2
        ee[cru] = s_acc + v * (ph[cru] - t_acc)
        obj[acc | cru] = ee[acc | cru]
        F[acc] = m * a + mu * m * G
        F[cru] = mu * m * G
        tr = t[aft] - t_rel
        ee[aft] = s + v * np.minimum(tr, v / 10.0) - 5.0 * np.minimum(tr, v / 10.0) ** 2   # hard stop (10 m/s^2)
        ts = np.minimum(tr, t_slide)
        decel = v ** 2 / (2 * max(slide, 1e-9))
        obj[aft] = s + v * ts - 0.5 * decel * ts ** 2
    contact = (F > 0.5).astype(np.float32)
    lateral = rng.normal(0, 0.03) * obj
    pos = lambda a1, a2=None: np.stack([sgn * a1, a2 if a2 is not None else np.zeros(T), np.zeros(T)], 1)
    force = np.stack([-sgn * F, np.zeros(T), np.zeros(T)], 1)
    x = sg.set(pos(ee), force=force, obj=pos(obj, lateral), contact=contact)
    travel = sgn * obj[-1]
    r_new = state["remaining"] - travel
    wall = state["remaining"] + 0.12   # fixture beyond the target line
    collision = travel > wall if sgn > 0 else False
    if collision:
        travel = wall
        r_new = state["remaining"] - travel
    succ = abs(r_new) < 0.015
    eff = _effort(F, np.abs(np.gradient(np.gradient(ee, sg.dt), sg.dt)), sg.dt, robot, vmax_used=v)
    prog = float(np.clip(1 - abs(r_new) / state["r0"], 0, 1))
    cont = np.array([travel, lateral[-1], 0.0, prog, F.max() / 50, dur, slide, v], np.float32)
    state["remaining"] = r_new
    return StepSim(x, eff, dur, cont, _ev(success=succ, collision=collision, stall=stall),
                   "guard_tripped" if stall else "done", succ,
                   dict(progress=prog, collision=int(collision), stall=int(stall)))


def step_lift(state, lat, p, robot: SynthRobot, rng, bias):
    m, mug, k = lat["mass"], lat["grip_friction"], lat["stiffness"]
    fmax = robot.spec.max_ee_force
    Fg = min(p["grip_force"], fmax * 1.2)
    a = p["accel"]
    vt = min(p["speed"], robot.spec.max_ee_speed)
    h = p["height"]
    stall = m * (G + a) > min(fmax, 9.81 * robot.spec.payload_kg * 1.3)
    cap = 2 * mug * Fg * float(np.exp(rng.normal(0, 0.04)))
    transport = h >= 0.05
    a_t = 6.0 * vt if transport else 0.0
    need_lift = m * (G + a)
    need_tr = m * math.sqrt(G ** 2 + a_t ** 2)
    rho = max(need_lift, need_tr) / cap
    drop = (rho > 1.0) and not stall
    slip = (rho > 0.8) and not stall
    t_grip, t_lift = 0.25, 2 * math.sqrt(2 * h / a) if not stall else 0.3
    t_tr = (0.3 / vt) if transport else 0.0
    dur = t_grip + t_lift + t_tr + 0.15
    sg = _Seg(dur, robot, rng, bias)
    t = sg.t
    gc = _ramp(t, 0, t_grip * 0.6, 0.0, 1.0)
    comp = Fg / k   # part compression under grip
    w0 = 0.04
    gp = np.where(t < t_grip * 0.6, _ramp(t, 0, t_grip * 0.6, 0.06, w0), w0 - comp * _ramp(t, t_grip * 0.6, t_grip, 0, 1))
    z = np.zeros(T)
    xx = np.zeros(T)
    lift = (t >= t_grip) & (t < t_grip + t_lift)
    tl = t[lift] - t_grip
    hz = h if not stall else 0.002
    z[lift] = np.where(tl < t_lift / 2, 0.5 * (a if not stall else 0.05) * tl ** 2,
                       hz - 0.5 * (a if not stall else 0.05) * (t_lift - tl) ** 2)
    z[t >= t_grip + t_lift] = hz
    trn = t >= t_grip + t_lift
    if transport:
        xx[trn] = _ramp(t[trn], t_grip + t_lift, t_grip + t_lift + t_tr, 0, 0.3)
    acc_z = np.gradient(np.gradient(z, sg.dt), sg.dt)
    acc_x = np.gradient(np.gradient(xx, sg.dt), sg.dt)
    held = t >= t_grip * 0.8
    # load on wrist: object weight + inertia while held (minus slip / drop)
    obj_z, obj_x = z.copy(), xx.copy()
    slip_disp = 0.0
    if slip:
        slip_disp = 0.004 * (rho - 0.8) / 0.2
        obj_z = z - slip_disp * np.clip((t - t_grip) / max(t_lift, 1e-3), 0, 1)
    load = np.where(held, 1.0, 0.0)
    if drop:
        # dropped at the moment of peak demand (lift accel phase if lift-limited else transport)
        td = t_grip + (0.25 * t_lift if need_lift >= need_tr else t_lift + 0.3 * t_tr)
        after = t >= td
        load[after] = 0.0
        obj_z[after] = np.maximum(obj_z[np.argmax(after)] - 0.5 * G * (t[after] - td) ** 2, 0)
        obj_x[after] = obj_x[np.argmax(after)]
    fz = -(m * (G + acc_z)) * load
    fx = -(m * acc_x) * load
    force = np.stack([fx, np.zeros(T), fz], 1)
    contact = (gc > 0.5).astype(np.float32) * np.maximum(load, (t < t_grip).astype(float))
    pos = np.stack([xx, np.zeros(T), z + 0.0], 1)
    x = sg.set(pos, force=force, grip_cmd=gc, grip_pos=gp, obj=np.stack([obj_x, np.zeros(T), obj_z], 1),
               contact=contact)
    succ = transport and not drop and not stall and h >= 0.1
    eff = _effort(np.abs(fz) + np.abs(fx), np.abs(acc_z) + np.abs(acc_x), sg.dt, robot,
                  extra_sq=(Fg / 100.0) ** 2 * dur * 4.0, vmax_used=max(vt, a * t_lift / 2))
    risk_drop = drop and h >= 0.05
    prog = 1.0 if succ else 0.0
    cont = np.array([obj_x[-1], 0.0, obj_z[-1], prog, np.abs(force).max() / 50, dur, slip_disp * 250,
                     comp * 100], np.float32)
    return StepSim(x, eff, dur, cont, _ev(success=succ, slip=slip, drop=risk_drop, stall=stall),
                   "guard_tripped" if stall else ("contact_lost" if drop else "done"), succ,
                   dict(progress=prog, drop=int(risk_drop), slip=int(slip), stall=int(stall), collision=0))


def step_insert(state, lat, p, robot: SynthRobot, rng, bias):
    off = np.array([lat["offset_x"], lat["offset_y"]])
    c, k = lat["clearance"], lat["stiffness"]
    aim = np.array([p["aim_x"], p["aim_y"]])
    vi, r = p["speed"], p["search_r"]
    e = off - aim
    en = float(np.linalg.norm(e))
    chamfer = 0.0012 * (1.0 - 0.6 * vi / 0.05)
    capture = c + chamfer
    direct = en < capture
    found = (not direct) and (r > 0) and (en < capture + r)
    succ = direct or found
    # jam: contact impact force with stiffness, worse when fast and misaligned
    t_app, depth = 0.15, 0.02
    impact = math.sqrt(k) * vi * 20.0
    jam = ((not succ) and impact > 45.0) or (direct and en > c and vi > 0.035 and k > 8000)
    t_search = (r / 0.0005) * 0.15 if found or ((not direct) and r > 0) else 0.0
    t_ins = depth / vi if succ else 0.004 / vi
    dur = t_app + t_ins + t_search + 0.1
    sg = _Seg(dur, robot, rng, bias)
    t = sg.t
    z = _ramp(t, 0, t_app, 0.01, 0.0)
    after = t >= t_app
    pen = np.zeros(T)
    lat_f = np.zeros((T, 2))
    xy = np.tile(aim, (T, 1))
    if succ:
        z[after] = -np.minimum(vi * (t[after] - t_app - t_search).clip(0), depth)
        if found:
            ss = (t >= t_app) & (t < t_app + t_search)
            ph = (t[ss] - t_app) / max(t_search, 1e-6)
            ang = ph * 6 * np.pi
            xy[ss] = aim + np.stack([np.cos(ang), np.sin(ang)], 1) * (r * ph)[:, None]
            xy[t >= t_app + t_search] = off
        else:
            xy[after] = aim + (e * np.clip((t[after] - t_app) / 0.05, 0, 1)[:, None])
        fz = np.where(after, 3.0 + np.minimum(k * 2e-4, 6.0) * np.exp(-np.clip(t - t_app, 0, None) / 0.05), 0.0)
        if en > c:  # chamfer-guided: lateral force toward hole during capture
            lat_f[after] = (e / (en + 1e-9)) * min(k * (en - c), 30.0) * np.exp(-(t[after] - t_app) / 0.05)[:, None]
    else:
        pen[after] = np.minimum(vi * (t[after] - t_app), 0.001)
        z[after] = -pen[after]
        fz = np.where(after, np.minimum(k * pen + impact * np.exp(-np.clip(t - t_app, 0, None) / 0.02), 80.0), 0.0)
        if en < capture + 0.0015:  # on the chamfer: lateral force reveals direction
            lat_f[after] = (e / (en + 1e-9)) * min(k * (capture + 0.0015 - en) * 0.5, 20.0)
        if r > 0:
            ss = (t >= t_app) & (t < t_app + t_search)
            ph = (t[ss] - t_app) / max(t_search, 1e-6)
            ang = ph * 6 * np.pi
            xy[ss] = aim + np.stack([np.cos(ang), np.sin(ang)], 1) * (r * ph)[:, None]
    force = np.concatenate([lat_f, np.abs(fz)[:, None]], 1)   # contact pushes the hand up
    pos = np.concatenate([xy, z[:, None]], 1)
    contact = (np.linalg.norm(force, axis=1) > 2.0).astype(np.float32)
    # target = the hole (static); observed "obj" channel is the peg pose (held), relative to the perceived hole
    x = sg.set(pos, force=force, obj=pos.copy(), contact=contact)
    eff = _effort(np.linalg.norm(force, axis=1), np.abs(np.gradient(np.gradient(z, sg.dt), sg.dt)), sg.dt, robot,
                  vmax_used=vi)
    prog = 1.0 if succ else 0.0
    cont = np.array([0.0, 0.0, z[-1], prog, np.linalg.norm(force, axis=1).max() / 50, dur,
                     (1.0 if succ else pen[-1] / 0.001), np.linalg.norm(lat_f.mean(0)) / 10], np.float32)
    return StepSim(x, eff, dur, cont, _ev(success=succ, jam=jam), "done" if succ else "guard_tripped", succ,
                   dict(progress=prog, jam=int(jam), collision=0))


def step_drawer(state, lat, p, robot: SynthRobot, rng, bias):
    b, fb = lat["damping"], lat["breakaway"]
    F = float(np.clip(p["force"], -robot.spec.max_ee_force, robot.spec.max_ee_force))
    Tp = p["duration"]
    sgn = 1.0 if F >= 0 else -1.0
    moves = abs(F) > fb
    v_ss = (abs(F) - 0.3 * fb) / b if moves else 0.0
    stall = False
    if v_ss > robot.spec.max_ee_speed:
        v_ss = robot.spec.max_ee_speed
    x0 = state["x"]
    dur = Tp + 0.2
    sg = _Seg(dur, robot, rng, bias)
    t = sg.t
    tau = 0.06
    tt = np.clip(t - 0.05, 0, Tp)
    xs = sgn * v_ss * (tt - tau * (1 - np.exp(-tt / tau)))
    xs = xs * float(np.exp(rng.normal(0, 0.03)))
    lo, hi = -x0, 0.40 - x0
    hit_speed = 0.0
    if xs.max() > hi or xs.min() < lo:
        hit_speed = v_ss
        xs = np.clip(xs, lo, hi)
    collision = hit_speed > 0.25
    if not moves and abs(F) > 0.5 * robot.spec.max_ee_force:
        stall = True
    Ft = np.where((t >= 0.05) & (t <= Tp + 0.05), F, 0.0)
    if moves:
        # damping force once moving, breakaway peak at start
        Ft = np.where((t >= 0.05) & (t <= Tp + 0.05), sgn * (0.3 * fb + b * np.abs(np.gradient(xs, sg.dt))), 0.0)
        Ft = Ft + np.where((t >= 0.05) & (t < 0.09), sgn * 0.7 * fb, 0.0)
        Ft = np.clip(Ft, -abs(F), abs(F))
    pos = np.stack([xs, np.zeros(T), np.zeros(T)], 1)
    force = np.stack([-Ft, np.zeros(T), np.zeros(T)], 1)
    x = sg.set(pos, force=force, grip_cmd=np.ones(T), grip_pos=np.full(T, 0.02), obj=pos.copy(),
               contact=(np.abs(Ft) > 0.5).astype(np.float32))
    x_new = x0 + xs[-1]
    state["x"] = x_new
    err = abs(x_new - state["goal"])
    succ = err < 0.02 and not collision
    eff = _effort(np.abs(Ft), np.abs(np.gradient(np.gradient(xs, sg.dt), sg.dt)), sg.dt, robot, vmax_used=v_ss)
    prog = float(np.clip(1 - err / state["goal"], 0, 1))
    cont = np.array([xs[-1], 0.0, 0.0, prog, np.abs(Ft).max() / 50, dur, v_ss, x_new], np.float32)
    return StepSim(x, eff, dur, cont, _ev(success=succ, collision=collision, stall=stall),
                   "done", succ, dict(progress=prog, collision=int(collision), stall=int(stall)))


def step_sweep(state, lat, p, robot: SynthRobot, rng, bias):
    mub, M = lat["bulk_friction"], lat["mass"]
    P = state["particles"]          # (N,2) x along sweep axis, y
    N = len(P)
    mp = M / state["n0"]
    depth, vs, L = p["depth"], min(p["speed"], robot.spec.max_ee_speed), p["stroke"]
    x0 = P[:, 0].min() - 0.01
    # engaged: particles within the stroke window, engaged with prob depth (blade depth into the pile)
    inwin = (P[:, 0] >= x0) & (P[:, 0] <= x0 + L) & (P[:, 0] < state["goal_lo"])
    eng = inwin & (rng.random(N) < depth)
    n_eng = int(eng.sum())
    Freq = mub * mp * n_eng * G * (1 + 0.5 * vs)
    fmax = robot.spec.max_ee_force
    stall = Freq > fmax
    frac = 1.0 if not stall else fmax / Freq
    Leff = L * frac
    t_app = 0.1
    t_str = Leff / vs + vs / 4.0
    dur = t_app + t_str + 0.3
    newP = P.copy()
    if n_eng:
        slide = vs ** 2 / (2 * mub * G) * rng.uniform(0.2, 1.0, n_eng)
        newP[eng, 0] = np.maximum(P[eng, 0], x0 + Leff) + slide
        newP[eng, 1] += rng.normal(0, 0.01, n_eng) * vs
    sg = _Seg(dur, robot, rng, bias)
    t = sg.t
    bx = x0 + _ramp(t, t_app, t_app + t_str, 0, Leff)
    # load grows as the blade gathers particles
    gathered = np.clip((bx - x0) / max(Leff, 1e-6), 0, 1)
    Ft = np.where((t >= t_app) & (t <= t_app + t_str), mub * mp * n_eng * G * (0.3 + 0.7 * gathered), 0.0)
    Ft = Ft + mp * n_eng * np.abs(np.gradient(np.gradient(bx, sg.dt), sg.dt)) * (t <= t_app + t_str)
    Ft = np.minimum(Ft, fmax)
    pos = np.stack([bx - P[:, 0].mean(), np.zeros(T), np.full(T, -0.01 * depth)], 1)
    cen = np.stack([_ramp(t, t_app, t_app + t_str + 0.2, P[:, 0].mean(), newP[:, 0].mean()) - P[:, 0].mean(),
                    np.zeros(T), np.zeros(T)], 1)
    force = np.stack([-Ft, np.zeros(T), np.zeros(T)], 1)
    x = sg.set(pos, force=force, obj=cen, contact=(Ft > 0.5).astype(np.float32))
    spill = newP[:, 0] > state["goal_hi"] + 0.05
    keep = ~spill
    in_goal = (newP[:, 0] >= state["goal_lo"]) & (newP[:, 0] <= state["goal_hi"])
    frac_goal = float(in_goal.sum() / state["n0"])
    state["particles"] = newP[keep]
    spill_frac = float(spill.sum() / state["n0"])
    state["spilled"] = state.get("spilled", 0.0) + spill_frac
    succ = frac_goal >= 0.8
    drop = spill_frac > 0.05
    eff = _effort(Ft, np.abs(np.gradient(np.gradient(bx, sg.dt), sg.dt)), sg.dt, robot, vmax_used=vs)
    cont = np.array([cen[-1, 0], 0.0, 0.0, frac_goal, Ft.max() / 50, dur, spill_frac, n_eng / state["n0"]],
                    np.float32)
    return StepSim(x, eff, dur, cont, _ev(success=succ, drop=drop, stall=stall), "done", succ,
                   dict(progress=frac_goal, drop=int(drop), stall=int(stall), collision=0))


STEP = {"slide": step_slide, "lift": step_lift, "insert": step_insert, "drawer": step_drawer, "sweep": step_sweep}


def init_state(family, rng, lat):
    if family == "slide":
        r0 = float(rng.uniform(0.12, 0.6))
        return dict(remaining=r0, r0=r0, size=rng.uniform(0.03, 0.08, 3))
    if family == "lift":
        return dict(size=rng.uniform(0.02, 0.06, 3))
    if family == "insert":
        return dict(size=np.array([0.006, 0.006, 0.02]))
    if family == "drawer":
        return dict(x=0.0, goal=float(rng.uniform(0.12, 0.3)), size=np.array([0.05, 0.01, 0.01]))
    if family == "sweep":
        n = int(rng.integers(20, 201))
        P = np.stack([rng.uniform(0.0, 0.10, n), rng.uniform(-0.05, 0.05, n)], 1)
        return dict(particles=P, n0=n, goal_lo=0.22, goal_hi=0.34, size=np.array([0.05, 0.05, 0.02]))
    raise KeyError(family)


def entity_frame(family, state, robot_name) -> S.EntityFrame:
    """actor-legal entity snapshot (positions relative to target)."""
    ET = S.ENT_TYPES.index
    rows = []  # (type, pos3, size3)
    sz = np.asarray(state.get("size", np.zeros(3)), float)
    if family == "slide":
        rows = [(ET("object"), [0, 0, 0], sz), (ET("goal_region"), [state["remaining"], 0, 0], [0.015, 0.1, 0.01]),
                (ET("effector"), [-0.05, 0, 0], [0.01] * 3),
                (ET("fixture"), [state["remaining"] + 0.12 + sz[0], 0, 0], [0.01, 0.3, 0.05])]
    elif family == "lift":
        rows = [(ET("object"), [0, 0, 0], sz), (ET("goal_region"), [0.3, 0, 0.1], [0.05, 0.05, 0.01]),
                (ET("effector"), [0, 0, 0.1], [0.01] * 3)]
    elif family == "insert":
        rows = [(ET("fixture"), [0, 0, 0], [0.007, 0.007, 0.02]), (ET("object"), [0, 0, 0.03], sz),
                (ET("effector"), [0, 0, 0.06], [0.01] * 3), (ET("goal_region"), [0, 0, -0.02], [0.002] * 3)]
    elif family == "drawer":
        rows = [(ET("link"), [0, 0, 0], sz), (ET("goal_region"), [state["goal"] - state["x"], 0, 0], [0.02, 0.05, 0.05]),
                (ET("fixture"), [0.4 - state["x"], 0, 0], [0.01, 0.2, 0.2]), (ET("effector"), [0, 0, 0], [0.01] * 3)]
    elif family == "sweep":
        P = state["particles"]
        c = np.r_[P.mean(0), 0] if len(P) else np.zeros(3)
        ext = np.r_[P.std(0) * 2, 0.02] if len(P) else np.full(3, 0.01)
        rows = [(ET("object"), [0, 0, 0], ext),
                (ET("goal_region"), [(state["goal_lo"] + state["goal_hi"]) / 2 - c[0], -c[1], 0],
                 [(state["goal_hi"] - state["goal_lo"]) / 2, 0.1, 0.02]),
                (ET("effector"), [P[:, 0].min() - 0.02 - c[0] if len(P) else -0.05, 0, 0], [0.005, 0.06, 0.03])]
        rows += [(ET("particle"), [px - c[0], py - c[1], 0], [0.004] * 3) for px, py in P]
    n = len(rows)
    return S.EntityFrame(types=np.array([r[0] for r in rows]), pos=np.array([r[1] for r in rows], float),
                         quat=np.tile([1.0, 0, 0, 0], (n, 1)), size=np.array([r[2] for r in rows], float),
                         vel=np.zeros((n, 3)), age=np.zeros(n), valid=np.ones(n, bool), target=0)


# ----------------------------------------------------------------- episode / dataset generation
@dataclass
class SynthEpisode:
    family: str
    robot: str
    seed: int
    latents: dict
    steps: list = field(default_factory=list)   # list of dict(frame, interaction)


def probe_params(family, rng=None) -> dict:
    """low-risk informative probe per family (used by the planner's VOI gate and by data collection)."""
    if family == "slide":
        return {"dir": 1.0, "dist": 0.02, "speed": 0.15}
    if family == "lift":
        return {"accel": 2.0, "grip_force": 25.0, "height": 0.01, "speed": 0.1}
    if family == "insert":
        return {"aim_x": 0.0, "aim_y": 0.0, "search_r": 0.0, "speed": 0.005}
    if family == "drawer":
        return {"duration": 0.2, "force": 30.0}
    if family == "sweep":
        return {"depth": 0.3, "speed": 0.2, "stroke": 0.05}
    raise KeyError(family)


def run_step(family, state, lat, params, robot_name, rng, bias, ep_id, k, t_now):
    robot = ROBOTS[robot_name]
    st = STEP[family](state, lat, params, robot, rng, bias)
    call = api.SkillCall(skill=FAMILY_SKILL[family], params=dict(params), target="target")
    em = np.array(EVENT_MASK[family], bool)
    inter = S.Interaction(episode_id=ep_id, step=k, family=family, robot=robot_name, call_vec=call.vector(),
                          skill=call.skill, params=dict(params), target_is_query=True, segment=st.seg,
                          effort_cost=st.effort.cost(), effort=st.effort.as_dict(), t_end=t_now + st.duration,
                          outcome_cont=st.cont, outcome_cont_mask=np.ones(S.N_CONT, bool), outcome_events=st.events,
                          outcome_events_mask=em, labels=dict(lat))
    return st, inter


def generate_episode(family, robot_name, seed, n_steps=None, policy="random", probe_prob=0.25, stress=False):
    rng = np.random.default_rng(seed)
    lat = sample_latents(family, rng, stress)
    state = init_state(family, rng, lat)
    robot = ROBOTS[robot_name]
    bias = rng.normal(0, robot.ft_bias, 6)
    ep = SynthEpisode(family, robot_name, seed, lat)
    t_now = 0.0
    for k in range(n_steps or MAX_STEPS[family]):
        frame = entity_frame(family, state, robot_name)
        if policy == "random":
            params = probe_params(family) if rng.random() < probe_prob else sample_params(family, rng)
            if family == "slide":   # bias pushes toward the goal (realistic data), keep some reversed
                params["dir"] = float(np.sign(state["remaining"]) if rng.random() < 0.85 else -np.sign(state["remaining"]))
            if family == "drawer":
                params["force"] = abs(params["force"]) * (np.sign(state["goal"] - state["x"]) if rng.random() < 0.85 else -1)
        else:
            params = policy(family, state, frame, ep, rng)
        st, inter = run_step(family, state, lat, params, robot_name, rng, bias, f"{family}-{robot_name}-{seed}", k,
                             t_now)
        t_now = inter.t_end + 0.5
        ep.steps.append(dict(frame=frame, inter=inter))
        if family == "sweep" and len(state["particles"]) == 0:
            break
    return ep


# ----------------------------------------------------------------- scripted heuristic (baseline a)
def heuristic_params(family, state, frame, ep_or_none, rng):
    """what a competent integrator would script WITHOUT knowing the hidden properties: nominal-property inverse
    model + conservative margins (the 'scripted heuristic skill baseline')."""
    if family == "slide":
        r = state["remaining"]
        mu0 = 0.5
        # push most of the way, then a nominal-friction release slide; s + v^2/(2 mu0 g) = |r|
        v = min(0.35, math.sqrt(max(abs(r) * 0.5, 1e-4) * 2 * mu0 * G))
        s = max(abs(r) - v ** 2 / (2 * mu0 * G), 0.01)
        return {"dir": float(np.sign(r) or 1.0), "dist": float(min(s, 0.3)), "speed": float(v)}
    if family == "lift":
        return {"accel": 3.0, "grip_force": 60.0, "height": 0.12, "speed": 0.4}
    if family == "insert":
        return {"aim_x": 0.0, "aim_y": 0.0, "search_r": 0.002, "speed": 0.01}
    if family == "drawer":
        e = state["goal"] - state["x"]
        b0, fb0 = 15.0, 8.0
        F = 25.0
        v = (F - 0.3 * fb0) / b0
        return {"duration": float(np.clip(abs(e) / v, 0.1, 2.0)), "force": float(np.sign(e) * F)}
    if family == "sweep":
        P = state["particles"]
        need = state["goal_lo"] + 0.03 - (P[:, 0].min() if len(P) else 0)
        return {"depth": 0.8, "speed": 0.4, "stroke": float(np.clip(need, 0.03, 0.4))}
    raise KeyError(family)


# ----------------------------------------------------------------- IndustrialEnv-compatible wrapper
class SynthEnv:
    """api.IndustrialEnv over the synthetic families (so the closed-loop harness is exercised before ENVS lands).
    Extra (documented) hooks used by the v2 harness: `frame()` (EntityFrame), `probe_skills()`, `param_bounds()`."""

    def __init__(self):
        self.task = None

    def reset(self, task: api.TaskSpec):
        self.task = task
        self.rng = np.random.default_rng(task.seed)
        fam = task.family
        self.lat = sample_latents(fam, self.rng, task.params.get("stress", False))
        self.state = init_state(fam, self.rng, self.lat)
        self.bias = self.rng.normal(0, ROBOTS[task.robot].ft_bias, 6)
        self.k, self.t, self.done = 0, 0.0, False
        # "lot" protocol: L items with the SAME hidden properties processed in sequence (evidence carries over)
        self.lot = int(task.params.get("lot", 1))
        self.items_done = 0
        self.exec_rng = np.random.default_rng(task.seed + 7919)
        return []

    def frame(self):
        return entity_frame(self.task.family, self.state, self.task.robot)

    def param_bounds(self):
        return PARAM_BOUNDS[self.task.family]

    def candidate_skills(self, belief, n, rng):
        fam = self.task.family
        return [api.SkillCall(FAMILY_SKILL[fam], sample_params(fam, rng), target="target", candidate_id=f"c{i}")
                for i in range(n)]

    def probe_skills(self, belief=None):
        fam = self.task.family
        return [api.SkillCall(FAMILY_SKILL[fam], probe_params(fam), target="target", candidate_id="probe")]

    def goal(self):
        return dict(family=self.task.family)

    def render(self, camera="overview"):
        return np.zeros((8, 8, 3), np.uint8)

    def execute(self, call: api.SkillCall) -> api.StepOut:
        fam = self.task.family
        b = PARAM_BOUNDS[fam]
        params = {k: float(np.clip(call.params.get(k, np.mean(b[k])), *b[k])) for k in b}
        st, inter = run_step(fam, self.state, self.lat, params, self.task.robot, self.exec_rng, self.bias,
                             f"env-{self.task.seed}", self.k, self.t)
        self.k += 1
        self.t = inter.t_end + 0.5
        item_success = st.success
        if item_success:
            self.items_done += 1
            if self.items_done < self.lot:           # next item from the same lot: fresh geometry, same latents
                self.state = init_state(fam, self.rng, self.lat)
        all_done = self.items_done >= self.lot
        budget = MAX_STEPS[fam] * self.lot
        self.done = all_done or self.k >= budget or (fam == "sweep" and len(self.state["particles"]) == 0 and not item_success)
        res = api.SkillResult(ok=st.status == "done", status=st.status, effort=st.effort, duration_s=st.duration,
                              log=dict(segment=dict(x=st.seg, t0=self.t, t1=self.t + st.duration),
                                       skill=call.skill, params=params, target=call.target, interaction=inter))
        inter.labels = {}  # the actor-side copy carries no labels
        metrics = dict(st.metrics, item_success=int(item_success), items_done=self.items_done, lot=self.lot)
        return api.StepOut(events=[], result=res, done=self.done, success=all_done, metrics=metrics,
                           labels=dict(self.lat))
