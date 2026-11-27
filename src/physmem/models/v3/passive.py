"""Passive third-person domain (V3 H3; owner: V3-MODEL). MuJoCo scenes observed ONLY through a camera keypoint
tracker (`tracks.CameraTracker`), never through simulator state. Optional single known-force poke (H3 scale claim).

Scenes (one per episode, N_WIN windows; each window starts with an environment event the observer does NOT know):
  spring  : body on a horizontal slide joint with hidden stiffness k and damping c, mass m; event = unknown kick
            (initial displacement and velocity). Passively identifiable: k/m (period), c/m (decay). Not: m, k.
  collide : two spheres on a frictionless rail; A is launched at unknown speed into B at rest; hidden masses and
            restitution. Passively identifiable: m_A/m_B and e (momentum + energy). Not: absolute masses.
  bounce  : ball dropped from an unknown height onto the floor; hidden restitution (contact solref damping).
            Identifiable: e. Not: mass.
  slide   : box launched along a table at unknown speed; hidden Coulomb friction. Identifiable: mu (absolute,
            because g is known and the RGB-D tracker is metric). Not: mass.
Poke (p = 0.5, window 0): instead of the environment event, the observer's own probe applies a KNOWN force pulse
(F for 0.1 s, measured by its force sensor) to the target -> absolute mass m = impulse / dv, hence k = (k/m) m.

Labels (evaluator only): mass, k, c, k_over_m, c_over_m, mass_ratio, restitution (realised, measured from the
simulator's own pre/post-impact velocities), friction.
"""
from __future__ import annotations

import math

import numpy as np

from .tracks import CameraTracker

DT = 0.001
WIN = 1.5
N_WIN = 4
REC = 0.01
SCENES = ["spring", "collide", "bounce", "slide"]
G = 9.81


def _solref(e_target):
    """MuJoCo direct solref (-stiffness, -damping) giving roughly the requested restitution for a light contact."""
    k = 2.0e4
    zeta = -math.log(max(e_target, 1e-3)) / math.sqrt(math.pi ** 2 + math.log(max(e_target, 1e-3)) ** 2)
    return -k, -2 * zeta * math.sqrt(k)


def sample_scene(rng, scene=None):
    scene = scene or SCENES[rng.integers(len(SCENES))]
    p = dict(scene=scene, mass=float(np.exp(rng.uniform(np.log(0.1), np.log(2.0)))), poke=bool(rng.random() < 0.5))
    if scene == "spring":
        p["k_over_m"] = float(np.exp(rng.uniform(np.log(4.0), np.log(400.0))))
        p["k"] = p["k_over_m"] * p["mass"]
        p["zeta"] = float(rng.uniform(0.02, 0.4))
        p["c"] = 2 * p["zeta"] * math.sqrt(p["k"] * p["mass"])
    elif scene == "collide":
        p["mass_b"] = float(p["mass"] * np.exp(rng.uniform(np.log(0.25), np.log(4.0))))
        p["e_set"] = float(rng.uniform(0.1, 0.9))
    elif scene == "bounce":
        p["e_set"] = float(rng.uniform(0.2, 0.9))
    else:
        p["mu"] = float(rng.uniform(0.1, 0.8))
    return p


def _xml(p):
    s = p["scene"]
    if s == "spring":
        body = f"""<body name="a" pos="0 0 0.05"><joint name="j" type="slide" axis="1 0 0" stiffness="{p['k']}"
          damping="{p['c']}"/><geom type="sphere" size="0.03" mass="{p['mass']}" contype="0" conaffinity="0"/></body>"""
    elif s == "collide":
        sr = "%.3f %.3f" % _solref(p["e_set"])
        body = f"""<body name="a" pos="-0.15 0 0.05"><joint name="ja" type="slide" axis="1 0 0"/>
          <geom type="sphere" size="0.03" mass="{p['mass']}" solref="{sr}" solimp="0.99 0.99 0.001"/></body>
          <body name="b" pos="0.05 0 0.05"><joint name="jb" type="slide" axis="1 0 0"/>
          <geom type="sphere" size="0.03" mass="{p['mass_b']}" solref="{sr}" solimp="0.99 0.99 0.001"/></body>"""
    elif s == "bounce":
        sr = "%.3f %.3f" % _solref(p["e_set"])
        body = f"""<body name="a" pos="0 0 0.12"><joint name="j" type="slide" axis="0 0 1"/>
          <geom type="sphere" size="0.03" mass="{p['mass']}" solref="{sr}" solimp="0.99 0.99 0.001"/></body>"""
    else:
        body = f"""<body name="a" pos="0 0 0.02"><joint name="jx" type="slide" axis="1 0 0"/>
          <joint name="jz" type="slide" axis="0 0 1"/><geom type="box" size="0.03 0.03 0.02" mass="{p['mass']}"
          friction="{p['mu']} 0.005 0.0001" priority="1"/></body>"""
    floor = '' if s in ("spring", "collide") else \
        '<geom name="floor" type="plane" size="3 1 0.1" friction="0.5 0.005 0.0001"/>'
    return f"""<mujoco><option timestep="{DT}" integrator="implicitfast" gravity="0 0 -{G}"/>
      <worldbody>{floor}{body}</worldbody></mujoco>"""


def simulate(p, rng):
    """-> truth dict: t (n,), pos (n, nb, 3) keypoints, events, realised labels; poke force record if any."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(_xml(p))
    d = mujoco.MjData(m)
    nb = 2 if p["scene"] == "collide" else 1
    bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in (["a", "b"] if nb == 2 else ["a"])]
    rec_t, rec_x, rec_f = [], [], []
    e_meas, events = [], []
    every = int(round(REC / DT))
    for w in range(N_WIN):
        poke = p["poke"] and w == 0
        mujoco.mj_resetData(m, d)
        F_poke = 0.0
        s = p["scene"]
        if poke:
            F_poke = float(rng.uniform(2.0, 8.0))
        elif s == "spring":
            d.qpos[0] = rng.uniform(-0.06, 0.06); d.qvel[0] = rng.uniform(-0.4, 0.4)
        elif s == "collide":
            d.qvel[0] = rng.uniform(0.3, 1.2)
        elif s == "bounce":
            d.qpos[0] = rng.uniform(0.0, 0.15)
        else:
            d.qvel[0] = rng.uniform(0.5, 2.0)
        events.append(dict(poke=poke, F=F_poke))
        mujoco.mj_forward(m, d)
        t0 = w * (WIN + 0.5)
        vprev = d.qvel.copy()
        v_app, in_con = None, False
        for i in range(int(round(WIN / DT))):
            tt = i * DT
            d.xfrc_applied[:] = 0
            f_now = F_poke if (poke and tt < 0.1) else 0.0
            if f_now:
                d.xfrc_applied[bids[0], 0 if s != "bounce" else 2] = f_now
            mujoco.mj_step(m, d)
            if s in ("collide", "bounce"):
                v = d.qvel.copy()
                # realised restitution at the first impact of the window (evaluator-side measurement)
                if s == "bounce" and vprev[0] < -0.05 and v[0] > 0 and len(e_meas) <= w:
                    e_meas.append(-v[0] / vprev[0])
                if s == "collide" and len(e_meas) <= w:
                    if d.ncon > 0 and not in_con:
                        in_con, v_app = True, vprev[0] - vprev[1]
                    elif d.ncon == 0 and in_con:
                        if v_app is not None and v_app > 0.05:
                            e_meas.append(-(v[0] - v[1]) / v_app)
                        in_con = False
                vprev = v
            if (i + 1) % every == 0:
                rec_t.append(t0 + tt + DT)
                rec_x.append(np.stack([d.xpos[b].copy() for b in bids]))
                rec_f.append(f_now)
    lab = {k: p[k] for k in ("mass", "k", "c", "k_over_m", "mu", "mass_b") if k in p}
    if p["scene"] == "spring":
        lab["c_over_m"] = p["c"] / p["mass"]
    if p["scene"] == "collide":
        lab["mass_ratio"] = p["mass"] / p["mass_b"]
    if e_meas:
        lab["restitution"] = float(np.median(e_meas))
    return dict(t=np.array(rec_t), pos=np.array(rec_x), f=np.array(rec_f), events=events, labels=lab, nb=nb)


def observe(tr, rng, cfg=None):
    """actor side: tracker detections of every body (back-projected, NaN = missed) + the probe's measured force."""
    cam = CameraTracker(rng, center=(0.0, 0.0, 0.08), cfg=cfg)
    ft = cam.frame_times(0.0, tr["t"][-1], rng)
    P = np.stack([np.stack([np.interp(ft, tr["t"], tr["pos"][:, b, j]) for j in range(3)], -1)
                  for b in range(tr["nb"])], 1)
    u, v, dpt = cam.observe(P, rng)
    xyz = cam.backproject(u, v, dpt)
    fmeas = tr["f"] + rng.normal(0, 0.05, len(tr["f"]))
    return dict(ft=ft, xyz=xyz, t=tr["t"], fmeas=fmeas, poke=[e["poke"] for e in tr["events"]],
                poke_F=[e["F"] for e in tr["events"]])


def generate_episode(seed, scene=None):
    rng = np.random.default_rng(seed)
    p = sample_scene(rng, scene)
    tr = simulate(p, rng)
    return p, tr, observe(tr, rng)
