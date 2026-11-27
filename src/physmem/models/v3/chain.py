"""Identifiable contact-chain family (V3 section 0; owner: V3-MODEL).

A pusher (the robot's own force-controlled effector) drives a 1-D chain of L boxes along +x on a table. Hidden
per-box mass m_i and Coulomb friction mu_i; optionally the last box is tied to a wall anchor by a spring of hidden
stiffness k (damping fixed and known). Each episode has N_WIN excitation windows. In each window the pusher applies
a KNOWN constant force F for T seconds, retracts (-4 N for 0.15 s) and coasts back under known joint damping; the
chain slides to rest after release (post-release slide). Between windows the scene settles for SETTLE s and the
pusher is re-placed behind box 1.

Observation modalities (all actor-legal, all noisy; generated here, simulator state never crosses):
  * proprio/force  : pusher encoder (100 Hz), finite-difference velocity, wrist force sensor along x (noise + per-
                     episode bias), thresholded contact flag, commanded force.
  * keypoint tracks: third-person RGB-D keypoint tracker of every box (tracks.CameraTracker: 30 Hz with jitter,
                     frame drops, dropout, outliers, occluder band, depth noise, calibration error).
  * pose anchors   : sparse "golden" motion-capture fixes of boxes (sd 0.3 mm), Poisson(1.2) per box per window.
Entity state at decision time is the actor's own estimate (anchor if fresh, else tracker median) with provenance.

Evaluator-only labels: true trajectories on the physical-time grid TAU (0..1.5 s every 0.1 s), contact flags and
interval-average contact forces per edge, and the hidden parameters.

Also here: a vectorised analytic 1-D chain simulator (`AnalyticChain`) used by the classical baseline (CEM system
identification on the same observations) and by the labelled oracle, and a slide-to-target planner task.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .tracks import CameraTracker

G = 9.81
DT = 0.002
REC = 0.01                       # 100 Hz record
WIN = 1.6                        # observed window length (s)
SETTLE = 0.8
N_WIN = 4
TAU = np.round(np.arange(16) * 0.1, 6)    # physical-time belief grid (s), 16 points, 15 intervals
N_TAU = len(TAU)
N_INT = N_TAU - 1
T_EV = 32                        # evidence bins per window
L_MAX = 4
N_ENT = L_MAX + 2                # pusher, boxes, wall
N_EDGE = L_MAX + 1               # pusher-B1, Bi-Bi+1, B_L-wall
PUSHER_M, PUSHER_R, PUSHER_B = 0.2, 0.015, 2.0
SPRING_C = 1.0                   # known spring-box joint damping (N s / m)
RETRACT_F, RETRACT_T = -2.0, 0.15
V_MAX, V_SOFT = 0.6, 0.1          # pusher admittance limit: force fades to 0 between V_MAX - V_SOFT and V_MAX
BOX_HY, BOX_HZ = 0.03, 0.02
F_RANGE, T_RANGE = (1.0, 15.0), (0.1, 0.3)
M_RANGE, MU_RANGE, K_RANGE = (0.2, 1.0), (0.15, 0.6), (15.0, 150.0)
SPRING_P = 0.4
ENT_PUSHER, ENT_BOX, ENT_WALL = 0, 1, 2
EDGE_CONTACT, EDGE_SPRING = 0, 1
MOD_TRACK, MOD_PROPRIO, MOD_ANCHOR = 0, 1, 2
N_MOD = 3
EV_C = 6                         # evidence channels: x, v, f, contact, f_cmd, valid
PROV = ["anchor", "tracker", "known", "stale"]
SIG_ANCHOR, SIG_ENC, SIG_F, SIG_FB = 3e-4, 5e-5, 0.2, 0.3


def cmd_force(t, F, T, vp, xp=np):
    """known pusher command (N) at window-relative time t and pusher velocity vp: constant force F for T s with a
    speed (admittance) limit, then a -2 N retract for 0.15 s (speed-limited), then 0. Works on numpy or torch."""
    fwd = F * xp.clip((V_MAX - vp) / V_SOFT, 0.0, 1.0)
    back = RETRACT_F * xp.clip((V_MAX + vp) / V_SOFT, 0.0, 1.0)
    zero = 0.0 * vp
    return xp.where(t < T, fwd, xp.where(t < T + RETRACT_T, back, zero))


@dataclass
class ChainParams:
    L: int
    mass: np.ndarray
    mu: np.ndarray
    half: np.ndarray                 # box half widths along x
    gaps: np.ndarray                 # initial gaps between consecutive boxes (L-1,)
    spring: bool
    k: float
    actions: list                    # [(F, T)] * N_WIN
    pusher_gap: list                 # extra gap behind box 1 at each window start

    def copy(self, **kw):
        d = dict(L=self.L, mass=self.mass.copy(), mu=self.mu.copy(), half=self.half.copy(), gaps=self.gaps.copy(),
                 spring=self.spring, k=self.k, actions=list(self.actions), pusher_gap=list(self.pusher_gap))
        d.update(kw)
        return ChainParams(**d)


def sample_params(rng, L=None, Ls=(1, 2, 3)):
    L = int(L if L is not None else Ls[rng.integers(len(Ls))])
    mass = np.exp(rng.uniform(np.log(M_RANGE[0]), np.log(M_RANGE[1]), L))
    mu = rng.uniform(*MU_RANGE, L)
    half = rng.uniform(0.02, 0.04, L)
    gaps = rng.uniform(0.0, 0.01, max(L - 1, 0))
    spring = bool(rng.random() < SPRING_P)
    k = float(np.exp(rng.uniform(np.log(K_RANGE[0]), np.log(K_RANGE[1])))) if spring else 0.0
    actions = [(float(rng.uniform(*F_RANGE)), float(rng.uniform(*T_RANGE))) for _ in range(N_WIN)]
    pg = [float(rng.uniform(0.001, 0.005)) for _ in range(N_WIN)]
    return ChainParams(L, mass, mu, half, gaps, spring, k, actions, pg)


def box_x0(p: ChainParams):
    x, out = 0.0, []
    for i in range(p.L):
        x = x + p.half[i] if i == 0 else x + p.half[i - 1] + p.gaps[i - 1] + p.half[i]
        out.append(x)
    return np.array(out)


def _xml(p: ChainParams):
    x0 = box_x0(p)
    boxes = []
    for i in range(p.L):
        stiff = f'stiffness="{p.k:.6f}" damping="{SPRING_C}"' if (p.spring and i == p.L - 1) else 'damping="0"'
        boxes.append(f"""
    <body name="b{i}" pos="{x0[i]:.6f} 0 {BOX_HZ}">
      <joint name="bx{i}" type="slide" axis="1 0 0" {stiff}/>
      <joint name="bz{i}" type="slide" axis="0 0 1"/>
      <geom name="g{i}" type="box" size="{p.half[i]:.6f} {BOX_HY} {BOX_HZ}" mass="{p.mass[i]:.6f}"
            friction="{p.mu[i]:.6f} 0.005 0.0001" priority="1" contype="3" conaffinity="3" condim="3"/>
    </body>""")
    return f"""<mujoco>
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" gravity="0 0 -{G}"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 1 0.1" friction="0.5 0.005 0.0001" contype="1" conaffinity="1"/>
    <body name="pusher" pos="{-PUSHER_R - p.pusher_gap[0]:.6f} 0 {BOX_HZ}">
      <joint name="px" type="slide" axis="1 0 0" damping="{PUSHER_B}"/>
      <geom name="pg" type="sphere" size="{PUSHER_R}" mass="{PUSHER_M}" contype="2" conaffinity="2" condim="1"
            priority="2"/>
    </body>{''.join(boxes)}
  </worldbody>
  <actuator><motor joint="px" gear="1" ctrlrange="-40 40"/></actuator>
</mujoco>"""


class ChainSim:
    """MuJoCo chain; evaluator side. run() -> truth dict (100 Hz records over the whole episode)."""

    def __init__(self, p: ChainParams):
        import mujoco
        self.mj = mujoco
        self.p = p
        self.m = mujoco.MjModel.from_xml_string(_xml(p))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.gid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ["pg"] + [f"g{i}" for i in range(p.L)]]
        self.qx = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                   for n in ["px"] + [f"bx{i}" for i in range(p.L)]]
        self.vx = [self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                   for n in ["px"] + [f"bx{i}" for i in range(p.L)]]
        self.body0 = np.r_[-PUSHER_R - p.pusher_gap[0], box_x0(p)]
        self._f6 = np.zeros(6)

    def state_x(self):
        return np.array([self.d.qpos[a] for a in self.qx]) + self.body0

    def state_v(self):
        return np.array([self.d.qvel[a] for a in self.vx])

    def edge_forces(self):
        """normal contact force on each chain edge (pusher-B1, B1-B2, ...), N."""
        L = self.p.L
        f = np.zeros(L)
        g2e = {}
        for e in range(L):
            g2e[frozenset((self.gid[e], self.gid[e + 1]))] = e
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            e = g2e.get(frozenset((c.geom1, c.geom2)))
            if e is not None:
                self.mj.mj_contactForce(self.m, self.d, i, self._f6)
                f[e] += abs(self._f6[0])
        return f

    def place_pusher(self, gap):
        x = self.state_x()
        target = x[1] - self.p.half[0] - PUSHER_R - gap
        self.d.qpos[self.qx[0]] = target - self.body0[0]
        self.d.qvel[self.vx[0]] = 0.0
        self.mj.mj_forward(self.m, self.d)

    def run_window(self, F, T, rec):
        """simulate one observed window, appending 100 Hz rows to rec; returns per-step-accumulated interval-average
        edge forces (L, N_INT) on the TAU grid."""
        nstep = int(round(WIN / DT))
        every = int(round(REC / DT))
        imp = np.zeros((self.p.L, N_INT))
        t0 = self.d.time
        for s in range(nstep):
            t = s * DT
            self.d.ctrl[0] = float(cmd_force(t, F, T, self.d.qvel[self.vx[0]]))
            self.mj.mj_step(self.m, self.d)
            tt = (s + 1) * DT
            j = int(tt / 0.1 - 1e-9)
            if j < N_INT:
                imp[:, j] += self.edge_forces() * DT
            if (s + 1) % every == 0:
                rec["t"].append(self.d.time)
                rec["x"].append(self.state_x())
                rec["v"].append(self.state_v())
                rec["f"].append(self.edge_forces())
                rec["u"].append(float(self.d.ctrl[0]))
        return imp / 0.1

    def settle(self, rec):
        nstep = int(round(SETTLE / DT))
        every = int(round(REC / DT))
        self.d.ctrl[0] = 0.0
        for s in range(nstep):
            self.mj.mj_step(self.m, self.d)
            if (s + 1) % every == 0:
                rec["t"].append(self.d.time)
                rec["x"].append(self.state_x())
                rec["v"].append(self.state_v())
                rec["f"].append(self.edge_forces())
                rec["u"].append(0.0)


def simulate(p: ChainParams, windows=None, init_state=None):
    """run the episode (all N_WIN windows, or a single window from init_state=(x, v) for counterfactuals)."""
    sim = ChainSim(p)
    rec = dict(t=[], x=[], v=[], f=[], u=[])
    starts, fimp, sx, sv, win_idx = [], [], [], [], []
    if init_state is not None:
        x, v = init_state
        for i, (a, b) in enumerate(zip(sim.qx, sim.vx)):
            sim.d.qpos[a] = x[i] - sim.body0[i]
            sim.d.qvel[b] = v[i]
        sim.mj.mj_forward(sim.m, sim.d)
    # an initial settle so the first window starts from a measured rest state
    if init_state is None:
        sim.settle(rec)
    for w in range(N_WIN if windows is None else windows):
        sim.place_pusher(p.pusher_gap[w])
        starts.append(sim.d.time)
        sx.append(sim.state_x()); sv.append(sim.state_v())
        F, T = p.actions[w]
        n0 = len(rec["t"])
        fimp.append(sim.run_window(F, T, rec))
        win_idx.append((n0, len(rec["t"])))
        if init_state is None:
            sim.settle(rec)
    out = {k: np.array(v) for k, v in rec.items()}
    out["starts"] = np.array(starts)
    out["fint"] = np.array(fimp)            # (W, L, N_INT) interval-average edge normal forces
    out["start_x"], out["start_v"] = np.array(sx), np.array(sv)
    out["win_idx"] = np.array(win_idx)
    return out


def grid_truth(tr, w):
    """true x, v (L+1 entities incl. pusher) and edge contact flags on the TAU grid of window w."""
    a, b = tr["win_idx"][w]
    s = tr["starts"][w]
    t = np.r_[s, tr["t"][a:b]]
    x = np.concatenate([tr["start_x"][w][None], tr["x"][a:b]])
    v = np.concatenate([tr["start_v"][w][None], tr["v"][a:b]])
    f = np.concatenate([tr["f"][a:a + 1], tr["f"][a:b]])
    ts = s + TAU
    X = np.stack([np.interp(ts, t, x[:, i]) for i in range(x.shape[1])], 1)   # (N_TAU, L+1)
    V = np.stack([np.interp(ts, t, v[:, i]) for i in range(v.shape[1])], 1)
    Fe = np.stack([np.interp(ts, t, f[:, i]) for i in range(f.shape[1])], 1)
    return X, V, Fe > 0.05


# ----------------------------------------------------------------------------- observation (actor side)
def observe(p: ChainParams, tr: dict, rng: np.random.Generator, cfg: dict | None = None):
    """-> dict of actor-legal arrays for the episode (canonical entity order: pusher, boxes, wall)."""
    cfg = cfg or {}
    L = p.L
    t, X = tr["t"], tr["x"]
    t_end = t[-1]
    cam = CameraTracker(rng, center=(0.08, 0.0, BOX_HZ), cfg=cfg.get("tracker"))
    ft = cam.frame_times(0.0, t_end, rng)
    bx = np.stack([np.interp(ft, t, X[:, 1 + i]) for i in range(L)], 1)                  # (Tf, L) true box x
    P3 = np.stack([bx, np.zeros_like(bx), np.full_like(bx, 2 * BOX_HZ)], -1)            # top-centre keypoints
    u, v, dpt = cam.observe(P3, rng)
    trk = cam.backproject(u, v, dpt)[..., 0]                                            # (Tf, L) actor x (NaN)
    # sparse golden anchors
    anc_t, anc_x, anc_i = [], [], []
    for w, s in enumerate(tr["starts"]):
        for i in range(L):
            n = rng.poisson(1.2)
            ts = list(s + rng.uniform(0, WIN, n))
            if rng.random() < 0.5:
                ts.append(s - rng.uniform(0.0, 0.08))
            for tt in ts:
                anc_t.append(tt); anc_i.append(i)
                anc_x.append(np.interp(tt, t, X[:, 1 + i]) + rng.normal(0, SIG_ANCHOR))
    anc_t, anc_x, anc_i = np.array(anc_t), np.array(anc_x), np.array(anc_i, int)
    # proprio / force at 100 Hz (the record grid)
    enc = X[:, 0] + rng.normal(0, SIG_ENC, len(t))
    fbias = rng.normal(0, SIG_FB)
    fmeas = tr["f"][:, 0] + fbias + rng.normal(0, SIG_F, len(t))
    size_hat = p.half + rng.normal(0, 0.001, L)
    enc_start = tr["start_x"][:, 0] + rng.normal(0, SIG_ENC, len(tr["starts"]))
    # spring mount (fixture with known rest length): map anchor of the spring rest position, sd 1 mm
    x_rest_map = float(box_x0(p)[-1] + rng.normal(0, 0.001))
    return dict(enc_start=enc_start, x_rest_map=x_rest_map, u=tr["u"], ft=ft, trk=trk, anc_t=anc_t, anc_x=anc_x, anc_i=anc_i, t=t, enc=enc, fmeas=fmeas,
                size_hat=size_hat, cam_occ=cam.occ is not None)


def _bin(ts, xs, t0, n=T_EV, span=WIN):
    """irregular samples -> (n,) bin means + valid flags over [t0, t0 + span)."""
    b = np.floor((np.asarray(ts) - t0) / span * n).astype(int)
    ok = (b >= 0) & (b < n) & np.isfinite(xs)
    s = np.zeros(n); c = np.zeros(n)
    np.add.at(s, b[ok], np.asarray(xs)[ok]); np.add.at(c, b[ok], 1)
    return np.where(c > 0, s / np.maximum(c, 1), 0.0), (c > 0).astype(float)


def window_start_estimate(obs, i, ts):
    """actor estimate of box i position just before ts: fresh anchor (<=0.1 s) else tracker median over 0.25 s.
    -> (x_hat, provenance index, age)."""
    m = (obs["anc_i"] == i) & (obs["anc_t"] <= ts) & (obs["anc_t"] >= ts - 0.1)
    if m.any():
        j = np.argmax(np.where(m, obs["anc_t"], -1e9))
        return obs["anc_x"][j], 0, ts - obs["anc_t"][j]
    ft, x = obs["ft"], obs["trk"][:, i]
    w = (ft <= ts) & (ft >= ts - 0.25) & np.isfinite(x)
    if w.sum() >= 2:
        return float(np.median(x[w])), 1, ts - ft[w].max()
    ok = (ft <= ts) & np.isfinite(x)
    if ok.any():
        j = np.nonzero(ok)[0][-1]
        return float(x[j]), 3, ts - ft[j]
    return float("nan"), 3, 10.0


def episode_arrays(p: ChainParams, tr: dict, obs: dict):
    """fixed-size arrays for one episode (padded to L_MAX). Positions are WORLD (sim frame); the tensorizer re-
    expresses them relative to the actor's estimate of box 1 at each window start."""
    L = p.L
    E = N_ENT
    W = N_WIN
    a = {}
    et = np.full(E, -1, np.int8)
    et[0] = ENT_PUSHER; et[1:L + 1] = ENT_BOX
    if p.spring:
        et[L + 1] = ENT_WALL
    a["ent_type"] = et
    size = np.zeros(E, np.float32); size[0] = PUSHER_R; size[1:L + 1] = obs["size_hat"]
    a["size"] = size
    edges = np.zeros((N_EDGE, 3), np.int8); emask = np.zeros(N_EDGE, bool)
    for e in range(L):
        edges[e] = (e, e + 1, EDGE_CONTACT); emask[e] = True
    if p.spring:
        edges[L] = (L, L + 1, EDGE_SPRING); emask[L] = True
    a["edges"], a["edge_mask"] = edges, emask
    # wall / spring rest (actor: map anchor + episode-start tracker estimate of the last box)
    x_rest = obs["x_rest_map"] if p.spring else 0.0
    a["x_rest"] = np.float32(x_rest)
    a["L"] = np.int8(L)
    a["action"] = np.array(p.actions, np.float32)                     # (W, 2) F, T
    xh = np.zeros((W, E), np.float32); prov = np.zeros((W, E, 4), np.float32); age = np.zeros((W, E), np.float32)
    ev = np.zeros((W, E, N_MOD, T_EV, EV_C), np.float16); evm = np.zeros((W, E, N_MOD), bool)
    X_t = np.zeros((W, E, N_TAU), np.float32); V_t = np.zeros((W, E, N_TAU), np.float32)
    C_t = np.zeros((W, N_EDGE, N_TAU), bool); F_t = np.zeros((W, N_EDGE, N_INT), np.float32)
    for w, s in enumerate(tr["starts"]):
        # pusher (known: encoder)
        xh[w, 0] = obs["enc_start"][w]; prov[w, 0, 2] = 1
        for i in range(L):
            x_, pv, ag = window_start_estimate(obs, i, s)
            xh[w, 1 + i] = x_; prov[w, 1 + i, pv] = 1; age[w, 1 + i] = ag
        if p.spring:
            xh[w, L + 1] = x_rest + 0.10; prov[w, L + 1, 2] = 1
        # evidence of this window (used as history by later rows)
        for i in range(L):
            xb, vb = _bin(obs["ft"], obs["trk"][:, i] - xh[w, 1 + i], s)
            ev[w, 1 + i, MOD_TRACK, :, 0] = xb; ev[w, 1 + i, MOD_TRACK, :, 5] = vb
            evm[w, 1 + i, MOD_TRACK] = vb.sum() > 0
            m = obs["anc_i"] == i
            xa, va = _bin(obs["anc_t"][m], obs["anc_x"][m] - xh[w, 1 + i], s)
            ev[w, 1 + i, MOD_ANCHOR, :, 0] = xa; ev[w, 1 + i, MOD_ANCHOR, :, 5] = va
            evm[w, 1 + i, MOD_ANCHOR] = va.sum() > 0
        sel = (obs["t"] >= s) & (obs["t"] < s + WIN)
        ts = obs["t"][sel]
        enc = obs["enc"][sel] - xh[w, 0]
        vel = np.gradient(obs["enc"][sel], REC)
        F, T = p.actions[w]
        ch = [enc, vel, obs["fmeas"][sel], (obs["fmeas"][sel] > 0.6).astype(float), obs["u"][sel]]
        for c, arr in enumerate(ch):
            ev[w, 0, MOD_PROPRIO, :, c] = _bin(ts, arr, s)[0]
        ev[w, 0, MOD_PROPRIO, :, 5] = _bin(ts, enc, s)[1]
        evm[w, 0, MOD_PROPRIO] = True
        # labels (evaluator only)
        Xg, Vg, Cg = grid_truth(tr, w)
        X_t[w, :L + 1] = Xg.T; V_t[w, :L + 1] = Vg.T
        if p.spring:
            X_t[w, L + 1] = xh[w, L + 1]
        C_t[w, :L] = Cg.T
        F_t[w, :L] = tr["fint"][w]
    a.update(xhat=xh, prov=prov, age=age, ev=ev, ev_mask=evm)
    lab = dict(X=X_t, V=V_t, C=C_t, Fint=F_t)
    pm = np.zeros(E, np.float32); pu = np.zeros(E, np.float32); pk = np.zeros(E, np.float32)
    pm[0] = PUSHER_M; pm[1:L + 1] = p.mass; pu[1:L + 1] = p.mu
    if p.spring:
        pk[L] = p.k
    lab.update(mass=pm, mu=pu, k=pk, spring=np.bool_(p.spring))
    return a, lab


def generate_episode(seed: int, L=None, Ls=(1, 2, 3), obs_cfg=None):
    rng = np.random.default_rng(seed)
    p = sample_params(rng, L=L, Ls=Ls)
    tr = simulate(p)
    obs = observe(p, tr, rng, obs_cfg)
    a, lab = episode_arrays(p, tr, obs)
    return p, tr, obs, a, lab


# ----------------------------------------------------------------------------- analytic 1-D chain (numpy)
class AnalyticChain:
    """vectorised over a population P: pusher + L boxes (+ spring on the last box). Penalty contacts, exact
    Coulomb stick/slip per box. Used by the classical estimator, the labelled oracle and the planner baselines."""
    KC, CC, DTA = 1.0e4, 20.0, 0.001

    @staticmethod
    def run(x0, half, mass, mu, k, spring, F, T, x_rest=0.0, v0=None, span=WIN, rec_dt=REC):
        """x0 (L+1,) initial positions (pusher first); half (L,); mass/mu (P,L); k (P,); F/T scalars or (P,).
        -> dict t (n,), x (P,n,L+1), fp (P,n) pusher-B1 contact force."""
        mass, mu = np.atleast_2d(mass), np.atleast_2d(mu)
        P, L = mass.shape
        F = np.broadcast_to(np.asarray(F, float), (P,)); T = np.broadcast_to(np.asarray(T, float), (P,))
        k = np.broadcast_to(np.asarray(k, float), (P,))
        x = np.tile(np.asarray(x0, float), (P, 1))
        v = np.zeros((P, L + 1)) if v0 is None else np.tile(np.asarray(v0, float), (P, 1))
        m_all = np.concatenate([np.full((P, 1), PUSHER_M), mass], 1)
        rad = np.r_[PUSHER_R, half]
        dt = AnalyticChain.DTA
        n = int(round(span / dt)); ev = int(round(rec_dt / dt))
        ts, xs, fps = [0.0], [x.copy()], [np.zeros(P)]
        for s in range(n):
            t = s * dt
            f = np.zeros((P, L + 1))
            fc = np.zeros((P, L))
            for e in range(L):
                pen = rad[e] + rad[e + 1] - (x[:, e + 1] - x[:, e])
                rel = v[:, e] - v[:, e + 1]
                fe = np.where(pen > 0, np.maximum(AnalyticChain.KC * pen + AnalyticChain.CC * rel, 0.0), 0.0)
                fc[:, e] = fe
                f[:, e] -= fe; f[:, e + 1] += fe
            u = cmd_force(t, F, T, v[:, 0])
            f[:, 0] += u - PUSHER_B * v[:, 0]
            if spring:
                f[:, L] += -k * (x[:, L] - x_rest) - SPRING_C * v[:, L]
            # pusher: no friction
            v[:, 0] += f[:, 0] / PUSHER_M * dt
            # boxes: Coulomb
            fb, vb, mb = f[:, 1:], v[:, 1:], m_all[:, 1:]
            fr = mu * mb * G
            moving = np.abs(vb) > 1e-4
            vn = vb + (fb - fr * np.sign(vb)) / mb * dt
            vn = np.where(moving & (np.sign(vn) != np.sign(vb)), 0.0, vn)
            start = ~moving & (np.abs(fb) > fr)
            vs = np.where(start, (fb - fr * np.sign(fb)) / mb * dt, 0.0)
            v[:, 1:] = np.where(moving, vn, vs)
            x += v * dt
            if (s + 1) % ev == 0:
                ts.append((s + 1) * dt); xs.append(x.copy()); fps.append(fc[:, 0].copy())
        return dict(t=np.array(ts), x=np.stack(xs, 1), fp=np.stack(fps, 1))


def analytic_grid(run, P_idx=None):
    """AnalyticChain output -> x on the TAU grid (P, N_TAU, L+1) (tau 0 = initial)."""
    idx = [int(round(tt / REC)) for tt in TAU]
    return run["x"][:, idx]
