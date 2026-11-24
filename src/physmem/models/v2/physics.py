"""Physics-structured features for the v2 outcome head (owner: MODEL-V2; lead request 2026-11-23 ~21:40Z).

Decision-relevant physical quantities computed from (typed-slot latent VALUES in SI) x (named skill params) x
(robot limits from the embodiment vector) x (goal geometry from the scene tokens). Each feature is a signed MARGIN
(positive = good) or a predicted physical quantity; the head derives event logits from these margins (small margin
network) and adds a gated, L2-regularized learned residual. Latent physics still enters ONLY through the typed slots
(these features read the slot values).

  grip margin        log(2 mu_grip F_grip) - log(m sqrt((g+a_lift)^2 + a_lat^2))
  payload/torque     log(min(F_max, 1.3 payload g)) - log(m (g + a))
  push force         log F_max - log(mu m g)              travel = s + v^2 / (2 mu g)  (vs goal distance)
  articulation       log|F| - log F_breakaway ; v = (|F| - 0.3 F_b)/b ; opening = v T (vs goal) ; slam speed
                     position-controlled doors/drawers: log(0.5 F_max) - log(F_b + b v) ; grip vs pull force
  insertion          (clearance + chamfer [+ search radius] - lateral error) / 0.5 mm ; jam ~ sqrt(k) v
  bulk sweep         carry = stroke + 0.6 v^2/(2 mu_b g) vs goal distance ; stall log F_max - log(mu_b M g (1+v/2) depth)
"""
from __future__ import annotations

import math

import torch

from . import spec as S

G = 9.81
K_PHYS = 23
PHYS_NAMES = ["grip_margin", "payload_margin", "log_grip_force", "lift_height", "push_force_margin",
              "push_goal_err", "push_goal_abs", "log_travel", "breakaway_margin", "open_goal_err", "open_goal_abs",
              "slam_speed", "open_grip_margin", "insert_capture", "insert_search", "jam", "sweep_goal_err",
              "sweep_goal_abs", "sweep_stall_margin", "log_mass",
              "crush_margin", "dent_margin", "overweight"]
NK = len(S.SKILLS)
SK = {n: i for i, n in enumerate(S.SKILLS)}
GOAL_T = S.ENT_TYPES.index("goal_region")
POS0 = len(S.ENT_TYPES)


def latents_si(z):
    """normalized typed-slot values (B,L) -> dict of SI tensors (clamped to the physical range |z| <= 4)."""
    z = z.clamp(-4, 4)
    out = {}
    for i, (name, tr, loc, sc, _) in enumerate(S.LATENTS):
        v = z[:, i] * sc + loc
        out[name] = torch.exp(v) if tr == "log" else v
    for k in ("friction", "grip_friction", "bulk_friction"):
        out[k] = out[k].clamp(min=0.02)
    return out


def physics_features(z, call_raw, emb, ent, ent_mask):
    """-> phi (B,K_PHYS), base_cont (B,N_CONT) raw SI physics predictions, base_mask (B,N_CONT) bool."""
    B = z.shape[0]
    lat = latents_si(z)
    o = NK
    P = lambda n: call_raw[:, o + S.P_INDEX[n]]
    H = lambda n: call_raw[:, o + S.N_PARAM + S.P_INDEX[n]] > 0.5
    sk = call_raw[:, :NK]
    is_ = lambda n: sk[:, SK[n]] > 0.5
    fmax = torch.exp(emb[:, 8 + 3]).clamp(min=1.0)
    payload = torch.exp(emb[:, 8 + 1]).clamp(min=0.05)
    vmax = emb[:, 8 + 2].clamp(min=0.05)
    m, mu, mug = lat["mass"], lat["friction"], lat["grip_friction"]
    # goal region (first goal entity), position relative to the target, metres
    gm = (ent[:, :, GOAL_T] > 0.5) & ent_mask
    gi = gm.float().argmax(1)
    goal = ent[torch.arange(B, device=z.device), gi, POS0:POS0 + 3] * 0.1 * gm.any(1, keepdim=True).float()
    gx = goal[:, 0]
    phi = torch.zeros(B, K_PHYS, device=z.device)
    base = torch.zeros(B, S.N_CONT, device=z.device)
    bmask = torch.zeros(B, S.N_CONT, dtype=torch.bool, device=z.device)
    lg = lambda x: torch.log(x.clamp(min=1e-4))

    # ---- pick / lift
    pk = is_("pick").float()
    a = torch.where(H("accel"), P("accel"), P("lift_speed").abs() / 0.25 + 0.5)
    a_lat = torch.where(H("accel") & (P("height") >= 0.05), 6.0 * P("speed").abs().clamp(max=vmax), torch.zeros_like(a))
    F = torch.where(H("grip_force"), P("grip_force").abs(), torch.where(H("suction"), 0.5 * fmax * P("suction").abs(),
                                                                       torch.zeros_like(a)))
    F = torch.minimum(F, 1.2 * fmax)
    need = m * torch.sqrt((G + a) ** 2 + a_lat ** 2)
    phi[:, 0] = pk * (lg(2 * mug * F) - lg(need))
    phi[:, 1] = pk * (lg(torch.minimum(fmax, 1.3 * payload * G)) - lg(m * (G + a)))
    phi[:, 2] = pk * torch.log1p(F) / 4
    phi[:, 3] = pk * (P("height") + P("lift_height")) / 0.1
    # ---- push / slide
    pu = is_("push").float()
    v = P("speed").abs().clamp(max=vmax)
    s = P("dist").abs() + P("distance").abs()
    slide = v ** 2 / (2 * mu * G)
    travel = s + slide
    dirn = torch.where(H("dir"), torch.sign(P("dir")) + (P("dir") == 0).float(), torch.ones_like(v))
    phi[:, 4] = pu * (lg(fmax) - lg(mu * m * G))
    err = torch.where(H("dir"), (dirn * travel - gx) / 0.015, torch.zeros_like(v))
    phi[:, 5] = pu * err
    phi[:, 6] = pu * err.abs()
    phi[:, 7] = pu * lg(travel)
    hd = (is_("push") & H("dir"))
    base[:, 0] = torch.where(hd, dirn * travel, base[:, 0])
    base[:, 6] = torch.where(hd, slide, base[:, 6])
    bmask[:, 0] |= hd
    bmask[:, 6] |= hd
    # ---- open / close (articulated)
    op = (is_("open") | is_("close")).float()
    b_, fb = lat["damping"], lat["breakaway"]
    Fo = P("force")
    synth_open = H("force") & H("duration")
    vss = ((Fo.abs() - 0.3 * fb).clamp(min=0) / b_).clamp(max=vmax)
    opening = torch.sign(Fo) * vss * P("duration")
    need_o = fb + b_ * P("speed").abs()
    phi[:, 8] = op * torch.where(synth_open, lg(Fo.abs()) - lg(fb), lg(0.5 * fmax) - lg(need_o))
    oerr = torch.where(synth_open, (opening - gx) / 0.02, torch.zeros_like(v))
    phi[:, 9] = op * oerr
    phi[:, 10] = op * oerr.abs()
    phi[:, 11] = op * torch.where(synth_open, vss / 0.25, P("speed").abs() / 0.25)
    phi[:, 12] = op * torch.where(H("grip_force"), lg(2 * mug * P("grip_force").abs()) - lg(need_o), torch.zeros_like(v))
    so = (is_("open") | is_("close")) & synth_open
    base[:, 0] = torch.where(so, opening, base[:, 0])
    base[:, 6] = torch.where(so, vss, base[:, 6])
    bmask[:, 0] |= so
    bmask[:, 6] |= so
    # ---- insert
    ins = is_("insert").float()
    c, k = lat["clearance"], lat["stiffness"]
    synth_ins = H("aim_x")
    e = torch.sqrt((lat["offset_x"] - P("aim_x")) ** 2 + (lat["offset_y"] - P("aim_y")) ** 2 + 1e-12)
    cap = c + 0.0012 * (1 - 0.6 * (v / 0.05).clamp(max=1))
    phi[:, 13] = ins * torch.where(synth_ins, (cap - e) / 5e-4, lg(c / 1e-3))
    phi[:, 14] = ins * torch.where(synth_ins, (cap + P("search_r").abs() - e) / 5e-4,
                                   lg((P("search_r").abs() + 5e-4) / (c + 1e-4)))
    phi[:, 15] = ins * (lg(torch.sqrt(k) * v * 20 / 45) + torch.where(H("compliance"), -P("compliance"), torch.zeros_like(v)))
    # ---- sweep (bulk)
    sw = is_("sweep").float()
    mub = lat["bulk_friction"]
    stroke = torch.where(H("stroke"), P("stroke").abs(), (P("x1") - P("x0")).abs())
    carry = stroke + 0.6 * v ** 2 / (2 * mub * G)
    serr = (carry - gx.abs() - 0.03) / 0.05
    phi[:, 16] = sw * serr
    phi[:, 17] = sw * serr.abs()
    depth = torch.where(H("depth"), P("depth").abs(), torch.ones_like(v))
    phi[:, 18] = sw * (lg(fmax) - lg(mub * m * G * (1 + 0.5 * v) * depth))
    phi[:, 19] = lg(m)
    # ---- r5: part force limits and the depalletize exception lane
    crush = lat["crush"]
    gf = torch.where(H("grip_force"), P("grip_force").abs(), torch.zeros_like(v))
    grips = (is_("pick") | is_("open") | is_("close")) & H("grip_force")
    phi[:, 20] = torch.where(grips, lg(crush) - lg(gf.clamp(min=0.5)), torch.zeros_like(v))      # > 0: squeeze below limit
    phi[:, 21] = torch.where(is_("insert") & H("down_force"), lg(crush) - lg(P("down_force").abs().clamp(min=0.5)),
                             torch.zeros_like(v))
    phi[:, 22] = torch.where(is_("handover"), lg(m * G) - lg(payload * G), torch.zeros_like(v))  # > 0: overweight -> flag
    return phi.clamp(-10, 10), base, bmask
