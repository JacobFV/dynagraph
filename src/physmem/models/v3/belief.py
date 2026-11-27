"""Looped typed belief-state core (V3 section 0; owner: V3-MODEL).

The PI's thesis, made explicit: the loop state IS an editable variable store. Key = (entity, property class),
value = a named physical belief (Gaussian mean + log-variance, or a contact logit / force). Each refinement pass k
  1. READS the store (every named value is re-embedded together with fixed known inputs and an opaque entity id),
  2. UPDATES it with graph-masked attention over [entity tokens, relation (edge) tokens, evidence tokens],
  3. DECODES proposals for every named belief and WRITES them back with a learned, uncertainty-aware convex update
     in natural parameters (recency-normalised: lambda' = (1-w) lambda + w lambda_prop, eta likewise),
  4. re-applies anchors (known/golden values) and interventions (clamped beliefs) under availability masks.
Pass k+1 reads only the written store: there is no hidden-state channel between passes (`LBS`). Refinement depth
k is separate from physical time t: the store holds beliefs on the physical-time grid TAU for the queried action
(the candidate action enters the loop as the pusher's known command; the horizon is the TAU grid).

Property classes in the store (per entity): log mass, Coulomb friction, spring stiffness (parameters), pose x(tau)
and velocity v(tau) for every tau in TAU (states), each with uncertainty; per relation: contact logit c(tau) and
interval-average normal force f(tau) >= 0. Provenance of the current pose (anchor / tracker / known / stale + age)
is a known input of the entity token.

Variants
  LBS        : per-pass decode -> update -> write back; law regularisation on EVERY pass (`law_w` > 0)
  LBS-nolaw  : the same with law_w = 0
  V1         : the v1/Inkling design: typed estimates decoded ONCE before the loop, the loop refines hidden tokens
               without writeback, the readout reuses the pre-loop typed estimates; law loss on final outputs only.
  dense      : LBS with dense attention (relation ablation: no graph mask)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import chain as C

# normalisation (documented constants, identical for every variant)
X_SC, V_SC, F_SC = 0.05, 0.5, 5.0
LM0, LM_SC = math.log(0.447), 0.6
MU0, MU_SC = 0.375, 0.15
LK0, LK_SC = math.log(47.4), 0.7
NT, NI = C.N_TAU, C.N_INT
P_FIELDS = ["log_mass", "friction", "log_k"]
NP = 3
EFD = 2 * NP + 4 * NT          # entity field dim: param mean/lv, x mean/lv, v mean/lv
GFD = NT + NI                  # edge field dim: contact logits, forces
ID_D = 16
TF_LV = math.log(0.01 ** 2)


def p_to_si(pm):
    """normalised param means (...,3) -> SI dict."""
    return dict(mass=torch.exp(pm[..., 0] * LM_SC + LM0), mu=(pm[..., 1] * MU_SC + MU0).clamp(min=0.01),
                k=torch.exp(pm[..., 2] * LK_SC + LK0))


def si_to_p(mass=None, mu=None, k=None):
    out = []
    for v, f in ((mass, lambda m: (np.log(np.maximum(m, 1e-6)) - LM0) / LM_SC), (mu, lambda u: (u - MU0) / MU_SC),
                 (k, lambda k_: (np.log(np.maximum(k_, 1e-6)) - LK0) / LK_SC)):
        out.append(None if v is None else f(v))
    return out


# ----------------------------------------------------------------------------- data (GPU-resident)
@dataclass
class Batch:
    ent_type: torch.Tensor   # (B,E) long, -1 pad
    size: torch.Tensor       # (B,E) m
    prov: torch.Tensor       # (B,E,4)
    age: torch.Tensor        # (B,E)
    xhat: torch.Tensor       # (B,E) m, relative to the pusher encoder at window start
    action: torch.Tensor     # (B,2) F, T of the queried window
    edges: torch.Tensor      # (B,Ed,3) long
    edge_mask: torch.Tensor  # (B,Ed) bool
    x_rest: torch.Tensor     # (B,) m, rel
    ev: torch.Tensor         # (B,Nev,T,C)
    ev_mask: torch.Tensor    # (B,Nev) bool
    ev_ent: torch.Tensor     # (B,Nev) long entity index
    ev_mod: torch.Tensor     # (B,Nev) long modality
    ev_age: torch.Tensor     # (B,Nev) windows ago
    ev_act: torch.Tensor     # (B,Nev,2) action of the evidence window


@dataclass
class Labels:
    X: torch.Tensor; V: torch.Tensor; Cn: torch.Tensor; Fint: torch.Tensor
    mass: torch.Tensor; mu: torch.Tensor; k: torch.Tensor; spring: torch.Tensor; k_row: torch.Tensor


class ChainStore:
    """npz from physmem.v3_chain gen -> device tensors; gather(rows) -> (Batch, Labels). rows index (episode, k)."""

    def __init__(self, path, device, max_episodes=None):
        z = np.load(path)
        n = len(z["L"]) if max_episodes is None else min(max_episodes, len(z["L"]))
        t = lambda k, dt=torch.float32: torch.as_tensor(np.asarray(z[k][:n]), device=device, dtype=dt)
        self.n_ep = n
        self.device = device
        self.L = t("L", torch.long)
        self.ent_type = t("ent_type", torch.long)
        self.size = t("size")
        self.edges = t("edges", torch.long)
        self.edge_mask = t("edge_mask", torch.bool)
        self.x_rest_w = t("x_rest")
        self.action = t("action")
        xh = np.asarray(z["xhat"][:n], np.float64)
        prov = np.asarray(z["prov"][:n], np.float32).copy()
        age = np.asarray(z["age"][:n], np.float32).copy()
        bad = ~np.isfinite(xh)
        prov[bad] = 0; prov[..., 3][bad] = 1; age[bad] = 10.0
        xh = np.where(bad, np.nan, xh)
        self.xhat_w = torch.as_tensor(xh, device=device, dtype=torch.float32)       # world, NaN = unknown
        self.prov = torch.as_tensor(prov, device=device)
        self.age = torch.as_tensor(age, device=device)
        self.ev = t("ev", torch.float16)
        self.ev_mask = t("ev_mask", torch.bool)
        self.X = t("lab_X"); self.V = t("lab_V"); self.Cn = t("lab_C", torch.bool); self.Fint = t("lab_Fint")
        self.mass = t("lab_mass"); self.mu = t("lab_mu"); self.k = t("lab_k"); self.spring = t("lab_spring", torch.bool)
        self.extra = {k: z[k][:n] for k in z.files if k.startswith("eval_")}
        cfp = path[:-4] + "_cf.npz"
        self.has_cf = os.path.exists(cfp)
        if self.has_cf:
            c = np.load(cfp)
            tt = lambda k, dt=torch.float32: torch.as_tensor(np.asarray(c[k][:n]), device=device, dtype=dt)
            self.cf_X, self.cf_V = tt("cf_X"), tt("cf_V")
            self.cf_ent, self.cf_prop, self.cf_val = tt("cf_ent", torch.long), tt("cf_prop", torch.long), tt("cf_val")

    def gather_cf(self, rows, b, lab, frac, gen=None):
        """-> (row mask, edited TRUE normalised params (B,E,3), X_cf rel origin (B,E,NT), V_cf)."""
        e, k = rows[:, 0], rows[:, 1]
        B = len(e)
        sel = torch.rand(B, device=e.device, generator=gen) < frac
        pm_t = label_targets(b, lab)[0].clone()
        ent, pr = self.cf_ent[e, k], self.cf_prop[e, k]
        pm_t[torch.arange(B, device=e.device), ent, pr] = self.cf_val[e, k]
        org = self.xhat_w[e, k, 0]
        return sel, pm_t, self.cf_X[e, k] - org[:, None, None], self.cf_V[e, k]

    def rows(self, ks=(0, 1, 2, 3)):
        e = torch.arange(self.n_ep, device=self.device)
        return torch.cat([torch.stack([e, torch.full_like(e, k)], 1) for k in ks])

    def gather(self, rows, drop=None):
        """rows (B,2) long [episode, window k]. drop: dict of availability manipulations (eval protocols)."""
        drop = drop or {}
        e, k = rows[:, 0], rows[:, 1]
        B = len(e)
        E, W, M = C.N_ENT, C.N_WIN, C.N_MOD
        org = self.xhat_w[e, k, 0]                                    # pusher encoder at window start
        xh = self.xhat_w[e, k] - org[:, None]
        prov, age = self.prov[e, k].clone(), self.age[e, k].clone()
        unk = ~torch.isfinite(xh)
        xh = torch.nan_to_num(xh, 0.0)
        # evidence of windows w < k  -> tokens (B, (W-1)*E*M)
        Wh = W - 1
        widx = (k[:, None] - 1 - torch.arange(Wh, device=e.device)[None])          # most recent first
        okw = widx >= 0
        wsafe = widx.clamp(min=0)
        ev = self.ev[e[:, None], wsafe]                                                # (B,Wh,E,M,T,C)
        evm = self.ev_mask[e[:, None], wsafe] & okw[..., None, None]
        if drop.get("no_anchor"):
            evm[..., C.MOD_ANCHOR] = False
        if drop.get("no_track"):
            evm[..., C.MOD_TRACK] = False
        if drop.get("no_proprio"):
            evm[..., C.MOD_PROPRIO] = False
        if drop.get("drop_last_window"):
            evm[:, 0] = False
        Tn, Cc = ev.shape[-2:]
        ev_ent = torch.arange(E, device=e.device)[None, None, :, None].expand(B, Wh, E, M)
        ev_mod = torch.arange(M, device=e.device)[None, None, None, :].expand(B, Wh, E, M)
        ev_age = (torch.arange(Wh, device=e.device) + 1)[None, :, None, None].expand(B, Wh, E, M)
        ev_act = self.action[e[:, None], wsafe][:, :, None, None, :].expand(B, Wh, E, M, 2)
        if drop.get("no_anchor"):
            # window-start pose estimates that came from an anchor fall back to "stale" (tracker value unavailable
            # offline here; we keep the value but mark it stale with a large age: conservative)
            was = prov[..., 0] > 0.5
            prov[..., 0][was] = 0; prov[..., 3][was] = 1; age[was] = 1.0
        prov[..., 3] = torch.where(unk, torch.ones_like(prov[..., 3]), prov[..., 3])
        b = Batch(ent_type=self.ent_type[e], size=self.size[e], prov=prov, age=age, xhat=xh,
                  action=self.action[e, k], edges=self.edges[e], edge_mask=self.edge_mask[e],
                  x_rest=self.x_rest_w[e] - org,
                  ev=ev.reshape(B, -1, Tn, Cc).float(), ev_mask=evm.reshape(B, -1),
                  ev_ent=ev_ent.reshape(B, -1), ev_mod=ev_mod.reshape(B, -1), ev_age=ev_age.reshape(B, -1),
                  ev_act=ev_act.reshape(B, -1, 2))
        X = self.X[e, k] - org[:, None, None]
        lab = Labels(X=X, V=self.V[e, k], Cn=self.Cn[e, k], Fint=self.Fint[e, k], mass=self.mass[e], mu=self.mu[e],
                     k=self.k[e], spring=self.spring[e], k_row=k)
        return b, lab


# ----------------------------------------------------------------------------- modules
def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class EvidenceEncoder(nn.Module):
    """one (entity, modality, window) evidence series (T, EV_C) -> token."""

    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(C.EV_C, 64, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.GELU(),
                                  nn.Conv1d(96, d, 5, stride=2, padding=2), nn.GELU())
        self.mod = nn.Embedding(C.N_MOD, d)
        self.age = nn.Embedding(C.N_WIN + 1, d)
        self.act = nn.Linear(2, d)
        self.out = mlp(3 * d, d, d)
        self.register_buffer("sc", torch.tensor([1 / X_SC, 1 / V_SC, 1 / F_SC, 1.0, 1 / F_SC, 1.0]))

    def forward(self, b: Batch):
        B, N, T, Cc = b.ev.shape
        x = (b.ev * self.sc).clamp(-50, 50).reshape(B * N, T, Cc).transpose(1, 2)
        h = self.conv(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], -1).reshape(B, N, -1)
        act = self.act(b.ev_act / torch.tensor([15.0, 0.3], device=b.ev.device))
        return self.out(torch.cat([h, act], -1)) + self.mod(b.ev_mod) + self.age(b.ev_age)


class CoreLayer(nn.Module):
    """state tokens attend (graph-masked) to [state tokens, evidence tokens]; pre-norm."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.n1, self.n2, self.nk = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = mlp(d, 2 * d, d)
        self.heads = heads

    def forward(self, s, ev, mask):
        kv = torch.cat([self.n1(s), self.nk(ev)], 1)
        am = (~mask).repeat_interleave(self.heads, 0)              # True = blocked
        a, _ = self.att(self.n1(s), kv, kv, attn_mask=am, need_weights=False)
        s = s + a
        return s + self.ff(self.n2(s))


def graph_mask(b: Batch, dense=False, block_edges=None):
    """(B, E+Ed, E+Ed+Nev) bool attention mask (True = may attend).
    entity i: itself, incident edges, own evidence; edge (s,d): itself, s, d, evidence of s and d."""
    B, E = b.ent_type.shape
    Ed = b.edges.shape[1]
    Nev = b.ev.shape[1]
    dv = b.ent_type.device
    ent_ok = b.ent_type >= 0
    em = b.edge_mask.clone()
    if block_edges is not None:
        em = em & ~block_edges
    src, dst = b.edges[..., 0], b.edges[..., 1]
    oh_s = F.one_hot(src, E).bool() & em[..., None]                 # (B,Ed,E)
    oh_d = F.one_hot(dst, E).bool() & em[..., None]
    inc = oh_s | oh_d                                                 # edge-entity incidence
    ev_own = F.one_hot(b.ev_ent, E).bool().transpose(1, 2) & b.ev_mask[:, None]   # (B,E,Nev)
    eye_e = torch.eye(E, dtype=torch.bool, device=dv)[None].expand(B, E, E)
    eye_g = torch.eye(Ed, dtype=torch.bool, device=dv)[None].expand(B, Ed, Ed)
    if dense:
        st_ok = torch.cat([ent_ok, em], 1)
        m = torch.cat([st_ok[:, None, :].expand(B, E + Ed, E + Ed), b.ev_mask[:, None].expand(B, E + Ed, Nev)], 2)
        m = m | torch.eye(E + Ed + Nev, dtype=torch.bool, device=dv)[None, :E + Ed]
        return m
    ent_rows = torch.cat([eye_e, inc.transpose(1, 2), ev_own], 2)                # (B,E,E+Ed+Nev)
    edge_ev = (inc.float() @ ev_own.float()) > 0                                 # (B,Ed,Nev)
    edge_rows = torch.cat([inc, eye_g, edge_ev], 2)
    return torch.cat([ent_rows, edge_rows], 1)


class Store:
    """named belief store. ent (B,E,EFD) = [pm(3), plv(3), xm(NT), xlv(NT), vm(NT), vlv(NT)]; edge (B,Ed,GFD)."""

    def __init__(self, ent, edge):
        self.ent, self.edge = ent, edge

    def pm(self): return self.ent[..., 0:NP]
    def plv(self): return self.ent[..., NP:2 * NP]
    def xm(self): return self.ent[..., 2 * NP:2 * NP + NT]
    def xlv(self): return self.ent[..., 2 * NP + NT:2 * NP + 2 * NT]
    def vm(self): return self.ent[..., 2 * NP + 2 * NT:2 * NP + 3 * NT]
    def vlv(self): return self.ent[..., 2 * NP + 3 * NT:]
    def cl(self): return self.edge[..., :NT]
    def f(self): return self.edge[..., NT:]

    def means(self):
        return torch.cat([self.pm(), self.xm(), self.vm()], -1)

    def lvs(self):
        return torch.cat([self.plv(), self.xlv(), self.vlv()], -1)

    @staticmethod
    def from_parts(mean, lv, edge):
        pm, xm, vm = mean[..., :NP], mean[..., NP:NP + NT], mean[..., NP + NT:]
        plv, xlv, vlv = lv[..., :NP], lv[..., NP:NP + NT], lv[..., NP + NT:]
        return Store(torch.cat([pm, plv, xm, xlv, vm, vlv], -1), edge)


PROV_LV = torch.tensor([math.log((3e-4 / X_SC) ** 2), math.log((3e-3 / X_SC) ** 2), math.log((1e-4 / X_SC) ** 2),
                        math.log((2e-2 / X_SC) ** 2)])


def init_store(b: Batch):
    """pass-0 store: parameter prior (0, 0) normalised; pose = current estimate at every tau with provenance
    variance; velocity 0 (+-1 normalised); contacts unknown (logit 0), forces 0."""
    B, E = b.ent_type.shape
    Ed = b.edges.shape[1]
    dv = b.xhat.device
    pm = torch.zeros(B, E, NP, device=dv); plv = torch.zeros(B, E, NP, device=dv)
    x0 = (b.xhat / X_SC)[..., None].expand(B, E, NT)
    lv0 = (b.prov @ PROV_LV.to(dv))[..., None] + torch.log1p(b.age)[..., None]
    xlv = torch.cat([lv0, torch.zeros(B, E, NT - 1, device=dv)], -1)
    vm = torch.zeros(B, E, NT, device=dv); vlv = torch.zeros(B, E, NT, device=dv)
    ent = torch.cat([pm, plv, x0, xlv, vm, vlv], -1)
    edge = torch.zeros(B, Ed, GFD, device=dv)
    return Store(ent, edge)


def anchor_mask(b: Batch):
    """fields fixed by known anchors: pusher pose/vel at tau0 (encoder, at rest after re-placement); wall pose and
    velocity at every tau (map anchor); pusher mass is a known constant (not a store field)."""
    B, E = b.ent_type.shape
    dv = b.xhat.device
    mean_mask = torch.zeros(B, E, NP + 2 * NT, dtype=torch.bool, device=dv)
    val = torch.zeros(B, E, NP + 2 * NT, device=dv)
    pus = b.ent_type == C.ENT_PUSHER
    wall = b.ent_type == C.ENT_WALL
    mean_mask[..., NP] |= pus; mean_mask[..., NP + NT] |= pus
    val[..., NP] = torch.where(pus, b.xhat / X_SC, val[..., NP])
    mean_mask[..., NP:NP + NT] |= wall[..., None]
    mean_mask[..., NP + NT:] |= wall[..., None]
    val[..., NP:NP + NT] = torch.where(wall[..., None], (b.xhat / X_SC)[..., None].expand(-1, -1, NT),
                                       val[..., NP:NP + NT])
    return mean_mask, val


def apply_clamps(store: Store, masks):
    """masks: list of (mean_mask (B,E,NP+2NT) bool, values) ; clamped fields get the value and a tiny variance."""
    mean, lv = store.means(), store.lvs()
    for mm, val in masks:
        mean = torch.where(mm, val, mean)
        lv = torch.where(mm, torch.full_like(lv, TF_LV), lv)
    return Store.from_parts(mean, lv, store.edge)


class Embedder(nn.Module):
    def __init__(self, d):
        super().__init__()
        kin = 3 + 1 + 4 + 1 + 1 + 1 + 2 + 1       # type, size, prov, age, xhat, spring-box, action (pusher), valid
        self.ent = mlp(EFD + kin + ID_D, 2 * d, d)
        self.edge = mlp(GFD + 2 + 1 + 1 + 2 * ID_D, 2 * d, d)

    def known(self, b: Batch):
        B, E = b.ent_type.shape
        t = F.one_hot(b.ent_type.clamp(min=0), 3).float() * (b.ent_type >= 0)[..., None]
        is_sb = torch.zeros(B, E, device=b.xhat.device)
        sp = b.edge_mask & (b.edges[..., 2] == C.EDGE_SPRING)
        is_sb.scatter_add_(1, b.edges[..., 0], sp.float())
        act = torch.where((b.ent_type == C.ENT_PUSHER)[..., None],
                          (b.action / torch.tensor([15.0, 0.3], device=b.xhat.device))[:, None], 0.0 * b.action[:, None])
        return torch.cat([t, (b.size / 0.03)[..., None], b.prov, torch.log1p(b.age)[..., None],
                          (b.xhat / X_SC)[..., None], is_sb[..., None], act, (b.ent_type >= 0).float()[..., None]], -1)

    def forward(self, store: Store, b: Batch, ids):
        ent_in = torch.cat([store.ent.clamp(-30, 30), self.known(b), ids], -1)
        src, dst = b.edges[..., 0], b.edges[..., 1]
        gi = lambda x, i: torch.gather(x, 1, i[..., None].expand(-1, -1, x.shape[-1]))
        et = F.one_hot(b.edges[..., 2].clamp(0, 1), 2).float()
        rest = (gi(b.size[..., None], src) + gi(b.size[..., None], dst)) / 0.03
        xr = torch.where(b.edges[..., 2:3] == C.EDGE_SPRING, (b.x_rest / X_SC)[:, None, None], 0.0 * rest)
        edge_in = torch.cat([store.edge.clamp(-30, 30), et, rest, xr, gi(ids, src), gi(ids, dst)], -1)
        return self.ent(ent_in), self.edge(edge_in)


class LBS(nn.Module):
    """looped belief store (variants 'lbs' and 'dense'); see module docstring."""

    def __init__(self, d=128, layers=2, dense=False, max_passes=12):
        super().__init__()
        self.kind = "dense" if dense else "lbs"
        self.dense = dense
        self.evenc = EvidenceEncoder(d)
        self.ev_id = nn.Linear(ID_D, d)
        self.emb = Embedder(d)
        self.pass_emb = nn.Embedding(max_passes, d)
        self.core = nn.ModuleList([CoreLayer(d) for _ in range(layers)])
        self.dec_ent = mlp(d, 2 * d, 2 * (NP + 2 * NT) + (NP + 2 * NT))
        self.dec_edge = mlp(d, 2 * d, GFD + 2)
        nn.init.zeros_(self.dec_ent[-1].bias)

    def evidence(self, b, ids):
        gi = torch.gather(ids, 1, b.ev_ent[..., None].expand(-1, -1, ids.shape[-1]))
        return self.evenc(b) + self.ev_id(gi)

    def step(self, store, b, ids, evtok, mask, k):
        se, sg = self.emb(store, b, ids)
        s = torch.cat([se, sg], 1) + self.pass_emb.weight[min(k, self.pass_emb.num_embeddings - 1)]
        for layer in self.core:
            s = layer(s, evtok, mask)
        E = se.shape[1]
        de, dg = self.dec_ent(s[:, :E]).float(), self.dec_edge(s[:, E:]).float()
        nf = NP + 2 * NT
        pmean, plv, gate = de[..., :nf], de[..., nf:2 * nf].clamp(-10, 4), de[..., 2 * nf:]
        w = torch.sigmoid(gate)
        mean, lv = store.means(), store.lvs()
        lam, lamp = torch.exp(-lv), torch.exp(-plv)
        lam_n = (1 - w) * lam + w * lamp
        mean_n = ((1 - w) * lam * mean + w * lamp * pmean) / lam_n
        lv_n = -torch.log(lam_n)
        wc, wf = torch.sigmoid(dg[..., GFD:GFD + 1]), torch.sigmoid(dg[..., GFD + 1:GFD + 2])
        cl = (1 - wc) * store.cl() + wc * dg[..., :NT]
        f = (1 - wf) * store.f() + wf * F.softplus(dg[..., NT:GFD])
        return Store.from_parts(mean_n, lv_n, torch.cat([cl, f], -1))

    proj_steps = 0

    def enable_projection(self, steps=4):
        """EXPLORATORY (V3 §0d): structural law projection inside the loop. After every write-back, `steps` learned-
        step gradient steps on the law residual move the STATE beliefs (x, v, contact force) given the current
        PARAMETER beliefs, so a parameter edit has to propagate through physics, not through a loss."""
        self.proj_steps = steps
        self.log_eta = nn.Parameter(torch.full((3,), math.log(math.expm1(0.3))))

    def project(self, store: Store, b: Batch, cl):
        train = self.training and torch.is_grad_enabled()
        with torch.enable_grad():
            xm, vm, f = store.xm(), store.vm(), store.f()
            pm = store.pm()
            if not train:
                xm, vm, f, pm = (t.detach().requires_grad_() for t in (xm, vm, f, pm))
            eta = F.softplus(self.log_eta)
            for _ in range(self.proj_steps):
                st = Store.from_parts(torch.cat([pm, xm, vm], -1), store.lvs(), torch.cat([store.cl(), f], -1))
                r = sum(store_law(st, b, per_row=True).values()).sum()
                gx, gv, gf = torch.autograd.grad(r, [xm, vm, f], create_graph=train)
                xm = xm - (eta[0] * gx).clamp(-1, 1)
                vm = vm - (eta[1] * gv).clamp(-1, 1)
                f = F.relu(f - (eta[2] * gf).clamp(-1, 1))
            if not train:
                xm, vm, f = xm.detach(), vm.detach(), f.detach()
        out = Store.from_parts(torch.cat([store.pm(), xm, vm], -1), store.lvs(), torch.cat([store.cl(), f], -1))
        return apply_clamps(out, cl)

    def forward(self, b: Batch, passes=4, ids=None, clamps=(), block_edges=None, late_ev=None, late_from=None):
        """-> list of Store, one per pass (index -1 = final). clamps: [(mean_mask, values)] interventions.
        late_ev: (B,Nev) bool evidence tokens that become AVAILABLE only from pass `late_from` (delayed evidence)."""
        B, E = b.ent_type.shape
        if ids is None:
            ids = torch.randn(B, E, ID_D, device=b.xhat.device)
        evtok = self.evidence(b, ids)
        am = anchor_mask(b)
        cl = [am] + list(clamps)
        store = apply_clamps(init_store(b), cl)
        out = []
        b0 = b
        if late_ev is not None:
            b0 = Batch(**{**b.__dict__, "ev_mask": b.ev_mask & ~late_ev})
        mask0 = graph_mask(b0, self.dense, block_edges)
        mask1 = graph_mask(b, self.dense, block_edges) if late_ev is not None else mask0
        for k in range(passes):
            mask = mask1 if (late_ev is not None and k >= late_from) else mask0
            store = self.step(store, b, ids, evtok, mask, k)
            store = apply_clamps(store, cl)
            if self.proj_steps:
                store = self.project(store, b, cl)
            out.append(store)
        return out


class V1(nn.Module):
    """v1/Inkling-style baseline: typed estimates decoded once before the loop (no writeback); hidden-token loop;
    readout from final hidden tokens + pre-loop typed estimates."""
    kind = "v1"

    def __init__(self, d=128, layers=2, max_passes=12):
        super().__init__()
        self.evenc = EvidenceEncoder(d)
        self.ev_id = nn.Linear(ID_D, d)
        self.emb = Embedder(d)
        self.pass_emb = nn.Embedding(max_passes, d)
        self.enc = nn.ModuleList([CoreLayer(d) for _ in range(layers)])
        self.core = nn.ModuleList([CoreLayer(d) for _ in range(layers)])
        self.dec_param = mlp(d, 2 * d, 2 * NP)
        self.wr = nn.Linear(2 * NP, d)
        self.read_ent = mlp(d + 2 * NP, 2 * d, 4 * NT)
        self.read_edge = mlp(d, 2 * d, GFD)

    def forward(self, b: Batch, passes=4, ids=None, clamps=(), block_edges=None, late_ev=None, late_from=None):
        B, E = b.ent_type.shape
        if ids is None:
            ids = torch.randn(B, E, ID_D, device=b.xhat.device)
        gi = torch.gather(ids, 1, b.ev_ent[..., None].expand(-1, -1, ID_D))
        evtok = self.evenc(b) + self.ev_id(gi)
        b0 = b if late_ev is None else Batch(**{**b.__dict__, "ev_mask": b.ev_mask & ~late_ev})
        mask0 = graph_mask(b0, False, block_edges)
        mask1 = graph_mask(b, False, block_edges)
        am = anchor_mask(b)
        st0 = apply_clamps(init_store(b), [am])
        se, sg = self.emb(st0, b, ids)
        s = torch.cat([se, sg], 1)
        for layer in self.enc:                                   # encoder (pre-loop): typed estimates read here
            s = layer(s, evtok, mask0)
        pp = self.dec_param(s[:, :E]).float()
        pm, plv = pp[..., :NP], pp[..., NP:].clamp(-10, 4)
        # interventions act on the typed estimates (as in v1)
        full = torch.cat([pm, torch.zeros(B, E, 2 * NT, device=pm.device)], -1)
        full_lv = torch.cat([plv, torch.zeros(B, E, 2 * NT, device=pm.device)], -1)
        for mm, val in clamps:
            full = torch.where(mm, val, full)
            full_lv = torch.where(mm, torch.full_like(full_lv, TF_LV), full_lv)
        pm, plv = full[..., :NP], full_lv[..., :NP]
        w_ = self.wr(torch.cat([pm, plv], -1))
        s = s + torch.cat([w_, torch.zeros_like(w_[:, :1]).expand(-1, s.shape[1] - E, -1)], 1)
        out = []
        for k in range(passes):
            mask = mask1 if (late_ev is not None and k >= late_from) else mask0
            s = s + self.pass_emb.weight[min(k, self.pass_emb.num_embeddings - 1)]
            for layer in self.core:
                s = layer(s, evtok, mask)
        r = self.read_ent(torch.cat([s[:, :E], pm, plv], -1)).float()
        xm, xlv, vm, vlv = r[..., :NT], r[..., NT:2 * NT].clamp(-10, 4), r[..., 2 * NT:3 * NT], \
            r[..., 3 * NT:].clamp(-10, 4)
        xm = xm + (b.xhat / X_SC)[..., None]                    # residual on the current estimate (as LBS init)
        mean = torch.cat([pm, xm, vm], -1); lv = torch.cat([plv, xlv, vlv], -1)
        g = self.read_edge(s[:, E:]).float()
        st = Store.from_parts(mean, lv, torch.cat([g[..., :NT], F.softplus(g[..., NT:])], -1))
        st = apply_clamps(st, [am] + list(clamps))
        return [st]


def build(kind, d=128):
    """kinds: lbs | dense | v1 | generic (LBS backbone whose parameter fields are an UNNAMED latent: no property
    supervision, no laws, never clamped by name -- the matched generic-memory twin)."""
    if kind == "v1":
        return V1(d)
    if kind == "arm_i":
        return LBSI(d)
    m = LBS(d, dense=(kind == "dense"))
    if kind == "proj":
        m.kind = "proj"
        m.enable_projection(4)
    if kind == "generic":
        m.kind, m.typed = "generic", False
    return m


# ----------------------------------------------------------------------------- losses
def gnll(m, lv, y, mask):
    l = 0.5 * (lv + (y - m) ** 2 * torch.exp(-lv))
    mf = mask.float()
    return (l * mf).sum() / mf.sum().clamp(min=1)


def label_targets(b: Batch, lab: Labels):
    """normalised targets + masks: params (B,E,3), x/v (B,E,NT), contacts (B,Ed,NT)."""
    box = b.ent_type == C.ENT_BOX
    pus = b.ent_type == C.ENT_PUSHER
    pm_t = torch.stack([(torch.log(lab.mass.clamp(min=1e-6)) - LM0) / LM_SC, (lab.mu - MU0) / MU_SC,
                        (torch.log(lab.k.clamp(min=1e-6)) - LK0) / LK_SC], -1)
    pmask = torch.stack([box, box, box & (lab.k > 0)], -1)
    xmask = (box | pus)[..., None].expand(-1, -1, NT)
    cmask = (b.edge_mask & (b.edges[..., 2] == C.EDGE_CONTACT))[..., None].expand(-1, -1, NT)
    return pm_t, pmask, lab.X / X_SC, lab.V / V_SC, xmask, lab.Cn.float(), cmask


def pred_loss(store: Store, b, lab, sample_w=None, typed=True):
    pm_t, pmask, xt, vt, xmask, ct, cmask = label_targets(b, lab)
    if not typed:
        pmask = torch.zeros_like(pmask)
    if sample_w is not None:
        pmask = pmask & sample_w[:, None, None]; xmask = xmask & sample_w[:, None, None]
        cmask = cmask & sample_w[:, None, None]
    l_p = gnll(store.pm(), store.plv(), pm_t, pmask)
    l_x = gnll(store.xm(), store.xlv(), xt, xmask)
    l_v = gnll(store.vm(), store.vlv(), vt, xmask)
    bce = F.binary_cross_entropy_with_logits(store.cl(), ct, reduction="none")
    l_c = (bce * cmask).sum() / cmask.sum().clamp(min=1)
    return l_p + l_x + l_v + 0.5 * l_c, dict(l_p=l_p.item(), l_x=l_x.item(), l_v=l_v.item(), l_c=l_c.item())


def law_residuals(xm_si, vm_si, f_si, mass, mu, k, b: Batch, per_row=False):
    """physics consistency of a decoded trajectory (SI): kinematics, pusher Newton, box Newton with the Coulomb
    friction cone / sliding equality and Hooke spring force, contact non-penetration and complementarity.
      xm_si, vm_si (B,E,NT); f_si (B,Ed,NI) interval-average normal force on each edge; mass/mu/k (B,E).
    Returns dict of mean squared residual terms (dimensionless)."""
    B, E, _ = xm_si.shape
    dv = xm_si.device
    tau = torch.as_tensor(C.TAU, device=dv, dtype=torch.float32)
    dt = (tau[1:] - tau[:-1])[None, None]
    tmid = 0.5 * (tau[1:] + tau[:-1])[None]
    box = (b.ent_type == C.ENT_BOX)
    pus = (b.ent_type == C.ENT_PUSHER)
    mov = box | pus
    x0, x1 = xm_si[..., :-1], xm_si[..., 1:]
    v0, v1 = vm_si[..., :-1], vm_si[..., 1:]
    vmid = 0.5 * (v0 + v1)
    r_kin = ((x1 - x0) - vmid * dt) / 0.01
    # edge forces -> per-entity net (contact edges only)
    con = b.edge_mask & (b.edges[..., 2] == C.EDGE_CONTACT)
    fc = f_si * con[..., None]
    net = torch.zeros(B, E, C.N_INT, device=dv)
    net.scatter_add_(1, b.edges[..., 1:2].expand(-1, -1, C.N_INT), fc)     # pushes dst forward
    net.scatter_add_(1, b.edges[..., 0:1].expand(-1, -1, C.N_INT), -fc)    # pushes src backward
    # spring on the spring box
    sp = b.edge_mask & (b.edges[..., 2] == C.EDGE_SPRING)
    sb = torch.zeros(B, E, device=dv).scatter_add_(1, b.edges[..., 0], sp.float()) > 0
    xmid = 0.5 * (x0 + x1)
    fs = (-k[..., None] * (xmid - b.x_rest[:, None, None]) - C.SPRING_C * vmid) * sb[..., None]
    # pusher
    Fq, Tq = b.action[:, 0:1], b.action[:, 1:2]
    u = C.cmd_force(tmid, Fq, Tq, vmid[:, 0], xp=torch)                       # pusher is entity 0
    vp = vmid[:, 0]
    r_p = (C.PUSHER_M * (v1[:, 0] - v0[:, 0]) / dt[:, 0] - (u - C.PUSHER_B * vp + net[:, 0])) / 3.0
    r_p = r_p * pus[:, 0:1]
    # boxes: required friction force
    m_ = mass.clamp(min=1e-3)[..., None]
    mg = m_ * C.G
    r_f = m_ * (v1 - v0) / dt - (net + fs)
    cone = F.relu(r_f.abs() - mu[..., None] * mg) / mg
    slide = torch.sigmoid((torch.minimum(v0.abs(), v1.abs()) - 0.02) / 0.005) * ((v0 * v1) > 0).float()
    eq = slide * (r_f + mu[..., None] * mg * torch.sign(vmid)) / mg
    # contacts
    src, dst = b.edges[..., 0], b.edges[..., 1]
    gx = lambda i: torch.gather(xm_si, 1, i[..., None].expand(-1, -1, C.N_TAU))
    gs = lambda i: torch.gather(b.size, 1, i)
    gap = gx(dst) - gx(src) - (gs(src) + gs(dst))[..., None]                  # (B,Ed,NT)
    pen = F.relu(-gap) / 0.005 * con[..., None]
    comp = (f_si / F_SC) * F.relu(torch.minimum(gap[..., :-1], gap[..., 1:])) / 0.005 * con[..., None]
    bm = box[..., None].float(); mm = mov[..., None].float()
    def mean(x, w):
        if per_row:
            return ((x ** 2) * w).flatten(1).sum(1) / w.expand_as(x).flatten(1).sum(1).clamp(min=1)
        return ((x ** 2) * w).sum() / w.expand_as(x).sum().clamp(min=1)
    return dict(kin=mean(r_kin, mm), pusher=mean(r_p, pus[:, 0:1].float()), cone=mean(cone, bm),
                slide=mean(eq, bm), pen=mean(pen, con[..., None].float()), comp=mean(comp, con[..., None].float()))


def store_law(store: Store, b: Batch, per_row=False):
    si = p_to_si(store.pm())
    mass = torch.where(b.ent_type == C.ENT_PUSHER, torch.full_like(si["mass"], C.PUSHER_M), si["mass"])
    return law_residuals(store.xm() * X_SC, store.vm() * V_SC, store.f() * F_SC, mass, si["mu"], si["k"], b,
                         per_row=per_row)


def random_param_edit(b: Batch, gen=None, frac=0.25):
    """counterfactual augmentation: for a random fraction of rows clamp one parameter of one random box to a random
    value from the normalised prior. -> (row mask (B,), clamp (mean_mask, values))."""
    B, E = b.ent_type.shape
    dv = b.xhat.device
    rows = torch.rand(B, device=dv, generator=gen) < frac
    box = (b.ent_type == C.ENT_BOX).float() + 1e-6
    ent = torch.multinomial(box, 1, generator=gen)[:, 0]
    sp = b.edge_mask & (b.edges[..., 2] == C.EDGE_SPRING)
    sb = torch.zeros(B, E, device=dv).scatter_add_(1, b.edges[..., 0], sp.float()) > 0
    prop = torch.randint(0, NP, (B,), device=dv, generator=gen)
    has_k = sb[torch.arange(B, device=dv), ent]
    prop = torch.where((prop == 2) & ~has_k, torch.randint(0, 2, (B,), device=dv, generator=gen), prop)
    mm = torch.zeros(B, E, NP + 2 * NT, dtype=torch.bool, device=dv)
    mm[torch.arange(B, device=dv), ent, prop] = rows
    val = torch.zeros(B, E, NP + 2 * NT, device=dv)
    val[torch.arange(B, device=dv), ent, prop] = torch.randn(B, device=dv, generator=gen).clamp(-2, 2)
    return rows, (mm, val)


def total_loss(model, b, lab, passes, law_w=1.0, cf_frac=0.25, ds_w=0.3, gen=None):
    """prediction loss on the final pass (+ ds_w deep supervision on earlier passes, LBS only) and law loss on every
    pass (LBS) / on the final outputs (V1). Counterfactually edited rows get the law loss only."""
    clamps, keep = (), None
    if not getattr(model, "typed", True):
        law_w, cf_frac = 0.0, 0.0
    if law_w > 0 and cf_frac > 0:
        rows, cl = random_param_edit(b, gen, cf_frac)
        clamps, keep = (cl,), ~rows
    outs = model(b, passes=passes, clamps=clamps)
    loss, logs = 0.0, {}
    n = len(outs)
    for i, st in enumerate(outs):
        w = 1.0 if i == n - 1 else ds_w
        lp, lg = pred_loss(st, b, lab, keep, typed=getattr(model, "typed", True))
        loss = loss + w * lp
        if law_w > 0:
            r = store_law(st, b)
            ll = sum(r.values())
            loss = loss + w * law_w * ll
            if i == n - 1:
                logs.update({f"law_{k}": v.item() for k, v in r.items()})
        if i == n - 1:
            logs.update(lg)
    logs["loss"] = float(loss.item())
    return loss, logs


# ----------------------------------------------------------------------------- Arm I: non-bypass learned rollout
class Rollout(nn.Module):
    """LEARNED dynamics on the named store only (no evidence access): from parameter slots + the state at the start
    of the horizon (x, v at tau0) + known inputs (sizes, types, the pusher's known command, relation graph), roll the
    chain forward over the physical-time grid. Per step: graph-masked attention among entity and relation tokens
    -> velocity increments (entities), contact logit + force (relations); positions integrate by construction."""

    def __init__(self, d=96, layers=2):
        super().__init__()
        self.ent = mlp(3 + 2 + 1 + 3 + 1 + 1 + 1 + ID_D, 2 * d, d)
        self.edge = mlp(2 + 1 + 1 + 2 + 2 * ID_D, 2 * d, d)
        self.core = nn.ModuleList([CoreLayer(d) for _ in range(layers)])
        self.dv = mlp(d, d, 2)            # dv (normalised), log-variance increment
        self.de = mlp(d, d, 2)            # contact logit, force
        self.d = d

    def forward(self, pm, x0, v0, b: Batch, ids, block_edges=None):
        """pm (B,E,3) normalised param VALUES; x0, v0 (B,E) normalised. -> xm, vm, xlv, vlv (B,E,NT), cl, f (B,Ed,NT/NI)."""
        B, E = b.ent_type.shape
        Ed = b.edges.shape[1]
        dv_ = b.xhat.device
        mask = graph_mask(b, False, block_edges)[:, :, :E + Ed]
        tau = torch.as_tensor(C.TAU, device=dv_, dtype=torch.float32)
        t_ = F.one_hot(b.ent_type.clamp(min=0), 3).float() * (b.ent_type >= 0)[..., None]
        sp = b.edge_mask & (b.edges[..., 2] == C.EDGE_SPRING)
        is_sb = (torch.zeros(B, E, device=dv_).scatter_add_(1, b.edges[..., 0], sp.float()) > 0).float()
        movable = ((b.ent_type == C.ENT_BOX) | (b.ent_type == C.ENT_PUSHER)).float()
        pus = (b.ent_type == C.ENT_PUSHER).float()
        src, dst = b.edges[..., 0], b.edges[..., 1]
        gi = lambda x, i: torch.gather(x, 1, i[..., None].expand(-1, -1, x.shape[-1]))
        et = F.one_hot(b.edges[..., 2].clamp(0, 1), 2).float()
        rad = b.size / X_SC
        xr = torch.where(b.edges[..., 2] == C.EDGE_SPRING, (b.x_rest / X_SC)[:, None], torch.zeros_like(src).float())
        xs, vs, xl, vl, cls, fs = [x0], [v0], [torch.full_like(x0, -6.0)], [torch.full_like(v0, -6.0)], [], []
        x, v, lvx, lvv = x0, v0, torch.full_like(x0, -6.0), torch.full_like(v0, -6.0)
        for j in range(NI):
            tm = 0.5 * (tau[j] + tau[j + 1])
            u = C.cmd_force(tm, b.action[:, 0], b.action[:, 1], v[:, 0] * V_SC, xp=torch) / 15.0
            ent_in = torch.cat([pm.clamp(-4, 4), x[..., None], v[..., None], (b.size / 0.03)[..., None], t_,
                                is_sb[..., None], (b.ent_type >= 0).float()[..., None], (u[:, None] * pus)[..., None],
                                ids], -1)
            gap = (gi(x[..., None], dst) - gi(x[..., None], src)).squeeze(-1) - (gi(rad[..., None], src) +
                                                                                 gi(rad[..., None], dst)).squeeze(-1)
            rel = (gi(v[..., None], src) - gi(v[..., None], dst)).squeeze(-1)
            edge_in = torch.cat([et, (gap * 5)[..., None], xr[..., None], rel[..., None],
                                 torch.full_like(rel, float(tm))[..., None], gi(ids, src), gi(ids, dst)], -1)
            s = torch.cat([self.ent(ent_in), self.edge(edge_in)], 1)
            for layer in self.core:
                s = layer(s, s[:, :0], mask)
            o = self.dv(s[:, :E]).float()
            g = self.de(s[:, E:]).float()
            v1 = v + o[..., 0] * movable
            x = x + 0.5 * (v + v1) * (tau[j + 1] - tau[j]) * V_SC / X_SC * movable
            v = v1
            lvx = torch.logaddexp(lvx, o[..., 1].clamp(-12, 2)); lvv = torch.logaddexp(lvv, o[..., 1].clamp(-12, 2))
            xs.append(x); vs.append(v); xl.append(lvx); vl.append(lvv); cls.append(g[..., 0])
            fs.append(F.softplus(g[..., 1]))
        cl = torch.stack([cls[0]] + cls, -1)
        return (torch.stack(xs, -1), torch.stack(vs, -1), torch.stack(xl, -1), torch.stack(vl, -1), cl,
                torch.stack(fs, -1))


class LBSI(LBS):
    """Arm I: learned loop (evidence -> parameter slots and the tau0 state) + non-bypass learned Rollout (future
    beliefs computable ONLY from the slots + tau0 state). Trained with teacher forcing and real MuJoCo do()-pairs."""

    def __init__(self, d=128):
        super().__init__(d)
        self.kind = "arm_i"
        self.roll = Rollout()

    def write_future(self, store, b, ids, pm_in=None, block_edges=None, cl=()):
        pm = store.pm() if pm_in is None else pm_in
        xm, vm, xlv, vlv, c_, f_ = self.roll(pm, store.xm()[..., 0], store.vm()[..., 0], b, ids, block_edges)
        out = Store.from_parts(torch.cat([store.pm(), xm, vm], -1), torch.cat([store.plv(), xlv, vlv], -1),
                               torch.cat([c_, f_], -1))
        return apply_clamps(out, cl)

    def loop(self, b, passes, ids, clamps=(), block_edges=None, late_ev=None, late_from=None):
        """the learned loop over evidence: writes parameter slots and the tau0 state only (future fields are reset
        to their uninformative init after every pass, so the loop carries no future-motion channel)."""
        evtok = self.evidence(b, ids)
        cl = [anchor_mask(b)] + list(clamps)
        init = apply_clamps(init_store(b), cl)
        store = init
        b0 = b if late_ev is None else Batch(**{**b.__dict__, "ev_mask": b.ev_mask & ~late_ev})
        mask0 = graph_mask(b0, False, block_edges)
        mask1 = graph_mask(b, False, block_edges)
        outs = []
        for k in range(passes):
            mask = mask1 if (late_ev is not None and k >= late_from) else mask0
            st = apply_clamps(self.step(store, b, ids, evtok, mask, k), cl)
            keep = torch.zeros_like(st.ent, dtype=torch.bool)
            keep[..., :2 * NP] = True
            keep[..., 2 * NP] = True; keep[..., 2 * NP + NT] = True              # x(tau0), xlv(tau0)
            keep[..., 2 * NP + 2 * NT] = True; keep[..., 2 * NP + 3 * NT] = True  # v(tau0), vlv(tau0)
            store = Store(torch.where(keep, st.ent, init.ent), init.edge)
            outs.append(store)
        return outs, cl

    def forward(self, b, passes=4, ids=None, clamps=(), block_edges=None, late_ev=None, late_from=None):
        B, E = b.ent_type.shape
        if ids is None:
            ids = torch.randn(B, E, ID_D, device=b.xhat.device)
        loops, cl = self.loop(b, passes, ids, clamps, block_edges, late_ev, late_from)
        return [self.write_future(st, b, ids, None, block_edges, cl) for st in loops]


def arm_i_loss(model: LBSI, b, lab, passes, tf_prob=0.5, cf=None, law_w=1.0, gen=None):
    """estimator (loop) trained by NLL on parameters + tau0 state (every pass); rollout trained on the final pass with
    sample-form parameter input: TRUE parameters w.p. tf_prob, else the DETACHED loop estimate (v2 recipe). cf =
    (rows mask, true edited params (B,E,3), X_cf, V_cf): those rows use the edited TRUE params and the MuJoCo
    counterfactual trajectory as target (real do()-data)."""
    pm_t, pmask, xt, vt, xmask, ct, cmask = label_targets(b, lab)
    B = len(pm_t)
    dv = pm_t.device
    ids = torch.randn(B, b.ent_type.shape[1], ID_D, device=dv, generator=gen)
    # loop passes (no rollout) -> estimator losses
    tmp, _ = model.loop(b, passes, ids)
    loss = 0.0
    logs = {}
    x0mask = xmask.clone(); x0mask[..., 1:] = False
    for i, st in enumerate(tmp):
        w = 1.0 if i == len(tmp) - 1 else 0.3
        l_p = gnll(st.pm(), st.plv(), pm_t, pmask)
        l_x0 = gnll(st.xm()[..., :1], st.xlv()[..., :1], xt[..., :1], x0mask[..., :1])
        loss = loss + w * (l_p + l_x0)
    est = tmp[-1].pm().detach()
    tf = (torch.rand(B, 1, 1, device=dv, generator=gen) < tf_prob) & pmask
    pm_in = torch.where(tf, pm_t, est)
    X, V, keep = xt, vt, torch.ones(B, dtype=torch.bool, device=dv)
    if cf is not None:
        rows, pm_cf, Xcf, Vcf = cf
        pm_in = torch.where(rows[:, None, None], torch.where(pmask, pm_cf, est), pm_in)
        X = torch.where(rows[:, None, None], Xcf / X_SC, X)
        V = torch.where(rows[:, None, None], Vcf / V_SC, V)
    st0 = tmp[-1]
    st = model.write_future(Store.from_parts(torch.cat([st0.pm().detach(), st0.xm().detach(), st0.vm().detach()], -1),
                                             st0.lvs().detach(), st0.edge.detach()), b, ids, pm_in,
                            cl=[anchor_mask(b)])
    fm = xmask.clone(); fm[..., 0] = False
    l_x = gnll(st.xm(), st.xlv(), X, fm)
    l_v = gnll(st.vm(), st.vlv(), V, fm)
    cm = cmask & (keep[:, None, None] if cf is None else ~cf[0][:, None, None])
    bce = F.binary_cross_entropy_with_logits(st.cl(), ct, reduction="none")
    l_c = (bce * cm).sum() / cm.sum().clamp(min=1)
    loss = loss + l_x + l_v + 0.5 * l_c
    if law_w > 0:
        si = p_to_si(pm_in)
        mass = torch.where(b.ent_type == C.ENT_PUSHER, torch.full_like(si["mass"], C.PUSHER_M), si["mass"])
        r = law_residuals(st.xm() * X_SC, st.vm() * V_SC, st.f() * F_SC, mass, si["mu"], si["k"], b)
        loss = loss + law_w * sum(r.values())
        logs.update({f"law_{k}": v.item() for k, v in r.items()})
    logs.update(l_p=l_p.item(), l_x=l_x.item(), l_v=l_v.item(), l_c=l_c.item(), loss=loss.item())
    return loss, logs
