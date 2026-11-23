"""v2 typed physical-memory world model (owner: MODEL-V2). See docs/MODEL_V2.md.

variants
  typed    : per-interaction sysID (conv+transformer over raw segments + analytic features) -> per-latent Gaussian,
             precision-fused (with a learned equicorrelation correction) into TYPED SLOTS; outcome head reads latent
             physics ONLY through the slots (FiLM + concat). Slots are readable and overwriteable (interventions).
  generic  : same encoders and head; interaction segments are pooled into an unsupervised memory vector that is
             projected into the same FiLM input width (no typed slots, no sysID loss). "no typed memory, no sysID".
  nomem    : same head, no interaction history at all (z input = 0).
  oracle   : typed model evaluated with the true latents written into the slots (labelled oracle; eval only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import spec as S
from .tensorize import V2Batch, V2Labels

L = S.N_LATENT
TF_LOGVAR = math.log(0.05 ** 2)   # slot log-variance used when a latent is "known" (teacher forcing / oracle)


def mlp(i, h, o, n=2, act=nn.GELU):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class SegmentEncoder(nn.Module):
    """raw interaction time series -> embedding. (B,K,T,C) -> (B,K,d)."""

    def __init__(self, d=128, layers=2):
        super().__init__()
        C = S.N_CH
        self.conv = nn.Sequential(
            nn.Conv1d(C, d, 5, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU())
        self.pos = nn.Parameter(torch.randn(1, S.SEG_T // 4, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(el, layers)
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.call = mlp(S.CALL_DIM, d, d)
        self.an = mlp(S.N_AN, d, d)
        self.emb = mlp(S.EMB_DIM, d, d)
        self.out = mlp(4 * d, 2 * d, d)

    def forward(self, seg, call, an, emb):
        B, K, T, C = seg.shape
        x = seg.reshape(B * K, T, C).transpose(1, 2)
        h = self.conv(x).transpose(1, 2) + self.pos                     # (BK, T/4, d)
        h = self.tf(h)
        p, _ = self.pool(self.pool_q.expand(B * K, -1, -1), h, h)
        p = p.reshape(B, K, -1)
        e = self.emb(emb)[:, None].expand(-1, K, -1)
        return self.out(torch.cat([p, self.call(call), self.an(an.clamp(-20, 20)), e], -1))


class SlotPool(nn.Module):
    """slot attention (Locatello et al.) over a variable number of entity tokens -> S slots."""

    def __init__(self, d=128, n_slots=8, iters=2):
        super().__init__()
        self.n, self.iters = n_slots, iters
        self.mu = nn.Parameter(torch.randn(1, n_slots, d) * 0.1)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d)
        self.ln_in, self.ln_s = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = mlp(d, 2 * d, d)

    def forward(self, x, mask):
        B, N, d = x.shape
        x = self.ln_in(x)
        k, v = self.k(x), self.v(x)
        s = self.mu.expand(B, -1, -1)
        for _ in range(self.iters):
            q = self.q(self.ln_s(s))
            logits = torch.einsum("bsd,bnd->bsn", q, k) / math.sqrt(d)
            logits = logits.masked_fill(~mask[:, None], -1e4)
            a = logits.softmax(1) + 1e-8                                  # compete over slots
            a = a * mask[:, None]
            a = a / a.sum(-1, keepdim=True).clamp(min=1e-6)
            upd = torch.einsum("bsn,bnd->bsd", a, v)
            s = self.gru(upd.reshape(-1, d), s.reshape(-1, d)).reshape(B, self.n, d)
            s = s + self.ff(s)
        return s


class SceneEncoder(nn.Module):
    """current entity state (+ embodiment) -> context. Never sees interaction histories."""

    def __init__(self, d=128, n_slots=8):
        super().__init__()
        self.ent = mlp(S.ENT_DIM + 1, d, d)
        self.pool = SlotPool(d, n_slots)
        self.emb = mlp(S.EMB_DIM, d, d)
        el = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(el, 2)
        self.out = mlp(3 * d, 2 * d, d)

    def forward(self, ent, ent_mask, target, emb):
        B, N, _ = ent.shape
        cnt = torch.log1p(ent_mask.sum(1).float())[:, None, None].expand(-1, N, 1)
        e = self.ent(torch.cat([ent, cnt], -1))
        tgt = e[torch.arange(B, device=e.device), target]
        slots = self.pool(e, ent_mask)
        toks = torch.cat([tgt[:, None], self.emb(emb)[:, None], slots], 1)
        h = self.tf(toks)
        return self.out(torch.cat([h[:, 0], h[:, 1], h[:, 2:].mean(1)], -1)), tgt


class FiLMHead(nn.Module):
    """skill-conditioned outcome / effort / risk head. Latent physics enters ONLY via zin (typed slots) through
    FiLM at every layer + concatenation at the input."""

    def __init__(self, d=128, h=256, n_layers=3, zdim=2 * L):
        super().__init__()
        self.call = mlp(S.CALL_DIM, d, d)
        self.inp = nn.Linear(2 * d + zdim, h)
        self.layers = nn.ModuleList([nn.Linear(h, h) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h, elementwise_affine=False) for _ in range(n_layers + 1)])
        self.film = mlp(zdim + S.CALL_DIM, h, 2 * h * (n_layers + 1))
        self.n_layers = n_layers
        self.cont = nn.Linear(h, 2 * S.N_CONT)
        self.ev = nn.Linear(h, S.N_EVENTS)
        self.eff = nn.Linear(h, 2)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, ctx, call, zin):
        x = self.inp(torch.cat([ctx, self.call(call), zin], -1))
        gb = self.film(torch.cat([zin, call], -1)).chunk(2 * (self.n_layers + 1), -1)
        for i in range(self.n_layers + 1):
            x = self.norms[i](x) * (1 + gb[2 * i]) + gb[2 * i + 1]
            x = F.gelu(x)
            if i < self.n_layers:
                x = x + self.layers[i](x)
        c = self.cont(x)
        e = self.eff(x)
        return dict(cont_mu=c[:, :S.N_CONT], cont_logvar=c[:, S.N_CONT:].clamp(-12, 6), ev_logits=self.ev(x),
                    eff_mu=e[:, 0], eff_logvar=e[:, 1].clamp(-8, 4))


class PhysMemV2(nn.Module):
    def __init__(self, variant="typed", d=128, n_slots=8, seg_layers=2):
        super().__init__()
        assert variant in ("typed", "generic", "nomem")
        self.variant = variant
        self.seg = SegmentEncoder(d, seg_layers) if variant != "nomem" else None
        self.scene = SceneEncoder(d, n_slots)
        self.head = FiLMHead(d)
        if variant == "typed":
            self.est = mlp(d, d, 2 * L)
            self.prior = mlp(d + S.EMB_DIM, d, 2 * L)            # prior from target token + embodiment
            self.rho = nn.Parameter(torch.full((L,), -2.0))      # equicorrelation of per-interaction errors
        elif variant == "generic":
            self.mem = mlp(d + 1, d, d)
            self.to_z = nn.Linear(d, 2 * L)
        # info head: predicted per-latent log-variance a candidate call would yield (planner VOI)
        self.info = mlp(d + S.CALL_DIM, d, L) if variant == "typed" else None
        # outcome scalers (raw = norm*std + mean), fit on train
        self.register_buffer("cont_mean", torch.zeros(S.N_CONT))
        self.register_buffer("cont_std", torch.ones(S.N_CONT))
        self.register_buffer("eff_mean", torch.zeros(()))
        self.register_buffer("eff_std", torch.ones(()))
        # per-skill call-parameter scalers (SkillCall.vector carries raw SI params of very different scales)
        self.register_buffer("ev_supervised", torch.ones(S.N_EVENTS, dtype=torch.bool))  # events with training labels
        self.register_buffer("z_var_scale", torch.ones(L))   # post-hoc per-latent variance temperature (fit on val)
        self.register_buffer("call_mean", torch.zeros(len(S.SKILLS), S.CALL_DIM))
        self.register_buffer("call_std", torch.ones(len(S.SKILLS), S.CALL_DIM))

    def norm_call(self, call):
        sk = call[..., :len(S.SKILLS)].argmax(-1)
        mu, sd = self.call_mean[sk], self.call_std[sk]
        nk = len(S.SKILLS)
        return torch.cat([call[..., :nk], ((call[..., nk:] - mu[..., nk:]) / sd[..., nk:]).clamp(-8, 8)], -1)

    def norm_batch(self, b: V2Batch) -> V2Batch:
        if getattr(b, "_normed", False):
            return b
        nb = V2Batch(**{f: getattr(b, f) for f in b.__dataclass_fields__})
        nb.call = self.norm_call(b.call)
        nb.seg_call = self.norm_call(b.seg_call)
        nb.seg = torch.nan_to_num(b.seg).clamp(-200, 200)         # robustness to simulator blow-ups
        nb.ent = torch.nan_to_num(b.ent).clamp(-100, 100)
        nb.seg_an = torch.nan_to_num(b.seg_an).clamp(-1e3, 1e3)
        nb._normed = True
        return nb

    # ------------------------------------------------------------- memory
    def per_interaction(self, b: V2Batch):
        """per-interaction latent estimates: (mu (B,K,L), logvar (B,K,L), h (B,K,d))."""
        h = self.seg(b.seg, b.seg_call, b.seg_an, b.emb)
        if self.variant != "typed":
            return None, None, h
        o = self.est(h)
        return o[..., :L], o[..., L:].clamp(-9, 6), h

    def memory(self, b: V2Batch, ctx_tgt=None):
        """-> dict(z_mu, z_logvar (B,L) typed slots ; seg_mu, seg_logvar per interaction ; zin (B,2L))."""
        B = b.call.shape[0]
        out = {}
        if self.variant == "nomem":
            out["zin"] = torch.zeros(B, 2 * L, device=b.call.device)
            return out
        mu_k, lv_k, h = self.per_interaction(b)
        m = (b.seg_mask & b.seg_on_target).float()
        if self.variant == "generic":
            pooled = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            mem = self.mem(torch.cat([pooled, torch.log1p(m.sum(1, keepdim=True))], -1))
            out["zin"] = self.to_z(mem)
            out["mem"] = mem
            return out
        pr = self.prior(torch.cat([ctx_tgt, b.emb], -1))
        mu0, lv0 = pr[:, :L], pr[:, L:].clamp(-6, 4)
        p0 = torch.exp(-lv0)
        pk = torch.exp(-lv_k) * m[..., None]
        n = m.sum(1, keepdim=True)
        rho = torch.sigmoid(self.rho)[None]
        infl = 1 + rho * (n - 1).clamp(min=0)               # n equicorrelated measurements ~ n/(1+rho(n-1)) effective
        psum = pk.sum(1) / infl
        musum = (pk * mu_k).sum(1) / infl
        prec = p0 + psum
        z_mu = (p0 * mu0 + musum) / prec
        z_lv = -torch.log(prec) + torch.log(self.z_var_scale)[None]
        out.update(z_mu=z_mu, z_logvar=z_lv, seg_mu=mu_k, seg_logvar=lv_k, prior_mu=mu0, prior_logvar=lv0, h=h)
        out["zin"] = self.slot_input(z_mu)          # plug-in posterior mean (planner uses posterior samples)
        return out

    @staticmethod
    def slot_input(z_mu, z_logvar=None):
        """head input for typed slots: a latent VALUE presented as known (sample form). The slot's posterior
        log-variance is kept in mem / used by the planner through sampling, not fed to the head."""
        return torch.cat([z_mu, torch.full_like(z_mu, 0.5 * TF_LOGVAR)], -1)

    # ------------------------------------------------------------- forward
    def forward(self, b: V2Batch, z_override: Optional[tuple] = None, z_mask_override=None, return_mem=False):
        """z_override=(mu (B,L), logvar (B,L)) overwrites typed slots (all latents, or only where
        z_mask_override (B,L) is True). Returns dict with raw-unit outcome predictions."""
        b = self.norm_batch(b)
        ctx, tgt = self.scene(b.ent, b.ent_mask, b.target, b.emb)
        mem = self.memory(b, tgt)
        zin = mem["zin"]
        if z_override is not None:
            assert self.variant == "typed", "interventions need typed slots"
            mu, lv = z_override
            if z_mask_override is not None:
                mu = torch.where(z_mask_override, mu, mem["z_mu"])
                lv = torch.where(z_mask_override, lv, mem["z_logvar"])
            zin = self.slot_input(mu, lv)
            mem = dict(mem, z_mu=mu, z_logvar=lv, zin=zin)
        o = self.head(ctx, b.call, zin)
        o["ctx"] = ctx
        o["mem"] = mem
        return o

    def decode(self, o):
        """normalized head outputs -> raw units: cont mean/var, event probs, effort (cost), risk prob."""
        cm = o["cont_mu"] * self.cont_std + self.cont_mean
        cv = torch.exp(o["cont_logvar"]) * self.cont_std ** 2
        p = torch.sigmoid(o["ev_logits"])
        ri = [S.OUT_EVENTS.index(e) for e in S.RISK_EVENTS if bool(self.ev_supervised[S.OUT_EVENTS.index(e)])]
        risk = 1 - torch.prod(1 - p[:, ri], -1) if ri else torch.zeros_like(p[:, 0])
        logc = o["eff_mu"] * self.eff_std + self.eff_mean
        eff = torch.exp(logc + 0.5 * torch.exp(o["eff_logvar"]) * self.eff_std ** 2) - 1e-3
        return dict(cont=cm, cont_var=cv, p=p, risk=risk, effort=eff.clamp(min=0), log_effort=logc,
                    success=p[:, S.OUT_EVENTS.index("success")])


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------- losses
def gauss_nll(mu, lv, y, mask):
    l = 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))
    m = mask.float()
    return (l * m).sum() / m.sum().clamp(min=1)


def v2_loss(model: PhysMemV2, b: V2Batch, lab: V2Labels, tf_prob=0.5, w_sysid=1.0, w_info=0.1, detach_est=True):
    """total loss + logs. teacher forcing: with prob tf_prob per sample, labelled latents are written into the slots
    (known, small variance) so the slot->outcome mapping is learned independent of estimation quality; otherwise the
    (detached) fused posterior is used so the head learns to act under real estimation uncertainty."""
    b = model.norm_batch(b)
    ctx, tgt = model.scene(b.ent, b.ent_mask, b.target, b.emb)
    mem = model.memory(b, tgt)
    logs = {}
    loss = 0.0
    if model.variant == "typed":
        zm = lab.z_mask
        seg_m = (b.seg_mask & b.seg_on_target)[..., None] & zm[:, None]
        l_seg = gauss_nll(mem["seg_mu"], mem["seg_logvar"], lab.z[:, None].expand_as(mem["seg_mu"]), seg_m)
        l_post = gauss_nll(mem["z_mu"], mem["z_logvar"], lab.z, zm)
        l_prior = gauss_nll(mem["prior_mu"], mem["prior_logvar"], lab.z, zm)
        loss = loss + w_sysid * (l_seg + l_post + 0.5 * l_prior)
        logs.update(l_seg=l_seg.item(), l_post=l_post.item(), l_prior=l_prior.item())
        z_mu, z_lv = mem["z_mu"], mem["z_logvar"]
        if detach_est:
            z_mu, z_lv = z_mu.detach(), z_lv.detach()
        B = z_mu.shape[0]
        # sample-form slot input (exactly what the planner feeds): each latent is either its TRUE value (teacher
        # forcing, labelled latents, prob tf_prob per sample) or a SAMPLE from the fused posterior, always presented
        # as a known value. Unlabelled / irrelevant latents are posterior(prior) samples -> the head learns to ignore
        # them where they do not matter, and expected costs over samples integrate latent uncertainty correctly.
        tf = (torch.rand(B, 1, device=z_mu.device) < tf_prob) & zm
        z_s = z_mu + torch.exp(0.5 * z_lv) * torch.randn_like(z_mu)
        z_in_mu = torch.where(tf, lab.z, z_s)
        zin = model.slot_input(z_in_mu, torch.full_like(z_lv, TF_LOGVAR))
        # info head regression: predicted per-interaction logvar for the call that produced it
        if model.info is not None and w_info > 0:
            hm = (b.seg_mask & b.seg_on_target)
            if hm.any():
                pred = model.info(torch.cat([ctx.detach()[:, None].expand(-1, b.seg_call.shape[1], -1),
                                             b.seg_call], -1))
                l_info = (((pred - mem["seg_logvar"].detach()) ** 2) * (hm[..., None] & zm[:, None])).sum() / \
                         ((hm[..., None] & zm[:, None]).sum().clamp(min=1))
                loss = loss + w_info * l_info
                logs["l_info"] = l_info.item()
    else:
        zin = mem["zin"]
    o = model.head(ctx, b.call, zin)
    yc = (lab.cont - model.cont_mean) / model.cont_std
    l_cont = gauss_nll(o["cont_mu"], o["cont_logvar"], yc, lab.cont_mask)
    em = lab.events_mask.float()
    l_ev = (F.binary_cross_entropy_with_logits(o["ev_logits"], lab.events.float(), reduction="none") * em).sum() / \
        em.sum().clamp(min=1)
    ye = (lab.effort - model.eff_mean) / model.eff_std
    l_eff = gauss_nll(o["eff_mu"], o["eff_logvar"], ye, torch.ones_like(ye, dtype=torch.bool))
    loss = loss + l_cont + l_ev + 0.5 * l_eff
    logs.update(l_cont=l_cont.item(), l_ev=l_ev.item(), l_eff=l_eff.item(), loss=float(loss.item()))
    return loss, logs
