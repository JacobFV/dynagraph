"""ACT-style imitation policy (Zhao et al. 2023, "Action Chunking with Transformers"), trained from scratch on il-v0
demos. Low-dim observation tokens instead of images (envs-r7 adds a 64x64 wrist camera -> optional CNN token).

  encoder: [context (embodiment + family), proprio(t, t-1), latent z, K_OBJ object tokens, K_REG region tokens]
           -> transformer encoder (object / region tokens masked when invalid)
  decoder: H learned chunk queries cross-attend the encoder -> H actions (continuous dims + controller-mode logits)
  CVAE:    training-time encoder over [CLS, proprio, action chunk] -> z (mean / logvar); z = 0 at test (ACT).
Loss: L1 on standardized continuous action dims (padded steps masked) + CE on the mode + beta * KL.
Execution: a new chunk every tick, temporal ensembling over overlapping chunks (weights exp(-m * age)).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D

MODE_SL = slice(8, 12)
CONT = np.r_[0:8, 12:17]      # continuous action dims (il-v0); il-v1 adds 17:20 (force target), see cont_dims()


def cont_dims(a_dim):
    return np.r_[0:8, 12:a_dim]


class ACTPolicy(nn.Module):
    def __init__(self, ctx_dim, d=192, heads=4, enc_layers=4, dec_layers=4, H=20, z_dim=32, drop=0.1,
                 a_mean=None, a_std=None, a_dim=D.A_DIM, img=0):
        super().__init__()
        self.H, self.z_dim, self.ctx_dim = H, z_dim, ctx_dim
        self.cfg = dict(ctx_dim=ctx_dim, d=d, heads=heads, enc_layers=enc_layers, dec_layers=dec_layers, H=H,
                        z_dim=z_dim, drop=drop, a_dim=a_dim, img=img)
        self.a_dim, self.img = a_dim, img
        self.register_buffer("a_mean", torch.zeros(a_dim) if a_mean is None else torch.as_tensor(a_mean).float())
        self.register_buffer("a_std", torch.ones(a_dim) if a_std is None else torch.as_tensor(a_std).float())
        if img:     # wrist camera (envs-r7): small CNN -> 4 tokens (2x2 spatial grid)
            self.cnn = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                                     nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(), nn.Conv2d(64, d, 3, 2, 1),
                                     nn.AdaptiveAvgPool2d(2))
            self.img_typ = nn.Parameter(torch.randn(1, 4, d) * 0.02)
        self.ctx = nn.Linear(ctx_dim, d)
        self.prop = nn.Sequential(nn.Linear(2 * D.P_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.obj = nn.Sequential(nn.Linear(D.O_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.reg = nn.Sequential(nn.Linear(D.R_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.zin = nn.Linear(z_dim, d)
        self.typ = nn.Parameter(torch.randn(5, d) * 0.02)       # ctx, prop, z, obj, reg
        L = lambda: nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(L(), enc_layers)
        self.q = nn.Parameter(torch.randn(H, d) * 0.02)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, drop, batch_first=True,
                                                                    norm_first=True), dec_layers)
        self.head = nn.Linear(d, a_dim)
        # CVAE posterior encoder
        self.cls = nn.Parameter(torch.randn(1, d) * 0.02)
        self.ain = nn.Linear(a_dim, d)
        self.apos = nn.Parameter(torch.randn(H, d) * 0.02)
        self.venc = nn.TransformerEncoder(L(), 2)
        self.zhead = nn.Linear(d, 2 * z_dim)

    def norm_act(self, a):
        return (a - self.a_mean) / self.a_std

    def encode(self, ctx, p2, obj, reg, z, img=None):
        B = ctx.shape[0]
        toks = [self.ctx(ctx)[:, None] + self.typ[0], self.prop(p2)[:, None] + self.typ[1],
                self.zin(z)[:, None] + self.typ[2], self.obj(obj) + self.typ[3], self.reg(reg) + self.typ[4]]
        pads = [torch.zeros(B, 3, dtype=torch.bool, device=ctx.device), obj[..., -1] < 0.5, reg[..., -1] < 0.5]
        if self.img and img is not None:
            f = self.cnn(img.permute(0, 3, 1, 2).float() / 255.0 - 0.5).flatten(2).transpose(1, 2)   # (B,4,d)
            toks.append(f + self.img_typ)
            pads.append(torch.zeros(B, 4, dtype=torch.bool, device=ctx.device))
        x = torch.cat(toks, 1)
        pad = torch.cat(pads, 1)
        return self.enc(x, src_key_padding_mask=pad), pad

    def posterior(self, p2, act_n, apad):
        B = p2.shape[0]
        x = torch.cat([self.cls.expand(B, 1, -1), self.prop(p2)[:, None], self.ain(act_n) + self.apos], 1)
        pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=x.device), apad], 1)
        h = self.venc(x, src_key_padding_mask=pad)[:, 0]
        mu, lv = self.zhead(h).chunk(2, -1)
        return mu, lv.clamp(-8, 4)

    def forward(self, ctx, p2, obj, reg, act=None, apad=None, img=None):
        B = ctx.shape[0]
        if act is not None:
            mu, lv = self.posterior(p2, self.norm_act(act), apad)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        else:
            mu = lv = None
            z = torch.zeros(B, self.z_dim, device=ctx.device)
        mem, pad = self.encode(ctx, p2, obj, reg, z, img)
        h = self.dec(self.q.expand(B, -1, -1), mem, memory_key_padding_mask=pad)
        return self.head(h), mu, lv

    def loss(self, batch, beta=10.0):
        out, mu, lv = self(batch["ctx"], batch["p2"], batch["obj"], batch["reg"], batch["act"], batch["apad"],
                           batch.get("img"))
        tgt = self.norm_act(batch["act"])
        m = (~batch["apad"]).float()[..., None]
        cd = cont_dims(self.a_dim)
        c = torch.as_tensor(cd, device=out.device)
        l1 = ((out[..., c] - tgt[..., c]).abs() * m).sum() / (m.sum() * len(cd)).clamp(min=1)
        mode_t = batch["act"][..., MODE_SL].argmax(-1)
        ce = (F.cross_entropy(out[..., MODE_SL].reshape(-1, 4), mode_t.reshape(-1), reduction="none")
              * m.reshape(-1)).sum() / m.sum().clamp(min=1)
        kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(-1)).mean() if beta > 0 else torch.zeros((), device=out.device)
        return l1 + ce + beta * kl, dict(l1=float(l1), ce=float(ce), kl=float(kl))

    @torch.no_grad()
    def act_chunk(self, ctx, p2, obj, reg, img=None):
        out, _, _ = self(ctx, p2, obj, reg, img=img)
        a = out * self.a_std + self.a_mean
        mode = out[..., MODE_SL].argmax(-1)
        a[..., MODE_SL] = F.one_hot(mode, 4).float()
        return a


def action_stats(acts):
    """per-dim mean/std for the continuous dims; the mode one-hot stays raw (mean 0, std 1)."""
    A = np.concatenate(acts, 0)
    mu, sd = A.mean(0), A.std(0) + 1e-3
    mu[MODE_SL], sd[MODE_SL] = 0.0, 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


class TemporalEnsembler:
    """ACT temporal ensembling: every tick a new H-step chunk; the executed action is the exp-weighted mean of all
    chunk predictions for this tick (older predictions weighted higher, w_i = exp(-m * i), i = 0 oldest)."""

    def __init__(self, H, m=0.01):
        self.H, self.m, self.buf = H, m, []      # list of (t_issued, chunk)

    def push(self, t, chunk):
        if self.m == "open":        # ACT without ensembling: execute each chunk open-loop, re-plan every H ticks
            if not self.buf or t - self.buf[-1][0] >= self.H:
                self.buf = [(t, np.asarray(chunk))]
            return
        self.buf.append((t, np.asarray(chunk)))
        self.buf = [(ti, c) for ti, c in self.buf if t - ti < self.H]

    def get(self, t):
        if self.m is None or self.m == "open":   # 'latest' / 'open': one chunk's own prediction (no ensembling)
            ti, c = self.buf[-1]
            return np.asarray(c[t - ti]).copy()
        preds = [c[t - ti] for ti, c in self.buf if 0 <= t - ti < len(c)]
        w = np.exp(-self.m * np.arange(len(preds)))
        a = (np.stack(preds) * (w / w.sum())[:, None]).sum(0)
        mode = int(np.argmax(sum(wi * p[MODE_SL] for wi, p in zip(w, preds))))
        a[MODE_SL] = np.eye(4)[mode]
        return a
