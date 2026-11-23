"""Compact recurrent physical-memory models (agent MODEL).

One shared interface (see docs/MODEL_INTERFACE.md):
    tokens, mask, slots = model.encode(events, query)
    tokens = model.refine(tokens, mask, passes)        # model.core is swappable
    out = model.readout(tokens, mask, query, slots)    # -> ModelOutput (raw SI units)

Encoders
- GenericMemoryEncoder: Perceiver-style latent array cross-attending to raw event tokens
  ("no structured memory" control). Entity/source identity only as equality flags.
- StructuredMemoryEncoder: per-(entity, property) slots addressed by equality with
  age/variance-aware gated updates and duplicate suppression, entity tokens with relational
  messages (relative geometry, contact evidence), and typed FRICTION/MASS estimate slots holding
  explicit (mean, logvar) + latent that can be overwritten (`intervene`).

Label: this is the "compact recurrent physical-memory model", never "inkling".
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from physmem.schema import (
    ACTION_DIM,
    HORIZONS,
    N_PROPS,
    STATE_DIM,
    VALUE_DIM,
    EntityType,
    EventTensors,
    ModelOutput,
    Prop,
    Provenance,
    QueryTensors,
)

N_PROV = len(Provenance)


def _fdt():
    return torch.get_default_dtype()

N_ETYPE = len(EntityType)
N_CONTACT = 4
G = 9.81

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def soft_clamp(x, lo, hi):
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    return mid + half * torch.tanh((x - mid) / half)


def mlp(i, h, o, n=2):
    layers = [nn.Linear(i, h), nn.GELU()]
    for _ in range(n - 2):
        layers += [nn.Linear(h, h), nn.GELU()]
    layers += [nn.Linear(h, o)]
    return nn.Sequential(*layers)


def _gather_ent(x_bn, idx_be):
    """x (B,N[,...]) gather at entity indices idx (B,E) (-1 -> 0 index, caller masks)."""
    idx = idx_be.clamp(min=0)
    if x_bn.dim() == 2:
        return torch.gather(x_bn, 1, idx)
    return torch.gather(x_bn, 1, idx.unsqueeze(-1).expand(-1, -1, x_bn.shape[-1]))


class Block(nn.Module):
    """pre-LN set-attention block (self-attention over tokens, key padding by mask)."""

    def __init__(self, d, heads=4, ff=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = mlp(d, ff * d, d)

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class CrossBlock(nn.Module):
    """queries attend to a key/value set (mask True = valid)."""

    def __init__(self, d, heads=4, ff=2):
        super().__init__()
        self.lnq = nn.LayerNorm(d)
        self.lnk = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = mlp(d, ff * d, d)

    def forward(self, q, kv, mask):
        # guard rows with no valid keys (avoid NaN): let them attend to everything, output zeroed
        empty = ~mask.any(-1)
        m = mask.clone()
        m[empty] = True
        kvn = self.lnk(kv)
        a, _ = self.attn(self.lnq(q), kvn, kvn, key_padding_mask=~m, need_weights=False)
        a = a * (~empty).to(_fdt())[:, None, None]
        q = q + a
        return q + self.ff(self.ln2(q))


# ----------------------------------------------------------------------------
# event featurisation (shared by both encoders; equality features only, no id embeddings)
# ----------------------------------------------------------------------------

EV_FEAT_DIM = VALUE_DIM * 2 + N_PROPS + N_PROV + 4 + 4 + N_ETYPE + (N_ETYPE + 1) + 1


def event_features(ev: EventTensors, target: torch.Tensor) -> torch.Tensor:
    B, E = ev.entity.shape
    m = ev.mask
    val = symlog(ev.value.to(_fdt()) * 4.0)  # metres*4: typical 0.1 m -> 0.4
    lv = soft_clamp(ev.logvar.to(_fdt()), -20.0, 6.0) / 10.0
    prop = F.one_hot(ev.prop.long().clamp(0, N_PROPS - 1), N_PROPS).to(_fdt())
    prov = F.one_hot(ev.provenance.long().clamp(0, N_PROV - 1), N_PROV).to(_fdt())
    age = ev.age.to(_fdt()).clamp(min=0)
    lat = ev.latency.to(_fdt()).clamp(min=0)
    dt = ev.dt_obs.to(_fdt()).clamp(min=1e-3)[:, None]
    tfe = torch.stack([age / 2.0, torch.log1p(age / dt) / 4.0, lat / 0.5, torch.log1p(lat / dt) / 2.0], -1)
    ent = ev.entity.long()
    rel = ev.related.long()
    t = target.long()[:, None]
    # source equality with the source of the target's most recent pose event
    tp = m & (ent == t) & (ev.prop.long() == int(Prop.POSE2))
    a_ = torch.where(tp, age, torch.full_like(age, 1e9))
    ref = torch.gather(ev.source.long(), 1, a_.argmin(1, keepdim=True))
    has_ref = tp.any(1, keepdim=True)
    same_src = ((ev.source.long() == ref) & has_ref).to(_fdt())
    eq = torch.stack([(ent == t).to(_fdt()), (rel == t).to(_fdt()), (rel < 0).to(_fdt()), same_src], -1)
    et = _gather_ent(ev.ent_type.long(), ent).clamp(0, N_ETYPE - 1)
    et_oh = F.one_hot(et, N_ETYPE).to(_fdt())
    ret = _gather_ent(ev.ent_type.long(), rel).clamp(0, N_ETYPE - 1)
    ret_oh = torch.cat([F.one_hot(ret, N_ETYPE).to(_fdt()) * (rel >= 0).to(_fdt())[..., None], (rel < 0).to(_fdt())[..., None]], -1)
    f = torch.cat([val, lv, prop, prov, tfe, eq, et_oh, ret_oh, m.to(_fdt())[..., None]], -1)
    return f * m.to(_fdt())[..., None]


# ----------------------------------------------------------------------------
# encoders
# ----------------------------------------------------------------------------


class GenericMemoryEncoder(nn.Module):
    """fixed latent array cross-attending to event tokens (no structured memory)."""

    kind = "generic"

    def __init__(self, d=128, n_latents=16, n_cross=2, ff_hidden=None):
        super().__init__()
        self.d = d
        h = ff_hidden or 2 * d
        self.ev = mlp(EV_FEAT_DIM, h, d, n=3)
        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.02)
        self.cross = nn.ModuleList([CrossBlock(d) for _ in range(n_cross)])
        self.n_latents = n_latents

    def forward(self, ev: EventTensors, query: QueryTensors, edge_lesion=None):
        B = ev.entity.shape[0]
        x = self.ev(event_features(ev, query.target))
        lat = self.latents[None].expand(B, -1, -1)
        for blk in self.cross:
            lat = blk(lat, x, ev.mask)
        M = self.n_latents
        mask = torch.ones(B, M, dtype=torch.bool, device=x.device)
        slots = {
            "token_entity": torch.full((B, M), -1, dtype=torch.long, device=x.device),
            "token_kind": torch.zeros(B, M, dtype=torch.long, device=x.device),
            "typed": {},
            "typed_index": {},
        }
        return lat, mask, slots


TYPED_PROPS = (Prop.FRICTION, Prop.MASS)


def typed_norm(prop, mean):
    if prop == Prop.FRICTION:
        return (mean - 0.5) / 0.25
    return torch.log(mean.clamp(min=1e-3)) / 1.0


class StructuredMemoryEncoder(nn.Module):
    """per-(entity,prop) slots with gated, age/variance-aware updates, entity tokens with
    relational messages, and typed FRICTION/MASS estimate slots."""

    kind = "structured"
    N_STATS = VALUE_DIM * 2 + 4

    def __init__(self, d=128, latent_typed=16, ff_hidden=None):
        super().__init__()
        self.d = d
        h = ff_hidden or 2 * d
        self.ev = mlp(EV_FEAT_DIM, h, d, n=3)
        self.kproj = nn.Linear(d, d)
        self.vproj = nn.Linear(d, d)
        self.prop_emb = nn.Embedding(N_PROPS, d)
        self.etype_emb = nn.Embedding(N_ETYPE, d)
        self.target_emb = nn.Embedding(2, d)
        # age / variance / provenance-aware attention bias (insertion vs correction weighting)
        self.bias = mlp(4 + N_PROV, 32, 1)
        self.stats = nn.Linear(self.N_STATS, d)
        self.gate = nn.Linear(3 * d, d)
        self.upd = nn.Linear(3 * d, d)
        self.empty = nn.Parameter(torch.zeros(N_PROPS, d))
        self.slot_ln = nn.LayerNorm(d)
        # entity pooling over own slots
        self.ent_q = nn.Linear(d, d)
        self.ent_out = mlp(2 * d, h, d)
        # relational messages
        self.n_rel = 9
        self.edge = mlp(2 * d + self.n_rel, h, d)
        self.edge_gate = nn.Linear(2 * d + self.n_rel, 1)
        self.ent_msg = mlp(2 * d, h, d)
        # typed estimate slots
        self.typed_q = nn.Parameter(torch.randn(len(TYPED_PROPS), d) * 0.02)
        self.typed_cross = CrossBlock(d)
        self.typed_head = nn.Linear(d, 2 + latent_typed)
        self.typed_enc = mlp(2 + latent_typed, h, d)
        self.typed_kind = nn.Embedding(len(TYPED_PROPS), d)
        self.latent_typed = latent_typed

    # -- slot construction ----------------------------------------------------
    @staticmethod
    def duplicate_mask(ev: EventTensors):
        """True for events that are exact duplicates of an earlier-index event
        (same entity, prop, source, measured time, value). Exact duplicates carry no new
        evidence; dropping all but one keeps results permutation-equivariant because the
        removed copies are identical to the kept one."""
        B, E = ev.entity.shape
        m = ev.mask
        g = torch.Generator(device="cpu").manual_seed(1234)
        proj = torch.randn(VALUE_DIM, generator=g).to(ev.value.device)
        vh = (ev.value.to(_fdt()) * proj).sum(-1)
        meas = ev.age.to(_fdt())  # same query time within a sample => equal age <=> equal measured_at
        same = (
            (ev.entity[:, :, None] == ev.entity[:, None, :])
            & (ev.prop[:, :, None] == ev.prop[:, None, :])
            & (ev.source[:, :, None] == ev.source[:, None, :])
            & ((meas[:, :, None] - meas[:, None, :]).abs() < 1e-6)
            & ((vh[:, :, None] - vh[:, None, :]).abs() < 1e-6)
            & m[:, :, None]
            & m[:, None, :]
        )
        earlier = torch.tril(torch.ones(E, E, dtype=torch.bool, device=m.device), diagonal=-1)
        return (same & earlier[None]).any(-1)

    def forward(self, ev: EventTensors, query: QueryTensors, edge_lesion=None):
        B, E = ev.entity.shape
        N = ev.ent_type.shape[1]
        P = N_PROPS
        dev = ev.value.device
        tgt = query.target.long()
        feats = event_features(ev, tgt)
        x = self.ev(feats)
        dup = self.duplicate_mask(ev)
        em = ev.mask & ~dup & (ev.entity >= 0) & (ev.entity < N)
        ent = ev.entity.long().clamp(0, N - 1)
        prop = ev.prop.long().clamp(0, P - 1)
        sid = ent * P + prop  # (B,E)
        S = N * P
        in_slot = (sid[:, None, :] == torch.arange(S, device=dev)[None, :, None]) & em[:, None, :]  # (B,S,E)
        slot_valid = in_slot.any(-1)
        # slot queries: prop type + entity type + target flag (equality only)
        s_prop = torch.arange(S, device=dev) % P
        s_ent = torch.arange(S, device=dev) // P
        et = ev.ent_type.long().clamp(0, N_ETYPE - 1)
        is_t = (torch.arange(N, device=dev)[None] == tgt[:, None]).long()  # (B,N)
        q = (
            self.prop_emb(s_prop)[None]
            + self.etype_emb(et)[:, s_ent]
            + self.target_emb(is_t)[:, s_ent]
        )  # (B,S,d)
        k = self.kproj(x)
        v = self.vproj(x)
        age = ev.age.to(_fdt()).clamp(min=0)
        lvm = soft_clamp(ev.logvar.to(_fdt()), -20, 6).mean(-1)
        bias_in = torch.cat(
            [
                torch.stack([age / 2.0, torch.log1p(age / 0.05) / 4.0, lvm / 10.0, ev.latency.to(_fdt()) / 0.5], -1),
                F.one_hot(ev.provenance.long().clamp(0, N_PROV - 1), N_PROV).to(_fdt()),
            ],
            -1,
        )
        b = self.bias(bias_in).squeeze(-1)  # (B,E)
        logits = torch.einsum("bsd,bed->bse", q, k) / math.sqrt(self.d) + b[:, None, :]
        logits = logits.masked_fill(~in_slot, -1e9)
        w = torch.softmax(logits, -1) * in_slot.to(_fdt())
        agg = torch.einsum("bse,bed->bsd", w, v)
        # explicit, permutation-invariant slot statistics
        lv = soft_clamp(ev.logvar.to(_fdt()), -20, 6)
        prec = torch.exp(-lv).clamp(max=1e8)  # (B,E,V)
        val = ev.value.to(_fdt())
        inf = in_slot.to(_fdt())
        psum = torch.einsum("bse,bev->bsv", inf, prec)
        pmean = torch.einsum("bse,bev->bsv", inf, prec * val) / psum.clamp(min=1e-12)
        rec = torch.softmax((-age / 0.02)[:, None, :].expand(-1, S, -1).masked_fill(~in_slot, -1e9), -1) * inf
        latest = torch.einsum("bse,bev->bsv", rec, val)
        cnt = inf.sum(-1)
        min_age = (rec * age[:, None, :]).sum(-1)
        stats = torch.cat(
            [
                symlog(pmean * 4.0),
                symlog(latest * 4.0),
                torch.stack(
                    [
                        torch.log1p(psum.mean(-1)) / 10.0,
                        torch.log1p(cnt) / 3.0,
                        min_age / 2.0,
                        slot_valid.to(_fdt()),
                    ],
                    -1,
                ),
            ],
            -1,
        )
        st = self.stats(stats)
        hcat = torch.cat([q, agg, st], -1)
        g = torch.sigmoid(self.gate(hcat))
        slot = q + g * torch.tanh(self.upd(hcat))
        slot = torch.where(slot_valid[..., None], slot, self.empty[s_prop][None].expand(B, -1, -1))
        slot = self.slot_ln(slot)
        # entity tokens: pool own valid slots
        slot_e = slot.view(B, N, P, self.d)
        sv = slot_valid.view(B, N, P)
        eq = self.ent_q(self.etype_emb(et) + self.target_emb(is_t))  # (B,N,d)
        al = torch.einsum("bnd,bnpd->bnp", eq, slot_e) / math.sqrt(self.d)
        al = al.masked_fill(~sv, -1e9)
        aw = torch.softmax(al, -1) * sv.to(_fdt())
        pooled = torch.einsum("bnp,bnpd->bnd", aw, slot_e)
        ent_tok = self.ent_out(torch.cat([eq, pooled], -1))
        ent_valid = ev.ent_mask.bool() & sv.any(-1)
        ent_valid = ent_valid | ev.ent_mask.bool()
        # relational features from latest pose + contact evidence
        latest_e = latest.view(B, N, P, VALUE_DIM)
        pose = latest_e[:, :, int(Prop.POSE2), :4]
        has_pose = sv[:, :, int(Prop.POSE2)]
        dxy = pose[:, None, :, :2] - pose[:, :, None, :2]  # (B,N,N,2): j - i
        dist = dxy.norm(dim=-1, keepdim=True)
        ci, si = pose[:, :, None, 2], pose[:, :, None, 3]
        cj, sj = pose[:, None, :, 2], pose[:, None, :, 3]
        rel_c = (ci * cj + si * sj)[..., None]
        rel_s = (ci * sj - si * cj)[..., None]
        # contact evidence: CONTACT/RELATION events between (entity i, related j)
        rel_idx = ev.related.long()
        cm = em & (rel_idx >= 0) & (rel_idx < N) & ((ev.prop == int(Prop.CONTACT)) | (ev.prop == int(Prop.RELATION)))
        pair = ent * N + rel_idx.clamp(0, N - 1)
        recw = torch.exp(-age / 0.25) * cm.to(_fdt())
        cval = ev.value[..., 0].to(_fdt()).clamp(0, 1)
        num = torch.zeros(B, N * N, device=dev).scatter_add(1, pair, recw * cval)
        den = torch.zeros(B, N * N, device=dev).scatter_add(1, pair, recw)
        contact = (num / den.clamp(min=1e-6)).view(B, N, N)
        contact = torch.maximum(contact, contact.transpose(1, 2))
        has_c = (den.view(B, N, N) + den.view(B, N, N).transpose(1, 2)) > 0
        pv = (has_pose[:, :, None] & has_pose[:, None, :]).to(_fdt())[..., None]
        relf = torch.cat(
            [
                symlog(dxy * 4.0) * pv,
                symlog(dist * 4.0) * pv,
                rel_c * pv,
                rel_s * pv,
                pv,
                contact[..., None],
                has_c.to(_fdt())[..., None],
                (torch.arange(N, device=dev)[None, :, None] == torch.arange(N, device=dev)[None, None, :]).to(_fdt()).expand(B, -1, -1)[..., None],
            ],
            -1,
        )  # (B,N,N,9)
        ei = ent_tok[:, :, None].expand(-1, -1, N, -1)
        ej = ent_tok[:, None, :].expand(-1, N, -1, -1)
        ein = torch.cat([ei, ej, relf], -1)
        gate = torch.sigmoid(self.edge_gate(ein)).squeeze(-1)
        emask = ent_valid[:, :, None] & ent_valid[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=dev)[None]
        if edge_lesion is not None:
            emask = emask & ~edge_lesion.bool()
        gate = gate * emask.to(_fdt())
        msg = (gate[..., None] * self.edge(ein)).sum(2) / emask.to(_fdt()).sum(2, keepdim=True).clamp(min=1)
        ent_tok = ent_tok + self.ent_msg(torch.cat([ent_tok, msg], -1))
        # typed estimate slots for the target
        kv = torch.cat([slot, ent_tok], 1)
        kvm = torch.cat([slot_valid, ent_valid], 1)
        kv_ent = torch.cat([s_ent[None].expand(B, -1), torch.arange(N, device=dev)[None].expand(B, -1)], 1)
        kv = kv + self.target_emb((kv_ent == tgt[:, None]).long())
        tq = self.typed_q[None].expand(B, -1, -1)
        th = self.typed_cross(tq, kv, kvm)
        raw = self.typed_head(th)  # (B,2,2+L)
        typed = {}
        for i, p in enumerate(TYPED_PROPS):
            r = raw[:, i]
            if p == Prop.FRICTION:
                mean = 1.5 * torch.sigmoid(r[:, 0])
            else:
                mean = torch.exp(soft_clamp(r[:, 0], -4.0, 3.0))
            logvar = soft_clamp(r[:, 1] - 2.0, -12.0, 4.0)
            typed[p] = {"mean": mean, "logvar": logvar, "latent": r[:, 2:]}
        typed_tok = torch.stack([self._encode_typed(p, typed[p]) for p in TYPED_PROPS], 1)
        tokens = torch.cat([slot, ent_tok, typed_tok], 1)
        mask = torch.cat([slot_valid, ent_valid, torch.ones(B, len(TYPED_PROPS), dtype=torch.bool, device=dev)], 1)
        M = tokens.shape[1]
        tok_ent = torch.cat(
            [kv_ent, torch.full((B, len(TYPED_PROPS)), -1, dtype=torch.long, device=dev)], 1
        )
        tok_kind = torch.cat(
            [
                torch.ones(B, S, dtype=torch.long, device=dev),
                torch.full((B, N), 2, dtype=torch.long, device=dev),
                torch.tensor([3, 4], device=dev)[None].expand(B, -1),
            ],
            1,
        )
        slots = {
            "token_entity": tok_ent,
            "token_kind": tok_kind,
            "typed": typed,
            "typed_index": {p: S + N + i for i, p in enumerate(TYPED_PROPS)},
            "edge_gate": gate,
            "n_slots": S,
            "n_entities": N,
        }
        return tokens, mask, slots

    def _encode_typed(self, p, t):
        i = TYPED_PROPS.index(p)
        z = torch.cat([typed_norm(p, t["mean"])[:, None], (t["logvar"] / 4.0)[:, None], t["latent"]], -1)
        return self.typed_enc(z) + self.typed_kind.weight[i][None]


class PushSummaryStructuredEncoder(StructuredMemoryEncoder):
    """b11p (post-hoc, exploratory): StructuredMemoryEncoder + a per-push event-sequence summarizer feeding the
    typed FRICTION and MASS slots. Past pushes are segmented by the actor's own COMMAND events (push start = a
    nonzero command preceded by a zero command); for each push window [start-0.05 s, start+window] the target's
    POSE2/VEL2 events (with times relative to the push start) and the push command are summarised by attention
    pooling into per-push evidence (mean, logvar) for friction and log-mass. The per-push evidences are fused with
    the base typed estimate by precision weighting, so the slot stays an explicit, interveneable (mean, logvar)."""

    MAX_PUSH = 4

    def __init__(self, d=128, latent_typed=16, ff_hidden=None, ds=64, window=1.5):
        super().__init__(d, latent_typed, ff_hidden)
        self.window = window
        self.p_ev = mlp(1 + 2 * VALUE_DIM + 2, ds, ds, n=3)
        self.p_k = nn.Linear(ds, ds)
        self.p_v = nn.Linear(ds, ds)
        self.p_ctx = mlp(3, ds, ds)
        self.p_out = mlp(2 * ds + 1, ds, 4)  # friction (raw, logvar), log-mass (mean, logvar)

    def push_evidence(self, ev: EventTensors, tgt):
        B, E = ev.entity.shape
        dev = ev.value.device
        fdt = _fdt()
        t = -ev.age.to(fdt)
        cm = ev.mask & (ev.prop == int(Prop.COMMAND))
        C = 16
        tc = torch.where(cm, t, torch.full_like(t, 1e9))
        tcs, oc = torch.sort(tc, 1)
        tcs, oc = tcs[:, :C], oc[:, :C]
        valid_c = tcs < 1e8
        cval = torch.gather(ev.value.to(fdt)[..., :2], 1, oc[..., None].expand(-1, -1, 2))
        nrm = cval.norm(dim=-1) * valid_c.to(fdt)
        prev = torch.cat([torch.zeros_like(nrm[:, :1]), nrm[:, :-1]], 1)
        start = valid_c & (nrm > 1e-6) & (prev < 1e-6)
        # first MAX_PUSH starts in time order
        key = torch.where(start, tcs, torch.full_like(tcs, 1e9))
        ks, ko = torch.sort(key, 1)
        K = self.MAX_PUSH
        ks, ko = ks[:, :K], ko[:, :K]
        wvalid = ks < 1e8
        wcmd = torch.gather(cval, 1, ko[..., None].expand(-1, -1, 2))
        ws = torch.where(wvalid, ks, torch.zeros_like(ks))
        isp = ev.prop == int(Prop.POSE2)
        isv = ev.prop == int(Prop.VEL2)
        te = ev.mask & (ev.entity == tgt[:, None]) & (isp | isv)
        tau = t[:, None, :] - ws[:, :, None]  # (B,K,E)
        inw = te[:, None, :] & wvalid[:, :, None] & (tau >= -0.05) & (tau < self.window)
        val = symlog(ev.value.to(fdt) * 4.0)
        lv = soft_clamp(ev.logvar.to(fdt), -20.0, 6.0) / 10.0
        f = torch.cat([tau[..., None] / self.window, val[:, None].expand(-1, K, -1, -1), lv[:, None].expand(-1, K, -1, -1),
                       isp.to(fdt)[:, None, :, None].expand(-1, K, -1, 1), isv.to(fdt)[:, None, :, None].expand(-1, K, -1, 1)], -1)
        h = self.p_ev(f)  # (B,K,E,ds)
        ctx = self.p_ctx(torch.cat([wcmd * 4.0, wcmd.norm(dim=-1, keepdim=True) * 4.0], -1))  # (B,K,ds)
        lg = (self.p_k(h) * ctx[:, :, None]).sum(-1) / math.sqrt(h.shape[-1])
        lg = lg.masked_fill(~inw, -1e9)
        w = torch.softmax(lg, -1) * inw.to(fdt)
        pooled = (w[..., None] * self.p_v(h)).sum(2)
        cnt = inw.to(fdt).sum(-1)
        o = self.p_out(torch.cat([pooled, ctx, torch.log1p(cnt)[..., None] / 3.0], -1))
        ok = wvalid & (cnt >= 3)
        return dict(fr_mean=1.5 * torch.sigmoid(o[..., 0]), fr_logvar=soft_clamp(o[..., 1] - 2.0, -12.0, 4.0),
                    lm_mean=soft_clamp(o[..., 2], -4.0, 3.0), lm_logvar=soft_clamp(o[..., 3], -12.0, 4.0), valid=ok,
                    count=cnt, start=ws)

    def forward(self, ev: EventTensors, query: QueryTensors, edge_lesion=None):
        tokens, mask, slots = super().forward(ev, query, edge_lesion=edge_lesion)
        pe = self.push_evidence(ev, query.target.long())
        okf = pe["valid"].to(tokens.dtype)
        typed = {p: dict(v) for p, v in slots["typed"].items()}
        # friction: precision fusion in raw space
        fb = typed[Prop.FRICTION]
        pb = torch.exp(-fb["logvar"])
        pk = torch.exp(-pe["fr_logvar"]) * okf
        P = pb + pk.sum(1)
        fm = (fb["mean"] * pb + (pe["fr_mean"] * pk).sum(1)) / P
        typed[Prop.FRICTION] = {"mean": fm, "logvar": -torch.log(P), "latent": fb["latent"]}
        # mass: precision fusion in log space, returned as raw mean/logvar (delta method)
        mb = typed[Prop.MASS]
        lmb = torch.log(mb["mean"].clamp(min=1e-3))
        lvb = mb["logvar"] - 2 * lmb
        qb = torch.exp(-lvb)
        qk = torch.exp(-pe["lm_logvar"]) * okf
        Q = qb + qk.sum(1)
        lm = (lmb * qb + (pe["lm_mean"] * qk).sum(1)) / Q
        mm = torch.exp(lm)
        typed[Prop.MASS] = {"mean": mm, "logvar": -torch.log(Q) + 2 * lm, "latent": mb["latent"]}
        tokens = tokens.clone()
        for p in TYPED_PROPS:
            tokens[:, slots["typed_index"][p]] = self._encode_typed(p, typed[p])
        slots = dict(slots)
        slots["typed"] = typed
        slots["push_evidence"] = pe
        return tokens, mask, slots


# ----------------------------------------------------------------------------
# recurrent core
# ----------------------------------------------------------------------------


class RecurrentCore(nn.Module):
    """shared-weight set-attention blocks applied `passes` times (with input injection).
    passes refine the belief; they do not advance physical time."""

    def __init__(self, d=128, n_blocks=2, heads=4, ff=2):
        super().__init__()
        self.d = d
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(n_blocks)])
        self.inject = nn.Linear(d, d)
        nn.init.zeros_(self.inject.weight)
        nn.init.zeros_(self.inject.bias)
        self.ln = nn.LayerNorm(d)

    def forward(self, tokens, mask, passes: int):
        x = tokens
        e = self.inject(tokens)
        for _ in range(int(passes)):
            x = x + e
            for blk in self.blocks:
                x = blk(x, mask)
        return self.ln(x)


class UntiedCore(nn.Module):
    """compute-matched untied control: n_blocks*ref_passes distinct blocks applied once
    (passes argument ignored; cost equals the tied core at ref_passes)."""

    def __init__(self, d=128, n_blocks=2, ref_passes=4, heads=4, ff=2):
        super().__init__()
        self.d = d
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(n_blocks * ref_passes)])
        self.ln = nn.LayerNorm(d)
        self.ref_passes = ref_passes

    def forward(self, tokens, mask, passes: int):
        x = tokens
        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln(x)


# ----------------------------------------------------------------------------
# readout head (common across variants)
# ----------------------------------------------------------------------------


def fourier(h, n=8):
    f = torch.exp(torch.linspace(math.log(0.5), math.log(16.0), n, device=h.device))
    a = h[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a), h[..., None]], -1)


class StateHead(nn.Module):
    def __init__(self, d=128, n_layers=2):
        super().__init__()
        self.d = d
        self.act = mlp(ACTION_DIM, d, d)
        self.hor = mlp(17, d, d)
        self.aux_q = nn.Parameter(torch.randn(2, d) * 0.02)  # friction, mass
        self.target_emb = nn.Embedding(2, d)
        self.cross = nn.ModuleList([CrossBlock(d) for _ in range(n_layers)])
        self.out = nn.Linear(d, 2 * STATE_DIM + N_CONTACT)
        self.aux_out = nn.Linear(d, 2)
        # target scaler (raw = norm*std + mean), fit on TRAIN data
        self.register_buffer("y_mean", torch.zeros(len(HORIZONS), STATE_DIM))
        self.register_buffer("y_std", torch.ones(len(HORIZONS), STATE_DIM))
        self.register_buffer("horizons", torch.tensor(HORIZONS, dtype=torch.float32))

    def scaler_for(self, horizons):
        """per-horizon scaler; interpolate if query horizons differ from HORIZONS."""
        H = horizons.shape[1]
        if H == len(HORIZONS):
            return self.y_mean[None].expand(horizons.shape[0], -1, -1), self.y_std[None].expand(horizons.shape[0], -1, -1)
        idx = torch.bucketize(horizons.to(_fdt()), self.horizons).clamp(0, len(HORIZONS) - 1)
        return self.y_mean[idx], self.y_std[idx]

    def forward(self, tokens, mask, query: QueryTensors, slots):
        B, M, d = tokens.shape
        hz = query.horizons.to(_fdt())
        H = hz.shape[1]
        a = self.act(query.action.to(_fdt()))
        qh = a[:, None] + self.hor(fourier(hz))
        q = torch.cat([qh, (self.aux_q[None] + a[:, None]).expand(B, -1, -1)], 1)
        te = slots.get("token_entity") if slots else None
        if te is not None:
            kv = tokens + self.target_emb((te == query.target.long()[:, None]).long())
        else:
            kv = tokens
        for blk in self.cross:
            q = blk(q, kv, mask)
        o = self.out(q[:, :H])
        mu_n = o[..., :STATE_DIM]
        lv_n = soft_clamp(o[..., STATE_DIM : 2 * STATE_DIM], -9.0, 5.0)
        contact = o[..., 2 * STATE_DIM :]
        ym, ys = self.scaler_for(hz)
        mean = mu_n * ys + ym
        logvar = lv_n + 2 * torch.log(ys)
        ax = self.aux_out(q[:, H:])  # (B,2,2)
        fr_mean = 1.5 * torch.sigmoid(ax[:, 0, 0])
        fr_lv = soft_clamp(ax[:, 0, 1] - 2.0, -12.0, 4.0)
        ms_mean = torch.exp(soft_clamp(ax[:, 1, 0], -4.0, 3.0))
        ms_lv = soft_clamp(ax[:, 1, 1] - 2.0, -12.0, 4.0)
        typed = (slots or {}).get("typed") or {}
        if Prop.FRICTION in typed:  # structured: the explicit typed slot IS the estimate
            fr_mean, fr_lv = typed[Prop.FRICTION]["mean"], typed[Prop.FRICTION]["logvar"]
        if Prop.MASS in typed:
            ms_mean, ms_lv = typed[Prop.MASS]["mean"], typed[Prop.MASS]["logvar"]
        return ModelOutput(
            mean=mean,
            logvar=logvar,
            friction_mean=fr_mean,
            friction_logvar=fr_lv,
            mass_mean=ms_mean,
            mass_logvar=ms_lv,
            contact_logits=contact,
            slots=slots,
        )


# ----------------------------------------------------------------------------
# full model
# ----------------------------------------------------------------------------


class PhysMemModel(nn.Module):
    def __init__(self, encoder, core, head, variant="", physics=False, passes_train=(1, 2, 4)):
        super().__init__()
        self.encoder = encoder
        self.core = core
        self.head = head
        self.variant = variant
        self.physics = physics
        self.passes_train = tuple(passes_train)

    @property
    def d(self):
        return self.encoder.d

    def set_target_scaler(self, mean, std):
        self.head.y_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.head.y_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-4))

    def encode(self, events: EventTensors, query: QueryTensors, edge_lesion=None):
        return self.encoder(events, query, edge_lesion=edge_lesion)

    def refine(self, tokens, mask, passes: int):
        return self.core(tokens, mask, passes)

    def readout(self, tokens, mask, query: QueryTensors, slots) -> ModelOutput:
        return self.head(tokens, mask, query, slots)

    def forward(self, events: EventTensors, query: QueryTensors, passes: int = 4) -> ModelOutput:
        tokens, mask, slots = self.encode(events, query)
        tokens = self.refine(tokens, mask, passes)
        out = self.readout(tokens, mask, query, slots)
        out.cost = {"passes": int(passes), "tokens": int(tokens.shape[1])}
        return out

    def intervene(self, tokens, mask, slots, prop, mean, logvar):
        """overwrite a typed estimate slot and re-encode its token (structured only)."""
        prop = Prop(int(prop))
        if prop not in slots.get("typed", {}):
            raise ValueError(f"model {self.variant} has no typed slot for {prop.name}")
        typed = {p: dict(v) for p, v in slots["typed"].items()}
        B = tokens.shape[0]
        typed[prop]["mean"] = torch.as_tensor(mean, dtype=tokens.dtype, device=tokens.device).expand(B).clone()
        typed[prop]["logvar"] = torch.as_tensor(logvar, dtype=tokens.dtype, device=tokens.device).expand(B).clone()
        new_tok = self.encoder._encode_typed(prop, typed[prop])
        tokens = tokens.clone()
        tokens[:, slots["typed_index"][prop]] = new_tok
        s2 = dict(slots)
        s2["typed"] = typed
        return tokens, s2

    def freeze(self, encoder=True, head=True, core=False):
        for mod, flag in ((self.encoder, encoder), (self.head, head), (self.core, core)):
            for p in mod.parameters():
                p.requires_grad_(not flag)
        return self


def count_params(m: nn.Module, trainable=True):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))


VARIANTS = {
    # name: (structured, physics, untied)
    "b00": (False, False, False),
    "b10": (True, False, False),
    "b01": (False, True, False),
    "b11": (True, True, False),
    "b10u": (True, False, True),
    "b11u": (True, True, True),
    "b00u": (False, False, True),
    "b11p": (True, True, False),  # post-hoc: + per-push friction/mass evidence summarizer (PushSummaryStructuredEncoder)
}


def _generic_hidden_for(d, target_params, n_latents=16):
    """choose generic encoder event-MLP width so that generic ~ structured params."""
    best = None
    for h in range(d, 8 * d + 1, 16):
        n = count_params(GenericMemoryEncoder(d, n_latents=n_latents, ff_hidden=h))
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (h, n)
    return best[0]


_HCACHE = {}


def build_model(variant: str, d: int = 128, passes_train=(1, 2, 4), core: Optional[nn.Module] = None, n_blocks=2):
    structured, physics, untied = VARIANTS[variant]
    if structured and variant.endswith("p"):
        enc = PushSummaryStructuredEncoder(d)
    elif structured:
        enc = StructuredMemoryEncoder(d)
    else:
        if d not in _HCACHE:
            _HCACHE[d] = _generic_hidden_for(d, count_params(StructuredMemoryEncoder(d)))
        enc = GenericMemoryEncoder(d, ff_hidden=_HCACHE[d])
    if core is None:
        core = UntiedCore(d, n_blocks, ref_passes=4) if untied else RecurrentCore(d, n_blocks)
    head = StateHead(d)
    return PhysMemModel(enc, core, head, variant=variant, physics=physics, passes_train=passes_train)
