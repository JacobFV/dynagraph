"""v2 tensor layer: interaction histories + entity frames -> V2Batch (actor) / V2Labels (evaluator/training only).

Storage is per FAMILY (family-homogeneous batches keep entity padding small: bulk families pad to their own N_max),
fully resident on the training device; batches are gathered with index ops (no python per-sample work).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import torch

from . import spec as S

K_MAX = 6            # max history interactions fed to the model (most recent kept)
N_ENT_MAX = 256      # entity cap per sample (bulk piles subsampled uniformly beyond this)


@dataclass
class V2Batch:
    """ACTOR-LEGAL inputs. B batch, K history, T=SEG_T, C=N_CH, N entities.
      seg (B,K,T,C) normalized segments ; seg_mask (B,K) bool ; seg_call (B,K,24) ; seg_an (B,K,N_AN) ;
      seg_age (B,K) s since that interaction ended ; seg_on_target (B,K) bool (acted on the queried entity)
      ent (B,N,ENT_DIM) ; ent_mask (B,N) ; target (B,) long index of query entity
      emb (B,EMB_DIM) embodiment vector ; call (B,24) candidate skill vector"""
    seg: torch.Tensor
    seg_mask: torch.Tensor
    seg_call: torch.Tensor
    seg_an: torch.Tensor
    seg_age: torch.Tensor
    seg_on_target: torch.Tensor
    ent: torch.Tensor
    ent_mask: torch.Tensor
    target: torch.Tensor
    emb: torch.Tensor
    call: torch.Tensor

    def to(self, device):
        return V2Batch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def expand_calls(self, calls: torch.Tensor) -> "V2Batch":
        """(B,...) batch x (B,Q,24) calls -> (B*Q) batch (memory repeated) for candidate scoring."""
        B, Q = calls.shape[:2]
        rep = lambda t: t.repeat_interleave(Q, 0)
        d = {f.name: rep(getattr(self, f.name)) for f in fields(self) if f.name != "call"}
        return V2Batch(**d, call=calls.reshape(B * Q, -1))


@dataclass
class V2Labels:
    """TRAINING / EVALUATOR ONLY (never passed to a forward):
      z (B,L) normalized latent truth ; z_mask (B,L) ; cont (B,N_CONT) raw ; cont_mask ; events (B,N_EVENTS) ;
      events_mask ; effort (B,) log(EffortMeter.cost + 1e-3) ; family (B,) long ; k (B,) step index"""
    z: torch.Tensor
    z_mask: torch.Tensor
    cont: torch.Tensor
    cont_mask: torch.Tensor
    events: torch.Tensor
    events_mask: torch.Tensor
    effort: torch.Tensor
    family: torch.Tensor
    k: torch.Tensor

    def to(self, device):
        return V2Labels(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


def seg_tensor(x: np.ndarray) -> np.ndarray:
    xr = x if x.shape[0] == S.SEG_T else S.resample_segment(x)
    return (xr / S.CH_SCALE).astype(np.float32)


def frame_tokens(frame: S.EntityFrame, rng: Optional[np.random.Generator] = None, cap: int = N_ENT_MAX):
    tok = frame.tokens()
    # relation edges -> incident relation counts appended into the spare validity column budget (sum per type)
    if len(tok) > cap:
        keep = np.r_[frame.target, np.setdiff1d(np.arange(len(tok)), [frame.target])]
        rest = keep[1:]
        if rng is not None:
            rest = rng.permutation(rest)
        # keep non-particles first, then a uniform subset of particles
        part = S.ENT_TYPES.index("particle")
        nonp = rest[frame.types[rest] != part]
        p = rest[frame.types[rest] == part]
        keep = np.r_[keep[0], nonp, p][:cap]
        tok = tok[keep]
        return tok, 0
    return tok, frame.target


# ----------------------------------------------------------------- dataset from episodes
FAMILY_IDS: dict = {}


def family_id(name: str) -> int:
    if name not in FAMILY_IDS:
        FAMILY_IDS[name] = len(FAMILY_IDS)
    return FAMILY_IDS[name]


def build_family_arrays(episodes, emb_of_robot, rng=None):
    """episodes: list of (family, robot, list of steps) where each step is dict(frame=EntityFrame, inter=Interaction).
    Returns numpy dict for one family (all episodes must share the family)."""
    rows = dict(seg=[], call=[], an=[], tend=[], on_t=[], cont=[], cont_mask=[], events=[], events_mask=[],
                effort=[], z=[], z_mask=[], ep_start=[], k=[], robot=[], ent=[], nent=[], emb=[], ep_index=[], tgt=[])
    nmax = 0
    r = 0
    for ei, (fam, robot, steps) in enumerate(episodes):
        start = r
        for k, st in enumerate(steps):
            it: S.Interaction = st["inter"]
            rows["seg"].append(seg_tensor(it.segment).astype(np.float16))
            rows["call"].append(S.named_call_vector(it.skill, it.params))
            rows["an"].append(S.analytic_features(it.segment if it.segment.shape[0] == S.SEG_T
                                                  else S.resample_segment(it.segment)))
            rows["tend"].append(it.t_end)
            rows["on_t"].append(it.target_is_query)
            # per-episode local target index by first appearance (equality only; no id semantics)
            tmap = rows.setdefault("_tmap", {}).setdefault(ei, {})
            rows["tgt"].append(tmap.setdefault(getattr(it, "target_id", "") or "", len(tmap)))
            rows["cont"].append(it.outcome_cont)
            rows["cont_mask"].append(it.outcome_cont_mask)
            rows["events"].append(it.outcome_events)
            rows["events_mask"].append(it.outcome_events_mask)
            rows["effort"].append(np.log(it.effort_cost + 1e-3))
            z, zm = S.labels_to_z(it.labels)
            rows["z"].append(z)
            rows["z_mask"].append(zm)
            rows["ep_start"].append(start)
            rows["k"].append(k)
            rows["robot"].append(robot)
            rows["emb"].append(emb_of_robot(robot))
            tok, _ = frame_tokens(st["frame"], rng)
            rows["ent"].append(tok)
            rows["nent"].append(len(tok))
            rows["ep_index"].append(ei)
            nmax = max(nmax, len(tok))
            r += 1
    n = r
    ent = np.zeros((n, nmax, S.ENT_DIM), np.float16)
    for i, t in enumerate(rows["ent"]):
        ent[i, :len(t)] = t
    out = {k: np.asarray(v) for k, v in rows.items() if k not in ("ent", "robot", "_tmap")}
    out["ent"] = ent
    out["robot"] = np.array(rows["robot"])
    return out


class FamilyStore:
    """device-resident arrays for one family + sample gathering."""

    def __init__(self, arrays: dict, name: str, device):
        self.name = name
        self.fid = family_id(name)
        self.n = len(arrays["k"])
        dv = device
        self.seg = torch.as_tensor(arrays["seg"], device=dv)                       # fp16 (n,T,C)
        self.call = torch.as_tensor(arrays["call"], device=dv)
        self.an = torch.as_tensor(arrays["an"], device=dv)
        self.tend = torch.as_tensor(arrays["tend"], device=dv, dtype=torch.float32)
        self.on_t = torch.as_tensor(arrays["on_t"], device=dv)
        self.tgt = torch.as_tensor(arrays["tgt"] if "tgt" in arrays else np.zeros(self.n, np.int64), device=dv,
                                   dtype=torch.long)
        self.cont = torch.as_tensor(arrays["cont"], device=dv)
        self.cont_mask = torch.as_tensor(arrays["cont_mask"], device=dv)
        self.events = torch.as_tensor(arrays["events"], device=dv)
        self.events_mask = torch.as_tensor(arrays["events_mask"], device=dv)
        self.effort = torch.as_tensor(arrays["effort"], device=dv, dtype=torch.float32)
        self.z = torch.as_tensor(arrays["z"], device=dv)
        self.z_mask = torch.as_tensor(arrays["z_mask"], device=dv)
        self.ep_start = torch.as_tensor(arrays["ep_start"], device=dv, dtype=torch.long)
        self.k = torch.as_tensor(arrays["k"], device=dv, dtype=torch.long)
        self.ent = torch.as_tensor(arrays["ent"], device=dv)
        self.nent = torch.as_tensor(arrays["nent"], device=dv, dtype=torch.long)
        self.emb = torch.as_tensor(arrays["emb"], device=dv)
        self.robot = arrays["robot"]
        self.ep_index = arrays["ep_index"]

    def gather(self, idx: torch.Tensor, k_override: Optional[torch.Tensor] = None, drop_hist: float = 0.0,
               gen: Optional[torch.Generator] = None):
        """rows idx (B,) -> (V2Batch, V2Labels). History = previous interactions of the same episode (most recent
        K_MAX). k_override truncates history length (e.g. evaluate with exactly j past interactions)."""
        dv = idx.device
        k = self.k[idx] if k_override is None else torch.minimum(k_override, self.k[idx])
        start = self.ep_start[idx]
        ar = torch.arange(K_MAX, device=dv)
        lo = torch.clamp(k - K_MAX, min=0)
        hidx = start[:, None] + lo[:, None] + ar[None]                    # (B,K)
        hmask = (lo[:, None] + ar[None]) < k[:, None]
        if drop_hist > 0:
            hmask = hmask & (torch.rand(hmask.shape, device=dv, generator=gen) >= drop_hist)
        hidx = torch.where(hmask, hidx, idx[:, None])                     # safe index; masked anyway
        now = self.tend[idx] - 0.0
        tq = torch.where(k > 0, self.tend[start + torch.clamp(k - 1, min=0)] + 0.5, self.tend[idx] * 0)
        age = (tq[:, None] - self.tend[hidx]).clamp(min=0) * hmask
        nmask = torch.arange(self.ent.shape[1], device=dv)[None] < self.nent[idx][:, None]
        b = V2Batch(seg=self.seg[hidx].float(), seg_mask=hmask, seg_call=self.call[hidx], seg_an=self.an[hidx],
                    seg_age=age, seg_on_target=(self.tgt[hidx] == self.tgt[idx][:, None]) & hmask, ent=self.ent[idx].float(), ent_mask=nmask,
                    target=torch.zeros_like(idx), emb=self.emb[idx], call=self.call[idx])
        lab = V2Labels(z=self.z[idx], z_mask=self.z_mask[idx], cont=self.cont[idx], cont_mask=self.cont_mask[idx],
                       events=self.events[idx], events_mask=self.events_mask[idx], effort=self.effort[idx],
                       family=torch.full_like(idx, self.fid), k=k)
        return b, lab


def single_batch(history, frame: S.EntityFrame, emb: np.ndarray, calls: np.ndarray, device="cpu",
                 now: Optional[float] = None, on_target=None, query_target: Optional[str] = None) -> V2Batch:
    """record path for closed loop: history = list of Interaction (actor copy), frame, emb vector, calls (Q,24)
    -> V2Batch with B=Q (memory repeated per candidate)."""
    hist = history[-K_MAX:]
    K = K_MAX
    seg = np.zeros((K, S.SEG_T, S.N_CH), np.float32)
    an = np.zeros((K, S.N_AN), np.float32)
    call = np.zeros((K, S.CALLN_DIM), np.float32)
    m = np.zeros(K, bool)
    age = np.zeros(K, np.float32)
    ont = np.zeros(K, bool)
    now = now if now is not None else (hist[-1].t_end + 0.5 if hist else 0.0)
    for i, it in enumerate(hist):
        xr = it.segment if it.segment.shape[0] == S.SEG_T else S.resample_segment(it.segment)
        seg[i] = xr / S.CH_SCALE
        an[i] = S.analytic_features(xr)
        call[i] = S.named_call_vector(it.skill, it.params)
        m[i] = True
        age[i] = max(now - it.t_end, 0.0)
        if on_target is not None:
            ont[i] = on_target[i]
        elif query_target is not None:
            ont[i] = (getattr(it, "target_id", "") or "") == query_target
        else:
            ont[i] = it.target_is_query
    tok, tgt = frame_tokens(frame)
    Q = len(calls)
    t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
    rep = lambda a: a.unsqueeze(0).expand(Q, *a.shape).contiguous()
    return V2Batch(seg=rep(t(seg)), seg_mask=rep(t(m, torch.bool)), seg_call=rep(t(call)), seg_an=rep(t(an)),
                   seg_age=rep(t(age)), seg_on_target=rep(t(ont, torch.bool)), ent=rep(t(tok)),
                   ent_mask=torch.ones(Q, len(tok), dtype=torch.bool, device=device),
                   target=torch.full((Q,), tgt, dtype=torch.long, device=device), emb=rep(t(emb)),
                   call=t(calls))


def arrays_hash(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = arrays[k]
        if isinstance(a, np.ndarray) and a.dtype != object:
            h.update(k.encode())
            h.update(np.ascontiguousarray(a).tobytes()[:1 << 20])
    return h.hexdigest()[:16]


def concat_arrays(parts: list) -> dict:
    """concatenate build_family_arrays outputs (same family): re-index episodes / ep_start, pad entities."""
    parts = [p for p in parts if p is not None and len(p["k"])]
    nmax = max(p["ent"].shape[1] for p in parts)
    out, row0, ep0 = {}, 0, 0
    acc = {k: [] for k in parts[0]}
    for p in parts:
        n = len(p["k"])
        for k, v in p.items():
            if k == "ent":
                e = np.zeros((n, nmax, v.shape[2]), v.dtype)
                e[:, :v.shape[1]] = v
                acc[k].append(e)
            elif k == "ep_start":
                acc[k].append(v + row0)
            elif k == "ep_index":
                acc[k].append(v + ep0)
            else:
                acc[k].append(v)
        row0 += n
        ep0 += int(p["ep_index"].max()) + 1
    return {k: np.concatenate(v) for k, v in acc.items()}
