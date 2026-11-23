"""losses for compact models (agent MODEL).

- observation loss (always on): Gaussian NLL of the future state in NORMALIZED units
  (per-horizon/per-dim train scaler held by the model head).
- physics terms (physics variants b?1 only), each divided by a scale fit on TRAIN labels:
  * coulomb: on consecutive horizon pairs where both are free-slide (LabelTensors.free_slide_mask),
    |v(t2)| ~= max(0, |v(t1)| - mu_hat*g*(t2-t1)) using the model's OWN friction estimate
  * kinematic: displacement between horizons ~= trapezoid of predicted velocities
  * friction / mass recovery: Gaussian NLL of the true value under the model's typed estimate
  * contact-mode cross entropy
Predictive accuracy and residual terms are always reported separately (`parts`).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

G = 9.81


@dataclass
class LossConfig:
    physics: bool = False
    w_coulomb: float = 0.3
    w_kinematic: float = 0.3
    w_friction: float = 0.3
    w_mass: float = 0.1
    w_contact: float = 0.1

    def to_dict(self):
        return asdict(self)


def gaussian_nll(mu, logvar, y):
    return 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar) + math.log(2 * math.pi))


def _pairs(t):
    return t[:, :-1], t[:, 1:]


def physics_residuals(mean, friction, horizons, free_slide, valid):
    """raw residuals (no scaling). mean (B,H,7); friction (B,); horizons (B,H).
    returns (coulomb (B,H-1), coulomb_mask, kinematic (B,H-1,2), kin_mask)."""
    v = mean[..., 4:6]
    sp = v.norm(dim=-1)
    s1, s2 = _pairs(sp)
    h1, h2 = _pairs(horizons.float())
    dt = (h2 - h1).clamp(min=1e-4)
    pred2 = torch.clamp(s1 - friction[:, None] * G * dt, min=0.0)
    coul = s2 - pred2
    fs1, fs2 = _pairs(free_slide.bool() & valid.bool())
    cmask = fs1 & fs2
    p1, p2 = _pairs(mean[..., 0:2])
    v1, v2 = _pairs(v)
    kin = (p2 - p1) - dt[..., None] * 0.5 * (v1 + v2)
    va, vb = _pairs(valid.bool())
    return coul, cmask, kin, va & vb


def fit_loss_scales(batches, max_batches=50):
    """scales from TRAIN labels: typical speed change on free-slide pairs and typical displacement
    increment between horizons (so normalized residuals are O(1) for an uninformative predictor)."""
    ds, dp, fr, ms = [], [], [], []
    for i, (ev, q, lab) in enumerate(batches):
        if i >= max_batches:
            break
        fut = lab.future.float()
        sp = fut[..., 4:6].norm(dim=-1)
        s1, s2 = _pairs(sp)
        fs = lab.free_slide_mask.bool() & lab.future_mask.bool()
        f1, f2 = _pairs(fs)
        m = f1 & f2 & (s1 > 0.005)  # moving pairs only (at-rest pairs are trivially consistent)
        if m.any():
            ds.append((s2 - s1)[m])
        va, vb = _pairs(lab.future_mask.bool())
        p1, p2 = _pairs(fut[..., 0:2])
        mm = va & vb
        if mm.any():
            dp.append((p2 - p1)[mm].reshape(-1))
        fr.append(lab.friction.float().reshape(-1))
        ms.append(lab.mass.float().reshape(-1))
    cat = lambda xs: torch.cat(xs) if xs else torch.ones(1)
    sc = {
        "coulomb": float(cat(ds).abs().mean().clamp(min=1e-3)),
        "kinematic": float(cat(dp).abs().mean().clamp(min=1e-3)),
        "friction_std": float(cat(fr).std().clamp(min=1e-3)) if len(cat(fr)) > 1 else 0.1,
        "mass_std": float(cat(ms).std().clamp(min=1e-3)) if len(cat(ms)) > 1 else 0.5,
    }
    return sc


def compute_loss(out, lab, query, cfg: LossConfig, scales: dict, y_mean, y_std):
    """returns (total_loss, parts dict of detached floats-as-tensors).
    y_mean/y_std: (H,7) or (B,H,7) scaler from the model head."""
    fm = lab.future_mask.bool()
    y = lab.future.float()
    mu = out.mean.float()
    lv = out.logvar.float()
    ys = y_std if y_std.dim() == 3 else y_std[None]
    ym = y_mean if y_mean.dim() == 3 else y_mean[None]
    yn = (y - ym) / ys
    mun = (mu - ym) / ys
    lvn = lv - 2 * torch.log(ys)
    nll = gaussian_nll(mun, lvn, yn).mean(-1)  # (B,H)
    denom = fm.float().sum().clamp(min=1)
    nll = (nll * fm.float()).sum() / denom
    parts = {"nll": nll.detach()}
    with torch.no_grad():
        parts["mse_norm"] = ((((mun - yn) ** 2).mean(-1)) * fm.float()).sum() / denom
    coul, cm, kin, km = physics_residuals(mu, out.friction_mean.float(), query.horizons, lab.free_slide_mask, fm)
    c_loss = ((coul / scales["coulomb"]) ** 2 * cm.float()).sum() / cm.float().sum().clamp(min=1)
    k_loss = (((kin / scales["kinematic"]) ** 2).sum(-1) * km.float()).sum() / km.float().sum().clamp(min=1)
    fr_n = (out.friction_mean.float() - lab.friction.float()) / scales["friction_std"]
    fr_lvn = out.friction_logvar.float() - 2 * math.log(scales["friction_std"])
    f_loss = gaussian_nll(fr_n, fr_lvn, torch.zeros_like(fr_n)).mean()
    ms_n = (torch.log(out.mass_mean.float().clamp(min=1e-3)) - torch.log(lab.mass.float().clamp(min=1e-3)))
    # mass logvar is of the raw quantity; delta-method to log-space
    ms_lv_log = out.mass_logvar.float() - 2 * torch.log(out.mass_mean.float().clamp(min=1e-3))
    m_loss = gaussian_nll(ms_n, ms_lv_log, torch.zeros_like(ms_n)).mean()
    cl = out.contact_logits.float()
    ct = lab.contact_mode.long().clamp(0, cl.shape[-1] - 1)
    ce = F.cross_entropy(cl.reshape(-1, cl.shape[-1]), ct.reshape(-1), reduction="none").view_as(fm.float())
    ce = (ce * fm.float()).sum() / denom
    parts.update(
        coulomb=c_loss.detach(), kinematic=k_loss.detach(), friction_nll=f_loss.detach(), mass_nll=m_loss.detach(),
        contact_ce=ce.detach(), friction_abs_err=(out.friction_mean.float() - lab.friction.float()).abs().mean().detach(),
    )
    total = nll
    if cfg.physics:
        total = (
            total
            + cfg.w_coulomb * c_loss
            + cfg.w_kinematic * k_loss
            + cfg.w_friction * f_loss
            + cfg.w_mass * m_loss
            + cfg.w_contact * ce
        )
    parts["total"] = total.detach()
    return total, parts
