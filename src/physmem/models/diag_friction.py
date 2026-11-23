"""post-hoc diagnosis: why does the readout ignore the typed FRICTION slot? (agent MODEL, exploratory)

python -m physmem.models.diag_friction --ckpt results/checkpoints/compact-main-b11-s0.pt --dataset v2
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from physmem.models.losses import LossConfig, compute_loss, fit_loss_scales, gaussian_nll
from physmem.schema import Prop


def head_attention_share(model, tokens, mask, query, slots):
    """mean attention mass the state-horizon queries put on each token kind, per head-layer (avg heads)."""
    h = model.head
    B = tokens.shape[0]
    hz = query.horizons.float()
    H = hz.shape[1]
    from physmem.models.compact import fourier

    a = h.act(query.action.float())
    q = torch.cat([a[:, None] + h.hor(fourier(hz)), (h.aux_q[None] + a[:, None]).expand(B, -1, -1)], 1)
    te = slots["token_entity"]
    kv = tokens + h.target_emb((te == query.target.long()[:, None]).long())
    out = {}
    kind = slots["token_kind"]
    for li, blk in enumerate(h.cross):
        kvn = blk.lnk(kv)
        _, w = blk.attn(blk.lnq(q), kvn, kvn, key_padding_mask=~mask, need_weights=True, average_attn_weights=True)
        w = w[:, :H]  # state queries (B,H,M)
        for k, name in ((3, "typed_friction"), (4, "typed_mass"), (1, "prop_slots"), (2, "entity_tokens")):
            sel = (kind == k) & mask
            share = (w * sel[:, None, :].float()).sum(-1).mean().item()
            n_tok = sel.float().sum(-1).mean().item()
            out[f"L{li}:{name}"] = dict(attn_share=share, n_tokens=n_tok,
                                        uniform_share=n_tok / mask.float().sum(-1).mean().item())
        q = blk(q, kv, mask)
    return out


def main(argv=None):
    from physmem.evaluate import load_compact
    from physmem.train import fit_target_scaler, load_dataset

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/checkpoints/compact-main-b11-s0.pt")
    p.add_argument("--dataset", default="v2")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--w-friction", type=float, default=0.03)
    a = p.parse_args(argv)
    cfg = {"compact": {"dataset": a.dataset}}
    m = load_compact(a.ckpt, "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    scales = ck["scales"]
    test = load_dataset("test", cfg, "cpu")
    ev, q, lab = test.batch(torch.arange(a.n))
    rep = {"ckpt": a.ckpt, "dataset": a.dataset}
    with torch.no_grad():
        t, mk, s = m.encode(ev, q)
        tr = m.refine(t, mk, 4)
        rep["attention"] = head_attention_share(m, tr, mk, q, s)
        fm = s["typed"][Prop.FRICTION]["mean"]
        rep["slot_mean_stats"] = dict(mean=float(fm.mean()), std_across_samples=float(fm.std()),
                                      corr_with_true=float(np.corrcoef(fm.numpy(), lab.friction.numpy())[0, 1]),
                                      true_std=float(lab.friction.std()))
        # how much of the typed token is the explicit mean vs the latent: token change for mean +/-0.15
        t_lo, _ = m.intervene(t, mk, s, Prop.FRICTION, torch.full((a.n,), 0.35), s["typed"][Prop.FRICTION]["logvar"])
        t_hi, _ = m.intervene(t, mk, s, Prop.FRICTION, torch.full((a.n,), 0.65), s["typed"][Prop.FRICTION]["logvar"])
        j = s["typed_index"][Prop.FRICTION]
        rep["token_change_rel"] = float((t_hi[:, j] - t_lo[:, j]).norm(dim=-1).mean() / t[:, j].norm(dim=-1).mean())
        # same token after the core (does the change survive refinement?)
        r_lo, r_hi = m.refine(t_lo, mk, 4), m.refine(t_hi, mk, 4)
        rep["refined_token_change_rel"] = float((r_hi[:, j] - r_lo[:, j]).norm(dim=-1).mean() / tr[:, j].norm(dim=-1).mean())
        rep["refined_all_tokens_change_rel"] = float(((r_hi - r_lo).norm(dim=-1) * mk).sum() / (tr.norm(dim=-1) * mk).sum())
    # gradients into the explicit friction mean: state-NLL path vs friction-supervision path
    t, mk, s = m.encode(ev, q)
    mean = s["typed"][Prop.FRICTION]["mean"].detach().clone().requires_grad_(True)
    t2, s2 = m.intervene(t, mk, s, Prop.FRICTION, mean, s["typed"][Prop.FRICTION]["logvar"].detach())
    s2["typed"][Prop.FRICTION]["mean"] = mean  # keep graph to the leaf
    out = m.readout(m.refine(t2, mk, 4), mk, q, s2)
    total, parts = compute_loss(out, lab, q, LossConfig(physics=True), scales, m.head.y_mean, m.head.y_std)
    g_state = torch.autograd.grad(_nll(out, lab, m), mean, retain_graph=True)[0]
    fr_n = (out.friction_mean - lab.friction) / scales["friction_std"]
    lvn = out.friction_logvar - 2 * math.log(scales["friction_std"])
    f_loss = gaussian_nll(fr_n, lvn, torch.zeros_like(fr_n)).mean()
    g_fric = torch.autograd.grad(f_loss, mean, retain_graph=True)[0]
    rep["grad_abs_mean"] = dict(state_nll=float(g_state.abs().mean()), friction_nll_unweighted=float(g_fric.abs().mean()),
                                friction_nll_weighted=float(a.w_friction * g_fric.abs().mean()))
    print(json.dumps(rep, indent=1))
    return rep


def _nll(out, lab, m):
    ys, ym = m.head.y_std[None], m.head.y_mean[None]
    yn = (lab.future - ym) / ys
    mun = (out.mean - ym) / ys
    lvn = out.logvar - 2 * torch.log(ys)
    return gaussian_nll(mun, lvn, yn).mean()


if __name__ == "__main__":
    main()
