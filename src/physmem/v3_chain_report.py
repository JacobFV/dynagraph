"""V3 section 0 report (owner: V3-MODEL): paired bootstrap tables from results/v3/eval/*.

  python -m physmem.v3_chain_report --split test --models lbs_law_s0,lbs_law_s1,...  > results/v3/report_test.md
Every number is recomputed from the per-row / per-episode eval files; pairing is by episode (same seeds, same
edits, same goals for every model). 95% percentile bootstrap over episodes (4000 resamples).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from physmem.v3_chain_eval import RESULTS, boot_ci, summarize_causal


def ev(name):
    return os.path.join(RESULTS, "eval", name)


def row_metric(npz, k_sel=None):
    """per-episode mean final-position error (m) over boxes and the selected rows. -> dict e -> value."""
    z = np.load(npz)
    rows, fe = z["rows"], z["final_err"]
    per_row = np.nanmean(fe, 1)
    out = {}
    for (e, k), v in zip(rows, per_row):
        if (k_sel is None or k in k_sel) and np.isfinite(v):
            out.setdefault(int(e), []).append(v)
    return {e: float(np.mean(v)) for e, v in out.items()}


def paired(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    lo, hi = boot_ci(d)
    return float(d.mean()), lo, hi, len(keys)


def causal_episode(recs):
    """per-episode causal indicators for paired differences: own-box sign agreement (final) and action sign."""
    own, act, real = {}, {}, {}
    for r in recs:
        i = r["box"]
        s, m = r["sim_dx_final"][i], r["mod_dx_final"][i]
        if abs(s) > 0.002:
            own[r["e"]] = float(np.sign(s) == np.sign(m))
        dFo, dFm = r["F_or1"] - r["F_or0"], r["F_m1"] - r["F_m0"]
        if abs(dFo) > 0.25:
            act[r["e"]] = float(np.sign(dFo) == np.sign(dFm))
        real[r["e"]] = r["real_err_plain"]
    return own, act, real


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--groups", required=True,
                    help="name=run1+run2+...;name2=... (seeds pooled by averaging per episode)")
    ap.add_argument("--ref", default="LBS")
    ap.add_argument("--cond", default="full")
    ap.add_argument("--K", default="4")
    a = ap.parse_args(argv)
    groups = {}
    for g in a.groups.split(";"):
        n, rs = g.split("=")
        groups[n] = rs.split("+")
    lines = [f"# V3 §0 report, split `{a.split}`, cond `{a.cond}`, K={a.K}", ""]
    # ---- prediction
    pred = {}
    for n, rs in groups.items():
        per = []
        for r in rs:
            f = ev(f"classical_{a.split}.npz") if r == "classical" else ev(f"{r}_{a.split}_{a.cond}_K{a.K}.npz")
            if os.path.exists(f):
                per.append(row_metric(f, k_sel=(1, 2, 3)))
        if per:
            keys = set.intersection(*[set(p) for p in per])
            pred[n] = {e: float(np.mean([p[e] for p in per])) for e in keys}
    lines += ["## Final-position error (mm, rows with history k ≥ 1; per-episode mean, seeds averaged)", "",
              f"| model | mean | Δ vs {a.ref} [95% CI] | n |", "|---|---|---|---|"]
    for n, v in pred.items():
        m = np.mean(list(v.values())) * 1e3
        if n != a.ref and a.ref in pred:
            d, lo, hi, k = paired(v, pred[a.ref])
            lines.append(f"| {n} | {m:.1f} | {d * 1e3:+.1f} [{lo * 1e3:+.1f}, {hi * 1e3:+.1f}] | {k} |")
        else:
            lines.append(f"| {n} | {m:.1f} | – | {len(v)} |")
    # ---- causal
    lines += ["", "## Causal acceptance (per seed) and closed loop", ""]
    hdr = ["sign_own_final", "corr_own_final", "ratio_own_final", "sign_other_final", "sign_own_t06", "sign_own_t03",
           "sign_imp_in", "sign_action", "real_cf_edit_minus_noedit_mm", "real_cf_ci_mm", "closed_loop_err_mm",
           "closed_loop_succ15", "p_err_ontime", "p_err_late", "p_err_never", "edge_block_frac_B2_suppressed",
           "edge_block_B1_sign"]
    lines += ["| run | " + " | ".join(hdr) + " |", "|" + "---|" * (len(hdr) + 1)]
    cep = {}
    for n, rs in groups.items():
        for r in rs:
            f = ev(f"{r}_{a.split}_causal.json")
            if not os.path.exists(f):
                continue
            recs = json.load(open(f))
            s = summarize_causal(recs)
            fmt = lambda x: (f"[{x[0]:+.1f}, {x[1]:+.1f}]" if isinstance(x, list) else
                             ("–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"))
            lines.append(f"| {r} | " + " | ".join(fmt(s.get(h)) for h in hdr) + " |")
            cep.setdefault(n, []).append(causal_episode(recs))
    # paired causal / closed-loop differences vs ref (seeds pooled per episode)
    def pool(lst, i):
        keys = set.intersection(*[set(x[i]) for x in lst])
        return {e: float(np.mean([x[i][e] for x in lst])) for e in keys}
    if a.ref in cep:
        lines += ["", f"## Paired differences vs {a.ref} (episodes; seeds averaged)", "",
                  "| model | own-box sign agreement Δ | action sign agreement Δ | closed-loop error Δ (mm) |",
                  "|---|---|---|---|"]
        for n in cep:
            if n == a.ref:
                continue
            cells = []
            for i, sc in ((0, 1.0), (1, 1.0), (2, 1e3)):
                d, lo, hi, k = paired(pool(cep[n], i), pool(cep[a.ref], i))
                cells.append(f"{d * sc:+.3f} [{lo * sc:+.3f}, {hi * sc:+.3f}] (n={k})")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
    # ---- A5 anchor removal, C6 modalities, C5 depth (per run)
    lines += ["", "## Conditions / depth (per run; rows k >= 1)", "",
              "| run | cond | K | final err mm | final sd mm | cov80 final | mass abs err | friction abs err | k abs err |"
              " mass corr | friction corr |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, rs in groups.items():
        for r in rs:
            for cond in ("full", "no_anchor", "no_track", "no_proprio"):
                for K in ("1", "2", "4", "8"):
                    f = ev(f"{r}_{a.split}_{cond}_K{K}.npz")
                    if not os.path.exists(f):
                        continue
                    z = np.load(f)
                    sel = z["rows"][:, 1] >= 1
                    fe, sd, fz = z["final_err"][sel], z["final_sd"][sel], z["final_z"][sel]
                    pe, pm, pz, ps = z["p_err"][sel], z["p_mean"][sel], z["p_z"][sel], z["p_sd"][sel]
                    def corr(j):
                        m_ = pm[..., j]; ok = np.isfinite(m_)
                        t_ = (m_ - pz[..., j] * ps[..., j])[ok]
                        return np.corrcoef(m_[ok], t_)[0, 1] if ok.sum() > 10 else np.nan
                    lines.append(f"| {r} | {cond} | {K} | {np.nanmean(fe) * 1e3:.1f} | {np.nanmean(sd) * 1e3:.1f} | "
                                 f"{np.nanmean(np.abs(fz) < 1.2816):.3f} | {np.nanmean(pe[..., 0]):.3f} | "
                                 f"{np.nanmean(pe[..., 1]):.3f} | {np.nanmean(pe[..., 2]):.3f} | {corr(0):.3f} | "
                                 f"{corr(1):.3f} |")
            if os.path.exists(ev(f"{r}_{a.split}_full_K4.npz")) and os.path.exists(ev(f"{r}_{a.split}_no_anchor_K4.npz")):
                z0, z1 = np.load(ev(f"{r}_{a.split}_full_K4.npz")), np.load(ev(f"{r}_{a.split}_no_anchor_K4.npz"))
                sel = z0["rows"][:, 1] >= 1
                d = np.nanmean(z1["final_sd"][sel], 1) - np.nanmean(z0["final_sd"][sel], 1)
                ok = np.isfinite(d)
                lo, hi = boot_ci(d[ok])
                lines.append(f"| {r} | A5 Δsd(no_anchor − full) | 4 | | {d[ok].mean() * 1e3:+.2f} "
                             f"[{lo * 1e3:+.2f}, {hi * 1e3:+.2f}] | | | | | | |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
