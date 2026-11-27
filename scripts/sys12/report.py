"""Build results/sys12/*.md tables from the run outputs (offline System I eval, language retention, gate test, closed
loop with the pre-registered paired comparisons). usage: python scripts/sys12/report.py"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from physmem.models.v2.harness import bootstrap_ci  # noqa: E402

R = "results/sys12"
out = []


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


# ------------------------------------------------------------------ offline System I
def pool(r):
    keys = sorted({k for v in r.values() for k in v if k != "n"})
    o = {}
    for k in keys:
        num = sum(v[k] * v["n"] for v in r.values() if k in v and np.isfinite(v[k]))
        den = sum(v["n"] for v in r.values() if k in v and np.isfinite(v[k]))
        if den:
            o[k] = num / den
    return o


ARMS = [("s1r1_inkling_gated_s0", "Inkling-2-Mini core (pretrained), gated System I, 1 pass"),
        ("s1r1_inkling_ungated_s0", "Inkling-2-Mini core (pretrained), UNGATED LoRA (control i)"),
        ("s1r1_inkling_random_s0", "Inkling random-init core, gated (control ii)"),
        ("s1r1_compact_s0", "compact System I, same frozen memory (control iii-a)"),
        ("v2_matched_typed_s0", "MODEL-V2 typed, sample-matched 128k (control iii-b)"),
        ("v2_r7_typed_s0", "MODEL-V2 r7 typed s0 (2.05 M samples, reference)")]
MET = ["success_auc", "success_brier", "damage_auc", "slip_auc", "drop_auc", "cont_nll", "effort_log_rmse"]
rows = []
for run, label in ARMS:
    f = f"{R}/eval/{run}.json"
    if not os.path.exists(f):
        continue
    ev = json.load(open(f))
    for sp in ("test", "heldout_robot", "heldout_family"):
        if sp in ev:
            p = pool(ev[sp])
            rows.append((label, sp, p, sum(v["n"] for v in ev[sp].values())))
if rows:
    out.append("## Offline System I outcome prediction (fixed row subsets, <=512 rows per family, row-weighted pool)\n")
    out.append("| model | split | rows | " + " | ".join(MET) + " |")
    out.append("|---|---|---|" + "---|" * len(MET))
    for label, sp, p, n in rows:
        out.append(f"| {label} | {sp} | {n} | " + " | ".join(fmt(p.get(k)) for k in MET) + " |")
    out.append("")
    # training budget
    out.append("| run | samples seen | train s | trainable params |\n|---|---|---|---|")
    for run, label in ARMS[:4]:
        f = f"{R}/curves/{run}.json"
        if os.path.exists(f):
            i = json.load(open(f))["info"]
            out.append(f"| {label} | {i.get('samples_seen')} | {i.get('train_seconds')} | {i.get('trainable_params')} |")
    out.append("")

# ------------------------------------------------------------------ gate + language
for gname in ("gate_test.json", "gate_test_trained.json"):
    f = f"{R}/{gname}"
    if os.path.exists(f):
        g = json.load(open(f))
        out.append(f"## Gate test ({gname}; adapters: {g.get('adapters')})\n")
        out.append("| check | value |\n|---|---|")
        for k, v in g["checks"].items():
            out.append(f"| {k} | {v} |")
        for k, v in g.get("latency", {}).items():
            out.append(f"| latency {k} | {v} |")
        out.append("")
f = f"{R}/lang_eval.json"
if os.path.exists(f):
    L = json.load(open(f))
    out.append(f"## Language retention ({L['text']['source']}, {L['text']['tokens']} tokens)\n")
    out.append("| condition | ppl | KL(base‖cond) top-64 | parse format | parse skill | parse target | parse exact | "
               "parse constrained | instr acc |\n|---|---|---|---|---|---|---|---|---|")
    for n, r in L["conditions"].items():
        pa = r["parse"]
        out.append(f"| {n} | {fmt(r['ppl'], 3)} | {fmt(r['kl_to_base_top64'], 5)} | {fmt(pa['format_valid'], 2)} | "
                   f"{fmt(pa['skill_acc'], 2)} | {fmt(pa['target_acc'], 2)} | {fmt(pa['exact'], 2)} | "
                   f"{fmt(pa['constrained_acc'], 2)} | {fmt(r['instr']['acc'], 2)} ({r['instr']['n']}) |")
    out.append("")

# ------------------------------------------------------------------ closed loop
COMP = [("s12_native", "v2"), ("s12_native", "pta"), ("s12_delegated", "v2"), ("s12_delegated", "pta"),
        ("s12_native", "s12_delegated"), ("s12_native", "s1native_scripted")]
for f in sorted(glob.glob(f"{R}/closed_loop/s12r1_*.json")):
    if f.endswith(".partial.json"):
        continue
    d = json.load(open(f))
    E = d["episodes"]
    prim = "good_frac" if d["family"] == "bin_clearing" else "success"
    val = lambda e: float(e.get(prim, np.nan)) if prim == "good_frac" else float(e["success"])
    out.append(f"## Closed loop {d['cell']} ({d['family']}, {d['robot']}, {d['split']}, n = {len(E['pta'])}; primary = "
               f"{'good parts per lot' if prim == 'good_frac' else 'success'})\n")
    out.append("| policy | primary | success | damage | probes/ep | effort | steps | timeouts |\n|---|---|---|---|---|---|---|---|")
    for k, eps in E.items():
        out.append(f"| {k} | {fmt(np.nanmean([val(e) for e in eps]), 2)} | {fmt(np.mean([e['success'] for e in eps]), 2)} | "
                   f"{fmt(np.mean([e.get('damage', 0) > 0 for e in eps]), 2)} | {fmt(np.mean([e.get('probes', 0) for e in eps]), 2)} | "
                   f"{fmt(np.mean([e['effort'] for e in eps]), 0)} | {fmt(np.mean([e['steps'] for e in eps]), 1)} | "
                   f"{int(sum(e.get('timeout', 0) for e in eps))} |")
    out.append("\n| pre-registered comparison | Δ primary [95% CI] | Δ damage [95% CI] |\n|---|---|---|")
    for i, (a, b) in enumerate(COMP, 1):
        if a in E and b in E:
            x = np.array([val(e) for e in E[a]]) - np.array([val(e) for e in E[b]])
            y = np.array([e.get("damage", 0) > 0 for e in E[a]], float) - np.array([e.get("damage", 0) > 0 for e in E[b]], float)
            m, lo, hi = bootstrap_ci(x[np.isfinite(x)])
            m2, lo2, hi2 = bootstrap_ci(y)
            sig = "**" if lo > 0 or hi < 0 else ""
            out.append(f"| C{i} {a} − {b} | {sig}{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig} | {m2:+.2f} [{lo2:+.2f}, {hi2:+.2f}] |")
    dl = f.replace(".json", ".declog.jsonl")
    if os.path.exists(dl):
        D = [json.loads(l) for l in open(dl)]
        out.append("\n| policy | decisions | System II agrees with scripted | forced (1 option) | sys2 ms (median, LM calls) | "
                   "sys1 ms (median) | probe decisions |\n|---|---|---|---|---|---|---|")
        for k in sorted({x["policy"] for x in D}):
            xs = [x for x in D if x["policy"] == k]
            lm = [x["sys2_ms"] for x in xs if x.get("n_opts", 1) > 1 and x["sys2_ms"] > 0]
            s1 = [x["sys1_ms"] for x in xs if "sys1_ms" in x]
            out.append(f"| {k} | {len(xs)} | {fmt(np.mean([x['sys2'] == x['scripted'] for x in xs]), 2)} | "
                       f"{fmt(np.mean([x.get('n_opts', 1) == 1 for x in xs]), 2)} | {fmt(np.median(lm), 0) if lm else '–'} | "
                       f"{fmt(np.median(s1), 0) if s1 else '–'} | {sum(x['kind'] == 'probe' for x in xs)} |")
    out.append("")

os.makedirs(R, exist_ok=True)
open(f"{R}/tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))

# ------------------------------------------------------------------ additions: Gen3, row-wise gate, gen256, exploratory
more = []
f = f"{R}/eval/gen3_heldout_robot_aux1masked.json"
if os.path.exists(f):
    G3 = json.load(open(f))
    more.append("## Offline held-out ROBOT (Gen3, aux1 masked; measured with the Gen3 base/shoulder self-collision defect, "
                "fixed in envs-r7.1)\n")
    more.append("| model | rows | " + " | ".join(MET) + " |\n|---|---|" + "---|" * len(MET))
    for run, r in G3.items():
        p = pool(r)
        more.append(f"| {run} | {sum(v['n'] for v in r.values())} | " + " | ".join(fmt(p.get(k)) for k in MET) + " |")
    more.append("")
f = f"{R}/gate_test_rowwise.json"
if os.path.exists(f):
    g = json.load(open(f))
    more.append(f"## Gate check (d), shape-invariant row-wise experts ({g['method']})\n\n| check | value |\n|---|---|")
    more += [f"| {k} | {v} |" for k, v in g["checks"].items()]
    more.append("")
f = f"{R}/lang_eval_gen256.json"
if os.path.exists(f):
    L2 = json.load(open(f))
    more.append("## Free-form System II generation, 256-token budget (answer parsed after the thinking span)\n")
    more.append("| condition | n parse | format | skill | target | exact | constrained | instr acc (n) |\n|---|---|---|---|---|---|---|---|")
    for n, r in L2["conditions"].items():
        pa = r["parse"]
        more.append(f"| {n} | {pa['n']} | {fmt(pa['format_valid'], 2)} | {fmt(pa['skill_acc'], 2)} | {fmt(pa['target_acc'], 2)} | "
                    f"{fmt(pa['exact'], 2)} | {fmt(pa['constrained_acc'], 2)} | {fmt(r['instr']['acc'], 2)} ({r['instr']['n']}) |")
    if "gated_system2" in L2["conditions"] and "base" in L2["conditions"]:
        gs, bs = L2["conditions"]["gated_system2"]["parse_raw"], L2["conditions"]["base"]["parse_raw"]
        more.append(f"\ngated System II free-form outputs identical to base on {sum(a == b for a, b in zip(gs, bs))}/{len(gs)} items")
    more.append("")
for f in sorted(glob.glob(f"{R}/closed_loop/explore1_*.json")):
    if f.endswith(".partial.json"):
        continue
    d = json.load(open(f))
    E = d["episodes"]
    prim = "good_frac" if d["family"] == "bin_clearing" else "success"
    val = lambda e: float(e.get(prim, np.nan)) if prim == "good_frac" else float(e["success"])
    more.append(f"## EXPLORATORY rule-vs-model {d['cell']} (MODEL-V2 decision rules, fresh seeds +{d['seed_off']}, "
                f"n = {len(E['pta'])}; primary = {prim})\n")
    more.append("| policy | primary | damage | probes/ep |\n|---|---|---|---|")
    for k, eps in E.items():
        more.append(f"| {k} | {fmt(np.nanmean([val(e) for e in eps]), 2)} | {fmt(np.mean([e.get('damage', 0) > 0 for e in eps]), 2)} | "
                    f"{fmt(np.mean([e.get('probes', 0) for e in eps]), 2)} |")
    more.append("\n| comparison | Δ primary [95% CI] | Δ damage [95% CI] |\n|---|---|---|")
    for a, b in (("native_v2rule", "v2_s0"), ("native_v2rule", "v2_ens3"), ("native_v2rule", "pta"), ("v2_s0", "v2_ens3")):
        x = np.array([val(e) for e in E[a]]) - np.array([val(e) for e in E[b]])
        y = np.array([e.get("damage", 0) > 0 for e in E[a]], float) - np.array([e.get("damage", 0) > 0 for e in E[b]], float)
        m_, lo, hi = bootstrap_ci(x[np.isfinite(x)])
        m2, lo2, hi2 = bootstrap_ci(y)
        sig = "**" if lo > 0 or hi < 0 else ""
        more.append(f"| {a} − {b} | {sig}{m_:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig} | {m2:+.2f} [{lo2:+.2f}, {hi2:+.2f}] |")
    more.append("")
# median effort per closed-loop policy (the mean is dominated by rare simulator blow-ups)
for f in sorted(glob.glob(f"{R}/closed_loop/s12r1_*.json")):
    if not f.endswith(".partial.json"):
        d = json.load(open(f))
        more.append(f"median effort {d['cell']}: " + ", ".join(
            f"{k} {np.median([e['effort'] for e in v]):.0f}" for k, v in d["episodes"].items()))
open(f"{R}/tables.md", "a").write("\n".join(more) + "\n")
print("\n".join(more))
