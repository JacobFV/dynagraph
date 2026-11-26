"""MODEL-V2 round 4 table (pre-registered, docs/MODEL_V2.md 11c) from results/v2/closed_loop/v2_r7_lot_<robot>.json.
Primary: good parts per lot. Per policy: mean good parts, Δ vs always-probe PTA [95% paired bootstrap CI over lots],
Δ vs generic [CI], Δ vs never-probe [CI]; secondary: success, damage rate, probes / lot, sim time, effort ratio vs PTA,
timeouts. usage: r7_table.py [out.md] [prefix=v2_r7_lot_]  (v2_r7v2_lot_ = exploratory 11e)"""
import glob, json, os, sys
import numpy as np
out = sys.argv[1] if len(sys.argv) > 1 else "results/v2/closed_loop/robot_table_r7.md"
prefix = sys.argv[2] if len(sys.argv) > 2 else "v2_r7_lot_"
rng = np.random.default_rng(0)
ORDER = ["heuristic", "heuristic_probe", "generic", "typed_nolot", "typed", "oracle(labelled)",
         "typed_nolot_v2", "typed_v2", "oracle_v2(labelled)"]
NAMES = {"heuristic": "never-probe (plain scripted)", "heuristic_probe": "always-probe (PTA)", "generic": "generic memory",
         "typed_nolot": "typed, lot memory OFF", "typed": "typed lot memory", "oracle(labelled)": "oracle (labelled)",
         "typed_nolot_v2": "v2 crush-gated, lot memory OFF", "typed_v2": "v2 crush-gated typed lot memory",
         "oracle_v2(labelled)": "v2 crush-gated oracle (labelled)"}


def boot(d):
    d = np.asarray(d, float)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(t):
    m, lo, hi = t
    star = "**" if lo > 0 else ("" if hi >= 0 else "_")
    return f"{star}{m:+.2f} [{lo:+.2f},{hi:+.2f}]{star}"


rows, summary = [], {}
for f in sorted(x for x in glob.glob(f"results/v2/closed_loop/{prefix}*.json") if not x.endswith(".partial.json")):
    robot = os.path.basename(f)[len(prefix):-5]
    d = json.load(open(f))["bin_clearing"]
    ep = d["episodes"]
    # a timed-out episode (wall-clock guard) is a failure: 0 good parts
    G = {k: np.array([0.0 if e.get("timeout") else e.get("good_frac", np.nan) for e in v], float) for k, v in ep.items()}
    rows.append(f"\n**{robot}** ({d.get('split')}, n = {len(ep['heuristic'])} lots, tau = {d.get('tau')})\n")
    rows.append("| policy | good parts / lot | Δ vs always-probe PTA | Δ vs generic | Δ vs memory OFF | Δ vs never-probe | success | damage | Δ damage vs PTA | probes / lot | Δ probes vs PTA | median sim t (s) | effort vs PTA | timeouts |")
    rows.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        if k not in ep:
            continue
        v = ep[k]
        g = G[k]
        dP = boot(g - G["heuristic_probe"]) if k != "heuristic_probe" else None
        dG = boot(g - G["generic"]) if "generic" in G and k != "generic" else None
        dH = boot(g - G["heuristic"]) if k != "heuristic" else None
        off = {"typed": "typed_nolot", "typed_v2": "typed_nolot_v2"}.get(k)
        dM = boot(g - G[off]) if off in G else None
        D = {k2: np.array([e.get("damage", 0) > 0 for e in v2], float) for k2, v2 in ep.items()}
        Pr = {k2: np.array([e.get("probes", 0) for e in v2], float) for k2, v2 in ep.items()}
        dD = boot(D[k] - D["heuristic_probe"]) if k != "heuristic_probe" else None
        dPr = boot(Pr[k] - Pr["heuristic_probe"]) if k != "heuristic_probe" else None
        er = float(np.exp(np.mean(np.log(np.array([e["effort"] for e in v]) + 1e-3)
                                  - np.log(np.array([e["effort"] for e in ep["heuristic_probe"]]) + 1e-3))))
        row = dict(good=float(np.nanmean(g)), d_pta=dP, d_generic=dG, d_never=dH, d_memory_off=dM,
                   success=float(np.mean([e["success"] for e in v])), damage=float(np.mean([e.get("damage", 0) > 0 for e in v])),
                   probes=float(np.mean([e.get("probes", 0) for e in v])), sim_t=float(np.median([e.get("sim_t", 0) for e in v])),
                   effort_ratio_vs_pta=er, d_damage_pta=dD, d_probes_pta=dPr, timeouts=int(sum(e.get("timeout", 0) for e in v)))
        summary[f"{robot}|{k}"] = row
        rows.append(f"| {NAMES[k]} | {row['good']:.2f} | {fmt(dP) if dP else '–'} | {fmt(dG) if dG else '–'} | {fmt(dM) if dM else '–'} | "
                    f"{fmt(dH) if dH else '–'} | {row['success']:.2f} | {row['damage']:.2f} | {fmt(dD) if dD else '–'} | {row['probes']:.1f} | "
                    f"{fmt(dPr) if dPr else '–'} | "
                    f"{row['sim_t']:.0f} | x{er:.2f} | {row['timeouts']} |")
open(out, "w").write("\n".join(rows) + "\n")
json.dump(summary, open(out.replace(".md", ".json"), "w"), indent=1)
print("\n".join(rows))
