"""MODEL-V2 round 3: real-robot table from results/v2/closed_loop/v2_tab5_<tag>_<robot>_<family>.json.
Per robot x family: plain heuristic (scripted) and probe-then-act heuristic success / damage rate; per policy group
(typed, generic, oracle pooled over training seeds s0-s2; residual_gated; residual_gated_probe):
Δ success vs plain heuristic [95% paired bootstrap CI], Δ vs probe-then-act heuristic [CI], damage rate, effort ratio.
usage: robot_table5.py <tag> [out.md]"""
import glob, json, os, sys
import numpy as np
tag = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else f"results/v2/closed_loop/robot_table5_{tag}.md"
rng = np.random.default_rng(0)
GROUPS = {"typed": "planner_typed_s", "generic": "planner_generic_s", "oracle": "planner_oracle(labelled)_s",
          "residual_gated": "planner_residual_gated", "residual_gated_probe": "planner_residual_gated_probe"}


def pooled(ep, prefix, exact):
    keys = [k for k in ep if (k == prefix if exact else k.startswith(prefix))]
    return keys


def paired(ref, eps_lists):
    """pooled paired differences: each seed's episodes pair with the reference episodes of the same task index."""
    d, e, dm = [], [], []
    for eps in eps_lists:
        for a, b in zip(eps, ref):
            d.append(float(a["success"]) - float(b["success"]))
            e.append(np.log(a["effort"] + 1e-3) - np.log(b["effort"] + 1e-3))
            dm.append(float(a.get("damage", 0) > 0))
    d, e = np.array(d), np.array(e)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(np.exp(e.mean())), float(np.mean(dm)), len(d)


rows = ["| robot | family | heuristic succ / dmg | probe-then-act succ / dmg | " + " | ".join(GROUPS) + " |",
        "|---|---|---|---|" + "---|" * len(GROUPS)]
summary = {}
for f in sorted(glob.glob(f"results/v2/closed_loop/v2_tab5_{tag}_*.json")):
    name = os.path.basename(f)[len(f"v2_tab5_{tag}_"):-5]
    d = json.load(open(f))
    for fam, v in d.items():
        robot = name[:-(len(fam) + 1)]
        ep = v["episodes"]
        Hh, Hp = ep["heuristic"], ep.get("heuristic_probe", ep["heuristic"])
        dmg = lambda E: float(np.mean([x.get("damage", 0) > 0 for x in E]))
        sr = lambda E: float(np.mean([x["success"] for x in E]))
        cells = []
        for g, pre in GROUPS.items():
            keys = [k for k in ep if (k.startswith(pre) if pre.endswith("_s") else k == pre)]
            if not keys:
                cells.append("–"); continue
            dh, lh, hh, er, dr, n = paired(Hh, [ep[k] for k in keys])
            dp, lp, hp, _, _, _ = paired(Hp, [ep[k] for k in keys])
            sig = "**" if lh > 0 else ""
            cells.append(f"{sig}{dh:+.2f} [{lh:+.2f},{hh:+.2f}]{sig} / vs PTA {dp:+.2f} [{lp:+.2f},{hp:+.2f}] / dmg {dr:.2f} / x{er:.2f} (n={n})")
            summary[f"{robot}|{fam}|{g}"] = dict(n=n, d_heur=dh, ci_heur=[lh, hh], d_probe=dp, ci_probe=[lp, hp],
                                                 damage=dr, effort_ratio=er,
                                                 success=float(np.mean([x["success"] for k in keys for x in ep[k]])),
                                                 per_seed={k: sr(ep[k]) for k in keys})
        summary[f"{robot}|{fam}|heuristic"] = dict(success=sr(Hh), damage=dmg(Hh), n=len(Hh))
        summary[f"{robot}|{fam}|heuristic_probe"] = dict(success=sr(Hp), damage=dmg(Hp), n=len(Hp))
        rows.append(f"| {robot} | {fam} | {sr(Hh):.2f} / {dmg(Hh):.2f} | {sr(Hp):.2f} / {dmg(Hp):.2f} | " + " | ".join(cells) + " |")
open(out, "w").write("\n".join(rows) + "\n")
json.dump(summary, open(out.replace(".md", ".json"), "w"), indent=1)
print("\n".join(rows))
