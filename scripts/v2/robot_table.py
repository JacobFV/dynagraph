"""MODEL-V2: real-robot paired table from results/v2/closed_loop/v2_tab_<tag>_<robot>_<family>.json ->
markdown: per robot x family, heuristic success and, per policy, Δ success vs heuristic [95% paired bootstrap CI],
effort ratio (geometric mean of per-episode effort ratios) and the policy's deviation rate.
usage: robot_table.py <tag> [out.md] [trust.json]"""
import glob, json, os, sys
import numpy as np
tag = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else f"results/v2/closed_loop/robot_table_{tag}.md"
rng = np.random.default_rng(0)
POL = ["planner_typed", "planner_residual", "planner_residual_gated", "planner_generic", "planner_oracle(labelled)"]
TRUST = json.load(open(sys.argv[3]))["families"] if len(sys.argv) > 3 else None
rows = ["| robot | family | n | heuristic succ | " + " | ".join(p.replace("planner_", "") for p in POL) + " |",
        "|---|---|---|---|" + "---|" * len(POL)]
summary = {}
for f in sorted(glob.glob(f"results/v2/closed_loop/v2_tab_{tag}_*.json")):
    name = os.path.basename(f)[len(f"v2_tab_{tag}_"):-5]
    d = json.load(open(f))
    for fam, v in d.items():
        robot = name[:-(len(fam) + 1)]
        ep = dict(v["episodes"])
        if TRUST is not None and "planner_residual_gated" not in ep and "planner_residual" in ep:
            # competence gate (VAL AUC): untrusted family -> scripted action every step == heuristic episodes
            # (identical policy, identical seeded tasks); trusted -> the residual planner's episodes
            ep["planner_residual_gated"] = ep["planner_residual"] if TRUST.get(fam, {}).get("trusted") else ep["heuristic"]
        H = np.array([e["success"] for e in ep["heuristic"]], float)
        He = np.array([e["effort"] for e in ep["heuristic"]])
        cells = []
        for p in POL:
            if p not in ep:
                cells.append("–"); continue
            S_ = np.array([e["success"] for e in ep[p]], float)
            E = np.array([e["effort"] for e in ep[p]])
            dd = S_ - H
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            er = float(np.exp(np.mean(np.log(E + 1e-3) - np.log(He + 1e-3))))
            dev = np.mean([e.get("deviations", 0) / max(e["steps"], 1) for e in ep[p]]) if p == "planner_residual" else None
            cells.append(f"{dd.mean():+.2f} [{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}] / x{er:.2f}"
                         + (f" (dev {dev:.2f})" if dev is not None else ""))
            summary[f"{robot}|{fam}|{p}"] = dict(n=len(S_), d=float(dd.mean()), ci=[float(np.percentile(bs, 2.5)),
                                                 float(np.percentile(bs, 97.5))], effort_ratio=er, success=float(S_.mean()),
                                                 heuristic=float(H.mean()))
        rows.append(f"| {robot} | {fam} | {len(H)} | {H.mean():.2f} | " + " | ".join(cells) + " |")
open(out, "w").write("\n".join(rows) + "\n")
json.dump(summary, open(out.replace(".md", ".json"), "w"), indent=1)
print("\n".join(rows))
