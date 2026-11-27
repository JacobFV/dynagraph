"""MODEL-V2 round 6 (r9) table, as pre-registered in docs/MODEL_V2.md 13a.
inputs: results/v2/closed_loop/v2_r9_<family>_<robot>.json (skill-level policies, r9_eval.py) and
        results/v2_il/test/il_<IL_RUN>_m<M>__<family>__<robot>__test.json (ACT, il_eval.py), same make_tasks seeds.
usage: r9_table.py IL_RUN ENS_M [out.md]
Per cell: ABSOLUTE success within env.time_budget() for every policy, median cycle time among successes, mean cycle
time with failures at the cap, damage rate; paired Δ success [95% bootstrap CI, 4000 resamples, seed 0]:
 P1 typed_full - best scripted (PTA; force_ramp on tight_insert), P2 typed_full - ACT, P3 typed_budget - ACT;
 secondary typed - generic, typed - nomem (both regimes)."""
import glob, json, os, sys
import numpy as np

il_run, ens_m = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "results/v2/closed_loop/robot_table_r9.md"
rng = np.random.default_rng(0)
ORDER = ["heuristic", "heuristic_probe", "force_ramp", "oracle(labelled)", "ACT", "nomem_budget", "generic_budget",
         "typed_budget", "nomem_full", "generic_full", "typed_full"]
NAMES = {"heuristic": "nominal scripted", "heuristic_probe": "PTA (probe-then-act)", "force_ramp": "force-ramp",
         "oracle(labelled)": "oracle-param scripted (labelled)", "ACT": "ACT imitation (tuned on VAL)",
         "nomem_budget": "no-memory, budget-matched", "generic_budget": "generic memory, budget-matched",
         "typed_budget": "typed memory, budget-matched", "nomem_full": "no-memory, full data",
         "generic_full": "generic memory, full data", "typed_full": "typed memory, full data"}


def boot(d):
    d = np.asarray(d, float)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(t):
    if t is None:
        return "–"
    m, lo, hi = t
    star = "**" if lo > 0 else ("_" if hi < 0 else "")
    return f"{star}{m:+.2f} [{lo:+.2f},{hi:+.2f}]{star}"


rows, summ = [], {}
for f in sorted(glob.glob("results/v2/closed_loop/v2_r9_*.json")):
    d = json.load(open(f))
    fam, robot = d["family"], d["robot"]
    ep = dict(d["episodes"])
    ilf = f"results/v2_il/test/il_{il_run}_m{ens_m}__{fam}__{robot}__test.json"
    if os.path.exists(ilf):
        il = json.load(open(ilf))
        ep["ACT"] = [dict(seed=e["seed"], success=e["success"], success_in_budget=e["success"] and e["cycle_t"] <= e["budget"],
                          cycle_t=e["cycle_t"], time_budget=e["budget"], damage=e["damage"]) for e in il["episodes"]]
    seeds = [e["seed"] for e in ep["heuristic"]]
    for k, v in ep.items():
        assert [e["seed"] for e in v] == seeds, f"unpaired seeds in {f}:{k}"
    S = {k: np.array([e["success_in_budget"] for e in v], float) for k, v in ep.items()}
    best = {"tight_insert": "force_ramp", "fixture_place": "heuristic"}.get(fam, "heuristic_probe")   # 13a, from VAL
    rows.append(f"\n**{fam} / {robot}** ({d['split']}, n = {len(seeds)}, env {d['env_version']}, best scripted = {NAMES[best]})\n")
    rows.append("| policy | success (in budget) | median cycle s | mean cycle s (fail = cap) | damage | Δ vs best scripted | Δ vs ACT | Δ vs generic (same regime) | Δ vs no-memory (same regime) |")
    rows.append("|---|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        if k not in ep:
            continue
        v = ep[k]
        cyc = [e["cycle_t"] for e in v if e["success_in_budget"]]
        cap = np.array([e["cycle_t"] if e["success_in_budget"] else e["time_budget"] for e in v], float)
        reg = k.split("_")[-1] if k.endswith(("_budget", "_full")) else None
        dB = boot(S[k] - S[best]) if k != best else None
        dA = boot(S[k] - S["ACT"]) if "ACT" in S and k != "ACT" else None
        dG = boot(S[k] - S[f"generic_{reg}"]) if reg and k.startswith("typed") and f"generic_{reg}" in S else None
        dN = boot(S[k] - S[f"nomem_{reg}"]) if reg and not k.startswith("nomem") and f"nomem_{reg}" in S else None
        r = dict(success=float(S[k].mean()), median_cycle=float(np.median(cyc)) if cyc else None,
                 mean_cycle_capped=float(cap.mean()), damage=float(np.mean([e.get("damage", 0) > 0 for e in v])),
                 d_best=dB, d_act=dA, d_generic=dG, d_nomem=dN)
        summ[f"{fam}|{robot}|{k}"] = r
        rows.append(f"| {NAMES[k]} | {r['success']:.2f} | {r['median_cycle']:.1f} | {r['mean_cycle_capped']:.1f} | "
                    f"{r['damage']:.2f} | {fmt(dB)} | {fmt(dA)} | {fmt(dG)} | {fmt(dN)} |"
                    if cyc else
                    f"| {NAMES[k]} | {r['success']:.2f} | – | {r['mean_cycle_capped']:.1f} | {r['damage']:.2f} | "
                    f"{fmt(dB)} | {fmt(dA)} | {fmt(dG)} | {fmt(dN)} |")
open(out, "w").write("\n".join(rows) + "\n")
json.dump(summ, open(out.replace(".md", ".json"), "w"), indent=1)
print("\n".join(rows))
