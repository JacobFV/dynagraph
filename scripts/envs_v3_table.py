"""envs-v3 FINAL CEILING TABLE with the VALUE OF THE LATENT (lead request, 2026-11-27):
per (family, robot): success of nominal / probe-then-act / oracle, and the paired differences oracle - nominal and
oracle - PTA (same seeds) with 95 % bootstrap CIs (10k resamples over seeds), failure causes, n.

    python scripts/envs_v3_table.py <tag> [min_n]      -> results/v2/envs/ceiling_<tag>_vol.{json,md}
A family whose oracle - nominal is < 0.15 on every robot is labelled 'competence / generalization' (the hidden
property barely changes the best action), per the lead's rule.
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")


def boot(d, rng, B=10000):
    d = np.asarray(d, float)
    if len(d) == 0:
        return (float("nan"),) * 3
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(tag, min_n=0):
    from envs_ceiling import classify
    rows = [json.loads(l) for l in open(f"results/v2/envs/ceiling_{tag}_rows.jsonl")]
    by = collections.defaultdict(dict)
    for r in rows:
        r["cause"] = classify(r)
        by[(r["family"], r["robot"])].setdefault(r["policy"], {})[r["seed"]] = r
    rng = np.random.default_rng(0)
    out, lines = [], []
    fams = collections.defaultdict(list)
    for (fam, rob), P in sorted(by.items()):
        ent = dict(family=fam, robot=rob)
        for pol in ("nominal", "probe_then_act", "oracle"):
            R = list(P.get(pol, {}).values())
            ent[pol] = dict(n=len(R), success=float(np.mean([r["success"] for r in R])) if R else None,
                            causes=dict(collections.Counter(r["cause"] for r in R if r["cause"] != "success")))
        for a in ("nominal", "probe_then_act"):
            common = sorted(set(P.get("oracle", {})) & set(P.get(a, {})))
            d = [float(P["oracle"][s]["success"]) - float(P[a][s]["success"]) for s in common]
            ent[f"oracle_minus_{a}"] = dict(n=len(d), mean_ci=boot(d, rng))
        out.append(ent)
        fams[fam].append(ent)
    label = {}
    for fam, E in fams.items():
        v = [e["oracle_minus_nominal"]["mean_ci"][0] for e in E if e["oracle_minus_nominal"]["n"]]
        label[fam] = "decision-relevant" if any(x >= 0.15 for x in v) else "competence / generalization"
    f = lambda x: "-" if x is None else f"{x:.2f}"
    g = lambda m: "-" if m[0] != m[0] else f"{m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}]"
    lines.append("| family | robot | n | nominal | PTA | oracle | oracle - nominal [95% CI] | oracle - PTA [95% CI] | "
                 "oracle failures | nominal failures | label |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in out:
        cs = lambda c: ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda x: -x[1]))
        lines.append(f"| {e['family']} | {e['robot']} | {e['oracle']['n']} | {f(e['nominal']['success'])} | "
                     f"{f(e['probe_then_act']['success'])} | {f(e['oracle']['success'])} | "
                     f"{g(e['oracle_minus_nominal']['mean_ci'])} | {g(e['oracle_minus_probe_then_act']['mean_ci'])} | "
                     f"{cs(e['oracle']['causes'])} | {cs(e['nominal']['causes'])} | {label[e['family']]} |")
    json.dump(dict(tag=tag, table=out, labels=label), open(f"results/v2/envs/ceiling_{tag}_vol.json", "w"), indent=1)
    open(f"results/v2/envs/ceiling_{tag}_vol.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
