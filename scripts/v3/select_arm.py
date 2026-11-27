"""V3 §0e pre-registered selection rule (VAL only): pick the arm-I configuration for test2.
score = number of A1..A4 passed on VAL (A1: own sign >= .85 & Spearman >= .6 & other sign >= .75; A2: t03 & t06 sign
>= .80 & impulse sign >= .75; A3: action sign >= .80; A4: CI upper < 0); ties -> arm_i2. Prints the chosen run stem."""
import json
import sys

from physmem.v3_chain_eval import summarize_causal


def score(path):
    s = summarize_causal(json.load(open(path)))
    a1 = s["sign_own_final"] >= .85 and s.get("corr_own_final", 0) >= .6 and s["sign_other_final"] >= .75
    a2 = s["sign_own_t03"] >= .80 and s["sign_own_t06"] >= .80 and s.get("sign_imp_in", 0) >= .75
    a3 = s["sign_action"] >= .80
    a4 = s["real_cf_ci_mm"][1] < 0
    return int(a1) + int(a2) + int(a3) + int(a4), dict(A1=a1, A2=a2, A3=a3, A4=a4)


cands = {"arm_i": "results/v3/eval/arm_i_s0_val_causal.json", "arm_i2": "results/v3/eval/arm_i2_s0_val_causal.json"}
sc = {k: score(p) for k, p in cands.items()}
best = max(sc, key=lambda k: (sc[k][0], k == "arm_i2"))
json.dump(dict(scores={k: v[1] for k, v in sc.items()}, chosen=best), open("results/v3/eval/arm_selection.json", "w"),
          indent=1)
print(best)
