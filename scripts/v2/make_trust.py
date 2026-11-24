"""val eval log -> trust json (deviate only where VAL success AUC >= 0.8). usage: make_trust.py <log> <out> <ckpt name>"""
import json, re, sys
t = {}
for line in open(sys.argv[1]):
    if line.startswith("val:model"):
        fam = line.split()[1]
        m = re.search(r"'success_auc': ([-0-9.e]+|nan)", line)
        auc = float(m.group(1)) if m and m.group(1) != "nan" else 0.0
        t[fam] = dict(val_auc=auc, trusted=auc >= 0.8)
json.dump(dict(rule=f"deviate from scripted only if VAL success AUC >= 0.8 ({sys.argv[3]})", families=t), open(sys.argv[2], "w"), indent=1)
print({k: (round(v["val_auc"], 2), v["trusted"]) for k, v in t.items()})
