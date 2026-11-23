"""dev tool: why do scripted episodes fail? status histogram + end state per family from generated shards.
    python scripts/envs_failures.py <out_dir> [family]"""
import collections, glob, gzip, pickle, sys
root = sys.argv[1]
fam = sys.argv[2] if len(sys.argv) > 2 else "*"
for path in sorted(glob.glob(f"{root}/{fam}/*.pkl.gz")):
    for r in pickle.load(gzip.open(path)):
        if r["success"]:
            continue
        st = collections.Counter(f"{s['call']['skill']}:{s['status']}" for s in r["steps"])
        m = r["final_metrics"]
        print(r["task"]["family"], r["task"]["split"], r["task"]["seed"], "prog", round(m.get("progress", 0), 2),
              "fatal", m.get("fatal"), "drops", m.get("drops"), "calls", len(r["steps"]), dict(st))
