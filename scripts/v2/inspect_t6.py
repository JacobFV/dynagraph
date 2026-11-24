import glob, collections
from physmem.envs.io import load_shard
hid = collections.Counter(); mk = collections.Counter(); dam = collections.Counter(); skills = collections.Counter(); n = 0
for path in sorted(glob.glob("data/v2_envs_large/tranche6_r5/*/*/train_0000.pkl.gz"))[:60]:
    for ep in load_shard(path, as_events=False)[:3]:
        n += 1
        fam = ep["task"]["family"]
        for st in ep["steps"]:
            skills[(fam, st["call"]["skill"])] += 1
            for e, h in st["labels"].get("hidden", {}).items():
                for k in h: hid[(fam, k)] += 1
            for k in st["metrics"]: mk[k] += 1
            if st["metrics"].get("damaged"): dam[fam] += 1
        if "damaged" in ep["steps"][-1]["labels"]: dam[(fam, "labels.damaged")] += 1
print(n, "episodes"); print(sorted({k for (f, k) in hid}))
print([k for k in hid if k[1] in ("crush_n",)][:20])
print("metrics keys", sorted(mk)); print("damaged steps", dict(dam)); print(sorted({s for s in skills}))
print("task params sample", load_shard(sorted(glob.glob("data/v2_envs_large/tranche6_r5/*/*/train_0000.pkl.gz"))[0], as_events=False)[0]["task"])
