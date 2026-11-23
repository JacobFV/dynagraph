import glob, collections, json
from physmem.envs.io import load_shard
keys = collections.defaultdict(set); hid = collections.defaultdict(set); goalk = collections.defaultdict(set); fams = collections.Counter()
for path in sorted(glob.glob("data/v2_envs_large/tranche1/*/*/train_0000.pkl.gz")):
    recs = load_shard(path, as_events=False)
    for ep in recs[:3]:
        fam = ep["task"]["family"]; fams[(fam, ep["task"]["robot"])] += 1
        goalk[fam] |= set(ep["goal"].keys())
        for st in ep["steps"]:
            c = st["call"]; keys[(fam, c["skill"])] |= {k for k, v in c["params"].items()}
            for e, h in st["labels"].get("hidden", {}).items():
                hid[fam] |= set(h.keys())
            hid[fam] |= {"task_hidden:" + k for k in st["labels"].get("task_hidden", {}) }
for k in sorted(keys): print("PARAMS", k, sorted(keys[k]))
for k in sorted(hid): print("LABELS", k, sorted(hid[k]))
for k in sorted(goalk): print("GOAL", k, sorted(goalk[k]))
print(sorted(fams))
