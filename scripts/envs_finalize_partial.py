"""Finalize a generation directory that was stopped early: manifest from the shards that exist.
    python scripts/envs_finalize_partial.py <dir> <plan.json> <new_dir_name> [env_version]
Renames <dir> -> <parent>/<new_dir_name>, writes manifest.json (planned vs completed per robot/family/split,
sha256 + bytes + seeds per shard) and manifest.sha256. Buffered (<50 per key) episodes of a killed run are lost."""
import collections
import glob
import gzip
import hashlib
import json
import os
import pickle
import sys
import time

src, plan, new = sys.argv[1], sys.argv[2], sys.argv[3]
env_version = sys.argv[4] if len(sys.argv) > 4 else "unknown"
dst = os.path.join(os.path.dirname(src.rstrip("/")), new)
os.rename(src, dst)
planned = collections.Counter((t["robot"], t["family"], t["split"]) for t in json.load(open(plan)))
files, done = {}, collections.Counter()
stats = collections.defaultdict(lambda: dict(n=0, succ=0, errors=0, viol=0))
for path in sorted(glob.glob(f"{dst}/**/*.pkl.gz", recursive=True)):
    recs = pickle.load(gzip.open(path))
    for r in recs:
        t = r["task"]
        done[(t["robot"], t["family"], t["split"])] += 1
        s = stats[f'{t["robot"]}/{t["family"]}']
        s["n"] += 1
        s["succ"] += int(r["success"])
        s["errors"] += int(r["error"] is not None)
        s["viol"] += int(r["final_metrics"].get("integrity_violations", 0) or 0)
    files[path] = dict(sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), n=len(recs),
                       bytes=os.path.getsize(path), seeds=[r["task"]["seed"] for r in recs])
man = dict(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), partial=True, env_version=env_version,
           planned=sum(planned.values()), completed=sum(done.values()),
           planned_by_robot=dict(collections.Counter({k[0]: v for k, v in planned.items()} and
                                                     sum((collections.Counter({k[0]: v}) for k, v in planned.items()),
                                                         collections.Counter()))),
           completed_by_robot=dict(sum((collections.Counter({k[0]: v}) for k, v in done.items()), collections.Counter())),
           files=files, families={k: dict(v, success_rate=v["succ"] / max(v["n"], 1)) for k, v in stats.items()},
           bytes_total=sum(f["bytes"] for f in files.values()),
           note="stopped early on lead's decision; episodes buffered in the parent (<50 per key) were lost")
old = json.load(open(os.path.join(dst, "manifest.json"))) if os.path.exists(os.path.join(dst, "manifest.json")) else {}
old.setdefault("runs", []).append(man)
json.dump(old, open(os.path.join(dst, "manifest.json"), "w"), indent=1)
h = hashlib.sha256(open(os.path.join(dst, "manifest.json"), "rb").read()).hexdigest()
open(os.path.join(dst, "manifest.sha256"), "w").write(f"{h}  manifest.json\n")
print(dst, "planned", man["planned"], "completed", man["completed"], man["completed_by_robot"],
      round(man["bytes_total"] / 1e9, 2), "GB", h)
