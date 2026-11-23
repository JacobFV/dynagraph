"""Batch headless generation of v2 environment episodes with the scripted heuristic baseline policy.

    python scripts/envs_generate.py --families all --splits train,val,test --n 8 --procs 6 --robot probe \
        --out data/v2_envs

Per (family, split): episodes are generated deterministically from make_tasks(...) root seeds (split by root seed
BEFORE anything else), one process per episode, results written as gzip-pickled shards:
  <out>/<family>/<split>.pkl.gz = list of episode dicts {task, events (all ObservationEvents, actor-legal),
  steps: [{call, result(ok,status,effort,duration), done, success, metrics, labels}], final_labels, wall_s}
and <out>/manifest.json with counts, seeds, sha256 per file, code hash, and episodes/hour per family.
Every episode outcome also goes to results/v2/envs/generation.jsonl (one row per episode, status=observed).
"""
import argparse
import gzip
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict
from multiprocessing import Pool

import numpy as np

from physmem.envs import FAMILIES, make_env
from physmem.envs.splits import HELDOUT_FAMILIES, make_tasks


def code_hash():
    h = hashlib.sha256()
    root = os.path.join(os.path.dirname(__file__), "..", "src", "physmem", "envs")
    for f in sorted(os.listdir(root)):
        if f.endswith(".py"):
            h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()[:16]


def run_episode(task, keep_events=True):
    t0 = time.time()
    env = make_env(task.family)
    ev0 = env.reset(task)
    steps = []
    out = None
    err = None
    try:
        while True:
            call = env.scripted()
            if call is None:
                break
            out = env.execute(call)
            steps.append(dict(call=asdict(call), ok=out.result.ok, status=out.result.status,
                              effort=out.result.effort.as_dict(), duration_s=out.result.duration_s, done=out.done,
                              success=out.success, metrics=out.metrics, labels=out.labels))
            if out.done:
                break
    except Exception as e:      # keep failures
        err = repr(e)
    success = bool(env.success()) if err is None else False
    rec = dict(task=asdict(task), episode_id=env.episode_id, events=env.events_all if keep_events else None,
               n_events=len(env.events_all), steps=steps, success=success, sim_t=env.t, wall_s=time.time() - t0,
               final_metrics=(out.metrics if out else {}), error=err, goal=env.goal())
    return rec


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _job(task, limit_s=300):
    """one episode with a wall-time guard: a hung episode is recorded as an error row, never silently dropped."""
    import signal
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(limit_s)
    try:
        return run_episode(task)
    except _Timeout:
        return dict(task=asdict(task), episode_id=f"{task.family}-{task.split}-{task.seed}", events=None, n_events=0,
                    steps=[], success=False, sim_t=0.0, wall_s=float(limit_s), final_metrics={},
                    error=f"timeout>{limit_s}s", goal={})
    finally:
        signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="all")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--robot", default="probe")
    ap.add_argument("--out", default="data/v2_envs")
    ap.add_argument("--base_seed", type=int, default=0)
    a = ap.parse_args()
    fams = list(FAMILIES) if a.families == "all" else a.families.split(",")
    splits = a.splits.split(",")
    tasks = []
    for f in fams:
        for s in splits:
            heldout = f in HELDOUT_FAMILIES.values()
            if heldout and s in ("train", "val"):
                continue
            ss = "heldout_family" if (heldout and s == "test") else s
            tasks += make_tasks(f, ss, a.n, a.robot, a.base_seed)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs("results/v2/envs", exist_ok=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        recs = pool.map(_job, tasks, chunksize=1)
    wall = time.time() - t0
    manifest = dict(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code_hash=code_hash(), robot=a.robot,
                    procs=a.procs, wall_s=wall, files={}, families={})
    rows = open("results/v2/envs/generation.jsonl", "a")
    for f in fams:
        fr = [r for r in recs if r["task"]["family"] == f]
        for s in sorted({r["task"]["split"] for r in fr}):
            sr = [r for r in fr if r["task"]["split"] == s]
            os.makedirs(os.path.join(a.out, f), exist_ok=True)
            path = os.path.join(a.out, f, f"{s}.pkl.gz")
            with gzip.open(path, "wb") as fh:
                pickle.dump(sr, fh, protocol=4)
            manifest["files"][path] = dict(sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), n=len(sr),
                                           seeds=[r["task"]["seed"] for r in sr])
        ws = np.array([r["wall_s"] for r in fr])
        manifest["families"][f] = dict(n=len(fr), success_rate=float(np.mean([r["success"] for r in fr])),
                                       mean_wall_s=float(ws.mean()), episodes_per_hour_per_proc=float(3600 / ws.mean()),
                                       mean_sim_s=float(np.mean([r["sim_t"] for r in fr])),
                                       mean_calls=float(np.mean([len(r["steps"]) for r in fr])),
                                       mean_events=float(np.mean([r["n_events"] for r in fr])),
                                       errors=sum(r["error"] is not None for r in fr),
                                       violations=int(sum(r["final_metrics"].get("integrity_violations", 0) for r in fr)))
        for r in fr:
            m = r["final_metrics"]
            rows.write(json.dumps(dict(run_id=f"envs-gen-{manifest['created']}", model="scripted_heuristic",
                                       family=f, split=r["task"]["split"], seed=r["task"]["seed"], robot=a.robot,
                                       config_hash=manifest["code_hash"], status="observed", success=r["success"],
                                       progress=m.get("progress"), calls=len(r["steps"]), sim_t=r["sim_t"],
                                       wall_s=r["wall_s"], effort_cost=sum(s["effort"]["cost"] for s in r["steps"]),
                                       collisions=m.get("collisions"), violations=m.get("integrity_violations"),
                                       drops=m.get("drops"), error=r["error"])) + "\n")
        print(f"{f:16s} n={len(fr):3d} success={manifest['families'][f]['success_rate']:.2f} "
              f"wall/ep={ws.mean():.1f}s eps/h/proc={3600 / ws.mean():.0f} calls={manifest['families'][f]['mean_calls']:.1f} "
              f"errors={manifest['families'][f]['errors']} viol={manifest['families'][f]['violations']}")
    rows.close()
    manifest["episodes_per_hour_total"] = len(recs) / wall * 3600
    mpath = os.path.join(a.out, "manifest.json")
    old = json.load(open(mpath)) if os.path.exists(mpath) else {}
    old.setdefault("runs", []).append(manifest)
    json.dump(old, open(mpath, "w"), indent=1)
    print(f"total {len(recs)} episodes in {wall:.0f}s on {a.procs} procs = {manifest['episodes_per_hour_total']:.0f} eps/h")


if __name__ == "__main__":
    main()
