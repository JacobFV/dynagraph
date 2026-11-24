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


from physmem.envs.rollout import EpisodeTimeout, choose, perturb, run_episode, run_with_timeout  # noqa: F401


LIMIT_S = 300


def _job(task, limit_s=None):
    """one episode with a wall-time guard: a timed-out episode ends at once (EpisodeTimeout is a BaseException, so
    skill-level `except Exception` handlers cannot swallow it) and is recorded with error='timeout', aborted=True."""
    return run_with_timeout(task, limit_s or LIMIT_S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="all")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--robot", default="probe")
    ap.add_argument("--out", default="data/v2_envs")
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--rows", default="results/v2/envs/generation.jsonl", help="per-episode results rows ('' = none)")
    ap.add_argument("--robots", default="", help="comma list; overrides --robot (multi-robot generation)")
    ap.add_argument("--matrix", default="", help="json {robot: [families]} of WORKING pairs (ROBOTS' matrix)")
    ap.add_argument("--heldout_robot", default="", help="robot held out ENTIRELY: only split=heldout_robot episodes")
    ap.add_argument("--limit_s", type=int, default=300, help="per-episode wall-time guard")
    ap.add_argument("--tasks_json", default="", help="explicit task list (TaskSpec dicts); overrides the grid")
    a = ap.parse_args()
    fams = list(FAMILIES) if a.families == "all" else a.families.split(",")
    splits = a.splits.split(",")
    robots = a.robots.split(",") if a.robots else [a.robot]
    matrix = json.load(open(a.matrix)) if a.matrix else None
    global LIMIT_S
    LIMIT_S = a.limit_s
    tasks = []
    for rob in robots:
        rfams = [f for f in fams if matrix is None or f in matrix.get(rob, [])]
        for f in rfams:
            if rob == a.heldout_robot:      # held-out robot: never in train/val/test; its own split only
                tasks += make_tasks(f, "heldout_robot", a.n, rob, a.base_seed)
                continue
            for s in splits:
                heldout = f in HELDOUT_FAMILIES.values()
                if heldout and s in ("train", "val"):
                    continue
                ss = "heldout_family" if (heldout and s == "test") else s
                tasks += make_tasks(f, ss, a.n, rob, a.base_seed)
    if a.tasks_json:
        from physmem.v2.api import TaskSpec
        tasks = [TaskSpec(**t) for t in json.load(open(a.tasks_json))]
        robots = sorted({t.robot for t in tasks})
        robots = robots if len(robots) > 1 else robots + ["_multi"]     # always <robot>/<family>/ layout
    os.makedirs(a.out, exist_ok=True)
    if a.rows:
        os.makedirs(os.path.dirname(a.rows) or ".", exist_ok=True)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    from physmem.envs import ENV_VERSION
    manifest = dict(created=created, env_version=ENV_VERSION, code_hash=code_hash(), robots=robots, heldout_robot=a.heldout_robot,
                    procs=a.procs, files={}, families={}, n_tasks=len(tasks), args=vars(a))
    rows = open(a.rows if a.rows else os.devnull, "a")
    buf, chunk_i, stats = {}, {}, {}
    CH = 50

    def flush(key, final=False):
        recs = buf.get(key, [])
        if not recs or (len(recs) < CH and not final):
            return
        rob, f, sp = key
        sub = os.path.join(a.out, f) if len(robots) == 1 else os.path.join(a.out, rob, f)
        os.makedirs(sub, exist_ok=True)
        k = chunk_i.get(key, 0)
        path = os.path.join(sub, f"{sp}_{k:04d}.pkl.gz")
        while os.path.exists(path):      # resumed run into the same --out: never overwrite an earlier shard
            k += 1
            path = os.path.join(sub, f"{sp}_{k:04d}.pkl.gz")
        chunk_i[key] = k + 1
        tmp = path + ".tmp"              # atomic: a stop mid-write never leaves a truncated shard
        with gzip.open(tmp, "wb", compresslevel=4) as fh:
            pickle.dump(recs, fh, protocol=4)
        os.replace(tmp, path)
        manifest["files"][path] = dict(sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), n=len(recs),
                                       bytes=os.path.getsize(path), seeds=[r["task"]["seed"] for r in recs])
        buf[key] = []

    t0 = time.time()
    n_done = 0
    import signal as _sig

    main_pid = os.getpid()
    state = dict(flushing=False, pending=False)

    def _stop(signum, frame):        # SIGTERM -> stop cleanly: flush every buffer + write a (partial) manifest
        if os.getpid() != main_pid:  # pool worker (pool.terminate()): just exit
            os._exit(0)
        if state["flushing"]:        # never interrupt a shard write: stop right after it
            state["pending"] = True
            return
        raise KeyboardInterrupt
    _sig.signal(_sig.SIGTERM, _stop)
    stopped = False
    try:
      with Pool(a.procs, maxtasksperchild=50) as pool:
        for r in pool.imap_unordered(_job, tasks, chunksize=1):
            if r.get("error", "") and str(r.get("error")).startswith("timeout"):
                r.setdefault("policy", r["task"]["params"].get("policy"))
            n_done += 1
            t = r["task"]
            key = (t["robot"], t["family"], t["split"])
            buf.setdefault(key, []).append(r)
            st = stats.setdefault((t["robot"], t["family"]), dict(n=0, succ=0, wall=0.0, sim=0.0, calls=0, events=0,
                                                                  errors=0, viol=0))
            m = r["final_metrics"]
            st["n"] += 1
            st["succ"] += int(r["success"])
            st["wall"] += r["wall_s"]
            st["sim"] += r["sim_t"]
            st["calls"] += len(r["steps"])
            st["events"] += r["n_events"]
            st["errors"] += int(r["error"] is not None)
            st["viol"] += int(m.get("integrity_violations", 0) or 0)
            rows.write(json.dumps(dict(run_id=f"envs-gen-{created}", model=f"policy_{r.get('policy', 'scripted')}",
                                       family=t["family"],
                                       split=t["split"], seed=t["seed"], robot=t["robot"],
                                       config_hash=manifest["code_hash"], status="observed", success=r["success"],
                                       progress=m.get("progress"), calls=len(r["steps"]), sim_t=r["sim_t"],
                                       wall_s=r["wall_s"], effort_cost=sum(s["effort"]["cost"] for s in r["steps"]),
                                       collisions=m.get("collisions"), violations=m.get("integrity_violations"),
                                       drops=m.get("drops"), error=r["error"])) + "\n")
            state["flushing"] = True
            flush(key)
            state["flushing"] = False
            if state["pending"]:
                raise KeyboardInterrupt
            if n_done % 200 == 0:
                rows.flush()
                print(f"[{n_done}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)
    except KeyboardInterrupt:
        stopped = True
        _sig.signal(_sig.SIGTERM, _sig.SIG_IGN)      # a second TERM must not interrupt the final flush
        print(f"STOPPED at {n_done}/{len(tasks)}: flushing buffers", flush=True)
    manifest["partial"] = stopped
    manifest["planned"] = len(tasks)
    for key in list(buf):
        flush(key, final=True)
    wall = time.time() - t0
    rows.close()
    for (rob, f), st in sorted(stats.items()):
        n = max(st["n"], 1)
        manifest["families"][f"{rob}/{f}"] = dict(n=st["n"], success_rate=st["succ"] / n, mean_wall_s=st["wall"] / n,
                                                 episodes_per_hour_per_proc=3600 * n / max(st["wall"], 1e-9),
                                                 mean_sim_s=st["sim"] / n, mean_calls=st["calls"] / n,
                                                 mean_events=st["events"] / n, errors=st["errors"], violations=st["viol"])
        print(f"{rob:14s} {f:16s} n={st['n']:4d} success={st['succ'] / n:.2f} wall/ep={st['wall'] / n:.1f}s "
              f"calls={st['calls'] / n:.1f} errors={st['errors']} viol={st['viol']}")
    manifest["wall_s"] = wall
    manifest["episodes"] = n_done
    manifest["episodes_per_hour_total"] = n_done / wall * 3600
    manifest["bytes_total"] = sum(v["bytes"] for v in manifest["files"].values())
    mpath = os.path.join(a.out, "manifest.json")
    old = json.load(open(mpath)) if os.path.exists(mpath) else {}
    old.setdefault("runs", []).append(manifest)
    json.dump(old, open(mpath, "w"), indent=1)
    print(f"total {n_done} episodes in {wall:.0f}s on {a.procs} procs = {manifest['episodes_per_hour_total']:.0f} eps/h, "
          f"{manifest['bytes_total'] / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
