"""Stop / resume / finish a large tranche without losing or redoing episodes (gb10-direct, run from the frozen copy).

    python scripts/envs_tranche_ctl.py stop    <tranche>            # SIGTERM the generator parent -> flush + partial manifest
    python scripts/envs_tranche_ctl.py resume  <tranche> <cap_s>    # remaining tasks (plan - stored seeds) into the SAME dir
    python scripts/envs_tranche_ctl.py finish  <tranche>            # background: when the generator exits COMPLETE, write
                                                                    # the aggregate manifest run + manifest.sha256
    python scripts/envs_tranche_ctl.py status  <tranche>

manifest.sha256 (which envs_status_watch.sh and MODEL's r6_pipeline.sh wait for) is written ONLY by `finish`, and only
when the last generator run is not partial and every planned task has a stored episode - a stop never looks FINAL.
Episodes are identified by (robot, family, split, seed); shards record their seeds, so a resume skips every stored
episode and reruns only the ones that were in flight. The pid of the running generator is kept in <dir>/generator.pid
(only that process is ever signalled; nothing outside ~/dynagraph* is touched).
"""
import collections
import glob
import gzip
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
LARGE = os.environ.get("ENVS_LARGE", os.path.join(HOME, "dynagraph/data/v2_envs_large"))
LOGS = os.path.join(HOME, "dynagraph/logs")
CODE = os.environ.get("ENVS_CODE", os.path.join(HOME, "dynagraph_r6"))


def paths(tr):
    d = os.path.join(LARGE, tr)
    return d, os.path.join(LARGE, "plan", f"{tr}.json"), os.path.join(d, "generator.pid")


def alive(pid):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
    except OSError:
        return False
    return "envs_generate.py" in cmd


def gen_pid(tr):
    d, _, pf = paths(tr)
    if os.path.exists(pf):
        pid = int(open(pf).read().split()[0])
        return pid if alive(pid) else None
    return None


def stored(d):
    """(robot, family, split, seed) of every episode in the shards on disk (shard paths: <rob>/<fam>/<split>_k)."""
    done, where = set(), collections.defaultdict(list)
    for path in sorted(glob.glob(f"{d}/**/*.pkl.gz", recursive=True)):
        try:
            recs = pickle.load(gzip.open(path))
        except Exception as e:           # truncated by a stop mid-write (pre-r6.1 generator): rerun its episodes
            os.rename(path, path + ".corrupt")
            print("QUARANTINED truncated shard", path, type(e).__name__)
            continue
        for r in recs:
            t = r["task"]
            k = (t["robot"], t["family"], t["split"], t["seed"])
            done.add(k)
            where[k].append(path)
    dup = {k: v for k, v in where.items() if len(v) > 1}
    if dup:
        print(f"WARNING {len(dup)} episodes stored twice, e.g.", list(dup.items())[:2])
    return done


def key(t):
    return (t["robot"], t["family"], t["split"], t["seed"])


def cmd_stop(tr):
    pid = gen_pid(tr)
    if pid is None:
        print("no running generator for", tr)
        return
    os.kill(pid, signal.SIGTERM)        # parent only: it flushes every buffer and appends a partial manifest run
    t0 = time.time()
    while alive(pid) and time.time() - t0 < 600:
        time.sleep(2)
    print("stopped" if not alive(pid) else "STILL RUNNING after 600 s", "pid", pid, f"({time.time() - t0:.0f}s)")
    cmd_status(tr)


def cmd_status(tr):
    d, plan, _ = paths(tr)
    P = json.load(open(plan))
    done = stored(d)
    rem = [t for t in P if key(t) not in done]
    m = json.load(open(os.path.join(d, "manifest.json"))) if os.path.exists(os.path.join(d, "manifest.json")) else {}
    runs = m.get("runs", [])
    print(f"{tr}: planned {len(P)}, stored {len(done & {key(t) for t in P})}, remaining {len(rem)}, runs {len(runs)}, "
          f"last run partial={runs[-1].get('partial') if runs else None}, generator pid {gen_pid(tr)}, "
          f"manifest.sha256 {'present' if os.path.exists(os.path.join(d, 'manifest.sha256')) else 'absent'}")


def cmd_resume(tr, cap_s):
    d, plan, pf = paths(tr)
    assert gen_pid(tr) is None, "generator still running"
    assert not os.path.exists(os.path.join(d, "manifest.sha256")), "already FINAL"
    P = json.load(open(plan))
    done = stored(d)
    rem = [t for t in P if key(t) not in done]
    if not rem:
        print("nothing remaining")
        return
    n = len(glob.glob(os.path.join(LARGE, "plan", f"{tr}_resume*.json")))
    rp = os.path.join(LARGE, "plan", f"{tr}_resume{n + 1}.json")
    os.makedirs(d, exist_ok=True)
    json.dump(rem, open(rp, "w"))
    log = open(os.path.join(LOGS, f"envs_large_{tr}.log"), "a")
    log.write(f"\n==== RESUME {n + 1} at {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())}: {len(rem)} remaining ====\n")
    log.flush()
    env = dict(os.environ, PYTHONPATH="src", MUJOCO_GL="egl")
    py = os.path.join(HOME, "dynagraph/.venv/bin/python")
    p = subprocess.Popen(["timeout", "-s", "TERM", str(int(cap_s)), "nice", py, "scripts/envs_generate.py",
                          "--tasks_json", rp, "--procs", os.environ.get("ENVS_PROCS", "8"), "--out", d, "--limit_s", "300"],
                         cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(50):                  # record the generator's own pid (child of timeout -> nice exec's python)
        time.sleep(0.2)
        kids = subprocess.run(["ps", "-o", "pid=", "--ppid", str(p.pid)], capture_output=True, text=True).stdout.split()
        if kids and alive(int(kids[0])):
            open(pf, "w").write(kids[0])
            break
    print("resumed", len(rem), "tasks ->", rp, "pid", open(pf).read() if os.path.exists(pf) else "?")
    subprocess.Popen([py, os.path.abspath(__file__), "finish", tr], cwd=CODE, start_new_session=True,
                     stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)


def cmd_finish(tr):
    """wait for the generator; FINAL only when the last run is complete and nothing planned is missing."""
    d, plan, _ = paths(tr)
    while gen_pid(tr) is not None:
        time.sleep(60)
    time.sleep(5)
    m = json.load(open(os.path.join(d, "manifest.json")))
    runs = [r for r in m["runs"] if not r.get("aggregate")]
    P = json.load(open(plan))
    done = stored(d)
    missing = [t for t in P if key(t) not in done]
    if runs[-1].get("partial") or missing:
        print(f"not final: partial={runs[-1].get('partial')}, missing {len(missing)}")
        return
    fams = collections.defaultdict(lambda: dict(n=0, errors=0, violations=0))
    for r in runs:
        for k, v in r.get("families", {}).items():
            for f in ("n", "errors", "violations"):
                fams[k][f] += v.get(f, 0)
    files = {}
    for r in runs:
        files.update(r.get("files", {}))
    agg = dict(aggregate=True, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               env_version=runs[0].get("env_version"), code_hash=runs[0].get("code_hash"),
               code_hashes=sorted({r.get("code_hash") for r in runs}), n_runs=len(runs), planned=len(P),
               episodes=len(done & {key(t) for t in P}), wall_s=sum(r.get("wall_s", 0) for r in runs),
               bytes_total=sum(v["bytes"] for v in files.values()), files=files, families=dict(fams), partial=False)
    m["runs"].append(agg)
    json.dump(m, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    h = hashlib.sha256(open(os.path.join(d, "manifest.json"), "rb").read()).hexdigest()
    open(os.path.join(d, "manifest.sha256"), "w").write(f"{h}  manifest.json\n")
    print("FINAL", agg["episodes"], "episodes", h)


if __name__ == "__main__":
    c, tr = sys.argv[1], sys.argv[2]
    {"stop": lambda: cmd_stop(tr), "status": lambda: cmd_status(tr), "finish": lambda: cmd_finish(tr),
     "resume": lambda: cmd_resume(tr, float(sys.argv[3]))}[c]()
