"""family x robot success matrix with REAL embodiments in the ENVS stations (scripted policy, envs_robot_smoke style).

python scripts/robots_env_matrix.py --robots panda,ur5e_2f85 --fams all --seeds 0-4 --procs 7 --out results/v2/robots/env_runs.jsonl
python scripts/robots_env_matrix.py --summarize results/v2/robots/env_runs.jsonl   # -> env_matrix.json + markdown
"""
import argparse, json, multiprocessing as mp, os, signal, sys, time, traceback


def run_one(args):
    fam, rob, seed, guard = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from physmem.envs import make_env
    from physmem.v2.api import TaskSpec
    t0 = time.time()
    row = dict(robot=rob, fam=fam, seed=seed)

    def _alarm(*_):
        raise TimeoutError(f'wall guard {guard}s')
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(guard)
    try:
        env = make_env(fam)
        env.reset(TaskSpec(fam, env.category, seed, rob, 'train'))
        steps, out, eff = [], None, 0.0
        while True:
            c = env.scripted()
            if c is None:
                break
            out = env.execute(c)
            eff += out.result.effort.cost()
            steps.append((c.skill, c.target, out.result.status, round(out.metrics.get('progress', 0), 3),
                          round(out.result.duration_s, 2)))
            if out.done:
                break
        m = out.metrics if out else {}
        row.update(success=bool(out and out.success), calls=len(steps), sim_t=round(env.t, 1),
                   progress=m.get('progress'), violations=m.get('integrity_violations'), collisions=m.get('collisions'),
                   drops=m.get('drops'), fatal=m.get('fatal'), effort_cost=round(eff, 2), steps=steps)
    except Exception as ex:
        row.update(success=False, error=f'{type(ex).__name__}: {ex}'[:300], tb=traceback.format_exc()[-600:])
    finally:
        signal.alarm(0)
    row['wall'] = round(time.time() - t0, 1)
    return row


def parse_seeds(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def summarize(path, out_json='results/v2/robots/env_matrix.json'):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    from collections import defaultdict
    cell = defaultdict(list)
    for r in rows:
        cell[(r['fam'], r['robot'])].append(r)
    fams = sorted({r['fam'] for r in rows})
    robs = []
    for r in rows:
        if r['robot'] not in robs:
            robs.append(r['robot'])
    mat = {}
    for (f, rb), rs in cell.items():
        mat.setdefault(f, {})[rb] = dict(n=len(rs), success=sum(r.get('success', False) for r in rs),
                                         progress=round(sum((r.get('progress') or 0) for r in rs) / len(rs), 2),
                                         errors=sum('error' in r for r in rs),
                                         violations=sum((r.get('violations') or 0) for r in rs),
                                         wall_mean=round(sum(r['wall'] for r in rs) / len(rs), 1),
                                         sim_mean=round(sum((r.get('sim_t') or 0) for r in rs) / len(rs), 1))
    json.dump(dict(matrix=mat, robots=robs, families=fams, source=path), open(out_json, 'w'), indent=1)
    print('| family | ' + ' | '.join(robs) + ' |')
    print('|---|' + '---|' * len(robs))
    for f in fams:
        cells = []
        for rb in robs:
            c = mat.get(f, {}).get(rb)
            cells.append('-' if c is None else f"{c['success']}/{c['n']} ({c['progress']:.2f})" + (f" E{c['errors']}" if c['errors'] else ''))
        print(f'| {f} | ' + ' | '.join(cells) + ' |')
    tot = {rb: (sum(mat[f][rb]['success'] for f in mat if rb in mat[f]), sum(mat[f][rb]['n'] for f in mat if rb in mat[f])) for rb in robs}
    print('| **total** | ' + ' | '.join(f'{a}/{b}' for a, b in tot.values()) + ' |')
    print('violations total:', sum(c['violations'] for f in mat for c in mat[f].values()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--robots', default='panda')
    ap.add_argument('--fams', default='all')
    ap.add_argument('--seeds', default='0-4')
    ap.add_argument('--procs', type=int, default=7)
    ap.add_argument('--guard', type=int, default=400)
    ap.add_argument('--out', default='results/v2/robots/env_runs.jsonl')
    ap.add_argument('--summarize', default=None)
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
        sys.exit(0)
    from physmem.envs import FAMILIES
    fams = list(FAMILIES) if a.fams == 'all' else a.fams.split(',')
    jobs = [(f, r, s, a.guard) for r in a.robots.split(',') for f in fams for s in parse_seeds(a.seeds)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with mp.get_context('fork').Pool(a.procs, maxtasksperchild=4) as pool, open(a.out, 'a') as fo:
        for row in pool.imap_unordered(run_one, jobs):
            fo.write(json.dumps(row, default=str) + '\n')
            fo.flush()
            print(row['robot'], row['fam'], row['seed'], row.get('success'), row.get('progress'), row.get('error', '')[:80],
                  row['wall'], flush=True)
