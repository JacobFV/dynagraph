"""ORACLE-parameter depalletize run (evaluator-only knowledge of carton mass is used ONLY to pick the skill's decision
knobs: suction level, lift speed, max_acc, transport speed). Shows that a correct choice can hold 12-22 kg cartons;
the model must infer these from interaction. python scripts/robots_depal_oracle.py robot seeds [heavy_p]"""
import json, sys, time
import numpy as np
from physmem.envs import make_env
from physmem.v2.api import TaskSpec, SkillCall

robot = sys.argv[1]
seeds = [int(s) for s in sys.argv[2].split(',')]
heavy_p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
for seed in seeds:
    env = make_env('depalletize')
    env.reset(TaskSpec('depalletize', env.category, seed, robot, 'val', params=dict(heavy_p=heavy_p)))
    t0, steps, heavy_now, out, last = time.time(), [], False, None, None
    for _ in range(40):
        c = env.scripted()
        if c is None:
            break
        p = dict(c.params)
        if c.skill == 'pick':
            m = float(env.scene.entities[c.target].hidden.get('mass', 0.0))   # ORACLE (evaluator label)
            heavy_now = m > 8.0
            p.update(suction=1.0, lift_speed=0.05 if heavy_now else 0.15, max_acc=1.0 if heavy_now else 3.0)
        elif c.skill == 'place' and heavy_now:
            p.update(speed=0.15, max_acc=1.0)
        out = env.execute(SkillCall(c.skill, p, target=c.target, effector=c.effector))
        tgt = c.target if c.skill == 'pick' else last
        last = tgt
        b = env.m.body(env.scene.entities[tgt].body).id
        steps.append((c.skill, c.target, out.result.status, round(out.metrics['progress'], 3),
                      env.d.xpos[b].round(2).tolist(), round(float(np.max(np.abs(out.result.effort.as_dict()['time_at_limit_s']))), 2)))
        if out.done:
            break
    masses = [round(float(env.scene.entities[e].hidden.get('mass', 0)), 1) for e in env.cartons()]
    print(json.dumps(dict(robot=robot, seed=seed, heavy_p=heavy_p, success=bool(out and out.success),
                          progress=None if out is None else round(out.metrics['progress'], 3), drops=None if out is None else out.metrics.get('drops'),
                          masses=masses, wall=round(time.time() - t0, 1), steps=steps)), flush=True)
