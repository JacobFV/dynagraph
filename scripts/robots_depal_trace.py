"""trace an ORACLE-knob gantry depalletize episode: per-cup seal / lip contacts / vacuum command, carton tilt and
height, TCP acceleration; prints the window around the first 'drops' increment. (debug; evaluator-only state)"""
import json, math, sys
import numpy as np, mujoco
from physmem.envs import make_env
from physmem.v2.api import TaskSpec, SkillCall
robot, seed, heavy_p = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
env = make_env('depalletize')
env.reset(TaskSpec('depalletize', env.category, seed, robot, 'val', params=dict(heavy_p=heavy_p)))
m, d, r = env.m, env.d, env.robot
e = r.emb.eff['hand']
log, cur = [], {'c': None}
prev_v = [np.zeros(3)]
orig = env.post_physics
def post(*a, **k):
    orig(*a, **k)
    if int(round(d.time / m.opt.timestep)) % 10:
        return
    v, _ = r.emb.tcp_vel(m, d, 'hand')
    acc = (v - prev_v[0]) / (10 * m.opt.timestep); prev_v[0] = v
    row = dict(t=round(d.time, 2), drops=env.metrics_acc.get('drops', 0))
    if cur['c']:
        b = m.body(env.scene.entities[cur['c']].body).id
        R = d.xmat[b].reshape(3, 3)
        row.update(cz=round(float(d.xpos[b][2]), 3), tilt=round(math.degrees(math.acos(min(1, abs(R[2, 2])))), 1))
    sealed, cnt = r.emb.cup_seals(m, d, 'hand', e.get('_held_seal'))
    row.update(seal=''.join('1' if x else '0' for x in sealed), lips=cnt.tolist(),
               u=np.round(d.ctrl[r.emb.act_ids[e['gacts']]], 2).tolist(), mask=''.join('1' if x else '0' for x in e.get('zone_mask', [])),
               tcp=np.round(r.emb.tcp_pose(d, 'hand')[0], 3).tolist(), acc=round(float(np.linalg.norm(acc)), 2),
               F=np.round(r.emb.wrist_ft(m, d, 'hand')[0], 0).tolist())
    if False:
        b = m.body(env.scene.entities[cur['c']].body).id
        f6 = np.zeros(6)
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact.geom1[i]], m.geom_bodyid[d.contact.geom2[i]]
            if b in (b1, b2):
                mujoco.mj_contactForce(m, d, i, f6)
                o = b2 if b1 == b else b1
                if abs(f6[0]) > 5:
                    print('CONTACT', m.body(o).name, round(f6[0], 1), 'n', d.contact.frame[i][:3].round(2), 'pos', d.contact.pos[i].round(3))
    log.append(row)
env.post_physics = post
for _ in range(30):
    c = env.scripted()
    if c is None: break
    p = dict(c.params)
    if c.skill == 'pick':
        cur['c'] = c.target
        mass = float(env.scene.entities[c.target].hidden.get('mass', 0))
        heavy = mass > 8
        p.update(suction=1.0, lift_speed=0.05 if heavy else 0.15, max_acc=1.0 if heavy else 3.0)
        print('PICK', c.target, 'mass', round(mass, 1), 'size', np.round(env.est_size(c.target), 3).tolist())
    elif c.skill == 'place':
        p.update(speed=0.15, max_acc=1.0)
    n0 = len(log)
    out = env.execute(SkillCall(c.skill, p, target=c.target))
    if c.skill == 'pick':
        print('zones', out.result.log.get('zones'), 'mask', e.get('zone_mask'))
    if out.result.status == 'dropped':
        print('slick', env.scene.entities[c.target].hidden.get('slick'), 'mu_grip', env.scene.entities[c.target].hidden.get('mu_grip'))
        L = log[n0:]
        on = [k for k, x in enumerate(L) if any(v > 0 for v in x['u'])]
        if on:
            last = on[-1]
            for x in L[max(0, last - 12):last + 3]:
                print(json.dumps(x))
        else:
            print('vacuum never on;', [ (x['t'], x['seal'], x['lips'], x['tcp'][2]) for x in L[::10]])
        break
    print(c.skill, out.result.status, 'prog', round(out.metrics['progress'], 3), 'drops', out.metrics.get('drops'))
    if out.metrics.get('drops'):
        i = next(k for k, x in enumerate(log) if x['drops'])
        for x in log[max(0, i - 6):i + 2:2]:
            print(json.dumps(x))
        for ce in env.cartons():
            b = m.body(env.scene.entities[ce].body).id
            print('carton', ce, np.round(d.xpos[b], 3).tolist(), 'on_outfeed', env._on_outfeed(ce) if hasattr(env, '_on_outfeed') else None)
        print('slots', env.slots, 'outfeed region', env.region('outfeed'))
        break
    if out.done: break
