"""suction hold bench (gantry_vacuum): one carton on a table, pick (oracle-free skill), lift, transfer 0.6 m, place.
grid: mass x lip friction (slick) x suction level x max_acc. Reports held-through-transfer and where it was lost.
python scripts/robots_suction_bench.py [robot]"""
import json, sys, itertools
import numpy as np, mujoco
import physmem.embodiments as E
from physmem.v2.api import SkillCall
from physmem.skills import SkillContext, run_skill, ObjectEstimate

robot = sys.argv[1] if len(sys.argv) > 1 else 'gantry_vacuum'
half = (0.14, 0.105, 0.10)
rows = []
for mass, mu, level, acc in itertools.product((5.0, 15.0, 22.0), (0.9, 0.2), (0.6, 1.0), (0.8, 2.0, 4.0)):
    spec = mujoco.MjSpec(); E.scene_defaults(spec, timestep=0.002)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.9, 0.5, 0.2], pos=[0, 0, 0.2], friction=[0.5, 0.005, 0.0001])
    b = spec.worldbody.add_body(name='carton', pos=[-0.3, 0, 0.4 + half[2] + 0.001])
    b.add_freejoint()
    g = b.add_geom(name='carton', type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), mass=mass, friction=[mu, 0.005, 0.0001], condim=4)
    emb = E.make(robot, prefix='r0')
    emb.build(spec, ((0, 0, 1.5), (1, 0, 0, 0)))
    m = spec.compile(); d = mujoco.MjData(m); emb.bind(m, d); emb.reset(m, d)
    ctx = SkillContext(m, d, emb); ctx.wait(0.3)
    emb.eff['hand']['suction_level'] = level
    est = ObjectEstimate(pos=tuple(d.xpos[m.body('carton').id]), half_size=half)
    r1 = run_skill(ctx, SkillCall('pick', dict(speed=0.5, lift=0.1, lift_speed=0.1, max_acc=acc), target='carton', effector='hand'), est)
    lost = None if r1.ok else 'pick:' + r1.status
    if r1.ok:
        r2 = run_skill(ctx, SkillCall('place', dict(x=0.3, y=0.0, z=0.4, speed=0.8, max_acc=acc), effector='hand'))
        cz = d.xpos[m.body('carton').id]
        ok = abs(cz[0] - 0.3) < 0.03 and abs(cz[2] - 0.4 - half[2]) < 0.01
        lost = None if ok else 'transfer/place:' + r2.status
    F_v = level * sum(m.actuator_gainprm[i, 0] for i in emb.act_ids if m.actuator(i).name.startswith('r0/suction'))
    rows.append(dict(mass=mass, mu=mu, level=level, max_acc=acc, ok=lost is None, lost=lost, F_vac_N=round(F_v, 1),
                     mg_N=round(mass * 9.81, 1), shear_cap_N=round(mu * max(0, F_v - mass * 9.81), 1), m_a_N=round(mass * acc, 1)))
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(f'results/v2/robots/suction_bench_{robot}.json', 'w'), indent=1)
