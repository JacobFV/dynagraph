import sys, math
import numpy as np, mujoco
from physmem.v2.api import SkillCall
from physmem.embodiments import testbench as TB
from physmem.skills import SkillContext, run_skill, ObjectEstimate
robot, cl, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(300 + seed + int(cl * 1e5))
B = TB.BENCH[robot]; (x0, x1), (y0, y1) = B['ws']; ph = (0.01, 0.01, 0.03)
peg = (x0 + 0.3 * (x1 - x0), y0 + 0.3 * (y1 - y0) + rng.uniform(-0.02, 0.02), B['table_z'] + ph[2])
hole = np.array([x0 + 0.7 * (x1 - x0) + rng.uniform(-0.02, 0.02), y0 + 0.7 * (y1 - y0), B['table_z'] + 0.03])
def extra(sp, b):
    TB.add_box(sp, 'peg', peg, ph, 0.05, friction=0.4); TB.add_hole_fixture(sp, 'hole', hole, ph[0] + cl / 2)
spec, emb, _ = TB.make_scene(robot, extra=extra); m, d = TB.compile_scene(spec, emb)
eff = next(iter(emb.eff)); log = []
def hook(m_, d_):
    if int(round(d_.time / 0.002)) % 50 == 0:
        pp, pq = TB.body_pose(m_, d_, 'peg'); R = TB.np.zeros(9); mujoco.mju_quat2Mat(R, pq)
        tilt = math.degrees(math.acos(min(1, R[8])))
        c = ctx.ctrl.st[eff]
        e_ = emb.eff[eff]; qn = d_.qpos[e_['qadr']]; jl = [(round(qn[k] - m_.jnt_range[m_.dof_jntid[e_['dofs'][k]]][0], 2), round(m_.jnt_range[m_.dof_jntid[e_['dofs'][k]]][1] - qn[k], 2)) for k in range(len(qn))]
        if c['mode'] == 'force' and abs(d_.time - 9.5) < 0.001:
            f6 = np.zeros(6)
            for k in range(d_.ncon):
                mujoco.mj_contactForce(m_, d_, k, f6)
                print('CON', m_.geom(d_.contact.geom1[k]).name, m_.body(m_.geom_bodyid[d_.contact.geom1[k]]).name, '|', m_.geom(d_.contact.geom2[k]).name, m_.body(m_.geom_bodyid[d_.contact.geom2[k]]).name, f6[:3].round(2), round(d_.contact.dist[k], 5))
        log.append((round(d_.time, 2), c['mode'], c.get('tripped'), min(min(a, b) for a, b in jl), c.get('touched'), np.round(c.get('a_lin', np.zeros(3)), 3).tolist(), round(float(emb.tcp_vel(m_, d_, eff)[0][2]), 4), c['kw'].get('axis'), c['kw'].get('approach_speed'), round(c.get('fi', 0), 3), round(c.get('alpha', 1), 2), (pp - hole).round(4).tolist(), round(tilt, 2), emb.tcp_pose(d_, eff)[0].round(4).tolist(), c['ft'].round(1).tolist()))
ctx = SkillContext(m, d, emb, on_step=hook); ctx.wait(0.2)
from physmem.skills import primitives as PR
_mv = PR._move
def mv(ctx_, eff_, pts, v, *a, **k):
    t0 = d.time; st = _mv(ctx_, eff_, pts, v, *a, **k)
    pl = ctx_.ctrl.st[eff_]['plans']
    print('MOVE', round(t0, 3), '->', round(d.time, 3), st, [np.round(x, 4).tolist() for x in pts], 'v', v, 'plans', [(round(x.t0, 3), round(x.duration, 3), [np.round(P_, 3).tolist() for P_ in x.P]) for x in pl])
    return st
PR._move = mv
est = TB.estimate(m, d, 'peg', ph, rng, pos_sd=0.001, yaw_sd=0.01)
r1 = run_skill(ctx, SkillCall('pick', dict(speed=0.3, depth=0.025, lift=0.08), target='peg', effector=eff), est)
print('pick', r1.status, r1.log)
hole_est = ObjectEstimate(pos=hole + np.r_[rng.normal(0, 0.001, 2), 0], half_size=(ph[0] + cl / 2,) * 2 + (0.015,))
r2 = run_skill(ctx, SkillCall('insert', dict(depth=0.02, search_r=0.004), target='hole', effector=eff), hole_est)
print('insert', r2.status, r2.log)
for l in log[int(sys.argv[4]) if len(sys.argv) > 4 else -60::3]: print(l)
