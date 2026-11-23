import sys, time
import numpy as np, mujoco
from physmem.v2.api import SkillCall
from physmem.embodiments import testbench as TB
from physmem.skills import SkillContext, run_skill

robot = sys.argv[1]; seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
for seed in seeds:
    rng = np.random.default_rng(seed)
    B = TB.BENCH[robot]
    hs = rng.uniform(*B['obj']); half = (hs, hs * rng.uniform(1, 1.3), hs * rng.uniform(1, 1.6))
    (x0, x1), (y0, y1) = B['ws']
    pos = (rng.uniform(x0, x1), rng.uniform(y0, y1), B['table_z'] + half[2])
    tgt = (rng.uniform(x0, x1), rng.uniform(y0, y1))
    def extra(spec, bench):
        TB.add_box(spec, 'obj', pos, half, rng.uniform(*B['mass']), yaw=rng.uniform(-1, 1))
    spec, emb, bench = TB.make_scene(robot, extra=extra)
    m, d = TB.compile_scene(spec, emb)
    eff = next(iter(emb.eff))
    snap = {}
    def hook(m_, d_):
        if 'img' not in snap and ctx.ctrl.st[eff].get('tripped') and len(sys.argv) > 3:
            r = mujoco.Renderer(m_, 480, 640); imgs = []
            for az in (90, 0):
                cam = mujoco.MjvCamera(); cam.lookat[:] = pos; cam.distance = 0.4; cam.azimuth = az; cam.elevation = -5
                r.update_scene(d_, cam); imgs.append(r.render())
            import imageio.v2 as iio; iio.imwrite(f'results/v2/robots/trip_{robot}_{seed}.png', np.concatenate(imgs, 1)); snap['img'] = 1
            print('trip: obj true', TB.body_pose(m_, d_, 'obj')[0].round(4), 'est', est.p.round(4), 'tcp', emb.tcp_pose(d_, eff)[0].round(4), 'ref', ctx.ctrl.reference(eff)[0].round(4), 'base q', d_.qpos[emb.base['qadr']].round(4) if emb.base is not None else None)
            F = ctx.ctrl.st[eff]['ft']; print('trip F', F.round(2), [(m_.geom(d_.contact.geom1[k]).name, m_.body(m_.geom_bodyid[d_.contact.geom1[k]]).name, m_.geom(d_.contact.geom2[k]).name, m_.body(m_.geom_bodyid[d_.contact.geom2[k]]).name) for k in range(d_.ncon)])
    ctx = SkillContext(m, d, emb, on_step=hook)
    ctx.wait(0.3)
    est = TB.estimate(m, d, 'obj', half, rng)
    t0 = time.time()
    r1 = run_skill(ctx, SkillCall('pick', dict(speed=0.4), target='obj', effector=eff), est)
    r2 = run_skill(ctx, SkillCall('place', dict(x=tgt[0], y=tgt[1], z=B['table_z'], speed=0.4), target='obj', effector=eff)) if r1.ok else None
    ctx.wait(0.3)
    p, _ = TB.body_pose(m, d, 'obj')
    err = float(np.linalg.norm(p[:2] - np.array(tgt)))
    print(robot, seed, 'pick', r1.status, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r1.log.items() if k not in ('peaks',)},
          'place', None if r2 is None else r2.status, 'err_xy', round(err, 4), 'z', round(p[2] - B['table_z'] - half[2], 4),
          'sim', round(d.time, 2), 'wall', round(time.time() - t0, 2), 'integ', TB.integrity(m, d))
