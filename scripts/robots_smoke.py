"""smoke: build each robot alone, hold home 1 s, report sag; pad midpoint in TCP frame; one 0.3 m stroke."""
import sys, time
import numpy as np, mujoco
import physmem.embodiments as E
from physmem.embodiments.trajectory import quat2mat

names = sys.argv[1:] or [n for n in E.ROBOTS]
for name in names:
    emb = E.make(name, prefix='r0')
    spec = mujoco.MjSpec(); E.scene_defaults(spec)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1])
    emb.build(spec, ((0, 0, 0), (1, 0, 0, 0)))
    m = spec.compile(); d = mujoco.MjData(m)
    emb.bind(m, d); emb.reset(m, d)
    c = emb.controller()
    out = {}
    for en, e in emb.eff.items():
        p, q = emb.tcp_pose(d, en)
        R = quat2mat(q)
        pads = [b for b in e['pad']]
        pm = np.mean([d.xpos[b] for b in pads], 0) if pads else p
        gpads = [g for g in e['ft_geoms'] if m.geom_bodyid[g] in e['pad'] and (m.geom_contype[g] or m.geom_conaffinity[g])]
        gp = np.array([d.geom_xpos[g] for g in gpads]) if gpads else np.zeros((1, 3))
        out[en] = dict(tcp=p.round(3).tolist(), z=R[:, 2].round(2).tolist(), y=R[:, 1].round(2).tolist(),
                       padgeom_in_tcp=((gp - p) @ R).round(3).tolist()[:8])
    p0 = {en: emb.tcp_pose(d, en)[0] for en in emb.eff}
    t0 = time.time()
    for k in range(500):
        c.apply(m, d); mujoco.mj_step(m, d)
    sag = {en: float(np.linalg.norm(emb.tcp_pose(d, en)[0] - p0[en])) for en in emb.eff}
    wall = time.time() - t0
    # stroke: 0.3 m along +y... direction depends; use horizontal perpendicular to base->tcp
    res = {}
    for en in emb.eff:
        p, q = emb.tcp_pose(d, en)
        r = p[:2] / (np.linalg.norm(p[:2]) + 1e-9)
        dvec = np.array([-r[1], r[0], 0.0])
        tr = E.make_traj(en, [(p + 0.15 * dvec, q), (p - 0.15 * dvec, q), (p, q)], durations=[0.6, 1.0, 0.6])
        pred = c.command(tr)
    errs = []
    while not c.done():
        c.apply(m, d); mujoco.mj_step(m, d)
        errs.append(max(c.st[en]['err'] for en in emb.eff))
    for k in range(100):
        c.apply(m, d); mujoco.mj_step(m, d)
    print(name, 'nu', len(emb.act_ids), 'dof', emb.spec.dof, 'sag_mm', {k: round(v * 1e3, 2) for k, v in sag.items()},
          'stroke max err mm', round(1e3 * max(errs), 2), 'final err mm', round(1e3 * max(c.st[en]['err'] for en in emb.eff), 2),
          'rt x', round(1.0 / wall, 1), 'effort', {k: round(v, 3) for k, v in c.effort.as_dict().items()}, 'peaks', {k: round(v, 1) for k, v in c.acc.peaks.items()}, 'pred', {k: round(v, 3) for k, v in pred.items()})
    for en, o in out.items():
        print('   ', en, o)
