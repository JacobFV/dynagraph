"""single-jaw grippers: find where the jaws meet when closed (in the TCP body frame) and the opening direction."""
import numpy as np, mujoco
import physmem.embodiments as E
for robot, fixed, moving, jq_open in (('z1', 'link06', 'gripperMover', -1.0), ('so100', 'Fixed_Jaw', 'Moving_Jaw', 1.0)):
    emb = E.make(robot); spec = mujoco.MjSpec(); E.scene_defaults(spec); emb.build(spec, ((0, 0, 0), (1, 0, 0, 0)))
    m = spec.compile(); d = mujoco.MjData(m); emb.bind(m, d); emb.reset(m, d)
    e = next(iter(emb.eff.values()))
    ja = m.actuator_trnid[emb.act_ids[e['gacts'][0]], 0]
    bf, bm = m.body('r0/' + fixed).id, m.body('r0/' + moving).id
    gf = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bf and (m.geom_contype[g] or m.geom_conaffinity[g])]
    gm = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bm and (m.geom_contype[g] or m.geom_conaffinity[g])]
    sb = m.site_bodyid[e['site']]
    for qj in (e['gclosed'][0], jq_open):
        d.qpos[m.jnt_qposadr[ja]] = qj; mujoco.mj_forward(m, d)
        best = None
        for a in gf:
            for b in gm:
                ft = np.zeros(6)
                dist = mujoco.mj_geomDistance(m, d, a, b, 0.2, ft)
                if best is None or dist < best[0]:
                    best = (dist, ft.copy(), a, b)
        dist, ft, a, b = best
        mid = 0.5 * (ft[:3] + ft[3:])
        R = d.xmat[sb].reshape(3, 3)
        print(robot, 'jaw q', qj, 'gap', round(dist, 4), 'mid in body', (R.T @ (mid - d.xpos[sb])).round(4),
              'fixed pt', (R.T @ (ft[:3] - d.xpos[sb])).round(4), 'moving pt', (R.T @ (ft[3:] - d.xpos[sb])).round(4),
              'geoms', m.geom(a).name, m.geom(b).name)
