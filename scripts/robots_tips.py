"""measure each gripper's extent beyond the TCP along the approach axis (open / closed) and its widest point."""
import numpy as np, mujoco
import physmem.embodiments as E


def extent(m, d, emb, eff):
    e = emb.eff[eff]
    R = d.site_xmat[e['site']].reshape(3, 3); p = d.site_xpos[e['site']]
    zs = []
    for g in e['ft_geoms']:
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        Rg = d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] @ Rg.T + d.geom_xpos[g]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            s = m.geom_size[g]
            V = np.array([[a * s[0], b * s[1], c * s[2]] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) @ Rg.T + d.geom_xpos[g]
        else:
            r = m.geom_rbound[g]
            V = d.geom_xpos[g] + np.array([[0, 0, r], [0, 0, -r]]) @ R.T
        zs.append((V - p) @ R)
    Z = np.vstack(zs)
    return float(Z[:, 2].max())


for name in E.ROBOTS:
    emb = E.make(name); spec = mujoco.MjSpec(); E.scene_defaults(spec); emb.build(spec, ((0, 0, 0), (1, 0, 0, 0)))
    m = spec.compile(); d = mujoco.MjData(m); emb.bind(m, d); emb.reset(m, d)
    c = emb.controller()
    out = []
    for g in (0.0, 1.0):
        for e in emb.eff:
            c.set_gripper(e, g)
        for _ in range(400):
            c.apply(m, d); mujoco.mj_step(m, d)
        out.append({e: round(extent(m, d, emb, e), 4) for e in emb.eff})
    print(name, 'tip open', out[0], 'tip closed', out[1])
