"""Dex3 grip calibration: fixed-base G1, a FIXED block between thumb and fingers at the TCP of one hand; close at
grip=1 and measure the block's contact normal force per side (sum |fn| / 2), per finger link, vs settings.
python scripts/robots_g1_grip_calib.py [width_m] [kp ...]"""
import sys, json
import numpy as np, mujoco
import physmem.embodiments as E

w = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
kps = [float(x) for x in sys.argv[2:]] or [None]
out = []
for kp in kps:
    for side in ('right', 'left'):
        emb0 = E.make('g1_dex3_fixed'); sp = mujoco.MjSpec(); E.scene_defaults(sp)
        emb0.build(sp, ((0, 0, 0), (1, 0, 0, 0))); m0 = sp.compile(); d0 = mujoco.MjData(m0)
        emb0.bind(m0, d0); emb0.reset(m0, d0)
        p, q = emb0.tcp_pose(d0, side)
        emb = E.make('g1_dex3_fixed'); sp = mujoco.MjSpec(); E.scene_defaults(sp)
        emb.build(sp, ((0, 0, 0), (1, 0, 0, 0)))
        R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
        off = np.array([-0.014, 0.021, 0.04])   # thumb-finger pinch point, same in each hand's TCP frame
        b = sp.worldbody.add_body(name='blk', pos=list(p + R @ off), quat=list(q))
        b.add_geom(name='blk', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.008, w / 2, 0.008], friction=[1, 0.005, 0.0001])
        m = sp.compile(); d = mujoco.MjData(m); emb.bind(m, d); emb.reset(m, d)
        c = emb.controller()
        if kp is not None:
            c.g['finger_kp'] = kp
        import os
        if os.environ.get('GRIP_N'):
            emb.set_grip_force(side, float(os.environ['GRIP_N']))
        c.set_gripper(side, 1.0)
        trace = []
        for it in range(1000):
            c.apply(m, d); mujoco.mj_step(m, d)
            if it % 100 == 99:
                trace.append((round(emb.pad_normal(m, d, side) / 2, 1), round(c.st[side].get('grip_kt', -1), 2)))
        gb = m.geom('blk').id
        per, f6 = {}, np.zeros(6)
        for k in range(d.ncon):
            if gb in (d.contact.geom1[k], d.contact.geom2[k]):
                o = d.contact.geom2[k] if d.contact.geom1[k] == gb else d.contact.geom1[k]
                mujoco.mj_contactForce(m, d, k, f6)
                n = m.body(m.geom_bodyid[o]).name.split('/')[-1]
                per[n] = per.get(n, 0.0) + abs(f6[0])
        e = emb.eff[side]
        tau = d.ctrl[emb.act_ids[e['gacts']]] * emb.act_gear[e['gacts']]
        th = sum(v for k, v in per.items() if 'thumb' in k)
        fi = sum(v for k, v in per.items() if ('index' in k or 'middle' in k))
        row = dict(side=side, kp=kp if kp is not None else c.g['finger_kp'], width=w,
                   pinch_n=round(min(th, fi), 2), thumb_n=round(th, 1), fingers_n=round(fi, 1), per_link={k: round(v, 1) for k, v in per.items()},
                   tau_frac=np.round(np.abs(tau) / (emb.act_hi[e['gacts']] * emb.act_gear[e['gacts']]), 2).tolist(),
                   closure=round(emb.gripper_state(m, d, side)[0], 2), trace=trace,
                   q_at_limit=[bool(abs(d.qpos[a] - m.jnt_range[j][0]) < 0.02 or abs(d.qpos[a] - m.jnt_range[j][1]) < 0.02)
                               for j, a in zip([m.dof_jntid[x] for x in emb.eff[side]['gdofs']], emb.eff[side]['gqadr'])])
        out.append(row)
        print(json.dumps(row))
