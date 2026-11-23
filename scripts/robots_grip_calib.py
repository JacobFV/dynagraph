"""grip-force calibration: close each gripper on a FIXED block at the TCP, measure the pad normal force per side
as a function of the finger-drive force. -> results/v2/robots/grip_calib.json"""
import json, sys
import numpy as np, mujoco
import physmem.embodiments as E

ROB = sys.argv[1:] or ['panda', 'ur5e_2f85', 'z1', 'so100', 'aloha']
out = {}
for name in ROB:
    emb0 = E.make(name); sp = mujoco.MjSpec(); E.scene_defaults(sp); emb0.build(sp, ((0, 0, 0), (1, 0, 0, 0)))
    m0 = sp.compile(); d0 = mujoco.MjData(m0); emb0.bind(m0, d0); emb0.reset(m0, d0)
    eff = next(iter(emb0.eff)); p, q = emb0.tcp_pose(d0, eff)
    gc = emb0.eff[eff]['cfg'].gripper
    w = min(0.04, 0.6 * gc.max_width)
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
    c = p + R[:, 1] * gc.jaw_dir * (w / 2 + 0.001)
    res = []
    lo, hi = emb0.act_lo[emb0.eff[eff]['gacts'][0]], emb0.act_hi[emb0.eff[eff]['gacts'][0]]
    for frac in (0.25, 0.5, 1.0):
        emb = E.make(name); sp = mujoco.MjSpec(); E.scene_defaults(sp); emb.build(sp, ((0, 0, 0), (1, 0, 0, 0)))
        b = sp.worldbody.add_body(name='blk', pos=list(c), quat=list(q))
        b.add_geom(name='blk', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, w / 2, 0.02], friction=[1, 0.005, 0.0001])
        m = sp.compile(); d = mujoco.MjData(m); emb.bind(m, d); emb.reset(m, d)
        e = emb.eff[eff]; e['gforce'] = e['gforce'] * frac
        ctl = emb.controller(); ctl.set_gripper(eff, 1.0)
        for _ in range(600):
            ctl.apply(m, d); mujoco.mj_step(m, d)
        gb = m.geom('blk').id
        Fn, f6 = 0.0, np.zeros(6)
        for k in range(d.ncon):
            if gb in (d.contact.geom1[k], d.contact.geom2[k]):
                mujoco.mj_contactForce(m, d, k, f6); Fn += abs(f6[0])
        res.append(dict(drive=float(e['gforce'][0]), pad_n_per_side=Fn / 2, closure=emb.gripper_state(m, d, eff)[0]))
    out[name] = dict(width=w, drive_limit=float(hi), rows=res, pad_per_drive=res[-1]['pad_n_per_side'] / res[-1]['drive'])
    print(name, out[name])
json.dump(out, open('results/v2/robots/grip_calib.json', 'w'), indent=1)
