"""SYNTHETIC 4-axis Cartesian palletizing gantry (x, y, z prismatic + wrist yaw), torque/force controlled.

This is a generic model with industry-typical numbers, NOT a vendor product or a calibrated CAD model:
- travel x/y +-1.2 m, z 0..-1.6 m below the beam (the station's `gantry` mount = beam height over the work area);
- moving masses: x bridge 60 kg, y carriage 30 kg, z quill 20 kg, yaw wrist 3 kg, tool 6-8 kg;
- reflected rotor inertia (armature) 1.5 / 1.0 / 0.8 kg (prismatic), 0.05 kg m^2 (yaw); viscous drive damping;
- drive force limits x 2500 N, y 2000 N, z 2000 N (gravity of quill + tool + 30 kg payload ~ 580 N), yaw 60 N m;
- speed/acc budget v 1.5 m/s, a 4 m/s^2 (belt/rack-and-pinion gantries: 1-3 m/s, 3-10 m/s^2);
- payload 30 kg.
Tools (choose by robot name):
- `gantry_vacuum`: vacuum plate, 2 x 2 cups r = 40 mm (zones 0-3, 60 kPa -> 302 N each) + a centre cup r = 15 mm
  (zone 4, 42 N) for small parts; zone control by the skill (cups off the face are disabled);
- `gantry_clamp`: parallel tote clamp, two jaws 0..0.15 m each (opening 0..0.30 m), 400 N per jaw.
"""
from __future__ import annotations

import numpy as np
import mujoco

G, J = mujoco.mjtGeom, mujoco.mjtJoint

AXES = dict(gx=dict(axis=[1, 0, 0], range=[-1.2, 1.2], mass=60.0, arm=1.5, damp=40.0, F=2500.0),
            gy=dict(axis=[0, 1, 0], range=[-1.2, 1.2], mass=30.0, arm=1.0, damp=30.0, F=2000.0),
            gz=dict(axis=[0, 0, 1], range=[-1.6, 0.1], mass=20.0, arm=0.8, damp=30.0, F=2000.0))


def _motor(spec, name, joint, F):
    a = spec.add_actuator(name=name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
    a.gainprm[0] = 1.0
    a.ctrllimited, a.ctrlrange = 1, [-F, F]
    a.forcelimited, a.forcerange = 1, [-F, F]


def build_gantry(tool='vacuum'):
    """-> child MjSpec of the gantry (+ tool), origin = beam height at the work-area centre."""
    from physmem.embodiments.core import add_suction_cup
    s = mujoco.MjSpec()
    base = s.worldbody.add_body(name='gantry_base')
    # (no frame visuals: the bridge/rails would sit between the station's tracker camera and the work area; the frame
    # is implied - masses live on small hidden spheres)
    base.add_geom(type=G.mjGEOM_SPHERE, size=[0.01, 0, 0], contype=0, conaffinity=0, rgba=[0, 0, 0, 0], mass=1.0)
    parent = base
    for name, a in AXES.items():
        b = parent.add_body(name=name)
        b.add_joint(name=name, type=J.mjJNT_SLIDE, axis=a['axis'], range=a['range'], limited=True,
                    armature=a['arm'], damping=a['damp'])
        if name in ('gx', 'gy'):   # bridge / carriage: mass only (see above)
            b.add_geom(type=G.mjGEOM_SPHERE, size=[0.01, 0, 0], contype=0, conaffinity=0, rgba=[0, 0, 0, 0],
                       mass=a['mass'])
        else:                 # vertical quill (visual) hanging from the carriage
            b.add_geom(type=G.mjGEOM_BOX, size=[0.03, 0.03, 0.3], pos=[0, 0, 0.32], contype=0, conaffinity=0,
                       rgba=[0.5, 0.5, 0.55, 1], mass=a['mass'])
        _motor(s, name, name, a['F'])
        parent = b
    yw = parent.add_body(name='gyaw')
    yw.add_joint(name='gyaw', type=J.mjJNT_HINGE, axis=[0, 0, 1], range=[-3.2, 3.2], limited=True, armature=0.05,
                 damping=1.0)
    yw.add_geom(type=G.mjGEOM_CYLINDER, size=[0.05, 0.02, 0], contype=0, conaffinity=0, rgba=[0.3, 0.3, 0.32, 1], mass=3.0)
    _motor(s, 'gyaw', 'gyaw', 60.0)
    # tool body, flipped so its +z points DOWN (approach)
    tool_b = yw.add_body(name='cup' if tool == 'vacuum' else 'clamp', pos=[0, 0, -0.03], quat=[0, 1, 0, 0])
    if tool == 'vacuum':
        add_suction_cup(s, parent_body=tool_b, stem=0.10, radius=0.04, dp=60e3, mass=6.0,
                        cups=[(-0.08, -0.05), (0.08, -0.05), (-0.08, 0.05), (0.08, 0.05)], plate=2.0)
        # centre small cup (zone 4) for small parts: its own body/actuator, 6 mm proud of the big lips
        cb = tool_b.add_body(name='cup4', pos=[0, 0, 0.10])   # flush with the big cups
        cb.add_geom(name='cup4_disk', type=G.mjGEOM_CYLINDER, size=[0.015, 0.004, 0], pos=[0, 0, 0.004], contype=0,
                    conaffinity=0, mass=0.01, rgba=[0.12, 0.12, 0.12, 1])
        from physmem.embodiments.core import RIM_N
        for j in range(RIM_N):
            a = 2 * np.pi * j / RIM_N
            cb.add_geom(name=f'cup4_lip{j}', type=G.mjGEOM_SPHERE, size=[0.003, 0, 0],
                        pos=[0.015 * np.cos(a), 0.015 * np.sin(a), 0.005], mass=0.001, rgba=[0.15, 0.15, 0.15, 1],
                        friction=[1.2, 0.01, 0.001], condim=4, margin=0.001, solref=[0.01, 1],
                        solimp=[0.9, 0.95, 0.002, 0.5, 2])
        act = s.add_actuator(name='suction4', trntype=mujoco.mjtTrn.mjTRN_BODY, target='cup4')
        act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
        act.gainprm[0] = 60e3 * np.pi * 0.015 ** 2
        act.ctrllimited, act.ctrlrange = 1, [0, 1]
    else:
        tool_b.add_geom(name='clamp_head', type=G.mjGEOM_BOX, size=[0.04, 0.17, 0.02], pos=[0, 0, 0.02],
                        rgba=[0.3, 0.3, 0.32, 1], mass=4.0)
        for side, sgn in (('l', 1), ('r', -1)):
            jb = tool_b.add_body(name=f'jaw_{side}', pos=[0, sgn * 0.005, 0.10])
            jb.add_joint(name=f'jaw_{side}', type=J.mjJNT_SLIDE, axis=[0, sgn, 0], range=[0.0, 0.15], limited=True,
                         armature=0.2, damping=5.0)
            jb.add_geom(name=f'jaw_{side}_pad', type=G.mjGEOM_BOX, size=[0.06, 0.006, 0.06], pos=[0, -sgn * 0.0, 0],
                        rgba=[0.15, 0.15, 0.15, 1], friction=[1.0, 0.01, 0.001], condim=4, mass=0.6)
        s.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1='jaw_l', name2='jaw_r', data=[0, 1, 0, 0, 0] + [0] * 7)
        t = s.add_tendon(name='clamp_split')
        t.wrap_joint('jaw_l', 0.5)
        t.wrap_joint('jaw_r', 0.5)
        a = s.add_actuator(name='clamp', target='clamp_split', trntype=mujoco.mjtTrn.mjTRN_TENDON)
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
        a.gainprm[0] = 1.0
        a.ctrllimited, a.ctrlrange = 1, [-1600.0, 1600.0]
        a.forcelimited, a.forcerange = 1, [-1600.0, 1600.0]
    return s
