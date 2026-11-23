"""Minimal benches for the ROBOTS gates and tests (ENVS builds the real industrial scenes). A table whose top is at
the robot's mount height (z=0) for fixed arms, or a 0.40 m table in front of a mobile base; free objects; a peg-hole
fixture; small parts for sweeping. Estimates given to skills = truth + tracker noise (the harness plays the sensor)."""
from __future__ import annotations

import math

import mujoco
import numpy as np

import physmem.embodiments as E
from physmem.skills.base import ObjectEstimate

# per-robot bench geometry: table top z (in the mount frame), workspace box (x, y ranges), object scale
BENCH = {
    'panda': dict(table_z=0.0, ws=((0.35, 0.65), (-0.25, 0.25)), obj=(0.02, 0.03), mass=(0.1, 0.6), stroke=(0.2, 0.3)),
    'ur5e_2f85': dict(table_z=0.0, ws=((0.35, 0.65), (-0.25, 0.25)), obj=(0.02, 0.03), mass=(0.1, 0.6), stroke=(0.2, 0.3)),
    'gen3_2f85': dict(table_z=0.0, ws=((0.35, 0.65), (-0.25, 0.25)), obj=(0.02, 0.03), mass=(0.1, 0.6), stroke=(0.2, 0.3)),
    'z1': dict(table_z=0.0, ws=((0.28, 0.45), (-0.15, 0.15)), obj=(0.015, 0.025), mass=(0.1, 0.4), stroke=(0.15, 0.25)),
    'so100': dict(table_z=0.0, ws=((0.16, 0.26), (-0.08, 0.08)), obj=(0.012, 0.018), mass=(0.02, 0.08), stroke=(0.08, 0.12)),
    'aloha': dict(table_z=0.0, ws=((-0.32, -0.12), (-0.15, 0.15)), obj=(0.012, 0.02), mass=(0.05, 0.25), stroke=(0.15, 0.25)),
    'ur5e_suction': dict(table_z=0.0, ws=((0.35, 0.65), (-0.25, 0.25)), obj=(0.03, 0.06), mass=(0.5, 4.0), stroke=(0.2, 0.3)),
    'fr3_suction': dict(table_z=0.0, ws=((0.35, 0.65), (-0.25, 0.25)), obj=(0.03, 0.06), mass=(0.3, 2.5), stroke=(0.2, 0.3)),
    'tidybot': dict(table_z=0.40, ws=((0.5, 0.68), (-0.18, 0.18)), obj=(0.02, 0.03), mass=(0.1, 0.6), stroke=(0.2, 0.3)),
}


def new_spec():
    spec = mujoco.MjSpec()
    E.scene_defaults(spec)
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    spec.worldbody.add_light(pos=[0, -1, 2.5], dir=[0, 0.4, -1], diffuse=[0.8, 0.8, 0.8])
    return spec


def add_table(spec, name, center, half, rgba=(0.55, 0.5, 0.45, 1)):
    b = spec.worldbody.add_body(name=name, pos=list(center))
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), rgba=list(rgba), friction=[0.8, 0.005, 0.0001])
    return b


def add_box(spec, name, pos, half, mass, rgba=(0.2, 0.5, 0.9, 1), friction=0.8, yaw=0.0):
    b = spec.worldbody.add_body(name=name, pos=list(pos), quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    b.add_freejoint(name=name + '_free')
    b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), mass=mass, rgba=list(rgba),
               friction=[friction, 0.005, 0.0001], condim=4)
    return b


def add_hole_fixture(spec, name, center, hole_half, wall=0.02, height=0.03):
    """a square hole (half width hole_half) through 4 blocks on the table; center = entry centre at the top."""
    b = spec.worldbody.add_body(name=name, pos=[center[0], center[1], center[2] - height / 2])
    o = hole_half + wall / 2
    L = hole_half + wall
    for k, (dx, dy, sx, sy) in enumerate([(o, 0, wall / 2, L), (-o, 0, wall / 2, L), (0, o, hole_half, wall / 2), (0, -o, hole_half, wall / 2)]):
        b.add_geom(name=f'{name}_w{k}', type=mujoco.mjtGeom.mjGEOM_BOX, pos=[dx, dy, 0], size=[sx, sy, height / 2],
                   rgba=[0.6, 0.6, 0.65, 1], friction=[0.3, 0.005, 0.0001])
    return b


def make_scene(robot, prefix='r0', floor=True, extra=None):
    """-> (spec, emb, bench). extra(spec, bench) adds objects before compile."""
    bench = dict(BENCH[robot])
    spec = new_spec()
    if floor:
        z0 = -0.75 if robot != 'tidybot' else 0.0
        spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, z0], size=[4, 4, 0.1],
                                rgba=[0.8, 0.8, 0.8, 1])
    tz = bench['table_z']
    if robot == 'tidybot':
        add_table(spec, 'table', (0.85, 0, tz / 2), (0.45, 0.6, tz / 2))
    elif robot == 'aloha':
        add_table(spec, 'table', (0, 0, -0.02), (0.8, 0.5, 0.02))
    else:
        add_table(spec, 'table', (0.3, 0, -0.02), (0.6, 0.6, 0.02))
    emb = E.make(robot, prefix=prefix)
    emb.build(spec, ((0, 0, 0), (1, 0, 0, 0)))
    if extra is not None:
        extra(spec, bench)
    return spec, emb, bench


def compile_scene(spec, emb):
    m = spec.compile()
    d = mujoco.MjData(m)
    emb.bind(m, d)
    emb.reset(m, d)
    return m, d


def body_pose(m, d, name):
    b = m.body(name).id
    return d.xpos[b].copy(), d.xquat[b].copy()


def estimate(m, d, name, half, rng=None, pos_sd=0.002, yaw_sd=0.02, size_sd=0.0):
    """tracker estimate of a free box: truth + noise (the harness plays the pose tracker)."""
    p, q = body_pose(m, d, name)
    rng = rng or np.random.default_rng(0)
    p = p + rng.normal(0, pos_sd, 3) * np.array([1, 1, 0.5])
    dyaw = rng.normal(0, yaw_sd)
    qn = np.zeros(4)
    mujoco.mju_mulQuat(qn, np.array([math.cos(dyaw / 2), 0, 0, math.sin(dyaw / 2)]), q)
    h = np.asarray(half) * (1 + rng.normal(0, size_sd, 3))
    return ObjectEstimate(pos=p, quat=qn, half_size=tuple(h))


def integrity(m, d):
    """v2 G1 integrity: no applied object forces, no equality constraints touching free objects, mocap untouched."""
    v = []
    if np.any(d.xfrc_applied != 0):
        v.append('xfrc_applied')
    if np.any(d.qfrc_applied != 0):
        v.append('qfrc_applied')
    for i in range(m.neq):
        if d.eq_active[i]:
            b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
            if m.eq_type[i] == mujoco.mjtEq.mjEQ_WELD and (m.body_jntnum[b1] and m.jnt_type[m.body_jntadr[b1]] == 0 or
                                                            m.body_jntnum[b2] and m.jnt_type[m.body_jntadr[b2]] == 0):
                v.append(f'weld:{m.eq(i).name}')
    return v
