"""v2 skills: run through the controller only, actor-legal inputs, SkillResult with effort."""
import ast
import inspect
import math

import numpy as np

from physmem.v2.api import EffortMeter, SkillCall, SkillResult
from physmem.embodiments import testbench as TB
from physmem.skills import ObjectEstimate, SkillContext, run_skill
from physmem.skills import primitives as P


def _bench(robot, half=(0.025, 0.025, 0.03), pos=(0.5, 0.0), mass=0.3):
    spec, emb, B = TB.make_scene(robot, extra=lambda sp, b: TB.add_box(sp, 'obj', (pos[0], pos[1], b['table_z'] + half[2]), half, mass))
    m, d = TB.compile_scene(spec, emb)
    ctx = SkillContext(m, d, emb)
    ctx.wait(0.2)
    return m, d, emb, ctx, B


def test_skills_never_touch_object_state():
    """the skill code reads only the robot (controller/proprio) and estimates: no xpos/qpos/xfrc/eq writes."""
    src = inspect.getsource(P)
    tree = ast.parse(src)
    banned = {'xfrc_applied', 'qfrc_applied', 'eq_active', 'mocap_pos', 'mocap_quat', 'qpos', 'xpos', 'xquat', 'body'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in banned:
            raise AssertionError(f'skills access data.{node.attr} (line {node.lineno})')


def test_pick_place_panda():
    m, d, emb, ctx, B = _bench('panda')
    est = TB.estimate(m, d, 'obj', (0.025, 0.025, 0.03), np.random.default_rng(0))
    r = run_skill(ctx, SkillCall('pick', dict(speed=0.4), target='obj', effector='hand'), est)
    assert isinstance(r, SkillResult) and r.ok, r.status
    assert isinstance(r.effort, EffortMeter) and r.effort.energy_j > 0
    r2 = run_skill(ctx, SkillCall('place', dict(x=0.45, y=0.15, z=B['table_z'], speed=0.4), target='obj', effector='hand'))
    assert r2.ok, r2.status
    ctx.wait(0.3)
    p, _ = TB.body_pose(m, d, 'obj')
    assert np.hypot(p[0] - 0.45, p[1] - 0.15) < 0.015
    assert TB.integrity(m, d) == []


def test_push_reports_speed_tracking():
    m, d, emb, ctx, B = _bench('ur5e_2f85')
    est = TB.estimate(m, d, 'obj', (0.025, 0.025, 0.03), np.random.default_rng(1))
    p0, _ = TB.body_pose(m, d, 'obj')
    r = run_skill(ctx, SkillCall('push', dict(theta=0.0, speed=0.3, distance=0.08), target='obj', effector='hand'), est)
    assert r.ok
    assert abs(r.log['speed_ratio'] - 1) < 0.1
    p1, _ = TB.body_pose(m, d, 'obj')
    assert p1[0] - p0[0] > 0.05


def test_too_wide_and_unsupported():
    m, d, emb, ctx, B = _bench('panda')
    r = run_skill(ctx, SkillCall('pick', {}, target='obj', effector='hand'),
                  ObjectEstimate(pos=(0.5, 0, 0.05), half_size=(0.1, 0.1, 0.05)))
    assert not r.ok and r.status == 'too_wide'
    r = run_skill(ctx, SkillCall('regrasp', {}, effector='hand'))
    assert not r.ok and r.status == 'unsupported'


def test_skill_vector_encoding():
    v = SkillCall('push', dict(theta=0.3, speed=0.4, distance=0.1)).vector()
    assert v.shape == (24,) and v[1] == 1.0


def test_suction_pick_place_ur5e():
    m, d, emb, ctx, B = _bench('ur5e_suction', half=(0.05, 0.05, 0.03), mass=2.0)   # cup lip 8 mm inside the face
    est = TB.estimate(m, d, 'obj', (0.05, 0.05, 0.03), np.random.default_rng(2))
    r = run_skill(ctx, SkillCall('pick', dict(speed=0.3), target='obj', effector='hand'), est)
    assert r.ok, r.status
    assert emb.gripper_state(m, d, 'hand')[0] > 0.5          # sealed and vacuum on
    p, _ = TB.body_pose(m, d, 'obj')
    assert p[2] > 0.08                                         # lifted by adhesion (no weld)
    r2 = run_skill(ctx, SkillCall('place', dict(x=0.45, y=0.15, z=0.0, speed=0.3), effector='hand'))
    assert r2.ok
    assert TB.integrity(m, d) == []


def test_suction_needs_seal():
    m, d, emb, ctx, B = _bench('ur5e_suction')
    c = emb.controller()
    c.set_gripper('hand', 1.0)
    ctx.wait(0.1)
    assert not emb.cup_seal(m, d, 'hand')[0]
    assert d.ctrl[emb.act_ids[emb.eff['hand']['gacts'][0]]] == 0.0   # vacuum on in free air -> no adhesion force


def test_grip_force_api():
    import physmem.embodiments as E
    emb = E.make('ur5e_2f85', grip_force=150)
    m, d, emb2, ctx, B = _bench('ur5e_2f85')
    assert abs(emb2.set_grip_force('hand', 50.0) - 50.0) < 1e-6
    assert emb2.max_grip_force('hand') > 200


def test_move_base_tidybot():
    import mujoco
    import physmem.embodiments as E
    spec = mujoco.MjSpec()
    E.scene_defaults(spec)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1])
    emb = E.make('tidybot', prefix='r0')
    emb.build(spec, ((0.0, -1.0, 0.0), (0.7071068, 0, 0, 0.7071068)))     # facing +y
    m = spec.compile()
    d = mujoco.MjData(m)
    emb.bind(m, d)
    emb.reset(m, d)
    ctx = SkillContext(m, d, emb)
    r = run_skill(ctx, SkillCall('move_base', dict(x=0.3, y=-0.6, yaw=math.pi / 2 + 0.3, speed=0.3), effector='hand'))
    assert r.ok, (r.status, r.log)
    assert r.effort.energy_j > 0
    p = d.xpos[m.body('r0/base_link').id]
    assert abs(p[0] - 0.3) < 0.03 and abs(p[1] + 0.6) < 0.03
    r2 = run_skill(SkillContext(*_bench('panda')[:3]), SkillCall('move_base', dict(x=0, y=0), effector='hand'))
    assert not r2.ok and r2.status == 'unsupported'
