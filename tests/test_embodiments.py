"""v2 embodiments: registry, torque-motor conversion, gravity hold, trajectory blending, tracking, effort, guards."""
import math

import mujoco
import numpy as np
import pytest

import physmem.embodiments as E
from physmem.embodiments import testbench as TB
from physmem.embodiments.trajectory import BlendedTrajectory, Plan
from physmem.embodiments.ik import down_quat

ARMS = ['panda', 'ur5e_2f85', 'gen3_2f85', 'z1', 'so100', 'aloha', 'tidybot']


def _scene(name):
    spec = mujoco.MjSpec()
    E.scene_defaults(spec)
    emb = E.make(name, prefix='r0')
    emb.build(spec, ((0, 0, 0), (1, 0, 0, 0)))
    m = spec.compile()
    d = mujoco.MjData(m)
    emb.bind(m, d)
    emb.reset(m, d)
    return emb, m, d


def test_registry():
    n = E.names()
    for k in ARMS + ['g1_dex3', 'g1_dex3_fixed']:
        assert k in n
    with pytest.raises(KeyError):
        E.make('nope')


@pytest.mark.parametrize('name', ARMS)
def test_torque_motors_and_gravity_hold(name):
    emb, m, d = _scene(name)
    for i in emb.act_ids:
        assert m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED and m.actuator_gainprm[i, 0] == 1
        assert m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
        assert m.actuator_ctrllimited[i] and m.actuator_ctrlrange[i, 1] > 0
    c = emb.controller()
    p0 = {e: emb.tcp_pose(d, e)[0] for e in emb.eff}
    for _ in range(250):
        c.apply(m, d)
        mujoco.mj_step(m, d)
    for e in emb.eff:
        assert np.linalg.norm(emb.tcp_pose(d, e)[0] - p0[e]) < 2e-3, (name, e)
    assert emb.spec.dof > 0 and emb.spec.name == name


def test_blended_plan_constant_speed_cruise():
    v = 0.5
    dv = np.array([1.0, 0, 0])
    q = np.array([0, 1.0, 0, 0])
    s = np.zeros(3)
    a, b, f = s + dv * 0.1, s + dv * 0.3, s + dv * 0.35
    pl = Plan(BlendedTrajectory(effector='h', waypoints=[(a, q), (b, q), (f, q)], durations=[0.4, 0.4, 0.2],
                                via_vel=[dv * v, dv * v, np.zeros(3)]), s, q, 0.0)
    for t in np.linspace(0.41, 0.79, 10):
        assert abs(pl.at(t)[1] @ dv - v) < 1e-9
    # continuity of velocity across via-points
    for tk in (0.4, 0.8):
        assert np.linalg.norm(pl.at(tk - 1e-6)[1] - pl.at(tk + 1e-6)[1]) < 1e-3
    assert np.allclose(pl.at(5.0)[0], f) and np.allclose(pl.at(5.0)[1], 0)


def test_retiming_respects_budget():
    q = np.array([0, 1.0, 0, 0])
    lim = dict(v=0.5, a=3.0, j=50.0, w=3.0)
    tr = BlendedTrajectory(effector='h', waypoints=[(np.array([0.5, 0, 0]), q)], durations=[0.3])
    pl = Plan(tr, np.zeros(3), q, 0.0, limits=lim)
    assert pl.scale > 1 and pl.pred['peak_v'] <= lim['v'] * 1.01 and pl.pred['peak_j'] <= lim['j'] * 1.01
    pl2 = Plan(tr, np.zeros(3), q, 0.0, limits=lim, effort_weight=1.0)
    assert pl2.duration > pl.duration


def test_stroke_tracking_and_effort_panda():
    emb, m, d = _scene('panda')
    c = emb.controller()
    p, _ = emb.tcp_pose(d, 'hand')
    q = down_quat(0.0)
    c.command(BlendedTrajectory(effector='hand', waypoints=[(p + [0, -0.12, 0], q)], durations=[0.6]))
    while not c.done():
        c.apply(m, d)
        mujoco.mj_step(m, d)
    v = 0.6
    dv = np.array([0, 1.0, 0])
    s, _ = emb.tcp_pose(d, 'hand')
    a, b, f = s + dv * 0.09, s + dv * 0.21, s + dv * 0.27
    t0 = d.time
    c.command(BlendedTrajectory(effector='hand', waypoints=[(a, q), (b, q), (f, q)], durations=[0.3, 0.2, 0.2],
                                via_vel=[dv * v, dv * v, np.zeros(3)]))
    sp, err = [], []
    while not c.done():
        c.apply(m, d)
        mujoco.mj_step(m, d)
        if t0 + 0.35 < d.time < t0 + 0.45:
            sp.append(emb.tcp_vel(m, d, 'hand')[0] @ dv)
            err.append(c.st['hand']['err'])
    assert abs(np.mean(sp) / v - 1) < 0.03
    assert max(err) < 0.005
    eff = c.effort
    assert eff.torque_sq > 0 and eff.energy_j > 0 and eff.peak_ee_speed > 0.5 and eff.cost() > 0


def test_guarded_trips_on_table():
    spec, emb, B = TB.make_scene('ur5e_2f85')
    m, d = TB.compile_scene(spec, emb)
    c = emb.controller()
    p, q = emb.tcp_pose(d, 'hand')
    c.set_gripper('hand', 1.0)
    c.set_mode('guarded', 'hand', threshold=8.0, axis=(0, 0, -1))
    c.command(BlendedTrajectory(effector='hand', waypoints=[(np.array([p[0], p[1], -0.05]), q)], durations=[4.0]))
    t_end = d.time + 6
    while not c.done() and d.time < t_end:
        c.apply(m, d)
        mujoco.mj_step(m, d)
    assert c.tripped('hand')
    assert d.time < t_end - 1.0                     # stopped early, no waiting
    assert emb.tcp_pose(d, 'hand')[0][2] > -0.01    # did not push through the table
    assert TB.integrity(m, d) == []


def test_g1_floating_builds_and_stands():
    spec = mujoco.MjSpec()
    E.scene_defaults(spec)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1])
    emb = E.make('g1_dex3', prefix='h0')
    emb.build(spec, ((0, 0, 0), (1, 0, 0, 0)))
    m = spec.compile()
    d = mujoco.MjData(m)
    emb.bind(m, d)
    emb.reset(m, d)
    c = emb.controller()
    assert emb.spec.floating_base and emb.spec.kind == 'humanoid'
    for _ in range(500):
        c.apply(m, d)
        mujoco.mj_step(m, d)
    assert d.xpos[emb.root_body][2] > 0.6
