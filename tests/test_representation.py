"""Record-layer representation contracts (e00): renaming, permutation, units,
frames, rotation representation, time origin. Tensor-layer/model equivariance
tests live in tests/test_models.py and tests/test_data.py."""
import math
import random

import numpy as np

from physmem.frames import Frame2, convert_length_units, galilean_shift, quat_canonical, quat_to_yaw, transform_value
from physmem.memory import WorldBelief
from physmem.schema import EntityDescriptor, EntityType, ObservationEvent, Prop

TOL = 1e-6


def _events(names, shift_t=0.0, perm_seed=None):
    rng = np.random.default_rng(0)
    evs = []
    for i, n in enumerate(names):
        for k in range(5):
            t = 0.05 * k + shift_t
            evs.append(ObservationEvent(f"{n}-{k}", "ep", n, Prop.POSE2, (i + 0.01 * k, -i, 1.0, 0.0), (1e-6,) * 4, "world", t, t, f"cam-{n}"))
        evs.append(ObservationEvent(f"{n}-m", "ep", n, Prop.MASS, (1.0 + i,), (0.01,), "world", shift_t, shift_t, f"scale-{n}"))
    if perm_seed is not None:
        random.Random(perm_seed).shuffle(evs)
    return evs


def _belief(names, evs, now):
    b = WorldBelief("ep")
    for n in names:
        b.add_entity(EntityDescriptor(n, EntityType.OBJECT))
    b.ingest_events(evs, now=now)
    return b


def test_opaque_renaming_preserves_values():
    a = ["box", "cup", "bolt"]
    r = ["zz91", "a", "Q-7"]
    ba = _belief(a, _events(a), 1.0)
    br = _belief(r, [ObservationEvent(e.event_id.replace(e.entity_id, r[a.index(e.entity_id)]), e.episode_id, r[a.index(e.entity_id)],
                                       e.property_type, e.value, e.variance, e.frame_id, e.measured_at, e.received_at,
                                       e.source_id + "x") for e in _events(a)], 1.0)
    for x, y in zip(a, r):
        for p in (Prop.POSE2, Prop.MASS):
            assert np.allclose(ba.resolve_query(x, p)[0], br.resolve_query(y, p)[0])
            assert np.allclose(ba.resolve_query(x, p)[1], br.resolve_query(y, p)[1])


def test_event_order_permutation_invariance_of_belief():
    a = ["box", "cup"]
    b1 = _belief(a, _events(a), 1.0)
    b2 = _belief(a, _events(a, perm_seed=3), 1.0)
    for x in a:
        assert np.allclose(b1.resolve_query(x, Prop.POSE2)[0], b2.resolve_query(x, Prop.POSE2)[0])


def test_time_origin_shift():
    a = ["box"]
    b1 = _belief(a, _events(a), 1.0)
    b2 = _belief(a, _events(a, shift_t=1000.0), 1001.0)
    for p in (Prop.POSE2, Prop.MASS):
        assert np.allclose(b1.resolve_query("box", p)[0], b2.resolve_query("box", p)[0])
        assert np.allclose(b1.resolve_query("box", p)[1], b2.resolve_query("box", p)[1])


def test_frame_round_trip_and_composition():
    rng = np.random.default_rng(1)
    for _ in range(50):
        T = Frame2(*rng.normal(size=2), rng.uniform(-math.pi, math.pi))
        yaw = rng.uniform(-math.pi, math.pi)
        pose = np.array([*rng.normal(size=2), math.cos(yaw), math.sin(yaw)])
        var = np.abs(rng.normal(size=4)) * 1e-4
        for prop, val, vv in ((Prop.POSE2, pose, var), (Prop.VEL2, rng.normal(size=3), np.abs(rng.normal(size=3))), (Prop.COMMAND, rng.normal(size=2), np.ones(2))):
            v2, var2 = transform_value(prop, val, vv, T)
            v3, var3 = transform_value(prop, v2, var2, T.inverse())
            assert np.allclose(v3, val, atol=TOL)
            if prop != Prop.POSE2:
                assert np.isclose(np.sum(var3[:2]), np.sum(vv[:2]))  # trace preserved
        # composition consistency: two paths to the same frame agree
        U = Frame2(*rng.normal(size=2), rng.uniform(-math.pi, math.pi))
        a1, _ = transform_value(Prop.POSE2, transform_value(Prop.POSE2, pose, var, T)[0], var, U)
        a2, _ = transform_value(Prop.POSE2, pose, var, U.compose(T))
        assert np.allclose(a1, a2, atol=TOL)


def test_belief_frame_paths_consistent():
    b = WorldBelief("ep")
    b.register_frame("table", "world", Frame2(1.0, 2.0, 0.3))
    b.register_frame("cam", "table", Frame2(0.2, -0.1, -1.1))
    b.add_entity(EntityDescriptor("box", EntityType.OBJECT))
    p_cam = (0.3, 0.4, math.cos(0.2), math.sin(0.2))
    b.ingest_events([ObservationEvent("e", "ep", "box", Prop.POSE2, p_cam, (1e-6,) * 4, "cam", 0.0, 0.0, "c")], now=0.0)
    m_world = b.resolve_query("box", Prop.POSE2)[0]
    m_cam = b.resolve_query("box", Prop.POSE2, target_frame="cam")[0]
    assert np.allclose(m_cam, p_cam, atol=TOL)
    direct, _ = transform_value(Prop.POSE2, p_cam, (0,) * 4, Frame2(1.0, 2.0, 0.3).compose(Frame2(0.2, -0.1, -1.1)))
    assert np.allclose(direct, m_world, atol=TOL)


def test_units_metres_vs_millimetres():
    pose_mm = (1234.0, -50.0, 1.0, 0.0)
    v, var = convert_length_units(Prop.POSE2, pose_mm, (4.0, 4.0, 1e-4, 1e-4), from_scale=0.001)
    assert np.allclose(v, (1.234, -0.05, 1.0, 0.0)) and np.allclose(var[:2], 4e-6)
    k, kv = convert_length_units(Prop.STIFFNESS, (2.0,), (0.01,), from_scale=0.001)  # 2 N/mm
    assert np.isclose(k[0], 2000.0)
    back, _ = convert_length_units(Prop.POSE2, v, var, from_scale=1.0, to_scale=0.001)
    assert np.allclose(back, pose_mm)


def test_quaternion_sign_equivalence():
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        assert np.allclose(quat_canonical(q), quat_canonical(-q))
        assert math.isclose(quat_to_yaw(q), quat_to_yaw(-q), abs_tol=1e-9)


def test_galilean_shift_preserves_relative_quantities():
    p1 = np.array([1.0, 0.0, 1, 0]); p2 = np.array([2.0, 1.0, 1, 0])
    u = np.array([0.3, -0.2]); t = 2.0
    a, _ = galilean_shift(Prop.POSE2, p1, np.zeros(4), u, t)
    b, _ = galilean_shift(Prop.POSE2, p2, np.zeros(4), u, t)
    assert np.allclose(b[:2] - a[:2], p2[:2] - p1[:2])
