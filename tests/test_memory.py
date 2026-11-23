import math

import numpy as np
import pytest

from physmem.memory import PrivilegedInputError, UnknownPropertyError, WorldBelief
from physmem.schema import EntityDescriptor, EntityType, ObservationEvent, Prop, Provenance

EP = "ep0"


def ev(eid, ent, prop, value, var, t, src="s1", recv=None, prov=Provenance.SENSOR):
    return ObservationEvent(eid, EP, ent, prop, tuple(value), tuple(var), "world", t, t if recv is None else recv, src, prov)


def make():
    b = WorldBelief(EP)
    b.add_entity(EntityDescriptor("box", EntityType.OBJECT))
    return b


def test_authoritative_correction_beats_many_stale_events():
    b = make()
    stale = [ev(f"e{i}", "box", Prop.MASS, (1.0,), (0.01,), 0.01 * i, src=f"s{i % 3}") for i in range(200)]
    b.ingest_events(stale, now=2.0)
    m, v, _ = b.resolve_query("box", Prop.MASS)
    assert abs(m[0] - 1.0) < 1e-9
    b.ingest_events([ev("auth", "box", Prop.MASS, (0.4,), (1e-4,), 2.5, src="scale")], now=3.0, authoritative_sources=["scale"])
    m, v, rec = b.resolve_query("box", Prop.MASS)
    assert abs(m[0] - 0.4) < 1e-9, "authoritative correction must be visible on the next query"
    # obsolete evidence replayed afterwards is ignored
    b.ingest_events([ev("late", "box", Prop.MASS, (1.0,), (0.01,), 1.0, src="s1")], now=3.1)
    assert abs(b.resolve_query("box", Prop.MASS)[0][0] - 0.4) < 1e-9


def test_weak_new_measurement_does_not_force_certainty():
    b = make()
    b.ingest_events([ev("a", "box", Prop.FRICTION, (0.5,), (0.0025,), 0.0, src="prior")], now=0.0)
    b.ingest_events([ev("b", "box", Prop.FRICTION, (0.9,), (0.25,), 1.0, src="noisy")], now=1.0)
    m, v, _ = b.resolve_query("box", Prop.FRICTION)
    assert m[0] < 0.52, "weak recent estimate must not override a strong prior"
    assert v[0] < 0.0025 and v[0] > 0.0024  # precision-weighted, not collapsed to 0


def test_duplicates_and_same_source_do_not_reduce_variance():
    b = make()
    e = ev("x", "box", Prop.SIZE, (0.03, 0.03, 0.03), (1e-4,) * 3, 0.0)
    b.ingest_events([e], now=0.0)
    v1 = b.resolve_query("box", Prop.SIZE)[1].copy()
    b.ingest_events([e, e, e], now=0.1)  # identical replays
    assert np.allclose(b.resolve_query("box", Prop.SIZE)[1], v1)
    # same source, new event ids: correlated -> no variance reduction
    b.ingest_events([ev(f"x{i}", "box", Prop.SIZE, (0.03, 0.03, 0.03), (1e-4,) * 3, 0.1 * i) for i in range(1, 20)], now=2.0)
    assert np.allclose(b.resolve_query("box", Prop.SIZE)[1], v1)
    # an independent second source does reduce it
    b.ingest_events([ev("y", "box", Prop.SIZE, (0.03, 0.03, 0.03), (1e-4,) * 3, 2.0, src="s2")], now=2.0)
    assert np.allclose(b.resolve_query("box", Prop.SIZE)[1], v1 / 2)


def test_deletion_removes_from_every_view_and_reinsertion_needs_new_evidence():
    b = make()
    b.ingest_events([ev("p0", "box", Prop.POSE2, (0, 0, 1, 0), (1e-6,) * 4, 0.0)], now=0.0)
    b.ingest_events([], now=1.0)
    b.delete("box", Prop.POSE2)
    with pytest.raises(UnknownPropertyError):
        b.resolve_query("box", Prop.POSE2)
    assert not any(e.property_type == Prop.POSE2 for e in b.available_events())
    assert not any(r.property_type == Prop.POSE2 for r in b.records())
    # stale (pre-deletion) evidence arriving late cannot resurrect it
    b.ingest_events([ev("p_old", "box", Prop.POSE2, (5, 5, 1, 0), (1e-6,) * 4, 0.5, recv=1.2)], now=1.2)
    assert not b.has("box", Prop.POSE2)
    b.ingest_events([ev("p1", "box", Prop.POSE2, (0.1, 0, 1, 0), (1e-6,) * 4, 1.3)], now=1.3)
    assert abs(b.resolve_query("box", Prop.POSE2)[0][0] - 0.1) < 1e-9


def test_retraction():
    b = make()
    b.ingest_events([ev("a", "box", Prop.MASS, (1.0,), (0.01,), 0.0, src="a"), ev("b", "box", Prop.MASS, (3.0,), (0.01,), 0.0, src="b")], now=0.0)
    assert abs(b.resolve_query("box", Prop.MASS)[0][0] - 2.0) < 1e-9
    b.retract("b")
    assert abs(b.resolve_query("box", Prop.MASS)[0][0] - 1.0) < 1e-9
    b.ingest_events([ev("b", "box", Prop.MASS, (3.0,), (0.01,), 0.0, src="b")], now=0.1)  # replay of retracted id
    assert abs(b.resolve_query("box", Prop.MASS)[0][0] - 1.0) < 1e-9


def test_out_of_order_delivery_does_not_overwrite_newer_pose():
    b = make()
    b.ingest_events([ev("new", "box", Prop.POSE2, (1, 0, 1, 0), (1e-6,) * 4, 1.0)], now=1.0)
    b.ingest_events([ev("old", "box", Prop.POSE2, (0, 0, 1, 0), (1e-6,) * 4, 0.5, recv=1.1)], now=1.1)
    assert abs(b.resolve_query("box", Prop.POSE2)[0][0] - 1.0) < 1e-12


def test_future_events_are_buffered_until_available():
    b = make()
    b.ingest_events([ev("f", "box", Prop.POSE2, (9, 9, 1, 0), (1e-6,) * 4, 0.9, recv=2.0)], now=1.0)
    assert not b.has("box", Prop.POSE2)
    b.ingest_events([], now=2.0)
    assert b.has("box", Prop.POSE2)


def test_stale_pose_grows_uncertain_but_mass_does_not_decay():
    b = make()
    b.ingest_events([ev("p", "box", Prop.POSE2, (0, 0, 1, 0), (1e-6,) * 4, 0.0), ev("m", "box", Prop.MASS, (1.0,), (0.01,), 0.0)], now=0.0)
    v0p = b.resolve_query("box", Prop.POSE2)[1][0]
    v0m = b.resolve_query("box", Prop.MASS)[1][0]
    b.ingest_events([], now=10.0)
    assert b.resolve_query("box", Prop.POSE2)[1][0] > 100 * v0p
    assert b.resolve_query("box", Prop.MASS)[1][0] == v0m


def test_privileged_rejected_at_actor_boundary():
    b = make()
    with pytest.raises(PrivilegedInputError):
        b.ingest_events([ev("t", "box", Prop.FRICTION, (0.5,), (0.0,), 0.0, prov=Provenance.PRIVILEGED_LABEL)], now=0.0)
    evaluator = WorldBelief(EP, actor_boundary=False)
    evaluator.ingest_events([ev("t", "box", Prop.FRICTION, (0.5,), (0.0,), 0.0, prov=Provenance.PRIVILEGED_LABEL)], now=0.0)


def test_model_estimate_write_and_provenance():
    b = make()
    b.ingest_events([], now=1.0)
    b.write_estimate("box", Prop.FRICTION, [0.4], [0.01], source="b11")
    m, v, rec = b.resolve_query("box", Prop.FRICTION)
    assert abs(m[0] - 0.4) < 1e-12 and rec.provenance_refs[0].startswith("estimate:b11")
    snap = b.snapshot()
    assert snap["records"][0]["kind"] == "PREDICTED"
