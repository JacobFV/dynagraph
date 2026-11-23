"""Tests for the v2 industrial environments (agent ENVS). Run on gb10-direct:
    scripts/remote.sh "python -m pytest -q tests/test_envs.py"
Robot = test-only probe unless stated."""
import math

import mujoco
import numpy as np
import pytest

from physmem.envs import FAMILIES, family_class, make_env
from physmem.envs.props import XProp, split_events
from physmem.envs.splits import HELDOUT_FAMILIES, make_tasks, seed_bucket
from physmem.memory import WorldBelief
from physmem.schema import Provenance
from physmem.v2.api import SkillCall, TaskSpec

FAST = ["singulation", "peg_insert", "drawer_open", "door_open", "fixture_place", "connector_insert"]


def task(fam, seed=0, split="train", **params):
    return TaskSpec(fam, family_class(fam).category, seed, "probe", split, params)


def _sig(env):
    return np.concatenate([env.d.qpos.copy(), [env.t]])


# ------------------------------------------------------------------ registry / API
def test_registry_counts():
    cats = {f: family_class(f).category for f in FAMILIES}
    assert len(FAMILIES) >= 12
    assert sum(c == "bulk" for c in cats.values()) >= 5
    assert sum(c == "fine" for c in cats.values()) >= 5


@pytest.mark.parametrize("fam", FAMILIES)
def test_reset_and_api(fam):
    env = make_env(fam)
    ev = env.reset(task(fam, 3))
    assert len(ev) > 0
    g = env.goal()
    assert g["family"] == fam and (g["regions"] or g.get("joint_goal"))
    rng = np.random.default_rng(0)
    c = env.candidate_skills(None, 5, rng)
    assert all(isinstance(x, SkillCall) for x in c)
    assert env.scripted() is not None
    img = env.render(width=160, height=120)
    assert img.shape == (120, 160, 3)


# ------------------------------------------------------------------ determinism
@pytest.mark.parametrize("fam", ["bin_clearing", "singulation", "conveyor_sort", "peg_insert"])
def test_determinism_by_seed(fam):
    outs = []
    for _ in range(2):
        env = make_env(fam)
        ev = env.reset(task(fam, 7))
        o = env.execute(env.scripted())
        outs.append((_sig(env), [e.value for e in ev], [e.value for e in o.events], o.metrics["progress"]))
    a, b = outs
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2] and a[3] == b[3]
    env = make_env(fam)
    env.reset(task(fam, 8))
    assert not np.array_equal(_sig(env), a[0])


# ------------------------------------------------------------------ integrity
@pytest.mark.parametrize("fam", FAST)
def test_scripted_episode_no_integrity_violations(fam):
    env = make_env(fam)
    env.reset(task(fam, 1))
    out = None
    for _ in range(4):
        c = env.scripted()
        if c is None:
            break
        out = env.execute(c)
        if out.done:
            break
    assert out is not None and out.metrics["integrity_violations"] == 0


def test_integrity_monitor_catches_pose_write_and_force():
    env = make_env("singulation")
    env.reset(task("singulation", 2))
    e = env.objects()[0]
    adr = env.qadr[e]
    env.run(0.05)
    assert env.metrics_acc["integrity_violations"] == 0
    env.d.qpos[adr] += 0.01          # teleport
    env.run(0.01)
    n1 = env.metrics_acc["integrity_violations"]
    assert n1 > 0
    env.d.xfrc_applied[env.body_id[e], 2] = 5.0
    env.run(0.01)
    assert env.metrics_acc["integrity_violations"] > n1
    env.d.xfrc_applied[:] = 0
    # post-step check alone (another module owning the loop) also sees a teleport
    n2 = env.metrics_acc["integrity_violations"]
    env.d.qpos[adr + 1] += 0.01
    mujoco.mj_step(env.m, env.d)
    env.post_physics()
    assert env.metrics_acc["integrity_violations"] > n2


def test_no_object_welds_in_models():
    for fam in FAST:
        env = make_env(fam)
        env.reset(task(fam, 0))
        assert env._bad_eq == []


# ------------------------------------------------------------------ actor legality
@pytest.mark.parametrize("fam", ["bin_clearing", "drawer_open", "pouch_transfer"])
def test_actor_events_not_privileged_and_ingestable(fam):
    env = make_env(fam)
    ev = env.reset(task(fam, 4))
    out = env.execute(env.scripted())
    evs = ev + out.events
    assert all(e.provenance in (Provenance.SENSOR, Provenance.DERIVED) for e in evs)
    std, ext = split_events(evs)
    assert std and not ext                 # approved schema: every env event is schema-standard
    b = WorldBelief(env.episode_id)
    b.ingest_events(evs, now=env.t)        # raises on privileged / malformed (incl. POSE3 / WRENCH ...)
    # no event value equals a hidden truth exactly (latent quantities never emitted)
    hidden_props = {XProp.GRIP_FRICTION, XProp.RESTITUTION, XProp.COMPLIANCE, XProp.CLEARANCE, XProp.JOINT_FRICTION}
    assert not any(e.property_type in hidden_props for e in evs)


def test_events_respect_latency_and_causality():
    env = make_env("singulation")
    env.reset(task("singulation", 5))
    out = env.execute(env.scripted())
    for e in out.events:
        assert e.received_at >= e.measured_at
        assert e.measured_at <= env.t + 1e-9
    trk = [e for e in out.events if e.source_id == "tracker" and e.property_type == XProp.POSE3]
    lat = np.array([e.received_at - e.measured_at for e in trk])
    assert lat.min() > 0.02


def test_candidates_use_no_truth():
    """candidate generation reads the ActorView only: corrupting the TRUE poses after the last sensor packet
    must not change the candidates."""
    env = make_env("bin_clearing")
    env.reset(task("bin_clearing", 6))
    c1 = env.candidate_skills(None, 6, np.random.default_rng(1))
    q = env.d.qpos.copy()
    env.d.qpos[env._obj_q] += 0.05
    mujoco.mj_forward(env.m, env.d)
    c2 = env.candidate_skills(None, 6, np.random.default_rng(1))
    env.d.qpos[:] = q
    assert [c.params for c in c1] == [c.params for c in c2]
    assert [c.target for c in c1] == [c.target for c in c2]


def test_occlusion_drops_tracker_packets():
    env = make_env("bin_clearing")
    env.reset(task("bin_clearing", 3))
    # a part deep in the bin is sometimes hidden by walls / other parts / the robot: count visibility
    vis = []
    for e in env.objects():
        p, _ = env.true_pose(e)
        vis.append(env.sensors.visible(env.body_id[e], p))
    assert any(vis)
    # put the camera under the bench: everything is occluded
    env.sensors.cam = np.array([0.0, 0.0, 0.3])
    assert not any(env.sensors.visible(env.body_id[e], env.true_pose(e)[0]) for e in env.objects())


# ------------------------------------------------------------------ hidden physics matters
def test_hidden_mass_changes_pick_outcome():
    res = {}
    for m in (0.3, 3.0):
        env = make_env("bin_clearing")
        env.reset(task("bin_clearing", 1000, n_parts=1, hidden={"mass": m, "mu_grip": 0.5}, shapes=("box",)))
        out = env.execute(SkillCall("pick", dict(grip_force=15.0, lift_speed=0.3, depth=0.5), target=env.objects()[0]))
        res[m] = out.result.ok
    assert res[0.3] and not res[3.0]


def test_hidden_friction_changes_push_distance():
    d = {}
    for mu in (0.15, 0.9):
        env = make_env("singulation")
        env.reset(task("singulation", 1000, n_objects=1, hidden={"mu": mu, "mass": 1.0}))
        e = env.objects()[0]
        p0 = env.true_pose(e)[0]
        env.execute(SkillCall("push", dict(theta=math.pi, phi=0.0, speed=0.6, distance=0.05), target=e))
        env.run(1.0)
        d[mu] = float(np.linalg.norm(env.true_pose(e)[0][:2] - p0[:2]))
    assert d[0.15] > 2 * d[0.9]


def test_hidden_joint_friction_changes_drawer_travel():
    q = {}
    for fl in (3.0, 40.0):
        env = make_env("drawer_open")
        env.reset(task("drawer_open", 1001, frictionloss=fl, damping=20.0, mass=5.0, mu_grip=0.6))
        env.execute(env._open(grip_force=15.0, travel=0.2, speed=0.15))
        q[fl] = env.q()
    assert q[3.0] > 0.1 and q[40.0] < 0.02


def test_conveyor_belt_is_driven():
    env = make_env("conveyor_sort")
    env.reset(task("conveyor_sort", 0, belt_speed=0.1))
    j = env.m.joint("belt_slide")
    q0 = env.d.qpos[env.m.jnt_qposadr[j.id]]
    env.run(1.0)
    v = (env.d.qpos[env.m.jnt_qposadr[j.id]] - q0) / 1.0
    assert abs(v - 0.1) < 0.01
    e = env.objects()[0]
    assert abs(env.d.qvel[env.dadr[e]] - 0.1) < 0.02     # parts ride the belt by friction


# ------------------------------------------------------------------ solvable by the scripted baseline
@pytest.mark.parametrize("fam", ["singulation", "peg_insert", "drawer_open", "door_open", "fixture_place",
                                 "connector_insert", "kitting_pocket", "bulk_sweep", "twist_lock", "depalletize"])
def test_scripted_solves_some_seeds(fam):
    ok = 0
    for seed in range(6 if fam == "kitting_pocket" else 3):     # kitting baseline succeeds on ~30 % of seeds
        env = make_env(fam)
        env.reset(task(fam, seed))
        while True:
            c = env.scripted()
            if c is None:
                break
            out = env.execute(c)
            if out.done:
                break
        ok += env.success()
    assert ok >= 1


# ------------------------------------------------------------------ splits
def test_splits_by_root_seed_disjoint():
    tr = {t.seed for t in make_tasks("singulation", "train", 50)}
    va = {t.seed for t in make_tasks("singulation", "val", 20)}
    te = {t.seed for t in make_tasks("singulation", "test", 20)}
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert all(seed_bucket(s) == "train" for s in tr)
    for fam in HELDOUT_FAMILIES.values():
        with pytest.raises(ValueError):
            make_tasks(fam, "train", 1)


def test_heldout_layout_rotates_station():
    a, b = make_env("peg_insert"), make_env("peg_insert")
    a.reset(task("peg_insert", 11))
    b.reset(TaskSpec("peg_insert", "fine", 11, "probe", "train", dict(layout="rot180")))
    ra, rb = a.region("hole")["center"], b.region("hole")["center"]
    assert np.allclose([ra[0], ra[1]], [-rb[0], -rb[1]], atol=1e-9)


def test_compact_event_storage_roundtrip():
    from physmem.envs.io import events_from_array, events_to_array
    env = make_env("drawer_open")
    ev = env.reset(task("drawer_open", 2))
    ev += env.execute(env.scripted()).events
    back = events_from_array(events_to_array(ev))
    assert len(back) == len(ev)
    for a, b in zip(ev, back):
        assert a.event_id == b.event_id and a.entity_id == b.entity_id and a.property_type == b.property_type
        assert np.allclose(a.value, b.value, atol=1e-5) and a.measured_at == b.measured_at
        assert a.source_id == b.source_id and a.provenance == b.provenance


def test_reset_redraws_scenes_that_drop_objects():
    for s in range(6):
        env = make_env("bin_clearing")
        env.reset(task("bin_clearing", s))
        assert env.metrics_acc["drops"] == 0 and env.metrics_acc["sim_instabilities"] == 0


def test_skill_log_has_commanded_reference():
    env = make_env("singulation")
    env.reset(task("singulation", 0))
    out = env.execute(env.scripted())
    t, pos, vel = out.result.log["cmd"]
    assert len(t) > 10 and pos.shape == (len(t), 3) and vel.shape == (len(t), 3)
    assert {"slip", "jam", "stall", "drop", "collision"} <= set(out.metrics)


def test_depalletize_flag_path_and_payload_rule():
    env = make_env("depalletize")
    env.reset(task("depalletize", 3, layers=1, hidden={"mass": 25.0}))
    heavy = [e for e in env.cartons() if env.overweight(e)]
    assert heavy                                        # probe payload 20 kg vs 12-22 kg cartons
    out = env.execute(SkillCall("handover", {}, target=heavy[0]))
    assert out.result.status == "flagged" and heavy[0] in env.flagged
    assert out.metrics["integrity_violations"] == 0


def test_conveyor_never_retargets_sorted_items():
    env = make_env("conveyor_sort")
    env.reset(task("conveyor_sort", 1))
    seen_sorted = set()
    for _ in range(14):
        c = env.scripted()
        if c is None:
            break
        if c.skill == "pick":
            assert c.target not in seen_sorted
        out = env.execute(c)
        seen_sorted |= env._sorted_ids()
        if out.done:
            break


def test_mobile_robot_gets_lower_bench():
    from physmem.envs.stations import MOBILE_BENCH_TOP
    env = make_env("peg_insert")
    env.reset(TaskSpec("peg_insert", "fine", 0, "probe", "train", dict(bench_top=MOBILE_BENCH_TOP)))
    assert abs(env.surface_z - MOBILE_BENCH_TOP) < 1e-9


def test_resimulate_is_deterministic_and_interventions_change_outcome():
    env = make_env("bin_clearing")
    env.reset(task("bin_clearing", 1000, n_parts=1, shapes=("box",), hidden={"mass": 0.3, "mu_grip": 0.6}))
    e = env.objects()[0]
    snap = env.snapshot()
    call = SkillCall("pick", dict(grip_force=15.0, lift_speed=0.3, depth=0.5), target=e)
    a = env.resimulate(snap, {}, call)
    qa = env.d.qpos.copy()
    b = env.resimulate(snap, {}, call)
    assert np.allclose(qa, env.d.qpos) and a.result.ok == b.result.ok and a.result.ok
    c = env.resimulate(snap, {e: {"mass": 3.0}}, call)          # heavy: slips
    assert not c.result.ok
    d = env.resimulate(snap, {e: {"mu_grip": 0.08}}, call)      # oily: slips
    assert not d.result.ok
    assert env.metrics_acc["integrity_violations"] == 0


def test_episode_timeout_aborts_cleanly_even_through_skill_exception_handlers(monkeypatch):
    """ROBOTS round 5: a wall-clock timeout fired mid-skill and the episode continued from a broken state. The
    timeout is a BaseException: skill code that catches Exception (like the real-robot adapter) cannot swallow it,
    and the runner issues no further calls."""
    from physmem.envs import base as B
    from physmem.envs import probe as PR
    from physmem.envs.rollout import EpisodeTimeout, run_episode
    calls = {"n": 0}
    orig_run = B.StationEnv.run

    def run(self, T, stop=None):
        if getattr(self, "_settling", False) is False and self.n_calls >= 2:
            raise EpisodeTimeout()
        return orig_run(self, T, stop)
    orig_skill = PR.ProbeSkills.run

    def skill(self, env, call):          # adapter-style: converts Exceptions into failed results
        calls["n"] += 1
        try:
            return orig_skill(self, env, call)
        except Exception as ex:          # noqa: BLE001
            from physmem.v2.api import EffortMeter, SkillResult
            return SkillResult(False, f"error:{ex!r}", EffortMeter(), 0.0)
    monkeypatch.setattr(B.StationEnv, "run", run)
    monkeypatch.setattr(PR.ProbeSkills, "run", skill)
    rec = run_episode(task("bin_clearing", 3, policy="scripted"), keep_events=False)
    assert rec["aborted"] and rec["error"] == "timeout"
    assert calls["n"] == 2 and len(rec["steps"]) == 1          # the interrupted call is not recorded, none after it
    assert not rec["success"]
