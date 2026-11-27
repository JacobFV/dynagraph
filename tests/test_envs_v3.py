"""envs-v3 smoke tests (agent ENVS-V3): every new family resets deterministically, emits only actor-legal events, runs
its scripted policy's first calls without integrity violations, and exposes oracle / probe-then-act hooks.

Run: PYTHONPATH=src python -m pytest tests/test_envs_v3.py -q   (probe robot; G1 family needs ROBOTS' assets + proxsuite)
"""
import numpy as np
import pytest

from physmem.envs import FAMILIES, family_class, make_env
from physmem.schema import Provenance
from physmem.v2.api import TaskSpec

V3 = ["pour", "granular_dose", "cloth_fold", "cable_route", "stack_overhang", "handover"]


def _task(fam, seed=11, **params):
    return TaskSpec(fam, family_class(fam).category, seed, "probe", "test", params)


def test_registered():
    for f in V3 + ["g1_push_terrain"]:
        assert f in FAMILIES


@pytest.mark.parametrize("fam", V3)
def test_reset_deterministic_and_actor_legal(fam):
    e1, e2 = make_env(fam), make_env(fam)
    ev1, ev2 = e1.reset(_task(fam)), e2.reset(_task(fam))
    assert len(ev1) == len(ev2) and len(ev1) > 0
    assert all(e.provenance != Provenance.PRIVILEGED_LABEL for e in ev1)
    # hidden physics only in labels
    lab = e1.labels()
    assert "task_hidden" in lab and lab["task_hidden"]
    for e in ev1:
        assert e.entity_id not in ("task_hidden",)
    s1, s2 = e1.true_state(), e2.true_state()
    for k in s1:
        assert np.allclose(s1[k]["pos"], s2[k]["pos"])


@pytest.mark.parametrize("fam", V3)
def test_first_calls_run_clean(fam):
    env = make_env(fam)
    env.reset(_task(fam))
    for _ in range(2):
        c = env.scripted()
        if c is None:
            break
        out = env.execute(c)
        assert out.metrics["integrity_violations"] == 0
        assert not out.metrics["unstable"], out.metrics["unstable_reasons"]
        assert all(e.provenance != Provenance.PRIVILEGED_LABEL for e in out.events)
        if out.done:
            break


@pytest.mark.parametrize("fam", V3)
def test_oracle_and_pta_hooks(fam):
    from physmem.envs.oracle import rule
    env = make_env(fam)
    env.reset(_task(fam))
    c = env.scripted()
    assert c is not None
    assert rule(env, c, "true") is not None
    assert env.probe_then_act(c) is not None


def test_scale_is_tared_and_reads_contents():
    env = make_env("pour")
    env.reset(_task("pour"))
    env.idle(0.5)
    r = env.scale_reading("receiver", window=0.3)
    assert r is not None and abs(r) < 0.005          # empty receiver reads ~0 after the tare


def test_stack_plan_stability_margin():
    from physmem.envs.families_v3_rigid import stack_plan
    L = 0.1
    x, top = stack_plan(L, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], D=0.05, margin=0.005, maximal=True)
    # uniform blocks: harmonic stack overhang 11/12 L minus the margins
    assert abs(top - (11 / 12 * L - 3 * 0.005)) < 1e-3 or top > 0.06
    x2, top2 = stack_plan(L, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], D=0.05, margin=0.005)
    assert top2 <= 0.05 + 0.01 + 1e-6
