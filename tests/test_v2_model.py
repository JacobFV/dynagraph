"""tests for the v2 typed physical-memory world model, synthetic world, tensorizer and planner (agent MODEL-V2)."""
import copy
import glob
import math
import os

import numpy as np
import pytest
import torch

from physmem.models.v2 import spec as S
from physmem.models.v2 import synth as Y
from physmem.models.v2.model import PhysMemV2, TF_LOGVAR, count_params, v2_loss
from physmem.models.v2.tensorize import (K_MAX, FamilyStore, V2Batch, V2Labels, build_family_arrays, single_batch)
from physmem.planner import CEMPlanner, PlannerConfig
from physmem.v2 import api


def _emb(r):
    return S.embodiment_vector(Y.ROBOTS[r].spec)


def _store(fam="slide", n=40, seed=0):
    eps = []
    for i in range(n):
        ep = Y.generate_episode(fam, Y.TRAIN_ROBOTS[i % 4], seed + i)
        eps.append((fam, ep.robot, ep.steps))
    return FamilyStore(build_family_arrays(eps, _emb), fam, "cpu")


def _model(variant="typed", store=None):
    torch.manual_seed(0)
    m = PhysMemV2(variant, d=32)
    if store is not None:
        from physmem.v2_train import fit_scalers
        fit_scalers(m, {"x": store})
    return m.eval()


# ----------------------------------------------------------------- spec
def test_latent_roundtrip_and_labels():
    for name, tr, loc, sc, _ in S.LATENTS:
        x = 2.5 if tr == "log" else 0.37
        assert abs(S.z_to_latent(name, S.latent_to_z(name, x)) - x) < 1e-9
    z, m = S.labels_to_z({"mass": 0.8, "friction": 0.5, "bogus": 3.0})
    assert m.sum() == 2 and abs(z[S.LATENT_INDEX["mass"]]) < 1e-6 and abs(z[S.LATENT_INDEX["friction"]]) < 1e-6


def test_segment_recorder_and_resample():
    rec = S.SegmentRecorder([1.0, 2.0, 0.0])
    for i in range(137):
        t = i / 200
        rec.tick(t, [1 + t, 2, 0], [1, 0, 0], [1 + t, 2, 0], [1, 0, 0], np.r_[-3.0, 0, 0, 0, 0, 0], 1.0, 0.03,
                 None if i % 7 else [1 + t, 2, 0], 1.0)
    seg = rec.finish()
    assert seg["x"].shape == (137, S.N_CH)
    assert abs(seg["x"][0, S.CH["cmd_x"]]) < 1e-6        # relative to target start
    xr = S.resample_segment(seg["x"])
    assert xr.shape == (S.SEG_T, S.N_CH)
    assert xr[:, S.CH["obj_valid"]].sum() >= 15          # sparse validity flags survive resampling
    f = S.analytic_features(xr)
    assert f.shape == (S.N_AN,) and np.isfinite(f).all()


def test_embodiment_vector():
    for r in Y.ROBOTS.values():
        v = S.embodiment_vector(r.spec)
        assert v.shape == (S.EMB_DIM,) and np.isfinite(v).all()


# ----------------------------------------------------------------- synthetic world physics
@pytest.mark.parametrize("fam", Y.FAMILIES)
def test_synth_episode_contract(fam):
    ep = Y.generate_episode(fam, "synth_arm_a", 3)
    for st in ep.steps:
        it = st["inter"]
        assert it.segment.shape == (S.SEG_T, S.N_CH) and np.isfinite(it.segment).all()
        assert it.outcome_cont.shape == (S.N_CONT,) and it.outcome_events.shape == (S.N_EVENTS,)
        assert set(it.labels) == set(Y.FAMILY_LATENTS[fam])
        assert it.effort_cost >= 0


def _sim_mean(fam, lat, params, R=12, state=None, robot="synth_arm_a"):
    out = []
    for r in range(R):
        st = copy.deepcopy(state) if state is not None else Y.init_state(fam, np.random.default_rng(0), lat)
        out.append(Y.STEP[fam](st, lat, params, Y.ROBOTS[robot], np.random.default_rng(r), np.zeros(6)))
    return out


def test_synth_latents_change_outcomes_physically():
    p = {"dir": 1.0, "dist": 0.1, "speed": 0.5}
    lo = np.mean([s.cont[0] for s in _sim_mean("slide", dict(mass=1.0, friction=0.2), p)])
    hi = np.mean([s.cont[0] for s in _sim_mean("slide", dict(mass=1.0, friction=0.8), p)])
    assert lo > hi + 0.03            # low friction slides much farther
    p = {"accel": 3.0, "grip_force": 20.0, "height": 0.12, "speed": 0.3}
    light = np.mean([s.events[0] for s in _sim_mean("lift", dict(mass=0.3, grip_friction=0.6, stiffness=5e3), p)])
    heavy = np.mean([s.events[0] for s in _sim_mean("lift", dict(mass=3.0, grip_friction=0.6, stiffness=5e3), p)])
    assert light > 0.9 and heavy < 0.1
    p = {"duration": 0.3, "force": 10.0}
    st = dict(x=0.0, goal=0.2)
    a = np.mean([s.cont[0] for s in _sim_mean("drawer", dict(damping=10.0, breakaway=5.0), p, state=st)])
    b = np.mean([s.cont[0] for s in _sim_mean("drawer", dict(damping=60.0, breakaway=5.0), p, state=st)])
    assert a > b


def test_analytic_features_track_physics():
    """least-squares / static-load features are monotone in the hidden property (actor-legal signals only)."""
    m_feat = []
    for m in (0.3, 1.0, 3.0):
        st = Y.STEP["lift"](Y.init_state("lift", np.random.default_rng(0), {}), dict(mass=m, grip_friction=1.0, stiffness=5e3),
                            {"accel": 2.0, "grip_force": 90.0, "height": 0.12, "speed": 0.2}, Y.ROBOTS["synth_arm_a"],
                            np.random.default_rng(0), np.zeros(6))
        m_feat.append(S.analytic_features(st.seg)[1])
    assert m_feat[0] < m_feat[1] < m_feat[2]


# ----------------------------------------------------------------- tensorizer
def test_store_history_is_causal_and_same_episode():
    st = _store("slide", n=12)
    idx = torch.arange(st.n)
    b, lab = st.gather(idx)
    assert b.seg.shape == (st.n, K_MAX, S.SEG_T, S.N_CH)
    # history mask count == step index (capped)
    assert torch.equal(b.seg_mask.sum(1), torch.clamp(st.k, max=K_MAX))
    # history segments equal the stored segments of earlier steps of the SAME episode
    for i in range(st.n):
        k = int(st.k[i])
        for j in range(min(k, K_MAX)):
            src = int(st.ep_start[i]) + max(k - K_MAX, 0) + j
            assert st.ep_index[src] == st.ep_index[i] and int(st.k[src]) < k
            assert torch.allclose(b.seg[i, j], st.seg[src].float())
    # actor batch carries no label fields
    assert not ({"z", "z_mask", "labels", "events"} & set(V2Batch.__dataclass_fields__))
    b0, _ = st.gather(idx, k_override=torch.zeros_like(idx))
    assert not b0.seg_mask.any()


def test_entity_cap_keeps_target_first():
    fr = Y.entity_frame("sweep", Y.init_state("sweep", np.random.default_rng(1), {}), "synth_arm_a")
    from physmem.models.v2.tensorize import frame_tokens
    tok, tgt = frame_tokens(fr, np.random.default_rng(0), cap=16)
    assert len(tok) == 16 and tgt == 0 and tok[0, len(S.ENT_TYPES) + 13] == 1


# ----------------------------------------------------------------- model structure
@pytest.mark.parametrize("variant", ["typed", "generic", "nomem"])
def test_forward_shapes_and_loss(variant):
    st = _store("lift", n=8)
    m = _model(variant, st).train()
    b, lab = st.gather(torch.arange(st.n))
    o = m(b)
    assert o["cont_mu"].shape == (st.n, S.N_CONT) and o["ev_logits"].shape == (st.n, S.N_EVENTS)
    d = m.decode(o)
    assert (d["risk"] >= 0).all() and (d["risk"] <= 1).all() and (d["effort"] >= 0).all()
    loss, logs = v2_loss(m, b, lab)
    loss.backward()
    assert math.isfinite(loss.item())


def test_param_budget_typed_vs_generic():
    a, b = count_params(PhysMemV2("typed")), count_params(PhysMemV2("generic"))
    assert abs(a - b) / a < 0.10


def test_typed_slots_are_the_only_latent_pathway():
    """non-bypass: with ALL typed slots overwritten, the prediction is independent of the interaction history
    (the outcome head cannot read latent physics from anywhere else)."""
    st = _store("slide", n=10)
    m = _model("typed", st)
    idx = torch.nonzero(st.k >= 2).flatten()[:6]
    b, lab = st.gather(idx)
    b2, _ = st.gather(idx, k_override=torch.zeros_like(idx))       # no history at all
    Z = torch.randn(len(idx), S.N_LATENT)
    LV = torch.full_like(Z, TF_LOGVAR)
    with torch.no_grad():
        o1 = m(b, z_override=(Z, LV))
        o2 = m(b2, z_override=(Z, LV))
        o3 = m(b, z_override=(Z + 1.0, LV))
    assert torch.allclose(o1["cont_mu"], o2["cont_mu"], atol=1e-5)
    assert torch.allclose(o1["ev_logits"], o2["ev_logits"], atol=1e-5)
    assert not torch.allclose(o1["cont_mu"], o3["cont_mu"], atol=1e-4)  # slots do enter the head


def test_history_order_and_entity_permutation_invariance():
    st = _store("sweep", n=6)
    m = _model("typed", st)
    idx = torch.nonzero(st.k >= 3).flatten()[:3]
    b, _ = st.gather(idx)
    with torch.no_grad():
        o = m(b)
        perm = torch.randperm(K_MAX)
        bp = V2Batch(**{f: getattr(b, f) for f in V2Batch.__dataclass_fields__})
        for f in ("seg", "seg_mask", "seg_call", "seg_an", "seg_age", "seg_on_target"):
            setattr(bp, f, getattr(b, f)[:, perm])
        op = m(bp)
        assert torch.allclose(o["mem"]["z_mu"], op["mem"]["z_mu"], atol=1e-4)
        # permute non-target entities
        N = b.ent.shape[1]
        ep = torch.cat([torch.zeros(1, dtype=torch.long), 1 + torch.randperm(N - 1)])
        be = V2Batch(**{f: getattr(b, f) for f in V2Batch.__dataclass_fields__})
        be.ent, be.ent_mask = b.ent[:, ep], b.ent_mask[:, ep]
        oe = m(be)
        assert torch.allclose(o["cont_mu"], oe["cont_mu"], atol=1e-3)


def test_precision_fusion_shrinks_with_evidence():
    st = _store("drawer", n=10)
    m = _model("typed", st)
    idx = torch.nonzero(st.k >= 3).flatten()[:8]
    with torch.no_grad():
        v = [m(st.gather(idx, k_override=torch.full_like(idx, j))[0])["mem"]["z_logvar"] for j in range(4)]
    for j in range(3):
        assert (v[j + 1] <= v[j] + 1e-5).all()


def test_generic_rejects_interventions():
    st = _store("slide", n=4)
    m = _model("generic", st)
    b, _ = st.gather(torch.arange(st.n))
    with pytest.raises(AssertionError):
        m(b, z_override=(torch.zeros(st.n, S.N_LATENT), torch.zeros(st.n, S.N_LATENT)))


# ----------------------------------------------------------------- record path + planner
def test_single_batch_matches_store_and_ignores_labels():
    st = _store("insert", n=3)
    i = int(torch.nonzero(st.k == 2).flatten()[0])
    ep_i = st.ep_index[i]
    eps = []
    ep = Y.generate_episode("insert", Y.TRAIN_ROBOTS[ep_i % 4], ep_i)
    hist = [s["inter"] for s in ep.steps[:2]]
    for h in hist:
        h.labels = {"offset_x": 123.0}    # must not matter (labels never enter V2Batch)
    b = single_batch(hist, ep.steps[2]["frame"], _emb(ep.robot), ep.steps[2]["inter"].call_vec[None])
    b2, _ = st.gather(torch.tensor([i]))
    assert torch.allclose(b.seg[0, :2], b2.seg[0, :2], atol=2e-3)
    assert torch.allclose(b.call, b2.call)


def test_planner_decides_within_bounds_and_voi_runs():
    st = _store("lift", n=6)
    m = _model("typed", st)
    pl = CEMPlanner(m, "typed", PlannerConfig(pop=32, iters=2, n_z=8))
    env = Y.SynthEnv()
    env.reset(api.TaskSpec("lift", "logistics", 5, "synth_arm_a", "test"))
    d = pl.decide([], env.frame(), _emb("synth_arm_a"), "pick", env.param_bounds(), np.random.default_rng(0),
                  probe_params=[c.params for c in env.probe_skills()])
    for k, (lo, hi) in env.param_bounds().items():
        assert lo - 1e-9 <= d.params[k] <= hi + 1e-9
    assert "evpi" in d.info and np.isfinite(d.info["evpi"])
    out = env.execute(d.call("target"))
    assert out.result.log["segment"]["x"].shape == (S.SEG_T, S.N_CH)


def test_harness_paired_episode_runs():
    from physmem.models.v2 import harness as H
    st = _store("drawer", n=6)
    m = _model("typed", st)
    pl = CEMPlanner(m, "typed", PlannerConfig(pop=16, iters=1, n_z=4))
    tasks = [api.TaskSpec("drawer", "articulated", s, "synth_arm_a", "test") for s in range(2)]
    res = H.paired_eval(Y.SynthEnv, tasks, {"heuristic": ("heuristic", None), "planner_typed": ("planner", pl)})
    s = H.summarize(res)
    assert s["heuristic"]["n"] == 2 and "paired_success_planner_typed_minus_this" in s["heuristic"]


# ----------------------------------------------------------------- learned-model gates (need a trained checkpoint)
CK = sorted(glob.glob("results/v2/checkpoints/v2_synth_r1_typed_s[0-9].pt"))


@pytest.mark.skipif(not CK or not os.path.exists("data/v2_synth/slide_test.npz"), reason="no trained synth checkpoint")
def test_trained_latent_recovery_gate():
    from physmem.v2_train import evaluate, load_model, load_stores
    m, _ = load_model(CK[-1], torch.device("cpu"))
    st = load_stores("data/v2_synth", "test", torch.device("cpu"), ["slide", "drawer"])
    for s in st.values():   # subsample for speed
        pass
    ev = evaluate(m, st, max_hist=(1, 2))
    assert ev["slide"]["latent:friction:corr@k2"] >= 0.7
    assert ev["drawer"]["latent:damping:corr@k1"] >= 0.7
    # uncertainty shrinks with evidence
    assert ev["slide"]["latent:friction:std@k2"] < ev["slide"]["latent:friction:std@k1"]


# ----------------------------------------------------------------- ENVS adapter
def test_env_adapter_params_segment_and_labels():
    from physmem.models.v2 import env_adapter as A
    from physmem.schema import ObservationEvent, Provenance
    if A.XProp is None:
        pytest.skip("envs.props not importable")
    p0 = np.array([0.5, -0.2, 0.8])
    params = {"x0": 0.6, "y1": -0.1, "speed": 0.3, "grip_z": 0.9, "path": [1, 2]}
    rel = A.rel_params(params, p0)
    assert abs(rel["x0"] - 0.1) < 1e-12 and abs(rel["y1"] - 0.1) < 1e-12 and rel["speed"] == 0.3
    assert abs(rel["grip_z"] - 0.1) < 1e-12 and rel["path"] == [1, 2]
    back = A.abs_params(rel, p0)
    assert all(abs(back[k] - params[k]) < 1e-12 for k in ("x0", "y1", "speed", "grip_z"))
    # synthetic event stream: hand moves +x at 0.2 m/s pushing with 5 N, object follows, 25 Hz F/T, 10 Hz tracker
    evs, n = [], 0

    def ev(eid, prop, val, t):
        nonlocal n
        n += 1
        return ObservationEvent(f"e{n}", "ep", eid, prop, tuple(val), (0.0,) * len(val), "world", t, t, "s",
                                Provenance.SENSOR)
    for i in range(51):
        t = i * 0.04
        x = 0.2 * t
        evs += [ev("robot/hand", A.XProp.EE_POSE, (p0[0] - 0.05 + x, p0[1], p0[2], 1, 0, 0, 0), t),
                ev("robot/hand", A.XProp.WRENCH, (-5.0, 0, 0, 0, 0, 0), t),
                ev("robot/hand", A.XProp.GRIPPER, (0.04, 0.0, 0.0), t),
                ev("robot/hand", A.XProp.TACTILE, (0, 0, 1, 5.0), t)]
        if i % 2 == 0:
            evs.append(ev("box", A.XProp.POSE3, (p0[0] + x, p0[1], p0[2], 1, 0, 0, 0), t))
    idx = A.EventIndex()
    idx.add(evs)
    seg = A.build_segment(idx, 0.0, 2.0, ["box"], p0)
    assert seg.shape == (51, S.N_CH)
    assert np.allclose(seg[:, S.CH["fx"]], -5.0) and seg[:, S.CH["contact"]].all()
    assert abs(seg[-1, S.CH["obj_x"]] - 0.4) < 1e-6 and abs(seg[-1, S.CH["ee_vx"]] - 0.2) < 1e-3
    assert 20 <= seg[:, S.CH["obj_valid"]].sum() <= 27
    f = S.analytic_features(S.resample_segment(seg))
    assert np.isfinite(f).all() and f[0] > 1.0           # sustained push force detected
    lab = A.map_labels({"hidden": {"box": {"mass": 1.2, "mu": 0.4, "mu_grip": 0.7, "restitution": 0.1}}}, ["box"])
    assert lab == {"mass": 1.2, "friction": 0.4, "grip_friction": 0.7}
    pile = A.map_labels({"hidden": {"a": {"mass": 0.1, "mu": 0.3}, "b": {"mass": 0.2, "mu": 0.5}}}, ["a", "b"])
    assert abs(pile["mass"] - 0.3) < 1e-9 and abs(pile["bulk_friction"] - 0.4) < 1e-9
