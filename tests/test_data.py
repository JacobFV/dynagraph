"""Worlds + data pipeline tests (agent DATA)."""
import copy
import math

import numpy as np
import pytest
import torch

from physmem import data as D
from physmem import worlds as W
from physmem.schema import EntityType, Prop, HORIZONS, STATE_DIM, VALUE_DIM


def test_determinism_by_seed():
    a = D.generate_root("train", 11, base_seed=3)
    b = D.generate_root("train", 11, base_seed=3)
    c = D.generate_root("train", 11, base_seed=4)
    for k in D.EV_FIELDS:
        np.testing.assert_array_equal(a["events"][k], b["events"][k])
    for la, lb in zip(a["labels"], b["labels"]):
        np.testing.assert_array_equal(la["future"], lb["future"])
    assert a["params"] == b["params"]
    assert a["params"] != c["params"]


def _ep(mu=0.5, mass=1.0, actions=None, **kw):
    acts = actions or [W.PushAction(0.0, 0.0, 0.3, 0.2) for _ in range(W.K_PUSHES)]
    return W.EpisodeParams(world="w2", friction=mu, mass=mass, half=(0.03, 0.03, 0.03), pos=(0.0, 0.0), yaw=0.3,
                           pusher="sphere15", actions=acts, pusher_park=(0.3, 0.3), **kw)


def test_box_at_rest_stays_at_rest():
    # pushes too slow/short to ever close the 3mm gap -> no contact -> box must not move
    tr = W.simulate(_ep(actions=[W.PushAction(0.0, 0.0, 0.002, 0.1)] * W.K_PUSHES))
    assert tr["f_push"].max() == 0.0
    disp = np.abs(tr["obj_pose"][:, 0, :] - tr["obj_pose"][0, 0, :]).max(0)
    assert disp[:2].max() < 2e-4 and disp[2] < 2e-3, disp
    assert np.abs(tr["obj_vel"][-200:, 0]).max() < 1e-3


def _slide_after_push(mu, mass=1.0):
    tr = W.simulate(_ep(mu=mu, mass=mass))
    i_end = int(round(tr["push_end"][0] / W.REC_DT))
    i_h = int(round((tr["push_start"][0] + 2.0) / W.REC_DT))
    assert tr["f_push"][i_end + 10: i_h].max() == 0.0  # pusher stopped, box slides free
    v_end = np.hypot(*tr["obj_vel"][i_end + 1, 0, :2])
    slide = np.hypot(*(tr["obj_pose"][i_h, 0, :2] - tr["obj_pose"][i_end + 1, 0, :2]))
    return slide, v_end


def test_higher_friction_shorter_slide():
    s_lo, v_lo = _slide_after_push(0.3)
    s_hi, v_hi = _slide_after_push(0.7)
    print(f"slide after push: mu=0.3 -> {s_lo*1000:.2f}mm (v_end {v_lo:.3f}), mu=0.7 -> {s_hi*1000:.2f}mm "
          f"(v_end {v_hi:.3f}); Coulomb v^2/2mug = {v_lo**2/(2*0.3*9.81)*1000:.2f} / {v_hi**2/(2*0.7*9.81)*1000:.2f}mm")
    assert s_lo > s_hi > 0
    # roughly Coulomb: within a factor 2 of v^2/(2 mu g)
    for s, v, mu in ((s_lo, v_lo, 0.3), (s_hi, v_hi, 0.7)):
        pred = v**2 / (2 * mu * 9.81)
        assert 0.5 * pred < s < 2.0 * pred, (s, pred)


def _se2_cols(cols, tx, ty, a):
    c, s = math.cos(a), math.sin(a)
    out = copy.deepcopy(cols)
    P = cols["prop"]
    for p in (Prop.POSE2,):
        m = P == p
        x, y = cols["val"][m, 0].astype(np.float64), cols["val"][m, 1].astype(np.float64)
        out["val"][m, 0] = c * x - s * y + tx
        out["val"][m, 1] = s * x + c * y + ty
        cc, ss = cols["val"][m, 2].astype(np.float64), cols["val"][m, 3].astype(np.float64)
        out["val"][m, 2] = c * cc - s * ss
        out["val"][m, 3] = s * cc + c * ss
    for p in (Prop.VEL2, Prop.COMMAND):
        m = P == p
        x, y = cols["val"][m, 0].astype(np.float64), cols["val"][m, 1].astype(np.float64)
        out["val"][m, 0] = c * x - s * y
        out["val"][m, 1] = s * x + c * y
    return out


def _se2_traj(tr, tx, ty, a):
    c, s = math.cos(a), math.sin(a)
    out = copy.deepcopy(tr)
    x, y = tr["obj_pose"][..., 0], tr["obj_pose"][..., 1]
    out["obj_pose"][..., 0] = c * x - s * y + tx
    out["obj_pose"][..., 1] = s * x + c * y + ty
    out["obj_pose"][..., 2] = tr["obj_pose"][..., 2] + a
    vx, vy = tr["obj_vel"][..., 0], tr["obj_vel"][..., 1]
    out["obj_vel"][..., 0] = c * vx - s * vy
    out["obj_vel"][..., 1] = s * vx + c * vy
    return out


def test_canonical_frame_equivariance_record_level():
    """Global SE(2) transform of the world (events + ground truth): tensors and labels unchanged."""
    r = D.generate_root("train", 21, base_seed=5)
    ep = W.EpisodeParams(**{**r["params"], "actions": [W.PushAction(*a) for a in r["params"]["actions"]]})
    tr = W.simulate(ep)
    cols = D.observe(ep, tr, r["sensor"], np.random.default_rng(0), r["root"])
    tx, ty, a = 0.37, -1.2, 2.1
    cols2, tr2 = _se2_cols(cols, tx, ty, a), _se2_traj(tr, tx, ty, a)
    for k in range(W.K_PUSHES):
        now = tr["push_start"][k] - D.DECISION_EPS
        s1, s2 = D.tensorize_sample(cols, now), D.tensorize_sample(cols2, now)
        np.testing.assert_allclose(s1["val"], s2["val"], atol=2e-5)
        np.testing.assert_allclose(s1["logvar"], s2["logvar"], atol=1e-3)
        np.testing.assert_array_equal(s1["mask"], s2["mask"])
        l1, l2 = D.make_labels_np(tr, ep, k, s1["est"]), D.make_labels_np(tr2, ep, k, s2["est"])
        np.testing.assert_allclose(l1["future"], l2["future"], atol=2e-5)
        np.testing.assert_array_equal(l1["free_slide_mask"], l2["free_slide_mask"])


def test_canonical_frame_equivariance_sim_level():
    """Re-simulating a globally SE(2)-transformed world gives (nearly) the same canonical labels."""
    acts = [W.PushAction(0.4, 0.1, 0.25, 0.3), W.PushAction(-2.0, -0.2, 0.15, 0.4),
            W.PushAction(1.5, 0.0, 0.35, 0.2), W.PushAction(3.0, 0.3, 0.1, 0.5)]
    ep1 = _ep(mu=0.45, mass=0.8, actions=acts)
    a, tx, ty = 1.1, 0.05, -0.07
    c, s = math.cos(a), math.sin(a)
    tf = lambda p: (c * p[0] - s * p[1] + tx, s * p[0] + c * p[1] + ty)
    ep2 = copy.deepcopy(ep1)
    ep2.pos, ep2.yaw, ep2.pusher_park = tf(ep1.pos), ep1.yaw + a, tf(ep1.pusher_park)
    t1, t2 = W.simulate(ep1), W.simulate(ep2)
    errs = []
    for k in range(W.K_PUSHES):
        i0 = int(round(t1["push_start"][k] / W.REC_DT))
        e1 = (*t1["obj_pose"][i0, 0, :2], t1["obj_pose"][i0, 0, 2])
        e2 = (*t2["obj_pose"][i0, 0, :2], t2["obj_pose"][i0, 0, 2])
        l1, l2 = D.make_labels_np(t1, ep1, k, e1), D.make_labels_np(t2, ep2, k, e2)
        errs.append(np.abs(l1["future"][:, :2] - l2["future"][:, :2]).max())
    print("sim-level SE(2) max canonical position deviation per push (m):", np.round(errs, 6))
    assert max(errs) < 5e-3


def test_duplicates_do_not_change_tensors():
    r = D.generate_root("train", 2, base_seed=9)
    cols = r["events"]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(cols["eid"]), 200, replace=False)
    dup = {k: np.concatenate([v, v[pick]]) for k, v in cols.items()}
    dup["recv"][-200:] += 0.2
    dup["avail"][-200:] += 0.2
    for k in range(W.K_PUSHES):
        now = r["push_start"][k] - D.DECISION_EPS
        a, b = D.tensorize_sample(cols, now), D.tensorize_sample(dup, now)
        for f in ("ent", "prop", "val", "logvar", "age", "latency", "src", "mask", "est"):
            np.testing.assert_array_equal(a[f], b[f], err_msg=f)
    # record path too
    evs = D.columnar_to_events(cols, r["root"], r["ent_types"])
    now = r["push_start"][3] - D.DECISION_EPS
    t1 = D.events_to_tensors([evs], [now], np.random.default_rng(1))
    t2 = D.events_to_tensors([evs + evs[:100]], [now], np.random.default_rng(1))
    for k, v in t1.__dict__.items():
        assert torch.equal(v, t2.__dict__[k]), k


def test_record_path_matches_cache_and_loader_contract(tmp_path):
    D.generate(dict(name="t", n_train=8, n_val=2, n_test=2, n_strata=2, procs=2, out_root=str(tmp_path),
                    strata=["w3"]))
    ds = D.load_split("train", "t", data_root=str(tmp_path))
    assert len(ds) == 8 * W.K_PUSHES
    L = D.make_loader(ds, batch_size=16, seed=0)
    ev, q, lab = next(iter(L))
    B, E = ev.mask.shape
    assert ev.value.shape == (B, E, VALUE_DIM) and lab.future.shape == (B, len(HORIZONS), STATE_DIM)
    assert q.action.shape == (B, 8) and q.horizons.shape == (B, len(HORIZONS))
    # target index points at an OBJECT entity that has a POSE2 near the canonical origin (latest estimate)
    for b in range(B):
        assert ev.ent_type[b, q.target[b]] == EntityType.OBJECT and ev.ent_mask[b, q.target[b]]
        m = ev.mask[b] & (ev.entity[b] == q.target[b]) & (ev.prop[b] == Prop.POSE2)
        newest = torch.argmin(torch.where(m, ev.age[b], torch.full_like(ev.age[b], 1e9)))
        assert ev.value[b, newest, :2].abs().max() < 1e-5
        assert (ev.age[b][ev.mask[b]] >= 0).all()
    # record path == cache for the same root (unpermuted)
    for i in (0, 5):
        evs, root, et = D.root_events("train", i // W.K_PUSHES, "t", data_root=str(tmp_path))
        now = float(ds.t["dt_obs"][i] * 0) + (W.T0 + W.PUSH_PERIOD * (i % W.K_PUSHES)) - D.DECISION_EPS
        t = D.events_to_tensors([evs], [now], np.random.default_rng(0), permute=False)
        m1, m2 = t.mask[0], ds.t["mask"][i]
        assert m1.sum() == m2.sum()
        np.testing.assert_allclose(np.sort(t.value[0][m1].numpy().ravel()), np.sort(ds.t["value"][i][m2].numpy().ravel()),
                                   atol=1e-6)
    # permutation: equality structure preserved, entity labels change across epochs
    ev2, q2, _ = next(iter(L))
    assert not torch.equal(ev.entity, ev2.entity) or not torch.equal(q.target, q2.target)


def _v2_slide(mu, speed=0.8, mass=1.0, pusher="sphere15"):
    acts = [W.PushAction(0.0, 0.0, speed, 0.3) for _ in range(W.K_PUSHES)]
    ep = W.EpisodeParams(world="w2", friction=mu, mass=mass, half=(0.03, 0.03, 0.02), pos=(0.0, 0.0), yaw=0.3,
                         pusher=pusher, actions=acts, pusher_park=(0.3, 0.3), retract=True, pusher_fmax=100.0)
    tr = W.simulate(ep)
    i_rel = int(round((tr["push_end"][0] + 0.02) / W.REC_DT))  # just after release
    i_h = int(round((tr["push_start"][0] + 2.0) / W.REC_DT))
    assert tr["f_push"][i_rel:i_h].max() == 0.0, "pusher must release at push end (v2 retract)"
    v = np.hypot(*tr["obj_vel"][i_rel, 0, :2])
    slide = np.hypot(*(tr["obj_pose"][i_h, 0, :2] - tr["obj_pose"][i_rel, 0, :2]))
    assert np.hypot(*tr["obj_vel"][i_h, 0, :2]) < 1e-3  # at rest by 2.0 s
    return slide, v


@pytest.mark.parametrize("mass", [0.3, 1.5])
def test_v2_free_slide_matches_coulomb(mass):
    res = {}
    for mu in (0.3, 0.7):
        s, v = _v2_slide(mu, mass=mass)
        pred = v**2 / (2 * mu * 9.81)
        res[mu] = (s, pred)
        print(f"v2 mass={mass} mu={mu}: release v={v:.3f} m/s slide={s*100:.2f} cm, v^2/(2 mu g)={pred*100:.2f} cm, "
              f"ratio={s/pred:.3f}")
        assert abs(s / pred - 1) < 0.2
    assert res[0.3][0] > 2 * res[0.7][0]
    assert res[0.3][0] > 0.05  # >= 5 cm at mu=0.3, 0.8 m/s


def test_v2_strata_config():
    st = D.STRATA_BY_VERSION["v2"]
    assert set(st) == set(D.STRATA_BY_VERSION["v1"])
    rng = np.random.default_rng(0)
    eps = [W.sample_episode(rng, st["train"]["world"]) for _ in range(200)]
    sp = np.array([a.speed for e in eps for a in e.actions])
    du = np.array([a.duration for e in eps for a in e.actions])
    assert sp.min() >= 0.05 and sp.max() <= 1.0 and np.median(sp) < 0.4
    assert du.min() >= 0.1 - 1e-9 and du.max() <= 0.5 + 1e-9
    assert all(e.retract and e.half[2] <= 0.9 * min(e.half[:2]) + 1e-9 for e in eps)


@pytest.mark.parametrize("L", [1, 2, 4])
def test_chain_target_is_last_box(L):
    r = D.generate_root(f"chain_L{L}", 0, base_seed=2, version="v2")
    ep = W.EpisodeParams(**{**r["params"], "actions": [W.PushAction(*a) for a in r["params"]["actions"]]})
    assert len(ep.others or []) == L - 1 and r["chain_L"] == L
    et = r["ent_types"]
    assert (et == EntityType.OBJECT).sum() == L and et[1] == EntityType.PUSHER
    # target is the box farthest along the chain axis; push applied to the first box
    u = np.array([math.cos(ep.chain_axis), math.sin(ep.chain_axis)])
    proj_t = np.dot(ep.pos, u)
    for o in ep.others or []:
        assert np.dot(o["pos"], u) < proj_t
    # the target moves on the first push although the pusher never touches it (L>1)
    tr = r["traj20"]
    i0 = int(round(r["push_start"][0] / 0.05))
    disp = [np.hypot(*(tr["obj_pose"][int(round(t / 0.05)) + 40, 0, :2] - tr["obj_pose"][int(round(t / 0.05)), 0, :2]))
            for t in r["push_start"]]
    assert max(disp) > 0.003, disp
    if L > 1:
        assert tr["f_other"][i0: i0 + 40].max() > 0.0  # contact propagated through the chain to the target


def test_agent_mode_is_latent_and_k0_uninformative():
    modes = set()
    for i in range(12):
        r = D.generate_root("agent", i, base_seed=3, version="v2")
        modes.add(r["agent_mode"])
        assert r["ent_types"][2] == EntityType.AGENT
        tr = r["traj20"]
        i0 = int(round(W.T0 / 0.05))
        assert np.abs(tr["agent_pose"][:i0] - tr["agent_pose"][0]).max() < 1e-9  # static before first push
        # mode never reaches the actor stream: no BEHAVIOR events, no privileged provenance
        assert not np.any(r["events"]["prop"] == Prop.BEHAVIOR)
        assert not np.any(r["events"]["prov"] == 3)
        if r["agent_mode"] in (1, 3):  # crossing / oscillating agents move after T0
            assert np.abs(tr["agent_pose"][i0 + 40] - tr["agent_pose"][0]).max() > 0.01
    assert len(modes) >= 3


def test_views_and_nested_subsets(tmp_path):
    D.generate(dict(name="s", version="v2", n_train=40, n_val=2, n_test=2, n_strata=4, procs=2, out_root=str(tmp_path),
                    strata=["agent", "chain_L2"]))
    sub = D.make_subsets("s", seed=1, data_root=str(tmp_path))
    r1, r10, r100 = (set(sub["subsets"][k]["roots"]) for k in ("1pct", "10pct", "100pct"))
    assert r1 < r10 < r100 and len(r100) == 40 and len(r1) == 1 and len(r10) == 4
    ds = D.load_split("train", "s", data_root=str(tmp_path), subset="10pct")
    assert set(ds.t["root"].tolist()) == r10 and len(ds) == 16
    k0 = D.load_split("agent_k0", "s", data_root=str(tmp_path))
    kl = D.load_split("agent_klate", "s", data_root=str(tmp_path))
    assert (k0.t["k"] == 0).all() and (kl.t["k"] >= 1).all() and len(k0) + len(kl) == 16
    assert "agent_mode" in k0.meta() and (k0.meta()["agent_mode"] >= 0).all()
    ch = D.load_split("chain_L2", "s", data_root=str(tmp_path))
    ev, q, lab = ch.batch(torch.arange(len(ch)))
    assert ev.ent_type.shape[1] == 4 and (ch.t["chain_L"] == 2).all()
    for b in range(len(ch)):
        assert ev.ent_type[b, q.target[b]] == EntityType.OBJECT
