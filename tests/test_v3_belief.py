"""V3 section 0: structural guarantees of the looped belief store and the chain family."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from physmem.models.v3 import belief as B, chain as C
from physmem.models.v3.tracks import CameraTracker


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from physmem import v3_chain
    d = tmp_path_factory.mktemp("chain")
    v3_chain.gen(str(d), "train", 12)
    return B.ChainStore(str(d / "train.npz"), "cpu")


def test_tracker_backprojection_exact_without_noise():
    rng = np.random.default_rng(0)
    cfg = dict(jitter=0, frame_drop=0, px=0, outlier_p=0, kp_drop=0, occ_p=0, depth_rel=0, depth_abs=0, calib_f=0,
               calib_rot=0, calib_t=0)
    cam = CameraTracker(rng, cfg=cfg)
    P = np.array([[[0.05, 0.0, 0.04], [0.2, 0.0, 0.04]]])
    u, v, d = cam.observe(P, rng)
    assert np.allclose(cam.backproject(u, v, d), P, atol=1e-9)


def test_law_residuals_small_on_truth(store):
    b, lab = store.gather(store.rows())
    mass = torch.where(b.ent_type == 0, torch.full_like(lab.mass, C.PUSHER_M), lab.mass)
    r = B.law_residuals(lab.X, lab.V, lab.Fint, mass, lab.mu, lab.k, b)
    assert r["cone"] < 0.01 and r["slide"] < 0.01 and r["comp"] < 0.05 and r["kin"] < 0.5


def test_lbs_state_passes_only_through_named_store(store):
    """editing a named belief between passes changes everything downstream; re-running a pass from the written
    store reproduces the model's next store exactly (no hidden channel)."""
    torch.manual_seed(0)
    m = B.build("lbs").eval()
    b, _ = store.gather(store.rows()[:8])
    ids = torch.randn(8, C.N_ENT, B.ID_D)
    with torch.no_grad():
        outs = m(b, passes=3, ids=ids)
        ev = m.evidence(b, ids)
        mask = B.graph_mask(b)
        cl = [B.anchor_mask(b)]
        nxt = B.apply_clamps(m.step(outs[1], b, ids, ev, mask, 2), cl)
    assert torch.allclose(nxt.ent, outs[2].ent, atol=1e-5) and torch.allclose(nxt.edge, outs[2].edge, atol=1e-5)


def test_clamps_hold_at_every_pass(store):
    m = B.build("lbs").eval()
    b, _ = store.gather(store.rows()[:4])
    mm = torch.zeros(4, C.N_ENT, B.NP + 2 * B.NT, dtype=torch.bool); mm[:, 1, 1] = True
    val = torch.full((4, C.N_ENT, B.NP + 2 * B.NT), 1.234)
    with torch.no_grad():
        outs = m(b, passes=4, clamps=((mm, val),))
    for s in outs:
        assert torch.allclose(s.pm()[:, 1, 1], torch.full((4,), 1.234))


def test_blocking_all_edges_isolates_entities(store):
    """with every relation blocked, entity 1's beliefs cannot depend on other entities' evidence."""
    m = B.build("lbs").eval()
    rows = store.rows(ks=(3,))[:6]
    b, _ = store.gather(rows)
    ids = torch.randn(len(rows), C.N_ENT, B.ID_D)
    blk = torch.ones(len(rows), C.N_EDGE, dtype=torch.bool)
    b2 = B.Batch(**{**b.__dict__, "ev": b.ev.clone()})
    other = b.ev_ent != 1
    b2.ev[other] += 0.05
    with torch.no_grad():
        s1 = m(b, passes=3, ids=ids, block_edges=blk)[-1]
        s2 = m(b2, passes=3, ids=ids, block_edges=blk)[-1]
        u1 = m(b, passes=3, ids=ids)[-1]
        u2 = m(b2, passes=3, ids=ids)[-1]
    assert torch.allclose(s1.ent[:, 1], s2.ent[:, 1], atol=1e-5)
    assert not torch.allclose(u1.ent[:, 1], u2.ent[:, 1], atol=1e-5)


def test_v1_has_no_writeback_typed_estimates_fixed_before_loop(store):
    m = B.build("v1").eval()
    b, _ = store.gather(store.rows()[:4])
    ids = torch.randn(4, C.N_ENT, B.ID_D)
    with torch.no_grad():
        p2 = m(b, passes=2, ids=ids)[-1].pm()
        p6 = m(b, passes=6, ids=ids)[-1].pm()
    assert torch.allclose(p2, p6)           # loop depth cannot change the typed estimates (the audited v1 gap)


def test_counterfactual_replay_close():
    p, tr, obs, a, lab = C.generate_episode(5, L=2)
    k = 2
    q = p.copy(actions=[p.actions[k]], pusher_gap=[p.pusher_gap[k]])
    tr2 = C.simulate(q, windows=1, init_state=(tr["start_x"][k], tr["start_v"][k]))
    X2, _, _ = C.grid_truth(tr2, 0)
    assert np.abs(X2[:, 1:] - lab["X"][k, 1:p.L + 1].T).max() < 0.01


def test_contract_is_superset_of_v2_latents():
    from physmem.models.v2 import spec
    from physmem.models.v3 import contract as K
    assert K.V2_LATENTS == spec.LATENT_NAMES
    for (n, tr, loc, sc, unit), p in zip(spec.LATENTS, K.PROPERTIES):
        assert (n, tr, unit) == (p[0], p[2], p[5]) and abs(loc - p[3]) < 1e-12 and abs(sc - p[4]) < 1e-12
    for n in ("spring_k", "k_over_m", "c_over_m", "mass_ratio", "restitution", "length_scale"):
        assert n in K.PROP_NAMES
