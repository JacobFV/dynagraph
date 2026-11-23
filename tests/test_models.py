"""tests for compact models, analytic baselines, losses (agent MODEL)."""
import copy

import numpy as np
import pytest
import torch

from physmem.schema import (
    ACTION_DIM,
    HORIZONS,
    N_PROPS,
    STATE_DIM,
    VALUE_DIM,
    EventTensors,
    LabelTensors,
    ModelOutput,
    Prop,
    QueryTensors,
)
from physmem.models.compact import build_model, count_params, VARIANTS


def fake_batch(B=4, E=40, N=3, seed=0, device="cpu"):
    """synthetic random tensors that obey the schema shapes (not physically meaningful)."""
    g = torch.Generator().manual_seed(seed)
    H = len(HORIZONS)
    ent = torch.randint(0, N, (B, E), generator=g)
    prop = torch.tensor([int(p) for p in (Prop.POSE2, Prop.VEL2, Prop.CONTACT, Prop.SIZE, Prop.COMMAND)])[
        torch.randint(0, 5, (B, E), generator=g)
    ]
    related = torch.where(prop == int(Prop.CONTACT), (ent + 1) % N, torch.full_like(ent, -1))
    value = torch.randn(B, E, VALUE_DIM, generator=g) * 0.1
    logvar = torch.full((B, E, VALUE_DIM), -6.0) + torch.randn(B, E, VALUE_DIM, generator=g) * 0.1
    age = torch.rand(B, E, generator=g) * 3
    lat = torch.rand(B, E, generator=g) * 0.05
    src = torch.randint(0, 3, (B, E), generator=g)
    prov = torch.zeros(B, E, dtype=torch.long)
    mask = torch.rand(B, E, generator=g) > 0.1
    ent_type = torch.zeros(B, N, dtype=torch.long)
    ent_type[:, 1] = 1
    ent_mask = torch.ones(B, N, dtype=torch.bool)
    ev = EventTensors(ent, related, prop, value, logvar, age, lat, src, prov, mask, ent_type, ent_mask, torch.full((B,), 0.05))
    q = QueryTensors(
        target=torch.zeros(B, dtype=torch.long),
        action=torch.randn(B, ACTION_DIM, generator=g),
        horizons=torch.tensor(HORIZONS, dtype=torch.float32)[None].expand(B, -1).clone(),
    )
    lab = LabelTensors(
        future=torch.randn(B, H, STATE_DIM, generator=g) * 0.05,
        future_mask=torch.ones(B, H, dtype=torch.bool),
        friction=torch.rand(B, generator=g) * 0.4 + 0.3,
        mass=torch.rand(B, generator=g) * 1.8 + 0.2,
        contact_mode=torch.randint(0, 4, (B, H), generator=g),
        free_slide_mask=torch.rand(B, H, generator=g) > 0.5,
        law=torch.zeros(B, dtype=torch.long),
    )
    return ev.to(device), q.to(device), lab.to(device)


def _permute(ev, q, perm_ent, perm_src, perm_ev):
    """relabel entity/source local indices and reorder events. perm_ent[old] = new."""
    inv = torch.argsort(perm_ent)
    ent = perm_ent[ev.entity]
    rel = torch.where(ev.related >= 0, perm_ent[ev.related.clamp(min=0)], ev.related)
    src = perm_src[ev.source]
    f = lambda x: x[:, perm_ev]
    ev2 = EventTensors(
        f(ent), f(rel), f(ev.prop), f(ev.value), f(ev.logvar), f(ev.age), f(ev.latency), f(src), f(ev.provenance),
        f(ev.mask), ev.ent_type[:, inv], ev.ent_mask[:, inv], ev.dt_obs,
    )
    q2 = QueryTensors(perm_ent[q.target], q.action, q.horizons)
    return ev2, q2


@pytest.mark.parametrize("variant", ["b00", "b10", "b01", "b11", "b10u", "b11p"])
def test_shapes(variant):
    torch.manual_seed(0)
    m = build_model(variant).eval()
    ev, q, _ = fake_batch()
    for k in (1, 2, 4, 8):
        out = m(ev, q, passes=k)
        assert isinstance(out, ModelOutput)
        assert out.mean.shape == (4, len(HORIZONS), STATE_DIM)
        assert out.logvar.shape == out.mean.shape
        assert out.friction_mean.shape == (4,) and out.mass_mean.shape == (4,)
        assert out.contact_logits.shape == (4, len(HORIZONS), 4)
        assert torch.isfinite(out.mean).all() and torch.isfinite(out.logvar).all()


def test_split_api_matches_forward():
    torch.manual_seed(0)
    m = build_model("b11").eval()
    ev, q, _ = fake_batch()
    out = m(ev, q, passes=2)
    t, mk, s = m.encode(ev, q)
    t = m.refine(t, mk, 2)
    out2 = m.readout(t, mk, q, s)
    assert torch.allclose(out.mean, out2.mean)


def test_param_counts_matched():
    counts = {v: count_params(build_model(v)) for v in ("b00", "b10")}
    print(counts)
    assert abs(counts["b00"] - counts["b10"]) / counts["b10"] < 0.10
    assert 1e6 <= counts["b10"] <= 5e6


@pytest.fixture
def f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.mark.parametrize("variant", ["b10", "b00", "b11p"])
def test_permutation_invariance(variant, f64):
    torch.manual_seed(0)
    m = build_model(variant).eval().double()
    ev, q, _ = fake_batch(B=3, E=30, N=4)
    ev = EventTensors(**{k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in ev.__dict__.items()})
    q = QueryTensors(q.target, q.action.double(), q.horizons.double())
    g = torch.Generator().manual_seed(5)
    perm_ent = torch.randperm(4, generator=g)
    perm_src = torch.randperm(3, generator=g)
    perm_ev = torch.randperm(30, generator=g)
    ev2, q2 = _permute(ev, q, perm_ent, perm_src, perm_ev)
    with torch.no_grad():
        a = m(ev, q, passes=4)
        b = m(ev2, q2, passes=4)
    assert torch.allclose(a.mean, b.mean, atol=1e-9), (a.mean - b.mean).abs().max()
    assert torch.allclose(a.friction_mean, b.friction_mean, atol=1e-9)
    if variant in ("b10", "b11p"):
        # token-level equivariance: entity tokens permute with the entity relabelling
        t1, m1, s1 = m.encode(ev, q)
        t2, m2, s2 = m.encode(ev2, q2)
        S, N = s1["n_slots"], s1["n_entities"]
        e1 = t1[:, S : S + N]
        e2 = t2[:, S : S + N]
        assert torch.allclose(e1, e2[:, perm_ent], atol=1e-9)


def test_duplicates_do_not_change_structured_output(f64):
    torch.manual_seed(0)
    m = build_model("b10").eval().double()
    ev, q, _ = fake_batch(B=2, E=20, N=3)
    ev = EventTensors(**{k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in ev.__dict__.items()})
    q = QueryTensors(q.target, q.action.double(), q.horizons.double())
    cat = lambda x: torch.cat([x, x[:, :10]], 1)
    ev2 = EventTensors(
        cat(ev.entity), cat(ev.related), cat(ev.prop), cat(ev.value), cat(ev.logvar), cat(ev.age), cat(ev.latency),
        cat(ev.source), cat(ev.provenance), cat(ev.mask), ev.ent_type, ev.ent_mask, ev.dt_obs,
    )
    with torch.no_grad():
        a = m(ev, q, 2)
        b = m(ev2, q, 2)
    assert torch.allclose(a.mean, b.mean, atol=1e-9)


def test_frozen_modules():
    m = build_model("b11")
    m.freeze(encoder=True, head=True, core=False)
    ev, q, lab = fake_batch()
    before_enc = copy.deepcopy(m.encoder.state_dict())
    before_core = copy.deepcopy(m.core.state_dict())
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    out = m(ev, q, 2)
    (out.mean.square().mean() + out.logvar.mean()).backward()
    opt.step()
    for k, v in m.encoder.state_dict().items():
        assert torch.equal(v, before_enc[k])
    assert any(not torch.equal(v, before_core[k]) for k, v in m.core.state_dict().items())
    assert all(p.grad is None for p in m.encoder.parameters())


def test_intervention_changes_friction_readout():
    torch.manual_seed(0)
    m = build_model("b11").eval()
    ev, q, _ = fake_batch()
    with torch.no_grad():
        t, mk, s = m.encode(ev, q)
        base = m.readout(m.refine(t, mk, 2), mk, q, s)
        t2, s2 = m.intervene(t, mk, s, Prop.FRICTION, torch.full((4,), 0.9), torch.full((4,), -6.0))
        out = m.readout(m.refine(t2, mk, 2), mk, q, s2)
    assert torch.allclose(out.friction_mean, torch.full((4,), 0.9))
    assert not torch.allclose(out.mean, base.mean)
    # original slots untouched
    assert not torch.allclose(s["typed"][Prop.FRICTION]["mean"], torch.full((4,), 0.9))
    with pytest.raises(ValueError):
        g = build_model("b00").eval()
        tg, mg, sg = g.encode(ev, q)
        g.intervene(tg, mg, sg, Prop.FRICTION, 0.5, -3.0)


def test_edge_lesion_changes_structured_tokens():
    m = build_model("b10").eval()
    ev, q, _ = fake_batch(N=3)
    les = torch.ones(4, 3, 3, dtype=torch.bool)
    with torch.no_grad():
        a = m(ev, q, 2)
        t, mk, s = m.encode(ev, q, edge_lesion=les)
        b = m.readout(m.refine(t, mk, 2), mk, q, s)
    assert not torch.allclose(a.mean, b.mean)


def test_deterministic_eval():
    m = build_model("b11").eval()
    ev, q, _ = fake_batch()
    with torch.no_grad():
        a = m(ev, q, 4)
        b = m(ev, q, 4)
    assert torch.equal(a.mean, b.mean) and torch.equal(a.logvar, b.logvar)


def test_losses_and_backward():
    from physmem.models.losses import LossConfig, compute_loss, fit_loss_scales

    m = build_model("b11")
    ev, q, lab = fake_batch()
    scales = fit_loss_scales([(ev, q, lab)])
    out = m(ev, q, 2)
    loss, parts = compute_loss(out, lab, q, LossConfig(physics=True), scales, y_std=m.head.y_std, y_mean=m.head.y_mean)
    loss.backward()
    assert torch.isfinite(loss)
    for k in ("nll", "coulomb", "kinematic", "friction_nll", "mass_nll", "contact_ce"):
        assert k in parts


def test_analytic_shapes():
    from physmem.models.analytic import ConstantVelocity, CoulombAnalytic, CoulombOracle

    ev, q, lab = fake_batch()
    for mdl in (ConstantVelocity(), CoulombAnalytic()):
        out = mdl(ev, q)
        assert out.mean.shape == (4, len(HORIZONS), STATE_DIM)
        assert torch.isfinite(out.mean).all() and torch.isfinite(out.logvar).all()
    out = CoulombOracle()(ev, q, labels=lab)
    assert torch.allclose(out.friction_mean, lab.friction.float())


@pytest.mark.parametrize("variant", ["b00", "b11"])
def test_variable_entity_count(variant):
    """chain strata have up to 9 entities; models must accept any N (trained at N=4)."""
    m = build_model(variant).eval()
    for N in (2, 4, 9):
        ev, q, _ = fake_batch(B=2, E=50, N=N)
        with torch.no_grad():
            out = m(ev, q, passes=2)
        assert torch.isfinite(out.mean).all()


def test_b11p_push_evidence_and_intervention():
    m = build_model("b11p").eval()
    ev, q, _ = fake_batch(B=4, E=60, N=3)
    with torch.no_grad():
        t, mk, s = m.encode(ev, q)
        assert "push_evidence" in s and s["push_evidence"]["fr_mean"].shape == (4, 4)
        t2, s2 = m.intervene(t, mk, s, Prop.FRICTION, torch.full((4,), 0.9), torch.full((4,), -6.0))
        out = m.readout(m.refine(t2, mk, 2), mk, q, s2)
    assert torch.allclose(out.friction_mean, torch.full((4,), 0.9))
    print("b11p params", count_params(m), "b11", count_params(build_model("b11")))
