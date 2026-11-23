"""Core-integration contracts for the Inkling adapter.

Fast tests use a tiny randomly-initialised Inkling2Loop of the same architecture
family. The real-checkpoint gate is scripts/inkling_gate.py (results in
results/inkling_gate.json); a light real-checkpoint test runs when
PHYSMEM_REAL_INKLING=1.
"""
import os

import pytest
import torch

from physmem.models.inkling import (CoreHooks, InklingCore, frozen_checksum, instrumented_forward, lora_modules,
                                    set_lora_enabled, tiny_text_model)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


@cuda
def test_instrumented_matches_native_full_forward():
    tm = tiny_text_model()
    ids = torch.randint(0, 500, (2, 24), device="cuda")
    with torch.no_grad():
        emb = tm.embed_tokens(ids)
        native = tm(inputs_embeds=emb).last_hidden_state
        mine, _ = instrumented_forward(tm, emb)
    assert torch.equal(native, mine), (native.float() - mine.float()).abs().max()


@cuda
def test_zero_gated_adapters_and_lora_preserve_native_core():
    tm = tiny_text_model()
    ids = torch.randint(0, 500, (1, 10), device="cuda")
    core = InklingCore(tm, d_mem=32, query_dim=8, prefix_ids=ids[0], n_readout=4)
    B = 3
    q = torch.randn(B, 8, device="cuda")
    with torch.no_grad():
        x = core.build_inputs(B, q)
        set_lora_enabled(tm, False)
        core.adapters_enabled = False
        ref, _ = instrumented_forward(tm, x, stop_after_core=True)
        set_lora_enabled(tm, True)
        core.adapters_enabled = True
        slots = torch.randn(B, 5, 32, device="cuda")
        recs = []
        h, _ = instrumented_forward(tm, x, CoreHooks(lambda p, hh: hh + 0 * hh, None), stop_after_core=True)
        assert torch.equal(h, ref)
        # with real (zero-gated) adapters: same core-exit state
        out, m = core(slots, torch.ones(B, 5, dtype=torch.bool, device="cuda"), q, keep_records=True)
        assert torch.equal(core.last_info["records"][-1], ref)


@cuda
def test_gradient_reaches_first_pass_adapter_through_core_and_base_frozen():
    tm = tiny_text_model()
    ids = torch.randint(0, 500, (1, 10), device="cuda")
    core = InklingCore(tm, d_mem=32, query_dim=8, prefix_ids=ids[0], n_readout=4)
    with torch.no_grad():  # open the gates so the read path is active
        core.read_gate.fill_(0.5)
        core.write_gate.fill_(0.5)
        for m in lora_modules(tm):
            m.B.normal_(0, 1e-3)
    before = frozen_checksum(tm)
    opt = torch.optim.AdamW(core.trainable_parameters(), lr=1e-3)
    B = 2
    slots = torch.randn(B, 5, 32, device="cuda", requires_grad=True)
    out, m = core(slots, torch.ones(B, 5, dtype=torch.bool, device="cuda"), torch.randn(B, 8, device="cuda"))
    loss = out[:, -4:].pow(2).mean()
    loss.backward()
    # pass-0 read adapter gradient must pass through passes 1..2 of the core;
    # check via the gate gradient of each pass and the read projection
    assert core.read_gate.grad is not None and torch.isfinite(core.read_gate.grad).all()
    assert (core.read_gate.grad.abs() > 0).all(), core.read_gate.grad
    assert core.read.q.weight.grad.abs().sum() > 0
    loras = lora_modules(tm)
    assert all(l.A.grad is not None and l.A.grad.abs().sum() > 0 for l in loras)
    assert slots.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in tm.named_parameters() if not (n.endswith(".A") or n.endswith(".B")))
    a0 = loras[0].A.detach().clone()
    opt.step()
    assert not torch.equal(a0, loras[0].A)
    assert frozen_checksum(tm) == before


@cuda
def test_pass0_read_gradient_is_mediated_by_later_passes():
    """Isolate pass 0: only read gate 0 open; readout taken at core exit (after
    pass 2). A non-zero gradient proves the dependency path goes through the
    recurrent core's later passes (messages), not a shortcut."""
    tm = tiny_text_model()
    ids = torch.randint(0, 500, (1, 10), device="cuda")
    core = InklingCore(tm, d_mem=32, query_dim=8, prefix_ids=ids[0], n_readout=4, core_lora=False)
    with torch.no_grad():
        core.read_gate.zero_(); core.read_gate[0] = 0.5
    slots = torch.randn(2, 5, 32, device="cuda", requires_grad=True)
    out, _ = core(slots, torch.ones(2, 5, dtype=torch.bool, device="cuda"), torch.randn(2, 8, device="cuda"))
    out[:, -4:].pow(2).mean().backward()
    assert slots.grad.abs().sum() > 0
    assert core.read.k.weight.grad.abs().sum() > 0


@pytest.mark.skipif(os.environ.get("PHYSMEM_REAL_INKLING") != "1", reason="real checkpoint test is opt-in (heavy)")
def test_real_checkpoint_equivalence():
    from physmem.models.inkling import load_inkling_text

    tm, tok, _, rep = load_inkling_text()
    ids = tok("The box slides on the table.", return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        emb = tm.embed_tokens(ids)
        native = tm(inputs_embeds=emb).last_hidden_state
        mine, _ = instrumented_forward(tm, emb)
    assert torch.equal(native, mine)


@cuda
def test_inkling_physmem_pipeline_on_tiny_core():
    import sys
    sys.path.insert(0, "tests")
    from test_models import fake_batch
    from physmem.models.inkling import build_inkling_model
    from physmem.models.losses import LossConfig, compute_loss

    tm = tiny_text_model()
    prefix = torch.randint(0, 500, (12,), device="cuda")
    ev, q, lab = fake_batch(B=4, E=64, N=4, seed=0)
    ev, q, lab = ev.to("cuda"), q.to("cuda"), lab.to("cuda")
    for v in ("i00", "i11"):
        m = build_inkling_model(v, tm, prefix, d=128, n_readout=4)
        with torch.no_grad():
            m.core.read_gate.fill_(0.3); m.core.write_gate.fill_(0.3)
        out = m(ev, q)
        from physmem.models.losses import fit_loss_scales
        scales = fit_loss_scales([(ev, q, lab)])
        loss, parts = compute_loss(out, lab, q, LossConfig(physics=m.physics), scales, m.head.y_mean, m.head.y_std)
        loss.backward()
        assert torch.isfinite(loss)
        assert m.core.read.k.weight.grad.abs().sum() > 0
