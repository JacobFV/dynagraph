"""System I / System II gate tests on a tiny randomly initialised Inkling2Loop text model (fast; GPU).
The same checks run on the real checkpoint in scripts/sys12/gate_test.py."""
import pytest
import torch

pytest.importorskip("transformers")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from physmem.models.inkling import instrumented_forward, tiny_text_model  # noqa: E402
from physmem.models.sys12 import (Sys1Core, format_handoff, gated_lora_modules, parse_handoff,  # noqa: E402
                                  set_lora_mode)


class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(ord(c) * 7 + i) % 480 for i, c in enumerate(text[:40])]}


def _core(tm, mode="gated", passes=1):
    core = Sys1Core(tm, FakeTok(), d_mem=32, n_readout=4, passes=passes, lora_mode=mode, checkpoint_layers=False)
    with torch.no_grad():                         # make every adapter non-trivial
        for m in core.lora:
            m.B.normal_(0, 0.05)
        core.read_gate.fill_(0.7)
        core.write_gate.fill_(0.7)
    return core


def test_system2_is_base_bit_exact():
    tm = tiny_text_model()
    ids = torch.randint(0, 480, (2, 23), device="cuda")
    with torch.no_grad():
        base = tm(input_ids=ids).last_hidden_state.clone()
        core = _core(tm)
        set_lora_mode(tm, "off")
        after = tm(input_ids=ids).last_hidden_state
        assert torch.equal(base, after), "System II (adapters off) must equal the base model bit for bit"
        set_lora_mode(tm, "always")
        ungated = tm(input_ids=ids).last_hidden_state
        set_lora_mode(tm, "off")
    assert not torch.equal(base, ungated), "ungated LoRA must change the base computation"
    assert len(gated_lora_modules(tm)) == 4 * len(core.core_layer_ids)


def test_instrumented_equals_native_with_adapters_present():
    tm = tiny_text_model()
    _core(tm)
    ids = torch.randint(0, 480, (2, 17), device="cuda")
    with torch.no_grad():
        nat = tm(input_ids=ids).last_hidden_state
        x = tm.embed_tokens(ids)
        ins, _ = instrumented_forward(tm, x)
    assert torch.equal(nat, ins)


def test_system1_prefix_states_are_base_and_readout_uses_memory():
    tm = tiny_text_model()
    core = _core(tm, passes=2)
    B = 2
    mem = torch.randn(B, 5, 32, device="cuda")
    mm = torch.ones(B, 5, dtype=torch.bool, device="cuda")
    act = torch.zeros(B, 11, device="cuda"); act[:, 3] = 1
    with torch.no_grad():
        x, s0 = core.build_inputs(B, "bin_clearing", act)
        ref, _ = instrumented_forward(tm, x[:, :s0], stop_after_core=True, num_passes=2)   # base on the prefix only
        ctx1 = core(mem, mm, act, "bin_clearing", keep_records=True)
        h_core = core.last_info["records"][-1]
        assert torch.equal(h_core[:, :s0], ref), "System I must not change System II prefix states (position gate)"
        ctx2 = core(mem * 3.0, mm, act, "bin_clearing")
        assert not torch.allclose(ctx1, ctx2), "System I output must depend on the memory"


def test_handoff_roundtrip():
    t = "Plan: clear the bin. " + format_handoff("pick", "part03") + " then place."
    assert parse_handoff(t) == ("pick", "part03")
    assert parse_handoff("<sys1> skill=fly target=x") is None
    assert parse_handoff("no call here") is None


def test_packed_nvfp4_matches_reference_dequant():
    from physmem.models.inkling_quant import PackedExpertWeight
    E, R, C = 3, 8, 64
    codes = torch.randint(0, 256, (E, R, C // 2), device="cuda").to(torch.bfloat16)
    scale = (torch.rand(E, R, C // 16, device="cuda") * 2).to(torch.float8_e4m3fn).to(torch.bfloat16)
    amax = torch.rand(E, 1, 1, device="cuda").to(torch.bfloat16)
    den = torch.full((E, 1, 1), 2688.0, device="cuda").to(torch.bfloat16)
    pw = PackedExpertWeight(codes, scale, amax, den, interleave=False)
    sel = torch.tensor([2, 0], device="cuda")
    a = pw.dequant(sel, use_triton=True)
    b = pw.dequant(sel, use_triton=False)
    assert torch.equal(a, b)
