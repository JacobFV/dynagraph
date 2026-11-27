"""Deliverable 1 gate test on the REAL Inkling-2-Mini checkpoint (packed experts).
(a) instrumented forward (hooks off) == native forward
(b) System II with the System I machinery installed and NON-trivial adapters (trained adapters if --ckpt, else random
    non-zero LoRA B / gates): logits bit-identical to the base model; greedy decode identical
(c) ungated LoRA ('always') changes the logits (magnitude reported)
(d) inside a System I call, the System II prefix positions keep their base core states (exact expert path)
(e) latency: System I call (bs 1) at 1/2/3 core passes, fast vs exact experts; System II option scoring
usage: python scripts/sys12/gate_test.py [--ckpt results/sys12/checkpoints/<run>.pt]"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, "src")
from physmem.models.inkling import instrumented_forward, mem_snapshot  # noqa: E402
from physmem.models.inkling_quant import QuantExperts, load_inkling_quant  # noqa: E402
from physmem.models.sys12 import Sys1Core, set_lora_mode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="")
ap.add_argument("--out", default="results/sys12/gate_test.json")
a = ap.parse_args()
model, tm, tok, rep = load_inkling_quant("ckpt/Inkling-2-Mini")
res = {"load": rep, "checks": {}}
C = res["checks"]
prompts = ["The robot picks up the part and places it in the tote because",
           "<|message_user|><|content_text|>Which skill should a robot use to move a box from a bin to a tote?<|end_message|><|message_model|>",
           "In 1905, Einstein published four papers that"]
enc = [torch.tensor(tok(p, add_special_tokens=False)["input_ids"], device="cuda")[None] for p in prompts]


def logits(ids):
    with torch.no_grad():
        return model(input_ids=ids).logits.float()


def greedy(ids, n=24):
    with torch.no_grad():
        return model.generate(input_ids=ids, max_new_tokens=n, do_sample=False)[0, ids.shape[1]:].tolist()


base = [logits(e) for e in enc]
base_gen = [greedy(e) for e in enc]
# (a)
with torch.no_grad():
    nat = tm(input_ids=enc[0]).last_hidden_state
    ins, _ = instrumented_forward(tm, tm.embed_tokens(enc[0]))
C["a_instrumented_equals_native"] = bool(torch.equal(nat, ins))

core = Sys1Core(tm, tok, d_mem=128, act_dim=11, n_readout=16, passes=1, lora_mode="gated")
src = "random non-zero adapters (LoRA B ~ N(0, 0.02), gates 0.5)"
if a.ckpt:
    c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    core.load_adapter_state(c["state"]["core"])
    src = f"trained adapters from {a.ckpt}"
else:
    with torch.no_grad():
        for m in core.lora:
            m.B.normal_(0, 0.02)
        core.read_gate.fill_(0.5); core.write_gate.fill_(0.5)
res["adapters"] = src
lora_norm = float(sum(float((m.B.float() @ m.A.float()).norm()) for m in core.lora))
res["lora_delta_norm_sum"] = lora_norm
# (b) System II = adapters off
set_lora_mode(tm, "off")
s2 = [logits(e) for e in enc]
C["b_system2_logits_bit_identical"] = all(torch.equal(x, y) for x, y in zip(base, s2))
C["b_system2_max_abs_logit_diff"] = max(float((x - y).abs().max()) for x, y in zip(base, s2))
C["b_system2_greedy_identical"] = all(g == greedy(e) for g, e in zip(base_gen, enc))
# (c) ungated
set_lora_mode(tm, "always")
ug = [logits(e) for e in enc]
set_lora_mode(tm, "off")
kl = []
for x, y in zip(base, ug):
    p = torch.log_softmax(x[0], -1); q = torch.log_softmax(y[0], -1)
    kl.append(float((p.exp() * (p - q)).sum(-1).mean()))
C["c_ungated_changes_logits"] = any(not torch.equal(x, y) for x, y in zip(base, ug))
C["c_ungated_max_abs_logit_diff"] = max(float((x - y).abs().max()) for x, y in zip(base, ug))
C["c_ungated_mean_KL_base_to_ungated"] = sum(kl) / len(kl)
# (d) System I prefix identity (exact expert path, 1 and 3 passes). Reference = the base model (LoRA off, no memory
#     hooks) on the SAME full input sequence (same GEMM shapes); also reported vs a prefix-only base run, where any
#     difference is GEMM-shape rounding (different number of routed rows), not information flow.
mem = torch.randn(1, 20, 128, device="cuda")
mm = torch.ones(1, 20, dtype=torch.bool, device="cuda")
act = torch.zeros(1, 11, device="cuda"); act[0, 3] = 1
core.fast_experts = False
for P in (1, 3):
    with torch.no_grad():
        x, s0 = core.build_inputs(1, "bin_clearing", act)
        set_lora_mode(tm, "off")
        ref_full, _ = instrumented_forward(tm, x, stop_after_core=True, num_passes=P)
        ref_pre, _ = instrumented_forward(tm, x[:, :s0], stop_after_core=True, num_passes=P)
        core(mem, mm, act, "bin_clearing", passes=P, keep_records=True)
        h = core.last_info["records"][-1]
    C[f"d_sys1_prefix_states_bit_identical_to_base_same_seq_p{P}"] = bool(torch.equal(h[:, :s0], ref_full[:, :s0]))
    C[f"d_sys1_prefix_max_abs_diff_same_seq_p{P}"] = float((h[:, :s0].float() - ref_full[:, :s0].float()).abs().max())
    C[f"d_sys1_readout_changed_by_adapters_p{P}"] = bool(not torch.equal(h[:, s0:], ref_full[:, s0:]))
    rel = float((ref_full[:, :s0].float() - ref_pre.float()).norm() / ref_pre.float().norm())
    C[f"d_base_full_vs_prefix_only_rel_diff_p{P}"] = rel
# (e) latency
lat = {}
for fast in (True, False):
    core.fast_experts = fast
    for P in (1, 2, 3):
        with torch.no_grad():
            core(mem, mm, act, "bin_clearing", passes=P)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(5):
                core(mem, mm, act, "bin_clearing", passes=P)
            torch.cuda.synchronize()
        lat[f"sys1_bs1_passes{P}_{'fast' if fast else 'exact'}_ms"] = round(1000 * (time.time() - t0) / 5, 1)
core.fast_experts = True
# System II option scoring: one forward over a 3-option menu prompt (logprob of each option continuation)
menu = enc[1]
with torch.no_grad():
    torch.cuda.synchronize(); t0 = time.time()
    model(input_ids=menu.expand(3, -1))
    torch.cuda.synchronize()
lat["sys2_menu_forward_bs3_ms"] = round(1000 * (time.time() - t0), 1)
res["latency"] = lat
res["mem_end"] = mem_snapshot()
res["seq_len_sys1"] = core.last_info.get("seq_len")
print(json.dumps(res, indent=1, default=str))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1, default=str)
