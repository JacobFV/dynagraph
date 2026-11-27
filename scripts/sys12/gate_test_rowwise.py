"""Gate check (d), shape-invariant version. The MoE expert loop batches every position routed to an expert into one
GEMM, so the rows a System I call appends change the GEMM M dimension for prefix rows (and can flip near-tied top-k
routing), i.e. prefix states can differ from the base by rounding even though no information flows backwards.
Here the expert computation is done ROW BY ROW (M = 1 per token and expert; monkeypatched in this test process only),
which removes that shape dependence. Then:
  (1) base full sequence vs System I call (trained gated adapters, memory hooks, LoRA gated) on the SAME sequence:
      prefix core states must be bit-identical (the position gate and causal attention leave them untouched);
  (2) control: base full sequence vs base on the prefix only (attention shapes differ -> may differ by rounding).
usage: python scripts/sys12/gate_test_rowwise.py --ckpt <gated ckpt> [--out results/sys12/gate_test_rowwise.json]"""
import argparse
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from physmem.models.inkling import instrumented_forward  # noqa: E402
from physmem.models.inkling_quant import QuantExperts, load_inkling_quant  # noqa: E402
from physmem.models.sys12 import Sys1Core, set_lora_mode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="results/sys12/gate_test_rowwise.json")
a = ap.parse_args()


def rowwise(self, hidden_states, top_k_index, top_k_weights):
    out = torch.zeros_like(hidden_states, dtype=torch.float32)
    hit = sorted({int(e) for e in top_k_index.flatten().tolist() if int(e) != self.num_experts})
    ids = torch.tensor(hit, device=hidden_states.device)
    W13, W2 = self.gate_up.dequant(ids), self.down.dequant(ids)
    pos = {e: i for i, e in enumerate(hit)}
    for t in range(hidden_states.shape[0]):
        x = hidden_states[t:t + 1]
        for k in range(top_k_index.shape[1]):
            e = int(top_k_index[t, k])
            if e == self.num_experts:
                continue
            h = self._apply_gate(F.linear(x, W13[pos[e]]))
            h = F.linear(h, W2[pos[e]]) * top_k_weights[t, k]
            out[t] += h[0].float()
    return out.to(hidden_states.dtype)


model, tm, tok, rep = load_inkling_quant("ckpt/Inkling-2-Mini")
QuantExperts.forward = rowwise
core = Sys1Core(tm, tok, d_mem=128, act_dim=11, n_readout=16, passes=1, lora_mode="gated")
c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
core.load_adapter_state(c["state"]["core"])
core.fast_experts = False          # FAST would bypass the patched forward
torch.manual_seed(0)
mem = torch.randn(1, 21, 128, device="cuda")
mm = torch.ones(1, 21, dtype=torch.bool, device="cuda")
act = torch.zeros(1, 11, device="cuda"); act[0, 3] = 1
res = {"ckpt": a.ckpt, "method": "row-wise expert GEMMs (M=1), exact dequant, LoRA mode gated", "checks": {}}
for fam in ("bin_clearing", "pouch_transfer"):
    for P in (1, 3):
        with torch.no_grad():
            x, s0 = core.build_inputs(1, fam, act)
            set_lora_mode(tm, "off")
            ref_full, _ = instrumented_forward(tm, x, stop_after_core=True, num_passes=P)
            ref_pre, _ = instrumented_forward(tm, x[:, :s0], stop_after_core=True, num_passes=P)
            core(mem, mm, act, fam, passes=P, keep_records=True)
            h = core.last_info["records"][-1]
        k = f"{fam}_p{P}"
        res["checks"][f"{k}_sys1_prefix_bit_identical_to_base_same_seq"] = bool(torch.equal(h[:, :s0], ref_full[:, :s0]))
        res["checks"][f"{k}_sys1_prefix_max_abs_diff"] = float((h[:, :s0].float() - ref_full[:, :s0].float()).abs().max())
        res["checks"][f"{k}_sys1_positions_changed"] = bool(not torch.equal(h[:, s0:], ref_full[:, s0:]))
        res["checks"][f"{k}_base_full_vs_prefix_only_bit_identical"] = bool(torch.equal(ref_full[:, :s0], ref_pre))
        res["checks"][f"{k}_base_full_vs_prefix_only_max_abs_diff"] = float((ref_full[:, :s0].float() - ref_pre.float()).abs().max())
        print(k, {kk: v for kk, v in res["checks"].items() if kk.startswith(k)}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
