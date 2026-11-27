"""Milestone 1 (fit): load Inkling-2-Mini with packed experts, verify exactness of the on-the-fly dequant against the
transformers conversion op applied to the raw checkpoint tensors, report memory, greedy decode, forward timing.
usage: python scripts/sys12/fit_check.py [--cache-core]"""
import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, "src")
from physmem.models.inkling_quant import load_inkling_quant  # noqa: E402
from physmem.models.inkling import mem_snapshot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache-core", action="store_true")
ap.add_argument("--out", default="results/sys12/fit_check.json")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
CK = "ckpt/Inkling-2-Mini"
model, tm, tok, rep = load_inkling_quant(CK, cache_layers=range(6, 18) if a.cache_core else ())
res = {"load": rep}

# 1. exactness of packed dequant vs transformers op (layer 3 nvfp4, layer 21 mxfp8), several experts
from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave  # noqa: E402

f = safe_open(os.path.join(CK, "model-00001-of-00001.safetensors"), "pt", device="cuda")
op = Inkling2LoopDequantizeExperts()
exact = {}
for li in (3, 21):
    ex = tm.layers[li].mlp.experts
    for wname, packed, il in (("w13", ex.gate_up, True), ("w2", ex.down, False)):
        pre = f"model.llm.layers.{li}.mlp.experts.{wname}_weight"
        srcs = [pre, pre + "_scale", pre + "_amax", pre + "_outer_scale_denominator"]
        keys = set(f.keys())
        d_raw = {s: f.get_tensor(s) for s in srcs if s in keys}
        d = {k: v.to(torch.bfloat16) for k, v in d_raw.items()}      # what the transformers loader feeds its op
        out = op.convert(d, srcs, ["x"], model=None, full_layer_name=None)["x"]
        out_raw = op.convert(d_raw, srcs, ["x"], model=None, full_layer_name=None)["x"]
        if il:
            out_raw = Interleave(dim=1).convert({"x": out_raw}, ["x"], ["x"])["x"]
        if il:
            out = Interleave(dim=1).convert({"x": out}, ["x"], ["x"])["x"]
        sel = torch.tensor([0, 7, 64, 127], device="cuda")
        mine = packed.dequant(sel)
        mine_t = packed.dequant(sel, use_triton=False)
        exact[f"L{li}.{wname}"] = {"kind": packed.kind,
                                   "triton_vs_transformers_loader_bit_equal": bool(torch.equal(mine, out[sel])),
                                   "torch_vs_transformers_loader_bit_equal": bool(torch.equal(mine_t, out[sel])),
                                   "max_abs_diff_vs_raw_fp32_amax": float((mine.float() - out_raw[sel].float()).abs().max())}
        if li == 3 and wname == "w13":
            ref13 = out
        if li == 3 and wname == "w2":
            ref2 = out
        del out, out_raw, d, d_raw
# full experts forward: QuantExperts vs native loop with bf16-dequantized weights (layer 3)
ex = tm.layers[3].mlp.experts
hs = torch.randn(512, ex.hidden_dim, device="cuda", dtype=torch.bfloat16)
tki = torch.randint(0, 128, (512, 8), device="cuda")
tkw = torch.rand(512, 8, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    mine_out = ex(hs, tki, tkw)
    fin = torch.zeros_like(hs, dtype=torch.float32)
    em = torch.nn.functional.one_hot(tki, num_classes=129).permute(2, 1, 0)
    for e in torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist():
        if e == 128:
            continue
        pos, ti = torch.where(em[e])
        c = ex._apply_gate(torch.nn.functional.linear(hs[ti], ref13[e]))
        c = torch.nn.functional.linear(c, ref2[e]) * tkw[ti, pos, None]
        fin.index_add_(0, ti, c.to(torch.float32))
    nat = fin.to(hs.dtype)
exact["experts_forward_L3_bit_equal"] = bool(torch.equal(mine_out, nat))
del ref13, ref2
# dequant timing
sel = torch.arange(128, device="cuda")
for name, fn in (("triton", lambda: ex.gate_up.dequant(sel)), ("torch", lambda: ex.gate_up.dequant(sel[:16], use_triton=False))):
    fn(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    exact[f"dequant_{name}_s"] = round((time.time() - t0) / 3, 4)
torch.cuda.empty_cache()
res["dequant_exactness"] = exact
print(json.dumps(exact), flush=True)

# 2. greedy decode (native generate with cache) sanity
msgs = [{"role": "user", "content": "In one sentence, what does a robot gripper do?"}]
try:
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
except Exception as e:  # noqa: BLE001
    print("chat template failed", e)
    ids = tok("Question: what does a robot gripper do?\nAnswer:", return_tensors="pt")["input_ids"]
ids = ids.cuda()
t0 = time.time()
with torch.no_grad():
    gen = model.generate(input_ids=ids, max_new_tokens=40, do_sample=False)
dt = time.time() - t0
txt = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=False)
res["greedy"] = {"text": txt, "s": round(dt, 2), "tok_per_s": round((gen.shape[1] - ids.shape[1]) / dt, 2)}
print(res["greedy"], flush=True)
res["mem_after_generate"] = mem_snapshot()

# 3. forward timing (no cache) at a System-I-like length
x = torch.randint(0, 200000, (32, 50), device="cuda")
with torch.no_grad():
    for i in range(3):
        torch.cuda.synchronize(); t0 = time.time()
        _ = tm(input_ids=x).last_hidden_state
        torch.cuda.synchronize()
        res.setdefault("fwd_bs32_L50_s", []).append(round(time.time() - t0, 3))
res["mem_end"] = mem_snapshot()
print(json.dumps(res, default=str), flush=True)
json.dump(res, open(a.out, "w"), indent=1, default=str)
