"""Inkling-2-Mini with QUANTIZED experts kept resident (owner: INKLING agent).

transformers 5.22's Inkling2Loop loader dequantizes the native NVFP4 (layers 0-19) / MXFP8 (layers 20-23) expert
weights to bf16 at load (52.8 GB resident). Here the dequant conversion op is intercepted: the PACKED tensors
(uint8 E2M1 codes / float8 values, block scales, per-expert amax / denominator) are kept on the GPU (~13 GB for all
24 layers) and each layer's experts module is replaced by `QuantExperts`, which dequantizes only the experts a batch
actually routes to, on the fly, with EXACTLY the arithmetic of `Inkling2LoopDequantizeExperts` (float32 values x
block scale x expert factor -> cast to the param dtype) and the native expert loop (same expert order, fp32
index_add accumulation). Outputs are therefore bit-identical to the bf16-dequantized model (tested in
scripts/sys12/fit_check.py against the transformers op on the raw checkpoint tensors).

Optionally `cache_layers` keeps chosen layers' experts dequantized in bf16 (speed / memory trade-off).
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_STASH: dict = {}


def _layer_of(name: str) -> int:
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")   # bundled ptxas lacks sm_121a (GB10)
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _nvfp4_kernel(codes, scale, factor, eids, out, R, CB, SC, BLOCK: tl.constexpr):
        """codes (E,R,CB) uint8 (low nibble = even column); scale (E,R,SC) bf16 (one per 16 values); factor (E,) fp32
        -> out (k,R,2CB) bf16. value = e2m1(code) * scale * factor in fp32, then round to bf16 (reference order)."""
        kk = tl.program_id(0)
        e = tl.load(eids + kk).to(tl.int64)
        offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        n = R * CB
        m = offs < n
        r = offs // CB
        j = offs % CB
        c = tl.load(codes + e * n + offs, mask=m, other=0).to(tl.int32)
        s = tl.load(scale + e * R * SC + r * SC + j // 8, mask=m, other=0).to(tl.float32)
        f = tl.load(factor + e)
        lo = c & 15
        hi = c >> 4
        mlo, mhi = lo & 7, hi & 7
        elo, ehi = (mlo >> 1).to(tl.float32), (mhi >> 1).to(tl.float32)
        flo, fhi = (mlo & 1).to(tl.float32), (mhi & 1).to(tl.float32)
        vlo = tl.where(elo == 0, 0.5 * flo, tl.exp2(elo - 1.0) * (1.0 + 0.5 * flo))
        vhi = tl.where(ehi == 0, 0.5 * fhi, tl.exp2(ehi - 1.0) * (1.0 + 0.5 * fhi))
        vlo = tl.where((lo >> 3) == 1, -vlo, vlo)
        vhi = tl.where((hi >> 3) == 1, -vhi, vhi)
        vlo = (vlo * s) * f
        vhi = (vhi * s) * f
        o = out + kk.to(tl.int64) * n * 2 + r * (2 * CB) + 2 * j
        tl.store(o, vlo.to(tl.bfloat16), mask=m)
        tl.store(o + 1, vhi.to(tl.bfloat16), mask=m)

    @triton.jit
    def _mxfp8_kernel(vals, scale, eids, out, R, C, SC, BLOCK: tl.constexpr):
        """vals (E,R,C) float8_e4m3fn; scale (E,R,SC) fp32 = 2**(s-127) per 32 values -> out (k,R,C) bf16."""
        kk = tl.program_id(0)
        e = tl.load(eids + kk).to(tl.int64)
        offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        n = R * C
        m = offs < n
        r = offs // C
        j = offs % C
        v = tl.load(vals + e * n + offs, mask=m).to(tl.float32)
        s = tl.load(scale + e * R * SC + r * SC + j // 32, mask=m, other=0)
        tl.store(out + kk.to(tl.int64) * n + offs, (v * s).to(tl.bfloat16), mask=m)

    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

USE_TRITON = HAVE_TRITON and os.environ.get("SYS12_NO_TRITON", "0") != "1"


class PackedExpertWeight:
    """one expert weight tensor in its checkpoint format, compact on the GPU.
    nvfp4: codes uint8 (E,R,C/2), block scale bf16 (E,R,C/16) (exact copy of the float8 scale), factor (E,) fp32.
    mxfp8: values float8_e4m3fn (E,R,C), block scale fp32 (E,R,C/32) = 2**(s-127) precomputed exactly.
    The transformers loader casts every source tensor to bf16 before its dequant op (lossless for the codes /
    float8 / exponents; LOSSY for the fp32 amax) - we receive the same cast tensors, so the result equals what
    transformers itself loads."""

    def __init__(self, weight, scale, amax, denom, interleave: bool, dtype=torch.bfloat16):
        self.dtype = dtype
        self.kind = "nvfp4" if amax is not None else "mxfp8"
        E, R = weight.shape[:2]
        self.E, self.R = E, R
        self.C = weight.shape[2] * 2 if self.kind == "nvfp4" else weight.shape[2]
        if interleave:
            # transformers applies Interleave(dim=1) AFTER dequant (rows 0,2,4.. then 1,3,5..). Blocks run along
            # columns, so permuting the packed rows first is exactly equivalent.
            perm = torch.cat([torch.arange(0, R, 2), torch.arange(1, R, 2)]).to(weight.device)
            weight, scale = weight.index_select(1, perm), scale.index_select(1, perm)
        if self.kind == "nvfp4":
            self.weight = weight.to(torch.uint8).contiguous()
            self.scale = scale.to(torch.bfloat16).contiguous()
            self.factor = (amax.float() / denom.float()).reshape(E).contiguous()
            e2m1 = torch.tensor(E2M1_VALUES, device=weight.device)
            byte = torch.arange(256, device=weight.device)
            self.lut = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)  # (256, 2) fp32 (torch fallback)
        else:
            self.weight = weight.to(torch.float8_e4m3fn).contiguous()
            self.scale = torch.exp2(scale.float() - 127).contiguous()
            self.factor = None

    def nbytes(self):
        n = self.weight.numel() * self.weight.element_size() + self.scale.numel() * self.scale.element_size()
        return n + (self.factor.numel() * 4 if self.factor is not None else 0)

    @torch.no_grad()
    def dequant(self, experts: torch.Tensor, use_triton=None) -> torch.Tensor:
        """experts (k,) long -> (k, R, C) bf16; same values/order as Inkling2LoopDequantizeExperts."""
        use_triton = USE_TRITON if use_triton is None else use_triton
        if use_triton and self.dtype == torch.bfloat16:
            k = experts.numel()
            out = torch.empty(k, self.R, self.C, dtype=torch.bfloat16, device=self.weight.device)
            eids = experts.to(torch.int32).contiguous()
            BLOCK = 2048
            if self.kind == "nvfp4":
                CB = self.C // 2
                grid = (k, triton.cdiv(self.R * CB, BLOCK))
                _nvfp4_kernel[grid](self.weight, self.scale, self.factor, eids, out, self.R, CB, self.C // 16,
                                    BLOCK=BLOCK)
            else:
                grid = (k, triton.cdiv(self.R * self.C, BLOCK))
                _mxfp8_kernel[grid](self.weight, self.scale, eids, out, self.R, self.C, self.C // 32, BLOCK=BLOCK)
            return out
        block_scale = self.scale[experts].float()
        if self.kind == "mxfp8":
            values = self.weight[experts].float().view(*block_scale.shape, 32)
            values = values * block_scale[..., None]
        else:
            codes = self.weight[experts].long()
            values = self.lut[codes].view(*block_scale.shape, 16) * block_scale[..., None]
            values = values * self.factor[experts, None, None, None]
        return values.reshape(-1, self.R, self.C).to(self.dtype)

    def permute_(self, gen: torch.Generator):
        """random-init control: shuffle packed codes and block scales within each expert (matched marginals,
        destroyed structure). In place, chunked per expert."""
        with torch.no_grad():
            for e in range(self.E):
                for t in (self.weight, self.scale):
                    w = t[e].reshape(-1)
                    w.copy_(w[torch.randperm(w.numel(), generator=gen, device="cpu").to(w.device)])


class QuantExperts(nn.Module):
    """drop-in for Inkling2LoopExperts with packed weights (forward signature and arithmetic identical).

    FAST (class flag, False by default): sort-by-expert + two grouped GEMMs over the hit experts instead of the
    native per-expert Python loop (~1.3k kernels/layer). Same weights; fp32 accumulation order of the top-k sum
    differs, so FAST is NOT bit-identical to the native loop. It is used ONLY inside System I calls (Sys1Core sets
    and restores it); System II / base / language evals always run the exact native loop."""

    CHUNK = 16
    FAST = False

    def __init__(self, native, gate_up: PackedExpertWeight, down: PackedExpertWeight, cache: bool = False):
        super().__init__()
        self.num_experts = native.num_experts
        self.hidden_dim, self.intermediate_dim = native.hidden_dim, native.intermediate_dim
        self.act_fn = native.act_fn
        self.gate_up, self.down = gate_up, down
        self.cached = None
        if cache:
            ar = torch.arange(self.num_experts, device=gate_up.weight.device)
            self.cached = (torch.cat([gate_up.dequant(ar[i:i + 8]) for i in range(0, self.num_experts, 8)]),
                           torch.cat([down.dequant(ar[i:i + 8]) for i in range(0, self.num_experts, 8)]))

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate, up)

    def _forward_fast(self, hidden_states, top_k_index, top_k_weights):
        E = self.num_experts
        T, K = top_k_index.shape
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat, stable=True)
        with torch.no_grad():
            counts = torch.bincount(flat, minlength=E + 1)[:E]
            hit = counts.nonzero()[:, 0]
            nvalid = int(counts.sum())
        order = order[:nvalid]                                    # index E (= no expert) sorts last
        tok = order // K
        offs = torch.cumsum(counts[hit], 0).to(torch.int32)
        if self.cached is not None:
            w13, w2 = self.cached[0][hit], self.cached[1][hit]
        else:
            w13, w2 = self.gate_up.dequant(hit), self.down.dequant(hit)
        xs = hidden_states[tok]
        g = torch._grouped_mm(xs, w13.transpose(1, 2), offs=offs)
        a = self._apply_gate(g)
        y = torch._grouped_mm(a, w2.transpose(1, 2), offs=offs)
        y = y * top_k_weights.reshape(-1)[order][:, None]
        out = torch.zeros_like(hidden_states, dtype=torch.float32)
        out = out.index_add(0, tok, y.to(torch.float32))
        return out.to(hidden_states.dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if QuantExperts.FAST:
            return self._forward_fast(hidden_states, top_k_index, top_k_weights)
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        hit = [int(e) for e in expert_hit[:, 0].tolist() if int(e) != self.num_experts]
        for c0 in range(0, len(hit), self.CHUNK):
            ids = hit[c0:c0 + self.CHUNK]
            if self.cached is not None:
                w13s = w2s = None
            else:
                it = torch.tensor(ids, device=hidden_states.device)
                w13s, w2s = self.gate_up.dequant(it), self.down.dequant(it)
            for j, expert_idx in enumerate(ids):
                w13 = self.cached[0][expert_idx] if w13s is None else w13s[j]
                w2 = self.cached[1][expert_idx] if w2s is None else w2s[j]
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                cur = self._apply_gate(F.linear(current_state, w13))
                cur = F.linear(cur, w2)
                cur = cur * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, cur.to(final_hidden_states.dtype))
            del w13s, w2s
        return final_hidden_states.to(hidden_states.dtype)


def _install_interceptors():
    """patch the transformers expert dequant op: stash packed tensors, return a zero-stride placeholder."""
    from transformers import core_model_loading as cml

    if getattr(cml, "_sys12_patched", False):
        return
    orig_deq, orig_il = cml.Inkling2LoopDequantizeExperts.convert, cml.Interleave.convert

    def deq(self, input_dict, source_patterns, target_patterns, model=None, full_layer_name=None, **kw):
        tensors = {k: v[0] if isinstance(v, list) else v for k, v in input_dict.items()}
        if len(tensors) == 1 or not _STASH.get("_active"):
            return orig_deq(self, input_dict, source_patterns, target_patterns, model=model,
                            full_layer_name=full_layer_name, **kw)
        weight, scale, amax, denom = (tensors.get(p) for p in source_patterns)
        rows = weight.shape[1]
        cols = weight.shape[2] * 2 if amax is not None else weight.shape[2]
        dev = _STASH["_device"]
        put = lambda t: None if t is None else t.to(dev)
        _STASH[full_layer_name] = PackedExpertWeight(put(weight), put(scale), put(amax), put(denom),
                                                     interleave="gate_up" in full_layer_name)
        ph = torch.zeros((), dtype=torch.bfloat16, device=dev).expand(weight.shape[0], rows, cols)
        del weight, scale, tensors
        ph._sys12_placeholder = True
        return {target_patterns[0]: ph}

    def il(self, input_dict, source_patterns, target_patterns, **kw):
        t = next(iter(input_dict.values()))
        t = t[0] if isinstance(t, list) else t
        if getattr(t, "_sys12_placeholder", False) or (t.dim() == 3 and t.stride() == (0, 0, 0)):
            return {target_patterns[0]: t}
        return orig_il(self, input_dict, source_patterns, target_patterns, **kw)

    cml.Inkling2LoopDequantizeExperts.convert = deq
    cml.Interleave.convert = il
    cml._sys12_patched = True



def load_inkling_quant(path="ckpt/Inkling-2-Mini", device="cuda", cache_layers=(), keep_lm_head=True, log=print):
    """-> (full model, text model, tokenizer, report). Experts stay packed except `cache_layers`."""
    from transformers import AutoTokenizer
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForConditionalGeneration
    from .inkling import mem_snapshot

    _install_interceptors()
    # synchronous loading: tensors are materialised only when their converter runs (the default async pool submits
    # every tensor up front -> ~53 GB transient of bf16-cast expert codes on GB10 unified memory)
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    # transformers pre-allocates the full EXPECTED (bf16-dequantized) model size in one block to warm the caching
    # allocator (52.8 GB, kept reserved afterwards): disable it
    import transformers.modeling_utils as mu
    mu.caching_allocator_warmup = lambda *a, **k: None
    _STASH.clear()
    _STASH["_active"] = True
    _STASH["_device"] = torch.device(device)
    t0 = time.time()
    rep = {"mem_before": mem_snapshot()}
    model = Inkling2LoopForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
    _STASH["_active"] = False
    rep["load_s"] = round(time.time() - t0, 1)
    tm = model.model.language_model
    model.model.vision_tower = None
    if not keep_lm_head:
        model.lm_head = None
    packed_bytes = 0
    keys = [k for k in _STASH if not k.startswith("_")]
    by_layer: dict = {}
    for k in keys:
        by_layer.setdefault(_layer_of(k), {})["gate_up" if "gate_up" in k else "down"] = _STASH.pop(k)
    assert sorted(by_layer) == list(range(tm.config.num_hidden_layers)), sorted(by_layer)
    for li, d in by_layer.items():
        mlp = tm.layers[li].mlp
        gu, dn = d["gate_up"], d["down"]
        packed_bytes += gu.nbytes() + dn.nbytes()
        mlp.experts = QuantExperts(mlp.experts, gu, dn, cache=li in set(cache_layers))
    torch.cuda.empty_cache()
    tm.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rep["packed_expert_GB"] = round(packed_bytes / 1e9, 2)
    rep["cache_layers"] = list(cache_layers)
    rep["dense_params"] = sum(p.numel() for p in tm.parameters())
    rep["mem_ready"] = mem_snapshot()
    tok = AutoTokenizer.from_pretrained(path)
    log(json.dumps(rep))
    return model, tm, tok, rep


def randomize_text_model_(tm, layer_ids, seed=0, embed=True):
    """random-init control with matched weight statistics: every tensor of the listed layers (and optionally the
    token embedding) is shuffled element-wise (dense) / code- and scale-wise within each expert (packed)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for li in layer_ids:
            layer = tm.layers[li]
            for n, p in layer.named_parameters():
                if p.dim() == 0 or getattr(p, "_sys12_placeholder", False) or p.stride() == (0,) * p.dim():
                    continue
                flat = p.data.reshape(-1)
                flat.copy_(flat[torch.randperm(flat.numel(), generator=gen).to(flat.device)])
            ex = layer.mlp.experts
            if isinstance(ex, QuantExperts):
                ex.gate_up.permute_(gen)
                ex.down.permute_(gen)
                ex.cached = None
        if embed:
            w = tm.embed_tokens.weight.data
            w.copy_(w[torch.randperm(w.shape[0], generator=gen).to(w.device)])  # permute token rows
