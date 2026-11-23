"""Actual pretrained Inkling-2-Mini recurrent core as the refinement module.

What this file does (and does not) claim:
- loads the pinned checkpoint thinkingmachines/Inkling-2-Mini @ b85c3cf through the
  transformers v5.24 implementation, which dequantizes the native NVFP4/MXFP8 experts
  to bf16 at load. All base weights are frozen.
- `instrumented_forward` re-implements Inkling2LoopTextModel.forward step by step
  (pre-loop layers, 3 passes over the shared core layers 6..17 with native loop
  messages, block attention residuals, convolutions, relative logits and sinks all
  executed by the native modules) and exposes hooks at each pass start/end. With
  hooks disabled it must match the native forward (tests/test_core_integration.py).
- unordered typed memory never enters the token sequence. A small fixed set of
  readout tokens (plus one action/query token) is appended to a fixed text prefix;
  at the start of every core pass, readout positions cross-attend to the memory
  slots (READ adapter); at the end of every pass, slots cross-attend to the readout
  hidden states (WRITE adapter, shared weights across passes). Gates are zero-init,
  so the adapted model starts exactly at the native computation.
- LoRA (rank 8, alpha 32) on the attention projections of the core layers. HF names:
  self_attn.{q_proj,k_proj,v_proj}.linear and self_attn.o_proj.linear. These
  correspond to the native checkpoint's attn.wq_du / wk_dv / wv_dv / wo_ud (NeMo's
  fused "qkvr" includes the relative-logit projection wr_du, which we do not adapt).
  Experts are not adapted.
- readout is taken at the CORE EXIT (after pass 3 of layer 17), not from final text
  embeddings; post-loop layers are skipped during training (they are executed in
  the equivalence test).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_util

INKLING_REPO = "thinkingmachines/Inkling-2-Mini"
INKLING_REVISION = "b85c3cfa615fc413cc9af3379f491b963f3b586a"
PHYS_PROMPT = (
    "<|message_system|><|content_text|>You maintain a physical scene memory of objects, contacts and "
    "friction. Read the memory and predict how the target object moves under the proposed push.<|end_message|>"
)


# ---------------------------------------------------------------------------
# native module access
# ---------------------------------------------------------------------------

def _mod():
    from transformers.models.inkling2_loop import modeling_inkling2_loop as m

    return m


def build_masks(tm, inputs_embeds, attention_mask=None):
    m = _mod()
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
    mask_kwargs = {"config": tm.config, "inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
                   "cache": None, "position_ids": position_ids}
    masks = {
        "full_attention": m.create_causal_mask(**mask_kwargs),
        "sliding_attention": m.create_sliding_window_causal_mask(**mask_kwargs),
        "linear_attention": m.create_recurrent_attention_mask(**mask_kwargs),
    }
    rel = {
        "full_attention": m.get_relative_distance(inputs_embeds, None, is_sliding=False),
        "sliding_attention": m.get_relative_distance(inputs_embeds, None, is_sliding=True),
    }
    return masks, rel, position_ids


@dataclass
class CoreHooks:
    """pass_start(p, hidden) -> hidden ; pass_end(p, hidden) -> None.
    `record` collects per-pass core-exit states for inspection."""

    pass_start: Optional[Callable] = None
    pass_end: Optional[Callable] = None
    record: list = field(default_factory=list)
    keep_records: bool = False


def instrumented_forward(tm, inputs_embeds, hooks: Optional[CoreHooks] = None, stop_after_core: bool = False,
                         num_passes: Optional[int] = None, checkpoint_layers: bool = False, attention_mask=None):
    """Step-by-step replica of Inkling2LoopTextModel.forward (no cache).

    Returns (output, info): output is the core-exit hidden state when
    stop_after_core else the final normed last_hidden_state (native output).
    """
    cfg = tm.config
    masks, rel, position_ids = build_masks(tm, inputs_embeds, attention_mask)
    loop_start, loop_end = cfg.loop_layer_index_start, cfg.loop_layer_index_end
    passes = cfg.loop_num_passes if num_passes is None else num_passes

    def call(layer, hs, bs, messages=None):
        kw = dict(attention_mask=masks[layer.attention_type], relative_distance=rel[layer.attention_type],
                  conv_mask=masks["linear_attention"], cache=None, position_ids=position_ids)
        args = (hs, bs) if messages is None else (hs, bs, messages)
        if checkpoint_layers and torch.is_grad_enabled():
            return ckpt_util.checkpoint(layer, *args, use_reentrant=False, **kw)
        return layer(*args, **kw)

    hidden = inputs_embeds
    block = inputs_embeds.new_zeros(*inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1])
    for layer in tm.layers[:loop_start]:
        new = call(layer, hidden, block)
        if layer.is_block_boundary:
            block = torch.cat([block.to(hidden.device), hidden.unsqueeze(-2)], dim=-2)
        hidden = new
    pre_hidden, pre_block = hidden, block
    n_pre_blocks = pre_block.shape[-2]

    # pass 0
    if hooks is not None and hooks.pass_start is not None:
        hidden = hooks.pass_start(0, hidden)
    for layer in tm.layers[loop_start:loop_end]:
        new = call(layer, hidden, block)
        if layer.is_block_boundary:
            block = torch.cat([block.to(hidden.device), hidden.unsqueeze(-2)], dim=-2)
        hidden = new
    if hooks is not None:
        if hooks.pass_end is not None:
            hooks.pass_end(0, hidden)
        if hooks.keep_records:
            hooks.record.append(hidden)
    for p in range(1, passes):
        prev_block = block[..., n_pre_blocks + 1:, :]
        prev_hidden = hidden
        hidden, block = pre_hidden, pre_block
        if hooks is not None and hooks.pass_start is not None:
            hidden = hooks.pass_start(p, hidden)
        for li, layer in enumerate(tm.layers[loop_start:loop_end]):
            messages = tm.message_input(prev_block, prev_hidden, li)
            new = call(layer, hidden, block, messages)
            if layer.is_block_boundary:
                block = torch.cat([block.to(hidden.device), hidden.unsqueeze(-2)], dim=-2)
            hidden = new
        if hooks is not None:
            if hooks.pass_end is not None:
                hooks.pass_end(p, hidden)
            if hooks.keep_records:
                hooks.record.append(hidden)
    info = {"passes": passes}
    if stop_after_core:
        return hidden, info
    for layer in tm.layers[loop_end:]:
        new = call(layer, hidden, block)
        if layer.is_block_boundary:
            block = torch.cat([block.to(hidden.device), hidden.unsqueeze(-2)], dim=-2)
        hidden = new
    hidden = tm.final_input(block, hidden)[0]
    hidden = tm.norm(hidden).to(inputs_embeds.dtype)
    return hidden, info


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 32.0):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.enabled = True

    @property
    def weight(self):  # some native code reads .weight.dtype
        return self.base.weight

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        d = F.linear(F.linear(x.float(), self.A), self.B) * self.scale
        return y + d.to(y.dtype)


LORA_TARGETS = ("self_attn.q_proj.linear", "self_attn.k_proj.linear", "self_attn.v_proj.linear", "self_attn.o_proj.linear")


def inject_lora(tm, layer_ids, r=8, alpha=32.0, targets=LORA_TARGETS):
    names = []
    for li in layer_ids:
        layer = tm.layers[li]
        for t in targets:
            parent_path, attr = t.rsplit(".", 1)
            parent = layer.get_submodule(parent_path)
            base = getattr(parent, attr)
            if isinstance(base, LoRALinear):
                continue
            setattr(parent, attr, LoRALinear(base, r, alpha))
            names.append(f"layers.{li}.{t}")
    return names


def lora_modules(tm):
    return [m for m in tm.modules() if isinstance(m, LoRALinear)]


def set_lora_enabled(tm, flag: bool):
    for m in lora_modules(tm):
        m.enabled = flag


def reset_lora(tm, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    for m in lora_modules(tm):
        with torch.no_grad():
            bound = 1.0 / math.sqrt(m.A.shape[1])
            m.A.copy_((torch.rand(m.A.shape, generator=g) * 2 - 1).mul_(bound).to(m.A.device))
            m.B.zero_()


class _XAttn(nn.Module):
    """Set cross-attention: queries (B,Q,dq) attend to keys/values (B,K,dk) with mask."""

    def __init__(self, dq, dk, dout, d_attn=256, heads=4):
        super().__init__()
        self.h, self.dh = heads, d_attn // heads
        self.nq, self.nk = nn.LayerNorm(dq), nn.LayerNorm(dk)
        self.q = nn.Linear(dq, d_attn, bias=False)
        self.k = nn.Linear(dk, d_attn, bias=False)
        self.v = nn.Linear(dk, d_attn, bias=False)
        self.o = nn.Linear(d_attn, dout, bias=False)

    def forward(self, xq, xk, kmask=None):
        B, Q, _ = xq.shape
        K = xk.shape[1]
        q = self.q(self.nq(xq)).view(B, Q, self.h, self.dh).transpose(1, 2)
        k = self.k(self.nk(xk)).view(B, K, self.h, self.dh).transpose(1, 2)
        v = self.v(self.nk(xk)).view(B, K, self.h, self.dh).transpose(1, 2)
        am = None
        if kmask is not None:
            am = kmask[:, None, None, :].to(torch.bool)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        out = torch.nan_to_num(out)  # rows with no valid key
        return self.o(out.transpose(1, 2).reshape(B, Q, -1))


class InklingCore(nn.Module):
    """refine(tokens, mask, query) with the pretrained Inkling recurrent core.

    tokens: (B,M,d) memory slot tokens from a physmem encoder (typed or generic).
    query : (B,A) action/query features (shared action contract).
    returns tokens' (B, M+R, d) = [written-back slots, projected readout tokens],
    mask' (B, M+R).
    """

    def __init__(self, tm, d_mem: int, query_dim: int, prefix_ids: torch.Tensor, n_readout: int = 16,
                 lora_rank: int = 8, lora_alpha: float = 32.0, core_lora: bool = True, checkpoint_layers: bool = False):
        super().__init__()
        self.tm = tm  # frozen base (registered so .to() works, but excluded from trainable params)
        for p in self.tm.parameters():
            p.requires_grad_(False)
        H = tm.config.hidden_size
        self.H, self.d, self.R = H, d_mem, n_readout
        self.passes = tm.config.loop_num_passes
        self.register_buffer("prefix_ids", prefix_ids.view(1, -1).long(), persistent=False)
        with torch.no_grad():
            emb = tm.embed_tokens(self.prefix_ids.to(tm.embed_tokens.weight.device)).float()
        scale = float(emb.pow(2).mean().sqrt())
        dev = tm.embed_tokens.weight.device
        self.readout_embed = nn.Parameter(torch.randn(n_readout, H, device=dev) * scale)
        self.query_proj = nn.Linear(query_dim, H).to(dev)
        nn.init.normal_(self.query_proj.weight, std=scale / math.sqrt(query_dim))
        nn.init.zeros_(self.query_proj.bias)
        self.read = _XAttn(H, d_mem, H).to(dev)
        self.write = _XAttn(d_mem, H, d_mem).to(dev)
        self.read_gate = nn.Parameter(torch.zeros(self.passes, device=dev))
        self.write_gate = nn.Parameter(torch.zeros(self.passes, device=dev))
        self.out_norm = nn.LayerNorm(H).to(dev)
        self.out_proj = nn.Linear(H, d_mem).to(dev)
        self.core_layer_ids = list(range(tm.config.loop_layer_index_start, tm.config.loop_layer_index_end))
        self.lora_names = inject_lora(tm, self.core_layer_ids, lora_rank, lora_alpha) if core_lora else []
        self.checkpoint_layers = checkpoint_layers
        self.bypass = False  # constant-message / core-bypass diagnostic
        self.adapters_enabled = True
        self.last_info = {}

    # trainable = everything except the frozen base (LoRA params live inside tm but require grad)
    def trainable_parameters(self):
        own = [p for n, p in self.named_parameters() if not n.startswith("tm.")]
        lora = [p for m in lora_modules(self.tm) for p in (m.A, m.B)]
        return own + lora

    def build_inputs(self, B, query):
        dt = self.tm.embed_tokens.weight.dtype
        prefix = self.tm.embed_tokens(self.prefix_ids.expand(B, -1))
        qtok = self.query_proj(query.float()).unsqueeze(1)
        ro = self.readout_embed.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([prefix, qtok.to(dt), ro.to(dt)], dim=1)

    def forward(self, tokens, mask, query, num_passes: Optional[int] = None, keep_records=False):
        B, M, d = tokens.shape
        x = self.build_inputs(B, query)
        L = x.shape[1]
        r0 = L - self.R
        slots = [tokens.float()]
        mask_f = mask

        if self.bypass:
            h_ro = self.readout_embed.unsqueeze(0).expand(B, -1, -1).float()
            ro = self.out_proj(self.out_norm(h_ro))
            return torch.cat([slots[-1], ro], 1), torch.cat([mask_f, torch.ones(B, self.R, dtype=torch.bool, device=mask.device)], 1)

        def pass_start(p, hidden):
            if not self.adapters_enabled:
                return hidden
            g = torch.tanh(self.read_gate[p])
            upd = self.read(hidden[:, r0:].float(), slots[-1], mask_f) * g
            pad = torch.zeros(B, r0, self.H, device=hidden.device, dtype=upd.dtype)
            return (hidden.float() + torch.cat([pad, upd], 1)).to(hidden.dtype)

        def pass_end(p, hidden):
            if not self.adapters_enabled:
                return
            g = torch.tanh(self.write_gate[p])
            ro = hidden[:, r0:].float()
            slots.append(slots[-1] + g * self.write(slots[-1], ro))

        hooks = CoreHooks(pass_start, pass_end, keep_records=keep_records)
        h, info = instrumented_forward(self.tm, x, hooks, stop_after_core=True, num_passes=num_passes,
                                       checkpoint_layers=self.checkpoint_layers)
        ro = self.out_proj(self.out_norm(h[:, r0:].float()))
        self.last_info = {**info, "records": hooks.record if keep_records else None, "seq_len": L}
        out = torch.cat([slots[-1], ro], 1)
        out_mask = torch.cat([mask_f, torch.ones(B, self.R, dtype=torch.bool, device=mask.device)], 1)
        return out, out_mask


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def mem_snapshot():
    out = {}
    try:
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
        out["MemAvailable_GB"] = round(mi["MemAvailable"] / 1e6, 2)
        out["SwapUsed_GB"] = round((mi["SwapTotal"] - mi["SwapFree"]) / 1e6, 2)
    except Exception:  # noqa: BLE001
        pass
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith(("VmRSS", "VmHWM")):
                out[l.split(":")[0] + "_GB"] = round(int(l.split()[1]) / 1e6, 2)
    if torch.cuda.is_available():
        out["cuda_alloc_GB"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        out["cuda_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        out["cuda_reserved_GB"] = round(torch.cuda.memory_reserved() / 1e9, 2)
    return out


def load_inkling_text(path="ckpt/Inkling-2-Mini", device="cuda", drop_vision=True, log=print):
    """Load the pinned checkpoint, return (text_model, tokenizer, report)."""
    from transformers import AutoTokenizer
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForConditionalGeneration

    t0 = time.time()
    rep = {"repo": INKLING_REPO, "revision": INKLING_REVISION, "path": path, "mem_before": mem_snapshot()}
    model = Inkling2LoopForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
    rep["load_s"] = round(time.time() - t0, 1)
    rep["mem_after_load"] = mem_snapshot()
    tm = model.model.language_model
    if drop_vision:
        model.model.vision_tower = None
        model.lm_head = None
    tm.eval()
    for p in tm.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in tm.parameters())
    rep["text_params"] = n
    rep["dtypes"] = sorted({str(p.dtype) for p in tm.parameters()})
    rep["config"] = {k: getattr(tm.config, k) for k in ("hidden_size", "num_hidden_layers", "loop_layer_index_start",
                                                        "loop_layer_index_end", "loop_num_passes", "n_routed_experts",
                                                        "num_experts_per_tok")}
    tok = AutoTokenizer.from_pretrained(path)
    torch.cuda.empty_cache()
    rep["mem_ready"] = mem_snapshot()
    log(json.dumps(rep))
    return tm, tok, model, rep


def prefix_ids_from_tokenizer(tok, text=PHYS_PROMPT):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return torch.tensor(ids, dtype=torch.long)


def frozen_checksum(tm, n_tensors=24):
    """Cheap frozen-weight check without cloning: per-tensor version counters and
    float64 sums of a fixed subset of base tensors."""
    out = {}
    names = [n for n, _ in tm.named_parameters() if ".A" not in n[-2:] and ".B" not in n[-2:]]
    step = max(1, len(names) // n_tensors)
    params = dict(tm.named_parameters())
    for n in names[::step]:
        p = params[n]
        out[n] = (p._version, float(p.detach().double().sum().item()))
    return out


def tiny_text_model(device="cuda", seed=0):
    """Randomly initialised Inkling2Loop text model with the same architecture
    family (for fast unit tests only; never used for results)."""
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopTextModel

    torch.manual_seed(seed)
    cfg = Inkling2LoopTextConfig(vocab_size=512, unpadded_vocab_size=500, hidden_size=128, num_hidden_layers=8,
                                 num_attention_heads=4, num_key_value_heads=2, head_dim=32, swa_num_attention_heads=4,
                                 swa_num_key_value_heads=2, swa_head_dim=32, sliding_window_size=16, rel_extent=32,
                                 moe_intermediate_size=64, n_routed_experts=8, num_experts_per_tok=2,
                                 n_layers_per_attn_res_block=2, loop_layer_index_start=2, loop_layer_index_end=6,
                                 loop_num_passes=3, loop_gate_rank=16, log_scaling_n_floor=16)
    cfg._attn_implementation = "eager"
    tm = Inkling2LoopTextModel(cfg).to(device=device, dtype=torch.bfloat16)
    keep = Inkling2LoopTextModel._keep_in_fp32_modules_strict
    with torch.no_grad():
        for n, p in list(tm.named_parameters()) + list(tm.named_buffers()):
            if any(k in n for k in keep):
                p.data = p.data.float()
        # modules constructed with explicit fp32 dtype in __init__
        for mod in tm.modules():
            if type(mod).__name__ in ("Inkling2LoopMessageGate",):
                mod.forget_gate_proj.float(); mod.message_gate_proj.float()
    # non-zero attention-residual / message queries so the loop actually mixes
    with torch.no_grad():
        for n, p in tm.named_parameters():
            if p.abs().sum() == 0 and p.dtype == torch.float32:
                p.normal_(0, 0.02)
    tm.eval()
    return tm


# ---------------------------------------------------------------------------
# full model: compact encoder/head (agent MODEL interface) + Inkling core
# ---------------------------------------------------------------------------

class InklingPhysMem(nn.Module):
    """i00 = generic memory encoder, observation loss only; i11 = typed structured
    memory + physics losses. Both: same Inkling core adapters/LoRA, same head.
    The compact model's own RecurrentCore is removed (not used, not counted)."""

    def __init__(self, compact, core: InklingCore, variant: str):
        super().__init__()
        self.compact = compact
        self.compact.core = nn.Identity()
        self.core = core
        self.variant = variant
        self.physics = compact.physics
        self.passes_train = (core.passes,)

    @property
    def d(self):
        return self.compact.d

    def set_target_scaler(self, mean, std):
        self.compact.set_target_scaler(mean, std)

    def encode(self, events, query, edge_lesion=None):
        return self.compact.encode(events, query, edge_lesion=edge_lesion)

    def refine(self, tokens, mask, query, passes=None, keep_records=False):
        return self.core(tokens, mask, query.action, num_passes=passes, keep_records=keep_records)

    def _extend_slots(self, slots, B, device):
        s2 = dict(slots)
        R = self.core.R
        for k, fill in (("token_entity", -1), ("token_kind", 0)):
            if k in s2 and s2[k] is not None:
                s2[k] = torch.cat([s2[k], torch.full((B, R), fill, dtype=s2[k].dtype, device=device)], 1)
        return s2

    def readout(self, tokens, mask, query, slots):
        return self.compact.readout(tokens, mask, query, slots)

    def forward(self, events, query, passes=None, core_passes=None):
        # `passes` (the compact-model refinement count used by shared eval code) is
        # deliberately ignored: the Inkling core always runs its native loop count
        # unless `core_passes` is given explicitly (out-of-distribution diagnostic).
        tokens, mask, slots = self.encode(events, query)
        t2, m2 = self.refine(tokens, mask, query, core_passes)
        out = self.readout(t2, m2, query, self._extend_slots(slots, tokens.shape[0], tokens.device))
        out.cost = {"passes": self.core.passes if core_passes is None else core_passes, "tokens": int(t2.shape[1]),
                    "core_seq_len": self.core.last_info.get("seq_len")}
        return out

    def intervene(self, tokens, mask, slots, prop, mean, logvar):
        return self.compact.intervene(tokens, mask, slots, prop, mean, logvar)

    def forward_from_tokens(self, tokens, mask, slots, query, passes=None):
        t2, m2 = self.refine(tokens, mask, query, passes)
        return self.readout(t2, m2, query, self._extend_slots(slots, tokens.shape[0], tokens.device))

    def trainable_parameters(self):
        return [p for p in self.compact.parameters() if p.requires_grad] + self.core.trainable_parameters()


def build_inkling_model(variant: str, tm, prefix_ids, d: int = 128, n_readout: int = 16, seed: int = 0,
                        core_lora: bool = True, checkpoint_layers: bool = False):
    """variant i00 -> compact b00 encoder/head; i11 -> compact b11. 'i11nolora' = adapters only (no LoRA)."""
    from .compact import build_model
    from ..schema import ACTION_DIM

    torch.manual_seed(seed)
    base = {"i00": "b00", "i11": "b11", "i11nolora": "b11", "i00nolora": "b00"}[variant]
    compact = build_model(base, d=d).cuda()
    # remove LoRA from previous runs sharing the same frozen base, then (re)inject
    for li in range(tm.config.loop_layer_index_start, tm.config.loop_layer_index_end):
        layer = tm.layers[li]
        for t in LORA_TARGETS:
            parent_path, attr = t.rsplit(".", 1)
            parent = layer.get_submodule(parent_path)
            mod = getattr(parent, attr)
            if isinstance(mod, LoRALinear):
                setattr(parent, attr, mod.base)
    use_lora = core_lora and not variant.endswith("nolora")
    core = InklingCore(tm, d_mem=d, query_dim=ACTION_DIM, prefix_ids=prefix_ids, n_readout=n_readout,
                       core_lora=use_lora, checkpoint_layers=checkpoint_layers)
    if use_lora:
        reset_lora(tm, seed)
    return InklingPhysMem(compact, core, variant)


# ---------------------------------------------------------------------------
# training suite (i00 vs i11 on the actual core) — shares data/loss/eval code
# with the compact factorial (agent MODEL's train.py / evaluate.py)
# ---------------------------------------------------------------------------

def _head_of(model):
    return model.compact.head


InklingPhysMem.head = property(_head_of)


def inkling_train_run(model, variant, seed, budget_s, train_ds, val_ds, scales, ym, ys, bs=8, lr=3e-4,
                      val_every_s=300, landmarks_s=(600, 1200, 1800), log=print, run_tag="main"):
    """Time-budgeted adapter training. Saves trainable-parameter state only."""
    from ..models.losses import LossConfig, compute_loss
    from ..train import RES, append_jsonl, utc

    run_id = f"inkling-{run_tag}-{variant}-s{seed}"
    curves = RES / "curves" / f"{run_id}.jsonl"
    curves.parent.mkdir(parents=True, exist_ok=True)
    ckdir = RES / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    model.set_target_scaler(ym, ys)
    lcfg = LossConfig(physics=model.physics)
    params = model.trainable_parameters()
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    g = torch.Generator(device=train_ds.device).manual_seed(seed)
    t0 = time.time()
    step, examples = 0, 0
    best = (float("inf"), -1)
    next_val = val_every_s
    lm = list(landmarks_s)
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    it = None
    row = dict(time_utc=utc(), run_id=run_id, kind="inkling_train", variant=variant, seed=seed, n_trainable=n_train,
               budget_s=budget_s, bs=bs, lr=lr, loss_cfg=lcfg.to_dict(), lora=bool(model.core.lora_names))

    def save(path, extra):
        sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(dict(trainable=sd, variant=variant, seed=seed, step=step, examples=examples, scales=scales,
                        y_mean=ym.cpu(), y_std=ys.cpu(), elapsed_s=time.time() - t0, **extra), path)

    try:
        model.train()
        model.core.tm.eval()
        while time.time() - t0 < budget_s:
            if it is None:
                it = train_ds.iterate(bs, shuffle=True, generator=g)
            try:
                ev, q, lab = next(it)
            except StopIteration:
                it = None
                continue
            ts = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(ev, q)
                loss, parts = compute_loss(out, lab, q, lcfg, scales, model.head.y_mean, model.head.y_std)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            torch.cuda.synchronize()
            step_times.append(time.time() - ts)
            step += 1
            examples += int(q.target.shape[0])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if step % 10 == 0 or step == 1:
                append_jsonl(curves, dict(run_id=run_id, split="train", step=step, examples=examples,
                                          elapsed_s=time.time() - t0, grad_norm=float(gn),
                                          read_gate=[float(x) for x in torch.tanh(model.core.read_gate).detach()],
                                          write_gate=[float(x) for x in torch.tanh(model.core.write_gate).detach()],
                                          **{k: float(v) for k, v in parts.items()}))
            el = time.time() - t0
            if el >= next_val or el >= budget_s:
                next_val += val_every_s
                vm = inkling_val(model, val_ds, scales, lcfg)
                append_jsonl(curves, dict(run_id=run_id, split="val", step=step, examples=examples, elapsed_s=el, **vm))
                log(f"[{run_id}] step {step} ex {examples} val_nll {vm['nll']:.4f} mse {vm['mse_norm']:.4f} t={el:.0f}s "
                    f"step_s={sum(step_times[-20:])/len(step_times[-20:]):.2f}")
                if vm["nll"] < best[0]:
                    best = (vm["nll"], step)
                    save(ckdir / f"{run_id}.pt", dict(val=vm))
            while lm and el >= lm[0]:
                save(ckdir / f"{run_id}_at{int(lm.pop(0))}s.pt", {})
        vm = inkling_val(model, val_ds, scales, lcfg)
        append_jsonl(curves, dict(run_id=run_id, split="val", step=step, examples=examples, elapsed_s=time.time() - t0, **vm))
        if vm["nll"] < best[0]:
            best = (vm["nll"], step)
            save(ckdir / f"{run_id}.pt", dict(val=vm))
        save(ckdir / f"{run_id}_final.pt", dict(val=vm))
        st = sorted(step_times[5:]) or step_times
        row.update(status="ok", steps=step, examples=examples, wall_s=time.time() - t0,
                   step_s_mean=sum(step_times[5:]) / max(1, len(step_times[5:])), step_s_p95=st[int(0.95 * (len(st) - 1))] if st else None,
                   peak_cuda_gb=torch.cuda.max_memory_allocated() / 1e9, best_val_nll=best[0], best_step=best[1],
                   mem=mem_snapshot())
    except Exception as e:  # noqa: BLE001
        import traceback

        row.update(status="failed", error=repr(e), traceback=traceback.format_exc(), wall_s=time.time() - t0, steps=step)
        log(row["traceback"])
    row["end_utc"] = utc()
    append_jsonl(RES / "ledger.jsonl", row)
    return row


@torch.no_grad()
def inkling_val(model, val_ds, scales, lcfg, bs=32):
    from ..models.losses import compute_loss

    was = model.training
    model.eval()
    agg, n = {}, 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for ev, q, lab in val_ds.iterate(bs):
            out = model(ev, q)
            _, parts = compute_loss(out, lab, q, lcfg, scales, model.head.y_mean, model.head.y_std)
            b = int(q.target.shape[0])
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v) * b
            n += b
    model.train(was)
    return {k: v / max(n, 1) for k, v in agg.items()}


def load_trainable(model, path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    named = dict(model.named_parameters())
    for n, v in ck["trainable"].items():
        named[n].data.copy_(v.to(named[n].device))
    model.set_target_scaler(ck["y_mean"], ck["y_std"])
    return ck


def train_suite(suite, config, budget_min=None, seeds=None, variants=None, val_n=256):
    """inkling-pair: i00 vs i11 (matched), time-budgeted per run; one model load."""
    from ..train import RES, append_jsonl, fit_target_scaler, load_dataset, utc
    from ..models.losses import fit_loss_scales

    ic = config.get("inkling", {})
    variants = variants or ic.get("variants", ["i00", "i11"])
    seeds = seeds or ic.get("seeds", [0])
    budget_min = budget_min or ic.get("run_minutes", 30)
    logf = open(RES / "inkling_train.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    tm, tok, full, rep = load_inkling_text(config.get("paths", {}).get("checkpoint", "ckpt/Inkling-2-Mini"), log=log)
    prefix = prefix_ids_from_tokenizer(tok).cuda()
    tr = load_dataset("train", config, "cuda")
    va = load_dataset("val", config, "cuda")
    g = torch.Generator().manual_seed(999)
    va = va.subset(torch.randperm(va.n, generator=g)[:val_n].to(va.device))
    ym, ys = fit_target_scaler(tr)
    scales = fit_loss_scales(tr.iterate(1024))
    rows = []
    for seed in seeds:
        for v in variants:  # interleave variants within a seed: matched conditions
            model = build_inkling_model(v, tm, prefix, seed=seed, checkpoint_layers=ic.get("checkpoint_layers", False))
            log(f"[suite] {v} s{seed} trainable={sum(p.numel() for p in model.trainable_parameters())} lora={len(model.core.lora_names)}")
            rows.append(inkling_train_run(model, v, seed, budget_min * 60, tr, va, scales, ym.cuda(), ys.cuda(),
                                          bs=ic.get("batch_size", 8), lr=ic.get("lr", 3e-4), log=log))
            del model
            torch.cuda.empty_cache()
    append_jsonl(RES / "ledger.jsonl", dict(time_utc=utc(), kind="inkling_suite", suite=suite, rows=[r.get("run_id") for r in rows],
                                            load=rep))
    return rows


if __name__ == "__main__":
    import argparse

    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sprint.yaml")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--variants", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    train_suite("inkling-pair", cfg, budget_min=a.minutes,
                seeds=[int(s) for s in a.seeds.split(",")] if a.seeds else None,
                variants=a.variants.split(",") if a.variants else None)


# ---------------------------------------------------------------------------
# evaluation of trained Inkling runs (same metrics/rows as evaluate.py)
# ---------------------------------------------------------------------------

INKLING_EVAL_STRATA = ("test", "ood_fric_0.15", "ood_fric_0.9", "occ_1.0", "misspec_velfric", "heldout_cells", "paddle")


def evaluate_inkling(config, runs=None, strata=INKLING_EVAL_STRATA, max_roots=None, diagnostics=True, tag="main"):
    """runs: list of (variant, seed). Loads base once, evaluates best-val adapters.
    Diagnostics on `test`: core bypass (constant readout message) and LoRA disabled
    (zero-shot lesion of the adapted core attention, not a retrained condition)."""
    import glob as _glob
    from pathlib import Path as _P

    import numpy as np

    from .. import evaluate as E
    from ..models.losses import fit_loss_scales
    from ..train import RES, config_hash, fit_target_scaler, load_dataset

    tm, tok, full, rep = load_inkling_text(config.get("paths", {}).get("checkpoint", "ckpt/Inkling-2-Mini"))
    prefix = prefix_ids_from_tokenizer(tok).cuda()
    tr = load_dataset("train", config, "cuda")
    y_mean, y_std = [x.numpy().astype(np.float64) for x in fit_target_scaler(tr)]
    scales = fit_loss_scales(tr.iterate(1024))
    del tr
    if runs is None:
        runs = []
        for p in sorted(_glob.glob(str(RES / "checkpoints" / f"inkling-{tag}-i*-s*.pt"))):
            stem = _P(p).stem
            if "_" in stem:
                continue
            parts = stem.split("-")
            runs.append((parts[-2], int(parts[-1][1:])))
    W = E.Writer(config.get("stats", {}).get("bootstrap_resamples", 2000),
                 config_hash(dict(suite="inkling", strata=list(strata), version=E.EVAL_VERSION)))
    lat = {}
    for variant, seed in runs:
        path = RES / "checkpoints" / f"inkling-{tag}-{variant}-s{seed}.pt"
        model = build_inkling_model(variant, tm, prefix, seed=seed)
        ck = load_trainable(model, path)
        model.eval()
        mname = f"inkling-{variant}"
        conds = [("", None)]
        if diagnostics:
            conds += [("bypass:", "bypass"), ("nolora:", "nolora")]
        for st in strata:
            try:
                ds = load_dataset(st, config, "cuda")
            except Exception as e:  # noqa: BLE001
                print(f"[ink-eval] skip {st}: {e!r}")
                continue
            idx = None
            if max_roots:
                roots = ds.roots
                keep = np.unique(roots)[:max_roots]
                idx = np.nonzero(np.isin(roots, keep))[0]
            lab = E.labels_np(ds, idx)
            for prefix_name, cond in conds:
                if cond is not None and st != "test":
                    continue
                model.core.bypass = cond == "bypass"
                set_lora_enabled(tm, cond != "nolora")
                t0 = time.time()
                pred = E.predict(model, ds, bs=32, idx=idx, bf16=True)
                dt = time.time() - t0
                met = E.sample_metrics(pred, lab, y_mean, y_std, scales)
                W.metric_rows("e08" if cond is None else "e08_diag", mname + (f"[{cond}]" if cond else ""), variant, seed, st,
                              met, lab["root"], extra_model="actual_inkling_core", ckpt_step=ck.get("step"))
                if cond is None and st == "test":
                    lat[(variant, seed)] = dt / max(1, len(lab["root"]))
                print(f"[ink-eval] {mname} s{seed} {st} {cond or 'normal'} pos@0.5={np.nanmean(met['pos_err_m'][:, 2]):.4f} t={dt:.0f}s", flush=True)
            model.core.bypass = False
            set_lora_enabled(tm, True)
        # e07-style typed-slot intervention for i11 (prediction sensitivity)
        if variant.startswith("i11") and diagnostics:
            ds = load_dataset("test", config, "cuda")
            outs = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for fr in (0.35, 0.65):
                    accum = []
                    for ev, q, lab_t in ds.iterate(32, idx=torch.arange(min(512, ds.n), device=ds.device)):
                        tokens, mask, slots = model.encode(ev, q)
                        B = tokens.shape[0]
                        t2, s2 = model.intervene(tokens, mask, slots, Prop_FRICTION(), torch.full((B,), fr, device=tokens.device),
                                                 torch.full((B,), -6.0, device=tokens.device))
                        o = model.forward_from_tokens(t2, mask, s2, q)
                        accum.append(o.mean[:, :, :2].float().cpu())
                    outs[fr] = torch.cat(accum)
            d = (outs[0.35] - outs[0.65]).norm(dim=-1)  # (n,H)
            for hi, h in enumerate((0.1, 0.25, 0.5, 1.0, 1.5, 2.0)):
                W.row("e07_inkling", mname, variant, seed, "test", f"pred_disp_change_fr0.35_vs_0.65@h{h}", float(d[:, hi].mean()),
                      len(np.unique(ds.roots[: d.shape[0]])))
        del model
        torch.cuda.empty_cache()
    for (v, s), x in lat.items():
        W.row("e08", f"inkling-{v}", v, s, "test", "latency_s_per_sample_bs32", x, 0)
    return W.rows


def Prop_FRICTION():
    from ..schema import Prop

    return Prop.FRICTION
