"""System I / System II on the pretrained Inkling-2-Mini looped core (owner: INKLING agent). See docs/INKLING_SYS12.md.

Mode tokens
  <sys2>  System II = ordinary language deliberation by the UNMODIFIED base LM (all adapters gated off). <sys2> is a
          router-level control token: it switches the adapters off and is NOT fed to the LM (an untrained embedding
          in the stream would change the base computation), so System II is the base LM bit for bit.
  <sys1>  System I = fast physical-memory mode. The router appends a trained <sys1> embedding, one ACTION token (the
          structured skill choice System II emitted: skill one-hot) and R readout tokens to the System II context and
          runs a REDUCED number of core passes (default 1 of the native 3; post-loop layers skipped). Only in this mode
          are the typed-memory cross-attention (read at pass start / write at pass end, zero-init tanh gates) and the
          core LoRA active, and the LoRA is additionally POSITION-gated: it only acts on positions from <sys1> onward,
          so every System II prefix position keeps its base hidden state (KV reuse) even inside a System I call.
Handoff: System II emits text `<sys1> skill=<skill> target=<entity>`; `parse_handoff` extracts the structured choice;
System I fills in the continuous params (grip force, lift speed, probe-or-not) with uncertainty.

System I readout -> a context vector ctx (d=128) that replaces MODEL-V2's scene-encoder context; the MODEL-V2 outcome
head (FiLM, physics margins, typed-slot values) maps (ctx, candidate call, slot sample) -> outcome / event / effort
distributions, so every MODEL-V2 planner (LotPlanner, ResidualPlanner, CEM) runs on top unchanged. The typed memory
front end (segment sysID, precision fusion, scene entity encoder) is the FROZEN r7 MODEL-V2 typed s0 memory, identical
for every arm (Inkling pretrained / random-init / compact), so arms differ only in the decision path.

LoRA modes (GatedLoRALinear.mode): 'off' (System II / base), 'gated' (System I, positions >= <sys1>), 'always'
(the UNGATED baseline fine-tune: LoRA on at every position and in every mode -> it also changes System II).
"""
from __future__ import annotations

import math
import re
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .inkling import LORA_TARGETS, CoreHooks, LoRALinear, _XAttn, instrumented_forward
from .v2 import spec as S
from .v2.model import TF_LOGVAR, PhysMemV2, gauss_nll, mlp

SYS1, SYS2 = "<sys1>", "<sys2>"
L = S.N_LATENT

# actor-legal task descriptions (what an operator would type; no hidden quantities)
FAMILY_TEXT = {
    "bin_clearing": "move every part from the source bin into the tote without damaging it",
    "pouch_transfer": "pick the soft pouch and place it in the tote without bursting it",
    "tote_transfer": "transfer the items from one tote to the other",
    "conveyor_sort": "pick parts off the conveyor and sort them into the bins",
    "depalletize": "lift the cartons off the pallet and deliver them, flag the overweight ones",
    "singulation": "push the parts apart so each one can be picked alone",
    "door_open": "open the cabinet door",
    "drawer_open": "pull the drawer open",
    "fixture_place": "place the part into the fixture",
    "tool_return": "return the tool to its holder",
    "peg_insert": "insert the peg into the hole",
    "small_peg_insert": "insert the small peg into the tight hole",
    "connector_insert": "plug the connector into its socket",
    "twist_lock": "insert and twist the part to lock it",
    "kitting_pocket": "place each part into its kitting pocket",
    "machine_tending": "load the part into the machine and unload it",
    "shelf_restock": "put the items onto the shelf",
    "small_parts_pick": "pick the small parts and place them in the tray",
    "bulk_sweep": "sweep the bulk material into the target area",
}


def sys1_prefix_text(family: str) -> str:
    return ("<|message_system|><|content_text|>You control an industrial robot cell. Task: "
            f"{FAMILY_TEXT.get(family, family.replace('_', ' '))}.<|end_message|>")


HANDOFF_RE = re.compile(r"<sys1>\s*skill\s*=\s*([A-Za-z_]+)\s+target\s*=\s*([A-Za-z0-9_\-/\.]+)")


def parse_handoff(text: str):
    """System II text -> (skill, target) or None. The FIRST well-formed <sys1> call wins."""
    m = HANDOFF_RE.search(text or "")
    if not m:
        return None
    sk = m.group(1).lower()
    return (sk, m.group(2)) if sk in S.SKILLS else None


def format_handoff(skill: str, target: str) -> str:
    return f"{SYS1} skill={skill} target={target}"


# ------------------------------------------------------------------------------------------- gated LoRA
class GatedLoRALinear(LoRALinear):
    mode = "off"          # class default; set per instance through set_lora_mode
    gate = None           # (B, L) float/bool position gate (gated mode); None -> all positions

    def forward(self, x):
        y = self.base(x)
        if self.mode == "off":
            return y
        d = F.linear(F.linear(x.float(), self.A), self.B) * self.scale
        if self.mode == "gated":
            g = self.gate
            if g is None:
                return y
            g = g.to(d.dtype)
            if d.dim() == 3 and g.shape[:2] == d.shape[:2]:
                d = d * g[..., None]
            elif d.dim() == 2 and g.numel() == d.shape[0]:
                d = d * g.reshape(-1, 1)
            else:
                raise RuntimeError(f"LoRA gate shape {tuple(g.shape)} vs activation {tuple(d.shape)}")
        return y + d.to(y.dtype)


def inject_gated_lora(tm, layer_ids, r=8, alpha=32.0, targets=LORA_TARGETS, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mods = []
    for li in layer_ids:
        layer = tm.layers[li]
        for t in targets:
            parent_path, attr = t.rsplit(".", 1)
            parent = layer.get_submodule(parent_path)
            base = getattr(parent, attr)
            if isinstance(base, LoRALinear):
                base = base.base
            m = GatedLoRALinear(base, r, alpha)
            with torch.no_grad():
                bound = 1.0 / math.sqrt(m.A.shape[1])
                m.A.copy_((torch.rand(m.A.shape, generator=g) * 2 - 1).mul_(bound).to(m.A.device))
                m.B.zero_()
            setattr(parent, attr, m)
            mods.append(m)
    return mods


def remove_lora(tm):
    for layer in tm.layers:
        for t in LORA_TARGETS:
            parent_path, attr = t.rsplit(".", 1)
            try:
                parent = layer.get_submodule(parent_path)
            except AttributeError:
                continue
            mod = getattr(parent, attr, None)
            if isinstance(mod, LoRALinear):
                setattr(parent, attr, mod.base)


def gated_lora_modules(tm):
    return [m for m in tm.modules() if isinstance(m, GatedLoRALinear)]


def set_lora_mode(tm, mode: str, gate=None):
    assert mode in ("off", "gated", "always")
    for m in gated_lora_modules(tm):
        m.mode = mode
        m.gate = gate


# ------------------------------------------------------------------------------------------- System I core
class Sys1Core(nn.Module):
    """Inkling core in System I mode. memory tokens (B,M,d) + action features -> ctx (B,d) (+ per-pass records).

    lora_mode_train: 'gated' (main) or 'always' (ungated baseline). passes: System I core passes (<= native 3)."""

    def __init__(self, tm, tok, d_mem=128, act_dim=None, n_readout=16, passes=1, lora_rank=8, lora_alpha=32.0,
                 lora_mode="gated", checkpoint_layers=True, seed=0, fast_experts=True):
        super().__init__()
        self.tm = tm
        for p in tm.parameters():
            p.requires_grad_(False)
        H = tm.config.hidden_size
        self.H, self.d, self.R, self.passes = H, d_mem, n_readout, passes
        self.native_passes = tm.config.loop_num_passes
        dev = tm.embed_tokens.weight.device
        self.dev = dev
        self.tok = tok
        self._prefix_cache = {}
        with torch.no_grad():
            ids = torch.tensor(tok(sys1_prefix_text("bin_clearing"), add_special_tokens=False)["input_ids"], device=dev)
            emb = tm.embed_tokens(ids).float()
        scale = float(emb.pow(2).mean().sqrt())
        self.emb_scale = scale
        g = torch.Generator(device="cpu").manual_seed(seed)
        rn = lambda *s: (torch.randn(*s, generator=g) * scale).to(dev)
        # mode token embeddings (new rows; <sys2> exists for the tokenizer / router, never fed to the LM)
        self.mode_embed = nn.Parameter(rn(2, H))                   # [<sys1>, <sys2>]
        self.readout_embed = nn.Parameter(rn(n_readout, H))
        act_dim = act_dim or len(S.SKILLS)
        self.act_proj = nn.Linear(act_dim, H).to(dev)
        with torch.no_grad():
            self.act_proj.weight.copy_((torch.randn(H, act_dim, generator=g) * scale).to(dev))
            self.act_proj.bias.zero_()
        self.read = _XAttn(H, d_mem, H).to(dev)
        self.write = _XAttn(d_mem, H, d_mem).to(dev)
        self.read_gate = nn.Parameter(torch.zeros(self.native_passes, device=dev))
        self.write_gate = nn.Parameter(torch.zeros(self.native_passes, device=dev))
        self.out_norm = nn.LayerNorm(H).to(dev)
        self.out_proj = nn.Linear(H, d_mem).to(dev)
        self.ctx_proj = mlp(2 * d_mem, 2 * d_mem, d_mem).to(dev)
        self.core_layer_ids = list(range(tm.config.loop_layer_index_start, tm.config.loop_layer_index_end))
        remove_lora(tm)
        self.lora = inject_gated_lora(tm, self.core_layer_ids, lora_rank, lora_alpha, seed=seed) if lora_rank > 0 else []
        self.lora_mode = lora_mode
        self.checkpoint_layers = checkpoint_layers
        self.fast_experts = fast_experts
        self.last_info = {}

    def trainable_parameters(self):
        own = [p for n, p in self.named_parameters() if not n.startswith("tm.")]
        return own + [p for m in self.lora for p in (m.A, m.B)]

    def disarm(self):
        """back to System II: LoRA off, exact native expert loop."""
        from .inkling_quant import QuantExperts
        set_lora_mode(self.tm, "off")
        QuantExperts.FAST = False

    def adapter_state(self):
        sd = {n: p.detach().cpu() for n, p in self.named_parameters() if not n.startswith("tm.")}
        for i, m in enumerate(self.lora):
            sd[f"lora.{i}.A"], sd[f"lora.{i}.B"] = m.A.detach().cpu(), m.B.detach().cpu()
        return sd

    def load_adapter_state(self, sd):
        own = dict(self.named_parameters())
        with torch.no_grad():
            for n, v in sd.items():
                if n.startswith("lora."):
                    _, i, w = n.split(".")
                    getattr(self.lora[int(i)], w).copy_(v)
                else:
                    own[n].copy_(v)

    def prefix_ids(self, family):
        if family not in self._prefix_cache:
            self._prefix_cache[family] = torch.tensor(
                self.tok(sys1_prefix_text(family), add_special_tokens=False)["input_ids"], device=self.dev)
        return self._prefix_cache[family]

    def build_inputs(self, B, family, act, prefix_ids=None):
        """[System II context tokens | <sys1> | action | R readout] -> (embeds (B,Lx,H), s0 = index of <sys1>)."""
        ids = self.prefix_ids(family) if prefix_ids is None else prefix_ids
        dt = self.tm.embed_tokens.weight.dtype
        pre = self.tm.embed_tokens(ids)[None].expand(B, -1, -1)
        m1 = self.mode_embed[0].to(dt)[None, None].expand(B, 1, -1)
        a = self.act_proj(act.float()).to(dt)[:, None]
        ro = self.readout_embed.to(dt)[None].expand(B, -1, -1)
        return torch.cat([pre, m1, a, ro], 1), pre.shape[1]

    def forward(self, mem_tokens, mem_mask, act, family, passes=None, prefix_ids=None, keep_records=False):
        B = mem_tokens.shape[0]
        x, s0 = self.build_inputs(B, family, act, prefix_ids)
        Lx = x.shape[1]
        r0 = Lx - self.R
        gate = torch.zeros(B, Lx, device=x.device)
        gate[:, s0:] = 1.0
        set_lora_mode(self.tm, self.lora_mode, gate if self.lora_mode == "gated" else None)
        mem = [mem_tokens.float()]

        def pass_start(p, hidden):
            g = torch.tanh(self.read_gate[p])
            upd = self.read(hidden[:, s0:].float(), mem[-1], mem_mask) * g
            pad = torch.zeros(B, s0, self.H, device=hidden.device, dtype=upd.dtype)
            return (hidden.float() + torch.cat([pad, upd], 1)).to(hidden.dtype)

        def pass_end(p, hidden):
            g = torch.tanh(self.write_gate[p])
            mem.append(mem[-1] + g * self.write(mem[-1], hidden[:, s0:].float()))

        hooks = CoreHooks(pass_start, pass_end, keep_records=keep_records)
        from .inkling_quant import QuantExperts
        QuantExperts.FAST = self.fast_experts
        grad = torch.is_grad_enabled()
        try:
            h, info = instrumented_forward(self.tm, x, hooks, stop_after_core=True,
                                           num_passes=passes or self.passes,
                                           checkpoint_layers=self.checkpoint_layers and grad)
        finally:
            if not grad:
                self.disarm()
        # with grad enabled the LoRA mode / gate and the FAST flag stay ARMED until disarm() (after backward):
        # checkpointed layers are recomputed during backward and must see the same System I configuration.
        ro = self.out_proj(self.out_norm(h[:, r0:].float())).mean(1)
        ac = self.out_proj(self.out_norm(h[:, s0 + 1].float()))
        ctx = self.ctx_proj(torch.cat([ro, ac], -1))
        self.last_info = dict(info, seq_len=Lx, s0=s0, records=hooks.record if keep_records else None)
        return ctx


class CompactSys1Core(nn.Module):
    """compact control (MODEL-V2-style, no LM): 2-layer d=128 transformer over [action token | memory tokens]."""

    def __init__(self, d_mem=128, act_dim=None, layers=2):
        super().__init__()
        act_dim = act_dim or len(S.SKILLS)
        self.act = mlp(act_dim, d_mem, d_mem)
        self.fam = nn.Embedding(64, d_mem)
        el = nn.TransformerEncoderLayer(d_mem, 4, 2 * d_mem, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(el, layers)
        self.out = mlp(2 * d_mem, 2 * d_mem, d_mem)
        self.passes, self.last_info = 0, {}

    def trainable_parameters(self):
        return list(self.parameters())

    def disarm(self):
        pass

    def adapter_state(self):
        return {n: p.detach().cpu() for n, p in self.state_dict().items()}

    def load_adapter_state(self, sd):
        self.load_state_dict(sd)

    def forward(self, mem_tokens, mem_mask, act, family, passes=None, prefix_ids=None, keep_records=False):
        from .v2.tensorize import family_id
        a = self.act(act.float()) + self.fam.weight[family_id(family) % 64][None]
        x = torch.cat([a[:, None], mem_tokens.float()], 1)
        m = torch.cat([torch.ones_like(mem_mask[:, :1]), mem_mask], 1)
        h = self.tf(x, src_key_padding_mask=~m)
        hm = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        return self.out(torch.cat([h[:, 0], hm], -1))


# ------------------------------------------------------------------------------------------- full System I model
class InklingSys1V2(PhysMemV2):
    """MODEL-V2-compatible typed world model whose context comes from System I (Inkling core or compact control).

    Memory front end (scene entity MLP + slot pool + embodiment MLP, segment sysID, prior, fusion) = frozen copy of a
    trained MODEL-V2 typed checkpoint. Trainable: memory-token adapters, the System I core adapters (or the compact
    transformer) and a freshly initialised outcome head."""

    def __init__(self, base: PhysMemV2, core: nn.Module, d=128):
        super().__init__("typed", d=d, phys=base.phys, n_skills=base.n_skills)
        sd = base.state_dict()
        self.load_state_dict(sd)                       # memory + scalers (+ head weights, re-initialised below)
        dev = next(base.parameters()).device
        self.to(dev)
        fresh = PhysMemV2("typed", d=d, phys=base.phys, n_skills=base.n_skills).to(dev)
        self.head.load_state_dict(fresh.head.state_dict())   # head trained from scratch in every arm
        del fresh
        self.scene.tf = None                           # MODEL-V2's context transformer is replaced by System I
        self.scene.out = None
        for n, p in self.named_parameters():
            p.requires_grad_(n.startswith("head."))
        self.type_emb = nn.Parameter(torch.randn(L, d, device=dev) * 0.1)
        self.typed_tok = mlp(3, d, d).to(dev)
        self.core = core
        self.family = "bin_clearing"     # System II context of the current call (set by the router / trainer)
        self.passes_override = None

    # the planners call scene() then memory(); scene() returns a placeholder ctx that memory() fills in place
    def scene_tokens(self, ent, ent_mask, target, emb):
        sc = self.scene
        B, N, _ = ent.shape
        cnt = torch.log1p(ent_mask.sum(1).float())[:, None, None].expand(-1, N, 1)
        e = sc.ent(torch.cat([ent, cnt], -1))
        tgt = e[torch.arange(B, device=e.device), target]
        slots = sc.pool(e, ent_mask)
        return torch.cat([tgt[:, None], sc.emb(emb)[:, None], slots], 1), tgt

    def memory_tokens(self, scene_tok, mem, seg_count):
        zt = torch.stack([mem["z_mu"], mem["z_logvar"], seg_count[:, None].expand(-1, L)], -1)   # (B,L,3)
        typed = self.typed_tok(zt) + self.type_emb[None]
        toks = torch.cat([scene_tok, typed], 1)
        return toks, torch.ones(toks.shape[:2], dtype=torch.bool, device=toks.device)

    def system1(self, b, act=None, family=None, passes=None):
        """b normalized V2Batch -> (ctx (B,d), mem dict, tgt). act: (B, n_skill) skill one-hot of the System II
        choice (default: the skill columns of b.call)."""
        with torch.no_grad():
            scene_tok, tgt = self.scene_tokens(b.ent, b.ent_mask, b.target, b.emb)
            mem = PhysMemV2.memory(self, b, tgt)
            n = torch.log1p((b.seg_mask & b.seg_on_target).float().sum(1))
        toks, tmask = self.memory_tokens(scene_tok, mem, n)
        if act is None:
            act = b.call[:, :self.n_skills]
        ctx = self.core(toks, tmask, act[:, :self.n_skills], family or self.family,
                        passes=passes or self.passes_override)
        return ctx, mem, tgt

    def forward(self, b, z_override=None, z_mask_override=None, return_mem=False, family=None):
        b = self.norm_batch(b)
        if z_override is not None:
            raise NotImplementedError("interventions: override mem then call head_eval (see sys12_train.interv)")
        ctx, mem, _ = self.system1(b, family=family)
        o = self.head_eval(ctx, b.call_raw, mem["zin"], b.emb, b.ent, b.ent_mask)
        o["ctx"], o["mem"] = ctx, mem
        return o


def sys1_loss(model: InklingSys1V2, b, lab, family, tf_prob=0.5):
    """MODEL-V2 head losses (same as v2_loss without the sysID terms: the memory is frozen)."""
    b = model.norm_batch(b)
    ctx, mem, _ = model.system1(b, family=family)
    z_mu, z_lv = mem["z_mu"].detach(), mem["z_logvar"].detach()
    B = z_mu.shape[0]
    zm = lab.z_mask
    tf = (torch.rand(B, 1, device=z_mu.device) < tf_prob) & zm
    z_s = z_mu + torch.exp(0.5 * z_lv) * torch.randn_like(z_mu)
    zin = model.slot_input(torch.where(tf, lab.z, z_s), torch.full_like(z_lv, TF_LOGVAR))
    o = model.head_eval(ctx, b.call_raw, zin, b.emb, b.ent, b.ent_mask)
    yc = (lab.cont - model.cont_mean) / model.cont_std
    l_cont = gauss_nll(o["cont_mu"], o["cont_logvar"], yc, lab.cont_mask)
    em = lab.events_mask.float()
    l_ev = (F.binary_cross_entropy_with_logits(o["ev_logits"], lab.events.float(), reduction="none") * em).sum() / \
        em.sum().clamp(min=1)
    ye = (lab.effort - model.eff_mean) / model.eff_std
    l_eff = gauss_nll(o["eff_mu"], o["eff_logvar"], ye, torch.ones_like(ye, dtype=torch.bool))
    loss = l_cont + l_ev + 0.5 * l_eff
    logs = dict(l_cont=float(l_cont), l_ev=float(l_ev), l_eff=float(l_eff))
    if "ev_res" in o:
        l_res = 0.01 * ((o["ev_res"] ** 2).mean() + (o["cont_res"] ** 2).mean())
        loss = loss + l_res
    logs["loss"] = float(loss)
    return loss, logs


# ------------------------------------------------------------------------------------------- planner plumbing
def sys1_planner(model: InklingSys1V2, cfg=None):
    """a MODEL-V2 CEMPlanner whose _prepare runs System I once per (decision, query) and caches the context."""
    from physmem.planner import CEMPlanner, PlannerConfig
    from physmem.models.v2.tensorize import single_batch

    class Sys1Planner(CEMPlanner):
        def _prepare(self, history, frame, emb, query_target=None):
            m = self.model
            import numpy as np
            key = (len(history), id(history[-1]) if history else None, query_target, getattr(self, "skill", "pick"),
                   id(frame))
            c = getattr(self, "_cache", None)
            if c is not None and c[0] == key:          # System1Head.readout re-asks the same memory: reuse it
                self._scene = c[2]
                return c[1]
            with torch.no_grad():
                b = single_batch(history, frame, emb, np.zeros((1, S.CALLN_DIM), np.float32), device=self.dev,
                                 query_target=query_target)
                b = m.norm_batch(b)
                act = torch.zeros(1, m.n_skills, device=self.dev)
                act[0, S.SKILLS.index(getattr(self, "skill", "pick"))] = 1.0
                ctx, mem, _ = m.system1(b, act=act)
            self._scene = (b.emb, b.ent, b.ent_mask)
            self.n_sys1_calls = getattr(self, "n_sys1_calls", 0) + 1
            self._cache = (key, (ctx, mem), self._scene)
            return ctx, mem

    return Sys1Planner(model, "typed", cfg or PlannerConfig(n_z=16, voi=False))


# ------------------------------------------------------------------------------------------- System II router
SYS2_INSTRUCTION = ("You are the task planner (System II) of an industrial robot cell. Read the task and the current "
                    "cell state, then choose the single next skill call. Answer with exactly one line of the form "
                    "`<sys1> skill=<skill> target=<entity>` and nothing else. A fast controller (System I) then "
                    "chooses the grip force, speeds and whether to probe first.")


def sys2_messages(task_text: str, state_lines, options):
    """chat messages for one System II decision. options: list of (skill, target) the cell currently allows."""
    opts = "\n".join(f"- {format_handoff(s, t)}" for s, t in options)
    state = "\n".join(f"- {l}" for l in state_lines) if state_lines else "- (no observations yet)"
    user = (f"Task: {task_text}.\nCell state:\n{state}\nAllowed next calls:\n{opts}\n"
            "Which call comes next?")
    return [{"role": "system", "content": SYS2_INSTRUCTION}, {"role": "user", "content": user}]


class Sys2Router:
    """System II = the base LM, adapters OFF (exact native expert loop). Two decision modes:
    choose(): constrained - log-likelihood of each allowed handoff line, argmax (robust; used in closed loop);
    generate(): free-form greedy decode + parse_handoff (format-following eval)."""

    def __init__(self, model, tm, tok, core=None):
        self.model, self.tm, self.tok, self.core = model, tm, tok, core

    def prompt_ids(self, messages):
        txt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                           reasoning_effort="none")
        txt = txt + "<|content_text|>"          # answer channel directly (no thinking span)
        return torch.tensor(self.tok(txt, add_special_tokens=False)["input_ids"], device=self.tm.embed_tokens.weight.device)

    def _sys2(self):
        if self.core is not None:
            self.core.disarm()
        else:
            set_lora_mode(self.tm, "off")

    @torch.no_grad()
    def choose(self, messages, options, sys2_mode=True):
        """-> (index, logprobs list). sys2_mode=False lets the caller keep an ungated LoRA on (control)."""
        if sys2_mode:
            self._sys2()
        p = self.prompt_ids(messages)
        outs = [torch.tensor(self.tok(format_handoff(s, t), add_special_tokens=False)["input_ids"], device=p.device)
                for s, t in options]
        Lm = max(len(o) for o in outs)
        ids = torch.full((len(outs), len(p) + Lm), int(p[-1]), dtype=torch.long, device=p.device)
        for i, o in enumerate(outs):
            ids[i, :len(p)] = p
            ids[i, len(p):len(p) + len(o)] = o
        lp = torch.log_softmax(self.model(input_ids=ids).logits.float(), -1)
        scores = []
        for i, o in enumerate(outs):
            pos = torch.arange(len(p) - 1, len(p) - 1 + len(o), device=p.device)
            scores.append(float(lp[i, pos, o].sum()))
        return int(max(range(len(scores)), key=lambda i: scores[i])), scores

    @torch.no_grad()
    def generate(self, messages, max_new_tokens=32, sys2_mode=True):
        if sys2_mode:
            self._sys2()
        p = self.prompt_ids(messages)[None]
        g = self.model.generate(input_ids=p, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(g[0, p.shape[1]:], skip_special_tokens=False)
