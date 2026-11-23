"""Actual-core decision gate (spec §4.2). Writes results/inkling_gate.json.

1 checkpoint/config/revision + finite native outputs (incl. a short greedy decode)
2 capture of the intended recurrent pass outputs (core exit after each pass)
3 instrumented == native forward with adapters disabled (and zero-gated adapters)
4 physical prediction loss -> finite nonzero grads in read/write adapters + LoRA through the core
5 optimizer step changes adapters; frozen base unchanged (version counters + checksums)
6 tiny fixed-batch overfit + separate validation behaviour (no generalization claim)
7 load peak, steady memory, warm/sustained step timing (fwd+bwd+opt)
8 projected complete-run budget
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, "src")
from physmem.models import inkling as I  # noqa: E402
from physmem.models.losses import LossConfig, compute_loss, fit_loss_scales  # noqa: E402
from physmem.train import RES, fit_target_scaler, load_dataset  # noqa: E402

OUT = RES / "inkling_gate.json"
gate = {"time_utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}}


def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gate, indent=2, default=str))


def check(name, ok, **info):
    gate["checks"][name] = {"pass": bool(ok), **info}
    print(f"[gate] {name}: {'PASS' if ok else 'FAIL'} {json.dumps(info, default=str)[:400]}", flush=True)
    save()


cfg = yaml.safe_load(open("configs/sprint.yaml"))
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
tm, tok, full, rep = I.load_inkling_text(cfg["paths"]["checkpoint"], drop_vision=False)
gate["load"] = rep
cfgjson = json.loads(Path(cfg["paths"]["checkpoint"], "config.json").read_text())

# ---- 1. native outputs: logits + greedy decode
msgs = [{"role": "user", "content": "In one short sentence: what happens to a box pushed across a rough table after the push stops?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", reasoning_effort="none")
if isinstance(ids, dict) or hasattr(ids, "keys"):
    ids = ids["input_ids"]
ids = ids.cuda()
gen_text = None
with torch.no_grad():
    t1 = time.time()
    try:
        out = full.generate(input_ids=ids, max_new_tokens=40, do_sample=False)
        gen_text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    except Exception as e:  # noqa: BLE001
        gen_text = f"generate failed: {e!r}"
    gen_s = time.time() - t1
    emb = tm.embed_tokens(ids)
    native = tm(inputs_embeds=emb).last_hidden_state
    logits = full.lm_head(native[:, -1:].to(full.lm_head.weight.dtype)).float()
check("1_checkpoint_native_outputs", torch.isfinite(native).all().item() and torch.isfinite(logits).all().item(),
      revision=I.INKLING_REVISION, arch=cfgjson["architectures"], quant=cfgjson.get("quantization_config"),
      loop=cfgjson["text_config"]["loop_config"], greedy_decode=gen_text, decode_s=round(gen_s, 1),
      param_dtypes=rep["dtypes"], text_params=rep["text_params"])
# free vision + lm_head
full.model.vision_tower = None
full.lm_head = None
torch.cuda.empty_cache()

# ---- 2+3. pass capture and native equivalence
with torch.no_grad():
    hooks = I.CoreHooks(keep_records=True)
    mine, _ = I.instrumented_forward(tm, emb, hooks)
    eq = torch.equal(mine, native)
    maxdiff = float((mine.float() - native.float()).abs().max())
    recs = hooks.record
    pass_diffs = [float((recs[i].float() - recs[i - 1].float()).norm() / recs[i - 1].float().norm()) for i in range(1, len(recs))]
check("2_pass_capture", len(recs) == tm.config.loop_num_passes and all(d > 0 for d in pass_diffs),
      n_passes=len(recs), core_layers=[tm.config.loop_layer_index_start, tm.config.loop_layer_index_end - 1],
      relative_change_between_passes=pass_diffs)
check("3a_instrumented_equals_native", eq, max_abs_diff=maxdiff)

# real data
tr = load_dataset("train", cfg, "cuda")
va = load_dataset("val", cfg, "cuda")
ym, ys = fit_target_scaler(tr)
scales = fit_loss_scales(tr.iterate(1024))
prefix = I.prefix_ids_from_tokenizer(tok).cuda()
model = I.build_inkling_model("i11", tm, prefix, seed=0)
model.set_target_scaler(ym.cuda(), ys.cuda())
ev, q, lab = tr.batch(torch.arange(8))
with torch.no_grad():
    B = q.action.shape[0]
    x = model.core.build_inputs(B, q.action)
    I.set_lora_enabled(tm, False); model.core.adapters_enabled = False
    ref, _ = I.instrumented_forward(tm, x, stop_after_core=True)
    I.set_lora_enabled(tm, True); model.core.adapters_enabled = True
    tokens, mask, slots = model.encode(ev, q)
    model.core(tokens, mask, q.action, keep_records=True)
    same = torch.equal(model.core.last_info["records"][-1], ref)
check("3b_zero_gated_adapters_preserve_core", same, lora_targets=I.LORA_TARGETS, n_lora=len(model.core.lora_names),
      seq_len=int(x.shape[1]))

# ---- 4. gradient path
lcfg = LossConfig(physics=True)
before = I.frozen_checksum(tm)
params = model.trainable_parameters()
opt = torch.optim.AdamW(params, lr=3e-4)
# step 0: zero gates -> only gates (and encoder via nothing) receive grad; take one step so the read path opens
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(ev, q)
    loss, parts = compute_loss(out, lab, q, lcfg, scales, model.head.y_mean, model.head.y_std)
opt.zero_grad(set_to_none=True)
loss.backward()
g_read_gate = model.core.read_gate.grad.detach().clone()
g_write_gate = model.core.write_gate.grad.detach().clone()
lora_A = [m.A.grad for m in I.lora_modules(tm)]
check("4a_gate_gradients_from_physical_loss", torch.isfinite(loss).item() and (g_read_gate.abs() > 0).all().item(),
      loss=float(loss), read_gate_grad=g_read_gate.tolist(), write_gate_grad=g_write_gate.tolist(),
      lora_grad_nonnull=sum(a is not None for a in lora_A), note="LoRA B=0 at init so A-grads are zero; B gets grads")
opt.step()
# second step: gates open -> read/write projections and LoRA A/B get grads through the core
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(ev, q)
    loss2, _ = compute_loss(out, lab, q, lcfg, scales, model.head.y_mean, model.head.y_std)
opt.zero_grad(set_to_none=True)
loss2.backward()
gr = {"read.k": float(model.core.read.k.weight.grad.abs().sum()), "read.q": float(model.core.read.q.weight.grad.abs().sum()),
      "write.q": float(model.core.write.q.weight.grad.abs().sum()),
      "lora_A_nonzero": sum(int(m.A.grad is not None and m.A.grad.abs().sum() > 0) for m in I.lora_modules(tm)),
      "lora_B_nonzero": sum(int(m.B.grad is not None and m.B.grad.abs().sum() > 0) for m in I.lora_modules(tm)),
      "n_lora": len(I.lora_modules(tm)),
      "base_grads_none": all(p.grad is None for n, p in tm.named_parameters() if not (n.endswith(".A") or n.endswith(".B")))}
finite = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
check("4b_adapter_gradients_through_core", finite and gr["read.k"] > 0 and gr["lora_B_nonzero"] == gr["n_lora"] and gr["base_grads_none"], **gr)

# pass-0 isolation: only pass-0 read gate open, LoRA off -> grads to read.k must flow through passes 1,2
with torch.no_grad():
    saved_rg = model.core.read_gate.detach().clone(); saved_wg = model.core.write_gate.detach().clone()
    model.core.read_gate.zero_(); model.core.read_gate[0] = 0.3; model.core.write_gate.zero_()
I.set_lora_enabled(tm, False)
opt.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(ev, q)
    l3, _ = compute_loss(out, lab, q, lcfg, scales, model.head.y_mean, model.head.y_std)
l3.backward()
check("4c_pass0_read_grad_via_later_passes", float(model.core.read.k.weight.grad.abs().sum()) > 0,
      read_k_grad=float(model.core.read.k.weight.grad.abs().sum()))
with torch.no_grad():
    model.core.read_gate.copy_(saved_rg); model.core.write_gate.copy_(saved_wg)
I.set_lora_enabled(tm, True)

# ---- 5. optimizer step changes adapters, frozen base unchanged
a0 = I.lora_modules(tm)[0].B.detach().clone()
r0 = model.core.read.k.weight.detach().clone()
opt.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(ev, q)
    l4, _ = compute_loss(out, lab, q, lcfg, scales, model.head.y_mean, model.head.y_std)
l4.backward()
opt.step()
after = I.frozen_checksum(tm)
check("5_step_changes_adapters_only", (not torch.equal(a0, I.lora_modules(tm)[0].B)) and (not torch.equal(r0, model.core.read.k.weight)) and after == before,
      frozen_tensors_checked=len(before), frozen_unchanged=after == before)

# ---- 6 + 7. overfit fixed batch + timing
fixed = tr.batch(torch.arange(16, 32))
vb = va.batch(torch.arange(16))
losses, vlosses, times = [], [], []
model2 = I.build_inkling_model("i11", tm, prefix, seed=1)
model2.set_target_scaler(ym.cuda(), ys.cuda())
p2 = model2.trainable_parameters()
opt2 = torch.optim.AdamW(p2, lr=1e-3)
torch.cuda.reset_peak_memory_stats()
mem_steps = []
for s in range(40):
    ts = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model2(fixed[0], fixed[1])
        l, parts = compute_loss(out, fixed[2], fixed[1], lcfg, scales, model2.head.y_mean, model2.head.y_std)
    opt2.zero_grad(set_to_none=True)
    l.backward()
    torch.nn.utils.clip_grad_norm_(p2, 1.0)
    opt2.step()
    torch.cuda.synchronize()
    times.append(time.time() - ts)
    losses.append(float(parts["nll"]))
    if s % 10 == 9:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vo = model2(vb[0], vb[1])
            _, vp = compute_loss(vo, vb[2], vb[1], lcfg, scales, model2.head.y_mean, model2.head.y_std)
        vlosses.append(float(vp["nll"]))
        mem_steps.append(I.mem_snapshot())
check("6_tiny_overfit", losses[-1] < losses[0] - 0.5, train_nll_first=losses[0], train_nll_last=losses[-1],
      train_nll_curve=losses[::5], val_nll_every10=vlosses, note="fixed 16-sample batch; no generalization claim")
warm = times[1]
sustained = sum(times[5:]) / len(times[5:])
peak = torch.cuda.max_memory_allocated() / 1e9
# batch-size scaling probe
bs_t = {}
for bsz in (8, 32):
    b = tr.batch(torch.arange(bsz))
    ts = time.time()
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model2(b[0], b[1])
            l, _ = compute_loss(out, b[2], b[1], lcfg, scales, model2.head.y_mean, model2.head.y_std)
        opt2.zero_grad(set_to_none=True); l.backward(); opt2.step()
    torch.cuda.synchronize()
    bs_t[bsz] = (time.time() - ts) / 3
check("7_timing_memory", True, load_s=rep["load_s"], mem_after_load=rep["mem_after_load"], mem_ready=rep["mem_ready"],
      first_step_s=times[0], warm_step_s=warm, sustained_step_s_bs16=sustained, step_s_by_bs=bs_t,
      peak_cuda_alloc_gb_train=peak, mem_during_train=mem_steps[-1] if mem_steps else None,
      trainable_params=sum(p.numel() for p in p2))

# ---- 8. budget
best_bs = max(bs_t, key=lambda k: k / bs_t[k])
run_min = cfg["inkling"].get("run_minutes", 30)
steps = int(run_min * 60 * 0.85 / bs_t[best_bs])
check("8_budget", steps * best_bs >= 2000, chosen_bs=best_bs, step_s=bs_t[best_bs], run_minutes=run_min,
      projected_steps_per_run=steps, projected_examples_per_run=steps * best_bs,
      note="pass if a run can see >=2000 training examples (~1/8 epoch); else downscope")
gate["passed"] = all(c["pass"] for c in gate["checks"].values())
gate["time_utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
gate["wall_s"] = time.time() - t0
save()
print("GATE", "PASSED" if gate["passed"] else "FAILED", flush=True)
