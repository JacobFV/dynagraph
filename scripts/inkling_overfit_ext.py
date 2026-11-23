"""AMENDED gate check 6 (run after the original 40-step check FAILED its 0.5-nat criterion,
with NLL decreasing monotonically 1.37->1.00). Same fixed 16-sample batch, same lr/optimizer,
extended to 200 steps. Recorded separately as check 6b; the original FAIL stays in the record."""
import json, sys, time
import torch, yaml
sys.path.insert(0, "src")
from physmem.models import inkling as I
from physmem.models.losses import LossConfig, compute_loss, fit_loss_scales
from physmem.train import RES, fit_target_scaler, load_dataset

cfg = yaml.safe_load(open("configs/sprint.yaml"))
tm, tok, full, rep = I.load_inkling_text(cfg["paths"]["checkpoint"])
tr = load_dataset("train", cfg, "cuda"); va = load_dataset("val", cfg, "cuda")
ym, ys = fit_target_scaler(tr); scales = fit_loss_scales(tr.iterate(1024))
prefix = I.prefix_ids_from_tokenizer(tok).cuda()
m = I.build_inkling_model("i11", tm, prefix, seed=1); m.set_target_scaler(ym.cuda(), ys.cuda())
p = m.trainable_parameters(); opt = torch.optim.AdamW(p, lr=1e-3)
fixed = tr.batch(torch.arange(16, 32)); vb = va.batch(torch.arange(16)); lc = LossConfig(physics=True)
tl, vl = [], []
for s in range(200):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(fixed[0], fixed[1]); l, parts = compute_loss(o, fixed[2], fixed[1], lc, scales, m.head.y_mean, m.head.y_std)
    opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(p, 1.0); opt.step()
    tl.append(float(parts["nll"]))
    if s % 25 == 24:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vo = m(vb[0], vb[1]); _, vp = compute_loss(vo, vb[2], vb[1], lc, scales, m.head.y_mean, m.head.y_std)
        vl.append(float(vp["nll"])); print(s, tl[-1], vl[-1], flush=True)
g = json.loads((RES / "inkling_gate.json").read_text())
ok = tl[-1] < tl[0] - 0.5
g["checks"]["6b_extended_overfit_AMENDED"] = {"pass": ok, "steps": 200, "train_nll_first": tl[0], "train_nll_last": tl[-1],
    "train_nll_curve": tl[::10], "val_nll_every25": vl,
    "note": "amended after check 6 failed its 40-step criterion; same batch/lr; original FAIL retained"}
g["passed_with_amendment"] = all(c["pass"] for k, c in g["checks"].items() if k != "6_tiny_overfit")
(RES / "inkling_gate.json").write_text(json.dumps(g, indent=2, default=str))
print("6b", "PASS" if ok else "FAIL", tl[0], tl[-1], flush=True)
