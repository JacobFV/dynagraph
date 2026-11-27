"""Deliverable 5: language retention. Conditions: base (no adapters installed), gated System I adapters installed but
System II mode (adapters off), ungated LoRA (always on). Metrics:
  ppl     : perplexity on held-out general text (wikitext-2-raw-v1 TEST, public; first N non-overlapping 512-token
            windows) + mean per-token KL(base || condition)
  parse   : task -> skill parsing on our industrial tasks (generated items with unique answers): free-form greedy
            decode -> parse_handoff (format validity, skill acc, target acc, exact) and constrained choice (loglik
            argmax over the allowed calls)
  instr   : small instruction-following set with automatic checkers
usage: python scripts/sys12/lang_eval.py --gated <ckpt> --ungated <ckpt> [--windows 48]"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
from physmem.models.inkling_quant import load_inkling_quant  # noqa: E402
from physmem.models.sys12 import (FAMILY_TEXT, Sys1Core, Sys2Router, parse_handoff, set_lora_mode,  # noqa: E402
                                  sys2_messages)

# ------------------------------------------------------------------------------------------- items
def parse_items(n=60, seed=0):
    """(task_text, state_lines, options, answer). Every item has exactly one correct call stated by the state."""
    rng = np.random.default_rng(seed)
    items = []
    while len(items) < n:
        kind = rng.integers(6)
        if kind == 0:      # bin clearing, holding a part -> place
            p = f"part{rng.integers(0, 6):02d}"
            st = [f"holding {p} in the gripper", "tote: empty space available"]
            opts = [("place", "tote"), ("pick", f"part{rng.integers(6, 9):02d}"), ("reach", "tote")]
            items.append(("bin_clearing", st, opts, ("place", "tote")))
        elif kind == 1:    # bin clearing, gripper empty, one part flagged accessible on top
            ps = [f"part{i:02d}" for i in rng.choice(8, 3, replace=False)]
            top = ps[int(rng.integers(3))]
            st = ["gripper empty"] + [f"{q} in the source bin" + (" (on top, most accessible)" if q == top else
                                                                    " (covered by other parts)") for q in ps]
            opts = [("pick", q) for q in ps] + [("place", "tote")]
            items.append(("bin_clearing", st, opts, ("pick", top)))
        elif kind == 2:    # pouch
            hold = bool(rng.integers(2))
            st = ["holding the pouch" if hold else "gripper empty", "pouch " + ("lifted" if hold else "on the bench")]
            opts = [("pick", "pouch"), ("place", "tote")]
            items.append(("pouch_transfer", st, opts, ("place", "tote") if hold else ("pick", "pouch")))
        elif kind == 3:    # door / drawer
            fam = ["door_open", "drawer_open"][int(rng.integers(2))]
            tgt = "door" if fam == "door_open" else "drawer"
            st = ["gripper empty", f"{tgt} closed, handle visible"]
            opts = [("open", tgt), ("close", tgt), ("push", tgt)]
            items.append((fam, st, opts, ("open", tgt)))
        elif kind == 4:    # peg insertion, holding peg
            st = ["holding peg0 aligned above hole0"]
            opts = [("insert", "hole0"), ("pick", "peg0"), ("place", "tray")]
            items.append(("peg_insert", st, opts, ("insert", "hole0")))
        else:              # depalletize: carton flagged overweight -> handover
            c = f"carton{rng.integers(0, 5)}"
            heavy = bool(rng.integers(2))
            st = [f"{c} hefted: " + ("too heavy for the gripper payload" if heavy else "within payload")]
            opts = [("handover", c), ("pick", c)]
            items.append(("depalletize", st, opts, ("handover", c) if heavy else ("pick", c)))
    out = []
    for fam, st, opts, ans in items:
        perm = list(rng.permutation(len(opts)))
        out.append(dict(family=fam, task=FAMILY_TEXT[fam], state=st, options=[opts[i] for i in perm], answer=ans))
    return out


INSTR = [
    ("What is 17 + 26? Reply with only the number.", lambda t: t.strip().rstrip(".") == "43"),
    ("What is 9 times 7? Reply with only the number.", lambda t: t.strip().rstrip(".") == "63"),
    ("Reply with the word ROBOT in uppercase and nothing else.", lambda t: t.strip().strip(".") == "ROBOT"),
    ("List exactly three primary colors, one per line, no other text.",
     lambda t: len([l for l in t.strip().splitlines() if l.strip()]) == 3),
    ('Return a JSON object with a single key "skill" whose value is "pick". No other text.',
     lambda t: _json_ok(t, {"skill": "pick"})),
    ("Answer yes or no only: is a gripper part of a robot arm?", lambda t: t.strip().lower().strip(".") == "yes"),
    ("Translate 'thank you' into French. Reply with only the translation.",
     lambda t: "merci" in t.strip().lower() and len(t.split()) <= 4),
    ("Write the word 'tote' backwards. Reply with only that.", lambda t: t.strip().strip(".").lower() == "etot"),
    ("What is the capital of Japan? One word.", lambda t: t.strip().strip(".").lower() == "tokyo"),
    ("Count the letters in the word 'gripper'. Reply with only the number.", lambda t: t.strip().rstrip(".") == "7"),
    ("Reply with a comma-separated list of the numbers 1 to 5 and nothing else.",
     lambda t: re.sub(r"\s", "", t.strip().rstrip(".")) == "1,2,3,4,5"),
    ("Is 0.3 larger than 0.25? Answer yes or no only.", lambda t: t.strip().lower().strip(".") == "yes"),
    ("Name the robot skill used to lift an object off a table. One word, lowercase.",
     lambda t: t.strip().strip(".").lower() in ("pick", "grasp", "lift")),
    ("Respond with exactly two words: the first is 'safe', the second is 'stop'.",
     lambda t: t.strip().strip(".").lower().split() == ["safe", "stop"]),
    ("What is 100 divided by 4? Reply with only the number.", lambda t: t.strip().rstrip(".") in ("25", "25.0")),
    ("Reply with the lowercase letter that comes after 'q' in the alphabet, nothing else.",
     lambda t: t.strip().strip(".").lower() == "r"),
]


def _json_ok(t, want):
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return m is not None and json.loads(m.group(0)) == want
    except Exception:  # noqa: BLE001
        return False


def clean(t):
    """final answer text: after the last <|content_text|> channel marker (a thinking span may precede it)."""
    if "<|content_text|>" in t:
        t = t.split("<|content_text|>")[-1]
    elif "<|content_thinking|>" in t:
        return ""                           # generation budget ended inside the thinking span: no answer
    return t.split("<|end_message|>")[0].strip().strip("`").strip()


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", default="")
    ap.add_argument("--ungated", default="")
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--n-parse", type=int, default=60)
    ap.add_argument("--out", default="results/sys12/lang_eval.json")
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--gated-subset", type=int, default=0, help="gated condition: only the first N parse/instr items")
    a = ap.parse_args()
    model, tm, tok, rep = load_inkling_quant("ckpt/Inkling-2-Mini")
    import pyarrow.parquet as pq
    text = "".join(pq.read_table("data/sys12_text/wikitext2_test.parquet").to_pydict()["text"])
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"][: a.windows * a.win])
    wins = ids.view(a.windows, a.win).cuda()
    items = parse_items(a.n_parse)
    router = Sys2Router(model, tm, tok)

    core = None
    conds = [("base", None, None)]
    if a.gated:
        conds.append(("gated_system2", a.gated, "off"))
    if a.ungated:
        conds.append(("ungated", a.ungated, "always"))
    res = dict(text=dict(source="Salesforce/wikitext wikitext-2-raw-v1 test (public)",
                         sha256="5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
                         windows=a.windows, win=a.win, tokens=int(wins.numel())), conditions={})
    base_lp = None
    for name, ck, mode in conds:
        t0 = time.time()
        if ck:
            c = torch.load(ck, map_location="cpu", weights_only=False)
            if core is None:
                core = Sys1Core(tm, tok, d_mem=128, act_dim=11, n_readout=16, passes=1, lora_mode="gated")
            core.load_adapter_state(c["state"]["core"])
        set_lora_mode(tm, mode or "off")
        # --- perplexity + KL to base
        nll, kls, store = [], [], []
        with torch.no_grad():
            for i in (range(0, a.windows, 4) if not a.no_ppl else []):
                x = wins[i:i + 4]
                lp = torch.log_softmax(model(input_ids=x).logits.float(), -1)
                tgt = x[:, 1:]
                nll.append(-lp[:, :-1].gather(-1, tgt[..., None])[..., 0].cpu())
                top = lp.topk(64, -1)          # KL on the base top-64 support (memory), renormalised
                if base_lp is None:
                    store.append((top.indices.cpu(), top.values.cpu()))
                else:
                    bi, bv = base_lp[i // 4]
                    bi, bv = bi.cuda(), bv.cuda()
                    pb = torch.softmax(bv, -1)
                    qc = torch.log_softmax(lp.gather(-1, bi), -1)
                    kls.append(float((pb * (torch.log_softmax(bv, -1) - qc)).sum(-1).mean()))
                del lp
        if base_lp is None:
            base_lp = store
        if nll:
            nll = torch.cat(nll).flatten()
            r = dict(ppl=float(torch.exp(nll.mean())), nll=float(nll.mean()),
                     kl_to_base_top64=(float(np.mean(kls)) if kls else 0.0), mode=mode or "no adapters")
        else:
            r = dict(mode=mode or "no adapters", gen_tokens=a.gen_tokens)
        sub = a.gated_subset if (name == "gated_system2" and a.gated_subset) else None
        its = items[:sub] if sub else items
        ins = INSTR[:max(1, sub // 4)] if sub else INSTR
        # --- parsing (free-form + constrained)
        sys2 = mode != "always"
        ok_fmt = ok_sk = ok_tg = ok_ex = ok_con = 0
        samples, outs_parse = [], []
        for it in its:
            msgs = sys2_messages(it["task"], it["state"], it["options"])
            set_lora_mode(tm, mode or "off")
            raw = router.generate(msgs, a.gen_tokens, sys2_mode=sys2)
            out = clean(raw)
            outs_parse.append(raw)
            ph = parse_handoff(out)
            ans = tuple(it["answer"])
            ok_fmt += ph is not None
            ok_sk += ph is not None and ph[0] == ans[0]
            ok_tg += ph is not None and ph[1] == ans[1]
            ok_ex += ph == ans
            set_lora_mode(tm, mode or "off")
            j, _ = router.choose(msgs, it["options"], sys2_mode=sys2)
            ok_con += tuple(it["options"][j]) == ans
            if len(samples) < 6:
                samples.append(dict(answer=ans, out=out))
        n = len(its)
        r["parse_raw"] = outs_parse
        r["parse"] = dict(n=n, format_valid=ok_fmt / n, skill_acc=ok_sk / n, target_acc=ok_tg / n, exact=ok_ex / n,
                          constrained_acc=ok_con / n, samples=samples)
        # --- instruction following
        ok, outs = 0, []
        for q, chk in ins:
            set_lora_mode(tm, mode or "off")
            o = clean(router.generate([{"role": "user", "content": q}], a.gen_tokens, sys2_mode=sys2))
            ok += bool(chk(o))
            outs.append(o)
        r["instr"] = dict(n=len(ins), acc=ok / len(ins), outputs=outs)
        r["seconds"] = round(time.time() - t0, 1)
        set_lora_mode(tm, "off")
        res["conditions"][name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k not in ("instr",)}, default=str)[:900], "instr", r["instr"]["acc"],
              flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
