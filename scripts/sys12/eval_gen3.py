"""Offline held-out-ROBOT (Gen3) eval for every System I arm + MODEL-V2 references, with the v2 sanitize limit on
cont column 7 ('aux1') lifted and aux1 masked out of the continuous NLL. Reason: every Gen3 row of data/v2_r7_final
has |aux1| up to ~1.2e4 (> SANITY_LIMITS['cont'] = 1000, a gen3 unit/scale issue), so the standard loader drops the
whole split. Event AUC / Brier are unaffected by aux1. Same fixed row subsets for every model (<= 512 per family).
usage: python scripts/sys12/eval_gen3.py"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import physmem.v2_train as VT  # noqa: E402
from physmem.models.v2 import spec as S  # noqa: E402
from physmem.sys12_train import (CKD, DATA, RES, build, evaluate_model, load_state, pool)  # noqa: E402

AUX1 = S.OUT_CONT.index("aux1")
lim0 = dict(VT.SANITY_LIMITS)


def stores():
    orig = VT.sanitize

    def san(arr, name=""):
        a = dict(arr)
        a["cont"] = a["cont"].copy()
        a["cont"][:, AUX1] = 0.0           # lifted for the sanity check only; masked below
        out = orig(a, name)
        return out
    VT.sanitize = san
    st = VT.load_stores(DATA, "heldout_robot", torch.device("cuda"))
    VT.sanitize = orig
    for s in st.values():
        s.cont_mask[:, AUX1] = False
    return st


st = stores()
print({k: v.n for k, v in st.items()}, flush=True)
res = {}
for run in ("s1r1_inkling_gated_s0", "s1r1_inkling_ungated_s0", "s1r1_inkling_random_s0", "s1r1_compact_s0"):
    f = os.path.join(CKD, f"{run}.pt")
    if not os.path.exists(f):
        continue
    c = torch.load(f, map_location="cpu", weights_only=False)
    m, _ = build(c["info"]["arm"], c["info"]["passes"], c["info"]["seed"])
    load_state(m, c["state"])
    r = evaluate_model(m, {k: v for k, v in st.items() if k != "drawer_open"}, max_rows=512)
    res[run] = r
    print(run, json.dumps({k: round(v, 4) for k, v in pool(r).items()}), flush=True)
    del m
    torch.cuda.empty_cache()
for tag, ck in (("v2_r7_typed_s0", "results/v2/checkpoints/v2_robots_r7_typed_s0.pt"),
                ("v2_matched_typed_s0", os.path.join(RES, "v2/checkpoints/v2_sys12_matched_typed_s0.pt"))):
    if os.path.exists(ck):
        m, _ = VT.load_model(ck, torch.device("cuda"))
        r = evaluate_model(m, {k: v for k, v in st.items() if k != "drawer_open"}, max_rows=512, sys1=False)
        res[tag] = r
        print(tag, json.dumps({k: round(v, 4) for k, v in pool(r).items()}), flush=True)
json.dump(res, open(os.path.join(RES, "eval", "gen3_heldout_robot_aux1masked.json"), "w"), indent=1)
