"""Re-evaluate the MODEL-V2 references on EXACTLY the System I arms' row subsets. tensorize.family_id assigns ids in
first-load order per process and eval_subset seeds the row sample from it; the arms' processes load the TRAIN split
first, the first evalv2 run loaded only test stores (drawer_open shifts the ids) -> different rows for 11/16 test
families. Here FAMILY_IDS is pre-populated in train-split order (as in the arm runs), and per-family label rates are
asserted equal to the gated arm's eval before anything is written. Old outputs are kept as *_misaligned.json."""
import json
import os
import shutil
import sys

sys.path.insert(0, "src")
from physmem.models.v2 import tensorize as T  # noqa: E402
from physmem.sys12_train import DATA, HOLDOUT_FAMILY, RES, eval_v2  # noqa: E402

suf = "_train.npz"
fams = sorted({f[:-len(suf)] for f in os.listdir(DATA) if f.endswith(suf)})
fams = [f for f in fams if f != HOLDOUT_FAMILY]
T.FAMILY_IDS.clear()
for f in fams:
    T.family_id(f)
ref = json.load(open(os.path.join(RES, "eval", "s1r1_inkling_gated_s0.json")))
for tag, ck in (("v2_r7_typed_s0", "results/v2/checkpoints/v2_robots_r7_typed_s0.pt"),
                ("v2_matched_typed_s0", os.path.join(RES, "v2/checkpoints/v2_sys12_matched_typed_s0.pt"))):
    out = os.path.join(RES, "eval", f"{tag}.json")
    if os.path.exists(out) and not os.path.exists(out.replace(".json", "_misaligned.json")):
        shutil.copy(out, out.replace(".json", "_misaligned.json"))
    eval_v2(ck, tag)
    new = json.load(open(out))
    for sp in ("test", "heldout_family"):
        for f, r in ref[sp].items():
            for k in ("success_rate", "damage_rate", "slip_rate"):
                if k in r:
                    assert abs(new[sp][f][k] - r[k]) < 1e-9, (tag, sp, f, k, new[sp][f][k], r[k])
    print(tag, "row subsets verified identical to the System I arms", flush=True)
