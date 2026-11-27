"""budget-matched memory-model data (r9 regime i): the memory-model view of EXACTLY the il-v1 demonstration
episodes (train) and the VAL demo-policy episodes (val), per family (robots concatenated).
usage: r9_mem_data.py TRAIN_MEM_DIR VAL_MEM_DIR OUT_DIR"""
import glob, os, sys
import numpy as np
from physmem.models.v2.tensorize import concat_arrays
tr, va, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
fams = sorted({os.path.basename(f)[:-4].split("__")[1] for f in glob.glob(f"{tr}/*__*.npz")})
for fam in fams:
    for split, d in (("train", tr), ("val", va)):
        parts = [dict(np.load(f)) for f in sorted(glob.glob(f"{d}/*__{fam}.npz"))]
        if not parts:
            continue
        arr = concat_arrays(parts) if len(parts) > 1 else parts[0]
        np.savez(f"{out}/{fam}_{split}.npz", **arr)
        print(fam, split, "rows", len(arr["k"]), "episodes", len(np.unique(arr["ep_index"])), flush=True)
