"""MODEL-V2 r5: data dir for round r = base train + all on-policy rollouts so far (per family); val/test/heldout from base.
usage: merge_onpolicy.py <base_dir> <out_dir> <rollout_dir1> [rollout_dir2 ...]"""
import glob, os, sys
import numpy as np
from physmem.models.v2.tensorize import concat_arrays
base, out, rdirs = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(os.path.join(base, "*.npz"))):
    name = os.path.basename(f)
    dst = os.path.join(out, name)
    if not name.endswith("_train.npz"):
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f), dst)
        continue
    fam = name[:-len("_train.npz")]
    parts = [dict(np.load(f))] + [dict(np.load(p)) for d in rdirs for p in glob.glob(os.path.join(d, f"*__{fam}.npz"))]
    arr = concat_arrays(parts) if len(parts) > 1 else parts[0]
    np.savez(dst, **arr)
    print(name, "rows", len(arr["k"]), "rollout parts", len(parts) - 1)
