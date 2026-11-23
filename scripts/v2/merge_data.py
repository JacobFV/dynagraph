"""MODEL-V2: merge converted v2 array dirs per (family, split). usage: merge_data.py <out> <dir1> <dir2> ..."""
import glob, json, os, sys
import numpy as np
from physmem.models.v2.tensorize import concat_arrays, arrays_hash
out, dirs = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
names = sorted({os.path.basename(f) for d in dirs for f in glob.glob(os.path.join(d, "*.npz"))})
man = {"_meta": {"sources": dirs}}
for n in names:
    parts = [dict(np.load(os.path.join(d, n))) for d in dirs if os.path.exists(os.path.join(d, n))]
    arr = concat_arrays(parts) if len(parts) > 1 else parts[0]
    np.savez(os.path.join(out, n), **arr)
    man[n] = dict(n=int(len(arr["k"])), hash=arrays_hash(arr), robots={r: int((arr["robot"] == r).sum()) for r in np.unique(arr["robot"])})
    print(n, man[n]["n"], man[n]["robots"])
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
