"""split il-v1 demo files for the < 6 GB small-GPU-job cap: OUT/<cell>.npz = episodes WITHOUT images (successful +
failed, as recorded) with 'img_off' for successful episodes, and OUT/<cell>.img5.npy = the wrist images of the
SUCCESSFUL episodes at 5 Hz (every second 10 Hz tick; the trainer uses frame t//2). usage: il_split_img.py IN OUT"""
import glob, json, os, shutil, sys
import numpy as np
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(f"{src}/*__*.npz")):
    if f.endswith(".partial.npz"):
        continue
    cell = os.path.basename(f)[:-4]
    z = np.load(f, allow_pickle=True)["eps"]
    eps, imgs, off = [], [], 0
    for e in z:
        e = dict(e)
        im = e.pop("img", None)
        if e.get("success") and im is not None:
            sub = im[::2]
            e["img_off"] = off
            off += len(sub)
            imgs.append(sub)
        eps.append(e)
    del z
    np.savez_compressed(f"{out}/{cell}.npz", eps=np.array(eps, dtype=object))
    if imgs:
        np.save(f"{out}/{cell}.img5.npy", np.concatenate(imgs).astype(np.uint8))
    shutil.copy(f[:-4] + ".json", f"{out}/{cell}.json")
    print(cell, "episodes", len(eps), "img frames (5 Hz)", off, flush=True)
