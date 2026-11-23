"""Render a still per family (reset + end of a scripted episode) into results/v2/envs/ + a contact sheet.
    python scripts/envs_render.py [families|all] [seed] [robot]"""
import os
import sys

import imageio.v2 as iio
import numpy as np

from physmem.envs import FAMILIES, family_class, make_env
from physmem.envs.splits import split_params
from physmem.v2.api import TaskSpec

fams = list(FAMILIES) if len(sys.argv) < 2 or sys.argv[1] == "all" else sys.argv[1].split(",")
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
robot = sys.argv[3] if len(sys.argv) > 3 else "probe"
os.makedirs("results/v2/envs", exist_ok=True)
tiles = []
for fam in fams:
    env = make_env(fam)
    env.reset(TaskSpec(fam, family_class(fam).category, seed, robot, "train", split_params(fam, "train")))
    a = env.render(width=640, height=480)
    succ = False
    for _ in range(40):
        c = env.scripted()
        if c is None:
            break
        out = env.execute(c)
        if out.done:
            break
    b = env.render(width=640, height=480)
    succ = env.success()
    iio.imwrite(f"results/v2/envs/{fam}_{robot}_reset.png", a)
    iio.imwrite(f"results/v2/envs/{fam}_{robot}_end.png", b)
    tiles.append(np.concatenate([a, b], 1)[::2, ::2])
    print(fam, "success", succ)
rows = [np.concatenate(tiles[i:i + 2], 1) if i + 1 < len(tiles) else np.concatenate([tiles[i], np.zeros_like(tiles[i])], 1)
        for i in range(0, len(tiles), 2)]
iio.imwrite(f"results/v2/envs/contact_sheet_{robot}.png", np.concatenate(rows, 0))
print("wrote contact sheet")
