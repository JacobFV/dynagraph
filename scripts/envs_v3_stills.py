"""envs-v3 stills: one scripted (oracle-parameter) episode per (family, robot), site-dressed; PNG frames at reset,
mid-episode and the end + a contact sheet. Output results/v2/envs/v3_stills/<family>_<robot>_{reset,mid,end}.png.

    python scripts/envs_v3_stills.py family:robot[,family:robot...] [seed]
"""
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")


def main():
    import imageio
    from physmem.envs import family_class, make_env
    from physmem.v2.api import TaskSpec
    from envs_ceiling import apply_policy
    out = "results/v2/envs/v3_stills"
    os.makedirs(out, exist_ok=True)
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1700003
    sheet = []
    for pair in sys.argv[1].split(","):
        fam, rob = pair.split(":")
        env = make_env(fam)
        env.reset(TaskSpec(fam, family_class(fam).category, seed, rob, "test", dict(dressing="site")))
        c = env.scene.camera
        lk = np.asarray(c.get("view_lookat", c["lookat"]), float)
        c["view_pos"] = (lk[0] + 0.55, lk[1] - 0.85, lk[2] + 0.6)
        frames = [env.render(width=960, height=720)]
        calls = 0
        for _ in range(4 * env.budget + 10):
            call = env.scripted()
            if call is None:
                break
            o = env.execute(apply_policy(env, call, "oracle"))
            calls += 1
            if calls == 1:
                frames.append(env.render(width=960, height=720))
            if o.done:
                break
        frames.append(env.render(width=960, height=720))
        for name, f in zip(("reset", "mid", "end"), frames):
            imageio.imwrite(f"{out}/{fam}_{rob}_{name}.png", f)
        sheet.append(np.concatenate([f[::2, ::2] for f in frames[:3]] + [np.zeros((360, 480 * (3 - len(frames[:3])), 3), np.uint8)]
                                    if len(frames) < 3 else [f[::2, ::2] for f in frames[:3]], axis=1))
        print(fam, rob, "solved", env.solved(), flush=True)
    imageio.imwrite(f"{out}/contact_sheet.png", np.concatenate(sheet, axis=0))


if __name__ == "__main__":
    main()
