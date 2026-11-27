"""envs-v3 dev runner: one scripted episode, per-call diagnostics (local light dev).

    python scripts/envs_v3_dev.py <family> <robot> <seed> [policy] [key=value params ...] [--render out.png]
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")


def main():
    fam, robot, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    pol = sys.argv[4] if len(sys.argv) > 4 and "=" not in sys.argv[4] and not sys.argv[4].startswith("--") else "nominal"
    params, render, sheet, every = {}, None, None, 1.0
    args = sys.argv[4:]
    for i, a in enumerate(args):
        if a == "--render":
            render = args[i + 1]
        elif a == "--sheet":
            sheet = args[i + 1]
        elif a == "--every":
            every = float(args[i + 1])
        elif "=" in a:
            k, v = a.split("=", 1)
            try:
                params[k] = json.loads(v)
            except Exception:
                params[k] = v
    from physmem.envs import family_class, make_env
    from physmem.v2.api import TaskSpec
    from envs_ceiling import apply_policy
    env = make_env(fam)
    t0 = time.time()
    env.reset(TaskSpec(fam, family_class(fam).category, seed, robot, "test", params))
    frames = []
    if "--close" in sys.argv:            # zoomed debug view
        c = env.scene.camera
        lk = np.asarray(c.get("view_lookat", c["lookat"]), float)
        c["view_pos"] = (lk[0] + 0.35, lk[1] - 0.55, lk[2] + 0.45)
    if sheet:
        import types
        orig = env.post_physics
        nxt = [0.0]

        def pp(self_, stop=None):
            r = orig(stop)
            if self_.t >= nxt[0]:
                nxt[0] = self_.t + every
                frames.append((round(self_.t, 1), self_.render(width=320, height=240)))
            return r
        env.post_physics = types.MethodType(pp, env)
    print(f"reset {time.time() - t0:.1f}s attempts {env.reset_attempts} hidden {getattr(env, 'hidden_task', {})}")
    for _ in range(4 * env.budget + 10):
        c = env.scripted()
        if c is None:
            break
        c = apply_policy(env, c, pol)
        t1, s1 = time.time(), env.t
        out = env.execute(c)
        mx = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.metrics.items()
              if k in ("progress", "poured_kg", "spilled_kg", "left_kg", "collisions", "damaged", "drops",
                       "unstable", "integrity_violations", "fold_err", "far_moved", "x_err", "seated", "route_ok", "tug_n", "stand_s") or k.startswith("x_")}
        lg = {k: v for k, v in (out.result.log or {}).items() if k not in ("params", "cmd", "peaks", "skill", "target")}
        print(f"{c.skill:14s} {out.result.status:14s} sim {env.t - s1:5.1f}s wall {time.time() - t1:5.1f}s "
              f"params {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.params.items()})}")
        print(f"   metrics {mx}  log {lg}")
        if out.done:
            break
    print("SOLVED", env.solved(), "progress", round(env.progress(), 3), "sim_t", round(env.t, 1),
          "wall", round(time.time() - t0, 1))
    if sheet and frames:
        import imageio
        fr = frames[:40]
        cols = 5
        rows_ = (len(fr) + cols - 1) // cols
        H, W = 240, 320
        img = np.zeros((rows_ * H, cols * W, 3), np.uint8)
        for k, (t, f) in enumerate(fr):
            img[(k // cols) * H:(k // cols + 1) * H, (k % cols) * W:(k % cols + 1) * W] = f
        imageio.imwrite(sheet, img)
        print("sheet times", [t for t, _ in fr])
    if render:
        import imageio
        imageio.imwrite(render, env.render(width=640, height=480))


if __name__ == "__main__":
    main()
