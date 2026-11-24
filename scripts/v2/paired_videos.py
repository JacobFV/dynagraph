"""MODEL-V2 r5: paired side-by-side videos, heuristic (scripted) vs gated residual planner, SAME seed, site-dressed scene,
label strip with policy, success / damage and effort; reuses ENVS' Recorder (scripts/envs_videos.py).
usage: MUJOCO_GL=egl python scripts/v2/paired_videos.py <tag> <trust.json> <robot:family:seed> [...]
outputs results/v2/videos_paired/<family>__<robot>__s<seed>.mp4 (+ .json) and 4 stills (--stills N)."""
import importlib.util, json, os, subprocess, sys
import numpy as np
from PIL import ImageDraw
spec = importlib.util.spec_from_file_location("envs_videos", "scripts/envs_videos.py")
EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
from physmem.envs import make_env
from physmem.v2 import api
from physmem.models.v2 import harness as H, spec as S
from physmem.models.v2.env_adapter import EnvAdapter
from physmem.planner import CEMPlanner, PlannerConfig
from physmem.v2_train import load_model
OUT = "results/v2/videos_paired"


def run(tag, trust, robot, fam, seed, policy, split="test", idx=0):
    env = make_env(fam)
    task = api.TaskSpec(fam, "", int(seed), robot, split, params={"dressing": "site"})
    ad = EnvAdapter(env)
    ad.reset(task)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"_{fam}__{robot}__s{seed}__{policy}.mp4")
    label = "scripted heuristic" if policy == "heuristic" else "MODEL-V2 residual planner (typed memory, gated)"
    info = dict(robot=robot, family=fam, seed=seed, split=f"test; policy: {label}; same seed")
    rec = EV.Recorder(env, path, info, EV.SPEED.get(fam, 2))
    orig = env.post_physics
    def post(stop=None):
        r = orig(stop); rec.tick(); return r
    env.post_physics = post
    planner = None
    if policy != "heuristic":
        cfg = PlannerConfig(pop=96, iters=3, n_z=16, voi=False, c_prog=2.0, lam_effort=0.03)
        C = "results/v2/checkpoints"
        pls = [CEMPlanner(load_model(f"{C}/v2_robots_{tag}_typed_s{s}.pt")[0], "typed", cfg) for s in (0, 1, 2)]
        tr = {f: v["trusted"] for f, v in json.load(open(trust))["families"].items()}
        planner = H.ResidualPlanner(pls, trust=tr)
    emb = S.embodiment_vector(ad.embodiment_spec())
    rng = np.random.default_rng(int(idx))      # identical to the table episode (harness rng_seed = task index)
    hist, effort, damaged = [], 0.0, False
    for k in range(getattr(env, "budget", 20)):
        if policy == "heuristic":
            c = env.scripted()
        else:
            c, _ = planner.decide(ad, hist, emb, rng)
        if c is None:
            break
        rec.skill = f"{c.skill} {c.target or ''}".strip() + (f"  grip {c.params['grip_force']:.0f} N" if "grip_force" in c.params else "")
        out = ad.execute(c, relative=False)
        hist.append(out.result.log["interaction"])
        effort += out.result.effort.cost(); rec.effort = effort
        damaged = damaged or H._damaged(out)
        if out.done:
            break
    ok = bool(env.solved())
    rec.status = ("SUCCESS" if ok else ("FAIL (damaged part)" if damaged else "FAIL")) + ("" if ok or damaged else "")
    rec.skill = "-"
    rec.close()
    return dict(path=path, success=ok, damaged=bool(damaged), effort=effort, policy=label, video_s=rec.n_frames / EV.FPS)


def main():
    tag, trust = sys.argv[1], sys.argv[2]
    stills = 0
    args = [a for a in sys.argv[3:] if not a.startswith("--stills")]
    for a in sys.argv[3:]:
        if a.startswith("--stills"):
            stills = int(a.split("=")[1])
    index = []
    for cell in args:
        robot, fam, seed, split, idx = (cell.split(":") + ["test", "0"])[:5]
        L = run(tag, trust, robot, fam, seed, "heuristic", split, int(idx))
        R = run(tag, trust, robot, fam, seed, "residual", split, int(idx))
        out = os.path.join(OUT, f"{fam}__{robot}__s{seed}.mp4")
        subprocess.run([EV.FFMPEG, "-y", "-loglevel", "error", "-i", L["path"], "-i", R["path"], "-filter_complex",
                        "[0:v]tpad=stop_mode=clone:stop_duration=600[a];[1:v]tpad=stop_mode=clone:stop_duration=600[b];"
                        "[a][b]hstack=inputs=2,scale=2560:720[v]", "-map", "[v]", "-t", f"{max(L['video_s'], R['video_s']):.2f}", "-c:v", "libx264",
                        "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
        meta = dict(robot=robot, family=fam, seed=int(seed), split=split, task_index=int(idx), left=L, right=R,
                    dressing="site", path=out, note="same seed, same scene; left = plain scripted heuristic, right = "
                    "MODEL-V2 gated residual planner (3-seed typed-memory ensemble)")
        json.dump(meta, open(out[:-4] + ".json", "w"), indent=1)
        index.append(meta)
        print(json.dumps(meta), flush=True)
        if stills:
            gal = os.path.expanduser("~/dynagraph_gallery_paired")      # copied to ~/Downloads/dynagraph_gallery/paired locally
            os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
            t = 0.7 * max(L["video_s"], R["video_s"])
            png = os.path.join(OUT, "stills", f"{fam}__{robot}__s{seed}.png")
            subprocess.run([EV.FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", out, "-frames:v", "1", png], check=True)
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
