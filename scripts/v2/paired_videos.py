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


class StripRecorder(EV.Recorder):
    """round 4 fix: the base strip packs robot/task/seed/split/policy into ONE line that clipped at the half-panel edge.
    Three short lines instead: (1) robot / task / seed / split, (2) policy, (3) time / effort / skill; outcome on the
    right of line 3; fonts sized for a 1280-px half panel."""
    policy_label = ""

    def frame(self, final=False):
        env = self.env
        self.main.update_scene(env.d, self.cam)
        img = EV.Image.fromarray(self.main.render())
        try:
            tcp = np.asarray(env.robot.tcp()[0], float)
            ic = EV.mujoco.MjvCamera()
            ic.lookat[:] = tcp - np.array([0, 0, 0.05])
            ic.distance, ic.azimuth, ic.elevation = 0.5, self.cam.azimuth + 25, -40
            self.inset.update_scene(env.d, ic)
            img.paste(EV.Image.fromarray(self.inset.render()), (EV.W - 372, 12))
        except Exception:
            pass
        canvas = EV.Image.new("RGB", (EV.W, EV.H), (18, 20, 24))
        canvas.paste(img, (0, 0))
        d = EV.ImageDraw.Draw(canvas)
        i = self.info
        y0 = EV.H - EV.STRIP + 4
        f1, f2 = EV._font(19, True), EV._font(17)
        d.text((12, y0), f"{i['robot']}  ·  {i['family']}  ·  seed {i['seed']} ({i['split']})  ·  site-dressed", font=f1,
               fill=(235, 235, 235))
        d.text((12, y0 + 27), f"policy: {self.policy_label}", font=f1, fill=(120, 190, 255))
        d.text((12, y0 + 55), f"sim t {env.t:5.1f} s · {self.speed}x · effort {self.effort:,.0f} · {self.skill}"[:78],
               font=f2, fill=(200, 200, 200))
        st = self.status
        col = (90, 220, 110) if st.startswith("SUCCESS") else ((240, 90, 80) if st.startswith("FAIL") else (230, 200, 90))
        d.text((EV.W - 330, y0 + 50), st, font=EV._font(24, True), fill=col)
        self.proc.stdin.write(canvas.tobytes())
        self.n_frames += 1


def _lot_params():
    """lot cells: n_parts (NPARTS, default 3) and the shared budget (BUDGET_PER_PART x parts; round 5 / 12a)."""
    n = int(os.environ.get("NPARTS", "3"))
    bpp = float(os.environ.get("BUDGET_PER_PART", "0") or 0)
    return dict(n_parts=n, **({"budget": int(round(bpp * n))} if bpp > 0 else {}))


def run(tag, trust, robot, fam, seed, policy, split="test", idx=0, variant="residual_gated"):
    env = make_env(fam)
    lot = variant.startswith("lot")
    task = api.TaskSpec(fam, "", int(seed), robot, split,
                        params={"dressing": "site", **(_lot_params() if lot else {})})
    if lot:
        from physmem.models.v2.lot import LotEnvAdapter
        ad = LotEnvAdapter(env)
    else:
        ad = EnvAdapter(env)
    ad.reset(task)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"_{fam}__{robot}__s{seed}__{policy}.mp4")
    label = ("scripted heuristic" if policy == "heuristic" else
             "always-probe scripted (probe-then-act)" if policy == "heuristic_probe" else
             ("MODEL-V2 typed lot memory, crush-gated probing (v2)" if variant == "lot_v2"
              else "MODEL-V2 typed lot memory: memory-conditioned probing") if lot else
             ("MODEL-V2 residual planner, gated, anchored on probe-then-act" if variant == "residual_gated_probe"
              else "MODEL-V2 residual planner, gated (plain scripted anchor)"))
    info = dict(robot=robot, family=fam, seed=seed, split=split)
    rec = StripRecorder(env, path, info, EV.SPEED.get(fam, 2))
    rec.policy_label = label + "  (same seed, paired)"
    orig = env.post_physics
    def post(stop=None):
        r = orig(stop); rec.tick(); return r
    env.post_physics = post
    planner = None
    if lot and policy == "residual":
        from physmem.models.v2.lot import LotPlanner
        cfg = PlannerConfig(n_z=16, voi=False)
        C = "results/v2/checkpoints"
        planner = LotPlanner([CEMPlanner(load_model(f"{C}/v2_robots_{tag}_typed_s{s}.pt")[0], "typed", cfg)
                              for s in (0, 1, 2)], tau=float(os.environ.get("TAU", "0.15")),
                             crush_gate=variant == "lot_v2")
    elif policy not in ("heuristic", "heuristic_probe"):
        cfg = PlannerConfig(pop=96, iters=3, n_z=16, voi=False, c_prog=2.0, lam_effort=0.03)
        C = "results/v2/checkpoints"
        pls = [CEMPlanner(load_model(f"{C}/v2_robots_{tag}_typed_s{s}.pt")[0], "typed", cfg) for s in (0, 1, 2)]
        tr = {f: v["trusted"] for f, v in json.load(open(trust))["families"].items()}
        planner = H.ResidualPlanner(pls, trust=tr)
        planner.anchor_probe = variant == "residual_gated_probe"
    emb = S.embodiment_vector(ad.embodiment_spec())
    rng = np.random.default_rng(int(idx))      # identical to the table episode (harness rng_seed = task index)
    hist, effort, damaged, probes = [], 0.0, False, 0
    for k in range(getattr(env, "budget", 20)):
        if policy in ("heuristic", "heuristic_probe"):
            c = env.scripted()
            if c is not None and policy == "heuristic_probe":
                c = env.probe_then_act(c)
        else:
            c, _ = planner.decide(ad, hist, emb, rng)
        if c is None:
            break
        is_probe = lot and c.skill == "pick" and float(c.params.get("grip_force", 99)) <= 4.0
        probes += int(is_probe)
        rec.skill = (f"{c.skill} {c.target or ''}".strip() + (f"  grip {c.params['grip_force']:.0f} N" if "grip_force" in c.params else "")
                     + ("  [PROBE]" if is_probe else "") + (f"  probes {probes}" if lot else ""))
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
    return dict(path=path, success=ok, damaged=bool(damaged), effort=effort, probes=probes, policy=label, video_s=rec.n_frames / EV.FPS)


def main():
    tag, trust = sys.argv[1], sys.argv[2]
    stills = 0
    args = [a for a in sys.argv[3:] if not a.startswith("--stills")]
    for a in sys.argv[3:]:
        if a.startswith("--stills"):
            stills = int(a.split("=")[1])
    index = []
    for cell in args:
        robot, fam, seed, split, idx, variant = (cell.split(":") + ["test", "0", "residual_gated"])[:6]
        L = run(tag, trust, robot, fam, seed, "heuristic_probe" if variant.startswith("lot") else "heuristic", split,
                int(idx), variant)
        R = run(tag, trust, robot, fam, seed, "residual", split, int(idx), variant)
        out = os.path.join(OUT, f"{fam}__{robot}__s{seed}.mp4")
        subprocess.run([EV.FFMPEG, "-y", "-loglevel", "error", "-i", L["path"], "-i", R["path"], "-filter_complex",
                        "[0:v]tpad=stop_mode=clone:stop_duration=600[a];[1:v]tpad=stop_mode=clone:stop_duration=600[b];"
                        "[a][b]hstack=inputs=2,scale=2560:720[v]", "-map", "[v]", "-t", f"{max(L['video_s'], R['video_s']):.2f}", "-c:v", "libx264",
                        "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
        meta = dict(robot=robot, family=fam, seed=int(seed), split=split, task_index=int(idx), left=L, right=R,
                    dressing="site", path=out, note=(f"same seed, same lot; left = {L['policy']}, right = {R['policy']} (3-seed typed-memory ensemble)"
                          if variant.startswith("lot") else "same seed, same scene; left = plain scripted heuristic, right = "
                          "MODEL-V2 gated residual planner (3-seed typed-memory ensemble)"))
        json.dump(meta, open(out[:-4] + ".json", "w"), indent=1)
        index.append(meta)
        print(json.dumps(meta), flush=True)
        if stills:
            gal = os.path.expanduser("~/dynagraph_gallery_paired")      # copied to ~/Downloads/dynagraph_gallery/paired locally
            os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
            for q in ((0.2, 0.45, 0.7, 0.95) if stills >= 4 else (0.7,)):
                t = q * max(L["video_s"], R["video_s"])
                png = os.path.join(OUT, "stills", f"{fam}__{robot}__s{seed}__t{int(q * 100):02d}.png")
                subprocess.run([EV.FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", out, "-frames:v", "1", png],
                               check=True)
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
