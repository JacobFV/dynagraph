"""Videos of real robots doing the v2 task families (scripted heuristic policy), 1280x720 H.264 with a label strip.

    MUJOCO_GL=egl python scripts/envs_videos.py --families peg_insert,door_open --procs 5
    python scripts/envs_videos.py --montage            # results/v2/videos/montage_all_families.mp4

Per family: the best robot of ROBOTS' env matrix (docs/ROBOTS.md section 11, results/v2/robots/env_matrix.json; highest
success rate, then progress), a PRE-REGISTERED seed = the first seed of the family's test split (make_tasks(...,"test",1);
held-out families: their heldout_family split) - never cherry-picked; failures are shown and labelled. Frames are
captured from inside the physics loop at a fixed sim-time interval (playback speed labelled), main view = close 3/4
free camera on the work area, inset = close-up following the robot's TCP. Rendering is evaluator-side.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import time
from multiprocessing import Pool

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FFMPEG = os.path.expanduser("~/.local/bin/ffmpeg")
W, H, STRIP = 1280, 720, 92
FPS = 30
OUT = "results/v2/videos"
SPEED = {"door_open": 1, "drawer_open": 1, "twist_lock": 1, "peg_insert": 1, "small_peg_insert": 1, "fixture_place": 1,
         "connector_insert": 1, "tool_return": 1, "singulation": 2, "pouch_transfer": 2, "machine_tending": 2,
         "kitting_pocket": 2, "conveyor_sort": 2, "tote_transfer": 3, "bulk_sweep": 3, "shelf_restock": 3,
         "small_parts_pick": 3, "bin_clearing": 4, "depalletize": 4}
MAX_SIM = 360.0
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def best_robot(fam):
    for p in (os.path.expanduser("~/dynagraph/results/v2/robots/env_matrix.json"), "results/v2/robots/env_matrix.json"):
        if os.path.exists(p):
            m = json.load(open(p))["matrix"].get(fam, {})
            if m:
                rob, st = max(m.items(), key=lambda kv: (kv[1].get("success", 0) / max(kv[1].get("n", 1), 1),
                                                         kv[1].get("progress", 0)))
                return rob, st
    return "probe", {}


def _font(sz, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, sz)
    except Exception:
        return ImageFont.load_default()


class Recorder:
    def __init__(self, env, path, info, speed):
        self.env, self.info, self.speed = env, info, speed
        m = env.m
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), W)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), H)
        self.main = mujoco.Renderer(m, H - STRIP, W)
        self.inset = mujoco.Renderer(m, 240, 360)
        self.every = max(1, int(round(speed / (FPS * m.opt.timestep))))
        self.k = 0
        self.status = "running"
        self.skill = ""
        self.effort = 0.0
        self.f1, self.f2, self.fb = _font(22, True), _font(19), _font(30, True)
        self.cam = self._main_cam()
        self.proc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                      "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset",
                                      "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                                     stdin=subprocess.PIPE)
        self.n_frames = 0

    def _main_cam(self):
        env = self.env
        pts = [env.true_pose(e)[0] for e in env.scene.entities]
        pts += [np.asarray(r["center"]) for r in env.scene.regions.values()]
        P = np.array(pts)
        lo, hi = P.min(0), P.max(0)
        look = (lo + hi) / 2
        look[2] = max(look[2], env.surface_z)
        spread = float(np.linalg.norm((hi - lo)[:2]))
        # 3/4 view from the side AWAY from the robot base (robots stand at the -y side of every station), so the arm
        # does not fill the frame: camera at +x, +y relative to the work area, looking back toward the robot
        try:
            base = np.asarray(env.robot.mount[0], float)
        except Exception:
            base = look - np.array([0.0, 0.6, 0.0])
        away = look[:2] - base[:2]
        away = away / max(np.linalg.norm(away), 1e-6)
        side = np.array([-away[1], away[0]])
        dirv = 0.75 * away + 0.66 * side
        cam = mujoco.MjvCamera()
        cam.lookat[:] = look
        cam.distance = float(np.clip(1.1 * spread + 0.6, 0.85, 2.8))
        cam.azimuth = math.degrees(math.atan2(dirv[1], dirv[0]))
        cam.elevation = -35.0
        return cam

    def frame(self, final=False):
        env = self.env
        self.main.update_scene(env.d, self.cam)
        img = Image.fromarray(self.main.render())
        try:
            tcp = np.asarray(env.robot.tcp()[0], float)
            ic = mujoco.MjvCamera()
            ic.lookat[:] = tcp - np.array([0, 0, 0.05])
            ic.distance, ic.azimuth, ic.elevation = 0.5, self.cam.azimuth + 25, -40
            self.inset.update_scene(env.d, ic)
            ins = Image.fromarray(self.inset.render())
            img.paste(ins, (W - 360 - 12, 12))
            dr = ImageDraw.Draw(img)
            dr.rectangle([W - 372, 12, W - 12, 252], outline=(255, 255, 255), width=2)
            dr.text((W - 366, 14), "TCP close-up", font=self.f2, fill=(255, 255, 255))
        except Exception:
            pass
        canvas = Image.new("RGB", (W, H), (18, 20, 24))
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        i = self.info
        y0 = H - STRIP + 8
        d.text((14, y0), f"robot: {i['robot']}   task: {i['family']}   seed {i['seed']} ({i['split']} split, "
               f"pre-registered)   policy: scripted heuristic", font=self.f1, fill=(235, 235, 235))
        st = self.status
        col = (90, 220, 110) if st.startswith("SUCCESS") else ((240, 90, 80) if st.startswith("FAIL") else (230, 200, 90))
        d.text((14, y0 + 32), f"sim t = {env.t:6.1f} s   playback {self.speed}x   effort cost {self.effort:,.0f}   "
               f"skill: {self.skill}", font=self.f2, fill=(200, 200, 200))
        d.text((14, y0 + 58), "simulation · synthetic sensors · MuJoCo", font=self.f2, fill=(150, 150, 150))
        d.text((W - 420, y0 + 28), st, font=self.fb, fill=col)
        self.proc.stdin.write(canvas.tobytes())
        self.n_frames += 1

    def tick(self):
        self.k += 1
        if self.k % self.every == 0:
            self.frame()

    def close(self, hold_s=2.5):
        for _ in range(int(hold_s * FPS)):
            self.frame(final=True)
        self.proc.stdin.close()
        self.proc.wait()
        self.main.close()
        self.inset.close()


def record(fam, out_dir=OUT):
    from physmem.envs import ENV_VERSION, family_class, make_env
    from physmem.envs.splits import HELDOUT_FAMILIES, make_tasks
    rob, mstat = best_robot(fam)
    split = "heldout_family" if fam in HELDOUT_FAMILIES.values() else "test"
    task = make_tasks(fam, split, 1, rob)[0]
    env = make_env(fam)
    t0 = time.time()
    env.reset(task)
    speed = SPEED.get(fam, 2)
    path = os.path.join(out_dir, f"{fam}__{rob}__s{task.seed}.mp4")
    info = dict(robot=rob, family=fam, seed=task.seed, split=split)
    rec = Recorder(env, path, info, speed)
    orig = env.post_physics

    def post(stop=None):
        r = orig(stop)
        rec.tick()
        return r
    env.post_physics = post
    steps, out, note = [], None, ""
    try:
        while env.t < MAX_SIM:
            c = env.scripted()
            if c is None:
                note = "policy finished"
                break
            rec.skill = f"{c.skill} {c.target or ''}".strip()
            out = env.execute(c)
            rec.effort += out.result.effort.cost()
            steps.append((c.skill, c.target, out.result.status))
            if out.done:
                break
        else:
            note = f"video cap {MAX_SIM:.0f} s sim"
    except Exception as e:      # show and label failures
        note = f"error: {type(e).__name__}"
    ok = bool(env.solved())
    why = ""
    if not ok:
        bad = [f"{sk}: {st}" for sk, _, st in steps if st != "done"]
        top = max(set(bad), key=bad.count) if bad else ""
        why = ("damaged part" if getattr(env, "damaged", None) else
               ("dropped object" if env.metrics_acc.get("drops") else
                (f"{top} x{bad.count(top)}" if top else (note or "not completed"))))
    rec.status = "SUCCESS" if ok else f"FAIL ({why})"
    rec.skill = "-"
    rec.close()
    return dict(family=fam, robot=rob, seed=task.seed, split=split, success=ok, failure_reason=why,
                progress=float(env.progress()), sim_t=round(env.t, 1), playback_speed=speed, frames=rec.n_frames,
                video_s=round(rec.n_frames / FPS, 1), effort_cost=round(rec.effort, 1), calls=len(steps),
                steps=steps, env_version=ENV_VERSION, matrix_cell=mstat, path=path,
                integrity_violations=env.metrics_acc.get("integrity_violations", 0), wall_s=round(time.time() - t0, 1))


def _job(fam):
    try:
        return record(fam)
    except Exception as e:
        return dict(family=fam, error=repr(e)[:300])


def montage(out_dir=OUT, per=9.0):
    """~9 s per family (19 families -> ~3 min): each clip sped up by an extra factor f to fit, with an overlay stating f."""
    idx = json.load(open(os.path.join(out_dir, "index.json")))
    clips, tmp = [], os.path.join(out_dir, "_montage")
    os.makedirs(tmp, exist_ok=True)
    f1 = _font(28, True)
    for e in sorted(idx.values(), key=lambda e: e["family"]):
        if "path" not in e or not os.path.exists(e["path"]):
            continue
        f = max(1.0, e["video_s"] / per)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        msg = f"{e['family']}  ({e['robot']})" + (f"   montage: {f:.1f}x faster than the speed in the strip" if f > 1.05 else "")
        dr.rectangle([0, 0, W, 46], fill=(0, 0, 0, 170))
        dr.text((14, 8), msg, font=f1, fill=(255, 255, 255, 255))
        png = os.path.join(tmp, f"{e['family']}.png")
        ov.save(png)
        clip = os.path.join(tmp, f"{e['family']}.mp4")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", e["path"], "-i", png, "-filter_complex",
                        f"[0:v]setpts=PTS/{f:.4f},fps={FPS}[v];[v][1:v]overlay=0:0", "-t", f"{per + 0.5:.1f}", "-an",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", clip], check=True)
        clips.append(clip)
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join(f"file '{os.path.abspath(c)}'\n" for c in clips))
    dst = os.path.join(out_dir, "montage_all_families.mp4")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", dst], check=True)
    return dst, len(clips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="all")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.montage:
        print(montage())
        return
    from physmem.envs import FAMILIES
    fams = list(FAMILIES) if a.families == "all" else a.families.split(",")
    with Pool(a.procs, maxtasksperchild=1) as pool:
        res = pool.map(_job, fams, chunksize=1)
    ipath = os.path.join(OUT, "index.json")
    idx = json.load(open(ipath)) if os.path.exists(ipath) else {}
    for r in res:
        idx[r["family"]] = r
        print(json.dumps({k: r.get(k) for k in ("family", "robot", "seed", "success", "failure_reason", "sim_t",
                                                 "video_s", "playback_speed", "error", "path")}))
    json.dump(idx, open(ipath, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
