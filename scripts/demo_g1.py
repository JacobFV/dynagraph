"""Paired G1 push episodes (same initial state/seed) for several predictors, with an
offscreen 1280x720 mp4 per predictor and a per-step JSON for the replay page.

  python scripts/demo_g1.py --seed 3 --predictors stub:0.5 analytic --tag paired_demo --video
  python scripts/demo_g1.py --seeds 0-9 --predictors stub:0.5 analytic --tag paired10   (json only)

Outputs: results/g1/<tag>/<pair_id>__<model>.json (+ .mp4), results/g1/<tag>/metrics.jsonl.
Everything the video shows that is not actor knowledge is labelled ORACLE / eval-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from physmem import control as C
from physmem.robots import g1

W, H = 1280, 720
CONFIG_HASH = None
MERGE_GLOBAL = False
EXPERIMENT = "e09"
NOTE = None
FPS = 30
HEADER = "unitree g1 · fixed-base simulation · synthetic sensors"


def _font(size):
    from PIL import ImageFont

    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class Recorder:
    def __init__(self, world, path, model_name, playback=1.0, oracle_pred=None):
        import mujoco

        self.mj = mujoco
        self.world = world
        self.sim = world._sim  # privileged: rendering only
        self.ren = mujoco.Renderer(self.sim.model, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.38, 0.0, 0.82]
        self.cam.distance = 1.0
        self.cam.azimuth = 215
        self.cam.elevation = -28
        self.model_name = model_name
        self.playback = playback
        self.sel = None
        self.oracle_pred = oracle_pred
        self.next_t = 0.0
        self.fonts = (_font(26), _font(20), _font(17))
        ff = shutil.which("ffmpeg") or os.path.expanduser("~/.local/bin/ffmpeg")
        self.proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                      f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                      "-crf", "23", str(path)], stdin=subprocess.PIPE)
        self.n = 0

    def _marker(self, pos, r, rgba, kind="sphere", h=0.001):
        sc = self.ren.scene
        if sc.ngeom >= sc.maxgeom:
            return
        g = sc.geoms[sc.ngeom]
        t = self.mj.mjtGeom.mjGEOM_SPHERE if kind == "sphere" else self.mj.mjtGeom.mjGEOM_CYLINDER
        self.mj.mjv_initGeom(g, t, np.array([r, h, 0.0]), np.asarray(pos, float), np.eye(3).flatten(),
                             np.asarray(rgba, np.float32))
        sc.ngeom += 1

    def maybe_frame(self, phase=""):
        t = self.sim.time
        if t + 1e-9 < self.next_t:
            return
        self.next_t += self.playback / FPS
        self.ren.update_scene(self.sim.data, camera=self.cam)
        z = g1.TABLE_TOP + 0.004
        if self.sel is not None:
            for (x, y), sd in zip(self.sel["pred_world"], self.sel["pred_sd"]):
                self._marker((x, y, z + 0.06), 0.008, (0.2, 0.6, 1.0, 0.9))
            (x, y), sd = self.sel["pred_world"][-1], self.sel["pred_sd"][-1]
            self._marker((x, y, z), max(sd, 0.003), (0.2, 0.6, 1.0, 0.25), "cyl")
        if self.oracle_pred is not None and self.sel is not None and self.sel["k"] in self.oracle_pred:
            for x, y in self.oracle_pred[self.sel["k"]]:
                self._marker((x, y, z + 0.09), 0.006, (1.0, 0.3, 0.8, 0.9))
        img = self.ren.render()
        self.proc.stdin.write(self._overlay(img, phase).tobytes())
        self.n += 1

    def _overlay(self, img, phase):
        from PIL import Image, ImageDraw

        im = Image.fromarray(img)
        d = ImageDraw.Draw(im, "RGBA")
        f0, f1, f2 = self.fonts
        d.rectangle([0, 0, W, 86], fill=(0, 0, 0, 150))
        d.text((16, 10), HEADER, font=f0, fill=(255, 255, 255))
        d.text((16, 48), f"predictor: {self.model_name}   sim t = {self.sim.time:5.2f} s   playback {self.playback:g}x   phase: {phase}",
               font=f1, fill=(220, 220, 220))
        y = 96
        lines = ["PELVIS FIXED (welded to world; orange mount)", "box pose: noisy synthetic sensor 2 mm / 1 deg @ 20 Hz",
                 "hand pose: FK of encoders; contact: synthetic tactile"]
        if self.sel is not None:
            th, ph, sp, du = self.sel["params"]
            bv = self.sel["belief_var"]
            fr = self.sel.get("friction")
            lines += [f"decision {self.sel['k']}: candidate {self.sel['chosen']}  theta {th:+.2f} phi {ph:+.2f} v {sp:.2f} m/s T {du:.2f} s",
                      f"score {self.sel['score']*100:.1f} cm   pred. final sd {self.sel['pred_sd'][-1]*100:.1f} cm   pose sd {math.sqrt(bv[0])*1000:.1f} mm"
                      + (f"   friction est {fr[0]:.2f} ± {math.sqrt(max(fr[1], 0)):.2f}" if fr else ""),
                      "blue: predicted box trajectory (chosen candidate) + final uncertainty disk"]
        if self.oracle_pred is not None:
            lines.append("magenta: ORACLE prediction with TRUE friction (eval-only overlay, not seen by actor)")
        lines.append("green disk: target region (r = 3 cm)")
        d.rectangle([8, y - 4, 820, y + 22 * len(lines) + 4], fill=(0, 0, 0, 110))
        for i, s in enumerate(lines):
            d.text((16, y + 22 * i), s, font=f2, fill=(255, 255, 255))
        return im

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def run_one(seed, spec, tag_dir, pair_id, video=False, playback=1.0, oracle_overlay=False, cfg=None,
            benchmark="tabletop-push-easy", capability=None):
    ep = g1.sample_slide_episode(seed) if benchmark == "slide-to-target" else g1.sample_push_episode(seed)
    world = g1.G1PushWorld(ep, seed=seed)
    if capability is not None:
        world.capability = capability
    pred = C.make_predictor(spec, oracle_friction=ep.box.friction if spec == "oracle" else None)
    name = getattr(pred, "name", spec)
    rec = None
    oracle_pred = {}
    if video:
        rec = Recorder(world, tag_dir / f"{pair_id}__{name}.mp4", name, playback, oracle_pred if oracle_overlay else None)
        orig = world.actor._tick

        def tick(t, phase=""):
            orig(t, phase)
            rec.maybe_frame(phase)

        world.actor._tick = tick
    oracle = C.PriorCoulombStub(ep.box.friction, name="ORACLE") if oracle_overlay else None

    def on_select(sel):
        if rec is not None:
            rec.sel = sel
        if oracle is not None:  # eval-only overlay: oracle rollout of the SAME chosen candidate
            from physmem import data as D

            evs = []  # oracle only needs the size; build from the belief-free size estimate
            th, ph, sp, du = sel["params"]
            cand = g1.ActionCandidate("o", "push", tuple(sel["params"]), g1.OBJ)
            import torch
            from physmem.schema import EventTensors, QueryTensors, Prop, HORIZONS

            ev = EventTensors(entity=torch.zeros(1, 1, dtype=torch.long), related=torch.full((1, 1), -1),
                              prop=torch.full((1, 1), int(Prop.SIZE)),
                              value=torch.tensor([[list(sel["half"]) + [0.0] * 5]], dtype=torch.float32),
                              logvar=torch.zeros(1, 1, 8), age=torch.zeros(1, 1), latency=torch.zeros(1, 1),
                              source=torch.zeros(1, 1, dtype=torch.long), provenance=torch.zeros(1, 1, dtype=torch.long),
                              mask=torch.ones(1, 1, dtype=torch.bool), ent_type=torch.zeros(1, 4, dtype=torch.long),
                              ent_mask=torch.ones(1, 4, dtype=torch.bool), dt_obs=torch.tensor([0.05]))
            q = QueryTensors(target=torch.zeros(1, dtype=torch.long), action=torch.tensor(cand.vector()[None]),
                             horizons=torch.tensor([HORIZONS], dtype=torch.float32))
            m = oracle(ev, q).mean[0].numpy()
            x0, y0, yaw0 = sel["est"]
            c, s = math.cos(yaw0), math.sin(yaw0)
            oracle_pred[sel["k"]] = [[x0 + c * r[0] - s * r[1], y0 + s * r[0] + c * r[1]] for r in m]

    t0 = time.time()
    res = C.run_episode(world, pred, cfg or C.ControllerConfig(), seed=seed, model_name=name, on_select=on_select)
    if rec is not None:
        rec.close()
    trace = world.trace()
    out = dict(pair_id=pair_id, tag=tag_dir.name, model=name, predictor_spec=spec, seed=seed, success=res.success,
               label=getattr(pred, "label", None),
               episode=ep.to_json(), metrics=res.metrics, decisions=res.decisions,
               oracle_overlay_eval_only=oracle_pred or None,
               steps=[dict(t=r["t"], box=[r["x"], r["y"], r["z"], r["yaw"]], box_speed=r["v"],
                           hand_l=r["hand_l"], hand_r=r["hand_r"], hand_xy=_near(r)[:2], hand_z=_near(r)[2],
                           f_hand=r["f_hand"]) for r in trace],
               phases=world.actor.log[::5], video=f"{pair_id}__{name}.mp4" if video else None,
               labels="box/hand truth in `steps` are simulator ground truth (eval/replay only); actor saw only noisy events")
    (tag_dir / f"{pair_id}__{name}.json").write_text(json.dumps(out, default=_jd))
    m = res.metrics
    row = dict(run_id=f"g1-{tag_dir.name}-{pair_id}-{name}", experiment=EXPERIMENT, note=NOTE, benchmark=benchmark, tag=tag_dir.name,
               model=name, seed=seed, pair_id=pair_id, config_hash=CONFIG_HASH, status="observed", world="g1_push",
               success=res.success, objective=m["objective"], push_cost=m["push_cost"],
               label=getattr(pred, "label", None), predictor_val_nll=getattr(pred, "val_nll", None),
               chosen_speeds=m["chosen_speeds"], max_speed=world.capability.max_push_speed,
               final_dist=m["final_dist"], n_pushes=m["n_pushes"], sim_time=m["sim_time"],
               mean_inference_latency=m["mean_inference_latency"], real_time_factor=m["real_time_factor"],
               audit_clean=m["clean"], dist_after_first_push=m["dist_after_first_push"],
               mean_final_pred_err=m["mean_final_pred_err"], friction=ep.box.friction, mass=ep.box.mass, wall=time.time() - t0)
    with open(tag_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row, default=_jd) + "\n")
    if MERGE_GLOBAL:
        with open("results/metrics.jsonl", "a") as f:
            f.write(json.dumps(dict(row, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), default=_jd) + "\n")
    print(f"[{pair_id}] {name:34s} success {res.success!s:5} pushes {m['n_pushes']} final {m['final_dist']*100:5.1f} cm "
          f"sim {m['sim_time']:5.1f}s mu {ep.box.friction:.2f} m {ep.box.mass:.2f} status {m['status']}", flush=True)
    return row


def _near(r):
    """hand (left/right pusher site) nearest to the box at this step."""
    dl = math.hypot(r["hand_l"][0] - r["x"], r["hand_l"][1] - r["y"])
    dr = math.hypot(r["hand_r"][0] - r["x"], r["hand_r"][1] - r["y"])
    return r["hand_l"] if dl < dr else r["hand_r"]


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--predictors", nargs="+", default=["stub:0.5", "analytic"])
    ap.add_argument("--tag", default="paired")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--playback", type=float, default=1.0)
    ap.add_argument("--oracle-overlay", action="store_true")
    ap.add_argument("--benchmark", default="slide-to-target", choices=["slide-to-target", "tabletop-push-easy"])
    ap.add_argument("--max-pushes", type=int, default=None)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--max-speed", type=float, default=None, help="capability override (slow-only ablation)")
    ap.add_argument("--merge-global", action="store_true", help="also append rows to results/metrics.jsonl")
    ap.add_argument("--experiment", default="e09")
    ap.add_argument("--note", default=None)
    ap.add_argument("--predictor-outer", action="store_true",
                    help="loop predictors outermost (load each heavy predictor once, e.g. inkling)")
    a = ap.parse_args()
    if a.seed is not None:
        seeds = [a.seed]
    elif "-" in a.seeds:
        lo, hi = a.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in a.seeds.split(",")]
    import dataclasses
    import hashlib

    global CONFIG_HASH, MERGE_GLOBAL, EXPERIMENT, NOTE
    MERGE_GLOBAL = a.merge_global
    EXPERIMENT, NOTE = a.experiment, a.note
    base = C.SLIDE_TO_TARGET if a.benchmark == "slide-to-target" else C.EASY
    cfg = dataclasses.replace(base, **{k: v for k, v in (("max_pushes", a.max_pushes), ("cap_s", a.cap)) if v is not None})
    cap = dataclasses.replace(g1.G1_CAPABILITY, max_push_speed=a.max_speed) if a.max_speed else None
    CONFIG_HASH = hashlib.sha1(json.dumps(dict(cfg=dataclasses.asdict(cfg), max_speed=a.max_speed,
                                               lam=cfg.uncertainty_lambda), sort_keys=True).encode()).hexdigest()[:12]
    pair_prefix = "g1slide" if a.benchmark == "slide-to-target" else "g1push"
    tag_dir = Path("results/g1") / a.tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    order = ([(spec, s) for spec in a.predictors for s in seeds] if a.predictor_outer
             else [(spec, s) for s in seeds for spec in a.predictors])
    for spec, s in order:
        pair_id = f"{pair_prefix}-s{s}"
        rows.append(run_one(s, spec, tag_dir, pair_id, a.video, a.playback, a.oracle_overlay, cfg, a.benchmark, cap))
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    summ = {k: dict(n=len(v), success=sum(x["success"] for x in v), mean_final_dist=float(np.mean([x["final_dist"] for x in v])),
                    mean_pushes=float(np.mean([x["n_pushes"] for x in v])),
                    mean_objective=float(np.mean([x["objective"] for x in v])),
                    frac_fast_pushes=float(np.mean([s_ >= C.FAST_SPEED[0] for x in v for s_ in x["chosen_speeds"]] or [0])),
                    mean_sim_time=float(np.mean([x["sim_time"] for x in v])),
                    mean_dist_after_first_push=float(np.nanmean([x["dist_after_first_push"] if x["dist_after_first_push"] is not None else np.nan for x in v])),
                    mean_final_pred_err=float(np.nanmean([x["mean_final_pred_err"] if x["mean_final_pred_err"] is not None else np.nan for x in v])))
            for k, v in by.items()}
    (tag_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
