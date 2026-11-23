"""Final demo video results/demo.mp4 (1280x720, 30 fps, H.264) assembled from REAL recorded G1 episodes.

Segments (spec 9.4), every one labelled on-screen:
 1 title card (hardware, models, checkpoint names)
 2 held-out slide-to-target episode, representative fixed seed 29 (chosen before looking at outcomes),
   compact b11 in the loop: object/hand/contact state, belief mean+sd, memory-update provenance,
   predicted short trajectory of the chosen candidate
 3 held-out surface parameter: b11 friction belief before vs after the first push (numbers)
 4 paired baseline from the same initial state: b00 vs b11 side-by-side
 5 relevant-slot intervention: b11 with the typed FRICTION slot overwritten (model.intervene before refine)
   vs normal b11: prediction change, chosen-candidate change, realized outcome
 + Inkling panel: decision log of the recorded slide_inkling seed-29 episode (no video was recorded)
 6 end panel: counts with denominators, latency, lift/place gate, limitations
Writes results/g1/demo_manifest.json and results/g1/demo/*.json (episode logs).

Episodes are re-simulated here with the same code/seeds (deterministic) and rendered offscreen (EGL);
compact models run on CPU. No Inkling model is loaded.
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from physmem import control as C
from physmem.robots import g1
from physmem.schema import Prop

W, H, FPS = 1280, 720, 30
SEED = 29
OUT = Path("results/g1/demo")
CLIPS = OUT / "clips"
CK = {"b11": "results/checkpoints/compact-main-b11-s1.pt", "b00": "results/checkpoints/compact-main-b00-s1.pt"}
FF = shutil.which("ffmpeg") or os.path.expanduser("~/.local/bin/ffmpeg")

spec = importlib.util.spec_from_file_location("demo_g1", "scripts/demo_g1.py")
DG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DG)
FONT = {s: DG._font(s) for s in (15, 17, 20, 24, 30, 40)}


def writer(path):
    return subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                             "-preset", "medium", str(path)], stdin=subprocess.PIPE)


def reader(path, w=W, h=H):
    p = subprocess.Popen([FF, "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                          f"{w}x{h}", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.wait()


def card(path, title, lines, seconds, sub=None, color=(18, 22, 30)):
    im = Image.new("RGB", (W, H), color)
    d = ImageDraw.Draw(im)
    d.text((60, 40), title, font=FONT[30], fill=(255, 255, 255))
    if sub:
        d.text((60, 86), sub, font=FONT[17], fill=(170, 190, 220))
    y = 130
    for ln in lines:
        if isinstance(ln, tuple):
            txt, col, f = ln
        else:
            txt, col, f = ln, (225, 225, 225), 17
        d.text((60, y), txt, font=FONT[f], fill=col)
        y += f + 9
    d.text((60, H - 34), "unitree g1 · fixed-base simulation · synthetic sensors", font=FONT[15], fill=(140, 140, 140))
    raw = np.asarray(im).tobytes()
    pw = writer(path)
    for _ in range(int(seconds * FPS)):
        pw.stdin.write(raw)
    pw.stdin.close()
    pw.wait()


# --------------------------------------------------------------------------------------------------
# predictors
# --------------------------------------------------------------------------------------------------


class FrictionIntervened:
    """b11 with the typed FRICTION slot overwritten (model.intervene before refine). Also computes the
    normal b11 output on the SAME tensors to log the prediction change (diagnostic only)."""

    def __init__(self, base, value, sd=0.01):
        self.base, self.value, self.lv = base, float(value), math.log(sd * sd)
        self.name = f"compact_b11_FRICTION_slot:={value:g}"
        self.calls = []

    def __call__(self, ev, q):
        m = self.base.model
        with torch.no_grad():
            tok, mask, slots = m.encode(ev, q)
            out_n = m.readout(m.refine(tok, mask, self.base.passes), mask, q, slots)
            tok2, slots2 = m.intervene(tok, mask, slots, Prop.FRICTION, torch.tensor(self.value), torch.tensor(self.lv))
            out_i = m.readout(m.refine(tok2, mask, self.base.passes), mask, q, slots2)
        dn = out_n.mean[:, -1, :2].numpy()
        di = out_i.mean[:, -1, :2].numpy()
        diff = np.linalg.norm(di - dn, axis=1)
        self.calls.append(dict(friction_normal=float(out_n.friction_mean.mean()), friction_intervened=float(out_i.friction_mean.mean()),
                               pred_change_mean_cm=float(100 * diff.mean()), pred_change_max_cm=float(100 * diff.max()),
                               argmin_normal_vs_intervened=None, _normal=out_n, _int=out_i))
        return out_i


class Tapped:
    """wrap a predictor to keep its last ModelOutput (for the overlay)."""

    def __init__(self, inner):
        self.inner, self.name = inner, getattr(inner, "name", "pred")

    def __call__(self, ev, q):
        return self.inner(ev, q)


# --------------------------------------------------------------------------------------------------
# episode recorder with the full state panel
# --------------------------------------------------------------------------------------------------


class PanelRecorder(DG.Recorder):
    def __init__(self, world, path, model_name, header_extra, **kw):
        super().__init__(world, path, model_name, **kw)
        self.header_extra = header_extra
        self.last_pose = None
        self.last_contact = None
        self.mem = []
        self.k = None
        self.fric_hist = []

    def tap(self, evs):
        for e in evs:
            if e.entity_id == g1.OBJ and e.property_type == Prop.POSE2:
                self.last_pose = e
            if e.property_type == Prop.CONTACT:
                self.last_contact = e

    def _overlay(self, img, phase):
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im, "RGBA")
        f = FONT
        d.rectangle([0, 0, W, 74], fill=(0, 0, 0, 165))
        d.text((14, 6), DG.HEADER + "   ·   " + self.header_extra, font=f[20], fill=(255, 255, 255))
        d.text((14, 40), f"predictor: {self.model_name}   sim t = {self.sim.time:5.2f} s   playback {self.playback:g}x (no cuts)   phase: {phase}",
               font=f[17], fill=(215, 215, 215))
        # right panel
        x0, y = 845, 84
        d.rectangle([x0 - 8, y - 6, W - 6, H - 8], fill=(0, 0, 0, 150))
        L = []
        st = self.sim.true_object_state()
        fh = self.sim.contact_forces()["hand"]
        L.append(("OBJECT / HAND / CONTACT", (255, 210, 120)))
        if self.last_pose is not None:
            v = self.last_pose.value
            L.append((f"belief pose (sensor) x {v[0]:.3f} y {v[1]:.3f} m  sd {math.sqrt(self.last_pose.variance[0])*1000:.1f} mm", (255, 255, 255)))
        L.append((f"truth (eval-only) x {st['pos'][0]:.3f} y {st['pos'][1]:.3f}  v {np.linalg.norm(st['lin_vel'][:2]):.2f} m/s", (170, 170, 170)))
        if self.last_contact is not None:
            L.append((f"tactile CONTACT (actor) = {int(self.last_contact.value[0])}  @ {self.last_contact.measured_at:.2f}s", (255, 255, 255)))
        L.append((f"hand-box normal force (eval-only) {fh:5.1f} N", (170, 170, 170)))
        gx, gy = self.world.goal["target_xy"]
        L.append((f"distance to target (truth): {100*math.hypot(st['pos'][0]-gx, st['pos'][1]-gy):.1f} cm", (170, 170, 170)))
        if self.sel is not None:
            s = self.sel
            th, ph, sp, du = s["params"]
            L.append(("", None))
            L.append((f"DECISION {s['k']}  (belief rev {s.get('belief_revision')}, t={s.get('belief_time', 0):.2f}s)", (255, 210, 120)))
            L.append((f"chosen {s['chosen']}: v {sp:.2f} m/s, T {du:.2f} s, theta {th:+.2f}", (255, 255, 255)))
            L.append((f"score {100*s['score']:.1f} cm; pred final sd {100*s['pred_sd'][-1]:.1f} cm", (255, 255, 255)))
            fr = s.get("friction")
            if fr:
                L.append((f"FRICTION belief {fr[0]:.2f} ± {math.sqrt(max(fr[1], 0)):.2f}   (true {self.world.ep.box.friction:.2f}, eval-only)", (120, 220, 255)))
            L.append(("memory updates since last decision:", (255, 210, 120)))
            for r in (s.get("memory_updates") or {}).get("records", [])[:6]:
                src = (r.get("sources") or ["?"])[0]
                mv = ", ".join(f"{x:.3f}" for x in r["mean"][:2])
                L.append((f" {r['entity'][:10]}.{r['prop']} rev{r['revision']} [{mv}] <- {src[:24]}", (210, 210, 210)))
        for txt, col in L:
            if col is not None:
                d.text((x0, y), txt, font=f[15], fill=col)
            y += 20
        d.rectangle([0, H - 54, 835, H], fill=(0, 0, 0, 140))
        d.text((14, H - 50), "blue: predicted box trajectory of chosen candidate + final uncertainty disk · green: 3 cm target", font=f[15], fill=(200, 220, 255))
        if self.oracle_pred is not None:
            d.text((14, H - 28), "magenta: ORACLE prediction (true friction) — eval-only overlay, not seen by the actor", font=f[15], fill=(255, 150, 230))
        else:
            d.text((14, H - 28), "PELVIS FIXED (orange mount) · box pose: synthetic 2 mm/1° sensor · hand: encoder FK", font=f[15], fill=(200, 200, 200))
        return im


def run_recorded(name, predictor, header_extra, oracle_overlay=False):
    ep = g1.sample_slide_episode(SEED)
    world = g1.G1PushWorld(ep, seed=SEED)
    clip = CLIPS / f"ep_{name}.mp4"
    oracle_pred = {} if oracle_overlay else None
    rec = PanelRecorder(world, clip, getattr(predictor, "name", name), header_extra, playback=1.0, oracle_pred=oracle_pred)
    orig_tick = world.actor._tick
    orig_drain = world.actor.drain_events

    def tick(t, phase=""):
        orig_tick(t, phase)
        rec.maybe_frame(phase)

    def drain():
        evs = orig_drain()
        rec.tap(evs)
        return evs

    world.actor._tick = tick
    world.actor.drain_events = drain

    def on_select(sel):
        rec.sel = sel

    res = C.run_episode(world, predictor, C.SLIDE_TO_TARGET, seed=SEED, model_name=getattr(predictor, "name", name),
                        on_select=on_select)
    for _ in range(int(1.0 * FPS)):  # 1 s hold on the final state
        rec.next_t = 0.0
        rec.maybe_frame("final")
    rec.close()
    out = dict(name=name, seed=SEED, model=res.model, success=res.success, metrics=res.metrics, decisions=res.decisions,
               episode=ep.to_json(), clip=str(clip))
    (OUT / f"{name}.json").write_text(json.dumps(out, default=DG._jd))
    print(f"[{name}] success {res.success} final {100*res.metrics['final_dist']:.1f} cm pushes {res.metrics['n_pushes']}", flush=True)
    return out


def side_by_side(path, clip_a, clip_b, label_a, label_b, title):
    pw = writer(path)
    ga, gb = reader(clip_a), reader(clip_b)
    fa = fb = None
    for _ in range(100000):
        na, nb = next(ga, None), next(gb, None)
        if na is None and nb is None:
            break
        fa = na if na is not None else fa
        fb = nb if nb is not None else fb
        im = Image.new("RGB", (W, H), (12, 14, 18))
        im.paste(Image.fromarray(fa).resize((636, 358)), (2, 150))
        im.paste(Image.fromarray(fb).resize((636, 358)), (642, 150))
        d = ImageDraw.Draw(im)
        d.text((20, 20), title, font=FONT[24], fill=(255, 255, 255))
        d.text((20, 62), "same initial state (seed 29), both played at 1x, no cuts; panels shrunk from the full-size recordings",
               font=FONT[17], fill=(190, 200, 220))
        d.text((10, 118), label_a, font=FONT[20], fill=(255, 210, 120))
        d.text((650, 118), label_b, font=FONT[20], fill=(120, 220, 255))
        d.text((20, H - 34), "unitree g1 · fixed-base simulation · synthetic sensors", font=FONT[15], fill=(140, 140, 140))
        pw.stdin.write(np.asarray(im).tobytes())
    pw.stdin.close()
    pw.wait()


def summ(tag):
    rows = [json.loads(l) for l in open(f"results/g1/{tag}/metrics.jsonl")]
    last = {}
    for r in rows:
        last[(r["pair_id"], r["model"])] = r
    by = {}
    for r in last.values():
        by.setdefault(r["model"], []).append(r)
    return {m: dict(n=len(v), succ=sum(x["success"] for x in v), final=100 * float(np.mean([x["final_dist"] for x in v])),
                    med=100 * float(np.median([x["final_dist"] for x in v])), lat=1000 * float(np.mean([x["mean_inference_latency"] for x in v])),
                    rtf=float(np.mean([x["real_time_factor"] for x in v]))) for m, v in by.items()}


def main():
    t0 = time.time()
    CLIPS.mkdir(parents=True, exist_ok=True)
    ep = g1.sample_slide_episode(SEED)
    mu = ep.box.friction
    wrong = 0.3 if mu > 0.5 else 0.7
    b11 = C.make_predictor("compact:" + CK["b11"])
    b00 = C.make_predictor("compact:" + CK["b00"])
    b11.name, b00.name = "compact b11 (structured + physics) s1", "compact b00 (generic, no physics) s1"
    manifest = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seed=SEED,
                    seed_note="representative fixed seed chosen before looking at outcomes (continuity with slide_video_s29)",
                    benchmark="slide-to-target", controller="v1", checkpoints=CK, segments=[])
    clips = []
    # 1 title
    c = CLIPS / "s1_title.mp4"
    card(c, "unitree g1 · fixed-base simulation · synthetic sensors",
         [("Physical-memory models controlling a simulated Unitree G1 (MuJoCo Menagerie f3b6013, pelvis welded)", (255, 255, 255), 20),
          "",
          "hardware: NVIDIA GB10 (gb10-direct); MuJoCo 4.0.1; offscreen EGL rendering",
          "task: slide-to-target — push a box with hidden friction (0.3–0.7) into a 3 cm target 15–30 cm away, ≤3 pushes",
          "sensors: synthetic noisy box pose (2 mm / 1°, 20 Hz), hand pose from encoder FK, synthetic tactile contact",
          "",
          ("models shown in video:", (255, 210, 120), 20),
          f"  compact recurrent physical-memory model b11 (structured typed memory + physics loss): {CK['b11']}",
          f"  compact recurrent physical-memory model b00 (generic memory, no physics loss): {CK['b00']}",
          "  (compact models run on CPU in the loop; the simulation pauses during inference)",
          ("Inkling-2-Mini core i00/i11: appears ONLY as a recorded decision log and in the results table", (255, 210, 120), 17),
          "  (actual Inkling-2-Mini core i11 in the loop, sim paused during inference; results/checkpoints/inkling-main-i11-s1.pt)",
          "",
          "all episodes below are real simulator runs; playback 1x; oracle overlays are labelled eval-only"],
         8, sub="final demo · results/demo.mp4")
    clips.append(c)
    manifest["segments"].append(dict(n=1, kind="title card", clip=str(c), seconds=8))
    # 2 held-out episode with b11
    ep_b11 = run_recorded("seed29_b11", b11, "held-out episode · representative fixed seed 29", oracle_overlay=False)
    c = CLIPS / "s2_title.mp4"
    card(c, "2 · held-out slide-to-target episode — representative fixed seed 29",
         ["test seed 29 (held out; chosen before looking at outcomes), hidden friction and mass not given to the model",
          "predictor: compact b11 in the loop; right panel: belief, contact, chosen candidate, memory-update provenance",
          f"outcome: success={ep_b11['success']}, final distance {100*ep_b11['metrics']['final_dist']:.1f} cm, "
          f"{ep_b11['metrics']['n_pushes']} pushes"], 4)
    clips += [c, Path(ep_b11["clip"])]
    manifest["segments"].append(dict(n=2, kind="held-out episode", model="b11", seed=SEED, json=str(OUT / "seed29_b11.json"),
                                     clip=ep_b11["clip"], playback="1x, no cuts"))
    # 3 friction belief before/after first push
    fe = [d.get("friction_estimate") for d in ep_b11["decisions"]]
    lines = [f"true friction (eval-only label): {mu:.3f}", ""]
    for k, f_ in enumerate(fe):
        if f_:
            lines.append((f"decision {k} ({'before any push' if k == 0 else f'after push {k}'}):  FRICTION belief {f_[0]:.3f} ± {math.sqrt(max(f_[1], 0)):.3f}"
                          f"   |error| {abs(f_[0]-mu):.3f}", (120, 220, 255), 20))
    lines += ["", "the first push is the informative interaction (h7): the typed FRICTION slot is read from the model's memory at",
              "each decision. Across 50 paired seeds (slide_compact) b11's mean |error| goes 0.092 -> 0.080 -> 0.071 over decisions",
              "0/1/2 (b00's friction readout: 0.19–0.23). Whether this changes decisions: see segment 5 (slot intervention)."]
    c = CLIPS / "s3_friction.mp4"
    card(c, "3 · held-out surface parameter: b11 friction belief, seed 29", lines, 9,
         sub="before vs after the first (informative) push; true value is an eval-only label")
    clips.append(c)
    manifest["segments"].append(dict(n=3, kind="friction belief", model="b11", seed=SEED, friction_true=mu, estimates=fe,
                                     json=str(OUT / "seed29_b11.json")))
    # 4 paired baseline b00, side by side
    ep_b00 = run_recorded("seed29_b00", b00, "paired baseline · same initial state, seed 29")
    c = CLIPS / "s4_paired.mp4"
    side_by_side(c, ep_b00["clip"], ep_b11["clip"],
                 f"b00 (generic): {'success' if ep_b00['success'] else 'miss'} {100*ep_b00['metrics']['final_dist']:.1f} cm, {ep_b00['metrics']['n_pushes']} pushes",
                 f"b11 (structured+physics): {'success' if ep_b11['success'] else 'miss'} {100*ep_b11['metrics']['final_dist']:.1f} cm, {ep_b11['metrics']['n_pushes']} pushes",
                 "4 · paired baseline from the same initial state: b00 vs b11")
    clips.append(c)
    manifest["segments"].append(dict(n=4, kind="paired baseline side-by-side", models=["b00", "b11"], seed=SEED,
                                     json=[str(OUT / "seed29_b00.json"), str(OUT / "seed29_b11.json")], clip=str(c)))
    # 5 intervention
    intv = FrictionIntervened(b11, wrong)
    ep_int = run_recorded("seed29_b11_friction_slot", intv, f"slot intervention · FRICTION := {wrong:g} (true {mu:.2f})")
    c0 = intv.calls[0] if intv.calls else {}
    ch_n = ep_b11["decisions"][0]["chosen"] if ep_b11["decisions"] else None
    ch_i = ep_int["decisions"][0]["chosen"] if ep_int["decisions"] else None
    lines5 = [f"same initial state (seed 29). b11 typed FRICTION slot overwritten to {wrong:g} via model.intervene before refine",
              f"(true friction {mu:.2f}; normal b11 belief {c0.get('friction_normal', float('nan')):.3f}; readout after intervention "
              f"{c0.get('friction_intervened', float('nan')):.3f})", "",
              (f"decision 0, all feasible candidates: predicted final-position change mean {c0.get('pred_change_mean_cm', float('nan')):.2f} cm, "
               f"max {c0.get('pred_change_max_cm', float('nan')):.2f} cm", (120, 220, 255), 20),
              (f"chosen candidate at decision 0: normal {ch_n}  vs  intervened {ch_i}  -> {'CHANGED' if ch_n != ch_i else 'unchanged'}", (120, 220, 255), 20),
              (f"realized: normal {100*ep_b11['metrics']['final_dist']:.1f} cm ({'success' if ep_b11['success'] else 'miss'})   vs   "
               f"intervened {100*ep_int['metrics']['final_dist']:.1f} cm ({'success' if ep_int['success'] else 'miss'})", (120, 220, 255), 20),
              "", "honest reading: the prediction change from a large friction-slot edit is small — consistent with e07 (the typed",
              "slot is barely used by the state decoder). Next: the intervened episode itself (1x, no cuts)."]
    c = CLIPS / "s5_card.mp4"
    card(c, "5 · relevant-slot intervention: FRICTION slot overwritten vs normal b11", lines5, 9)
    clips += [c, Path(ep_int["clip"])]
    manifest["segments"].append(dict(n=5, kind="FRICTION slot intervention", model="b11", seed=SEED, value=wrong, true=mu,
                                     decision0={k: v for k, v in c0.items() if not k.startswith("_")},
                                     chosen_normal=ch_n, chosen_intervened=ch_i,
                                     realized_normal_cm=100 * ep_b11["metrics"]["final_dist"],
                                     realized_intervened_cm=100 * ep_int["metrics"]["final_dist"],
                                     json=str(OUT / "seed29_b11_friction_slot.json"), clip=ep_int["clip"]))
    # Inkling decision-log panel (no video recorded)
    ik = json.load(open("results/g1/slide_inkling/g1slide-s29__inkling_i11_s1.json"))
    L = ["no video was recorded for the Inkling runs — this panel shows the recorded decision log of the SAME seed",
         "(actual Inkling-2-Mini core i11 in the loop, sim paused during inference; checkpoint results/checkpoints/inkling-main-i11-s1.pt)", ""]
    for d in ik["decisions"]:
        th, ph, sp, du = d["chosen_params"]
        fr = d.get("friction_estimate")
        L.append((f"decision {d['k']}: chosen {d['chosen']} v {sp:.2f} m/s T {du:.2f} s  score {100*d['score']:.1f} cm  "
                  f"latency {d['inference_latency_s']:.2f} s" + (f"  friction {fr[0]:.2f}" if fr else ""), (255, 255, 255), 17))
        L.append((f"     predicted final error {100*d.get('eval_final_pred_err', float('nan')):.1f} cm; distance after push {100*d.get('eval_dist_to_goal_after', float('nan')):.1f} cm", (190, 190, 190), 15))
    m = ik["metrics"]
    L += ["", (f"outcome: success={ik['success']}  final {100*m['final_dist']:.1f} cm  pushes {m['n_pushes']}  real-time factor {m['real_time_factor']:.2f}", (120, 220, 255), 20)]
    c = CLIPS / "s5b_inkling.mp4"
    card(c, "Inkling-2-Mini core i11 on seed 29 — recorded decision log (no video)", L, 8)
    clips.append(c)
    manifest["segments"].append(dict(n="5b", kind="inkling decision-log panel (no video recorded)",
                                     json="results/g1/slide_inkling/g1slide-s29__inkling_i11_s1.json"))
    # 6 end panels
    s30, sc, si = summ("slide30"), summ("slide_compact"), summ("slide_inkling")

    def row(name, s):
        return f"{name:44s} {s['succ']:>3d}/{s['n']:<3d}  final {s['final']:4.1f} cm (median {s['med']:4.1f})  latency {s['lat']:7.1f} ms  RTF {s['rtf']:5.1f}"

    lines6 = [("slide-to-target, paired seeds (success = within 3 cm and still for 0.5 s; ≤3 pushes)", (255, 210, 120), 20)]
    for tag, s in (("slide30 (seeds 0–29)", s30), ("slide_compact (seeds 0–49)", sc), ("slide_inkling (seeds 0–29)", si)):
        lines6.append((tag, (170, 190, 220), 17))
        for k, v in s.items():
            lab = k.replace("compact_compact-main-", "compact ").replace("_p4", "").replace("ORACLE_true_friction_coulomb", "ORACLE (true friction, eval-only)")
            lab = lab.replace("inkling_i", "Inkling core i").replace("stub_prior_coulomb_mu0.5", "stub (prior friction 0.5)")
            lines6.append((row(lab, v), (255, 255, 255), 15))
    lines6 += ["", "integrity: 0 object-state writes, 0 constraints, 0 applied forces in every episode; Inkling/compact pause the sim (RTF shown)"]
    c = CLIPS / "s6_results.mp4"
    card(c, "6 · results (from results/g1 summaries; denominators shown)", lines6, 12)
    clips.append(c)
    lim = [("lift/place gate: FAILED — 3/10 seeded rail trials (needed 8/10) -> PUSH locked as the G1 endpoint", (255, 160, 160), 20),
           "",
           ("limitations", (255, 210, 120), 20),
           "• fixed base: pelvis welded to the world; no balance/locomotion",
           "• synthetic sensors: simulated noisy pose / tactile streams, not real cameras or skin",
           "• embodiment gap: models trained on a sphere pusher (w2); the G1 flat hand has a 2.5 cm pre-contact gap, ~2 s",
           "  approach and 0.83–1.16x speed tracking; fast-push execution scatter ~2–4 cm (controller-v2 attempt not adopted)",
           "• friction not causally used: the oracle (true friction) barely beats a fixed prior; b11's friction belief improves",
           "  with interaction but a large friction-slot edit changes predictions little (e07) and decisions rarely",
           "• no learned model beats the simple prior-friction stub on the G1 slide-to-target benchmark",
           "",
           "data: results/g1/*, results/metrics.jsonl (e09), docs/G1.md · manifest: results/g1/demo_manifest.json"]
    c = CLIPS / "s6_limits.mp4"
    card(c, "6 · gate outcome and limitations", lim, 10)
    clips.append(c)
    manifest["segments"].append(dict(n=6, kind="end panels", sources=["results/g1/slide30/metrics.jsonl",
                                     "results/g1/slide_compact/metrics.jsonl", "results/g1/slide_inkling/metrics.jsonl",
                                     "results/g1/rail_gate.json"], tables=dict(slide30=s30, slide_compact=sc, slide_inkling=si)))
    # concat
    lst = CLIPS / "concat.txt"
    lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in clips))
    subprocess.check_call([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "22", "-r", str(FPS), "results/demo.mp4"])
    dur = float(subprocess.check_output([FF.replace("ffmpeg", "ffprobe") if os.path.exists(FF.replace("ffmpeg", "ffprobe")) else FF,
                                         "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", "results/demo.mp4"]).decode().strip()) \
        if os.path.exists(FF.replace("ffmpeg", "ffprobe")) else None
    manifest["output"] = dict(path="results/demo.mp4", duration_s=dur, size=os.path.getsize("results/demo.mp4"),
                              build_wall_s=round(time.time() - t0, 1), clips=[str(p) for p in clips])
    Path("results/g1/demo_manifest.json").write_text(json.dumps(manifest, indent=1, default=DG._jd))
    print(json.dumps(manifest["output"], indent=1))


if __name__ == "__main__":
    main()
