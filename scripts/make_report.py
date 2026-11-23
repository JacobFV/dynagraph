#!/usr/bin/env python3
"""Build the sprint report (agent REPORT).

    python scripts/make_report.py --results results/ [--out report/] [--config configs/sprint.yaml]
    # or from python: make_report.main(["--results", "results/"])

Reads EVERY result file that exists (all optional):
  results/metrics.jsonl     evaluation rows (evaluate.py / inkling.evaluate_inkling); preferred source
  results/summary.csv       same rows in csv (used for rows missing from metrics.jsonl)
  results/ledger.jsonl      training / suite ledger (compact_train, inkling_train, pilot, inkling_suite, ...)
  results/curves/*.jsonl    learning curves
  results/inkling_gate.json actual-core decision gate
  results/doctor.json       host / library facts
  results/gpu_ledger.log    GPU lock START/END lines (falls back to logs/gpu_ledger.log)
  results/g1/**             G1 episodes (EpisodeResult dicts, {episode,result,trace} dumps, *.jsonl), sweeps, pngs
  results/demo.mp4, results/g1/**/*.mp4
  results/verdicts.json     lead-edited hypothesis verdicts (template written to report/verdicts.json if missing)

Writes report/data.json, report/data.js (same data for file:// viewing), report/index.html (from
report/_template.html with the data inlined) and report/assets/*.png (matplotlib, optional).
Every number shown comes from a saved row; derived numbers (seed means, Wilson intervals, factorial
seed averages) are labelled as derived and list their source rows.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = [f"e{i:02d}" for i in range(26)]
STATUS_ORDER = ["observed", "projected", "target", "skipped", "untested"]
# short titles only where the code base defines what the experiment is; the lead can override in verdicts.json
EXP_TITLES = {
    "e03": "sensor corruption strata (occlusion / delay / drop / rate)",
    "e04": "OOD + held-out strata (friction 0.15/0.9, held-out cells, paddle, w1)",
    "e05": "memory edits (duplicate / stale flood, conflicting source, correction, deletion)",
    "e06": "misspecified laws (velocity-dependent friction, actuation drift)",
    "e07": "typed-slot interventions vs simulator counterfactuals",
    "e08": "actual Inkling-2-Mini core: i00 vs i11 (+ bypass / no-LoRA diagnostics)",
    "e10": "recurrent refinement passes 1/2/4/8 (8 = extrapolation)",
    "e11": "lesions (typed token, edges) + restoration",
    "e14": "relational: added distractor, w3 second object",
    "e16": "uncertainty vs history length (push index k)",
}
HYPOTHESES = [f"h{i}" for i in range(1, 8)]
H_TITLES = {"h7": "uncertainty is high at k=0 (no informative history) and falls with evidence"}
COLORS = {"b00": "#2a78d6", "b10": "#eb6834", "b01": "#1baf7a", "b11": "#eda100", "coulomb": "#e87ba4",
          "constvel": "#6250d6", "coulomb_oracle": "#52514e", "i00": "#008300", "i11": "#e34948"}
EVAL_VERSION = "eval-1"
INKLING_EVAL_STRATA = ("test", "ood_fric_0.15", "ood_fric_0.9", "occ_1.0", "misspec_velfric", "heldout_cells", "paddle")
MAX_REPLAY_FRAMES = 2400


# ----------------------------------------------------------------------------- io helpers

def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def read_jsonl(path):
    out, bad = [], 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
    except OSError:
        pass
    return out, bad


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def fnum(x):
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def config_hash(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def clean(o):
    """json-safe (nan/inf -> None)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


# ----------------------------------------------------------------------------- rows

def tag_table(ledger):
    """config_hash -> eval tag, by recomputing evaluate.py's ev_cfg hash for plausible tags."""
    tags = {"main", "pilot", "pilotpar", "devcheck", "faketest", "smoke", "dev", "test", "final", "wpilot", "wpilot-v1"}
    tags |= {r.get("tag") for r in ledger if r.get("tag")}
    for r in ledger:
        rid = r.get("run_id") or ""
        m = re.match(r"(?:compact|inkling)-(.+)-[bi]\d\d\w*-s\d+$", rid)
        if m:
            tags.add(m.group(1))
    out = {}
    for tag in tags:
        for suite in ("primary", "diagnostics", "ood", "all", "pilot", "main"):
            for nb in (50, 100, 200, 250, 500, 1000, 2000, 5000, 10000):
                for ds in ("v1", "v2", "v0"):
                    out[config_hash(dict(suite=suite, dataset=ds, tag=tag, n_boot=nb, version=EVAL_VERSION))] = \
                        f"{tag}" + ("" if suite == "primary" else f"/{suite}") + ("" if nb >= 1000 else f" (n_boot={nb})")
    out[config_hash(dict(suite="inkling", strata=list(INKLING_EVAL_STRATA), version=EVAL_VERSION))] = "inkling-eval"
    return out


def load_rows(res, ledger, files):
    rows, seen = [], {}
    tags = tag_table(ledger)

    def add(r, src):
        r = dict(r)
        meta_keys = {"experiment", "stratum", "model", "variant", "seed", "metric", "value", "ci_lo", "ci_hi", "n_roots",
                     "status", "config_hash", "run_id", "time_utc", "eval_version"}
        row = dict(exp=str(r.get("experiment", "")), stratum=str(r.get("stratum", "")), model=str(r.get("model", "")),
                   variant=str(r.get("variant", "")), metric=str(r.get("metric", "")),
                   seed=int(fnum(r.get("seed")) if fnum(r.get("seed")) is not None else -1),
                   value=fnum(r.get("value")), lo=fnum(r.get("ci_lo")), hi=fnum(r.get("ci_hi")),
                   n=int(fnum(r.get("n_roots")) or 0), status=str(r.get("status") or "untested"),
                   hash=str(r.get("config_hash") or ""), run_id=r.get("run_id"), t=r.get("time_utc"), src=src)
        row["tag"] = str(r.get("tag") or tags.get(row["hash"], f"unknown:{row['hash']}" if row["hash"] else "unknown"))
        extra = {k: v for k, v in r.items() if k not in meta_keys}
        if extra:
            row["x"] = clean(extra)
        key = (row["hash"], row["exp"], row["stratum"], row["model"], row["variant"], row["seed"], row["metric"], str(r.get("tag") or ""),
               str(r.get("pair_id") or ""), str(r.get("adaptation") or ""))
        if key in seen:
            old = rows[seen[key]]
            if old["src"] == "metrics.jsonl" and src != "metrics.jsonl":
                return
            rows[seen[key]] = row
        else:
            seen[key] = len(rows)
            rows.append(row)

    mp = res / "metrics.jsonl"
    if mp.exists():
        rr, bad = read_jsonl(mp)
        files.append(dict(path="metrics.jsonl", sha=sha12(mp), rows=len(rr), bad_lines=bad))
        for r in rr:
            add(r, "metrics.jsonl")
    base = {(r["hash"], r["exp"], r["stratum"], r["model"], r["variant"], r["seed"], r["metric"]) for r in rows}
    hash_tag = {}
    for r in rows:
        if not r["tag"].startswith("unknown"):
            hash_tag.setdefault(r["hash"], r["tag"])
    sp = res / "summary.csv"
    if sp.exists():
        with open(sp, newline="") as f:
            rr = [r for r in csv.DictReader(f) if r.get("experiment") and r.get("experiment") != "experiment"]
        files.append(dict(path="summary.csv", sha=sha12(sp), rows=len(rr)))
        n_dup = 0
        for r in rr:
            bk = (str(r.get("config_hash") or ""), str(r.get("experiment", "")), str(r.get("stratum", "")), str(r.get("model", "")),
                  str(r.get("variant", "")), int(fnum(r.get("seed")) if fnum(r.get("seed")) is not None else -1), str(r.get("metric", "")))
            if bk in base:  # already present (with full metadata) in metrics.jsonl
                n_dup += 1
                continue
            if not r.get("tag") and bk[0] in hash_tag:
                r = dict(r, tag=hash_tag[bk[0]])
            add(r, "summary.csv")
        files[-1]["dup_of_metrics"] = n_dup
    for r in rows:
        if not r["run_id"]:
            r["run_id"] = f"eval-{r['exp']}-{r['model']}-s{r['seed']}-{r['stratum']}"
        r["tier"] = row_tier(r)
        if r.get("x"):  # keep only the extras the page uses (size)
            r["x"] = {k: v for k, v in r["x"].items() if k in KEEP_EXTRA} or None
            if r["x"] is None:
                del r["x"]
    return rows


KEEP_EXTRA = {"note", "n_seeds", "post_hoc", "adaptation", "label", "lesion", "edit", "passes", "extrapolation", "benchmark",
              "pair_id", "success", "final_dist", "n_pushes", "oracle", "chain_L", "ablation", "extra_model", "ckpt_step",
              "predictor_val_nll", "adapter_data", "world"}


def pack_rows(rows):
    """columnar + interned strings (keeps data.json small); the page unpacks to objects."""
    cols = ["exp", "stratum", "model", "variant", "seed", "metric", "value", "lo", "hi", "n", "status", "hash", "run_id", "tag",
            "tier", "src", "t", "x"]
    strs, idx = [], {}

    def S(v):
        if v is None:
            return None
        if v not in idx:
            idx[v] = len(strs)
            strs.append(v)
        return idx[v]

    def F(v):
        return None if v is None else float(f"{v:.6g}")

    strcols = {"exp", "stratum", "model", "variant", "metric", "status", "hash", "run_id", "tag", "tier", "src", "t"}
    data = []
    for r in rows:
        data.append([S(r.get(c)) if c in strcols else (F(r.get(c)) if c in ("value", "lo", "hi") else r.get(c)) for c in cols])
    return dict(cols=cols, strcols=sorted(strcols), strings=strs, data=data)


def exp_id(e):
    m = re.match(r"^(e\d\d[a-z]*)", e)
    return m.group(1) if m else e


DEV_TAGS = ("devcheck", "faketest", "pilot", "wpilot", "smoke", "dev", "unknown")


def row_tier(r):
    """main | exploratory (post-hoc, experiment ids ending in x / fp) | pilot/dev (engineering checks)."""
    x = r.get("x") or {}
    if x.get("post_hoc") or re.match(r"^e\d\d[a-z]*(x|fp|fpx)$", r["exp"]) or r["tag"].startswith("v3x"):
        return "exploratory"
    t = r["tag"].split("/")[0].split(" ")[0]
    if t.startswith(DEV_TAGS) or t == "test":
        return "pilot/dev"
    return "main"


TIER_RANK = {"main": 0, "exploratory": 1, "pilot/dev": 2}


def tag_rank(tag):
    t = tag.split("/")[0].split(" ")[0]
    if t.startswith("v3x"):
        return 1
    if t.startswith(DEV_TAGS) or t == "test":
        return 2
    return 0


# ----------------------------------------------------------------------------- G1 episodes

def _get(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None


def _xy(v):
    if v is None:
        return None
    if isinstance(v, dict):
        x, y = _get(v, "x"), _get(v, "y")
        if x is None and "pos" in v:
            return _xy(v["pos"])
        return [float(x), float(y)] if x is not None and y is not None else None
    if isinstance(v, (list, tuple)) and len(v) >= 2 and all(isinstance(a, (int, float)) for a in v[:2]):
        return [float(v[0]), float(v[1])]
    return None


def _canon_to_world(frame, dx, dy):
    x0, y0, yaw0 = frame
    c, s = math.cos(yaw0), math.sin(yaw0)
    return [x0 + c * dx - s * dy, y0 + s * dx + c * dy]


def normalize_episode(obj, src, want_replay=True):
    """returns (summary, replay|None) or None if obj is not an episode record."""
    if not isinstance(obj, dict):
        return None
    res = obj.get("result") if isinstance(obj.get("result"), dict) else obj
    decisions = _get(res, "decisions") or _get(obj, "decisions") or []
    trace = _get(obj, "steps", "trace", "frames") or _get(res, "steps", "trace", "frames") or []
    if not isinstance(trace, list):
        trace = []
    if not decisions and len(trace) < 2:
        return None
    if "success" not in res and not decisions and not any(isinstance(s, dict) and "t" in s for s in trace[:3]):
        return None
    spec = _get(obj, "episode", "spec", "episode_spec") or _get(res, "episode", "spec") or {}
    if not isinstance(spec, dict):
        spec = {}
    met = res.get("metrics") if isinstance(res.get("metrics"), dict) else {}
    model = str(_get(res, "model") or _get(obj, "model") or "unknown")
    seed = _get(res, "seed") if _get(res, "seed") is not None else _get(obj, "seed")
    eid = str(_get(res, "episode_id") or _get(obj, "episode_id") or f"{Path(src).stem}")
    tag = str(_get(obj, "tag") or _get(res, "tag") or (Path(src).parent.name if "/" in src else ""))
    summary = dict(id=eid, model=model, seed=seed, tag=tag, success=res.get("success"), status=met.get("status"),
                   n_pushes=met.get("n_pushes", len(decisions)), final_dist=fnum(met.get("final_dist")),
                   still=met.get("still"), clean=met.get("clean"), sim_time=fnum(met.get("sim_time")),
                   wall_time=fnum(met.get("wall_time")), rtf=fnum(met.get("real_time_factor")),
                   latency=fnum(met.get("mean_inference_latency")), audit=met.get("audit"),
                   hidden=met.get("hidden_eval_only"), pair=_get(obj, "pair_id", "pair") or _get(res, "pair_id"),
                   benchmark=met.get("benchmark") or _get(obj, "benchmark"),
                   variant=_get(obj, "variant", "condition") or spec.get("variant"), src=src)
    # decisions log (compact)
    dl = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        ch = d.get("chosen")
        cands = d.get("candidates") or []
        chc = next((c for c in cands if c.get("id") == ch), {})
        dl.append(dict(k=d.get("k"), t=fnum(d.get("belief_time")), rev=d.get("belief_revision"),
                       belief=[fnum(v) for v in (d.get("belief_pose") or [])][:4], chosen=ch, model_top=d.get("model_top"),
                       params=d.get("chosen_params"), score=fnum(d.get("score")), pred_dist=fnum(chc.get("pred_dist")),
                       pred_sd=fnum(chc.get("pred_sd")), n_cand=len(cands), n_feas=sum(1 for c in cands if c.get("feasible")),
                       ik_rejected=d.get("ik_rejected"), friction=d.get("friction_estimate"),
                       latency=fnum(d.get("inference_latency_s")), err=d.get("pred_pos_err_at_horizons"),
                       adapter=d.get("tensor_adapter")))
    summary["decisions"] = dl if want_replay else []
    summary["n_decisions"] = len(dl)
    if not want_replay:
        return clean(summary), None
    # ---- replay
    frames = []
    for s in trace:
        if not isinstance(s, dict) or fnum(s.get("t")) is None:
            continue
        bx = s.get("box")
        if isinstance(bx, (list, tuple)) and len(bx) >= 4:  # G1 dumps: box = [x, y, z, yaw] (simulator truth)
            obj_xy, yaw = [float(bx[0]), float(bx[1])], float(bx[3])
        else:
            obj_xy = _xy(_get(s, "obj", "object", "true_obj")) or ([float(s["x"]), float(s["y"])] if "x" in s and "y" in s else None)
            if obj_xy is None:
                continue
            o = _get(s, "obj", "object", "true_obj")
            yaw = fnum(s.get("yaw")) if s.get("yaw") is not None else fnum(_get(o, "yaw") if isinstance(o, dict) else None)
        fr = dict(t=float(s["t"]), o=[round(obj_xy[0], 5), round(obj_xy[1], 5), round(yaw or 0.0, 4)])
        hand = _xy(_get(s, "hand", "hand_xy", "pusher", "pusher_xy", "effector", "ee"))
        if hand is None and "hand_x" in s:
            hand = [float(s["hand_x"]), float(s["hand_y"])]
        if hand:
            fr["h"] = [round(hand[0], 5), round(hand[1], 5)]
        if fnum(s.get("f_hand")) is not None:
            fr["f"] = round(float(s["f_hand"]), 3)
        b = _get(s, "belief", "belief_pose", "belief_mean")
        if b is not None:
            bxy = _xy(b)
            bv = _get(s, "belief_var", "belief_cov") or (_get(b, "var") if isinstance(b, dict) else None)
            if bxy:
                fr["b"] = [round(bxy[0], 5), round(bxy[1], 5)]
                if isinstance(bv, (list, tuple)) and len(bv) >= 2:
                    fr["bv"] = [float(bv[0]), float(bv[1])] if not isinstance(bv[0], list) else [float(bv[0][0]), float(bv[1][1])]
        if s.get("phase"):
            fr["p"] = str(s["phase"])
        frames.append(fr)
    if len(frames) > MAX_REPLAY_FRAMES:
        stride = math.ceil(len(frames) / MAX_REPLAY_FRAMES)
        frames = frames[::stride]
    replay = None
    if len(frames) >= 2:
        has_hand = any("h" in f for f in frames)
        rdec = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
            ex = d.get("execution") or {}
            frame = d.get("canonical_frame")
            pred, pred_sd, real = [], [], []
            hz = d.get("horizons") or []
            if d.get("predicted_world_xy"):
                pred = [[float(a), float(b)] for a, b in d["predicted_world_xy"]]
                pred_sd = [fnum(v) for v in (d.get("predicted_sd_xy") or [None] * len(pred))]
                if frame:
                    real = [_canon_to_world(frame, rr[0], rr[1]) if rr else None for rr in d.get("realized_canonical_eval_only") or []]
            elif frame and d.get("predicted_mean"):
                for i, m in enumerate(d["predicted_mean"]):
                    pred.append(_canon_to_world(frame, m[0], m[1]))
                    v = (d.get("predicted_var") or [None] * len(d["predicted_mean"]))[i]
                    pred_sd.append(math.sqrt(max(v[0], 0) + max(v[1], 0)) if v else None)
                for rr in d.get("realized_canonical_eval_only") or []:
                    real.append(_canon_to_world(frame, rr[0], rr[1]) if rr else None)
            bp = d.get("belief_pose") or []
            bv = d.get("belief_pose_var") or []
            cands = []
            for c in d.get("candidates") or []:
                if c.get("pred_world_xy") is None:
                    continue
                cands.append(dict(id=c.get("id"), xy=[round(v, 5) for v in c["pred_world_xy"]], s=fnum(c.get("score")),
                                  sd=fnum(c.get("pred_sd")), f=bool(c.get("feasible"))))
            mem = _get(d, "memory_updates", "provenance", "memory_log") or []
            if isinstance(mem, dict):
                mem = mem.get("records") or mem.get("updates") or []
            rdec.append(clean(dict(k=d.get("k"), t=fnum(d.get("belief_time")), rev=d.get("belief_revision"),
                                   belief=bp[:2], yaw=math.atan2(bp[3], bp[2]) if len(bp) >= 4 else None, var=bv[:2],
                                   frame=frame, chosen=d.get("chosen"), params=d.get("chosen_params"),
                                   t_start=fnum(ex.get("t_start")), t_push=fnum(ex.get("t_push")), t_end=fnum(ex.get("t_push_end")),
                                   h0=_xy(ex.get("start")), h1=_xy(ex.get("end")), hz=hz, pred=pred, pred_sd=pred_sd, real=real,
                                   cands=cands, friction=d.get("friction_estimate"), score=fnum(d.get("score")),
                                   latency=fnum(d.get("inference_latency_s")), mem=mem[:200] if isinstance(mem, list) else [],
                                   ik_rejected=d.get("ik_rejected"))))
        box = spec.get("box") or {}
        mem_log = _get(obj, "memory_updates", "provenance", "memory_log") or []
        replay = clean(dict(id=eid, model=model, seed=seed, success=res.get("success"), src=src, frames=frames,
                            hand_source="logged" if has_hand else ("reconstructed from executed push plan (start/end, t_push..t_push_end)"
                                                                   if any(d.get("h0") for d in rdec) else "not logged"),
                            half=box.get("half"), target=spec.get("target"), target_radius=spec.get("target_radius"),
                            rail=spec.get("rail_xy"), tray=spec.get("tray_xy"), tray_half=spec.get("tray_half"),
                            decisions=rdec, memory=mem_log[:2000] if isinstance(mem_log, list) else [],
                            hidden=met.get("hidden_eval_only"), final_dist=summary["final_dist"]))
    return clean(summary), replay


REPLAY_GLOBS = ("slide_compact/*s29__*.json", "slide_compact_g1cal/*s29__*.json", "slide_inkling/*s29__*inkling_i11*.json")
SKIP_G1 = ("rail_gate.json", "speed_tracking.json", "demo_manifest.json")


def load_g1(res, assets, files, notes, replay_globs=REPLAY_GLOBS):
    out = dict(episodes=[], replays=[], sweep=None, baseline=None, images=[], videos=[], other=[])
    g1dir = res / "g1"
    want = set()
    for gpat in replay_globs:
        want |= {os.path.normpath(p) for p in glob.glob(str(g1dir / gpat))}
    paths = sorted(glob.glob(str(g1dir / "**" / "*.json"), recursive=True)) + \
        sorted(glob.glob(str(g1dir / "**" / "*.jsonl"), recursive=True))
    n_files = 0
    for p in paths:
        rel = os.path.relpath(p, res)
        name = Path(p).name
        if name == "friction_sweep.json":
            out["sweep"] = clean(read_json(p))
            files.append(dict(path=rel, sha=sha12(p)))
            continue
        if name == "baseline.json":
            b = read_json(p) or {}
            out["baseline"] = clean({k: b.get(k) for k in ("utc", "displacement_m", "dyaw", "max_hand_force", "audit", "candidate", "box_start", "box_end")})
            files.append(dict(path=rel, sha=sha12(p)))
            continue
        if name in SKIP_G1 or "servo_calib" in rel:
            out["other"].append(rel)
            continue
        if p.endswith(".jsonl"):
            objs, _ = read_jsonl(p)
        else:
            o = read_json(p)
            objs = o if isinstance(o, list) and o and isinstance(o[0], dict) and ("decisions" in o[0] or "result" in o[0]) else [o]
        n_files += 1
        sel = os.path.normpath(p) in want or "/demo/" in p
        n_ep = 0
        for i, o in enumerate(objs):
            r = normalize_episode(o, rel + (f"#{i}" if len(objs) > 1 else ""), want_replay=sel)
            if r is None:
                continue
            s_, rp = r
            s_["src_file"] = rel
            out["episodes"].append(s_)
            if rp is not None and os.path.normpath(p) in want:
                out["replays"].append(rp)
            n_ep += 1
        if n_ep == 0:
            out["other"].append(rel)
    files.append(dict(path="g1/**/*.json (episode files, hashes not listed individually)", sha="-", rows=n_files))
    for p in sorted(glob.glob(str(g1dir / "**" / "*.png"), recursive=True)):
        rel = os.path.relpath(p, res)
        dst = assets / ("g1_" + rel.replace("/", "_").replace("g1_", "", 1))
        shutil.copyfile(p, dst)
        out["images"].append(dict(src=rel, asset=f"assets/{dst.name}"))
    # videos are linked, not copied: report/ sits next to results/ in the repo
    for stale in glob.glob(str(assets / "video_*.mp4")):  # older versions copied videos; now linked
        os.remove(stale)
    vids = sorted(set(glob.glob(str(res / "*.mp4")) + glob.glob(str(g1dir / "**" / "*.mp4"), recursive=True)))
    for p in vids:
        rel = os.path.relpath(p, res)
        out["videos"].append(dict(src=rel, href=f"../results/{rel}", mb=round(os.path.getsize(p) / 1e6, 1)))
    # success rates per (tag, model) and pairs per (tag, pair_id) - derived from episode records
    by = defaultdict(list)
    for e in out["episodes"]:
        by[(e.get("tag") or "", e["model"])].append(e)
    rates = []
    for (tg, m), eps in sorted(by.items()):
        k = sum(1 for e in eps if e.get("success"))
        n = sum(1 for e in eps if e.get("success") is not None)
        lo, hi = wilson(k, n)
        fd = sorted(e["final_dist"] for e in eps if e.get("final_dist") is not None)
        rates.append(dict(tag=tg, model=m, k=k, n=n, rate=k / n if n else None, lo=lo, hi=hi,
                          median_final_dist=fd[len(fd) // 2] if fd else None,
                          mean_pushes=sum(e.get("n_pushes") or 0 for e in eps) / len(eps) if eps else None))
    out["rates"] = rates
    pairs = defaultdict(dict)
    for e in out["episodes"]:
        key = e.get("pair") or e["id"]
        pairs[(e.get("tag") or "", str(key))][e["model"]] = dict(success=e.get("success"), final_dist=e.get("final_dist"),
                                                                 n_pushes=e.get("n_pushes"), status=e.get("status"))
    out["pairs"] = [dict(tag=t, key=k, by_model=v) for (t, k), v in sorted(pairs.items(), key=lambda kv: (kv[0][0], _natkey(kv[0][1])))]
    out["tags"] = sorted({e.get("tag") or "" for e in out["episodes"]})
    out["models"] = sorted({e["model"] for e in out["episodes"]})
    return out


def _natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ----------------------------------------------------------------------------- curves / ledger / gpu

def load_curves(res, files):
    curves = {}
    for p in sorted(glob.glob(str(res / "curves" / "*.jsonl"))):
        stem = Path(p).stem
        if ".prev" in stem:
            continue
        rr, _ = read_jsonl(p)
        files.append(dict(path=os.path.relpath(p, res), sha=sha12(p), rows=len(rr)))
        keep = ("step", "examples", "elapsed_s", "nll", "mse_norm", "friction_abs_err", "total", "coulomb", "kinematic",
                "read_gate", "write_gate", "grad_norm", "passes")
        tr = [{k: r[k] for k in keep if k in r} for r in rr if r.get("split") == "train"]
        va = [{k: r[k] for k in keep if k in r} for r in rr if r.get("split") == "val"]
        if len(tr) > 300:
            tr = tr[:: math.ceil(len(tr) / 300)]
        curves[stem] = clean(dict(train=tr, val=va))
    return curves


def parse_gpu_ledger(paths, files):
    for p in paths:
        if p.exists():
            files.append(dict(path=str(p.name), sha=sha12(p)))
            ev = []
            with open(p) as f:
                for line in f:
                    m = re.match(r"\[(\S+)\]\s+(START|END)\s+(\S+)(?:\s+rc=(\S+))?", line.strip())
                    if m:
                        ev.append(dict(t=m.group(1), kind=m.group(2), name=m.group(3), rc=m.group(4)))
            spans, open_ = [], {}
            for e in ev:
                if e["kind"] == "START":
                    open_[e["name"]] = e["t"]
                else:
                    spans.append(dict(name=e["name"], start=open_.pop(e["name"], None), end=e["t"], rc=e["rc"]))
            for n, t in open_.items():
                spans.append(dict(name=n, start=t, end=None, rc="running?"))
            for s in spans:
                s["dur_s"] = (iso(s["end"]) - iso(s["start"])) if s["start"] and s["end"] else None
            return dict(source=str(p), spans=spans)
    return None


def iso(t):
    try:
        return time.mktime(time.strptime(t, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- verdicts

def load_verdicts(res, out, notes):
    for p in (res / "verdicts.json", out / "verdicts.json"):
        v = read_json(p)
        if isinstance(v, dict):
            return v, str(p)
        if p.exists():
            notes.append(f"{p} exists but is not valid json; using template")
    tmpl = dict(
        _readme="Edited by the lead. verdict in {supported, unsupported, contradicted, untested}. evidence: list of row filters "
                "(keys exp/stratum/model/variant/seed/metric/tag; '*' suffix = prefix match) whose rows are shown next to the verdict. "
                "experiments.<id>: optional title / status override (observed|projected|target|skipped|untested) / note. "
                "Copy to results/verdicts.json to make it travel with the results.",
        hypotheses={h: dict(title=H_TITLES.get(h, ""), verdict="untested", rationale="", evidence=[]) for h in HYPOTHESES},
        experiments={e: dict(title=EXP_TITLES.get(e, ""), status=None, note="") for e in EXPERIMENTS},
        limitations=[],
    )
    p = out / "verdicts.json"
    p.write_text(json.dumps(tmpl, indent=1))
    notes.append(f"verdicts template written to {p} (all untested)")
    return tmpl, str(p) + " (template)"


def match_filter(r, flt):
    for k, v in flt.items():
        if k == "note":
            continue
        rv = str(r.get(k, ""))
        v = str(v)
        if v.endswith("*"):
            if not rv.startswith(v[:-1]):
                return False
        elif rv != v:
            return False
    return True


# ----------------------------------------------------------------------------- plots

def make_plots(D, rows, assets, notes):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        notes.append(f"matplotlib unavailable ({e.__class__.__name__}); PNG plots skipped (tables still complete)")
        return {}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                         "axes.labelcolor": "#2b2b2a", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "figure.dpi": 110, "savefig.bbox": "tight", "lines.linewidth": 2, "legend.frameon": False})
    made = {}

    def color(model, variant):
        if model.startswith("inkling-"):
            return COLORS.get(variant[:3], "#e34948")
        return COLORS.get(variant if model == "compact" else model, "#8a8984")

    def label(model, variant, seed=None):
        if model == "compact":
            s = f"compact {variant}"
        elif model == "coulomb_oracle":
            s = "coulomb ORACLE (true friction)"
        elif model.startswith("inkling-"):
            s = f"actual Inkling core {model[8:]}"
        else:
            s = f"{model} (analytic)"
        return s if seed is None or seed < 0 else f"{s} s{seed}"

    def save(fig, name, title):
        p = assets / f"{name}.png"
        fig.savefig(p)
        plt.close(fig)
        made[name] = dict(asset=f"assets/{name}.png", title=title)

    def best_tag(rs):
        if not rs:
            return None
        return min({r["tag"] for r in rs}, key=lambda t: (tag_rank(t), t))

    def series(rs, metric, strata):
        """(model, variant) -> list per stratum of (mean over seeds, lo, hi, nseeds) using the single seed's CI when n=1."""
        acc = defaultdict(lambda: defaultdict(list))
        for r in rs:
            if r["metric"] == metric and r["stratum"] in strata and r["value"] is not None:
                acc[(r["model"], r["variant"])][r["stratum"]].append(r)
        out = {}
        for key, d in acc.items():
            pts = []
            for st in strata:
                lst = d.get(st, [])
                if not lst:
                    pts.append(None)
                    continue
                vals = [x["value"] for x in lst]
                m = sum(vals) / len(vals)
                if len(lst) == 1:
                    pts.append((m, lst[0]["lo"], lst[0]["hi"], 1))
                else:
                    pts.append((m, min(vals), max(vals), len(lst)))
            out[key] = pts
        return out

    def cat_plot(rs, strata, metrics, name, title, xlabels=None, logy=False, connect=False):
        if not rs:
            return
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.2), squeeze=False)
        any_pts = False
        for ax, (metric, ylab) in zip(axes[0], metrics):
            ser = series(rs, metric, strata)
            keys = sorted(ser, key=lambda k: (k[0] != "compact", k[0].startswith("inkling"), k))
            nk = max(1, len(keys))
            for j, key in enumerate(keys):
                pts = ser[key]
                xs = [i + (j - (nk - 1) / 2) * 0.6 / nk for i in range(len(strata))]
                xx = [x for x, p in zip(xs, pts) if p]
                yy = [p[0] for p in pts if p]
                if not xx:
                    continue
                any_pts = True
                el = [[max(0, p[0] - p[1]) if p[1] is not None else 0 for p in pts if p],
                      [max(0, p[2] - p[0]) if p[2] is not None else 0 for p in pts if p]]
                ls = "--" if key[0] == "coulomb_oracle" else "-"
                ax.errorbar(xx, yy, yerr=el, fmt="o", ms=4, color=color(*key), capsize=2, elinewidth=1, label=label(*key))
                if connect and len(xx) > 1:
                    ax.plot(xx, yy, ls, color=color(*key), lw=1, alpha=0.6)
            ax.set_xticks(range(len(strata)))
            ax.set_xticklabels(xlabels or strata, rotation=30 if len(strata) > 4 else 0, ha="right" if len(strata) > 4 else "center")
            ax.set_ylabel(ylab)
            if logy:
                ax.set_yscale("log")
        if not any_pts:
            plt.close(fig)
            return
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.tight_layout()
        nrow = math.ceil(len(lab) / 4)
        fig.legend(h, lab, loc="lower center", ncol=min(4, len(lab)), bbox_to_anchor=(0.5, 1.0), fontsize=8)
        fig.suptitle(title, y=1.1 + 0.06 * nrow, fontsize=10)
        save(fig, name, title + f" [eval tag: {best_tag(rs)}]")

    # ---- primary: error vs horizon
    prim = [r for r in rows if r["exp"] == "primary" and r["stratum"] == "test"]
    t = best_tag(prim)
    prim = [r for r in prim if r["tag"] == t]
    if prim:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
        hs = ["0.1", "0.5", "1.0"]
        for ax, (m, yl) in zip(axes, (("pos_err_m", "position error (m)"), ("yaw_err_rad", "yaw error (rad)"), ("vel_err_mps", "velocity error (m/s)"))):
            acc = defaultdict(lambda: defaultdict(list))
            for r in prim:
                mm = re.match(rf"^{m}@h(\S+)$", r["metric"])
                if mm and mm.group(1) in hs and r["value"] is not None:
                    acc[(r["model"], r["variant"])][mm.group(1)].append(r)
            for key, d in sorted(acc.items()):
                xs, ys, lo, hi = [], [], [], []
                for i, h in enumerate(hs):
                    if d.get(h):
                        v = [x["value"] for x in d[h]]
                        xs.append(float(h))
                        ys.append(sum(v) / len(v))
                        lo.append(d[h][0]["lo"] if len(v) == 1 else min(v))
                        hi.append(d[h][0]["hi"] if len(v) == 1 else max(v))
                ax.plot(xs, ys, "--" if key[0] == "coulomb_oracle" else "-", marker="o", ms=4, color=color(*key), label=label(*key))
                if all(a is not None for a in lo + hi):
                    ax.fill_between(xs, lo, hi, color=color(*key), alpha=0.12, lw=0)
            ax.set_xlabel("horizon after push start (s)")
            ax.set_ylabel(yl)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13), fontsize=8)
        save(fig, "primary_horizons", f"test stratum: error vs horizon (bands = 95% bootstrap CI over root episodes, single seed; min-max across seeds otherwise)")
    # ---- factorial effects forest
    eff = [r for r in rows if re.match(r"^(structure_main|physics_main|interaction):", r["metric"]) and r["value"] is not None]
    if eff:
        t = best_tag(eff)
        eff = [r for r in eff if r["tag"] == t and r["stratum"] == "test"] or [r for r in eff if r["tag"] == t]
        metrics = sorted({r["metric"].split(":", 1)[1] for r in eff})
        fig, axes = plt.subplots(1, 3, figsize=(12, 0.35 * len(metrics) + 1.2), sharey=True)
        for ax, en in zip(axes, ("structure_main", "physics_main", "interaction")):
            for i, m in enumerate(metrics):
                r = next((x for x in eff if x["metric"] == f"{en}:{m}"), None)
                if r:
                    ax.errorbar([r["value"]], [i], xerr=[[r["value"] - (r["lo"] or r["value"])], [(r["hi"] or r["value"]) - r["value"]]],
                                fmt="o", color="#2a78d6", ms=4, capsize=2)
            ax.axvline(0, color="#8a8984", lw=1)
            ax.set_title(en.replace("_", " "))
            ax.set_yticks(range(len(metrics)))
            ax.set_yticklabels(metrics)
            ax.set_xscale("symlog", linthresh=1e-3)
        save(fig, "factorial_effects", f"factorial effects (seed-averaged per root, 95% bootstrap CI over roots; negative = lower metric)")
    # ---- e03 sensor strata
    e03 = [r for r in rows if r["exp"] == "e03"]
    if e03:
        t = best_tag(e03)
        e03 = [r for r in e03 if r["tag"] == t]
        for fac, strata in (("occ", ["occ_0.0", "occ_0.2", "occ_0.5", "occ_1.0"]), ("delay", ["delay_0", "delay_1", "delay_3"]),
                            ("drop", ["drop_0.0", "drop_0.25", "drop_0.5"]), ("rate", ["rate_10", "rate_20", "rate_40"])):
            cat_plot(e03, strata, [("pos_err_m@h0.5", "pos err @0.5 s (m)"), ("nll@all", "NLL (norm.)"), ("cover80@all", "80% coverage")],
                     f"e03_{fac}", f"e03 sensor corruption: {fac}", connect=True)
    e04 = [r for r in rows if r["exp"] == "e04" and "_k" not in r["stratum"]]
    if e04:
        t = best_tag(e04)
        cat_plot([r for r in e04 if r["tag"] == t], ["heldout_cells", "ood_fric_0.15", "ood_fric_0.9", "paddle", "w1"],
                 [("pos_err_m@h0.5", "pos err @0.5 s (m)"), ("nll@all", "NLL"), ("friction_abs_err", "|friction err|")], "e04", "e04 OOD / held-out strata")
    e06 = [r for r in rows if r["exp"] == "e06"]
    if e06:
        t = best_tag(e06)
        cat_plot([r for r in e06 if r["tag"] == t], ["misspec_velfric", "misspec_drift"],
                 [("pos_err_m@h0.5", "pos err @0.5 s (m)"), ("nll@all", "NLL"), ("cover80@all", "80% coverage")], "e06", "e06 misspecified laws")
    e05 = [r for r in rows if r["exp"] == "e05"]
    if e05:
        t = best_tag(e05)
        e05 = [r for r in e05 if r["tag"] == t]
        st = ["test_duplicate_flood", "test_stale_flood", "test_conflicting_source", "test_authoritative_correction", "test_delete_velocity"]
        cat_plot(e05, st, [("delta_pos_err_m@all", "Δ pos err (m)"), ("delta_nll@all", "Δ NLL"), ("pred_change_pos_m@all", "prediction change (m)")],
                 "e05", "e05 memory edits (paired deltas vs unedited)", xlabels=[s[5:] for s in st])
    e10 = [r for r in rows if r["exp"] == "e10" and re.match(r"^test_passes\d+$", r["stratum"])]
    if e10:
        t = best_tag(e10)
        st = ["test_passes1", "test_passes2", "test_passes4", "test_passes8"]
        cat_plot([r for r in e10 if r["tag"] == t], st, [("nll@all", "NLL"), ("pos_err_m@h0.5", "pos err @0.5 s (m)"), ("friction_abs_err", "|friction err|")],
                 "e10", "e10 refinement passes (8 = extrapolation beyond training {1,2,4})", xlabels=["1", "2", "4", "8*"], connect=True)
    e11 = [r for r in rows if r["exp"] == "e11" and r["stratum"].startswith("test_lesion")]
    if e11:
        t = best_tag(e11)
        st = ["test_lesion_typed_friction_token", "test_lesion_random_token", "test_lesion_edge_target_pusher", "test_lesion_edge_random_other"]
        cat_plot([r for r in e11 if r["tag"] == t], st, [("delta_pos_err_m@all", "Δ pos err (m)"), ("delta_nll@all", "Δ NLL"), ("pred_change_pos_m", "prediction change (m)")],
                 "e11", "e11 lesions (targeted vs matched random)", xlabels=["typed friction", "random token", "edge tgt-pusher", "random edge"])
    e14 = [r for r in rows if r["exp"] == "e14"]
    if e14:
        t = best_tag(e14)
        cat_plot([r for r in e14 if r["tag"] == t], ["test_add_distractor"], [("delta_pos_err_m@all", "Δ pos err (m)"), ("delta_nll@all", "Δ NLL")],
                 "e14", "e14 added static distractor (paired deltas)", xlabels=["add distractor"])
    e16 = [r for r in rows if r["exp"] == "e16"]
    if e16:
        t = best_tag(e16)
        cat_plot([r for r in e16 if r["tag"] == t], ["test_k0", "test_k1", "test_k2", "test_k3"],
                 [("friction_abs_err", "|friction err|"), ("friction_std", "predicted friction sd"), ("nll@all", "NLL")],
                 "e16", "e16 vs push index k (history length)", xlabels=["k=0", "k=1", "k=2", "k=3"], connect=True)
    e07 = [r for r in rows if r["exp"] == "e07" and r["value"] is not None]
    if e07:
        t = best_tag(e07)
        e07 = [r for r in e07 if r["tag"] == t]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for key in sorted({(r["model"], r["variant"], r["seed"]) for r in e07}):
            for nm, ls in (("friction", "-"), ("mass_nuisance", ":")):
                pp = sorted([(float(re.search(r"@h(\S+)", r["metric"]).group(1)), r) for r in e07
                             if (r["model"], r["variant"], r["seed"]) == key and r["metric"].startswith(f"{nm}:pred_change_pos_m@h")])
                ss = sorted([(float(re.search(r"@h(\S+)", r["metric"]).group(1)), r) for r in e07
                             if (r["model"], r["variant"], r["seed"]) == key and r["metric"].startswith(f"{nm}:sim_change_pos_m@h")])
                if pp:
                    ax.plot([a for a, _ in pp], [r["value"] for _, r in pp], ls, marker="o", ms=4, color=color(key[0], key[1]),
                            label=f"{label(key[0], key[1], key[2])} predicted ({nm})")
                if ss and key == sorted({(r["model"], r["variant"], r["seed"]) for r in e07})[0]:
                    ax.plot([a for a, _ in ss], [r["value"] for _, r in ss], ls, marker="s", ms=4, color="#0b0b0b",
                            label=f"simulator counterfactual ({nm}) [ground truth]")
        ax.set_xlabel("horizon (s)")
        ax.set_ylabel("|Δ displacement| (m)")
        ax.legend(fontsize=7)
        save(fig, "e07", "e07 intervention: friction 0.35→0.65 (mass 0.4→1.2 nuisance) predicted vs simulated displacement change")
    # ---- inkling
    ink = [r for r in rows if r["model"].startswith("inkling-") and r["exp"] in ("e08", "e08_diag")]
    if ink:
        cat_plot(ink, list(INKLING_EVAL_STRATA), [("pos_err_m@h0.5", "pos err @0.5 s (m)"), ("nll@all", "NLL")], "inkling_strata",
                 "actual Inkling-2-Mini core: i00 vs i11 (+ bypass / no-LoRA on test)")
    # ---- learning curves
    for grp, title in (("compact", "compact recurrent physical-memory model: validation NLL"), ("inkling", "actual Inkling-2-Mini core (adapters): validation NLL")):
        cs = {k: v for k, v in D["curves"].items() if k.startswith(grp + "-") and v.get("val")}
        if not cs:
            continue
        tags = sorted({re.sub(r"-[bi]\d\d\w*-s\d+$", "", k) for k in cs})
        nc = min(4, len(tags))
        nr = math.ceil(len(tags) / nc)
        fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), squeeze=False)
        for ax in axes.flat[len(tags):]:
            ax.set_visible(False)
        for ax, tg in zip(axes.flat, tags):
            for k, v in sorted(cs.items()):
                if re.sub(r"-[bi]\d\d\w*-s\d+$", "", k) != tg:
                    continue
                var = re.search(r"-([bi]\d\d)\w*-s\d+$", k).group(1)
                xk = "elapsed_s" if grp == "inkling" else "step"
                ax.plot([p.get(xk) for p in v["val"]], [p.get("nll") for p in v["val"]], color=COLORS.get(var, "#888"), marker=".", label=k.split(tg + "-")[-1])
            ax.set_title(tg, fontsize=9)
            ax.set_xlabel("elapsed (s)" if grp == "inkling" else "step")
            ax.set_ylabel("val NLL")
            ax.legend(fontsize=7)
        fig.tight_layout()
        save(fig, f"curves_{grp}", title)
    # ---- gpu timeline
    g = D["cost"].get("gpu")
    if g and g["spans"]:
        spans = [s for s in g["spans"] if s["start"]]
        t0 = min(iso(s["start"]) for s in spans)
        now = time.time()
        fig, ax = plt.subplots(figsize=(10, 0.3 * len(spans) + 1))
        for i, s in enumerate(spans):
            a = (iso(s["start"]) - t0) / 3600
            b = ((iso(s["end"]) if s["end"] else now) - t0) / 3600
            ax.barh(i, b - a, left=a, color="#2a78d6" if s["rc"] == "0" else ("#eda100" if not s["end"] else "#e34948"), height=0.6)
        ax.set_yticks(range(len(spans)))
        ax.set_yticklabels([s["name"] for s in spans], fontsize=7)
        ax.set_xlabel(f"hours since {spans[0]['start']} (UTC)")
        ax.invert_yaxis()
        save(fig, "gpu_timeline", "GPU lock timeline (blue rc=0, red failed, yellow still open)")
    # ---- g1 friction sweep
    sw = D["g1"].get("sweep")
    if sw:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for i, th in enumerate(sorted({r["theta"] for r in sw})):
            g_ = sorted([r for r in sw if r["theta"] == th], key=lambda r: r["friction"])
            ax.plot([r["friction"] for r in g_], [r["disp"] for r in g_], marker="o", ms=4,
                    color=list(COLORS.values())[i % 8], label=f"θ={th:.2f} v={g_[0]['speed']} T={g_[0]['duration']}")
        ax.set_xlabel("hidden friction μ (simulator)")
        ax.set_ylabel("final box displacement (m)")
        ax.legend(fontsize=7)
        save(fig, "g1_friction_sweep", "G1 push primitive: hidden-friction sensitivity (no learned model)")
    return made


# ----------------------------------------------------------------------------- main

def _inline_md(t):
    import html as H

    t = H.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', t)
    return t


def md_to_html(md):
    """small markdown subset: headings, lists, tables, code fences, paragraphs, inline code/bold/italic/links."""
    out, para, lst, tbl, code = [], [], None, [], None

    def flush():
        nonlocal para, lst, tbl
        if para:
            out.append("<p>" + _inline_md(" ".join(para)) + "</p>")
            para = []
        if lst:
            tag, items = lst
            out.append(f"<{tag}>" + "".join(f"<li>{_inline_md(i)}</li>" for i in items) + f"</{tag}>")
            lst = None
        if tbl:
            rows_ = [[c.strip() for c in r.strip().strip("|").split("|")] for r in tbl if not re.match(r"^\s*\|?\s*:?-{2,}", r)]
            h = "<div class='tbl'><table><thead><tr>" + "".join(f"<th>{_inline_md(c)}</th>" for c in rows_[0]) + "</tr></thead><tbody>"
            h += "".join("<tr>" + "".join(f"<td>{_inline_md(c)}</td>" for c in r) + "</tr>" for r in rows_[1:])
            out.append(h + "</tbody></table></div>")
            tbl = []

    for line in md.splitlines():
        if code is not None:
            if line.strip().startswith("```"):
                import html as H

                out.append("<pre class='mono'>" + H.escape("\n".join(code)) + "</pre>")
                code = None
            else:
                code.append(line)
            continue
        if line.strip().startswith("```"):
            flush()
            code = []
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            lvl = min(6, len(m.group(1)) + 2)
            out.append(f"<h{lvl}>{_inline_md(m.group(2))}</h{lvl}>")
            continue
        if line.strip().startswith("|"):
            if para or lst:
                flush()
            tbl.append(line)
            continue
        m = re.match(r"^\s*([-*]|\d+\.)\s+(.*)", line)
        if m:
            kind = "ol" if m.group(1)[0].isdigit() else "ul"
            if para or tbl or (lst and lst[0] != kind):
                flush()
            if lst is None:
                lst = (kind, [])
            lst[1].append(m.group(2))
            continue
        if not line.strip():
            flush()
            continue
        if lst and line.startswith((" ", "\t")):
            lst[1][-1] += " " + line.strip()
            continue
        if tbl or lst:
            flush()
        para.append(line.strip())
    flush()
    return "\n".join(out)


def build(results, out, config, replay_globs=REPLAY_GLOBS):
    res, out = Path(results).resolve(), Path(out).resolve()
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    notes, files = [], []
    ledger, _ = read_jsonl(res / "ledger.jsonl")
    if (res / "ledger.jsonl").exists():
        files.append(dict(path="ledger.jsonl", sha=sha12(res / "ledger.jsonl"), rows=len(ledger)))
    rows = load_rows(res, ledger, files)
    gate = read_json(res / "inkling_gate.json")
    if gate is not None:
        files.append(dict(path="inkling_gate.json", sha=sha12(res / "inkling_gate.json")))
    doctor = read_json(res / "doctor.json") or {}
    if doctor:
        files.append(dict(path="doctor.json", sha=sha12(res / "doctor.json")))
    cfg = {}
    try:
        import yaml

        cfg = yaml.safe_load(open(config)) or {}
    except Exception as e:  # noqa: BLE001
        notes.append(f"config {config} not read ({e.__class__.__name__})")
    verdicts, vsrc = load_verdicts(res, out, notes)
    D = dict(meta={}, rows=rows)
    # ---- meta / header
    ink = cfg.get("inkling", {})
    load = (gate or {}).get("load") or {}
    D["meta"] = clean(dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), results_dir=str(res), files=files,
        verdicts_source=vsrc, notes=notes,
        sprint=cfg.get("sprint", {}),
        hardware=dict(device=doctor.get("device") or doctor.get("nvidia_smi"), capability=doctor.get("capability"),
                      host=doctor.get("host"), machine=doctor.get("machine"), mem_total=doctor.get("mem_total"),
                      torch=doctor.get("torch"), mujoco=doctor.get("mujoco"), transformers=doctor.get("transformers"),
                      python=doctor.get("python"), doctor_utc=doctor.get("time_utc")),
        checkpoint=dict(repo=ink.get("repo", "thinkingmachines/Inkling-2-Mini"), revision=ink.get("revision"),
                        quantization=ink.get("quantization"), core_layers=ink.get("core_layers"), num_passes=ink.get("num_passes"),
                        present_at_doctor=doctor.get("inkling_checkpoint_present"),
                        gate_revision=(((gate or {}).get("checks") or {}).get("1_checkpoint_native_outputs") or {}).get("revision"),
                        load_report={k: load.get(k) for k in ("load_s", "text_params", "mem_after_load") if k in load}),
    ))
    # ---- experiment status
    by_exp = defaultdict(list)
    for r in rows:
        by_exp[exp_id(r["exp"])].append(r)
    vexp = verdicts.get("experiments") or {}
    exps = []
    for e in ["primary"] + EXPERIMENTS + sorted(k for k in by_exp if k not in EXPERIMENTS and k != "primary"):
        rs = by_exp.get(e, [])
        st = {r["status"] for r in rs}
        status = next((s for s in STATUS_ORDER if s in st), "untested")
        ov = (vexp.get(e) or {}) if isinstance(vexp.get(e), dict) else {}
        tags = sorted({r["tag"] for r in rs}, key=lambda t: (tag_rank(t), t))
        exps.append(dict(id=e, title=ov.get("title") or EXP_TITLES.get(e, "primary factorial (test stratum)" if e == "primary" else ""),
                         status=ov.get("status") or status, status_from_rows=status, overridden=bool(ov.get("status")),
                         note=ov.get("note", ""), n_rows=len(rs), n_observed=sum(1 for r in rs if r["status"] == "observed"),
                         tags=tags, models=sorted({r["model"] for r in rs}),
                         tier=min((r["tier"] for r in rs), key=lambda t: TIER_RANK[t]) if rs else ""))
    D["experiments"] = exps
    # ---- hypotheses
    hyps = []
    vh = verdicts.get("hypotheses") or {}
    for h in HYPOTHESES:
        v = vh.get(h) or {}
        ev_rows = []
        for flt in v.get("evidence") or []:
            if isinstance(flt, dict):
                ev_rows += [i for i, r in enumerate(rows) if match_filter(r, flt)]
        hyps.append(dict(id=h, title=v.get("title") or H_TITLES.get(h, ""), verdict=v.get("verdict", "untested"),
                         rationale=v.get("rationale", ""), evidence=v.get("evidence") or [],
                         rows=sorted(set(ev_rows), key=lambda i: (TIER_RANK[rows[i]["tier"]], rows[i]["model"] != "compact",
                                                                    rows[i]["model"].startswith(("constvel", "coulomb")), i))[:200]))
    D["hypotheses"] = hyps
    # ---- other sections
    D["inkling"] = clean(dict(gate=gate, ledger=[r for r in ledger if str(r.get("kind", "")).startswith("inkling")]))
    D["curves"] = load_curves(res, files)
    D["g1"] = load_g1(res, assets, files, notes, replay_globs)
    gpu = parse_gpu_ledger([res / "gpu_ledger.log", res / "logs" / "gpu_ledger.log", res.parent / "logs" / "gpu_ledger.log"], files)
    led = [{k: r.get(k) for k in ("time_utc", "end_utc", "run_id", "kind", "variant", "seed", "tag", "config_hash", "status", "wall_s",
                                    "steps", "examples", "step_s", "step_s_mean", "step_s_p95", "peak_mem_mib", "peak_cuda_gb", "n_params",
                                    "n_trainable", "best_val_nll", "best_step", "error", "budget_s", "suite")} for r in ledger]
    tot = defaultdict(float)
    for r in ledger:
        if fnum(r.get("wall_s")):
            tot[str(r.get("kind"))] += float(r["wall_s"])
    D["cost"] = clean(dict(ledger=led, wall_by_kind=dict(tot), gpu=gpu))
    if D["cost"]["gpu"] is None:
        notes.append("no gpu_ledger.log in results/ (lead: copy logs/gpu_ledger.log to results/gpu_ledger.log)")
    # ---- limitations / skipped
    lim = list(verdicts.get("limitations") or [])
    auto = []
    if not any(r["tier"] == "main" for r in rows):
        auto.append("No rows from the main evaluation tag yet: every evaluation number shown comes from pilot / dev-check runs "
                    "(tag shown on each table) and is NOT a result.")
    seeds = {r["seed"] for r in rows if r["model"] == "compact" and r["seed"] >= 0}
    if rows and len(seeds) < 3:
        auto.append(f"compact rows exist for {len(seeds)} seed(s) {sorted(seeds)}; the plan is 3 seeds. Episode-level CIs do not capture seed variance.")
    if gate is None:
        auto.append("Actual-core gate (results/inkling_gate.json) not present: no Inkling-2-Mini claim can be made.")
    elif not gate.get("passed") and gate.get("passed_with_amendment"):
        auto.append("Actual-core gate passed only with an amendment (an original check failed and an amended check was added); both are recorded in inkling_gate.json.")
    elif not gate.get("passed"):
        auto.append("Actual-core gate has not passed (partial or failed); Inkling numbers, if any, are diagnostics only.")
    if D["g1"]["replays"] and any(rp["hand_source"] != "logged" for rp in D["g1"]["replays"]):
        auto.append("G1 replay: hand/pusher position is reconstructed from the executed push plan where the per-step log lacks it.")
    auto.append("G1 is a fixed-base MuJoCo simulation with synthetic sensors; no real robot, no locomotion, no vision.")
    D["limitations"] = dict(lead=lim, auto=auto)
    D["skipped"] = [dict(exp=r["exp"], model=r["model"], variant=r["variant"], seed=r["seed"], stratum=r["stratum"],
                         metric=r["metric"], note=(r.get("x") or {}).get("note", ""), tag=r["tag"]) for r in rows if r["status"] == "skipped"]
    D["plots"] = make_plots(D, rows, assets, notes)
    D["meta"]["notes"] = notes
    fm = res / "findings.md"
    if fm.exists():
        files.append(dict(path="findings.md", sha=sha12(fm)))
        D["findings_html"] = md_to_html(fm.read_text())
    D["demo_video"] = "../results/demo.mp4" if (res / "demo.mp4").exists() else None
    D = clean(D)
    D["rows"] = pack_rows(rows)
    js = json.dumps(D, separators=(",", ":"))
    if len(js) > 15e6:
        notes.append(f"data.json is {len(js)/1e6:.1f} MB (> 15 MB)")
    (out / "data.json").write_text(js)
    (out / "data.js").write_text("window.REPORT_DATA=" + js + ";\n")
    tpl = out / "_template.html"
    if tpl.exists():
        html = tpl.read_text().replace("/*__REPORT_DATA__*/null", js.replace("</", "<\\/"))
        (out / "index.html").write_text(html)
    else:
        notes.append(f"{tpl} missing; index.html not regenerated")
    return D


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "report"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "sprint.yaml"))
    ap.add_argument("--replays", default=",".join(REPLAY_GLOBS),
                    help="comma-separated globs under results/g1/ whose episodes get a replay (keeps data.json small)")
    a = ap.parse_args(argv)
    t0 = time.time()
    D = build(a.results, a.out, a.config, [g for g in a.replays.split(",") if g])
    n = len(D["rows"]["data"])
    obs = sum(1 for e in D["experiments"] if e["status"] == "observed")
    print(f"[report] rows={n} experiments_observed={obs} plots={len(D['plots'])} g1_episodes={len(D['g1']['episodes'])} "
          f"replays={len(D['g1']['replays'])} -> {Path(a.out).resolve()}/index.html ({time.time()-t0:.1f}s)")
    for m in D["meta"]["notes"]:
        print("[report] note:", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
