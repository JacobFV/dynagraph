"""Validate the MediaPipe hand front end on DexYCB (owner: agent EXT). docs/EXT.md §5a.

For each local sequence and the chosen camera(s): run HandTracker (VIDEO mode, 1 hand -- DexYCB sequences are
single-hand) on the RGB stream, lift to 3-D with the aligned depth, and compare against the DexYCB labels
(joint_2d px, joint_3d m camera frame; mocap-fitted MANO joints, evaluator only).

Metrics (frames where the labelled hand is present):
  recall        fraction of frames with a detection
  2-D           median / mean per-joint pixel error, PCK@10px, PCK@20px (640x480 images)
  3-D lifted    MPJPE (mm) of depth-lifted joints, absolute camera frame and root(wrist)-relative
  3-D world     MediaPipe world landmarks, root-relative MPJPE after removing the wrist (no scale fitting)
Joint order: DexYCB/MANO and MediaPipe both list wrist, then thumb (4), index, middle, ring, little (4 each), so
they correspond index for index.

usage: python -I scripts/ext/dexycb_pose_validate.py [--max-seq 20] [--serials 0] [--stride 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from physmem.ext import dexycb as D  # noqa: E402
from physmem.ext.human_pose import HandTracker, lift_depth  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "ext"


def run(a):
    seqs = D.sequences()
    # deterministic spread over objects: every k-th sequence
    seqs = seqs[:: max(1, len(seqs) // a.max_seq)][: a.max_seq]
    rows, agg = [], dict(e2=[], e3=[], e3r=[], ew=[], present=0, det=0)
    t0 = time.time()
    for q in seqs:
        for si in a.serials:
            serial = q.serials[si]
            fr = range(0, q.num_frames, a.stride)
            rgb, dep, K = D.load_view(q, serial, fr)
            lab = D.load_labels(q, serial, fr)
            ht = HandTracker(num_hands=1)
            ts = ht(rgb, fps=D.FPS / a.stride, source=f"dexycb/{q.subject}/{q.name}/{serial}")
            ht.close()
            # single hand: take whichever entity was detected per frame
            uv = np.where(np.isfinite(ts.uv[:, 0]), ts.uv[:, 0], ts.uv[:, 1])           # (T,21,2)
            wl = ts.extra["mp_world"]
            wl = np.where(np.isfinite(wl[:, 0]), wl[:, 0], wl[:, 1])
            xyz = lift_depth(uv[:, None], dep, K, 1e-3)[:, 0]
            gt2, gt3 = lab["joint_2d"], lab["joint_3d"]
            present = (gt2 >= 0).all(-1).all(-1)
            det = np.isfinite(uv).all(-1).all(-1)
            both = present & det
            e2 = np.linalg.norm(uv - gt2, axis=-1)[both]                                # (n,21)
            e3 = np.linalg.norm(xyz - gt3, axis=-1)[both] * 1000
            e3r = np.linalg.norm((xyz - xyz[:, :1]) - (gt3 - gt3[:, :1]), axis=-1)[both] * 1000
            ew = np.linalg.norm((wl - wl[:, :1]) - (gt3 - gt3[:, :1]), axis=-1)[both] * 1000
            agg["present"] += int(present.sum()); agg["det"] += int(both.sum())
            for k, v in (("e2", e2), ("e3", e3), ("e3r", e3r), ("ew", ew)):
                agg[k].append(v.reshape(-1))
            rows.append(dict(seq=f"{q.subject}/{q.name}", serial=serial, object=q.grasped, side=q.mano_side,
                             frames=int(len(present)), present=int(present.sum()), detected=int(both.sum()),
                             med_px=float(np.median(e2)) if e2.size else None,
                             med_mm_rootrel=float(np.nanmedian(e3r)) if e3r.size else None))
            print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    cat = {k: np.concatenate(agg[k]) if agg[k] else np.zeros(0) for k in ("e2", "e3", "e3r", "ew")}
    summ = dict(
        n_sequences=len(seqs), serials=a.serials, stride=a.stride, frames_hand_present=agg["present"],
        recall=agg["det"] / max(agg["present"], 1),
        px_median=float(np.median(cat["e2"])), px_mean=float(np.mean(cat["e2"])),
        pck10=float(np.mean(cat["e2"] < 10)), pck20=float(np.mean(cat["e2"] < 20)),
        mpjpe_lifted_abs_mm_median=float(np.nanmedian(cat["e3"])),
        mpjpe_lifted_rootrel_mm_median=float(np.nanmedian(cat["e3r"])),
        mpjpe_lifted_rootrel_mm_mean=float(np.nanmean(cat["e3r"])),
        mpjpe_mpworld_rootrel_mm_median=float(np.nanmedian(cat["ew"])),
        lifted_valid_frac=float(np.mean(np.isfinite(cat["e3"]))),
        note="DexYCB labels are mocap-fitted MANO joints (evaluator only); MediaPipe hand_landmarker.task float16; "
             "single-hand mode; depth lift = median of 5x5 aligned-depth window")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dexycb_pose_validation.json").write_text(json.dumps(dict(summary=summ, per_view=rows), indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq", type=int, default=20)
    ap.add_argument("--serials", default="0")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    a.serials = [int(s) for s in a.serials.split(",")]
    run(a)
