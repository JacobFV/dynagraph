"""Tier-B object-track front end: SAM 2.1 automatic masks on frame 0 + video propagation (owner: agent EXT).

SAM 2 code and checkpoints: Apache-2.0 (github.com/facebookresearch/sam2 README "License"). Runs on gb10 under
~/dynagraph/scripts/v2/gpu_small.sh (< 6 GB). Input: a list file of mp4 paths; output: one TrackSet .npz per movie
(entities obj0..objN, keypoints centroid/bbox_min/bbox_max, extra area + frame-0 mean RGB of each mask, which lets
downstream code identify the red agent / yellow patient by colour without simulator state).

usage (gb10): python -I sam2_tracks.py --list movies.txt --out tracks_dir --ckpt sam2.1_hiera_small.pt
              [--cfg configs/sam2.1/sam2.1_hiera_s.yaml] [--max-obj 8] [--stride 1]
List lines: "<mp4>" (auto masks on frame 0) or "<mp4> <map.png> <frame>" (Physion++: the official red/yellow cue map
gives the agent and patient masks at the cue frame -- the same cue humans get -- entities "agent", "patient" first,
then auto masks for the remaining objects; propagation runs forward and backward from that frame).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def read_frames(path, stride=1):
    import imageio.v2 as iio
    return np.stack([np.asarray(f)[..., :3] for i, f in enumerate(iio.get_reader(str(path))) if i % stride == 0])


def select_masks(masks, H, W, max_obj):
    """drop background-sized and tiny masks, then greedy non-overlap by predicted IoU."""
    keep, occ = [], np.zeros((H, W), bool)
    for m in sorted(masks, key=lambda m: -m["predicted_iou"]):
        a = m["area"] / (H * W)
        if not (2e-4 <= a <= 0.15):
            continue
        seg = m["segmentation"]
        if (seg & occ).sum() > 0.2 * seg.sum():
            continue
        keep.append(seg)
        occ |= seg
        if len(keep) >= max_obj:
            break
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--max-obj", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()

    import imageio.v2 as iio
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2, build_sam2_video_predictor

    from physmem.ext.tracks import KIND_OBJECT, TrackSet

    dev = "cuda"
    # shared-host rule: < 6 GB GPU per job. Hard cap the caching allocator and offload frames/state to CPU.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4.5e9 / total))
    tag = f"sam2.1-small ckpt_sha256={_sha(a.ckpt)} | frontend=sam2-v1"
    gen = SAM2AutomaticMaskGenerator(build_sam2(a.cfg, a.ckpt, device=dev), points_per_side=16,
                                     pred_iou_thresh=0.8, stability_score_thresh=0.9)
    vp = build_sam2_video_predictor(a.cfg, a.ckpt, device=dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [ln.split() for ln in Path(a.list).read_text().splitlines() if ln.strip()]
    t0 = time.time()
    for i, job in enumerate(jobs):
        mp4 = Path(job[0]).expanduser()
        dst = out / (mp4.parent.parent.name + "__" + mp4.parent.name + "__" + mp4.stem + ".npz")
        if dst.exists():
            continue
        rgb = read_frames(mp4, a.stride)
        T, H, W, _ = rgb.shape
        f0, names, cue = 0, [], []
        if len(job) == 3:
            cm = iio.imread(str(Path(job[1]).expanduser()))
            f0 = min(int(job[2]) // a.stride, T - 1)
            red = (cm[..., 0] > 200) & (cm[..., 1] < 60) & (cm[..., 3] > 0)
            yel = (cm[..., 0] > 200) & (cm[..., 1] > 200) & (cm[..., 3] > 0)
            for nm, mk in (("agent", red), ("patient", yel)):
                if mk.sum() >= 8:
                    cue.append(mk)
                    names.append(nm)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            auto = select_masks(gen.generate(rgb[f0]), H, W, a.max_obj)
            occ = np.any(cue, 0) if cue else np.zeros((H, W), bool)
            auto = [m for m in auto if (m & occ).sum() < 0.2 * m.sum()]
            sel = cue + auto[: max(a.max_obj - len(cue), 0)]
            names = names + [f"obj{k}" for k in range(len(sel) - len(names))]
            uv = np.full((T, max(len(sel), 1), 3, 2), np.nan, np.float32)
            area = np.zeros((T, max(len(sel), 1)), np.float32)
            col0 = np.zeros((max(len(sel), 1), 3), np.float32)
            if sel:
                with tempfile.TemporaryDirectory() as td:
                    for t in range(T):
                        iio.imwrite(f"{td}/{t:05d}.jpg", rgb[t], quality=95)
                    st = vp.init_state(video_path=td, offload_video_to_cpu=True, offload_state_to_cpu=True)
                    for k, m in enumerate(sel):
                        vp.add_new_mask(st, frame_idx=f0, obj_id=k, mask=m)
                        col0[k] = rgb[f0][m].mean(0)
                    for rev in ((False, True) if f0 > 0 else (False,)):
                        for f, ids, logits in vp.propagate_in_video(st, start_frame_idx=f0, reverse=rev):
                            ms = (logits > 0).cpu().numpy()[:, 0]
                            for k, m in zip(ids, ms):
                                if m.sum() < 8:
                                    continue
                                ys, xs = np.nonzero(m)
                                uv[f, k] = [(xs.mean(), ys.mean()), (xs.min(), ys.min()), (xs.max(), ys.max())]
                                area[f, k] = m.sum()
                    vp.reset_state(st)
        ents = names if sel else ["obj0"]
        ts = TrackSet(KIND_OBJECT, np.arange(T) * a.stride / a.fps, uv, ents,
                      ["centroid", "bbox_min", "bbox_max"], (H, W), f"{mp4} | {tag} | prompt_frame={f0}",
                      extra=dict(area=area, color0=col0, prompt_frame=np.array(f0)))
        ts.save(dst)
        print(f"[sam2] {i + 1}/{len(jobs)} {mp4.name} objs={len(sel)} T={T} "
              f"peak_gpu={torch.cuda.max_memory_allocated() / 1e9:.2f}GB {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
