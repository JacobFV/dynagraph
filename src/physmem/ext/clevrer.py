"""CLEVRER loader (Yi et al., ICLR 2020; data CC0). Owner: agent EXT. docs/EXT.md §1/§3c.

Layout after extracting the official validation archives into $EXT_DATA_ROOT/clevrer/extracted:
  video_10000-11000/video_XXXXX.mp4            480x320, 128 frames, 25 fps (validation ids 10000-14999)
  annotation_10000-11000/annotation_XXXXX.json  (the official zips have no split-level top directory)
Annotation fields (official README): object_property (id, color, material, shape), motion_trajectory (per frame:
objects with location, orientation, velocity, angular_velocity, inside_camera_view), collision (object_ids, frame_id,
location). Annotations are SIMULATOR state: evaluator-only labels under the V3 rules; model inputs come from the track
front end run on the mp4.

The official CLEVRER task is question answering (results/ext/clevrer_published_baselines.json); any collision-
prediction number computed from these annotations is a derived, non-official task.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("EXT_DATA_ROOT", "/home/brandonin/datasets")) / "clevrer" / "extracted"
FPS = 25.0


def videos(root: Path = ROOT):
    """-> list of (video_id:int, mp4 path or None, annotation path or None) for every id found locally."""
    mp4 = {int(p.stem.split("_")[1]): p for p in root.glob("video_*/video_*.mp4")}
    ann = {int(p.stem.split("_")[1]): p for p in root.glob("annotation_*/annotation_*.json")}
    return [(k, mp4.get(k), ann.get(k)) for k in sorted(set(mp4) | set(ann))]


def load_annotation(path) -> dict:
    """EVALUATOR ONLY: objects, per-frame trajectories (dict frame -> {object_id: location}), collisions."""
    d = json.loads(Path(path).read_text())
    traj = {fr["frame_id"]: {o["object_id"]: o["location"] for o in fr["objects"]} for fr in d["motion_trajectory"]}
    return dict(objects=d["object_property"], trajectory=traj,
                collisions=[(tuple(c["object_ids"]), c["frame_id"]) for c in d["collision"]])
