"""Physion v1 loader (Bear et al., NeurIPS 2021 D&B; data + code MIT). Owner: agent EXT. See docs/EXT.md §3.

Layout under $EXT_DATA_ROOT/physion (default /home/brandonin/datasets/physion):
  extracted/Physion/<Scenario>/mp4s-redyellow/*.mp4   official test movies shown to MODELS (agent red, patient yellow)
  extracted/Physion/labels.csv                        official test ground-truth OCP outcome (1200 rows)
  readout/<Scenario>/*.hdf5                           readout-fitting stimuli (only a prefix subset is local)
  results/csv/...                                     model_human_results.zip (official human + model responses)

Label rule (official, cogtoolslab/physics-benchmarking-neurips2021 data/extract_labels.ipynb):
    label = any(frame['labels']['target_contacting_zone'] for frame in f['frames'].values())
Observed window: first 1.5 s = 45 frames at 30 fps (paper §2.4, "tvis").
"""
from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCENARIOS = ["Dominoes", "Support", "Collide", "Contain", "Drop", "Link", "Roll", "Drape"]
FPS = 30
T_VIS = 45                    # observed frames (1.5 s)
ROOT = Path(os.environ.get("EXT_DATA_ROOT", "/home/brandonin/datasets")) / "physion"


@dataclass
class Stimulus:
    scenario: str
    name: str                 # canonical stimulus id, e.g. pilot_it2_collision_simple_box_0009_img (test) or hdf5 stem
    split: str                # "test" | "readout"
    path: Path                # mp4 (test) or hdf5 (readout)
    label: bool | None        # OCP ground truth (evaluator only)


def _test_key(mp4_name: str) -> str:
    # '<base>-redyellow_<idx>_img.mp4' -> '<base>_<idx>_img'  (labels.csv key)
    return re.sub(r"-redyellow_(\d+)_img\.mp4$", r"_\1_img", mp4_name)


def test_set(root: Path = ROOT, scenarios=SCENARIOS) -> list[Stimulus]:
    base = root / "extracted" / "Physion"
    labels = {}
    with open(base / "labels.csv") as f:
        for row in csv.reader(f):
            if row and row[0]:
                labels[row[0]] = row[1].strip() == "True"
    out = []
    for s in scenarios:
        for p in sorted((base / s / "mp4s-redyellow").glob("*.mp4")):
            k = _test_key(p.name)
            out.append(Stimulus(s, k, "test", p, labels.get(k)))
    missing = [x.name for x in out if x.label is None]
    assert not missing, f"test stimuli without labels: {missing[:5]}"
    return out


def readout_set(root: Path = ROOT, scenarios=SCENARIOS) -> list[Stimulus]:
    out = []
    for s in scenarios:
        for p in sorted((root / "readout" / s).glob("*.hdf5")):
            out.append(Stimulus(s, p.stem, "readout", p, None))
    return out


def read_mp4(path, max_frames=None) -> np.ndarray:
    """(T, H, W, 3) uint8 RGB."""
    import imageio.v3 as iio
    frames = []
    for i, fr in enumerate(iio.imiter(path, plugin="pyav") if _has_pyav() else iio.imiter(path)):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(np.asarray(fr)[..., :3])
    return np.stack(frames)


def _has_pyav():
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


def _decode_png(buf) -> np.ndarray:
    import imageio.v3 as iio
    return np.asarray(iio.imread(io.BytesIO(bytes(buf)), extension=".png"))


def hdf5_label(path) -> bool:
    """official OCP label rule (see module docstring)."""
    import h5py
    with h5py.File(path, "r") as f:
        return bool(any(bool(fr["labels"]["target_contacting_zone"][()]) for fr in f["frames"].values()))


def read_hdf5(path, max_frames=None, with_seg=False):
    """readout/test HDF5 -> dict(rgb (T,H,W,3) uint8, seg (T,H,W,3) uint8 or None, label, meta).
    rgb is the red/yellow model stimulus (the `_img` pass). seg and meta are SIMULATOR state: evaluator /
    diagnostic only, never inputs to a model under the V3 rules."""
    import h5py
    with h5py.File(path, "r") as f:
        keys = sorted(f["frames"].keys())
        label = bool(any(bool(f["frames"][k]["labels"]["target_contacting_zone"][()]) for k in keys))
        keys = keys[:max_frames] if max_frames else keys
        rgb = np.stack([_decode_png(f["frames"][k]["images"]["_img"][()])[..., :3] for k in keys])
        seg = (np.stack([_decode_png(f["frames"][k]["images"]["_id"][()])[..., :3] for k in keys])
               if with_seg else None)
        st = f["static"]
        meta = dict(n_frames=len(f["frames"]),
                    mass=st["mass"][()] if "mass" in st else None,
                    target_id=int(st["target_id"][()]) if "target_id" in st else None,
                    zone_id=int(st["zone_id"][()]) if "zone_id" in st else None,
                    object_ids=st["object_ids"][()] if "object_ids" in st else None,
                    seg_colors=st["object_segmentation_colors"][()] if "object_segmentation_colors" in st else None,
                    stimulus_name=st["stimulus_name"][()].decode() if "stimulus_name" in st else None)
    return dict(rgb=rgb, seg=seg, label=label, meta=meta)


# ----------------------------------------------------------------- official human data (model_human_results.zip)
def human_tables(root: Path = ROOT):
    """official human responses (results/csv/humans/human_responses-*.csv; participants listed in
    excluded_games.csv dropped, as in the official analysis inputs). Returns
      per_stim:     stim_ID -> np.array of 0/1 responses (YES = 1) over included participants
      participants: gameID -> {stim_ID: 0/1}
    stim_ID matches labels.csv keys (e.g. pilot_it2_collision_..._0002_img). {} , {} if not extracted."""
    import pandas as pd
    d = root / "results" / "csv" / "humans"
    files = sorted(d.glob("human_responses-*.csv"))
    if not files:
        return {}, {}
    excl = set(pd.read_csv(d / "excluded_games.csv")["gameID"]) if (d / "excluded_games.csv").exists() else set()
    df = pd.concat([pd.read_csv(p, usecols=["gameID", "stim_ID", "response", "condition"]) for p in files])
    df = df[(df["condition"] == "prediction") & ~df["gameID"].isin(excl)]
    df["y"] = (df["response"].astype(str).str.upper() == "YES").astype(int)
    per_stim = {k: g["y"].to_numpy() for k, g in df.groupby("stim_ID")}
    parts = {k: dict(zip(g["stim_ID"], g["y"])) for k, g in df.groupby("gameID")}
    return per_stim, parts


def human_responses(root: Path = ROOT):
    return human_tables(root)[0]
