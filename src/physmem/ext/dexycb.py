"""DexYCB loader (Chao et al., CVPR 2021; data CC BY-NC 4.0, research use). Owner: agent EXT. docs/EXT.md §5-6.

Our own reader (the official dex-ycb-toolkit is GPL-3.0 and needs MANO; we only follow its documented layout):
  <root>/<subject>/<sequence>/meta.yml            serials, ycb_ids, ycb_grasp_ind, mano_sides, num_frames
  <root>/<subject>/<sequence>/<serial>/color_%06d.jpg, aligned_depth_to_color_%06d.png (uint16 mm),
                                       labels_%06d.npz (seg, pose_y, pose_m, joint_3d, joint_2d)
  <root>/calibration/intrinsics/<serial>_640x480.yml  ('color': fx, fy, ppx, ppy)
Sequence index i (sorted, 100 per subject) grasps object i // 5 (20 objects x 5 trials), matching the toolkit's
's3' (unseen object) split rule.

ACTOR / EVALUATOR split: color + depth + intrinsics are actor-legal (an RGB-D camera). joint_2d/joint_3d (mocap-fitted
MANO joints), pose_y (object 6-D poses), seg and the object masses below are EVALUATOR-ONLY labels.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("EXT_DATA_ROOT", "/home/brandonin/datasets")) / "dexycb" / "extracted"
FPS = 30.0

YCB_CLASSES = {1: "002_master_chef_can", 2: "003_cracker_box", 3: "004_sugar_box", 4: "005_tomato_soup_can",
               5: "006_mustard_bottle", 6: "007_tuna_fish_can", 7: "008_pudding_box", 8: "009_gelatin_box",
               9: "010_potted_meat_can", 10: "011_banana", 11: "019_pitcher_base", 12: "021_bleach_cleanser",
               13: "024_bowl", 14: "025_mug", 15: "035_power_drill", 16: "036_wood_block", 17: "037_scissors",
               18: "040_large_marker", 19: "051_large_clamp", 20: "052_extra_large_clamp", 21: "061_foam_brick"}
# nominal masses (g): Calli et al., "Benchmarking in Manipulation Research", IEEE RAM 22(3) 2015, Table 2
# (PDF sha256 9fa0bc33...; transcribed 2026-11-26). Evaluator-only labels.
YCB_MASS_G = {"002_master_chef_can": 414, "003_cracker_box": 411, "004_sugar_box": 514, "005_tomato_soup_can": 349,
              "006_mustard_bottle": 603, "007_tuna_fish_can": 171, "008_pudding_box": 187, "009_gelatin_box": 97,
              "010_potted_meat_can": 370, "011_banana": 66, "019_pitcher_base": 178, "021_bleach_cleanser": 1131,
              "024_bowl": 147, "025_mug": 118, "035_power_drill": 895, "036_wood_block": 729, "037_scissors": 82,
              "040_large_marker": 15.8, "051_large_clamp": 125, "052_extra_large_clamp": 202,
              "061_foam_brick": 28}


@dataclass
class Sequence:
    subject: str
    name: str
    path: Path
    serials: list
    ycb_ids: list
    grasp_ind: int
    mano_side: str
    num_frames: int

    @property
    def grasped(self) -> str:
        return YCB_CLASSES[self.ycb_ids[self.grasp_ind]]

    @property
    def mass_kg(self) -> float:                 # evaluator only
        return YCB_MASS_G[self.grasped] / 1000.0


def _yaml(text):
    """safe YAML load that also accepts the !!python/tuple tags DexYCB's meta files use (as lists)."""
    import yaml

    class L(yaml.SafeLoader):
        pass
    L.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, node: ld.construct_sequence(node))
    return yaml.load(text, Loader=L)


def sequences(root: Path = ROOT, subjects=None) -> list[Sequence]:
    out = []
    for sd in sorted(p for p in root.glob("*-subject-*") if p.is_dir()):
        if subjects and sd.name not in subjects:
            continue
        for q in sorted(p for p in sd.iterdir() if (p / "meta.yml").exists()):
            m = _yaml((q / "meta.yml").read_text())
            out.append(Sequence(sd.name, q.name, q, m["serials"], m["ycb_ids"], m["ycb_grasp_ind"],
                                m["mano_sides"][0], m["num_frames"]))
    return out


def intrinsics(serial: str, root: Path = ROOT) -> np.ndarray:
    c = _yaml((root / "calibration" / "intrinsics" / f"{serial}_640x480.yml").read_text())["color"]
    return np.array([[c["fx"], 0, c["ppx"]], [0, c["fy"], c["ppy"]], [0, 0, 1]], np.float64)


def load_view(seq: Sequence, serial: str, frames=None, depth=True):
    """-> rgb (T,480,640,3) uint8, depth (T,480,640) uint16 mm or None, K (3,3)."""
    import imageio.v3 as iio
    d = seq.path / serial
    fr = range(seq.num_frames) if frames is None else frames
    rgb = np.stack([iio.imread(d / f"color_{f:06d}.jpg") for f in fr])
    dep = np.stack([iio.imread(d / f"aligned_depth_to_color_{f:06d}.png") for f in fr]) if depth else None
    return rgb, dep, intrinsics(serial)


def load_labels(seq: Sequence, serial: str, frames=None):
    """EVALUATOR ONLY: joint_2d (T,21,2) px, joint_3d (T,21,3) m (camera frame), pose_y (T,n_obj,3,4)."""
    d = seq.path / serial
    fr = range(seq.num_frames) if frames is None else frames
    j2, j3, py = [], [], []
    for f in fr:
        z = np.load(d / f"labels_{f:06d}.npz")
        j2.append(z["joint_2d"][0])
        j3.append(z["joint_3d"][0])
        py.append(z["pose_y"])
    return dict(joint_2d=np.stack(j2), joint_3d=np.stack(j3), pose_y=np.stack(py))
