"""Physion++ loader (Tung et al., NeurIPS 2023 D&B; MIT). Owner: agent EXT. docs/EXT.md §3b.

Layout after unzipping the official readout_data.zip / test_data.zip into $EXT_DATA_ROOT/physionpp/extracted:
  readout_data_v1/<scenario>_pp/<config>/<NNNN>_img.mp4   model stimulus (agent red / patient yellow)
                                        <NNNN>_id.mp4    segmentation video   (simulator; evaluator/diagnostic)
                                        <NNNN>_id.json   per-instance masks   (simulator; evaluator/diagnostic)
                                        <NNNN>.pkl       TDW metadata: frames/*, static (mass, friction, ...),
                                                         static/does_target_contact_zone (OCP label),
                                                         static/start_frame_for_prediction, star_mass/star_deform ...
                                        <NNNN>_map.png   red/yellow cue map
  testdata_v1/<scenario>_pp-copy{0,1}/<config>/...       same per-trial files
Scenarios (mechanical property in the name): mass_{collision,dominoes,waterpush}, friction_{collision,platform,cloth},
bouncy_{platform,wall} (elasticity), deform_{clothhang,clothhit} (deformability).
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("EXT_DATA_ROOT", "/home/brandonin/datasets")) / "physionpp" / "extracted"
PROPERTY = {"mass": "mass", "friction": "friction", "bouncy": "elasticity", "deform": "deformability"}


@dataclass
class Trial:
    split: str            # readout | test
    scenario: str         # e.g. mass_dominoes_pp
    config: str
    idx: str
    mp4: Path
    pkl: Path

    @property
    def prop(self) -> str:
        return PROPERTY[self.scenario.split("_")[0]]

    def meta(self) -> dict:
        """EVALUATOR ONLY: label, prediction start frame, hidden-property values."""
        with open(self.pkl, "rb") as f:
            d = pickle.load(f)
        st = d["static"]
        keys = ("does_target_contact_zone", "start_frame_for_prediction", "start_frame_of_the_clip",
                "start_frame_after_curtain", "star_mass", "star_deform", "target_id", "zone_id", "mass",
                "static_friction", "dynamic_friction", "bounciness", "stimulus_name")
        out = {k: st[k] for k in keys if k in st}
        out["label"] = bool(st.get("does_target_contact_zone", False))
        out["n_frames"] = len(d.get("frames", {}))
        return out


def trials(split: str, root: Path = ROOT) -> list[Trial]:
    base = root / ("readout_data_v1" if split == "readout" else "testdata_v1")
    out = []
    for sd in sorted(p for p in base.iterdir() if p.is_dir()):
        for cd in sorted(p for p in sd.iterdir() if p.is_dir()):
            for mp4 in sorted(cd.glob("*_img.mp4")):
                idx = mp4.name[:-8]
                pkl = cd / f"{idx}.pkl"
                if pkl.exists():
                    out.append(Trial(split, sd.name, cd.name, idx, mp4, pkl))
    return out


def stim_id(t: Trial) -> str:
    """human-data stimulus id: '<scenario dir>_<config>_<idx>_img' (e.g. mass_dominoes_pp-copy1_mass_dominoes-..._0000_img)."""
    return f"{t.scenario}_{t.config}_{t.idx}_img"


def official_test_ids(root: Path = ROOT) -> dict:
    """The paper's 192-per-property test list is not linked from the project page; we reconstruct it from the
    official human data (human_data.zip, 'run2' files: 50 participants x 192 stimuli per property, paper §4.2).
    Returns {property: set(stim_id)}; documented as a reconstruction in docs/EXT.md §3b."""
    import pandas as pd
    out = {}
    for f in sorted((root / "humans").glob("human_responses-*_pilot-pilot_run2_*.csv")):
        prop = PROPERTY[f.name.split("-")[1].split("_")[0]]
        d = pd.read_csv(f, usecols=["stim_ID", "condition"])
        out[prop] = set(d.loc[d.condition == "prediction", "stim_ID"])
    return out
