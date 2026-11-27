"""Render our Physion OCP results next to the published Table S1 numbers (markdown to stdout)."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results" / "ext"
ours = json.loads((R / sys.argv[1]).read_text()) if len(sys.argv) > 1 else \
    json.loads(sorted(R.glob("physion_ocp_color-*.json"))[-1].read_text())
pub = json.loads((R / "physion_published_baselines.json").read_text())["table_s1"]
scen = ["Dominoes", "Support", "Collide", "Contain", "Drop", "Link", "Roll", "Drape"]
print("| row | " + " | ".join(scen) + " |")
print("|---|" + "---|" * len(scen))
for m in ("Human", "pDEIT-lstm", "RPIN", "DPI"):
    print(f"| {m} (published) | " + " | ".join(f"{pub[m][s]['acc']:.3f}" for s in scen) + " |")
for prot in ("observed", "observed+simulated", "full"):
    cells = []
    for s in scen:
        r = ours["scenarios"].get(s)
        cells.append(f"{r[prot]['mean']['accuracy']:.3f}" if r else "–")
    print(f"| ours {ours['frontend']} {prot} | " + " | ".join(cells) + " |")
cells = [str(ours["scenarios"][s]["n_readout"]) if s in ours["scenarios"] else "0" for s in scen]
print("| n readout stimuli (reduced set) | " + " | ".join(cells) + " |")
