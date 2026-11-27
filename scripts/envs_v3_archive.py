"""move a family's calibration rows / oracle table aside (superseded by a task change): envs_v3_archive.py <family> <why>"""
import os
import shutil
import sys
import time

fam, why = sys.argv[1], sys.argv[2]
d = f"results/v2/envs/_superseded_{fam}_{why}_{time.strftime('%Y%m%dT%H%M%S')}"
os.makedirs(d, exist_ok=True)
for f in (f"v3_cal_{fam}_rows.jsonl", f"v3_oracle_{fam}.json"):
    if os.path.exists(f"results/v2/envs/{f}"):
        shutil.move(f"results/v2/envs/{f}", f"{d}/{f}")
print("archived to", d)
