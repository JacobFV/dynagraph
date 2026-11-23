"""Wait for compact-main checkpoints, pick the best seed per variant by VALIDATION nll (ck['val']['nll']),
then run paired slide-to-target episodes: b00 vs b11 (+ stub:0.5 + oracle) on seeds 0-29, then 30-49.
Rows are merged into results/metrics.jsonl (experiment e09). Usage (remote, nohup):
  python scripts/demo_g1_compact_paired.py [--variants b00 b11] [--quiet-min 15] [--deadline-utc 2026-11-23T13:00]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

CK = Path("results/checkpoints")


def finals_ready(variants):
    return all(glob.glob(str(CK / f"compact-main-{v}-s*_final.pt")) for v in variants)


def newest_mtime():
    fs = glob.glob(str(CK / "compact-main-*.pt"))
    return max((os.path.getmtime(f) for f in fs), default=0.0)


def best_by_val(variant):
    best = None
    for p in glob.glob(str(CK / f"compact-main-{variant}-s*.pt")):
        if not re.search(r"-s\d+\.pt$", p):
            continue  # skip _final/_atXX; the un-suffixed file is the train-loop's best-on-val checkpoint
        if not os.path.exists(p.replace(".pt", "_final.pt")):
            continue  # seed still training
        ck = torch.load(p, map_location="cpu", weights_only=False)
        nll = float(ck["val"]["nll"])
        if best is None or nll < best[0]:
            best = (nll, p, ck.get("seed"), ck.get("config_hash"))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["b00", "b11"])
    ap.add_argument("--quiet-min", type=float, default=15.0)
    ap.add_argument("--deadline-utc", default="2026-11-23T14:30")
    ap.add_argument("--tag", default="slide_compact")
    a = ap.parse_args()
    deadline = datetime.fromisoformat(a.deadline_utc).replace(tzinfo=timezone.utc).timestamp()
    while True:
        quiet = time.time() - newest_mtime() > a.quiet_min * 60
        if finals_ready(a.variants) and quiet:
            break
        if time.time() > deadline:
            print("deadline reached without checkpoints", flush=True)
            sys.exit(1)
        time.sleep(60)
    picks = {v: best_by_val(v) for v in a.variants}
    print("selected (best validation nll):", json.dumps(picks), flush=True)
    Path("results/g1", a.tag).mkdir(parents=True, exist_ok=True)
    Path("results/g1", a.tag, "selection.json").write_text(json.dumps(dict(
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), rule="min ck['val']['nll'] over seeds with _final",
        picks={v: dict(val_nll=p[0], path=p[1], seed=p[2], config_hash=p[3]) for v, p in picks.items()}), indent=1))
    preds = [f"compact:{picks[v][1]}" for v in a.variants] + ["stub:0.5", "oracle"]
    for seeds in ("0-29", "30-49"):
        procs = [subprocess.Popen([sys.executable, "scripts/demo_g1.py", "--benchmark", "slide-to-target", "--seeds", seeds,
                                   "--predictors", p, "--tag", a.tag, "--merge-global"]) for p in preds]
        for pr in procs:
            pr.wait()
        subprocess.run([sys.executable, "scripts/demo_g1_summarize.py", a.tag])


if __name__ == "__main__":
    main()
