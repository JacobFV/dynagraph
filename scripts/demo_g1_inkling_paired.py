"""Inkling core in the G1 loop (e09). Waits until results/checkpoints/inkling-main-i11-s0.pt exists AND
logs/gpu_ledger.log contains "END inkling-eval", picks the best i00 / i11 seed by checkpoint val nll, then
runs both on the slide-to-target paired seeds under scripts/gpu_heavy.sh (lock name "g1-inkling"), one
model load per process (predictor-outer order). Label: "inkling core in the loop, sim paused during inference".
  nohup python scripts/demo_g1_inkling_paired.py [--seeds 0-29] [--deadline-utc 2026-11-23T15:00] &
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")
from physmem import control as C  # noqa: E402


def ready():
    if not os.path.exists("results/checkpoints/inkling-main-i11-s0.pt"):
        return False
    try:
        return "END inkling-eval" in Path("logs/gpu_ledger.log").read_text()
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--fallback-seeds", default="0-19", help="used if less than --min-hours remain before the deadline")
    ap.add_argument("--min-hours", type=float, default=1.5)
    ap.add_argument("--deadline-utc", default="2026-11-23T15:30")
    ap.add_argument("--tag", default="slide_inkling")
    a = ap.parse_args()
    deadline = datetime.fromisoformat(a.deadline_utc).replace(tzinfo=timezone.utc).timestamp()
    while not ready():
        if time.time() > deadline:
            print("deadline reached; inkling checkpoints / eval not ready", flush=True)
            sys.exit(1)
        time.sleep(60)
    picks = {v: C.best_inkling_seed(v) for v in ("i00", "i11")}
    seeds = a.seeds if (deadline - time.time()) / 3600 >= a.min_hours else a.fallback_seeds
    tag_dir = Path("results/g1") / a.tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    sel = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), rule="min checkpoint['val'] nll per variant",
               picks={v: dict(val_nll=p[0], seed=p[1], path=p[2]) for v, p in picks.items() if p},
               seeds=seeds, label=C.InklingPredictor.label)
    (tag_dir / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel), flush=True)
    preds = [f"inkling:{v}:{p[1]}" for v, p in picks.items() if p]
    cmd = ("python scripts/demo_g1.py --benchmark slide-to-target --seeds " + seeds + " --predictors " + " ".join(preds)
           + f" --predictor-outer --tag {a.tag} --merge-global")
    rc = subprocess.call(["scripts/gpu_heavy.sh", "g1-inkling", cmd])
    subprocess.call([sys.executable, "scripts/demo_g1_summarize.py", a.tag])
    sys.exit(rc)


if __name__ == "__main__":
    main()
