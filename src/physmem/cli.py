"""physmem command line. The CLI and python SDK call the same functions.

  physmem doctor
  physmem generate --config configs/sprint.yaml
  physmem train --suite compact-factorial --config configs/sprint.yaml
  physmem train --suite inkling-pair --config configs/sprint.yaml
  physmem evaluate --suite primary --config configs/sprint.yaml
  physmem demo --robot g1 --task rail-to-table|tabletop-push --mode simulation
  physmem report --results results/
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def load_config(path):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def doctor(out="results/doctor.json"):
    rep = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(),
           "machine": platform.machine(), "python": sys.version.split()[0]}
    try:
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: l.split(":")[1].strip() for l in f}
        rep["mem_total"] = mi.get("MemTotal"); rep["mem_available"] = mi.get("MemAvailable"); rep["swap_free"] = mi.get("SwapFree")
    except OSError:
        pass
    du = shutil.disk_usage(".")
    rep["disk_free_gb"] = round(du.free / 1e9, 1)
    try:
        rep["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        rep["nvidia_smi"] = f"unavailable: {e}"
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rep["device"] = torch.cuda.get_device_name(0)
            rep["capability"] = list(torch.cuda.get_device_capability(0))
            x = torch.randn(256, 256, device="cuda", requires_grad=True)
            w = torch.randn(256, 256, device="cuda")
            y = (x @ w).tanh().sum()
            y.backward()
            ref = (x.detach().cpu() @ w.cpu()).tanh().sum()
            rep["cuda_fwd_bwd_ok"] = bool(torch.isfinite(x.grad).all().item()) and abs(float(y) - float(ref)) < 1e-2 * abs(float(ref)) + 1e-2
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = (x @ w).float().sum()
            rep["bf16_autocast_ok"] = bool(torch.isfinite(z).item())
    except Exception as e:  # noqa: BLE001
        rep["torch_error"] = repr(e)
    for mod in ("mujoco", "transformers", "numpy", "safetensors"):
        try:
            rep[mod] = importlib.import_module(mod).__version__
        except Exception as e:  # noqa: BLE001
            rep[mod] = f"missing: {e}"
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>")
        d = mujoco.MjData(m)
        for _ in range(100):
            mujoco.mj_step(m, d)
        rep["mujoco_step_ok"] = bool(d.qpos[2] < 0)
    except Exception as e:  # noqa: BLE001
        rep["mujoco_error"] = repr(e)
    ck = Path("ckpt/Inkling-2-Mini/model-00001-of-00001.safetensors")
    rep["inkling_checkpoint_present"] = ck.exists()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rep


def _call(module, fn, *a, **k):
    mod = importlib.import_module(f"physmem.{module}")
    if not hasattr(mod, fn):
        raise SystemExit(f"physmem.{module}.{fn} is not implemented")
    return getattr(mod, fn)(*a, **k)


def main(argv=None):
    p = argparse.ArgumentParser("physmem")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    g = sub.add_parser("generate"); g.add_argument("--config", default="configs/sprint.yaml")
    t = sub.add_parser("train"); t.add_argument("--suite", required=True); t.add_argument("--config", default="configs/sprint.yaml")
    e = sub.add_parser("evaluate"); e.add_argument("--suite", default="primary"); e.add_argument("--config", default="configs/sprint.yaml")
    d = sub.add_parser("demo"); d.add_argument("--robot", default="g1"); d.add_argument("--task", default="tabletop-push")
    d.add_argument("--mode", default="simulation"); d.add_argument("--model", default="b11"); d.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("report"); r.add_argument("--results", default="results/")
    a = p.parse_args(argv)
    if a.cmd == "doctor":
        doctor()
    elif a.cmd == "generate":
        _call("data", "generate", load_config(a.config))
    elif a.cmd == "train":
        if a.suite.startswith("inkling"):
            _call("models.inkling", "train_suite", a.suite, load_config(a.config))
        else:
            _call("train", "train_suite", a.suite, load_config(a.config))
    elif a.cmd == "evaluate":
        _call("evaluate", "evaluate_suite", a.suite, load_config(a.config))
    elif a.cmd == "demo":
        if a.mode != "simulation":
            raise SystemExit("only --mode simulation is available: no physical robot access was established in this sprint")
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        demo = importlib.import_module("demo_g1")
        demo.main(["--task", a.task, "--model", a.model, "--seed", str(a.seed)])
    elif a.cmd == "report":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        importlib.import_module("make_report").main(["--results", a.results])


if __name__ == "__main__":
    main()
