"""Env-drift guard for the pre-registered closed-loop cells (docs/INKLING_SYS12.md section a / erratum a1).
Run from the frozen tree ~/dynagraph_sys12 (git HEAD 3364798 export = envs-r6 + sys12 overlay):
  cd ~/dynagraph_sys12 && PYTHONPATH=src CELL=... python scripts/sys12/cl_guard.py <tag>
Aborts (exit 3) unless: physmem is imported from this tree, physmem.envs.ENV_VERSION == "envs-r6", every file under
envs/, skills/, embodiments/ is byte-identical to ~/dynagraph_r6, and the pre-registered System I/II files have their
registered sha256. Writes a provenance json next to the cell output, then runs scripts/sys12/closed_loop.py."""
import hashlib
import json
import os
import runpy
import sys
import time

HERE = os.path.abspath(os.getcwd())
R6 = os.path.expanduser("~/dynagraph_r6")
REGISTERED = {  # sha256 at registration (section a), files whose content is unaffected by the env contamination
    "scripts/sys12/closed_loop.py": "3880f26ee941e61128f813f57dac7511ace9a1e6712fca75e8a409ce648f6d38",
    "src/physmem/models/sys12.py": "ca14dae0fa396d3ba6a819949e284c23d742de74185355c4dd5e391d24f52f7e",
    "src/physmem/models/inkling_quant.py": "74f95f32094054f46263248ca0d84dfda1d882ffd8b858d3b094b291d0193d01",
    "src/physmem/models/v2/system1.py": "6a7366b8a31d7b920a0971879538bc41d8cb38753b1c8aa77db0150cf1c8b3d8",
    "src/physmem/models/v2/lot.py": "7b8817707ca87e1c668bd1815b86544cfa59a718355b512879c907d0df970343",
    "src/physmem/sys12_train.py": "3ab4d0b411dd718aaea414a87b3ee6579e143a7fe5f3a8556f1b43db410e322d",
}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fail(msg):
    print(f"[cl_guard] ABORT: {msg}", flush=True)
    sys.exit(3)


sys.path.insert(0, os.path.join(HERE, "src"))
import physmem  # noqa: E402
import physmem.envs as E  # noqa: E402

if not os.path.abspath(physmem.__file__).startswith(os.path.join(HERE, "src")):
    fail(f"physmem imported from {physmem.__file__}, not {HERE}/src")
if E.ENV_VERSION != "envs-r6":
    fail(f"ENV_VERSION = {E.ENV_VERSION!r}")
prov = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), tree=HERE, env_version=E.ENV_VERSION,
            physmem=physmem.__file__, files={})
for d in ("envs", "skills", "embodiments"):
    mine = os.path.join(HERE, "src/physmem", d)
    ref = os.path.join(R6, "src/physmem", d)
    names = set()
    for root in (mine, ref):
        for dp, _, fs in os.walk(root):
            for f in fs:
                if "__pycache__" not in dp and not f.endswith(".pyc"):
                    names.add(os.path.relpath(os.path.join(dp, f), root))
    for n in sorted(names):
        a, b = os.path.join(mine, n), os.path.join(ref, n)
        if not (os.path.exists(a) and os.path.exists(b)):
            fail(f"{d}/{n} present in only one of {HERE}, {R6}")
        if sha(a) != sha(b):
            fail(f"{d}/{n} differs from {R6}")
        prov["files"][f"src/physmem/{d}/{n}"] = sha(a)
for f, h in REGISTERED.items():
    if sha(os.path.join(HERE, f)) != h:
        fail(f"{f} sha256 differs from the registered one")
    prov["files"][f] = h
for f in ("src/physmem/models/v2/harness.py", "src/physmem/models/v2/env_adapter.py", "src/physmem/planner.py",
          "src/physmem/models/v2/spec.py", "src/physmem/models/v2/tensorize.py", "src/physmem/models/v2/model.py"):
    prov["files"][f] = sha(os.path.join(HERE, f))
cell = os.environ.get("CELL", "bin_ur5e")
os.makedirs("results/sys12/closed_loop", exist_ok=True)
json.dump(prov, open(f"results/sys12/closed_loop/provenance_{sys.argv[1]}_{cell}.json", "w"), indent=1)
print(f"[cl_guard] OK envs-r6, {len(prov['files'])} files verified -> running closed_loop.py {cell}", flush=True)
if os.environ.get("GUARD_ONLY"):
    sys.exit(0)
sys.argv = ["scripts/sys12/closed_loop.py"] + sys.argv[1:]
runpy.run_path("scripts/sys12/closed_loop.py", run_name="__main__")
