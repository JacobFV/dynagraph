"""results/manifest.json: artifact provenance + experiment statuses."""
import hashlib, json, glob, os, subprocess, time
from pathlib import Path
R = Path("results")
def sha(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()
v = json.loads((R / "verdicts.json").read_text())
ledger = [json.loads(l) for l in open(R / "ledger.jsonl")]
data_manifests = {d: json.loads(Path(f"data/{d}/manifest.json").read_text()).get("hashes", "see file") if Path(f"data/{d}/manifest.json").exists() else None
                  for d in ("v1", "v2", "v3", "g1cal")}
m = {
  "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "sprint": {"start_utc": "2026-11-23T04:45:53Z", "hard_stop_utc": "2026-11-23T16:45:53Z", "host": "gb10-direct (promaxgb10-4dfb), NVIDIA GB10 cc 12.1"},
  "environment": {"python": "3.12.13", "torch": "2.9.1+cu130", "transformers": "5.24.0", "accelerate": "1.16.0", "mujoco": "4.0.1",
                  "doctor": json.loads((R / "doctor.json").read_text()) if (R / "doctor.json").exists() else None},
  "inkling": {"repo": "thinkingmachines/Inkling-2-Mini", "revision": "b85c3cfa615fc413cc9af3379f491b963f3b586a",
              "weights_sha256": (R / "inkling_ckpt_sha256.txt").read_text().split()[0] if (R / "inkling_ckpt_sha256.txt").exists() else None,
              "gate": {k: json.loads((R / "inkling_gate.json").read_text()).get(k) for k in ("passed", "passed_with_amendment")}},
  "g1_asset": "mujoco_menagerie unitree_g1 (g1_with_hands) commit f3b6013; modifications in src/physmem/robots/assets/README.md; license BSD-3-Clause upstream (not verified in local copy)",
  "datasets": {d: f"data/{d}/manifest.json" for d in data_manifests if data_manifests[d] is not None},
  "experiments": {k: e for k, e in v["experiments"].items()},
  "hypotheses": {k: h["verdict"] for k, h in v["hypotheses"].items()},
  "runs": [{k: r.get(k) for k in ("run_id", "kind", "variant", "seed", "status", "config_hash", "steps", "examples", "wall_s", "best_val_nll", "tag")}
           for r in ledger if r.get("run_id")],
  "artifacts": {},
  "labels": {"compact": "compact recurrent physical-memory model (NOT inkling)",
             "inkling": "actual Inkling-2-Mini recurrent core, frozen base + adapters/LoRA",
             "g1": "unitree g1 · fixed-base simulation · synthetic sensors; push endpoint (lift/place gate failed 3/10)"},
}
for p in ["demo.mp4", "metrics.jsonl", "summary.csv", "ledger.jsonl", "findings.md", "verdicts.json", "inkling_gate.json", "reproduction.json", "gpu_ledger.log"]:
    if (R / p).exists():
        m["artifacts"][p] = {"sha256": sha(R / p), "bytes": (R / p).stat().st_size}
for p in sorted(glob.glob("results/checkpoints/*.pt")):
    if "_at" in p:
        continue
    m["artifacts"][p[len("results/"):]] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
(R / "manifest.json").write_text(json.dumps(m, indent=1, default=str))
print("artifacts", len(m["artifacts"]), "runs", len(m["runs"]))
