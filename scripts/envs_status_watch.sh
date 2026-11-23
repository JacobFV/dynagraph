#!/usr/bin/env bash
# remote: append a status line to results/v2/envs/large_status.md when each tranche's manifest.sha256 appears.
D=~/dynagraph/data/v2_envs_large
OUT=~/dynagraph/results/v2/envs/large_status.md
for t in "$@"; do
  until test -s "$D/$t/manifest.sha256"; do sleep 120; done
  line=$(python3 - "$D/$t" <<'PY'
import json, sys, time
d = sys.argv[1]
m = json.load(open(d + "/manifest.json"))["runs"][-1]
fam = m.get("families", {})
err = sum(v.get("errors", 0) for v in fam.values()); viol = sum(v.get("violations", 0) for v in fam.values())
sha = open(d + "/manifest.sha256").read().split()[0]
print(f"- {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())} FINAL {d.split('/')[-1]}: env {m.get('env_version', '?')}, "
      f"{m.get('episodes', m.get('completed'))} episodes, {m.get('bytes_total', 0) / 1e9:.2f} GB, "
      f"{m.get('wall_s', 0) / 3600:.2f} h, errors {err}, integrity violations {viol}, manifest sha256 {sha}")
PY
)
  echo "$line" >> "$OUT"
done
