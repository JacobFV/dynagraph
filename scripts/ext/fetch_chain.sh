#!/usr/bin/env bash
# EXT download order (lead decision 2026-11-26): Physion++ readout+test -> CLEVRER val -> DexYCB s01 ->
# Physion v1 readout prefixes (5%) -> DexYCB s02-08. One queue so bandwidth follows the priority order.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
prune() {  # extract + verify + delete (lead rule: footprint <= 300 GB; sha256 already in manifest)
  for f in "$@"; do [ -f "/home/brandonin/datasets/dexycb/raw/$f" ] && \
    $PY scripts/ext/extract_and_prune.py dexycb "$f" /home/brandonin/datasets/dexycb/extracted; done; }
$PY scripts/ext/fetch.py physionpp clevrer_val dexycb_s01
prune 20200709-subject-01.tar.gz models.tar.gz
nice -n 10 $PY scripts/ext/physion_readout_prefix.py --frac 0.05
for s in 20200813-subject-02 20200820-subject-03 20200903-subject-04 20200908-subject-05 20200918-subject-06 \
         20200928-subject-07 20201002-subject-08; do
  $PY - "$s" <<'PYEOF'
import sys; sys.path.insert(0, "scripts/ext"); import fetch
n, lic, src, files = fetch.SOURCES["dexycb_s02_08"]
fetch.SOURCES["one"] = (n, lic, src, [f for f in files if f[0].startswith(sys.argv[1])])
fetch.fetch("one")
PYEOF
  prune "$s.tar.gz"
done
echo "[chain done]"
