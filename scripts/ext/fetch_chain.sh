#!/usr/bin/env bash
# EXT download order (lead decision 2026-11-26): Physion++ readout+test -> CLEVRER val -> DexYCB s01 ->
# Physion v1 readout prefixes (5%) -> DexYCB s02-08. One queue so bandwidth follows the priority order.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
$PY scripts/ext/fetch.py physionpp clevrer_val dexycb_s01
nice -n 10 $PY scripts/ext/physion_readout_prefix.py --frac 0.05
$PY scripts/ext/fetch.py dexycb_s02_08
echo "[chain done]"
