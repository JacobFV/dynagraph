#!/usr/bin/env bash
# remote: while tranche $1 runs, if `free` shows < $2 GB available twice in a row (60 s apart), pause it cleanly and
# resume with $3 processes (once). Only touches the generator recorded in <dir>/generator.pid (envs_tranche_ctl.py).
T=$1; LIM=${2:-45}; P2=${3:-4}
D=~/dynagraph/data/v2_envs_large/$T; LOG=~/dynagraph/logs/envs_mem_guard_$T.log
cd ~/dynagraph_r6 || exit 1
PY=~/dynagraph/.venv/bin/python
low=0
while true; do
  pid=$(cat $D/generator.pid 2>/dev/null)
  if [ -z "$pid" ] || ! grep -q envs_generate /proc/$pid/cmdline 2>/dev/null; then echo "$(date -u +%FT%TZ) generator gone, guard exits" >> $LOG; exit 0; fi
  av=$(free -g | awk '/^Mem:/{print $7}')
  if [ "$av" -lt "$LIM" ]; then low=$((low+1)); else low=0; fi
  if [ $low -ge 2 ]; then
    echo "$(date -u +%FT%TZ) available ${av} GB < ${LIM}: pausing and resuming with $P2 procs" >> $LOG
    $PY scripts/envs_tranche_ctl.py stop $T >> $LOG 2>&1
    ENVS_PROCS=$P2 $PY scripts/envs_tranche_ctl.py resume $T 27000 >> $LOG 2>&1
    echo "- $(date -u +%Y-%m-%dT%H:%MZ) $T: memory guard (available ${av} GB < ${LIM} GB) -> resumed with $P2 procs" >> ~/dynagraph/results/v2/envs/large_status.md
    exit 0
  fi
  sleep 60
done
