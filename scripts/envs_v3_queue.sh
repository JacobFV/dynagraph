#!/usr/bin/env bash
# envs-v3 gb10 job queue: runs the given steps SEQUENTIALLY in ONE niced process slot (ENVS-V3 share: 1 process).
# Each step is a python argument string; the queue stops on a STOP file (logs/v3_queue.stop).
# usage (on gb10, from ~/dynagraph_envs_v3): nohup nice -n 19 bash scripts/envs_v3_queue.sh steps.txt > logs/v3_queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${PY:-$HOME/dynagraph/.venv/bin/python}
while IFS= read -r step; do
  [ -z "$step" ] && continue
  case "$step" in \#*) continue;; esac
  if [ -e logs/v3_queue.stop ]; then echo "$(date -u +%FT%TZ) STOP file: halting"; break; fi
  echo "$(date -u +%FT%TZ) START $step"
  $PY $step < /dev/null
  echo "$(date -u +%FT%TZ) END rc=$? $step"
done < "$1"
echo "$(date -u +%FT%TZ) QUEUE DONE"
