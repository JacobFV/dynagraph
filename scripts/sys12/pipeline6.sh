#!/usr/bin/env bash
# EXPLORATORY rule-vs-model diagnostic (lead request), after pipeline5. Frozen tree; guard checked first.
cd ~/dynagraph_sys12 && source ~/dynagraph/.venv/bin/activate
export PYTHONPATH=$HOME/dynagraph_sys12/src MUJOCO_GL=egl PYTORCH_ALLOC_CONF=expandable_segments:True EPISODE_TIMEOUT=600
while pgrep -f "[p]ipeline5.sh" > /dev/null; do sleep 120; done
GUARD_ONLY=1 CELL=explore python scripts/sys12/cl_guard.py explore1 > logs/sys12/explore_guard.log 2>&1 || { echo GUARD_FAIL; exit 3; }
for cell in bin_ur5e pouch_ur5e; do
  [ -f results/sys12/closed_loop/explore1_$cell.json ] && continue
  while [ -f ~/dynagraph/HOLD ] || [ -f ~/dynagraph/HOLD_SYS12 ]; do sleep 60; done
  until [ $(free -g | awk '/Mem:/ {print $7}') -ge 40 ]; do sleep 60; done
  scripts/gpu_heavy.sh sys12_explore_$cell "CELL=$cell N=30 nice -n 10 python scripts/sys12/explore_rule.py" >> logs/sys12/explore_$cell.log 2>&1
done
echo SYS12_PIPELINE6_DONE
