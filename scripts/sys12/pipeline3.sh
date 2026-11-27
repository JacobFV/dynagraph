#!/usr/bin/env bash
# after pipeline2: Gen3 offline eval (aux1-masked) + tables. Frozen tree.
cd ~/dynagraph_sys12 && source ~/dynagraph/.venv/bin/activate
export PYTHONPATH=$HOME/dynagraph_sys12/src PYTORCH_ALLOC_CONF=expandable_segments:True
while pgrep -f "[p]ipeline2.sh" > /dev/null; do sleep 120; done
while [ -f ~/dynagraph/HOLD ] || [ -f ~/dynagraph/HOLD_SYS12 ]; do sleep 60; done
[ -f results/sys12/eval/gen3_heldout_robot_aux1masked.json ] || scripts/gpu_heavy.sh sys12_gen3eval "nice -n 10 python scripts/sys12/eval_gen3.py" > logs/sys12/eval_gen3.log 2>&1
nice -n 10 python scripts/sys12/report.py > logs/sys12/report.log 2>&1
echo SYS12_PIPELINE3_DONE
