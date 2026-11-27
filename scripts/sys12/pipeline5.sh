#!/usr/bin/env bash
# after pipeline4: MODEL-V2 references re-evaluated on the arms' exact row subsets, then tables.
cd ~/dynagraph_sys12 && source ~/dynagraph/.venv/bin/activate
export PYTHONPATH=$HOME/dynagraph_sys12/src PYTORCH_ALLOC_CONF=expandable_segments:True
while pgrep -f "[p]ipeline4.sh" > /dev/null; do sleep 120; done
while [ -f ~/dynagraph/HOLD ] || [ -f ~/dynagraph/HOLD_SYS12 ]; do sleep 60; done
[ -f results/sys12/eval/v2_r7_typed_s0_misaligned.json ] || scripts/gpu_heavy.sh sys12_evalv2_aligned "nice -n 10 python scripts/sys12/eval_v2_aligned.py" > logs/sys12/evalv2_aligned.log 2>&1
nice -n 10 python scripts/sys12/report.py > logs/sys12/report.log 2>&1
echo SYS12_PIPELINE5_DONE
