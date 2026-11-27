#!/usr/bin/env bash
# INKLING System I round 1: 4 arms, identical sample sequence (seed 0), 4000 steps x bs 32 = 128k samples each.
# resumable (partial checkpoints), HOLD-aware, waits for >= 40 GB available before each Inkling job. Then evals.
cd ~/dynagraph && source .venv/bin/activate
export PYTHONPATH=src PYTORCH_ALLOC_CONF=expandable_segments:True
STEPS=${STEPS:-4000}
wait_ok() { while [ -f ~/dynagraph/HOLD ] || [ -f ~/dynagraph/HOLD_SYS12 ]; do sleep 60; done
            until [ $(free -g | awk '/Mem:/ {print $7}') -ge ${1:-40} ]; do sleep 60; done; }
for arm in inkling_gated inkling_ungated inkling_random compact; do
  run=s1r1_${arm}_s0
  [ -f results/sys12/checkpoints/$run.pt ] && continue
  wait_ok 40
  scripts/gpu_heavy.sh sys12_$run "nice -n 10 python -m physmem.sys12_train train --arm $arm --run $run --steps $STEPS --bs 32 --seed 0" >> logs/sys12/$run.log 2>&1
done
echo SYS12_TRAIN_DONE
