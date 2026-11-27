#!/usr/bin/env bash
# V3 §0d exploratory training (Arm I, Arm P); one gpu_small job at a time. Same budget as the frozen matrix.
cd ~/dynagraph_v3
until [ -f data/v3_chain/train_cf.npz ]; do sleep 60; done
for spec in "arm_i_s0 --kind arm_i --cf-frac 0.3 --seed 0" "proj_s0 --kind proj --seed 0"; do
  set -- $spec; name=$1; shift
  [ -f results/v3/checkpoints/$name.pt ] && continue
  scripts/v3/gpu_job.sh $name -m physmem.v3_chain train --run $name --steps 4000 --bs 512 --log-every 200 --eval-every 1000 "$@"
  while kill -0 $(cat logs/$name.pid) 2>/dev/null; do sleep 30; done
  grep -q "done" logs/$name.log && echo "[$(date -u +%T)] $name done" || { echo "[$(date -u +%T)] $name FAILED"; tail -5 logs/$name.log; }
done
echo TRAIN_EXPLORE_DONE
