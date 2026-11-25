#!/usr/bin/env bash
# MODEL-V2 round 4 (r7) training: base r2-r6 + tranche8 (tactile feature) + on-policy rounds 1-2; typed & generic x 3 seeds.
# niced CPU, GPU via gpu_heavy, checks free memory (>= 20 GB available) before each GPU job; idempotent.
cd ~/dynagraph
hold() { while [ -f ~/dynagraph/HOLD ]; do sleep 60; done; }
[ -d data/v2_r7_final ] || { nice -n 10 python scripts/v2/merge_data.py data/v2_r7_t8base data/v2_r6_base data/v2_r7_t8 > logs/v2_r7_merge.log 2>&1 &&
  nice -n 10 python scripts/v2/merge_onpolicy.py data/v2_r7_t8base data/v2_r7_final data/v2_r6_rollouts/round1 data/v2_r6_rollouts/round2 >> logs/v2_r7_merge.log 2>&1; }
for s in 0 1 2; do for v in typed generic; do
  run=v2_robots_r7_${v}_s$s
  [ -f results/v2/checkpoints/$run.pt ] && continue
  hold
  until [ $(free -g | awk '/Mem:/ {print $7}') -ge 20 ]; do sleep 60; done
  scripts/gpu_heavy.sh $run "nice -n 10 python -m physmem.v2_train train --data data/v2_r7_final --variant $v --run $run --seed $s --steps 8000 --bs 256 --eval-every 2000 --robot-balance --eval-splits test,heldout_robot,heldout_family" > logs/$run.log 2>&1
done; done
echo R7_TRAIN_DONE
