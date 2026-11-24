#!/usr/bin/env bash
# MODEL-V2: multi-robot training on converted ENVS/ROBOTS tranches, robot-balanced, 3 seeds (remote side)
data=${1:-data/v2_robots_t12}; tag=${2:-t12b}
for s in 0 1 2; do for v in typed generic nomem; do
  [ $s -gt 0 ] && [ $v = nomem ] && continue          # r5: typed + generic with 3 seeds (headline); nomem seed 0
  scripts/gpu_heavy.sh v2_robots_${tag}_${v}_s$s "python -m physmem.v2_train train --data $data --variant $v --run v2_robots_${tag}_${v}_s$s --seed $s --steps 8000 --bs 256 --eval-every 4000 --robot-balance --eval-splits test,heldout_robot,heldout_layout,heldout_objects,heldout_family" > logs/v2_robots_${tag}_${v}_s$s.log 2>&1
done; done
