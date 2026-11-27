#!/usr/bin/env bash
# r9 memory models: after both seed-0 grids, take each variant's VAL-selected (lr, d) (r9_select.py), train seeds 1
# and 2 with it (small GPU lock), and link canonical names results/v2/checkpoints/v2_robots_r9_<regime>_<v>_s<k>.pt
cd ~/dynagraph && source .venv/bin/activate
until grep -q R9_MEM_GRIDS_COMPLETE logs/v2_r9_mem_grids.log 2>/dev/null; do sleep 120; done
export PYTHONPATH=$HOME/dynagraph_m6run71/src OMP_NUM_THREADS=2 EXPECT_ENV=envs-r7.1
for REG in budget full; do
  DATA=$([ $REG = budget ] && echo data/v2_r9_budget || echo data/v2_r9_full)
  STEPS=$([ $REG = budget ] && echo 4000 || echo 8000)
  CUDA_VISIBLE_DEVICES= python scripts/v2/r9_select.py $REG > results/v2/r9_select_$REG.txt
  grep ^SELECT results/v2/r9_select_$REG.txt | while read _ v lr d sc; do
    for s in 1 2; do
      run=v2_robots_r9_${REG}_${v}_s${s}_lr${lr}_d${d}
      [ -f results/v2/checkpoints/$run.pt ] || scripts/v2/gpu_small.sh $run "nice -n 10 python -m physmem.v2_train train --data $DATA --variant $v --run $run --seed $s --lr $lr --d $d --steps $STEPS --bs 256 --eval-every 2000 --robot-balance --eval-splits val --no-test" > logs/$run.log 2>&1 < /dev/null
    done
    for s in 0 1 2; do ln -sf v2_robots_r9_${REG}_${v}_s${s}_lr${lr}_d${d}.pt results/v2/checkpoints/v2_robots_r9_${REG}_${v}_s${s}.pt; done
  done
done
echo R9_MEM_SEEDS_DONE >> logs/v2_r9_mem_seeds.log
