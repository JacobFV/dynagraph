#!/usr/bin/env bash
# MODEL-V2 round 6 memory models (13a), compute-matched typed / generic / nomem.
#   REGIME=budget: data = memory-model view of exactly the il-v1 demo episodes (+ VAL demo episodes for selection)
#   REGIME=full  : DATA=<dir> (r2-r6 + tranche9_r7(.1) + demos), set by the caller
# Grid on seed 0 per variant: lr {3e-4, 1e-3} x d {128, 192}; selection = lowest VAL outcome NLL (v2_train eval);
# then seeds 1, 2 with the selected (lr, d). GPU via gpu_heavy, each job <= ~1 h, resumable (.partial.pt).
cd ~/dynagraph && source .venv/bin/activate
scripts/v2/m6_overlay.sh > /dev/null || exit 1
export PYTHONPATH=$HOME/dynagraph_m6run71/src OMP_NUM_THREADS=2 EXPECT_ENV=envs-r7.1
REGIME=${REGIME:-budget}; STEPS=${STEPS:-$([ "${REGIME:-budget}" = budget ] && echo 4000 || echo 8000)}
if [ "$REGIME" = budget ]; then
  DATA=data/v2_r9_budget
  [ -d $DATA ] || CUDA_VISIBLE_DEVICES= nice -n 10 python scripts/v2/r9_mem_data.py data/v2_il_r7/mem data/v2_il_r7_val/mem $DATA
fi
: "${DATA:?}"
train() {  # variant seed lr d
  run=v2_robots_r9_${REGIME}_$1_s$2_lr$3_d$4
  [ -f results/v2/checkpoints/$run.pt ] && return
  until [ $(free -g | awk '/Mem:/ {print $7}') -ge 30 ]; do sleep 60; done
  scripts/v2/gpu_small.sh $run "nice -n 10 python -m physmem.v2_train train --data $DATA --variant $1 --run $run --seed $2 --lr $3 --d $4 --steps $STEPS --bs 256 --eval-every 2000 --robot-balance --eval-splits val --no-test" > logs/$run.log 2>&1
}
for v in typed generic nomem; do for lr in 3e-4 1e-3; do for d in 128 192; do train $v 0 $lr $d; done; done; done
echo R9_MEM_GRID_DONE $REGIME >> logs/v2_r9_mem_$REGIME.log
