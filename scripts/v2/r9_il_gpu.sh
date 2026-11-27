#!/usr/bin/env bash
# GPU loop for the ACT VAL grid: trains each listed config (gpu_heavy lock, resumable) if its checkpoint is missing.
# usage: CONFIGS="name:--H 10;name2:--H 20 --img 64" STEPS=20000 BETA=10 PREFIX=r9g r9_il_gpu.sh
cd ~/dynagraph && source .venv/bin/activate
export PYTHONPATH=$HOME/dynagraph_m6run71/src OMP_NUM_THREADS=1 EXPECT_ENV=envs-r7.1
IFS=';' read -ra CF <<< "$CONFIGS"
for cfg in "${CF[@]}"; do
  R=${PREFIX}_${cfg%%:*}
  [ -f results/v2/checkpoints/il_$R.pt ] && continue
  
  scripts/v2/gpu_small.sh il_$R "nice -n 10 python scripts/v2/il_train.py ${DATA:-data/v2_il_r7s} $R --pos abs --d 192 --bs 128 ${cfg#*:} --steps ${STEPS:-20000} --beta ${BETA:-10} --drop-cmd 1" >> logs/v2_il_$R.train.log 2>&1
done
echo GPU_LOOP_DONE $PREFIX
