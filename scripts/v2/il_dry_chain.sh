#!/usr/bin/env bash
# round-6 DRY RUN chain (envs-r6): wait for demos -> train ACT (GPU under the lock if it is free and >= 25 GB RAM
# free, else CPU, 1 niced proc) -> closed-loop VAL eval (2 procs). Resumable (each step skips finished outputs).
cd ~/dynagraph && source .venv/bin/activate
export PYTHONPATH=src OMP_NUM_THREADS=2 EXPECT_ENV=envs-r6
until grep -q IL_DRY_RECORD_DONE logs/v2_il_dry_record.log; do sleep 30; done
RUN=${RUN:-dry_r6a}
ARGS="data/v2_il_dry $RUN --steps ${STEPS:-4000} --bs 128 --d 128 --H 20 --beta 10"
if [ ! -f results/v2/checkpoints/il_$RUN.pt ]; then
  if flock -n .gpu_heavy.lock true && [ $(free -g | awk '/Mem:/ {print $7}') -ge 25 ]; then
    scripts/gpu_heavy.sh il_$RUN "nice -n 10 python scripts/v2/il_train.py $ARGS" >> logs/v2_il_$RUN.train.log 2>&1
  else
    CUDA_VISIBLE_DEVICES= nice -n 10 python scripts/v2/il_train.py $ARGS >> logs/v2_il_$RUN.train.log 2>&1
  fi
fi
PROCS=2 scripts/v2/il_dry_eval.sh $RUN
echo IL_DRY_CHAIN_DONE >> logs/v2_il_$RUN.train.log
