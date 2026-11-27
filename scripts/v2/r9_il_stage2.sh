#!/usr/bin/env bash
# ACT VAL tuning stages 2-3 (13a). Stage-1 winner by mean VAL success, ties -> lowest offline val L1: r9t_h10v.
# stage 2: beta 1 (20k) and 50k steps (beta 10); stage 3 (eval only): ensembling m in {0.01, open} on the winner.
cd ~/dynagraph
( BETA=1 STEPS=20000 CONFIGS="h10v_b1:--H 10 --img 64" PREFIX=r9t scripts/v2/r9_il_gpu.sh
  BETA=10 STEPS=50000 CONFIGS="h10v_50k:--H 10 --img 64" PREFIX=r9t scripts/v2/r9_il_gpu.sh ) >> logs/v2_r9t_stage2_gpu.log 2>&1 &
for n in h10v_b1 h10v_50k; do RUN=r9t_$n NO_TRAIN=1 ENS_M=0.1 NVAL=20 PROCS=1 scripts/v2/r9_il_tune.sh; done
echo R9_STAGE2_DONE >> logs/v2_r9_il_stage2.log
