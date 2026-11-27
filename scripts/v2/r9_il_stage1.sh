#!/usr/bin/env bash
# ACT VAL tuning stage 1 (13a grid; r9t = controller-command dims dropped + idle start trimmed): H in {10,20,50} x vision {off,on}, 20k steps, beta 10, ENS_M 0.1. Waits for all
# demos and the memory-view backfill; trains each config on the GPU (lock), then evaluates on VAL (2 procs).
cd ~/dynagraph
until [ -f data/v2_il_r7/bin_clearing__panda.npz ] && [ -f data/v2_il_r7/bin_clearing__ur5e_2f85.npz ] && grep -q "MEM_FILL_DONE tight_insert" logs/v2_r9_memfill.log; do sleep 60; done
for cfg in "h10:--H 10" "h20:--H 20" "h50:--H 50" "h10v:--H 10 --img 64" "h20v:--H 20 --img 64" "h50v:--H 50 --img 64"; do
  RUN=r9t_${cfg%%:*} TRAIN_ARGS="${cfg#*:} --steps 20000 --beta 10" NO_TRAIN=1 ENS_M=0.1 NVAL=20 PROCS=${EVAL_PROCS:-1} scripts/v2/r9_il_tune.sh
done
echo R9_STAGE1_DONE >> logs/v2_r9_il_stage1.log
