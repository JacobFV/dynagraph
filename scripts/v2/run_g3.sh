#!/usr/bin/env bash
# MODEL-V2: paired closed-loop G3 runs on the synthetic world, one CPU process per family (remote side).
# usage: scripts/v2/run_g3.sh <tag> <splits> <episodes> [extra planner flags...]
tag=$1; splits=$2; eps=$3; shift 3
ck=results/v2/checkpoints
mkdir -p logs
for fam in slide lift insert drawer sweep; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nohup python -m physmem.v2_train closedloop \
    --typed $ck/v2_synth_r1_typed_s0.pt --generic $ck/v2_synth_r1_generic_s0.pt --nomem $ck/v2_synth_r1_nomem_s0.pt \
    --run v2_g3_${tag}_${fam} --families $fam --splits $splits --episodes $eps "$@" > logs/v2_g3_${tag}_${fam}.log 2>&1 &
done
wait
