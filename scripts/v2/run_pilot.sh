#!/usr/bin/env bash
# MODEL-V2: val-seed planner pilot (typed + oracle only), one CPU process per family (remote side).
tag=$1; shift
mkdir -p logs
for fam in slide lift insert drawer sweep; do
  V2_RESULTS=results/v2_pilot CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nohup python -m physmem.v2_train closedloop \
    --typed results/v2/checkpoints/v2_synth_r1_typed_s0.pt --run pilot_${tag}_$fam --families $fam --splits val \
    --episodes 30 --pop 192 --iters 3 --n-z 24 "$@" > logs/v2_pilot_${tag}_$fam.log 2>&1 &
done
wait
