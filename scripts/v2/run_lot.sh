#!/usr/bin/env bash
# MODEL-V2: "lot" protocol (L items, same hidden properties, sequential) closed loop, one CPU process per family.
# usage: scripts/v2/run_lot.sh <tag> <splits> <episodes> <lot> [extra flags]
tag=$1; splits=$2; eps=$3; lot=$4; shift 4
ck=results/v2/checkpoints
mkdir -p logs
for fam in ${FAMS:-slide lift insert drawer sweep}; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nohup python -m physmem.v2_train closedloop \
    --typed $ck/v2_synth_r1_typed_s0.pt --generic $ck/v2_synth_r1_generic_s0.pt --nomem $ck/v2_synth_r1_nomem_s0.pt \
    --run v2_lot_${tag}_${fam} --families $fam --splits $splits --episodes $eps --lot $lot --no-voi-variant "$@" \
    > logs/v2_lot_${tag}_${fam}.log 2>&1 &
done
wait
