#!/usr/bin/env bash
# MODEL-V2: val-seed cost-weight sweep (typed planner only), remote side
mkdir -p logs
for cp in 0.5 1.0 2.0; do for le in 0.03 0.1; do for fam in drawer slide; do
  V2_RESULTS=results/v2_pilot CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nohup python -m physmem.v2_train closedloop \
    --typed results/v2/checkpoints/v2_synth_r1_typed_s0.pt --run sweep_cp${cp}_le${le}_$fam --families $fam --splits val \
    --episodes 24 --pop 128 --iters 3 --n-z 16 --typed-only --no-voi-variant --c-prog $cp --lam-effort $le \
    > logs/v2_sweep_cp${cp}_le${le}_$fam.log 2>&1 &
done; done; done
wait
