#!/usr/bin/env bash
# MODEL-V2: r4 synthetic G3 (test + held-out robot), single-item and lot-4, all policies incl. prior-mean + oracle.
# usage (remote): scripts/v2/run_g3_r4.sh <seed> [typed_ckpt_override]
s=${1:-0}; C=results/v2/checkpoints
T=${2:-$C/v2_synth_r4_typed_s$s.pt}
tag=$(basename $T .pt)
for lot in 1 4; do
  python -m physmem.v2_train closedloop --typed $T --generic $C/v2_synth_r4_generic_s$s.pt --nomem $C/v2_synth_r4_nomem_s$s.pt \
    --run v2_g3_${tag}_lot$lot --splits test,test_robot --episodes 100 --lot $lot --no-voi-variant --c-prog 2.0 --lam-effort 0.03 \
    > logs/v2_g3_${tag}_lot$lot.log 2>&1
done
