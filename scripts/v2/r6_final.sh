#!/usr/bin/env bash
# MODEL-V2 round 3 FINAL stage (remote), resumable + HOLD-aware (shared host: touch ~/dynagraph/HOLD to pause).
# every step is skipped if its output exists; CPU work is niced; GPU work goes through scripts/gpu_heavy.sh.
set -x
cd ~/dynagraph
export EXCLUDE_TRAINVAL_FAMILY=drawer_open
hold() { while [ -f ~/dynagraph/HOLD ]; do sleep 60; done; }
conv() { [ -f $2/manifest.json ] && return; hold; nice -n 10 python -m physmem.v2_train convert --pattern "data/v2_envs_large/$1/**/*.pkl.gz" --out $2 --workers ${CONV_W:-6} > logs/v2_r6_convert_$(basename $2).log 2>&1; }
gpu_train() {  # $1 data $2 variant $3 run $4 seed
  [ -f results/v2/checkpoints/$3.pt ] && return; hold
  scripts/gpu_heavy.sh $3 "python -m physmem.v2_train train --data $1 --variant $2 --run $3 --seed $4 --steps 8000 --bs 256 --eval-every 2000 --robot-balance --eval-splits test,heldout_robot,heldout_family" > logs/$3.log 2>&1
}
until [ -f data/v2_envs_large/tranche8_r6_site/manifest.sha256 ]; do sleep 120; done
conv tranche8_r6_site data/v2_r6_t8
hold
[ -d data/v2_r6_final ] || { nice -n 10 python scripts/v2/merge_data.py data/v2_r6_t8base data/v2_r6_base data/v2_r6_t8 > logs/v2_r6_merge_t8.log 2>&1 &&
  nice -n 10 python scripts/v2/merge_onpolicy.py data/v2_r6_t8base data/v2_r6_final data/v2_r6_rollouts/round1 data/v2_r6_rollouts/round2 > logs/v2_r6_merge_final.log 2>&1; }
for s in 0 1 2; do for v in typed generic; do gpu_train data/v2_r6_final $v v2_robots_r6f_${v}_s$s $s; done; done
gpu_train data/v2_r6_final nomem v2_robots_r6f_nomem_s0 0
if [ ! -f results/v2/trust_r6f.json ]; then hold
  nice -n 10 python -m physmem.v2_train eval --ckpt results/v2/checkpoints/v2_robots_r6f_typed_s0.pt --data data/v2_r6_final --splits val > logs/v2_eval_r6f_val.log 2>&1
  python scripts/v2/make_trust.py logs/v2_eval_r6f_val.log results/v2/trust_r6f.json v2_robots_r6f_typed_s0; fi
echo TRAINING_DONE
PROCS=${PROCS:-8} scripts/v2/run_robot_table_r6.sh r6f results/v2/trust_r6f.json
echo FINAL_DONE
