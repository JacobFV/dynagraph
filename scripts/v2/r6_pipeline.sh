#!/usr/bin/env bash
# MODEL-V2 round 3 / envs-r6 pipeline (remote). G5: drawer_open train/val excluded everywhere; Gen3 = held-out robot.
set -x
P=${PROCS:-8}
export EXCLUDE_TRAINVAL_FAMILY=drawer_open
conv() { python -m physmem.v2_train convert --pattern "data/v2_envs_large/$1/**/*.pkl.gz" --out $2 --workers 6 > logs/v2_r6_convert_$(basename $2).log 2>&1; }
# 1. r2-r6 tranches with the zero-length-window guard (no more whole-episode drops) + r6 schema
for t in tranche1:t1 tranche2_envs_r3_partial:t2 tranche5_r4:t5 tranche6_r5:t6 tranche3_heldout_robot_r4:t3; do conv ${t%%:*} data/v2_r6_${t##*:}; done
python scripts/v2/merge_data.py data/v2_r6_base data/v2_r6_t1 data/v2_r6_t2 data/v2_r6_t5 data/v2_r6_t6 data/v2_r6_t3 > logs/v2_r6_merge_base.log 2>&1
trust_for() { python -m physmem.v2_train eval --ckpt results/v2/checkpoints/v2_robots_$1_typed_s0.pt --data $2 --splits val > logs/v2_eval_$1_val.log 2>&1
              python scripts/v2/make_trust.py logs/v2_eval_$1_val.log results/v2/trust_$1.json v2_robots_$1_typed_s0; }
typed3() { for s in 0 1 2; do scripts/gpu_heavy.sh v2_robots_$2_typed_s$s "python -m physmem.v2_train train --data $1 --variant typed --run v2_robots_$2_typed_s$s --seed $s --steps 8000 --bs 256 --eval-every 4000 --robot-balance --eval-splits test,heldout_robot" > logs/v2_robots_$2_typed_s$s.log 2>&1; done; }
rollouts() {  # $1 round, $2 model tag, $3 out dir
  cells=""
  for rb in ur5e_2f85 panda; do for f in conveyor_sort bin_clearing singulation door_open tool_return pouch_transfer twist_lock machine_tending fixture_place; do cells="$cells $rb:$f"; done; done
  cells="$cells probe:depalletize ur5e_suction:depalletize probe:conveyor_sort probe:singulation"
  echo $cells | tr ' ' '\n' | grep . | xargs -P $P -I{} bash -c 'c={}; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl python scripts/v2/env_onpolicy.py '$1' ${c%%:*} ${c##*:} ${N_OP:-24} '$3' '$2' results/v2/trust_'$2'.json >> logs/v2_r6_onpolicy_round'$1'.log 2>&1'
}
# 2. interim models + on-policy rounds 1, 2 (while tranche8 generates)
typed3 data/v2_r6_base r6i0 ; trust_for r6i0 data/v2_r6_base
rollouts 1 r6i0 data/v2_r6_rollouts/round1
python scripts/v2/merge_onpolicy.py data/v2_r6_base data/v2_r6_op1 data/v2_r6_rollouts/round1 > logs/v2_r6_merge_op1.log 2>&1
typed3 data/v2_r6_op1 r6i1 ; trust_for r6i1 data/v2_r6_op1
rollouts 2 r6i1 data/v2_r6_rollouts/round2
echo ROUNDS_DONE
# 3. tranche8 (final): wait for its manifest, convert, merge with base + all rollouts, final 3-seed models, table
until [ -f data/v2_envs_large/tranche8_r6_site/manifest.sha256 ]; do sleep 120; done
conv tranche8_r6_site data/v2_r6_t8
python scripts/v2/merge_data.py data/v2_r6_t8base data/v2_r6_base data/v2_r6_t8 > logs/v2_r6_merge_t8.log 2>&1
python scripts/v2/merge_onpolicy.py data/v2_r6_t8base data/v2_r6_final data/v2_r6_rollouts/round1 data/v2_r6_rollouts/round2 > logs/v2_r6_merge_final.log 2>&1
scripts/v2/train_robots.sh data/v2_r6_final r6f ; trust_for r6f data/v2_r6_final
scripts/v2/run_robot_table_r5.sh r6f results/v2/trust_r6f.json
echo PIPELINE_DONE
