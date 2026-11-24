#!/usr/bin/env bash
# MODEL-V2 r5 pipeline (remote): base training -> trust (val) -> 2 on-policy rounds on real stations -> final table.
set -x
P=${PROCS:-9}
base=data/v2_r5_all
wait_file() { until [ -f "$1" ]; do sleep 30; done; }
wait_file logs/v2_r5_merge.done
trust_for() {  # $1 = tag -> results/v2/trust_$1.json from VAL success AUC of typed s0
  python -m physmem.v2_train eval --ckpt results/v2/checkpoints/v2_robots_$1_typed_s0.pt --data $2 --splits val > logs/v2_eval_$1_val.log 2>&1
  python scripts/v2/make_trust.py logs/v2_eval_$1_val.log results/v2/trust_$1.json v2_robots_$1_typed_s0
}
# round 0: base models
scripts/v2/train_robots.sh $base r5 ; trust_for r5 $base
prev=r5; rdirs=""
for r in 1 2; do
  out=data/v2_r5_rollouts/round$r
  cells=""
  for rb in ur5e_2f85 panda; do for f in tool_return conveyor_sort tote_transfer pouch_transfer twist_lock machine_tending bin_clearing fixture_place; do cells="$cells $rb:$f"; done; done
  cells="$cells probe:depalletize ur5e_suction:depalletize"
  echo $cells | tr ' ' '\n' | grep . | xargs -P $P -I{} bash -c 'c={}; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl python scripts/v2/env_onpolicy.py '$r' ${c%%:*} ${c##*:} ${N_OP:-24} '$out' '$prev' results/v2/trust_'$prev'.json >> logs/v2_onpolicy_env_round'$r'.log 2>&1'
  rdirs="$rdirs $out"
  python scripts/v2/merge_onpolicy.py $base data/v2_r5_op$r $rdirs > logs/v2_merge_op$r.log 2>&1
  scripts/v2/train_robots.sh data/v2_r5_op$r r5op$r ; trust_for r5op$r data/v2_r5_op$r
  prev=r5op$r
done
scripts/v2/run_robot_table_r5.sh $prev results/v2/trust_$prev.json
echo PIPELINE_DONE
