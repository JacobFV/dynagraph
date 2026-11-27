#!/usr/bin/env bash
# round-6 DRY RUN eval: ACT run $1 on the 6 r6 cells, VAL seeds 2950000+i (same as the scripted VAL record), 20 each.
cd ~/dynagraph && source .venv/bin/activate
export PYTHONPATH=src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl RUN=$1 EXPECT_ENV=${EXPECT_ENV:-envs-r6}
ev() { [ -f results/v2_il_dry/il_${RUN}__$1__$2__val.json ] || nice -n 10 python scripts/v2/il_eval.py $RUN $1 $2 val 20 2950000 results/v2_il_dry/il_${RUN}__$1__$2__val.json >> logs/v2_il_dry_eval_$RUN.log 2>&1; }
export -f ev
printf '%s\n' "fixture_place ur5e_2f85" "fixture_place panda" "twist_lock ur5e_2f85" "twist_lock panda" "door_open ur5e_2f85" "door_open panda" | xargs -P ${PROCS:-2} -L 1 bash -c 'ev $0 $1'
echo IL_DRY_EVAL_DONE >> logs/v2_il_dry_eval_$RUN.log
