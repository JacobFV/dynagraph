#!/usr/bin/env bash
# MODEL-V2 round 6 (r9): il-v1 demos on envs-r7 (asserted), demo policy = env.demo_call (oracle-param scripted),
# train split, make_tasks(base_seed=3_000_000), N per family x robot (<= 200), wrist camera 64x64 at 10 Hz.
# Also writes the memory-model view of the same episodes (data/v2_il_r7/mem). <= PROCS niced procs; resumable.
# usage: CELLS="fam:robot ..." N=200 PROCS=2 scripts/v2/r9_record.sh
cd ~/dynagraph && source .venv/bin/activate
scripts/v2/m6_overlay.sh || exit 1
export PYTHONPATH=$HOME/dynagraph_m6run71/src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl EXPECT_ENV=envs-r7.1 \
       DEMO_POLICY=${DEMO_POLICY:-demo} IL_IMG=64 TASKS=make_tasks N=${N:-200} SPLIT=${SPLIT:-train}
rec() { f=${1%%:*}; r=${1##*:}; [ -f ${OUT:-data/v2_il_r7}/${f}__${r}.npz ] || nice -n 10 python scripts/v2/il_record.py $f $r $N ${OUT:-data/v2_il_r7} 3000000 >> logs/v2_r9_record${TAG}.log 2>&1; }
export -f rec
printf '%s\n' ${CELLS} | xargs -P ${PROCS:-2} -L 1 bash -c 'rec $0'
echo R9_RECORD_DONE >> logs/v2_r9_record${TAG}.log
