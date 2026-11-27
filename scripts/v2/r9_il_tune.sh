#!/usr/bin/env bash
# MODEL-V2 round 6: ACT VAL tuning on envs-r7.1 (il-v1 demos, data/v2_il_r7). One config per call:
#   RUN=<name> TRAIN_ARGS="--H 20 --img 64 --steps 20000 --beta 10" [ENS_M=0.01] [NVAL=20] [PROCS=2] r9_il_tune.sh
# trains on the GPU (gpu_heavy lock, resumable) unless the checkpoint exists, then evaluates on the VAL cells
# (make_tasks val, base 3_000_000) on <= PROCS niced CPU procs; results/v2_il/val/il_<RUN>_m<ENS_M>__<cell>__val.json
cd ~/dynagraph && source .venv/bin/activate
scripts/v2/m6_overlay.sh > /dev/null || exit 1
export PYTHONPATH=$HOME/dynagraph_m6run71/src OMP_NUM_THREADS=1 MUJOCO_GL=egl EXPECT_ENV=envs-r7.1 TASKS=make_tasks
: "${RUN:?}"; ENS_M=${ENS_M:-0.01}; NVAL=${NVAL:-20}
CELLS=${CELLS:-"tight_insert:ur5e_2f85 tight_insert:panda fixture_place:ur5e_2f85 fixture_place:panda bin_clearing:ur5e_2f85 bin_clearing:panda"}
if [ -n "$NO_TRAIN" ]; then       # a separate GPU loop trains; just wait for the checkpoint
  until [ -f results/v2/checkpoints/il_$RUN.pt ]; do sleep 60; done
elif [ ! -f results/v2/checkpoints/il_$RUN.pt ]; then
  until [ $(free -g | awk '/Mem:/ {print $7}') -ge 30 ]; do sleep 60; done
  scripts/gpu_heavy.sh il_$RUN "nice -n 10 python scripts/v2/il_train.py data/v2_il_r7 $RUN --pos abs --d 192 --bs 128 $TRAIN_ARGS" >> logs/v2_il_$RUN.train.log 2>&1
fi
[ -f results/v2/checkpoints/il_$RUN.pt ] || { echo "no checkpoint for $RUN"; exit 1; }
export RUN ENS_M NVAL CUDA_VISIBLE_DEVICES=
ev() { f=${1%%:*}; r=${1##*:}; o=results/v2_il/val/il_${RUN}_m${ENS_M}__${f}__${r}__val.json
       [ -f $o ] || nice -n 10 python scripts/v2/il_eval.py $RUN $f $r val $NVAL 3000000 $o >> logs/v2_il_${RUN}_m${ENS_M}.eval.log 2>&1; }
export -f ev
printf '%s\n' $CELLS | xargs -P ${PROCS:-2} -L 1 bash -c 'ev $0'
echo IL_TUNE_DONE $RUN $ENS_M >> logs/v2_il_${RUN}_m${ENS_M}.eval.log
