#!/usr/bin/env bash
# 13a secondary S-lot: bin_clearing lots (5 parts, 12 shared calls) on envs-r7.1, n=30, seeds test base + 950000,
# typed v2 vs classical cached estimator vs PTA ...; models r9 full. <= PROCS niced procs; resumable.
cd ~/dynagraph && source .venv/bin/activate
scripts/v2/m6_overlay.sh > /dev/null || exit 1
export PYTHONPATH=$HOME/dynagraph_m6run71/src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl EXPECT_ENV=envs-r7.1 \
       TAG=r9_full TAU=0.15 EPISODE_TIMEOUT=300 SEED_OFF=950000 NPARTS=5 BUDGET_PER_PART=2.4 \
       POLICIES="heuristic,heuristic_probe,generic,typed_v2,typed_nolot_v2,cached_estimator,oracle_v2(labelled)"
run() { [ -f results/v2/closed_loop/v2_r9lot_$1.json ] || ROBOT=$1 SPLIT=$2 nice -n 10 python scripts/v2/lot_eval.py v2_r9lot_$1 bin_clearing 30 >> logs/v2_r9lot_$1.log 2>&1; }
export -f run
printf '%s\n' "probe test" "ur5e_2f85 test" "panda test" "gen3_2f85 heldout_robot" | xargs -P ${PROCS:-1} -L 1 bash -c 'run $0 $1'
echo R9LOT_DONE >> logs/v2_r9lot.log
