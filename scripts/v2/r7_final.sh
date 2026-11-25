#!/usr/bin/env bash
# MODEL-V2 round 4 FINAL (pre-registered, 11c): bin_clearing lots, n=30, tau=0.15 (VAL), 4 cells x 6 policies,
# <= 4 niced CPU processes, idempotent (cells resume from partial json), HOLD-aware.
cd ~/dynagraph
export PYTHONPATH=src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl TAG=r7 TAU=0.15 EPISODE_TIMEOUT=300
run() { [ -f results/v2/closed_loop/v2_r7_lot_$1.json ] && return; while [ -f ~/dynagraph/HOLD ]; do sleep 60; done
        ROBOT=$1 SPLIT=$2 nice -n 10 python scripts/v2/lot_eval.py v2_r7_lot_$1 bin_clearing 30 > logs/v2_r7_lot_$1.log 2>&1; }
export -f run
printf '%s\n' "ur5e_2f85 test" "panda test" "gen3_2f85 heldout_robot" "probe test" | xargs -P ${PROCS:-4} -L 1 bash -c 'run $0 $1'
echo R7_FINAL_DONE
