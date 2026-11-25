#!/usr/bin/env bash
# MODEL-V2 round 4 EXPLORATORY v2 (11e): crush-gated probing, FRESH lot seeds (+800000), n=30, same cells.
# <= 4 niced CPU processes, idempotent (cells resume from partial json), HOLD-aware.
cd ~/dynagraph
export PYTHONPATH=src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl TAG=r7 TAU=0.15 EPISODE_TIMEOUT=300 SEED_OFF=800000 POLICIES="heuristic,heuristic_probe,generic,typed_v2,typed_nolot_v2,oracle_v2(labelled)"
run() { [ -f results/v2/closed_loop/v2_r7v2_lot_$1.json ] && return; while [ -f ~/dynagraph/HOLD ]; do sleep 60; done
        ROBOT=$1 SPLIT=$2 nice -n 10 python scripts/v2/lot_eval.py v2_r7v2_lot_$1 bin_clearing 30 > logs/v2_r7v2_lot_$1.log 2>&1; }
export -f run
printf '%s\n' "probe test" "panda test" "ur5e_2f85 test" "gen3_2f85 heldout_robot" | xargs -P ${PROCS:-4} -L 1 bash -c 'run $0 $1'
echo R7V2_DONE
