#!/usr/bin/env bash
# MODEL-V2 round 5 (r8) TEST, pre-registered in docs/MODEL_V2.md 12a: 5-part lots, shared budget BPP x 5 calls,
# fresh seeds (+900000), n=30, tau 0.15, 4 cells x 6 policies; <= PROCS niced procs; resumable; HOLD-aware.
# usage: BPP=<frozen VAL choice> PROCS=4 scripts/v2/r8_final.sh
cd ~/dynagraph && source .venv/bin/activate
: "${BPP:?set BPP to the budget-per-part frozen in 12a}"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl TAG=r7 TAU=0.15 EPISODE_TIMEOUT=60 \
       SEED_OFF=900000 NPARTS=5 BUDGET_PER_PART=$BPP \
       POLICIES="heuristic,heuristic_probe,generic,typed_v2,typed_nolot_v2,oracle_v2(labelled)"
run() { [ -f results/v2/closed_loop/v2_r8_lot_$1.json ] && return; while [ -f ~/dynagraph/HOLD ]; do sleep 60; done
        ROBOT=$1 SPLIT=$2 nice -n 10 python scripts/v2/lot_eval.py v2_r8_lot_$1 bin_clearing 30 >> logs/v2_r8_lot_$1.log 2>&1; }
export -f run
printf '%s\n' "probe test" "ur5e_2f85 test" "gen3_2f85 heldout_robot" "panda test" | xargs -P ${PROCS:-4} -L 1 bash -c 'run $0 $1'
echo R8_FINAL_DONE
