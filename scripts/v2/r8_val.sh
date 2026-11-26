#!/usr/bin/env bash
# MODEL-V2 round 5 (r8) VAL budget sweep: baselines ONLY (never-probe, always-probe PTA), 5-part lots, shared budget
# {2.4, 3.0, 4.0} calls/part, ur5e + probe gantry, VAL split, 20 lots, <= 4 niced procs, resumable (partial json).
cd ~/dynagraph && source .venv/bin/activate
export PYTHONPATH=src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl TAG=r7 TAU=0.15 EPISODE_TIMEOUT=60 \
       SEED_OFF=900000 NPARTS=5 SPLIT=val POLICIES=heuristic,heuristic_probe OUT_DIR=results/v2_val_r8 V2_RESULTS=results/v2_val_r8
run() { [ -f results/v2_val_r8/val_r8_$1_b$2.json ] && return
        ROBOT=$1 BUDGET_PER_PART=$2 nice -n 10 python scripts/v2/lot_eval.py val_r8_$1_b$2 bin_clearing 20 > logs/v2_r8_val_$1_b$2.log 2>&1; }
export -f run
printf '%s\n' "ur5e_2f85 2.4" "probe 2.4" "ur5e_2f85 3.0" "probe 3.0" "ur5e_2f85 4.0" "probe 4.0" | xargs -P ${PROCS:-4} -L 1 bash -c 'run $0 $1'
echo R8_VAL_DONE
