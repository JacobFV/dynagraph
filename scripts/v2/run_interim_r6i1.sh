#!/usr/bin/env bash
# MODEL-V2 interim (labelled INTERIM, superseded by the r6f table): round-1 models, priority fragile cells, n=30,
# heuristic / probe-then-act / typed s0 / oracle s0 / residual_gated / residual_gated_probe; site-dressed; niced.
C=results/v2/checkpoints; N=30
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 NO_PRIOR=1 DRESSING=site NO_UNGATED=1 TRUST=results/v2/trust_r6i1.json
export RESIDUAL=$C/v2_robots_r6i1_typed_s0.pt,$C/v2_robots_r6i1_typed_s1.pt,$C/v2_robots_r6i1_typed_s2.pt
run() { [ -f results/v2/closed_loop/v2_tab5_r6i1_$1_$3.json ] && return; while [ -f ~/dynagraph/HOLD ]; do sleep 60; done
        ROBOT=$1 SPLIT=$2 nice -n 10 python scripts/v2/env_closed_loop.py v2_tab5_r6i1_$1_$3 $3 $N $C/v2_robots_r6i1_typed_s0.pt > logs/v2_tab5_r6i1_$1_$3.log 2>&1; }
export -f run; export C N
printf '%s\n' "ur5e_2f85 test conveyor_sort" "panda test conveyor_sort" "ur5e_2f85 test tool_return" "panda test tool_return" \
  "ur5e_2f85 test fixture_place" "panda test fixture_place" "ur5e_2f85 test bin_clearing" "probe test depalletize" | xargs -P ${PROCS:-6} -L 1 bash -c 'run $0 $1 $2'
